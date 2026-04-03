"""SQLite helpers for the Pi data-node task queue and partition ledger."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator
import json


UTC = timezone.utc


def utc_now() -> datetime:
    """Return the current UTC time."""
    return datetime.now(tz=UTC)


@dataclass(slots=True)
class BackfillTask:
    """SQLite-backed queue task."""

    id: int
    dataset: str
    symbol: str | None
    start_date: str
    end_date: str
    kind: str
    priority: int
    status: str
    attempts: int
    next_not_before: str | None
    last_error: str | None
    created_at: str
    updated_at: str


@dataclass(slots=True)
class VendorAttempt:
    """Persistent per-vendor attempt state for canonical and supplemental tasks."""

    task_key: str
    task_family: str
    planner_group: str
    vendor: str
    lease_owner: str | None
    status: str
    attempts: int
    last_error: str | None
    next_eligible_at: str | None
    leased_at: str | None
    lease_expires_at: str | None
    rows_returned: int | None
    payload_json: str | None
    updated_at: str

    @property
    def payload(self) -> dict:
        """Deserialize the optional planner payload."""
        if not self.payload_json:
            return {}
        try:
            return json.loads(self.payload_json)
        except json.JSONDecodeError:
            return {}


@dataclass(slots=True)
class PlannerTask:
    """Persistent planner-native work unit."""

    task_key: str
    task_family: str
    planner_group: str
    dataset: str
    tier: str
    priority: int
    start_date: str
    end_date: str
    symbols_json: str
    eligible_vendors_json: str
    output_name: str | None
    payload_json: str | None
    status: str
    lease_owner: str | None
    leased_at: str | None
    lease_expires_at: str | None
    next_eligible_at: str | None
    attempts: int
    last_error: str | None
    created_at: str
    updated_at: str

    @property
    def symbols(self) -> tuple[str, ...]:
        """Return the symbol scope for the task."""
        try:
            raw = json.loads(self.symbols_json)
        except json.JSONDecodeError:
            return ()
        return tuple(str(value) for value in raw)

    @property
    def eligible_vendors(self) -> tuple[str, ...]:
        """Return the vendors allowed to serve the task."""
        try:
            raw = json.loads(self.eligible_vendors_json)
        except json.JSONDecodeError:
            return ()
        return tuple(str(value) for value in raw)

    @property
    def payload(self) -> dict:
        """Deserialize the planner payload."""
        if not self.payload_json:
            return {}
        try:
            return json.loads(self.payload_json)
        except json.JSONDecodeError:
            return {}


@dataclass(slots=True)
class PlannerTaskProgress:
    """Persistent progress state for a planner task."""

    task_key: str
    expected_units: int
    completed_units: int
    remaining_units: int
    completed_symbols_json: str | None
    remaining_symbols_json: str | None
    state_json: str | None
    updated_at: str

    @property
    def completed_symbols(self) -> tuple[str, ...]:
        """Return the symbols fully covered for the task."""
        if not self.completed_symbols_json:
            return ()
        try:
            raw = json.loads(self.completed_symbols_json)
        except json.JSONDecodeError:
            return ()
        return tuple(str(value) for value in raw)

    @property
    def remaining_symbols(self) -> tuple[str, ...]:
        """Return the symbols still missing coverage for the task."""
        if not self.remaining_symbols_json:
            return ()
        try:
            raw = json.loads(self.remaining_symbols_json)
        except json.JSONDecodeError:
            return ()
        return tuple(str(value) for value in raw)

    @property
    def state(self) -> dict:
        """Return the planner progress payload."""
        if not self.state_json:
            return {}
        try:
            return json.loads(self.state_json)
        except json.JSONDecodeError:
            return {}


class DataNodeDB:
    """Small transactional wrapper around the local SQLite state store."""

    REQUIRED_TABLES = frozenset(
        {"backfill_queue", "partition_status", "vendor_attempts", "planner_tasks", "planner_task_progress"}
    )

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._initialize()

    @classmethod
    def recreate(cls, path: str | Path) -> "DataNodeDB":
        """Replace the SQLite database and sidecars with a clean initialized store."""
        db_path = Path(path)
        for candidate in [db_path, db_path.with_name(f"{db_path.name}-wal"), db_path.with_name(f"{db_path.name}-shm")]:
            if candidate.exists():
                candidate.unlink()
        return cls(db_path)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 5000")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS backfill_queue (
                  id               INTEGER PRIMARY KEY AUTOINCREMENT,
                  dataset          TEXT NOT NULL,
                  symbol           TEXT,
                  start_date       DATE NOT NULL,
                  end_date         DATE NOT NULL,
                  kind             TEXT NOT NULL,
                  priority         INTEGER NOT NULL,
                  status           TEXT NOT NULL,
                  attempts         INTEGER DEFAULT 0,
                  next_not_before  TIMESTAMP,
                  last_error       TEXT,
                  created_at       TIMESTAMP NOT NULL,
                  updated_at       TIMESTAMP NOT NULL,
                  UNIQUE(dataset, symbol, start_date, end_date, kind)
                )
                """
            )
            connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_backfill_queue_unique
                ON backfill_queue(dataset, IFNULL(symbol, '__ALL__'), start_date, end_date, kind)
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS partition_status (
                  source           TEXT NOT NULL,
                  dataset          TEXT NOT NULL,
                  date             DATE NOT NULL,
                  status           TEXT NOT NULL,
                  row_count        INTEGER,
                  expected_rows    INTEGER,
                  qc_code          TEXT,
                  note             TEXT,
                  updated_at       TIMESTAMP NOT NULL,
                  PRIMARY KEY (source, dataset, date)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS vendor_attempts (
                  task_key         TEXT NOT NULL,
                  task_family      TEXT NOT NULL,
                  planner_group    TEXT NOT NULL,
                  vendor           TEXT NOT NULL,
                  lease_owner      TEXT,
                  status           TEXT NOT NULL,
                  attempts         INTEGER DEFAULT 0,
                  last_error       TEXT,
                  next_eligible_at TIMESTAMP,
                  leased_at        TIMESTAMP,
                  lease_expires_at TIMESTAMP,
                  rows_returned    INTEGER,
                  payload_json     TEXT,
                  updated_at       TIMESTAMP NOT NULL,
                  PRIMARY KEY (task_key, vendor)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS planner_tasks (
                  task_key             TEXT PRIMARY KEY,
                  task_family          TEXT NOT NULL,
                  planner_group        TEXT NOT NULL,
                  dataset              TEXT NOT NULL,
                  tier                 TEXT NOT NULL,
                  priority             INTEGER NOT NULL,
                  start_date           DATE NOT NULL,
                  end_date             DATE NOT NULL,
                  symbols_json         TEXT NOT NULL,
                  eligible_vendors_json TEXT NOT NULL,
                  output_name          TEXT,
                  payload_json         TEXT,
                  status               TEXT NOT NULL,
                  lease_owner          TEXT,
                  leased_at            TIMESTAMP,
                  lease_expires_at     TIMESTAMP,
                  next_eligible_at     TIMESTAMP,
                  attempts             INTEGER DEFAULT 0,
                  last_error           TEXT,
                  created_at           TIMESTAMP NOT NULL,
                  updated_at           TIMESTAMP NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_planner_tasks_status
                ON planner_tasks(status, priority, updated_at)
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS planner_task_progress (
                  task_key             TEXT PRIMARY KEY,
                  expected_units       INTEGER NOT NULL,
                  completed_units      INTEGER NOT NULL,
                  remaining_units      INTEGER NOT NULL,
                  completed_symbols_json TEXT,
                  remaining_symbols_json TEXT,
                  state_json           TEXT,
                  updated_at           TIMESTAMP NOT NULL
                )
                """
            )
        self._validate_schema()

    def _validate_schema(self) -> None:
        """Verify the required tables exist after initialization."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        existing = {str(row["name"]) for row in rows}
        missing = self.REQUIRED_TABLES.difference(existing)
        if missing:
            raise sqlite3.OperationalError(f"missing sqlite tables: {', '.join(sorted(missing))}")

    def enqueue_task(
        self,
        dataset: str,
        symbol: str | None,
        start_date: str,
        end_date: str,
        kind: str,
        priority: int,
    ) -> int:
        """Insert a new task and return its identifier."""
        timestamp = utc_now().isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO backfill_queue (
                  dataset, symbol, start_date, end_date, kind, priority, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'PENDING', ?, ?)
                """,
                (dataset, symbol, start_date, end_date, kind, priority, timestamp, timestamp),
            )
            return int(cursor.lastrowid)

    def lease_next_task(self, now: datetime | None = None) -> BackfillTask | None:
        """Lease the next available task ordered by priority then age."""
        lease_time = (now or utc_now()).isoformat()
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM backfill_queue
                WHERE status IN ('PENDING', 'FAILED')
                  AND COALESCE(next_not_before, '1970-01-01T00:00:00+00:00') <= ?
                ORDER BY priority ASC, created_at ASC, id ASC
                LIMIT 1
                """,
                (lease_time,),
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                """
                UPDATE backfill_queue
                SET status = 'LEASED', updated_at = ?
                WHERE id = ?
                """,
                (lease_time, row["id"]),
            )
            leased_row = connection.execute("SELECT * FROM backfill_queue WHERE id = ?", (row["id"],)).fetchone()
            assert leased_row is not None
            return BackfillTask(**dict(leased_row))

    def peek_next_tasks(self, *, limit: int = 25, now: datetime | None = None) -> list[BackfillTask]:
        """Return eligible tasks without leasing them."""
        lease_time = (now or utc_now()).isoformat()
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM backfill_queue
                WHERE status IN ('PENDING', 'FAILED')
                  AND COALESCE(next_not_before, '1970-01-01T00:00:00+00:00') <= ?
                ORDER BY priority ASC, created_at ASC, id ASC
                LIMIT ?
                """,
                (lease_time, limit),
            ).fetchall()
        return [BackfillTask(**dict(row)) for row in rows]

    def lease_task_by_id(self, task_id: int, now: datetime | None = None) -> BackfillTask | None:
        """Lease a specific task if it is still eligible."""
        lease_time = (now or utc_now()).isoformat()
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM backfill_queue
                WHERE id = ?
                  AND status IN ('PENDING', 'FAILED')
                  AND COALESCE(next_not_before, '1970-01-01T00:00:00+00:00') <= ?
                """,
                (task_id, lease_time),
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                "UPDATE backfill_queue SET status = 'LEASED', updated_at = ? WHERE id = ?",
                (lease_time, task_id),
            )
            leased_row = connection.execute("SELECT * FROM backfill_queue WHERE id = ?", (task_id,)).fetchone()
        return BackfillTask(**dict(leased_row)) if leased_row is not None else None

    def mark_task_done(self, task_id: int) -> None:
        """Mark a leased task as complete."""
        with self._connect() as connection:
            connection.execute(
                "UPDATE backfill_queue SET status = 'DONE', updated_at = ? WHERE id = ?",
                (utc_now().isoformat(), task_id),
            )

    def mark_task_failed(self, task_id: int, error: str, backoff_minutes: int) -> None:
        """Mark a task failed and defer it until the backoff expires."""
        next_not_before = utc_now() + timedelta(minutes=backoff_minutes)
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE backfill_queue
                SET status = 'FAILED',
                    attempts = attempts + 1,
                    last_error = ?,
                    next_not_before = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (error, next_not_before.isoformat(), utc_now().isoformat(), task_id),
            )

    def defer_task(self, task_id: int, *, reason: str, backoff_minutes: int) -> None:
        """Return a leased task to pending state with a future eligibility time."""
        next_not_before = utc_now() + timedelta(minutes=backoff_minutes)
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE backfill_queue
                SET status = 'PENDING',
                    last_error = ?,
                    next_not_before = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (reason, next_not_before.isoformat(), utc_now().isoformat(), task_id),
            )

    def requeue_retryable_failures(self) -> int:
        """Move retryable failed tasks back to pending without discarding backoff."""
        patterns = (
            "%budget exhausted%",
            "%request failed: 429%",
            "%hourly request allocation%",
            "%daily request allocation%",
            "%too many requests%",
            "%rate limit%",
        )
        where_clause = " OR ".join("lower(coalesce(last_error, '')) LIKE ?" for _ in patterns)
        with self._connect() as connection:
            cursor = connection.execute(
                f"""
                UPDATE backfill_queue
                SET status = 'PENDING',
                    updated_at = ?
                WHERE status = 'FAILED'
                  AND ({where_clause})
                """,
                (utc_now().isoformat(), *patterns),
            )
        return int(cursor.rowcount)

    def update_partition_status(
        self,
        source: str,
        dataset: str,
        date: str,
        status: str,
        row_count: int | None,
        expected_rows: int | None = None,
        qc_code: str | None = None,
        note: str | None = None,
    ) -> None:
        """Upsert the partition QC ledger."""
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO partition_status (
                  source, dataset, date, status, row_count, expected_rows, qc_code, note, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source, dataset, date)
                DO UPDATE SET
                  status = excluded.status,
                  row_count = excluded.row_count,
                  expected_rows = excluded.expected_rows,
                  qc_code = excluded.qc_code,
                  note = excluded.note,
                  updated_at = excluded.updated_at
                """,
                (
                    source,
                    dataset,
                    date,
                    status,
                    row_count,
                    expected_rows,
                    qc_code,
                    note,
                    utc_now().isoformat(),
                ),
            )

    def fetch_partition_status(self) -> list[sqlite3.Row]:
        """Return the mirrored partition ledger for testing and sync."""
        with self._connect() as connection:
            return connection.execute(
                "SELECT * FROM partition_status ORDER BY source, dataset, date"
            ).fetchall()

    def has_pending_backfill(self) -> bool:
        """Return whether any backfill work is currently eligible to run."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT 1
                FROM backfill_queue
                WHERE status IN ('PENDING', 'FAILED')
                  AND COALESCE(next_not_before, '1970-01-01T00:00:00+00:00') <= ?
                LIMIT 1
                """,
                (utc_now().isoformat(),),
            ).fetchone()
            return row is not None

    def has_pending_datewide_backfill(self) -> bool:
        """Return whether any legacy datewide backlog remains."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT 1
                FROM backfill_queue
                WHERE symbol IS NULL
                  AND status IN ('PENDING', 'FAILED')
                  AND COALESCE(next_not_before, '1970-01-01T00:00:00+00:00') <= ?
                LIMIT 1
                """,
                (utc_now().isoformat(),),
            ).fetchone()
        return row is not None

    def mark_legacy_datewide_backfill_migrated(self) -> int:
        """Retire legacy backlog rows after planner migration is active."""
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE backfill_queue
                SET status = 'DONE',
                    last_error = 'migrated_to_planner',
                    updated_at = ?
                WHERE status IN ('PENDING', 'FAILED', 'LEASED')
                """,
                (utc_now().isoformat(),),
            )
        return int(cursor.rowcount)

    def latest_queue_update(self, *, statuses: tuple[str, ...] | None = None) -> str | None:
        """Return the most recent queue update timestamp, optionally filtered by status."""
        with self._connect() as connection:
            if statuses:
                placeholders = ",".join("?" for _ in statuses)
                row = connection.execute(
                    f"SELECT MAX(updated_at) AS updated_at FROM backfill_queue WHERE status IN ({placeholders})",
                    statuses,
                ).fetchone()
            else:
                row = connection.execute("SELECT MAX(updated_at) AS updated_at FROM backfill_queue").fetchone()
        if row is None:
            return None
        return row["updated_at"]

    def has_pending_planner_tasks(self, *, task_families: tuple[str, ...] | None = None) -> bool:
        """Return whether any planner task is eligible to run."""
        query = """
            SELECT 1
            FROM planner_tasks
            WHERE status IN ('PENDING', 'PARTIAL', 'FAILED')
              AND COALESCE(next_eligible_at, '1970-01-01T00:00:00+00:00') <= ?
        """
        params: list[object] = [utc_now().isoformat()]
        if task_families:
            placeholders = ",".join("?" for _ in task_families)
            query += f" AND task_family IN ({placeholders})"
            params.extend(task_families)
        query += " LIMIT 1"
        with self._connect() as connection:
            row = connection.execute(query, params).fetchone()
        return row is not None

    def latest_planner_update(self, *, statuses: tuple[str, ...] | None = None, task_families: tuple[str, ...] | None = None) -> str | None:
        """Return the latest planner task update timestamp."""
        query = "SELECT MAX(updated_at) AS updated_at FROM planner_tasks"
        clauses: list[str] = []
        params: list[object] = []
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            clauses.append(f"status IN ({placeholders})")
            params.extend(statuses)
        if task_families:
            placeholders = ",".join("?" for _ in task_families)
            clauses.append(f"task_family IN ({placeholders})")
            params.extend(task_families)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        with self._connect() as connection:
            row = connection.execute(query, params).fetchone()
        if row is None:
            return None
        return row["updated_at"]

    def lease_vendor_attempt(
        self,
        *,
        task_key: str,
        task_family: str,
        planner_group: str,
        vendor: str,
        lease_owner: str,
        payload: dict | None = None,
        lease_ttl_seconds: int = 90,
        now: datetime | None = None,
    ) -> VendorAttempt | None:
        """Lease a vendor attempt if it is eligible and not already complete."""
        current = (now or utc_now())
        lease_time = current.isoformat()
        expires_at = (current + timedelta(seconds=lease_ttl_seconds)).isoformat()
        payload_json = json.dumps(payload, sort_keys=True) if payload else None
        for _attempt in range(3):
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT * FROM vendor_attempts WHERE task_key = ? AND vendor = ?",
                    (task_key, vendor),
                ).fetchone()
                if row is not None:
                    status = str(row["status"])
                    lease_expiry = row["lease_expires_at"]
                    if status == "SUCCESS":
                        return None
                    if status == "LEASED" and lease_expiry and lease_expiry > lease_time and row["lease_owner"] != lease_owner:
                        return None
                    if row["next_eligible_at"] and row["next_eligible_at"] > lease_time:
                        return None
                    connection.execute(
                        """
                        UPDATE vendor_attempts
                        SET task_family = ?,
                            planner_group = ?,
                            lease_owner = ?,
                            status = 'LEASED',
                            leased_at = ?,
                            lease_expires_at = ?,
                            payload_json = COALESCE(?, payload_json),
                            updated_at = ?
                        WHERE task_key = ? AND vendor = ?
                        """,
                        (
                            task_family,
                            planner_group,
                            lease_owner,
                            lease_time,
                            expires_at,
                            payload_json,
                            lease_time,
                            task_key,
                            vendor,
                        ),
                    )
                else:
                    try:
                        connection.execute(
                            """
                            INSERT INTO vendor_attempts (
                              task_key, task_family, planner_group, vendor, lease_owner, status, attempts,
                              leased_at, lease_expires_at, payload_json, updated_at
                            ) VALUES (?, ?, ?, ?, ?, 'LEASED', 0, ?, ?, ?, ?)
                            """,
                            (
                                task_key,
                                task_family,
                                planner_group,
                                vendor,
                                lease_owner,
                                lease_time,
                                expires_at,
                                payload_json,
                                lease_time,
                            ),
                        )
                    except sqlite3.IntegrityError:
                        continue
                leased = connection.execute(
                    "SELECT * FROM vendor_attempts WHERE task_key = ? AND vendor = ?",
                    (task_key, vendor),
                ).fetchone()
            return VendorAttempt(**dict(leased)) if leased is not None else None
        return None

    def mark_vendor_attempt_success(
        self,
        *,
        task_key: str,
        vendor: str,
        rows_returned: int = 0,
    ) -> None:
        """Mark a vendor attempt successful."""
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE vendor_attempts
                SET status = 'SUCCESS',
                    rows_returned = ?,
                    lease_owner = NULL,
                    next_eligible_at = NULL,
                    last_error = NULL,
                    lease_expires_at = NULL,
                    updated_at = ?
                WHERE task_key = ? AND vendor = ?
                """,
                (rows_returned, utc_now().isoformat(), task_key, vendor),
            )

    def mark_vendor_attempt_failed(
        self,
        *,
        task_key: str,
        vendor: str,
        error: str,
        backoff_minutes: int,
        permanent: bool = False,
    ) -> None:
        """Mark a vendor attempt failed with backoff or permanently."""
        status = "PERMANENT_FAILED" if permanent else "FAILED"
        next_eligible_at = None if permanent else (utc_now() + timedelta(minutes=backoff_minutes)).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE vendor_attempts
                SET status = ?,
                    attempts = attempts + 1,
                    last_error = ?,
                    next_eligible_at = ?,
                    lease_owner = NULL,
                    lease_expires_at = NULL,
                    updated_at = ?
                WHERE task_key = ? AND vendor = ?
                """,
                (status, error, next_eligible_at, utc_now().isoformat(), task_key, vendor),
            )

    def vendor_attempts_for_task(self, task_key: str) -> list[VendorAttempt]:
        """Return all vendor attempts for a deterministic task key."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM vendor_attempts WHERE task_key = ? ORDER BY vendor",
                (task_key,),
            ).fetchall()
        return [VendorAttempt(**dict(row)) for row in rows]

    def fetch_vendor_attempts(self, *, planner_group: str | None = None) -> list[VendorAttempt]:
        """Return all vendor attempts for dashboard/status views."""
        with self._connect() as connection:
            if planner_group:
                rows = connection.execute(
                    "SELECT * FROM vendor_attempts WHERE planner_group = ? ORDER BY updated_at DESC, task_key, vendor",
                    (planner_group,),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM vendor_attempts ORDER BY updated_at DESC, task_key, vendor"
                ).fetchall()
        return [VendorAttempt(**dict(row)) for row in rows]

    def upsert_planner_task(
        self,
        *,
        task_key: str,
        task_family: str,
        planner_group: str,
        dataset: str,
        tier: str,
        priority: int,
        start_date: str,
        end_date: str,
        symbols: list[str] | tuple[str, ...],
        eligible_vendors: list[str] | tuple[str, ...],
        output_name: str | None = None,
        payload: dict | None = None,
    ) -> None:
        """Insert or refresh a planner task without clobbering in-flight state."""
        timestamp = utc_now().isoformat()
        symbols_json = json.dumps(list(symbols), sort_keys=True)
        vendors_json = json.dumps(list(eligible_vendors), sort_keys=True)
        payload_json = json.dumps(payload or {}, sort_keys=True)
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT status, created_at FROM planner_tasks WHERE task_key = ?",
                (task_key,),
            ).fetchone()
            status = str(existing["status"]) if existing is not None else "PENDING"
            created_at = str(existing["created_at"]) if existing is not None else timestamp
            connection.execute(
                """
                INSERT INTO planner_tasks (
                  task_key, task_family, planner_group, dataset, tier, priority,
                  start_date, end_date, symbols_json, eligible_vendors_json, output_name,
                  payload_json, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_key)
                DO UPDATE SET
                  task_family = excluded.task_family,
                  planner_group = excluded.planner_group,
                  dataset = excluded.dataset,
                  tier = excluded.tier,
                  priority = excluded.priority,
                  start_date = excluded.start_date,
                  end_date = excluded.end_date,
                  symbols_json = excluded.symbols_json,
                  eligible_vendors_json = excluded.eligible_vendors_json,
                  output_name = excluded.output_name,
                  payload_json = excluded.payload_json,
                  updated_at = excluded.updated_at
                """,
                (
                    task_key,
                    task_family,
                    planner_group,
                    dataset,
                    tier,
                    priority,
                    start_date,
                    end_date,
                    symbols_json,
                    vendors_json,
                    output_name,
                    payload_json,
                    status,
                    created_at,
                    timestamp,
                ),
            )

    def bulk_upsert_planner_tasks(self, tasks: list[dict[str, object]]) -> None:
        """Insert or refresh planner tasks in a single transaction."""
        if not tasks:
            return
        timestamp = utc_now().isoformat()
        with self._connect() as connection:
            existing_map: dict[str, tuple[str, str]] = {}
            task_keys = [str(task["task_key"]) for task in tasks]
            for index in range(0, len(task_keys), 500):
                chunk = task_keys[index : index + 500]
                placeholders = ",".join("?" for _ in chunk)
                rows = connection.execute(
                    f"SELECT task_key, status, created_at FROM planner_tasks WHERE task_key IN ({placeholders})",
                    chunk,
                ).fetchall()
                for row in rows:
                    existing_map[str(row["task_key"])] = (str(row["status"]), str(row["created_at"]))
            prepared: list[tuple[object, ...]] = []
            for task in tasks:
                status, created_at = existing_map.get(str(task["task_key"]), ("PENDING", timestamp))
                prepared.append(
                    (
                        task["task_key"],
                        task["task_family"],
                        task["planner_group"],
                        task["dataset"],
                        task["tier"],
                        int(task["priority"]),
                        task["start_date"],
                        task["end_date"],
                        json.dumps(list(task["symbols"]), sort_keys=True),
                        json.dumps(list(task["eligible_vendors"]), sort_keys=True),
                        task.get("output_name"),
                        json.dumps(task.get("payload") or {}, sort_keys=True),
                        status,
                        created_at,
                        timestamp,
                    )
                )
            connection.executemany(
                """
                INSERT INTO planner_tasks (
                  task_key, task_family, planner_group, dataset, tier, priority,
                  start_date, end_date, symbols_json, eligible_vendors_json, output_name,
                  payload_json, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_key)
                DO UPDATE SET
                  task_family = excluded.task_family,
                  planner_group = excluded.planner_group,
                  dataset = excluded.dataset,
                  tier = excluded.tier,
                  priority = excluded.priority,
                  start_date = excluded.start_date,
                  end_date = excluded.end_date,
                  symbols_json = excluded.symbols_json,
                  eligible_vendors_json = excluded.eligible_vendors_json,
                  output_name = excluded.output_name,
                  payload_json = excluded.payload_json,
                  updated_at = excluded.updated_at
                """,
                prepared,
            )

    def get_planner_task(self, task_key: str) -> PlannerTask | None:
        """Return a planner task by key."""
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM planner_tasks WHERE task_key = ?", (task_key,)).fetchone()
        return PlannerTask(**dict(row)) if row is not None else None

    def fetch_planner_tasks(
        self,
        *,
        task_family: str | None = None,
        task_families: tuple[str, ...] | None = None,
        planner_group: str | None = None,
        statuses: tuple[str, ...] | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[PlannerTask]:
        """Return planner tasks filtered for status views."""
        query = "SELECT * FROM planner_tasks"
        clauses: list[str] = []
        params: list[object] = []
        if task_family:
            clauses.append("task_family = ?")
            params.append(task_family)
        if task_families:
            placeholders = ",".join("?" for _ in task_families)
            clauses.append(f"task_family IN ({placeholders})")
            params.extend(task_families)
        if planner_group:
            clauses.append("planner_group = ?")
            params.append(planner_group)
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            clauses.append(f"status IN ({placeholders})")
            params.extend(statuses)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY priority ASC, created_at ASC, task_key ASC"
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        if offset:
            query += " OFFSET ?"
            params.append(offset)
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [PlannerTask(**dict(row)) for row in rows]

    def lease_next_planner_task(
        self,
        *,
        lease_owner: str,
        task_families: tuple[str, ...] | None = None,
        vendor: str | None = None,
        now: datetime | None = None,
        lease_ttl_seconds: int = 300,
        limit: int = 256,
        scan_pages: int = 16,
    ) -> PlannerTask | None:
        """Lease the next eligible planner task, optionally filtered for a vendor."""
        lease_time = (now or utc_now())
        lease_time_iso = lease_time.isoformat()
        for page in range(max(1, scan_pages)):
            candidates = self.fetch_planner_tasks(
                task_families=task_families,
                statuses=("PENDING", "PARTIAL", "FAILED", "LEASED"),
                limit=limit,
                offset=page * limit,
            )
            if not candidates:
                break
            for candidate in candidates:
                if task_families and candidate.task_family not in task_families:
                    continue
                if vendor and vendor not in candidate.eligible_vendors:
                    continue
                if candidate.status == "LEASED" and candidate.lease_expires_at and candidate.lease_expires_at > lease_time_iso:
                    continue
                if candidate.next_eligible_at and candidate.next_eligible_at > lease_time_iso:
                    continue
                expires_at = (lease_time + timedelta(seconds=lease_ttl_seconds)).isoformat()
                with self._connect() as connection:
                    updated = connection.execute(
                        """
                        UPDATE planner_tasks
                        SET status = 'LEASED',
                            lease_owner = ?,
                            leased_at = ?,
                            lease_expires_at = ?,
                            updated_at = ?
                        WHERE task_key = ?
                          AND (
                            status IN ('PENDING', 'PARTIAL', 'FAILED')
                            OR (status = 'LEASED' AND COALESCE(lease_expires_at, '1970-01-01T00:00:00+00:00') <= ?)
                          )
                          AND COALESCE(next_eligible_at, '1970-01-01T00:00:00+00:00') <= ?
                        """,
                        (
                            lease_owner,
                            lease_time_iso,
                            expires_at,
                            lease_time_iso,
                            candidate.task_key,
                            lease_time_iso,
                            lease_time_iso,
                        ),
                    ).rowcount
                    if updated:
                        row = connection.execute(
                            "SELECT * FROM planner_tasks WHERE task_key = ?",
                            (candidate.task_key,),
                        ).fetchone()
                        return PlannerTask(**dict(row))
        return None

    def lease_planner_task_by_key(
        self,
        *,
        task_key: str,
        lease_owner: str,
        now: datetime | None = None,
        lease_ttl_seconds: int = 300,
    ) -> PlannerTask | None:
        """Lease a specific planner task if it is still eligible."""
        lease_time = (now or utc_now())
        lease_time_iso = lease_time.isoformat()
        expires_at = (lease_time + timedelta(seconds=lease_ttl_seconds)).isoformat()
        with self._connect() as connection:
            updated = connection.execute(
                """
                UPDATE planner_tasks
                SET status = 'LEASED',
                    lease_owner = ?,
                    leased_at = ?,
                    lease_expires_at = ?,
                    updated_at = ?
                WHERE task_key = ?
                  AND (
                    status IN ('PENDING', 'PARTIAL', 'FAILED')
                    OR (status = 'LEASED' AND COALESCE(lease_expires_at, '1970-01-01T00:00:00+00:00') <= ?)
                  )
                  AND COALESCE(next_eligible_at, '1970-01-01T00:00:00+00:00') <= ?
                """,
                (
                    lease_owner,
                    lease_time_iso,
                    expires_at,
                    lease_time_iso,
                    task_key,
                    lease_time_iso,
                    lease_time_iso,
                ),
            ).rowcount
            if not updated:
                return None
            row = connection.execute("SELECT * FROM planner_tasks WHERE task_key = ?", (task_key,)).fetchone()
        return PlannerTask(**dict(row)) if row is not None else None

    def mark_planner_task_success(self, task_key: str) -> None:
        """Mark a planner task successful."""
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE planner_tasks
                SET status = 'SUCCESS',
                    lease_owner = NULL,
                    leased_at = NULL,
                    lease_expires_at = NULL,
                    next_eligible_at = NULL,
                    last_error = NULL,
                    updated_at = ?
                WHERE task_key = ?
                """,
                (utc_now().isoformat(), task_key),
            )

    def mark_planner_task_partial(self, task_key: str, *, error: str | None = None, backoff_minutes: int = 1) -> None:
        """Return a planner task to partial state with optional backoff."""
        next_eligible = (utc_now() + timedelta(minutes=backoff_minutes)).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE planner_tasks
                SET status = 'PARTIAL',
                    attempts = attempts + 1,
                    lease_owner = NULL,
                    leased_at = NULL,
                    lease_expires_at = NULL,
                    next_eligible_at = ?,
                    last_error = ?,
                    updated_at = ?
                WHERE task_key = ?
                """,
                (next_eligible, error, utc_now().isoformat(), task_key),
            )

    def mark_planner_task_failed(
        self,
        task_key: str,
        *,
        error: str,
        backoff_minutes: int,
        permanent: bool = False,
    ) -> None:
        """Mark a planner task failed with backoff or permanently."""
        status = "PERMANENT_FAILED" if permanent else "FAILED"
        next_eligible = None if permanent else (utc_now() + timedelta(minutes=backoff_minutes)).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE planner_tasks
                SET status = ?,
                    attempts = attempts + 1,
                    lease_owner = NULL,
                    leased_at = NULL,
                    lease_expires_at = NULL,
                    next_eligible_at = ?,
                    last_error = ?,
                    updated_at = ?
                WHERE task_key = ?
                """,
                (status, next_eligible, error, utc_now().isoformat(), task_key),
            )

    def update_planner_task_progress(
        self,
        *,
        task_key: str,
        expected_units: int,
        completed_units: int,
        remaining_units: int,
        completed_symbols: list[str] | tuple[str, ...] | None = None,
        remaining_symbols: list[str] | tuple[str, ...] | None = None,
        state: dict | None = None,
    ) -> None:
        """Upsert planner task progress."""
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO planner_task_progress (
                  task_key, expected_units, completed_units, remaining_units,
                  completed_symbols_json, remaining_symbols_json, state_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_key)
                DO UPDATE SET
                  expected_units = excluded.expected_units,
                  completed_units = excluded.completed_units,
                  remaining_units = excluded.remaining_units,
                  completed_symbols_json = excluded.completed_symbols_json,
                  remaining_symbols_json = excluded.remaining_symbols_json,
                  state_json = excluded.state_json,
                  updated_at = excluded.updated_at
                """,
                (
                    task_key,
                    int(expected_units),
                    int(completed_units),
                    int(remaining_units),
                    json.dumps(list(completed_symbols or ()), sort_keys=True) if completed_symbols is not None else None,
                    json.dumps(list(remaining_symbols or ()), sort_keys=True) if remaining_symbols is not None else None,
                    json.dumps(state or {}, sort_keys=True) if state is not None else None,
                    utc_now().isoformat(),
                ),
            )

    def bulk_update_planner_task_progress(self, rows: list[dict[str, object]]) -> None:
        """Upsert planner progress rows in a single transaction."""
        if not rows:
            return
        timestamp = utc_now().isoformat()
        payloads = [
            (
                row["task_key"],
                int(row["expected_units"]),
                int(row["completed_units"]),
                int(row["remaining_units"]),
                json.dumps(list(row.get("completed_symbols") or ()), sort_keys=True) if row.get("completed_symbols") is not None else None,
                json.dumps(list(row.get("remaining_symbols") or ()), sort_keys=True) if row.get("remaining_symbols") is not None else None,
                json.dumps(row.get("state") or {}, sort_keys=True) if row.get("state") is not None else None,
                timestamp,
            )
            for row in rows
        ]
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO planner_task_progress (
                  task_key, expected_units, completed_units, remaining_units,
                  completed_symbols_json, remaining_symbols_json, state_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_key)
                DO UPDATE SET
                  expected_units = excluded.expected_units,
                  completed_units = excluded.completed_units,
                  remaining_units = excluded.remaining_units,
                  completed_symbols_json = excluded.completed_symbols_json,
                  remaining_symbols_json = excluded.remaining_symbols_json,
                  state_json = excluded.state_json,
                  updated_at = excluded.updated_at
                """,
                payloads,
            )

    def fetch_planner_task_progress(self, task_key: str) -> PlannerTaskProgress | None:
        """Return progress for a planner task."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM planner_task_progress WHERE task_key = ?",
                (task_key,),
            ).fetchone()
        return PlannerTaskProgress(**dict(row)) if row is not None else None

    def planner_task_progress_map(self) -> dict[str, PlannerTaskProgress]:
        """Return planner progress keyed by task key."""
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM planner_task_progress").fetchall()
        return {str(row["task_key"]): PlannerTaskProgress(**dict(row)) for row in rows}

    def planner_summary(self) -> dict[str, object]:
        """Return aggregate planner counts and progress."""
        with self._connect() as connection:
            task_rows = connection.execute(
                """
                SELECT task_family, status, COUNT(*) AS count
                FROM planner_tasks
                GROUP BY task_family, status
                """
            ).fetchall()
            progress_rows = connection.execute(
                """
                SELECT planner_tasks.task_family AS task_family,
                       SUM(planner_task_progress.expected_units) AS expected_units,
                       SUM(planner_task_progress.completed_units) AS completed_units,
                       SUM(planner_task_progress.remaining_units) AS remaining_units
                FROM planner_tasks
                LEFT JOIN planner_task_progress
                  ON planner_tasks.task_key = planner_task_progress.task_key
                GROUP BY planner_tasks.task_family
                """
            ).fetchall()
        counts: dict[str, dict[str, int]] = {}
        for row in task_rows:
            family = str(row["task_family"])
            counts.setdefault(family, {})[str(row["status"])] = int(row["count"])
        progress: dict[str, dict[str, int]] = {}
        for row in progress_rows:
            progress[str(row["task_family"])] = {
                "expected_units": int(row["expected_units"] or 0),
                "completed_units": int(row["completed_units"] or 0),
                "remaining_units": int(row["remaining_units"] or 0),
            }
        return {"counts": counts, "progress": progress}
