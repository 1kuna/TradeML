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


class DataNodeDB:
    """Small transactional wrapper around the local SQLite state store."""

    REQUIRED_TABLES = frozenset({"backfill_queue", "partition_status", "vendor_attempts"})

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
                connection.execute(
                    """
                    INSERT INTO vendor_attempts (
                      task_key, task_family, planner_group, vendor, lease_owner, status, attempts,
                      leased_at, lease_expires_at, payload_json, updated_at
                    ) VALUES (?, ?, ?, ?, ?, 'LEASED', 0, ?, ?, ?, ?)
                    """,
                    (task_key, task_family, planner_group, vendor, lease_owner, lease_time, expires_at, payload_json, lease_time),
                )
            leased = connection.execute(
                "SELECT * FROM vendor_attempts WHERE task_key = ? AND vendor = ?",
                (task_key, vendor),
            ).fetchone()
        return VendorAttempt(**dict(leased)) if leased is not None else None

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
