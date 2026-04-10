"""SQLite helpers for the Pi data-node task queue and partition ledger."""

from __future__ import annotations

import sqlite3
import threading
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


@dataclass(slots=True)
class CanonicalUnit:
    """Durable canonical symbol-date coverage unit."""

    dataset: str
    symbol: str
    trading_date: str
    status: str
    source_name: str | None
    task_key: str | None
    written_at: str | None
    partition_revision: int
    last_error: str | None


@dataclass(slots=True)
class RawPartitionManifest:
    """Metadata describing the compacted raw partition for a trading date."""

    dataset: str
    trading_date: str
    partition_revision: int
    symbol_count: int
    row_count: int
    symbols_json: str | None
    content_hash: str | None
    last_compacted_at: str | None
    status: str

    @property
    def symbols(self) -> tuple[str, ...]:
        """Return the compacted symbol set for this partition."""
        if not self.symbols_json:
            return ()
        try:
            raw = json.loads(self.symbols_json)
        except json.JSONDecodeError:
            return ()
        return tuple(str(value) for value in raw)


@dataclass(slots=True)
class VendorLaneHealth:
    """Durable scheduler health state for a vendor/dataset lane."""

    vendor: str
    dataset: str
    state: str
    cooldown_until: str | None
    last_state_change: str
    recent_outbound_requests: int
    recent_success_units: int
    recent_remote_429s: int
    recent_local_budget_blocks: int
    recent_empty_valid: int
    recent_permanent_failures: int
    updated_at: str


class DataNodeDB:
    """Small transactional wrapper around the local SQLite state store."""

    REQUIRED_TABLES = frozenset(
        {
            "backfill_queue",
            "partition_status",
            "vendor_attempts",
            "planner_tasks",
            "planner_task_progress",
            "canonical_units",
            "raw_partition_manifest",
            "vendor_lane_health",
        }
    )

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._connection_lock = threading.RLock()
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
        with self._connection_lock:
            connection = sqlite3.connect(self.path, timeout=30.0)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA busy_timeout = 30000")
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
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS canonical_units (
                  dataset              TEXT NOT NULL,
                  symbol               TEXT NOT NULL,
                  trading_date         DATE NOT NULL,
                  status               TEXT NOT NULL,
                  source_name          TEXT,
                  task_key             TEXT,
                  written_at           TIMESTAMP,
                  partition_revision   INTEGER NOT NULL DEFAULT 0,
                  last_error           TEXT,
                  PRIMARY KEY (dataset, symbol, trading_date)
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_canonical_units_date
                ON canonical_units(dataset, trading_date, status)
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS raw_partition_manifest (
                  dataset              TEXT NOT NULL,
                  trading_date         DATE NOT NULL,
                  partition_revision   INTEGER NOT NULL DEFAULT 0,
                  symbol_count         INTEGER NOT NULL DEFAULT 0,
                  row_count            INTEGER NOT NULL DEFAULT 0,
                  symbols_json         TEXT,
                  content_hash         TEXT,
                  last_compacted_at    TIMESTAMP,
                  status               TEXT NOT NULL,
                  PRIMARY KEY (dataset, trading_date)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS vendor_lane_health (
                  vendor                   TEXT NOT NULL,
                  dataset                  TEXT NOT NULL,
                  state                    TEXT NOT NULL,
                  cooldown_until           TIMESTAMP,
                  last_state_change        TIMESTAMP NOT NULL,
                  recent_outbound_requests INTEGER NOT NULL DEFAULT 0,
                  recent_success_units     INTEGER NOT NULL DEFAULT 0,
                  recent_remote_429s       INTEGER NOT NULL DEFAULT 0,
                  recent_local_budget_blocks INTEGER NOT NULL DEFAULT 0,
                  recent_empty_valid       INTEGER NOT NULL DEFAULT 0,
                  recent_permanent_failures INTEGER NOT NULL DEFAULT 0,
                  updated_at               TIMESTAMP NOT NULL,
                  PRIMARY KEY (vendor, dataset)
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

    def replace_canonical_units_for_date(
        self,
        *,
        dataset: str,
        trading_date: str,
        symbols: list[str] | tuple[str, ...],
        partition_revision: int,
        source_names: dict[str, str] | None = None,
        task_key: str | None = None,
    ) -> None:
        """Replace the durable symbol-date ledger for a single compacted partition."""
        timestamp = utc_now().isoformat()
        normalized = sorted({str(symbol).upper() for symbol in symbols if str(symbol).strip()})
        with self._connect() as connection:
            existing_rows = connection.execute(
                """
                SELECT symbol
                FROM canonical_units
                WHERE dataset = ? AND trading_date = ?
                """,
                (dataset, trading_date),
            ).fetchall()
            existing = {str(row["symbol"]).upper() for row in existing_rows}
            missing = sorted(existing.difference(normalized))
            if missing:
                placeholders = ",".join("?" for _ in missing)
                connection.execute(
                    f"""
                    UPDATE canonical_units
                    SET status = 'MISSING',
                        written_at = ?,
                        partition_revision = ?,
                        last_error = 'missing from compacted partition'
                    WHERE dataset = ?
                      AND trading_date = ?
                      AND symbol IN ({placeholders})
                    """,
                    [timestamp, int(partition_revision), dataset, trading_date, *missing],
                )
            payloads = [
                (
                    dataset,
                    symbol,
                    trading_date,
                    "WRITTEN",
                    str((source_names or {}).get(symbol) or ""),
                    task_key,
                    timestamp,
                    int(partition_revision),
                    None,
                )
                for symbol in normalized
            ]
            connection.executemany(
                """
                INSERT INTO canonical_units (
                  dataset, symbol, trading_date, status, source_name, task_key,
                  written_at, partition_revision, last_error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(dataset, symbol, trading_date)
                DO UPDATE SET
                  status = excluded.status,
                  source_name = excluded.source_name,
                  task_key = COALESCE(excluded.task_key, canonical_units.task_key),
                  written_at = excluded.written_at,
                  partition_revision = excluded.partition_revision,
                  last_error = excluded.last_error
                """,
                payloads,
            )

    def mark_canonical_units_status(
        self,
        *,
        dataset: str,
        trading_date: str,
        symbols: list[str] | tuple[str, ...] | None,
        status: str,
        last_error: str | None = None,
    ) -> int:
        """Mark canonical units for a date/symbol scope with an explicit status."""
        timestamp = utc_now().isoformat()
        with self._connect() as connection:
            params: list[object] = [status, last_error, timestamp, dataset, trading_date]
            query = """
                UPDATE canonical_units
                SET status = ?,
                    last_error = ?,
                    written_at = ?
                WHERE dataset = ?
                  AND trading_date = ?
            """
            if symbols:
                normalized = sorted({str(symbol).upper() for symbol in symbols})
                placeholders = ",".join("?" for _ in normalized)
                query += f" AND symbol IN ({placeholders})"
                params.extend(normalized)
            cursor = connection.execute(query, params)
        return int(cursor.rowcount)

    def fetch_canonical_progress(
        self,
        *,
        dataset: str,
        symbols: list[str] | tuple[str, ...],
        trading_days: list[str] | tuple[str, ...],
    ) -> dict[str, object]:
        """Return canonical progress for a symbol/date scope from the durable ledger."""
        normalized_symbols = sorted({str(symbol).upper() for symbol in symbols})
        normalized_days = sorted({str(day) for day in trading_days})
        counts_by_symbol = {symbol: 0 for symbol in normalized_symbols}
        if not normalized_symbols or not normalized_days:
            return {
                "trading_days": normalized_days,
                "completed_symbols": [],
                "remaining_symbols": normalized_symbols,
                "expected_units": len(normalized_symbols) * len(normalized_days),
                "completed_units": 0,
                "remaining_units": len(normalized_symbols) * len(normalized_days),
            }
        with self._connect() as connection:
            symbol_placeholders = ",".join("?" for _ in normalized_symbols)
            day_placeholders = ",".join("?" for _ in normalized_days)
            rows = connection.execute(
                f"""
                SELECT symbol, trading_date
                FROM canonical_units
                WHERE dataset = ?
                  AND status = 'WRITTEN'
                  AND symbol IN ({symbol_placeholders})
                  AND trading_date IN ({day_placeholders})
                """,
                [dataset, *normalized_symbols, *normalized_days],
            ).fetchall()
        completed_pairs: set[tuple[str, str]] = set()
        for row in rows:
            symbol = str(row["symbol"]).upper()
            trading_date = str(row["trading_date"])
            pair = (trading_date, symbol)
            if pair in completed_pairs:
                continue
            completed_pairs.add(pair)
            counts_by_symbol[symbol] = counts_by_symbol.get(symbol, 0) + 1
        completed_symbols = sorted(symbol for symbol, count in counts_by_symbol.items() if count >= len(normalized_days))
        remaining_symbols = sorted(symbol for symbol, count in counts_by_symbol.items() if count < len(normalized_days))
        expected_units = len(normalized_symbols) * len(normalized_days)
        completed_units = len(completed_pairs)
        return {
            "trading_days": normalized_days,
            "completed_symbols": completed_symbols,
            "remaining_symbols": remaining_symbols,
            "expected_units": expected_units,
            "completed_units": completed_units,
            "remaining_units": max(0, expected_units - completed_units),
        }

    def fetch_canonical_units_for_date(
        self,
        *,
        dataset: str,
        trading_date: str,
        statuses: tuple[str, ...] | None = None,
    ) -> list[CanonicalUnit]:
        """Return canonical units for a trading date."""
        query = "SELECT * FROM canonical_units WHERE dataset = ? AND trading_date = ?"
        params: list[object] = [dataset, trading_date]
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            query += f" AND status IN ({placeholders})"
            params.extend(statuses)
        query += " ORDER BY symbol"
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [CanonicalUnit(**dict(row)) for row in rows]

    def upsert_raw_partition_manifest(
        self,
        *,
        dataset: str,
        trading_date: str,
        partition_revision: int,
        symbol_count: int,
        row_count: int,
        symbols: list[str] | tuple[str, ...],
        content_hash: str | None,
        status: str,
        last_compacted_at: str | None = None,
    ) -> None:
        """Upsert the durable manifest row for a compacted raw partition."""
        timestamp = last_compacted_at or utc_now().isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO raw_partition_manifest (
                  dataset, trading_date, partition_revision, symbol_count, row_count,
                  symbols_json, content_hash, last_compacted_at, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(dataset, trading_date)
                DO UPDATE SET
                  partition_revision = excluded.partition_revision,
                  symbol_count = excluded.symbol_count,
                  row_count = excluded.row_count,
                  symbols_json = excluded.symbols_json,
                  content_hash = excluded.content_hash,
                  last_compacted_at = excluded.last_compacted_at,
                  status = excluded.status
                """,
                (
                    dataset,
                    trading_date,
                    int(partition_revision),
                    int(symbol_count),
                    int(row_count),
                    json.dumps(sorted({str(symbol).upper() for symbol in symbols}), sort_keys=True),
                    content_hash,
                    timestamp,
                    status,
                ),
            )

    def get_raw_partition_manifest(self, *, dataset: str, trading_date: str) -> RawPartitionManifest | None:
        """Return the raw partition manifest for a date when present."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM raw_partition_manifest
                WHERE dataset = ? AND trading_date = ?
                """,
                (dataset, trading_date),
            ).fetchone()
        return RawPartitionManifest(**dict(row)) if row is not None else None

    def fetch_raw_partition_manifests(
        self,
        *,
        dataset: str | None = None,
        statuses: tuple[str, ...] | None = None,
    ) -> list[RawPartitionManifest]:
        """Return raw partition manifests filtered by dataset/status."""
        query = "SELECT * FROM raw_partition_manifest"
        clauses: list[str] = []
        params: list[object] = []
        if dataset:
            clauses.append("dataset = ?")
            params.append(dataset)
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            clauses.append(f"status IN ({placeholders})")
            params.extend(statuses)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY trading_date ASC"
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [RawPartitionManifest(**dict(row)) for row in rows]

    def mark_raw_partition_manifest_status(
        self,
        *,
        dataset: str,
        trading_date: str,
        status: str,
    ) -> None:
        """Update only the manifest health status for a date."""
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE raw_partition_manifest
                SET status = ?,
                    last_compacted_at = ?
                WHERE dataset = ? AND trading_date = ?
                """,
                (status, utc_now().isoformat(), dataset, trading_date),
            )

    def upsert_vendor_lane_health(
        self,
        *,
        vendor: str,
        dataset: str,
        state: str,
        cooldown_until: str | None = None,
        recent_outbound_requests: int = 0,
        recent_success_units: int = 0,
        recent_remote_429s: int = 0,
        recent_local_budget_blocks: int = 0,
        recent_empty_valid: int = 0,
        recent_permanent_failures: int = 0,
    ) -> None:
        """Persist the current scheduler health state for a vendor lane."""
        timestamp = utc_now().isoformat()
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT state, last_state_change FROM vendor_lane_health WHERE vendor = ? AND dataset = ?",
                (vendor, dataset),
            ).fetchone()
            last_state_change = timestamp
            if existing is not None and str(existing["state"]) == state:
                last_state_change = str(existing["last_state_change"])
            connection.execute(
                """
                INSERT INTO vendor_lane_health (
                  vendor, dataset, state, cooldown_until, last_state_change,
                  recent_outbound_requests, recent_success_units, recent_remote_429s,
                  recent_local_budget_blocks, recent_empty_valid, recent_permanent_failures, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(vendor, dataset)
                DO UPDATE SET
                  state = excluded.state,
                  cooldown_until = excluded.cooldown_until,
                  last_state_change = excluded.last_state_change,
                  recent_outbound_requests = excluded.recent_outbound_requests,
                  recent_success_units = excluded.recent_success_units,
                  recent_remote_429s = excluded.recent_remote_429s,
                  recent_local_budget_blocks = excluded.recent_local_budget_blocks,
                  recent_empty_valid = excluded.recent_empty_valid,
                  recent_permanent_failures = excluded.recent_permanent_failures,
                  updated_at = excluded.updated_at
                """,
                (
                    vendor,
                    dataset,
                    state,
                    cooldown_until,
                    last_state_change,
                    int(recent_outbound_requests),
                    int(recent_success_units),
                    int(recent_remote_429s),
                    int(recent_local_budget_blocks),
                    int(recent_empty_valid),
                    int(recent_permanent_failures),
                    timestamp,
                ),
            )

    def get_vendor_lane_health(self, *, vendor: str, dataset: str) -> VendorLaneHealth | None:
        """Return current lane-health state for a vendor/dataset."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM vendor_lane_health WHERE vendor = ? AND dataset = ?",
                (vendor, dataset),
            ).fetchone()
        return VendorLaneHealth(**dict(row)) if row is not None else None

    def vendor_lane_health_map(self, *, dataset: str) -> dict[str, VendorLaneHealth]:
        """Return lane-health rows keyed by vendor."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM vendor_lane_health WHERE dataset = ? ORDER BY vendor",
                (dataset,),
            ).fetchall()
        return {str(row["vendor"]): VendorLaneHealth(**dict(row)) for row in rows}

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
        allow_success_retry: bool = False,
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
                    if status == "SUCCESS" and not allow_success_retry:
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

    def vendor_attempts_for_symbol(
        self,
        *,
        vendor: str,
        symbol: str,
        task_family: str = "canonical_bars",
    ) -> list[VendorAttempt]:
        """Return vendor attempts whose payload scope includes the requested symbol."""
        pattern = f'%"{str(symbol).upper()}"%'
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM vendor_attempts
                WHERE vendor = ?
                  AND task_family = ?
                  AND payload_json LIKE ?
                ORDER BY updated_at DESC
                """,
                (vendor, task_family, pattern),
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

    def prune_planner_tasks(self, *, task_families: tuple[str, ...], valid_task_keys: set[str]) -> int:
        """Delete planner tasks and associated state that are no longer part of the planned backlog."""
        if not task_families:
            return 0
        placeholders = ",".join("?" for _ in task_families)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT task_key FROM planner_tasks WHERE task_family IN ({placeholders})",
                list(task_families),
            ).fetchall()
            stale_keys = [str(row["task_key"]) for row in rows if str(row["task_key"]) not in valid_task_keys]
            if not stale_keys:
                return 0
            for index in range(0, len(stale_keys), 500):
                chunk = stale_keys[index : index + 500]
                chunk_placeholders = ",".join("?" for _ in chunk)
                connection.execute(
                    f"DELETE FROM vendor_attempts WHERE task_key IN ({chunk_placeholders})",
                    chunk,
                )
                connection.execute(
                    f"DELETE FROM planner_task_progress WHERE task_key IN ({chunk_placeholders})",
                    chunk,
                )
                connection.execute(
                    f"DELETE FROM planner_tasks WHERE task_key IN ({chunk_placeholders})",
                    chunk,
                )
        return len(stale_keys)

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

    def reopen_planner_tasks(self, task_keys: list[str] | tuple[str, ...], *, reason: str) -> int:
        """Reopen planner tasks whose coverage regressed and clear stale attempt state."""
        if not task_keys:
            return 0
        timestamp = utc_now().isoformat()
        reopened = 0
        with self._connect() as connection:
            for index in range(0, len(task_keys), 500):
                chunk = [str(task_key) for task_key in task_keys[index : index + 500]]
                placeholders = ",".join("?" for _ in chunk)
                connection.execute(
                    f"DELETE FROM vendor_attempts WHERE task_key IN ({placeholders})",
                    chunk,
                )
                cursor = connection.execute(
                    f"""
                    UPDATE planner_tasks
                    SET status = 'PENDING',
                        attempts = 0,
                        lease_owner = NULL,
                        leased_at = NULL,
                        lease_expires_at = NULL,
                        next_eligible_at = NULL,
                        last_error = ?,
                        updated_at = ?
                    WHERE task_key IN ({placeholders})
                      AND status IN ('SUCCESS', 'PERMANENT_FAILED')
                    """,
                    [reason, timestamp, *chunk],
                )
                reopened += int(cursor.rowcount)
        return reopened

    def clear_planner_task_backoff(self, task_keys: list[str] | tuple[str, ...], *, reason: str) -> int:
        """Clear task-level planner backoff while preserving vendor attempt history."""
        if not task_keys:
            return 0
        timestamp = utc_now().isoformat()
        updated = 0
        with self._connect() as connection:
            for index in range(0, len(task_keys), 500):
                chunk = [str(task_key) for task_key in task_keys[index : index + 500]]
                placeholders = ",".join("?" for _ in chunk)
                cursor = connection.execute(
                    f"""
                    UPDATE planner_tasks
                    SET status = CASE WHEN status = 'FAILED' THEN 'PARTIAL' ELSE status END,
                        lease_owner = NULL,
                        leased_at = NULL,
                        lease_expires_at = NULL,
                        next_eligible_at = NULL,
                        last_error = ?,
                        updated_at = ?
                    WHERE task_key IN ({placeholders})
                      AND status IN ('PARTIAL', 'FAILED', 'LEASED')
                    """,
                    [reason, timestamp, *chunk],
                )
                updated += int(cursor.rowcount)
        return updated

    def release_planner_leases_for_owner(
        self,
        *,
        lease_owner: str,
        task_families: tuple[str, ...] | None = None,
    ) -> int:
        """Release any in-flight planner leases owned by a restarting worker."""
        clauses = ["status = 'LEASED'", "lease_owner = ?"]
        params: list[object] = [lease_owner]
        if task_families:
            placeholders = ",".join("?" for _ in task_families)
            clauses.append(f"task_family IN ({placeholders})")
            params.extend(task_families)
        with self._connect() as connection:
            cursor = connection.execute(
                f"""
                UPDATE planner_tasks
                SET status = CASE WHEN attempts > 0 THEN 'PARTIAL' ELSE 'PENDING' END,
                    lease_owner = NULL,
                    leased_at = NULL,
                    lease_expires_at = NULL,
                    next_eligible_at = NULL,
                    updated_at = ?
                WHERE {' AND '.join(clauses)}
                """,
                [utc_now().isoformat(), *params],
            )
        return int(cursor.rowcount)

    def release_vendor_attempt_leases_for_owner(
        self,
        *,
        lease_owner: str,
        task_families: tuple[str, ...] | None = None,
        reason: str,
    ) -> int:
        """Release any in-flight vendor attempts owned by a restarting worker."""
        clauses = ["status = 'LEASED'", "lease_owner = ?"]
        params: list[object] = [lease_owner]
        if task_families:
            placeholders = ",".join("?" for _ in task_families)
            clauses.append(f"task_family IN ({placeholders})")
            params.extend(task_families)
        with self._connect() as connection:
            cursor = connection.execute(
                f"""
                UPDATE vendor_attempts
                SET status = 'FAILED',
                    lease_owner = NULL,
                    leased_at = NULL,
                    lease_expires_at = NULL,
                    next_eligible_at = NULL,
                    last_error = COALESCE(last_error, ?),
                    updated_at = ?
                WHERE {' AND '.join(clauses)}
                """,
                [reason, utc_now().isoformat(), *params],
            )
        return int(cursor.rowcount)

    def count_repairable_stale_success_canonical_tasks(self, *, only_future_blocked: bool) -> int:
        """Count incomplete canonical tasks blocked by stale budget failures despite prior success."""
        now_iso = utc_now().isoformat()
        where = [
            "pt.task_family = 'canonical_bars'",
            "pt.status IN ('PARTIAL', 'FAILED', 'LEASED')",
            "pt.last_error LIKE '%budget exhausted%'",
            "prog.remaining_units > 0",
            "EXISTS (SELECT 1 FROM vendor_attempts va WHERE va.task_key = pt.task_key AND va.status = 'SUCCESS')",
        ]
        params: list[object] = []
        if only_future_blocked:
            where.append("pt.next_eligible_at IS NOT NULL")
            where.append("pt.next_eligible_at > ?")
            params.append(now_iso)
        else:
            where.append("(pt.next_eligible_at IS NULL OR pt.next_eligible_at <= ?)")
            params.append(now_iso)
        query = f"""
            SELECT COUNT(*)
            FROM planner_tasks pt
            JOIN planner_task_progress prog
              ON prog.task_key = pt.task_key
            WHERE {' AND '.join(where)}
        """
        with self._connect() as connection:
            row = connection.execute(query, params).fetchone()
        return int(row[0] if row is not None else 0)

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
            backlog_rows = connection.execute(
                """
                SELECT CASE
                         WHEN planner_tasks.planner_group = 'phase1_pinned_canonical' THEN 'phase1_pinned'
                         WHEN planner_tasks.planner_group = 'rolling_canonical' THEN 'rolling'
                         WHEN planner_tasks.planner_group = 'canonical_repair' THEN 'repair'
                         ELSE planner_tasks.planner_group
                       END AS backlog_class,
                       SUM(planner_task_progress.expected_units) AS expected_units,
                       SUM(planner_task_progress.completed_units) AS completed_units,
                       SUM(planner_task_progress.remaining_units) AS remaining_units
                FROM planner_tasks
                LEFT JOIN planner_task_progress
                  ON planner_tasks.task_key = planner_task_progress.task_key
                WHERE planner_tasks.task_family IN ('canonical_bars', 'canonical_repair')
                GROUP BY backlog_class
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
        backlog_progress: dict[str, dict[str, int]] = {}
        for row in backlog_rows:
            backlog_progress[str(row["backlog_class"])] = {
                "expected_units": int(row["expected_units"] or 0),
                "completed_units": int(row["completed_units"] or 0),
                "remaining_units": int(row["remaining_units"] or 0),
            }
        return {"counts": counts, "progress": progress, "backlog_progress": backlog_progress}
