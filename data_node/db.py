"""
SQLite control database for the unified Pi data-node.

Manages:
- backfill_queue: Unified ingest queue with kind ∈ {BOOTSTRAP, GAP, FORWARD, QC_PROBE}
- partition_status: Mirror of QC ledger for local queries

See SSOT_V2.md §2.8 and updated_node_spec.md §2 for schema details.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Generator, Optional

from loguru import logger
from .vendor_limits import massive_window


# Register SQLite adapters for date/datetime (Python 3.12+ compatible)
def _adapt_date(val: date) -> str:
    """Convert date to ISO format string for SQLite."""
    return val.isoformat()


def _adapt_datetime(val: datetime) -> str:
    """Convert datetime to ISO format string for SQLite."""
    return val.isoformat()


def _convert_date(val: bytes) -> date:
    """Convert SQLite date string to Python date."""
    return date.fromisoformat(val.decode())


def _convert_timestamp(val: bytes) -> datetime:
    """Convert SQLite timestamp string to Python datetime."""
    return datetime.fromisoformat(val.decode())


# Register adapters and converters
sqlite3.register_adapter(date, _adapt_date)
sqlite3.register_adapter(datetime, _adapt_datetime)
sqlite3.register_converter("DATE", _convert_date)
sqlite3.register_converter("TIMESTAMP", _convert_timestamp)


class TaskKind(str, Enum):
    """Task kinds for the unified ingest queue."""
    BOOTSTRAP = "BOOTSTRAP"
    GAP = "GAP"
    FORWARD = "FORWARD"
    QC_PROBE = "QC_PROBE"


class TaskStatus(str, Enum):
    """Task statuses."""
    PENDING = "PENDING"
    LEASED = "LEASED"
    DONE = "DONE"
    FAILED = "FAILED"


class PartitionStatus(str, Enum):
    """Partition QC status."""
    GREEN = "GREEN"
    AMBER = "AMBER"
    RED = "RED"


# Priority mappings per spec §0.2
PRIORITY_MAP = {
    (TaskKind.BOOTSTRAP, True): 0,   # BOOTSTRAP inside stage window
    (TaskKind.GAP, True): 1,         # GAP inside stage window
    (TaskKind.QC_PROBE, True): 2,    # QC_PROBE
    (TaskKind.QC_PROBE, False): 2,
    (TaskKind.BOOTSTRAP, False): 3,  # BOOTSTRAP outside stage window
    (TaskKind.GAP, False): 3,        # GAP outside stage window (same as old bootstrap)
    (TaskKind.FORWARD, True): 5,     # FORWARD
    (TaskKind.FORWARD, False): 5,
}

# Max days per task to prevent head-of-line blocking
# Minute data: smaller chunks (more data per day)
# EOD data: larger chunks (less data per day)
MAX_TASK_DAYS = {
    "equities_minute": 7,    # ~2,730 bars per task (7 × 390)
    "equities_eod": 30,      # 30 bars per task
    "fundamentals": 90,      # Quarterly data, larger chunks OK
    "default": 14,           # Reasonable default for unknown datasets
}


@dataclass
class Task:
    """A task from the backfill_queue."""
    id: int
    dataset: str
    symbol: Optional[str]
    start_date: str
    end_date: str
    kind: TaskKind
    priority: int
    status: TaskStatus
    attempts: int
    lease_owner: Optional[str]
    lease_expires_at: Optional[datetime]
    next_not_before: Optional[datetime]
    last_error: Optional[str]
    created_at: datetime
    updated_at: datetime


@dataclass
class PartitionRecord:
    """A partition status record."""
    source_name: str
    table_name: str
    symbol: Optional[str]
    dt: str
    status: PartitionStatus
    qc_score: float
    row_count: int
    expected_rows: int
    qc_code: Optional[str]
    first_observed_at: datetime
    last_observed_at: datetime


# Default database path (on SSD)
DEFAULT_DB_PATH = Path("data_layer/control/node.sqlite")


class NodeDB:
    """
    Thread-safe SQLite database manager for the Pi data node.

    Uses WAL mode for better concurrent access and provides
    transaction-safe helpers for queue operations.
    """

    _local = threading.local()

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize the database manager.

        Args:
            db_path: Path to SQLite database. Defaults to data_layer/control/node.sqlite
        """
        if db_path is None:
            # Check for DATA_ROOT env var
            data_root = os.environ.get("DATA_ROOT", ".")
            db_path = Path(data_root) / "data_layer" / "control" / "node.sqlite"

        self.db_path = Path(db_path)
        self._lock = threading.Lock()

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local connection with proper settings."""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(
                str(self.db_path),
                timeout=30.0,
                check_same_thread=False,
                detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
            )
            conn.row_factory = sqlite3.Row
            # Enable WAL mode for better concurrency
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=30000")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return self._local.conn

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for a transaction."""
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def init_db(self) -> None:
        """Create tables if they don't exist."""
        with self.transaction() as conn:
            # backfill_queue table (unified ingest queue)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS backfill_queue (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset          TEXT NOT NULL,
                    symbol           TEXT,
                    start_date       DATE NOT NULL,
                    end_date         DATE NOT NULL,
                    kind             TEXT NOT NULL CHECK (kind IN ('BOOTSTRAP', 'GAP', 'FORWARD', 'QC_PROBE')),
                    priority         INTEGER NOT NULL,
                    status           TEXT NOT NULL CHECK (status IN ('PENDING', 'LEASED', 'DONE', 'FAILED')),
                    attempts         INTEGER NOT NULL DEFAULT 0,
                    lease_owner      TEXT,
                    lease_expires_at TIMESTAMP,
                    next_not_before  TIMESTAMP,
                    last_error       TEXT,
                    created_at       TIMESTAMP NOT NULL,
                    updated_at       TIMESTAMP NOT NULL,
                    UNIQUE(dataset, symbol, start_date, end_date, kind)
                )
            """)

            # Index for efficient lease_next_task queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_backfill_queue_status
                ON backfill_queue(status, priority, created_at)
            """)

            # Index for vendor-aware lease queries (filters by dataset)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_backfill_queue_dataset
                ON backfill_queue(dataset, status, priority, created_at)
            """)

            # partition_status mirror table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS partition_status (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_name      TEXT NOT NULL,
                    table_name       TEXT NOT NULL,
                    symbol           TEXT,
                    dt               DATE,
                    partition_key    TEXT,
                    status           TEXT NOT NULL CHECK (status IN ('GREEN', 'AMBER', 'RED')),
                    qc_score         REAL,
                    row_count        INTEGER,
                    expected_rows    INTEGER,
                    qc_code          TEXT,
                    first_observed_at TIMESTAMP NOT NULL,
                    last_observed_at  TIMESTAMP NOT NULL,
                    notes            TEXT,
                    UNIQUE(source_name, table_name, symbol, dt)
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_partition_status_lookup
                ON partition_status(table_name, status, dt)
            """)

            # qc_probes table (cross-vendor QC results)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS qc_probes (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset          TEXT NOT NULL,
                    symbol           TEXT,
                    dt               DATE NOT NULL,
                    primary_vendor   TEXT,
                    secondary_vendor TEXT,
                    status           TEXT NOT NULL,
                    qc_code          TEXT,
                    details          TEXT,
                    created_at       TIMESTAMP NOT NULL,
                    updated_at       TIMESTAMP NOT NULL,
                    UNIQUE(dataset, symbol, dt, primary_vendor, secondary_vendor)
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_qc_probes_lookup
                ON qc_probes(dataset, dt, status)
            """)

            # Migration: Add fetch_params column for tracking API parameters
            # This enables selective re-fetch when config changes (e.g., IEX→SIP)
            try:
                conn.execute("ALTER TABLE partition_status ADD COLUMN fetch_params TEXT")
                logger.info("Added fetch_params column to partition_status")
            except sqlite3.OperationalError as e:
                if "duplicate column" not in str(e).lower():
                    raise

            logger.info(f"Initialized control database at {self.db_path}")

    # -------------------------------------------------------------------------
    # Queue operations
    # -------------------------------------------------------------------------

    def enqueue_task(
        self,
        dataset: str,
        symbol: Optional[str],
        start_date: str,
        end_date: str,
        kind: TaskKind,
        priority: int,
        chunk: bool = True,
        allow_overlap: bool = False,
    ) -> Optional[int]:
        """
        Enqueue a task into the backfill_queue.

        Uses INSERT OR IGNORE to avoid duplicates (thanks to UNIQUE constraint).
        Automatically chunks large date ranges to prevent head-of-line blocking.

        Args:
            dataset: Logical dataset name (equities_eod, equities_minute, etc.)
            symbol: Symbol or None for non-symbol datasets
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            kind: Task kind (BOOTSTRAP, GAP, FORWARD, QC_PROBE)
            priority: Priority (lower = more important)
            chunk: If True, automatically chunk large date ranges (default True)

        Returns:
            Task ID of first chunk if inserted, None if already exists
        """
        # Parse dates
        start_dt = date.fromisoformat(start_date) if isinstance(start_date, str) else start_date
        end_dt = date.fromisoformat(end_date) if isinstance(end_date, str) else end_date

        # Skip if overlapping pending/leased task already exists (used by gap audit)
        if not allow_overlap and self.task_exists_overlapping(
            dataset=dataset,
            symbol=symbol,
            start_date=start_dt,
            end_date=end_dt,
        ):
            logger.debug(f"Skipped enqueue (overlap pending/leased) for {dataset}/{symbol} [{start_dt}..{end_dt}]")
            return None

        # Get max days for this dataset
        max_days = MAX_TASK_DAYS.get(dataset, MAX_TASK_DAYS["default"])

        # Calculate date range
        total_days = (end_dt - start_dt).days + 1

        # If small enough or chunking disabled, insert directly
        if not chunk or total_days <= max_days:
            return self._enqueue_single_task(dataset, symbol, start_date, end_date, kind, priority)

        # Chunk the date range
        first_task_id = None
        chunk_start = start_dt
        chunk_num = 0

        while chunk_start <= end_dt:
            chunk_end = min(chunk_start + timedelta(days=max_days - 1), end_dt)

            task_id = self._enqueue_single_task(
                dataset,
                symbol,
                chunk_start.isoformat(),
                chunk_end.isoformat(),
                kind,
                priority + chunk_num,  # Slightly lower priority for later chunks
            )

            if first_task_id is None and task_id is not None:
                first_task_id = task_id

            chunk_start = chunk_end + timedelta(days=1)
            chunk_num += 1

        if chunk_num > 1:
            logger.info(f"Chunked {dataset}/{symbol} [{start_date}..{end_date}] into {chunk_num} tasks (max {max_days} days each)")

        return first_task_id

    def _enqueue_single_task(
        self,
        dataset: str,
        symbol: Optional[str],
        start_date: str,
        end_date: str,
        kind: TaskKind,
        priority: int,
    ) -> Optional[int]:
        """Enqueue a single task without chunking."""
        now = datetime.now(timezone.utc).isoformat()

        with self.transaction() as conn:
            cursor = conn.execute("""
                INSERT OR IGNORE INTO backfill_queue
                (dataset, symbol, start_date, end_date, kind, priority, status, attempts, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, 'PENDING', 0, ?, ?)
            """, (dataset, symbol, start_date, end_date, kind.value, priority, now, now))

            if cursor.rowcount > 0:
                task_id = cursor.lastrowid
                logger.debug(f"Enqueued task {task_id}: {kind.value} {dataset}/{symbol} [{start_date}..{end_date}]")
                return task_id
            return None

    def task_exists_overlapping(
        self,
        dataset: str,
        symbol: Optional[str],
        start_date,
        end_date,
        statuses: tuple[str, ...] = ("PENDING", "LEASED"),
    ) -> bool:
        """
        Check if there is any pending/leased task that overlaps the given date range,
        regardless of kind.
        """
        if not statuses:
            return False

        # Normalize dates to ISO strings for lexicographic comparison in SQLite
        if hasattr(start_date, "isoformat"):
            start_date = start_date.isoformat()
        if hasattr(end_date, "isoformat"):
            end_date = end_date.isoformat()

        placeholders = ",".join(["?"] * len(statuses))
        # Order of params matches SQL: dataset, symbol, start_date, end_date, statuses...
        params: list = [dataset, symbol, start_date, end_date, *statuses]

        with self.transaction() as conn:
            row = conn.execute(
                f"""
                SELECT 1
                FROM backfill_queue
                WHERE dataset = ?
                  AND symbol IS ?
                  AND NOT (end_date < ? OR start_date > ?)
                  AND status IN ({placeholders})
                LIMIT 1
                """,
                params,
            ).fetchone()
            return row is not None

    def release_task(self, task_id: int) -> None:
        """Release a leased task back to PENDING."""
        now = datetime.now(timezone.utc).isoformat()
        with self.transaction() as conn:
            conn.execute(
                """
                UPDATE backfill_queue
                SET status = 'PENDING',
                    lease_owner = NULL,
                    lease_expires_at = NULL,
                    updated_at = ?
                WHERE id = ?
                """,
                (now, task_id),
            )

    def lease_next_task(
        self,
        node_id: str,
        lease_ttl_seconds: int = 300,
        now: Optional[datetime] = None,
    ) -> Optional[Task]:
        """
        Lease the next available task.

        Finds the oldest PENDING task by (priority, created_at) that:
        - Has status='PENDING' OR has an expired lease
        - Has next_not_before <= now (or NULL)

        Args:
            node_id: This node's identifier for the lease
            lease_ttl_seconds: How long to hold the lease (default 5 min)
            now: Current time (for testing); defaults to utcnow()

        Returns:
            Task if one was leased, None otherwise
        """
        if now is None:
            now = datetime.now(timezone.utc)

        lease_expires = now + timedelta(seconds=lease_ttl_seconds)
        now_str = now.isoformat()
        lease_expires_str = lease_expires.isoformat()

        with self._lock:
            with self.transaction() as conn:
                # Find next eligible task
                # Treat expired leases as PENDING
                row = conn.execute("""
                    SELECT *
                    FROM backfill_queue
                    WHERE (status = 'PENDING' OR (status = 'LEASED' AND lease_expires_at < ?))
                      AND (next_not_before IS NULL OR next_not_before <= ?)
                    ORDER BY priority ASC, created_at ASC
                    LIMIT 1
                """, (now_str, now_str)).fetchone()

                if row is None:
                    return None

                task_id = row['id']

                # Attempt to lease it
                cursor = conn.execute("""
                    UPDATE backfill_queue
                    SET status = 'LEASED',
                        lease_owner = ?,
                        lease_expires_at = ?,
                        updated_at = ?
                    WHERE id = ?
                      AND (status = 'PENDING' OR (status = 'LEASED' AND lease_expires_at < ?))
                """, (node_id, lease_expires_str, now_str, task_id, now_str))

                if cursor.rowcount == 0:
                    # Race condition - someone else got it
                    return None

                # Re-fetch to confirm
                row = conn.execute(
                    "SELECT * FROM backfill_queue WHERE id = ?",
                    (task_id,)
                ).fetchone()

                if row is None or row['lease_owner'] != node_id:
                    return None

                task = self._row_to_task(row)
                logger.debug(f"Leased task {task.id}: {task.kind.value} {task.dataset}/{task.symbol}")
                return task

    def lease_next_task_for_vendor(
        self,
        vendor: str,
        datasets: list[str],
        node_id: str,
        lease_ttl_seconds: int = 300,
        now: Optional[datetime] = None,
    ) -> Optional[Task]:
        """
        Lease the next available task for a specific vendor.

        Only considers tasks for datasets that this vendor can handle.

        Args:
            vendor: Vendor name (for logging)
            datasets: List of datasets this vendor can handle
            node_id: This node's identifier for the lease
            lease_ttl_seconds: How long to hold the lease (default 5 min)
            now: Current time (for testing); defaults to utcnow()

        Returns:
            Task if one was leased, None otherwise
        """
        if not datasets:
            return None

        if now is None:
            now = datetime.now(timezone.utc)

        lease_expires = now + timedelta(seconds=lease_ttl_seconds)
        now_str = now.isoformat()
        lease_expires_str = lease_expires.isoformat()

        # Build dataset IN clause
        placeholders = ",".join("?" * len(datasets))

        # Vendor-specific eligibility filters (avoid leasing tasks Massive cannot serve)
        extra_where = ""
        extra_params: tuple[str, ...] = ()
        if vendor == "massive":
            earliest, latest = massive_window(now.date())
            # Skip any tasks outside Massive free-tier window: older than 2y or newer than T+1
            extra_where = " AND start_date >= ? AND end_date >= ? AND start_date <= ? AND end_date <= ?"
            extra_params = (
                earliest.isoformat(),
                earliest.isoformat(),
                latest.isoformat(),
                latest.isoformat(),
            )

        import time as _time
        t_start = _time.time()

        with self._lock:
            t_lock_acquired = _time.time()
            lock_wait_ms = (t_lock_acquired - t_start) * 1000

            with self.transaction() as conn:
                # Find next eligible task for this vendor's datasets
                t_query_start = _time.time()
                row = conn.execute(f"""
                    SELECT *
                    FROM backfill_queue
                    WHERE dataset IN ({placeholders})
                      AND (status = 'PENDING' OR (status = 'LEASED' AND lease_expires_at < ?))
                      AND (next_not_before IS NULL OR next_not_before <= ?)
                      {extra_where}
                    ORDER BY priority ASC, created_at ASC
                    LIMIT 1
                """, (*datasets, now_str, now_str, *extra_params)).fetchone()
                t_query_end = _time.time()
                query_ms = (t_query_end - t_query_start) * 1000

                if row is None:
                    total_ms = (_time.time() - t_start) * 1000
                    if lock_wait_ms > 20:  # Log if lock wait was significant
                        logger.info(f"[DB-LEASE] {vendor}: no task, total={total_ms:.0f}ms (lock_wait={lock_wait_ms:.0f}ms, query={query_ms:.0f}ms)")
                    return None

                task_id = row['id']

                # Attempt to lease it
                t_update_start = _time.time()
                cursor = conn.execute("""
                    UPDATE backfill_queue
                    SET status = 'LEASED',
                        lease_owner = ?,
                        lease_expires_at = ?,
                        updated_at = ?
                    WHERE id = ?
                      AND (status = 'PENDING' OR (status = 'LEASED' AND lease_expires_at < ?))
                """, (node_id, lease_expires_str, now_str, task_id, now_str))
                t_update_end = _time.time()
                update_ms = (t_update_end - t_update_start) * 1000

                if cursor.rowcount == 0:
                    # Race condition - someone else got it
                    return None

                # Re-fetch to confirm
                row = conn.execute(
                    "SELECT * FROM backfill_queue WHERE id = ?",
                    (task_id,)
                ).fetchone()

                if row is None or row['lease_owner'] != node_id:
                    return None

                task = self._row_to_task(row)
                total_ms = (_time.time() - t_start) * 1000
                logger.info(f"[DB-LEASE] {vendor}: task={task.id}, total={total_ms:.0f}ms (lock_wait={lock_wait_ms:.0f}ms, query={query_ms:.0f}ms, update={update_ms:.0f}ms)")
                return task

    def mark_task_done(self, task_id: int, conn: Optional[sqlite3.Connection] = None) -> bool:
        """
        Mark a task as DONE.

        Args:
            task_id: Task ID

        Returns:
            True if updated, False if task not found
        """
        now = datetime.now(timezone.utc).isoformat()

        if conn is None:
            context = self.transaction()
        else:
            context = nullcontext(conn)

        with context as _conn:
            cursor = _conn.execute("""
                UPDATE backfill_queue
                SET status = 'DONE',
                    lease_owner = NULL,
                    lease_expires_at = NULL,
                    updated_at = ?
                WHERE id = ?
            """, (now, task_id))

            if cursor.rowcount > 0:
                logger.debug(f"Marked task {task_id} as DONE")
                return True
            return False

    def mark_task_failed(
        self,
        task_id: int,
        error: str,
        backoff_until: Optional[datetime] = None,
        max_attempts: int = 5,
    ) -> bool:
        """
        Mark a task as failed or set up for retry.

        If attempts < max_attempts, task stays PENDING with backoff.
        If attempts >= max_attempts, task is marked FAILED.

        Args:
            task_id: Task ID
            error: Error message
            backoff_until: When to retry (None for immediate retry)
            max_attempts: Maximum retry attempts before marking FAILED

        Returns:
            True if updated, False if task not found
        """
        now = datetime.now(timezone.utc)
        now_str = now.isoformat()
        backoff_str = backoff_until.isoformat() if backoff_until else None

        with self.transaction() as conn:
            # Get current attempts
            row = conn.execute(
                "SELECT attempts FROM backfill_queue WHERE id = ?",
                (task_id,)
            ).fetchone()

            if row is None:
                return False

            new_attempts = row['attempts'] + 1

            if new_attempts >= max_attempts:
                # Mark as FAILED
                conn.execute("""
                    UPDATE backfill_queue
                    SET status = 'FAILED',
                        attempts = ?,
                        last_error = ?,
                        lease_owner = NULL,
                        lease_expires_at = NULL,
                        updated_at = ?
                    WHERE id = ?
                """, (new_attempts, error[:2000], now_str, task_id))
                logger.warning(f"Task {task_id} marked FAILED after {new_attempts} attempts: {error[:100]}")
            else:
                # Reset to PENDING with backoff
                conn.execute("""
                    UPDATE backfill_queue
                    SET status = 'PENDING',
                        attempts = ?,
                        last_error = ?,
                        next_not_before = ?,
                        lease_owner = NULL,
                        lease_expires_at = NULL,
                        updated_at = ?
                    WHERE id = ?
                """, (new_attempts, error[:2000], backoff_str, now_str, task_id))
                logger.debug(f"Task {task_id} will retry (attempt {new_attempts}), backoff until {backoff_str}")

            return True

    def defer_task_until(self, task_id: int, backoff_until: datetime) -> bool:
        """
        Defer a task without consuming an attempt.

        Sets next_not_before and clears the lease, leaving attempts unchanged.
        """
        now_str = datetime.now(timezone.utc).isoformat()
        backoff_str = backoff_until.isoformat()

        with self.transaction() as conn:
            cursor = conn.execute(
                """
                UPDATE backfill_queue
                SET status = 'PENDING',
                    next_not_before = ?,
                    lease_owner = NULL,
                    lease_expires_at = NULL,
                    updated_at = ?
                WHERE id = ?
                """,
                (backoff_str, now_str, task_id),
            )
            if cursor.rowcount > 0:
                logger.debug(f"Deferred task {task_id} until {backoff_str} (no attempt increment)")
                return True
            return False

    def get_queue_stats(self) -> dict:
        """Get queue statistics."""
        conn = self._get_connection()

        stats = {"total": 0, "by_status": {}, "by_kind": {}}

        for row in conn.execute("""
            SELECT status, COUNT(*) as cnt FROM backfill_queue GROUP BY status
        """):
            stats["by_status"][row['status']] = row['cnt']
            stats["total"] += row['cnt']

        for row in conn.execute("""
            SELECT kind, COUNT(*) as cnt FROM backfill_queue WHERE status IN ('PENDING', 'LEASED') GROUP BY kind
        """):
            stats["by_kind"][row['kind']] = row['cnt']

        return stats

    def get_pending_count(self) -> int:
        """Get count of PENDING tasks."""
        conn = self._get_connection()
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM backfill_queue WHERE status = 'PENDING'"
        ).fetchone()
        return row['cnt'] if row else 0

    def get_failed_count(self) -> int:
        """Get count of FAILED tasks."""
        conn = self._get_connection()
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM backfill_queue WHERE status = 'FAILED'"
        ).fetchone()
        return row['cnt'] if row else 0

    # -------------------------------------------------------------------------
    # Partition status operations
    # -------------------------------------------------------------------------

    def upsert_partition_status(
        self,
        source_name: str,
        table_name: str,
        symbol: Optional[str],
        dt: str,
        status: PartitionStatus,
        qc_score: float,
        row_count: int,
        expected_rows: int,
        qc_code: Optional[str] = None,
        notes: Optional[str] = None,
        fetch_params: Optional[dict] = None,
        conn: Optional[sqlite3.Connection] = None,
    ) -> None:
        """
        Upsert a partition status record.

        Preserves first_observed_at for existing records.

        Args:
            fetch_params: API parameters used for this fetch (e.g., {"feed": "sip", "timeframe": "1Min"}).
                          Enables selective re-fetch when config changes.
        """
        now = datetime.now(timezone.utc).isoformat()
        fetch_params_json = json.dumps(fetch_params) if fetch_params else None

        # Allow caller to supply an open transaction to keep task updates atomic
        if conn is None:
            context = self.transaction()
        else:
            context = nullcontext(conn)

        with context as _conn:
            # Check if exists
            existing = _conn.execute("""
                SELECT first_observed_at FROM partition_status
                WHERE source_name = ? AND table_name = ? AND symbol IS ? AND dt = ?
            """, (source_name, table_name, symbol, dt)).fetchone()

            if existing:
                # Update
                _conn.execute("""
                    UPDATE partition_status
                    SET status = ?, qc_score = ?, row_count = ?, expected_rows = ?,
                        qc_code = ?, last_observed_at = ?, notes = ?, fetch_params = ?
                    WHERE source_name = ? AND table_name = ? AND symbol IS ? AND dt = ?
                """, (
                    status.value, qc_score, row_count, expected_rows, qc_code, now, notes,
                    fetch_params_json, source_name, table_name, symbol, dt
                ))
            else:
                # Insert
                _conn.execute("""
                    INSERT INTO partition_status
                    (source_name, table_name, symbol, dt, status, qc_score, row_count,
                     expected_rows, qc_code, first_observed_at, last_observed_at, notes, fetch_params)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    source_name, table_name, symbol, dt, status.value, qc_score, row_count,
                    expected_rows, qc_code, now, now, notes, fetch_params_json
                ))

    def get_green_coverage(
        self,
        table_name: str,
        symbols: Optional[list[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        source_name: Optional[str] = None,
    ) -> float:
        """
        Calculate GREEN coverage for a table over a date range.

        Args:
            table_name: Table name to query
            symbols: Optional list of symbols to filter by
            start_date: Start date (YYYY-MM-DD or date object)
            end_date: End date (YYYY-MM-DD or date object)
            source_name: Optional source filter

        Returns:
            GREEN fraction (0.0 to 1.0)
        """
        conn = self._get_connection()

        # Convert dates to strings if needed
        if hasattr(start_date, 'isoformat'):
            start_date = start_date.isoformat()
        if hasattr(end_date, 'isoformat'):
            end_date = end_date.isoformat()

        # Build query dynamically
        conditions = ["table_name = ?"]
        params: list = [table_name]

        if start_date:
            conditions.append("dt >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("dt <= ?")
            params.append(end_date)
        if source_name:
            conditions.append("source_name = ?")
            params.append(source_name)
        if symbols:
            placeholders = ",".join("?" * len(symbols))
            conditions.append(f"symbol IN ({placeholders})")
            params.extend(symbols)

        where_clause = " AND ".join(conditions)

        counts = {"GREEN": 0, "AMBER": 0, "RED": 0}
        total = 0

        for row in conn.execute(f"""
            SELECT status, COUNT(*) as cnt
            FROM partition_status
            WHERE {where_clause}
            GROUP BY status
        """, params):
            counts[row['status']] = row['cnt']
            total += row['cnt']

        return counts["GREEN"] / total if total > 0 else 0.0

    def get_partition_status_batch(
        self,
        table_name: str,
        symbols: list[str],
        start_date,
        end_date,
    ) -> list[dict]:
        """
        Get partition status for multiple symbols in a date range.

        Args:
            table_name: Table name to query
            symbols: List of symbols
            start_date: Start date (YYYY-MM-DD or date object)
            end_date: End date (YYYY-MM-DD or date object)

        Returns:
            List of dicts with symbol, dt, status, etc.
        """
        conn = self._get_connection()

        # Convert dates to strings if needed
        if hasattr(start_date, 'isoformat'):
            start_date = start_date.isoformat()
        if hasattr(end_date, 'isoformat'):
            end_date = end_date.isoformat()

        if not symbols:
            return []

        placeholders = ",".join("?" * len(symbols))
        params = [table_name, start_date, end_date] + symbols

        rows = conn.execute(f"""
            SELECT symbol, dt, status, qc_score, row_count, expected_rows, qc_code, source_name
            FROM partition_status
            WHERE table_name = ? AND dt >= ? AND dt <= ?
              AND symbol IN ({placeholders})
        """, params).fetchall()

        return [dict(row) for row in rows]

    def get_latest_partition_status(
        self,
        table_name: str,
        symbol: Optional[str],
        dt: str,
        status: Optional[PartitionStatus] = None,
    ) -> Optional[dict]:
        """
        Get the latest partition_status record for a dataset/symbol/date.

        Args:
            table_name: Dataset name
            symbol: Symbol or None
            dt: Date (YYYY-MM-DD)
            status: Optional status filter (GREEN/AMBER/RED)

        Returns:
            Dict row or None
        """
        conn = self._get_connection()

        conditions = ["table_name = ?", "symbol IS ?", "dt = ?"]
        params: list = [table_name, symbol, dt]
        if status is not None:
            conditions.append("status = ?")
            params.append(status.value)

        where_clause = " AND ".join(conditions)

        row = conn.execute(f"""
            SELECT * FROM partition_status
            WHERE {where_clause}
            ORDER BY last_observed_at DESC
            LIMIT 1
        """, params).fetchone()

        return dict(row) if row else None

    def upsert_qc_probe(
        self,
        dataset: str,
        symbol: Optional[str],
        dt: str,
        primary_vendor: Optional[str],
        secondary_vendor: Optional[str],
        status: str,
        qc_code: Optional[str] = None,
        details: Optional[dict] = None,
        conn: Optional[sqlite3.Connection] = None,
    ) -> None:
        """Upsert a QC probe result."""
        now = datetime.now(timezone.utc).isoformat()
        details_json = json.dumps(details) if details else None

        if conn is None:
            context = self.transaction()
        else:
            context = nullcontext(conn)

        with context as _conn:
            _conn.execute(
                """
                INSERT INTO qc_probes
                (dataset, symbol, dt, primary_vendor, secondary_vendor, status, qc_code, details, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(dataset, symbol, dt, primary_vendor, secondary_vendor)
                DO UPDATE SET status = excluded.status,
                              qc_code = excluded.qc_code,
                              details = excluded.details,
                              updated_at = excluded.updated_at
                """,
                (
                    dataset,
                    symbol,
                    dt,
                    primary_vendor,
                    secondary_vendor,
                    status,
                    qc_code,
                    details_json,
                    now,
                    now,
                ),
            )

    def prune_failed_tasks(self, before: datetime) -> int:
        """
        Delete failed tasks older than the given timestamp.

        Args:
            before: Delete tasks created before this time

        Returns:
            Number of tasks deleted
        """
        before_str = before.isoformat()

        with self.transaction() as conn:
            cursor = conn.execute("""
                DELETE FROM backfill_queue
                WHERE status = 'FAILED' AND created_at < ?
            """, (before_str,))

            deleted = cursor.rowcount
            if deleted > 0:
                logger.debug(f"Pruned {deleted} failed tasks created before {before_str}")

            return deleted

    def get_gaps(
        self,
        table_name: str,
        start_date: str,
        end_date: str,
        statuses: Optional[list[PartitionStatus]] = None,
    ) -> list[PartitionRecord]:
        """
        Get partition records with specified statuses (gaps).

        Args:
            table_name: Table name to query
            start_date: Start date
            end_date: End date
            statuses: Statuses to include (default: [RED, AMBER])

        Returns:
            List of PartitionRecord objects
        """
        if statuses is None:
            statuses = [PartitionStatus.RED, PartitionStatus.AMBER]

        status_list = ",".join(f"'{s.value}'" for s in statuses)

        conn = self._get_connection()
        rows = conn.execute(f"""
            SELECT * FROM partition_status
            WHERE table_name = ? AND dt >= ? AND dt <= ?
              AND status IN ({status_list})
            ORDER BY dt DESC
        """, (table_name, start_date, end_date)).fetchall()

        return [self._row_to_partition(row) for row in rows]

    def get_partitions_by_status(
        self,
        table_name: str,
        status: PartitionStatus,
        limit: int = 1000,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> list[PartitionRecord]:
        """
        Get partitions with a given status for QC review.

        Used by maintenance QC to find partitions that need row-count validation.

        Args:
            table_name: Table name to query
            status: Status to filter by (GREEN, AMBER, RED)
            limit: Maximum number of records to return
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            List of PartitionRecord objects
        """
        conn = self._get_connection()

        # Build query dynamically
        conditions = ["table_name = ?", "status = ?"]
        params: list = [table_name, status.value]

        if start_date:
            conditions.append("dt >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("dt <= ?")
            params.append(end_date)

        where_clause = " AND ".join(conditions)
        params.append(limit)

        rows = conn.execute(f"""
            SELECT * FROM partition_status
            WHERE {where_clause}
            ORDER BY dt DESC
            LIMIT ?
        """, params).fetchall()

        return [self._row_to_partition(row) for row in rows]

    def count_partitions_by_status(
        self,
        table_name: str,
        status: PartitionStatus,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> int:
        """
        Count partitions with a given status.

        Args:
            table_name: Table name to query
            status: Status to filter by
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            Count of matching partitions
        """
        conn = self._get_connection()

        conditions = ["table_name = ?", "status = ?"]
        params: list = [table_name, status.value]

        if start_date:
            conditions.append("dt >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("dt <= ?")
            params.append(end_date)

        where_clause = " AND ".join(conditions)

        row = conn.execute(f"""
            SELECT COUNT(*) as cnt FROM partition_status
            WHERE {where_clause}
        """, params).fetchone()

        return row['cnt'] if row else 0

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _parse_datetime(val) -> Optional[datetime]:
        """Parse a datetime from SQLite (may be str or datetime due to converters)."""
        if val is None:
            return None
        if isinstance(val, datetime):
            return val
        return datetime.fromisoformat(val)

    def _row_to_task(self, row: sqlite3.Row) -> Task:
        """Convert a database row to a Task object."""
        return Task(
            id=row['id'],
            dataset=row['dataset'],
            symbol=row['symbol'],
            start_date=row['start_date'],
            end_date=row['end_date'],
            kind=TaskKind(row['kind']),
            priority=row['priority'],
            status=TaskStatus(row['status']),
            attempts=row['attempts'],
            lease_owner=row['lease_owner'],
            lease_expires_at=self._parse_datetime(row['lease_expires_at']),
            next_not_before=self._parse_datetime(row['next_not_before']),
            last_error=row['last_error'],
            created_at=self._parse_datetime(row['created_at']),
            updated_at=self._parse_datetime(row['updated_at']),
        )

    def _row_to_partition(self, row: sqlite3.Row) -> PartitionRecord:
        """Convert a database row to a PartitionRecord object."""
        return PartitionRecord(
            source_name=row['source_name'],
            table_name=row['table_name'],
            symbol=row['symbol'],
            dt=row['dt'],
            status=PartitionStatus(row['status']),
            qc_score=row['qc_score'] or 0.0,
            row_count=row['row_count'] or 0,
            expected_rows=row['expected_rows'] or 0,
            qc_code=row['qc_code'],
            first_observed_at=self._parse_datetime(row['first_observed_at']),
            last_observed_at=self._parse_datetime(row['last_observed_at']),
        )

    def close(self) -> None:
        """Close the database connection."""
        if hasattr(self._local, 'conn') and self._local.conn:
            self._local.conn.close()
            self._local.conn = None


# Module-level singleton for convenience
_default_db: Optional[NodeDB] = None


def get_db(db_path: Optional[Path] = None) -> NodeDB:
    """
    Get the default NodeDB instance.

    Creates and initializes the database if needed.
    """
    global _default_db

    if _default_db is None:
        _default_db = NodeDB(db_path)
        _default_db.init_db()

    return _default_db


def reset_db() -> None:
    """Reset the default database instance (for testing)."""
    global _default_db
    if _default_db:
        _default_db.close()
        _default_db = None
