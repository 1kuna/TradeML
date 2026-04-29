"""SQLite helpers for the Pi data-node task queue and partition ledger."""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from trademl.data_node.db_models import (
    UTC,
    CanonicalUnit,
    PlannerTask,
    PlannerTaskProgress,
    RawPartitionManifest,
    VendorAttempt,
    VendorLaneHealth,
    utc_now,
)
from trademl.data_node.db_stores import (
    CanonicalLedgerStore,
    PlannerTaskStore,
    RuntimeStore,
    VendorAttemptStore,
)

__all__ = [
    "UTC",
    "CanonicalUnit",
    "DataNodeDB",
    "PlannerTask",
    "PlannerTaskProgress",
    "RawPartitionManifest",
    "VendorAttempt",
    "VendorLaneHealth",
    "utc_now",
]


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
        def clock() -> datetime:
            return utc_now()

        self.runtime = RuntimeStore(self._connect, clock)
        self.canonical = CanonicalLedgerStore(self._connect, clock)
        self.vendors = VendorAttemptStore(self._connect, clock)
        self.planner = PlannerTaskStore(self._connect, clock)
        self._stores = (self.runtime, self.canonical, self.vendors, self.planner)

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
                CREATE INDEX IF NOT EXISTS idx_planner_tasks_lease_order
                ON planner_tasks(status, priority, created_at, task_key)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_vendor_attempts_updated_lane
                ON vendor_attempts(updated_at, vendor, task_family, planner_group)
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


    def __getattr__(self, name: str) -> Any:
        """Delegate stable facade methods to focused stores."""
        for store in self._stores:
            try:
                return object.__getattribute__(store, name)
            except AttributeError:
                continue
        raise AttributeError(f"{type(self).__name__!s} object has no attribute {name!r}")


def _delegate_store_method(store_attr: str, method_name: str) -> Any:
    def delegated(self: DataNodeDB, *args: Any, **kwargs: Any) -> Any:
        store = object.__getattribute__(self, store_attr)
        method = object.__getattribute__(store, method_name)
        return method(*args, **kwargs)

    delegated.__name__ = method_name
    delegated.__qualname__ = f"{DataNodeDB.__name__}.{method_name}"
    delegated.__doc__ = f"Delegate to {store_attr}.{method_name}."
    return delegated


for _store_attr, _store_type in (
    ("runtime", RuntimeStore),
    ("canonical", CanonicalLedgerStore),
    ("vendors", VendorAttemptStore),
    ("planner", PlannerTaskStore),
):
    for _method_name, _method in _store_type.__dict__.items():
        if _method_name.startswith("_") or not callable(_method):
            continue
        setattr(DataNodeDB, _method_name, _delegate_store_method(_store_attr, _method_name))

del _method, _method_name, _store_attr, _store_type
