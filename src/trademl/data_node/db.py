"""SQLite helpers for the Pi data-node task queue and partition ledger."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator


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


class DataNodeDB:
    """Small transactional wrapper around the local SQLite state store."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
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
