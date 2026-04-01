"""Partition completeness auditing against the exchange calendar."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date as date_type

from trademl.calendars.exchange import ExchangeCalendarStore, get_trading_days
from trademl.data_node.db import DataNodeDB


@dataclass(slots=True)
class AuditResult:
    """Summary of an audit pass."""

    missing_dates: list[date_type]
    amber_dates: list[date_type]
    green_dates: list[date_type]


class PartitionAuditor:
    """Compare expected exchange sessions to the partition ledger."""

    def __init__(self, db: DataNodeDB, calendar_store: ExchangeCalendarStore) -> None:
        self.db = db
        self.calendar_store = calendar_store

    def audit_range(
        self,
        *,
        exchange: str,
        source: str,
        dataset: str,
        start_date: str,
        end_date: str,
        expected_rows: int,
        gap_priority: int = 1,
    ) -> AuditResult:
        """Audit the requested window and enqueue gap tasks for missing trading sessions."""
        missing_dates: list[date_type] = []
        amber_dates: list[date_type] = []
        green_dates: list[date_type] = []

        frame = self.calendar_store.build_calendar_frame(exchange=exchange, start=start_date, end=end_date)
        trading_days = set(get_trading_days(exchange, start_date, end_date))
        statuses = {
            (row["source"], row["dataset"], row["date"]): row for row in self.db.fetch_partition_status()
        }

        for row in frame.itertuples(index=False):
            key = (source, dataset, row.date.isoformat())
            existing = statuses.get(key)
            if row.date not in trading_days:
                self.db.update_partition_status(
                    source=source,
                    dataset=dataset,
                    date=row.date.isoformat(),
                    status="GREEN",
                    row_count=0,
                    expected_rows=0,
                    qc_code="NO_SESSION",
                )
                green_dates.append(row.date)
                continue

            if existing is None:
                self.db.update_partition_status(
                    source=source,
                    dataset=dataset,
                    date=row.date.isoformat(),
                    status="RED",
                    row_count=None,
                    expected_rows=expected_rows,
                    qc_code="MISSING",
                )
                try:
                    self.db.enqueue_task(
                        dataset=dataset,
                        symbol=None,
                        start_date=row.date.isoformat(),
                        end_date=row.date.isoformat(),
                        kind="GAP",
                        priority=gap_priority,
                    )
                except sqlite3.IntegrityError:
                    pass
                missing_dates.append(row.date)
                continue

            row_count = existing["row_count"] or 0
            if expected_rows and row_count < expected_rows:
                self.db.update_partition_status(
                    source=source,
                    dataset=dataset,
                    date=row.date.isoformat(),
                    status="AMBER",
                    row_count=row_count,
                    expected_rows=expected_rows,
                    qc_code="LOW_ROW_COUNT",
                )
                amber_dates.append(row.date)
            else:
                self.db.update_partition_status(
                    source=source,
                    dataset=dataset,
                    date=row.date.isoformat(),
                    status="GREEN",
                    row_count=row_count,
                    expected_rows=expected_rows,
                    qc_code="OK",
                )
                green_dates.append(row.date)

        return AuditResult(missing_dates=missing_dates, amber_dates=amber_dates, green_dates=green_dates)
