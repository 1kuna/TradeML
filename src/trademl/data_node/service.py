"""Main data-node service loop."""

from __future__ import annotations

import signal
import threading
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from trademl.connectors.base import BaseConnector, ConnectorError
from trademl.data_node.auditor import PartitionAuditor
from trademl.data_node.curator import Curator, CuratorResult
from trademl.data_node.db import DataNodeDB


@dataclass(slots=True)
class DataNodePaths:
    """Filesystem paths for the node."""

    root: Path

    @property
    def raw_equities(self) -> Path:
        return self.root / "data" / "raw" / "equities_bars"

    @property
    def curated_equities(self) -> Path:
        return self.root / "data" / "curated" / "equities_ohlcv_adj"

    @property
    def qc_root(self) -> Path:
        return self.root / "data" / "qc"


class DataNodeService:
    """Collect raw bars, audit completeness, curate data, and sync QC state."""

    def __init__(
        self,
        *,
        db: DataNodeDB,
        connectors: dict[str, BaseConnector],
        auditor: PartitionAuditor,
        curator: Curator,
        paths: DataNodePaths,
        source_name: str = "alpaca",
    ) -> None:
        self.db = db
        self.connectors = connectors
        self.auditor = auditor
        self.curator = curator
        self.paths = paths
        self.source_name = source_name
        self._stop_event = threading.Event()

    def stop(self) -> None:
        """Request a graceful shutdown."""
        self._stop_event.set()

    def install_signal_handlers(self) -> None:
        """Install SIGINT and SIGTERM handlers for graceful shutdown."""
        signal.signal(signal.SIGINT, lambda *_: self.stop())
        signal.signal(signal.SIGTERM, lambda *_: self.stop())

    def _write_raw_partition(self, frame: pd.DataFrame) -> list[str]:
        changed_dates: list[str] = []
        for day, day_frame in frame.groupby("date"):
            day_value = pd.Timestamp(day).strftime("%Y-%m-%d")
            partition = self.paths.raw_equities / f"date={day_value}"
            partition.mkdir(parents=True, exist_ok=True)
            day_frame.to_parquet(partition / "data.parquet", index=False)
            changed_dates.append(day_value)
            self.db.update_partition_status(
                source=self.source_name,
                dataset="equities_eod",
                date=day_value,
                status="GREEN",
                row_count=len(day_frame),
                expected_rows=len(day_frame),
                qc_code="OK",
            )
        return changed_dates

    def collect_forward(self, *, trading_date: str, symbols: list[str]) -> list[str]:
        """Fetch and persist the primary daily bars."""
        frame = self.connectors[self.source_name].fetch("equities_eod", symbols, trading_date, trading_date)
        if frame.empty:
            return []
        return self._write_raw_partition(frame)

    def process_backfill_queue(self) -> list[str]:
        """Process queued backfill or gap tasks until the queue is drained."""
        changed_dates: list[str] = []
        while task := self.db.lease_next_task():
            connector = self.connectors[self.source_name]
            symbols = [task.symbol] if task.symbol else []
            try:
                frame = connector.fetch(task.dataset, symbols, task.start_date, task.end_date)
            except ConnectorError as exc:
                self.db.mark_task_failed(task.id, str(exc), backoff_minutes=30)
                continue
            changed_dates.extend(self._write_raw_partition(frame))
            self.db.mark_task_done(task.id)
        return changed_dates

    def curate_dates(self, corp_actions: pd.DataFrame | None = None) -> CuratorResult:
        """Rebuild curated partitions from the current raw dataset."""
        raw_files = sorted(self.paths.raw_equities.glob("date=*/data.parquet"))
        raw_frame = pd.concat((pd.read_parquet(path) for path in raw_files), ignore_index=True) if raw_files else pd.DataFrame()
        result = self.curator.write_curated(
            raw_bars=raw_frame,
            corp_actions=corp_actions if corp_actions is not None else pd.DataFrame(),
            output_root=self.paths.curated_equities,
        )
        return result

    def sync_partition_status(self) -> Path:
        """Mirror the local SQLite QC ledger to parquet."""
        self.paths.qc_root.mkdir(parents=True, exist_ok=True)
        output = self.paths.qc_root / "partition_status.parquet"
        rows = [dict(row) for row in self.db.fetch_partition_status()]
        pd.DataFrame(rows).to_parquet(output, index=False)
        return output

    def run_cycle(
        self,
        *,
        trading_date: str,
        symbols: list[str],
        exchange: str,
        audit_start: str,
        audit_end: str,
        corp_actions: pd.DataFrame | None = None,
    ) -> dict[str, object]:
        """Run a single deterministic collection cycle."""
        forward_dates = self.collect_forward(trading_date=trading_date, symbols=symbols)
        audit_result = self.auditor.audit_range(
            exchange=exchange,
            source=self.source_name,
            dataset="equities_eod",
            start_date=audit_start,
            end_date=audit_end,
            expected_rows=len(symbols),
        )
        backfill_dates = self.process_backfill_queue()
        curated = self.curate_dates(corp_actions=corp_actions)
        qc_path = self.sync_partition_status()
        return {
            "forward_dates": forward_dates,
            "backfill_dates": backfill_dates,
            "audit_result": audit_result,
            "curated_rows": len(curated.frame),
            "qc_path": qc_path,
        }
