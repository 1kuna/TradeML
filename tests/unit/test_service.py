from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from trademl.calendars.exchange import ExchangeCalendarStore
from trademl.data_node.auditor import PartitionAuditor
from trademl.data_node.curator import Curator
from trademl.data_node.db import DataNodeDB
from trademl.data_node.service import DataNodePaths, DataNodeService


class _NoopConnector:
    vendor_name = "alpaca"

    def fetch(self, dataset: str, symbols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        return pd.DataFrame()


class _ClusterCoordinatorStub:
    def __init__(self) -> None:
        self._lease_calls: list[str] = []

    def heartbeat_worker(self) -> dict[str, str]:
        return {"worker_id": "worker-a"}

    def sync_shard_leases(self):
        return []

    def acquire_singleton(self, task_name: str, bucket_key: str) -> bool:
        self._lease_calls.append(f"{task_name}:{bucket_key}")
        return task_name == "backfill"

    def mark_singleton_success(self, task_name: str, bucket_key: str, metadata: dict | None = None) -> None:
        self._lease_calls.append(f"success:{task_name}:{bucket_key}:{metadata or {}}")


def test_run_forever_executes_collection_cycle_and_stops(tmp_path: Path) -> None:
    db = DataNodeDB(tmp_path / "control" / "node.sqlite")
    service = DataNodeService(
        db=db,
        connectors={"alpaca": _NoopConnector(), "fred": _NoopConnector()},
        auditor=PartitionAuditor(db=db, calendar_store=ExchangeCalendarStore(root=tmp_path / "reference" / "calendars")),
        curator=Curator(),
        paths=DataNodePaths(root=tmp_path),
    )

    calls: list[tuple[str, tuple, dict]] = []

    def _record(name: str):
        def inner(*args, **kwargs):
            calls.append((name, args, kwargs))
            if name == "run_cycle":
                service.stop()
            if name == "sync_partition_status":
                return tmp_path / "data" / "qc" / "partition_status.parquet"
            return []

        return inner

    service.run_cycle = _record("run_cycle")  # type: ignore[method-assign]
    service.collect_macro_data = _record("collect_macro_data")  # type: ignore[method-assign]
    service.collect_reference_data = _record("collect_reference_data")  # type: ignore[method-assign]
    service.run_cross_vendor_price_checks = _record("run_cross_vendor_price_checks")  # type: ignore[method-assign]
    service.sync_partition_status = _record("sync_partition_status")  # type: ignore[method-assign]

    timestamps = iter(
        [
            datetime.fromisoformat("2026-03-31T20:35:00+00:00"),
            datetime.fromisoformat("2026-03-31T20:35:00+00:00"),
        ]
    )
    service.run_forever(
        symbols=["AAPL", "MSFT"],
        exchange="XNYS",
        now_fn=lambda: next(timestamps),
        sleep_fn=lambda _seconds: None,
        poll_seconds=0.0,
        macro_series_ids=["DGS10"],
        reference_jobs=[{"source": "alpaca", "dataset": "equities_eod", "symbols": ["AAPL"], "start_date": "2026-03-31", "end_date": "2026-03-31", "output_name": "noop"}],
        price_check_symbols=["AAPL"],
    )

    executed = [name for name, _args, _kwargs in calls]
    assert "run_cycle" in executed
    assert "collect_macro_data" in executed
    assert "collect_reference_data" in executed
    assert "run_cross_vendor_price_checks" in executed


def test_run_cluster_forever_drains_backlog_even_outside_maintenance_window(tmp_path: Path) -> None:
    db = DataNodeDB(tmp_path / "control" / "node.sqlite")
    db.enqueue_task("equities_eod", "AAPL", "2026-03-31", "2026-03-31", "GAP", 1)
    service = DataNodeService(
        db=db,
        connectors={"alpaca": _NoopConnector()},
        auditor=PartitionAuditor(db=db, calendar_store=ExchangeCalendarStore(root=tmp_path / "reference" / "calendars")),
        curator=Curator(),
        paths=DataNodePaths(root=tmp_path),
    )
    coordinator = _ClusterCoordinatorStub()
    calls: list[str] = []

    def process_backfill_queue() -> list[str]:
        calls.append("process_backfill_queue")
        service.stop()
        return []

    service.process_backfill_queue = process_backfill_queue  # type: ignore[method-assign]
    service.sync_partition_status = lambda: tmp_path / "data" / "qc" / "partition_status.parquet"  # type: ignore[method-assign]

    service.run_cluster_forever(
        coordinator=coordinator,  # type: ignore[arg-type]
        symbols=["AAPL"],
        exchange="XNYS",
        collection_time_et="16:30",
        maintenance_hour_local=23,
        poll_seconds=0.0,
        now_fn=lambda: datetime.fromisoformat("2026-03-31T20:35:00+00:00"),
        sleep_fn=lambda _seconds: None,
    )

    assert calls == ["process_backfill_queue"]
