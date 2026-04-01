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
