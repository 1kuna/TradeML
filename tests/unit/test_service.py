from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from trademl.calendars.exchange import ExchangeCalendarStore
from trademl.connectors.base import TemporaryConnectorError
from trademl.data_node.auditor import PartitionAuditor
from trademl.data_node.curator import Curator
from trademl.data_node.db import DataNodeDB
from trademl.data_node.service import DataNodePaths, DataNodeService


class _NoopConnector:
    vendor_name = "alpaca"

    def fetch(self, dataset: str, symbols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        return pd.DataFrame()


class _BudgetFailConnector:
    vendor_name = "massive"

    def fetch(self, dataset: str, symbols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        raise TemporaryConnectorError("budget exhausted for vendor=massive")


class _PartialReferenceConnector:
    vendor_name = "alpha_vantage"

    def fetch(self, dataset: str, symbols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        symbol = symbols[0]
        if symbol == "MSFT":
            raise TemporaryConnectorError("budget exhausted for vendor=alpha_vantage")
        return pd.DataFrame(
            [
                {
                    "symbol": symbol,
                    "event_type": "dividend",
                    "ex_date": start_date,
                    "ratio": pd.NA,
                    "amount": 1.0,
                    "source": "alpha_vantage",
                }
            ]
        )


class _ListingConnector:
    vendor_name = "alpha_vantage"

    def fetch(self, dataset: str, symbols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        if dataset == "listings":
            return pd.DataFrame(
                [
                    {
                        "symbol": "AAPL",
                        "name": "Apple Inc.",
                        "exchange": "NASDAQ",
                        "assetType": "Stock",
                        "ipoDate": "1980-12-12",
                        "delistingDate": None,
                        "status": "Active",
                    }
                ]
            )
        raise ValueError(dataset)


class _DelistingConnector:
    vendor_name = "fmp"

    def fetch(self, dataset: str, symbols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        if dataset == "delistings":
            return pd.DataFrame([{"symbol": "OLD", "companyName": "Old Co", "delistedDate": "2024-01-05", "reason": "acquired"}])
        if dataset == "symbol_changes":
            return pd.DataFrame([{"oldSymbol": "FB", "newSymbol": "META", "date": "2022-06-09"}])
        raise ValueError(dataset)


class _ClusterCoordinatorStub:
    def __init__(self) -> None:
        self._lease_calls: list[str] = []
        self.allowed: set[str] = {"backfill"}

    def heartbeat_worker(self) -> dict[str, str]:
        return {"worker_id": "worker-a"}

    def sync_shard_leases(self):
        return []

    def acquire_singleton(self, task_name: str, bucket_key: str) -> bool:
        self._lease_calls.append(f"{task_name}:{bucket_key}")
        return task_name in self.allowed

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


def test_run_cluster_forever_opportunistically_runs_auxiliary_jobs_after_backfill(tmp_path: Path) -> None:
    db = DataNodeDB(tmp_path / "control" / "node.sqlite")
    raw_partition = tmp_path / "data" / "raw" / "equities_bars" / "date=2026-03-31"
    raw_partition.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"symbol": "AAPL", "date": "2026-03-31", "close": 100.0}]).to_parquet(
        raw_partition / "data.parquet",
        index=False,
    )
    service = DataNodeService(
        db=db,
        connectors={"alpaca": _NoopConnector(), "fred": _NoopConnector(), "massive": _NoopConnector(), "finnhub": _NoopConnector()},
        auditor=PartitionAuditor(db=db, calendar_store=ExchangeCalendarStore(root=tmp_path / "reference" / "calendars")),
        curator=Curator(),
        paths=DataNodePaths(root=tmp_path),
    )
    coordinator = _ClusterCoordinatorStub()
    coordinator.allowed = {"macro", "reference", "price_checks"}
    calls: list[str] = []

    service.collect_macro_data = lambda *args, **kwargs: calls.append("macro") or []  # type: ignore[method-assign]
    service.collect_reference_data = lambda *args, **kwargs: calls.append("reference") or []  # type: ignore[method-assign]

    def _price_checks(*args, **kwargs):
        calls.append("price_checks")
        service.stop()
        return tmp_path / "data" / "qc" / "price_checks_2026-03-31.parquet"

    service.run_cross_vendor_price_checks = _price_checks  # type: ignore[method-assign]

    service.run_cluster_forever(
        coordinator=coordinator,  # type: ignore[arg-type]
        symbols=["AAPL"],
        exchange="XNYS",
        collection_time_et="16:30",
        maintenance_hour_local=23,
        poll_seconds=0.0,
        now_fn=lambda: datetime.fromisoformat("2026-03-31T18:00:00+00:00"),
        sleep_fn=lambda _seconds: None,
        macro_series_ids=["DGS10"],
        reference_jobs=[{"source": "alpaca", "dataset": "equities_eod", "symbols": ["AAPL"], "output_name": "noop"}],
        price_check_symbols=["AAPL"],
    )

    assert calls == ["macro", "reference", "price_checks"]


def test_run_cluster_forever_reference_budget_failure_does_not_crash_worker(tmp_path: Path) -> None:
    db = DataNodeDB(tmp_path / "control" / "node.sqlite")
    raw_partition = tmp_path / "data" / "raw" / "equities_bars" / "date=2026-03-31"
    raw_partition.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"symbol": "AAPL", "date": "2026-03-31", "close": 100.0}]).to_parquet(
        raw_partition / "data.parquet",
        index=False,
    )
    service = DataNodeService(
        db=db,
        connectors={"alpaca": _NoopConnector(), "fred": _NoopConnector(), "massive": _BudgetFailConnector(), "finnhub": _NoopConnector()},
        auditor=PartitionAuditor(db=db, calendar_store=ExchangeCalendarStore(root=tmp_path / "reference" / "calendars")),
        curator=Curator(),
        paths=DataNodePaths(root=tmp_path),
    )
    coordinator = _ClusterCoordinatorStub()
    coordinator.allowed = {"reference", "price_checks"}
    calls: list[str] = []

    def _price_checks(*args, **kwargs):
        calls.append("price_checks")
        service.stop()
        return tmp_path / "data" / "qc" / "price_checks_2026-03-31.parquet"

    service.run_cross_vendor_price_checks = _price_checks  # type: ignore[method-assign]

    service.run_cluster_forever(
        coordinator=coordinator,  # type: ignore[arg-type]
        symbols=["AAPL"],
        exchange="XNYS",
        collection_time_et="16:30",
        maintenance_hour_local=23,
        poll_seconds=0.0,
        now_fn=lambda: datetime.fromisoformat("2026-03-31T18:00:00+00:00"),
        sleep_fn=lambda _seconds: None,
        reference_jobs=[{"source": "massive", "dataset": "reference_splits", "symbols": ["AAPL"], "output_name": "splits"}],
        price_check_symbols=["AAPL"],
    )

    assert calls == ["price_checks"]


def test_collect_reference_data_persists_partial_results_before_budget_failure(tmp_path: Path) -> None:
    db = DataNodeDB(tmp_path / "control" / "node.sqlite")
    service = DataNodeService(
        db=db,
        connectors={"alpaca": _NoopConnector(), "alpha_vantage": _PartialReferenceConnector()},
        auditor=PartitionAuditor(db=db, calendar_store=ExchangeCalendarStore(root=tmp_path / "reference" / "calendars")),
        curator=Curator(),
        paths=DataNodePaths(root=tmp_path),
    )

    try:
        service.collect_reference_data(
            [
                {
                    "source": "alpha_vantage",
                    "dataset": "corp_actions",
                    "symbols": ["AAPL", "MSFT"],
                    "start_date": "2025-01-07",
                    "end_date": "2025-01-07",
                    "output_name": "corp_actions",
                }
            ]
        )
    except TemporaryConnectorError as exc:
        assert "reference collection incomplete" in str(exc)
    else:
        raise AssertionError("expected TemporaryConnectorError")

    stored = pd.read_parquet(tmp_path / "data" / "reference" / "corp_actions.parquet")
    assert stored["symbol"].tolist() == ["AAPL"]


def test_collect_reference_data_rebuilds_security_master_outputs(tmp_path: Path) -> None:
    db = DataNodeDB(tmp_path / "control" / "node.sqlite")
    service = DataNodeService(
        db=db,
        connectors={"alpaca": _NoopConnector(), "alpha_vantage": _ListingConnector(), "fmp": _DelistingConnector()},
        auditor=PartitionAuditor(db=db, calendar_store=ExchangeCalendarStore(root=tmp_path / "reference" / "calendars")),
        curator=Curator(),
        paths=DataNodePaths(root=tmp_path),
    )

    outputs = service.collect_reference_data(
        [
            {
                "source": "alpha_vantage",
                "dataset": "listings",
                "symbols": [],
                "start_date": "2026-04-01",
                "end_date": "2026-04-01",
                "output_name": "listings",
            },
            {
                "source": "fmp",
                "dataset": "delistings",
                "symbols": [],
                "start_date": "2026-04-01",
                "end_date": "2026-04-01",
                "output_name": "delistings",
            },
            {
                "source": "fmp",
                "dataset": "symbol_changes",
                "symbols": [],
                "start_date": "2026-04-01",
                "end_date": "2026-04-01",
                "output_name": "symbol_changes",
            },
        ]
    )

    assert tmp_path / "data" / "reference" / "listing_history.parquet" in outputs
    assert tmp_path / "data" / "reference" / "ticker_changes.parquet" in outputs
