from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
import threading
import time

import numpy as np
import pandas as pd
import sqlite3

from trademl.calendars.exchange import ExchangeCalendarStore
from trademl.connectors.base import PermanentConnectorError, TemporaryConnectorError
from trademl.data_node.auxiliary_runtime import AuxiliaryRuntime
from trademl.data_node.auditor import PartitionAuditor
from trademl.data_node.budgets import BudgetManager
from trademl.data_node.canonical_runtime import CanonicalRuntime
from trademl.data_node.curator import Curator
from trademl.data_node.db import DataNodeDB
from trademl.data_node.planner import PlannedTask
from trademl.data_node import service as service_module
from trademl.data_node.service import DataNodePaths, DataNodeService


class _NoopConnector:
    vendor_name = "alpaca"

    def fetch(self, dataset: str, symbols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        return pd.DataFrame()


class _BudgetFailConnector:
    vendor_name = "massive"

    def fetch(self, dataset: str, symbols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        raise TemporaryConnectorError("budget exhausted for vendor=massive")


class _ForwardFailConnector:
    def __init__(self, vendor_name: str) -> None:
        self.vendor_name = vendor_name

    def fetch(self, dataset: str, symbols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        raise TemporaryConnectorError(f"{self.vendor_name} unavailable")


class _ForwardSuccessConnector:
    def __init__(self, vendor_name: str) -> None:
        self.vendor_name = vendor_name

    def fetch(self, dataset: str, symbols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "date": end_date,
                    "symbol": symbols[0],
                    "open": 10.0,
                    "high": 11.0,
                    "low": 9.0,
                    "close": 10.5,
                    "volume": 100,
                    "source_name": self.vendor_name,
                }
            ]
        )


class _PriceCheckConnector:
    def __init__(self, vendor_name: str) -> None:
        self.vendor_name = vendor_name
        self.calls: list[str] = []

    def fetch(self, dataset: str, symbols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        self.calls.append(self.vendor_name)
        return pd.DataFrame([{"symbol": symbol, "close": 10.0} for symbol in symbols])


class _FredConnector:
    vendor_name = "fred"

    def fetch(self, dataset: str, symbols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        if dataset == "macros_treasury":
            return pd.DataFrame(
                [{"series_id": symbols[0], "observation_date": end_date, "value": 4.0, "vintage_date": end_date}]
            )
        if dataset == "vintagedates":
            return pd.DataFrame([{"series_id": symbols[0], "vintage_date": end_date}])
        raise ValueError(dataset)


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


class _BackfillConnector:
    def __init__(self, vendor_name: str) -> None:
        self.vendor_name = vendor_name
        self.calls: list[tuple[str, list[str], str, str]] = []

    def fetch(self, dataset: str, symbols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        self.calls.append((dataset, symbols, start_date, end_date))
        return pd.DataFrame(
            [
                {
                    "date": start_date,
                    "symbol": symbols[0],
                    "open": 10.0,
                    "high": 11.0,
                    "low": 9.0,
                    "close": 10.5,
                    "vwap": pd.NA,
                    "volume": 100,
                    "trade_count": pd.NA,
                    "ingested_at": pd.Timestamp.now(tz="UTC"),
                    "source_name": self.vendor_name,
                    "source_uri": "/bars",
                    "vendor_ts": pd.Timestamp(start_date),
                }
            ]
        )


class _BudgetedBackfillConnector(_BackfillConnector):
    def __init__(self, vendor_name: str, budget_manager: BudgetManager) -> None:
        super().__init__(vendor_name)
        self.budget_manager = budget_manager


class _MultiSymbolBackfillConnector:
    def __init__(self, vendor_name: str) -> None:
        self.vendor_name = vendor_name
        self.calls: list[tuple[str, list[str], str, str]] = []

    def fetch(self, dataset: str, symbols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        self.calls.append((dataset, list(symbols), start_date, end_date))
        return pd.DataFrame(
            [
                {
                    "date": start_date,
                    "symbol": symbol,
                    "open": 10.0,
                    "high": 11.0,
                    "low": 9.0,
                    "close": 10.5,
                    "vwap": pd.NA,
                    "volume": 100,
                    "trade_count": pd.NA,
                    "ingested_at": pd.Timestamp.now(tz="UTC"),
                    "source_name": self.vendor_name,
                    "source_uri": "/bars",
                    "vendor_ts": pd.Timestamp(start_date),
                }
                for symbol in symbols
            ]
        )


class _SelectiveBackfillConnector:
    def __init__(self, vendor_name: str, available_symbols: list[str], *, empty_for_unknown: bool = True) -> None:
        self.vendor_name = vendor_name
        self.available_symbols = {symbol.upper() for symbol in available_symbols}
        self.empty_for_unknown = empty_for_unknown
        self.calls: list[tuple[str, list[str], str, str]] = []

    def fetch(self, dataset: str, symbols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        self.calls.append((dataset, list(symbols), start_date, end_date))
        rows: list[dict[str, object]] = []
        for symbol in symbols:
            if symbol.upper() not in self.available_symbols:
                continue
            rows.append(
                {
                    "date": start_date,
                    "symbol": symbol,
                    "open": 10.0,
                    "high": 11.0,
                    "low": 9.0,
                    "close": 10.5,
                    "vwap": pd.NA,
                    "volume": 100,
                    "trade_count": pd.NA,
                    "ingested_at": pd.Timestamp.now(tz="UTC"),
                    "source_name": self.vendor_name,
                    "source_uri": "/bars",
                    "vendor_ts": pd.Timestamp(start_date),
                }
            )
        if not rows and self.empty_for_unknown:
            return pd.DataFrame(columns=_BackfillConnector(self.vendor_name).fetch(dataset, ["AAPL"], start_date, end_date).columns)
        return pd.DataFrame(rows)


class _PermanentFailBackfillConnector:
    def __init__(self, vendor_name: str, message: str = "403 forbidden") -> None:
        self.vendor_name = vendor_name
        self.message = message

    def fetch(self, dataset: str, symbols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        raise PermanentConnectorError(self.message)


class _ClusterCoordinatorStub:
    def __init__(self) -> None:
        self._lease_calls: list[str] = []
        self.allowed: set[str] = {"backfill"}
        self.last_success: dict[str, dict] = {}

    def heartbeat_worker(self) -> dict[str, str]:
        return {"worker_id": "worker-a"}

    def sync_shard_leases(self):
        return []

    def acquire_singleton(self, task_name: str, bucket_key: str) -> bool:
        self._lease_calls.append(f"{task_name}:{bucket_key}")
        return task_name in self.allowed

    def acquire_or_renew_lease(self, lease_id: str) -> bool:
        self._lease_calls.append(f"renew:{lease_id}")
        return True

    def mark_singleton_success(self, task_name: str, bucket_key: str, metadata: dict | None = None) -> None:
        self._lease_calls.append(f"success:{task_name}:{bucket_key}:{metadata or {}}")
        self.last_success[task_name] = {"bucket": bucket_key, "updated_at": "2026-03-31T18:00:00+00:00", "metadata": metadata or {}}

    def read_last_success(self) -> dict[str, dict]:
        return self.last_success


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


def test_run_forever_does_not_drain_backlog_outside_maintenance_window(tmp_path: Path) -> None:
    db = DataNodeDB(tmp_path / "control" / "node.sqlite")
    db.bulk_upsert_planner_tasks(
        [
            {
                "task_key": "canonical_bars::equities_eod::00000::0000::2026-03-01::2026-03-31",
                "task_family": "canonical_bars",
                "planner_group": "canonical_bars_backlog",
                "dataset": "equities_eod",
                "tier": "A",
                "priority": 10,
                "start_date": "2026-03-01",
                "end_date": "2026-03-31",
                "symbols": ["AAPL"],
                "eligible_vendors": ["alpaca"],
                "output_name": "equities_bars",
                "payload": {"scope_kind": "symbol_range", "trading_days": ["2026-03-31"]},
            }
        ]
    )
    service = DataNodeService(
        db=db,
        connectors={"alpaca": _NoopConnector()},
        auditor=PartitionAuditor(db=db, calendar_store=ExchangeCalendarStore(root=tmp_path / "reference" / "calendars")),
        curator=Curator(),
        paths=DataNodePaths(root=tmp_path),
    )

    calls: list[str] = []
    service.run_cycle = lambda *args, **kwargs: calls.append("run_cycle")  # type: ignore[method-assign]
    service.process_backfill_queue = lambda: calls.append("backfill") or []  # type: ignore[method-assign]
    service.sync_partition_status = lambda: tmp_path / "data" / "qc" / "partition_status.parquet"  # type: ignore[method-assign]

    timestamps = iter(
        [
            datetime.fromisoformat("2026-03-31T20:35:00+00:00"),
            datetime.fromisoformat("2026-03-31T20:35:00+00:00"),
        ]
    )

    def _sleep(_seconds: float) -> None:
        service.stop()

    service.run_forever(
        symbols=["AAPL"],
        exchange="XNYS",
        maintenance_hour_local=23,
        now_fn=lambda: next(timestamps),
        sleep_fn=_sleep,
        poll_seconds=0.0,
    )

    assert "run_cycle" in calls
    assert "backfill" not in calls


def test_collect_forward_falls_back_to_secondary_vendor(tmp_path: Path) -> None:
    db = DataNodeDB(tmp_path / "control" / "node.sqlite")
    service = DataNodeService(
        db=db,
        connectors={"alpaca": _ForwardFailConnector("alpaca"), "tiingo": _ForwardSuccessConnector("tiingo")},
        auditor=PartitionAuditor(db=db, calendar_store=ExchangeCalendarStore(root=tmp_path / "reference" / "calendars")),
        curator=Curator(),
        paths=DataNodePaths(root=tmp_path),
    )

    changed = service.collect_forward(trading_date="2026-03-31", symbols=["AAPL"])

    assert changed == ["2026-03-31"]
    stored = pd.read_parquet(tmp_path / "data" / "raw" / "equities_bars" / "date=2026-03-31" / "data.parquet")
    assert stored["symbol"].tolist() == ["AAPL"]


def test_seed_planner_tasks_passes_freeze_cutoff_to_planner(tmp_path: Path, monkeypatch) -> None:
    db = DataNodeDB(tmp_path / "control" / "node.sqlite")
    service = DataNodeService(
        db=db,
        connectors={"alpaca": _NoopConnector()},
        auditor=PartitionAuditor(db=db, calendar_store=ExchangeCalendarStore(root=tmp_path / "reference" / "calendars")),
        curator=Curator(),
        paths=DataNodePaths(root=tmp_path),
    )
    service.default_symbols = ["AAPL", "MSFT"]

    captured: dict[str, object] = {}

    monkeypatch.setattr(
        service_module,
        "recommended_training_cutoff",
        lambda **kwargs: {"date": "2026-03-06", "coverage_ratio": 1.0},
    )

    def fake_plan_coverage_tasks(**kwargs):  # noqa: ANN001
        captured.update(kwargs)
        return []

    monkeypatch.setattr(service_module, "plan_coverage_tasks", fake_plan_coverage_tasks)

    service._seed_planner_tasks(trading_date="2026-04-07")

    assert captured["freeze_report_date"] == "2026-03-06"
    assert captured["current_date"] == "2026-04-07"


def test_seed_planner_tasks_reopens_regressed_canonical_tasks(tmp_path: Path, monkeypatch) -> None:
    db = DataNodeDB(tmp_path / "control" / "node.sqlite")
    service = DataNodeService(
        db=db,
        connectors={"alpaca": _NoopConnector(), "tiingo": _NoopConnector()},
        auditor=PartitionAuditor(db=db, calendar_store=ExchangeCalendarStore(root=tmp_path / "reference" / "calendars")),
        curator=Curator(),
        paths=DataNodePaths(root=tmp_path),
    )
    service.default_symbols = ["AAPL"]
    task = PlannedTask(
        task_key="canonical::AAPL::2025-01-03",
        task_family="canonical_bars",
        planner_group="canonical_bars_backlog",
        dataset="equities_eod",
        tier="A",
        priority=5,
        start_date="2025-01-03",
        end_date="2025-01-03",
        symbols=("AAPL",),
        output_name=None,
        preferred_vendors=("alpaca", "tiingo"),
        payload={"scope_kind": "symbol_range", "trading_days": ["2025-01-03"]},
    )
    db.upsert_planner_task(
        task_key=task.task_key,
        task_family=task.task_family,
        planner_group=task.planner_group,
        dataset=task.dataset,
        tier=task.tier,
        priority=task.priority,
        start_date=task.start_date,
        end_date=task.end_date,
        symbols=task.symbols,
        eligible_vendors=task.preferred_vendors,
        payload=task.payload,
    )
    db.mark_planner_task_success(task.task_key)
    db.update_planner_task_progress(
        task_key=task.task_key,
        expected_units=1,
        completed_units=1,
        remaining_units=0,
        completed_symbols=["AAPL"],
        remaining_symbols=[],
        state={"scope_kind": "symbol_range"},
    )
    leased = db.lease_vendor_attempt(
        task_key=task.task_key,
        task_family="canonical_bars",
        planner_group=task.planner_group,
        vendor="alpaca",
        lease_owner=service.worker_id,
        payload={"symbols": ["AAPL"], "start_date": task.start_date, "end_date": task.end_date},
    )
    assert leased is not None
    db.mark_vendor_attempt_success(task_key=task.task_key, vendor="alpaca", rows_returned=1)

    monkeypatch.setattr(
        service_module,
        "recommended_training_cutoff",
        lambda **kwargs: {"date": "2025-01-03", "coverage_ratio": 1.0},
    )
    monkeypatch.setattr(service_module, "plan_coverage_tasks", lambda **kwargs: [task])

    service._seed_planner_tasks(trading_date="2025-01-04")

    refreshed = db.get_planner_task(task.task_key)
    progress = db.fetch_planner_task_progress(task.task_key)
    attempts = db.vendor_attempts_for_task(task.task_key)

    assert refreshed is not None
    assert refreshed.status == "PENDING"
    assert refreshed.last_error == "canonical coverage regressed"
    assert progress is not None
    assert progress.remaining_units == 1
    assert progress.completed_units == 0
    assert attempts == []


def test_backfill_budget_exhaustion_defers_task_instead_of_marking_failed(tmp_path: Path) -> None:
    db = DataNodeDB(tmp_path / "control" / "node.sqlite")
    task_id = db.enqueue_task("equities_eod", "AAPL", "2025-01-01", "2025-01-02", "GAP", 1)
    leased = db.lease_task_by_id(task_id)
    assert leased is not None
    service = DataNodeService(
        db=db,
        connectors={"finnhub": _BudgetFailConnector()},
        auditor=PartitionAuditor(db=db, calendar_store=ExchangeCalendarStore(root=tmp_path / "reference" / "calendars")),
        curator=Curator(),
        paths=DataNodePaths(root=tmp_path),
        source_name="finnhub",
    )

    changed = service._canonical_runtime._process_backfill_task_for_vendor(leased, "finnhub")
    with sqlite3.connect(tmp_path / "control" / "node.sqlite") as connection:
        row = connection.execute("SELECT status, last_error, next_not_before FROM backfill_queue WHERE id = ?", (task_id,)).fetchone()

    assert changed == []
    assert row is not None
    assert row[0] == "PENDING"
    assert "budget exhausted" in (row[1] or "")
    assert row[2] is not None


def test_lease_next_task_for_vendor_skips_single_symbol_vendors_for_datewide_backfill(tmp_path: Path) -> None:
    db = DataNodeDB(tmp_path / "control" / "node.sqlite")
    db.enqueue_task("equities_eod", None, "2025-01-01", "2025-01-02", "GAP", 1)
    service = DataNodeService(
        db=db,
        connectors={"alpaca": _BackfillConnector("alpaca"), "tiingo": _BackfillConnector("tiingo")},
        auditor=PartitionAuditor(db=db, calendar_store=ExchangeCalendarStore(root=tmp_path / "reference" / "calendars")),
        curator=Curator(),
        paths=DataNodePaths(root=tmp_path),
        source_name="alpaca",
    )

    tiingo_task = service._canonical_runtime._lease_next_task_for_vendor("tiingo")
    alpaca_task = service._canonical_runtime._lease_next_task_for_vendor("alpaca")

    assert tiingo_task is None
    assert alpaca_task is not None
    assert alpaca_task.symbol is None


def test_datewide_backfill_merges_parallel_vendor_results_and_marks_task_done(tmp_path: Path) -> None:
    db = DataNodeDB(tmp_path / "control" / "node.sqlite")
    task_id = db.enqueue_task("equities_eod", None, "2025-01-01", "2025-01-01", "GAP", 1)
    leased = db.lease_task_by_id(task_id)
    assert leased is not None
    alpaca = _SelectiveBackfillConnector("alpaca", ["AAPL", "MSFT"])
    tiingo = _SelectiveBackfillConnector("tiingo", ["NVDA"])
    service = DataNodeService(
        db=db,
        connectors={"alpaca": alpaca, "tiingo": tiingo},
        auditor=PartitionAuditor(db=db, calendar_store=ExchangeCalendarStore(root=tmp_path / "reference" / "calendars")),
        curator=Curator(),
        paths=DataNodePaths(root=tmp_path),
        source_name="alpaca",
    )
    service.default_symbols = ["AAPL", "MSFT", "NVDA"]

    changed = service._canonical_runtime._process_backfill_task_for_vendor(leased, "alpaca")

    assert changed == ["2025-01-01"]
    stored = pd.read_parquet(tmp_path / "data" / "raw" / "equities_bars" / "date=2025-01-01" / "data.parquet")
    assert sorted(stored["symbol"].tolist()) == ["AAPL", "MSFT", "NVDA"]
    with sqlite3.connect(tmp_path / "control" / "node.sqlite") as connection:
        status_row = connection.execute("SELECT status FROM backfill_queue WHERE id = ?", (task_id,)).fetchone()
    assert status_row == ("DONE",)
    assert alpaca.calls
    assert tiingo.calls


def test_datewide_backfill_keeps_partial_progress_and_defers_remaining_symbols(tmp_path: Path) -> None:
    db = DataNodeDB(tmp_path / "control" / "node.sqlite")
    task_id = db.enqueue_task("equities_eod", None, "2025-01-02", "2025-01-02", "GAP", 1)
    leased = db.lease_task_by_id(task_id)
    assert leased is not None
    alpaca = _SelectiveBackfillConnector("alpaca", ["AAPL"])
    service = DataNodeService(
        db=db,
        connectors={"alpaca": alpaca},
        auditor=PartitionAuditor(db=db, calendar_store=ExchangeCalendarStore(root=tmp_path / "reference" / "calendars")),
        curator=Curator(),
        paths=DataNodePaths(root=tmp_path),
        source_name="alpaca",
    )
    service.default_symbols = ["AAPL", "MSFT"]

    changed = service._canonical_runtime._process_backfill_task_for_vendor(leased, "alpaca")

    assert changed == ["2025-01-02"]
    stored = pd.read_parquet(tmp_path / "data" / "raw" / "equities_bars" / "date=2025-01-02" / "data.parquet")
    assert stored["symbol"].tolist() == ["AAPL"]
    with sqlite3.connect(tmp_path / "control" / "node.sqlite") as connection:
        row = connection.execute("SELECT status, last_error FROM backfill_queue WHERE id = ?", (task_id,)).fetchone()
    assert row is not None
    assert row[0] == "PENDING"
    assert "remaining_symbols=1" in (row[1] or "")


def test_canonical_planner_batch_uses_atomic_symbol_tasks_and_multisymbol_fetch(tmp_path: Path) -> None:
    db = DataNodeDB(tmp_path / "control" / "node.sqlite")
    service = DataNodeService(
        db=db,
        connectors={"alpaca": _MultiSymbolBackfillConnector("alpaca")},
        auditor=PartitionAuditor(db=db, calendar_store=ExchangeCalendarStore(root=tmp_path / "reference" / "calendars")),
        curator=Curator(),
        paths=DataNodePaths(root=tmp_path),
        source_name="alpaca",
    )
    tasks = []
    for symbol in ["AAPL", "MSFT", "NVDA"]:
        task_key = f"canonical::{symbol}"
        db.upsert_planner_task(
            task_key=task_key,
            task_family="canonical_bars",
            planner_group="canonical_bars_backlog",
            dataset="equities_eod",
            tier="A",
            priority=10,
            start_date="2025-01-02",
            end_date="2025-01-02",
            symbols=[symbol],
            eligible_vendors=["alpaca", "tiingo"],
            payload={"scope_kind": "symbol_range", "trading_days": ["2025-01-02"]},
        )
        db.update_planner_task_progress(
            task_key=task_key,
            expected_units=1,
            completed_units=0,
            remaining_units=1,
            remaining_symbols=[symbol],
            state={"scope_kind": "symbol_range"},
        )
    tasks = service._canonical_runtime._lease_canonical_batch("alpaca")

    changed = service._canonical_runtime._process_canonical_planner_batch(batch=tasks, vendor="alpaca", exchange="XNYS")

    assert changed == ["2025-01-02"]
    connector = service.connectors["alpaca"]
    assert isinstance(connector, _MultiSymbolBackfillConnector)
    assert connector.calls == [("equities_eod", ["AAPL", "MSFT", "NVDA"], "2025-01-02", "2025-01-02")]
    stored = pd.read_parquet(tmp_path / "data" / "raw" / "equities_bars" / "date=2025-01-02" / "data.parquet")
    assert sorted(stored["symbol"].tolist()) == ["AAPL", "MSFT", "NVDA"]
    statuses = {task.task_key: db.get_planner_task(task.task_key).status for task in tasks}
    assert set(statuses.values()) == {"SUCCESS"}


def test_materialize_job_rotates_symbol_subset_deterministically() -> None:
    job = {
        "source": "finnhub",
        "dataset": "company_profile",
        "symbols": ["AAPL", "MSFT", "NVDA", "META", "TSLA"],
        "max_symbols_per_run": 2,
        "rotation_key": "finnhub:company_profile",
    }

    materialized = DataNodeService._materialize_job(job, "2026-04-01")

    assert materialized["symbols"] == ["MSFT", "NVDA"]
    assert materialized["start_date"] == "2026-04-01"
    assert materialized["end_date"] == "2026-04-01"


def test_expand_reference_jobs_respects_batch_jobs() -> None:
    expanded = AuxiliaryRuntime._expand_reference_jobs(
        [
            {
                "source": "tiingo",
                "dataset": "corp_actions_dividends",
                "symbols": ["AAPL", "MSFT"],
                "explode_symbols": False,
            }
        ]
    )

    assert len(expanded) == 1
    assert expanded[0]["symbols"] == ["AAPL", "MSFT"]


def test_append_reference_frame_deduplicates_dict_valued_rows_without_crashing(tmp_path: Path) -> None:
    db = DataNodeDB(tmp_path / "control" / "node.sqlite")
    service = DataNodeService(
        db=db,
        connectors={"alpaca": _NoopConnector()},
        auditor=PartitionAuditor(db=db, calendar_store=ExchangeCalendarStore(root=tmp_path / "reference" / "calendars")),
        curator=Curator(),
        paths=DataNodePaths(root=tmp_path),
    )
    output = tmp_path / "data" / "reference" / "sec_filings.parquet"
    frame = pd.DataFrame(
        [
            {
                "cik": "0000320193",
                "accession": "0000320193-26-000001",
                "facts": {"revenue": {"value": 1}},
                "items": [{"name": "10-K"}],
            },
            {
                "cik": "0000320193",
                "accession": "0000320193-26-000001",
                "facts": {"revenue": {"value": 1}},
                "items": [{"name": "10-K"}],
            },
        ]
    )

    service._auxiliary_runtime._append_reference_frame(output, frame)

    stored = pd.read_parquet(output)
    assert len(stored) == 1
    assert stored.iloc[0]["accession"] == "0000320193-26-000001"


def test_append_reference_frame_deduplicates_ndarray_valued_rows_without_crashing(tmp_path: Path) -> None:
    db = DataNodeDB(tmp_path / "control" / "node.sqlite")
    service = DataNodeService(
        db=db,
        connectors={"alpaca": _NoopConnector()},
        auditor=PartitionAuditor(db=db, calendar_store=ExchangeCalendarStore(root=tmp_path / "reference" / "calendars")),
        curator=Curator(),
        paths=DataNodePaths(root=tmp_path),
    )
    output = tmp_path / "data" / "reference" / "companyfacts.parquet"
    frame = pd.DataFrame(
        [
            {
                "symbol": "AAPL",
                "facts": {
                    "segments": np.array([1, 2, 3], dtype=np.int64),
                    "meta": {"weights": np.array([0.25, 0.75], dtype=np.float64)},
                },
            },
            {
                "symbol": "AAPL",
                "facts": {
                    "segments": np.array([1, 2, 3], dtype=np.int64),
                    "meta": {"weights": np.array([0.25, 0.75], dtype=np.float64)},
                },
            },
        ]
    )

    service._auxiliary_runtime._append_reference_frame(output, frame)

    stored = pd.read_parquet(output)
    assert len(stored) == 1
    assert stored.iloc[0]["symbol"] == "AAPL"


def test_process_planner_queue_survives_auxiliary_lane_exception(tmp_path: Path) -> None:
    db = DataNodeDB(tmp_path / "control" / "node.sqlite")
    service = DataNodeService(
        db=db,
        connectors={"alpaca": _NoopConnector()},
        auditor=PartitionAuditor(db=db, calendar_store=ExchangeCalendarStore(root=tmp_path / "reference" / "calendars")),
        curator=Curator(),
        paths=DataNodePaths(root=tmp_path),
    )

    service.default_symbols = ["AAPL"]
    service._seed_planner_tasks = lambda trading_date=None: None  # type: ignore[method-assign]
    service._canonical_runtime._backfill_lane_widths = lambda: {}  # type: ignore[method-assign]
    service._aux_lane_widths = lambda task_kinds=None: {"alpaca": 1}  # type: ignore[method-assign]
    service._drain_auxiliary_lane = lambda vendor: (_ for _ in ()).throw(RuntimeError("aux boom"))  # type: ignore[method-assign]

    changed = service.process_planner_queue()

    assert changed == []


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


def test_ensure_planner_backlog_seeded_when_planner_is_empty(tmp_path: Path) -> None:
    db = DataNodeDB(tmp_path / "control" / "node.sqlite")
    service = DataNodeService(
        db=db,
        connectors={"alpaca": _NoopConnector()},
        auditor=PartitionAuditor(db=db, calendar_store=ExchangeCalendarStore(root=tmp_path / "reference" / "calendars")),
        curator=Curator(),
        paths=DataNodePaths(root=tmp_path),
    )
    service.default_symbols = ["AAPL", "MSFT"]
    service.stage_years = 1

    service._ensure_planner_backlog_seeded(trading_date="2026-04-03")

    planner_tasks = db.fetch_planner_tasks(limit=5)
    assert planner_tasks
    assert planner_tasks[0].task_family == "canonical_bars"


def test_ensure_planner_backlog_seeded_refreshes_existing_backlog_once_per_day(tmp_path: Path, monkeypatch) -> None:
    db = DataNodeDB(tmp_path / "control" / "node.sqlite")
    service = DataNodeService(
        db=db,
        connectors={"alpaca": _NoopConnector()},
        auditor=PartitionAuditor(db=db, calendar_store=ExchangeCalendarStore(root=tmp_path / "reference" / "calendars")),
        curator=Curator(),
        paths=DataNodePaths(root=tmp_path),
    )
    service.default_symbols = ["AAPL", "MSFT"]

    calls: list[str] = []

    def fake_seed(*, trading_date: str | None = None) -> None:
        calls.append(str(trading_date))

    monkeypatch.setattr(service, "_seed_planner_tasks", fake_seed)
    db.upsert_planner_task(
        task_key="canonical::existing",
        task_family="canonical_bars",
        planner_group="canonical_bars_backlog",
        dataset="equities_eod",
        tier="A",
        priority=5,
        start_date="2026-04-01",
        end_date="2026-04-30",
        symbols=["AAPL"],
        eligible_vendors=["alpaca"],
        output_name="equities_bars",
        payload={"scope_kind": "symbol_range"},
    )

    service._ensure_planner_backlog_seeded(trading_date="2026-04-03")
    service._ensure_planner_backlog_seeded(trading_date="2026-04-03")
    service._ensure_planner_backlog_seeded(trading_date="2026-04-04")

    assert calls == ["2026-04-03", "2026-04-04"]


def test_lease_canonical_batch_scans_past_front_slice_starvation(tmp_path: Path) -> None:
    db = DataNodeDB(tmp_path / "control" / "node.sqlite")
    service = DataNodeService(
        db=db,
        connectors={
            "alpaca": _MultiSymbolBackfillConnector("alpaca"),
            "tiingo": _BackfillConnector("tiingo"),
        },
        auditor=PartitionAuditor(db=db, calendar_store=ExchangeCalendarStore(root=tmp_path / "reference" / "calendars")),
        curator=Curator(),
        paths=DataNodePaths(root=tmp_path),
        source_name="alpaca",
    )
    for idx in range(300):
        task_key = f"canonical::{idx:04d}"
        db.upsert_planner_task(
            task_key=task_key,
            task_family="canonical_bars",
            planner_group="canonical_bars_backlog",
            dataset="equities_eod",
            tier="A",
            priority=10,
            start_date="2025-01-02",
            end_date="2025-01-31",
            symbols=[f"SYM{idx:04d}"],
            eligible_vendors=["alpaca", "tiingo"],
            payload={"scope_kind": "symbol_range", "trading_days": ["2025-01-02"]},
        )
        db.update_planner_task_progress(
            task_key=task_key,
            expected_units=1,
            completed_units=0,
            remaining_units=1,
            remaining_symbols=[f"SYM{idx:04d}"],
            state={"scope_kind": "symbol_range"},
        )
        if idx < 256:
            leased = db.lease_planner_task_by_key(task_key=task_key, lease_owner="alpaca-worker")
            assert leased is not None

    batch = service._canonical_runtime._lease_canonical_batch("tiingo")

    assert batch
    assert batch[0].symbols == ("SYM0256",)


def test_lease_canonical_batch_prioritizes_small_partial_tail(tmp_path: Path) -> None:
    db = DataNodeDB(tmp_path / "control" / "node.sqlite")
    service = DataNodeService(
        db=db,
        connectors={
            "alpaca": _MultiSymbolBackfillConnector("alpaca"),
            "tiingo": _BackfillConnector("tiingo"),
        },
        auditor=PartitionAuditor(db=db, calendar_store=ExchangeCalendarStore(root=tmp_path / "reference" / "calendars")),
        curator=Curator(),
        paths=DataNodePaths(root=tmp_path),
        source_name="alpaca",
    )
    db.upsert_planner_task(
        task_key="canonical::older_fresh",
        task_family="canonical_bars",
        planner_group="canonical_bars_backlog",
        dataset="equities_eod",
        tier="A",
        priority=5,
        start_date="2026-03-16",
        end_date="2026-04-10",
        symbols=["BIG"],
        eligible_vendors=["alpaca", "tiingo"],
        payload={"scope_kind": "symbol_range", "trading_days": ["2026-03-16"]},
    )
    db.update_planner_task_progress(
        task_key="canonical::older_fresh",
        expected_units=19,
        completed_units=0,
        remaining_units=19,
        remaining_symbols=["BIG"],
        state={"scope_kind": "symbol_range"},
    )
    db.upsert_planner_task(
        task_key="canonical::small_partial",
        task_family="canonical_bars",
        planner_group="canonical_bars_backlog",
        dataset="equities_eod",
        tier="A",
        priority=5,
        start_date="2026-03-16",
        end_date="2026-04-10",
        symbols=["TAIL"],
        eligible_vendors=["alpaca", "tiingo"],
        payload={"scope_kind": "symbol_range", "trading_days": ["2026-03-16"]},
    )
    db.update_planner_task_progress(
        task_key="canonical::small_partial",
        expected_units=19,
        completed_units=17,
        remaining_units=2,
        remaining_symbols=["TAIL"],
        state={"scope_kind": "symbol_range"},
    )
    db.mark_planner_task_partial(
        "canonical::small_partial",
        error="tiingo: budget exhausted for vendor=tiingo",
        backoff_minutes=0,
    )

    batch = service._canonical_runtime._lease_canonical_batch("alpaca")

    assert batch
    assert batch[0].task_key == "canonical::small_partial"


def test_permanent_vendor_failure_does_not_terminally_poison_task_when_alternatives_remain(tmp_path: Path) -> None:
    db = DataNodeDB(tmp_path / "control" / "node.sqlite")
    service = DataNodeService(
        db=db,
        connectors={
            "alpaca": _PermanentFailBackfillConnector("alpaca", "alpaca request failed: 403 forbidden"),
            "tiingo": _BackfillConnector("tiingo"),
        },
        auditor=PartitionAuditor(db=db, calendar_store=ExchangeCalendarStore(root=tmp_path / "reference" / "calendars")),
        curator=Curator(),
        paths=DataNodePaths(root=tmp_path),
        source_name="alpaca",
    )
    db.upsert_planner_task(
        task_key="canonical::AAPL",
        task_family="canonical_bars",
        planner_group="canonical_bars_backlog",
        dataset="equities_eod",
        tier="A",
        priority=10,
        start_date="2025-01-02",
        end_date="2025-01-02",
        symbols=["AAPL"],
        eligible_vendors=["alpaca", "tiingo"],
        payload={"scope_kind": "symbol_range", "trading_days": ["2025-01-02"]},
    )
    db.update_planner_task_progress(
        task_key="canonical::AAPL",
        expected_units=1,
        completed_units=0,
        remaining_units=1,
        remaining_symbols=["AAPL"],
        state={"scope_kind": "symbol_range"},
    )
    task = db.lease_planner_task_by_key(task_key="canonical::AAPL", lease_owner=service.worker_id)

    assert task is not None
    changed = service._canonical_runtime._process_canonical_planner_batch(batch=[task], vendor="alpaca", exchange="XNYS")

    assert changed == []
    refreshed = db.get_planner_task("canonical::AAPL")
    assert refreshed is not None
    assert refreshed.status == "PARTIAL"
    attempts = {(attempt.vendor, attempt.status) for attempt in db.vendor_attempts_for_task("canonical::AAPL")}
    assert ("alpaca", "PERMANENT_FAILED") in attempts


def test_canonical_partial_success_is_immediately_reeligible_when_alternative_vendor_remains(tmp_path: Path) -> None:
    db = DataNodeDB(tmp_path / "control" / "node.sqlite")
    service = DataNodeService(
        db=db,
        connectors={
            "alpaca": _SelectiveBackfillConnector("alpaca", ["AAPL"]),
            "tiingo": _BackfillConnector("tiingo"),
        },
        auditor=PartitionAuditor(db=db, calendar_store=ExchangeCalendarStore(root=tmp_path / "reference" / "calendars")),
        curator=Curator(),
        paths=DataNodePaths(root=tmp_path),
        source_name="alpaca",
    )
    db.upsert_planner_task(
        task_key="canonical::AAPL::chunk0",
        task_family="canonical_bars",
        planner_group="canonical_bars_backlog",
        dataset="equities_eod",
        tier="A",
        priority=5,
        start_date="2025-01-02",
        end_date="2025-01-03",
        symbols=["AAPL"],
        eligible_vendors=["alpaca", "tiingo"],
        payload={"scope_kind": "symbol_range", "trading_days": ["2025-01-02", "2025-01-03"]},
    )
    db.update_planner_task_progress(
        task_key="canonical::AAPL::chunk0",
        expected_units=2,
        completed_units=0,
        remaining_units=2,
        remaining_symbols=["AAPL"],
        state={"scope_kind": "symbol_range"},
    )
    task = db.lease_planner_task_by_key(task_key="canonical::AAPL::chunk0", lease_owner=service.worker_id)

    assert task is not None
    changed = service._canonical_runtime._process_canonical_planner_batch(batch=[task], vendor="alpaca", exchange="XNYS")

    assert changed == ["2025-01-02"]
    refreshed = db.get_planner_task("canonical::AAPL::chunk0")
    assert refreshed is not None
    assert refreshed.status == "PARTIAL"
    assert refreshed.next_eligible_at is not None
    delta = datetime.fromisoformat(refreshed.next_eligible_at) - datetime.now(tz=service_module.UTC)
    assert delta.total_seconds() < 60


def test_canonical_partial_task_can_retry_same_vendor_after_stale_success(tmp_path: Path) -> None:
    db = DataNodeDB(tmp_path / "control" / "node.sqlite")
    alpaca = _BackfillConnector("alpaca")
    service = DataNodeService(
        db=db,
        connectors={"alpaca": alpaca},
        auditor=PartitionAuditor(db=db, calendar_store=ExchangeCalendarStore(root=tmp_path / "reference" / "calendars")),
        curator=Curator(),
        paths=DataNodePaths(root=tmp_path),
        source_name="alpaca",
    )
    db.upsert_planner_task(
        task_key="canonical::TECK::stale_success",
        task_family="canonical_bars",
        planner_group="canonical_bars_backlog",
        dataset="equities_eod",
        tier="A",
        priority=5,
        start_date="2025-12-16",
        end_date="2025-12-16",
        symbols=["TECK"],
        eligible_vendors=["alpaca"],
        payload={"scope_kind": "symbol_range", "trading_days": ["2025-12-16"]},
    )
    db.update_planner_task_progress(
        task_key="canonical::TECK::stale_success",
        expected_units=1,
        completed_units=0,
        remaining_units=1,
        remaining_symbols=["TECK"],
        state={"scope_kind": "symbol_range"},
    )
    seeded = db.lease_vendor_attempt(
        task_key="canonical::TECK::stale_success",
        task_family="canonical_bars",
        planner_group="canonical_bars_backlog",
        vendor="alpaca",
        lease_owner="worker-0",
        payload={"symbols": ["TECK"], "start_date": "2025-12-16", "end_date": "2025-12-16"},
    )
    assert seeded is not None
    db.mark_vendor_attempt_success(task_key="canonical::TECK::stale_success", vendor="alpaca", rows_returned=1)
    db.mark_planner_task_partial("canonical::TECK::stale_success", error="remaining_units=1", backoff_minutes=0)

    batch = service._canonical_runtime._lease_canonical_batch("alpaca")

    assert len(batch) == 1
    changed = service._canonical_runtime._process_canonical_planner_batch(batch=batch, vendor="alpaca", exchange="XNYS")

    assert changed == ["2025-12-16"]
    refreshed = db.get_planner_task("canonical::TECK::stale_success")
    assert refreshed is not None
    assert refreshed.status == "SUCCESS"
    progress = db.fetch_planner_task_progress("canonical::TECK::stale_success")
    assert progress is not None
    assert progress.remaining_units == 0


def test_canonical_vendor_budget_failure_does_not_back_off_other_available_vendors(tmp_path: Path) -> None:
    db = DataNodeDB(tmp_path / "control" / "node.sqlite")
    service = DataNodeService(
        db=db,
        connectors={
            "tiingo": _BudgetFailConnector(),
            "alpaca": _BackfillConnector("alpaca"),
        },
        auditor=PartitionAuditor(db=db, calendar_store=ExchangeCalendarStore(root=tmp_path / "reference" / "calendars")),
        curator=Curator(),
        paths=DataNodePaths(root=tmp_path),
        source_name="alpaca",
    )
    db.upsert_planner_task(
        task_key="canonical::AAPL::budget",
        task_family="canonical_bars",
        planner_group="canonical_bars_backlog",
        dataset="equities_eod",
        tier="A",
        priority=5,
        start_date="2025-01-02",
        end_date="2025-01-02",
        symbols=["AAPL"],
        eligible_vendors=["tiingo", "alpaca"],
        payload={"scope_kind": "symbol_range", "trading_days": ["2025-01-02"]},
    )
    db.update_planner_task_progress(
        task_key="canonical::AAPL::budget",
        expected_units=1,
        completed_units=0,
        remaining_units=1,
        remaining_symbols=["AAPL"],
        state={"scope_kind": "symbol_range"},
    )
    task = db.lease_planner_task_by_key(task_key="canonical::AAPL::budget", lease_owner=service.worker_id)

    assert task is not None
    changed = service._canonical_runtime._process_canonical_planner_batch(batch=[task], vendor="tiingo", exchange="XNYS")

    assert changed == []
    refreshed = db.get_planner_task("canonical::AAPL::budget")
    assert refreshed is not None
    assert refreshed.status == "PARTIAL"
    assert refreshed.next_eligible_at is not None
    delta = datetime.fromisoformat(refreshed.next_eligible_at) - datetime.now(tz=service_module.UTC)
    assert delta.total_seconds() < 60


def test_canonical_vendor_selection_skips_symbol_windows_before_learned_vendor_floor(tmp_path: Path) -> None:
    db = DataNodeDB(tmp_path / "control" / "node.sqlite")
    alpaca = _BackfillConnector("alpaca")
    tiingo = _BackfillConnector("tiingo")
    service = DataNodeService(
        db=db,
        connectors={"alpaca": alpaca, "tiingo": tiingo},
        auditor=PartitionAuditor(db=db, calendar_store=ExchangeCalendarStore(root=tmp_path / "reference" / "calendars")),
        curator=Curator(),
        paths=DataNodePaths(root=tmp_path),
        source_name="alpaca",
    )
    for task_key, start_date, end_date, status, error in [
        ("canonical::APLD::early", "2021-05-05", "2021-06-02", "PERMANENT_FAILED", "empty planner canonical result"),
        ("canonical::APLD::late", "2022-03-18", "2022-04-14", "SUCCESS", None),
    ]:
        attempt = db.lease_vendor_attempt(
            task_key=task_key,
            task_family="canonical_bars",
            planner_group="canonical_bars_backlog",
            vendor="alpaca",
            lease_owner="worker-1",
            payload={"symbols": ["APLD"], "start_date": start_date, "end_date": end_date},
        )
        assert attempt is not None
        if status == "SUCCESS":
            db.mark_vendor_attempt_success(task_key=task_key, vendor="alpaca", rows_returned=20)
        else:
            db.mark_vendor_attempt_failed(task_key=task_key, vendor="alpaca", error=str(error), backoff_minutes=0, permanent=True)
    db.upsert_planner_task(
        task_key="canonical::APLD::current",
        task_family="canonical_bars",
        planner_group="canonical_bars_backlog",
        dataset="equities_eod",
        tier="A",
        priority=5,
        start_date="2021-07-01",
        end_date="2021-07-29",
        symbols=["APLD"],
        eligible_vendors=["alpaca", "tiingo"],
        payload={"scope_kind": "symbol_range", "trading_days": ["2021-07-01"]},
    )
    db.update_planner_task_progress(
        task_key="canonical::APLD::current",
        expected_units=1,
        completed_units=0,
        remaining_units=1,
        remaining_symbols=["APLD"],
        state={"scope_kind": "symbol_range"},
    )
    task = db.lease_planner_task_by_key(task_key="canonical::APLD::current", lease_owner=service.worker_id)

    assert task is not None
    changed = service._canonical_runtime._process_canonical_planner_task(task, exchange="XNYS")

    assert changed == ["2021-07-01"]
    assert alpaca.calls == []
    assert len(tiingo.calls) == 1


def test_canonical_vendor_selection_skips_tiingo_before_supported_ticker_start_date(tmp_path: Path) -> None:
    db = DataNodeDB(tmp_path / "control" / "node.sqlite")
    alpaca = _BackfillConnector("alpaca")
    tiingo = _BackfillConnector("tiingo")
    service = DataNodeService(
        db=db,
        connectors={"alpaca": alpaca, "tiingo": tiingo},
        auditor=PartitionAuditor(db=db, calendar_store=ExchangeCalendarStore(root=tmp_path / "reference" / "calendars")),
        curator=Curator(),
        paths=DataNodePaths(root=tmp_path),
        source_name="alpaca",
    )
    reference_root = tmp_path / "data" / "reference"
    reference_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"symbol": "APLD", "exchange": "XNAS", "start_date": "2022-03-18", "end_date": "2026-04-09"}]
    ).to_parquet(reference_root / "tiingo_supported_tickers.parquet", index=False)
    db.upsert_planner_task(
        task_key="canonical::APLD::tiingo_meta",
        task_family="canonical_bars",
        planner_group="canonical_bars_backlog",
        dataset="equities_eod",
        tier="A",
        priority=5,
        start_date="2021-07-01",
        end_date="2021-07-29",
        symbols=["APLD"],
        eligible_vendors=["tiingo", "alpaca"],
        payload={"scope_kind": "symbol_range", "trading_days": ["2021-07-01"]},
    )
    db.update_planner_task_progress(
        task_key="canonical::APLD::tiingo_meta",
        expected_units=1,
        completed_units=0,
        remaining_units=1,
        remaining_symbols=["APLD"],
        state={"scope_kind": "symbol_range"},
    )
    task = db.lease_planner_task_by_key(task_key="canonical::APLD::tiingo_meta", lease_owner=service.worker_id)

    assert task is not None
    changed = service._canonical_runtime._process_canonical_planner_task(task, exchange="XNYS")

    assert changed == ["2021-07-01"]
    assert alpaca.calls == [("equities_eod", ["APLD"], "2021-07-01", "2021-07-29")]
    assert tiingo.calls == []


def test_lease_canonical_batch_skips_budget_blocked_vendor_lane(tmp_path: Path) -> None:
    db = DataNodeDB(tmp_path / "control" / "node.sqlite")
    tiingo_budget = BudgetManager({"tiingo": {"rpm": 1, "daily_cap": 10}})
    tiingo_budget.record_spend("tiingo", task_kind="FORWARD")
    alpaca = _BudgetedBackfillConnector("alpaca", BudgetManager({"alpaca": {"rpm": 10, "daily_cap": 100}}))
    tiingo = _BudgetedBackfillConnector("tiingo", tiingo_budget)
    service = DataNodeService(
        db=db,
        connectors={"alpaca": alpaca, "tiingo": tiingo},
        auditor=PartitionAuditor(db=db, calendar_store=ExchangeCalendarStore(root=tmp_path / "reference" / "calendars")),
        curator=Curator(),
        paths=DataNodePaths(root=tmp_path),
        source_name="alpaca",
    )
    db.upsert_planner_task(
        task_key="canonical::AAPL::budget",
        task_family="canonical_bars",
        planner_group="canonical_bars_backlog",
        dataset="equities_eod",
        tier="A",
        priority=5,
        start_date="2025-12-16",
        end_date="2026-01-14",
        symbols=["AAPL"],
        eligible_vendors=["tiingo", "alpaca"],
        payload={"scope_kind": "symbol_range", "trading_days": ["2025-12-16"]},
    )
    db.update_planner_task_progress(
        task_key="canonical::AAPL::budget",
        expected_units=1,
        completed_units=0,
        remaining_units=1,
        remaining_symbols=["AAPL"],
        state={"scope_kind": "symbol_range"},
    )

    assert service._canonical_runtime._lease_canonical_batch("tiingo") == []
    alpaca_batch = service._canonical_runtime._lease_canonical_batch("alpaca")
    assert len(alpaca_batch) == 1
    assert alpaca_batch[0].task_key == "canonical::AAPL::budget"


def test_release_budget_blocked_canonical_tasks_clears_task_backoff_for_spendable_alternate(tmp_path: Path) -> None:
    db = DataNodeDB(tmp_path / "control" / "node.sqlite")
    tiingo_budget = BudgetManager({"tiingo": {"rpm": 1, "daily_cap": 10}})
    tiingo_budget.record_spend("tiingo", task_kind="FORWARD")
    alpaca = _BudgetedBackfillConnector("alpaca", BudgetManager({"alpaca": {"rpm": 10, "daily_cap": 100}}))
    tiingo = _BudgetedBackfillConnector("tiingo", tiingo_budget)
    service = DataNodeService(
        db=db,
        connectors={"alpaca": alpaca, "tiingo": tiingo},
        auditor=PartitionAuditor(db=db, calendar_store=ExchangeCalendarStore(root=tmp_path / "reference" / "calendars")),
        curator=Curator(),
        paths=DataNodePaths(root=tmp_path),
        source_name="alpaca",
    )
    db.upsert_planner_task(
        task_key="canonical::AAPL::release",
        task_family="canonical_bars",
        planner_group="canonical_bars_backlog",
        dataset="equities_eod",
        tier="A",
        priority=5,
        start_date="2025-12-16",
        end_date="2026-01-14",
        symbols=["AAPL"],
        eligible_vendors=["tiingo", "alpaca"],
        payload={"scope_kind": "symbol_range", "trading_days": ["2025-12-16"]},
    )
    db.mark_planner_task_partial(
        "canonical::AAPL::release",
        error="tiingo: budget exhausted for vendor=tiingo",
        backoff_minutes=30,
    )
    db.mark_vendor_attempt_failed(
        task_key="canonical::AAPL::release",
        vendor="tiingo",
        error="budget exhausted for vendor=tiingo",
        backoff_minutes=30,
    )

    released = service._release_budget_blocked_canonical_tasks()
    task = next(task for task in db.fetch_planner_tasks(task_family="canonical_bars") if task.task_key == "canonical::AAPL::release")

    assert released == 1
    assert task.next_eligible_at is None
    assert task.status == "PARTIAL"
    assert task.last_error == "released budget-blocked canonical task to alternate vendor"


def test_release_budget_blocked_canonical_tasks_clears_task_backoff_for_reusable_success_vendor(tmp_path: Path) -> None:
    db = DataNodeDB(tmp_path / "control" / "node.sqlite")
    alpaca = _BudgetedBackfillConnector("alpaca", BudgetManager({"alpaca": {"rpm": 10, "daily_cap": 100}}))
    tiingo_budget = BudgetManager({"tiingo": {"rpm": 1, "daily_cap": 10}})
    tiingo_budget.record_spend("tiingo", task_kind="FORWARD")
    tiingo = _BudgetedBackfillConnector("tiingo", tiingo_budget)
    service = DataNodeService(
        db=db,
        connectors={"alpaca": alpaca, "tiingo": tiingo},
        auditor=PartitionAuditor(db=db, calendar_store=ExchangeCalendarStore(root=tmp_path / "reference" / "calendars")),
        curator=Curator(),
        paths=DataNodePaths(root=tmp_path),
        source_name="alpaca",
    )
    db.upsert_planner_task(
        task_key="canonical::TECK::release",
        task_family="canonical_bars",
        planner_group="canonical_bars_backlog",
        dataset="equities_eod",
        tier="A",
        priority=5,
        start_date="2025-12-16",
        end_date="2026-01-14",
        symbols=["TECK"],
        eligible_vendors=["alpaca", "tiingo"],
        payload={"scope_kind": "symbol_range", "trading_days": ["2025-12-16"]},
    )
    db.update_planner_task_progress(
        task_key="canonical::TECK::release",
        expected_units=1,
        completed_units=0,
        remaining_units=1,
        remaining_symbols=["TECK"],
        state={"scope_kind": "symbol_range"},
    )
    attempt = db.lease_vendor_attempt(
        task_key="canonical::TECK::release",
        task_family="canonical_bars",
        planner_group="canonical_bars_backlog",
        vendor="alpaca",
        lease_owner="worker-0",
        payload={"symbols": ["TECK"], "start_date": "2025-12-16", "end_date": "2026-01-14"},
    )
    assert attempt is not None
    db.mark_vendor_attempt_success(task_key="canonical::TECK::release", vendor="alpaca", rows_returned=20)
    db.mark_planner_task_partial(
        "canonical::TECK::release",
        error="tiingo: budget exhausted for vendor=tiingo",
        backoff_minutes=30,
    )
    db.mark_vendor_attempt_failed(
        task_key="canonical::TECK::release",
        vendor="tiingo",
        error="budget exhausted for vendor=tiingo",
        backoff_minutes=30,
    )

    released = service._release_budget_blocked_canonical_tasks()
    task = next(task for task in db.fetch_planner_tasks(task_family="canonical_bars") if task.task_key == "canonical::TECK::release")

    assert released == 1
    assert task.next_eligible_at is None
    assert task.status == "PARTIAL"
    assert task.last_error == "released budget-blocked canonical task to alternate vendor"


def test_backfill_lane_widths_prioritize_alpaca_before_tiingo(tmp_path: Path) -> None:
    runtime = CanonicalRuntime(
        db=DataNodeDB(tmp_path / "control" / "node.sqlite"),
        connectors={"alpaca": _BudgetedBackfillConnector("alpaca", BudgetManager({"alpaca": {"rpm": 10, "daily_cap": 100}})),
                    "tiingo": _BudgetedBackfillConnector("tiingo", BudgetManager({"tiingo": {"rpm": 10, "daily_cap": 100}}))},
        paths=DataNodePaths(root=tmp_path),
        source_name="alpaca",
        capability_audit_state={},
        worker_id="test",
        default_symbols_getter=lambda: [],
        stop_requested=lambda: False,
        write_raw_partition_fn=lambda frame, source_name: [],
    )

    assert list(runtime._backfill_lane_widths())[:2] == ["alpaca", "tiingo"]


def test_write_raw_partition_serializes_concurrent_same_day_writes(tmp_path: Path, monkeypatch) -> None:
    db = DataNodeDB(tmp_path / "control" / "node.sqlite")
    service = DataNodeService(
        db=db,
        connectors={"alpaca": _NoopConnector()},
        auditor=PartitionAuditor(db=db, calendar_store=ExchangeCalendarStore(root=tmp_path / "reference" / "calendars")),
        curator=Curator(),
        paths=DataNodePaths(root=tmp_path),
        source_name="alpaca",
    )
    entered_stale_read = threading.Event()
    original_merge = service._canonical_runtime._merge_partition_frame

    def delayed_merge(*, partition: Path, frame: pd.DataFrame) -> pd.DataFrame:
        if not (partition / "data.parquet").exists() and not entered_stale_read.is_set():
            entered_stale_read.set()
            time.sleep(0.2)
        return original_merge(partition=partition, frame=frame)

    monkeypatch.setattr(service._canonical_runtime, "_merge_partition_frame", delayed_merge)

    base_row = {
        "date": "2025-12-16",
        "open": 10.0,
        "high": 11.0,
        "low": 9.0,
        "close": 10.5,
        "vwap": pd.NA,
        "volume": 100,
        "trade_count": pd.NA,
        "ingested_at": pd.Timestamp.now(tz="UTC"),
        "source_name": "alpaca",
        "source_uri": "/bars",
        "vendor_ts": pd.Timestamp("2025-12-16"),
    }
    frame_a = pd.DataFrame([{**base_row, "symbol": "AAPL"}])
    frame_b = pd.DataFrame([{**base_row, "symbol": "MSFT"}])

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_a = executor.submit(service._write_raw_partition, frame_a, source_name="alpaca")
        assert entered_stale_read.wait(timeout=5)
        future_b = executor.submit(service._write_raw_partition, frame_b, source_name="alpaca")
        future_a.result(timeout=5)
        future_b.result(timeout=5)

    output = pd.read_parquet(tmp_path / "data" / "raw" / "equities_bars" / "date=2025-12-16" / "data.parquet")
    assert set(output["symbol"]) == {"AAPL", "MSFT"}


def test_load_corp_actions_reference_skips_corrupt_parquet(tmp_path: Path) -> None:
    db = DataNodeDB(tmp_path / "control" / "node.sqlite")
    service = DataNodeService(
        db=db,
        connectors={"alpaca": _NoopConnector()},
        auditor=PartitionAuditor(db=db, calendar_store=ExchangeCalendarStore(root=tmp_path / "reference" / "calendars")),
        curator=Curator(),
        paths=DataNodePaths(root=tmp_path),
        source_name="alpaca",
    )
    reference_root = tmp_path / "data" / "reference"
    reference_root.mkdir(parents=True, exist_ok=True)
    (reference_root / "corp_actions.parquet").write_text("not parquet")
    pd.DataFrame(
        [{"symbol": "AAPL", "ex_date": "2025-01-02", "amount": 1.0, "ratio": pd.NA, "event_type": "dividend", "source": "alpha_vantage"}]
    ).to_parquet(reference_root / "dividends.parquet", index=False)

    frame = service.load_corp_actions_reference()

    assert not frame.empty
    assert sorted(frame["symbol"].astype(str).str.upper().unique().tolist()) == ["AAPL"]


def test_load_corp_actions_reference_ignores_unrelated_reference_parquet(tmp_path: Path) -> None:
    db = DataNodeDB(tmp_path / "control" / "node.sqlite")
    service = DataNodeService(
        db=db,
        connectors={"alpaca": _NoopConnector()},
        auditor=PartitionAuditor(db=db, calendar_store=ExchangeCalendarStore(root=tmp_path / "reference" / "calendars")),
        curator=Curator(),
        paths=DataNodePaths(root=tmp_path),
        source_name="alpaca",
    )
    reference_root = tmp_path / "data" / "reference"
    reference_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"symbol": "AAPL", "ex_date": "2025-01-02", "amount": 1.0, "ratio": pd.NA, "event_type": "dividend", "source": "alpha_vantage"}]
    ).to_parquet(reference_root / "dividends.parquet", index=False)
    unrelated_path = reference_root / "fred_vintagedates.parquet"
    unrelated_path.write_text("not parquet", encoding="utf-8")

    frame = service.load_corp_actions_reference()

    assert sorted(frame["symbol"].astype(str).str.upper().unique().tolist()) == ["AAPL"]
    assert unrelated_path.exists()


def test_curate_dates_reads_only_changed_raw_partitions(tmp_path: Path) -> None:
    db = DataNodeDB(tmp_path / "control" / "node.sqlite")
    service = DataNodeService(
        db=db,
        connectors={"alpaca": _NoopConnector()},
        auditor=PartitionAuditor(db=db, calendar_store=ExchangeCalendarStore(root=tmp_path / "reference" / "calendars")),
        curator=Curator(),
        paths=DataNodePaths(root=tmp_path),
        source_name="alpaca",
    )
    raw_root = tmp_path / "data" / "raw" / "equities_bars"
    for day, symbol in [("2025-01-02", "AAPL"), ("2025-01-03", "MSFT")]:
        partition = raw_root / f"date={day}"
        partition.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "date": day,
                    "symbol": symbol,
                    "open": 10.0,
                    "high": 11.0,
                    "low": 9.0,
                    "close": 10.5,
                    "volume": 100,
                    "source_name": "alpaca",
                }
            ]
        ).to_parquet(partition / "data.parquet", index=False)

    result = service.curate_dates(corp_actions=pd.DataFrame(), changed_dates=["2025-01-03"])

    assert sorted(pd.to_datetime(result.frame["date"]).dt.strftime("%Y-%m-%d").unique().tolist()) == ["2025-01-03"]


def test_curate_dates_quarantines_corrupt_changed_partition_and_continues(tmp_path: Path) -> None:
    db = DataNodeDB(tmp_path / "control" / "node.sqlite")
    service = DataNodeService(
        db=db,
        connectors={"alpaca": _NoopConnector()},
        auditor=PartitionAuditor(db=db, calendar_store=ExchangeCalendarStore(root=tmp_path / "reference" / "calendars")),
        curator=Curator(),
        paths=DataNodePaths(root=tmp_path),
        source_name="alpaca",
    )
    raw_root = tmp_path / "data" / "raw" / "equities_bars"
    good_partition = raw_root / "date=2025-01-03"
    bad_partition = raw_root / "date=2025-01-04"
    good_partition.mkdir(parents=True, exist_ok=True)
    bad_partition.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "date": "2025-01-03",
                "symbol": "MSFT",
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.5,
                "volume": 100,
                "source_name": "alpaca",
            }
        ]
    ).to_parquet(good_partition / "data.parquet", index=False)
    (bad_partition / "data.parquet").write_text("not parquet", encoding="utf-8")

    result = service.curate_dates(corp_actions=pd.DataFrame(), changed_dates=["2025-01-03", "2025-01-04"])

    assert sorted(pd.to_datetime(result.frame["date"]).dt.strftime("%Y-%m-%d").unique().tolist()) == ["2025-01-03"]
    assert not (bad_partition / "data.parquet").exists()
    assert list(bad_partition.glob("data.corrupt.*.parquet"))


def test_curate_dates_streams_multiple_changed_dates_without_full_batch_concat(tmp_path: Path) -> None:
    db = DataNodeDB(tmp_path / "control" / "node.sqlite")
    service = DataNodeService(
        db=db,
        connectors={"alpaca": _NoopConnector()},
        auditor=PartitionAuditor(db=db, calendar_store=ExchangeCalendarStore(root=tmp_path / "reference" / "calendars")),
        curator=Curator(),
        paths=DataNodePaths(root=tmp_path),
        source_name="alpaca",
    )
    raw_root = tmp_path / "data" / "raw" / "equities_bars"
    for day, symbol in [("2025-01-02", "AAPL"), ("2025-01-03", "MSFT")]:
        partition = raw_root / f"date={day}"
        partition.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "date": day,
                    "symbol": symbol,
                    "open": 10.0,
                    "high": 11.0,
                    "low": 9.0,
                    "close": 10.5,
                    "volume": 100,
                    "source_name": "alpaca",
                }
            ]
        ).to_parquet(partition / "data.parquet", index=False)

    result = service.curate_dates(corp_actions=pd.DataFrame(), changed_dates=["2025-01-02", "2025-01-03"])

    assert sorted(pd.to_datetime(result.frame["date"]).dt.strftime("%Y-%m-%d").unique().tolist()) == ["2025-01-02", "2025-01-03"]


def test_run_cluster_forever_reruns_backfill_when_same_day_queue_is_reseeded(tmp_path: Path) -> None:
    db = DataNodeDB(tmp_path / "control" / "node.sqlite")
    db.enqueue_task("equities_eod", None, "2026-04-02", "2026-04-02", "GAP", 1)
    service = DataNodeService(
        db=db,
        connectors={"alpaca": _NoopConnector()},
        auditor=PartitionAuditor(db=db, calendar_store=ExchangeCalendarStore(root=tmp_path / "reference" / "calendars")),
        curator=Curator(),
        paths=DataNodePaths(root=tmp_path),
    )
    coordinator = _ClusterCoordinatorStub()
    coordinator.allowed = set()
    coordinator.last_success["backfill"] = {"bucket": "2026-04-02", "updated_at": "2026-04-02T16:11:09+00:00"}
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
        now_fn=lambda: datetime.fromisoformat("2026-04-02T20:35:00+00:00"),
        sleep_fn=lambda _seconds: None,
    )

    assert calls == ["process_backfill_queue"]
    assert "renew:singleton::backfill::2026-04-02" in coordinator._lease_calls


def test_run_cluster_forever_renews_backfill_lease_while_backlog_remains(tmp_path: Path) -> None:
    db = DataNodeDB(tmp_path / "control" / "node.sqlite")
    db.enqueue_task("equities_eod", None, "2026-04-02", "2026-04-02", "GAP", 1)
    service = DataNodeService(
        db=db,
        connectors={"alpaca": _NoopConnector()},
        auditor=PartitionAuditor(db=db, calendar_store=ExchangeCalendarStore(root=tmp_path / "reference" / "calendars")),
        curator=Curator(),
        paths=DataNodePaths(root=tmp_path),
    )
    coordinator = _ClusterCoordinatorStub()
    coordinator.allowed = set()
    coordinator.last_success["backfill"] = {"bucket": "2026-04-02", "updated_at": "2999-04-02T16:11:09+00:00"}
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
        now_fn=lambda: datetime.fromisoformat("2026-04-02T20:35:00+00:00"),
        sleep_fn=lambda _seconds: None,
    )

    assert calls == ["process_backfill_queue"]
    assert "renew:singleton::backfill::2026-04-02" in coordinator._lease_calls


def test_run_cluster_forever_curates_only_trading_date_during_audit_curate(tmp_path: Path) -> None:
    db = DataNodeDB(tmp_path / "control" / "node.sqlite")
    service = DataNodeService(
        db=db,
        connectors={"alpaca": _NoopConnector()},
        auditor=PartitionAuditor(db=db, calendar_store=ExchangeCalendarStore(root=tmp_path / "reference" / "calendars")),
        curator=Curator(),
        paths=DataNodePaths(root=tmp_path),
    )
    coordinator = _ClusterCoordinatorStub()
    coordinator.allowed = {"audit_curate"}
    captured: dict[str, object] = {}

    service._collect_cluster_shard = lambda **kwargs: None  # type: ignore[method-assign]
    service._auxiliary_runtime._run_cluster_auxiliary_tasks = lambda **kwargs: None  # type: ignore[method-assign]
    service.sync_partition_status = lambda: tmp_path / "data" / "qc" / "partition_status.parquet"  # type: ignore[method-assign]
    service.auditor.audit_range = lambda **kwargs: []  # type: ignore[method-assign]
    service.load_corp_actions_reference = lambda: pd.DataFrame()  # type: ignore[method-assign]

    def capture_curate(*, corp_actions=None, changed_dates=None):  # noqa: ANN001
        captured["changed_dates"] = changed_dates
        service.stop()
        return service_module.CuratorResult(frame=pd.DataFrame(), adjustment_log=pd.DataFrame())

    service.curate_dates = capture_curate  # type: ignore[method-assign]

    service.run_cluster_forever(
        coordinator=coordinator,  # type: ignore[arg-type]
        symbols=["AAPL", "MSFT"],
        exchange="XNYS",
        collection_time_et="16:30",
        maintenance_hour_local=23,
        poll_seconds=0.0,
        now_fn=lambda: datetime.fromisoformat("2026-04-07T20:35:00+00:00"),
        sleep_fn=lambda _seconds: None,
    )

    assert captured["changed_dates"] == ["2026-04-07"]


def test_run_cluster_forever_backfill_curates_only_changed_dates(tmp_path: Path) -> None:
    db = DataNodeDB(tmp_path / "control" / "node.sqlite")
    db.enqueue_task("equities_eod", None, "2026-04-02", "2026-04-02", "GAP", 1)
    service = DataNodeService(
        db=db,
        connectors={"alpaca": _NoopConnector()},
        auditor=PartitionAuditor(db=db, calendar_store=ExchangeCalendarStore(root=tmp_path / "reference" / "calendars")),
        curator=Curator(),
        paths=DataNodePaths(root=tmp_path),
    )
    coordinator = _ClusterCoordinatorStub()
    captured: dict[str, object] = {}

    service.load_corp_actions_reference = lambda: pd.DataFrame()  # type: ignore[method-assign]
    service.sync_partition_status = lambda: tmp_path / "data" / "qc" / "partition_status.parquet"  # type: ignore[method-assign]

    def process_backfill_queue() -> list[str]:
        return ["2026-04-01", "2026-04-02"]

    def capture_curate(*, corp_actions=None, changed_dates=None):  # noqa: ANN001
        captured["changed_dates"] = changed_dates
        service.stop()
        return service_module.CuratorResult(frame=pd.DataFrame(), adjustment_log=pd.DataFrame())

    service.process_backfill_queue = process_backfill_queue  # type: ignore[method-assign]
    service.curate_dates = capture_curate  # type: ignore[method-assign]

    service.run_cluster_forever(
        coordinator=coordinator,  # type: ignore[arg-type]
        symbols=["AAPL"],
        exchange="XNYS",
        collection_time_et="23:59",
        maintenance_hour_local=23,
        poll_seconds=0.0,
        now_fn=lambda: datetime.fromisoformat("2026-04-02T23:35:00+00:00"),
        sleep_fn=lambda _seconds: None,
    )

    assert captured["changed_dates"] == ["2026-04-01", "2026-04-02"]


def test_process_planner_queue_skips_auxiliary_lanes_while_canonical_backlog_exists(tmp_path: Path) -> None:
    db = DataNodeDB(tmp_path / "control" / "node.sqlite")
    db.bulk_upsert_planner_tasks(
        [
            {
                "task_key": "canonical_bars::equities_eod::00000::0000::2026-03-01::2026-03-31",
                "task_family": "canonical_bars",
                "planner_group": "canonical_bars_backlog",
                "dataset": "equities_eod",
                "tier": "A",
                "priority": 10,
                "start_date": "2026-03-01",
                "end_date": "2026-03-31",
                "symbols": ["AAPL"],
                "eligible_vendors": ["alpaca"],
                "output_name": "equities_bars",
                "payload": {"scope_kind": "symbol_range", "trading_days": ["2026-03-31"]},
            }
        ]
    )
    service = DataNodeService(
        db=db,
        connectors={"alpaca": _NoopConnector(), "alpha_vantage": _NoopConnector()},
        auditor=PartitionAuditor(db=db, calendar_store=ExchangeCalendarStore(root=tmp_path / "reference" / "calendars")),
        curator=Curator(),
        paths=DataNodePaths(root=tmp_path),
    )
    service.default_symbols = ["AAPL"]
    calls: list[str] = []

    service._seed_planner_tasks = lambda **kwargs: None  # type: ignore[method-assign]
    service._canonical_runtime._backfill_lane_widths = lambda: {"alpaca": 1}  # type: ignore[method-assign]
    service._aux_lane_widths = lambda **kwargs: {"alpha_vantage": 1}  # type: ignore[method-assign]
    service._drain_canonical_lane = lambda vendor, exchange: calls.append(f"canonical:{vendor}") or []  # type: ignore[method-assign]
    service._drain_auxiliary_lane = lambda vendor: calls.append(f"aux:{vendor}") or []  # type: ignore[method-assign]

    service.process_planner_queue(exchange="XNYS")

    assert calls == ["canonical:alpaca"]


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
    service._ensure_planner_backlog_seeded = lambda **kwargs: None  # type: ignore[method-assign]

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

    assert "macro" in calls
    assert "price_checks" in calls


def test_collect_macro_data_persists_vintage_reference(tmp_path: Path) -> None:
    db = DataNodeDB(tmp_path / "control" / "node.sqlite")
    service = DataNodeService(
        db=db,
        connectors={"fred": _FredConnector()},
        auditor=PartitionAuditor(db=db, calendar_store=ExchangeCalendarStore(root=tmp_path / "reference" / "calendars")),
        curator=Curator(),
        paths=DataNodePaths(root=tmp_path),
    )

    outputs = service.collect_macro_data(["DGS10"], "2026-03-01", "2026-03-31")

    assert tmp_path / "data" / "reference" / "fred_vintagedates.parquet" in outputs
    vintages = pd.read_parquet(tmp_path / "data" / "reference" / "fred_vintagedates.parquet")
    assert vintages["series_id"].tolist() == ["DGS10"]


def test_collect_macro_data_persists_vintages_even_when_observations_are_empty(tmp_path: Path) -> None:
    class _VintagesOnlyFredConnector:
        def fetch(self, dataset: str, symbols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
            if dataset == "macros_treasury":
                return pd.DataFrame(columns=["series_id", "observation_date", "value", "vintage_date", "ingested_at"])
            if dataset == "vintagedates":
                return pd.DataFrame([{"series_id": symbols[0], "vintage_date": end_date}])
            raise ValueError(dataset)

    db = DataNodeDB(tmp_path / "control" / "node.sqlite")
    service = DataNodeService(
        db=db,
        connectors={"fred": _VintagesOnlyFredConnector()},
        auditor=PartitionAuditor(db=db, calendar_store=ExchangeCalendarStore(root=tmp_path / "reference" / "calendars")),
        curator=Curator(),
        paths=DataNodePaths(root=tmp_path),
    )

    outputs = service.collect_macro_data(["DGS10"], "2026-03-01", "2026-03-31")

    assert tmp_path / "data" / "reference" / "fred_vintagedates.parquet" in outputs
    assert not (tmp_path / "data" / "raw" / "macros_fred" / "series=DGS10" / "data.parquet").exists()
    vintages = pd.read_parquet(tmp_path / "data" / "reference" / "fred_vintagedates.parquet")
    assert vintages["series_id"].tolist() == ["DGS10"]


def test_run_cross_vendor_price_checks_excludes_finnhub(tmp_path: Path) -> None:
    db = DataNodeDB(tmp_path / "control" / "node.sqlite")
    service = DataNodeService(
        db=db,
        connectors={
            "alpaca": _PriceCheckConnector("alpaca"),
            "massive": _PriceCheckConnector("massive"),
            "twelve_data": _PriceCheckConnector("twelve_data"),
            "tiingo": _PriceCheckConnector("tiingo"),
            "finnhub": _PriceCheckConnector("finnhub"),
        },
        auditor=PartitionAuditor(db=db, calendar_store=ExchangeCalendarStore(root=tmp_path / "reference" / "calendars")),
        curator=Curator(),
        paths=DataNodePaths(root=tmp_path),
        source_name="alpaca",
    )

    output = service.run_cross_vendor_price_checks(trading_date="2026-04-02", sample_symbols=["AAPL", "MSFT"])
    frame = pd.read_parquet(output)

    assert sorted(frame["vendor"].unique().tolist()) == ["massive", "tiingo"]


def test_run_cross_vendor_price_checks_logs_backup_fetch_failures(tmp_path: Path, caplog) -> None:
    class _FailingPriceCheckConnector:
        vendor_name = "massive"

        def fetch(self, dataset: str, symbols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
            raise TemporaryConnectorError("massive unavailable")

    db = DataNodeDB(tmp_path / "control" / "node.sqlite")
    service = DataNodeService(
        db=db,
        connectors={
            "alpaca": _PriceCheckConnector("alpaca"),
            "massive": _FailingPriceCheckConnector(),
            "tiingo": _PriceCheckConnector("tiingo"),
            "finnhub": _PriceCheckConnector("finnhub"),
        },
        auditor=PartitionAuditor(db=db, calendar_store=ExchangeCalendarStore(root=tmp_path / "reference" / "calendars")),
        curator=Curator(),
        paths=DataNodePaths(root=tmp_path),
        source_name="alpaca",
    )

    with caplog.at_level("WARNING"):
        output = service.run_cross_vendor_price_checks(trading_date="2026-04-02", sample_symbols=["AAPL", "MSFT"])

    frame = pd.read_parquet(output)
    assert sorted(frame["vendor"].unique().tolist()) == ["tiingo"]
    assert any(
        "price_check_backup_failed" in record.message and "vendor=massive" in record.message
        for record in caplog.records
    )


def test_aux_lane_widths_hold_bar_first_vendors_back_when_canonical_backlog_exists(tmp_path: Path) -> None:
    db = DataNodeDB(tmp_path / "control" / "node.sqlite")
    service = DataNodeService(
        db=db,
        connectors={
            "alpaca": _NoopConnector(),
            "twelve_data": _NoopConnector(),
            "massive": _NoopConnector(),
            "alpha_vantage": _NoopConnector(),
            "fred": _FredConnector(),
            "fmp": _NoopConnector(),
            "sec_edgar": _NoopConnector(),
            "finnhub": _NoopConnector(),
        },
        auditor=PartitionAuditor(db=db, calendar_store=ExchangeCalendarStore(root=tmp_path / "reference" / "calendars")),
        curator=Curator(),
        paths=DataNodePaths(root=tmp_path),
    )
    service.default_symbols = ["AAPL"]
    db.bulk_upsert_planner_tasks(
        [
            {
                "task_key": "canonical_bars::equities_eod::00000::0000::2026-03-01::2026-03-31",
                "task_family": "canonical_bars",
                "planner_group": "canonical_bars_backlog",
                "dataset": "equities_eod",
                "tier": "A",
                "priority": 10,
                "start_date": "2026-03-01",
                "end_date": "2026-03-31",
                "symbols": ["AAPL"],
                "eligible_vendors": ["alpaca", "twelve_data", "massive"],
                "output_name": "equities_bars",
                "payload": {"scope_kind": "symbol_range", "trading_days": ["2026-03-31"]},
            }
        ]
    )

    widths = service._aux_lane_widths(task_kinds={"REFERENCE", "EVENT", "MACRO"})

    assert "alpaca" not in widths
    assert widths["twelve_data"] == 1
    assert "massive" not in widths
    assert widths["alpha_vantage"] == 1
    assert widths["fred"] == 2
    assert widths["fmp"] == 1
    assert widths["sec_edgar"] == 2


def test_planned_auxiliary_work_materializes_reference_and_macro_tasks(tmp_path: Path) -> None:
    db = DataNodeDB(tmp_path / "control" / "node.sqlite")
    reference_root = tmp_path / "data" / "reference"
    reference_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"ticker": "AAPL", "cik_str": "320193"}]).to_parquet(reference_root / "sec_company_tickers.parquet", index=False)
    service = DataNodeService(
        db=db,
        connectors={"alpha_vantage": _NoopConnector(), "fred": _FredConnector(), "sec_edgar": _NoopConnector()},
        auditor=PartitionAuditor(db=db, calendar_store=ExchangeCalendarStore(root=tmp_path / "reference" / "calendars")),
        curator=Curator(),
        paths=DataNodePaths(root=tmp_path),
        stage_years=5,
    )
    service.default_symbols = ["AAPL", "MSFT"]

    macro_series, reference_jobs = service._auxiliary_runtime._planned_auxiliary_work(trading_date="2026-03-31")

    assert "DGS10" in macro_series
    datasets = {job["dataset"] for job in reference_jobs}
    assert "listings" in datasets
    assert "filing_index" in datasets


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


def test_build_canonical_coverage_index_streams_partitions_and_skips_unreadable_ones(tmp_path: Path) -> None:
    db = DataNodeDB(tmp_path / "control" / "node.sqlite")
    service = DataNodeService(
        db=db,
        connectors={"alpaca": _NoopConnector()},
        auditor=PartitionAuditor(db=db, calendar_store=ExchangeCalendarStore(root=tmp_path / "reference" / "calendars")),
        curator=Curator(),
        paths=DataNodePaths(root=tmp_path),
    )
    good_partition = tmp_path / "data" / "raw" / "equities_bars" / "date=2026-03-31"
    bad_partition = tmp_path / "data" / "raw" / "equities_bars" / "date=2026-04-01"
    good_partition.mkdir(parents=True, exist_ok=True)
    bad_partition.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"symbol": "AAPL", "date": "2026-03-31", "close": 100.0},
            {"symbol": "aapl", "date": "2026-03-31", "close": 101.0},
            {"symbol": "MSFT", "date": "2026-03-31", "close": 200.0},
        ]
    ).to_parquet(
        good_partition / "data.parquet",
        index=False,
    )
    (bad_partition / "data.parquet").write_text("not parquet", encoding="utf-8")

    coverage = service._canonical_runtime._build_canonical_coverage_index(trading_days=["2026-03-31", "2026-04-01"])

    assert coverage == {"AAPL": {"2026-03-31"}, "MSFT": {"2026-03-31"}}


def test_build_canonical_coverage_index_logs_unreadable_partition_once(tmp_path: Path, caplog) -> None:
    db = DataNodeDB(tmp_path / "control" / "node.sqlite")
    service = DataNodeService(
        db=db,
        connectors={"alpaca": _NoopConnector()},
        auditor=PartitionAuditor(db=db, calendar_store=ExchangeCalendarStore(root=tmp_path / "reference" / "calendars")),
        curator=Curator(),
        paths=DataNodePaths(root=tmp_path),
    )
    bad_partition = tmp_path / "data" / "raw" / "equities_bars" / "date=2026-04-01"
    bad_partition.mkdir(parents=True, exist_ok=True)
    bad_path = bad_partition / "data.parquet"
    bad_path.write_text("not parquet", encoding="utf-8")

    with caplog.at_level("WARNING"):
        first = service._canonical_runtime._build_canonical_coverage_index(trading_days=["2026-04-01"])
        second = service._canonical_runtime._build_canonical_coverage_index(trading_days=["2026-04-01"])

    assert first == {}
    assert second == {}
    matches = [record for record in caplog.records if "skipping_unreadable_raw_partition" in record.message]
    assert len(matches) == 1
    assert str(bad_path) in matches[0].message


def test_merge_partition_frame_quarantines_zero_byte_existing_partition(tmp_path: Path) -> None:
    db = DataNodeDB(tmp_path / "control" / "node.sqlite")
    service = DataNodeService(
        db=db,
        connectors={"alpaca": _NoopConnector()},
        auditor=PartitionAuditor(db=db, calendar_store=ExchangeCalendarStore(root=tmp_path / "reference" / "calendars")),
        curator=Curator(),
        paths=DataNodePaths(root=tmp_path),
    )
    partition = tmp_path / "data" / "raw" / "equities_bars" / "date=2026-04-01"
    partition.mkdir(parents=True, exist_ok=True)
    existing_path = partition / "data.parquet"
    existing_path.write_bytes(b"")
    incoming = pd.DataFrame(
        [
            {
                "date": "2026-04-01",
                "symbol": "AAPL",
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.5,
                "volume": 100,
                "source_name": "alpaca",
            }
        ]
    )

    merged = service._canonical_runtime._merge_partition_frame(partition=partition, frame=incoming)

    assert len(merged) == 1
    assert merged.iloc[0]["symbol"] == "AAPL"
    assert not existing_path.exists()
    quarantined = list(partition.glob("data.corrupt.*.parquet"))
    assert quarantined


def test_run_cluster_auxiliary_tasks_uses_daily_reference_bucket_until_core_ready(tmp_path: Path) -> None:
    db = DataNodeDB(tmp_path / "control" / "node.sqlite")
    service = DataNodeService(
        db=db,
        connectors={"alpaca": _NoopConnector(), "fred": _NoopConnector()},
        auditor=PartitionAuditor(db=db, calendar_store=ExchangeCalendarStore(root=tmp_path / "reference" / "calendars")),
        curator=Curator(),
        paths=DataNodePaths(root=tmp_path),
    )
    coordinator = _ClusterCoordinatorStub()
    coordinator.allowed = {"reference"}
    service.collect_reference_data = lambda *args, **kwargs: []  # type: ignore[method-assign]

    service._auxiliary_runtime._run_cluster_auxiliary_tasks(
        coordinator=coordinator,  # type: ignore[arg-type]
        trading_date="2026-03-31",
        current_et=datetime.fromisoformat("2026-03-31T18:00:00+00:00"),
        macro_series_ids=["DGS10"],
        reference_jobs=[{"source": "alpaca", "dataset": "assets", "symbols": [], "output_name": "alpaca_assets"}],
        price_check_symbols=["AAPL"],
        source_name=service.source_name,
    )

    assert "reference:2026-03-31" in coordinator._lease_calls


def test_collect_reference_data_persists_partial_results_before_budget_failure(tmp_path: Path) -> None:
    db = DataNodeDB(tmp_path / "control" / "node.sqlite")
    service = DataNodeService(
        db=db,
        connectors={"alpaca": _NoopConnector(), "alpha_vantage": _PartialReferenceConnector()},
        auditor=PartitionAuditor(db=db, calendar_store=ExchangeCalendarStore(root=tmp_path / "reference" / "calendars")),
        curator=Curator(),
        paths=DataNodePaths(root=tmp_path),
    )

    result = service.collect_reference_data(
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

    stored = pd.read_parquet(tmp_path / "data" / "reference" / "corp_actions.parquet")
    assert stored["symbol"].tolist() == ["AAPL"]
    assert result.failures == []
    assert result.deferred


def test_collect_reference_data_rebuilds_security_master_outputs(tmp_path: Path) -> None:
    db = DataNodeDB(tmp_path / "control" / "node.sqlite")
    service = DataNodeService(
        db=db,
        connectors={"alpaca": _NoopConnector(), "alpha_vantage": _ListingConnector(), "fmp": _DelistingConnector()},
        auditor=PartitionAuditor(db=db, calendar_store=ExchangeCalendarStore(root=tmp_path / "reference" / "calendars")),
        curator=Curator(),
        paths=DataNodePaths(root=tmp_path),
    )

    result = service.collect_reference_data(
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

    assert tmp_path / "data" / "reference" / "listing_history.parquet" in result.outputs
    assert tmp_path / "data" / "reference" / "ticker_changes.parquet" in result.outputs


def test_process_auxiliary_planner_task_marks_reference_success_and_deferred_consistently(tmp_path: Path) -> None:
    db = DataNodeDB(tmp_path / "control" / "node.sqlite")
    service = DataNodeService(
        db=db,
        connectors={"alpha_vantage": _PartialReferenceConnector()},
        auditor=PartitionAuditor(db=db, calendar_store=ExchangeCalendarStore(root=tmp_path / "reference" / "calendars")),
        curator=Curator(),
        paths=DataNodePaths(root=tmp_path),
    )
    for symbol in ["AAPL", "MSFT"]:
        db.upsert_planner_task(
            task_key=f"aux::{symbol}",
            task_family="corp_actions",
            planner_group="reference_events_backlog",
            dataset="corp_actions",
            tier="A",
            priority=5,
            start_date="2025-01-07",
            end_date="2025-01-07",
            symbols=[symbol],
            eligible_vendors=["alpha_vantage"],
            output_name="corp_actions",
            payload={"scope_kind": "symbol_range"},
        )
        db.update_planner_task_progress(
            task_key=f"aux::{symbol}",
            expected_units=1,
            completed_units=0,
            remaining_units=1,
            remaining_symbols=[symbol],
            state={"scope_kind": "symbol_range"},
        )

    success_task = db.lease_planner_task_by_key(task_key="aux::AAPL", lease_owner=service.worker_id)
    deferred_task = db.lease_planner_task_by_key(task_key="aux::MSFT", lease_owner=service.worker_id)

    assert success_task is not None
    assert deferred_task is not None

    service._auxiliary_runtime._process_auxiliary_planner_task(success_task, "alpha_vantage")
    service._auxiliary_runtime._process_auxiliary_planner_task(deferred_task, "alpha_vantage")

    assert db.get_planner_task("aux::AAPL").status == "SUCCESS"
    assert db.get_planner_task("aux::MSFT").status == "PARTIAL"
    stored = pd.read_parquet(tmp_path / "data" / "reference" / "corp_actions.parquet")
    assert stored["symbol"].tolist() == ["AAPL"]


def test_process_backfill_queue_prefers_tiingo_for_equities_history(tmp_path: Path) -> None:
    db = DataNodeDB(tmp_path / "control" / "node.sqlite")
    db.enqueue_task("equities_eod", "AAPL", "2019-01-02", "2019-01-02", "BOOTSTRAP", 1)
    alpaca = _NoopConnector()
    tiingo = _BackfillConnector("tiingo")
    service = DataNodeService(
        db=db,
        connectors={"alpaca": alpaca, "tiingo": tiingo},
        auditor=PartitionAuditor(db=db, calendar_store=ExchangeCalendarStore(root=tmp_path / "reference" / "calendars")),
        curator=Curator(),
        paths=DataNodePaths(root=tmp_path),
    )
    service.default_symbols = ["AAPL"]

    changed = service.process_backfill_queue()

    assert changed == ["2019-01-02"]
    assert tiingo.calls == [("equities_eod", ["AAPL"], "2019-01-02", "2019-01-02")]
    stored = pd.read_parquet(tmp_path / "data" / "raw" / "equities_bars" / "date=2019-01-02" / "data.parquet")
    assert stored.iloc[0]["source_name"] == "tiingo"
