from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import sqlite3

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

    changed = service._process_backfill_task_for_vendor(leased, "finnhub")
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

    tiingo_task = service._lease_next_task_for_vendor("tiingo")
    alpaca_task = service._lease_next_task_for_vendor("alpaca")

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

    changed = service._process_backfill_task_for_vendor(leased, "alpaca")

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

    changed = service._process_backfill_task_for_vendor(leased, "alpaca")

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
    tasks = service._lease_canonical_batch("alpaca")

    changed = service._process_canonical_planner_batch(batch=tasks, vendor="alpaca", exchange="XNYS")

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
    expanded = DataNodeService._expand_reference_jobs(
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

    batch = service._lease_canonical_batch("tiingo")

    assert batch
    assert batch[0].symbols == ("SYM0256",)


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

    assert "macro" in calls
    assert "reference" in calls
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

    macro_series, reference_jobs = service._planned_auxiliary_work(trading_date="2026-03-31")

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

    service._run_cluster_auxiliary_tasks(
        coordinator=coordinator,  # type: ignore[arg-type]
        trading_date="2026-03-31",
        current_et=datetime.fromisoformat("2026-03-31T18:00:00+00:00"),
        macro_series_ids=["DGS10"],
        reference_jobs=[{"source": "alpaca", "dataset": "assets", "symbols": [], "output_name": "alpaca_assets"}],
        price_check_symbols=["AAPL"],
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
