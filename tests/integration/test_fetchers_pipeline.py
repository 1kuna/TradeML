from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from data_node.db import PartitionStatus, TaskKind
from data_node.worker import QueueWorker
from data_node.fetchers import FetchResult, FetchStatus


def _bdates(start: date, days: int) -> list[date]:
    return pd.bdate_range(start=start, periods=days).date.tolist()


def _write_partitioned(df: pd.DataFrame, base_path: Path, partition_cols: list[str]) -> None:
    for _, sub in df.groupby(partition_cols, dropna=False):
        part_path = Path(base_path)
        for col in partition_cols:
            val = sub.iloc[0][col]
            if hasattr(val, "isoformat"):
                val = val.isoformat()
            part_path = part_path / f"{col}={val}"
        part_path.mkdir(parents=True, exist_ok=True)
        sub.to_parquet(part_path / "data.parquet", index=False)


class _FakeBudget:
    def can_spend(self, vendor: str, kind: TaskKind, tokens: int = 1) -> bool:  # noqa: ARG002
        return True


def test_worker_processes_equities_task_with_stubbed_alpaca(node_db, temp_data_root, monkeypatch):
    start = _bdates(date.today() - timedelta(days=5), 2)
    start_date, end_date = start[0], start[-1]

    class FakeAlpacaConnector:
        def __init__(self, *args, **kwargs):  # noqa: ARG002
            self.source_name = "alpaca"
            self.last_feed = None

        def fetch_bars(self, symbols, start_date, end_date, timeframe):  # noqa: ARG002
            dates = pd.bdate_range(start=start_date, end=end_date).date
            rows = []
            for sym in symbols:
                for dt in dates:
                    rows.append(
                        {
                            "date": dt,
                            "symbol": sym,
                            "open": 10.0,
                            "high": 11.0,
                            "low": 9.5,
                            "close": 10.5,
                            "vwap": 10.5,
                            "volume": 1_000,
                        }
                    )
            return pd.DataFrame(rows)

        def write_parquet(self, df, path, partition_cols=None, schema=None):  # noqa: ARG002
            base = Path(path)
            if partition_cols:
                _write_partitioned(df, base, partition_cols)
            else:
                _write_partitioned(df, base, ["date", "symbol"])

    monkeypatch.setenv("DATA_ROOT", str(temp_data_root))
    monkeypatch.setenv("REQUEST_PACING_ENABLED", "false")
    monkeypatch.setenv("ALPACA_API_KEY", "dummy")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "dummy")
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("CURATED_EQUITY_BARS_ADJ_DIR", str(temp_data_root / "data_layer/curated/equities_ohlcv_adj"))
    monkeypatch.setenv("REQUEST_PACING_ENABLED", "false")
    monkeypatch.setenv("REQUEST_PACING_JITTER_MS", "0,0")

    monkeypatch.setattr("data_layer.connectors.alpaca_connector.AlpacaConnector", FakeAlpacaConnector)

    node_db.enqueue_task(
        dataset="equities_eod",
        symbol="AAPL",
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        kind=TaskKind.BOOTSTRAP,
        priority=0,
    )

    worker = QueueWorker(db=node_db, budgets=_FakeBudget())
    processed = worker.process_one()
    assert processed is True

    # Parquet written under DATA_ROOT
    for dt in (start_date, end_date):
        raw_path = (
            temp_data_root
            / "data_layer"
            / "raw"
            / "alpaca"
            / "equities_bars"
            / f"date={dt.isoformat()}"
            / "symbol=AAPL"
            / "data.parquet"
        )
        assert raw_path.exists()

    # Partition status updated to GREEN
    parts = node_db.get_partition_status_batch(
        table_name="equities_eod",
        symbols=["AAPL"],
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
    )
    assert parts
    assert all(p["status"] == PartitionStatus.GREEN.value for p in parts)


def test_handle_partial_creates_followup_task(node_db, temp_data_root, monkeypatch):
    monkeypatch.setenv("DATA_ROOT", str(temp_data_root))
    task_id = node_db.enqueue_task(
        dataset="equities_eod",
        symbol="MSFT",
        start_date="2024-01-01",
        end_date="2024-01-03",
        kind=TaskKind.FORWARD,
        priority=2,
    )
    task = node_db.lease_next_task(node_id="node-test")
    assert task and task.id == task_id

    worker = QueueWorker(db=node_db, budgets=_FakeBudget())
    result = FetchResult(
        status=FetchStatus.PARTIAL,
        rows=2,
        rows_by_date={
            date(2024, 1, 1): 100,
            date(2024, 1, 2): 0,  # will be GREEN NO_SESSION if weekend
        },
        qc_code="OK",
        vendor_used="alpaca",
        failed_at_date="2024-01-03",
        original_end_date="2024-01-03",
    )

    worker.handle_result(task, result)

    stats = node_db.get_queue_stats()
    # Original task done, follow-up queued
    assert stats["by_status"].get("PENDING", 0) >= 1

    parts = node_db.get_partition_status_batch(
        table_name="equities_eod",
        symbols=["MSFT"],
        start_date="2024-01-01",
        end_date="2024-01-03",
    )
    assert parts


def test_options_and_macros_fetchers_with_stubs(node_db, temp_data_root, monkeypatch):
    start_dt = date.today() - timedelta(days=3)
    monkeypatch.setenv("DATA_ROOT", str(temp_data_root))

    class FakeFinnhubConnector:
        def __init__(self, *args, **kwargs):  # noqa: ARG002
            self.source_name = "finnhub"

        def fetch_options_chain(self, symbol, date=None):  # noqa: ARG002
            return pd.DataFrame(
                [
                    {
                        "date": date or start_dt,
                        "underlier": symbol,
                        "expiry": start_dt + timedelta(days=30),
                        "strike": 100.0,
                        "cp_flag": "C",
                        "bid": 1.0,
                        "ask": 1.2,
                    }
                ]
            )

        def write_parquet(self, df, path, partition_cols=None, schema=None):  # noqa: ARG002
            base = Path(path)
            _write_partitioned(df, base, partition_cols or ["date", "underlier"])

    class FakeFREDConnector:
        def __init__(self, *args, **kwargs):  # noqa: ARG002
            self.source_name = "fred"

        def fetch_treasury_curve(self, start_date, end_date):  # noqa: ARG002
            return pd.DataFrame(
                [
                    {"date": start_date, "maturity": "10y", "rate": 0.03},
                    {"date": end_date, "maturity": "2y", "rate": 0.02},
                ]
            )

        def write_parquet(self, df, path, partition_cols=None, schema=None):  # noqa: ARG002
            base = Path(path)
            _write_partitioned(df, base, partition_cols or ["date"])

    monkeypatch.setattr("data_layer.connectors.finnhub_connector.FinnhubConnector", FakeFinnhubConnector)
    monkeypatch.setattr("data_layer.connectors.fred_connector.FREDConnector", FakeFREDConnector)

    # Enqueue options and macro tasks
    node_db.enqueue_task(
        dataset="options_chains",
        symbol="AAPL",
        start_date=start_dt.isoformat(),
        end_date=start_dt.isoformat(),
        kind=TaskKind.BOOTSTRAP,
        priority=0,
    )
    node_db.enqueue_task(
        dataset="macros_fred",
        symbol=None,
        start_date=start_dt.isoformat(),
        end_date=start_dt.isoformat(),
        kind=TaskKind.BOOTSTRAP,
        priority=0,
    )

    worker = QueueWorker(db=node_db, budgets=_FakeBudget())
    # Process both tasks
    assert worker.process_one() is True
    assert worker.process_one() is True

    # Partition status entries exist
    options_parts = node_db.get_partition_status_batch(
        table_name="options_chains",
        symbols=["AAPL"],
        start_date=start_dt.isoformat(),
        end_date=start_dt.isoformat(),
    )
    assert options_parts
    macro_records = node_db.get_partitions_by_status(
        table_name="macros_fred",
        status=PartitionStatus.GREEN,
        start_date=start_dt.isoformat(),
        end_date=start_dt.isoformat(),
    )
    assert macro_records

    # Parquet outputs exist
    opt_path = (
        temp_data_root
        / "data_layer"
        / "raw"
        / "finnhub"
        / "options_chains"
        / f"date={start_dt.isoformat()}"
        / "underlier=AAPL"
        / "data.parquet"
    )
    fred_path = (
        temp_data_root
        / "data_layer"
        / "raw"
        / "fred"
        / "macro_treasury"
        / f"date={start_dt.isoformat()}"
        / "data.parquet"
    )
    assert opt_path.exists()
    assert fred_path.exists()
