from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from data_node.db import TaskKind
from data_node.worker import QueueWorker, QueueWorkerLoop


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


class _FakeAlpacaConnector:
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
        _write_partitioned(df, base, partition_cols or ["date", "symbol"])


class _FakeFinnhubConnector:
    def __init__(self, *args, **kwargs):  # noqa: ARG002
        self.source_name = "finnhub"

    def fetch_options_chain(self, symbol, date=None):  # noqa: ARG002
        return pd.DataFrame(
            [
                {
                    "date": date or date.today(),
                    "underlier": symbol,
                    "expiry": date.today() + timedelta(days=30),
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


class _FakeFREDConnector:
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


def test_worker_loop_processes_end_to_end(node_db, temp_data_root, monkeypatch):
    today = date.today()
    start_dt = today - timedelta(days=2)
    monkeypatch.setenv("DATA_ROOT", str(temp_data_root))
    monkeypatch.setenv("REQUEST_PACING_ENABLED", "false")
    monkeypatch.setenv("REQUEST_PACING_JITTER_MS", "0,0")

    monkeypatch.setattr("data_layer.connectors.alpaca_connector.AlpacaConnector", _FakeAlpacaConnector)
    monkeypatch.setattr("data_layer.connectors.finnhub_connector.FinnhubConnector", _FakeFinnhubConnector)
    monkeypatch.setattr("data_layer.connectors.fred_connector.FREDConnector", _FakeFREDConnector)

    # Seed mixed tasks
    node_db.enqueue_task(
        dataset="equities_eod",
        symbol="AAPL",
        start_date=start_dt.isoformat(),
        end_date=today.isoformat(),
        kind=TaskKind.BOOTSTRAP,
        priority=0,
    )
    node_db.enqueue_task(
        dataset="options_chains",
        symbol="AAPL",
        start_date=today.isoformat(),
        end_date=today.isoformat(),
        kind=TaskKind.BOOTSTRAP,
        priority=1,
    )
    node_db.enqueue_task(
        dataset="macros_fred",
        symbol=None,
        start_date=today.isoformat(),
        end_date=today.isoformat(),
        kind=TaskKind.BOOTSTRAP,
        priority=2,
    )

    worker = QueueWorker(db=node_db, budgets=_FakeBudget())

    # Run loop manually for a few iterations
    processed = 0
    for _ in range(5):
        processed += 1 if worker.process_one() else 0

    assert processed >= 3
    stats = node_db.get_queue_stats()
    assert stats["by_status"].get("PENDING", 0) == 0
    assert stats["by_status"].get("LEASED", 0) == 0
