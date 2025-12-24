from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from data_node.db import PartitionStatus, TaskKind
from data_node.worker import QueueWorker


class _FakeBudget:
    def can_spend(self, vendor: str, kind: TaskKind, tokens: int = 1) -> bool:  # noqa: ARG002
        return True


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


def test_qc_probe_mismatch_marks_amber_and_gap(node_db, temp_data_root, monkeypatch):
    probe_date = date(2024, 1, 3)
    dt_str = probe_date.isoformat()

    class FakeAlpacaConnector:
        def __init__(self, *args, **kwargs):  # noqa: ARG002
            self.source_name = "alpaca"
            self.last_feed = None

        def fetch_bars(self, symbols, start_date, end_date, timeframe):  # noqa: ARG002
            rows = []
            for sym in symbols:
                rows.append(
                    {
                        "date": start_date,
                        "symbol": sym,
                        "open": 20.0,
                        "high": 21.0,
                        "low": 19.0,
                        "close": 20.5,
                        "vwap": 20.5,
                        "volume": 2_000,
                    }
                )
            return pd.DataFrame(rows)

        def write_parquet(self, df, path, partition_cols=None, schema=None):  # noqa: ARG002
            base = Path(path)
            _write_partitioned(df, base, partition_cols or ["date", "symbol"])

    monkeypatch.setenv("DATA_ROOT", str(temp_data_root))
    monkeypatch.setenv("REQUEST_PACING_ENABLED", "false")
    monkeypatch.setenv("REQUEST_PACING_JITTER_MS", "0,0")
    monkeypatch.setattr("data_layer.connectors.alpaca_connector.AlpacaConnector", FakeAlpacaConnector)

    primary_root = (
        temp_data_root
        / "data_layer"
        / "raw"
        / "massive"
        / "equities_bars"
        / f"date={dt_str}"
        / "symbol=AAPL"
    )
    primary_root.mkdir(parents=True, exist_ok=True)
    primary_df = pd.DataFrame(
        [
            {
                "date": probe_date,
                "symbol": "AAPL",
                "open": 10.0,
                "high": 11.0,
                "low": 9.5,
                "close": 10.5,
                "vwap": 10.5,
                "volume": 1_000,
            }
        ]
    )
    primary_df.to_parquet(primary_root / "data.parquet", index=False)

    node_db.upsert_partition_status(
        source_name="massive",
        table_name="equities_eod",
        symbol="AAPL",
        dt=dt_str,
        status=PartitionStatus.GREEN,
        qc_score=1.0,
        row_count=1,
        expected_rows=1,
        qc_code="OK",
    )

    node_db.enqueue_task(
        dataset="equities_eod",
        symbol="AAPL",
        start_date=dt_str,
        end_date=dt_str,
        kind=TaskKind.QC_PROBE,
        priority=1,
    )

    worker = QueueWorker(db=node_db, budgets=_FakeBudget())
    assert worker.process_one() is True

    conn = node_db._get_connection()
    qc_row = conn.execute(
        """
        SELECT status, primary_vendor, secondary_vendor
        FROM qc_probes
        WHERE dataset = ? AND symbol = ? AND dt = ?
        """,
        ("equities_eod", "AAPL", dt_str),
    ).fetchone()
    assert qc_row is not None
    assert qc_row["status"] == "MISMATCH"
    assert qc_row["primary_vendor"] == "massive"
    assert qc_row["secondary_vendor"] == "alpaca"

    primary_status = conn.execute(
        """
        SELECT status
        FROM partition_status
        WHERE source_name = ? AND table_name = ? AND symbol = ? AND dt = ?
        """,
        ("massive", "equities_eod", "AAPL", dt_str),
    ).fetchone()
    assert primary_status is not None
    assert primary_status["status"] == PartitionStatus.AMBER.value

    stats = node_db.get_queue_stats()
    assert stats["by_kind"].get("GAP", 0) >= 1
