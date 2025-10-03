import os
from datetime import date
from pathlib import Path

import pandas as pd


def _write_parquet(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def test_audit_equities_minute_and_options(tmp_path, monkeypatch):
    # Local storage
    os.environ["STORAGE_BACKEND"] = "local"

    root = Path.cwd()
    # Prepare raw partitions for one day
    ds = date.today().isoformat()

    # equities_eod
    df_eod = pd.DataFrame([
        {"date": ds, "symbol": "AAPL", "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1},
        {"date": ds, "symbol": "MSFT", "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1},
    ])
    _write_parquet(df_eod, Path("data_layer/raw/alpaca/equities_bars") / f"date={ds}" / "data.parquet")

    # equities_minute
    df_min = pd.DataFrame([
        {"date": ds, "symbol": "AAPL", "open": 1, "high": 1, "low": 1, "close": 1, "volume": 100},
        {"date": ds, "symbol": "MSFT", "open": 1, "high": 1, "low": 1, "close": 1, "volume": 100},
    ])
    _write_parquet(df_min, Path("data_layer/raw/alpaca/equities_bars_minute") / f"date={ds}" / "data.parquet")

    # options_chains (underlier level)
    df_opt = pd.DataFrame([
        {"underlier": "AAPL", "expiry": pd.to_datetime(ds).date(), "strike": 100.0, "cp_flag": "C", "ts_ns": 0},
        {"underlier": "MSFT", "expiry": pd.to_datetime(ds).date(), "strike": 100.0, "cp_flag": "P", "ts_ns": 0},
    ])
    _write_parquet(df_opt, Path("data_layer/raw/options_chains/finnhub") / f"date={ds}" / "underlier=AAPL" / "data.parquet")

    # Audit
    from ops.ssot import audit_scan
    audit_scan(["equities_eod", "equities_minute", "options_chains"])

    # Verify ledger written and contains recent statuses
    ledger_path = Path("data_layer/qc/partition_status.parquet")
    assert ledger_path.exists()
    df_ledger = pd.read_parquet(ledger_path)
    assert set(df_ledger["table_name"].unique()) & {"equities_eod", "equities_minute", "options_chains"}

    # Coverage heatmap renders
    from ops.monitoring.coverage import coverage_heatmap
    out = coverage_heatmap(str(ledger_path), out_dir="ops/reports")
    # If matplotlib is unavailable, function returns empty string; otherwise a file path
    assert (not out) or Path(out).exists()
