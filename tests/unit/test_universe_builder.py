from datetime import date, timedelta
from pathlib import Path
import pandas as pd


def _write(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def test_build_universe_from_curated(tmp_path):
    # Create two dates with two symbols
    d1 = date(2020, 1, 1).isoformat()
    d2 = date(2020, 1, 2).isoformat()
    base = Path("data_layer/curated/equities_ohlcv_adj")
    _write(pd.DataFrame([
        {"date": d1, "symbol": "AAA", "close_adj": 10.0, "close_raw": 10.0, "volume_adj": 1_000_000},
        {"date": d1, "symbol": "BBB", "close_adj": 100.0, "close_raw": 100.0, "volume_adj": 100_000},
    ]), base / f"date={d1}" / "data.parquet")
    _write(pd.DataFrame([
        {"date": d2, "symbol": "AAA", "close_adj": 10.0, "close_raw": 10.0, "volume_adj": 1_000_000},
        {"date": d2, "symbol": "BBB", "close_adj": 100.0, "close_raw": 100.0, "volume_adj": 100_000},
    ]), base / f"date={d2}" / "data.parquet")

    from ops.reference.universe import build_universe_from_curated
    symbols = build_universe_from_curated(n_top=1, lookback_days=3650)
    assert symbols and symbols[0] == "AAA"  # higher ADV

