import os
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


def _write(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def test_compute_equity_features_and_labels(tmp_path):
    # Prepare curated history for one symbol
    base = Path("data_layer/curated/equities_ohlcv_adj")
    start = date.today() - timedelta(days=90)
    days = pd.bdate_range(start=start, end=date.today()).date
    # Synthetic rising price, steady volume
    prices = np.linspace(10, 12, num=len(days))
    df = pd.DataFrame({
        "date": days,
        "symbol": "AAPL",
        "session_id": [d.strftime("%Y%m%d") for d in days],
        "open_adj": prices,
        "high_adj": prices * 1.01,
        "low_adj": prices * 0.99,
        "close_adj": prices,
        "vwap_adj": prices,
        "volume_adj": 1_000_000,
        "close_raw": prices,
        "adjustment_factor": 1.0,
        "last_adjustment_date": None,
        "ingested_at": pd.Timestamp.utcnow(),
        "source_name": "curator",
        "source_uri": "curator://test",
        "transform_id": "tst",
    })
    _write(df, base / "AAPL_adj.parquet")

    from feature_store.equities.features import compute_equity_features
    from labeling.triple_barrier.triple_barrier import triple_barrier

    asof = date.today()
    feats = compute_equity_features(asof, ["AAPL"])  # should find file and produce features
    assert not feats.empty
    assert set(["symbol", "asof"]).issubset(set(feats.columns))
    assert feats.iloc[0]["symbol"] == "AAPL"

    labels = triple_barrier(asof, ["AAPL"], tp_sigma=2.0, sl_sigma=1.0, max_h=5)
    # Might be empty if no forward beyond asof exists; allow empty but not error
    assert labels is not None

