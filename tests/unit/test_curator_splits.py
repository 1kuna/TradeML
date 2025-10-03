import os
from datetime import date
from pathlib import Path

import pandas as pd


def _write(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def test_curator_split_day_adjustment(tmp_path):
    os.environ["STORAGE_BACKEND"] = "local"

    ds = date.today().isoformat()
    # Raw bar for ABC on split day
    raw = pd.DataFrame([
        {"date": ds, "symbol": "ABC", "open": 20.0, "high": 22.0, "low": 19.5, "close": 21.0, "vwap": 20.5, "volume": 1000, "ingested_at": pd.Timestamp.utcnow(), "source_name": "alpaca", "source_uri": "alpaca://test"}
    ])
    _write(raw, Path("data_layer/raw/alpaca/equities_bars") / f"date={ds}" / "data.parquet")

    # Corporate actions: 2-for-1 split on ds
    ca = pd.DataFrame([
        {"symbol": "ABC", "event_type": "split", "ex_date": pd.to_datetime(ds).date(), "ratio": 2.0, "record_date": None, "pay_date": None, "source_name": "alpha_vantage", "ingested_at": pd.Timestamp.utcnow(), "source_uri": "alpha_vantage://SPLITS/ABC"}
    ])
    _write(ca, Path("data_layer/reference/corp_actions/ABC_corp_actions.parquet"))

    # Curator config (local)
    cfg = tmp_path / "curator.yml"
    cfg.write_text(
        """
watermark:
  bookmark_key: manifests/curator_it.json
jobs:
  - name: equities_bars_ohlcv
    source: alpaca
    table: equities_bars
    input_prefix: raw/alpaca/equities_bars
    output_prefix: curated/equities_ohlcv_adj
    partition: date
    idempotent: true
"""
    )

    from scripts.curator import Curator
    Curator(str(cfg)).run()

    # Verify curated output and adjusted columns
    out_path = Path("data_layer/curated/equities_ohlcv_adj") / f"date={ds}" / "data.parquet"
    assert out_path.exists()
    cur = pd.read_parquet(out_path)
    assert cur.loc[0, "open_adj"] == 10.0  # 20 / 2
    assert cur.loc[0, "close_adj"] == 10.5  # 21 / 2
    assert cur.loc[0, "volume_adj"] == 2000  # 1000 * 2
    assert cur.loc[0, "adjustment_factor"] == 2.0

