from __future__ import annotations

"""
Universe builder from curated equities.

Computes top-N symbols by 60-business-day ADV (USD) from
data_layer/curated/equities_ohlcv_adj and writes:
- data_layer/reference/universe.csv
- data_layer/reference/universe_symbols.txt

Falls back gracefully when curated data is limited.
"""

from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd
from loguru import logger


def _list_curated_dates() -> List[str]:
    root = Path("data_layer/curated/equities_ohlcv_adj")
    if not root.exists():
        return []
    return sorted({p.parent.name.split("=")[-1] for p in root.glob("date=*/data.parquet")})


def build_universe_from_curated(n_top: int = 1000, lookback_days: int = 90) -> List[str]:
    dates = _list_curated_dates()
    if not dates:
        logger.warning("No curated dates found for universe build")
        return []
    # Use last ~60 business days (~90 calendar days as approximation)
    last = pd.to_datetime(dates[-1]).date()
    start = last - timedelta(days=lookback_days)
    keep = [d for d in dates if pd.to_datetime(d).date() >= start]
    if not keep:
        keep = dates[-5:]

    frames = []
    for ds in keep:
        p = Path("data_layer/curated/equities_ohlcv_adj") / f"date={ds}" / "data.parquet"
        try:
            df = pd.read_parquet(p)
            frames.append(df[["symbol", "close_adj", "close_raw", "volume_adj"]].assign(date=pd.to_datetime(ds).date()))
        except Exception:
            continue
    if not frames:
        logger.warning("No curated frames loaded for universe build")
        return []
    df = pd.concat(frames, ignore_index=True)
    price = df["close_adj"].fillna(df["close_raw"]).astype(float)
    vol = df["volume_adj"].astype(float)
    df["dollar_volume"] = price * vol
    adv = df.groupby("symbol")["dollar_volume"].mean().sort_values(ascending=False)
    top = adv.head(n_top).index.tolist()

    # Write outputs
    out_dir = Path("data_layer/reference")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "universe.csv"
    adv_df = adv.reset_index().rename(columns={"dollar_volume": "adv_usd"})
    adv_df.to_csv(csv_path, index=False)
    (out_dir / "universe_symbols.txt").write_text("\n".join(top))
    logger.info(f"Universe built from curated: {len(top)} symbols → {csv_path}")
    return top

