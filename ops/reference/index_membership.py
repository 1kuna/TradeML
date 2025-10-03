from __future__ import annotations

"""
Build and persist index membership history from free curated data.

We approximate "index membership" using a reproducible universe definition:
the ADV_TOP{N} membership by 60-day average dollar volume, computed daily from
curated equities OHLCV. This avoids paid sources for proprietary index lists.

Writes partitions to:
  data_layer/reference/index_membership/date=YYYY-MM-DD/data.parquet
with columns: date, index_name, symbol, member
"""

from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, List

import pandas as pd
from loguru import logger


def _list_dates(root: Path) -> List[date]:
    if not root.exists():
        return []
    out = []
    for p in root.glob("date=*"):
        try:
            ds = pd.to_datetime(p.name.split("=", 1)[-1]).date()
            out.append(ds)
        except Exception:
            continue
    return sorted(out)


def _load_panel(dates: List[date]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    root = Path("data_layer/curated/equities_ohlcv_adj")
    for ds in dates:
        p = root / f"date={ds.isoformat()}" / "data.parquet"
        try:
            df = pd.read_parquet(p)
            frames.append(df[["symbol", "close_adj", "close_raw", "volume_adj"]].assign(date=ds))
        except Exception:
            continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def update_index_membership(n_top: int = 1000, lookback_days: int = 90, index_name: str | None = None) -> int:
    idx_name = index_name or f"ADV_TOP{n_top}"
    root = Path("data_layer/curated/equities_ohlcv_adj")
    ds_all = _list_dates(root)
    if not ds_all:
        logger.info("No curated OHLCV partitions; skipping index membership update")
        return 0
    last = ds_all[-1]
    # Look back ~60 business days
    start = last - timedelta(days=lookback_days)
    ds_lb = [d for d in ds_all if d >= start]
    if not ds_lb:
        ds_lb = ds_all[-5:]
    panel = _load_panel(ds_lb)
    if panel.empty:
        logger.info("No panel loaded for index membership")
        return 0
    panel["price"] = panel["close_adj"].fillna(panel["close_raw"]).astype(float)
    panel["dollar_volume"] = panel["price"] * panel["volume_adj"].astype(float)
    adv = panel.groupby("symbol")["dollar_volume"].mean().sort_values(ascending=False)
    members = set(adv.head(n_top).index.tolist())

    out_dir = Path("data_layer/reference/index_membership") / f"date={last.isoformat()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame({
        "date": [last] * len(members),
        "index_name": [idx_name] * len(members),
        "symbol": list(members),
        "member": [True] * len(members),
    })
    df_out.to_parquet(out_dir / "data.parquet", index=False)
    logger.info(f"Index membership updated for {idx_name} at {out_dir}")
    return len(members)

