from __future__ import annotations

"""
Tick size regime builder (free-only approximation).

Approximates tick size regime per symbol per date using simple rules:
  - Base tick: $0.01
  - Half-penny ($0.005) for highly liquid names (top-N by ADV) and price >= $1

Writes partitions to:
  data_layer/reference/tick_size_regime/date=YYYY-MM-DD/data.parquet
with columns: date, symbol, tick_size
"""

from datetime import date, timedelta
from pathlib import Path
from typing import List

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


def update_tick_size(top_n_half_penny: int = 250, lookback_days: int = 90) -> int:
    root = Path("data_layer/curated/equities_ohlcv_adj")
    ds_all = _list_dates(root)
    if not ds_all:
        logger.info("No curated OHLCV partitions; skipping tick size update")
        return 0
    last = ds_all[-1]
    start = last - timedelta(days=lookback_days)
    ds_lb = [d for d in ds_all if d >= start]
    if not ds_lb:
        ds_lb = ds_all[-5:]
    frames = []
    for ds in ds_lb:
        p = root / f"date={ds.isoformat()}" / "data.parquet"
        try:
            df = pd.read_parquet(p)
            frames.append(df[["symbol", "close_adj", "close_raw", "volume_adj"]])
        except Exception:
            continue
    if not frames:
        return 0
    panel = pd.concat(frames, ignore_index=True)
    price = panel["close_adj"].fillna(panel["close_raw"]).astype(float)
    panel["dollar_volume"] = price * panel["volume_adj"].astype(float)
    adv = panel.groupby("symbol")["dollar_volume"].mean().sort_values(ascending=False)
    top_liq = set(adv.head(top_n_half_penny).index.tolist())

    latest_p = root / f"date={last.isoformat()}" / "data.parquet"
    dfl = pd.read_parquet(latest_p)
    dfl["price"] = dfl["close_adj"].fillna(dfl["close_raw"]).astype(float)
    dfl["tick_size"] = dfl.apply(lambda r: 0.005 if (r["symbol"] in top_liq and r["price"] >= 1.0) else 0.01, axis=1)
    out_dir = Path("data_layer/reference/tick_size_regime") / f"date={last.isoformat()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    dfl[["symbol", "tick_size"]].assign(date=last).to_parquet(out_dir / "data.parquet", index=False)
    logger.info(f"Tick size regime updated for {len(dfl)} symbols at {out_dir}")
    return int(len(dfl))

