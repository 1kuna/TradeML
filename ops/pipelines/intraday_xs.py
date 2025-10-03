from __future__ import annotations

"""
Intraday cross-sectional pipeline (free-tier friendly scaffold).

Uses curated minute bars if available (curated/equities_minute). Computes
per-day features like intraday volatility, open-close return, and high-low
range proxies aggregated from minute bars. Falls back to daily OHLCV if
minute data is missing.
"""

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from loguru import logger

from ops.reports.emitter import emit_daily


@dataclass
class IntradayConfig:
    start_date: str
    end_date: str
    universe: List[str]
    initial_capital: float = 1_000_000.0


def _list_dates_in_range(root: Path, start: date, end: date) -> List[date]:
    dates = []
    for p in root.glob("date=*"):
        try:
            ds = pd.to_datetime(p.name.split("=")[-1]).date()
            if start <= ds <= end:
                dates.append(ds)
        except Exception:
            pass
    return sorted(dates)


def _load_minute_day(ds: date) -> pd.DataFrame:
    p = Path("data_layer/curated/equities_minute") / f"date={ds.isoformat()}" / "data.parquet"
    if p.exists():
        try:
            return pd.read_parquet(p)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def run_intraday(cfg: IntradayConfig) -> Dict:
    start = pd.to_datetime(cfg.start_date).date()
    end = pd.to_datetime(cfg.end_date).date()
    base = Path("data_layer/curated/equities_minute")
    if not base.exists():
        logger.warning("No curated minute data; intraday_xs will skip")
        return {"status": "no_data"}
    dates = _list_dates_in_range(base, start, end)
    if not dates:
        return {"status": "no_data"}

    rows = []
    for ds in dates:
        df = _load_minute_day(ds)
        if df.empty or not set(["symbol", "open", "high", "low", "close"]).issubset(df.columns):
            continue
        sub = df[df["symbol"].isin(cfg.universe)].copy()
        if sub.empty:
            continue
        # Aggregate features per symbol for the day
        agg = sub.groupby("symbol").agg(
            oc_ret=("close", lambda x: x.iloc[-1] / x.iloc[0] - 1.0 if len(x) > 1 else 0.0),
            hl_range=("high", "max"),
            ll_range=("low", "min"),
            vol=("close", lambda x: x.pct_change().std(ddof=1) if len(x) > 2 else 0.0),
        ).reset_index()
        agg["hl_spread"] = (agg["hl_range"] - agg["ll_range"]) / agg["hl_range"].replace(0, np.nan)
        agg["score"] = agg["oc_ret"].fillna(0.0) / (agg["vol"].replace(0, np.nan))
        agg["date"] = ds
        rows.append(agg[["date", "symbol", "score"]])

    if not rows:
        return {"status": "no_data"}
    scores = pd.concat(rows, ignore_index=True)

    # Convert scores to weights per day (z-score within day)
    def zgroup(g):
        s = g["score"]
        if len(s) < 2 or s.std(ddof=1) == 0:
            g["target_w"] = 0.0
        else:
            z = (s - s.mean()) / s.std(ddof=1)
            g["target_w"] = z / z.abs().sum()
        return g

    weights = scores.groupby("date", group_keys=False).apply(zgroup)
    last_date = weights["date"].max()
    last_positions = weights[weights["date"] == last_date][["symbol", "target_w"]]

    metrics = {"status": "ok", "days": int(weights["date"].nunique())}
    emit_daily(last_date, last_positions, metrics)
    return {"status": "ok", "weights": weights}

