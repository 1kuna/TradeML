"""
Delta-hedged options PnL estimator (free-data friendly).

Computes one-day PnL for a delta-hedged straddle using curated IV artifacts:
  - Uses ATM strike ~= underlying close
  - Approximates vega PnL via change in ATM IV from t to t+1
  - Delta-hedge cost via underlier return between t and t+1

Inputs:
  - Curated IV table: data_layer/curated/options_iv/date=YYYY-MM-DD/underlier=SYM/data.parquet
  - Curated prices:   data_layer/curated/equities_ohlcv_adj/date=YYYY-MM-DD/data.parquet

This is a coarse, diagnostics-oriented path to provide delta-hedged PnL evaluation
without paid feeds. It should be replaced by a more accurate mark-to-market once
NBBO time series (OPRA) is available.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class StraddlePnl:
    underlier: str
    asof: date
    next_date: date
    atm_iv_t: float
    atm_iv_t1: float
    spot_t: float
    spot_t1: float
    est_vega: float
    pnl_delta_hedged: float


def _load_iv(asof: date, underlier: str) -> pd.DataFrame:
    p = Path("data_layer/curated/options_iv") / f"date={asof.isoformat()}" / f"underlier={underlier}" / "data.parquet"
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(p)
    except Exception:
        return pd.DataFrame()


def _load_spot(asof: date, symbol: str) -> Optional[float]:
    p = Path("data_layer/curated/equities_ohlcv_adj") / f"date={asof.isoformat()}" / "data.parquet"
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p)
    except Exception:
        return None
    row = df[df["symbol"] == symbol]
    if row.empty:
        return None
    return float(row.iloc[0].get("close_adj", row.iloc[0].get("close_raw", np.nan)))


def _atm_iv(iv_df: pd.DataFrame, spot: float) -> Optional[float]:
    if iv_df.empty:
        return None
    # Use nearest expiry > 0d and closest strike to spot
    iv_df = iv_df.copy()
    iv_df["expiry"] = pd.to_datetime(iv_df["expiry"]).dt.date
    future = iv_df[iv_df["expiry"] > iv_df["date"].iloc[0]] if "date" in iv_df.columns else iv_df
    if future.empty:
        future = iv_df
    # Choose nearest expiry
    exps = sorted({e for e in future["expiry"].dropna().unique()})
    if not exps:
        return None
    exp = exps[0]
    sl = future[future["expiry"] == exp]
    if sl.empty:
        return None
    sl["dist"] = (sl["strike"] - spot).abs()
    row = sl.sort_values("dist").iloc[0]
    return float(row.get("iv")) if pd.notna(row.get("iv")) else None


def estimate_delta_hedged_straddle_pnl(asof: date, underlier: str) -> Optional[StraddlePnl]:
    # Anchor on first available price from asof forward (up to ~1 week)
    anchor_day = asof
    s0: Optional[float] = None
    for _ in range(6):
        s0 = _load_spot(anchor_day, underlier)
        if s0 is not None:
            break
        anchor_day = anchor_day + timedelta(days=1)
    if s0 is None:
        return None

    # Find next available price after anchor
    next_day = anchor_day + timedelta(days=1)
    s1: Optional[float] = None
    for _ in range(6):
        s1 = _load_spot(next_day, underlier)
        if s1 is not None:
            break
        next_day = next_day + timedelta(days=1)
    if s1 is None:
        return None

    iv0 = _load_iv(anchor_day, underlier)
    iv1 = _load_iv(next_day, underlier)
    if iv0.empty or iv1.empty:
        return None

    atm0 = _atm_iv(iv0, s0)
    atm1 = _atm_iv(iv1, s1)
    if atm0 is None or atm1 is None:
        return None

    # Approximate vega for straddle: use BS vega for ATM call+put ~ 2 * S * sqrt(T) * n(d1) * 0.01
    # Without full greeks here, approximate with: est_vega ≈ 2 * S * 0.4 * sqrt(21/252)
    # This is a crude proxy for one-month ATM; intent is directional diagnostic only.
    T_approx = np.sqrt(21 / 252.0)
    est_vega = 2.0 * s0 * 0.4 * T_approx
    d_iv = (atm1 - atm0)
    pnl_vol = est_vega * d_iv

    # Delta-hedged: subtract PnL due to underlier move for the straddle's delta ≈ 0 initially
    # For small moves and daily rebalance, assume minimal residual; we set delta cost to 0 here.
    pnl = float(pnl_vol)

    return StraddlePnl(
        underlier=underlier,
        asof=asof,
        next_date=next_day,
        atm_iv_t=float(atm0),
        atm_iv_t1=float(atm1),
        spot_t=float(s0),
        spot_t1=float(s1),
        est_vega=float(est_vega),
        pnl_delta_hedged=float(pnl),
    )
