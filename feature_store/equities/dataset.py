"""
Equity dataset builder for Phase 2.

Builds timeseries features and labels over a date range for a universe,
ready for CPCV training and backtesting integration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Literal, Optional, Tuple
from datetime import date as Date

import numpy as np
import pandas as pd
from loguru import logger

from data_layer.curated.loaders import load_price_panel


@dataclass
class Dataset:
    X: pd.DataFrame  # features incl. date,symbol
    y: pd.Series     # target aligned with X (index matches X)
    meta: pd.DataFrame  # columns: date, symbol, horizon_days


def _compute_features_panel(panel: pd.DataFrame, standardize_window: int = 252) -> pd.DataFrame:
    df = panel.sort_values(["symbol", "date"]).copy()
    # Returns
    df["ret_1d"] = df.groupby("symbol")["close"].pct_change()
    # Momentum - use transform to preserve index alignment
    for k in (5, 20, 60):
        df[f"mom_{k}d"] = df.groupby("symbol")["close"].transform(lambda s: s / s.shift(k) - 1.0)
    # Realized volatility
    for k in (20, 60):
        df[f"rv_{k}d"] = df.groupby("symbol")["ret_1d"].rolling(window=k, min_periods=max(5, k // 2)).std(ddof=1).reset_index(level=0, drop=True)
    # Liquidity ADV
    df["dollar_volume"] = df["close"] * df["volume"].astype(float)
    df["adv_20d"] = df.groupby("symbol")["dollar_volume"].rolling(window=20, min_periods=10).mean().reset_index(level=0, drop=True)
    # Seasonality
    dow = pd.to_datetime(df["date"]).dt.weekday
    df["feature_dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    df["feature_dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    # Rolling standardization (z-scores) per symbol
    def zscore(g: pd.Series) -> pd.Series:
        mu = g.rolling(standardize_window, min_periods=20).mean()
        sd = g.rolling(standardize_window, min_periods=20).std(ddof=1)
        return (g - mu) / sd

    feature_cols = ["mom_5d", "mom_20d", "mom_60d", "rv_20d", "rv_60d", "adv_20d"]
    for c in feature_cols:
        df[f"feature_{c}"] = df.groupby("symbol")[c].transform(zscore)

    out_cols = ["date", "symbol", "feature_dow_sin", "feature_dow_cos"] + [f"feature_{c}" for c in feature_cols]
    features = df[out_cols].dropna(subset=[f for f in out_cols if f.startswith("feature_")])
    return features


def _horizon_labels(panel: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    df = panel.sort_values(["symbol", "date"]).copy()
    # Shift fwd close per symbol
    fwd_close = df.groupby("symbol")["close"].shift(-horizon_days)
    fwd_ret = (fwd_close / df["close"]) - 1.0
    labels = pd.DataFrame(
        {
            "symbol": df["symbol"],
            "entry_date": df["date"],
            "exit_date": df.groupby("symbol")["date"].shift(-horizon_days),
            "horizon_days": horizon_days,
            "forward_return": fwd_ret,
        }
    )
    labels = labels.dropna(subset=["forward_return"]).reset_index(drop=True)
    return labels


def _rolling_sigma(ret: pd.Series, window: int) -> pd.Series:
    return ret.rolling(window=window, min_periods=max(5, window // 2)).std(ddof=1)


def _triple_barrier_labels(
    panel: pd.DataFrame,
    tp_sigma: float,
    sl_sigma: float,
    max_h: int,
    vol_window: int,
) -> pd.DataFrame:
    df = panel.sort_values(["symbol", "date"]).copy()
    df["ret_1d"] = df.groupby("symbol")["close"].pct_change()
    df["sigma"] = df.groupby("symbol")["ret_1d"].transform(lambda s: _rolling_sigma(s, vol_window))

    rows: List[dict] = []
    for sym, g in df.groupby("symbol", sort=False):
        g = g.reset_index(drop=True)
        for i in range(1, len(g) - 1):
            entry_dt = g.loc[i, "date"]
            entry_px = float(g.loc[i, "close"])
            sigma = g.loc[i - 1, "sigma"]  # prior day sigma (PIT)
            if pd.isna(sigma) or sigma <= 0:
                continue
            tp_level = entry_px * (1.0 + tp_sigma * sigma)
            sl_level = entry_px * (1.0 - sl_sigma * sigma)
            # forward path up to max_h
            end_i = min(i + max_h, len(g) - 1)
            path = g.loc[i + 1 : end_i]
            label = 0
            outcome = 0.0
            exit_dt = g.loc[end_i, "date"]
            hit = False
            for _, row in path.iterrows():
                hi = float(row.get("high", row["close"]))
                lo = float(row.get("low", row["close"]))
                exit_dt = row["date"]
                if hi >= tp_level:
                    label = 1
                    outcome = (tp_level / entry_px) - 1.0
                    hit = True
                    break
                if lo <= sl_level:
                    label = -1
                    outcome = (sl_level / entry_px) - 1.0
                    hit = True
                    break
            if not hit:
                exit_px = float(g.loc[end_i, "close"])  # settle at horizon
                outcome = (exit_px / entry_px) - 1.0
                label = 0
            rows.append(
                {
                    "symbol": sym,
                    "entry_date": entry_dt,
                    "exit_date": exit_dt,
                    "label": label,
                    "outcome": outcome,
                    "horizon_days": int((pd.to_datetime(exit_dt) - pd.to_datetime(entry_dt)).days),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["symbol", "entry_date", "exit_date", "label", "outcome", "horizon_days"])
    return pd.DataFrame(rows)


def build_training_dataset(
    universe: Iterable[str],
    start_date: object,
    end_date: object,
    label_type: Literal["horizon", "triple_barrier"] = "horizon",
    horizon_days: int = 5,
    tp_sigma: float = 2.0,
    sl_sigma: float = 1.0,
    max_h: int = 10,
    vol_window: int = 20,
    standardize_window: int = 252,
) -> Dataset:
    """Build X, y, meta dataset for Phase 2 training.

    Returns:
        Dataset(X=features_with_keys, y=target_series, meta=labels_meta)
    """
    logger.info("Loading curated price panel...")
    panel = load_price_panel(universe, start_date, end_date)
    if panel.empty:
        return Dataset(X=pd.DataFrame(), y=pd.Series(dtype=float), meta=pd.DataFrame())

    logger.info("Computing features panel...")
    feats = _compute_features_panel(panel, standardize_window=standardize_window)

    logger.info(f"Generating '{label_type}' labels...")
    if label_type == "horizon":
        labels = _horizon_labels(panel, horizon_days=horizon_days)
        target_col = "forward_return"
    else:
        labels = _triple_barrier_labels(panel, tp_sigma=tp_sigma, sl_sigma=sl_sigma, max_h=max_h, vol_window=vol_window)
        target_col = "label"

    # Align features (as-of == entry_date)
    feats_ren = feats.rename(columns={"date": "entry_date"})
    df = feats_ren.merge(labels, on=["symbol", "entry_date"], how="inner")

    # Prepare outputs
    feature_cols = [c for c in df.columns if c.startswith("feature_")]
    X = df[["entry_date", "symbol"] + feature_cols].rename(columns={"entry_date": "date"})
    y = df[target_col]
    # meta should use entry_date before renaming
    meta = df[["entry_date", "symbol"]].rename(columns={"entry_date": "date"}).copy()
    meta["horizon_days"] = df.get("horizon_days", horizon_days)

    return Dataset(X=X.reset_index(drop=True), y=y.reset_index(drop=True), meta=meta.reset_index(drop=True))
