"""Intraday feature engineering utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class IntradayFeatureConfig:
    """Configuration for intraday feature engineering."""

    resample_minutes: int = 5
    min_bars: int = 30
    ofi_window: int = 10
    rolling_vol_window: int = 30


def _prepare(df: pd.DataFrame, cfg: IntradayFeatureConfig) -> pd.DataFrame:
    expected_cols = {"symbol", "timestamp", "open", "high", "low", "close", "volume"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Intraday data missing required columns: {sorted(missing)}")
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df.sort_values(["symbol", "timestamp"], inplace=True)
    if cfg.resample_minutes > 1:
        df = (
            df.set_index("timestamp")
            .groupby("symbol")
            .resample(f"{cfg.resample_minutes}T")
            .agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            })
            .dropna(subset=["open", "high", "low", "close"])
            .reset_index()
        )
        df.rename(columns={"timestamp": "ts"}, inplace=True)
        df["timestamp"] = df["ts"]
        df.drop(columns=["ts"], inplace=True)
    return df


def _order_flow_imbalance(df: pd.DataFrame, window: int) -> pd.Series:
    price_diff = df["close"].diff().fillna(0.0)
    volume = df["volume"].fillna(0.0)
    return (np.sign(price_diff) * volume).rolling(window=window, min_periods=max(3, window // 2)).sum()


def build_intraday_features(df: pd.DataFrame, cfg: Optional[IntradayFeatureConfig] = None) -> pd.DataFrame:
    """Aggregate minute data into daily cross-sectional features."""
    cfg = cfg or IntradayFeatureConfig()
    if df.empty:
        return pd.DataFrame(columns=[
            "symbol",
            "date",
            "vwap_dislocation",
            "intraday_vol",
            "ofi",
            "gap_open",
            "close_ret",
            "volume_z",
        ])

    df = _prepare(df, cfg)
    df["date"] = df["timestamp"].dt.tz_convert("UTC").dt.date

    feats = []
    for (sym, asof), sub in df.groupby(["symbol", "date"]):
        if len(sub) < cfg.min_bars:
            continue
        sub = sub.sort_values("timestamp")
        price = sub["close"].values.astype(float)
        vol = sub["volume"].values.astype(float)
        if vol.sum() <= 0:
            continue
        vwap = float((price * vol).sum() / vol.sum())
        close = float(price[-1])
        open_px = float(sub["open"].iloc[0])
        high = float(sub["high"].max())
        low = float(sub["low"].min())
        minute_returns = pd.Series(price).pct_change().dropna()
        intraday_vol = float(minute_returns.std(ddof=1)) if not minute_returns.empty else 0.0
        ofi = float(_order_flow_imbalance(sub, cfg.ofi_window).iloc[-1]) if len(sub) >= cfg.ofi_window else 0.0
        gap_open = (open_px / float(sub["close"].shift(1).dropna().iloc[-1]) - 1.0) if sub["close"].shift(1).dropna().size else 0.0
        close_ret = (close / open_px) - 1.0 if open_px else 0.0
        range_spread = (high - low) / vwap if vwap else 0.0
        volume_z = (vol.sum() - vol.mean()) / (vol.std(ddof=1) + 1e-6)
        feats.append({
            "symbol": sym,
            "date": asof,
            "vwap_dislocation": (close - vwap) / vwap,
            "intraday_vol": intraday_vol,
            "ofi": ofi,
            "gap_open": gap_open,
            "close_ret": close_ret,
            "range_spread": range_spread,
            "volume_z": volume_z,
        })

    return pd.DataFrame(feats)
