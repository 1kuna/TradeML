"""
Intraday Feature Engineering.

SSOT v2 Section 3.3: Intraday_xs - features (optional sleeve)

Per minute/aggregated daily features on a liquid subset:
- VWAP dislocation vs last trade
- Order-flow imbalance and signed volume
- Short-horizon realized volatility and microstructure noise proxies
- Time-of-day seasonality encoding
- TOB/LOB integration (bid-ask spread, depth imbalance) when available

Minimal API:
    build_intraday_features(df, cfg) -> pd.DataFrame

Output columns: symbol, date, feature_* (standardized, lag-safe)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time as Time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


# Trading session boundaries (Eastern Time)
MARKET_OPEN = Time(9, 30)
MARKET_CLOSE = Time(16, 0)
TRADING_MINUTES = 390  # 6.5 hours


@dataclass
class IntradayFeatureConfig:
    """Configuration for intraday feature engineering."""

    resample_minutes: int = 5
    min_bars: int = 30
    ofi_window: int = 10
    rolling_vol_window: int = 30
    # Microstructure settings
    roll_window: int = 20  # Window for Roll spread estimator
    realized_kernel_window: int = 30  # Parzen kernel window for RK variance
    # Time-of-day buckets for seasonality
    tod_buckets: int = 6  # Number of time-of-day buckets (e.g., 6 = hourly in 6.5h session)
    # TOB/LOB columns (optional - used if present)
    bid_col: str = "bid"
    ask_col: str = "ask"
    bid_size_col: str = "bid_size"
    ask_size_col: str = "ask_size"


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
            .resample(f"{cfg.resample_minutes}min")
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
    """Compute rolling order flow imbalance (signed volume)."""
    price_diff = df["close"].diff().fillna(0.0)
    volume = df["volume"].fillna(0.0)
    return (np.sign(price_diff) * volume).rolling(window=window, min_periods=max(3, window // 2)).sum()


def _signed_volume(df: pd.DataFrame) -> pd.Series:
    """Compute signed volume based on price direction."""
    price_diff = df["close"].diff().fillna(0.0)
    volume = df["volume"].fillna(0.0)
    return np.sign(price_diff) * volume


def _roll_spread_estimator(returns: pd.Series, window: int) -> float:
    """
    Estimate bid-ask spread using Roll (1984) model.

    The Roll spread is 2 * sqrt(-cov(r_t, r_{t-1})) when autocovariance is negative.
    Returns NaN if autocovariance is positive (spread undefined).
    """
    if len(returns) < window:
        return np.nan

    recent = returns.tail(window).dropna()
    if len(recent) < 5:
        return np.nan

    # Autocovariance at lag 1
    mean_r = recent.mean()
    demeaned = recent - mean_r
    autocov = (demeaned.iloc[:-1].values * demeaned.iloc[1:].values).mean()

    if autocov >= 0:
        # Positive autocov means Roll model doesn't apply (momentum regime)
        return 0.0

    return 2.0 * np.sqrt(-autocov)


def _realized_kernel_variance(returns: pd.Series, window: int) -> float:
    """
    Compute realized kernel variance using Parzen kernel.

    This provides a noise-robust estimate of integrated variance,
    accounting for microstructure noise in high-frequency returns.
    """
    if len(returns) < window:
        return np.nan

    recent = returns.tail(window).dropna()
    n = len(recent)
    if n < 5:
        return np.nan

    # Bandwidth selection (following Barndorff-Nielsen et al.)
    H = int(np.ceil(4 * (n / 100) ** 0.4))
    H = min(H, n - 1)

    # Compute realized variance (sum of squared returns)
    rv = (recent ** 2).sum()

    # Add kernel-weighted autocovariances
    for h in range(1, H + 1):
        # Parzen kernel weight
        x = h / (H + 1)
        if x <= 0.5:
            k_h = 1 - 6 * x**2 + 6 * x**3
        else:
            k_h = 2 * (1 - x)**3

        # Autocovariance at lag h
        gamma_h = (recent.iloc[:-h].values * recent.iloc[h:].values).sum()
        rv += 2 * k_h * gamma_h

    return max(rv, 0.0)  # Ensure non-negative


def _microstructure_noise_ratio(returns: pd.Series, window: int) -> float:
    """
    Estimate microstructure noise as ratio of RV to RK variance.

    Higher values indicate more noise relative to true volatility.
    """
    if len(returns) < window:
        return np.nan

    recent = returns.tail(window).dropna()
    if len(recent) < 10:
        return np.nan

    rv = (recent ** 2).sum()
    rk = _realized_kernel_variance(returns, window)

    if rk is None or np.isnan(rk) or rk <= 0:
        return np.nan

    # Noise ratio: (RV - RK) / RK
    # Positive means noise inflating RV, negative means negative autocorrelation
    return (rv - rk) / rk


def _time_of_day_features(timestamp: pd.Timestamp) -> Tuple[float, float, float]:
    """
    Compute time-of-day seasonality features.

    Returns:
        (tod_sin, tod_cos, minutes_from_open)
    """
    try:
        # Convert to Eastern time for US markets
        if timestamp.tzinfo is not None:
            ts_eastern = timestamp.tz_convert("America/New_York")
        else:
            ts_eastern = timestamp

        # Minutes from market open
        market_open_mins = MARKET_OPEN.hour * 60 + MARKET_OPEN.minute
        current_mins = ts_eastern.hour * 60 + ts_eastern.minute
        mins_from_open = current_mins - market_open_mins

        # Normalize to [0, 1] range within trading session
        progress = mins_from_open / TRADING_MINUTES
        progress = np.clip(progress, 0, 1)

        # Sin/cos encoding for cyclical time-of-day
        tod_sin = np.sin(2 * np.pi * progress)
        tod_cos = np.cos(2 * np.pi * progress)

        return tod_sin, tod_cos, float(mins_from_open)
    except Exception:
        return 0.0, 1.0, 0.0


def _compute_lob_features(
    df: pd.DataFrame,
    cfg: IntradayFeatureConfig,
) -> Dict[str, float]:
    """
    Compute LOB (Limit Order Book) features if bid/ask data available.

    Returns dict with:
        - avg_spread: Average bid-ask spread
        - spread_vol: Spread volatility
        - depth_imbalance: (bid_size - ask_size) / (bid_size + ask_size)
        - queue_pressure: Directional queue imbalance signal
    """
    features = {
        "avg_spread": np.nan,
        "spread_vol": np.nan,
        "depth_imbalance": np.nan,
        "queue_pressure": np.nan,
    }

    has_quotes = cfg.bid_col in df.columns and cfg.ask_col in df.columns
    has_depth = cfg.bid_size_col in df.columns and cfg.ask_size_col in df.columns

    if not has_quotes:
        return features

    bid = df[cfg.bid_col].astype(float)
    ask = df[cfg.ask_col].astype(float)

    # Filter valid quotes
    valid = (bid > 0) & (ask > 0) & (ask > bid)
    if valid.sum() < 5:
        return features

    # Spread features
    mid = (bid + ask) / 2
    spread = (ask - bid) / mid  # Relative spread

    features["avg_spread"] = float(spread[valid].mean())
    features["spread_vol"] = float(spread[valid].std())

    if has_depth:
        bid_size = df[cfg.bid_size_col].astype(float)
        ask_size = df[cfg.ask_size_col].astype(float)

        # Depth imbalance
        total_depth = bid_size + ask_size
        valid_depth = valid & (total_depth > 0)
        if valid_depth.sum() > 0:
            imbalance = (bid_size - ask_size) / total_depth
            features["depth_imbalance"] = float(imbalance[valid_depth].mean())

            # Queue pressure: correlation of imbalance with future returns
            # (simplified: just use last imbalance value)
            features["queue_pressure"] = float(imbalance[valid_depth].iloc[-1])

    return features


def build_intraday_features(df: pd.DataFrame, cfg: Optional[IntradayFeatureConfig] = None) -> pd.DataFrame:
    """
    Aggregate minute data into daily cross-sectional features.

    SSOT v2 Section 3.3 features:
    - VWAP dislocation vs last trade
    - Order-flow imbalance and signed volume
    - Short-horizon realized volatility and microstructure noise proxies
    - Time-of-day seasonality (sin/cos encoding)
    - TOB/LOB features (spread, depth imbalance) when available

    Args:
        df: Minute-level OHLCV data with columns:
            Required: symbol, timestamp, open, high, low, close, volume
            Optional: bid, ask, bid_size, ask_size (for LOB features)
        cfg: Feature configuration

    Returns:
        DataFrame with columns: symbol, date, feature_*
    """
    cfg = cfg or IntradayFeatureConfig()
    if df.empty:
        return pd.DataFrame(columns=[
            "symbol",
            "date",
            # Core features
            "feature_vwap_dislocation",
            "feature_intraday_vol",
            "feature_ofi",
            "feature_signed_volume",
            "feature_gap_open",
            "feature_close_ret",
            "feature_range_spread",
            "feature_volume_z",
            # Microstructure noise proxies
            "feature_roll_spread",
            "feature_realized_kernel_vol",
            "feature_noise_ratio",
            # Time-of-day seasonality
            "feature_tod_sin",
            "feature_tod_cos",
            "feature_session_progress",
            # LOB features (NaN if not available)
            "feature_avg_spread",
            "feature_spread_vol",
            "feature_depth_imbalance",
            "feature_queue_pressure",
        ])

    df = _prepare(df, cfg)
    df["date"] = df["timestamp"].dt.tz_convert("UTC").dt.date

    # Precompute prior-day closes per symbol for gap calculation
    prev_close_by_sym: Dict[str, pd.Series] = {}
    for sym, g in df.groupby("symbol"):
        g = g.sort_values("timestamp")
        day_close = g.groupby("date")["close"].last().sort_index()
        prev_close_by_sym[sym] = day_close.shift(1)

    feats = []
    for (sym, asof), sub in df.groupby(["symbol", "date"]):
        if len(sub) < cfg.min_bars:
            continue
        sub = sub.sort_values("timestamp")
        price = sub["close"].values.astype(float)
        vol = sub["volume"].values.astype(float)
        if vol.sum() <= 0:
            continue

        # Basic price/volume stats
        vwap = float((price * vol).sum() / vol.sum())
        close = float(price[-1])
        open_px = float(sub["open"].iloc[0])
        high = float(sub["high"].max())
        low = float(sub["low"].min())

        # Returns series for microstructure analysis
        minute_returns = pd.Series(price).pct_change().dropna()
        intraday_vol = float(minute_returns.std(ddof=1)) if not minute_returns.empty else 0.0

        # Order flow imbalance
        ofi = float(_order_flow_imbalance(sub, cfg.ofi_window).iloc[-1]) if len(sub) >= cfg.ofi_window else 0.0

        # Signed volume (cumulative for the day)
        signed_vol_series = _signed_volume(sub)
        signed_volume = float(signed_vol_series.sum()) if not signed_vol_series.empty else 0.0

        # Gap open calculation
        prior_day_close = None
        prev_map = prev_close_by_sym.get(sym)
        if prev_map is not None:
            prior_day_close = prev_map.get(asof, None)
        if prior_day_close is not None and np.isfinite(prior_day_close) and prior_day_close != 0:
            gap_open = (open_px / float(prior_day_close)) - 1.0
        elif sub["close"].shift(1).dropna().size:
            gap_open = (open_px / float(sub["close"].shift(1).dropna().iloc[-1]) - 1.0)
        else:
            gap_open = 0.0

        # Basic derived features
        close_ret = (close / open_px) - 1.0 if open_px else 0.0
        range_spread = (high - low) / vwap if vwap else 0.0
        volume_z = (vol.sum() - vol.mean()) / (vol.std(ddof=1) + 1e-6)

        # Microstructure noise proxies
        roll_spread = _roll_spread_estimator(minute_returns, cfg.roll_window)
        rk_var = _realized_kernel_variance(minute_returns, cfg.realized_kernel_window)
        realized_kernel_vol = np.sqrt(rk_var) if rk_var is not None and not np.isnan(rk_var) else np.nan
        noise_ratio = _microstructure_noise_ratio(minute_returns, cfg.realized_kernel_window)

        # Time-of-day seasonality (use last timestamp of the day)
        last_ts = sub["timestamp"].iloc[-1]
        tod_sin, tod_cos, session_progress = _time_of_day_features(last_ts)
        # Normalize session progress to [0, 1]
        session_progress_norm = session_progress / TRADING_MINUTES if session_progress > 0 else 0.0

        # LOB features (if available)
        lob_feats = _compute_lob_features(sub, cfg)

        feats.append({
            "symbol": sym,
            "date": asof,
            # Core features
            "feature_vwap_dislocation": (close - vwap) / vwap,
            "feature_intraday_vol": intraday_vol,
            "feature_ofi": ofi,
            "feature_signed_volume": signed_volume,
            "feature_gap_open": gap_open,
            "feature_close_ret": close_ret,
            "feature_range_spread": range_spread,
            "feature_volume_z": volume_z,
            # Microstructure noise proxies
            "feature_roll_spread": roll_spread,
            "feature_realized_kernel_vol": realized_kernel_vol,
            "feature_noise_ratio": noise_ratio,
            # Time-of-day seasonality
            "feature_tod_sin": tod_sin,
            "feature_tod_cos": tod_cos,
            "feature_session_progress": session_progress_norm,
            # LOB features
            "feature_avg_spread": lob_feats["avg_spread"],
            "feature_spread_vol": lob_feats["spread_vol"],
            "feature_depth_imbalance": lob_feats["depth_imbalance"],
            "feature_queue_pressure": lob_feats["queue_pressure"],
        })

    return pd.DataFrame(feats)
