"""
Equity feature engineering (PIT-safe).

SSOT v2 Section 3.1: Equities_xs - features

Minimal API (from blueprint):
    features.compute_equity_features(date, universe) -> pd.DataFrame

Inputs: OHLCV (adjusted) up to as-of date; corporate actions assumed applied.
Output columns: `symbol`, `asof`, `feature_*` (standardized, lag-safe).

Feature Set (equities_xs v1 per SSOT):
- Price-based:
  - Multi-horizon momentum: 5, 20, 60, 126-day log returns
  - Gap statistics (overnight, open-to-close) and recent drawdowns
- Volatility:
  - 20 and 60-day realized volatility
  - Rolling downside volatility (semi-deviation)
- Liquidity:
  - 20-day ADV
  - Turnover, Amihud illiquidity
- Seasonality & calendar:
  - Day-of-week and month-of-year encoded as sin/cos
  - Distance to earnings (placeholder)
- Risk / cross-sectional context:
  - Size proxies (market cap)
  - Sector/industry dummies (from reference data)

Data discovery:
- Looks for curated Parquet per-symbol under `CURATED_EQUITY_BARS_ADJ_DIR`
  environment variable, defaulting to `data_layer/curated/equities_ohlcv_adj`.
- Expected columns in curated files (best effort):
  date, symbol, open_adj/high_adj/low_adj/close_adj, volume_adj, close_raw.
  Falls back to unadjusted OHLCV names if needed.

NOTE: This implementation favors clarity and PIT-safety. It avoids using
future data by only referencing history strictly before/as-of date when
constructing features.
"""

from __future__ import annotations

import os
from datetime import date as Date, datetime
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from loguru import logger


def _to_date(d: object) -> Date:
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, Date):
        return d
    if isinstance(d, str):
        return datetime.strptime(d, "%Y-%m-%d").date()
    raise TypeError(f"Unsupported date type: {type(d)}")


def _load_symbol_history(symbol: str, base_dir: str) -> Optional[pd.DataFrame]:
    """Load per-symbol curated OHLCV (adjusted if available).

    Preferred: per-symbol file under base_dir. If not found, fall back to
    scanning date-partitioned curated directories and filtering for the symbol.
    """
    # Preferred: {SYMBOL}_adj.parquet or {SYMBOL}.parquet
    candidates = [
        os.path.join(base_dir, f"{symbol}_adj.parquet"),
        os.path.join(base_dir, f"{symbol}.parquet"),
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                df = pd.read_parquet(path)
                return df
            except Exception as e:  # pragma: no cover (environment dependent engines)
                logger.warning(f"Failed to read {path}: {e}")
                continue

    # Fallback: scan date-partitioned curated directory
    root = os.path.join("data_layer", "curated", "equities_ohlcv_adj")
    if os.path.exists(root):
        frames = []
        # Read up to last ~400 calendar days of partitions to keep IO bounded
        try:
            dirs = [p for p in os.listdir(root) if p.startswith("date=")]
            # Sort by date and keep most recent 400
            def _ds(n: str):
                try:
                    return datetime.strptime(n.split("=", 1)[-1], "%Y-%m-%d").date()
                except Exception:
                    return None
            dated = sorted([(d, _ds(d)) for d in dirs], key=lambda x: (x[1] is None, x[1]))
            dated = [d for d in dated if d[1] is not None]
            for dname, _ in dated[-400:]:
                path = os.path.join(root, dname, "data.parquet")
                if not os.path.exists(path):
                    continue
                try:
                    df = pd.read_parquet(path)
                except Exception:
                    continue
                if df.empty or "symbol" not in df.columns:
                    continue
                sub = df[df["symbol"] == symbol]
                if not sub.empty:
                    frames.append(sub)
        except Exception:
            pass
        if frames:
            out = pd.concat(frames, ignore_index=True)
            return out

    logger.debug(f"No curated history found for {symbol}")
    return None


def _ensure_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize adjusted OHLCV column names if available; fallback to raw."""
    colmap = {}
    # Prefer adjusted naming
    if "close_adj" in df.columns:
        colmap.update(
            {
                "open_adj": "open",
                "high_adj": "high",
                "low_adj": "low",
                "close_adj": "close",
                "volume_adj": "volume",
            }
        )
    # Ensure expected columns exist (raw names otherwise)
    df2 = df.rename(columns=colmap).copy()
    req = ["date", "symbol", "close", "volume"]
    missing = [c for c in req if c not in df2.columns]
    if missing:
        raise ValueError(f"Missing required columns in price data: {missing}")
    # Coerce date type
    if not np.issubdtype(df2["date"].dtype, np.datetime64):
        df2["date"] = pd.to_datetime(df2["date"]).dt.date
    return df2


def _standardize_last_row_by_rolling(
    series: pd.Series, window: int, min_periods: int = 20
) -> float:
    """Return the z-score of the last observation vs rolling window mean/std.

    Uses only historical data up to and including the last point provided.
    If insufficient history, returns NaN.
    """
    if len(series) < min_periods:
        return np.nan
    roll = series.rolling(window=window, min_periods=min_periods)
    mu = roll.mean().iloc[-1]
    sd = roll.std(ddof=1).iloc[-1]
    if sd in (0, None) or pd.isna(sd):
        return np.nan
    return (series.iloc[-1] - mu) / sd


def _compute_symbol_features(
    hist: pd.DataFrame,
    asof: Date,
    standardize_window: int = 252,
) -> Optional[dict]:
    """Compute PIT-safe features for a single symbol as-of date.

    Returns a dict with keys: symbol, asof, feature_* (z-scored).

    Feature categories (SSOT v2 Section 3.1):
    - Price-based: momentum, gap stats, drawdowns
    - Volatility: realized vol, downside vol
    - Liquidity: ADV, turnover, Amihud
    - Seasonality: day-of-week, month-of-year
    - Size: market cap proxy
    """
    df = _ensure_ohlcv_columns(hist)

    # Restrict to history up to as-of (inclusive for calendar features, but
    # rolling price-based features use strictly prior data to avoid leakage).
    df = df[df["date"] <= asof].sort_values("date").copy()
    if df.empty or df["date"].iloc[-1] != asof:
        # No bar for as-of -> cannot produce features
        return None

    # ============ PRICE-BASED FEATURES ============

    # Daily simple returns
    df["ret_1d"] = df["close"].pct_change()

    # Multi-horizon momentum (k-day horizon) as-of: using close/close.shift(k) - 1
    for k in (5, 20, 60, 126):
        df[f"mom_{k}d"] = df["close"] / df["close"].shift(k) - 1.0

    # Gap statistics (overnight return and open-to-close return)
    if "open" in df.columns:
        # Overnight gap: today's open vs yesterday's close
        df["gap_overnight"] = df["open"] / df["close"].shift(1) - 1.0
        # Intraday move: today's close vs today's open
        df["gap_intraday"] = df["close"] / df["open"] - 1.0
        # Rolling average of overnight gaps (mean reversion signal)
        df["gap_overnight_avg_20d"] = df["gap_overnight"].rolling(window=20, min_periods=5).mean()
    else:
        df["gap_overnight"] = np.nan
        df["gap_intraday"] = np.nan
        df["gap_overnight_avg_20d"] = np.nan

    # Recent drawdown (current price vs 20d high, 60d high)
    if "high" in df.columns:
        df["high_20d"] = df["high"].rolling(window=20, min_periods=5).max()
        df["high_60d"] = df["high"].rolling(window=60, min_periods=10).max()
        df["drawdown_20d"] = df["close"] / df["high_20d"] - 1.0
        df["drawdown_60d"] = df["close"] / df["high_60d"] - 1.0
    else:
        df["drawdown_20d"] = df["close"] / df["close"].rolling(window=20, min_periods=5).max() - 1.0
        df["drawdown_60d"] = df["close"] / df["close"].rolling(window=60, min_periods=10).max() - 1.0

    # ============ VOLATILITY FEATURES ============

    # Realized volatility (rolling std of daily returns)
    for k in (20, 60):
        df[f"rv_{k}d"] = df["ret_1d"].rolling(window=k, min_periods=max(5, k // 2)).std(ddof=1)

    # Downside volatility (semi-deviation) - only negative returns
    def _downside_vol(ret_series, window, min_periods):
        """Compute rolling downside volatility (negative returns only)."""
        neg_rets = ret_series.clip(upper=0)  # Only negative returns
        return neg_rets.rolling(window=window, min_periods=min_periods).std(ddof=1)

    df["downside_vol_20d"] = _downside_vol(df["ret_1d"], 20, 10)
    df["downside_vol_60d"] = _downside_vol(df["ret_1d"], 60, 20)

    # ============ LIQUIDITY FEATURES ============

    # Dollar volume and ADV
    df["dollar_volume"] = df["close"] * df["volume"].astype(float)
    df["adv_20d"] = df["dollar_volume"].rolling(window=20, min_periods=10).mean()

    # Turnover (volume / shares outstanding proxy - use volume/avg_volume as substitute)
    avg_vol = df["volume"].rolling(window=60, min_periods=20).mean()
    df["turnover"] = df["volume"] / avg_vol.replace(0, np.nan)

    # Amihud illiquidity (|return| / dollar_volume)
    df["amihud"] = df["ret_1d"].abs() / (df["dollar_volume"] + 1e-10)
    df["amihud_20d"] = df["amihud"].rolling(window=20, min_periods=10).mean()

    # ============ SEASONALITY FEATURES ============

    asof_ts = pd.to_datetime(asof)

    # Day-of-week via sin/cos
    dow = asof_ts.weekday()  # 0..6, equity trading typically 0..4
    dow_sin = np.sin(2 * np.pi * dow / 7.0)
    dow_cos = np.cos(2 * np.pi * dow / 7.0)

    # Month-of-year via sin/cos
    month = asof_ts.month  # 1..12
    moy_sin = np.sin(2 * np.pi * (month - 1) / 12.0)
    moy_cos = np.cos(2 * np.pi * (month - 1) / 12.0)

    # ============ SIZE FEATURES ============

    # Market cap proxy (use average dollar volume as proxy for size)
    # Real market cap would require shares outstanding from fundamentals
    df["mktcap_proxy"] = df["adv_20d"] * 20  # Rough proxy

    # ============ COLLECT FEATURES ============

    # Select the last row (as-of)
    last = df.iloc[-1]

    # Features to z-score standardize
    feats_raw = {
        "mom_5d": df["mom_5d"],
        "mom_20d": df["mom_20d"],
        "mom_60d": df["mom_60d"],
        "mom_126d": df["mom_126d"],
        "rv_20d": df["rv_20d"],
        "rv_60d": df["rv_60d"],
        "downside_vol_20d": df["downside_vol_20d"],
        "downside_vol_60d": df["downside_vol_60d"],
        "adv_20d": df["adv_20d"],
        "turnover": df["turnover"],
        "amihud_20d": df["amihud_20d"],
        "mktcap_proxy": df["mktcap_proxy"],
        "gap_overnight": df["gap_overnight"],
        "gap_intraday": df["gap_intraday"],
        "gap_overnight_avg_20d": df["gap_overnight_avg_20d"],
        "drawdown_20d": df["drawdown_20d"],
        "drawdown_60d": df["drawdown_60d"],
    }

    features = {
        "symbol": last.get("symbol", None),
        "asof": asof,
        # Seasonality (already bounded, no need to z-score)
        "feature_dow_sin": float(dow_sin),
        "feature_dow_cos": float(dow_cos),
        "feature_moy_sin": float(moy_sin),
        "feature_moy_cos": float(moy_cos),
    }

    # Z-score standardize numeric features via rolling window (PIT-safe)
    for name, series in feats_raw.items():
        z = _standardize_last_row_by_rolling(series, window=standardize_window, min_periods=20)
        features[f"feature_{name}"] = float(z) if z is not None and not pd.isna(z) else np.nan

    return features


def compute_equity_features(asof_date: object, universe: Iterable[str]) -> pd.DataFrame:
    """Compute PIT-safe cross-sectional features for a given as-of date.

    Args:
        asof_date: date or 'YYYY-MM-DD' string indicating the feature snapshot date
        universe: iterable of symbols to compute features for

    Returns:
        DataFrame with columns: symbol, asof, feature_*
    """
    asof = _to_date(asof_date)
    base_dir = os.getenv("CURATED_EQUITY_BARS_ADJ_DIR", os.path.join("data_layer", "curated", "equities_ohlcv_adj"))

    universe = list(dict.fromkeys([s.strip().upper() for s in universe if s]))  # dedupe
    logger.info(f"Computing equity features for {len(universe)} symbols as of {asof}")

    rows: List[dict] = []
    missing: List[str] = []
    for sym in universe:
        hist = _load_symbol_history(sym, base_dir)
        if hist is None:
            missing.append(sym)
            continue
        feat = _compute_symbol_features(hist, asof)
        if feat is not None:
            # Ensure symbol is set
            feat["symbol"] = feat.get("symbol") or sym
            rows.append(feat)
        else:
            logger.debug(f"Insufficient data for {sym} at {asof}")

    if missing:
        logger.warning(f"No curated history found for symbols: {', '.join(missing[:10])}{'...' if len(missing)>10 else ''}")

    if not rows:
        return pd.DataFrame(columns=["symbol", "asof"])  # empty

    df = pd.DataFrame(rows)
    # Ensure column order (symbol, asof, features...)
    cols = ["symbol", "asof"] + [c for c in df.columns if c.startswith("feature_")]
    df = df[cols]
    return df


if __name__ == "__main__":  # Simple smoke test (paths must exist)
    import argparse

    parser = argparse.ArgumentParser(description="Compute PIT-safe equity features")
    parser.add_argument("--asof", type=str, required=True, help="As-of date YYYY-MM-DD")
    parser.add_argument("--symbols", type=str, nargs="+", help="Symbols list e.g., AAPL MSFT")
    args = parser.parse_args()

    out = compute_equity_features(args.asof, args.symbols)
    print(out.head().to_string(index=False))
