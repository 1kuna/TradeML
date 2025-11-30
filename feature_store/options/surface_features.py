"""
Options Surface Feature Engineering.

SSOT v2 Section 3.2: Options_vol - features

Extracts features from IV surfaces for the options_vol model:
- Surface shape: ATM IV level, skew, curvature
- Term structure: slope (short vs long tenor IV)
- Relative level: IV rank/percentile vs trailing distribution
- Structure/context: time to expiry, moneyness buckets, distance to events

Minimal API:
    compute_options_features(asof_date, universe) -> pd.DataFrame

Output columns: `underlier`, `asof`, `feature_*`
"""

from __future__ import annotations

import os
from datetime import date as Date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from .svi import SVIParams


def _to_date(d: object) -> Date:
    """Convert to date object."""
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, Date):
        return d
    if isinstance(d, str):
        return datetime.strptime(d, "%Y-%m-%d").date()
    raise TypeError(f"Unsupported date type: {type(d)}")


def _load_surface_history(
    underlier: str,
    base_dir: str = "data_layer/curated/options_surface",
    lookback_days: int = 252,
) -> Optional[pd.DataFrame]:
    """
    Load options surface history for an underlier.

    Scans date-partitioned directories under base_dir for the underlier's surfaces.
    """
    root = Path(base_dir)
    if not root.exists():
        return None

    frames = []
    cutoff = datetime.now().date() - timedelta(days=lookback_days)

    # Scan date partitions
    for date_dir in sorted(root.glob("date=*")):
        try:
            date_str = date_dir.name.split("=")[-1]
            dt = datetime.strptime(date_str, "%Y-%m-%d").date()
            if dt < cutoff:
                continue
        except Exception:
            continue

        # Look for underlier-specific file or combined file
        underlier_file = date_dir / f"underlier={underlier}" / "data.parquet"
        if underlier_file.exists():
            try:
                df = pd.read_parquet(underlier_file)
                df["date"] = dt
                frames.append(df)
            except Exception:
                pass
        else:
            # Try combined file
            combined_file = date_dir / "data.parquet"
            if combined_file.exists():
                try:
                    df = pd.read_parquet(combined_file)
                    if "underlier" in df.columns:
                        df = df[df["underlier"] == underlier]
                    if not df.empty:
                        if "date" not in df.columns:
                            df["date"] = dt
                        frames.append(df)
                except Exception:
                    pass

    if not frames:
        return None

    return pd.concat(frames, ignore_index=True)


def _load_iv_history(
    underlier: str,
    base_dir: str = "data_layer/curated/options_iv",
    lookback_days: int = 252,
) -> Optional[pd.DataFrame]:
    """
    Load options IV history for an underlier.

    Falls back to options_iv data if surface data is not available.
    """
    root = Path(base_dir)
    if not root.exists():
        return None

    frames = []
    cutoff = datetime.now().date() - timedelta(days=lookback_days)

    for date_dir in sorted(root.glob("date=*")):
        try:
            date_str = date_dir.name.split("=")[-1]
            dt = datetime.strptime(date_str, "%Y-%m-%d").date()
            if dt < cutoff:
                continue
        except Exception:
            continue

        # Look for underlier-specific or combined file
        for pattern in [f"underlier={underlier}/data.parquet", "data.parquet"]:
            path = date_dir / pattern
            if path.exists():
                try:
                    df = pd.read_parquet(path)
                    if "underlier" in df.columns:
                        df = df[df["underlier"] == underlier]
                    elif "symbol" in df.columns:
                        df = df[df["symbol"] == underlier]
                    if not df.empty:
                        if "date" not in df.columns:
                            df["date"] = dt
                        frames.append(df)
                    break
                except Exception:
                    pass

    if not frames:
        return None

    return pd.concat(frames, ignore_index=True)


def _extract_atm_iv(surface_df: pd.DataFrame, asof: Date) -> Optional[float]:
    """Extract ATM IV from surface data for a specific date."""
    df = surface_df[surface_df["date"] == asof]
    if df.empty:
        return None

    # Try to find ATM IV from SVI parameters or direct IV
    if "a" in df.columns and "b" in df.columns:
        # SVI parameterization: ATM is at k=m, so w(0) ≈ a + b*sigma for k≈m≈0
        row = df.iloc[0]
        try:
            a, b, m, sigma = row["a"], row["b"], row.get("m", 0), row.get("sigma", 0.1)
            T = row.get("T", row.get("tenor", 30 / 365))
            atm_var = a + b * np.sqrt(sigma**2)
            return np.sqrt(max(atm_var / T, 0)) if T > 0 else None
        except Exception:
            pass

    if "atm_iv" in df.columns:
        return float(df["atm_iv"].iloc[0])

    if "iv" in df.columns:
        # Use average as proxy
        return float(df["iv"].mean())

    return None


def _extract_skew(surface_df: pd.DataFrame, asof: Date) -> Optional[float]:
    """Extract skew from surface data (25-delta put - 25-delta call spread)."""
    df = surface_df[surface_df["date"] == asof]
    if df.empty:
        return None

    if "rho" in df.columns:
        # SVI rho is related to skew
        return float(df["rho"].iloc[0])

    if "skew_25d" in df.columns:
        return float(df["skew_25d"].iloc[0])

    # Try to compute from strike-level data
    if "strike" in df.columns and "iv" in df.columns and "moneyness" in df.columns:
        otm_puts = df[df["moneyness"] < -0.1]["iv"].mean()
        otm_calls = df[df["moneyness"] > 0.1]["iv"].mean()
        if not pd.isna(otm_puts) and not pd.isna(otm_calls):
            return float(otm_puts - otm_calls)

    return None


def _extract_curvature(surface_df: pd.DataFrame, asof: Date) -> Optional[float]:
    """Extract curvature (butterfly) from surface data."""
    df = surface_df[surface_df["date"] == asof]
    if df.empty:
        return None

    if "b" in df.columns:
        # SVI b parameter controls curvature
        return float(df["b"].iloc[0])

    if "curvature" in df.columns or "butterfly" in df.columns:
        col = "curvature" if "curvature" in df.columns else "butterfly"
        return float(df[col].iloc[0])

    return None


def _extract_term_structure_slope(surface_df: pd.DataFrame, asof: Date) -> Optional[float]:
    """Extract term structure slope (long tenor - short tenor ATM IV)."""
    df = surface_df[surface_df["date"] == asof]
    if df.empty:
        return None

    if "tenor" in df.columns and "atm_iv" in df.columns:
        short_tenors = df[df["tenor"] <= 30 / 365]["atm_iv"]
        long_tenors = df[df["tenor"] >= 90 / 365]["atm_iv"]
        if not short_tenors.empty and not long_tenors.empty:
            return float(long_tenors.mean() - short_tenors.mean())

    if "term_slope" in df.columns:
        return float(df["term_slope"].iloc[0])

    return None


def _compute_iv_rank(atm_iv_series: pd.Series, current_iv: float, window: int = 252) -> float:
    """Compute IV rank (percentile) vs trailing distribution."""
    if len(atm_iv_series) < 20:
        return np.nan

    recent = atm_iv_series.tail(window).dropna()
    if len(recent) < 20 or current_iv is None or pd.isna(current_iv):
        return np.nan

    # Percentile rank
    rank = (recent < current_iv).sum() / len(recent)
    return float(rank)


def _compute_underlier_features(
    underlier: str,
    asof: Date,
    surface_hist: Optional[pd.DataFrame],
    iv_hist: Optional[pd.DataFrame],
    standardize_window: int = 252,
) -> Optional[dict]:
    """Compute PIT-safe options features for a single underlier as-of date."""

    # Use surface history if available, else fall back to IV history
    hist = surface_hist if surface_hist is not None else iv_hist
    if hist is None or hist.empty:
        return None

    # Restrict to history up to as-of
    hist = hist[hist["date"] <= asof].copy()
    if hist.empty:
        return None

    # Check if we have data for as-of date
    asof_data = hist[hist["date"] == asof]
    if asof_data.empty:
        return None

    # Extract surface shape features
    atm_iv = _extract_atm_iv(hist, asof)
    skew = _extract_skew(hist, asof)
    curvature = _extract_curvature(hist, asof)
    term_slope = _extract_term_structure_slope(hist, asof)

    # Compute IV historical series for ranking
    atm_iv_series = pd.Series([
        _extract_atm_iv(hist, d) for d in hist["date"].unique()
    ])

    # IV rank
    iv_rank = _compute_iv_rank(atm_iv_series, atm_iv, standardize_window)

    # Rolling standardization
    def _rolling_zscore(current: Optional[float], series: pd.Series, window: int) -> float:
        if current is None or pd.isna(current) or len(series) < 20:
            return np.nan
        recent = series.tail(window).dropna()
        if len(recent) < 20:
            return np.nan
        mu = recent.mean()
        sd = recent.std()
        if sd == 0 or pd.isna(sd):
            return np.nan
        return float((current - mu) / sd)

    features = {
        "underlier": underlier,
        "asof": asof,
        # Surface shape (z-scored)
        "feature_atm_iv": _rolling_zscore(atm_iv, atm_iv_series, standardize_window),
        "feature_skew": float(skew) if skew is not None and not pd.isna(skew) else np.nan,
        "feature_curvature": float(curvature) if curvature is not None and not pd.isna(curvature) else np.nan,
        "feature_term_slope": float(term_slope) if term_slope is not None and not pd.isna(term_slope) else np.nan,
        # Relative level
        "feature_iv_rank": float(iv_rank) if not pd.isna(iv_rank) else np.nan,
        # Raw values for reference
        "atm_iv_raw": float(atm_iv) if atm_iv is not None else np.nan,
    }

    return features


def compute_options_features(
    asof_date: object,
    universe: Iterable[str],
    surface_dir: str = "data_layer/curated/options_surface",
    iv_dir: str = "data_layer/curated/options_iv",
) -> pd.DataFrame:
    """
    Compute PIT-safe options surface features for a given as-of date.

    Args:
        asof_date: date or 'YYYY-MM-DD' string
        universe: iterable of underliers to compute features for
        surface_dir: path to curated options surfaces
        iv_dir: path to curated options IV (fallback)

    Returns:
        DataFrame with columns: underlier, asof, feature_*
    """
    asof = _to_date(asof_date)
    universe = list(dict.fromkeys([s.strip().upper() for s in universe if s]))

    logger.info(f"Computing options features for {len(universe)} underliers as of {asof}")

    rows: List[dict] = []
    missing: List[str] = []

    for underlier in universe:
        # Try surface data first, then IV data
        surface_hist = _load_surface_history(underlier, surface_dir)
        iv_hist = _load_iv_history(underlier, iv_dir) if surface_hist is None else None

        if surface_hist is None and iv_hist is None:
            missing.append(underlier)
            continue

        feat = _compute_underlier_features(underlier, asof, surface_hist, iv_hist)
        if feat is not None:
            rows.append(feat)
        else:
            logger.debug(f"Insufficient options data for {underlier} at {asof}")

    if missing:
        logger.warning(
            f"No options data found for: {', '.join(missing[:10])}{'...' if len(missing) > 10 else ''}"
        )

    if not rows:
        return pd.DataFrame(columns=["underlier", "asof"])

    df = pd.DataFrame(rows)
    cols = ["underlier", "asof"] + [c for c in df.columns if c.startswith("feature_")]
    df = df[[c for c in cols if c in df.columns]]

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute options surface features")
    parser.add_argument("--asof", type=str, required=True, help="As-of date YYYY-MM-DD")
    parser.add_argument("--underliers", type=str, nargs="+", help="Underliers e.g., AAPL MSFT SPY")
    args = parser.parse_args()

    out = compute_options_features(args.asof, args.underliers)
    print(out.to_string(index=False))
