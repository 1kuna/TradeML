"""
Delta-Hedged PnL Labeling for Options.

SSOT v2 Section 3.4: Options-specific labels

Computes delta-hedged PnL for options positions, which isolates the
volatility component from directional exposure. This is the primary
label for the options_vol model.

Delta-hedged PnL = Option PnL - Delta × Underlying PnL

Minimal API:
    compute_delta_hedged_pnl(asof_date, underliers, horizon_days) -> pd.DataFrame

Output columns: underlier, entry_date, exit_date, horizon_days, delta_hedged_pnl
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date as Date, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from feature_store.options.iv import BlackScholesIV, Greeks


@dataclass
class DeltaHedgedConfig:
    """Configuration for delta-hedged PnL calculation."""

    horizon_days: int = 5
    risk_free_rate: float = 0.03
    dividend_yield: float = 0.0
    # Moneyness range for ATM options (log-moneyness)
    atm_range: tuple = (-0.05, 0.05)
    # Target tenor in days for primary label
    target_tenor_days: int = 30
    # Tenor tolerance (±days)
    tenor_tolerance: int = 7


def _to_date(d: object) -> Date:
    """Convert to date object."""
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, Date):
        return d
    if isinstance(d, str):
        return datetime.strptime(d, "%Y-%m-%d").date()
    raise TypeError(f"Unsupported date type: {type(d)}")


def _load_options_iv(
    underlier: str,
    asof: Date,
    base_dir: str = "data_layer/curated/options_iv",
) -> Optional[pd.DataFrame]:
    """Load curated options IV data for an underlier on a date."""
    root = Path(base_dir) / f"date={asof.isoformat()}"

    # Try underlier-specific path
    path = root / f"underlier={underlier}" / "data.parquet"
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception as e:
            logger.warning(f"Failed to read {path}: {e}")

    # Try combined file
    combined = root / "data.parquet"
    if combined.exists():
        try:
            df = pd.read_parquet(combined)
            if "underlier" in df.columns:
                return df[df["underlier"] == underlier]
            elif "symbol" in df.columns:
                return df[df["symbol"] == underlier]
        except Exception as e:
            logger.warning(f"Failed to read {combined}: {e}")

    return None


def _load_underlying_prices(
    underlier: str,
    start_date: Date,
    end_date: Date,
    base_dir: str = "data_layer/curated/equities_ohlcv_adj",
) -> Optional[pd.DataFrame]:
    """Load underlying equity prices for a date range."""
    # Try per-symbol file
    candidates = [
        Path(base_dir) / f"{underlier}_adj.parquet",
        Path(base_dir) / f"{underlier}.parquet",
    ]

    for path in candidates:
        if path.exists():
            try:
                df = pd.read_parquet(path)
                df = _normalize_equity_df(df)
                return df[(df["date"] >= start_date) & (df["date"] <= end_date)]
            except Exception:
                continue

    # Try date-partitioned layout
    root = Path(base_dir)
    if not root.exists():
        return None

    frames = []
    current = start_date
    while current <= end_date:
        date_dir = root / f"date={current.isoformat()}"
        data_file = date_dir / "data.parquet"
        if data_file.exists():
            try:
                df = pd.read_parquet(data_file)
                if "symbol" in df.columns:
                    df = df[df["symbol"] == underlier]
                if not df.empty:
                    frames.append(df)
            except Exception:
                pass
        current = current + pd.Timedelta(days=1)

    if frames:
        df = pd.concat(frames, ignore_index=True)
        return _normalize_equity_df(df)

    return None


def _normalize_equity_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize equity dataframe column names."""
    colmap = {}
    if "close_adj" in df.columns:
        colmap.update({
            "open_adj": "open",
            "high_adj": "high",
            "low_adj": "low",
            "close_adj": "close",
            "volume_adj": "volume",
        })

    df = df.rename(columns=colmap).copy()

    if "date" in df.columns and not np.issubdtype(df["date"].dtype, np.datetime64):
        df["date"] = pd.to_datetime(df["date"]).dt.date

    return df


def _select_atm_options(
    iv_df: pd.DataFrame,
    spot: float,
    cfg: DeltaHedgedConfig,
) -> pd.DataFrame:
    """
    Select ATM options within target tenor range.

    Returns options near ATM (within moneyness range) and near target tenor.
    """
    if iv_df.empty:
        return pd.DataFrame()

    # Compute log-moneyness
    if "strike" not in iv_df.columns:
        return pd.DataFrame()

    df = iv_df.copy()
    df["log_moneyness"] = np.log(df["strike"] / spot)

    # Filter to ATM range
    atm_min, atm_max = cfg.atm_range
    df = df[(df["log_moneyness"] >= atm_min) & (df["log_moneyness"] <= atm_max)]

    if df.empty:
        return df

    # Compute days to expiry
    if "expiry" in df.columns:
        df["days_to_expiry"] = (pd.to_datetime(df["expiry"]) - pd.Timestamp.now()).dt.days
    elif "dte" in df.columns:
        df["days_to_expiry"] = df["dte"]
    else:
        return pd.DataFrame()

    # Filter to target tenor range
    min_tenor = cfg.target_tenor_days - cfg.tenor_tolerance
    max_tenor = cfg.target_tenor_days + cfg.tenor_tolerance
    df = df[(df["days_to_expiry"] >= min_tenor) & (df["days_to_expiry"] <= max_tenor)]

    return df


def _compute_option_value_change(
    iv_entry: float,
    iv_exit: float,
    spot_entry: float,
    spot_exit: float,
    strike: float,
    T_entry: float,
    T_exit: float,
    r: float,
    q: float,
    option_type: str = "call",
) -> Dict[str, float]:
    """
    Compute option value change and delta-hedged PnL.

    Returns:
        Dict with option_pnl, underlying_pnl, delta_hedged_pnl
    """
    # Entry values
    if option_type == "call":
        price_entry = BlackScholesIV.call_price(spot_entry, strike, T_entry, r, iv_entry, q)
        price_exit = BlackScholesIV.call_price(spot_exit, strike, T_exit, r, iv_exit, q)
    else:
        price_entry = BlackScholesIV.put_price(spot_entry, strike, T_entry, r, iv_entry, q)
        price_exit = BlackScholesIV.put_price(spot_exit, strike, T_exit, r, iv_exit, q)

    # Entry delta for hedging
    greeks = BlackScholesIV.calculate_greeks(spot_entry, strike, T_entry, r, iv_entry, option_type, q)
    delta = greeks.delta

    # PnL components
    option_pnl = price_exit - price_entry
    underlying_pnl = delta * (spot_exit - spot_entry)
    delta_hedged_pnl = option_pnl - underlying_pnl

    return {
        "option_pnl": option_pnl,
        "underlying_pnl": underlying_pnl,
        "delta_hedged_pnl": delta_hedged_pnl,
        "entry_delta": delta,
        "entry_iv": iv_entry,
        "exit_iv": iv_exit,
    }


def compute_delta_hedged_pnl(
    asof_date: object,
    underliers: Iterable[str],
    cfg: Optional[DeltaHedgedConfig] = None,
) -> pd.DataFrame:
    """
    Compute delta-hedged PnL labels for options positions.

    This isolates the volatility component of options returns by
    removing directional exposure through delta hedging.

    Args:
        asof_date: Entry date for positions
        underliers: Iterable of underlying symbols
        cfg: Configuration for label computation

    Returns:
        DataFrame with columns:
            underlier, entry_date, exit_date, horizon_days,
            delta_hedged_pnl, option_pnl, underlying_pnl,
            entry_iv, exit_iv, entry_delta
    """
    cfg = cfg or DeltaHedgedConfig()
    asof = _to_date(asof_date)
    underliers = list(dict.fromkeys([u.strip().upper() for u in underliers if u]))

    logger.info(f"Computing delta-hedged PnL for {len(underliers)} underliers as of {asof}")

    # Determine exit date
    exit_date = asof + pd.Timedelta(days=cfg.horizon_days)

    rows: List[dict] = []
    missing_data: List[str] = []

    for underlier in underliers:
        # Load entry-date options IV
        iv_entry_df = _load_options_iv(underlier, asof)
        if iv_entry_df is None or iv_entry_df.empty:
            missing_data.append(f"{underlier}:iv_entry")
            continue

        # Load underlying prices
        underlying_df = _load_underlying_prices(underlier, asof, exit_date)
        if underlying_df is None or underlying_df.empty:
            missing_data.append(f"{underlier}:underlying")
            continue

        # Get spot prices
        entry_row = underlying_df[underlying_df["date"] == asof]
        exit_row = underlying_df[underlying_df["date"] == exit_date]

        if entry_row.empty or exit_row.empty:
            # Try to find closest available exit date
            available_dates = sorted(underlying_df["date"].unique())
            valid_exits = [d for d in available_dates if d > asof]
            if not valid_exits:
                missing_data.append(f"{underlier}:exit_date")
                continue
            actual_exit = valid_exits[min(cfg.horizon_days - 1, len(valid_exits) - 1)]
            exit_row = underlying_df[underlying_df["date"] == actual_exit]
            if exit_row.empty or entry_row.empty:
                missing_data.append(f"{underlier}:prices")
                continue
        else:
            actual_exit = exit_date

        spot_entry = float(entry_row["close"].iloc[0])
        spot_exit = float(exit_row["close"].iloc[0])

        # Select ATM options
        atm_options = _select_atm_options(iv_entry_df, spot_entry, cfg)
        if atm_options.empty:
            missing_data.append(f"{underlier}:atm_options")
            continue

        # Load exit-date options IV
        iv_exit_df = _load_options_iv(underlier, actual_exit)
        if iv_exit_df is None or iv_exit_df.empty:
            # Use entry IV as proxy (assume IV unchanged)
            iv_exit_df = iv_entry_df.copy()

        # Compute average delta-hedged PnL across ATM options
        pnl_samples = []

        for _, opt in atm_options.iterrows():
            strike = opt["strike"]
            iv_entry = opt.get("iv", opt.get("implied_volatility"))
            if iv_entry is None or pd.isna(iv_entry):
                continue

            dte_entry = opt.get("days_to_expiry", cfg.target_tenor_days)
            T_entry = max(dte_entry, 1) / 365.0
            T_exit = max(dte_entry - cfg.horizon_days, 1) / 365.0

            # Try to find matching exit IV
            exit_match = iv_exit_df[
                (iv_exit_df["strike"] == strike) &
                (abs(iv_exit_df.get("days_to_expiry", pd.Series([dte_entry - cfg.horizon_days])) - (dte_entry - cfg.horizon_days)) <= 2)
            ]

            if not exit_match.empty:
                iv_exit = exit_match["iv"].iloc[0] if "iv" in exit_match.columns else exit_match.get("implied_volatility", pd.Series([iv_entry])).iloc[0]
            else:
                iv_exit = iv_entry  # Assume unchanged

            if pd.isna(iv_exit):
                iv_exit = iv_entry

            # Compute for both calls and puts, average
            for opt_type in ["call", "put"]:
                try:
                    result = _compute_option_value_change(
                        iv_entry=iv_entry,
                        iv_exit=iv_exit,
                        spot_entry=spot_entry,
                        spot_exit=spot_exit,
                        strike=strike,
                        T_entry=T_entry,
                        T_exit=T_exit,
                        r=cfg.risk_free_rate,
                        q=cfg.dividend_yield,
                        option_type=opt_type,
                    )
                    pnl_samples.append(result)
                except Exception as e:
                    logger.debug(f"Failed to compute PnL for {underlier} {strike} {opt_type}: {e}")

        if not pnl_samples:
            missing_data.append(f"{underlier}:pnl_calc")
            continue

        # Aggregate
        avg_delta_hedged_pnl = np.mean([s["delta_hedged_pnl"] for s in pnl_samples])
        avg_option_pnl = np.mean([s["option_pnl"] for s in pnl_samples])
        avg_underlying_pnl = np.mean([s["underlying_pnl"] for s in pnl_samples])
        avg_entry_iv = np.mean([s["entry_iv"] for s in pnl_samples])
        avg_exit_iv = np.mean([s["exit_iv"] for s in pnl_samples])
        avg_entry_delta = np.mean([abs(s["entry_delta"]) for s in pnl_samples])

        rows.append({
            "underlier": underlier,
            "entry_date": asof,
            "exit_date": actual_exit,
            "horizon_days": cfg.horizon_days,
            "delta_hedged_pnl": float(avg_delta_hedged_pnl),
            "option_pnl": float(avg_option_pnl),
            "underlying_pnl": float(avg_underlying_pnl),
            "entry_iv": float(avg_entry_iv),
            "exit_iv": float(avg_exit_iv),
            "entry_delta": float(avg_entry_delta),
            "num_options": len(pnl_samples),
        })

    if missing_data:
        logger.warning(f"Missing data for: {', '.join(missing_data[:10])}{'...' if len(missing_data) > 10 else ''}")

    if not rows:
        return pd.DataFrame(columns=[
            "underlier", "entry_date", "exit_date", "horizon_days",
            "delta_hedged_pnl", "option_pnl", "underlying_pnl",
            "entry_iv", "exit_iv", "entry_delta", "num_options"
        ])

    return pd.DataFrame(rows)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute delta-hedged PnL labels")
    parser.add_argument("--asof", type=str, required=True, help="Entry date YYYY-MM-DD")
    parser.add_argument("--underliers", type=str, nargs="+", help="Underliers e.g., AAPL MSFT SPY")
    parser.add_argument("--horizon", type=int, default=5, help="Horizon in days")
    args = parser.parse_args()

    cfg = DeltaHedgedConfig(horizon_days=args.horizon)
    out = compute_delta_hedged_pnl(args.asof, args.underliers, cfg)
    print(out.to_string(index=False))
