"""
Triple-barrier classification labels (PIT-safe).

Minimal API (from blueprint):
    labels.triple_barrier(date, universe, tp_sigma, sl_sigma, max_h) -> pd.DataFrame

Output columns: symbol, entry_date, exit_date, label, outcome, meta

Rules:
- Barriers scale with rolling volatility (sigma) estimated from past returns.
- Entry at as-of close; walk forward up to max_h trading days.
- If upper barrier hit first -> label=+1; lower first -> label=-1; else 0 at horizon.
- Outcome = realized return at exit (barrier touch or horizon reach).
"""

from __future__ import annotations

import os
from datetime import date as Date, datetime
from typing import Dict, Iterable, List, Optional, Tuple

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
    paths = [
        os.path.join(base_dir, f"{symbol}_adj.parquet"),
        os.path.join(base_dir, f"{symbol}.parquet"),
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                return pd.read_parquet(p)
            except Exception as e:  # pragma: no cover
                logger.warning(f"Failed to read {p}: {e}")
    return None


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    colmap = {}
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
    out = df.rename(columns=colmap).copy()
    if not np.issubdtype(out["date"].dtype, np.datetime64):
        out["date"] = pd.to_datetime(out["date"]).dt.date
    return out


def _rolling_sigma(ret: pd.Series, window: int = 20) -> pd.Series:
    return ret.rolling(window=window, min_periods=max(5, window // 2)).std(ddof=1)


def _scan_barriers(
    path: pd.DataFrame,
    entry_px: float,
    tp_mult: float,
    sl_mult: float,
) -> Tuple[int, float, Date]:
    """Walk forward and detect first barrier touch.

    path must have columns: date, high, low, close (normalized names).
    Returns (label, outcome_return, exit_date).
    """
    tp_level = entry_px * (1.0 + tp_mult)
    sl_level = entry_px * (1.0 - sl_mult)

    for _, row in path.iterrows():
        hi = float(row.get("high", row["close"]))
        lo = float(row.get("low", row["close"]))
        dt = row["date"]

        if not np.isnan(hi) and hi >= tp_level:
            # TP touched
            outcome = (tp_level / entry_px) - 1.0
            return 1, outcome, dt

        if not np.isnan(lo) and lo <= sl_level:
            # SL touched
            outcome = (sl_level / entry_px) - 1.0
            return -1, outcome, dt

    # Neither touched: settle at last close
    last = path.iloc[-1]
    exit_px = float(last["close"]) if "close" in last else float(last["high"])  # fallback
    outcome = (exit_px / entry_px) - 1.0
    return 0, outcome, last["date"]


def triple_barrier(
    asof_date: object,
    universe: Iterable[str],
    tp_sigma: float,
    sl_sigma: float,
    max_h: int,
    vol_window: int = 20,
) -> pd.DataFrame:
    """Compute triple-barrier classification labels for a given snapshot.

    Args:
        asof_date: date or 'YYYY-MM-DD'
        universe: iterable of symbols
        tp_sigma: take-profit multiple of rolling sigma (e.g., 2.0)
        sl_sigma: stop-loss multiple of rolling sigma (e.g., 1.0)
        max_h: maximum holding period in trading days
        vol_window: trailing window for sigma estimation

    Returns:
        DataFrame: symbol, entry_date, exit_date, label, outcome, meta
    """
    asof = _to_date(asof_date)
    base_dir = os.getenv("CURATED_EQUITY_BARS_ADJ_DIR", os.path.join("data_layer", "curated", "equities_ohlcv_adj"))

    rows: List[Dict] = []
    for sym in {s.strip().upper() for s in universe if s}:
        hist = _load_symbol_history(sym, base_dir)
        if hist is None:
            logger.debug(f"Missing history for {sym}")
            continue
        df = _normalize(hist).sort_values("date").copy()

        # Must have as-of bar
        if asof not in set(df["date"]):
            logger.debug(f"No as-of bar for {sym} at {asof}")
            continue

        # Compute rolling sigma from strictly prior returns
        df["ret_1d"] = df["close"].pct_change()
        df["sigma"] = _rolling_sigma(df["ret_1d"], window=vol_window)

        # Locate as-of index and entry context
        idx = df.index[df["date"] == asof][0]
        pos = df.index.get_loc(idx)
        if pos < 1:
            logger.debug(f"Insufficient history for sigma at {sym} {asof}")
            continue

        entry_px = float(df.loc[idx, "close"])  # entry at as-of close
        sigma = float(df.iloc[pos - 1]["sigma"])  # use sigma up to prior bar (PIT)
        if pd.isna(sigma) or sigma <= 0:
            logger.debug(f"Invalid sigma for {sym} at {asof}")
            continue

        # Define forward path up to max_h trading days (exclude as-of row)
        fwd = df.iloc[pos + 1 : pos + 1 + max_h].copy()
        if fwd.empty:
            logger.debug(f"No forward path for {sym} at {asof}")
            continue

        label, outcome, exit_dt = _scan_barriers(
            fwd, entry_px=entry_px, tp_mult=tp_sigma * sigma, sl_mult=sl_sigma * sigma
        )

        rows.append(
            {
                "symbol": sym,
                "entry_date": asof,
                "exit_date": exit_dt,
                "label": int(label),
                "outcome": float(outcome),
                "meta": {
                    "tp_mult": float(tp_sigma * sigma),
                    "sl_mult": float(sl_sigma * sigma),
                    "sigma": float(sigma),
                    "vol_window": int(vol_window),
                    "max_h": int(max_h),
                },
            }
        )

    if not rows:
        return pd.DataFrame(columns=["symbol", "entry_date", "exit_date", "label", "outcome", "meta"])  # empty

    out = pd.DataFrame(rows)
    return out


if __name__ == "__main__":  # smoke test
    import argparse

    p = argparse.ArgumentParser(description="Triple-barrier labeling")
    p.add_argument("--asof", required=True)
    p.add_argument("--tp", type=float, default=2.0)
    p.add_argument("--sl", type=float, default=1.0)
    p.add_argument("--max_h", type=int, default=10)
    p.add_argument("--symbols", nargs="+", required=True)
    a = p.parse_args()
    out = triple_barrier(a.asof, a.symbols, a.tp, a.sl, a.max_h)
    print(out.head().to_string(index=False))

