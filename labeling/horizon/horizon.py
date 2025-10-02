"""
Horizon return labels (execution-aligned, PIT-safe).

Generates k-step forward returns for each symbol as-of a given date.

Output columns: symbol, entry_date, exit_date, horizon_days, forward_return
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
    candidates = [
        os.path.join(base_dir, f"{symbol}_adj.parquet"),
        os.path.join(base_dir, f"{symbol}.parquet"),
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return pd.read_parquet(path)
            except Exception as e:  # pragma: no cover
                logger.warning(f"Failed to read {path}: {e}")
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


def horizon_returns(asof_date: object, universe: Iterable[str], horizon_days: int = 5) -> pd.DataFrame:
    """Compute k-step forward returns for each symbol at a snapshot date.

    Args:
        asof_date: date or 'YYYY-MM-DD' string
        universe: iterable of symbols
        horizon_days: forward horizon in trading days

    Returns:
        DataFrame: symbol, entry_date, exit_date, horizon_days, forward_return
    """
    asof = _to_date(asof_date)
    base_dir = os.getenv("CURATED_EQUITY_BARS_ADJ_DIR", os.path.join("data_layer", "curated", "equities_ohlcv_adj"))

    rows: List[dict] = []
    for sym in {s.strip().upper() for s in universe if s}:
        hist = _load_symbol_history(sym, base_dir)
        if hist is None:
            logger.debug(f"Missing history for {sym}")
            continue
        df = _normalize(hist).sort_values("date")
        # ensure we have an entry row at as-of
        if asof not in set(df["date"]):
            logger.debug(f"No as-of bar for {sym} at {asof}")
            continue

        # Locate index of as-of and compute exit index = as-of idx + horizon_days
        idx = df.index[df["date"] == asof][0]
        pos = df.index.get_loc(idx)
        exit_pos = pos + horizon_days

        if exit_pos >= len(df.index):
            # not enough future bars to compute realized forward return
            logger.debug(f"Insufficient future bars for {sym} at {asof} (k={horizon_days})")
            continue

        entry_px = float(df.loc[idx, "close"])  # use close at entry
        exit_idx = df.index[exit_pos]
        exit_px = float(df.loc[exit_idx, "close"])  # close at horizon

        fwd = (exit_px / entry_px) - 1.0

        rows.append(
            {
                "symbol": sym,
                "entry_date": asof,
                "exit_date": df.loc[exit_idx, "date"],
                "horizon_days": int(horizon_days),
                "forward_return": float(fwd),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["symbol", "entry_date", "exit_date", "horizon_days", "forward_return"])  # empty

    return pd.DataFrame(rows)


if __name__ == "__main__":  # smoke test
    import argparse

    p = argparse.ArgumentParser(description="Horizon return labeling")
    p.add_argument("--asof", required=True)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--symbols", nargs="+", required=True)
    a = p.parse_args()
    out = horizon_returns(a.asof, a.symbols, a.k)
    print(out.head().to_string(index=False))

