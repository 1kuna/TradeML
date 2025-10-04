"""
Curated data loaders for equities OHLCV panels.

Utilities to load per-symbol curated Parquet into a unified panel.
Normalizes adjusted column names into standard: date, symbol, open, high, low, close, volume.
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


def _read_symbol(base_dir: str, symbol: str) -> Optional[pd.DataFrame]:
    paths = [
        os.path.join(base_dir, f"{symbol}_adj.parquet"),
        os.path.join(base_dir, f"{symbol}.parquet"),
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                df = pd.read_parquet(p)
                df["symbol"] = symbol
                return df
            except Exception as e:  # pragma: no cover
                logger.warning(f"Failed to read {p}: {e}")
                continue
    return None


def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
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
    req = ["date", "symbol", "close", "volume"]
    missing = [c for c in req if c not in out.columns]
    if missing:
        raise ValueError(f"Missing columns in curated data: {missing}")
    if not np.issubdtype(out["date"].dtype, np.datetime64):
        out["date"] = pd.to_datetime(out["date"]).dt.date
    # Ensure high/low exist for triple-barrier; fallback to close
    for c in ("high", "low", "open"):
        if c not in out.columns:
            out[c] = out["close"]
    return out[["date", "symbol", "open", "high", "low", "close", "volume"]]


def load_price_panel(
    universe: Iterable[str],
    start_date: object,
    end_date: object,
    base_dir_env: str = "CURATED_EQUITY_BARS_ADJ_DIR",
    default_dir: str = os.path.join("data_layer", "curated", "equities_ohlcv_adj"),
) -> pd.DataFrame:
    """Load normalized OHLCV panel for symbols in [start_date, end_date]."""
    start = _to_date(start_date)
    end = _to_date(end_date)
    base_dir = os.getenv(base_dir_env, default_dir)

    frames: List[pd.DataFrame] = []
    for sym in {s.strip().upper() for s in universe if s}:
        raw = _read_symbol(base_dir, sym)
        if raw is None:
            logger.debug(f"No curated file for {sym}")
            continue
        df = _normalize_cols(raw)
        df = df[(df["date"] >= start) & (df["date"] <= end)]
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["date", "symbol", "open", "high", "low", "close", "volume"])

    panel = pd.concat(frames, ignore_index=True).sort_values(["date", "symbol"])  # type: ignore
    return panel

