from __future__ import annotations

"""
Generate simple options strategies from IV/surface artifacts.

Current simple policy: for each underlier with an IV slice for nearest expiry,
emit a delta-hedged straddle with a placeholder expected edge metric.
"""

from datetime import date
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from loguru import logger

from ops.reports.emitter import emit_options_daily


def _nearest_expiry(df: pd.DataFrame, asof: date) -> date | None:
    exps = sorted({pd.to_datetime(x).date() for x in df["expiry"].dropna().unique()})
    exps = [e for e in exps if e > asof]
    return exps[0] if exps else None


def build_and_emit(asof: date, underliers: List[str]) -> Dict[str, str] | None:
    base = Path("data_layer/curated/options_iv") / f"date={asof.isoformat()}"
    if not base.exists():
        logger.info("No IV artifacts found; skipping options strategies")
        return None
    strategies = []
    for ul in underliers:
        p = base / f"underlier={ul}" / "data.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        if df.empty:
            continue
        exp = _nearest_expiry(df, asof)
        if not exp:
            continue
        # Expected edge proxy: more contracts with valid IV => higher confidence
        slice_df = df[df["expiry"] == exp]
        contracts = int(len(slice_df))
        expected_bps = min(50, max(5, contracts // 5))  # placeholder
        strategies.append({
            "underlier": ul,
            "type": "delta_hedged_straddle",
            "expiry": exp,
            "qty": 1,
            "target_vega": None,
            "expected_pnl_bps": expected_bps,
            "conf": 0.5 + min(0.4, contracts / 200.0),
            "hedge_rules": {"rebalance": "daily", "delta_threshold": 0.2},
        })
    if not strategies:
        logger.info("No strategies generated from IV")
        return None
    metrics = {"count": len(strategies)}
    return emit_options_daily(asof, strategies, metrics)

