from __future__ import annotations

"""
Evaluate delta-hedged PnL for previously emitted options strategies using free curated data.

Looks for ops/reports/options_YYYY-MM-DD.json and estimates one-day PnL per underlier
for delta-hedged straddles.
"""

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Dict

import pandas as pd
from loguru import logger

from backtest.engine.options_pnl import estimate_delta_hedged_straddle_pnl


def evaluate_options_pnl(asof: date, reports_dir: str = "ops/reports") -> Dict:
    rep = Path(reports_dir) / f"options_{asof.isoformat()}.json"
    if not rep.exists():
        logger.info(f"No options report found for {asof}")
        return {"status": "no_data"}
    data = json.loads(rep.read_text())
    strategies = data.get("strategies", [])
    if not strategies:
        return {"status": "no_strategies"}
    rows = []
    for s in strategies:
        ul = s.get("underlier")
        if not ul:
            continue
        res = estimate_delta_hedged_straddle_pnl(asof, ul)
        if res is None:
            continue
        rows.append({
            "underlier": res.underlier,
            "asof": res.asof,
            "next": res.next_date,
            "pnl": res.pnl_delta_hedged,
            "atm_iv_t": res.atm_iv_t,
            "atm_iv_t1": res.atm_iv_t1,
            "spot_t": res.spot_t,
            "spot_t1": res.spot_t1,
        })
    if not rows:
        return {"status": "no_pnl"}
    df = pd.DataFrame(rows)
    out = {"status": "ok", "count": int(len(df)), "pnl_mean": float(df["pnl"].mean())}
    # Persist next to original report
    out_path = Path(reports_dir) / f"options_eval_{asof.isoformat()}.json"
    out_path.write_text(json.dumps(out, indent=2))
    logger.info(f"Options PnL evaluated for {len(df)} strategies → {out_path}")
    return out

