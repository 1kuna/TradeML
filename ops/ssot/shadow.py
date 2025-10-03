from __future__ import annotations

"""
Shadow trading logger and evaluator (free-tier friendly).

Logs signals (weights) and later evaluates realized PnL from curated
equities_ohlcv_adj closes.
"""

import json
from datetime import date
from pathlib import Path
from typing import Dict

import pandas as pd
from loguru import logger


def log_signals(asof: date, weights: pd.DataFrame, out_dir: str = "ops/reports/shadow/equities_xs") -> str:
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"signals_{asof.isoformat()}.json"
    payload = {"asof": asof.isoformat(), "weights": weights.to_dict(orient="records")}
    path.write_text(json.dumps(payload, indent=2))
    logger.info(f"Shadow signals logged: {path}")
    return str(path)


def evaluate_shadow(start: date, end: date, in_dir: str = "ops/reports/shadow/equities_xs") -> Dict:
    files = sorted(Path(in_dir).glob("signals_*.json"))
    if not files:
        return {"status": "no_data"}
    # Load curated closes into panel
    base = Path("data_layer/curated/equities_ohlcv_adj")
    dates = [pd.to_datetime(p.name.split("=")[-1]).date() for p in base.glob("date=*")]
    panel = []
    for ds in dates:
        if not (start <= ds <= end):
            continue
        p = base / f"date={ds.isoformat()}" / "data.parquet"
        try:
            df = pd.read_parquet(p)
            panel.append(df[["symbol", "close_adj"]].assign(date=ds))
        except Exception:
            pass
    if not panel:
        return {"status": "no_prices"}
    px = pd.concat(panel, ignore_index=True)
    px_pivot = px.pivot(index="date", columns="symbol", values="close_adj").sort_index()

    # Evaluate day-over-day returns since signals
    pnl_rows = []
    for f in files:
        data = json.loads(f.read_text())
        asof = pd.to_datetime(data["asof"]).date()
        if not (start <= asof < end):
            continue
        w = pd.DataFrame(data["weights"])
        w = w.set_index("symbol")["target_w"].to_dict()
        # Next day return from asof to next trading day available
        dates_sorted = list(px_pivot.index)
        if asof not in dates_sorted:
            continue
        idx = dates_sorted.index(asof)
        if idx + 1 >= len(dates_sorted):
            continue
        dnext = dates_sorted[idx + 1]
        p0 = px_pivot.loc[asof]
        p1 = px_pivot.loc[dnext]
        rets = (p1 / p0 - 1.0).fillna(0.0)
        pnl = sum(rets.get(sym, 0.0) * w.get(sym, 0.0) for sym in w.keys())
        pnl_rows.append({"asof": asof, "next": dnext, "pnl": float(pnl)})

    if not pnl_rows:
        return {"status": "no_pnl"}
    dfp = pd.DataFrame(pnl_rows)
    return {"status": "ok", "pnl_mean": float(dfp["pnl"].mean()), "pnl_count": int(len(dfp))}

