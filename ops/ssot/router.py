from __future__ import annotations

"""
Router/meta-blender per SSOT. Gating on completeness and regime flags.

Exposes:
- route(asof, symbol, ctx): returns available signals by sleeve given completeness
- meta_blend(model_scores, weights=None): blends scores with weights from config

Stacking (CPCV-consistent) can be enabled via configs/router.yml when multiple
signals are present and an OOS dataset is available (not implemented here).
"""

from datetime import date
from typing import Dict

import numpy as np
import pandas as pd
from loguru import logger


class RouterContext:
    def __init__(self, ledger: pd.DataFrame):
        self.ledger = ledger

    def completeness(self, symbol: str, asof: date) -> Dict[str, bool]:
        # Simple windowed check: is today GREEN for equities_eod for symbol
        rows = self.ledger[(self.ledger.symbol == symbol) & (self.ledger.dt == asof)]
        out = {
            "equities_eod": any((rows.table_name == "equities_eod") & (rows.status == "GREEN")),
            "equities_minute": any((rows.table_name == "equities_minute") & (rows.status == "GREEN")),
            "options_chains": any((rows.table_name == "options_chains") & (rows.status == "GREEN")),
        }
        return out

    def regime_flags(self, asof: date) -> Dict[str, bool]:
        # Placeholder regime flags; extend with real masks
        return {"modern": asof.year >= 2012}


def route(asof: date, symbol: str, ctx: RouterContext):
    comp = ctx.completeness(symbol, asof)
    regime = ctx.regime_flags(asof)

    signals = {}
    if comp.get("equities_eod", False):
        signals["equities_xs"] = f"score_{symbol}_{asof.isoformat()}"
    if comp.get("equities_minute", False):
        signals["intraday_xs"] = f"intraday_{symbol}_{asof.isoformat()}"
    if comp.get("options_chains", False):
        signals["options_vol"] = f"ov_{symbol}_{asof.isoformat()}"

    logger.debug(f"Routing {symbol} on {asof}: available={list(signals)} regime={regime}")
    return signals


def meta_blend(model_scores: Dict[str, float], weights: Dict[str, float] | None = None) -> float:
    """Blend scores from multiple models with simple weights and fallbacks.

    Defaults: equities_xs=0.6, intraday_xs=0.3, options_vol=0.1
    """
    if not model_scores:
        return 0.0
    # Load router weights from config if not provided
    stack_cfg = {}
    if weights is None:
        try:
            with open("configs/router.yml") as f:
                cfg = yaml.safe_load(f) or {}
            rcfg = (cfg.get("router") or {})
            weights = rcfg.get("weights", {})
            stack_cfg = rcfg.get("stacker", {}) or {}
        except Exception:
            weights = {}
    # If a linear stacker is configured with explicit per-model weights, prefer those
    lw = (stack_cfg.get("linear_weights") or {}) if isinstance(stack_cfg, dict) else {}
    if stack_cfg.get("enabled") and lw:
        w = lw
    else:
        w = weights or {"equities_xs": 0.6, "intraday_xs": 0.3, "options_vol": 0.1}
    tot = 0.0
    denom = 0.0
    for k, v in model_scores.items():
        wk = float(w.get(k, 0.0))
        tot += wk * float(v)
        denom += wk
    return tot / denom if denom > 0 else float(np.mean(list(model_scores.values())))
import yaml
