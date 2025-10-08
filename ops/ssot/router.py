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
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from loguru import logger

from models.meta import load_stacker, stack_scores

_ROUTER_CFG_CACHE: Dict[str, object] = {"mtime": None, "config": {}}
_STACKER_CACHE: Dict[str, object] = {"path": None, "obj": None, "meta": {}}


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


def _route_signals(asof: date, symbol: str, ctx: RouterContext):
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


@lru_cache(maxsize=1)
def _endpoints_config(path: str = "configs/endpoints.yml") -> Dict:
    try:
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get("providers", {}) or {}
    except FileNotFoundError:
        logger.debug("endpoints.yml missing; dataset routing will fall back to defaults")
        return {}
    except Exception as exc:
        logger.warning(f"Failed to load endpoints config: {exc}")
        return {}


def _unique_score(value) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, str):
        flag = value.lower()
        if flag in {"true", "yes"}:
            return 1.0
        if flag == "partial":
            return 0.5
    return 0.0


def _router_config(path: str = "configs/router.yml") -> Dict:
    cfg_path = Path(path)
    try:
        mtime = cfg_path.stat().st_mtime
    except FileNotFoundError:
        if _ROUTER_CFG_CACHE["config"]:
            _ROUTER_CFG_CACHE.update({"mtime": None, "config": {}})
        return {}

    if _ROUTER_CFG_CACHE.get("mtime") != mtime:
        try:
            with open(cfg_path) as f:
                _ROUTER_CFG_CACHE["config"] = yaml.safe_load(f) or {}
        except Exception as exc:
            logger.warning(f"Failed to load router config: {exc}")
            _ROUTER_CFG_CACHE["config"] = {}
        _ROUTER_CFG_CACHE["mtime"] = mtime
    return _ROUTER_CFG_CACHE.get("config", {})


def _get_stacker(stack_cfg: Dict) -> Tuple[object | None, Dict]:
    if not stack_cfg.get("enabled"):
        weights = stack_cfg.get("linear_weights")
        if weights:
            return weights, {"feature_columns": list(weights.keys()), "type": "linear_weights"}
        return None, {}

    metadata_path = stack_cfg.get("metadata_path")
    if metadata_path:
        if _STACKER_CACHE.get("path") != metadata_path:
            try:
                obj, meta = load_stacker(metadata_path)
                _STACKER_CACHE.update({"path": metadata_path, "obj": obj, "meta": meta})
            except Exception as exc:
                logger.warning(f"Failed to load stacker metadata {metadata_path}: {exc}")
                _STACKER_CACHE.update({"path": metadata_path, "obj": None, "meta": {}})
        return _STACKER_CACHE.get("obj"), _STACKER_CACHE.get("meta", {})

    weights = stack_cfg.get("linear_weights")
    if weights:
        return weights, {"feature_columns": list(weights.keys()), "type": "linear_weights"}
    return None, {}


def route_dataset(dataset: str, want_date: date, universe: List[str] | None = None, endpoints_path: str = "configs/endpoints.yml") -> List[str]:
    """Return providers ordered by preference for a dataset.

    Args:
        dataset: Dataset name (e.g., 'equities_eod')
        want_date: Target date for the data pull
        universe: Optional list of symbols to size demand
        endpoints_path: Override config path when testing

    Returns:
        Ordered list of provider keys
    """

    cfg = _endpoints_config(endpoints_path)
    details = []
    universe = universe or []

    for provider, pdata in cfg.items():
        datasets = (pdata or {}).get("datasets", {}) or {}
        if dataset not in datasets:
            continue
        entry = datasets[dataset] or {}
        uniq = _unique_score(entry.get("unique"))
        weight = float(entry.get("weight", 1))
        daily_cap = pdata.get("daily_cap") or pdata.get("rpm")
        budget_headroom = 1.0
        if daily_cap:
            try:
                cap_val = float(daily_cap)
                demand = max(1.0, len(universe) or 1.0)
                budget_headroom = 1.0 if cap_val >= demand else 0.5
            except Exception:
                budget_headroom = 1.0
        recency_need = 1.0 if abs((date.today() - want_date).days) <= 2 else 0.2
        coverage_gap = 1.0  # placeholder until completeness ledger integrated
        details.append({
            "provider": provider,
            "uniq": uniq,
            "weight": weight,
            "budget_headroom": budget_headroom,
            "recency_need": recency_need,
            "coverage_gap": coverage_gap,
        })

    if not details:
        return []

    multi_provider = len(details) > 1
    entries = []
    for det in details:
        duplication_risk = 1.0 if det["uniq"] < 0.5 and multi_provider else 0.0
        score = (
            (3 * det["uniq"]) +
            (2 * det["coverage_gap"]) +
            det["budget_headroom"] +
            det["recency_need"] -
            (2 * duplication_risk) +
            det["weight"]
        )
        entries.append((det["provider"], score, det["weight"]))

    entries.sort(key=lambda tpl: (-tpl[1], -tpl[2], tpl[0]))
    return [prov for prov, _, _ in entries]


def route(*args, **kwargs):  # type: ignore[override]
    """Route either model signals or dataset providers based on signature."""
    if not args:
        raise TypeError("route() missing required arguments")
    if isinstance(args[0], date):
        if len(args) < 3:
            raise TypeError("route(date, symbol, ctx) expects three positional arguments")
        return _route_signals(args[0], args[1], args[2])
    if isinstance(args[0], str):
        dataset = args[0]
        want_date = kwargs.pop("want_date", None)
        universe = kwargs.pop("universe", None)
        if want_date is None:
            want_date = args[1] if len(args) > 1 and isinstance(args[1], date) else date.today()
        if universe is None:
            universe = args[2] if len(args) > 2 and isinstance(args[2], list) else []
        return route_dataset(dataset, want_date, universe)
    raise TypeError("Unsupported signature for route()")


def meta_blend(model_scores: Dict[str, float], weights: Dict[str, float] | None = None) -> float:
    """Blend scores from multiple models with simple weights and fallbacks.

    Defaults: equities_xs=0.6, intraday_xs=0.3, options_vol=0.1
    """
    if not model_scores:
        return 0.0
    cfg = _router_config()
    router_section = (cfg.get("router") or {})
    stack_cfg = router_section.get("stacker") or {}

    stacker_obj, metadata = _get_stacker(stack_cfg)
    if stacker_obj is not None:
        try:
            df_scores = pd.DataFrame([model_scores])
            blended = stack_scores(df_scores, stacker_obj, metadata)
            return float(blended[0])
        except Exception as exc:
            logger.warning(f"Stacker blend failed, reverting to static weights: {exc}")

    if weights is None:
        weights = router_section.get("weights", {})
    if not weights:
        weights = stack_cfg.get("linear_weights", {}) or {"equities_xs": 0.6, "intraday_xs": 0.3, "options_vol": 0.1}

    tot = 0.0
    denom = 0.0
    for k, v in model_scores.items():
        wk = float(weights.get(k, 0.0))
        tot += wk * float(v)
        denom += wk
    return tot / denom if denom > 0 else float(np.mean(list(model_scores.values())))
