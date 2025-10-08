from __future__ import annotations

"""
Options Volatility Pipeline (scaffold)

Goal: Ingest options chains (NBBO mids), compute IVs, fit SVI slices per
expiry with QC, and emit daily strategies or diagnostics. This scaffold wires
existing iv and svi utilities and provides a minimal entry point for future
expansion.
"""

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
from loguru import logger

from feature_store.options.iv import calculate_iv_from_price, BlackScholesIV
from feature_store.options.svi import fit_svi_slice


@dataclass
class OptionsVolConfig:
    asof: str
    underliers: List[str]
    risk_free_rate: float = 0.03  # placeholder; wire FRED later
    dividend_yield: float = 0.0
    min_contracts: int = 50


def _load_latest_chains(asof: date) -> pd.DataFrame:
    # Local raw finnhub chains layout: data_layer/raw/options_chains/finnhub/date=YYYY-MM-DD/underlier=SYM/data.parquet
    base = Path("data_layer/raw/options_chains/finnhub") / f"date={asof.isoformat()}"
    if not base.exists():
        return pd.DataFrame()
    frames = []
    for p in base.glob("underlier=*/data.parquet"):
        try:
            frames.append(pd.read_parquet(p))
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return df


def _time_to_expiry(asof: date, expiry: date) -> float:
    days = (expiry - asof).days
    return max(days, 0) / 365.0


def run_options_vol(cfg: OptionsVolConfig) -> Dict:
    asof_d = pd.to_datetime(cfg.asof).date()
    # Consume curated IV written by ops/ssot/options.build_iv
    base = Path("data_layer/curated/options_iv") / f"date={asof_d.isoformat()}"
    if not base.exists():
        logger.warning("No curated IV artifacts for as-of; run IV builder first")
        return {"status": "no_data"}

    results = {}
    for ul in cfg.underliers:
        p = base / f"underlier={ul}" / "data.parquet"
        if not p.exists():
            continue
        iv_df = pd.read_parquet(p)
        if iv_df.empty:
            continue
        spot = float(iv_df.iloc[0].get("underlying_price", 100.0))
        svi_results = {}
        for exp, slice_df in iv_df.groupby("expiry"):
            if len(slice_df) < cfg.min_contracts:
                continue
            strikes = slice_df["strike"].values
            ivs = slice_df["iv"].values
            T = _time_to_expiry(asof_d, pd.to_datetime(exp).date())
            if T <= 0:
                continue
            fit = fit_svi_slice(strikes=np.asarray(strikes), spot=spot, ivs=np.asarray(ivs), T=T)
            svi_results[str(pd.to_datetime(exp).date())] = {
                "fit_successful": fit["fit_successful"],
                "rmse": getattr(fit.get("metrics"), "rmse", None) if fit.get("metrics") is not None else None,
            }
        if svi_results:
            results[ul] = {"svi": svi_results}

    status = "ok" if results else "no_data"
    return {"status": status, "asof": cfg.asof, "underliers": list(results.keys()), "results": results}
