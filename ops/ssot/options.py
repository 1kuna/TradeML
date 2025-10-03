from __future__ import annotations

"""
Options IV builder and SVI fitter for curated outputs.

Writes:
- curated/options_iv/date=YYYY-MM-DD/underlier=SYM/data.parquet
- curated/options_surface/date=YYYY-MM-DD/underlier=SYM/data.parquet
"""

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from feature_store.options.iv import calculate_iv_from_price, BlackScholesIV
from feature_store.options.svi import fit_svi_slice


def _load_spot(asof: date, symbol: str) -> Optional[float]:
    path = Path("data_layer/curated/equities_ohlcv_adj") / f"date={asof.isoformat()}" / "data.parquet"
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
    except Exception:
        return None
    row = df[df["symbol"] == symbol]
    if row.empty:
        return None
    return float(row.iloc[0].get("close_adj", row.iloc[0].get("close_raw", np.nan)))


def _load_risk_free(asof: date) -> float:
    # Use 1Y tenor if available, else 3M; fallback 3%
    p = Path("data_layer/raw/macro_treasury/fred") / f"date={asof.isoformat()}" / "data.parquet"
    if not p.exists():
        return 0.03
    try:
        df = pd.read_parquet(p)
    except Exception:
        return 0.03
    if "tenor" in df.columns and "value" in df.columns:
        for t in ["1y", "3m"]:
            hit = df[df["tenor"] == t]
            if not hit.empty:
                # Convert percent to rate
                return float(hit.iloc[0]["value"]) / 100.0
    return 0.03


def build_iv(asof: date, underliers: List[str], min_contracts: int = 10) -> Dict:
    # Support both table-first and source-first local layouts
    base_primary = Path("data_layer/raw/finnhub/options_chains") / f"date={asof.isoformat()}"
    base_alt = Path("data_layer/raw/options_chains/finnhub") / f"date={asof.isoformat()}"
    base = base_primary if base_primary.exists() else base_alt
    if not base.exists():
        logger.warning("No raw options chains for as-of; skipping IV build")
        return {"status": "no_data"}

    total = 0
    for ul in underliers:
        spot = _load_spot(asof, ul)
        if spot is None or not np.isfinite(spot):
            logger.warning(f"Missing spot for {ul} at {asof}")
            continue
        r = _load_risk_free(asof)

        path = base / f"underlier={ul}" / "data.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if df.empty or "nbbo_mid" not in df.columns:
            continue
        rows = []
        for _, row in df.iterrows():
            exp = pd.to_datetime(row["expiry"]).date() if pd.notna(row.get("expiry")) else None
            if not exp:
                continue
            T = max((exp - asof).days, 0) / 365.0
            if T <= 0:
                continue
            price = row.get("nbbo_mid")
            if pd.isna(price):
                continue
            K = float(row.get("strike"))
            cp = str(row.get("cp_flag", "C")).upper()
            opt_type = "call" if cp == "C" else "put"
            iv = calculate_iv_from_price(price, S=spot, K=K, T=T, r=r, option_type=opt_type)
            if iv is None or not np.isfinite(iv) or iv <= 0:
                continue
            delta = BlackScholesIV.calculate_greeks(spot, K, T, r, iv, opt_type).delta
            rows.append({
                "date": asof,
                "underlier": ul,
                "expiry": exp,
                "strike": K,
                "cp_flag": cp,
                "iv": iv,
                "delta": delta,
                "gamma": None,
                "theta": None,
                "vega": None,
                "rho": None,
                "underlying_price": spot,
                "risk_free_rate": r,
                "dividend_yield": 0.0,
                "time_to_expiry": T,
                "nbbo_mid": price,
                "is_itm": (cp == 'C' and spot >= K) or (cp == 'P' and spot <= K),
                "iv_valid": True,
                "is_crossed": False,
                "ingested_at": pd.Timestamp.utcnow(),
                "source_name": "finnhub",
                "source_uri": f"finnhub://option-chain/{ul}",
                "transform_id": "options_iv_v1",
            })
        if not rows or len(rows) < min_contracts:
            continue
        iv_df = pd.DataFrame(rows)
        out = Path("data_layer/curated/options_iv") / f"date={asof.isoformat()}" / f"underlier={ul}"
        out.mkdir(parents=True, exist_ok=True)
        iv_df.to_parquet(out / "data.parquet", index=False)
        total += len(iv_df)
        logger.info(f"Wrote IVs for {ul}: {len(iv_df)} rows")
    return {"status": "ok", "count": total}


def fit_surfaces(asof: date, underliers: List[str], min_contracts: int = 20) -> Dict:
    in_base = Path("data_layer/curated/options_iv") / f"date={asof.isoformat()}"
    if not in_base.exists():
        return {"status": "no_data"}
    total = 0
    for ul in underliers:
        p = in_base / f"underlier={ul}" / "data.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        if df.empty:
            continue
        results = []
        spot = float(df.iloc[0]["underlying_price"]) if "underlying_price" in df.columns else None
        for exp, sl in df.groupby("expiry"):
            if len(sl) < min_contracts:
                continue
            strikes = sl["strike"].values
            ivs = sl["iv"].values
            T = max((pd.to_datetime(exp).date() - asof).days, 0) / 365.0
            if T <= 0 or spot is None:
                continue
            fit = fit_svi_slice(np.asarray(strikes), float(spot), np.asarray(ivs), T)
            params = fit.get("params")
            metrics = fit.get("metrics")
            results.append({
                "date": asof,
                "underlier": ul,
                "expiry": pd.to_datetime(exp).date(),
                "svi_a": getattr(params, "a", None) if params else None,
                "svi_b": getattr(params, "b", None) if params else None,
                "svi_rho": getattr(params, "rho", None) if params else None,
                "svi_m": getattr(params, "m", None) if params else None,
                "svi_sigma": getattr(params, "sigma", None) if params else None,
                "fit_rmse": getattr(metrics, "rmse", None) if metrics else None,
                "num_options_fitted": int(len(sl)),
                "butterfly_arb": getattr(metrics, "has_butterfly_arb", False) if metrics else False,
                "vertical_arb": False,
                "calendar_arb": getattr(metrics, "has_calendar_arb", False) if metrics else False,
                "atm_iv": float(np.nan) if spot is None else float(np.nan),
                "skew_25d": float(np.nan),
                "slope": float(np.nan),
                "ingested_at": pd.Timestamp.utcnow(),
                "source_name": "svi",
                "source_uri": f"svi://fit/{ul}/{exp}",
                "transform_id": "options_surface_v1",
            })
        if results:
            out = Path("data_layer/curated/options_surface") / f"date={asof.isoformat()}" / f"underlier={ul}"
            out.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(results).to_parquet(out / "data.parquet", index=False)
            total += len(results)
    return {"status": "ok", "count": total}
