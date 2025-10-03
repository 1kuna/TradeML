import os
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


def _write(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def test_build_iv_and_surface(tmp_path):
    os.environ["STORAGE_BACKEND"] = "local"
    asof = date.today()

    # Curated equities for spot
    eq = pd.DataFrame([
        {"date": asof, "symbol": "AAPL", "close_adj": 100.0, "close_raw": 100.0}
    ])
    _write(eq, Path("data_layer/curated/equities_ohlcv_adj") / f"date={asof.isoformat()}" / "data.parquet")

    # Macro treasury 1y = 5%
    rf = pd.DataFrame([
        {"date": asof, "tenor": "1y", "value": 5.0}
    ])
    _write(rf, Path("data_layer/raw/macro_treasury/fred") / f"date={asof.isoformat()}" / "data.parquet")

    # Raw chains with nbbo_mid
    exp = asof + timedelta(days=30)
    strikes = np.linspace(80, 120, 25)
    rows = []
    for K in strikes:
        rows.append({
            "underlier": "AAPL",
            "expiry": exp,
            "strike": float(K),
            "cp_flag": "C",
            "bid": None,
            "ask": None,
            "nbbo_mid": max(0.1, 5.0 - abs(100 - float(K)) * 0.1),
            "ts_ns": 0,
        })
    chain = pd.DataFrame(rows)
    _write(chain, Path("data_layer/raw/options_chains/finnhub") / f"date={asof.isoformat()}" / "underlier=AAPL" / "data.parquet")

    from ops.ssot.options import build_iv, fit_surfaces
    out_iv = build_iv(asof, ["AAPL"], min_contracts=10)
    assert out_iv["status"] in ("ok", "no_data")

    out_surf = fit_surfaces(asof, ["AAPL"], min_contracts=10)
    assert out_surf["status"] in ("ok", "no_data")

