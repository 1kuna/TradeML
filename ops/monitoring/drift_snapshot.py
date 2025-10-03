from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import List

import pandas as pd
from loguru import logger

from .drift import DriftDetector
from feature_store.equities.features import compute_equity_features


def drift_snapshot(asof: date, universe: List[str], baseline_path: str = "ops/reports/baseline_features.parquet") -> str:
    asof_d = pd.to_datetime(asof).date()
    # Compute current features
    X = compute_equity_features(asof_d, universe)
    if X.empty:
        logger.warning("No features for drift snapshot; skipping")
        return ""
    # Drop non-numeric columns for drift calc
    keep = [c for c in X.columns if c.startswith("feature_")]
    if not keep:
        return ""
    Xc = X[keep]

    base_p = Path(baseline_path)
    out_dir = Path("ops/reports")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not base_p.exists():
        X.to_parquet(base_p, index=False)
        logger.info(f"Baseline features saved: {baseline_path}")
        return ""

    try:
        Xb = pd.read_parquet(base_p)
    except Exception:
        logger.warning("Failed to read baseline; resetting baseline")
        X.to_parquet(base_p, index=False)
        return ""

    # Align columns
    keep_b = [c for c in Xb.columns if c.startswith("feature_")]
    Xb = Xb[keep_b].dropna()
    Xc = Xc[keep_b].dropna()

    dd = DriftDetector()
    dd.set_baseline(Xb)
    results = dd.detect(Xc, metric="PSI")
    df = dd.summary(results)
    out_path = out_dir / f"drift_{asof_d.isoformat()}.json"
    df.to_json(out_path, orient="records")
    logger.info(f"Drift snapshot written: {out_path}")
    return str(out_path)

