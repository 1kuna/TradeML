"""
Portfolio construction (minimal Phase 2 baseline).

API (blueprint): portfolio.build(scores_df, risk_cfg) -> dict

Scores per date,symbol are converted to target weights with:
- Cross-sectional z-score per date
- Gross exposure cap
- Optional per-name cap
- Optional Kelly fraction scaling (default 1.0 means raw weights)

Output dict includes target_weights DataFrame and metadata.
"""

from __future__ import annotations

from typing import Dict, Optional
import numpy as np
import pandas as pd
from loguru import logger


def _per_date_zscore(df: pd.DataFrame, score_col: str = "score") -> pd.Series:
    def z(g: pd.Series) -> pd.Series:
        mu = g.mean()
        sd = g.std(ddof=1)
        if sd == 0 or np.isnan(sd):
            return pd.Series(0.0, index=g.index)
        return (g - mu) / sd

    return df.groupby("date")[score_col].transform(z)


def build(scores_df: pd.DataFrame, risk_cfg: Optional[Dict] = None) -> Dict:
    """Convert scores to target weights.

    Expects columns: date, symbol, score. Returns dict with DataFrame
    target_weights having columns: date, symbol, target_w, and metadata.
    """
    if risk_cfg is None:
        risk_cfg = {}

    gross_cap = float(risk_cfg.get("gross_cap", 1.0))  # total |w| per date
    max_name = float(risk_cfg.get("max_name", 0.05))   # per-name cap
    kelly_fraction = float(risk_cfg.get("kelly_fraction", 1.0))

    df = scores_df.copy()
    if not {"date", "symbol", "score"}.issubset(df.columns):
        raise ValueError("scores_df must have columns: date, symbol, score")

    df["z"] = _per_date_zscore(df, "score")

    # Convert z-scores to provisional weights by softmax-like normalization of absolute z
    def scale_group(g: pd.DataFrame) -> pd.DataFrame:
        if (g["z"].abs().sum() == 0) or np.isnan(g["z"].abs().sum()):
            g["target_w"] = 0.0
            return g
        w = g["z"] / g["z"].abs().sum() * gross_cap
        # Cap per-name and renormalize gross
        w = w.clip(-max_name, max_name)
        if w.abs().sum() > 0:
            w = w / w.abs().sum() * gross_cap
        g["target_w"] = w * kelly_fraction
        return g

    out = df.groupby("date", group_keys=False).apply(scale_group).reset_index(drop=True)
    result = {
        "target_weights": out[["date", "symbol", "target_w"]],
        "gross_cap": gross_cap,
        "kelly_fraction": kelly_fraction,
        "max_name": max_name,
    }
    logger.info(
        f"Portfolio built: {out['date'].nunique()} dates, gross_cap={gross_cap}, "
        f"kelly={kelly_fraction}, max_name={max_name}"
    )
    return result

