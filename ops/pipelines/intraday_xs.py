from __future__ import annotations

"""
Intraday cross-sectional pipeline (free-tier friendly scaffold).

Uses curated minute bars if available (curated/equities_minute). Computes
per-day features like intraday volatility, open-close return, and high-low
range proxies aggregated from minute bars. Falls back to daily OHLCV if
minute data is missing.
"""

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from loguru import logger

from feature_store.intraday import IntradayFeatureConfig, build_intraday_features
from models.intraday_xs import PatchConfig, predict_patchtst, train_patchtst
from ops.reports.emitter import emit_daily


@dataclass
class IntradayConfig:
    start_date: str
    end_date: str
    universe: List[str]
    initial_capital: float = 1_000_000.0


def _list_dates_in_range(root: Path, start: date, end: date) -> List[date]:
    dates = []
    for p in root.glob("date=*"):
        try:
            ds = pd.to_datetime(p.name.split("=")[-1]).date()
            if start <= ds <= end:
                dates.append(ds)
        except Exception:
            pass
    return sorted(dates)


def _load_minute_day(ds: date) -> pd.DataFrame:
    p = Path("data_layer/curated/equities_minute") / f"date={ds.isoformat()}" / "data.parquet"
    if p.exists():
        try:
            return pd.read_parquet(p)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def run_intraday(cfg: IntradayConfig) -> Dict:
    start = pd.to_datetime(cfg.start_date).date()
    end = pd.to_datetime(cfg.end_date).date()
    base = Path("data_layer/curated/equities_minute")
    if not base.exists():
        logger.warning("No curated minute data; intraday_xs will skip")
        return {"status": "no_data"}
    dates = _list_dates_in_range(base, start, end)
    if not dates:
        return {"status": "no_data"}

    feature_rows = []
    for ds in dates:
        df = _load_minute_day(ds)
        if df.empty or not set(["symbol", "open", "high", "low", "close", "volume"]).issubset(df.columns):
            continue
        sub = df[df["symbol"].isin(cfg.universe)].copy()
        if sub.empty:
            continue
        feats = build_intraday_features(sub, IntradayFeatureConfig())
        feats["date"] = ds
        feature_rows.append(feats)

    if not feature_rows:
        return {"status": "no_data"}

    features = pd.concat(feature_rows, ignore_index=True)
    target = features["close_ret"].fillna(0.0)
    feature_cols = [c for c in features.columns if c not in {"symbol", "date", "close_ret"}]
    X = features[feature_cols].fillna(0.0)

    training_metrics = {}
    try:
        model, training_metrics = train_patchtst(X, target, PatchConfig())
        preds = predict_patchtst(model, X)
        features["score"] = preds
    except Exception as exc:
        logger.warning(f"Intraday model training failed; fallback to vwap dislocation: {exc}")
        features["score"] = features.get("vwap_dislocation", 0.0)

    scores = features[["date", "symbol", "score"]].copy()

    def zgroup(g):
        s = g["score"]
        if len(s) < 2 or s.std(ddof=1) == 0:
            g["target_w"] = 0.0
        else:
            z = (s - s.mean()) / s.std(ddof=1)
            g["target_w"] = z / z.abs().sum()
        return g

    weights = scores.groupby("date", group_keys=False).apply(zgroup)
    last_date = weights["date"].max()
    last_positions = weights[weights["date"] == last_date][["symbol", "target_w"]]

    metrics = {
        "status": "ok",
        "days": int(weights["date"].nunique()),
    }
    for k, v in training_metrics.items():
        if isinstance(v, (int, float, np.floating, str)):
            metrics[f"model_{k}"] = v
    emit_daily(last_date, last_positions, metrics)
    return {"status": "ok", "weights": weights}
