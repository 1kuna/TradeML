from __future__ import annotations

"""
Intraday Cross-Sectional Pipeline.

SSOT v2 Section 4.1: Specialist models - intraday_xs

Goal: Predict intraday price movements using microstructure features.
Architecture: PatchTST (or fallback GBM when torch unavailable)

Pipeline steps:
1. Load curated minute bars
2. Compute intraday features (VWAP dislocation, OFI, microstructure noise, etc.)
3. Generate intraday return labels
4. Train model with CPCV validation
5. Generate position weights and emit daily report

Minimal API:
    run_intraday(cfg) -> Dict with status, weights, metrics
"""

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

import numpy as np
import pandas as pd
from loguru import logger

from feature_store.intraday import IntradayFeatureConfig, build_intraday_features
from models.intraday_xs import PatchConfig, predict_patchtst, train_patchtst
from labeling.horizon.horizon import horizon_returns
from validation.cpcv import run_cpcv
from ops.reports.emitter import emit_daily


@dataclass
class IntradayConfig:
    """Configuration for intraday cross-sectional pipeline."""

    start_date: str
    end_date: str
    universe: List[str]
    initial_capital: float = 1_000_000.0
    # Training settings
    lookback_days: int = 60  # Shorter lookback for intraday
    horizon_days: int = 1  # Intraday horizon (next-day return)
    min_train_samples: int = 50
    # CPCV settings
    n_splits: int = 5
    n_test_splits: int = 2
    embargo_days: int = 2
    purge_days: int = 1
    # Model settings
    patch_config: Optional[PatchConfig] = None


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


def _build_feature_label_panel(
    cfg: IntradayConfig,
    dates: List[date],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build feature and label panels from minute data.

    Returns:
        (features_df, labels_df)
    """
    feature_rows = []

    for ds in dates:
        df = _load_minute_day(ds)
        if df.empty:
            continue

        required_cols = {"symbol", "open", "high", "low", "close", "volume"}
        if not required_cols.issubset(df.columns):
            continue

        # Filter to universe
        sub = df[df["symbol"].isin(cfg.universe)].copy()
        if sub.empty:
            continue

        # Build intraday features (now with feature_ prefix)
        feats = build_intraday_features(sub, IntradayFeatureConfig())
        if not feats.empty:
            feature_rows.append(feats)

    if not feature_rows:
        return pd.DataFrame(), pd.DataFrame()

    features_df = pd.concat(feature_rows, ignore_index=True)

    # Generate horizon return labels
    all_labels = []
    for ds in features_df["date"].unique():
        try:
            labels = horizon_returns(ds, cfg.universe, horizon_days=cfg.horizon_days)
            if not labels.empty:
                all_labels.append(labels)
        except Exception as e:
            logger.debug(f"Label generation failed for {ds}: {e}")

    if not all_labels:
        return features_df, pd.DataFrame()

    labels_df = pd.concat(all_labels, ignore_index=True)

    return features_df, labels_df


def _merge_features_labels(
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge features and labels on (symbol, date)."""
    if features_df.empty or labels_df.empty:
        return pd.DataFrame()

    # Normalize column names
    labels_df = labels_df.copy()
    if "entry_date" in labels_df.columns:
        labels_df = labels_df.rename(columns={"entry_date": "date"})

    # Merge
    merged = pd.merge(
        features_df,
        labels_df[["symbol", "date", "forward_return"]],
        on=["symbol", "date"],
        how="inner",
    )

    return merged


def run_intraday(cfg: IntradayConfig) -> Dict:
    """
    Run the intraday cross-sectional pipeline.

    Steps:
    1. Load curated minute bars for date range
    2. Compute intraday features (microstructure, LOB, seasonality)
    3. Generate forward return labels
    4. Train PatchTST model with CPCV validation
    5. Generate position weights and emit daily report

    Args:
        cfg: Pipeline configuration

    Returns:
        Dict with status, weights, metrics, artifact paths
    """
    start = pd.to_datetime(cfg.start_date).date()
    end = pd.to_datetime(cfg.end_date).date()

    logger.info(f"Running intraday_xs pipeline from {start} to {end}")

    base = Path("data_layer/curated/equities_minute")
    if not base.exists():
        logger.warning("No curated minute data; intraday_xs will skip")
        return {"status": "no_data"}

    dates = _list_dates_in_range(base, start, end)
    if not dates:
        return {"status": "no_data", "reason": "no_dates_in_range"}

    # Build feature and label panels
    features_df, labels_df = _build_feature_label_panel(cfg, dates)

    if features_df.empty:
        logger.warning("No features generated from minute data")
        return {"status": "no_data", "reason": "no_features"}

    # Merge features and labels
    train_df = _merge_features_labels(features_df, labels_df)

    if len(train_df) < cfg.min_train_samples:
        logger.warning(f"Insufficient training samples: {len(train_df)} < {cfg.min_train_samples}")
        # Fallback: use feature_close_ret as target
        train_df = features_df.copy()
        train_df["forward_return"] = train_df.get("feature_close_ret", 0.0)

    logger.info(f"Training panel: {len(train_df)} samples")

    # Prepare feature matrix
    feature_cols = [c for c in train_df.columns if c.startswith("feature_")]
    X = train_df[feature_cols].fillna(0.0)
    y = train_df["forward_return"].fillna(0.0)

    # CPCV validation
    cpcv_results = None
    try:
        cpcv_results = run_cpcv(
            X=X,
            y=y,
            dates=pd.Series(train_df["date"]),
            n_splits=cfg.n_splits,
            n_test_splits=cfg.n_test_splits,
            embargo_days=cfg.embargo_days,
            purge_days=cfg.purge_days,
        )
        logger.info(f"CPCV validation: mean_score={cpcv_results.get('mean_score', 'N/A'):.4f}")
    except Exception as e:
        logger.warning(f"CPCV validation failed: {e}")

    # Train model
    training_metrics = {}
    model = None
    patch_cfg = cfg.patch_config or PatchConfig()

    try:
        model, training_metrics = train_patchtst(X, y, patch_cfg)
        preds = predict_patchtst(model, X)
        train_df = train_df.copy()
        train_df["score"] = preds
        logger.info(f"PatchTST trained: rmse={training_metrics.get('rmse', 'N/A'):.6f}")
    except Exception as exc:
        logger.warning(f"Intraday model training failed; fallback to vwap dislocation: {exc}")
        train_df = train_df.copy()
        train_df["score"] = train_df.get("feature_vwap_dislocation", 0.0)
        training_metrics["backend"] = "fallback"
        training_metrics["error"] = str(exc)

    # Generate position weights (cross-sectional z-score)
    scores = train_df[["date", "symbol", "score"]].copy()

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

    # Save artifacts
    artifact_dir = Path("models/intraday_xs/artifacts")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "start_date": cfg.start_date,
        "end_date": cfg.end_date,
        "train_samples": int(len(train_df)),
        "features": feature_cols,
        "training_metrics": training_metrics,
        "cpcv_results": cpcv_results,
        "timestamp": pd.Timestamp.utcnow().isoformat(),
    }
    (artifact_dir / "intraday_xs_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    # Emit daily report
    metrics = {
        "status": "ok",
        "days": int(weights["date"].nunique()),
        "train_samples": len(train_df),
    }
    for k, v in training_metrics.items():
        if isinstance(v, (int, float, np.floating, str)):
            metrics[f"model_{k}"] = v

    emit_daily(last_date, last_positions, metrics)

    return {
        "status": "ok",
        "weights": weights,
        "train_samples": len(train_df),
        "features": feature_cols,
        "training_metrics": training_metrics,
        "cpcv_results": cpcv_results,
    }
