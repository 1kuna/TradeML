from __future__ import annotations

"""
Options Volatility Pipeline.

SSOT v2 Section 4.1: Specialist models - options_vol

Goal: Predict delta-hedged PnL (volatility edge) using IV surface features.
Pipeline steps:
1. Load curated options IV data
2. Fit SVI surfaces with QC
3. Compute surface features (ATM IV, skew, curvature, term structure)
4. Generate delta-hedged PnL labels
5. Train model with CPCV validation
6. Evaluate and persist artifacts

Minimal API:
    run_options_vol(cfg) -> Dict with status, metrics, artifact paths
"""

from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import json
import numpy as np
import pandas as pd
from loguru import logger

from feature_store.options.iv import calculate_iv_from_price, BlackScholesIV
from feature_store.options.svi import fit_svi_slice, SVIParams
from feature_store.options.surface_features import compute_options_features
from labeling.options.delta_hedged_pnl import compute_delta_hedged_pnl, DeltaHedgedConfig
from models.options_vol import OptionsModelConfig, save_model, train_options_model
from validation.cpcv import run_cpcv


@dataclass
class OptionsVolConfig:
    """Configuration for options volatility pipeline."""

    asof: str
    underliers: List[str]
    risk_free_rate: float = 0.03
    dividend_yield: float = 0.0
    min_contracts: int = 50
    # Training settings
    lookback_days: int = 252  # Feature/label lookback
    horizon_days: int = 5  # Label horizon
    min_train_samples: int = 100  # Minimum samples for training
    # CPCV settings
    n_splits: int = 5
    n_test_splits: int = 2
    embargo_days: int = 5
    purge_days: int = 5


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


def _build_training_panel(
    cfg: OptionsVolConfig,
    asof_d: date,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Build feature and label panels for training.

    Returns:
        (features_df, labels_df, svi_diagnostics)
    """
    # Generate training dates (lookback from asof)
    train_dates = pd.date_range(
        end=asof_d - timedelta(days=cfg.horizon_days),  # Leave room for labels
        periods=cfg.lookback_days,
        freq="B",  # Business days
    ).date.tolist()

    logger.info(f"Building training panel for {len(train_dates)} dates, {len(cfg.underliers)} underliers")

    # Compute features for each date
    all_features = []
    all_labels = []
    svi_diagnostics = {}

    for train_date in train_dates:
        # Compute surface features
        features = compute_options_features(train_date, cfg.underliers)
        if not features.empty:
            all_features.append(features)

        # Compute delta-hedged PnL labels
        label_cfg = DeltaHedgedConfig(
            horizon_days=cfg.horizon_days,
            risk_free_rate=cfg.risk_free_rate,
            dividend_yield=cfg.dividend_yield,
        )
        labels = compute_delta_hedged_pnl(train_date, cfg.underliers, label_cfg)
        if not labels.empty:
            all_labels.append(labels)

    if not all_features or not all_labels:
        return pd.DataFrame(), pd.DataFrame(), svi_diagnostics

    features_df = pd.concat(all_features, ignore_index=True)
    labels_df = pd.concat(all_labels, ignore_index=True)

    # Also run SVI fitting for diagnostics on asof date
    base = Path("data_layer/curated/options_iv") / f"date={asof_d.isoformat()}"
    if base.exists():
        for ul in cfg.underliers:
            p = base / f"underlier={ul}" / "data.parquet"
            if not p.exists():
                continue
            try:
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
                    rmse = getattr(fit.get("metrics"), "rmse", None) if fit.get("metrics") is not None else None
                    svi_results[str(pd.to_datetime(exp).date())] = {
                        "fit_successful": fit["fit_successful"],
                        "rmse": rmse,
                    }
                if svi_results:
                    svi_diagnostics[ul] = svi_results
            except Exception as e:
                logger.debug(f"SVI fitting failed for {ul}: {e}")

    return features_df, labels_df, svi_diagnostics


def _merge_features_labels(features_df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    """Merge features and labels on (underlier, date)."""
    if features_df.empty or labels_df.empty:
        return pd.DataFrame()

    # Normalize column names
    features_df = features_df.copy()
    labels_df = labels_df.copy()

    # Ensure consistent key columns
    if "underlier" not in features_df.columns and "symbol" in features_df.columns:
        features_df = features_df.rename(columns={"symbol": "underlier"})
    if "asof" in features_df.columns:
        features_df = features_df.rename(columns={"asof": "date"})
    if "entry_date" in labels_df.columns:
        labels_df = labels_df.rename(columns={"entry_date": "date"})

    # Merge
    merged = pd.merge(
        features_df,
        labels_df[["underlier", "date", "delta_hedged_pnl", "entry_iv", "exit_iv"]],
        on=["underlier", "date"],
        how="inner",
    )

    return merged


def run_options_vol(cfg: OptionsVolConfig) -> Dict:
    """
    Run the options volatility pipeline.

    Steps:
    1. Build training panel (features + labels) over lookback period
    2. Merge features with delta-hedged PnL labels
    3. Train model with CPCV validation
    4. Evaluate and save artifacts

    Args:
        cfg: Pipeline configuration

    Returns:
        Dict with status, metrics, model path, diagnostics
    """
    asof_d = pd.to_datetime(cfg.asof).date()
    logger.info(f"Running options_vol pipeline as of {asof_d}")

    # Build training panel
    features_df, labels_df, svi_diagnostics = _build_training_panel(cfg, asof_d)

    if features_df.empty or labels_df.empty:
        logger.warning("No features or labels available for training")
        return {
            "status": "no_data",
            "asof": cfg.asof,
            "underliers": cfg.underliers,
            "svi_diagnostics": svi_diagnostics,
        }

    # Merge features and labels
    train_df = _merge_features_labels(features_df, labels_df)
    logger.info(f"Training panel: {len(train_df)} samples")

    if len(train_df) < cfg.min_train_samples:
        logger.warning(f"Insufficient training samples: {len(train_df)} < {cfg.min_train_samples}")
        return {
            "status": "insufficient_data",
            "asof": cfg.asof,
            "samples": len(train_df),
            "min_required": cfg.min_train_samples,
            "svi_diagnostics": svi_diagnostics,
        }

    # Prepare feature matrix
    feature_cols = [c for c in train_df.columns if c.startswith("feature_")]
    X = train_df[feature_cols].copy()
    y = train_df["delta_hedged_pnl"].copy()

    # Handle NaN features
    X = X.fillna(0)

    # CPCV validation
    cpcv_results = None
    try:
        cpcv_results = run_cpcv(
            X=X,
            y=y,
            dates=train_df["date"],
            n_splits=cfg.n_splits,
            n_test_splits=cfg.n_test_splits,
            embargo_days=cfg.embargo_days,
            purge_days=cfg.purge_days,
        )
        logger.info(f"CPCV validation: mean_score={cpcv_results.get('mean_score', 'N/A'):.4f}")
    except Exception as e:
        logger.warning(f"CPCV validation failed: {e}")

    # Train final model on all data
    artifact = None
    try:
        model, train_metrics = train_options_model(X, y, OptionsModelConfig())

        # Save artifacts
        artifact_dir = Path("models/options_vol/artifacts")
        artifact_dir.mkdir(parents=True, exist_ok=True)
        model_path = artifact_dir / f"options_vol_{asof_d.isoformat()}.pkl"
        save_model(model, model_path)

        # Also save as latest
        latest_path = artifact_dir / "options_vol_latest.pkl"
        save_model(model, latest_path)

        # Summary
        summary = {
            "asof": cfg.asof,
            "train_samples": int(len(X)),
            "features": feature_cols,
            "train_metrics": train_metrics,
            "cpcv_results": cpcv_results,
            "model_path": str(model_path),
            "timestamp": pd.Timestamp.utcnow().isoformat(),
        }
        (artifact_dir / "options_vol_summary.json").write_text(json.dumps(summary, indent=2, default=str))

        artifact = {
            "model_path": str(model_path),
            "train_metrics": train_metrics,
            "cpcv_results": cpcv_results,
        }

        logger.info(f"Options vol model saved to {model_path}")

    except Exception as e:
        logger.error(f"Options model training failed: {e}")
        return {
            "status": "training_failed",
            "error": str(e),
            "asof": cfg.asof,
            "svi_diagnostics": svi_diagnostics,
        }

    return {
        "status": "ok",
        "asof": cfg.asof,
        "underliers": cfg.underliers,
        "train_samples": len(train_df),
        "features": feature_cols,
        "artifact": artifact,
        "svi_diagnostics": svi_diagnostics,
    }
