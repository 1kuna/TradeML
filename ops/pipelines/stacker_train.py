"""Meta stacker training pipeline (SSOT v2 compliant).

Per SSOT v2 §4.5, the stacker must see only OOS fold predictions (CPCV-consistent).
This means when training the stacker on fold K, it must only see OOF predictions
from folds != K to prevent information leakage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd
from loguru import logger

from feature_store.equities.dataset import build_training_dataset
from models.meta import stack_scores, train_stacker


def _load_oof(path: str, require_fold: bool = False) -> pd.DataFrame:
    """Load OOF predictions from file.

    Parameters
    ----------
    path : str
        Path to OOF file (parquet or csv)
    require_fold : bool
        If True, raise error if 'fold' column is missing (needed for CPCV stacking)

    Returns
    -------
    DataFrame with columns: date, symbol, score, and optionally fold
    """
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(f"OOF source not found: {path}")
    if file.suffix == ".csv":
        df = pd.read_csv(file)
    else:
        df = pd.read_parquet(file)

    expected = {"date", "symbol", "score"}
    if not expected.issubset(df.columns):
        raise ValueError(f"OOF file {path} missing required columns {expected}")

    if require_fold and "fold" not in df.columns:
        raise ValueError(
            f"OOF file {path} missing 'fold' column required for CPCV-consistent stacking. "
            "Re-run sleeve training to generate fold IDs."
        )

    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def _update_router_config(path: str, metadata_path: str, sleeves: List[str]) -> None:
    import yaml

    cfg_path = Path(path)
    if not cfg_path.exists():
        logger.warning("router.yml not found; skipping stacker auto-wire")
        return
    with open(cfg_path) as f:
        data = yaml.safe_load(f) or {}
    router = data.setdefault("router", {})
    stacker_cfg = router.setdefault("stacker", {})
    stacker_cfg["enabled"] = True
    stacker_cfg["metadata_path"] = metadata_path
    stacker_cfg["sleeves"] = sleeves
    with open(cfg_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    logger.info(f"Updated router config with stacker metadata: {metadata_path}")


@dataclass
class StackerTrainConfig:
    start_date: str
    end_date: str
    universe: List[str]
    oof_sources: Dict[str, str]
    label_type: Literal["horizon", "triple_barrier"] = "horizon"
    horizon_days: int = 5
    artifact_dir: str = "models/meta/artifacts"
    algorithm: str = "ridge"
    update_router: bool = True
    router_cfg_path: str = "configs/router.yml"
    drop_columns: Optional[List[str]] = None
    linear_weights: Optional[Dict[str, float]] = None
    # CPCV consistency per SSOT v2 §4.5
    cpcv_consistent: bool = True  # enforce fold-aware stacker training
    fallback_to_simple: bool = True  # if fold info missing, fall back to simple merge


def _run_cpcv_stacker_training(
    stack_df: pd.DataFrame,
    sleeves: List[str],
    train_cfg: Dict,
) -> tuple[Dict, pd.DataFrame]:
    """Train stacker with CPCV consistency (leave-one-fold-out).

    Per SSOT v2 §4.5, stacker for fold K only sees OOF from folds != K.

    Parameters
    ----------
    stack_df : DataFrame
        Must have columns: date, symbol, target, fold, and sleeve score columns
    sleeves : List[str]
        Names of sleeve score columns
    train_cfg : Dict
        Training configuration for train_stacker()

    Returns
    -------
    tuple[Dict, DataFrame] : (final_result, stacker_oof_predictions)
    """
    if "fold" not in stack_df.columns:
        raise ValueError("stack_df must have 'fold' column for CPCV-consistent training")

    folds = sorted(stack_df["fold"].unique())
    n_folds = len(folds)
    logger.info(f"Running CPCV-consistent stacker training with {n_folds} folds")

    stacker_oof_rows = []
    fold_results = []

    for fold_id in folds:
        # Train on all folds except fold_id
        train_mask = stack_df["fold"] != fold_id
        test_mask = stack_df["fold"] == fold_id

        train_data = stack_df[train_mask]
        test_data = stack_df[test_mask]

        if len(train_data) < 10 or len(test_data) < 1:
            logger.warning(f"Fold {fold_id}: insufficient data (train={len(train_data)}, test={len(test_data)})")
            continue

        # Train stacker on folds != fold_id
        fold_train_cfg = {**train_cfg, "artifact_dir": f"{train_cfg['artifact_dir']}/fold_{fold_id}"}
        try:
            result = train_stacker(
                train_data[["date", "symbol", *sleeves]],
                train_data["target"],
                fold_train_cfg,
            )
            fold_results.append({"fold": fold_id, "result": result})

            # Generate OOF predictions for fold_id using this stacker
            if result.get("metadata_path"):
                scores = stack_scores(
                    test_data[["date", "symbol", *sleeves]],
                    result["metadata_path"],
                )
                oof_chunk = test_data[["date", "symbol", "fold"]].copy()
                oof_chunk["stacked_score"] = scores
                stacker_oof_rows.append(oof_chunk)
            elif result.get("weights"):
                # Linear weights case
                weights = result["weights"]
                scores = sum(test_data[s].values * weights.get(s, 0.0) for s in sleeves)
                oof_chunk = test_data[["date", "symbol", "fold"]].copy()
                oof_chunk["stacked_score"] = scores
                stacker_oof_rows.append(oof_chunk)

        except Exception as e:
            logger.warning(f"Fold {fold_id} stacker training failed: {e}")
            continue

    if not fold_results:
        raise RuntimeError("All fold stacker trainings failed")

    # Train final stacker on all data for inference
    logger.info("Training final stacker on all folds for inference model")
    final_result = train_stacker(
        stack_df[["date", "symbol", *sleeves]],
        stack_df["target"],
        train_cfg,
    )

    # Combine OOF predictions
    stacker_oof = pd.concat(stacker_oof_rows, ignore_index=True) if stacker_oof_rows else pd.DataFrame()

    # Add fold metrics to result
    final_result["cpcv_folds"] = len(fold_results)
    final_result["cpcv_consistent"] = True

    return final_result, stacker_oof


def run_stacker_pipeline(cfg: StackerTrainConfig) -> Dict:
    """Run meta stacker training pipeline.

    Per SSOT v2 §4.5, when cpcv_consistent=True (default), the stacker is trained
    using leave-one-fold-out to ensure no information leakage.
    """
    if not cfg.oof_sources:
        raise ValueError("No OOF sources provided for stacker training.")

    # Determine if we need fold information for CPCV consistency
    require_fold = cfg.cpcv_consistent

    oof_frames = []
    sleeves: List[str] = []
    has_fold_info = True

    for sleeve, path in cfg.oof_sources.items():
        try:
            df = _load_oof(path, require_fold=require_fold)
        except ValueError as e:
            if cfg.fallback_to_simple and "fold" in str(e):
                logger.warning(f"Fold info missing for {sleeve}; falling back to simple merge")
                df = _load_oof(path, require_fold=False)
                has_fold_info = False
            else:
                raise

        df = df.rename(columns={"score": sleeve})
        cols_to_keep = ["date", "symbol", sleeve]
        if "fold" in df.columns:
            cols_to_keep.append("fold")
        oof_frames.append(df[cols_to_keep])
        sleeves.append(sleeve)

    # Merge OOF frames
    merged = oof_frames[0]
    for df in oof_frames[1:]:
        merge_cols = ["date", "symbol"]
        if "fold" in merged.columns and "fold" in df.columns:
            # Keep fold from first frame if they match, otherwise drop
            df = df.drop(columns=["fold"], errors="ignore")
        merged = merged.merge(df, on=merge_cols, how="outer")

    # Check fold consistency across sleeves
    if cfg.cpcv_consistent and has_fold_info and "fold" not in merged.columns:
        if cfg.fallback_to_simple:
            logger.warning("Fold information lost during merge; falling back to simple training")
            has_fold_info = False
        else:
            raise ValueError("CPCV consistency requested but fold information not preserved in merge")

    # Build labels
    dataset = build_training_dataset(
        universe=cfg.universe,
        start_date=cfg.start_date,
        end_date=cfg.end_date,
        label_type=cfg.label_type,
        horizon_days=cfg.horizon_days,
    )
    label_frame = pd.DataFrame(
        {
            "date": pd.to_datetime(dataset.X["date"]).dt.date,
            "symbol": dataset.X["symbol"],
            "target": dataset.y,
        }
    )

    stack_df = merged.merge(label_frame, on=["date", "symbol"], how="inner")
    stack_df = stack_df.dropna(subset=sleeves + ["target"])
    if stack_df.empty:
        raise RuntimeError("No overlapping rows between OOF sources and labels.")

    train_cfg = {
        "algorithm": cfg.algorithm,
        "artifact_dir": cfg.artifact_dir,
        "drop_columns": (cfg.drop_columns or []) + ["target"],
    }
    if cfg.linear_weights:
        train_cfg["linear_weights"] = cfg.linear_weights

    # Choose training mode based on CPCV consistency
    stacker_oof = None
    if cfg.cpcv_consistent and has_fold_info and "fold" in stack_df.columns:
        # CPCV-consistent training per SSOT v2 §4.5
        logger.info("Using CPCV-consistent stacker training (leave-one-fold-out)")
        result, stacker_oof = _run_cpcv_stacker_training(stack_df, sleeves, train_cfg)
    else:
        # Simple training (all data at once)
        if cfg.cpcv_consistent:
            logger.warning("CPCV consistency requested but not available; using simple training")
        result = train_stacker(stack_df[["date", "symbol", *sleeves]], stack_df["target"], train_cfg)
        result["cpcv_consistent"] = False

    metadata_path = result.get("metadata_path")
    if cfg.update_router and metadata_path:
        _update_router_config(cfg.router_cfg_path, metadata_path, sleeves)

    # Persist stacked OOF predictions for diagnostics
    out_dir = Path(cfg.artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if stacker_oof is not None and not stacker_oof.empty:
        # Use CPCV-generated OOF predictions
        stacker_oof.to_parquet(out_dir / "stacker_oof_scores.parquet", index=False)
        logger.info(f"Saved CPCV-consistent stacker OOF predictions: {len(stacker_oof)} rows")
    elif metadata_path:
        # Generate OOF predictions (not strictly CPCV-consistent but useful for diagnostics)
        try:
            combined = stack_df[["date", "symbol"]].copy()
            if "fold" in stack_df.columns:
                combined["fold"] = stack_df["fold"]
            combined["stacked_score"] = stack_scores(
                stack_df[["date", "symbol", *sleeves]], metadata_path
            )
            combined.to_parquet(out_dir / "stacker_oof_scores.parquet", index=False)
        except Exception as exc:
            logger.warning(f"Failed to persist stacked OOF scores: {exc}")

    summary = {
        "n_samples": int(len(stack_df)),
        "sleeves": sleeves,
        "metrics": result.get("metrics"),
        "metadata_path": metadata_path,
        "cpcv_consistent": result.get("cpcv_consistent", False),
        "cpcv_folds": result.get("cpcv_folds"),
    }
    logger.info(f"Stacker training complete: {json.dumps(summary, indent=2, default=str)}")
    return summary


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Train meta stacker")
    parser.add_argument("--config", default="configs/training/stacker.yml")
    args = parser.parse_args()

    with open(args.config) as f:
        raw_cfg = yaml.safe_load(f) or {}
    cfg = StackerTrainConfig(**raw_cfg)
    run_stacker_pipeline(cfg)
