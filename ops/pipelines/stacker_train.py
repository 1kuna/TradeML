"""Meta stacker training pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional

import pandas as pd
from loguru import logger

from feature_store.equities.dataset import build_training_dataset
from models.meta import stack_scores, train_stacker


def _load_oof(path: str) -> pd.DataFrame:
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


def run_stacker_pipeline(cfg: StackerTrainConfig) -> Dict:
    if not cfg.oof_sources:
        raise ValueError("No OOF sources provided for stacker training.")

    oof_frames = []
    sleeves: List[str] = []
    for sleeve, path in cfg.oof_sources.items():
        df = _load_oof(path).rename(columns={"score": sleeve})
        oof_frames.append(df[["date", "symbol", sleeve]])
        sleeves.append(sleeve)

    merged = oof_frames[0]
    for df in oof_frames[1:]:
        merged = merged.merge(df, on=["date", "symbol"], how="outer")

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

    result = train_stacker(stack_df[["date", "symbol", *sleeves]], stack_df["target"], train_cfg)

    metadata_path = result.get("metadata_path")
    if cfg.update_router and metadata_path:
        _update_router_config(cfg.router_cfg_path, metadata_path, sleeves)

    # Persist stacked OOF predictions for diagnostics
    if metadata_path:
        try:
            combined = stack_df[["date", "symbol"]].copy()
            combined["stacked_score"] = stack_scores(
                stack_df[["date", "symbol", *sleeves]], metadata_path
            )
            out_dir = Path(metadata_path).parent
            out_dir.mkdir(parents=True, exist_ok=True)
            combined.to_parquet(out_dir / "stacker_oof_scores.parquet", index=False)
        except Exception as exc:
            logger.warning(f"Failed to persist stacked OOF scores: {exc}")

    summary = {
        "n_samples": int(len(stack_df)),
        "sleeves": sleeves,
        "metrics": result.get("metrics"),
        "metadata_path": metadata_path,
    }
    logger.info(f"Stacker training complete: {json.dumps(summary, indent=2)}")
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
