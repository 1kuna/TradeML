"""
Meta-stacking utilities for blending sleeve OOF predictions.

The stacker is intentionally flexible: it supports quick linear weights,
classical ridge/logistic stacks, or a LightGBM-based meta learner.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from loguru import logger

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None  # type: ignore

from models.equities_xs.lgbm import LGBMParams, fit_lgbm, predict_lgbm, save_pickle as save_lgbm_pickle


def _detect_problem_type(y: pd.Series) -> str:
    clean = y.dropna().unique()
    if len(clean) <= 2 and set(clean) <= {0, 1}:
        return "binary"
    return "regression"


def _feature_columns(oof_df: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.Index]:
    drop_cols = set(cfg.get("drop_columns", [])) | {"date", "symbol"}
    cols = [c for c in oof_df.columns if c not in drop_cols]
    if not cols:
        raise ValueError("No feature columns available for stacker training.")
    return oof_df[cols].copy(), pd.Index(cols)


def _artifact_dir(cfg: Dict[str, Any]) -> Path:
    base = Path(cfg.get("artifact_dir", "models/meta/artifacts"))
    ts = cfg.get("artifact_time") or time.strftime("%Y%m%d_%H%M%S")
    out = base / ts
    out.mkdir(parents=True, exist_ok=True)
    return out


def train_stacker(oof_df: pd.DataFrame, y: pd.Series, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Train a meta learner on OOF predictions.

    cfg keys:
        algorithm: 'ridge' (default), 'logistic', 'lgbm', or 'linear_weights'
        artifact_dir: base directory for outputs
        linear_weights: optional manual weights when algorithm == linear_weights
    """
    if oof_df.empty:
        raise ValueError("OOF dataframe is empty; cannot train stacker.")
    if len(oof_df) != len(y):
        raise ValueError("OOF dataframe and target series must align by length.")

    algorithm = (cfg.get("algorithm") or "ridge").lower()
    problem_type = _detect_problem_type(y)

    features, cols = _feature_columns(oof_df, cfg)
    metadata: Dict[str, Any] = {
        "algorithm": algorithm,
        "problem_type": problem_type,
        "feature_columns": list(cols),
        "config": cfg,
    }

    artifact_dir = _artifact_dir(cfg)
    model_path = artifact_dir / "stacker.pkl"
    metadata_path = artifact_dir / "metadata.json"

    if algorithm == "linear_weights":
        weights = cfg.get("linear_weights") or {}
        if not weights:
            raise ValueError("linear_weights algorithm requires cfg['linear_weights'].")
        weights = {k: float(v) for k, v in weights.items()}
        metadata.update({"weights": weights, "type": "linear_weights"})
        metadata_path.write_text(json.dumps(metadata, indent=2))
        logger.info(f"Stacker linear weights persisted to {metadata_path}")
        return {
            "model_path": None,
            "metadata_path": str(metadata_path),
            "metrics": {},
            "weights": weights,
        }

    if algorithm == "lgbm":
        model, metrics = fit_lgbm(features, y, params=cfg.get("params") or LGBMParams())
        save_lgbm_pickle(model, model_path)
        try:
            feature_names = list(cols)
            from models.equities_xs.lgbm import export_onnx

            export_onnx(model, feature_names, artifact_dir / "stacker.onnx")
            metadata["onnx_exported"] = True
        except Exception as exc:  # pragma: no cover - optional dependency
            metadata["onnx_exported"] = False
            metadata["onnx_error"] = str(exc)
            logger.warning(f"Stacker ONNX export skipped: {exc}")
    else:
        if joblib is None:
            raise ImportError("joblib is required to persist stacker estimators.")
        if algorithm == "logistic" and problem_type != "binary":
            raise ValueError("Logistic stacker requires binary targets.")

        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        if algorithm == "logistic":
            est = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=int(cfg.get("max_iter", 500)), class_weight="balanced")),
                ]
            )
        else:  # ridge
            est = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("reg", Ridge(alpha=float(cfg.get("alpha", 1.0)), random_state=42)),
                ]
            )
        est.fit(features, y)
        joblib.dump(est, model_path)
        preds = (
            est.predict_proba(features)[:, 1]
            if hasattr(est, "predict_proba")
            else est.predict(features)
        )
        metrics = {
            "rmse": float(np.sqrt(np.mean((preds - y.values) ** 2))),
            "mae": float(np.mean(np.abs(preds - y.values))),
        }

    metadata.update({"type": "model", "model_path": str(model_path), "metrics": metrics})
    metadata_path.write_text(json.dumps(metadata, indent=2))
    logger.info(f"Stacker model persisted to {artifact_dir}")
    return {
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "metrics": metrics,
    }


def load_stacker(path: str | Path) -> Tuple[Any, Dict[str, Any]]:
    """
    Load a stacker artifact.

    Returns (model_or_weights, metadata).
    """
    path = Path(path)
    if path.is_dir():
        metadata_path = path / "metadata.json"
    elif path.suffix == ".json":
        metadata_path = path
    else:
        raise ValueError("Provide a metadata.json path or artifact directory for stacker load.")

    if not metadata_path.exists():
        raise FileNotFoundError(f"Stacker metadata not found: {metadata_path}")

    metadata = json.loads(metadata_path.read_text())
    if metadata.get("type") == "linear_weights":
        return metadata["weights"], metadata

    model_path = metadata.get("model_path")
    if not model_path:
        raise ValueError("Stacker metadata missing model_path.")

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Stacker model not found: {model_path}")

    if joblib is None:
        raise ImportError("joblib is required to load stacker models.")
    model = joblib.load(model_path)
    return model, metadata


def stack_scores(df_scores: pd.DataFrame, model_or_weights: Any, metadata: Dict[str, Any] | None = None) -> np.ndarray:
    """
    Apply a trained stacker or linear weights to a score dataframe.

    df_scores should contain the feature columns used during training.
    """
    if isinstance(model_or_weights, (str, Path)):
        model_or_weights, metadata = load_stacker(model_or_weights)

    if isinstance(model_or_weights, dict):
        weights = {k: float(v) for k, v in model_or_weights.items()}
        missing = [k for k in weights if k not in df_scores.columns]
        if missing:
            logger.warning(f"Stacker weights missing columns {missing}; treating as zeros.")
        total = 0.0
        denom = 0.0
        for sleeve, w in weights.items():
            if sleeve not in df_scores:
                continue
            total += w * df_scores[sleeve].astype(float)
            denom += abs(w)
        if denom == 0:
            return np.zeros(len(df_scores))
        return (total / denom).to_numpy()

    feature_columns: Iterable[str] = (
        (metadata or {}).get("feature_columns") if metadata else None
    ) or df_scores.columns
    X = df_scores[list(feature_columns)].copy()

    if hasattr(model_or_weights, "predict_proba"):
        return np.asarray(model_or_weights.predict_proba(X)[:, 1])
    return np.asarray(model_or_weights.predict(X))
