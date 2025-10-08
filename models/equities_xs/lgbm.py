"""
LightGBM-based learners for the equities cross-sectional sleeve.

The helpers here provide a thin abstraction over the scikit-learn style
LightGBM estimators so the pipelines can swap in tree ensembles without
duplicating problem-type detection or export logic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

try:  # LightGBM is optional but strongly encouraged for this sleeve.
    import lightgbm as lgb  # type: ignore
except Exception:  # pragma: no cover - handled via runtime checks
    lgb = None  # type: ignore

try:
    import joblib
except ImportError:  # pragma: no cover - joblib is in requirements but double check
    joblib = None  # type: ignore


@dataclass
class LGBMParams:
    """Default hyper-parameters aligned with the architecture blueprint."""

    num_leaves: int = 63
    max_depth: int = -1
    learning_rate: float = 0.05
    min_data_in_leaf: int = 200
    colsample_bytree: float = 0.8
    subsample: float = 0.8
    lambda_l1: float = 0.0
    lambda_l2: float = 0.0
    n_estimators: int = 400
    objective: Optional[str] = None
    random_state: int = 42


def _detect_problem_type(y: pd.Series) -> str:
    """Infer whether we are performing regression or binary classification."""
    clean = y.dropna().unique()
    if len(clean) <= 2 and set(clean) <= {0, 1}:
        return "binary"
    return "regression"


def _build_estimator(params: LGBMParams, problem_type: str):
    if lgb is None:
        raise ImportError(
            "lightgbm is required for the LightGBM sleeve but is not installed. "
            "Install lightgbm>=4.1.0 or select a different estimator."
        )

    param_dict = asdict(params).copy()
    objective = param_dict.pop("objective") or (
        "binary" if problem_type == "binary" else "regression"
    )
    param_dict.pop("random_state", None)  # handled via init kwargs

    common_kwargs = {
        "n_estimators": params.n_estimators,
        "learning_rate": params.learning_rate,
        "num_leaves": params.num_leaves,
        "max_depth": params.max_depth,
        "min_child_samples": params.min_data_in_leaf,
        "colsample_bytree": params.colsample_bytree,
        "subsample": params.subsample,
        "reg_alpha": params.lambda_l1,
        "reg_lambda": params.lambda_l2,
        "random_state": params.random_state,
    }

    if problem_type == "binary":
        return lgb.LGBMClassifier(objective=objective, **common_kwargs)  # type: ignore[arg-type]
    if problem_type == "regression":
        return lgb.LGBMRegressor(objective=objective, **common_kwargs)  # type: ignore[arg-type]
    raise ValueError(f"Unsupported problem type: {problem_type}")


def fit_lgbm(
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: Optional[np.ndarray] = None,
    params: LGBMParams | None = None,
) -> Tuple[object, Dict[str, float]]:
    """Fit a LightGBM model returning the estimator and basic in-sample metrics."""
    params = params or LGBMParams()
    problem_type = _detect_problem_type(y)
    est = _build_estimator(params, problem_type)

    est.fit(X, y, sample_weight=sample_weight)

    metrics: Dict[str, float] = {}
    preds = est.predict(X)
    if problem_type == "binary":
        try:
            from sklearn.metrics import accuracy_score, roc_auc_score

            proba = est.predict_proba(X)[:, 1]
            metrics["accuracy"] = float(accuracy_score(y, (proba >= 0.5).astype(int)))
            # Guard against degenerate labels
            if len(np.unique(y)) > 1:
                metrics["roc_auc"] = float(roc_auc_score(y, proba))
        except Exception as exc:  # pragma: no cover
            logger.debug(f"Failed to compute LightGBM classification metrics: {exc}")
    else:
        try:
            from sklearn.metrics import mean_squared_error, r2_score

            metrics["rmse"] = float(np.sqrt(mean_squared_error(y, preds)))
            if len(np.unique(y)) > 1:
                metrics["r2"] = float(r2_score(y, preds))
        except Exception as exc:  # pragma: no cover
            logger.debug(f"Failed to compute LightGBM regression metrics: {exc}")

    metrics["problem_type"] = problem_type
    logger.info(
        "LightGBM fitted "
        + ", ".join([f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, (int, float))])
        + f", problem={problem_type}"
    )
    return est, metrics


def predict_lgbm(model: object, X: pd.DataFrame, as_probability: bool = True) -> np.ndarray:
    """Predict using a LightGBM estimator."""
    if not hasattr(model, "predict"):
        raise TypeError("Model appears invalid for LightGBM prediction.")
    problem_type = None

    # If estimator exposes predict_proba, assume binary classification
    if hasattr(model, "predict_proba") and as_probability:
        try:
            return np.asarray(model.predict_proba(X)[:, 1])
        except Exception:
            problem_type = "regression"

    # Fall back to raw prediction
    preds = model.predict(X)
    if problem_type == "regression" or not isinstance(preds, np.ndarray):
        return np.asarray(preds)
    return preds


def save_pickle(model: object, path: str | Path) -> None:
    """Persist a fitted estimator to disk via joblib."""
    if joblib is None:
        raise ImportError("joblib is required to persist LightGBM models.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def export_onnx(model: object, feature_names: list[str], path: str | Path) -> Path:
    """
    Export a fitted LightGBM model to ONNX format.

    Requires skl2onnx + onnxruntime. If unavailable, raises RuntimeError to let
    the caller decide whether to treat it as fatal or to skip the export.
    """
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "ONNX export requires skl2onnx. Install skl2onnx>=1.15 to enable export."
        ) from exc

    initial_type = [("features", FloatTensorType([None, len(feature_names)]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    logger.info(f"LightGBM model exported to ONNX: {path}")
    return path
