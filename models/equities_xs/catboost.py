"""
CatBoost-based learners for the equities cross-sectional sleeve.

CatBoost is optional for this repository; these helpers wrap the API so
pipelines can opt in when the dependency is installed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

try:
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool  # type: ignore
except Exception:  # pragma: no cover - dependency optional
    CatBoostClassifier = None  # type: ignore
    CatBoostRegressor = None  # type: ignore
    Pool = None  # type: ignore

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None  # type: ignore


@dataclass
class CatBoostParams:
    depth: int = 7
    learning_rate: float = 0.05
    l2_leaf_reg: float = 3.0
    iterations: int = 400
    random_seed: int = 42
    loss_function: Optional[str] = None
    allow_writing_files: bool = False


def _detect_problem_type(y: pd.Series) -> str:
    clean = y.dropna().unique()
    if len(clean) <= 2 and set(clean) <= {0, 1}:
        return "binary"
    return "regression"


def fit_catboost(
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: Optional[np.ndarray] = None,
    params: CatBoostParams | None = None,
) -> Tuple[object, Dict[str, float]]:
    """Fit a CatBoost model; raises if the dependency is missing."""
    if CatBoostClassifier is None or CatBoostRegressor is None:
        raise ImportError("catboost is not installed. Install catboost>=1.2.0 to enable this sleeve.")

    params = params or CatBoostParams()
    cfg = asdict(params)
    allow_writing_files = cfg.pop("allow_writing_files", False)
    loss_function = cfg.pop("loss_function")

    problem_type = _detect_problem_type(y)
    if problem_type == "binary":
        model = CatBoostClassifier(
            loss_function=loss_function or "Logloss",
            allow_writing_files=allow_writing_files,
            **cfg,
        )
    else:
        model = CatBoostRegressor(
            loss_function=loss_function or "RMSE",
            allow_writing_files=allow_writing_files,
            **cfg,
        )

    pool = Pool(X, y, weight=sample_weight)
    model.fit(pool, verbose=False)

    metrics: Dict[str, float] = {}
    preds = model.predict(X)
    if problem_type == "binary":
        try:
            from sklearn.metrics import accuracy_score, roc_auc_score

            proba = model.predict_proba(X)[:, 1]
            metrics["accuracy"] = float(accuracy_score(y, (proba >= 0.5).astype(int)))
            if len(np.unique(y)) > 1:
                metrics["roc_auc"] = float(roc_auc_score(y, proba))
        except Exception as exc:  # pragma: no cover
            logger.debug(f"CatBoost classification metrics failed: {exc}")
    else:
        try:
            from sklearn.metrics import mean_squared_error, r2_score

            metrics["rmse"] = float(np.sqrt(mean_squared_error(y, preds)))
            if len(np.unique(y)) > 1:
                metrics["r2"] = float(r2_score(y, preds))
        except Exception as exc:  # pragma: no cover
            logger.debug(f"CatBoost regression metrics failed: {exc}")

    metrics["problem_type"] = problem_type
    logger.info(
        "CatBoost fitted "
        + ", ".join([f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, (int, float))])
        + f", problem={problem_type}"
    )
    return model, metrics


def predict_catboost(model: object, X: pd.DataFrame, as_probability: bool = True) -> np.ndarray:
    if model is None or not hasattr(model, "predict"):
        raise TypeError("Model is not a fitted CatBoost estimator.")
    if hasattr(model, "predict_proba") and as_probability:
        try:
            return np.asarray(model.predict_proba(X)[:, 1])
        except Exception:
            pass
    return np.asarray(model.predict(X))


def save_pickle(model: object, path: str | Path) -> None:
    if joblib is None:
        raise ImportError("joblib is required to persist CatBoost models.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def export_onnx(model: object, path: str | Path) -> Path:
    """Export CatBoost model to ONNX using the native saver."""
    if model is None or not hasattr(model, "save_model"):
        raise TypeError("Model is not a CatBoost estimator.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path), format="onnx")
    logger.info(f"CatBoost model exported to ONNX: {path}")
    return path
