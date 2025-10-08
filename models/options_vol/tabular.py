"""
Tabular learners for the options volatility sleeve.

The sleeve expects delta-hedged PnL or edge targets at the underlier/day
granularity. These helpers train a gradient boosted tree (LightGBM when
available, otherwise a RandomForest baseline).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

try:
    import lightgbm as lgb  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    lgb = None  # type: ignore

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None  # type: ignore


@dataclass
class OptionsModelConfig:
    objective: str = "regression"
    learning_rate: float = 0.05
    num_leaves: int = 63
    n_estimators: int = 300
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    random_state: int = 42


def _train_lightgbm(X: pd.DataFrame, y: pd.Series, cfg: OptionsModelConfig) -> Tuple[object, Dict[str, float]]:
    model = lgb.LGBMRegressor(
        objective="regression",
        learning_rate=cfg.learning_rate,
        num_leaves=cfg.num_leaves,
        n_estimators=cfg.n_estimators,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        random_state=cfg.random_state,
    )
    model.fit(X, y)
    preds = model.predict(X)
    metrics = {
        "rmse": float(np.sqrt(np.mean((preds - y.values) ** 2))),
        "mae": float(np.mean(np.abs(preds - y.values))),
    }
    return model, metrics


def _train_forest(X: pd.DataFrame, y: pd.Series) -> Tuple[object, Dict[str, float]]:
    from sklearn.ensemble import RandomForestRegressor

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)
    preds = model.predict(X)
    metrics = {
        "rmse": float(np.sqrt(np.mean((preds - y.values) ** 2))),
        "mae": float(np.mean(np.abs(preds - y.values))),
    }
    return model, metrics


def train_options_model(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: Optional[OptionsModelConfig] = None,
) -> Tuple[object, Dict[str, float]]:
    """Train an options sleeve model returning estimator and quick metrics."""
    if X.empty:
        raise ValueError("Options model dataset is empty.")
    cfg = cfg or OptionsModelConfig()
    if lgb is not None:
        model, metrics = _train_lightgbm(X, y, cfg)
        backend = "lightgbm"
    else:
        model, metrics = _train_forest(X, y)
        backend = "random_forest"

    metrics["backend"] = backend
    logger.info(
        f"Options model trained via {backend}; rmse={metrics.get('rmse', float('nan')):.6f}"
    )
    return model, metrics


def predict_options_model(model: object, X: pd.DataFrame) -> np.ndarray:
    if not hasattr(model, "predict"):
        raise TypeError("Options model must expose predict().")
    return np.asarray(model.predict(X))


def save_model(model: object, path: str | Path) -> None:
    if joblib is None:
        raise ImportError("joblib is required to persist options models.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str | Path) -> object:
    if joblib is None:
        raise ImportError("joblib is required to load options models.")
    return joblib.load(path)
