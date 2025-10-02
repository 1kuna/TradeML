"""
Baseline cross-sectional models for equities.

Implements simple Ridge regression (for returns) and Logistic regression
(for classification labels) with standardization and sensible defaults.

These functions are intentionally lightweight and do not run CPCV or
hyper-parameter searches. Use `validation` utilities to evaluate under
purged, embargoed CV and compute PBO/DSR as needed.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    accuracy_score,
    roc_auc_score,
)


def _split_xy(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def train_ridge_regression(X: pd.DataFrame, y: pd.Series, alpha: float = 1.0) -> Tuple[Pipeline, Dict[str, float]]:
    """Train a Ridge regression baseline with standardization.

    Returns fitted pipeline and IS metrics on provided data (for quick checks).
    """
    pipe = Pipeline(
        [
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("model", Ridge(alpha=alpha, random_state=42)),
        ]
    )
    pipe.fit(X, y)
    preds = pipe.predict(X)
    metrics = {
        "r2": float(r2_score(y, preds)) if len(np.unique(y)) > 1 else 0.0,
        "rmse": float(np.sqrt(mean_squared_error(y, preds))),
    }
    logger.info(f"Ridge trained: R2={metrics['r2']:.3f}, RMSE={metrics['rmse']:.4f}")
    return pipe, metrics


def train_logistic_regression(
    X: pd.DataFrame,
    y: pd.Series,
    C: float = 1.0,
    class_weight: str | dict | None = "balanced",
) -> Tuple[Pipeline, Dict[str, float]]:
    """Train a Logistic regression baseline with standardization.

    Returns fitted pipeline and IS metrics on provided data (for quick checks).
    """
    pipe = Pipeline(
        [
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "model",
                LogisticRegression(
                    C=C,
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=200,
                    class_weight=class_weight,
                    random_state=42,
                ),
            ),
        ]
    )
    pipe.fit(X, y)
    preds = pipe.predict(X)
    metrics = {
        "accuracy": float(accuracy_score(y, preds)),
    }
    # AUC only for binary labels and with positive class present
    try:
        if set(np.unique(y)) <= {0, 1} and len(np.unique(y)) == 2:
            proba = pipe.predict_proba(X)[:, 1]
            metrics["roc_auc"] = float(roc_auc_score(y, proba))
    except Exception:
        pass
    logger.info(
        "Logistic trained: "
        + ", ".join([f"{k}={v:.3f}" for k, v in metrics.items()])
    )
    return pipe, metrics

