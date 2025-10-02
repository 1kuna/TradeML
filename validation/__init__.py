"""Validation framework for anti-overfitting.

Exposes convenience wrapper `run_cpcv` to align with the blueprint's
minimal API, alongside CPCV, PBO, and DSR primitives.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .cpcv import CPCV
from .pbo import PBOCalculator, calculate_pbo
from .dsr import DSRCalculator, calculate_dsr

__all__ = [
    "CPCV",
    "PBOCalculator",
    "calculate_pbo",
    "DSRCalculator",
    "calculate_dsr",
    "run_cpcv",
]


def run_cpcv(
    X: pd.DataFrame,
    y: pd.Series,
    groups: Optional[pd.DataFrame] = None,
    embargo_days: int = 10,
    n_folds: int = 8,
) -> Dict:
    """Run CPCV with a simple baseline estimator and return split metrics.

    This is a lightweight convenience to quickly exercise CPCV mechanics.
    For full experiments, users should build explicit model pipelines, then
    compute PBO/DSR using validation primitives.

    Args:
        X: Features (may include columns like 'date'/'symbol')
        y: Target series
        groups: DataFrame with at least 'date' and 'horizon_days' columns.
                If None, attempts to infer 'date' from X and uses horizon_days=5.
        embargo_days: Embargo window after test set
        n_folds: Number of folds

    Returns:
        Dict with per-fold metrics and summary statistics.
    """
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import accuracy_score, roc_auc_score, r2_score, mean_squared_error

    # Prepare labels DataFrame for CPCV (date + horizon)
    if groups is None:
        if "date" in X.columns:
            dates = X["date"]
        else:
            raise ValueError("groups is None and 'date' column not found in X")
        labels_df = pd.DataFrame({"date": pd.to_datetime(dates).dt.date, "horizon_days": 5})
    else:
        if "date" not in groups or "horizon_days" not in groups:
            raise ValueError("groups must contain 'date' and 'horizon_days'")
        labels_df = groups.copy()
        labels_df["date"] = pd.to_datetime(labels_df["date"]).dt.date

    # Determine problem type
    y_values = y.values
    unique = np.unique(y_values[~pd.isna(y_values)])
    is_classification = len(unique) <= 10 and set(unique) <= {0, 1}

    # Drop non-feature columns if present
    X_model = X.copy()
    for col in ("date", "symbol", "asof"):
        if col in X_model:
            X_model = X_model.drop(columns=[col])

    # Build model pipeline
    if is_classification:
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=200, class_weight="balanced", random_state=42)),
            ]
        )
    else:
        model = Pipeline([("scaler", StandardScaler()), ("reg", Ridge(random_state=42))])

    # Run CPCV splits
    cv = CPCV(n_folds=n_folds, embargo_days=embargo_days)
    splits = cv.split(X.assign(date=labels_df["date"]), labels_df)

    per_split = []
    for (train_idx, test_idx) in splits:
        model.fit(X_model.iloc[train_idx], y.iloc[train_idx])
        y_pred = model.predict(X_model.iloc[test_idx])
        if is_classification:
            m = {
                "accuracy": float(accuracy_score(y.iloc[test_idx], y_pred)),
            }
            try:
                proba = model.predict_proba(X_model.iloc[test_idx])[:, 1]
                m["roc_auc"] = float(roc_auc_score(y.iloc[test_idx], proba))
            except Exception:
                pass
        else:
            m = {
                "r2": float(r2_score(y.iloc[test_idx], y_pred)) if len(np.unique(y.iloc[test_idx])) > 1 else 0.0,
                "rmse": float(np.sqrt(mean_squared_error(y.iloc[test_idx], y_pred))),
            }
        per_split.append(m)

    # Summarize
    summary: Dict[str, float] = {}
    if per_split:
        keys = per_split[0].keys()
        for k in keys:
            vals = np.array([d[k] for d in per_split if k in d])
            if len(vals):
                summary[f"{k}_mean"] = float(np.mean(vals))
                summary[f"{k}_std"] = float(np.std(vals))

    return {
        "problem_type": "classification" if is_classification else "regression",
        "n_folds": int(n_folds),
        "embargo_days": int(embargo_days),
        "per_split_metrics": per_split,
        "summary": summary,
    }
