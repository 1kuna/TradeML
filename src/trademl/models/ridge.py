"""Ridge baseline model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


@dataclass
class RidgeModel:
    """Thin sklearn Ridge wrapper with fit/predict interface."""

    alpha: float = 1.0

    def __post_init__(self) -> None:
        self.model = Ridge(alpha=self.alpha, random_state=42, solver="lsqr")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RidgeModel":
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)


def tune_ridge_via_walk_forward(
    frame: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    validation_runner,
    validation_config: dict,
    *,
    alphas: list[float] | None = None,
) -> float:
    """Select the Ridge alpha with the best mean walk-forward rank IC."""
    candidate_alphas = alphas or [0.01, 0.1, 1.0, 10.0, 100.0]
    best_alpha = candidate_alphas[0]
    best_score = float("-inf")

    for alpha in candidate_alphas:
        folds = validation_runner(
            frame,
            feature_cols,
            label_col,
            lambda a=alpha: RidgeModel(alpha=a),
            validation_config,
        )
        if not folds:
            continue
        score = float(np.mean([fold.rank_ic for fold in folds]))
        if score > best_score:
            best_score = score
            best_alpha = alpha
    return best_alpha
