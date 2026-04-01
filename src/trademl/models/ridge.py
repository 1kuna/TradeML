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
