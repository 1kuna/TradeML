"""LightGBM challenger model."""

from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd


@dataclass
class LightGBMModel:
    """LightGBM regressor with lightweight Optuna tuning."""

    n_trials: int = 10
    random_state: int = 42
    best_params: dict[str, float | int] = field(default_factory=dict)
    trials_run: int = 0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LightGBMModel":
        if not self.best_params:
            self.best_params = self._tune(X, y)
        self.model = lgb.LGBMRegressor(
            objective="regression",
            random_state=self.random_state,
            n_estimators=100,
            **self.best_params,
        )
        weights = self._time_decay_weights(len(X))
        self.model.fit(X, y, sample_weight=weights)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def _tune(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float | int]:
        split = max(1, int(len(X) * 0.8))
        train_X, valid_X = X.iloc[:split], X.iloc[split:]
        train_y, valid_y = y.iloc[:split], y.iloc[split:]
        if valid_X.empty:
            valid_X, valid_y = train_X, train_y

        def objective(trial: optuna.Trial) -> float:
            params = {
                "num_leaves": trial.suggest_categorical("num_leaves", [31, 63]),
                "learning_rate": trial.suggest_categorical("learning_rate", [0.01, 0.05]),
                "min_child_samples": trial.suggest_categorical("min_child_samples", [50, 200]),
                "colsample_bytree": trial.suggest_categorical("colsample_bytree", [0.6, 0.9]),
                "subsample": trial.suggest_categorical("subsample", [0.6, 0.9]),
            }
            model = lgb.LGBMRegressor(objective="regression", random_state=self.random_state, n_estimators=75, **params)
            model.fit(train_X, train_y)
            predictions = model.predict(valid_X)
            return float(np.mean((predictions - valid_y) ** 2))

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=min(self.n_trials, 20), show_progress_bar=False)
        self.trials_run = len(study.trials)
        return study.best_params

    def _time_decay_weights(self, n_rows: int) -> np.ndarray:
        if n_rows <= 1:
            return np.ones(n_rows)
        idx = np.arange(n_rows)
        half_life = max(1, int(252 * 1.5))
        return 0.5 ** ((n_rows - 1 - idx) / half_life)
