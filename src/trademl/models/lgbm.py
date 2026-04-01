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
    best_params: dict[str, float | int] = field(
        default_factory=lambda: {
            "num_leaves": 31,
            "learning_rate": 0.05,
            "min_child_samples": 50,
            "colsample_bytree": 0.9,
            "subsample": 0.9,
        }
    )
    trials_run: int = 0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LightGBMModel":
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

    def _time_decay_weights(self, n_rows: int) -> np.ndarray:
        if n_rows <= 1:
            return np.ones(n_rows)
        idx = np.arange(n_rows)
        half_life = max(1, int(252 * 1.5))
        return 0.5 ** ((n_rows - 1 - idx) / half_life)


def tune_lightgbm_via_walk_forward(
    frame: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    validation_runner,
    validation_config: dict,
    *,
    n_trials: int = 10,
    random_state: int = 42,
) -> dict[str, float | int]:
    """Tune LightGBM hyperparameters using walk-forward rank IC."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            "num_leaves": trial.suggest_categorical("num_leaves", [31, 63]),
            "learning_rate": trial.suggest_categorical("learning_rate", [0.01, 0.05]),
            "min_child_samples": trial.suggest_categorical("min_child_samples", [50, 200]),
            "colsample_bytree": trial.suggest_categorical("colsample_bytree", [0.6, 0.9]),
            "subsample": trial.suggest_categorical("subsample", [0.6, 0.9]),
        }
        folds = validation_runner(
            frame,
            feature_cols,
            label_col,
            lambda: LightGBMModel(n_trials=0, random_state=random_state, best_params=params),
            validation_config,
        )
        if not folds:
            return -1.0
        return float(np.mean([fold.rank_ic for fold in folds]))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=min(n_trials, 20), show_progress_bar=False)
    return study.best_params
