"""CatBoost advanced challenger model."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import optuna
import pandas as pd

try:
    from catboost import CatBoostRegressor
except ModuleNotFoundError:  # pragma: no cover - exercised when dependency is absent at runtime
    CatBoostRegressor = None  # type: ignore[assignment]


@dataclass
class CatBoostModel:
    """CatBoost regressor with lightweight Optuna tuning."""

    n_trials: int = 10
    random_state: int = 42
    best_params: dict[str, float | int] = field(
        default_factory=lambda: {
            "depth": 6,
            "learning_rate": 0.05,
            "l2_leaf_reg": 3.0,
            "subsample": 0.9,
            "iterations": 200,
        }
    )
    trials_run: int = 0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "CatBoostModel":
        if CatBoostRegressor is None:
            raise ModuleNotFoundError("catboost is required for the advanced challenger lane")
        params = dict(self.best_params)
        iterations = int(params.pop("iterations", 200))
        self.model = CatBoostRegressor(
            loss_function="RMSE",
            random_seed=self.random_state,
            iterations=iterations,
            allow_writing_files=False,
            verbose=False,
            **params,
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


def tune_catboost_via_walk_forward(
    frame: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    validation_runner,
    validation_config: dict,
    *,
    n_trials: int = 10,
    random_state: int = 42,
) -> dict[str, float | int]:
    """Tune CatBoost hyperparameters using walk-forward rank IC."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            "depth": trial.suggest_categorical("depth", [4, 6, 8]),
            "learning_rate": trial.suggest_categorical("learning_rate", [0.02, 0.05, 0.1]),
            "l2_leaf_reg": trial.suggest_categorical("l2_leaf_reg", [3.0, 5.0, 7.0]),
            "subsample": trial.suggest_categorical("subsample", [0.7, 0.9]),
            "iterations": trial.suggest_categorical("iterations", [150, 200]),
        }
        folds = validation_runner(
            frame,
            feature_cols,
            label_col,
            lambda: CatBoostModel(n_trials=0, random_state=random_state, best_params=params),
            validation_config,
        )
        if not folds:
            return -1.0
        return float(np.mean([fold.rank_ic for fold in folds]))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=min(n_trials, 20), show_progress_bar=False)
    return study.best_params
