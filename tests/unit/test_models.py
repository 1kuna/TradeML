from __future__ import annotations

import numpy as np
import pandas as pd

from trademl.models.lgbm import LightGBMModel
from trademl.models.ridge import RidgeModel


def _model_frame() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    X = pd.DataFrame({"feature_1": rng.normal(size=200), "feature_2": rng.normal(size=200)})
    y = X["feature_1"] * 0.3 - X["feature_2"] * 0.2 + rng.normal(0, 0.05, size=200)
    return X, y


def test_ridge_and_lightgbm_train_and_predict() -> None:
    X, y = _model_frame()
    ridge = RidgeModel(alpha=1.0).fit(X, y)
    lgbm = LightGBMModel(n_trials=3).fit(X, y)

    ridge_predictions = ridge.predict(X.head(5))
    lgbm_predictions = lgbm.predict(X.head(5))

    assert len(ridge_predictions) == 5
    assert len(lgbm_predictions) == 5
    assert lgbm.trials_run <= 20
