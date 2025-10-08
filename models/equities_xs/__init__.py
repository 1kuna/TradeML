"""Cross-sectional equities models (baselines and challengers)."""

from .baselines import train_logistic_regression, train_ridge_regression
from .catboost import (
    CatBoostParams,
    export_onnx as export_catboost_onnx,
    fit_catboost,
    predict_catboost,
    save_pickle as save_catboost_pickle,
)
from .lgbm import (
    LGBMParams,
    export_onnx as export_lgbm_onnx,
    fit_lgbm,
    predict_lgbm,
    save_pickle as save_lgbm_pickle,
)

__all__ = [
    "train_ridge_regression",
    "train_logistic_regression",
    "LGBMParams",
    "fit_lgbm",
    "predict_lgbm",
    "export_lgbm_onnx",
    "save_lgbm_pickle",
    "CatBoostParams",
    "fit_catboost",
    "predict_catboost",
    "export_catboost_onnx",
    "save_catboost_pickle",
]
