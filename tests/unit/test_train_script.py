from __future__ import annotations

import importlib.util
import sys
import types
from types import SimpleNamespace
from pathlib import Path

import pandas as pd


def test_train_module_import_does_not_require_catboost_for_baseline_paths() -> None:
    module = _load_train_module()

    assert callable(module.run_training)


def test_fold_report_preserves_training_report_shape() -> None:
    module = _load_train_module()
    folds = [
        SimpleNamespace(rank_ic=0.1, decile_spread=0.02, hit_rate=0.6, bucket_returns={"1": -0.01, "10": 0.01}),
        SimpleNamespace(rank_ic=0.3, decile_spread=0.04, hit_rate=0.7, bucket_returns={"1": -0.02, "10": 0.02}),
    ]

    report = module._fold_report(folds)

    assert report == {
        "folds": [
            {"rank_ic": 0.1, "decile_spread": 0.02, "hit_rate": 0.6, "bucket_returns": {"1": -0.01, "10": 0.01}},
            {"rank_ic": 0.3, "decile_spread": 0.04, "hit_rate": 0.7, "bucket_returns": {"1": -0.02, "10": 0.02}},
        ],
        "mean_rank_ic": 0.2,
        "decile_chart_data": {
            "fold_1": {"1": -0.01, "10": 0.01},
            "fold_2": {"1": -0.02, "10": 0.02},
        },
    }


def test_ensemble_predictions_rank_average_by_date() -> None:
    module = _load_train_module()
    ridge = pd.DataFrame(
        {
            "date": ["2026-04-24", "2026-04-24", "2026-04-24"],
            "symbol": ["A", "B", "C"],
            "prediction": [0.1, 0.3, 0.2],
            "label_5d": [0.0, 0.1, 0.2],
        }
    )
    lightgbm = pd.DataFrame(
        {
            "date": ["2026-04-24", "2026-04-24", "2026-04-24"],
            "symbol": ["A", "B", "C"],
            "prediction": [0.9, 0.1, 0.4],
            "label_5d": [0.0, 0.1, 0.2],
        }
    )

    ensemble = module._ensemble_predictions({"ridge": ridge, "lightgbm": lightgbm})

    assert ensemble["symbol"].tolist() == ["A", "B", "C"]
    assert ensemble.loc[ensemble["symbol"] == "A", "prediction"].iloc[0] == (1 / 3 + 1.0) / 2
    assert ensemble.loc[ensemble["symbol"] == "B", "prediction"].iloc[0] == (1.0 + 1 / 3) / 2


def test_prediction_report_scores_prediction_frame() -> None:
    module = _load_train_module()
    predictions = pd.DataFrame(
        {
            "date": ["2026-04-24", "2026-04-24", "2026-04-24"],
            "symbol": ["A", "B", "C"],
            "prediction": [0.1, 0.2, 0.3],
            "label_5d": [0.1, 0.2, 0.3],
        }
    )

    report = module._prediction_report(predictions)

    assert report["mean_rank_ic"] == 1.0
    assert report["folds"][0]["rank_ic"] == 1.0


def test_prediction_report_supports_non_primary_label_horizon() -> None:
    module = _load_train_module()
    predictions = pd.DataFrame(
        {
            "date": ["2026-04-24", "2026-04-24", "2026-04-24"],
            "symbol": ["A", "B", "C"],
            "prediction": [0.1, 0.2, 0.3],
            "label_1d": [0.1, 0.2, 0.3],
        }
    )

    report = module._prediction_report(predictions, label_col="label_1d")

    assert report["mean_rank_ic"] == 1.0


def _load_train_module():
    module_path = Path(__file__).resolve().parents[2] / "src" / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("train_under_test", module_path)
    assert spec is not None
    assert spec.loader is not None

    poisoned_catboost = types.ModuleType("trademl.models.catboost")
    prior = sys.modules.get("trademl.models.catboost")
    sys.modules["trademl.models.catboost"] = poisoned_catboost
    try:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if prior is None:
            sys.modules.pop("trademl.models.catboost", None)
        else:
            sys.modules["trademl.models.catboost"] = prior
    return module
