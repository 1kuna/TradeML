from __future__ import annotations

import pandas as pd

from trademl.validation.metrics import bucket_metrics, mean_daily_bucket_spread, prediction_buckets, rank_ic


def test_rank_ic_normalizes_constant_or_nan_inputs_to_zero() -> None:
    assert rank_ic(pd.Series([1.0, 1.0, 1.0]), pd.Series([0.1, 0.2, 0.3])) == 0.0
    assert rank_ic(pd.Series([float("nan"), float("nan")]), pd.Series([0.1, 0.2])) == 0.0


def test_prediction_buckets_are_stable_rank_buckets() -> None:
    buckets = prediction_buckets(pd.Series([0.2, 0.2, 0.5, 0.1]), max_buckets=4)

    assert buckets.tolist() == [1, 2, 3, 0]


def test_bucket_metrics_match_top_bottom_contract() -> None:
    frame = pd.DataFrame(
        {
            "prediction": [0.1, 0.2, 0.3, 0.4],
            "label": [-0.04, -0.01, 0.02, 0.05],
        }
    )

    spread, hit_rate, returns = bucket_metrics(frame, label_col="label")

    assert spread == 0.09
    assert hit_rate == 1.0
    assert returns == {"1": -0.04, "2": -0.01, "3": 0.02, "4": 0.05}


def test_mean_daily_bucket_spread_averages_valid_dates_only() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-02", "2026-01-02", "2026-01-03"]),
            "prediction": [0.1, 0.9, 0.5],
            "label": [-0.02, 0.03, 0.10],
        }
    )

    assert mean_daily_bucket_spread(frame, label_col="label") == 0.05
