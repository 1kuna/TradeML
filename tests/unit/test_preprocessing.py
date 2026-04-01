from __future__ import annotations

import pandas as pd

from trademl.features.preprocessing import rank_normalize


def test_rank_normalize_maps_into_unit_range_without_nans() -> None:
    frame = pd.DataFrame(
        {
            "date": ["2026-01-02"] * 4 + ["2026-01-03"] * 4,
            "symbol": list("ABCD") * 2,
            "feature_a": [1.0, 2.0, None, 4.0, 10.0, 30.0, 20.0, None],
            "feature_b": [5.0, 4.0, 3.0, 2.0, 1.0, None, 0.5, 0.1],
        }
    )

    normalized = rank_normalize(frame, ["feature_a", "feature_b"])

    assert normalized[["feature_a", "feature_b"]].max().max() <= 1.0
    assert normalized[["feature_a", "feature_b"]].min().min() >= -1.0
    assert normalized[["feature_a", "feature_b"]].isna().sum().sum() == 0


def test_feature_dates_over_missing_threshold_are_dropped() -> None:
    frame = pd.DataFrame(
        {
            "date": ["2026-01-02"] * 4,
            "symbol": list("ABCD"),
            "feature_a": [1.0, None, None, None],
        }
    )

    normalized = rank_normalize(frame, ["feature_a"], missing_threshold=0.30)

    assert normalized["feature_a"].isna().all()
