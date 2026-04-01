from __future__ import annotations

import numpy as np
import pandas as pd

from trademl.models.ridge import RidgeModel
from trademl.validation.cpcv import combinatorially_purged_cv
from trademl.validation.diagnostics import placebo_test


def _cpcv_frame() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2023-01-01", periods=240)
    rows = []
    for symbol_idx, symbol in enumerate(["AAPL", "MSFT", "NVDA", "AMZN"]):
        feature = rng.normal(symbol_idx, 1.0, len(dates))
        label = feature * 0.1 + rng.normal(0, 0.01, len(dates))
        for date, f_value, y_value in zip(dates, feature, label, strict=False):
            rows.append({"date": date, "symbol": symbol, "feature_1": f_value, "label": y_value})
    return pd.DataFrame(rows)


def test_cpcv_retains_majority_of_training_rows() -> None:
    frame = _cpcv_frame()
    results = combinatorially_purged_cv(frame, ["feature_1"], "label", lambda: RidgeModel(alpha=1.0), n_folds=8, embargo_days=10)

    assert results
    assert min(result.retention for result in results) > 0.7


def test_placebo_labels_are_near_zero() -> None:
    frame = _cpcv_frame()
    scores = placebo_test(frame, ["feature_1"], "label", lambda: RidgeModel(alpha=1.0), n_shuffles=5)
    assert max(abs(score) for score in scores) < 0.1
