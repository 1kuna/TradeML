from __future__ import annotations

import numpy as np
import pandas as pd

from trademl.models.ridge import RidgeModel
from trademl.validation.walk_forward import expanding_walk_forward


def _walk_forward_frame() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", "2024-12-31")
    symbols = ["AAPL", "MSFT", "NVDA", "AMZN", "META"]
    rows = []
    for symbol_idx, symbol in enumerate(symbols):
        feature = np.sin(np.arange(len(dates)) / 30) + symbol_idx * 0.1 + rng.normal(0, 0.1, len(dates))
        label = feature * 0.5 + rng.normal(0, 0.05, len(dates))
        for date, f_value, y_value in zip(dates, feature, label, strict=False):
            rows.append({"date": date, "symbol": symbol, "feature_1": f_value, "label": y_value})
    return pd.DataFrame(rows)


def test_walk_forward_fold_count_and_purge() -> None:
    frame = _walk_forward_frame()
    results = expanding_walk_forward(
        frame,
        ["feature_1"],
        "label",
        lambda: RidgeModel(alpha=1.0),
        {"initial_train_years": 2, "step_months": 3, "purge_days": 5},
    )

    assert len(results) == 12
    assert all(result.train_end < result.test_start for result in results)
    assert (results[0].test_start - results[0].train_end).days >= 5
    assert all(isinstance(result.rank_ic, float) for result in results)
    assert all(isinstance(result.decile_spread, float) for result in results)
