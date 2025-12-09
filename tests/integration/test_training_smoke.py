from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from feature_store.equities.dataset import build_training_dataset


def test_training_smoke_with_ridge(curated_prices):
    start = curated_prices["dates"][0]
    end = curated_prices["dates"][-10]

    ds = build_training_dataset(
        universe=curated_prices["symbols"],
        start_date=start,
        end_date=end,
        label_type="triple_barrier",
        tp_sigma=1.0,
        sl_sigma=0.5,
        max_h=5,
        vol_window=10,
        standardize_window=30,
    )

    if ds.X.empty or ds.y.empty:
        pytest.skip("Synthetic curated data did not produce a dataset")

    X_num = ds.X.select_dtypes(include=["number"]).copy()
    y = ds.y.loc[X_num.index]

    # Drop rows with NaNs introduced by PIT windows
    mask = ~(X_num.isna().any(axis=1) | y.isna())
    X_num = X_num[mask]
    y = y[mask]

    assert len(X_num) >= 1

    model = Ridge(alpha=1.0)
    model.fit(X_num, y)
    preds = model.predict(X_num.head(5))

    assert preds.shape[0] == min(5, len(X_num))
    assert np.isfinite(preds).all()
