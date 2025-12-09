from __future__ import annotations

from datetime import date

from feature_store.equities.dataset import build_training_dataset
from feature_store.equities.features import compute_equity_features
from labeling.triple_barrier.triple_barrier import triple_barrier


def test_compute_equity_features_uses_curated_history(curated_prices):
    asof = curated_prices["dates"][-1]
    feats = compute_equity_features(asof, curated_prices["symbols"])

    assert not feats.empty
    assert set(feats["symbol"].unique()) == set(curated_prices["symbols"])
    assert feats["asof"].iloc[0] == asof
    feature_cols = [c for c in feats.columns if c.startswith("feature_")]
    assert feature_cols  # ensure actual feature columns are present


def test_build_training_dataset_aligns_features_and_labels(curated_prices):
    start = curated_prices["dates"][0]
    end = curated_prices["dates"][-2]

    ds = build_training_dataset(
        universe=curated_prices["symbols"],
        start_date=start,
        end_date=end,
        label_type="triple_barrier",
        tp_sigma=1.5,
        sl_sigma=0.75,
        max_h=5,
        vol_window=10,
        standardize_window=30,
    )

    assert not ds.X.empty
    assert len(ds.X) == len(ds.y) == len(ds.meta)
    assert set(ds.X["symbol"].unique()) <= set(curated_prices["symbols"])


def test_triple_barrier_labels_generate_from_curated_history(curated_prices):
    asof = curated_prices["dates"][-7]  # leave forward path for max_h
    labels = triple_barrier(
        asof_date=asof,
        universe=curated_prices["symbols"],
        tp_sigma=1.0,
        sl_sigma=0.5,
        max_h=5,
        vol_window=10,
    )

    assert not labels.empty
    assert set(labels["symbol"].unique()) <= set(curated_prices["symbols"])
    assert labels["entry_date"].iloc[0] == asof
