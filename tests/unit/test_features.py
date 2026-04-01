from __future__ import annotations

import numpy as np
import pandas as pd

from feature_store.equities.dataset import build_training_dataset
from feature_store.equities.features import compute_equity_features
from labeling.triple_barrier.triple_barrier import triple_barrier
from trademl.features.equities import build_features


def _panel() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", "2025-12-31")
    symbols = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL"]
    rows = []
    for symbol_idx, symbol in enumerate(symbols):
        close = 50 + symbol_idx * 5 + np.cumsum(0.05 + 0.2 * np.sin(np.arange(len(dates)) / 20) + rng.normal(0, 0.5, len(dates)))
        close = np.maximum(close, 5)
        open_ = close * (1 + rng.normal(0, 0.002, len(dates)))
        volume = 1_000_000 + symbol_idx * 50_000 + rng.integers(0, 50_000, len(dates))
        for date, open_price, close_price, vol in zip(dates, open_, close, volume, strict=False):
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": float(open_price),
                    "high": float(max(open_price, close_price) * 1.01),
                    "low": float(min(open_price, close_price) * 0.99),
                    "close": float(close_price),
                    "vwap": float((open_price + close_price) / 2),
                    "volume": int(vol),
                }
            )
    return pd.DataFrame(rows)


def _feature_config() -> dict:
    return {
        "price": {"momentum": [5, 20, 60, 126], "reversal": [1, 5], "drawdown": [20, 60]},
        "volatility": {"realized": [20, 60], "idiosyncratic": [60]},
        "liquidity": {"adv_dollar": [20], "amihud": [20]},
        "controls": {"log_price": True},
    }


def test_features_compute_expected_columns_and_shape() -> None:
    panel = _panel()
    features = build_features(panel, _feature_config())

    expected = {
        "momentum_5d",
        "momentum_20d",
        "momentum_60d",
        "momentum_126d",
        "reversal_1d",
        "reversal_5d",
        "drawdown_20d",
        "drawdown_60d",
        "gap_overnight",
        "realized_vol_20d",
        "realized_vol_60d",
        "idiosyncratic_vol_60d",
        "adv_dollar_20d",
        "amihud_20d",
        "log_price",
    }
    assert expected.issubset(features.columns)
    assert len(features) == panel["date"].nunique() * panel["symbol"].nunique()


def test_features_are_pit_safe_outside_future_window() -> None:
    panel = _panel()
    mutated = panel.copy()
    mutated.loc[mutated["date"] >= pd.Timestamp("2025-06-01"), "close"] *= 2.0

    baseline = build_features(panel, _feature_config())
    shifted = build_features(mutated, _feature_config())

    cutoff = pd.Timestamp("2025-01-15")
    columns = ["momentum_5d", "momentum_20d", "gap_overnight", "log_price"]
    left = baseline.loc[baseline["date"] < cutoff, ["date", "symbol", *columns]].reset_index(drop=True)
    right = shifted.loc[shifted["date"] < cutoff, ["date", "symbol", *columns]].reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)


def test_compute_equity_features_uses_curated_history(curated_prices) -> None:
    asof = curated_prices["dates"][-1]
    features = compute_equity_features(asof, curated_prices["symbols"])

    assert not features.empty
    assert set(features["symbol"].unique()) == set(curated_prices["symbols"])
    assert features["asof"].iloc[0] == asof
    feature_cols = [column for column in features.columns if column.startswith("feature_")]
    assert feature_cols


def test_build_training_dataset_aligns_features_and_labels(curated_prices) -> None:
    start = curated_prices["dates"][0]
    end = curated_prices["dates"][-2]

    dataset = build_training_dataset(
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

    assert not dataset.X.empty
    assert len(dataset.X) == len(dataset.y) == len(dataset.meta)
    assert set(dataset.X["symbol"].unique()) <= set(curated_prices["symbols"])


def test_triple_barrier_labels_generate_from_curated_history(curated_prices) -> None:
    asof = curated_prices["dates"][-7]
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
