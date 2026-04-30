from __future__ import annotations

import numpy as np
import pandas as pd

from trademl.features.equities import build_features
from trademl.modeling import build_modeling_artifacts, feature_label_preflight, load_modeling_dataset
from trademl.modeling.factory import _feature_groups_for


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


def test_reversal_features_are_negative_of_recent_returns_not_momentum_duplicates() -> None:
    panel = pd.DataFrame(
        {
            "date": pd.bdate_range("2026-01-01", periods=8).tolist() * 2,
            "symbol": ["AAPL"] * 8 + ["MSFT"] * 8,
            "open": [10, 11, 12, 13, 12, 11, 10, 9, 20, 19, 18, 17, 18, 19, 20, 21],
            "high": [10, 11, 12, 13, 12, 11, 10, 9, 20, 19, 18, 17, 18, 19, 20, 21],
            "low": [10, 11, 12, 13, 12, 11, 10, 9, 20, 19, 18, 17, 18, 19, 20, 21],
            "close": [10, 11, 12, 13, 12, 11, 10, 9, 20, 19, 18, 17, 18, 19, 20, 21],
            "vwap": [10, 11, 12, 13, 12, 11, 10, 9, 20, 19, 18, 17, 18, 19, 20, 21],
            "volume": [1_000_000] * 16,
        }
    )

    features = build_features(panel, _feature_config())
    aligned = features.dropna(subset=["momentum_5d", "reversal_5d", "reversal_1d"]).reset_index(drop=True)

    assert not aligned.empty
    pd.testing.assert_series_equal(
        aligned["reversal_5d"],
        -aligned["momentum_5d"],
        check_names=False,
    )
    assert not aligned["reversal_1d"].equals(aligned["momentum_5d"])


def test_modeling_feature_label_factory_writes_pit_versioned_artifacts(tmp_path) -> None:
    data_root = tmp_path / "nas"
    panel = _panel().loc[lambda frame: frame["date"] <= pd.Timestamp("2020-10-30")].copy()
    for date_value, day_frame in panel.groupby(panel["date"].dt.strftime("%Y-%m-%d")):
        partition = data_root / "data" / "curated" / "equities_ohlcv_adj" / f"date={date_value}"
        partition.mkdir(parents=True)
        day_frame.to_parquet(partition / "data.parquet", index=False)

    payload = build_modeling_artifacts(
        data_root=data_root,
        feature_config=_feature_config(),
        feature_version="price_liquidity_test_v1",
        label_horizons=[1, 5, 20],
    )
    frame, metadata = load_modeling_dataset(
        data_root=data_root,
        feature_version="price_liquidity_test_v1",
        label_version="universe_relative_forward_return_v1",
        label_horizon=5,
    )
    preflight = feature_label_preflight(
        data_root=data_root,
        feature_version="price_liquidity_test_v1",
        label_version="universe_relative_forward_return_v1",
        label_horizon=5,
    )

    assert payload["feature_version"] == "price_liquidity_test_v1"
    assert {"feature_available_at", "feature_version", "data_revision", "label_5d"}.issubset(frame.columns)
    assert (pd.to_datetime(frame["feature_available_at"], utc=True).dt.tz_convert(None) <= pd.to_datetime(frame["date"])).all()
    assert metadata["label_horizon"] == 5
    assert preflight["ok"] is True

    cutoff_frame, _ = load_modeling_dataset(
        data_root=data_root,
        feature_version="price_liquidity_test_v1",
        label_version="universe_relative_forward_return_v1",
        label_horizon=5,
        report_date="2020-10-23",
    )
    immature = pd.to_datetime(cutoff_frame["target_date_5d"]) > pd.Timestamp("2020-10-23")
    assert cutoff_frame.loc[immature, "label_5d"].isna().all()


def test_modeling_feature_factory_adds_multisource_pit_features(tmp_path) -> None:
    data_root = tmp_path / "nas"
    panel = _panel().loc[lambda frame: frame["date"] <= pd.Timestamp("2020-02-14")].copy()
    for date_value, day_frame in panel.groupby(panel["date"].dt.strftime("%Y-%m-%d")):
        partition = data_root / "data" / "curated" / "equities_ohlcv_adj" / f"date={date_value}"
        partition.mkdir(parents=True)
        day_frame.to_parquet(partition / "data.parquet", index=False)
    reference = data_root / "data" / "reference"
    reference.mkdir(parents=True)
    pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "metric_date": ["2019-12-31"],
            "metric_name": ["assets"],
            "metric_value": [100.0],
            "last_verified": ["2020-01-10"],
        }
    ).to_parquet(reference / "fundamentals_daily.parquet", index=False)
    pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "form": ["10-Q"],
            "accepted_at": ["2020-01-15T22:00:00Z"],
        }
    ).to_parquet(reference / "sec_filing_index.parquet", index=False)
    news_partition = data_root / "data" / "raw" / "ticker_news" / "date=2020-01-20"
    news_partition.mkdir(parents=True)
    pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "published_at": ["2020-01-20T15:00:00Z"],
            "source": ["alpaca"],
            "headline": ["test"],
        }
    ).to_parquet(news_partition / "data.parquet", index=False)
    minute_partition = data_root / "data" / "raw" / "equities_minute" / "date=2020-01-21"
    minute_partition.mkdir(parents=True)
    pd.DataFrame(
        {
            "symbol": ["AAPL"] * 35,
            "timestamp": pd.date_range("2020-01-21 14:30", periods=35, freq="min", tz="UTC"),
            "open": [100.0] * 35,
            "high": [101.0] * 35,
            "low": [99.0] * 35,
            "close": [100.0 + idx * 0.1 for idx in range(35)],
            "volume": [1000] * 35,
        }
    ).to_parquet(minute_partition / "data.parquet", index=False)

    payload = build_modeling_artifacts(
        data_root=data_root,
        feature_config=_feature_config(),
        feature_set="daily_multi_source_v1",
        feature_version="daily_multi_source_test_v1",
        label_horizons=[1, 5],
    )
    frame, _ = load_modeling_dataset(
        data_root=data_root,
        feature_version="daily_multi_source_test_v1",
        label_version="universe_relative_forward_return_v1",
        label_horizon=1,
    )
    apple = frame.loc[(frame["symbol"] == "AAPL") & (pd.to_datetime(frame["date"]) >= pd.Timestamp("2020-01-22"))]

    assert {"fundamentals_sec", "news_events", "minute_daily"}.issubset(set(payload["feature_groups"]))
    assert {
        "fundamental_metric_count",
        "sec_filings_30d",
        "news_count_7d",
        "news_abnormal_volume_7d",
        "news_novelty_proxy_7d",
        "minute_intraday_return",
        "minute_volume_ratio",
        "minute_close_imbalance_proxy",
    }.issubset(frame.columns)
    assert payload["feature_group_metadata"]["news_events"]["safety_delay"] == "1d"
    assert payload["feature_group_metadata"]["minute_daily"]["feature_available_at_policy"] == "prior-session minute aggregates are shifted to the next modeling date"
    assert not apple.empty
    assert apple["news_count_7d"].max() >= 1
    assert apple["minute_intraday_return"].max() > 0
    assert (pd.to_datetime(frame["feature_available_at"], utc=True).dt.tz_convert(None) <= pd.to_datetime(frame["date"])).all()


def test_feature_version_names_select_only_their_intended_optional_groups() -> None:
    assert _feature_groups_for(
        feature_set="daily_price_liquidity_v1",
        feature_version="sec_filing_events_v1",
    ) == ["price_liquidity", "fundamentals_sec"]
    assert _feature_groups_for(
        feature_set="daily_price_liquidity_v1",
        feature_version="news_event_aggregates_v1",
    ) == ["price_liquidity", "news_events"]
    assert _feature_groups_for(
        feature_set="daily_price_liquidity_v1",
        feature_version="minute_daily_aggregates_v1",
    ) == ["price_liquidity", "minute_daily"]
    assert _feature_groups_for(
        feature_set="daily_price_liquidity_v1",
        feature_version="multi_source_daily_v1",
    ) == ["price_liquidity", "fundamentals_sec", "news_events", "minute_daily"]
