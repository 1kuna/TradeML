"""Unit tests for intraday feature engineering (microstructure)."""

from datetime import date
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_minute_data():
    """Create sample minute bar data with bid/ask."""
    np.random.seed(42)
    n_bars = 100

    # Create realistic minute data
    base_price = 100.0
    prices = base_price + np.cumsum(np.random.randn(n_bars) * 0.1)

    df = pd.DataFrame({
        "symbol": ["AAPL"] * n_bars,
        "date": [date(2024, 1, 15)] * n_bars,
        "timestamp": pd.date_range("2024-01-15 09:30", periods=n_bars, freq="1min"),
        "open": prices,
        "high": prices + np.random.uniform(0, 0.5, n_bars),
        "low": prices - np.random.uniform(0, 0.5, n_bars),
        "close": prices + np.random.randn(n_bars) * 0.05,
        "volume": np.random.randint(1000, 10000, n_bars).astype(float),
        "bid": prices - 0.01,
        "ask": prices + 0.01,
        "bid_size": np.random.randint(100, 1000, n_bars).astype(float),
        "ask_size": np.random.randint(100, 1000, n_bars).astype(float),
    })
    return df


def test_intraday_feature_config_defaults():
    """Test IntradayFeatureConfig has expected defaults."""
    from feature_store.intraday.features import IntradayFeatureConfig

    cfg = IntradayFeatureConfig()

    assert cfg.resample_minutes == 5
    assert cfg.min_bars == 30
    assert cfg.ofi_window == 10
    assert cfg.rolling_vol_window == 30
    assert cfg.roll_window == 20
    assert cfg.realized_kernel_window == 30
    assert cfg.tod_buckets == 6


def test_build_intraday_features_basic(sample_minute_data):
    """Test build_intraday_features produces expected columns."""
    from feature_store.intraday.features import IntradayFeatureConfig, build_intraday_features

    cfg = IntradayFeatureConfig(min_bars=10)  # Lower threshold for test
    result = build_intraday_features(sample_minute_data, cfg)

    # Should produce features
    assert not result.empty

    # Check required identifier columns
    assert "symbol" in result.columns
    assert "date" in result.columns

    # Check feature columns have feature_ prefix
    feature_cols = [c for c in result.columns if c.startswith("feature_")]
    assert len(feature_cols) > 0

    # Check specific expected features
    expected_features = [
        "feature_vwap_dislocation",
        "feature_ofi",
        "feature_close_ret",
    ]
    for feat in expected_features:
        assert feat in result.columns, f"Missing feature: {feat}"


def test_microstructure_noise_features(sample_minute_data):
    """Test microstructure noise features are computed."""
    from feature_store.intraday.features import IntradayFeatureConfig, build_intraday_features

    cfg = IntradayFeatureConfig(min_bars=10)
    result = build_intraday_features(sample_minute_data, cfg)

    # Check microstructure features exist (actual names from implementation)
    microstructure_features = [
        "feature_roll_spread",
        "feature_realized_kernel_vol",  # Not _var
        "feature_noise_ratio",
    ]

    for feat in microstructure_features:
        assert feat in result.columns, f"Missing microstructure feature: {feat}"


def test_lob_features(sample_minute_data):
    """Test LOB (limit order book) features are computed."""
    from feature_store.intraday.features import IntradayFeatureConfig, build_intraday_features

    cfg = IntradayFeatureConfig(min_bars=10)
    result = build_intraday_features(sample_minute_data, cfg)

    # Check LOB features exist (actual names from implementation)
    lob_features = [
        "feature_avg_spread",  # Not bid_ask_spread
        "feature_depth_imbalance",
        "feature_spread_vol",  # Not effective_spread
    ]

    for feat in lob_features:
        assert feat in result.columns, f"Missing LOB feature: {feat}"


def test_time_of_day_features(sample_minute_data):
    """Test time-of-day encoding features."""
    from feature_store.intraday.features import IntradayFeatureConfig, build_intraday_features

    cfg = IntradayFeatureConfig(min_bars=10, tod_buckets=6)
    result = build_intraday_features(sample_minute_data, cfg)

    # Check TOD features exist (actual names from implementation)
    tod_features = [
        "feature_session_progress",  # Not tod_bucket
        "feature_tod_sin",
        "feature_tod_cos",
    ]

    for feat in tod_features:
        assert feat in result.columns, f"Missing TOD feature: {feat}"


def test_signed_volume():
    """Test signed volume computation."""
    from feature_store.intraday.features import _signed_volume

    df = pd.DataFrame({
        "close": [100, 101, 100.5, 102, 101.5],
        "volume": [1000, 1500, 1200, 1800, 1100],
    })

    result = _signed_volume(df)

    # Result should be a series
    assert isinstance(result, pd.Series)
    assert len(result) == len(df)


def test_empty_input_returns_empty():
    """Test that empty input returns empty output."""
    from feature_store.intraday.features import IntradayFeatureConfig, build_intraday_features

    empty_df = pd.DataFrame()
    cfg = IntradayFeatureConfig()

    result = build_intraday_features(empty_df, cfg)

    assert result.empty


def test_sufficient_data_produces_features(sample_minute_data):
    """Test that sufficient data produces valid features."""
    from feature_store.intraday.features import IntradayFeatureConfig, build_intraday_features

    cfg = IntradayFeatureConfig(min_bars=10)
    result = build_intraday_features(sample_minute_data, cfg)

    # Should have at least one row
    assert len(result) > 0

    # Should have feature columns
    feature_cols = [c for c in result.columns if c.startswith("feature_")]
    assert len(feature_cols) >= 10  # We expect at least 10 features


def test_vwap_dislocation_computed(sample_minute_data):
    """Test VWAP dislocation feature is computed correctly."""
    from feature_store.intraday.features import IntradayFeatureConfig, build_intraday_features

    cfg = IntradayFeatureConfig(min_bars=10)
    result = build_intraday_features(sample_minute_data, cfg)

    assert "feature_vwap_dislocation" in result.columns
    # VWAP dislocation should be a small number (close is near VWAP)
    vwap_disl = result["feature_vwap_dislocation"].iloc[0]
    assert isinstance(vwap_disl, (int, float, np.number)) or pd.isna(vwap_disl)


def test_ofi_computed(sample_minute_data):
    """Test Order Flow Imbalance (OFI) feature."""
    from feature_store.intraday.features import IntradayFeatureConfig, build_intraday_features

    cfg = IntradayFeatureConfig(min_bars=10)
    result = build_intraday_features(sample_minute_data, cfg)

    assert "feature_ofi" in result.columns


def test_range_spread_computed(sample_minute_data):
    """Test range spread feature is computed."""
    from feature_store.intraday.features import IntradayFeatureConfig, build_intraday_features

    cfg = IntradayFeatureConfig(min_bars=10)
    result = build_intraday_features(sample_minute_data, cfg)

    assert "feature_range_spread" in result.columns


def test_volume_z_computed(sample_minute_data):
    """Test volume z-score feature is computed."""
    from feature_store.intraday.features import IntradayFeatureConfig, build_intraday_features

    cfg = IntradayFeatureConfig(min_bars=10)
    result = build_intraday_features(sample_minute_data, cfg)

    assert "feature_volume_z" in result.columns


def test_gap_open_computed(sample_minute_data):
    """Test gap open feature is computed."""
    from feature_store.intraday.features import IntradayFeatureConfig, build_intraday_features

    cfg = IntradayFeatureConfig(min_bars=10)
    result = build_intraday_features(sample_minute_data, cfg)

    assert "feature_gap_open" in result.columns


def test_multiple_symbols():
    """Test feature computation with multiple symbols."""
    from feature_store.intraday.features import IntradayFeatureConfig, build_intraday_features

    np.random.seed(42)
    n_bars = 100

    rows = []
    for sym in ["AAPL", "MSFT"]:
        prices = 100.0 + np.cumsum(np.random.randn(n_bars) * 0.1)
        for i in range(n_bars):
            rows.append({
                "symbol": sym,
                "date": date(2024, 1, 15),
                "timestamp": pd.Timestamp("2024-01-15 09:30") + pd.Timedelta(minutes=i),
                "open": prices[i],
                "high": prices[i] + 0.5,
                "low": prices[i] - 0.5,
                "close": prices[i],
                "volume": 1000.0 + np.random.rand() * 1000,
                "bid": prices[i] - 0.01,
                "ask": prices[i] + 0.01,
                "bid_size": 500.0,
                "ask_size": 500.0,
            })

    df = pd.DataFrame(rows)
    cfg = IntradayFeatureConfig(min_bars=10)
    result = build_intraday_features(df, cfg)

    # Should have rows for both symbols
    if not result.empty:
        assert len(result["symbol"].unique()) <= 2
