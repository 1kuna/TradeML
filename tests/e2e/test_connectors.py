"""
End-to-end tests for data connectors.

Tests actual API connectivity, response parsing, and schema validation.
Tests are skipped if required API keys are not available.

Run with: pytest tests/e2e/test_connectors.py -v
Or specific connector: pytest tests/e2e/test_connectors.py::TestAlpacaConnector -v
"""

from __future__ import annotations

import os
from datetime import date, timedelta

import pandas as pd
import pytest

# Skip all tests in this module if no API keys are configured
pytestmark = pytest.mark.e2e


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def recent_date_range():
    """Return a short, recent date range for testing."""
    end = date.today() - timedelta(days=1)  # Yesterday
    start = end - timedelta(days=5)
    return start, end


@pytest.fixture
def test_symbols():
    """Return commonly available test symbols."""
    return ["AAPL", "MSFT"]


# ============================================================================
# Alpaca Connector Tests
# ============================================================================

class TestAlpacaConnector:
    """E2E tests for Alpaca Markets connector."""

    @pytest.fixture
    def alpaca_connector(self):
        """Create Alpaca connector if credentials available."""
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")

        if not api_key or not secret_key:
            pytest.skip("ALPACA_API_KEY or ALPACA_SECRET_KEY not set")

        from data_layer.connectors.alpaca_connector import AlpacaConnector
        return AlpacaConnector(api_key=api_key, secret_key=secret_key)

    def test_fetch_bars_returns_data(self, alpaca_connector, test_symbols, recent_date_range):
        """Verify fetch_bars returns non-empty DataFrame with expected columns."""
        start, end = recent_date_range

        df = alpaca_connector.fetch_bars(
            symbols=test_symbols,
            start_date=start,
            end_date=end,
            timeframe="1Day",
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0, "Expected non-empty DataFrame"

        # Verify expected columns
        expected_cols = {"date", "symbol", "open", "high", "low", "close", "volume"}
        assert expected_cols.issubset(df.columns), f"Missing columns: {expected_cols - set(df.columns)}"

    def test_fetch_bars_schema_types(self, alpaca_connector, test_symbols, recent_date_range):
        """Verify returned data has correct types."""
        start, end = recent_date_range

        df = alpaca_connector.fetch_bars(
            symbols=test_symbols,
            start_date=start,
            end_date=end,
            timeframe="1Day",
        )

        if df.empty:
            pytest.skip("No data returned (market may have been closed)")

        # Check numeric columns
        assert df["open"].dtype in ["float64", "float32"]
        assert df["close"].dtype in ["float64", "float32"]
        assert df["volume"].dtype in ["int64", "float64"]

        # Check symbol is string
        assert df["symbol"].dtype == "object"

    def test_fetch_bars_metadata_added(self, alpaca_connector, test_symbols, recent_date_range):
        """Verify metadata columns are added by connector."""
        start, end = recent_date_range

        df = alpaca_connector.fetch_bars(
            symbols=test_symbols[:1],
            start_date=start,
            end_date=end,
            timeframe="1Day",
        )

        if df.empty:
            pytest.skip("No data returned")

        # Metadata columns from _add_metadata
        assert "ingested_at" in df.columns
        assert "source_name" in df.columns
        assert "source_uri" in df.columns
        assert df["source_name"].iloc[0] == "alpaca"

    def test_invalid_timeframe_raises(self, alpaca_connector, test_symbols, recent_date_range):
        """Verify invalid timeframe raises ConnectorError."""
        start, end = recent_date_range

        from data_layer.connectors.base import ConnectorError

        with pytest.raises(ConnectorError, match="Invalid timeframe"):
            alpaca_connector.fetch_bars(
                symbols=test_symbols,
                start_date=start,
                end_date=end,
                timeframe="InvalidTF",
            )

    def test_fetch_universe_returns_symbols(self, alpaca_connector):
        """Verify fetch_universe returns list of symbols."""
        symbols = alpaca_connector.fetch_universe()

        assert isinstance(symbols, list)
        assert len(symbols) > 0
        assert all(isinstance(s, str) for s in symbols)


# ============================================================================
# Massive Connector Tests
# ============================================================================

class TestMassiveConnector:
    """E2E tests for Massive.com connector."""

    @pytest.fixture
    def massive_connector(self):
        """Create Massive connector if credentials available."""
        api_key = os.getenv("MASSIVE_API_KEY")

        if not api_key:
            pytest.skip("MASSIVE_API_KEY not set")

        from data_layer.connectors.massive_connector import MassiveConnector
        return MassiveConnector(api_key=api_key)

    def test_fetch_aggregates_returns_data(self, massive_connector, recent_date_range):
        """Verify fetch_aggregates returns non-empty DataFrame."""
        start, end = recent_date_range

        df = massive_connector.fetch_aggregates(
            symbol="AAPL",
            start_date=start,
            end_date=end,
            timespan="day",
        )

        # May be empty if rate limited or no data
        if df.empty:
            pytest.skip("No data returned (rate limit or no data)")

        expected_cols = {"date", "symbol", "open", "high", "low", "close", "volume"}
        assert expected_cols.issubset(df.columns)

    def test_fetch_splits_returns_dataframe(self, massive_connector):
        """Verify fetch_splits returns DataFrame (even if empty)."""
        df = massive_connector.fetch_splits("AAPL")

        assert isinstance(df, pd.DataFrame)
        # AAPL has had splits, should have data
        if not df.empty:
            assert "symbol" in df.columns
            assert "event_type" in df.columns
            assert df["event_type"].iloc[0] == "split"

    def test_fetch_dividends_returns_dataframe(self, massive_connector):
        """Verify fetch_dividends returns DataFrame."""
        df = massive_connector.fetch_dividends("AAPL")

        assert isinstance(df, pd.DataFrame)
        # AAPL pays dividends
        if not df.empty:
            assert "amount" in df.columns
            assert df["event_type"].iloc[0] == "dividend"

    def test_market_status_returns_dict(self, massive_connector):
        """Verify market_status_now returns dict or None."""
        status = massive_connector.market_status_now()

        # May be None if rate limited
        if status is not None:
            assert isinstance(status, dict)

    def test_list_active_tickers(self, massive_connector):
        """Verify list_active_tickers returns tickers."""
        df, next_cursor = massive_connector.list_active_tickers(limit=10)

        if df.empty:
            pytest.skip("No tickers returned (rate limit)")

        assert "ticker" in df.columns
        assert len(df) <= 10


# ============================================================================
# FRED Connector Tests
# ============================================================================

class TestFREDConnector:
    """E2E tests for FRED (Federal Reserve) connector."""

    @pytest.fixture
    def fred_connector(self):
        """Create FRED connector if credentials available."""
        api_key = os.getenv("FRED_API_KEY")

        if not api_key:
            pytest.skip("FRED_API_KEY not set")

        from data_layer.connectors.fred_connector import FREDConnector
        return FREDConnector(api_key=api_key)

    def test_fetch_series_returns_data(self, fred_connector, recent_date_range):
        """Verify fetch_series returns time series data."""
        # Use a longer lookback for FRED (data may be sparse)
        end = date.today()
        start = end - timedelta(days=30)

        df = fred_connector.fetch_series(
            series_id="DGS10",  # 10-Year Treasury
            start_date=start,
            end_date=end,
        )

        assert isinstance(df, pd.DataFrame)
        if df.empty:
            pytest.skip("No data returned (weekend/holiday period)")

        assert "series_id" in df.columns
        assert "date" in df.columns
        assert "value" in df.columns
        assert df["series_id"].iloc[0] == "DGS10"

    def test_fetch_treasury_curve(self, fred_connector):
        """Verify fetch_treasury_curve returns multiple tenors."""
        end = date.today()
        start = end - timedelta(days=30)

        df = fred_connector.fetch_treasury_curve(
            tenors=["1y", "10y"],
            start_date=start,
            end_date=end,
        )

        if df.empty:
            pytest.skip("No data returned")

        assert "tenor" in df.columns
        assert set(df["tenor"].unique()).issubset({"1y", "10y"})

    def test_get_series_info(self, fred_connector):
        """Verify get_series_info returns metadata."""
        info = fred_connector.get_series_info("DGS10")

        assert isinstance(info, dict)
        assert "id" in info or "title" in info

    def test_unknown_series_raises(self, fred_connector):
        """Verify unknown series raises ConnectorError."""
        from data_layer.connectors.base import ConnectorError

        with pytest.raises(ConnectorError):
            fred_connector.fetch_series("INVALID_SERIES_12345")


# ============================================================================
# Alpha Vantage Connector Tests
# ============================================================================

class TestAlphaVantageConnector:
    """E2E tests for Alpha Vantage connector."""

    @pytest.fixture
    def av_connector(self):
        """Create Alpha Vantage connector if credentials available."""
        api_key = os.getenv("ALPHA_VANTAGE_API_KEY")

        if not api_key:
            pytest.skip("ALPHA_VANTAGE_API_KEY not set")

        from data_layer.connectors.alpha_vantage_connector import AlphaVantageConnector
        return AlphaVantageConnector(api_key=api_key)

    def test_fetch_daily_returns_data(self, av_connector):
        """Verify fetch_daily returns OHLCV data."""
        df = av_connector.fetch_daily("AAPL")

        # May be rate limited
        if df.empty:
            pytest.skip("No data returned (likely rate limited)")

        expected_cols = {"date", "symbol", "open", "high", "low", "close", "volume"}
        assert expected_cols.issubset(df.columns)

    def test_fetch_corporate_actions(self, av_connector):
        """Verify fetch_corporate_actions returns DataFrame."""
        df = av_connector.fetch_corporate_actions("AAPL")

        assert isinstance(df, pd.DataFrame)
        # AAPL should have corp actions
        if not df.empty:
            assert "symbol" in df.columns


# ============================================================================
# Finnhub Connector Tests
# ============================================================================

class TestFinnhubConnector:
    """E2E tests for Finnhub connector."""

    @pytest.fixture
    def finnhub_connector(self):
        """Create Finnhub connector if credentials available."""
        api_key = os.getenv("FINNHUB_API_KEY")

        if not api_key:
            pytest.skip("FINNHUB_API_KEY not set")

        from data_layer.connectors.finnhub_connector import FinnhubConnector
        return FinnhubConnector(api_key=api_key)

    def test_fetch_candle_daily_returns_data(self, finnhub_connector, recent_date_range):
        """Verify fetch_candle_daily returns OHLCV data."""
        start, end = recent_date_range

        df = finnhub_connector.fetch_candle_daily(
            symbol="AAPL",
            start_date=start,
            end_date=end,
        )

        if df.empty:
            pytest.skip("No data returned")

        expected_cols = {"date", "symbol", "open", "high", "low", "close", "volume"}
        assert expected_cols.issubset(df.columns)

    def test_fetch_company_profile(self, finnhub_connector):
        """Verify fetch_company_profile returns company info."""
        profile = finnhub_connector.fetch_company_profile("AAPL")

        if profile is None or not profile:
            pytest.skip("No profile returned (rate limit)")

        assert isinstance(profile, dict)


# ============================================================================
# FMP Connector Tests
# ============================================================================

class TestFMPConnector:
    """E2E tests for Financial Modeling Prep connector."""

    @pytest.fixture
    def fmp_connector(self):
        """Create FMP connector if credentials available."""
        api_key = os.getenv("FMP_API_KEY")

        if not api_key:
            pytest.skip("FMP_API_KEY not set")

        from data_layer.connectors.fmp_connector import FMPConnector
        return FMPConnector(api_key=api_key)

    def test_fetch_historical_price(self, fmp_connector):
        """Verify fetch_historical_price returns data."""
        df = fmp_connector.fetch_historical_price("AAPL")

        if df.empty:
            pytest.skip("No data returned")

        assert isinstance(df, pd.DataFrame)
        expected_cols = {"date", "symbol", "open", "high", "low", "close", "volume"}
        assert expected_cols.issubset(df.columns)

    def test_fetch_statements(self, fmp_connector):
        """Verify fetch_statements returns fundamental data."""
        df = fmp_connector.fetch_statements("AAPL", kind="income")

        if df.empty:
            pytest.skip("No data returned")

        assert isinstance(df, pd.DataFrame)
        # Should have revenue or net income
        has_fundamentals = any(col in df.columns for col in ["revenue", "netIncome", "grossProfit"])
        assert has_fundamentals, f"Missing fundamental columns, got: {df.columns.tolist()}"


# ============================================================================
# Cross-Connector Tests
# ============================================================================

class TestConnectorBase:
    """Test BaseConnector functionality."""

    def test_checksum_computation(self):
        """Verify checksum computation is deterministic."""
        from data_layer.connectors.base import BaseConnector

        data = b"test data for checksum"
        checksum1 = BaseConnector._compute_checksum(data)
        checksum2 = BaseConnector._compute_checksum(data)

        assert checksum1 == checksum2
        assert len(checksum1) == 64  # SHA256 hex digest length

    def test_checksum_different_data(self):
        """Verify different data produces different checksum."""
        from data_layer.connectors.base import BaseConnector

        checksum1 = BaseConnector._compute_checksum(b"data1")
        checksum2 = BaseConnector._compute_checksum(b"data2")

        assert checksum1 != checksum2


class TestConnectorErrorHandling:
    """Test error handling across connectors."""

    def test_alpaca_missing_credentials_raises(self):
        """Verify Alpaca raises error with missing credentials."""
        from data_layer.connectors.base import ConnectorError

        # Temporarily remove env vars
        old_key = os.environ.pop("ALPACA_API_KEY", None)
        old_secret = os.environ.pop("ALPACA_SECRET_KEY", None)

        try:
            from data_layer.connectors.alpaca_connector import AlpacaConnector
            with pytest.raises(ConnectorError, match="credentials not found"):
                AlpacaConnector()
        finally:
            # Restore
            if old_key:
                os.environ["ALPACA_API_KEY"] = old_key
            if old_secret:
                os.environ["ALPACA_SECRET_KEY"] = old_secret

    def test_polygon_missing_credentials_raises(self):
        """Verify Polygon raises error with missing credentials."""
        from data_layer.connectors.base import ConnectorError

        old_key = os.environ.pop("POLYGON_API_KEY", None)

        try:
            from data_layer.connectors.polygon_connector import PolygonConnector
            with pytest.raises(ConnectorError, match="not found"):
                PolygonConnector()
        finally:
            if old_key:
                os.environ["POLYGON_API_KEY"] = old_key

    def test_fred_missing_credentials_raises(self):
        """Verify FRED raises error with missing credentials."""
        from data_layer.connectors.base import ConnectorError

        old_key = os.environ.pop("FRED_API_KEY", None)

        try:
            from data_layer.connectors.fred_connector import FREDConnector
            with pytest.raises(ConnectorError, match="not found"):
                FREDConnector()
        finally:
            if old_key:
                os.environ["FRED_API_KEY"] = old_key
