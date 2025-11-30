import os
from datetime import date, timedelta

import pandas as pd
import pytest

from data_layer.connectors.alpaca_connector import AlpacaConnector
from data_layer.connectors.polygon_connector import PolygonConnector
from data_layer.connectors.finnhub_connector import FinnhubConnector
from data_layer.connectors.fred_connector import FREDConnector
from data_layer.connectors.alpha_vantage_connector import AlphaVantageConnector
from data_layer.connectors.fmp_connector import FMPConnector
from data_layer.connectors.base import ConnectorError


def _require_env(*keys):
    missing = [k for k in keys if not os.getenv(k)]
    if missing:
        pytest.skip(f"Missing env vars: {', '.join(missing)}")


def _recent_weekday(days_back: int = 5) -> date:
    d = date.today() - timedelta(days=days_back)
    while d.weekday() >= 5:  # Saturday/Sunday
        d -= timedelta(days=1)
    return d


@pytest.mark.liveapi
def test_alpaca_fetch_bars_live():
    _require_env("ALPACA_API_KEY", "ALPACA_SECRET_KEY")
    conn = AlpacaConnector(
        api_key=os.getenv("ALPACA_API_KEY"),
        secret_key=os.getenv("ALPACA_SECRET_KEY"),
    )
    end = _recent_weekday(5)
    start = end - timedelta(days=5)
    df = conn.fetch_bars(["AAPL"], start, end, timeframe="1Day")
    assert not df.empty, "Alpaca bars returned empty"
    assert {"date", "symbol", "open", "high", "low", "close"}.issubset(df.columns)


@pytest.mark.liveapi
def test_polygon_fetch_aggregates_live():
    _require_env("POLYGON_API_KEY")
    conn = PolygonConnector(api_key=os.getenv("POLYGON_API_KEY"))
    end = _recent_weekday(5)
    start = end - timedelta(days=5)
    df = conn.fetch_aggregates("AAPL", start, end, timespan="day")
    assert not df.empty, "Polygon aggregates returned empty"
    assert {"date", "symbol", "open", "high", "low", "close", "volume"}.issubset(df.columns)


@pytest.mark.liveapi
def test_finnhub_fetch_candles_live():
    _require_env("FINNHUB_API_KEY")
    conn = FinnhubConnector(api_key=os.getenv("FINNHUB_API_KEY"))
    end = _recent_weekday(5)
    start = end - timedelta(days=5)
    try:
        df = conn.fetch_candle_daily("AAPL", start, end)
    except ConnectorError as e:
        msg = str(e)
        if "HTTP 401" in msg or "HTTP 403" in msg or "access" in msg.lower():
            pytest.skip(f"Finnhub daily candles not accessible with this key: {msg}")
        raise
    assert not df.empty, "Finnhub daily candles returned empty"
    assert {"date", "symbol", "open", "high", "low", "close", "volume"}.issubset(df.columns)


@pytest.mark.liveapi
def test_fred_fetch_series_live():
    _require_env("FRED_API_KEY")
    conn = FREDConnector(api_key=os.getenv("FRED_API_KEY"))
    end = date.today()
    start = end - timedelta(days=30)
    df = conn.fetch_series("DGS1MO", start_date=start, end_date=end)
    assert not df.empty, "FRED series returned empty"
    assert {"date", "value", "series_id"}.issubset(df.columns)


@pytest.mark.liveapi
def test_alpha_vantage_listing_status_live():
    _require_env("ALPHA_VANTAGE_API_KEY")
    conn = AlphaVantageConnector(api_key=os.getenv("ALPHA_VANTAGE_API_KEY"))
    df = conn.fetch_listing_status(state="active")
    assert not df.empty, "Alpha Vantage listing status returned empty"
    assert "symbol" in df.columns


@pytest.mark.liveapi
def test_fmp_eod_day_live():
    _require_env("FMP_API_KEY")
    conn = FMPConnector(api_key=os.getenv("FMP_API_KEY"))
    d = _recent_weekday(7)
    df = conn.fetch_eod_day("AAPL", d)
    assert not df.empty, "FMP EOD day returned empty"
    assert {"date", "open", "high", "low", "close", "volume"}.issubset(df.columns)
