from __future__ import annotations

import os
from datetime import date, timedelta

import pytest

from data_layer.connectors.alpaca_connector import AlpacaConnector
from data_layer.connectors.alpha_vantage_connector import AlphaVantageConnector
from data_layer.connectors.base import ConnectorError
from data_layer.connectors.finnhub_connector import FinnhubConnector
from data_layer.connectors.fred_connector import FREDConnector
from data_layer.connectors.massive_connector import MassiveConnector


def _prev_business_day() -> date:
    d = date.today() - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def _require_env(keys: list[str]):
    missing = [k for k in keys if not os.getenv(k)]
    if missing:
        pytest.skip(f"Missing credentials: {', '.join(missing)}")


@pytest.mark.liveapi
def test_alpaca_daily_bars_endpoint():
    _require_env(["ALPACA_API_KEY", "ALPACA_SECRET_KEY"])
    try:
        conn = AlpacaConnector()
    except ConnectorError as exc:
        pytest.skip(f"Alpaca connector unavailable: {exc}")

    asof = _prev_business_day()
    df = conn.fetch_bars(["AAPL"], start_date=asof, end_date=asof, timeframe="1Day")

    assert not df.empty
    assert {"symbol", "open", "close", "ingested_at"}.issubset(df.columns)


@pytest.mark.liveapi
def test_finnhub_option_chain_endpoint():
    _require_env(["FINNHUB_API_KEY"])
    try:
        conn = FinnhubConnector()
    except ConnectorError as exc:
        pytest.skip(f"Finnhub connector unavailable: {exc}")

    df = conn.fetch_options_chain("AAPL")
    assert not df.empty
    assert {"underlier", "expiry", "strike", "cp_flag"}.issubset(df.columns)


@pytest.mark.liveapi
def test_fred_treasury_curve_endpoint():
    _require_env(["FRED_API_KEY"])
    try:
        conn = FREDConnector()
    except ConnectorError as exc:
        pytest.skip(f"FRED connector unavailable: {exc}")

    day = _prev_business_day()
    df = conn.fetch_treasury_curve(start_date=day, end_date=day)

    assert not df.empty
    assert {"date", "maturity", "rate"}.issubset(df.columns)


@pytest.mark.liveapi
def test_alpha_vantage_listing_status_endpoint():
    _require_env(["ALPHA_VANTAGE_API_KEY"])
    try:
        conn = AlphaVantageConnector()
    except ConnectorError as exc:
        pytest.skip(f"Alpha Vantage connector unavailable: {exc}")

    day = _prev_business_day().isoformat()
    df = conn.fetch_listing_status(state="active", date=day)

    assert not df.empty
    assert "symbol" in df.columns


@pytest.mark.liveapi
def test_massive_day_bars_endpoint():
    _require_env(["MASSIVE_API_KEY"])
    try:
        conn = MassiveConnector()
    except ConnectorError as exc:
        pytest.skip(f"Massive connector unavailable: {exc}")

    asof = _prev_business_day()
    df = conn.fetch_aggregates(symbol="AAPL", start_date=asof, end_date=asof, timespan="day")

    assert not df.empty
    assert {"date", "symbol", "close"}.issubset(df.columns)
