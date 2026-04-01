from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path

import pytest

from data_layer.connectors.alpaca_connector import AlpacaConnector as LegacyAlpacaConnector
from data_layer.connectors.alpha_vantage_connector import AlphaVantageConnector as LegacyAlphaVantageConnector
from data_layer.connectors.base import ConnectorError
from data_layer.connectors.finnhub_connector import FinnhubConnector as LegacyFinnhubConnector
from data_layer.connectors.fred_connector import FREDConnector as LegacyFREDConnector
from data_layer.connectors.massive_connector import MassiveConnector as LegacyMassiveConnector
from trademl.connectors.alpaca import AlpacaConnector
from trademl.connectors.alpha_vantage import AlphaVantageConnector
from trademl.connectors.base import PermanentConnectorError
from trademl.connectors.finnhub import FinnhubConnector
from trademl.connectors.fmp import FMPConnector
from trademl.connectors.fred import FredConnector
from trademl.connectors.massive import MassiveConnector
from trademl.connectors.sec_edgar import SecEdgarConnector
from trademl.data_node.budgets import BudgetManager


def _load_dotenv() -> None:
    dotenv_path = Path(".env")
    if not dotenv_path.exists():
        return
    for line in dotenv_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key, value)
    if os.getenv("ALPACA_API_SECRET") and not os.getenv("ALPACA_SECRET_KEY"):
        os.environ["ALPACA_SECRET_KEY"] = os.environ["ALPACA_API_SECRET"]
    if os.getenv("ALPACA_DATA_BASE_URL") and not os.getenv("ALPACA_DATA_URL"):
        os.environ["ALPACA_DATA_URL"] = os.environ["ALPACA_DATA_BASE_URL"]


def _prev_business_day() -> date:
    day = date.today() - timedelta(days=1)
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return day


def _recent_window(days_back: int = 5) -> tuple[str, str]:
    end = _prev_business_day()
    start = end - timedelta(days=days_back)
    return start.isoformat(), end.isoformat()


def _require_env(*names: str | list[str]) -> None:
    flattened: list[str] = []
    for name in names:
        if isinstance(name, list):
            flattened.extend(name)
        else:
            flattened.append(name)
    missing = [name for name in flattened if not os.getenv(name)]
    if missing:
        pytest.skip(f"missing credentials: {', '.join(missing)}")


def _budgets() -> BudgetManager:
    return BudgetManager(
        {
            "alpaca": {"rpm": 50, "daily_cap": 500},
            "massive": {"rpm": 5, "daily_cap": 100},
            "finnhub": {"rpm": 50, "daily_cap": 500},
            "alpha_vantage": {"rpm": 4, "daily_cap": 100},
            "fred": {"rpm": 80, "daily_cap": 500},
            "fmp": {"rpm": 5, "daily_cap": 100},
            "sec_edgar": {"rpm": 5, "daily_cap": 100},
        }
    )


@pytest.mark.liveapi
def test_live_alpaca_bars() -> None:
    _load_dotenv()
    _require_env("ALPACA_API_KEY", "ALPACA_API_SECRET")
    end = _prev_business_day().isoformat()
    connector = AlpacaConnector(
        base_url=os.getenv("ALPACA_DATA_BASE_URL", "https://data.alpaca.markets"),
        api_key=os.environ["ALPACA_API_KEY"],
        secret_key=os.environ["ALPACA_API_SECRET"],
        budget_manager=_budgets(),
    )
    frame = connector.fetch("equities_eod", ["AAPL"], end, end)
    assert not frame.empty
    assert frame.iloc[0]["symbol"] == "AAPL"


@pytest.mark.liveapi
def test_live_massive_bars() -> None:
    _load_dotenv()
    _require_env("MASSIVE_API_KEY")
    end = _prev_business_day().isoformat()
    connector = MassiveConnector(
        base_url="https://api.polygon.io",
        api_key=os.environ["MASSIVE_API_KEY"],
        budget_manager=_budgets(),
    )
    frame = connector.fetch("equities_eod", ["AAPL"], end, end)
    assert not frame.empty
    assert frame.iloc[0]["symbol"] == "AAPL"


@pytest.mark.liveapi
def test_live_finnhub_reference_endpoints() -> None:
    _load_dotenv()
    _require_env("FINNHUB_API_KEY")
    start, end = _recent_window()
    connector = FinnhubConnector(
        base_url="https://finnhub.io",
        api_key=os.environ["FINNHUB_API_KEY"],
        budget_manager=_budgets(),
    )
    profile = connector.fetch("company_profile", ["AAPL"], start, end)
    earnings = connector.fetch("earnings_calendar", [], start, end)
    assert not profile.empty
    assert "symbol" in earnings.columns or "date" in earnings.columns


@pytest.mark.liveapi
def test_live_finnhub_bars_if_entitled() -> None:
    _load_dotenv()
    _require_env("FINNHUB_API_KEY")
    end = _prev_business_day().isoformat()
    connector = FinnhubConnector(
        base_url="https://finnhub.io",
        api_key=os.environ["FINNHUB_API_KEY"],
        budget_manager=_budgets(),
    )
    try:
        frame = connector.fetch("equities_eod", ["AAPL"], end, end)
    except PermanentConnectorError as exc:
        pytest.skip(f"Finnhub candle entitlement unavailable: {exc}")
    assert not frame.empty


@pytest.mark.liveapi
def test_live_alpha_vantage_listings() -> None:
    _load_dotenv()
    _require_env("ALPHA_VANTAGE_API_KEY")
    start, end = _recent_window()
    connector = AlphaVantageConnector(
        base_url="https://www.alphavantage.co",
        api_key=os.environ["ALPHA_VANTAGE_API_KEY"],
        budget_manager=_budgets(),
    )
    frame = connector.fetch("listings", [], start, end)
    assert not frame.empty
    assert "symbol" in frame.columns


@pytest.mark.liveapi
def test_live_fred_series() -> None:
    _load_dotenv()
    _require_env("FRED_API_KEY")
    start, end = _recent_window()
    connector = FredConnector(
        base_url="https://api.stlouisfed.org",
        api_key=os.environ["FRED_API_KEY"],
        budget_manager=_budgets(),
    )
    frame = connector.fetch("macros_treasury", ["DGS10"], start, end)
    assert not frame.empty
    assert frame.iloc[0]["series_id"] == "DGS10"


@pytest.mark.liveapi
def test_live_fmp_stable_endpoints() -> None:
    _load_dotenv()
    _require_env("FMP_API_KEY")
    start, end = _recent_window()
    connector = FMPConnector(
        base_url="https://financialmodelingprep.com",
        api_key=os.environ["FMP_API_KEY"],
        budget_manager=_budgets(),
    )
    delistings = connector.fetch("delistings", [], start, end)
    earnings = connector.fetch("earnings_calendar", [], start, end)
    assert not delistings.empty
    assert not earnings.empty


@pytest.mark.liveapi
def test_live_sec_edgar_filings() -> None:
    _load_dotenv()
    start, end = _recent_window(90)
    connector = SecEdgarConnector(
        base_url="https://data.sec.gov",
        user_agent=os.getenv("SEC_EDGAR_USER_AGENT", "TradeML/0.1 test@example.com"),
        budget_manager=_budgets(),
    )
    frame = connector.fetch("filing_index", ["320193"], start, end)
    assert not frame.empty
    assert set(frame["form"]).intersection({"8-K", "10-K", "10-Q"})


@pytest.mark.liveapi
def test_legacy_alpaca_daily_bars_endpoint() -> None:
    _load_dotenv()
    _require_env(["ALPACA_API_KEY", "ALPACA_SECRET_KEY"])
    try:
        connector = LegacyAlpacaConnector()
    except ConnectorError as exc:
        pytest.skip(f"legacy alpaca connector unavailable: {exc}")

    asof = _prev_business_day()
    frame = connector.fetch_bars(["AAPL"], start_date=asof, end_date=asof, timeframe="1Day")
    assert not frame.empty
    assert {"symbol", "open", "close", "ingested_at"}.issubset(frame.columns)


@pytest.mark.liveapi
def test_legacy_finnhub_option_chain_endpoint() -> None:
    _load_dotenv()
    _require_env(["FINNHUB_API_KEY"])
    try:
        connector = LegacyFinnhubConnector()
    except ConnectorError as exc:
        pytest.skip(f"legacy finnhub connector unavailable: {exc}")

    try:
        frame = connector.fetch_options_chain("AAPL")
    except ConnectorError as exc:
        pytest.skip(f"legacy finnhub option-chain entitlement unavailable: {exc}")
    assert not frame.empty
    assert {"underlier", "expiry", "strike", "cp_flag"}.issubset(frame.columns)


@pytest.mark.liveapi
def test_legacy_fred_treasury_curve_endpoint() -> None:
    _load_dotenv()
    _require_env(["FRED_API_KEY"])
    try:
        connector = LegacyFREDConnector()
    except ConnectorError as exc:
        pytest.skip(f"legacy fred connector unavailable: {exc}")

    day = _prev_business_day()
    frame = connector.fetch_treasury_curve(start_date=day, end_date=day)
    assert not frame.empty
    assert {"date"}.issubset(frame.columns)
    assert {"maturity", "rate"}.issubset(frame.columns) or {"tenor", "value"}.issubset(frame.columns)


@pytest.mark.liveapi
def test_legacy_alpha_vantage_listing_status_endpoint() -> None:
    _load_dotenv()
    _require_env(["ALPHA_VANTAGE_API_KEY"])
    try:
        connector = LegacyAlphaVantageConnector()
    except ConnectorError as exc:
        pytest.skip(f"legacy alpha vantage connector unavailable: {exc}")

    frame = connector.fetch_listing_status(state="active", date=_prev_business_day().isoformat())
    assert not frame.empty
    assert "symbol" in frame.columns


@pytest.mark.liveapi
def test_legacy_massive_day_bars_endpoint() -> None:
    _load_dotenv()
    _require_env(["MASSIVE_API_KEY"])
    try:
        connector = LegacyMassiveConnector()
    except ConnectorError as exc:
        pytest.skip(f"legacy massive connector unavailable: {exc}")

    asof = _prev_business_day()
    frame = connector.fetch_aggregates(symbol="AAPL", start_date=asof, end_date=asof, timespan="day")
    assert not frame.empty
    assert {"date", "symbol", "close"}.issubset(frame.columns)
