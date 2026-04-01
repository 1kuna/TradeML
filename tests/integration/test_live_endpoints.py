from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path

import pandas as pd
import pytest

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


def _require_env(*names: str) -> None:
    missing = [name for name in names if not os.getenv(name)]
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


def _latest_session_date() -> str:
    return (pd.Timestamp.utcnow().normalize() - pd.offsets.BDay(1)).strftime("%Y-%m-%d")


def _recent_window(days: int = 5) -> tuple[str, str]:
    end = pd.Timestamp(_latest_session_date())
    start = (end - pd.offsets.BDay(days)).strftime("%Y-%m-%d")
    return start, end.strftime("%Y-%m-%d")


@pytest.mark.liveapi
def test_live_alpaca_bars() -> None:
    _load_dotenv()
    _require_env("ALPACA_API_KEY", "ALPACA_API_SECRET")
    session_date = _latest_session_date()
    connector = AlpacaConnector(
        base_url=os.getenv("ALPACA_DATA_BASE_URL", "https://data.alpaca.markets"),
        api_key=os.environ["ALPACA_API_KEY"],
        secret_key=os.environ["ALPACA_API_SECRET"],
        budget_manager=_budgets(),
    )
    frame = connector.fetch("equities_eod", ["AAPL"], session_date, session_date)
    assert not frame.empty
    assert frame.iloc[0]["symbol"] == "AAPL"


@pytest.mark.liveapi
def test_live_massive_bars() -> None:
    _load_dotenv()
    _require_env("MASSIVE_API_KEY")
    session_date = _latest_session_date()
    connector = MassiveConnector(
        base_url="https://api.polygon.io",
        api_key=os.environ["MASSIVE_API_KEY"],
        budget_manager=_budgets(),
    )
    try:
        frame = connector.fetch("equities_eod", ["AAPL"], session_date, session_date)
    except PermanentConnectorError as exc:
        pytest.skip(f"Massive timeframe entitlement unavailable: {exc}")
    assert not frame.empty
    assert frame.iloc[0]["symbol"] == "AAPL"


@pytest.mark.liveapi
def test_live_finnhub_reference_endpoints() -> None:
    _load_dotenv()
    _require_env("FINNHUB_API_KEY")
    start_date, end_date = _recent_window()
    connector = FinnhubConnector(
        base_url="https://finnhub.io",
        api_key=os.environ["FINNHUB_API_KEY"],
        budget_manager=_budgets(),
    )
    profile = connector.fetch("company_profile", ["AAPL"], start_date, end_date)
    earnings = connector.fetch("earnings_calendar", [], start_date, end_date)
    assert not profile.empty
    assert "symbol" in earnings.columns or "date" in earnings.columns


@pytest.mark.liveapi
def test_live_finnhub_bars_if_entitled() -> None:
    _load_dotenv()
    _require_env("FINNHUB_API_KEY")
    session_date = _latest_session_date()
    connector = FinnhubConnector(
        base_url="https://finnhub.io",
        api_key=os.environ["FINNHUB_API_KEY"],
        budget_manager=_budgets(),
    )
    try:
        frame = connector.fetch("equities_eod", ["AAPL"], session_date, session_date)
    except PermanentConnectorError as exc:
        pytest.skip(f"Finnhub candle entitlement unavailable: {exc}")
    assert not frame.empty


@pytest.mark.liveapi
def test_live_alpha_vantage_listings() -> None:
    _load_dotenv()
    _require_env("ALPHA_VANTAGE_API_KEY")
    start_date, end_date = _recent_window()
    connector = AlphaVantageConnector(
        base_url="https://www.alphavantage.co",
        api_key=os.environ["ALPHA_VANTAGE_API_KEY"],
        budget_manager=_budgets(),
    )
    frame = connector.fetch("listings", [], start_date, end_date)
    assert not frame.empty
    assert "symbol" in frame.columns


@pytest.mark.liveapi
def test_live_fred_series() -> None:
    _load_dotenv()
    _require_env("FRED_API_KEY")
    start_date, end_date = _recent_window()
    connector = FredConnector(
        base_url="https://api.stlouisfed.org",
        api_key=os.environ["FRED_API_KEY"],
        budget_manager=_budgets(),
    )
    frame = connector.fetch("macros_treasury", ["DGS10"], start_date, end_date)
    assert not frame.empty
    assert frame.iloc[0]["series_id"] == "DGS10"


@pytest.mark.liveapi
def test_live_fmp_stable_endpoints() -> None:
    _load_dotenv()
    _require_env("FMP_API_KEY")
    start_date, end_date = _recent_window()
    connector = FMPConnector(
        base_url="https://financialmodelingprep.com",
        api_key=os.environ["FMP_API_KEY"],
        budget_manager=_budgets(),
    )
    delistings = connector.fetch("delistings", [], start_date, end_date)
    earnings = connector.fetch("earnings_calendar", [], start_date, end_date)
    assert not delistings.empty
    assert not earnings.empty


@pytest.mark.liveapi
def test_live_sec_edgar_filings() -> None:
    _load_dotenv()
    end_date = _latest_session_date()
    start_date = (pd.Timestamp(end_date) - timedelta(days=90)).strftime("%Y-%m-%d")
    connector = SecEdgarConnector(
        base_url="https://data.sec.gov",
        user_agent=os.getenv("SEC_EDGAR_USER_AGENT", "TradeML/0.1 test@example.com"),
        budget_manager=_budgets(),
    )
    frame = connector.fetch("filing_index", ["320193"], start_date, end_date)
    assert not frame.empty
    assert set(frame["form"]).intersection({"8-K", "10-K", "10-Q"})
