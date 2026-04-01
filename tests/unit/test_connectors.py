from __future__ import annotations

import json

import pandas as pd
import pytest

from trademl.connectors.alpaca import AlpacaConnector
from trademl.connectors.alpha_vantage import AlphaVantageConnector
from trademl.connectors.base import BaseConnector, PermanentConnectorError, RetryConfig
from trademl.connectors.finnhub import FinnhubConnector
from trademl.connectors.fmp import FMPConnector
from trademl.connectors.fred import FredConnector
from trademl.connectors.massive import MassiveConnector
from trademl.connectors.sec_edgar import SecEdgarConnector
from trademl.data_node.budgets import BudgetManager


class FakeResponse:
    def __init__(self, status_code: int, payload: object, text: str | None = None) -> None:
        self.status_code = status_code
        self.payload = payload
        self._text = text

    def json(self) -> object:
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload

    @property
    def text(self) -> str:
        if self._text is not None:
            return self._text
        if isinstance(self.payload, (dict, list)):
            return json.dumps(self.payload)
        return str(self.payload)


class FakeSession:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, str, dict | None, dict | None]] = []

    def request(self, method: str, url: str, params: dict | None, headers: dict | None, timeout: int) -> FakeResponse:
        self.calls.append((method, url, params, headers))
        if not self.responses:
            raise AssertionError("no fake responses left")
        return self.responses.pop(0)


def _budget_manager() -> BudgetManager:
    return BudgetManager(
        {
            "alpaca": {"rpm": 100, "daily_cap": 1000},
            "massive": {"rpm": 100, "daily_cap": 1000},
            "finnhub": {"rpm": 100, "daily_cap": 1000},
            "alpha_vantage": {"rpm": 100, "daily_cap": 1000},
            "fred": {"rpm": 100, "daily_cap": 1000},
            "fmp": {"rpm": 100, "daily_cap": 1000},
            "sec_edgar": {"rpm": 100, "daily_cap": 1000},
        }
    )


def test_alpaca_connector_normalizes_bars() -> None:
    session = FakeSession(
        [
            FakeResponse(
                200,
                {
                    "bars": {
                        "AAPL": [{"t": "2026-01-02T00:00:00Z", "o": 1.0, "h": 2.0, "l": 0.5, "c": 1.5, "vw": 1.4, "v": 10, "n": 2}]
                    },
                    "next_page_token": None,
                },
            )
        ]
    )
    connector: BaseConnector = AlpacaConnector(
        base_url="https://data.alpaca.markets",
        api_key="key",
        budget_manager=_budget_manager(),
        session=session,
    )
    frame = connector.fetch("equities_eod", ["AAPL"], "2026-01-02", "2026-01-02")

    assert list(frame.columns[:6]) == ["date", "symbol", "open", "high", "low", "close"]
    assert frame.iloc[0]["symbol"] == "AAPL"
    assert frame.iloc[0]["volume"] == 10


def test_massive_connector_normalizes_bars() -> None:
    session = FakeSession([FakeResponse(200, {"results": [{"t": 1704153600000, "o": 10, "h": 11, "l": 9, "c": 10.5, "vw": 10.2, "v": 20, "n": 4}]})])
    connector = MassiveConnector(base_url="https://api.polygon.io", api_key="key", budget_manager=_budget_manager(), session=session)

    frame = connector.fetch("equities_eod", ["MSFT"], "2024-01-02", "2024-01-02")

    assert frame.iloc[0]["symbol"] == "MSFT"
    assert frame.iloc[0]["trade_count"] == 4


def test_finnhub_connector_normalizes_equities_and_earnings() -> None:
    session = FakeSession(
        [
            FakeResponse(200, {"s": "ok", "t": [1704153600], "o": [10], "h": [12], "l": [9], "c": [11], "v": [100]}),
            FakeResponse(200, {"earningsCalendar": [{"symbol": "AAPL", "date": "2026-01-29"}]}),
        ]
    )
    connector = FinnhubConnector(base_url="https://finnhub.io", api_key="key", budget_manager=_budget_manager(), session=session)

    bars = connector.fetch("equities_eod", ["AAPL"], "2024-01-02", "2024-01-02")
    earnings = connector.fetch("earnings_calendar", [], "2026-01-01", "2026-01-31")

    assert bars.iloc[0]["close"] == 11
    assert earnings.iloc[0]["symbol"] == "AAPL"


def test_alpha_vantage_and_fred_connectors() -> None:
    av_session = FakeSession(
        [
            FakeResponse(200, payload=[], text="symbol,name,exchange,assetType,ipoDate,delistingDate,status\nAAPL,Apple,NASDAQ,Stock,1980-12-12,,Active\n"),
            FakeResponse(200, {"data": [{"ex_dividend_date": "2024-01-05", "amount": "0.24"}]}),
            FakeResponse(200, {"data": [{"effective_date": "2024-01-10", "split_factor": "0.5"}]}),
        ]
    )
    fred_session = FakeSession(
        [FakeResponse(200, {"observations": [{"date": "2024-01-02", "value": "4.2", "realtime_start": "2024-01-02"}]})]
    )

    av = AlphaVantageConnector(base_url="https://www.alphavantage.co", api_key="key", budget_manager=_budget_manager(), session=av_session)
    fred = FredConnector(base_url="https://api.stlouisfed.org", api_key="key", budget_manager=_budget_manager(), session=fred_session)

    listings = av.fetch("listings", [], "2024-01-01", "2024-01-31")
    corp_actions = av.fetch("corp_actions", ["AAPL"], "2024-01-01", "2024-01-31")
    observations = fred.fetch("macros_treasury", ["DGS10"], "2024-01-01", "2024-01-31")

    assert listings.iloc[0]["symbol"] == "AAPL"
    assert set(corp_actions["event_type"]) == {"dividend", "split"}
    assert "amount" in corp_actions.columns
    assert observations.iloc[0]["series_id"] == "DGS10"
    assert observations.iloc[0]["value"] == pytest.approx(4.2)


def test_alpha_vantage_corp_actions_normalize_splits_and_dividends() -> None:
    av_session = FakeSession(
        [
            FakeResponse(
                200,
                {
                    "data": [
                        {"symbol": "AAPL", "ex_dividend_date": "2024-01-15", "amount": "0.24"},
                    ]
                },
            ),
            FakeResponse(
                200,
                {
                    "data": [
                        {"symbol": "AAPL", "effective_date": "2024-02-01", "split_factor": "0.5"},
                    ]
                },
            ),
        ]
    )
    av = AlphaVantageConnector(base_url="https://www.alphavantage.co", api_key="key", budget_manager=_budget_manager(), session=av_session)

    actions = av.fetch("corp_actions", ["AAPL"], "2024-01-01", "2024-03-01")

    assert set(actions["event_type"]) == {"dividend", "split"}
    assert set(actions["symbol"]) == {"AAPL"}
    assert set(actions.columns) >= {"symbol", "event_type", "ex_date", "ratio", "source"}


def test_fmp_and_sec_edgar_connectors() -> None:
    fmp_session = FakeSession([FakeResponse(200, [{"symbol": "XYZ", "delistedDate": "2024-01-05"}])])
    sec_session = FakeSession(
        [
            FakeResponse(
                200,
                {
                    "filings": {
                        "recent": {
                            "filingDate": ["2024-01-03", "2024-02-10"],
                            "form": ["8-K", "4"],
                            "accessionNumber": ["1", "2"],
                        }
                    }
                },
            )
        ]
    )

    fmp = FMPConnector(base_url="https://financialmodelingprep.com", api_key="key", budget_manager=_budget_manager(), session=fmp_session)
    sec = SecEdgarConnector(
        base_url="https://data.sec.gov",
        user_agent="TradeML/0.1 test@example.com",
        budget_manager=_budget_manager(),
        session=sec_session,
    )

    delistings = fmp.fetch("delistings", [], "2024-01-01", "2024-01-31")
    filings = sec.fetch("filing_index", ["320193"], "2024-01-01", "2024-01-31")

    assert delistings.iloc[0]["symbol"] == "XYZ"
    assert filings.iloc[0]["form"] == "8-K"


def test_retry_and_permanent_error_behavior() -> None:
    retry_session = FakeSession(
        [
            FakeResponse(429, {"error": "too many requests"}),
            FakeResponse(200, {"results": [{"t": 1704153600000, "o": 10, "h": 11, "l": 9, "c": 10.5, "vw": 10.2, "v": 20, "n": 4}]}),
        ]
    )
    connector = MassiveConnector(
        base_url="https://api.polygon.io",
        api_key="key",
        budget_manager=_budget_manager(),
        session=retry_session,
        retry_config=RetryConfig(max_attempts=2, base_delay_seconds=0.0, max_delay_seconds=0.0),
        sleep_fn=lambda _: None,
    )

    frame = connector.fetch("equities_eod", ["MSFT"], "2024-01-02", "2024-01-02")
    assert not frame.empty

    permanent_session = FakeSession([FakeResponse(403, {"error": "denied"}, text="NOT_ENTITLED")])
    permanent = MassiveConnector(base_url="https://api.polygon.io", api_key="key", budget_manager=_budget_manager(), session=permanent_session)

    with pytest.raises(PermanentConnectorError):
        permanent.fetch("equities_eod", ["MSFT"], "2024-01-02", "2024-01-02")
