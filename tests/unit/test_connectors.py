from __future__ import annotations

import json

import pandas as pd
import pytest
import requests

from trademl.connectors.alpaca import AlpacaConnector
from trademl.connectors.alpha_vantage import AlphaVantageConnector
from trademl.connectors.base import BaseConnector, PermanentConnectorError, RetryConfig, TemporaryConnectorError
from trademl.connectors.finnhub import FinnhubConnector
from trademl.connectors.fmp import FMPConnector
from trademl.connectors.fred import FredConnector
from trademl.connectors.massive import MassiveConnector
from trademl.connectors.sec_edgar import SecEdgarConnector
from trademl.connectors.tiingo import TiingoConnector
from trademl.connectors.twelve_data import TwelveDataConnector
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


class ErrorSession:
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.calls: list[tuple[str, str, dict | None, dict | None]] = []

    def request(self, method: str, url: str, params: dict | None, headers: dict | None, timeout: int) -> FakeResponse:
        self.calls.append((method, url, params, headers))
        raise self.error


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
            "tiingo": {"rpm": 100, "daily_cap": 1000},
            "twelve_data": {"rpm": 100, "daily_cap": 1000},
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
        trading_base_url="https://paper-api.alpaca.markets/v2",
        api_key="key",
        budget_manager=_budget_manager(),
        session=session,
    )
    frame = connector.fetch("equities_eod", ["AAPL"], "2026-01-02", "2026-01-02")

    assert list(frame.columns[:6]) == ["date", "symbol", "open", "high", "low", "close"]
    assert frame.iloc[0]["symbol"] == "AAPL"
    assert frame.iloc[0]["volume"] == 10


def test_alpaca_connector_normalizes_assets() -> None:
    session = FakeSession(
        [
            FakeResponse(
                200,
                [
                    {
                        "symbol": "AAPL",
                        "name": "Apple Inc.",
                        "exchange": "NASDAQ",
                        "status": "active",
                        "tradable": True,
                        "class": "us_equity",
                    }
                ],
            )
        ]
    )
    connector: BaseConnector = AlpacaConnector(
        base_url="https://data.alpaca.markets",
        trading_base_url="https://paper-api.alpaca.markets/v2",
        api_key="key",
        budget_manager=_budget_manager(),
        session=session,
    )

    frame = connector.fetch("assets", [], "2026-01-02", "2026-01-02")

    assert frame.iloc[0]["symbol"] == "AAPL"
    assert frame.iloc[0]["asset_class"] == "us_equity"
    assert bool(frame.iloc[0]["tradable"]) is True
    assert session.calls[0][1] == "https://paper-api.alpaca.markets/v2/assets"


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
            FakeResponse(200, payload=[], text="symbol,name,exchange,assetType,ipoDate,delistingDate,status\nOLD,Old Co,NASDAQ,Stock,1980-12-12,2024-01-15,Delisted\n"),
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

    assert set(listings["symbol"]) == {"AAPL", "OLD"}
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
    fmp_session = FakeSession(
        [
            FakeResponse(200, [{"symbol": "XYZ", "delistedDate": "2024-01-05"}]),
            FakeResponse(200, [{"oldSymbol": "FB", "newSymbol": "META", "date": "2022-06-09"}]),
        ]
    )
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
    symbol_changes = fmp.fetch("symbol_changes", [], "2024-01-01", "2024-01-31")
    filings = sec.fetch("filing_index", ["320193"], "2024-01-01", "2024-01-31")

    assert delistings.iloc[0]["symbol"] == "XYZ"
    assert symbol_changes.iloc[0]["newSymbol"] == "META"
    assert filings.iloc[0]["form"] == "8-K"


def test_tiingo_connector_normalizes_prices_actions_and_fundamentals() -> None:
    session = FakeSession(
        [
            FakeResponse(
                200,
                [
                    {
                        "date": "2024-01-02T00:00:00.000Z",
                        "open": 10.0,
                        "high": 11.0,
                        "low": 9.5,
                        "close": 10.5,
                        "adjClose": 10.4,
                        "adjVolume": 95,
                        "volume": 100,
                        "divCash": 0.12,
                        "splitFactor": 1.0,
                    }
                ],
            ),
            FakeResponse(
                200,
                [
                    {
                        "ticker": "AAPL",
                        "exDate": "2024-01-05",
                        "payDate": "2024-01-15",
                        "distribution": 0.24,
                    }
                ],
            ),
            FakeResponse(
                200,
                [
                    {
                        "ticker": "AAPL",
                        "exDate": "2024-02-01",
                        "splitFrom": 1,
                        "splitTo": 2,
                        "splitFactor": 0.5,
                    }
                ],
            ),
            FakeResponse(
                200,
                [{"ticker": "AAPL", "date": "2024-01-02", "marketCap": 1000000}],
            ),
            FakeResponse(
                200,
                [{"ticker": "AAPL", "date": "2023-12-31", "statementType": "annual", "revenue": 1000}],
            ),
        ]
    )
    connector = TiingoConnector(
        base_url="https://api.tiingo.com",
        api_key="key",
        budget_manager=_budget_manager(),
        session=session,
    )

    bars = connector.fetch("equities_eod", ["AAPL"], "2024-01-02", "2024-01-02")
    dividends = connector.fetch("corp_actions_dividends", ["AAPL"], "2024-01-01", "2024-01-31")
    splits = connector.fetch("corp_actions_splits", ["AAPL"], "2024-02-01", "2024-02-01")
    fundamentals = connector.fetch("fundamentals", ["AAPL"], "2024-01-01", "2024-01-31")

    assert bars.iloc[0]["symbol"] == "AAPL"
    assert bars.iloc[0]["adj_close"] == pytest.approx(10.4)
    assert dividends.iloc[0]["symbol"] == "AAPL"
    assert dividends.iloc[0]["event_type"] == "dividend"
    assert dividends.iloc[0]["amount"] == pytest.approx(0.24)
    assert splits.iloc[0]["ratio"] == pytest.approx(0.5)
    assert fundamentals.iloc[0]["symbol"] == "AAPL"
    assert set(fundamentals["statement_type"]) == {"daily", "statements"}
    assert session.calls[0][3]["Authorization"] == "Token key"


def test_twelve_data_connector_normalizes_prices_actions_and_statements() -> None:
    session = FakeSession(
        [
            FakeResponse(
                200,
                {
                    "meta": {"symbol": "AAPL", "interval": "1day"},
                    "values": [{"datetime": "2024-01-02", "open": "10", "high": "11", "low": "9", "close": "10.5", "volume": "100"}],
                },
            ),
            FakeResponse(
                200,
                {"meta": {"symbol": "AAPL"}, "dividends": [{"ex_date": "2024-01-05", "amount": "0.24", "payment_date": "2024-01-15"}]},
            ),
            FakeResponse(
                200,
                {"meta": {"symbol": "AAPL"}, "splits": [{"date": "2024-02-01", "ratio": "2:1"}]},
            ),
            FakeResponse(
                200,
                {"earnings": [{"symbol": "AAPL", "date": "2024-01-25", "eps_estimate": "2.10"}]},
            ),
            FakeResponse(
                200,
                {
                    "meta": {"symbol": "AAPL"},
                    "income_statement": [{"fiscal_date": "2023-12-31", "period": "annual", "revenue": "1000"}],
                },
            ),
            FakeResponse(
                200,
                {
                    "meta": {"symbol": "AAPL"},
                    "balance_sheet": [{"fiscal_date": "2023-12-31", "period": "annual", "total_assets": "2000"}],
                },
            ),
            FakeResponse(
                200,
                {
                    "meta": {"symbol": "AAPL"},
                    "cash_flow": [{"fiscal_date": "2023-12-31", "period": "annual", "operating_cash_flow": "500"}],
                },
            ),
        ]
    )
    connector = TwelveDataConnector(
        base_url="https://api.twelvedata.com",
        api_key="key",
        budget_manager=_budget_manager(),
        session=session,
    )

    bars = connector.fetch("equities_eod", ["AAPL"], "2024-01-02", "2024-01-02")
    dividends = connector.fetch("dividends", ["AAPL"], "2024-01-01", "2024-01-31")
    splits = connector.fetch("splits", ["AAPL"], "2024-02-01", "2024-02-01")
    earnings = connector.fetch("earnings_calendar", [], "2024-01-01", "2024-01-31")
    statements = connector.fetch("financial_statements", ["AAPL"], "2024-01-01", "2024-01-31")

    assert bars.iloc[0]["symbol"] == "AAPL"
    assert bars.iloc[0]["close"] == pytest.approx(10.5)
    assert dividends.iloc[0]["symbol"] == "AAPL"
    assert dividends.iloc[0]["amount"] == pytest.approx(0.24)
    assert splits.iloc[0]["ratio"] == pytest.approx(0.5)
    assert earnings.iloc[0]["symbol"] == "AAPL"
    assert set(statements["statement_type"]) == {"income_statement", "balance_sheet", "cash_flow"}
    assert session.calls[0][2]["apikey"] == "key"


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


def test_http_connector_wraps_request_exception_as_temporary_error() -> None:
    connector = MassiveConnector(
        base_url="https://api.polygon.io",
        api_key="key",
        budget_manager=_budget_manager(),
        session=ErrorSession(requests.ReadTimeout("timed out")),
        retry_config=RetryConfig(max_attempts=1, base_delay_seconds=0.0, max_delay_seconds=0.0),
        sleep_fn=lambda _: None,
    )

    with pytest.raises(TemporaryConnectorError):
        connector.fetch("equities_eod", ["MSFT"], "2024-01-02", "2024-01-02")


def test_http_connector_normalizes_exhausted_429_as_budget_error() -> None:
    connector = MassiveConnector(
        base_url="https://api.polygon.io",
        api_key="key",
        budget_manager=_budget_manager(),
        session=FakeSession([FakeResponse(429, {"error": "too many requests"})]),
        retry_config=RetryConfig(max_attempts=1, base_delay_seconds=0.0, max_delay_seconds=0.0),
        sleep_fn=lambda _: None,
    )

    with pytest.raises(TemporaryConnectorError, match="budget exhausted for vendor=massive"):
        connector.fetch("equities_eod", ["MSFT"], "2024-01-02", "2024-01-02")
