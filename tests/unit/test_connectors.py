from __future__ import annotations

import json
from pathlib import Path

import pytest
import requests

import trademl.connectors.base as base_module
from trademl.connectors.alpaca import AlpacaConnector
from trademl.connectors.alpha_vantage import AlphaVantageConnector
from trademl.connectors.base import (
    BaseConnector,
    BudgetBlockedConnectorError,
    PermanentConnectorError,
    RemoteRateLimitConnectorError,
    RetryConfig,
    TemporaryConnectorError,
)
from trademl.connectors.finnhub import FinnhubConnector
from trademl.connectors.fmp import FMPConnector
from trademl.connectors.fred import FredConnector
from trademl.connectors.massive import MassiveConnector
from trademl.connectors.sec_edgar import SecEdgarConnector
from trademl.connectors.sec_edgar import MissingCompanyfactsError
from trademl.connectors.sec_edgar import accession_directory_index_json_url
from trademl.connectors.sec_edgar import accession_index_url
from trademl.connectors.sec_edgar import complete_txt_url_from_index_filename
from trademl.connectors.sec_edgar import complete_txt_url_from_parts
from trademl.connectors.sec_edgar import raw_primary_xml_url
from trademl.connectors.sec_edgar import sec_archive_dir
from trademl.connectors.tiingo import TiingoConnector
from trademl.connectors.twelve_data import TwelveDataConnector
from trademl.data_node.budgets import BudgetManager
from trademl.events.form4 import Form4ManifestRow


class FakeResponse:
    def __init__(
        self,
        status_code: int,
        payload: object,
        text: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self.payload = payload
        self._text = text
        self.headers = headers or {}

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

    def iter_content(self, chunk_size: int):
        raw = self.text.encode("utf-8")
        for offset in range(0, len(raw), chunk_size):
            yield raw[offset : offset + chunk_size]

    def close(self) -> None:
        return None


class FakeSession:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, str, dict | None, dict | None]] = []
        self.kwargs: list[dict[str, object]] = []

    def request(
        self,
        method: str,
        url: str,
        params: dict | None,
        headers: dict | None,
        timeout: int,
        **kwargs: object,
    ) -> FakeResponse:
        self.calls.append((method, url, params, headers))
        self.kwargs.append(dict(kwargs))
        if not self.responses:
            raise AssertionError("no fake responses left")
        return self.responses.pop(0)


class ErrorSession:
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.calls: list[tuple[str, str, dict | None, dict | None]] = []

    def request(
        self,
        method: str,
        url: str,
        params: dict | None,
        headers: dict | None,
        timeout: int,
        **kwargs: object,
    ) -> FakeResponse:
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
                        "AAPL": [
                            {
                                "t": "2026-01-02T00:00:00Z",
                                "o": 1.0,
                                "h": 2.0,
                                "l": 0.5,
                                "c": 1.5,
                                "vw": 1.4,
                                "v": 10,
                                "n": 2,
                            }
                        ]
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


def test_alpaca_connector_normalizes_minute_bars() -> None:
    session = FakeSession(
        [
            FakeResponse(
                200,
                {
                    "bars": {
                        "AAPL": [
                            {
                                "t": "2026-04-09T14:31:00Z",
                                "o": 100.0,
                                "h": 101.0,
                                "l": 99.5,
                                "c": 100.5,
                                "vw": 100.2,
                                "v": 50,
                                "n": 3,
                            }
                        ]
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

    frame = connector.fetch("equities_minute", ["AAPL"], "2026-04-09", "2026-04-09")

    assert frame.iloc[0]["symbol"] == "AAPL"
    assert frame.iloc[0]["volume"] == 50
    assert frame.iloc[0]["feed"] == "iex"
    assert str(frame.iloc[0]["date"]) == "2026-04-09"
    assert frame.iloc[0]["vendor_ts"].isoformat() == "2026-04-09T14:31:00+00:00"
    assert session.calls[0][2]["timeframe"] == "1Min"
    assert session.calls[0][2]["start"] == "2026-04-09T00:00:00Z"
    assert session.calls[0][2]["end"] == "2026-04-09T23:59:59Z"
    assert session.calls[0][2]["limit"] == 10000


def test_alpaca_news_requests_content_and_preserves_raw_payload() -> None:
    session = FakeSession(
        [
            FakeResponse(
                200,
                {
                    "news": [
                        {
                            "id": 123,
                            "created_at": "2026-04-29T12:00:00Z",
                            "updated_at": "2026-04-29T12:01:00Z",
                            "headline": "AAPL news",
                            "summary": "summary",
                            "content": "full article text",
                            "url": "https://example.com/a",
                            "symbols": ["AAPL", "BTC/USD"],
                            "source": "Benzinga",
                        }
                    ],
                    "next_page_token": None,
                },
            )
        ]
    )
    connector = AlpacaConnector(
        base_url="https://data.alpaca.markets",
        trading_base_url="https://paper-api.alpaca.markets/v2",
        api_key="key",
        secret_key="secret",
        budget_manager=_budget_manager(),
        session=session,
    )

    frame = connector.fetch("news", ["AAPL"], "2026-04-29", "2026-04-29")

    assert session.calls[0][2]["include_content"] == "true"
    assert frame.iloc[0]["content"] == "full article text"
    assert frame.iloc[0]["news_id"] == "123"
    assert frame.iloc[0]["raw_payload_hash"]
    assert set(frame.iloc[0]["related_symbols"]) == {"AAPL", "BTC/USD"}


def test_alpaca_connector_normalizes_stock_trades_quotes_and_snapshots() -> None:
    session = FakeSession(
        [
            FakeResponse(200, {"trades": {"AAPL": [{"t": "2026-04-29T14:30:00Z", "p": 190.1, "s": 10, "x": "V", "c": ["@"], "i": 1}]}}),
            FakeResponse(200, {"quotes": {"AAPL": [{"t": "2026-04-29T14:30:01Z", "bp": 190.0, "bs": 2, "ap": 190.2, "as": 3, "bx": "V", "ax": "V", "c": ["R"]}]}}),
            FakeResponse(
                200,
                {
                    "snapshots": {
                        "AAPL": {
                            "latestTrade": {"t": "2026-04-29T14:30:02Z", "p": 190.3, "s": 1},
                            "latestQuote": {"bp": 190.2, "bs": 2, "ap": 190.4, "as": 3},
                            "minuteBar": {"o": 190.0, "h": 190.5, "l": 190.0, "c": 190.3, "v": 100},
                        }
                    }
                },
            ),
        ]
    )
    connector = AlpacaConnector(
        base_url="https://data.alpaca.markets",
        trading_base_url="https://paper-api.alpaca.markets/v2",
        api_key="key",
        secret_key="secret",
        budget_manager=_budget_manager(),
        session=session,
    )

    trades = connector.fetch("stock_trades", ["AAPL"], "2026-04-29", "2026-04-29")
    quotes = connector.fetch("stock_quotes", ["AAPL"], "2026-04-29", "2026-04-29")
    snapshots = connector.fetch("stock_snapshots", ["AAPL"], "2026-04-29", "2026-04-29")

    assert trades.iloc[0]["event_type"] == "trade"
    assert trades.iloc[0]["price"] == 190.1
    assert quotes.iloc[0]["event_type"] == "quote"
    assert quotes.iloc[0]["bid_price"] == 190.0
    assert snapshots.iloc[0]["latest_trade_price"] == 190.3
    assert snapshots.iloc[0]["asset_class"] == "us_equity"


def test_alpaca_audit_sample_bounds_pagination_for_high_volume_lanes() -> None:
    session = FakeSession(
        [
            FakeResponse(
                200,
                {
                    "trades": {"AAPL": [{"t": "2026-04-29T14:30:00Z", "p": 190.1, "s": 10}]},
                    "next_page_token": "next",
                },
            )
        ]
    )
    connector = AlpacaConnector(
        base_url="https://data.alpaca.markets",
        trading_base_url="https://paper-api.alpaca.markets/v2",
        api_key="key",
        secret_key="secret",
        budget_manager=_budget_manager(),
        session=session,
    )

    frame = connector.fetch_audit_sample("stock_trades", ["AAPL"], "2026-04-29", "2026-04-29")

    assert len(frame) == 1
    assert session.calls[0][2]["limit"] == 100
    assert session.responses == []


def test_alpaca_connector_normalizes_crypto_and_option_archives() -> None:
    session = FakeSession(
        [
            FakeResponse(200, {"bars": {"BTC/USD": [{"t": "2026-04-29T14:30:00Z", "o": 1, "h": 2, "l": 1, "c": 2, "v": 3, "n": 4, "vw": 1.5}]}}),
            FakeResponse(200, {"quotes": {"BTC/USD": [{"t": "2026-04-29T14:30:01Z", "bp": 1.0, "bs": 2.0, "ap": 1.1, "as": 3.0}]}}),
            FakeResponse(200, {"snapshots": {"BTC/USD": {"latestQuote": {"t": "2026-04-29T14:30:02Z", "bp": 1.0, "ap": 1.1}}}}),
            FakeResponse(
                200,
                {
                    "snapshots": {
                        "SPY260116C00500000": {
                            "latestTrade": {"t": "2026-04-29T14:30:00Z", "p": 10.0, "s": 1},
                            "greeks": {"delta": 0.5},
                            "impliedVolatility": 0.2,
                        }
                    }
                },
            ),
        ]
    )
    connector = AlpacaConnector(
        base_url="https://data.alpaca.markets",
        trading_base_url="https://paper-api.alpaca.markets/v2",
        api_key="key",
        secret_key="secret",
        budget_manager=_budget_manager(),
        session=session,
    )

    bars = connector.fetch("crypto_bars", ["BTC/USD"], "2026-04-29", "2026-04-29")
    quotes = connector.fetch("crypto_quotes", ["BTC/USD"], "2026-04-29", "2026-04-29")
    crypto_snapshot = connector.fetch("crypto_snapshots", ["BTC/USD"], "2026-04-29", "2026-04-29")
    chain = connector.fetch("option_chain_reference", ["SPY"], "2026-04-29", "2026-04-29")

    assert bars.iloc[0]["feed"] == "alpaca_crypto_us"
    assert quotes.iloc[0]["asset_class"] == "crypto"
    assert crypto_snapshot.iloc[0]["asset_class"] == "crypto"
    assert chain.iloc[0]["underlying_symbol"] == "SPY"
    assert bool(chain.iloc[0]["indicative"]) is True
    assert bool(chain.iloc[0]["not_live_trade_approved"]) is True


def test_alpaca_connector_normalizes_news() -> None:
    session = FakeSession(
        [
            FakeResponse(
                200,
                {
                    "news": [
                        {
                            "id": 123,
                            "created_at": "2026-04-09T13:30:00Z",
                            "updated_at": "2026-04-09T13:35:00Z",
                            "headline": "Apple news",
                            "summary": "Summary",
                            "url": "https://example.com/apple",
                            "source": "ExampleWire",
                            "symbols": ["AAPL", "MSFT"],
                            "images": [{"url": "https://example.com/apple.png"}],
                        }
                    ],
                    "next_page_token": None,
                },
            )
        ]
    )
    connector = AlpacaConnector(
        base_url="https://data.alpaca.markets",
        trading_base_url="https://paper-api.alpaca.markets/v2",
        api_key="key",
        secret_key="secret",
        budget_manager=_budget_manager(),
        session=session,
    )

    frame = connector.fetch("news", ["AAPL", "MSFT"], "2026-04-09", "2026-04-09")

    assert frame.iloc[0]["symbol"] == "AAPL"
    assert frame.iloc[0]["related_symbols"] == ("AAPL", "MSFT")
    assert frame.iloc[0]["headline"] == "Apple news"
    assert session.calls[0][1] == "https://data.alpaca.markets/v1beta1/news"
    assert session.calls[0][2]["symbols"] == "AAPL,MSFT"


def test_twelve_data_batch_fetch_records_weighted_request_telemetry() -> None:
    session = FakeSession(
        [
            FakeResponse(
                200,
                {
                    "AAPL": {
                        "meta": {"symbol": "AAPL"},
                        "values": [
                            {
                                "datetime": "2026-01-02",
                                "open": "1",
                                "high": "2",
                                "low": "0.5",
                                "close": "1.5",
                                "volume": "10",
                            }
                        ],
                    },
                    "MSFT": {
                        "meta": {"symbol": "MSFT"},
                        "values": [
                            {
                                "datetime": "2026-01-02",
                                "open": "3",
                                "high": "4",
                                "low": "2.5",
                                "close": "3.5",
                                "volume": "11",
                            }
                        ],
                    },
                },
            )
        ]
    )
    budget_manager = _budget_manager()
    connector = TwelveDataConnector(
        base_url="https://api.twelvedata.com",
        api_key="key",
        budget_manager=budget_manager,
        session=session,
    )

    frame = connector.fetch(
        "equities_eod", ["AAPL", "MSFT"], "2026-01-02", "2026-01-02"
    )

    snapshot = budget_manager.snapshot()
    telemetry = snapshot["vendors"]["twelve_data"]["telemetry"]
    assert len(frame) == 2
    assert telemetry["totals"]["outbound_requests"] == 1
    assert telemetry["totals"]["logical_units"] == 2
    assert telemetry["totals"]["request_cost_units"] == 2


def test_twelve_data_connector_normalizes_minute_bars_with_weighted_credits() -> None:
    session = FakeSession(
        [
            FakeResponse(
                200,
                {
                    "meta": {"symbol": "AAPL"},
                    "values": [
                        {
                            "datetime": "2026-04-09 09:31:00",
                            "open": "100",
                            "high": "101",
                            "low": "99",
                            "close": "100.5",
                            "volume": "2500",
                        }
                    ],
                },
            )
        ]
    )
    budget_manager = _budget_manager()
    connector = TwelveDataConnector(
        base_url="https://api.twelvedata.com",
        api_key="key",
        budget_manager=budget_manager,
        session=session,
    )

    frame = connector.fetch("equities_minute", ["AAPL"], "2026-04-09", "2026-04-09")

    assert frame.iloc[0]["symbol"] == "AAPL"
    assert str(frame.iloc[0]["date"]) == "2026-04-09"
    assert frame.iloc[0]["vendor_ts"].isoformat() == "2026-04-09T09:31:00+00:00"
    assert session.calls[0][2]["interval"] == "1min"
    assert session.calls[0][2]["outputsize"] == 5000
    telemetry = budget_manager.snapshot()["vendors"]["twelve_data"]["telemetry"]
    assert telemetry["totals"]["request_cost_units"] == 1


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
    session = FakeSession(
        [
            FakeResponse(
                200,
                {
                    "results": [
                        {
                            "t": 1704153600000,
                            "o": 10,
                            "h": 11,
                            "l": 9,
                            "c": 10.5,
                            "vw": 10.2,
                            "v": 20,
                            "n": 4,
                        }
                    ]
                },
            )
        ]
    )
    connector = MassiveConnector(
        base_url="https://api.polygon.io",
        api_key="key",
        budget_manager=_budget_manager(),
        session=session,
    )

    frame = connector.fetch("equities_eod", ["MSFT"], "2024-01-02", "2024-01-02")

    assert frame.iloc[0]["symbol"] == "MSFT"
    assert frame.iloc[0]["trade_count"] == 4


def test_massive_connector_paginates_bars_via_next_url() -> None:
    session = FakeSession(
        [
            FakeResponse(
                200,
                {
                    "results": [
                        {
                            "t": 1704153600000,
                            "o": 10,
                            "h": 11,
                            "l": 9,
                            "c": 10.5,
                            "vw": 10.2,
                            "v": 20,
                            "n": 4,
                        }
                    ],
                    "next_url": "https://api.massive.com/v2/aggs/ticker/MSFT/range/1/day/2024-01-02/2024-01-03?cursor=abc",
                },
            ),
            FakeResponse(
                200,
                {
                    "results": [
                        {
                            "t": 1704240000000,
                            "o": 11,
                            "h": 12,
                            "l": 10,
                            "c": 11.5,
                            "vw": 11.2,
                            "v": 21,
                            "n": 5,
                        }
                    ],
                },
            ),
        ]
    )
    connector = MassiveConnector(
        base_url="https://api.polygon.io",
        api_key="key",
        budget_manager=_budget_manager(),
        session=session,
    )

    frame = connector.fetch("equities_eod", ["MSFT"], "2024-01-02", "2024-01-03")

    assert frame["date"].astype(str).tolist() == ["2024-01-02", "2024-01-03"]
    assert (
        session.calls[1][1]
        == "https://api.massive.com/v2/aggs/ticker/MSFT/range/1/day/2024-01-02/2024-01-03?cursor=abc"
    )


def test_massive_connector_normalizes_minute_aggregates() -> None:
    session = FakeSession(
        [
            FakeResponse(
                200,
                {
                    "results": [
                        {
                            "t": 1775745060000,
                            "o": 100,
                            "h": 101,
                            "l": 99,
                            "c": 100.5,
                            "vw": 100.2,
                            "v": 20,
                            "n": 4,
                        }
                    ]
                },
            )
        ]
    )
    connector = MassiveConnector(
        base_url="https://api.polygon.io",
        api_key="key",
        budget_manager=_budget_manager(),
        session=session,
    )

    frame = connector.fetch("equities_minute", ["AAPL"], "2026-04-09", "2026-04-09")

    assert frame.iloc[0]["symbol"] == "AAPL"
    assert frame.iloc[0]["source_name"] == "massive"
    assert "/range/1/minute/" in session.calls[0][1]
    assert session.calls[0][2]["limit"] == 50000


def test_massive_connector_paginates_reference_tickers_via_next_url() -> None:
    session = FakeSession(
        [
            FakeResponse(
                200,
                {
                    "results": [{"ticker": "AAPL"}],
                    "next_url": "https://api.massive.com/v3/reference/tickers?cursor=abc",
                },
            ),
            FakeResponse(200, {"results": [{"ticker": "MSFT"}]}),
        ]
    )
    connector = MassiveConnector(
        base_url="https://api.polygon.io",
        api_key="key",
        budget_manager=_budget_manager(),
        session=session,
    )

    frame = connector.fetch("reference_tickers", [], "2024-01-02", "2024-01-03")

    assert frame["ticker"].tolist() == ["AAPL", "MSFT"]
    assert (
        session.calls[1][1] == "https://api.massive.com/v3/reference/tickers?cursor=abc"
    )


def test_finnhub_connector_normalizes_equities_and_earnings() -> None:
    session = FakeSession(
        [
            FakeResponse(
                200,
                {
                    "s": "ok",
                    "t": [1704153600],
                    "o": [10],
                    "h": [12],
                    "l": [9],
                    "c": [11],
                    "v": [100],
                },
            ),
            FakeResponse(
                200, {"earningsCalendar": [{"symbol": "AAPL", "date": "2026-01-29"}]}
            ),
        ]
    )
    connector = FinnhubConnector(
        base_url="https://finnhub.io",
        api_key="key",
        budget_manager=_budget_manager(),
        session=session,
    )

    bars = connector.fetch("equities_eod", ["AAPL"], "2024-01-02", "2024-01-02")
    earnings = connector.fetch("earnings_calendar", [], "2026-01-01", "2026-01-31")

    assert bars.iloc[0]["close"] == 11
    assert earnings.iloc[0]["symbol"] == "AAPL"


def test_tiingo_connector_normalizes_company_news() -> None:
    session = FakeSession(
        [
            FakeResponse(
                200,
                [
                    {
                        "id": 42,
                        "publishedDate": "2026-04-09T13:30:00Z",
                        "crawlDate": "2026-04-09T13:35:00Z",
                        "title": "Apple launches something",
                        "description": "Headline summary",
                        "url": "https://example.com/apple",
                        "image": "https://example.com/apple.png",
                        "source": "ExampleWire",
                        "tickers": ["AAPL", "MSFT"],
                        "tags": ["technology", "hardware"],
                    }
                ],
            )
        ]
    )
    connector = TiingoConnector(
        base_url="https://api.tiingo.com",
        api_key="key",
        budget_manager=_budget_manager(),
        session=session,
    )

    frame = connector.fetch("news", ["AAPL", "MSFT"], "2026-04-09", "2026-04-09")

    assert frame.iloc[0]["symbol"] == "AAPL"
    assert frame.iloc[0]["related_symbols"] == ("AAPL", "MSFT")
    assert frame.iloc[0]["tags"] == ("HARDWARE", "TECHNOLOGY")
    assert frame.iloc[0]["headline"] == "Apple launches something"
    assert session.calls[0][2]["tickers"] == "AAPL,MSFT"
    assert session.calls[0][2]["startDate"] == "2026-04-09"
    assert session.calls[0][2]["endDate"] == "2026-04-09"


def test_finnhub_connector_normalizes_company_news() -> None:
    session = FakeSession(
        [
            FakeResponse(
                200,
                [
                    {
                        "id": 7,
                        "datetime": 1775741400,
                        "headline": "Apple supplier update",
                        "summary": "Supplier summary",
                        "url": "https://example.com/supplier",
                        "image": "https://example.com/supplier.png",
                        "category": "company",
                        "source": "ExampleWire",
                        "related": "AAPL,TSM",
                    }
                ],
            )
        ]
    )
    connector = FinnhubConnector(
        base_url="https://finnhub.io",
        api_key="key",
        budget_manager=_budget_manager(),
        session=session,
    )

    frame = connector.fetch("company_news", ["AAPL"], "2026-04-08", "2026-04-09")

    assert frame.iloc[0]["symbol"] == "AAPL"
    assert frame.iloc[0]["related_symbols"] == ("AAPL", "TSM")
    assert frame.iloc[0]["category"] == "company"
    assert session.calls[0][2]["from"] == "2026-04-08"
    assert session.calls[0][2]["to"] == "2026-04-09"


def test_alpha_vantage_connector_normalizes_news_sentiment() -> None:
    session = FakeSession(
        [
            FakeResponse(
                200,
                {
                    "feed": [
                        {
                            "title": "Apple sentiment",
                            "url": "https://example.com/av",
                            "time_published": "20260409T133000",
                            "summary": "AV summary",
                            "source": "ExampleWire",
                            "banner_image": "https://example.com/av.png",
                            "ticker_sentiment": [
                                {"ticker": "AAPL"},
                                {"ticker": "MSFT"},
                            ],
                            "topics": [{"topic": "Technology"}],
                        }
                    ]
                },
            )
        ]
    )
    connector = AlphaVantageConnector(
        base_url="https://www.alphavantage.co",
        api_key="key",
        budget_manager=_budget_manager(),
        session=session,
    )

    frame = connector.fetch(
        "news_sentiment", ["AAPL", "MSFT"], "2026-04-09", "2026-04-09"
    )

    assert frame.iloc[0]["symbol"] == "AAPL"
    assert frame.iloc[0]["related_symbols"] == ("AAPL", "MSFT")
    assert frame.iloc[0]["tags"] == ("TECHNOLOGY",)
    assert session.calls[0][2]["function"] == "NEWS_SENTIMENT"
    assert session.calls[0][2]["limit"] == 1000


def test_fmp_connector_normalizes_stock_news() -> None:
    session = FakeSession(
        [
            FakeResponse(
                200,
                [
                    {
                        "publishedDate": "2026-04-09T13:30:00Z",
                        "title": "Apple FMP news",
                        "text": "FMP summary",
                        "url": "https://example.com/fmp",
                        "image": "https://example.com/fmp.png",
                        "site": "ExampleWire",
                        "symbols": "AAPL,MSFT",
                    }
                ],
            )
        ]
    )
    connector = FMPConnector(
        base_url="https://financialmodelingprep.com",
        api_key="key",
        budget_manager=_budget_manager(),
        session=session,
    )

    frame = connector.fetch("stock_news", ["AAPL", "MSFT"], "2026-04-09", "2026-04-09")

    assert frame.iloc[0]["symbol"] == "AAPL"
    assert frame.iloc[0]["related_symbols"] == ("AAPL", "MSFT")
    assert frame.iloc[0]["headline"] == "Apple FMP news"
    assert session.calls[0][1] == "https://financialmodelingprep.com/stable/news/stock"
    assert session.calls[0][2]["symbols"] == "AAPL,MSFT"


def test_alpha_vantage_and_fred_connectors() -> None:
    av_session = FakeSession(
        [
            FakeResponse(
                200,
                payload=[],
                text="symbol,name,exchange,assetType,ipoDate,delistingDate,status\nAAPL,Apple,NASDAQ,Stock,1980-12-12,,Active\n",
            ),
            FakeResponse(
                200,
                payload=[],
                text="symbol,name,exchange,assetType,ipoDate,delistingDate,status\nOLD,Old Co,NASDAQ,Stock,1980-12-12,2024-01-15,Delisted\n",
            ),
            FakeResponse(
                200, {"data": [{"ex_dividend_date": "2024-01-05", "amount": "0.24"}]}
            ),
            FakeResponse(
                200, {"data": [{"effective_date": "2024-01-10", "split_factor": "0.5"}]}
            ),
        ]
    )
    fred_session = FakeSession(
        [
            FakeResponse(
                200,
                {
                    "observations": [
                        {
                            "date": "2024-01-02",
                            "value": "4.2",
                            "realtime_start": "2024-01-02",
                        }
                    ]
                },
            )
        ]
    )

    av = AlphaVantageConnector(
        base_url="https://www.alphavantage.co",
        api_key="key",
        budget_manager=_budget_manager(),
        session=av_session,
    )
    fred = FredConnector(
        base_url="https://api.stlouisfed.org",
        api_key="key",
        budget_manager=_budget_manager(),
        session=fred_session,
    )

    listings = av.fetch("listings", [], "2024-01-01", "2024-01-31")
    corp_actions = av.fetch("corp_actions", ["AAPL"], "2024-01-01", "2024-01-31")
    observations = fred.fetch("macros_treasury", ["DGS10"], "2024-01-01", "2024-01-31")

    assert set(listings["symbol"]) == {"AAPL", "OLD"}
    assert set(corp_actions["event_type"]) == {"dividend", "split"}
    assert "amount" in corp_actions.columns
    assert observations.iloc[0]["series_id"] == "DGS10"
    assert observations.iloc[0]["value"] == pytest.approx(4.2)
    assert fred_session.calls[0][2]["limit"] == 10000
    assert fred_session.calls[0][2]["offset"] == 0


def test_fred_vintagedates_uses_supported_limit() -> None:
    fred_session = FakeSession(
        [FakeResponse(200, {"vintage_dates": ["2024-01-02", "2024-01-03"]})]
    )
    fred = FredConnector(
        base_url="https://api.stlouisfed.org",
        api_key="key",
        budget_manager=_budget_manager(),
        session=fred_session,
    )

    vintages = fred.fetch("vintagedates", ["DGS10"], "2024-01-01", "2024-01-31")

    assert vintages["series_id"].tolist() == ["DGS10", "DGS10"]
    assert fred_session.calls[0][2]["limit"] == 10000
    assert fred_session.calls[0][2]["offset"] == 0


def test_fred_observations_paginates_at_supported_limit() -> None:
    fred_session = FakeSession(
        [
            FakeResponse(
                200,
                {
                    "observations": [
                        {
                            "date": f"2024-01-{(index % 28) + 1:02d}",
                            "value": "4.2",
                            "realtime_start": "2024-01-01",
                        }
                        for index in range(10000)
                    ]
                },
            ),
            FakeResponse(
                200,
                {
                    "observations": [
                        {
                            "date": "2024-02-01",
                            "value": "4.3",
                            "realtime_start": "2024-02-01",
                        }
                    ]
                },
            ),
        ]
    )
    fred = FredConnector(
        base_url="https://api.stlouisfed.org",
        api_key="key",
        budget_manager=_budget_manager(),
        session=fred_session,
    )

    observations = fred.fetch("macros_treasury", ["DGS10"], "2024-01-01", "2024-02-01")

    assert len(observations) == 10001
    assert [call[2]["offset"] for call in fred_session.calls] == [0, 10000]
    assert {call[2]["limit"] for call in fred_session.calls} == {10000}


def test_fred_vintagedates_paginates_at_supported_limit() -> None:
    fred_session = FakeSession(
        [
            FakeResponse(200, {"vintage_dates": ["2024-01-01"] * 10000}),
            FakeResponse(200, {"vintage_dates": ["2024-02-01"]}),
        ]
    )
    fred = FredConnector(
        base_url="https://api.stlouisfed.org",
        api_key="key",
        budget_manager=_budget_manager(),
        session=fred_session,
    )

    vintages = fred.fetch("vintagedates", ["DGS10"], "2024-01-01", "2024-02-01")

    assert len(vintages) == 10001
    assert [call[2]["offset"] for call in fred_session.calls] == [0, 10000]
    assert {call[2]["limit"] for call in fred_session.calls} == {10000}


def test_alpha_vantage_corp_actions_normalize_splits_and_dividends() -> None:
    av_session = FakeSession(
        [
            FakeResponse(
                200,
                {
                    "data": [
                        {
                            "symbol": "AAPL",
                            "ex_dividend_date": "2024-01-15",
                            "amount": "0.24",
                        },
                    ]
                },
            ),
            FakeResponse(
                200,
                {
                    "data": [
                        {
                            "symbol": "AAPL",
                            "effective_date": "2024-02-01",
                            "split_factor": "0.5",
                        },
                    ]
                },
            ),
        ]
    )
    av = AlphaVantageConnector(
        base_url="https://www.alphavantage.co",
        api_key="key",
        budget_manager=_budget_manager(),
        session=av_session,
    )

    actions = av.fetch("corp_actions", ["AAPL"], "2024-01-01", "2024-03-01")

    assert set(actions["event_type"]) == {"dividend", "split"}
    assert set(actions["symbol"]) == {"AAPL"}
    assert set(actions.columns) >= {
        "symbol",
        "event_type",
        "ex_date",
        "ratio",
        "source",
    }


def test_alpha_vantage_corp_actions_accept_named_top_level_arrays() -> None:
    av_session = FakeSession(
        [
            FakeResponse(
                200,
                {
                    "symbol": "AAPL",
                    "dividends": [
                        {"date": "2024-01-15", "amount": "0.24"},
                    ],
                },
            ),
            FakeResponse(
                200,
                {
                    "symbol": "AAPL",
                    "splits": [
                        {"date": "2024-02-01", "split_factor": "2:1"},
                    ],
                },
            ),
        ]
    )
    av = AlphaVantageConnector(
        base_url="https://www.alphavantage.co",
        api_key="key",
        budget_manager=_budget_manager(),
        session=av_session,
    )

    actions = av.fetch("corp_actions", ["AAPL"], "2024-01-01", "2024-03-01")

    assert set(actions["event_type"]) == {"dividend", "split"}
    split_row = actions.loc[actions["event_type"] == "split"].iloc[0]
    assert split_row["ratio"] == pytest.approx(0.5)


def test_fmp_and_sec_edgar_connectors() -> None:
    fmp_session = FakeSession(
        [
            FakeResponse(200, [{"symbol": "XYZ", "delistedDate": "2024-01-05"}]),
            FakeResponse(
                200, [{"oldSymbol": "FB", "newSymbol": "META", "date": "2022-06-09"}]
            ),
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

    fmp = FMPConnector(
        base_url="https://financialmodelingprep.com",
        api_key="key",
        budget_manager=_budget_manager(),
        session=fmp_session,
    )
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
    assert sec_session.calls[0][3]["User-Agent"] == "TradeML/0.1 test@example.com"
    assert "Host" not in sec_session.calls[0][3]


def test_sec_edgar_streams_companyfacts_without_json_materialization(
    tmp_path: Path,
) -> None:
    session = FakeSession(
        [
            FakeResponse(
                200,
                {
                    "cik": 320193,
                    "entityName": "Apple Inc.",
                    "facts": {"us-gaap": {"Revenue": {"units": {"USD": []}}}},
                },
            )
        ]
    )
    sec = SecEdgarConnector(
        base_url="https://data.sec.gov",
        user_agent="TradeML/0.1 test@example.com",
        budget_manager=_budget_manager(),
        session=session,
    )
    output = tmp_path / "companyfacts.json.gz"

    metadata = sec.stream_companyfacts_to_gzip(cik="320193", output=output)

    assert output.exists()
    assert metadata["cik"] == "0000320193"
    assert metadata["raw_bytes"] > 0
    assert session.calls[0][1].endswith("/api/xbrl/companyfacts/CIK0000320193.json")
    assert session.calls[0][3]["User-Agent"] == "TradeML/0.1 test@example.com"


def test_sec_edgar_companyfacts_404_is_typed_missing_payload(
    tmp_path: Path,
) -> None:
    session = FakeSession(
        [
            FakeResponse(
                404,
                {},
                text=(
                    '<?xml version="1.0" encoding="UTF-8"?>'
                    "<Error><Code>NoSuchKey</Code><Message>The specified key does not exist.</Message>"
                    "<Key>api/xbrl/companyfacts/CIK0001103838.json</Key></Error>"
                ),
            )
        ]
    )
    sec = SecEdgarConnector(
        base_url="https://data.sec.gov",
        user_agent="TradeML/0.1 test@example.com",
        budget_manager=_budget_manager(),
        session=session,
    )

    with pytest.raises(MissingCompanyfactsError) as exc_info:
        sec.stream_companyfacts_to_gzip(
            cik="1103838", output=tmp_path / "companyfacts.json.gz"
        )

    assert exc_info.value.cik == "0001103838"
    assert not (tmp_path / "companyfacts.json.gz").exists()


def test_fmp_delistings_follow_documented_page_limit_pagination() -> None:
    fmp_session = FakeSession(
        [
            FakeResponse(
                200,
                [
                    {"symbol": f"SYM{i}", "delistedDate": "2024-01-05"}
                    for i in range(100)
                ],
            ),
            FakeResponse(200, [{"symbol": "TAIL", "delistedDate": "2024-01-06"}]),
        ]
    )
    fmp = FMPConnector(
        base_url="https://financialmodelingprep.com",
        api_key="key",
        budget_manager=_budget_manager(),
        session=fmp_session,
    )

    delistings = fmp.fetch("delistings", [], "2024-01-01", "2024-01-31")

    assert len(delistings) == 101
    assert fmp_session.calls[0][2]["page"] == 0
    assert fmp_session.calls[0][2]["limit"] == 100
    assert fmp_session.calls[1][2]["page"] == 1
    assert fmp_session.calls[1][2]["limit"] == 100


def test_sec_edgar_connector_fetches_archived_submission_segments() -> None:
    sec_session = FakeSession(
        [
            FakeResponse(
                200,
                {
                    "filings": {
                        "recent": {
                            "filingDate": ["2024-02-10"],
                            "form": ["4"],
                            "accessionNumber": ["2"],
                        },
                        "files": [
                            {
                                "name": "CIK0000320193-submissions-001.json",
                                "filingFrom": "2024-01-01",
                                "filingTo": "2024-01-31",
                            }
                        ],
                    }
                },
            ),
            FakeResponse(
                200,
                {
                    "filingDate": ["2024-01-03", "2024-01-20"],
                    "form": ["8-K", "10-Q"],
                    "accessionNumber": ["1", "3"],
                },
            ),
        ]
    )
    sec = SecEdgarConnector(
        base_url="https://data.sec.gov",
        user_agent="TradeML/0.1 test@example.com",
        budget_manager=_budget_manager(),
        session=sec_session,
    )

    filings = sec.fetch("filing_index", ["320193"], "2024-01-01", "2024-01-31")

    assert filings["form"].tolist() == ["8-K", "10-Q"]
    assert (
        sec_session.calls[1][1]
        == "https://data.sec.gov/submissions/CIK0000320193-submissions-001.json"
    )


def test_sec_form4_index_manifest_parses_archive_cik_and_accession() -> None:
    sec_session = FakeSession([])
    sec = SecEdgarConnector(
        base_url="https://data.sec.gov",
        user_agent="TradeML/0.1 test@example.com",
        budget_manager=_budget_manager(),
        session=sec_session,
    )
    text = """Description:           Master Index of EDGAR Dissemination Feed
---------------------------------------------------------------------------
CIK|Company Name|Form Type|Date Filed|Filename
769993|TIPTREE INC.|4|2015-04-15|edgar/data/769993/000076999315000534/0000769993-15-000534.txt
1393726|TIPTREE INC.|8-K|2015-04-15|edgar/data/1393726/0001393726-15-000001.txt
1971213|SINCLAIR INC.|4/A|2025-02-04|edgar/data/1971213/000125084225000026/0001250842-25-000026.txt
"""

    rows = sec.parse_form4_index_manifest(
        text, index_year=2015, index_quarter=2, index_crawled_at="2026-05-05T00:00:00Z"
    )

    assert [row.form for row in rows] == ["4", "4/A"]
    assert rows[0].archive_cik == "769993"
    assert rows[0].accession == "0000769993-15-000534"
    assert rows[0].accession_no_dashes == "000076999315000534"
    assert (
        rows[0].index_filename
        == "edgar/data/769993/000076999315000534/0000769993-15-000534.txt"
    )


def test_sec_form4_index_manifest_parses_fixed_width_form_index() -> None:
    sec_session = FakeSession([])
    sec = SecEdgarConnector(
        base_url="https://data.sec.gov",
        user_agent="TradeML/0.1 test@example.com",
        budget_manager=_budget_manager(),
        session=sec_session,
    )
    text = """Form Type   Company Name                                                  CIK         Date Filed  File Name
---------------------------------------------------------------------------------------------------------------------------------------------
1-A         AIBOTICS, INC.                                                1763329     2025-04-02  edgar/data/1763329/0001096906-25-000399.txt
4           SINCLAIR, INC.                                                1971213     2025-04-07  edgar/data/1971213/0001250842-25-000026.txt
4/A         SUPER MICRO COMPUTER, INC.                                    1375365     2019-03-01  edgar/data/1375365/0001758554-19-000046.txt
"""

    rows = sec.parse_form4_index_manifest(
        text, index_year=2025, index_quarter=2, index_crawled_at="2026-05-05T00:00:00Z"
    )

    assert [row.form for row in rows] == ["4", "4/A"]
    assert rows[0].archive_cik == "1971213"
    assert rows[0].accession == "0001250842-25-000026"
    assert rows[0].index_filename == (
        "edgar/data/1971213/0001250842-25-000026.txt"
    )
    assert rows[1].archive_cik == "1375365"
    assert rows[1].accession == "0001758554-19-000046"


def test_sec_form4_url_helpers_use_archive_cik_not_issuer_cik() -> None:
    accession = "0000769993-15-000534"

    assert (
        sec_archive_dir("769993", accession)
        == "https://www.sec.gov/Archives/edgar/data/769993/000076999315000534/"
    )
    assert (
        raw_primary_xml_url("769993", accession, "ownership doc.xml")
        == "https://www.sec.gov/Archives/edgar/data/769993/000076999315000534/ownership%20doc.xml"
    )
    assert (
        complete_txt_url_from_index_filename(
            "edgar/data/769993/000076999315000534/0000769993-15-000534.txt"
        )
        == "https://www.sec.gov/Archives/edgar/data/769993/000076999315000534/0000769993-15-000534.txt"
    )
    assert (
        complete_txt_url_from_index_filename(
            "edgar/data/769993/0000769993-15-000534.txt"
        )
        == "https://www.sec.gov/Archives/edgar/data/769993/000076999315000534/0000769993-15-000534.txt"
    )
    assert (
        complete_txt_url_from_parts("769993", accession)
        == "https://www.sec.gov/Archives/edgar/data/769993/000076999315000534/0000769993-15-000534.txt"
    )
    assert accession_index_url("769993", accession).endswith(
        "/0000769993-15-000534-index.htm"
    )
    assert accession_directory_index_json_url("769993", accession).endswith(
        "/index.json"
    )
    assert "1393726" not in raw_primary_xml_url(
        "769993", accession, "ownershipdoc03152015012806.xml"
    )


def test_sec_form4_fetch_uses_full_index_not_submissions_json() -> None:
    sec_session = FakeSession(
        [
            FakeResponse(
                200,
                {},
                text="""CIK|Company Name|Form Type|Date Filed|Filename
769993|TIPTREE INC.|4|2015-04-15|edgar/data/769993/000076999315000534/0000769993-15-000534.txt
1393726|TIPTREE INC.|8-K|2015-04-15|edgar/data/1393726/0001393726-15-000001.txt
""",
            )
        ]
    )
    sec = SecEdgarConnector(
        base_url="https://data.sec.gov",
        user_agent="TradeML/0.1 test@example.com",
        budget_manager=_budget_manager(),
        session=sec_session,
    )

    frame = sec.fetch("form4_ownership", [], "2015-04-01", "2015-04-30")

    assert frame.iloc[0]["archive_cik"] == "769993"
    assert frame.iloc[0]["accession"] == "0000769993-15-000534"
    assert "/full-index/2015/QTR2/form.idx" in sec_session.calls[0][1]
    assert sec_session.calls[0][3]["User-Agent"] == "TradeML/0.1 test@example.com"


def test_sec_8k_fetch_uses_full_index_and_exact_form_filter() -> None:
    sec_session = FakeSession(
        [
            FakeResponse(
                200,
                {},
                text="""CIK|Company Name|Form Type|Date Filed|Filename
320193|APPLE INC.|8-K|2025-04-07|edgar/data/320193/000032019325000001/0000320193-25-000001.txt
320193|APPLE INC.|8-K/A|2025-04-08|edgar/data/320193/000032019325000002/0000320193-25-000002.txt
320193|APPLE INC.|10-Q|2025-04-09|edgar/data/320193/000032019325000003/0000320193-25-000003.txt
""",
            )
        ]
    )
    sec = SecEdgarConnector(
        base_url="https://data.sec.gov",
        user_agent="TradeML/0.1 test@example.com",
        budget_manager=_budget_manager(),
        session=sec_session,
    )

    frame = sec.fetch("sec8k_index", [], "2025-04-01", "2025-04-30")

    assert frame["form"].tolist() == ["8-K"]
    assert frame.iloc[0]["archive_cik"] == "320193"
    assert frame.iloc[0]["accession"] == "0000320193-25-000001"
    assert "/full-index/2025/QTR2/form.idx" in sec_session.calls[0][1]


def test_sec_retrieves_complete_submission_text_from_index_filename() -> None:
    sec_session = FakeSession([FakeResponse(200, {}, text="<SEC-DOCUMENT />")])
    sec = SecEdgarConnector(
        base_url="https://data.sec.gov",
        user_agent="TradeML/0.1 test@example.com",
        budget_manager=_budget_manager(),
        session=sec_session,
    )

    status, text, url = sec.retrieve_complete_submission_text(
        index_filename="edgar/data/320193/0000320193-25-000001.txt",
        endpoint_key="sec8k_complete_txt",
    )

    assert status == 200
    assert text == "<SEC-DOCUMENT />"
    assert url.endswith("/320193/000032019325000001/0000320193-25-000001.txt")
    assert sec_session.calls[0][3]["User-Agent"] == "TradeML/0.1 test@example.com"
    assert sec_session.kwargs[0]["stream"] is True


def test_sec_complete_submission_text_rejects_oversized_stream() -> None:
    sec_session = FakeSession([FakeResponse(200, {}, text="0123456789")])
    sec = SecEdgarConnector(
        base_url="https://data.sec.gov",
        user_agent="TradeML/0.1 test@example.com",
        budget_manager=_budget_manager(),
        session=sec_session,
        max_complete_text_bytes=4,
        max_complete_text_seconds=120,
    )

    with pytest.raises(PermanentConnectorError, match="exceeded max bytes"):
        sec.retrieve_complete_submission_text(
            index_filename="edgar/data/320193/0000320193-25-000001.txt",
            endpoint_key="sec8k_complete_txt",
        )


def test_sec_complete_submission_text_default_guard_allows_large_real_8k_headers() -> None:
    sec_session = FakeSession(
        [
            FakeResponse(
                200,
                {},
                text="<SEC-DOCUMENT />",
                headers={"Content-Length": str(180 * 1024 * 1024)},
            )
        ]
    )
    sec = SecEdgarConnector(
        base_url="https://data.sec.gov",
        user_agent="TradeML/0.1 test@example.com",
        budget_manager=_budget_manager(),
        session=sec_session,
    )

    status, text, _url = sec.retrieve_complete_submission_text(
        index_filename="edgar/data/320193/0000320193-25-000001.txt",
        endpoint_key="sec8k_complete_txt",
    )

    assert status == 200
    assert text == "<SEC-DOCUMENT />"


def test_sec_form4_retrieval_falls_back_to_complete_txt_and_records_metadata() -> None:
    complete_txt = """<SEC-DOCUMENT>
<SEC-HEADER>
<ACCEPTANCE-DATETIME>20150415170104
</SEC-HEADER>
<DOCUMENT>
<TYPE>4
<FILENAME>ownershipdoc03152015012806.xml
<XML>
<ownershipDocument>
  <documentType>4</documentType>
  <issuer><issuerCik>1393726</issuerCik></issuer>
</ownershipDocument>
</XML>
</DOCUMENT>
</SEC-DOCUMENT>
"""
    sec_session = FakeSession(
        [
            FakeResponse(404, {}, text="not found"),
            FakeResponse(200, {}, text=complete_txt),
        ]
    )
    sec = SecEdgarConnector(
        base_url="https://data.sec.gov",
        user_agent="TradeML/0.1 test@example.com",
        budget_manager=_budget_manager(),
        session=sec_session,
    )
    manifest = Form4ManifestRow(
        archive_cik="769993",
        form="4",
        filed_date="2015-04-15",
        index_filename="edgar/data/769993/0000769993-15-000534.txt",
        accession="0000769993-15-000534",
        accession_no_dashes="000076999315000534",
        discovery_source="sec_full_index",
        index_year=2015,
        index_quarter=2,
        index_file_hash="fixture",
        index_crawled_at="2026-05-05T00:00:00Z",
    )

    result = sec.retrieve_form4_ownership_xml(
        manifest, primary_document="missing.xml"
    )

    assert result.ownership_xml is not None
    assert "<ownershipDocument>" in result.ownership_xml
    assert result.metadata.primary_xml_http_status == 404
    assert result.metadata.complete_txt_http_status == 200
    assert result.metadata.xml_source == "complete_txt_extracted"
    assert result.metadata.accepted_at_source == "sgml_header"
    assert result.metadata.accepted_at_raw == "20150415170104"
    assert "raw_primary_404" in result.metadata.quality_flags
    assert "used_complete_txt_fallback" in result.metadata.quality_flags
    assert "/769993/000076999315000534/missing.xml" in sec_session.calls[0][1]


def test_sec_form4_retrieval_uses_sgml_then_index_for_accepted_time() -> None:
    sec_session = FakeSession(
        [
            FakeResponse(
                200,
                {},
                text="<ownershipDocument><documentType>4</documentType></ownershipDocument>",
            ),
            FakeResponse(404, {}, text="missing header"),
            FakeResponse(200, {}, text="<html><body>Accepted 2025-02-04 18:01:02</body></html>"),
        ]
    )
    sec = SecEdgarConnector(
        base_url="https://data.sec.gov",
        user_agent="TradeML/0.1 test@example.com",
        budget_manager=_budget_manager(),
        session=sec_session,
    )
    manifest = Form4ManifestRow(
        archive_cik="1971213",
        form="4",
        filed_date="2025-02-04",
        index_filename="edgar/data/1971213/0001250842-25-000026.txt",
        accession="0001250842-25-000026",
        accession_no_dashes="000125084225000026",
        discovery_source="sec_full_index",
        index_year=2025,
        index_quarter=1,
        index_file_hash="fixture",
        index_crawled_at="2026-05-05T00:00:00Z",
    )

    result = sec.retrieve_form4_ownership_xml(
        manifest, primary_document="primary_doc.xml"
    )

    assert result.metadata.xml_source == "raw_primary"
    assert result.metadata.accepted_at_source == "accession_index"
    assert result.metadata.accepted_at_raw == "2025-02-04 18:01:02"
    assert "sgml_header_404" in result.metadata.quality_flags
    assert sec_session.calls[1][1].endswith("/0001250842-25-000026.hdr.sgml")
    assert sec_session.calls[2][1].endswith("/0001250842-25-000026-index.htm")


def test_sec_edgar_company_tickers_uses_sec_host_without_forced_data_host() -> None:
    sec_session = FakeSession(
        [
            FakeResponse(
                200,
                {
                    "0": {"ticker": "AAPL", "cik_str": 320193, "title": "Apple Inc."},
                },
            )
        ]
    )
    sec = SecEdgarConnector(
        base_url="https://data.sec.gov",
        user_agent="TradeML/0.1 test@example.com",
        budget_manager=_budget_manager(),
        session=sec_session,
    )

    frame = sec.fetch("company_tickers", [], "2024-01-01", "2024-01-31")

    assert frame.iloc[0]["ticker"] == "AAPL"
    method, url, _params, headers = sec_session.calls[0]
    assert method == "GET"
    assert url == "https://www.sec.gov/files/company_tickers.json"
    assert headers["User-Agent"] == "TradeML/0.1 test@example.com"
    assert "Host" not in headers


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
                [
                    {
                        "ticker": "AAPL",
                        "date": "2023-12-31",
                        "statementType": "annual",
                        "revenue": 1000,
                    }
                ],
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
    dividends = connector.fetch(
        "corp_actions_dividends", ["AAPL"], "2024-01-01", "2024-01-31"
    )
    splits = connector.fetch(
        "corp_actions_splits", ["AAPL"], "2024-02-01", "2024-02-01"
    )
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
    budget_manager = _budget_manager()
    session = FakeSession(
        [
            FakeResponse(
                200,
                {
                    "AAPL": {
                        "meta": {"symbol": "AAPL", "interval": "1day"},
                        "values": [
                            {
                                "datetime": "2024-01-02",
                                "open": "10",
                                "high": "11",
                                "low": "9",
                                "close": "10.5",
                                "volume": "100",
                            }
                        ],
                    },
                    "MSFT": {
                        "meta": {"symbol": "MSFT", "interval": "1day"},
                        "values": [
                            {
                                "datetime": "2024-01-02",
                                "open": "20",
                                "high": "21",
                                "low": "19",
                                "close": "20.5",
                                "volume": "200",
                            }
                        ],
                    },
                },
            ),
            FakeResponse(
                200,
                {
                    "meta": {"symbol": "AAPL"},
                    "dividends": [
                        {
                            "ex_date": "2024-01-05",
                            "amount": "0.24",
                            "payment_date": "2024-01-15",
                        }
                    ],
                },
            ),
            FakeResponse(
                200,
                {
                    "meta": {"symbol": "AAPL"},
                    "splits": [{"date": "2024-02-01", "ratio": "2:1"}],
                },
            ),
            FakeResponse(
                200,
                {
                    "earnings": [
                        {"symbol": "AAPL", "date": "2024-01-25", "eps_estimate": "2.10"}
                    ]
                },
            ),
            FakeResponse(
                200,
                {
                    "meta": {"symbol": "AAPL"},
                    "income_statement": [
                        {
                            "fiscal_date": "2023-12-31",
                            "period": "annual",
                            "revenue": "1000",
                        }
                    ],
                },
            ),
            FakeResponse(
                200,
                {
                    "meta": {"symbol": "AAPL"},
                    "balance_sheet": [
                        {
                            "fiscal_date": "2023-12-31",
                            "period": "annual",
                            "total_assets": "2000",
                        }
                    ],
                },
            ),
            FakeResponse(
                200,
                {
                    "meta": {"symbol": "AAPL"},
                    "cash_flow": [
                        {
                            "fiscal_date": "2023-12-31",
                            "period": "annual",
                            "operating_cash_flow": "500",
                        }
                    ],
                },
            ),
        ]
    )
    connector = TwelveDataConnector(
        base_url="https://api.twelvedata.com",
        api_key="key",
        budget_manager=budget_manager,
        session=session,
    )

    bars = connector.fetch("equities_eod", ["AAPL", "MSFT"], "2024-01-02", "2024-01-02")
    dividends = connector.fetch("dividends", ["AAPL"], "2024-01-01", "2024-01-31")
    splits = connector.fetch("splits", ["AAPL"], "2024-02-01", "2024-02-01")
    earnings = connector.fetch("earnings_calendar", [], "2024-01-01", "2024-01-31")
    statements = connector.fetch(
        "financial_statements", ["AAPL"], "2024-01-01", "2024-01-31"
    )

    assert sorted(bars["symbol"].tolist()) == ["AAPL", "MSFT"]
    assert bars.loc[bars["symbol"] == "AAPL", "close"].iloc[0] == pytest.approx(10.5)
    assert session.calls[0][2]["symbol"] == "AAPL,MSFT"
    assert dividends.iloc[0]["symbol"] == "AAPL"
    assert dividends.iloc[0]["amount"] == pytest.approx(0.24)
    assert splits.iloc[0]["ratio"] == pytest.approx(0.5)
    assert earnings.iloc[0]["symbol"] == "AAPL"
    assert set(statements["statement_type"]) == {
        "income_statement",
        "balance_sheet",
        "cash_flow",
    }
    assert session.calls[0][2]["apikey"] == "key"
    snapshot = budget_manager.snapshot()
    assert snapshot["vendors"]["twelve_data"]["daily_spend"]["FORWARD"] == 2
    assert snapshot["vendors"]["twelve_data"]["daily_requests"]["FORWARD"] == 1


def test_twelve_data_connector_parses_exchange_suffixed_batch_keys() -> None:
    session = FakeSession(
        [
            FakeResponse(
                200,
                {
                    "COST:NYSE": {
                        "meta": {"symbol": "COST", "interval": "1day"},
                        "values": [
                            {
                                "datetime": "2024-01-02",
                                "open": "10",
                                "high": "11",
                                "low": "9",
                                "close": "10.5",
                                "volume": "100",
                            }
                        ],
                    },
                    "DVN:NYSE": {
                        "meta": {"symbol": "DVN", "interval": "1day"},
                        "values": [
                            {
                                "datetime": "2024-01-02",
                                "open": "20",
                                "high": "21",
                                "low": "19",
                                "close": "20.5",
                                "volume": "200",
                            }
                        ],
                    },
                },
            )
        ]
    )
    connector = TwelveDataConnector(
        base_url="https://api.twelvedata.com",
        api_key="key",
        budget_manager=_budget_manager(),
        session=session,
    )

    bars = connector.fetch("equities_eod", ["COST", "DVN"], "2024-01-02", "2024-01-02")

    assert sorted(bars["symbol"].tolist()) == ["COST", "DVN"]


def test_retry_and_permanent_error_behavior() -> None:
    retry_session = FakeSession(
        [
            FakeResponse(429, {"error": "too many requests"}),
            FakeResponse(
                200,
                {
                    "results": [
                        {
                            "t": 1704153600000,
                            "o": 10,
                            "h": 11,
                            "l": 9,
                            "c": 10.5,
                            "vw": 10.2,
                            "v": 20,
                            "n": 4,
                        }
                    ]
                },
            ),
        ]
    )
    connector = MassiveConnector(
        base_url="https://api.polygon.io",
        api_key="key",
        budget_manager=_budget_manager(),
        session=retry_session,
        retry_config=RetryConfig(
            max_attempts=2, base_delay_seconds=0.0, max_delay_seconds=0.0
        ),
        sleep_fn=lambda _: None,
    )

    frame = connector.fetch("equities_eod", ["MSFT"], "2024-01-02", "2024-01-02")
    assert not frame.empty

    permanent_session = FakeSession(
        [FakeResponse(403, {"error": "denied"}, text="NOT_ENTITLED")]
    )
    permanent = MassiveConnector(
        base_url="https://api.polygon.io",
        api_key="key",
        budget_manager=_budget_manager(),
        session=permanent_session,
    )

    with pytest.raises(PermanentConnectorError):
        permanent.fetch("equities_eod", ["MSFT"], "2024-01-02", "2024-01-02")


def test_http_connector_uses_documented_reset_header_for_retry_delay() -> None:
    sleep_calls: list[float] = []
    session = FakeSession(
        [
            FakeResponse(
                429,
                {"error": "too many requests"},
                headers={"X-RateLimit-Reset": "103"},
            ),
            FakeResponse(
                200,
                {
                    "bars": {
                        "AAPL": [
                            {
                                "t": "2026-01-02T00:00:00Z",
                                "o": 1.0,
                                "h": 2.0,
                                "l": 0.5,
                                "c": 1.5,
                                "vw": 1.4,
                                "v": 10,
                                "n": 2,
                            }
                        ]
                    },
                    "next_page_token": None,
                },
            ),
        ]
    )
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(base_module.time, "time", lambda: 100.0)
    connector = AlpacaConnector(
        base_url="https://data.alpaca.markets",
        trading_base_url="https://paper-api.alpaca.markets/v2",
        api_key="key",
        budget_manager=_budget_manager(),
        session=session,
        retry_config=RetryConfig(
            max_attempts=2, base_delay_seconds=0.0, max_delay_seconds=0.0
        ),
        sleep_fn=lambda seconds: sleep_calls.append(seconds),
    )
    try:
        frame = connector.fetch("equities_eod", ["AAPL"], "2026-01-02", "2026-01-02")
    finally:
        monkeypatch.undo()

    telemetry = connector.budget_manager.snapshot()["vendors"]["alpaca"]["telemetry"]
    assert not frame.empty
    assert sleep_calls == [3.0]
    assert telemetry["totals"]["remote_rate_limits"] == 1


def test_http_connector_wraps_request_exception_as_temporary_error() -> None:
    connector = MassiveConnector(
        base_url="https://api.polygon.io",
        api_key="key",
        budget_manager=_budget_manager(),
        session=ErrorSession(requests.ReadTimeout("timed out")),
        retry_config=RetryConfig(
            max_attempts=1, base_delay_seconds=0.0, max_delay_seconds=0.0
        ),
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
        retry_config=RetryConfig(
            max_attempts=1, base_delay_seconds=0.0, max_delay_seconds=0.0
        ),
        sleep_fn=lambda _: None,
    )

    with pytest.raises(RemoteRateLimitConnectorError) as exc_info:
        connector.fetch("equities_eod", ["MSFT"], "2024-01-02", "2024-01-02")

    assert exc_info.value.vendor == "massive"


def test_http_connector_raises_typed_local_budget_block() -> None:
    budget_manager = BudgetManager({"massive": {"rpm": 1, "daily_cap": 10}})
    budget_manager.record_spend("massive")
    connector = MassiveConnector(
        base_url="https://api.polygon.io",
        api_key="key",
        budget_manager=budget_manager,
        session=FakeSession([]),
    )

    with pytest.raises(BudgetBlockedConnectorError) as exc_info:
        connector.fetch("equities_eod", ["MSFT"], "2024-01-02", "2024-01-02")

    assert exc_info.value.decision.blocked_dimension == "minute"
