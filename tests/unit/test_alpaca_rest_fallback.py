from datetime import date

import pytest

from data_layer.connectors import alpaca_connector


def test_alpaca_rest_fallback_on_sdk_nameerror(monkeypatch):
    """Ensure SDK NameError triggers REST fallback when SDK path is enabled."""
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    monkeypatch.setenv("ALPACA_BARS_USE_SDK", "1")

    class FakeClient:
        def __init__(self, *_args, **_kwargs):
            pass

        def get_stock_bars(self, *_args, **_kwargs):
            raise NameError("local variable 'json' referenced before assignment")

    monkeypatch.setattr(alpaca_connector, "StockHistoricalDataClient", FakeClient)

    rest_payload = {
        "ABC": [
            {
                "timestamp": "2024-01-01T00:00:00Z",
                "open": 1.0,
                "high": 2.0,
                "low": 0.5,
                "close": 1.5,
                "vwap": 1.25,
                "volume": 100,
                "trade_count": 4,
            }
        ]
    }

    def _fake_rest(self, symbols, start_date, end_date, timeframe):
        assert symbols == ["ABC"]
        assert start_date == date(2024, 1, 1)
        assert end_date == date(2024, 1, 1)
        assert timeframe == "1Min"
        return rest_payload

    monkeypatch.setattr(alpaca_connector.AlpacaConnector, "_fetch_raw_rest", _fake_rest, raising=False)

    conn = alpaca_connector.AlpacaConnector()
    data = conn._fetch_raw(["ABC"], date(2024, 1, 1), date(2024, 1, 1), timeframe="1Min")

    assert data == rest_payload


def test_fetch_raw_rest_handles_multibars_payload(monkeypatch):
    """Multi-bars REST payload (dict of symbol -> bars) should parse without attribute errors."""
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    monkeypatch.setenv("ALPACA_BARS_USE_SDK", "0")

    payloads = [
        {
            "bars": {
                "AAPL": [
                    {"t": "2024-01-01T00:01:00Z", "o": 101.0, "h": 102.0, "l": 99.0, "c": 100.5, "v": 10, "n": 5},
                    {"t": "2024-01-01T00:02:00Z", "o": 100.5, "h": 101.0, "l": 99.5, "c": 100.0, "v": 20, "n": 7},
                ],
                "MSFT": [
                    {"t": "2024-01-01T00:02:00Z", "o": 201.0, "h": 202.0, "l": 200.0, "c": 201.5, "v": 30, "n": 9}
                ],
            },
            "next_page_token": "token-1",
        },
        {
            "bars": {
                "GOOGL": [
                    {"t": "2024-01-02T00:02:00Z", "o": 301.0, "h": 302.0, "l": 300.0, "c": 301.5, "v": 300, "n": 11}
                ]
            },
            "next_page_token": None,
        },
    ]

    class FakeResp:
        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

    def fake_get(self, url, params=None, headers=None):
        assert url.endswith("/v2/stocks/bars")
        if not payloads:
            raise AssertionError("Exhausted payloads")
        return FakeResp(payloads.pop(0))

    monkeypatch.setattr(alpaca_connector.AlpacaConnector, "_get", fake_get, raising=False)

    conn = alpaca_connector.AlpacaConnector()
    data = conn._fetch_raw_rest(
        ["AAPL", "MSFT", "GOOGL"], date(2024, 1, 1), date(2024, 1, 2), timeframe="1Min"
    )

    assert set(data.keys()) == {"AAPL", "MSFT", "GOOGL"}
    assert data["AAPL"][0]["open"] == 101.0
    assert data["MSFT"][0]["timestamp"] == "2024-01-01T00:02:00Z"
    assert data["GOOGL"][0]["volume"] == 300
