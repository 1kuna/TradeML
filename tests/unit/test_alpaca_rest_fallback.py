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
