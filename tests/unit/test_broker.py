from __future__ import annotations

import pytest

from trademl.broker.alpaca_paper import AlpacaPaperCredentials, AlpacaPaperTradingClient


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self.payload


class FakeSession:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def request(self, method, url, **kwargs):  # noqa: ANN001
        self.calls.append({"method": method, "url": url, **kwargs})
        return FakeResponse({"ok": True})


def test_alpaca_paper_client_refuses_live_base_url() -> None:
    with pytest.raises(ValueError, match="paper trading base URL"):
        AlpacaPaperTradingClient(
            credentials=AlpacaPaperCredentials(api_key="key", api_secret="secret"),
            base_url="https://api.alpaca.markets/v2",
        )


def test_alpaca_paper_client_reads_account_from_paper_endpoint() -> None:
    session = FakeSession()
    client = AlpacaPaperTradingClient(
        credentials=AlpacaPaperCredentials(api_key="key", api_secret="secret"),
        session=session,
    )

    assert client.account() == {"ok": True}
    assert session.calls[0]["method"] == "GET"
    assert session.calls[0]["url"] == "https://paper-api.alpaca.markets/v2/account"
    assert session.calls[0]["headers"]["APCA-API-KEY-ID"] == "key"


def test_alpaca_paper_order_submission_is_disabled_by_default() -> None:
    session = FakeSession()
    client = AlpacaPaperTradingClient(
        credentials=AlpacaPaperCredentials(api_key="key", api_secret="secret"),
        session=session,
    )

    with pytest.raises(RuntimeError, match="disabled"):
        client.submit_order({"symbol": "AAPL", "qty": "1", "side": "buy", "type": "market", "time_in_force": "day"})

    assert session.calls == []


def test_alpaca_paper_order_submission_uses_orders_endpoint_when_enabled() -> None:
    session = FakeSession()
    client = AlpacaPaperTradingClient(
        credentials=AlpacaPaperCredentials(api_key="key", api_secret="secret"),
        session=session,
        submit_orders_enabled=True,
    )

    payload = {"symbol": "AAPL", "qty": "1", "side": "buy", "type": "market", "time_in_force": "day"}
    assert client.submit_order(payload) == {"ok": True}

    assert session.calls[0]["method"] == "POST"
    assert session.calls[0]["url"] == "https://paper-api.alpaca.markets/v2/orders"
    assert session.calls[0]["json"] == payload

