from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from trademl.broker.alpaca_paper import (
    AlpacaPaperCredentials,
    AlpacaPaperTradingClient,
    build_alpaca_order_payloads_from_deltas,
    paper_account_smoke_check,
    submit_alpaca_paper_payloads,
)


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


class FakeAccountSession(FakeSession):
    def request(self, method, url, **kwargs):  # noqa: ANN001
        self.calls.append({"method": method, "url": url, **kwargs})
        return FakeResponse(
            {
                "id": "paper-account",
                "currency": "USD",
                "buying_power": "100000",
                "portfolio_value": "100000",
                "client_order_id": (kwargs.get("json") or {}).get("client_order_id"),
                "symbol": (kwargs.get("json") or {}).get("symbol"),
                "status": "accepted",
            }
        )


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


def test_paper_account_smoke_is_read_only_and_skips_without_credentials() -> None:
    session = FakeAccountSession()

    skipped = paper_account_smoke_check(environ={}, session=session)

    assert skipped["status"] == "skipped"
    assert skipped["read_only"] is True
    assert session.calls == []

    ok = paper_account_smoke_check(
        policy={"broker": {"api_key_env": "KEY", "api_secret_env": "SECRET"}},
        environ={"KEY": "paper-key", "SECRET": "paper-secret"},
        session=session,
    )

    assert ok["status"] == "ok"
    assert ok["account_id"] == "paper-account"
    assert session.calls[0]["method"] == "GET"
    assert session.calls[0]["url"].endswith("/account")
    assert "paper-secret" not in repr(ok)


def test_build_alpaca_order_payloads_from_deltas_is_deterministic(tmp_path: Path) -> None:
    orders_path = tmp_path / "paper_orders.parquet"
    pd.DataFrame(
        [
            {"symbol": "MSFT", "order_delta": -0.10},
            {"symbol": "AAPL", "order_delta": 0.25},
            {"symbol": "ZERO", "order_delta": 0.0},
        ]
    ).to_parquet(orders_path, index=False)

    payloads = build_alpaca_order_payloads_from_deltas(
        orders_path=orders_path,
        portfolio_notional=100_000,
        client_order_prefix="canary",
    )

    assert payloads == [
        {
            "symbol": "AAPL",
            "side": "buy",
            "type": "market",
            "time_in_force": "day",
            "notional": "25000.00",
            "client_order_id": "canary-AAPL-buy",
        },
        {
            "symbol": "MSFT",
            "side": "sell",
            "type": "market",
            "time_in_force": "day",
            "notional": "10000.00",
            "client_order_id": "canary-MSFT-sell",
        },
    ]


def test_submit_alpaca_paper_payloads_requires_explicit_paper_enablement() -> None:
    payloads = [{"symbol": "AAPL", "side": "buy", "type": "market", "time_in_force": "day", "notional": "100.00"}]

    with pytest.raises(RuntimeError, match="disabled"):
        submit_alpaca_paper_payloads(
            payloads=payloads,
            policy={
                "no_live_orders": True,
                "broker": {"provider": "alpaca_paper", "submit_orders_enabled": False},
            },
            environ={"ALPACA_API_KEY": "paper-key", "ALPACA_API_SECRET": "paper-secret"},
        )

    session = FakeAccountSession()
    result = submit_alpaca_paper_payloads(
        payloads=payloads,
        policy={
            "no_live_orders": True,
            "broker": {"provider": "alpaca_paper", "submit_orders_enabled": True},
        },
        environ={"ALPACA_API_KEY": "paper-key", "ALPACA_API_SECRET": "paper-secret"},
        session=session,
    )

    assert result["status"] == "submitted"
    assert result["submitted_count"] == 1
    assert session.calls[0]["url"] == "https://paper-api.alpaca.markets/v2/orders"
