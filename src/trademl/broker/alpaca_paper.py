"""Alpaca paper-trading client guarded for research-only use."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests


PAPER_BASE_URL = "https://paper-api.alpaca.markets/v2"


@dataclass(frozen=True)
class AlpacaPaperCredentials:
    """Alpaca paper-trading API credentials."""

    api_key: str
    api_secret: str


class AlpacaPaperTradingClient:
    """Small paper-only wrapper around Alpaca Trading API endpoints."""

    def __init__(
        self,
        *,
        credentials: AlpacaPaperCredentials,
        base_url: str = PAPER_BASE_URL,
        submit_orders_enabled: bool = False,
        session: requests.Session | None = None,
        timeout: float = 30.0,
    ) -> None:
        if "paper-api.alpaca.markets" not in base_url:
            raise ValueError("AlpacaPaperTradingClient only accepts the paper trading base URL")
        self.credentials = credentials
        self.base_url = base_url.rstrip("/")
        self.submit_orders_enabled = submit_orders_enabled
        self.session = session or requests.Session()
        self.timeout = timeout

    def account(self) -> dict[str, Any]:
        """Return the current paper account payload."""
        return self._request("GET", "/account")

    def submit_order(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Submit one paper order when explicitly enabled."""
        if not self.submit_orders_enabled:
            raise RuntimeError("paper order submission is disabled by config")
        return self._request("POST", "/orders", json_payload=payload)

    def _request(self, method: str, endpoint: str, *, json_payload: dict[str, Any] | None = None) -> dict[str, Any]:
        headers = {
            "APCA-API-KEY-ID": self.credentials.api_key,
            "APCA-API-SECRET-KEY": self.credentials.api_secret,
            "accept": "application/json",
        }
        if json_payload is not None:
            headers["content-type"] = "application/json"
        response = self.session.request(
            method,
            f"{self.base_url}{endpoint}",
            headers=headers,
            json=json_payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, dict) else {"data": payload}


def alpaca_paper_credentials_from_env(
    *,
    api_key_env: str = "ALPACA_API_KEY",
    api_secret_env: str = "ALPACA_API_SECRET",
    environ: dict[str, str] | None = None,
) -> AlpacaPaperCredentials | None:
    """Return paper credentials from environment variables when both are present."""
    values = environ if environ is not None else os.environ
    api_key = str(values.get(api_key_env) or "").strip()
    api_secret = str(values.get(api_secret_env) or "").strip()
    if not api_key or not api_secret:
        return None
    return AlpacaPaperCredentials(api_key=api_key, api_secret=api_secret)


def paper_account_smoke_check(
    *,
    policy: dict[str, Any] | None = None,
    environ: dict[str, str] | None = None,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """Run a read-only Alpaca paper account smoke check when credentials exist."""
    broker = dict((policy or {}).get("broker") or policy or {})
    credentials = alpaca_paper_credentials_from_env(
        api_key_env=str(broker.get("api_key_env") or "ALPACA_API_KEY"),
        api_secret_env=str(broker.get("api_secret_env") or "ALPACA_API_SECRET"),
        environ=environ,
    )
    if credentials is None:
        return {"status": "skipped", "reason": "missing Alpaca paper credentials", "read_only": True}
    client = AlpacaPaperTradingClient(
        credentials=credentials,
        base_url=str(broker.get("base_url") or PAPER_BASE_URL),
        submit_orders_enabled=False,
        session=session,
    )
    try:
        account = client.account()
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "reason": str(exc), "read_only": True}
    return {
        "status": "ok",
        "account_id": account.get("id") or account.get("account_number"),
        "currency": account.get("currency"),
        "buying_power": account.get("buying_power"),
        "portfolio_value": account.get("portfolio_value"),
        "read_only": True,
        "base_url": client.base_url,
    }


def build_alpaca_order_payloads_from_deltas(
    *,
    orders_path: Path,
    portfolio_notional: float,
    client_order_prefix: str,
) -> list[dict[str, Any]]:
    """Convert paper target deltas into deterministic Alpaca paper market-order payloads."""
    orders = pd.read_parquet(orders_path)
    if orders.empty:
        return []
    payloads: list[dict[str, Any]] = []
    frame = orders.copy()
    frame["order_delta"] = pd.to_numeric(frame["order_delta"], errors="coerce").fillna(0.0)
    for row in frame.sort_values(["symbol"]).to_dict("records"):
        delta = float(row.get("order_delta") or 0.0)
        if abs(delta) <= 1e-12:
            continue
        symbol = str(row.get("symbol") or "").upper().strip()
        if not symbol:
            continue
        side = "buy" if delta > 0 else "sell"
        notional = round(abs(delta) * float(portfolio_notional), 2)
        if notional <= 0:
            continue
        payloads.append(
            {
                "symbol": symbol,
                "side": side,
                "type": "market",
                "time_in_force": "day",
                "notional": f"{notional:.2f}",
                "client_order_id": f"{client_order_prefix}-{symbol}-{side}",
            }
        )
    return payloads


def submit_alpaca_paper_payloads(
    *,
    payloads: list[dict[str, Any]],
    policy: dict[str, Any],
    environ: dict[str, str] | None = None,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """Submit prepared paper payloads only when policy explicitly enables it."""
    if not bool(policy.get("no_live_orders", True)):
        raise ValueError("paper submit requires no_live_orders=true")
    broker = dict(policy.get("broker") or {})
    if str(broker.get("provider") or "alpaca_paper") != "alpaca_paper":
        raise ValueError("paper submit only supports alpaca_paper")
    if not bool(broker.get("submit_orders_enabled", False)):
        raise RuntimeError("paper order submission is disabled by config")
    credentials = alpaca_paper_credentials_from_env(
        api_key_env=str(broker.get("api_key_env") or "ALPACA_API_KEY"),
        api_secret_env=str(broker.get("api_secret_env") or "ALPACA_API_SECRET"),
        environ=environ,
    )
    if credentials is None:
        raise RuntimeError("missing Alpaca paper credentials")
    client = AlpacaPaperTradingClient(
        credentials=credentials,
        base_url=str(broker.get("base_url") or PAPER_BASE_URL),
        submit_orders_enabled=True,
        session=session,
    )
    submissions = []
    for payload in payloads:
        result = client.submit_order(payload)
        submissions.append(
            {
                "request": payload,
                "status": result.get("status") or "submitted",
                "id": result.get("id"),
                "client_order_id": result.get("client_order_id") or payload.get("client_order_id"),
                "symbol": result.get("symbol") or payload.get("symbol"),
            }
        )
    return {"status": "submitted", "submitted_count": len(submissions), "submissions": submissions}
