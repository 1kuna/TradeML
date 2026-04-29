"""Alpaca paper-trading client guarded for research-only use."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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

