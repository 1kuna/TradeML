"""Base protocols and retry-aware HTTP helpers for vendor connectors."""

from __future__ import annotations

import csv
import io
import logging
import random
import time
from dataclasses import dataclass
from datetime import date as date_type
from typing import Any, Protocol, runtime_checkable

import pandas as pd
import requests

from trademl.data_node.budgets import BudgetManager


LOGGER = logging.getLogger(__name__)
DEFAULT_TIMEOUT = 30


class ConnectorError(RuntimeError):
    """Base connector error."""


class PermanentConnectorError(ConnectorError):
    """Non-retryable connector error."""


class TemporaryConnectorError(ConnectorError):
    """Retryable connector error."""


@runtime_checkable
class BaseConnector(Protocol):
    """Protocol implemented by all vendor connectors."""

    vendor_name: str

    def fetch(
        self,
        dataset: str,
        symbols: list[str],
        start_date: str | date_type,
        end_date: str | date_type,
    ) -> pd.DataFrame:
        """Fetch a normalized dataframe for the requested dataset."""


@dataclass(slots=True)
class RetryConfig:
    """Retry policy for transient vendor failures."""

    max_attempts: int = 4
    base_delay_seconds: float = 0.2
    max_delay_seconds: float = 2.0


class HTTPConnector:
    """Shared HTTP behavior for vendor-specific connectors."""

    vendor_name = "base"

    def __init__(
        self,
        *,
        base_url: str,
        budget_manager: BudgetManager,
        api_key: str | None = None,
        session: requests.Session | None = None,
        retry_config: RetryConfig | None = None,
        sleep_fn: Any = time.sleep,
        rng: random.Random | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.budget_manager = budget_manager
        self.api_key = api_key
        self.session = session or requests.Session()
        self.retry_config = retry_config or RetryConfig()
        self.sleep_fn = sleep_fn
        self.rng = rng or random.Random(42)
        self.logger = logger or LOGGER

    def _headers(self) -> dict[str, str]:
        """Return request headers for the connector."""
        return {}

    def _auth_params(self) -> dict[str, str]:
        """Return default auth query params for the connector."""
        return {}

    def _log_request(self, endpoint: str, symbols: list[str], rows: int, elapsed_ms: float) -> None:
        self.logger.info(
            "vendor_request vendor=%s endpoint=%s symbols=%s rows=%s elapsed_ms=%.2f",
            self.vendor_name,
            endpoint,
            ",".join(symbols),
            rows,
            elapsed_ms,
        )

    def _sleep_duration(self, attempt: int) -> float:
        delay = min(
            self.retry_config.base_delay_seconds * (2 ** (attempt - 1)),
            self.retry_config.max_delay_seconds,
        )
        return delay + self.rng.uniform(0, delay / 4 if delay else 0.01)

    def _request(
        self,
        *,
        method: str,
        endpoint: str,
        base_url: str | None = None,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        task_kind: str = "OTHER",
        timeout: int = DEFAULT_TIMEOUT,
    ) -> requests.Response:
        params = {**self._auth_params(), **(params or {})}
        request_headers = {**self._headers(), **(headers or {})}

        for attempt in range(1, self.retry_config.max_attempts + 1):
            if not self.budget_manager.can_spend(self.vendor_name, task_kind=task_kind):
                raise TemporaryConnectorError(f"budget exhausted for vendor={self.vendor_name}")

            start = time.perf_counter()
            try:
                response = self.session.request(
                    method=method,
                    url=f"{(base_url or self.base_url).rstrip('/')}{endpoint}",
                    params=params,
                    headers=request_headers,
                    timeout=timeout,
                )
            except requests.RequestException as exc:
                self.budget_manager.record_spend(self.vendor_name, task_kind=task_kind)
                if attempt < self.retry_config.max_attempts:
                    self.sleep_fn(self._sleep_duration(attempt))
                    continue
                raise TemporaryConnectorError(f"{self.vendor_name} request failed: {exc}") from exc
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.budget_manager.record_spend(self.vendor_name, task_kind=task_kind)

            response_text = response.text[:512] if response.text else ""
            if "NOT_ENTITLED" in response_text or "NOT_SUPPORTED" in response_text:
                raise PermanentConnectorError(response_text)

            if response.status_code < 400:
                self._log_request(endpoint=endpoint, symbols=[], rows=0, elapsed_ms=elapsed_ms)
                return response

            if response.status_code in {429, 500, 502, 503, 504} and attempt < self.retry_config.max_attempts:
                self.sleep_fn(self._sleep_duration(attempt))
                continue
            if response.status_code in {429, 500, 502, 503, 504}:
                raise TemporaryConnectorError(f"{self.vendor_name} request failed: {response.status_code} {response_text}")
            raise PermanentConnectorError(f"{self.vendor_name} request failed: {response.status_code} {response_text}")

        raise TemporaryConnectorError(f"{self.vendor_name} request failed after retries")

    def request_json(
        self,
        *,
        endpoint: str,
        base_url: str | None = None,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        task_kind: str = "OTHER",
    ) -> dict[str, Any] | list[Any]:
        """Issue a JSON request."""
        response = self._request(
            method="GET",
            endpoint=endpoint,
            base_url=base_url,
            params=params,
            headers=headers,
            task_kind=task_kind,
        )
        return response.json()

    def request_csv(
        self,
        *,
        endpoint: str,
        base_url: str | None = None,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        task_kind: str = "OTHER",
    ) -> pd.DataFrame:
        """Issue a CSV request."""
        response = self._request(
            method="GET",
            endpoint=endpoint,
            base_url=base_url,
            params=params,
            headers=headers,
            task_kind=task_kind,
        )
        reader = csv.DictReader(io.StringIO(response.text))
        return pd.DataFrame(reader)
