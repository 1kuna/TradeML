"""Base protocols and retry-aware HTTP helpers for vendor connectors."""

from __future__ import annotations

import csv
import email.utils
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
from trademl.data_node.provider_contracts import dataset_contract, provider_contract


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

    def _rate_limit_error(self) -> TemporaryConnectorError:
        """Return the normalized retryable error for vendor budget/rate exhaustion."""
        return TemporaryConnectorError(f"budget exhausted for vendor={self.vendor_name}")

    def _entitlement_failure_markers(self, endpoint_key: str | None = None) -> tuple[str, ...]:
        """Return known entitlement or unsupported-plan markers for an endpoint."""
        markers: list[str] = ["NOT_ENTITLED", "NOT_SUPPORTED"]
        provider = provider_contract(self.vendor_name)
        if provider is not None:
            for contract in provider.datasets:
                if endpoint_key is not None and contract.endpoint_key != endpoint_key and contract.dataset != endpoint_key:
                    continue
                markers.extend(contract.entitlement_failure_markers)
        return tuple(marker for marker in markers if marker)

    def _retryable_statuses(self, endpoint_key: str | None = None) -> tuple[int, ...]:
        """Return the documented retryable status codes for an endpoint."""
        contract = dataset_contract(self.vendor_name, endpoint_key or "")
        if contract is not None:
            return contract.retryable_statuses
        return (429, 500, 502, 503, 504)

    def _retry_after_header(self, endpoint_key: str | None = None) -> str | None:
        """Return the documented retry-after or reset header for an endpoint."""
        contract = dataset_contract(self.vendor_name, endpoint_key or "")
        if contract is not None:
            return contract.retry_after_header
        return "Retry-After"

    def _retry_after_seconds(self, response: requests.Response, endpoint_key: str | None = None) -> float | None:
        """Parse a documented retry-after/reset header into seconds."""
        header_name = self._retry_after_header(endpoint_key)
        if not header_name:
            return None
        raw_value = response.headers.get(header_name)
        if raw_value is None:
            return None
        text = str(raw_value).strip()
        if not text:
            return None
        if header_name.lower() == "x-ratelimit-reset":
            try:
                reset_epoch = float(text)
            except ValueError:
                return None
            return max(0.0, reset_epoch - time.time())
        try:
            return max(0.0, float(text))
        except ValueError:
            try:
                parsed = email.utils.parsedate_to_datetime(text)
            except (TypeError, ValueError, IndexError):
                return None
            return max(0.0, parsed.timestamp() - time.time())

    def _request(
        self,
        *,
        method: str,
        endpoint: str,
        endpoint_key: str | None = None,
        absolute_url: str | None = None,
        base_url: str | None = None,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        task_kind: str = "OTHER",
        budget_units: int = 1,
        logical_units: int = 1,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> requests.Response:
        params = {**self._auth_params(), **(params or {})}
        request_headers = {**self._headers(), **(headers or {})}
        normalized_units = max(1, int(budget_units))
        normalized_logical_units = max(0, int(logical_units))
        telemetry_key = endpoint_key or endpoint
        retryable_statuses = set(self._retryable_statuses(endpoint_key))
        entitlement_markers = tuple(marker.lower() for marker in self._entitlement_failure_markers(endpoint_key))

        for attempt in range(1, self.retry_config.max_attempts + 1):
            if not self.budget_manager.can_spend(self.vendor_name, task_kind=task_kind, units=normalized_units):
                self.budget_manager.record_local_budget_block(self.vendor_name, endpoint=telemetry_key)
                raise TemporaryConnectorError(f"budget exhausted for vendor={self.vendor_name}")

            start = time.perf_counter()
            try:
                response = self.session.request(
                    method=method,
                    url=absolute_url or f"{(base_url or self.base_url).rstrip('/')}{endpoint}",
                    params=params,
                    headers=request_headers,
                    timeout=timeout,
                )
            except requests.RequestException as exc:
                self.budget_manager.record_spend(
                    self.vendor_name,
                    task_kind=task_kind,
                    units=normalized_units,
                    endpoint=telemetry_key,
                    logical_units=normalized_logical_units,
                )
                if attempt < self.retry_config.max_attempts:
                    self.sleep_fn(self._sleep_duration(attempt))
                    continue
                raise TemporaryConnectorError(f"{self.vendor_name} request failed: {exc}") from exc
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.budget_manager.record_spend(
                self.vendor_name,
                task_kind=task_kind,
                units=normalized_units,
                endpoint=telemetry_key,
                logical_units=normalized_logical_units,
            )

            response_text = response.text[:512] if response.text else ""
            if any(marker in response_text.lower() for marker in entitlement_markers):
                self.budget_manager.record_permanent_failure(self.vendor_name, endpoint=telemetry_key)
                raise PermanentConnectorError(response_text)

            if response.status_code < 400:
                self._log_request(endpoint=endpoint, symbols=[], rows=0, elapsed_ms=elapsed_ms)
                return response

            if response.status_code == 429:
                self.budget_manager.record_remote_rate_limit(self.vendor_name, endpoint=telemetry_key)
                if attempt < self.retry_config.max_attempts:
                    self.sleep_fn(self._retry_after_seconds(response, endpoint_key) or self._sleep_duration(attempt))
                    continue
                raise self._rate_limit_error()
            if response.status_code in retryable_statuses and attempt < self.retry_config.max_attempts:
                self.sleep_fn(self._retry_after_seconds(response, endpoint_key) or self._sleep_duration(attempt))
                continue
            if response.status_code in retryable_statuses:
                raise TemporaryConnectorError(f"{self.vendor_name} request failed: {response.status_code} {response_text}")
            self.budget_manager.record_permanent_failure(self.vendor_name, endpoint=telemetry_key)
            raise PermanentConnectorError(f"{self.vendor_name} request failed: {response.status_code} {response_text}")

        raise TemporaryConnectorError(f"{self.vendor_name} request failed after retries")

    def request_json(
        self,
        *,
        endpoint: str,
        endpoint_key: str | None = None,
        absolute_url: str | None = None,
        base_url: str | None = None,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        task_kind: str = "OTHER",
        budget_units: int = 1,
        logical_units: int = 1,
    ) -> dict[str, Any] | list[Any]:
        """Issue a JSON request."""
        response = self._request(
            method="GET",
            endpoint=endpoint,
            endpoint_key=endpoint_key,
            absolute_url=absolute_url,
            base_url=base_url,
            params=params,
            headers=headers,
            task_kind=task_kind,
            budget_units=budget_units,
            logical_units=logical_units,
        )
        return response.json()

    def request_csv(
        self,
        *,
        endpoint: str,
        endpoint_key: str | None = None,
        absolute_url: str | None = None,
        base_url: str | None = None,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        task_kind: str = "OTHER",
        budget_units: int = 1,
        logical_units: int = 1,
    ) -> pd.DataFrame:
        """Issue a CSV request."""
        response = self._request(
            method="GET",
            endpoint=endpoint,
            endpoint_key=endpoint_key,
            absolute_url=absolute_url,
            base_url=base_url,
            params=params,
            headers=headers,
            task_kind=task_kind,
            budget_units=budget_units,
            logical_units=logical_units,
        )
        reader = csv.DictReader(io.StringIO(response.text))
        return pd.DataFrame(reader)

    def request_json_url(
        self,
        *,
        url: str,
        endpoint_key: str | None = None,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        task_kind: str = "OTHER",
        budget_units: int = 1,
        logical_units: int = 1,
    ) -> dict[str, Any] | list[Any]:
        """Issue a JSON request against an absolute URL."""
        return self.request_json(
            endpoint="",
            endpoint_key=endpoint_key,
            absolute_url=url,
            params=params,
            headers=headers,
            task_kind=task_kind,
            budget_units=budget_units,
            logical_units=logical_units,
        )
