"""Docs-backed provider contracts for connector/runtime policy."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True, frozen=True)
class DatasetContract:
    """Docs-backed policy for one provider dataset lane."""

    dataset: str
    endpoint_key: str
    max_batch_symbols: int
    request_cost_units: int = 1
    request_cost_basis: str = "request"
    pagination_mode: str = "none"
    pagination_limit: int | None = None
    history_floor_policy: str = "observed_or_listing"
    max_history_years: int | None = None
    retryable_statuses: tuple[int, ...] = (429, 500, 502, 503, 504)
    retry_after_header: str | None = "Retry-After"
    entitlement_failure_markers: tuple[str, ...] = ()
    empty_result_policy: str = "valid_empty"
    critical_path_allowed: bool = False
    docs_urls: tuple[str, ...] = ()
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize the contract for audits and dashboards."""
        payload = asdict(self)
        payload["retryable_statuses"] = list(self.retryable_statuses)
        payload["entitlement_failure_markers"] = list(self.entitlement_failure_markers)
        payload["docs_urls"] = list(self.docs_urls)
        return payload


@dataclass(slots=True, frozen=True)
class ProviderContract:
    """Docs-backed provider-level request contract."""

    vendor: str
    rpm: int
    daily_cap: int
    docs_urls: tuple[str, ...]
    notes: str = ""
    datasets: tuple[DatasetContract, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize the provider contract."""
        payload = asdict(self)
        payload["docs_urls"] = list(self.docs_urls)
        payload["datasets"] = [dataset.to_dict() for dataset in self.datasets]
        return payload


_PROVIDER_CONTRACTS: dict[str, ProviderContract] = {
    "alpaca": ProviderContract(
        vendor="alpaca",
        rpm=190,
        daily_cap=273600,
        docs_urls=(
            "https://docs.alpaca.markets/reference/stockbars",
            "https://docs.alpaca.markets/reference/getassets-1",
        ),
        notes="Use multi-symbol stock bars with pagination; prioritize Alpaca on canonical frozen windows.",
        datasets=(
            DatasetContract(
                dataset="equities_eod",
                endpoint_key="equities_eod",
                max_batch_symbols=100,
                pagination_mode="page_token",
                pagination_limit=10000,
                max_history_years=10,
                critical_path_allowed=True,
                entitlement_failure_markers=("NOT_ENTITLED", "subscription", "not permitted"),
                docs_urls=("https://docs.alpaca.markets/reference/stockbars",),
                notes="REST bars support multi-symbol requests and next_page_token pagination.",
            ),
            DatasetContract(
                dataset="assets",
                endpoint_key="assets",
                max_batch_symbols=1,
                docs_urls=("https://docs.alpaca.markets/reference/getassets-1",),
            ),
            DatasetContract(
                dataset="corp_actions",
                endpoint_key="corp_actions",
                max_batch_symbols=1,
                docs_urls=("https://docs.alpaca.markets/reference/corporateactionsannouncements-1",),
            ),
        ),
    ),
    "tiingo": ProviderContract(
        vendor="tiingo",
        rpm=158,
        daily_cap=95000,
        docs_urls=(
            "https://www.tiingo.com/documentation/end-of-day",
            "https://www.tiingo.com/documentation/corporate-actions/dividends",
            "https://www.tiingo.com/documentation/corporate-actions/splits",
        ),
        notes="Daily price endpoint is ticker-specific; adjusted fields are returned alongside raw OHLCV.",
        datasets=(
            DatasetContract(
                dataset="equities_eod",
                endpoint_key="equities_eod",
                max_batch_symbols=1,
                pagination_mode="none",
                max_history_years=30,
                critical_path_allowed=True,
                docs_urls=("https://www.tiingo.com/documentation/end-of-day",),
            ),
            DatasetContract(
                dataset="corp_actions_dividends",
                endpoint_key="corp_actions_dividends",
                max_batch_symbols=500,
                pagination_mode="none",
                docs_urls=("https://www.tiingo.com/documentation/corporate-actions/dividends",),
            ),
            DatasetContract(
                dataset="corp_actions_splits",
                endpoint_key="corp_actions_splits",
                max_batch_symbols=500,
                pagination_mode="none",
                docs_urls=("https://www.tiingo.com/documentation/corporate-actions/splits",),
            ),
            DatasetContract(
                dataset="supported_tickers",
                endpoint_key="supported_tickers",
                max_batch_symbols=1,
                docs_urls=("https://www.tiingo.com/documentation/end-of-day",),
            ),
        ),
    ),
    "twelve_data": ProviderContract(
        vendor="twelve_data",
        rpm=7,
        daily_cap=760,
        docs_urls=(
            "https://support.twelvedata.com/en/articles/5203360-batch-api-requests",
            "https://support.twelvedata.com/en/articles/5609168-introduction-to-twelve-data",
            "https://support.twelvedata.com/en/articles/9935903-us-equities-market-data",
        ),
        notes="Free/basic plans are credit-sensitive; batch requests are weighted by symbol count in our runtime.",
        datasets=(
            DatasetContract(
                dataset="equities_eod",
                endpoint_key="equities_eod",
                max_batch_symbols=8,
                request_cost_basis="symbol",
                pagination_mode="none",
                max_history_years=10,
                critical_path_allowed=False,
                entitlement_failure_markers=("code", "message", "not available", "not found", "plan"),
                docs_urls=(
                    "https://support.twelvedata.com/en/articles/5203360-batch-api-requests",
                    "https://support.twelvedata.com/en/articles/9935903-us-equities-market-data",
                ),
            ),
            DatasetContract(
                dataset="dividends",
                endpoint_key="dividends",
                max_batch_symbols=1,
                docs_urls=("https://twelvedata.com/docs#dividends",),
            ),
            DatasetContract(
                dataset="splits",
                endpoint_key="splits",
                max_batch_symbols=1,
                docs_urls=("https://twelvedata.com/docs#splits",),
            ),
        ),
    ),
    "massive": ProviderContract(
        vendor="massive",
        rpm=4,
        daily_cap=6840,
        docs_urls=(
            "https://polygon.io/docs/rest/stocks/aggregates/custom-bars",
            "https://polygon.io/docs/rest/stocks/tickers/all-tickers",
        ),
        notes="Aggregates are ticker-scoped; basic plans should stay off the critical frozen-window path.",
        datasets=(
            DatasetContract(
                dataset="equities_eod",
                endpoint_key="equities_eod",
                max_batch_symbols=1,
                pagination_mode="limit",
                pagination_limit=50000,
                max_history_years=10,
                critical_path_allowed=False,
                entitlement_failure_markers=("not authorized", "upgrade", "subscription", "plan"),
                docs_urls=("https://polygon.io/docs/rest/stocks/aggregates/custom-bars",),
            ),
            DatasetContract(
                dataset="reference_splits",
                endpoint_key="reference_splits",
                max_batch_symbols=1,
                docs_urls=("https://polygon.io/docs/rest/stocks/corporate-actions/splits",),
            ),
            DatasetContract(
                dataset="reference_dividends",
                endpoint_key="reference_dividends",
                max_batch_symbols=1,
                docs_urls=("https://polygon.io/docs/rest/stocks/corporate-actions/dividends",),
            ),
            DatasetContract(
                dataset="reference_tickers",
                endpoint_key="reference_tickers",
                max_batch_symbols=1,
                pagination_mode="limit",
                pagination_limit=1000,
                docs_urls=("https://polygon.io/docs/rest/stocks/tickers/all-tickers",),
            ),
        ),
    ),
    "fred": ProviderContract(
        vendor="fred",
        rpm=114,
        daily_cap=164160,
        docs_urls=(
            "https://fred.stlouisfed.org/docs/api/fred/series_observations.html",
            "https://fred.stlouisfed.org/docs/api/fred/series_vintagedates.html",
        ),
        notes="Macro observations and vintages are separate lanes; vintages remain phase-blocking reference data.",
        datasets=(
            DatasetContract(
                dataset="macros_treasury",
                endpoint_key="macros_treasury",
                max_batch_symbols=1,
                pagination_mode="none",
                docs_urls=("https://fred.stlouisfed.org/docs/api/fred/series_observations.html",),
            ),
            DatasetContract(
                dataset="vintagedates",
                endpoint_key="vintagedates",
                max_batch_symbols=1,
                pagination_mode="none",
                docs_urls=("https://fred.stlouisfed.org/docs/api/fred/series_vintagedates.html",),
            ),
        ),
    ),
    "sec_edgar": ProviderContract(
        vendor="sec_edgar",
        rpm=570,
        daily_cap=820800,
        docs_urls=("https://www.sec.gov/search-filings/edgar-application-programming-interfaces",),
        notes="Respect SEC fair-access guidance and always send a valid user-agent.",
        datasets=(
            DatasetContract(
                dataset="filing_index",
                endpoint_key="filing_index",
                max_batch_symbols=1,
                docs_urls=("https://www.sec.gov/search-filings/edgar-application-programming-interfaces",),
            ),
            DatasetContract(
                dataset="companyfacts",
                endpoint_key="companyfacts",
                max_batch_symbols=1,
                docs_urls=("https://www.sec.gov/search-filings/edgar-application-programming-interfaces",),
            ),
            DatasetContract(
                dataset="submissions",
                endpoint_key="submissions",
                max_batch_symbols=1,
                docs_urls=("https://www.sec.gov/search-filings/edgar-application-programming-interfaces",),
            ),
        ),
    ),
    "fmp": ProviderContract(
        vendor="fmp",
        rpm=1,
        daily_cap=237,
        docs_urls=("https://site.financialmodelingprep.com/developer/docs",),
        notes="Treat FMP as low-throughput reference-only in this runtime.",
        datasets=(
            DatasetContract(dataset="delistings", endpoint_key="delistings", max_batch_symbols=1, docs_urls=("https://site.financialmodelingprep.com/developer/docs/stable-delisted-companies-api",)),
            DatasetContract(dataset="symbol_changes", endpoint_key="symbol_changes", max_batch_symbols=1, docs_urls=("https://site.financialmodelingprep.com/developer/docs/stable-symbol-change-api",)),
        ),
    ),
    "finnhub": ProviderContract(
        vendor="finnhub",
        rpm=57,
        daily_cap=82080,
        docs_urls=("https://finnhub.io/docs/api",),
        notes="Use as supplemental reference/research lane, not canonical bar closer.",
        datasets=(
            DatasetContract(dataset="equities_eod", endpoint_key="equities_eod", max_batch_symbols=1, critical_path_allowed=False, docs_urls=("https://finnhub.io/docs/api/stock-candles",)),
            DatasetContract(dataset="earnings_calendar", endpoint_key="earnings_calendar", max_batch_symbols=1, docs_urls=("https://finnhub.io/docs/api/company-earnings-calendar",)),
            DatasetContract(dataset="profile", endpoint_key="profile", max_batch_symbols=1, docs_urls=("https://finnhub.io/docs/api/company-profile2",)),
        ),
    ),
    "alpha_vantage": ProviderContract(
        vendor="alpha_vantage",
        rpm=1,
        daily_cap=23,
        docs_urls=("https://www.alphavantage.co/documentation/",),
        notes="Free tier remains very low-throughput; reference-only in this runtime.",
        datasets=(
            DatasetContract(dataset="listings", endpoint_key="listings", max_batch_symbols=1, docs_urls=("https://www.alphavantage.co/documentation/",)),
            DatasetContract(dataset="corp_actions", endpoint_key="corp_actions", max_batch_symbols=1, docs_urls=("https://www.alphavantage.co/documentation/",)),
        ),
    ),
}


def provider_contract(vendor: str) -> ProviderContract | None:
    """Return the provider contract for a vendor."""
    return _PROVIDER_CONTRACTS.get(vendor)


def dataset_contract(vendor: str, dataset: str) -> DatasetContract | None:
    """Return the dataset contract for a provider dataset lane."""
    contract = provider_contract(vendor)
    if contract is None:
        return None
    for candidate in contract.datasets:
        if candidate.dataset == dataset or candidate.endpoint_key == dataset:
            return candidate
    return None


def provider_contract_rows() -> list[dict[str, Any]]:
    """Return provider contracts serialized for docs/dashboard use."""
    return [contract.to_dict() for _, contract in sorted(_PROVIDER_CONTRACTS.items())]


def default_vendor_limits() -> dict[str, dict[str, int]]:
    """Return docs-backed default vendor limits derived from the provider contracts."""
    return {
        vendor: {
            "rpm": int(contract.rpm),
            "daily_cap": int(contract.daily_cap),
        }
        for vendor, contract in sorted(_PROVIDER_CONTRACTS.items())
    }


def clone_provider_contract_rows() -> list[dict[str, Any]]:
    """Return a deep-copied provider contract table for mutation-safe consumers."""
    return deepcopy(provider_contract_rows())
