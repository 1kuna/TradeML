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
        rpm=200,
        daily_cap=288000,
        docs_urls=(
            "https://docs.alpaca.markets/reference/stockbars",
            "https://docs.alpaca.markets/reference/getassets-1",
        ),
        notes="Use multi-symbol stock bars with pagination; free/basic historical data is IEX-backed at 200 calls/min with no documented daily cap.",
        datasets=(
            DatasetContract(
                dataset="equities_eod",
                endpoint_key="equities_eod",
                max_batch_symbols=100,
                pagination_mode="page_token",
                pagination_limit=10000,
                max_history_years=10,
                retry_after_header="X-RateLimit-Reset",
                critical_path_allowed=True,
                entitlement_failure_markers=(
                    "NOT_ENTITLED",
                    "subscription",
                    "not permitted",
                ),
                docs_urls=("https://docs.alpaca.markets/reference/stockbars",),
                notes="REST bars support multi-symbol requests, limit up to 10000, next_page_token pagination, and X-RateLimit-* headers.",
            ),
            DatasetContract(
                dataset="equities_minute",
                endpoint_key="equities_minute",
                max_batch_symbols=100,
                pagination_mode="page_token",
                pagination_limit=10000,
                max_history_years=1,
                retry_after_header="X-RateLimit-Reset",
                critical_path_allowed=False,
                docs_urls=("https://docs.alpaca.markets/reference/stockbars",),
                notes="The same stock-bars endpoint supports 1Min bars with symbols, timeframe, start/end, limit, and page_token; use it as the rolling minute archive lane.",
            ),
            DatasetContract(
                dataset="news",
                endpoint_key="news",
                max_batch_symbols=50,
                pagination_mode="page_token",
                retry_after_header="X-RateLimit-Reset",
                entitlement_failure_markers=(
                    "NOT_ENTITLED",
                    "subscription",
                    "not permitted",
                    "forbidden",
                ),
                docs_urls=("https://docs.alpaca.markets/reference/news-3",),
                notes="Alpaca News is the preferred free-key news replacement when the live audit verifies the account entitlement.",
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
                docs_urls=(
                    "https://docs.alpaca.markets/reference/corporateactionsannouncements-1",
                ),
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
                docs_urls=(
                    "https://www.tiingo.com/documentation/corporate-actions/dividends",
                ),
            ),
            DatasetContract(
                dataset="corp_actions_splits",
                endpoint_key="corp_actions_splits",
                max_batch_symbols=500,
                pagination_mode="none",
                docs_urls=(
                    "https://www.tiingo.com/documentation/corporate-actions/splits",
                ),
            ),
            DatasetContract(
                dataset="supported_tickers",
                endpoint_key="supported_tickers",
                max_batch_symbols=1,
                docs_urls=("https://www.tiingo.com/documentation/end-of-day",),
            ),
            DatasetContract(
                dataset="news",
                endpoint_key="news",
                max_batch_symbols=50,
                pagination_mode="none",
                entitlement_failure_markers=(
                    "permission to access the news api",
                    "not permitted",
                    "forbidden",
                    "subscription",
                ),
                docs_urls=("https://www.tiingo.com/documentation/news",),
                notes="Current free key is entitlement-blocked for Tiingo News; keep this lane audit-gated instead of retrying it in production.",
            ),
        ),
    ),
    "twelve_data": ProviderContract(
        vendor="twelve_data",
        rpm=8,
        daily_cap=800,
        docs_urls=(
            "https://support.twelvedata.com/en/articles/5203360-batch-api-requests",
            "https://support.twelvedata.com/en/articles/5609168-introduction-to-twelve-data",
            "https://support.twelvedata.com/en/articles/9935903-us-equities-market-data",
        ),
        notes="Basic/free entitlement is 8 API credits/min and 800/day; batch requests are weighted by symbol count in our runtime.",
        datasets=(
            DatasetContract(
                dataset="equities_eod",
                endpoint_key="equities_eod",
                max_batch_symbols=8,
                request_cost_basis="symbol",
                pagination_mode="none",
                max_history_years=10,
                critical_path_allowed=False,
                entitlement_failure_markers=(
                    "code",
                    "message",
                    "not available",
                    "not found",
                    "plan",
                ),
                docs_urls=(
                    "https://support.twelvedata.com/en/articles/5203360-batch-api-requests",
                    "https://support.twelvedata.com/en/articles/9935903-us-equities-market-data",
                    "https://api.twelvedata.com/doc/swagger/openapi.json",
                ),
                notes="Comma-separated symbols are supported where allowed; batch credits are consumed per requested endpoint and quota exhaustion can return partial results.",
            ),
            DatasetContract(
                dataset="equities_minute",
                endpoint_key="equities_minute",
                max_batch_symbols=1,
                request_cost_basis="symbol",
                pagination_mode="next_api_query",
                max_history_years=1,
                critical_path_allowed=False,
                docs_urls=(
                    "https://twelvedata.com/docs/llms/introduction.md",
                    "https://api.twelvedata.com/doc/swagger/openapi.json",
                ),
                notes="The time_series endpoint supports intraday intervals like 1min and returns next_api_query metadata, but free-plan credits are tight so this is a secondary minute-archive lane.",
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
        rpm=5,
        daily_cap=7200,
        docs_urls=(
            "https://polygon.io/docs/rest/stocks/aggregates/custom-bars",
            "https://polygon.io/docs/rest/stocks/tickers/all-tickers",
        ),
        notes="Basic/free entitlement is 5 calls/min; aggregates are ticker-scoped and should stay off the critical frozen-window path.",
        datasets=(
            DatasetContract(
                dataset="equities_eod",
                endpoint_key="equities_eod",
                max_batch_symbols=1,
                pagination_mode="next_url",
                pagination_limit=50000,
                max_history_years=10,
                critical_path_allowed=False,
                entitlement_failure_markers=(
                    "not authorized",
                    "upgrade",
                    "subscription",
                    "plan",
                ),
                docs_urls=(
                    "https://polygon.io/docs/rest/stocks/aggregates/custom-bars",
                ),
                notes="Custom bars use limit with next_url cursor pagination; max limit 50000 and default 5000.",
            ),
            DatasetContract(
                dataset="equities_minute",
                endpoint_key="equities_minute",
                max_batch_symbols=1,
                pagination_mode="next_url",
                pagination_limit=50000,
                max_history_years=2,
                critical_path_allowed=False,
                entitlement_failure_markers=(
                    "not authorized",
                    "upgrade",
                    "subscription",
                    "plan",
                ),
                docs_urls=(
                    "https://polygon.io/docs/rest/stocks/aggregates/custom-bars",
                ),
                notes="Minute aggregates are a low-rate independent QC/fill lane under the 5 calls/min basic limit.",
            ),
            DatasetContract(
                dataset="reference_splits",
                endpoint_key="reference_splits",
                max_batch_symbols=1,
                docs_urls=(
                    "https://polygon.io/docs/rest/stocks/corporate-actions/splits",
                ),
            ),
            DatasetContract(
                dataset="reference_dividends",
                endpoint_key="reference_dividends",
                max_batch_symbols=1,
                docs_urls=(
                    "https://polygon.io/docs/rest/stocks/corporate-actions/dividends",
                ),
            ),
            DatasetContract(
                dataset="reference_tickers",
                endpoint_key="reference_tickers",
                max_batch_symbols=1,
                pagination_mode="next_url",
                pagination_limit=1000,
                docs_urls=("https://polygon.io/docs/rest/stocks/tickers/all-tickers",),
                notes="Reference tickers paginate through next_url cursor links with max limit 1000.",
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
                docs_urls=(
                    "https://fred.stlouisfed.org/docs/api/fred/series_observations.html",
                ),
                notes="Observations support explicit limit and offset; use limit 100000 and ascending sort for deterministic pulls.",
            ),
            DatasetContract(
                dataset="vintagedates",
                endpoint_key="vintagedates",
                max_batch_symbols=1,
                pagination_mode="none",
                docs_urls=(
                    "https://fred.stlouisfed.org/docs/api/fred/series_vintagedates.html",
                ),
                notes="Vintage dates are series-scoped and support explicit limits; use large single pulls for current macro lanes.",
            ),
        ),
    ),
    "sec_edgar": ProviderContract(
        vendor="sec_edgar",
        rpm=600,
        daily_cap=864000,
        docs_urls=(
            "https://www.sec.gov/search-filings/edgar-application-programming-interfaces",
        ),
        notes="Respect SEC fair-access guidance at max 10 requests/sec, always send a valid user-agent, and fetch archived submission segments when present.",
        datasets=(
            DatasetContract(
                dataset="filing_index",
                endpoint_key="filing_index",
                max_batch_symbols=1,
                docs_urls=(
                    "https://www.sec.gov/search-filings/edgar-application-programming-interfaces",
                ),
            ),
            DatasetContract(
                dataset="companyfacts",
                endpoint_key="companyfacts",
                max_batch_symbols=1,
                docs_urls=(
                    "https://www.sec.gov/search-filings/edgar-application-programming-interfaces",
                ),
            ),
            DatasetContract(
                dataset="submissions",
                endpoint_key="submissions",
                max_batch_symbols=1,
                docs_urls=(
                    "https://www.sec.gov/search-filings/edgar-application-programming-interfaces",
                ),
            ),
        ),
    ),
    "fmp": ProviderContract(
        vendor="fmp",
        rpm=1,
        daily_cap=250,
        docs_urls=("https://site.financialmodelingprep.com/developer/docs",),
        notes="Treat FMP basic/free as 250 calls/day and reference-only in this runtime.",
        datasets=(
            DatasetContract(
                dataset="delistings",
                endpoint_key="delistings",
                max_batch_symbols=1,
                pagination_mode="page",
                pagination_limit=100,
                docs_urls=(
                    "https://site.financialmodelingprep.com/developer/docs/stable-delisted-companies-api",
                ),
                notes="Stable delisted-companies uses page/limit pagination with documented limit 100.",
            ),
            DatasetContract(
                dataset="symbol_changes",
                endpoint_key="symbol_changes",
                max_batch_symbols=1,
                docs_urls=(
                    "https://site.financialmodelingprep.com/developer/docs/stable-symbol-change-api",
                ),
            ),
            DatasetContract(
                dataset="stock_news",
                endpoint_key="stock_news",
                max_batch_symbols=50,
                pagination_mode="page",
                docs_urls=("https://site.financialmodelingprep.com/developer/docs/",),
                notes="Stable stock-news endpoints are live-audit gated and capped by the free 250 calls/day budget.",
            ),
            DatasetContract(
                dataset="press_releases",
                endpoint_key="press_releases",
                max_batch_symbols=50,
                pagination_mode="page",
                docs_urls=("https://site.financialmodelingprep.com/developer/docs/",),
                notes="Stable press-release news endpoints are live-audit gated and capped by the free 250 calls/day budget.",
            ),
        ),
    ),
    "finnhub": ProviderContract(
        vendor="finnhub",
        rpm=57,
        daily_cap=82080,
        docs_urls=("https://finnhub.io/docs/api",),
        notes="Use as supplemental reference/research lane, not canonical bar closer; candle endpoint documents s=no_data as a valid empty response.",
        datasets=(
            DatasetContract(
                dataset="equities_eod",
                endpoint_key="equities_eod",
                max_batch_symbols=1,
                critical_path_allowed=False,
                docs_urls=("https://finnhub.io/docs/api/stock-candles",),
            ),
            DatasetContract(
                dataset="company_news",
                endpoint_key="company_news",
                max_batch_symbols=1,
                docs_urls=("https://finnhub.io/docs/api/company-news",),
                notes="Company news is symbol-scoped and date-bounded, making it a useful supplemental historical news archive lane.",
            ),
            DatasetContract(
                dataset="earnings_calendar",
                endpoint_key="earnings_calendar",
                max_batch_symbols=1,
                docs_urls=("https://finnhub.io/docs/api/company-earnings-calendar",),
            ),
            DatasetContract(
                dataset="profile",
                endpoint_key="profile",
                max_batch_symbols=1,
                docs_urls=("https://finnhub.io/docs/api/company-profile2",),
            ),
        ),
    ),
    "alpha_vantage": ProviderContract(
        vendor="alpha_vantage",
        rpm=1,
        daily_cap=25,
        docs_urls=("https://www.alphavantage.co/documentation/",),
        notes="Free tier remains very low-throughput at 25 requests/day; reference-only in this runtime.",
        datasets=(
            DatasetContract(
                dataset="listings",
                endpoint_key="listings",
                max_batch_symbols=1,
                docs_urls=("https://www.alphavantage.co/documentation/",),
                notes="LISTING_STATUS returns CSV and accepts optional date/state filters.",
            ),
            DatasetContract(
                dataset="corp_actions",
                endpoint_key="corp_actions",
                max_batch_symbols=1,
                docs_urls=("https://www.alphavantage.co/documentation/",),
                notes="DIVIDENDS and SPLITS are single-symbol lanes and may return named top-level arrays or CSV depending on datatype.",
            ),
            DatasetContract(
                dataset="news_sentiment",
                endpoint_key="news_sentiment",
                max_batch_symbols=10,
                pagination_mode="none",
                docs_urls=("https://www.alphavantage.co/documentation/",),
                notes="NEWS_SENTIMENT supports tickers, time_from, time_to, and limit up to 1000; free daily budget makes it a low-rate supplemental news lane.",
            ),
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
