"""Shared runtime helpers for vendor budgets and connector construction."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from trademl.connectors.alpaca import AlpacaConnector
from trademl.connectors.alpha_vantage import AlphaVantageConnector
from trademl.connectors.base import BaseConnector
from trademl.connectors.finnhub import FinnhubConnector
from trademl.connectors.fmp import FMPConnector
from trademl.connectors.fred import FredConnector
from trademl.connectors.massive import MassiveConnector
from trademl.connectors.sec_edgar import SecEdgarConnector
from trademl.connectors.tiingo import TiingoConnector
from trademl.connectors.twelve_data import TwelveDataConnector
from trademl.data_node.budgets import BudgetManager
from trademl.data_node.vendor_limits import DEFAULT_VENDOR_LIMITS

RUNTIME_VENDOR_ORDER: tuple[str, ...] = (
    "alpaca",
    "tiingo",
    "twelve_data",
    "massive",
    "finnhub",
    "alpha_vantage",
    "fred",
    "fmp",
    "sec_edgar",
)


def resolve_vendor_budgets(config: Mapping[str, object]) -> dict[str, dict[str, int]]:
    """Resolve configured vendor budgets with researched defaults as fallback."""
    resolved = {name: limits.copy() for name, limits in DEFAULT_VENDOR_LIMITS.items()}
    for vendor, values in (config.get("vendors", {}) or {}).items():
        if not isinstance(values, Mapping):
            continue
        existing = resolved.get(str(vendor), {"rpm": 1, "daily_cap": 1})
        resolved[str(vendor)] = {
            "rpm": int(values.get("rpm", existing["rpm"])),
            "daily_cap": int(values.get("daily_cap", existing["daily_cap"])),
        }
    return resolved


def resolve_vendor_budget(vendor_limits: Mapping[str, Mapping[str, int]], vendor: str) -> dict[str, int]:
    """Resolve one vendor budget from limits with defaults as fallback."""
    source = dict(vendor_limits.get(vendor, DEFAULT_VENDOR_LIMITS[vendor]))
    return {
        "rpm": int(source.get("rpm", DEFAULT_VENDOR_LIMITS[vendor]["rpm"])),
        "daily_cap": int(source.get("daily_cap", DEFAULT_VENDOR_LIMITS[vendor]["daily_cap"])),
    }


def build_connector(
    *,
    vendor: str,
    env_values: Mapping[str, str],
    vendor_limits: Mapping[str, Mapping[str, int]],
    budget_manager_factory: Callable[[str], BudgetManager] | None = None,
    sec_edgar_user_agent: str | None = None,
) -> BaseConnector | None:
    """Build one connector when the required credentials are present."""

    def _budget_manager(target_vendor: str) -> BudgetManager:
        if budget_manager_factory is not None:
            return budget_manager_factory(target_vendor)
        return BudgetManager({target_vendor: resolve_vendor_budget(vendor_limits, target_vendor)})

    if vendor == "alpaca":
        if not env_values.get("ALPACA_API_KEY"):
            return None
        return AlpacaConnector(
            base_url=env_values.get("ALPACA_DATA_BASE_URL", "https://data.alpaca.markets"),
            trading_base_url=env_values.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets/v2"),
            api_key=env_values.get("ALPACA_API_KEY", ""),
            secret_key=env_values.get("ALPACA_API_SECRET", ""),
            budget_manager=_budget_manager("alpaca"),
        )
    if vendor == "tiingo":
        if not env_values.get("TIINGO_API_KEY"):
            return None
        return TiingoConnector(
            base_url=env_values.get("TIINGO_BASE_URL", "https://api.tiingo.com"),
            api_key=env_values.get("TIINGO_API_KEY", ""),
            budget_manager=_budget_manager("tiingo"),
        )
    if vendor == "twelve_data":
        if not env_values.get("TWELVE_DATA_API_KEY"):
            return None
        return TwelveDataConnector(
            base_url=env_values.get("TWELVE_DATA_BASE_URL", "https://api.twelvedata.com"),
            api_key=env_values.get("TWELVE_DATA_API_KEY", ""),
            budget_manager=_budget_manager("twelve_data"),
        )
    if vendor == "massive":
        if not env_values.get("MASSIVE_API_KEY"):
            return None
        return MassiveConnector(
            base_url=env_values.get("MASSIVE_BASE_URL", "https://api.polygon.io"),
            api_key=env_values.get("MASSIVE_API_KEY", ""),
            budget_manager=_budget_manager("massive"),
        )
    if vendor == "finnhub":
        if not env_values.get("FINNHUB_API_KEY"):
            return None
        return FinnhubConnector(
            base_url=env_values.get("FINNHUB_BASE_URL", "https://finnhub.io"),
            api_key=env_values.get("FINNHUB_API_KEY", ""),
            budget_manager=_budget_manager("finnhub"),
        )
    if vendor == "alpha_vantage":
        if not env_values.get("ALPHA_VANTAGE_API_KEY"):
            return None
        return AlphaVantageConnector(
            base_url=env_values.get("ALPHA_VANTAGE_BASE_URL", "https://www.alphavantage.co"),
            api_key=env_values.get("ALPHA_VANTAGE_API_KEY", ""),
            budget_manager=_budget_manager("alpha_vantage"),
        )
    if vendor == "fred":
        if not env_values.get("FRED_API_KEY"):
            return None
        return FredConnector(
            base_url=env_values.get("FRED_BASE_URL", "https://api.stlouisfed.org"),
            api_key=env_values.get("FRED_API_KEY", ""),
            budget_manager=_budget_manager("fred"),
        )
    if vendor == "fmp":
        if not env_values.get("FMP_API_KEY"):
            return None
        return FMPConnector(
            base_url=env_values.get("FMP_BASE_URL", "https://financialmodelingprep.com"),
            api_key=env_values.get("FMP_API_KEY", ""),
            budget_manager=_budget_manager("fmp"),
        )
    if vendor == "sec_edgar":
        user_agent = env_values.get("SEC_EDGAR_USER_AGENT") or sec_edgar_user_agent
        if not user_agent:
            return None
        return SecEdgarConnector(
            base_url=env_values.get("SEC_EDGAR_BASE_URL", "https://data.sec.gov"),
            user_agent=user_agent,
            budget_manager=_budget_manager("sec_edgar"),
        )
    return None


def build_connectors(
    *,
    env_values: Mapping[str, str],
    vendor_limits: Mapping[str, Mapping[str, int]],
    vendors: Sequence[str] | None = None,
    budget_manager_factory: Callable[[str], BudgetManager] | None = None,
    sec_edgar_user_agent: str | None = None,
) -> dict[str, BaseConnector]:
    """Build all configured connectors for the runtime."""
    connectors: dict[str, BaseConnector] = {}
    for vendor in vendors or RUNTIME_VENDOR_ORDER:
        connector = build_connector(
            vendor=vendor,
            env_values=env_values,
            vendor_limits=vendor_limits,
            budget_manager_factory=budget_manager_factory,
            sec_edgar_user_agent=sec_edgar_user_agent,
        )
        if connector is not None:
            connectors[vendor] = connector
    return connectors
