from __future__ import annotations

from trademl.data_node.budgets import BudgetManager
from trademl.data_node.runtime import build_connectors, resolve_vendor_budgets


def test_resolve_vendor_budgets_keeps_defaults_for_new_connectors() -> None:
    budgets = resolve_vendor_budgets(
        {
            "vendors": {
                "alpaca": {"rpm": 200, "daily_cap": 20000},
            }
        }
    )

    assert budgets["alpaca"] == {"rpm": 200, "daily_cap": 20000}
    assert budgets["tiingo"] == {"rpm": 158, "daily_cap": 95000}
    assert budgets["twelve_data"] == {"rpm": 7, "daily_cap": 760}


def test_build_connectors_uses_endpoint_compatible_base_urls() -> None:
    connectors = build_connectors(
        env_values={
            "ALPACA_API_KEY": "alpaca-key",
            "ALPACA_API_SECRET": "alpaca-secret",
            "FRED_API_KEY": "fred-key",
            "FINNHUB_API_KEY": "finnhub-key",
            "FMP_API_KEY": "fmp-key",
            "SEC_EDGAR_USER_AGENT": "TradeML/1.0 test@example.com",
        },
        vendor_limits=resolve_vendor_budgets({"vendors": {}}),
    )

    assert connectors["alpaca"].base_url == "https://data.alpaca.markets"
    assert connectors["fred"].base_url == "https://api.stlouisfed.org"
    assert connectors["finnhub"].base_url == "https://finnhub.io"
    assert connectors["fmp"].base_url == "https://financialmodelingprep.com"
    assert connectors["sec_edgar"].base_url == "https://data.sec.gov"


def test_build_connectors_can_share_one_budget_manager() -> None:
    shared_budget_manager = BudgetManager(resolve_vendor_budgets({"vendors": {}}))

    connectors = build_connectors(
        env_values={
            "ALPACA_API_KEY": "alpaca-key",
            "ALPACA_API_SECRET": "alpaca-secret",
            "TIINGO_API_KEY": "tiingo-key",
        },
        vendor_limits=resolve_vendor_budgets({"vendors": {}}),
        budget_manager_factory=lambda _vendor: shared_budget_manager,
    )

    assert connectors["alpaca"].budget_manager is shared_budget_manager
    assert connectors["tiingo"].budget_manager is shared_budget_manager
