from __future__ import annotations

from trademl.data_node.__main__ import _resolve_vendor_budgets


def test_resolve_vendor_budgets_keeps_defaults_for_new_connectors() -> None:
    budgets = _resolve_vendor_budgets(
        {
            "vendors": {
                "alpaca": {"rpm": 200, "daily_cap": 20000},
            }
        }
    )

    assert budgets["alpaca"] == {"rpm": 200, "daily_cap": 20000}
    assert budgets["tiingo"] == {"rpm": 40, "daily_cap": 400}
    assert budgets["twelve_data"] == {"rpm": 6, "daily_cap": 600}
