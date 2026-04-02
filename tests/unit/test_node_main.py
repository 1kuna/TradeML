from __future__ import annotations

from trademl.data_node.__main__ import _build_reference_jobs, _resolve_vendor_budgets


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


def test_build_reference_jobs_uses_verified_defaults_and_caps_symbol_fanout() -> None:
    jobs = _build_reference_jobs(
        connectors={
            "alpaca": object(),
            "massive": object(),
            "alpha_vantage": object(),
            "tiingo": object(),
            "twelve_data": object(),
            "finnhub": object(),
            "fmp": object(),
            "sec_edgar": object(),
        },
        symbols=["AAPL", "MSFT", "NVDA"],
    )

    assert all(job["dataset"] != "supported_tickers" for job in jobs)
    assert all(job["source"] != "tiingo" for job in jobs)
    assert all(job["dataset"] != "symbol_changes" for job in jobs)

    finnhub_profiles = next(job for job in jobs if job["source"] == "finnhub" and job["dataset"] == "company_profile")
    assert finnhub_profiles["max_symbols_per_run"] == 50

    twelve_financials = next(
        job for job in jobs if job["source"] == "twelve_data" and job["dataset"] == "financial_statements"
    )
    assert twelve_financials["max_symbols_per_run"] == 20
