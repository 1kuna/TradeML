from __future__ import annotations

from pathlib import Path

from trademl.data_node.__main__ import _build_reference_jobs, _resolve_vendor_budgets, _should_rebuild_local_state


def test_resolve_vendor_budgets_keeps_defaults_for_new_connectors() -> None:
    budgets = _resolve_vendor_budgets(
        {
            "vendors": {
                "alpaca": {"rpm": 200, "daily_cap": 20000},
            }
        }
    )

    assert budgets["alpaca"] == {"rpm": 200, "daily_cap": 20000}
    assert budgets["tiingo"] == {"rpm": 158, "daily_cap": 95000}
    assert budgets["twelve_data"] == {"rpm": 7, "daily_cap": 760}


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

    assert all(job["dataset"] != "symbol_changes" for job in jobs)

    tiingo_supported = next(job for job in jobs if job["source"] == "tiingo" and job["dataset"] == "supported_tickers")
    assert tiingo_supported["output_name"] == "tiingo_supported_tickers"
    assert tiingo_supported["symbols"] == []

    finnhub_profiles = next(job for job in jobs if job["source"] == "finnhub" and job["dataset"] == "company_profile")
    assert finnhub_profiles["max_symbols_per_run"] == 50

    twelve_financials = next(
        job for job in jobs if job["source"] == "twelve_data" and job["dataset"] == "financial_statements"
    )
    assert twelve_financials["max_symbols_per_run"] == 20


def test_should_rebuild_local_state_only_when_missing_or_forced(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "node.sqlite"

    assert _should_rebuild_local_state(local_db_path=db_path) is True

    db_path.write_text("sqlite-placeholder", encoding="utf-8")
    assert _should_rebuild_local_state(local_db_path=db_path) is False

    monkeypatch.setenv("TRADEML_FORCE_REBUILD", "1")
    assert _should_rebuild_local_state(local_db_path=db_path) is True
