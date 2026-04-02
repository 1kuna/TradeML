from __future__ import annotations

from pathlib import Path

from trademl.data_node.audit import run_capability_audit
from trademl.data_node.capabilities import (
    backfill_capabilities,
    build_reference_jobs,
    capability_map,
    default_macro_series,
    effective_enable_status,
    forward_capabilities,
)


class _Connector:
    def __init__(self, vendor_name: str, failures: dict[str, Exception] | None = None) -> None:
        self.vendor_name = vendor_name
        self.failures = failures or {}

    def fetch(self, dataset: str, symbols: list[str], start_date: str, end_date: str):
        if dataset in self.failures:
            raise self.failures[dataset]
        import pandas as pd

        return pd.DataFrame([{"dataset": dataset, "symbol": (symbols or [""])[0], "date": end_date}])


def test_backfill_capabilities_follow_priority_order() -> None:
    ordered = backfill_capabilities(
        dataset="equities_eod",
        connectors={
            "alpaca": object(),
            "tiingo": object(),
            "twelve_data": object(),
            "massive": object(),
            "finnhub": object(),
        },
    )

    assert [capability.vendor for capability in ordered] == ["tiingo", "alpaca", "twelve_data", "massive", "finnhub"]


def test_forward_capabilities_follow_priority_order() -> None:
    ordered = forward_capabilities(
        dataset="equities_eod",
        connectors={
            "alpaca": object(),
            "tiingo": object(),
            "twelve_data": object(),
            "massive": object(),
            "finnhub": object(),
        },
    )

    assert [capability.vendor for capability in ordered] == ["alpaca", "tiingo", "twelve_data", "massive", "finnhub"]


def test_reference_jobs_only_include_enabled_verified_lanes() -> None:
    jobs = build_reference_jobs(
        connectors={
            "alpaca": object(),
            "massive": object(),
            "alpha_vantage": object(),
            "twelve_data": object(),
            "finnhub": object(),
            "fmp": object(),
            "sec_edgar": object(),
            "fred": object(),
            "tiingo": object(),
        },
        symbols=["AAPL", "MSFT", "NVDA"],
    )

    assert all(job["dataset"] != "news" for job in jobs)
    assert all(job["dataset"] != "price_target" for job in jobs)
    assert any(job["dataset"] == "company_profile" for job in jobs)
    assert any(job["dataset"] == "filing_index" for job in jobs)


def test_effective_enable_status_disables_doc_unverified_or_live_failed() -> None:
    capability = capability_map()["twelve_data.price_target.research"]
    assert effective_enable_status(capability) == "research_only"
    failed_override = {"capabilities": {capability.capability_id: {"doc_status": "doc_verified", "live_status": "live_failed", "enable_status": "research_only"}}}
    assert effective_enable_status(capability, audit_state=failed_override) == "disabled"


def test_run_capability_audit_persists_summary(tmp_path: Path) -> None:
    output = tmp_path / "vendor_audit.json"
    report = run_capability_audit(
        connectors={
            "alpaca": _Connector("alpaca"),
            "tiingo": _Connector("tiingo"),
            "twelve_data": _Connector("twelve_data"),
            "massive": _Connector("massive"),
            "finnhub": _Connector("finnhub"),
            "alpha_vantage": _Connector("alpha_vantage"),
            "fred": _Connector("fred"),
            "fmp": _Connector("fmp"),
            "sec_edgar": _Connector("sec_edgar"),
        },
        output_path=output,
    )

    assert output.exists()
    assert report["summary"]["live_status"]["live_verified"] > 0
    assert "alpaca.equities_eod.forward" in report["capabilities"]


def test_default_macro_series_covers_minimum_pack() -> None:
    series = default_macro_series()

    assert "DGS10" in series
    assert "UNRATE" in series
