from __future__ import annotations

from pathlib import Path

import pandas as pd

from trademl.data_node.capabilities import default_macro_series
from trademl.data_node.db import BackfillTask, VendorAttempt
from trademl.data_node.planner import (
    canonical_task_key,
    choose_vendor_for_canonical_task,
    plan_auxiliary_tasks,
    training_readiness,
)


def test_choose_vendor_for_canonical_task_uses_fallback_order_and_skips_failed_attempts() -> None:
    task = BackfillTask(
        id=1,
        dataset="equities_eod",
        symbol="AAPL",
        start_date="2020-01-01",
        end_date="2020-01-31",
        kind="GAP",
        priority=1,
        status="LEASED",
        attempts=0,
        next_not_before=None,
        last_error=None,
        created_at="2026-04-02T00:00:00+00:00",
        updated_at="2026-04-02T00:00:00+00:00",
    )
    attempts = [
        VendorAttempt(
            task_key=canonical_task_key(task),
            task_family="canonical",
            planner_group="canonical_bars_backlog",
            vendor="tiingo",
            lease_owner=None,
            status="FAILED",
            attempts=1,
            last_error="timeout",
            next_eligible_at="2999-01-01T00:00:00+00:00",
            leased_at=None,
            lease_expires_at=None,
            rows_returned=None,
            payload_json=None,
            updated_at="2026-04-02T00:00:00+00:00",
        )
    ]

    chosen = choose_vendor_for_canonical_task(
        task=task,
        connectors={"alpaca": object(), "tiingo": object(), "twelve_data": object()},
        audit_state=None,
        attempts=attempts,
    )

    assert chosen == "alpaca"


def test_choose_vendor_for_canonical_task_retries_vendor_after_backoff_expires() -> None:
    task = BackfillTask(
        id=1,
        dataset="equities_eod",
        symbol="AAPL",
        start_date="2020-01-01",
        end_date="2020-01-31",
        kind="GAP",
        priority=1,
        status="PENDING",
        attempts=0,
        next_not_before=None,
        last_error=None,
        created_at="2026-04-02T00:00:00+00:00",
        updated_at="2026-04-02T00:00:00+00:00",
    )
    attempts = [
        VendorAttempt(
            task_key=canonical_task_key(task),
            task_family="canonical",
            planner_group="canonical_bars_backlog",
            vendor="tiingo",
            lease_owner=None,
            status="FAILED",
            attempts=1,
            last_error="budget exhausted",
            next_eligible_at="2026-04-02T00:00:00+00:00",
            leased_at=None,
            lease_expires_at=None,
            rows_returned=None,
            payload_json=None,
            updated_at="2026-04-02T00:00:00+00:00",
        )
    ]

    chosen = choose_vendor_for_canonical_task(
        task=task,
        connectors={"tiingo": object(), "alpaca": object()},
        audit_state=None,
        attempts=attempts,
        now=pd.Timestamp("2026-04-02T01:00:00+00:00").to_pydatetime(),
    )

    assert chosen == "tiingo"


def test_plan_auxiliary_tasks_includes_macro_and_reference_chunks(tmp_path: Path) -> None:
    reference_root = tmp_path / "data" / "reference"
    reference_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"ticker": "AAPL", "cik_str": "320193"}]).to_parquet(reference_root / "sec_company_tickers.parquet", index=False)

    tasks = plan_auxiliary_tasks(
        data_root=tmp_path,
        stage_symbols=["AAPL", "MSFT", "NVDA"],
        stage_years=5,
        connectors={"alpha_vantage": object(), "fred": object(), "sec_edgar": object()},
        audit_state=None,
    )

    datasets = {task.dataset for task in tasks}
    assert "listings" in datasets
    assert "filing_index" in datasets
    assert "macros_treasury" in datasets
    assert "vintagedates" in datasets
    macro_tasks = [task for task in tasks if task.dataset == "macros_treasury"]
    assert len(macro_tasks) == len(default_macro_series())


def test_training_readiness_requires_core_datasets() -> None:
    readiness = training_readiness(
        raw_green_ratio=0.99,
        has_corp_actions=True,
        has_listing_history=True,
        has_delistings=True,
        has_sec_filings=False,
        has_macro_vintages=False,
        macro_series_count=3,
        required_macro_series=7,
    )

    assert readiness["ready"] is False
    assert readiness["blockers"] == ["sec_filings", "macro_vintages", "macro_pack"]
