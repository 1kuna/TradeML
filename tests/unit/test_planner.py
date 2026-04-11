from __future__ import annotations

from pathlib import Path

import pandas as pd
from trademl.data_node.budgets import BudgetManager

from trademl.data_node.capabilities import default_macro_series
from trademl.data_node.db import BackfillTask, VendorAttempt
from trademl.data_node.planner import (
    canonical_task_key,
    choose_vendor_for_canonical_task,
    plan_auxiliary_tasks,
    plan_canonical_bar_tasks,
    plan_coverage_tasks,
    training_readiness,
)


def test_choose_vendor_for_canonical_task_uses_priority_order_and_skips_failed_attempts() -> None:
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


def test_choose_vendor_for_canonical_task_skips_single_symbol_vendors_for_datewide_tasks() -> None:
    task = BackfillTask(
        id=1,
        dataset="equities_eod",
        symbol=None,
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

    chosen = choose_vendor_for_canonical_task(
        task=task,
        connectors={"alpaca": object(), "tiingo": object(), "twelve_data": object(), "finnhub": object()},
        audit_state=None,
        attempts=[],
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


def test_choose_vendor_for_canonical_task_skips_budget_blocked_vendor() -> None:
    class _BudgetedConnector:
        def __init__(self, vendor_name: str, budget_manager: BudgetManager) -> None:
            self.vendor_name = vendor_name
            self.budget_manager = budget_manager

    task = BackfillTask(
        id=1,
        dataset="equities_eod",
        symbol="AAPL",
        start_date="2025-12-16",
        end_date="2026-01-14",
        kind="GAP",
        priority=1,
        status="PENDING",
        attempts=0,
        next_not_before=None,
        last_error=None,
        created_at="2026-04-02T00:00:00+00:00",
        updated_at="2026-04-02T00:00:00+00:00",
    )
    tiingo_budget = BudgetManager({"tiingo": {"rpm": 1, "daily_cap": 1}})
    tiingo_budget.record_spend("tiingo", task_kind="FORWARD")
    chosen = choose_vendor_for_canonical_task(
        task=task,
        connectors={
            "alpaca": object(),
            "tiingo": _BudgetedConnector("tiingo", tiingo_budget),
        },
        audit_state=None,
        attempts=[],
    )

    assert chosen == "alpaca"


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


def test_plan_auxiliary_tasks_includes_research_archive_lanes_with_short_windows(tmp_path: Path) -> None:
    reference_root = tmp_path / "data" / "reference"
    reference_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"ticker": "AAPL", "cik_str": "320193"}]).to_parquet(reference_root / "sec_company_tickers.parquet", index=False)

    tasks = plan_auxiliary_tasks(
        data_root=tmp_path,
        stage_symbols=["AAPL", "MSFT", "NVDA"],
        stage_years=5,
        connectors={"alpaca": object(), "tiingo": object(), "finnhub": object()},
        audit_state=None,
        include_research=True,
        current_date="2026-04-10",
    )

    minute_tasks = [task for task in tasks if task.dataset == "equities_minute"]
    tiingo_news_tasks = [task for task in tasks if task.dataset == "news"]
    finnhub_news_tasks = [task for task in tasks if task.dataset == "company_news"]

    assert minute_tasks
    assert tiingo_news_tasks
    assert finnhub_news_tasks
    assert all(task.planner_group == "supplemental_research_backlog" for task in minute_tasks + tiingo_news_tasks + finnhub_news_tasks)
    assert all(task.start_date == "2026-04-05" and task.end_date == "2026-04-10" for task in minute_tasks)
    assert all(task.start_date == "2026-04-03" and task.end_date == "2026-04-10" for task in tiingo_news_tasks + finnhub_news_tasks)


def test_plan_canonical_bar_tasks_uses_symbol_range_windows() -> None:
    tasks = plan_canonical_bar_tasks(
        stage_symbols=["AAPL", "MSFT", "NVDA"],
        stage_years=1,
        connectors={"alpaca": object(), "tiingo": object()},
        current_date="2026-04-02",
        symbol_batch_size=2,
        trading_day_chunk_size=10,
    )

    assert tasks
    first = tasks[0]
    assert first.task_family == "canonical_bars"
    assert first.scope_kind == "symbol_range"
    assert first.preferred_vendors == ("tiingo", "alpaca")
    assert len(first.symbols) <= 2
    assert len(first.payload["trading_days"]) <= 10


def test_plan_canonical_bar_tasks_expands_and_prioritizes_frozen_training_window() -> None:
    tasks = plan_canonical_bar_tasks(
        stage_symbols=["AAPL"],
        stage_years=1,
        connectors={"alpaca": object(), "tiingo": object(), "twelve_data": object(), "massive": object()},
        current_date="2026-04-07",
        freeze_report_date="2026-03-06",
        symbol_batch_size=1,
        trading_day_chunk_size=20,
    )

    assert tasks
    assert min(task.start_date for task in tasks) <= "2025-03-06"
    frozen = [task for task in tasks if task.end_date <= "2026-03-06"]
    tail = [task for task in tasks if task.start_date > "2026-03-06"]
    assert frozen
    assert tail
    assert all(task.priority == 5 for task in frozen)
    assert all(task.payload["freeze_priority"] is True for task in frozen)
    assert all(task.preferred_vendors == ("alpaca", "tiingo") for task in frozen)
    assert all(task.priority == 10 for task in tail)


def test_plan_canonical_bar_tasks_respects_listing_history_windows(tmp_path: Path) -> None:
    reference_root = tmp_path / "data" / "reference"
    reference_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"symbol": "AAPL", "ipo_date": "1980-12-12", "delist_date": None},
            {"symbol": "SN", "ipo_date": "2023-07-31", "delist_date": None},
        ]
    ).to_parquet(reference_root / "listing_history.parquet", index=False)

    tasks = plan_canonical_bar_tasks(
        data_root=tmp_path,
        stage_symbols=["AAPL", "SN"],
        stage_years=1,
        connectors={"alpaca": object(), "tiingo": object(), "twelve_data": object()},
        current_date="2024-01-10",
        freeze_report_date="2023-12-29",
        symbol_batch_size=2,
        trading_day_chunk_size=20,
    )

    assert tasks
    sn_tasks = [task for task in tasks if task.symbols == ("SN",)]
    assert sn_tasks
    assert min(task.start_date for task in sn_tasks) >= "2023-07-31"
    assert all(task.preferred_vendors == ("alpaca", "tiingo") for task in sn_tasks if task.payload["freeze_priority"] is True)


def test_plan_coverage_tasks_orders_canonical_before_auxiliary(tmp_path: Path) -> None:
    reference_root = tmp_path / "data" / "reference"
    reference_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"ticker": "AAPL", "cik_str": "320193"}]).to_parquet(reference_root / "sec_company_tickers.parquet", index=False)

    tasks = plan_coverage_tasks(
        data_root=tmp_path,
        stage_symbols=["AAPL", "MSFT"],
        stage_years=1,
        connectors={"alpaca": object(), "tiingo": object(), "alpha_vantage": object(), "fred": object(), "sec_edgar": object()},
        current_date="2026-04-02",
        symbol_batch_size=2,
        trading_day_chunk_size=10,
    )

    assert tasks
    assert tasks[0].task_family == "canonical_bars"
    assert any(task.task_family == "auxiliary" for task in tasks)


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
