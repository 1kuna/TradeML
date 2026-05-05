from __future__ import annotations

from pathlib import Path

import pandas as pd

from trademl.data_node.capabilities import default_macro_series
from trademl.data_node.planner import (
    plan_auxiliary_tasks,
    plan_canonical_bar_tasks,
    plan_coverage_tasks,
    training_readiness,
)


def test_plan_auxiliary_tasks_includes_macro_and_reference_chunks(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "data" / "reference"
    reference_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"ticker": "AAPL", "cik_str": "320193"}]).to_parquet(
        reference_root / "sec_company_tickers.parquet", index=False
    )

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
    assert "companyfacts" in datasets
    assert "macros_treasury" in datasets
    assert "vintagedates" in datasets
    macro_tasks = [task for task in tasks if task.dataset == "macros_treasury"]
    assert len(macro_tasks) == len(default_macro_series())


def test_plan_auxiliary_tasks_includes_research_archive_lanes_with_historical_windows(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "data" / "reference"
    reference_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"ticker": "AAPL", "cik_str": "320193"}]).to_parquet(
        reference_root / "sec_company_tickers.parquet", index=False
    )

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
    assert not tiingo_news_tasks
    assert finnhub_news_tasks
    assert all(
        task.planner_group == "supplemental_research_backlog"
        for task in minute_tasks + tiingo_news_tasks + finnhub_news_tasks
    )
    assert any(
        task.start_date == "2025-04-10" and task.end_date == "2025-04-14"
        for task in minute_tasks
    )
    assert any(
        task.start_date == "2026-04-10" and task.end_date == "2026-04-10"
        for task in minute_tasks
    )
    assert any(
        task.start_date == "2025-04-10" and task.end_date == "2025-04-16"
        for task in finnhub_news_tasks
    )
    assert any(
        task.start_date == "2026-04-09" and task.end_date == "2026-04-10"
        for task in finnhub_news_tasks
    )
    assert all(
        task.payload["capture_family"] == "research_archive"
        for task in minute_tasks + tiingo_news_tasks + finnhub_news_tasks
    )


def test_plan_auxiliary_tasks_uses_vendor_scoped_news_task_keys(
    tmp_path: Path,
) -> None:
    audit_state = {
        "capabilities": {
            "alpaca.news.research": {
                "doc_status": "doc_verified",
                "live_status": "live_verified",
                "enable_status": "research_only",
            },
            "fmp.stock_news.research": {
                "doc_status": "doc_verified",
                "live_status": "live_verified",
                "enable_status": "research_only",
            },
        }
    }

    tasks = plan_auxiliary_tasks(
        data_root=tmp_path,
        stage_symbols=["AAPL", "MSFT"],
        stage_years=1,
        connectors={"alpaca": object(), "fmp": object(), "finnhub": object()},
        audit_state=audit_state,
        include_research=True,
        current_date="2026-04-10",
    )

    news_tasks = [
        task
        for task in tasks
        if task.output_name == "ticker_news"
        and task.dataset in {"news", "stock_news", "company_news"}
    ]
    assert {task.preferred_vendors[0] for task in news_tasks} == {
        "alpaca",
        "fmp",
        "finnhub",
    }
    assert len({task.task_key for task in news_tasks}) == len(news_tasks)
    assert any(task.task_key.startswith("research_only::alpaca::news::") for task in news_tasks)
    assert any(task.task_key.startswith("research_only::fmp::stock_news::") for task in news_tasks)
    assert all(
        task.payload["retention_class"] == "raw_archive"
        for task in news_tasks
    )


def test_plan_auxiliary_tasks_includes_audited_alpaca_free_plan_expansion_lanes(
    tmp_path: Path,
) -> None:
    audit_state = {
        "capabilities": {
            capability: {
                "doc_status": "doc_verified",
                "live_status": "live_verified",
                "enable_status": "research_only",
            }
            for capability in [
                "alpaca.stock_trades.research",
                "alpaca.stock_quotes.research",
                "alpaca.stock_snapshots.research",
                "alpaca.crypto_bars.research",
                "alpaca.crypto_quotes.research",
                "alpaca.crypto_snapshots.research",
                "alpaca.option_chain_reference.research",
            ]
        }
    }

    tasks = plan_auxiliary_tasks(
        data_root=tmp_path,
        stage_symbols=["AAPL", "MSFT"],
        stage_years=1,
        connectors={"alpaca": object()},
        audit_state=audit_state,
        include_research=True,
        current_date="2026-04-10",
    )
    by_dataset = {task.dataset: task for task in tasks}

    assert {"stock_trades", "stock_quotes", "stock_snapshots", "crypto_bars", "crypto_quotes", "crypto_snapshots", "option_chain_reference"}.issubset(by_dataset)
    assert by_dataset["crypto_bars"].symbols == ("BTC/USD", "ETH/USD", "SOL/USD")
    assert any(task.dataset == "option_chain_reference" and task.symbols == ("SPY",) for task in tasks)
    assert by_dataset["stock_snapshots"].start_date == "2026-04-10"
    assert by_dataset["stock_trades"].output_name == "alpaca_market_events"
    assert by_dataset["option_chain_reference"].output_name == "option_snapshots"


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
    assert any(task.symbols == ("AAPL", "MSFT") for task in tasks)
    assert any(task.symbols == ("NVDA",) for task in tasks)


def test_plan_canonical_bar_tasks_expands_and_prioritizes_frozen_training_window() -> (
    None
):
    tasks = plan_canonical_bar_tasks(
        stage_symbols=["AAPL"],
        stage_years=1,
        connectors={
            "alpaca": object(),
            "tiingo": object(),
            "twelve_data": object(),
            "massive": object(),
        },
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


def test_plan_canonical_bar_tasks_respects_listing_history_windows(
    tmp_path: Path,
) -> None:
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
    assert all(
        task.preferred_vendors == ("alpaca", "tiingo")
        for task in sn_tasks
        if task.payload["freeze_priority"] is True
    )


def test_plan_coverage_tasks_orders_canonical_before_auxiliary(tmp_path: Path) -> None:
    reference_root = tmp_path / "data" / "reference"
    reference_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"ticker": "AAPL", "cik_str": "320193"}]).to_parquet(
        reference_root / "sec_company_tickers.parquet", index=False
    )

    tasks = plan_coverage_tasks(
        data_root=tmp_path,
        stage_symbols=["AAPL", "MSFT"],
        stage_years=1,
        connectors={
            "alpaca": object(),
            "tiingo": object(),
            "alpha_vantage": object(),
            "fred": object(),
            "sec_edgar": object(),
        },
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
