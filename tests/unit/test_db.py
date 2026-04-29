from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from trademl.data_node import db as db_module
from trademl.data_node.budgets import BudgetManager
from trademl.data_node.db import DataNodeDB


def test_data_node_db_facade_delegates_to_focused_stores(tmp_path: Path) -> None:
    database = DataNodeDB(tmp_path / "node.sqlite")

    database.update_partition_status(
        source="alpaca",
        dataset="equities_eod",
        date="2026-04-02",
        status="OK",
        row_count=1,
    )
    database.upsert_planner_task(
        task_key="canonical::equities_eod::AAPL::2026-04-02",
        task_family="canonical_bars",
        planner_group="rolling_canonical",
        dataset="equities_eod",
        tier="A",
        priority=1,
        start_date="2026-04-02",
        end_date="2026-04-02",
        symbols=["AAPL"],
        eligible_vendors=["alpaca"],
    )
    database.upsert_vendor_lane_health(vendor="alpaca", dataset="equities_eod", state="OK")
    database.replace_canonical_units_for_date(
        dataset="equities_eod",
        trading_date="2026-04-02",
        symbols=["AAPL"],
        partition_revision=1,
    )

    assert database.runtime.fetch_partition_status()[0]["source"] == "alpaca"
    assert database.planner.get_planner_task("canonical::equities_eod::AAPL::2026-04-02") is not None
    assert database.vendors.get_vendor_lane_health(vendor="alpaca", dataset="equities_eod") is not None
    assert database.canonical.fetch_canonical_units_for_date(
        dataset="equities_eod",
        trading_date="2026-04-02",
    )[0].symbol == "AAPL"


def test_vendor_attempt_lease_success_and_backoff(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    database = DataNodeDB(tmp_path / "node.sqlite")
    now = db_module.utc_now()
    monkeypatch.setattr(db_module, "utc_now", lambda: now)

    leased = database.lease_vendor_attempt(
        task_key="reference::listings::2026-04-02",
        task_family="reference",
        planner_group="reference_events_backlog",
        vendor="alpha_vantage",
        lease_owner="worker-a",
        payload={"dataset": "listings"},
        lease_ttl_seconds=30,
    )

    assert leased is not None
    assert leased.status == "LEASED"
    database.mark_vendor_attempt_failed(
        task_key="reference::listings::2026-04-02",
        vendor="alpha_vantage",
        error="429",
        backoff_minutes=30,
    )
    assert database.lease_vendor_attempt(
        task_key="reference::listings::2026-04-02",
        task_family="reference",
        planner_group="reference_events_backlog",
        vendor="alpha_vantage",
        lease_owner="worker-a",
        payload={"dataset": "listings"},
        now=now + timedelta(minutes=29),
    ) is None
    assert database.lease_vendor_attempt(
        task_key="reference::listings::2026-04-02",
        task_family="reference",
        planner_group="reference_events_backlog",
        vendor="alpha_vantage",
        lease_owner="worker-a",
        payload={"dataset": "listings"},
        now=now + timedelta(minutes=31),
    ) is not None


def test_vendor_attempt_success_is_terminal(tmp_path: Path) -> None:
    database = DataNodeDB(tmp_path / "node.sqlite")
    leased = database.lease_vendor_attempt(
        task_key="canonical::equities_eod::AAPL::2024-01-01::2024-01-31::GAP",
        task_family="canonical",
        planner_group="canonical_bars_backlog",
        vendor="tiingo",
        lease_owner="worker-a",
    )

    assert leased is not None
    database.mark_vendor_attempt_success(
        task_key="canonical::equities_eod::AAPL::2024-01-01::2024-01-31::GAP",
        vendor="tiingo",
        rows_returned=22,
    )
    assert database.lease_vendor_attempt(
        task_key="canonical::equities_eod::AAPL::2024-01-01::2024-01-31::GAP",
        task_family="canonical",
        planner_group="canonical_bars_backlog",
        vendor="tiingo",
        lease_owner="worker-a",
    ) is None


def test_vendor_attempt_leasing_is_race_safe(tmp_path: Path) -> None:
    database = DataNodeDB(tmp_path / "node.sqlite")
    barrier = threading.Barrier(2)
    results: list[object] = []

    def worker(name: str) -> None:
        barrier.wait()
        try:
            leased = database.lease_vendor_attempt(
                task_key="canonical::equities_eod::AAPL::2024-01-01::2024-01-31::GAP",
                task_family="canonical",
                planner_group="canonical_bars_backlog",
                vendor="alpaca",
                lease_owner=name,
            )
            results.append(leased)
        except Exception as exc:  # pragma: no cover - assertion after join
            results.append(exc)

    first = threading.Thread(target=worker, args=("worker-a",))
    second = threading.Thread(target=worker, args=("worker-b",))
    first.start()
    second.start()
    first.join()
    second.join()

    assert not any(isinstance(result, Exception) for result in results)
    attempts = database.fetch_vendor_attempts()
    assert len(attempts) == 1
    assert attempts[0].vendor == "alpaca"


def test_fetch_vendor_attempts_filters_by_updated_after(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    database = DataNodeDB(tmp_path / "node.sqlite")
    first_now = db_module.utc_now()
    monkeypatch.setattr(db_module, "utc_now", lambda: first_now)
    leased = database.lease_vendor_attempt(
        task_key="canonical::equities_eod::AAPL::2024-01-01::2024-01-31::GAP",
        task_family="canonical_bars",
        planner_group="canonical_bars_backlog",
        vendor="alpaca",
        lease_owner="worker-a",
        payload={"symbols": ["AAPL"]},
    )
    assert leased is not None
    database.mark_vendor_attempt_success(
        task_key="canonical::equities_eod::AAPL::2024-01-01::2024-01-31::GAP",
        vendor="alpaca",
        rows_returned=22,
    )

    second_now = first_now + timedelta(minutes=20)
    monkeypatch.setattr(db_module, "utc_now", lambda: second_now)
    leased = database.lease_vendor_attempt(
        task_key="canonical::equities_eod::MSFT::2024-01-01::2024-01-31::GAP",
        task_family="canonical_bars",
        planner_group="canonical_bars_backlog",
        vendor="alpaca",
        lease_owner="worker-a",
        payload={"symbols": ["MSFT"]},
    )
    assert leased is not None
    database.mark_vendor_attempt_success(
        task_key="canonical::equities_eod::MSFT::2024-01-01::2024-01-31::GAP",
        vendor="alpaca",
        rows_returned=21,
    )

    recent = database.fetch_vendor_attempts(updated_after=(first_now + timedelta(minutes=10)).isoformat())

    assert len(recent) == 1
    assert recent[0].task_key.endswith("MSFT::2024-01-01::2024-01-31::GAP")


def test_summarize_vendor_attempts_returns_counts_vendor_rows_and_recent_failures(tmp_path: Path) -> None:
    database = DataNodeDB(tmp_path / "node.sqlite")
    leased = database.lease_vendor_attempt(
        task_key="reference::listings::2026-04-02",
        task_family="reference_events",
        planner_group="reference_events_backlog",
        vendor="alpaca",
        lease_owner="worker-a",
        payload={"dataset": "listings"},
    )
    assert leased is not None
    database.mark_vendor_attempt_failed(
        task_key="reference::listings::2026-04-02",
        vendor="alpaca",
        error="429",
        backoff_minutes=30,
    )

    summary = database.summarize_vendor_attempts()

    assert summary["counts"] == {"FAILED": 1}
    assert summary["by_vendor"] == [
        {
            "vendor": "alpaca",
            "total": 1,
            "leased": 0,
            "success": 0,
            "failed": 1,
            "permanent_failed": 0,
            "latest_update": summary["by_vendor"][0]["latest_update"],
        }
    ]
    assert summary["recent_failures"][0]["vendor"] == "alpaca"
    assert summary["recent_failures"][0]["status"] == "FAILED"


def test_count_planner_tasks_by_family_filters_by_status_and_updated_after(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    database = DataNodeDB(tmp_path / "node.sqlite")
    first_now = db_module.utc_now()
    monkeypatch.setattr(db_module, "utc_now", lambda: first_now)
    database.upsert_planner_task(
        task_key="reference_events::earnings::2026-04-02",
        task_family="reference_events",
        planner_group="reference_events_backlog",
        dataset="earnings",
        tier="A",
        priority=1,
        start_date="2026-04-02",
        end_date="2026-04-02",
        symbols=["AAPL"],
        eligible_vendors=["alpaca"],
        payload={"backlog_class": "reference_events"},
    )
    leased = database.lease_next_planner_task(lease_owner="worker-a", task_families=("reference_events",))
    assert leased is not None
    database.mark_planner_task_success(leased.task_key)

    second_now = first_now + timedelta(minutes=20)
    monkeypatch.setattr(db_module, "utc_now", lambda: second_now)
    database.upsert_planner_task(
        task_key="security_master::listings::2026-04-03",
        task_family="security_master",
        planner_group="security_master_backlog",
        dataset="listings",
        tier="A",
        priority=1,
        start_date="2026-04-03",
        end_date="2026-04-03",
        symbols=["AAPL"],
        eligible_vendors=["alpaca"],
        payload={"backlog_class": "security_master"},
    )
    leased = database.lease_next_planner_task(lease_owner="worker-a", task_families=("security_master",))
    assert leased is not None
    database.mark_planner_task_success(leased.task_key)

    counts = database.count_planner_tasks_by_family(
        statuses=("SUCCESS",),
        updated_after=(first_now + timedelta(minutes=10)).isoformat(),
    )

    assert counts == {"security_master": 1}


def test_vendor_attempt_release_clears_stale_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    database = DataNodeDB(tmp_path / "node.sqlite")
    now = db_module.utc_now()
    monkeypatch.setattr(db_module, "utc_now", lambda: now)
    first = database.lease_vendor_attempt(
        task_key="macro::DGS10",
        task_family="macro",
        planner_group="reference_events_backlog",
        vendor="fred",
        lease_owner="worker-a",
    )
    assert first is not None
    database.mark_vendor_attempt_failed(
        task_key="macro::DGS10",
        vendor="fred",
        error="fred request failed: old 400",
        backoff_minutes=5,
    )

    monkeypatch.setattr(db_module, "utc_now", lambda: now + timedelta(minutes=6))
    leased = database.lease_vendor_attempt(
        task_key="macro::DGS10",
        task_family="macro",
        planner_group="reference_events_backlog",
        vendor="fred",
        lease_owner="worker-a",
    )

    assert leased is not None
    assert leased.status == "LEASED"
    assert leased.last_error is None
    assert leased.next_eligible_at is None


def test_summarize_lane_throughput_reports_rows_per_minute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = DataNodeDB(tmp_path / "node.sqlite")
    now = db_module.utc_now()
    monkeypatch.setattr(db_module, "utc_now", lambda: now)
    database.upsert_planner_task(
        task_key="minute::AAPL",
        task_family="supplemental_research",
        planner_group="supplemental_research_backlog",
        dataset="equities_minute",
        tier="B",
        priority=205,
        start_date="2026-04-01",
        end_date="2026-04-01",
        symbols=["AAPL"],
        eligible_vendors=["alpaca"],
        output_name="equities_minute",
        payload={"scope_kind": "symbol_range"},
    )
    assert database.lease_vendor_attempt(
        task_key="minute::AAPL",
        task_family="supplemental_research",
        planner_group="supplemental_research_backlog",
        vendor="alpaca",
        lease_owner="worker-a",
    )
    database.mark_vendor_attempt_success(
        task_key="minute::AAPL", vendor="alpaca", rows_returned=300
    )

    throughput = database.summarize_lane_throughput(minutes=15)

    assert throughput[0]["vendor"] == "alpaca"
    assert throughput[0]["dataset"] == "equities_minute"
    assert throughput[0]["rows_returned"] == 300
    assert throughput[0]["rows_per_minute"] == pytest.approx(20.0)


def test_summarize_lane_throughput_includes_active_budget_cooldown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = DataNodeDB(tmp_path / "node.sqlite")
    now = datetime(2026, 4, 28, 12, 0, tzinfo=UTC)
    monkeypatch.setattr(db_module, "utc_now", lambda: now)
    database.upsert_planner_task(
        task_key="news::AAPL",
        task_family="supplemental_research",
        planner_group="supplemental_research_backlog",
        dataset="company_news",
        tier="B",
        priority=220,
        start_date="2026-04-01",
        end_date="2026-04-28",
        symbols=["AAPL"],
        eligible_vendors=["finnhub"],
        output_name="ticker_news",
        payload={"scope_kind": "symbol_range"},
    )
    assert database.lease_vendor_attempt(
        task_key="news::AAPL",
        task_family="supplemental_research",
        planner_group="supplemental_research_backlog",
        vendor="finnhub",
        lease_owner="worker-a",
    )
    database.mark_vendor_attempt_failed(
        task_key="news::AAPL",
        vendor="finnhub",
        error="budget exhausted for vendor=finnhub",
        backoff_minutes=30,
    )
    database.upsert_vendor_lane_health(
        vendor="finnhub",
        dataset="company_news",
        state="BUDGET_BLOCKED",
        cooldown_until=(now + timedelta(minutes=30)).isoformat(),
        recent_local_budget_blocks=2,
    )

    throughput = database.summarize_lane_throughput(minutes=15)

    assert throughput[0]["vendor"] == "finnhub"
    assert throughput[0]["dataset"] == "company_news"
    assert throughput[0]["lane_state"] == "BUDGET_BLOCKED"
    assert throughput[0]["cooldown_until"] == "2026-04-28T12:30:00+00:00"
    assert throughput[0]["blocked_reason"] == "budget_exhausted"
    assert throughput[0]["recent_local_budget_blocks"] == 2


def test_mark_completed_planner_progress_success_clears_stale_active_tasks(
    tmp_path: Path,
) -> None:
    database = DataNodeDB(tmp_path / "node.sqlite")
    database.upsert_planner_task(
        task_key="canonical::done",
        task_family="canonical_bars",
        planner_group="phase1_pinned_canonical",
        dataset="equities_eod",
        tier="A",
        priority=5,
        start_date="2026-04-01",
        end_date="2026-04-01",
        symbols=["AAPL"],
        eligible_vendors=["alpaca"],
        output_name="equities_bars",
        payload={"scope_kind": "symbol_range"},
    )
    database.update_planner_task_progress(
        task_key="canonical::done",
        expected_units=1,
        completed_units=1,
        remaining_units=0,
        completed_symbols=["AAPL"],
        remaining_symbols=[],
        state={"trading_days": ["2026-04-01"]},
    )
    database.mark_planner_task_partial(
        "canonical::done", error="old partial", backoff_minutes=30
    )

    completed = database.mark_completed_planner_progress_success(
        task_families=("canonical_bars",)
    )
    task = database.get_planner_task("canonical::done")

    assert completed == 1
    assert task is not None
    assert task.status == "SUCCESS"
    assert task.last_error is None


def test_planner_summary_excludes_terminal_tasks_from_remaining_units(
    tmp_path: Path,
) -> None:
    database = DataNodeDB(tmp_path / "node.sqlite")
    database.upsert_planner_task(
        task_key="canonical_repair::terminal",
        task_family="canonical_repair",
        planner_group="canonical_repair",
        dataset="equities_eod",
        tier="A",
        priority=8,
        start_date="2026-04-20",
        end_date="2026-04-20",
        symbols=["AL", "HOLX"],
        eligible_vendors=["alpaca"],
        output_name="equities_bars",
        payload={"scope_kind": "symbol_range"},
    )
    database.update_planner_task_progress(
        task_key="canonical_repair::terminal",
        expected_units=2,
        completed_units=0,
        remaining_units=2,
        remaining_symbols=["AL", "HOLX"],
        state={"trading_days": ["2026-04-20"]},
    )
    database.mark_planner_task_failed(
        "canonical_repair::terminal",
        error="canonical repair uncollectable",
        backoff_minutes=0,
        permanent=True,
    )

    summary = database.planner_summary()

    assert summary["counts"]["canonical_repair"]["PERMANENT_FAILED"] == 1
    assert summary["progress"]["canonical_repair"]["remaining_units"] == 0
    assert summary["backlog_progress"]["repair"]["remaining_units"] == 0


def test_planner_task_lifecycle_and_progress(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    database = DataNodeDB(tmp_path / "node.sqlite")
    now = db_module.utc_now()
    monkeypatch.setattr(db_module, "utc_now", lambda: now)

    database.upsert_planner_task(
        task_key="canonical::equities_eod::chunk000::2025-01-01::2025-01-31",
        task_family="canonical_bars",
        planner_group="canonical_bars_backlog",
        dataset="equities_eod",
        tier="A",
        priority=10,
        start_date="2025-01-01",
        end_date="2025-01-31",
        symbols=["AAPL", "MSFT"],
        eligible_vendors=["alpaca", "tiingo"],
        payload={"scope_kind": "symbol_range"},
    )
    leased = database.lease_next_planner_task(
        lease_owner="worker-a",
        task_families=("canonical_bars",),
    )

    assert leased is not None
    assert leased.task_family == "canonical_bars"
    assert leased.symbols == ("AAPL", "MSFT")
    assert leased.eligible_vendors == ("alpaca", "tiingo")

    database.update_planner_task_progress(
        task_key=leased.task_key,
        expected_units=40,
        completed_units=20,
        remaining_units=20,
        completed_symbols=["AAPL"],
        remaining_symbols=["MSFT"],
        state={"trading_days": 20},
    )
    database.mark_planner_task_partial(leased.task_key, error="remaining_symbols=1", backoff_minutes=30)

    assert database.lease_next_planner_task(
        lease_owner="worker-a",
        task_families=("canonical_bars",),
        now=now + timedelta(minutes=29),
    ) is None
    leased_again = database.lease_next_planner_task(
        lease_owner="worker-a",
        task_families=("canonical_bars",),
        now=now + timedelta(minutes=31),
    )
    assert leased_again is not None
    direct_lease = database.lease_planner_task_by_key(
        task_key=leased.task_key,
        lease_owner="worker-b",
        now=now + timedelta(minutes=31),
    )
    assert direct_lease is None
    progress = database.fetch_planner_task_progress(leased.task_key)
    assert progress is not None
    assert progress.completed_symbols == ("AAPL",)
    assert progress.remaining_symbols == ("MSFT",)
    assert progress.state["trading_days"] == 20

    database.mark_planner_task_success(leased.task_key)
    summary = database.planner_summary()
    assert summary["counts"]["canonical_bars"]["SUCCESS"] == 1
    assert summary["progress"]["canonical_bars"]["completed_units"] == 20


def test_lease_next_planner_task_filters_task_family_before_scan_window(tmp_path: Path) -> None:
    database = DataNodeDB(tmp_path / "node.sqlite")
    for index in range(300):
        database.upsert_planner_task(
            task_key=f"canonical::{index:04d}",
            task_family="canonical_bars",
            planner_group="canonical_bars_backlog",
            dataset="equities_eod",
            tier="A",
            priority=10,
            start_date="2025-01-01",
            end_date="2025-01-31",
            symbols=[f"SYM{index:04d}"],
            eligible_vendors=["alpaca"],
            payload={"scope_kind": "symbol_range"},
        )
    database.upsert_planner_task(
        task_key="macro::DGS10",
        task_family="macro",
        planner_group="macro_backlog",
        dataset="macros_treasury",
        tier="A",
        priority=20,
        start_date="2025-01-01",
        end_date="2025-01-31",
        symbols=["DGS10"],
        eligible_vendors=["fred"],
        payload={"scope_kind": "series_range"},
    )

    leased = database.lease_next_planner_task(
        lease_owner="worker-a",
        task_families=("macro",),
        vendor="fred",
        limit=64,
        scan_pages=2,
    )

    assert leased is not None
    assert leased.task_key == "macro::DGS10"


def test_update_partition_status_is_thread_safe(tmp_path: Path) -> None:
    database = DataNodeDB(tmp_path / "node.sqlite")
    barrier = threading.Barrier(8)
    errors: list[Exception] = []

    def worker(index: int) -> None:
        barrier.wait()
        try:
            database.update_partition_status(
                "alpaca",
                "equities_eod",
                f"2025-01-{index + 1:02d}",
                "GREEN",
                10,
                10,
                "OK",
            )
        except Exception as exc:  # pragma: no cover - asserted after join
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    rows = database.fetch_partition_status()
    assert len(rows) == 8


def test_prune_planner_tasks_removes_stale_tasks_and_attempts(tmp_path: Path) -> None:
    database = DataNodeDB(tmp_path / "node.sqlite")
    database.upsert_planner_task(
        task_key="keep",
        task_family="canonical_bars",
        planner_group="canonical_bars_backlog",
        dataset="equities_eod",
        tier="A",
        priority=5,
        start_date="2025-01-01",
        end_date="2025-01-31",
        symbols=["AAPL"],
        eligible_vendors=["alpaca"],
        payload={"scope_kind": "symbol_range"},
    )
    database.upsert_planner_task(
        task_key="drop",
        task_family="canonical_bars",
        planner_group="canonical_bars_backlog",
        dataset="equities_eod",
        tier="A",
        priority=5,
        start_date="2025-01-01",
        end_date="2025-01-31",
        symbols=["SN"],
        eligible_vendors=["twelve_data"],
        payload={"scope_kind": "symbol_range"},
    )
    leased = database.lease_planner_task_by_key(task_key="drop", lease_owner="worker-a")
    assert leased is not None
    database.lease_vendor_attempt(
        task_key="drop",
        task_family="canonical_bars",
        planner_group="canonical_bars_backlog",
        vendor="twelve_data",
        lease_owner="worker-a",
        payload={"symbols": ["SN"]},
    )
    database.update_planner_task_progress(
        task_key="drop",
        expected_units=20,
        completed_units=0,
        remaining_units=20,
        completed_symbols=[],
        remaining_symbols=["SN"],
        state={"trading_days": 20},
    )

    removed = database.prune_planner_tasks(task_families=("canonical_bars",), valid_task_keys={"keep"})

    assert removed == 1
    assert database.get_planner_task("drop") is None
    assert database.fetch_planner_task_progress("drop") is None
    assert database.get_planner_task("keep") is not None


def test_bulk_upsert_planner_tasks_skips_unchanged_row_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = DataNodeDB(tmp_path / "node.sqlite")
    row = {
        "task_key": "canonical::2026-04-01::000",
        "task_family": "canonical_bars",
        "planner_group": "rolling_canonical",
        "dataset": "equities_eod",
        "tier": "A",
        "priority": 5,
        "start_date": "2026-04-01",
        "end_date": "2026-04-01",
        "symbols": ["AAPL"],
        "eligible_vendors": ["alpaca"],
        "output_name": "equities_bars",
        "payload": {"scope_kind": "symbol_range"},
    }
    first_now = datetime(2026, 4, 1, 12, 0, tzinfo=UTC)
    second_now = datetime(2026, 4, 1, 12, 5, tzinfo=UTC)
    third_now = datetime(2026, 4, 1, 12, 10, tzinfo=UTC)
    monkeypatch.setattr(db_module, "utc_now", lambda: first_now)
    database.bulk_upsert_planner_tasks([row])
    first = database.get_planner_task("canonical::2026-04-01::000")
    assert first is not None

    monkeypatch.setattr(db_module, "utc_now", lambda: second_now)
    database.bulk_upsert_planner_tasks([dict(row)])
    unchanged = database.get_planner_task("canonical::2026-04-01::000")
    assert unchanged is not None
    assert unchanged.updated_at == first.updated_at

    changed = {**row, "priority": 4}
    monkeypatch.setattr(db_module, "utc_now", lambda: third_now)
    database.bulk_upsert_planner_tasks([changed])
    refreshed = database.get_planner_task("canonical::2026-04-01::000")
    assert refreshed is not None
    assert refreshed.updated_at == third_now.isoformat()


def test_canonical_progress_uses_durable_unit_ledger(tmp_path: Path) -> None:
    database = DataNodeDB(tmp_path / "node.sqlite")

    database.replace_canonical_units_for_date(
        dataset="equities_eod",
        trading_date="2025-01-02",
        symbols=["AAPL", "MSFT"],
        partition_revision=1,
        source_names={"AAPL": "alpaca", "MSFT": "alpaca"},
    )
    database.replace_canonical_units_for_date(
        dataset="equities_eod",
        trading_date="2025-01-03",
        symbols=["AAPL"],
        partition_revision=2,
        source_names={"AAPL": "alpaca"},
    )

    progress = database.fetch_canonical_progress(
        dataset="equities_eod",
        symbols=["AAPL", "MSFT"],
        trading_days=["2025-01-02", "2025-01-03"],
    )

    assert progress["expected_units"] == 4
    assert progress["completed_units"] == 3
    assert progress["remaining_units"] == 1
    assert progress["completed_symbols"] == ["AAPL"]
    assert progress["remaining_symbols"] == ["MSFT"]


def test_raw_partition_manifest_and_vendor_lane_health_round_trip(tmp_path: Path) -> None:
    database = DataNodeDB(tmp_path / "node.sqlite")

    database.upsert_raw_partition_manifest(
        dataset="equities_eod",
        trading_date="2025-01-02",
        partition_revision=3,
        symbol_count=2,
        row_count=2,
        symbols=["AAPL", "MSFT"],
        content_hash="abc123",
        status="HEALTHY",
    )
    database.upsert_vendor_lane_health(
        vendor="tiingo",
        dataset="equities_eod",
        state="COOLDOWN",
        cooldown_until="2026-04-10T12:00:00+00:00",
        recent_outbound_requests=4,
        recent_remote_429s=4,
    )

    manifest = database.get_raw_partition_manifest(dataset="equities_eod", trading_date="2025-01-02")
    lane = database.get_vendor_lane_health(vendor="tiingo", dataset="equities_eod")

    assert manifest is not None
    assert manifest.partition_revision == 3
    assert manifest.symbols == ("AAPL", "MSFT")
    assert lane is not None
    assert lane.state == "COOLDOWN"
    assert lane.recent_remote_429s == 4


def test_clear_expired_vendor_lane_cooldowns_resolves_current_state(
    tmp_path: Path,
) -> None:
    database = DataNodeDB(tmp_path / "node.sqlite")
    now = datetime.now(tz=UTC).replace(microsecond=0)
    database.upsert_vendor_lane_health(
        vendor="twelve_data",
        dataset="equities_minute",
        state="BUDGET_BLOCKED",
        cooldown_until=(now - timedelta(seconds=1)).isoformat(),
    )

    cleared = database.clear_expired_vendor_lane_cooldowns(now=now)
    lane = database.get_vendor_lane_health(
        vendor="twelve_data", dataset="equities_minute"
    )

    assert cleared == 1
    assert lane is not None
    assert lane.state == "HEALTHY"
    assert lane.cooldown_until is None


def test_clear_stale_idle_vendor_lanes_resolves_old_idle_marker(
    tmp_path: Path,
) -> None:
    database = DataNodeDB(tmp_path / "node.sqlite")
    now = datetime(2026, 4, 29, 12, tzinfo=UTC)
    database.upsert_vendor_lane_health(
        vendor="alpaca",
        dataset="news",
        state="IDLE_BUDGET",
    )

    cleared = database.clear_stale_idle_vendor_lanes(
        older_than=now + timedelta(days=1)
    )
    lane = database.get_vendor_lane_health(vendor="alpaca", dataset="news")

    assert cleared == 1
    assert lane is not None
    assert lane.state == "HEALTHY"


def test_claim_next_vendor_task_leases_task_and_attempt_atomically(
    tmp_path: Path,
) -> None:
    database = DataNodeDB(tmp_path / "node.sqlite")
    database.upsert_planner_task(
        task_key="minute::alpaca::AAPL",
        task_family="supplemental_research",
        planner_group="supplemental_research_backlog",
        dataset="equities_minute",
        tier="B",
        priority=205,
        start_date="2026-04-01",
        end_date="2026-04-01",
        symbols=["AAPL"],
        eligible_vendors=["alpaca"],
        output_name="equities_minute",
        payload={"request_units": 1},
    )
    budget = BudgetManager({"alpaca": {"rpm": 200, "daily_cap": 1000}})

    claim = database.claim_next_vendor_task(
        vendor="alpaca",
        lease_owner="worker-1",
        task_families=("supplemental_research",),
        budget_decision_provider=lambda _task: budget.budget_decision("alpaca"),
    )

    assert claim.task is not None
    assert claim.task.task_key == "minute::alpaca::AAPL"
    attempt = database.vendor_attempts_for_task("minute::alpaca::AAPL")[0]
    assert attempt.status == "LEASED"
    assert attempt.lease_owner == "worker-1"


def test_claim_next_vendor_task_returns_explicit_budget_skip_reason(
    tmp_path: Path,
) -> None:
    database = DataNodeDB(tmp_path / "node.sqlite")
    database.upsert_planner_task(
        task_key="minute::twelve::AAPL",
        task_family="supplemental_research",
        planner_group="supplemental_research_backlog",
        dataset="equities_minute",
        tier="B",
        priority=205,
        start_date="2026-04-01",
        end_date="2026-04-01",
        symbols=["AAPL"],
        eligible_vendors=["twelve_data"],
        output_name="equities_minute",
        payload={"request_units": 1},
    )
    budget = BudgetManager({"twelve_data": {"rpm": 1, "daily_cap": 100}})
    now = datetime.now(tz=UTC).replace(microsecond=0)
    budget.record_spend("twelve_data", now=now)

    claim = database.claim_next_vendor_task(
        vendor="twelve_data",
        lease_owner="worker-1",
        task_families=("supplemental_research",),
        budget_decision_provider=lambda _task: budget.budget_decision(
            "twelve_data", now=now + timedelta(seconds=1)
        ),
        now=now + timedelta(seconds=1),
    )

    assert claim.task is None
    assert claim.skip_reason == "budget:minute"
    assert claim.dataset == "equities_minute"
    assert claim.budget_decision.next_eligible_at == now + timedelta(seconds=60)
    assert database.vendor_attempts_for_task("minute::twelve::AAPL") == []


def test_claim_next_vendor_task_clears_stale_budget_cooldown_when_affordable(
    tmp_path: Path,
) -> None:
    database = DataNodeDB(tmp_path / "node.sqlite")
    now = datetime(2026, 4, 29, 12, tzinfo=UTC)
    database.upsert_vendor_lane_health(
        vendor="finnhub",
        dataset="company_news",
        state="BUDGET_BLOCKED",
        cooldown_until=(now + timedelta(minutes=20)).isoformat(),
        recent_local_budget_blocks=3,
    )
    database.upsert_planner_task(
        task_key="news::finnhub::AAPL",
        task_family="supplemental_research",
        planner_group="supplemental_research_backlog",
        dataset="company_news",
        tier="B",
        priority=220,
        start_date="2026-04-01",
        end_date="2026-04-01",
        symbols=["AAPL"],
        eligible_vendors=["finnhub"],
        output_name="ticker_news",
        payload={"request_units": 1},
    )
    budget = BudgetManager({"finnhub": {"rpm": 57, "daily_cap": 82080}})

    claim = database.claim_next_vendor_task(
        vendor="finnhub",
        lease_owner="worker-1",
        task_families=("supplemental_research",),
        budget_decision_provider=lambda _task: budget.budget_decision(
            "finnhub", now=now
        ),
        now=now,
    )

    lane = database.get_vendor_lane_health(vendor="finnhub", dataset="company_news")
    assert claim.task is not None
    assert lane is not None
    assert lane.state == "HEALTHY"
    assert lane.cooldown_until is None


def test_claim_next_vendor_task_clears_stale_budget_task_backoff_when_affordable(
    tmp_path: Path,
) -> None:
    database = DataNodeDB(tmp_path / "node.sqlite")
    now = datetime(2026, 4, 29, 12, tzinfo=UTC)
    database.upsert_planner_task(
        task_key="news::finnhub::MSFT",
        task_family="supplemental_research",
        planner_group="supplemental_research_backlog",
        dataset="company_news",
        tier="B",
        priority=220,
        start_date="2026-04-01",
        end_date="2026-04-01",
        symbols=["MSFT"],
        eligible_vendors=["finnhub"],
        output_name="ticker_news",
        payload={"request_units": 1},
    )
    database.mark_planner_task_failed(
        "news::finnhub::MSFT",
        error="finnhub:company_news: budget exhausted",
        backoff_minutes=30,
    )
    budget = BudgetManager({"finnhub": {"rpm": 57, "daily_cap": 82080}})

    claim = database.claim_next_vendor_task(
        vendor="finnhub",
        lease_owner="worker-1",
        task_families=("supplemental_research",),
        budget_decision_provider=lambda _task: budget.budget_decision(
            "finnhub", now=now
        ),
        now=now,
    )

    assert claim.task is not None
    assert claim.task.task_key == "news::finnhub::MSFT"


def test_claim_next_vendor_task_retries_empty_success_when_task_still_partial(
    tmp_path: Path,
) -> None:
    database = DataNodeDB(tmp_path / "node.sqlite")
    database.upsert_planner_task(
        task_key="minute::alpaca::today",
        task_family="supplemental_research",
        planner_group="supplemental_research_backlog",
        dataset="equities_minute",
        tier="B",
        priority=205,
        start_date="2026-04-29",
        end_date="2026-04-29",
        symbols=["AAPL"],
        eligible_vendors=["alpaca"],
        output_name="equities_minute",
        payload={"request_units": 1},
    )
    database.mark_planner_task_partial(
        "minute::alpaca::today", error="empty result", backoff_minutes=0
    )
    claim_now = datetime.now(tz=UTC) + timedelta(seconds=1)
    assert (
        database.lease_vendor_attempt(
            task_key="minute::alpaca::today",
            task_family="supplemental_research",
            planner_group="supplemental_research_backlog",
            vendor="alpaca",
            lease_owner="worker-1",
            payload={},
            now=claim_now - timedelta(minutes=1),
        )
        is not None
    )
    database.mark_vendor_attempt_success(
        task_key="minute::alpaca::today", vendor="alpaca", rows_returned=0
    )
    budget = BudgetManager({"alpaca": {"rpm": 200, "daily_cap": 288000}})

    claim = database.claim_next_vendor_task(
        vendor="alpaca",
        lease_owner="worker-1",
        task_families=("supplemental_research",),
        budget_decision_provider=lambda _task: budget.budget_decision(
            "alpaca", now=claim_now
        ),
        now=claim_now,
    )

    assert claim.task is not None
    attempt = database.vendor_attempts_for_task("minute::alpaca::today")[0]
    assert attempt.status == "LEASED"
    assert attempt.rows_returned == 0


def test_fetch_recent_raw_partition_dates_returns_newest_dates_first(tmp_path: Path) -> None:
    database = DataNodeDB(tmp_path / "node.sqlite")
    for trading_date, status in [
        ("2025-01-01", "GREEN"),
        ("2025-01-02", "INCOMPLETE"),
        ("2025-01-03", "UNREADABLE"),
        ("2025-01-04", "QUARANTINED"),
        ("2025-01-05", "GREEN"),
    ]:
        database.upsert_raw_partition_manifest(
            dataset="equities_eod",
            trading_date=trading_date,
            partition_revision=1,
            symbol_count=1,
            row_count=1,
            symbols=["AAPL"],
            content_hash=f"hash-{trading_date}",
            status=status,
        )

    recent = database.fetch_recent_raw_partition_dates(
        dataset="equities_eod",
        statuses=("INCOMPLETE", "UNREADABLE", "QUARANTINED"),
        limit=2,
    )

    assert recent == ["2025-01-04", "2025-01-03"]


def test_scheduler_decision_and_archive_telemetry_summaries(tmp_path: Path) -> None:
    database = DataNodeDB(tmp_path / "node.sqlite")

    database.record_scheduler_decision(
        vendor="alpaca",
        dataset="equities_minute",
        decision="claimed",
        task_key="minute:aapl",
    )
    database.record_scheduler_decision(
        vendor="alpaca",
        dataset="equities_minute",
        decision="budget_blocked",
        reason="budget:minute",
    )
    database.record_archive_write_telemetry(
        output_name="ticker_news",
        partition_date="2026-04-29",
        status="success",
        rows_in=3,
        rows_written=2,
        duplicates_dropped=1,
        coerced_columns=["news_id", "published_at"],
    )

    scheduler = database.summarize_scheduler_decisions(minutes=15)
    archive = database.summarize_archive_write_telemetry(minutes=60)

    assert {row["decision"] for row in scheduler["rows"]} == {
        "claimed",
        "budget_blocked",
    }
    assert archive["rows"][0]["output_name"] == "ticker_news"
    assert archive["rows"][0]["rows_in"] == 3
    assert archive["rows"][0]["duplicates_dropped"] == 1
