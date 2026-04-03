from __future__ import annotations

import sqlite3
import threading
from datetime import timedelta
from pathlib import Path

import pytest

from trademl.data_node import db as db_module
from trademl.data_node.db import DataNodeDB


def test_enqueue_lease_done_cycle(tmp_path: Path) -> None:
    database = DataNodeDB(tmp_path / "node.sqlite")
    task_id = database.enqueue_task("equities_eod", "AAPL", "2024-01-01", "2024-01-31", "GAP", 1)

    leased = database.lease_next_task()

    assert leased is not None
    assert leased.id == task_id
    assert leased.status == "LEASED"

    database.mark_task_done(task_id)
    assert database.lease_next_task() is None


def test_failed_task_respects_backoff(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    database = DataNodeDB(tmp_path / "node.sqlite")
    task_id = database.enqueue_task("equities_eod", "MSFT", "2024-02-01", "2024-02-29", "GAP", 1)
    first_lease = database.lease_next_task()
    assert first_lease is not None

    now = db_module.utc_now()
    monkeypatch.setattr(db_module, "utc_now", lambda: now)
    database.mark_task_failed(task_id, "429", backoff_minutes=30)

    assert database.lease_next_task(now=now + timedelta(minutes=29)) is None
    leased_again = database.lease_next_task(now=now + timedelta(minutes=31))
    assert leased_again is not None
    assert leased_again.id == task_id


def test_duplicate_tasks_rejected(tmp_path: Path) -> None:
    database = DataNodeDB(tmp_path / "node.sqlite")
    database.enqueue_task("equities_eod", "NVDA", "2024-01-01", "2024-01-05", "BOOTSTRAP", 5)

    with pytest.raises(sqlite3.IntegrityError):
        database.enqueue_task("equities_eod", "NVDA", "2024-01-01", "2024-01-05", "BOOTSTRAP", 5)


def test_duplicate_gap_tasks_with_null_symbol_are_rejected(tmp_path: Path) -> None:
    database = DataNodeDB(tmp_path / "node.sqlite")
    database.enqueue_task("equities_eod", None, "2024-01-01", "2024-01-01", "GAP", 1)

    with pytest.raises(sqlite3.IntegrityError):
        database.enqueue_task("equities_eod", None, "2024-01-01", "2024-01-01", "GAP", 1)


def test_priority_ordering(tmp_path: Path) -> None:
    database = DataNodeDB(tmp_path / "node.sqlite")
    database.enqueue_task("equities_eod", "LOW", "2024-01-01", "2024-01-05", "GAP", 5)
    database.enqueue_task("equities_eod", "MID", "2024-01-01", "2024-01-05", "GAP", 1)
    database.enqueue_task("equities_eod", "HIGH", "2024-01-01", "2024-01-05", "GAP", 0)

    leased = [database.lease_next_task(), database.lease_next_task(), database.lease_next_task()]

    assert [task.symbol for task in leased if task is not None] == ["HIGH", "MID", "LOW"]


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


def test_requeue_retryable_failures_moves_rate_limited_rows_back_to_pending(tmp_path: Path) -> None:
    database = DataNodeDB(tmp_path / "node.sqlite")
    retryable_id = database.enqueue_task("equities_eod", None, "2024-01-01", "2024-01-01", "GAP", 1)
    permanent_id = database.enqueue_task("equities_eod", None, "2024-01-02", "2024-01-02", "GAP", 1)
    database.mark_task_failed(retryable_id, 'tiingo request failed: 429 {"detail":"hourly request allocation"}', backoff_minutes=30)
    database.mark_task_failed(permanent_id, "alpaca: permanent data schema failure", backoff_minutes=30)

    moved = database.requeue_retryable_failures()

    assert moved == 1
    with sqlite3.connect(tmp_path / "node.sqlite") as connection:
        statuses = dict(connection.execute("SELECT id, status FROM backfill_queue").fetchall())
    assert statuses[retryable_id] == "PENDING"
    assert statuses[permanent_id] == "FAILED"


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
