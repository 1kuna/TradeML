from __future__ import annotations

from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor
import json
import os

from trademl.data_node.budgets import BudgetManager


UTC = timezone.utc


def test_rpm_limiting() -> None:
    manager = BudgetManager({"alpaca": {"rpm": 2, "daily_cap": 10}})
    now = datetime(2026, 1, 2, 12, 0, tzinfo=UTC)

    assert manager.can_spend("alpaca", now=now)
    manager.record_spend("alpaca", now=now)
    manager.record_spend("alpaca", now=now + timedelta(seconds=1))
    assert not manager.can_spend("alpaca", now=now + timedelta(seconds=2))
    assert manager.can_spend("alpaca", now=now + timedelta(seconds=61))


def test_daily_cap_enforcement() -> None:
    manager = BudgetManager({"fmp": {"rpm": 10, "daily_cap": 5}})
    now = datetime(2026, 1, 2, 12, 0, tzinfo=UTC)

    for offset in range(4):
        assert manager.can_spend("fmp", now=now + timedelta(minutes=offset))
        manager.record_spend("fmp", now=now + timedelta(minutes=offset))

    assert not manager.can_spend("fmp", now=now + timedelta(minutes=5))
    manager.reset_daily(now=now + timedelta(days=1))
    assert manager.can_spend("fmp", now=now + timedelta(days=1))


def test_daily_cap_enforcement_uses_weighted_units() -> None:
    manager = BudgetManager({"twelve_data": {"rpm": 10, "daily_cap": 5}})
    now = datetime(2026, 1, 2, 12, 0, tzinfo=UTC)

    assert manager.can_spend("twelve_data", task_kind="FORWARD", units=4, now=now)
    manager.record_spend("twelve_data", task_kind="FORWARD", units=4, now=now)

    assert not manager.can_spend("twelve_data", task_kind="FORWARD", units=2, now=now + timedelta(minutes=1))
    assert manager.can_spend("twelve_data", task_kind="FORWARD", units=1, now=now + timedelta(minutes=1))


def test_forward_reserve_allows_forward_when_other_tasks_are_blocked() -> None:
    manager = BudgetManager({"massive": {"rpm": 10, "daily_cap": 10}})
    now = datetime(2026, 1, 2, 12, 0, tzinfo=UTC)

    for offset in range(9):
        manager.record_spend("massive", task_kind="GAP", now=now + timedelta(minutes=offset))

    assert not manager.can_spend("massive", task_kind="GAP", now=now + timedelta(minutes=10))
    assert manager.can_spend("massive", task_kind="FORWARD", now=now + timedelta(minutes=10))


def test_budget_manager_persists_snapshot(tmp_path) -> None:
    snapshot_path = tmp_path / "budget_state.json"
    manager = BudgetManager(
        {"alpaca": {"rpm": 2, "daily_cap": 10}},
        snapshot_path=snapshot_path,
    )
    now = datetime(2026, 1, 2, 12, 0, tzinfo=UTC)

    manager.record_spend("alpaca", task_kind="FORWARD", now=now)
    manager.record_spend("alpaca", task_kind="OTHER", now=now + timedelta(seconds=1))

    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert payload["day_anchor"] == "2026-01-02"
    assert payload["writer_pid"] == os.getpid()
    assert payload["vendors"]["alpaca"]["daily_spend"]["FORWARD"] == 1
    assert payload["vendors"]["alpaca"]["daily_spend"]["OTHER"] == 1
    assert payload["vendors"]["alpaca"]["daily_spend"]["TOTAL"] == 2
    assert payload["vendors"]["alpaca"]["daily_requests"]["TOTAL"] == 2
    assert len(payload["vendors"]["alpaca"]["window_timestamps"]) == 2


def test_budget_manager_persists_weighted_units_and_request_counts(tmp_path) -> None:
    snapshot_path = tmp_path / "budget_state.json"
    manager = BudgetManager(
        {"twelve_data": {"rpm": 10, "daily_cap": 100}},
        snapshot_path=snapshot_path,
    )
    now = datetime(2026, 1, 2, 12, 0, tzinfo=UTC)

    manager.record_spend("twelve_data", task_kind="FORWARD", units=8, now=now)
    manager.record_spend("twelve_data", task_kind="OTHER", units=3, now=now + timedelta(seconds=1))

    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert payload["vendors"]["twelve_data"]["daily_spend"]["TOTAL"] == 11
    assert payload["vendors"]["twelve_data"]["daily_requests"]["TOTAL"] == 2


def test_budget_manager_snapshot_persistence_is_thread_safe(tmp_path) -> None:
    snapshot_path = tmp_path / "budget_state.json"
    manager = BudgetManager(
        {"alpaca": {"rpm": 1000, "daily_cap": 1000}},
        snapshot_path=snapshot_path,
    )
    now = datetime(2026, 1, 2, 12, 0, tzinfo=UTC)

    def _spend(offset: int) -> None:
        manager.record_spend("alpaca", task_kind="OTHER", now=now + timedelta(seconds=offset))

    with ThreadPoolExecutor(max_workers=16) as executor:
        list(executor.map(_spend, range(64)))

    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert payload["vendors"]["alpaca"]["daily_spend"]["TOTAL"] == 64


def test_budget_manager_restores_same_day_snapshot_on_restart(tmp_path) -> None:
    snapshot_path = tmp_path / "budget_state.json"
    now = datetime.now(tz=UTC).replace(microsecond=0)

    first = BudgetManager(
        {"alpaca": {"rpm": 2, "daily_cap": 10}},
        snapshot_path=snapshot_path,
    )
    first.record_spend("alpaca", task_kind="FORWARD", now=now)
    first.record_spend("alpaca", task_kind="OTHER", now=now + timedelta(seconds=1))

    restored = BudgetManager(
        {"alpaca": {"rpm": 2, "daily_cap": 10}},
        snapshot_path=snapshot_path,
    )

    payload = restored.snapshot(now=now + timedelta(seconds=2))
    vendor = payload["vendors"]["alpaca"]
    assert vendor["daily_spend"]["TOTAL"] == 2
    assert vendor["daily_requests"]["TOTAL"] == 2
    assert vendor["window_used"] == 2
    assert not restored.can_spend("alpaca", now=now + timedelta(seconds=2))
