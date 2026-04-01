from __future__ import annotations

from datetime import datetime, timedelta, timezone

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


def test_forward_reserve_allows_forward_when_other_tasks_are_blocked() -> None:
    manager = BudgetManager({"massive": {"rpm": 10, "daily_cap": 10}})
    now = datetime(2026, 1, 2, 12, 0, tzinfo=UTC)

    for offset in range(9):
        manager.record_spend("massive", task_kind="GAP", now=now + timedelta(minutes=offset))

    assert not manager.can_spend("massive", task_kind="GAP", now=now + timedelta(minutes=10))
    assert manager.can_spend("massive", task_kind="FORWARD", now=now + timedelta(minutes=10))
