from __future__ import annotations

import pytest

from data_node.budgets import BudgetManager, reset_budget_manager
from data_node.db import TaskKind


@pytest.fixture()
def budget_manager(temp_data_root, tmp_path):
    """
    Budget manager backed by a small test config and isolated state file.
    """
    reset_budget_manager()
    cfg_path = tmp_path / "backfill.yml"
    cfg_path.write_text(
        """
policy:
  rate_limits:
    alpaca:
      hard_rpm: 20
      soft_daily_cap: 10
"""
    )
    state_path = temp_data_root / "data_layer" / "control" / "budgets.json"
    mgr = BudgetManager(config_path=cfg_path, state_path=state_path)
    yield mgr, cfg_path, state_path
    reset_budget_manager()


def test_budget_daily_and_rpm_enforced(budget_manager):
    manager, _, _ = budget_manager

    budget = manager._budgets["alpaca"]
    budget.tokens = 100  # bypass RPM for this check

    # Daily slice: BOOTSTRAP limited to 85% of cap (8.5 → floor at 8 tokens)
    assert manager.can_spend("alpaca", TaskKind.BOOTSTRAP, tokens=8)
    assert not manager.can_spend("alpaca", TaskKind.BOOTSTRAP, tokens=9)

    # RPM bucket decrements on spend and blocks oversized bursts
    budget.tokens = 5
    assert manager.try_spend("alpaca", TaskKind.BOOTSTRAP, tokens=2)
    status = manager.get_budget_status("alpaca")
    assert status["spent_today"] == 2
    assert status["tokens_rpm"] <= 20
    assert not manager.can_spend("alpaca", TaskKind.BOOTSTRAP, tokens=4)  # more than remaining RPM tokens


def test_budget_state_persists_and_recovers(budget_manager):
    manager, cfg_path, state_path = budget_manager
    assert manager.try_spend("alpaca", TaskKind.FORWARD, tokens=3)
    manager.save()

    restored = BudgetManager(config_path=cfg_path, state_path=state_path)
    restored_status = restored.get_budget_status("alpaca")

    assert restored_status["spent_today"] >= 3
    assert restored_status["hard_rpm"] == 20
