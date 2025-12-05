"""
Tests for data_node.budgets module.

Tests:
- Token bucket rate limiting
- Daily budget slices (85/90/100%)
- Budget persistence
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
import time

from data_node.db import TaskKind
from data_node.budgets import BudgetManager, SLICE_BACKFILL, SLICE_QC, SLICE_FORWARD


class TestBudgetSlices:
    """Tests for budget slice enforcement."""

    def test_bootstrap_uses_backfill_slice(self, budget_manager):
        """Test that BOOTSTRAP uses the 85% slice."""
        # Should be able to spend initially
        assert budget_manager.can_spend("alpaca", TaskKind.BOOTSTRAP)

        # Get the budget
        status = budget_manager.get_budget_status("alpaca")
        daily_cap = status["soft_daily_cap"]
        threshold = int(SLICE_BACKFILL * daily_cap)

        # Simulate spending up to threshold
        for _ in range(threshold):
            budget_manager.spend("alpaca")

        # Now BOOTSTRAP should be blocked
        assert not budget_manager.can_spend("alpaca", TaskKind.BOOTSTRAP)

        # But FORWARD should still work (uses 100% slice)
        assert budget_manager.can_spend("alpaca", TaskKind.FORWARD)

    def test_gap_uses_backfill_slice(self, budget_manager):
        """Test that GAP uses the 85% slice."""
        assert budget_manager.can_spend("alpaca", TaskKind.GAP)

        status = budget_manager.get_budget_status("alpaca")
        daily_cap = status["soft_daily_cap"]
        threshold = int(SLICE_BACKFILL * daily_cap)

        for _ in range(threshold):
            budget_manager.spend("alpaca")

        assert not budget_manager.can_spend("alpaca", TaskKind.GAP)

    def test_qc_probe_uses_qc_slice(self, budget_manager):
        """Test that QC_PROBE uses the 90% slice."""
        status = budget_manager.get_budget_status("alpaca")
        daily_cap = status["soft_daily_cap"]

        # Spend up to 85%
        backfill_threshold = int(SLICE_BACKFILL * daily_cap)
        for _ in range(backfill_threshold):
            budget_manager.spend("alpaca")

        # BOOTSTRAP should be blocked, but QC_PROBE should work
        assert not budget_manager.can_spend("alpaca", TaskKind.BOOTSTRAP)
        assert budget_manager.can_spend("alpaca", TaskKind.QC_PROBE)

        # Spend up to 90%
        qc_threshold = int(SLICE_QC * daily_cap) - backfill_threshold
        for _ in range(qc_threshold):
            budget_manager.spend("alpaca")

        # Now QC_PROBE should be blocked
        assert not budget_manager.can_spend("alpaca", TaskKind.QC_PROBE)

        # But FORWARD should still work
        assert budget_manager.can_spend("alpaca", TaskKind.FORWARD)

    def test_forward_uses_full_slice(self, budget_manager):
        """Test that FORWARD can use the full 100% budget."""
        status = budget_manager.get_budget_status("alpaca")
        daily_cap = status["soft_daily_cap"]

        # Spend 99% of budget
        almost_all = int(0.99 * daily_cap)
        for _ in range(almost_all):
            budget_manager.spend("alpaca")

        # FORWARD should still work
        assert budget_manager.can_spend("alpaca", TaskKind.FORWARD)

        # Spend the rest
        remaining = daily_cap - almost_all
        for _ in range(remaining):
            budget_manager.spend("alpaca")

        # Now FORWARD should be blocked
        assert not budget_manager.can_spend("alpaca", TaskKind.FORWARD)


class TestTokenBucket:
    """Tests for RPM rate limiting."""

    def test_rpm_limit_blocks_burst(self, budget_manager):
        """Test that RPM limit blocks rapid requests."""
        status = budget_manager.get_budget_status("alpaca")
        hard_rpm = status["hard_rpm"]

        # Spend tokens quickly
        for _ in range(hard_rpm):
            if budget_manager.can_spend("alpaca", TaskKind.FORWARD):
                budget_manager.spend("alpaca")

        # Should be rate limited now (tokens exhausted)
        assert not budget_manager.can_spend("alpaca", TaskKind.FORWARD)

    def test_rpm_refills_over_time(self, budget_manager_rpm):
        """Test that tokens refill over time.

        Uses budget_manager_rpm fixture with high daily_cap to test RPM in isolation.
        """
        status = budget_manager_rpm.get_budget_status("alpaca")
        hard_rpm = status["hard_rpm"]  # 60 in this fixture

        # Exhaust all tokens
        for _ in range(hard_rpm):
            if budget_manager_rpm.can_spend("alpaca", TaskKind.FORWARD):
                budget_manager_rpm.spend("alpaca")

        # Should be rate limited now (tokens exhausted)
        assert not budget_manager_rpm.can_spend("alpaca", TaskKind.FORWARD)

        # Wait for some refill (1.5 seconds = 1.5 * 60/60 = 1.5 tokens)
        time.sleep(1.5)

        # Should have some tokens now (daily cap is 10000, so not a limit)
        assert budget_manager_rpm.can_spend("alpaca", TaskKind.FORWARD)


class TestTrySpend:
    """Tests for atomic try_spend operation."""

    def test_try_spend_success(self, budget_manager):
        """Test successful atomic spend."""
        initial = budget_manager.get_budget_status("alpaca")

        result = budget_manager.try_spend("alpaca", TaskKind.FORWARD)

        assert result is True

        after = budget_manager.get_budget_status("alpaca")
        assert after["spent_today"] == initial["spent_today"] + 1

    def test_try_spend_failure_budget(self, budget_manager):
        """Test atomic spend failure due to budget."""
        status = budget_manager.get_budget_status("alpaca")
        daily_cap = status["soft_daily_cap"]

        # Exhaust budget
        for _ in range(daily_cap):
            budget_manager.spend("alpaca")

        result = budget_manager.try_spend("alpaca", TaskKind.FORWARD)

        assert result is False


class TestBudgetStatus:
    """Tests for budget status reporting."""

    def test_get_all_status(self, budget_manager):
        """Test getting status for all vendors."""
        all_status = budget_manager.get_all_status()

        assert "alpaca" in all_status
        assert "finnhub" in all_status

        for vendor, status in all_status.items():
            assert "spent_today" in status
            assert "soft_daily_cap" in status
            assert "remaining" in status
            assert "pct_used" in status

    def test_eligible_vendors_for_kind(self, budget_manager):
        """Test getting eligible vendors for a task kind."""
        eligible = budget_manager.eligible_vendors_for_kind(TaskKind.FORWARD)

        assert len(eligible) > 0
        assert "alpaca" in eligible
