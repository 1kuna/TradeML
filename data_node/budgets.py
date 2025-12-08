"""
Budget manager for the unified Pi data-node.

Implements:
- Token bucket for per-vendor RPM (requests-per-minute) rate limiting
- Daily budget caps with 85/90/100% slices by task kind
- Persistence to JSON for crash recovery

See updated_node_spec.md §1.2 for budget slice semantics.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml
from loguru import logger

from .db import TaskKind


@dataclass
class VendorBudget:
    """Budget state for a single vendor."""
    hard_rpm: int                    # Max requests per minute
    soft_daily_cap: int              # Daily budget cap
    spent_today: int = 0             # Requests spent today
    last_reset: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Token bucket state for RPM
    tokens: float = field(default=0.0)
    last_token_update: float = field(default_factory=time.time)


# Budget slice thresholds (fraction of daily cap)
SLICE_BACKFILL = 0.85   # BOOTSTRAP + GAP can spend up to 85%
SLICE_QC = 0.90         # QC_PROBE can spend up to 90%
SLICE_FORWARD = 1.00    # FORWARD can spend up to 100%


class BudgetManager:
    """
    Thread-safe budget manager for vendor API limits.

    Enforces:
    - Per-vendor RPM via token bucket
    - Daily caps with kind-based slices:
      - 0-85%: BOOTSTRAP + GAP
      - 85-90%: QC_PROBE + FORWARD
      - 90-100%: FORWARD only
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        state_path: Optional[Path] = None,
    ):
        """
        Initialize the budget manager.

        Args:
            config_path: Path to backfill.yml (default: configs/backfill.yml)
            state_path: Path to persist state (default: data_layer/control/budgets.json)
        """
        self._lock = threading.RLock()  # Reentrant lock for try_spend()
        self._budgets: dict[str, VendorBudget] = {}

        # Resolve paths
        if config_path is None:
            config_path = Path("configs/backfill.yml")
        if state_path is None:
            data_root = os.environ.get("DATA_ROOT", ".")
            state_path = Path(data_root) / "data_layer" / "control" / "budgets.json"

        self.config_path = Path(config_path)
        self.state_path = Path(state_path)

        # Load config and state
        self._load_config()
        self._load_state()

    def _load_config(self) -> None:
        """Load rate limits from config file."""
        if not self.config_path.exists():
            logger.warning(f"Config file not found: {self.config_path}, using defaults")
            self._set_defaults()
            return

        with open(self.config_path) as f:
            config = yaml.safe_load(f)

        rate_limits = config.get("policy", {}).get("rate_limits", {})

        if not rate_limits:
            logger.warning("No rate_limits in config, using defaults")
            self._set_defaults()
            return

        for vendor, limits in rate_limits.items():
            rpm = limits.get("hard_rpm", 10)
            self._budgets[vendor] = VendorBudget(
                hard_rpm=rpm,
                soft_daily_cap=limits.get("soft_daily_cap", 1000),
                tokens=float(rpm),  # Start with full token bucket
            )
            logger.debug(f"Loaded budget for {vendor}: rpm={rpm}, daily={limits.get('soft_daily_cap')}")

    def _set_defaults(self) -> None:
        """Set default budget values based on verified free tier limits (Dec 2025).

        Daily caps = rpm * 60 * 24 (theoretical max), except AV which has hard 25/day.
        """
        defaults = {
            # Alpaca: 200 rpm free tier → 288,000/day theoretical
            "alpaca": VendorBudget(hard_rpm=200, soft_daily_cap=288_000, tokens=200.0),
            # Finnhub: 60 rpm free tier → 86,400/day theoretical
            "finnhub": VendorBudget(hard_rpm=60, soft_daily_cap=86_400, tokens=60.0),
            # Alpha Vantage: 5 rpm, 25/day HARD CAP
            "av": VendorBudget(hard_rpm=5, soft_daily_cap=25, tokens=5.0),
            # FRED: ~120 rpm → 172,800/day theoretical
            "fred": VendorBudget(hard_rpm=120, soft_daily_cap=172_800, tokens=120.0),
            # Massive: 5 rpm free tier → 7,200/day theoretical
            "massive": VendorBudget(hard_rpm=5, soft_daily_cap=7_200, tokens=5.0),
            # FMP removed - free tier too limited
        }
        self._budgets = defaults

    def _load_state(self) -> None:
        """Load persisted state from JSON file."""
        if not self.state_path.exists():
            logger.debug("No budget state file found, starting fresh")
            return

        try:
            with open(self.state_path) as f:
                state = json.load(f)

            now = datetime.now(timezone.utc)

            for vendor, data in state.items():
                if vendor not in self._budgets:
                    continue

                # Parse last_reset
                last_reset = datetime.fromisoformat(data.get("last_reset", now.isoformat()))

                # Check if we need to reset (new day in UTC)
                if last_reset.date() < now.date():
                    logger.info(f"Budget for {vendor} reset for new day")
                    self._budgets[vendor].spent_today = 0
                    self._budgets[vendor].last_reset = now
                else:
                    self._budgets[vendor].spent_today = data.get("spent_today", 0)
                    self._budgets[vendor].last_reset = last_reset

                # Restore token bucket state
                self._budgets[vendor].tokens = data.get("tokens", self._budgets[vendor].hard_rpm)
                self._budgets[vendor].last_token_update = data.get("last_token_update", time.time())

            logger.debug(f"Loaded budget state from {self.state_path}")
        except Exception as e:
            logger.warning(f"Failed to load budget state: {e}, starting fresh")

    def _save_state(self) -> None:
        """Persist state to JSON file."""
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)

            state = {}
            for vendor, budget in self._budgets.items():
                state[vendor] = {
                    "spent_today": budget.spent_today,
                    "last_reset": budget.last_reset.isoformat(),
                    "tokens": budget.tokens,
                    "last_token_update": budget.last_token_update,
                }

            # Atomic write
            tmp_path = self.state_path.with_suffix(".tmp")
            with open(tmp_path, "w") as f:
                json.dump(state, f, indent=2)
            tmp_path.rename(self.state_path)
        except Exception as e:
            logger.warning(f"Failed to save budget state: {e}")

    def reset(self) -> None:
        """Reset all budgets to defaults by deleting state file and reloading."""
        with self._lock:
            # Delete persisted state
            if self.state_path.exists():
                self.state_path.unlink()
                logger.info(f"Deleted budget state file: {self.state_path}")

            # Reload config (which will set defaults if no config)
            self._budgets.clear()
            self._load_config()
            logger.info("Budget state reset to defaults")

            # Log new values
            for vendor, budget in self._budgets.items():
                logger.info(f"  {vendor}: {budget.hard_rpm} rpm, {budget.soft_daily_cap:,}/day")

    def _refill_tokens(self, vendor: str) -> None:
        """Refill tokens based on elapsed time (token bucket algorithm)."""
        budget = self._budgets.get(vendor)
        if budget is None:
            return

        now = time.time()
        elapsed = now - budget.last_token_update

        # Refill rate: hard_rpm tokens per 60 seconds
        refill = elapsed * (budget.hard_rpm / 60.0)
        budget.tokens = min(budget.hard_rpm, budget.tokens + refill)
        budget.last_token_update = now

    def _check_daily_reset(self, vendor: str) -> None:
        """Reset daily counter if we've crossed into a new UTC day."""
        budget = self._budgets.get(vendor)
        if budget is None:
            return

        now = datetime.now(timezone.utc)
        if budget.last_reset.date() < now.date():
            logger.info(f"Daily budget reset for {vendor}")
            budget.spent_today = 0
            budget.last_reset = now

    def can_spend(self, vendor: str, kind: TaskKind, tokens: int = 1) -> bool:
        """
        Check if we can spend tokens for a given vendor and task kind.

        Implements the 85/90/100% budget slices:
        - BOOTSTRAP/GAP: only if spent < 85% of daily cap
        - QC_PROBE: only if spent < 90% of daily cap
        - FORWARD: only if spent < 100% of daily cap

        Args:
            vendor: Vendor name
            kind: Task kind
            tokens: Number of requests/tokens to spend

        Returns:
            True if spending is allowed
        """
        with self._lock:
            budget = self._budgets.get(vendor)
            if budget is None:
                logger.debug(f"Rejecting unknown vendor: {vendor}")
                return False

            self._check_daily_reset(vendor)
            self._refill_tokens(vendor)

            # Check token bucket (RPM)
            if budget.tokens < tokens:
                logger.debug(f"{vendor} RPM limit: {budget.tokens:.1f} tokens < {tokens}")
                return False

            # Check daily slice based on kind
            if kind in (TaskKind.BOOTSTRAP, TaskKind.GAP):
                threshold = SLICE_BACKFILL * budget.soft_daily_cap
            elif kind == TaskKind.QC_PROBE:
                threshold = SLICE_QC * budget.soft_daily_cap
            else:  # FORWARD
                threshold = SLICE_FORWARD * budget.soft_daily_cap

            if budget.spent_today + tokens > threshold:
                pct = budget.spent_today / budget.soft_daily_cap * 100
                logger.debug(f"{vendor} daily limit for {kind.value}: {pct:.1f}% spent, threshold {threshold}")
                return False

            return True

    def spend(self, vendor: str, tokens: int = 1) -> bool:
        """
        Record spending tokens for a vendor.

        Should be called after a successful API call.

        Args:
            vendor: Vendor name
            tokens: Number of requests/tokens spent

        Returns:
            True if recorded, False if vendor unknown
        """
        with self._lock:
            budget = self._budgets.get(vendor)
            if budget is None:
                return False

            self._check_daily_reset(vendor)
            self._refill_tokens(vendor)

            # Deduct from token bucket
            budget.tokens = max(0, budget.tokens - tokens)

            # Increment daily spent
            budget.spent_today += tokens

            # Persist periodically (every 10 calls)
            if budget.spent_today % 10 == 0:
                self._save_state()

            return True

    def try_spend(self, vendor: str, kind: TaskKind, tokens: int = 1) -> bool:
        """
        Atomically check and spend if allowed.

        Combines can_spend() and spend() in one atomic operation.

        Args:
            vendor: Vendor name
            kind: Task kind
            tokens: Number of requests/tokens

        Returns:
            True if spending was allowed and recorded
        """
        with self._lock:
            if not self.can_spend(vendor, kind, tokens):
                return False
            return self.spend(vendor, tokens)

    def get_budget_status(self, vendor: str) -> dict:
        """
        Get current budget status for a vendor.

        Returns:
            Dict with spent_today, soft_daily_cap, remaining, pct_used, tokens_rpm
        """
        with self._lock:
            budget = self._budgets.get(vendor)
            if budget is None:
                return {}

            self._check_daily_reset(vendor)
            self._refill_tokens(vendor)

            remaining = budget.soft_daily_cap - budget.spent_today
            pct_used = (budget.spent_today / budget.soft_daily_cap * 100) if budget.soft_daily_cap > 0 else 0

            return {
                "spent_today": budget.spent_today,
                "soft_daily_cap": budget.soft_daily_cap,
                "remaining": remaining,
                "pct_used": pct_used,
                "tokens_rpm": budget.tokens,
                "hard_rpm": budget.hard_rpm,
            }

    def get_all_status(self) -> dict[str, dict]:
        """Get budget status for all vendors."""
        return {vendor: self.get_budget_status(vendor) for vendor in self._budgets}

    def eligible_vendors_for_kind(self, kind: TaskKind) -> list[str]:
        """
        Get list of vendors that have budget for a given task kind.

        Args:
            kind: Task kind

        Returns:
            List of vendor names with available budget
        """
        eligible = []
        for vendor in self._budgets:
            if self.can_spend(vendor, kind):
                eligible.append(vendor)
        return eligible

    def save(self) -> None:
        """Force save current state."""
        with self._lock:
            self._save_state()


# Module-level singleton
_default_manager: Optional[BudgetManager] = None


def get_budget_manager(
    config_path: Optional[Path] = None,
    state_path: Optional[Path] = None,
) -> BudgetManager:
    """Get the default BudgetManager instance."""
    global _default_manager

    if _default_manager is None:
        _default_manager = BudgetManager(config_path, state_path)

    return _default_manager


def reset_budget_manager() -> None:
    """Reset the default manager (for testing)."""
    global _default_manager
    _default_manager = None
