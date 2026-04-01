"""Per-vendor request budget management."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone


UTC = timezone.utc


@dataclass(slots=True, frozen=True)
class VendorBudget:
    """Static per-vendor configuration."""

    rpm: int
    daily_cap: int


class BudgetManager:
    """Track per-vendor RPM and daily caps with FORWARD-task reserve support."""

    def __init__(self, config: dict[str, dict[str, int]], reserve_fraction: float = 0.10) -> None:
        self.reserve_fraction = reserve_fraction
        self.vendor_limits = {
            vendor: VendorBudget(rpm=values["rpm"], daily_cap=values["daily_cap"]) for vendor, values in config.items()
        }
        self.request_windows: dict[str, deque[datetime]] = defaultdict(deque)
        self.daily_spend: dict[str, dict[str, int]] = defaultdict(lambda: {"FORWARD": 0, "OTHER": 0})
        self.day_anchor = datetime.now(tz=UTC).date()

    def _normalize(self, now: datetime | None) -> datetime:
        current = now or datetime.now(tz=UTC)
        if current.tzinfo is None:
            current = current.replace(tzinfo=UTC)
        if current.date() != self.day_anchor:
            self.reset_daily(current)
        return current

    def _trim_window(self, vendor: str, now: datetime) -> None:
        window = self.request_windows[vendor]
        while window and (now - window[0]).total_seconds() >= 60:
            window.popleft()

    def can_spend(self, vendor: str, task_kind: str = "OTHER", now: datetime | None = None) -> bool:
        """Return whether a request can be issued for the vendor now."""
        current = self._normalize(now)
        self._trim_window(vendor, current)
        limits = self.vendor_limits[vendor]
        spend = self.daily_spend[vendor]

        if len(self.request_windows[vendor]) >= limits.rpm:
            return False

        total_spend = spend["FORWARD"] + spend["OTHER"]
        reserved_units = max(1, int(limits.daily_cap * self.reserve_fraction))
        non_forward_ceiling = max(0, limits.daily_cap - reserved_units)

        if task_kind == "FORWARD":
            return total_spend < limits.daily_cap
        return spend["OTHER"] < non_forward_ceiling and total_spend < limits.daily_cap

    def record_spend(self, vendor: str, task_kind: str = "OTHER", now: datetime | None = None) -> None:
        """Record a completed request."""
        current = self._normalize(now)
        self._trim_window(vendor, current)
        self.request_windows[vendor].append(current)
        bucket = "FORWARD" if task_kind == "FORWARD" else "OTHER"
        self.daily_spend[vendor][bucket] += 1

    def reset_daily(self, now: datetime | None = None) -> None:
        """Reset all daily counters, typically at midnight local-to-UTC boundary."""
        current = now or datetime.now(tz=UTC)
        if current.tzinfo is None:
            current = current.replace(tzinfo=UTC)
        self.day_anchor = current.date()
        self.daily_spend = defaultdict(lambda: {"FORWARD": 0, "OTHER": 0})
