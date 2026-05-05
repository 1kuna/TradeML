"""Adaptive collection concurrency control for Pi vendor lanes."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_VENDOR_CAPS = {
    "alpaca": 16,
    "tiingo": 4,
    "finnhub": 4,
    "fred": 4,
    "sec_edgar": 2,
    "twelve_data": 1,
    "massive": 1,
    "fmp": 1,
    "alpha_vantage": 1,
}


@dataclass(slots=True)
class LaneControlInput:
    """Inputs used to compute one vendor/dataset target width."""

    vendor: str
    dataset: str
    base_width: int
    eligible_tasks: int
    active_width: int
    rpm: int
    remaining_minute: int
    remaining_daily: int
    recent_429s: int = 0
    p95_latency_ms: float | None = None
    rows_per_credit: float | None = None


@dataclass(slots=True)
class LaneControlDecision:
    """Controller decision for one vendor/dataset lane."""

    vendor: str
    dataset: str
    eligible_tasks: int
    target_width: int
    active_width: int
    budget_remaining: dict[str, int]
    latency_ms: float | None
    rows_per_credit: float | None
    action: str
    reason: str


class SaturationControllerV2:
    """Compute greedy but safe collection lane widths from live telemetry."""

    def __init__(
        self,
        *,
        target_utilization: float = 0.98,
        global_worker_multiplier: int = 4,
        vendor_caps: dict[str, int] | None = None,
        enabled: bool = True,
    ) -> None:
        self.enabled = bool(enabled)
        self.target_utilization = min(1.0, max(0.1, float(target_utilization)))
        self.global_worker_multiplier = max(1, int(global_worker_multiplier))
        self.vendor_caps = {**DEFAULT_VENDOR_CAPS, **dict(vendor_caps or {})}

    @property
    def global_cap(self) -> int:
        """Return the safe process-wide I/O concurrency cap."""
        return max(8, int(os.cpu_count() or 1) * self.global_worker_multiplier)

    @classmethod
    def from_environment(cls) -> "SaturationControllerV2":
        """Build the controller from config-exported runtime environment values."""
        return cls(
            enabled=_env_bool("TRADEML_SATURATION_CONTROLLER_V2_ENABLED", default=True),
            target_utilization=float(
                os.getenv("TRADEML_SATURATION_CONTROLLER_V2_TARGET_UTILIZATION", "0.98")
            ),
            global_worker_multiplier=int(
                os.getenv("TRADEML_SATURATION_CONTROLLER_V2_GLOBAL_WORKER_MULTIPLIER", "4")
            ),
        )

    def decide(
        self,
        lane: LaneControlInput,
        *,
        resource_state: dict[str, Any] | None = None,
    ) -> LaneControlDecision:
        """Return the target concurrency and reason for one vendor/dataset lane."""
        resources = resource_state or {}
        budget = {
            "minute": max(0, int(lane.remaining_minute)),
            "daily": max(0, int(lane.remaining_daily)),
        }
        if not self.enabled:
            return self._decision(lane, max(1, int(lane.base_width)), "static", "controller_disabled", budget)
        if lane.eligible_tasks <= 0:
            return self._decision(lane, 0, "intentional_idle", "no_eligible_work", budget)
        if budget["minute"] <= 0:
            return self._decision(lane, 0, "budget_blocked", "minute_budget_exhausted", budget)
        if budget["daily"] <= 0:
            return self._decision(lane, 0, "budget_blocked", "daily_budget_exhausted", budget)
        cap = min(self.global_cap, self.vendor_caps.get(lane.vendor, 1))
        if int(lane.recent_429s or 0) > 0:
            target = min(cap, budget["minute"], max(1, int(lane.base_width)))
            return self._decision(lane, target, "backoff", "recent_remote_rate_limit", budget)
        if _resource_pressure(resources):
            target = min(cap, budget["minute"], max(1, int(lane.base_width)))
            return self._decision(lane, target, "backoff", str(resources.get("reason") or "resource_pressure"), budget)
        if _slow_nas(resources):
            target = min(cap, budget["minute"], max(1, int(lane.base_width)))
            return self._decision(lane, target, "backoff", "slow_nas_writes", budget)

        latency_ms = max(2500.0, float(lane.p95_latency_ms or 2500.0))
        requests_per_worker_per_minute = max(1.0, 60000.0 / latency_ms)
        target_rpm = max(1.0, min(float(lane.rpm or budget["minute"]), float(budget["minute"])) * self.target_utilization)
        needed = int(math.ceil(target_rpm / requests_per_worker_per_minute))
        target = max(int(lane.base_width), needed)
        target = min(cap, budget["minute"], max(1, int(lane.eligible_tasks)), target)
        if cap <= 1 and lane.vendor in {"twelve_data", "fmp", "alpha_vantage", "massive"}:
            action = "paced"
            reason = "low_rpm_vendor_paced"
        elif target > max(1, int(lane.base_width)):
            action = "scale_up"
            reason = "budget_backlog_latency_allow_more_lanes"
        else:
            action = "hold"
            reason = "target_width_matches_current_policy"
        return self._decision(lane, max(1, target), action, reason, budget)

    @staticmethod
    def _decision(
        lane: LaneControlInput,
        target_width: int,
        action: str,
        reason: str,
        budget: dict[str, int],
    ) -> LaneControlDecision:
        return LaneControlDecision(
            vendor=lane.vendor,
            dataset=lane.dataset,
            eligible_tasks=max(0, int(lane.eligible_tasks)),
            target_width=max(0, int(target_width)),
            active_width=max(0, int(lane.active_width)),
            budget_remaining=budget,
            latency_ms=float(lane.p95_latency_ms) if lane.p95_latency_ms is not None else None,
            rows_per_credit=float(lane.rows_per_credit) if lane.rows_per_credit is not None else None,
            action=action,
            reason=reason,
        )


def _env_bool(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _resource_pressure(resources: dict[str, Any]) -> bool:
    status = str(resources.get("status") or "").lower()
    if status in {"pressure", "critical"}:
        return True
    load = resources.get("load_average_1m")
    cpu_count = int(resources.get("cpu_count") or os.cpu_count() or 1)
    try:
        return float(load) > float(cpu_count) * 2.0
    except (TypeError, ValueError):
        return False


def _slow_nas(resources: dict[str, Any]) -> bool:
    try:
        return float(resources.get("nas_write_latency_ms") or 0.0) >= 5000.0
    except (TypeError, ValueError):
        return False


def local_resource_state(*, data_root: Path | None = None) -> dict[str, Any]:
    """Return lightweight local resource state for the controller."""
    cpu_count = int(os.cpu_count() or 1)
    load = os.getloadavg()[0] if hasattr(os, "getloadavg") else None
    memory = _linux_memory_state()
    state: dict[str, Any] = {
        "cpu_count": cpu_count,
        "load_average_1m": load,
        **memory,
    }
    if data_root is not None and data_root.exists():
        state["nas_write_latency_ms"] = None
    if _resource_pressure(state):
        state["reason"] = "high_load_or_memory_pressure"
    return state


def _linux_memory_state() -> dict[str, Any]:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return {"status": "unknown"}
    values: dict[str, int] = {}
    for line in meminfo.read_text(encoding="utf-8", errors="ignore").splitlines():
        key, _, value = line.partition(":")
        if not value:
            continue
        pieces = value.strip().split()
        if not pieces:
            continue
        with_value = pieces[0]
        try:
            values[key] = int(with_value) * 1024
        except ValueError:
            continue
    total = values.get("MemTotal") or 0
    available = values.get("MemAvailable") or 0
    if total <= 0:
        return {"status": "unknown"}
    available_ratio = available / total
    status = "pressure" if available_ratio < 0.10 or available < 256 * 1024 * 1024 else "ok"
    return {
        "status": status,
        "memory_total_bytes": total,
        "memory_available_bytes": available,
        "memory_available_ratio": round(available_ratio, 6),
    }
