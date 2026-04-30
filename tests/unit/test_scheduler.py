from __future__ import annotations

import logging
import threading
import time
from types import SimpleNamespace

from trademl.data_node.scheduler import PlannerLaneScheduler


class _SchedulerDB:
    def __init__(self) -> None:
        self.decisions: list[dict[str, str]] = []

    def has_pending_planner_tasks(self, *args, **kwargs) -> bool:  # noqa: ANN002, ANN003
        return True

    def clear_expired_vendor_lane_cooldowns(self) -> int:
        return 0

    def clear_stale_idle_vendor_lanes(self) -> int:
        return 0

    def record_scheduler_decision(self, **kwargs) -> None:  # noqa: ANN003
        self.decisions.append(dict(kwargs))


class _SchedulerService:
    worker_id = "worker-test"

    def __init__(self, *, actionable: bool, reason: str) -> None:
        self.db = _SchedulerDB()
        self._stop_event = threading.Event()
        self._actionable = actionable
        self._reason = reason
        self._canonical_runtime = SimpleNamespace(_backfill_lane_widths=lambda: {})

    def _ensure_planner_backlog_seeded(self, **kwargs) -> None:  # noqa: ANN003
        return None

    def _reconcile_budget_blocked_auxiliary_lanes(self) -> int:
        return 0

    def _reconcile_current_lane_health(self) -> int:
        return 0

    def _reclaim_expired_runtime_leases(self) -> dict[str, int]:
        return {}

    def _terminalize_unservable_canonical_repairs(self) -> int:
        return 0

    def _aux_lane_widths(self, **kwargs) -> dict[str, int]:  # noqa: ANN003
        return {} if self._stop_event.is_set() else {"twelve_data": 1}

    def _auxiliary_vendor_has_eligible_work(self, vendor: str) -> bool:
        return vendor == "twelve_data" and not self._stop_event.is_set()

    def _canonical_vendor_has_eligible_work(self, vendor: str) -> bool:
        return False

    def _planner_lane_wait_context(self, **kwargs) -> dict[str, object]:  # noqa: ANN003
        return {"actionable": self._actionable, "reason": self._reason}

    def _drain_auxiliary_lane(self, vendor: str) -> list[str]:
        time.sleep(0.65)
        self._stop_event.set()
        return []

    def _drain_canonical_lane(self, vendor: str, exchange: str) -> list[str]:
        return []


def test_scheduler_logs_paced_vendor_wait_without_false_stall(caplog) -> None:  # noqa: ANN001
    service = _SchedulerService(actionable=False, reason="active_vendor_attempt")
    caplog.set_level(logging.INFO, logger="trademl.data_node.scheduler")

    PlannerLaneScheduler(service).process_planner_queue(lane_stall_seconds=0.01)

    messages = [record.getMessage() for record in caplog.records]
    assert any("planner_queue_waiting" in message and "active_vendor_attempt" in message for message in messages)
    assert not any("planner_queue_stalled" in message for message in messages)
    assert service.db.decisions[0]["decision"] == "paced_wait"


def test_scheduler_records_actionable_stall(caplog) -> None:  # noqa: ANN001
    service = _SchedulerService(actionable=True, reason="pending_lane_exceeded_wait_window")
    caplog.set_level(logging.WARNING, logger="trademl.data_node.scheduler")

    PlannerLaneScheduler(service).process_planner_queue(lane_stall_seconds=0.01)

    messages = [record.getMessage() for record in caplog.records]
    assert any("planner_queue_stalled" in message and "pending_lane_exceeded_wait_window" in message for message in messages)
    assert service.db.decisions[0]["decision"] == "stalled"
