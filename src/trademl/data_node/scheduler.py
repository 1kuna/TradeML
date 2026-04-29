"""Planner-native lane scheduler for the Pi data node."""

from __future__ import annotations

import logging
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import UTC, datetime
from time import monotonic
from typing import Any, Callable

LOGGER = logging.getLogger(__name__)


class PlannerLaneScheduler:
    """Coordinate canonical and auxiliary planner lanes for a data-node service."""

    def __init__(self, service: Any) -> None:
        self._service = service

    def process_planner_queue(
        self,
        *,
        trading_date: str | None = None,
        exchange: str = "XNYS",
        heartbeat_fn: Callable[[], object] | None = None,
        heartbeat_interval_seconds: float = 30.0,
        lane_stall_seconds: float = 45.0,
    ) -> list[str]:
        """Process planner-native canonical and auxiliary work."""
        service = self._service
        if not service.db.has_pending_planner_tasks():
            service._ensure_planner_backlog_seeded(  # noqa: SLF001
                trading_date=trading_date or datetime.now(tz=UTC).date().isoformat()
            )
        service.db.clear_expired_vendor_lane_cooldowns()
        service.db.clear_stale_idle_vendor_lanes()
        service._reconcile_budget_blocked_auxiliary_lanes()  # noqa: SLF001
        service._reclaim_expired_runtime_leases()  # noqa: SLF001
        service._terminalize_unservable_canonical_repairs()  # noqa: SLF001
        changed_dates: list[str] = []
        futures: dict[object, str] = {}
        canonical_lane_widths = service._canonical_runtime._backfill_lane_widths()  # noqa: SLF001
        canonical_backlog_active = service.db.has_pending_planner_tasks(
            task_families=("canonical_bars", "canonical_repair")
        )
        aux_lane_widths = service._aux_lane_widths(  # noqa: SLF001
            task_kinds={"REFERENCE", "EVENT", "MACRO", "RESEARCH_ONLY"},
            canonical_pressure=canonical_backlog_active,
        )
        configured_lanes = len(canonical_lane_widths) + len(aux_lane_widths)
        max_workers = max(
            1,
            sum(canonical_lane_widths.values())
            + sum(aux_lane_widths.values())
            + configured_lanes,
        )
        executor = ThreadPoolExecutor(max_workers=max_workers)
        try:
            aux_active: dict[str, int] = defaultdict(int)
            canonical_active: dict[str, int] = defaultdict(int)
            aux_submitted: dict[str, int] = defaultdict(int)
            canonical_submitted: dict[str, int] = defaultdict(int)
            aux_submit_cap = {
                vendor: max(1, int(width)) * 32
                for vendor, width in aux_lane_widths.items()
            }
            canonical_submit_cap = {
                vendor: max(1, int(width)) * 16
                for vendor, width in canonical_lane_widths.items()
            }

            def submit_auxiliary_lane(vendor: str) -> None:
                futures[executor.submit(service._drain_auxiliary_lane, vendor)] = (  # noqa: SLF001
                    f"auxiliary:{vendor}"
                )
                aux_active[vendor] += 1
                aux_submitted[vendor] += 1

            def submit_canonical_lane(vendor: str) -> None:
                futures[
                    executor.submit(service._drain_canonical_lane, vendor, exchange)  # noqa: SLF001
                ] = f"canonical:{vendor}"
                canonical_active[vendor] += 1
                canonical_submitted[vendor] += 1

            def can_submit_auxiliary(vendor: str) -> bool:
                return aux_submitted[vendor] < aux_submit_cap.get(vendor, 32)

            def can_submit_canonical(vendor: str) -> bool:
                return canonical_submitted[vendor] < canonical_submit_cap.get(
                    vendor, 16
                )

            def refresh_lane_widths() -> None:
                canonical_lane_widths.update(
                    service._canonical_runtime._backfill_lane_widths()  # noqa: SLF001
                )
                canonical_active_backlog = service.db.has_pending_planner_tasks(
                    task_families=("canonical_bars", "canonical_repair")
                )
                aux_lane_widths.update(
                    service._aux_lane_widths(  # noqa: SLF001
                        task_kinds={"REFERENCE", "EVENT", "MACRO", "RESEARCH_ONLY"},
                        canonical_pressure=canonical_active_backlog,
                    )
                )
                for vendor, width in aux_lane_widths.items():
                    aux_submit_cap.setdefault(vendor, max(1, int(width)) * 32)
                for vendor, width in canonical_lane_widths.items():
                    canonical_submit_cap.setdefault(vendor, max(1, int(width)) * 16)

            def refill_ready_lanes() -> None:
                if service._stop_event.is_set():  # noqa: SLF001
                    return
                refresh_lane_widths()
                for vendor, width in canonical_lane_widths.items():
                    while (
                        canonical_active[vendor] < max(1, int(width))
                        and can_submit_canonical(vendor)
                        and service._canonical_vendor_has_eligible_work(vendor)  # noqa: SLF001
                    ):
                        submit_canonical_lane(vendor)
                for vendor, width in aux_lane_widths.items():
                    while (
                        aux_active[vendor] < max(1, int(width))
                        and can_submit_auxiliary(vendor)
                        and service._auxiliary_vendor_has_eligible_work(vendor)  # noqa: SLF001
                    ):
                        submit_auxiliary_lane(vendor)

            refill_ready_lanes()
            last_heartbeat = monotonic()
            last_pending_log = monotonic()
            queue_started = monotonic()
            while futures:
                done, _pending = wait(
                    list(futures), return_when=FIRST_COMPLETED, timeout=0.5
                )
                if not done:
                    now = monotonic()
                    if (now - queue_started) >= max(0.01, float(lane_stall_seconds)):
                        lane_counts: dict[str, int] = {}
                        for lane in futures.values():
                            lane_counts[lane] = lane_counts.get(lane, 0) + 1
                        LOGGER.warning(
                            "planner_queue_stalled worker_id=%s elapsed_seconds=%.1f pending=%s",
                            service.worker_id,
                            now - queue_started,
                            lane_counts,
                        )
                        service._reclaim_expired_runtime_leases()  # noqa: SLF001
                        service._terminalize_unservable_canonical_repairs()  # noqa: SLF001
                        refill_ready_lanes()
                        queue_started = now
                    if (
                        heartbeat_fn is not None
                        and (now - last_heartbeat) >= heartbeat_interval_seconds
                    ):
                        try:
                            heartbeat_fn()
                        except Exception:
                            LOGGER.exception(
                                "planner_queue_heartbeat_failed worker_id=%s",
                                service.worker_id,
                            )
                        last_heartbeat = now
                    if (now - last_pending_log) >= max(5.0, heartbeat_interval_seconds):
                        service._reclaim_expired_runtime_leases()  # noqa: SLF001
                        lane_counts: dict[str, int] = {}
                        for lane in futures.values():
                            lane_counts[lane] = lane_counts.get(lane, 0) + 1
                        LOGGER.warning(
                            "planner_queue_waiting worker_id=%s pending=%s",
                            service.worker_id,
                            lane_counts,
                        )
                        last_pending_log = now
                    continue
                for future in done:
                    task_type = futures.pop(future)
                    lane_kind, _, vendor = task_type.partition(":")
                    if lane_kind == "auxiliary":
                        aux_active[vendor] = max(0, aux_active[vendor] - 1)
                    elif lane_kind == "canonical":
                        canonical_active[vendor] = max(0, canonical_active[vendor] - 1)
                    try:
                        result = future.result()
                    except Exception:
                        LOGGER.exception(
                            "planner_lane_failed lane=%s worker_id=%s",
                            task_type,
                            service.worker_id,
                        )
                        if (
                            lane_kind == "auxiliary"
                            and not service._stop_event.is_set()  # noqa: SLF001
                            and can_submit_auxiliary(vendor)
                            and aux_active[vendor]
                            < max(1, int(aux_lane_widths.get(vendor, 1)))
                            and service._auxiliary_vendor_has_eligible_work(vendor)  # noqa: SLF001
                        ):
                            submit_auxiliary_lane(vendor)
                        continue
                    if lane_kind == "canonical":
                        changed_dates.extend(result)
                        if (
                            result
                            and not service._stop_event.is_set()  # noqa: SLF001
                            and can_submit_canonical(vendor)
                            and canonical_active[vendor]
                            < max(1, int(canonical_lane_widths.get(vendor, 1)))
                            and service._canonical_vendor_has_eligible_work(vendor)  # noqa: SLF001
                        ):
                            submit_canonical_lane(vendor)
                    elif (
                        not service._stop_event.is_set()  # noqa: SLF001
                        and aux_active[vendor]
                        < max(1, int(aux_lane_widths.get(vendor, 1)))
                        and can_submit_auxiliary(vendor)
                        and (
                            result
                            or service._auxiliary_vendor_has_eligible_work(vendor)  # noqa: SLF001
                        )
                    ):
                        submit_auxiliary_lane(vendor)
        finally:
            executor.shutdown(wait=True, cancel_futures=False)
        return sorted(set(changed_dates))
