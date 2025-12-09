"""
Queue worker for the unified Pi data-node.

Implements the dispatcher algorithm (spec §3):
1. Lease next PENDING task by (priority, created_at)
2. Select eligible vendor based on budgets and entitlements
3. Fetch data using the appropriate connector
4. Update partition_status and mark task done/failed

See updated_node_spec.md §3 for algorithm details.
"""

from __future__ import annotations

import os
import threading
import time
from datetime import date, datetime, timedelta, timezone
from typing import Callable, Optional

from loguru import logger

from .budgets import BudgetManager, get_budget_manager
from .db import NodeDB, Task, TaskKind, TaskStatus, PartitionStatus, get_db, PRIORITY_MAP
from .fetchers import (
    FetchResult,
    FetchStatus,
    fetch_task,
    get_vendors_for_dataset,
    VENDOR_PRIORITY,
)
from .qc import validate_partition_row_count
from .stages import get_expected_rows, get_qc_thresholds
from .vendor_limits import massive_in_window, massive_window


def _is_trading_day(dt_value) -> bool:
    """Best-effort trading day check using XNYS calendar; fallback to weekday."""
    try:
        from datetime import date as dt_date
        from data_layer.reference.calendars import get_calendar

        dt_obj = dt_date.fromisoformat(dt_value) if isinstance(dt_value, str) else dt_value
        cal = get_calendar("XNYS")
        return bool(cal.is_trading_day(dt_obj))
    except Exception:
        # Fallback: weekday (Mon-Fri) treated as trading
        try:
            return dt_value.weekday() < 5
        except Exception:
            return True


def _base_priority_for_kind(kind: TaskKind) -> int:
    """Get the lowest configured priority for a task kind."""
    candidates = [v for (k, _), v in PRIORITY_MAP.items() if k == kind]
    return min(candidates) if candidates else 0


def _capped_followup_priority(task: Task) -> int:
    """Cap follow-up priority to avoid indefinite inflation."""
    base = _base_priority_for_kind(task.kind)
    cap = base + 5
    return min(task.priority + 1, cap)


# Default concurrency limits per vendor
# Note: FMP removed - free tier too limited
DEFAULT_MAX_INFLIGHT = {
    "alpaca": 8,   # Increased for higher throughput (200 rpm limit)
    "finnhub": 2,
    "fred": 1,
    "av": 1,
    "massive": 2,
}

# Default lease TTL in seconds
DEFAULT_LEASE_TTL = 300  # 5 minutes

# Exponential backoff parameters
BACKOFF_BASE_SECONDS = 60
BACKOFF_MAX_SECONDS = 3600  # 1 hour

def _massive_in_window(task: Task) -> bool:
    """Check if task is within Massive free-tier window (2y history, T+1 delay)."""
    try:
        start_date = date.fromisoformat(task.start_date) if isinstance(task.start_date, str) else task.start_date
        end_date = date.fromisoformat(task.end_date) if isinstance(task.end_date, str) else task.end_date
    except Exception as exc:
        logger.info(f"Massive window check failed for task {task.id}: bad dates {task.start_date}->{task.end_date} ({exc})")
        return False

    earliest, latest = massive_window()

    if end_date < earliest or start_date < earliest:
        logger.info(f"Massive window miss (too old) task {task.id}: {start_date}->{end_date}, earliest {earliest}")
        return False
    if start_date > latest:
        logger.info(f"Massive window miss (too recent) task {task.id}: {start_date}->{end_date}, latest {latest}")
        return False
    return True


def _vendor_supports_task(task: Task, vendor: str) -> bool:
    """
    Check if a vendor can service the task based on temporal limits.

    Currently enforces Massive free-tier window (2y history, T+1 delay) for equities.
    """
    if vendor == "massive" and task.dataset in ("equities_eod", "equities_minute"):
        return massive_in_window(task.start_date, task.end_date)
    return True


class QueueWorker:
    """
    Worker that processes tasks from the backfill_queue.

    Implements the dispatcher algorithm:
    - Finds oldest PENDING task by (priority, created_at)
    - Selects best vendor with available budget
    - Fetches data and updates partition_status
    - Handles success, empty, rate-limit, and error outcomes
    """

    def __init__(
        self,
        db: Optional[NodeDB] = None,
        budgets: Optional[BudgetManager] = None,
        node_id: Optional[str] = None,
        max_inflight: Optional[dict[str, int]] = None,
    ):
        """
        Initialize the queue worker.

        Args:
            db: Database manager (default: get_db())
            budgets: Budget manager (default: get_budget_manager())
            node_id: This node's ID for leasing (default: EDGE_NODE_ID env or hostname)
            max_inflight: Per-vendor concurrency limits (default: from env or DEFAULT_MAX_INFLIGHT)
        """
        self.db = db or get_db()
        self.budgets = budgets or get_budget_manager()
        self.node_id = node_id or os.environ.get("EDGE_NODE_ID", os.uname().nodename)

        # Load max inflight from env or use defaults
        self.max_inflight = max_inflight or {}
        for vendor, default in DEFAULT_MAX_INFLIGHT.items():
            env_key = f"NODE_MAX_INFLIGHT_{vendor.upper()}"
            self.max_inflight[vendor] = int(os.environ.get(env_key, default))

        # Track currently running tasks per vendor
        self._inflight: dict[str, int] = {v: 0 for v in self.max_inflight}
        self._inflight_lock = threading.Lock()

        # Vendor ineligibility cache (for hard 4xx errors)
        self._vendor_ineligible: dict[tuple[str, str], datetime] = {}  # (dataset, vendor) -> expire time
        self._ineligible_ttl = timedelta(hours=24)

    def pick_vendor(self, task: Task) -> Optional[str]:
        """
        Pick the best vendor for a task based on budgets and eligibility.

        Args:
            task: Task to process

        Returns:
            Vendor name, or None if no vendor available
        """
        vendors = get_vendors_for_dataset(task.dataset)

        if not vendors:
            logger.warning(f"No vendors configured for dataset: {task.dataset}")
            return None

        logger.debug(f"Task {task.id} vendors: {vendors} for dataset {task.dataset}")
        eligible = []
        now = datetime.now(timezone.utc)

        for vendor in vendors:
            # Skip vendors that cannot service this task (e.g., Massive history window)
            if not _vendor_supports_task(task, vendor):
                logger.info(f"Vendor {vendor} skipped for task {task.id} window {task.start_date}->{task.end_date}")
                continue

            # Check ineligibility cache
            cache_key = (task.dataset, vendor)
            if cache_key in self._vendor_ineligible:
                if now < self._vendor_ineligible[cache_key]:
                    logger.debug(f"Vendor {vendor} temporarily ineligible for {task.dataset}")
                    continue
                else:
                    # Expired, remove from cache
                    del self._vendor_ineligible[cache_key]

            # Check budget
            if not self.budgets.can_spend(vendor, task.kind):
                logger.debug(f"Vendor {vendor} budget exhausted for {task.kind.value}")
                continue

            # Check inflight limit
            with self._inflight_lock:
                if self._inflight.get(vendor, 0) >= self.max_inflight.get(vendor, 1):
                    logger.debug(f"Vendor {vendor} at max inflight ({self.max_inflight.get(vendor, 1)})")
                    continue

            eligible.append(vendor)

        if not eligible:
            return None

        # Return highest priority (lowest number) vendor
        return min(eligible, key=lambda v: VENDOR_PRIORITY.get(v, 99))

    def process_task(self, task: Task, vendor: str) -> FetchResult:
        """
        Process a single task using the specified vendor.

        Args:
            task: Task to process
            vendor: Vendor to use

        Returns:
            FetchResult with outcome details
        """
        # Track inflight
        with self._inflight_lock:
            self._inflight[vendor] = self._inflight.get(vendor, 0) + 1

        try:
            # Budget is tracked per HTTP request in base.py._get()
            # (not per task, since multi-day tasks make many API calls)

            # Execute fetch
            result = fetch_task(task, vendor)
            result.vendor_used = vendor

            return result

        finally:
            # Release inflight slot
            with self._inflight_lock:
                self._inflight[vendor] = max(0, self._inflight.get(vendor, 0) - 1)

    def handle_result(self, task: Task, result: FetchResult) -> None:
        """
        Handle the result of a fetch operation.

        Updates partition_status and marks task done/failed.
        For multi-day tasks, creates per-day partition entries with proper validation.

        Args:
            task: The task that was processed
            result: Result from the fetch operation
        """
        now = datetime.now(timezone.utc)

        if result.status == FetchStatus.SUCCESS:
            # Get expected rows per day from config
            expected_per_day = get_expected_rows(task.dataset, num_days=1)
            thresholds = get_qc_thresholds()
            updates: list[dict] = []

            # Process per-day entries if available
            if result.rows_by_date:
                for dt, row_count in result.rows_by_date.items():
                    # Non-trading day with zero rows → NO_SESSION GREEN
                    if row_count == 0 and not _is_trading_day(dt):
                        updates.append(
                            dict(
                                source_name=result.vendor_used or "unknown",
                                table_name=task.dataset,
                                symbol=task.symbol,
                                dt=dt.isoformat() if hasattr(dt, "isoformat") else str(dt),
                                status=PartitionStatus.GREEN,
                                qc_score=1.0,
                                row_count=0,
                                expected_rows=0,
                                qc_code="NO_SESSION",
                                fetch_params=result.fetch_params,
                            )
                        )
                        continue

                    # Calculate status for this day
                    status, qc_code = validate_partition_row_count(
                        row_count=row_count,
                        expected_rows=expected_per_day,
                        qc_code=result.qc_code,
                        thresholds=thresholds,
                    )

                    updates.append(
                        dict(
                            source_name=result.vendor_used or "unknown",
                            table_name=task.dataset,
                            symbol=task.symbol,
                            dt=dt.isoformat() if hasattr(dt, "isoformat") else str(dt),
                            status=status,
                            qc_score=1.0 if status == PartitionStatus.GREEN else 0.5,
                            row_count=row_count,
                            expected_rows=expected_per_day,
                            qc_code=qc_code,
                            fetch_params=result.fetch_params,
                        )
                    )
            else:
                # Fallback: single entry for single-day tasks or legacy fetchers
                if result.rows == 0 and not _is_trading_day(task.start_date):
                    updates.append(
                        dict(
                            source_name=result.vendor_used or "unknown",
                            table_name=task.dataset,
                            symbol=task.symbol,
                            dt=task.start_date,
                            status=PartitionStatus.GREEN,
                            qc_score=1.0,
                            row_count=0,
                            expected_rows=0,
                            qc_code="NO_SESSION",
                            fetch_params=result.fetch_params,
                        )
                    )
                    with self.db.transaction() as conn:
                        for params in updates:
                            self.db.upsert_partition_status(conn=conn, **params)
                        self.db.mark_task_done(task.id, conn=conn)
                    logger.debug(f"Task {task.id} empty/non-trading single day")
                    return

                status, qc_code = validate_partition_row_count(
                    row_count=result.rows,
                    expected_rows=expected_per_day,
                    qc_code=result.qc_code,
                    thresholds=thresholds,
                )

                updates.append(
                    dict(
                        source_name=result.vendor_used or "unknown",
                        table_name=task.dataset,
                        symbol=task.symbol,
                        dt=task.start_date,
                        status=status,
                        qc_score=1.0 if status == PartitionStatus.GREEN else 0.5,
                        row_count=result.rows,
                        expected_rows=expected_per_day,
                        qc_code=qc_code,
                        fetch_params=result.fetch_params,
                    )
                )

            with self.db.transaction() as conn:
                for params in updates:
                    self.db.upsert_partition_status(conn=conn, **params)
                self.db.mark_task_done(task.id, conn=conn)
            logger.info(f"Task {task.id} completed: {result.rows} rows from {result.vendor_used}")

        elif result.status == FetchStatus.EMPTY:
            updates: list[dict] = []
            if result.qc_code == "STORAGE_NOT_IMPLEMENTED":
                # Data was fetched but not persisted; surface as AMBER gap
                updates.append(
                    dict(
                        source_name=result.vendor_used or "unknown",
                        table_name=task.dataset,
                        symbol=task.symbol,
                        dt=task.start_date,
                        status=PartitionStatus.AMBER,
                        qc_score=0.5,
                        row_count=0,
                        expected_rows=0,
                        qc_code=result.qc_code,
                        fetch_params=result.fetch_params,
                    )
                )
                logger.warning(f"Task {task.id} empty due to missing storage {task.start_date}: {result.qc_code}")
            else:
                # Check if this is a trading day - empty on trading day is suspicious
                from data_layer.reference.calendars import get_calendar
                try:
                    from datetime import date as dt_date
                    task_date = dt_date.fromisoformat(task.start_date) if isinstance(task.start_date, str) else task.start_date
                    is_trading = get_calendar("XNYS").is_trading_day(task_date)
                except Exception:
                    is_trading = True  # Assume trading day if calendar fails

                if is_trading and result.qc_code != "NO_SESSION":
                    # Empty on a trading day - mark AMBER for re-check, not GREEN
                    expected_rows = get_expected_rows(task.dataset, num_days=1)
                    updates.append(
                        dict(
                            source_name=result.vendor_used or "unknown",
                            table_name=task.dataset,
                            symbol=task.symbol,
                            dt=task.start_date,
                            status=PartitionStatus.AMBER,
                            qc_score=0.5,
                            row_count=0,
                            expected_rows=expected_rows,
                            qc_code=result.qc_code or "EMPTY_ON_TRADING_DAY",
                            fetch_params=result.fetch_params,
                        )
                    )
                    logger.warning(f"Task {task.id} empty on trading day {task.start_date}: {result.qc_code}")
                elif is_trading:
                    # Trading day with explicit no-session: treat as AMBER to avoid false GREENs
                    expected_rows = get_expected_rows(task.dataset, num_days=1)
                    updates.append(
                        dict(
                            source_name=result.vendor_used or "unknown",
                            table_name=task.dataset,
                            symbol=task.symbol,
                            dt=task.start_date,
                            status=PartitionStatus.AMBER,
                            qc_score=0.5,
                            row_count=0,
                            expected_rows=expected_rows,
                            qc_code=result.qc_code or "NO_SESSION",
                            fetch_params=result.fetch_params,
                        )
                    )
                    logger.warning(f"Task {task.id} empty on trading day {task.start_date} (NO_SESSION)")
                else:
                    # Non-trading day (weekend/holiday) - empty is expected, mark GREEN
                    updates.append(
                        dict(
                            source_name=result.vendor_used or "unknown",
                            table_name=task.dataset,
                            symbol=task.symbol,
                            dt=task.start_date,
                            status=PartitionStatus.GREEN,
                            qc_score=1.0,
                            row_count=0,
                            expected_rows=0,
                            qc_code=result.qc_code or "NO_SESSION",
                            fetch_params=result.fetch_params,
                        )
                    )
                    logger.debug(f"Task {task.id} empty/no-session: {result.qc_code}")

            with self.db.transaction() as conn:
                for params in updates:
                    self.db.upsert_partition_status(conn=conn, **params)
                self.db.mark_task_done(task.id, conn=conn)

        elif result.status == FetchStatus.PARTIAL:
            # Partial success - some days fetched, but failed partway through
            expected_per_day = get_expected_rows(task.dataset, num_days=1)
            thresholds = get_qc_thresholds()
            updates: list[dict] = []

            if result.rows_by_date:
                for dt, row_count in result.rows_by_date.items():
                    if row_count == 0 and not _is_trading_day(dt):
                        updates.append(
                            dict(
                                source_name=result.vendor_used or "unknown",
                                table_name=task.dataset,
                                symbol=task.symbol,
                                dt=dt.isoformat() if hasattr(dt, "isoformat") else str(dt),
                                status=PartitionStatus.GREEN,
                                qc_score=1.0,
                                row_count=0,
                                expected_rows=0,
                                qc_code="NO_SESSION",
                                fetch_params=result.fetch_params,
                            )
                        )
                        continue

                    status, qc_code = validate_partition_row_count(
                        row_count=row_count,
                        expected_rows=expected_per_day,
                        qc_code=result.qc_code,
                        thresholds=thresholds,
                    )

                    updates.append(
                        dict(
                            source_name=result.vendor_used or "unknown",
                            table_name=task.dataset,
                            symbol=task.symbol,
                            dt=dt.isoformat() if hasattr(dt, "isoformat") else str(dt),
                            status=status,
                            qc_score=1.0 if status == PartitionStatus.GREEN else 0.5,
                            row_count=row_count,
                            expected_rows=expected_per_day,
                            qc_code=qc_code,
                            fetch_params=result.fetch_params,
                        )
                    )

            # Create follow-up task for the remaining range
            if result.failed_at_date and result.original_end_date:
                new_task_id = self.db.enqueue_task(
                    dataset=task.dataset,
                    symbol=task.symbol,
                    start_date=result.failed_at_date,
                    end_date=result.original_end_date,
                    kind=task.kind,  # Preserve original kind
                    priority=_capped_followup_priority(task),
                    allow_overlap=True,  # Follow-up overlaps original range
                )
                if new_task_id:
                    logger.info(f"Created follow-up task {new_task_id} for {task.symbol} {result.failed_at_date} to {result.original_end_date}")

            with self.db.transaction() as conn:
                for params in updates:
                    self.db.upsert_partition_status(conn=conn, **params)
                self.db.mark_task_done(task.id, conn=conn)
            logger.info(f"Task {task.id} partial: {result.rows} rows from {result.vendor_used}, failed at {result.failed_at_date}")

        elif result.status == FetchStatus.RATE_LIMITED:
            # Rate limited - retry with backoff
            backoff = timedelta(seconds=BACKOFF_BASE_SECONDS * (2 ** task.attempts))
            backoff = min(backoff, timedelta(seconds=BACKOFF_MAX_SECONDS))
            backoff_until = now + backoff

            self.db.mark_task_failed(
                task.id,
                error=result.error or "Rate limited",
                backoff_until=backoff_until,
            )
            logger.warning(f"Task {task.id} rate limited, retry after {backoff_until}")

        elif result.status == FetchStatus.NOT_ENTITLED:
            # Not entitled - mark vendor ineligible for this dataset
            if result.vendor_used:
                cache_key = (task.dataset, result.vendor_used)
                self._vendor_ineligible[cache_key] = now + self._ineligible_ttl
                logger.warning(f"Vendor {result.vendor_used} marked ineligible for {task.dataset}: {result.error}")

            # Retry with another vendor (don't count against max attempts)
            self.db.mark_task_failed(
                task.id,
                error=result.error or "Not entitled",
                backoff_until=now + timedelta(seconds=10),  # Quick retry with different vendor
                max_attempts=task.attempts + 5,  # Give extra attempts for vendor switching
            )

        elif result.status == FetchStatus.NOT_SUPPORTED:
            # Not supported - mark task failed permanently
            self.db.mark_task_failed(
                task.id,
                error=result.error or "Not supported",
                max_attempts=0,  # Immediate failure
            )
            logger.error(f"Task {task.id} not supported: {result.error}")

        else:  # ERROR
            # Transient error - retry with exponential backoff
            backoff = timedelta(seconds=BACKOFF_BASE_SECONDS * (2 ** task.attempts))
            backoff = min(backoff, timedelta(seconds=BACKOFF_MAX_SECONDS))
            backoff_until = now + backoff

            self.db.mark_task_failed(
                task.id,
                error=result.error or "Unknown error",
                backoff_until=backoff_until,
            )
            logger.warning(f"Task {task.id} error, retry after {backoff_until}: {result.error}")

    def process_one(self) -> bool:
        """
        Process a single task from the queue.

        Returns:
            True if a task was processed, False if queue is empty or no vendors available
        """
        # Lease next task
        task = self.db.lease_next_task(
            node_id=self.node_id,
            lease_ttl_seconds=DEFAULT_LEASE_TTL,
        )

        if task is None:
            return False

        # Pick vendor
        vendor = self.pick_vendor(task)

        if vendor is None:
            # No vendor available right now - set backoff and release
            logger.debug(f"No vendor available for task {task.id}, backing off")
            self.db.mark_task_failed(
                task.id,
                error="No vendor available (budget/inflight limits)",
                backoff_until=datetime.now(timezone.utc) + timedelta(minutes=5),
                max_attempts=task.attempts + 10,  # Don't count towards failure
            )
            return False

        # Process
        logger.debug(f"Processing task {task.id} with {vendor}")
        result = self.process_task(task, vendor)

        # Handle result
        self.handle_result(task, result)

        return True


class QueueWorkerLoop:
    """
    Continuous loop that processes tasks from the queue.

    Can be run in a thread or as the main loop.
    """

    def __init__(
        self,
        worker: Optional[QueueWorker] = None,
        poll_interval: float = 1.0,
        idle_interval: float = 5.0,
        on_task_complete: Optional[Callable[[Task, FetchResult], None]] = None,
    ):
        """
        Initialize the worker loop.

        Args:
            worker: QueueWorker instance (default: create new)
            poll_interval: Seconds between polls when busy
            idle_interval: Seconds between polls when idle
            on_task_complete: Optional callback after each task
        """
        self.worker = worker or QueueWorker()
        self.poll_interval = poll_interval
        self.idle_interval = idle_interval
        self.on_task_complete = on_task_complete

        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self, threaded: bool = True) -> None:
        """
        Start the worker loop.

        Args:
            threaded: If True, run in a background thread
        """
        self._running = True

        if threaded:
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
            logger.info("QueueWorkerLoop started in background thread")
        else:
            self._run_loop()

    def stop(self) -> None:
        """Stop the worker loop."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
            self._thread = None
        logger.info("QueueWorkerLoop stopped")

    def _run_loop(self) -> None:
        """Main worker loop."""
        logger.info("QueueWorkerLoop running")

        while self._running:
            try:
                processed = self.worker.process_one()

                if processed:
                    time.sleep(self.poll_interval)
                else:
                    # No task available - wait longer
                    time.sleep(self.idle_interval)

            except KeyboardInterrupt:
                logger.info("QueueWorkerLoop interrupted")
                break
            except Exception as e:
                logger.exception(f"Error in worker loop: {e}")
                time.sleep(self.idle_interval)

        logger.info("QueueWorkerLoop exiting")

    @property
    def is_running(self) -> bool:
        """Check if the loop is running."""
        return self._running


def run_worker(
    poll_interval: float = 1.0,
    idle_interval: float = 5.0,
) -> None:
    """
    Run the queue worker loop in the foreground.

    Args:
        poll_interval: Seconds between polls when busy
        idle_interval: Seconds between polls when idle
    """
    loop = QueueWorkerLoop(
        poll_interval=poll_interval,
        idle_interval=idle_interval,
    )

    try:
        loop.start(threaded=False)
    except KeyboardInterrupt:
        pass
    finally:
        loop.stop()


# =============================================================================
# Per-Vendor Worker Pool (Multi-threaded)
# =============================================================================

# Default thread counts per vendor (sized for latency resilience)
# Note: FMP removed - free tier too limited
DEFAULT_THREAD_COUNTS = {
    "alpaca": 3,    # 3.25 rps × ~0.5s latency = 2-3 threads saturates API
    "finnhub": 2,   # 60 rpm
    "fred": 2,      # 120 rpm
    "av": 1,        # 5 rpm, 25/day - so slow 1 is plenty
    "massive": 1,   # 5 rpm
}


class VendorWorker:
    """
    Single worker thread dedicated to one vendor.

    Continuously leases and processes tasks for datasets this vendor can handle.
    """

    def __init__(
        self,
        vendor: str,
        worker_id: int,
        db: Optional[NodeDB] = None,
        budgets: Optional[BudgetManager] = None,
        node_id: Optional[str] = None,
        poll_interval: float = 0.5,
        idle_interval: float = 5.0,
    ):
        """
        Initialize a vendor worker.

        Args:
            vendor: Vendor name (alpaca, fred, etc.)
            worker_id: Unique ID for this worker (for logging)
            db: Database manager
            budgets: Budget manager
            node_id: Node identifier for leasing
            poll_interval: Seconds between polls when busy
            idle_interval: Seconds between polls when idle
        """
        self.vendor = vendor
        self.worker_id = worker_id
        self.db = db or get_db()
        self.budgets = budgets or get_budget_manager()
        self.node_id = node_id or os.environ.get("EDGE_NODE_ID", os.uname().nodename)
        self.poll_interval = poll_interval
        self.idle_interval = idle_interval

        # Get datasets this vendor can handle (excluding those they're not entitled to)
        from .fetchers import get_datasets_for_vendor, get_excluded_datasets
        all_datasets = get_datasets_for_vendor(vendor)
        excluded = get_excluded_datasets(vendor)
        self.datasets = [d for d in all_datasets if d not in excluded]
        self.excluded_datasets = excluded

        if excluded:
            logger.info(f"VendorWorker {vendor}: excluding datasets {excluded}")

        self._running = False
        self._tasks_processed = 0
        self._tasks_failed = 0

    def run(self) -> None:
        """Main worker loop."""
        self._running = True
        worker_name = f"{self.vendor}-{self.worker_id}"
        logger.info(f"VendorWorker {worker_name} starting (datasets: {self.datasets})")

        while self._running:
            try:
                processed = self._process_one()

                if processed:
                    time.sleep(self.poll_interval)
                else:
                    # No task available - wait longer
                    time.sleep(self.idle_interval)

            except Exception as e:
                logger.exception(f"VendorWorker {worker_name} error: {e}")
                time.sleep(self.idle_interval)

        logger.info(f"VendorWorker {worker_name} stopped (processed {self._tasks_processed} tasks)")

    def _process_one(self) -> bool:
        """
        Process a single task.

        Returns:
            True if a task was processed, False if none available
        """
        import time as _time
        t_start = _time.time()

        # Lease next task for this vendor
        t_lease_start = _time.time()
        task = self.db.lease_next_task_for_vendor(
            vendor=self.vendor,
            datasets=self.datasets,
            node_id=self.node_id,
            lease_ttl_seconds=DEFAULT_LEASE_TTL,
        )
        t_lease_end = _time.time()
        lease_ms = (t_lease_end - t_lease_start) * 1000

        if task is None:
            if lease_ms > 50:  # Only log if lease took unusually long
                logger.info(f"[TIMING] {self.vendor}-{self.worker_id}: lease (no task) took {lease_ms:.0f}ms")
            return False

        # Budget check using the actual task kind
        if not self.budgets.can_spend(self.vendor, task.kind):
            logger.debug(
                f"VendorWorker {self.vendor}-{self.worker_id}: budget exhausted for {task.kind.value}, releasing task {task.id}"
            )
            self.db.release_task(task.id)
            return False

        # Budget is tracked per HTTP request in base.py._get()
        # (not per task, since multi-day tasks make many API calls)

        # Execute fetch
        from .fetchers import fetch_task, FetchResult, FetchStatus

        t_fetch_start = _time.time()
        try:
            result = fetch_task(task, self.vendor)
            result.vendor_used = self.vendor
        except Exception as e:
            logger.exception(f"VendorWorker {self.vendor}-{self.worker_id}: fetch error: {e}")
            result = FetchResult(
                status=FetchStatus.ERROR,
                error=str(e),
                vendor_used=self.vendor,
            )
        t_fetch_end = _time.time()
        fetch_ms = (t_fetch_end - t_fetch_start) * 1000

        # Handle result
        t_handle_start = _time.time()
        self._handle_result(task, result)
        t_handle_end = _time.time()
        handle_ms = (t_handle_end - t_handle_start) * 1000

        self._tasks_processed += 1

        total_ms = (_time.time() - t_start) * 1000
        logger.info(f"[TIMING] {self.vendor}-{self.worker_id}: task={task.id} total={total_ms:.0f}ms (lease={lease_ms:.0f}ms, fetch={fetch_ms:.0f}ms, handle={handle_ms:.0f}ms)")

        return True

    def _handle_result(self, task: Task, result) -> None:
        """Handle the result of a fetch operation.

        For multi-day tasks, creates per-day partition entries with proper validation.
        """
        from .fetchers import FetchStatus

        now = datetime.now(timezone.utc)

        if result.status == FetchStatus.SUCCESS:
            # Get expected rows per day from config
            expected_per_day = get_expected_rows(task.dataset, num_days=1)
            thresholds = get_qc_thresholds()
            updates: list[dict] = []

            # Process per-day entries if available
            if result.rows_by_date:
                # Get trading calendar to check for weekends/holidays
                from data_layer.reference.calendars import get_calendar
                try:
                    cal = get_calendar("XNYS")
                except Exception:
                    cal = None

                for dt, row_count in result.rows_by_date.items():
                    # Convert to date if needed for calendar check
                    from datetime import date as dt_date
                    if isinstance(dt, str):
                        check_date = dt_date.fromisoformat(dt)
                    elif hasattr(dt, 'date'):
                        check_date = dt.date()
                    else:
                        check_date = dt

                    # Check if this is a trading day
                    is_trading = True  # Default to trading (safer)
                    if cal is not None:
                        try:
                            is_trading = cal.is_trading_day(check_date)
                        except Exception:
                            pass

                    # For non-trading days with 0 rows, skip validation - mark GREEN
                    if not is_trading and row_count == 0:
                        updates.append(
                            dict(
                                source_name=result.vendor_used or self.vendor,
                                table_name=task.dataset,
                                symbol=task.symbol,
                                dt=dt.isoformat() if hasattr(dt, "isoformat") else str(dt),
                                status=PartitionStatus.GREEN,
                                qc_score=1.0,
                                row_count=0,
                                expected_rows=0,
                                qc_code="NO_SESSION",
                                fetch_params=result.fetch_params,
                            )
                        )
                        continue

                    # Calculate status for this trading day
                    status, qc_code = validate_partition_row_count(
                        row_count=row_count,
                        expected_rows=expected_per_day,
                        qc_code=result.qc_code,
                        thresholds=thresholds,
                    )

                    updates.append(
                        dict(
                            source_name=result.vendor_used or self.vendor,
                            table_name=task.dataset,
                            symbol=task.symbol,
                            dt=dt.isoformat() if hasattr(dt, "isoformat") else str(dt),
                            status=status,
                            qc_score=1.0 if status == PartitionStatus.GREEN else 0.5,
                            row_count=row_count,
                            expected_rows=expected_per_day,
                            qc_code=qc_code,
                            fetch_params=result.fetch_params,
                        )
                    )
            else:
                # Fallback: single entry for single-day tasks or legacy fetchers
                if result.rows == 0 and not _is_trading_day(task.start_date):
                    updates.append(
                        dict(
                            source_name=result.vendor_used or self.vendor,
                            table_name=task.dataset,
                            symbol=task.symbol,
                            dt=task.start_date,
                            status=PartitionStatus.GREEN,
                            qc_score=1.0,
                            row_count=0,
                            expected_rows=0,
                            qc_code="NO_SESSION",
                            fetch_params=result.fetch_params,
                        )
                    )
                    with self.db.transaction() as conn:
                        for params in updates:
                            self.db.upsert_partition_status(conn=conn, **params)
                        self.db.mark_task_done(task.id, conn=conn)
                    logger.debug(f"Task {task.id} empty/non-trading single day")
                    return

                status, qc_code = validate_partition_row_count(
                    row_count=result.rows,
                    expected_rows=expected_per_day,
                    qc_code=result.qc_code,
                    thresholds=thresholds,
                )

                updates.append(
                    dict(
                        source_name=result.vendor_used or self.vendor,
                        table_name=task.dataset,
                        symbol=task.symbol,
                        dt=task.start_date,
                        status=status,
                        qc_score=1.0 if status == PartitionStatus.GREEN else 0.5,
                        row_count=result.rows,
                        expected_rows=expected_per_day,
                        qc_code=qc_code,
                        fetch_params=result.fetch_params,
                    )
                )

            with self.db.transaction() as conn:
                for params in updates:
                    self.db.upsert_partition_status(conn=conn, **params)
                self.db.mark_task_done(task.id, conn=conn)
            logger.info(f"Task {task.id} completed: {result.rows} rows from {self.vendor}")

        elif result.status == FetchStatus.EMPTY:
            updates: list[dict] = []
            if result.qc_code == "STORAGE_NOT_IMPLEMENTED":
                updates.append(
                    dict(
                        source_name=result.vendor_used or self.vendor,
                        table_name=task.dataset,
                        symbol=task.symbol,
                        dt=task.start_date,
                        status=PartitionStatus.AMBER,
                        qc_score=0.5,
                        row_count=0,
                        expected_rows=0,
                        qc_code=result.qc_code,
                        fetch_params=result.fetch_params,
                    )
                )
                logger.warning(f"Task {task.id} empty due to missing storage {task.start_date}: {result.qc_code}")
            else:
                is_trading = _is_trading_day(task.start_date)
                if is_trading and result.qc_code != "NO_SESSION":
                    expected_rows = get_expected_rows(task.dataset, num_days=1)
                    updates.append(
                        dict(
                            source_name=result.vendor_used or self.vendor,
                            table_name=task.dataset,
                            symbol=task.symbol,
                            dt=task.start_date,
                            status=PartitionStatus.AMBER,
                            qc_score=0.5,
                            row_count=0,
                            expected_rows=expected_rows,
                            qc_code=result.qc_code or "EMPTY_ON_TRADING_DAY",
                            fetch_params=result.fetch_params,
                        )
                    )
                    logger.warning(f"Task {task.id} empty on trading day {task.start_date}: {result.qc_code}")
                elif is_trading:
                    expected_rows = get_expected_rows(task.dataset, num_days=1)
                    updates.append(
                        dict(
                            source_name=result.vendor_used or self.vendor,
                            table_name=task.dataset,
                            symbol=task.symbol,
                            dt=task.start_date,
                            status=PartitionStatus.AMBER,
                            qc_score=0.5,
                            row_count=0,
                            expected_rows=expected_rows,
                            qc_code=result.qc_code or "NO_SESSION",
                            fetch_params=result.fetch_params,
                        )
                    )
                    logger.warning(f"Task {task.id} empty on trading day {task.start_date} (NO_SESSION)")
                else:
                    updates.append(
                        dict(
                            source_name=result.vendor_used or self.vendor,
                            table_name=task.dataset,
                            symbol=task.symbol,
                            dt=task.start_date,
                            status=PartitionStatus.GREEN,
                            qc_score=1.0,
                            row_count=0,
                            expected_rows=0,
                            qc_code=result.qc_code or "NO_SESSION",
                            fetch_params=result.fetch_params,
                        )
                    )
                    logger.debug(f"Task {task.id} empty/no-session: {result.qc_code}")

            with self.db.transaction() as conn:
                for params in updates:
                    self.db.upsert_partition_status(conn=conn, **params)
                self.db.mark_task_done(task.id, conn=conn)

        elif result.status == FetchStatus.PARTIAL:
            # Partial success - some days fetched, but failed partway through
            # 1) Process the successfully fetched days
            expected_per_day = get_expected_rows(task.dataset, num_days=1)
            thresholds = get_qc_thresholds()
            updates: list[dict] = []

            if result.rows_by_date:
                # Get trading calendar
                from data_layer.reference.calendars import get_calendar
                try:
                    cal = get_calendar("XNYS")
                except Exception:
                    cal = None

                for dt, row_count in result.rows_by_date.items():
                    from datetime import date as dt_date
                    if isinstance(dt, str):
                        check_date = dt_date.fromisoformat(dt)
                    elif hasattr(dt, 'date'):
                        check_date = dt.date()
                    else:
                        check_date = dt

                    is_trading = True
                    if cal is not None:
                        try:
                            is_trading = cal.is_trading_day(check_date)
                        except Exception:
                            pass

                    if not is_trading and row_count == 0:
                        updates.append(
                            dict(
                                source_name=result.vendor_used or self.vendor,
                                table_name=task.dataset,
                                symbol=task.symbol,
                                dt=dt.isoformat() if hasattr(dt, "isoformat") else str(dt),
                                status=PartitionStatus.GREEN,
                                qc_score=1.0,
                                row_count=0,
                                expected_rows=0,
                                qc_code="NO_SESSION",
                                fetch_params=result.fetch_params,
                            )
                        )
                        continue

                    status, qc_code = validate_partition_row_count(
                        row_count=row_count,
                        expected_rows=expected_per_day,
                        qc_code=result.qc_code,
                        thresholds=thresholds,
                    )

                    updates.append(
                        dict(
                            source_name=result.vendor_used or self.vendor,
                            table_name=task.dataset,
                            symbol=task.symbol,
                            dt=dt.isoformat() if hasattr(dt, "isoformat") else str(dt),
                            status=status,
                            qc_score=1.0 if status == PartitionStatus.GREEN else 0.5,
                            row_count=row_count,
                            expected_rows=expected_per_day,
                            qc_code=qc_code,
                            fetch_params=result.fetch_params,
                        )
                    )

            # 2) Create follow-up task for the remaining days
            if result.failed_at_date and result.original_end_date:
                new_task_id = self.db.enqueue_task(
                    dataset=task.dataset,
                    symbol=task.symbol,
                    start_date=result.failed_at_date,
                    end_date=result.original_end_date,
                    kind=task.kind,  # Preserve original kind
                    priority=_capped_followup_priority(task),
                    allow_overlap=True,  # Follow-up overlaps original range
                )
                if new_task_id:
                    logger.info(f"Created follow-up task {new_task_id} for {task.symbol} {result.failed_at_date} to {result.original_end_date}")

            # 3) Mark original task as done (the partial data is saved)
            with self.db.transaction() as conn:
                for params in updates:
                    self.db.upsert_partition_status(conn=conn, **params)
                self.db.mark_task_done(task.id, conn=conn)
            logger.info(f"Task {task.id} partial: {result.rows} rows from {self.vendor}, failed at {result.failed_at_date}")

        elif result.status == FetchStatus.RATE_LIMITED:
            backoff = timedelta(seconds=BACKOFF_BASE_SECONDS * (2 ** task.attempts))
            backoff = min(backoff, timedelta(seconds=BACKOFF_MAX_SECONDS))
            backoff_until = now + backoff

            self.db.mark_task_failed(
                task.id,
                error=result.error or "Rate limited",
                backoff_until=backoff_until,
            )
            self._tasks_failed += 1
            logger.warning(f"Task {task.id} rate limited, retry after {backoff_until}")

        else:  # ERROR, NOT_ENTITLED, NOT_SUPPORTED
            backoff = timedelta(seconds=BACKOFF_BASE_SECONDS * (2 ** task.attempts))
            backoff = min(backoff, timedelta(seconds=BACKOFF_MAX_SECONDS))
            backoff_until = now + backoff

            self.db.mark_task_failed(
                task.id,
                error=result.error or "Unknown error",
                backoff_until=backoff_until,
            )
            self._tasks_failed += 1
            logger.warning(f"Task {task.id} error, retry after {backoff_until}: {result.error}")

    def stop(self) -> None:
        """Stop the worker."""
        self._running = False

    @property
    def is_running(self) -> bool:
        """Check if the worker is running."""
        return self._running

    @property
    def tasks_processed(self) -> int:
        """Get count of tasks processed by this worker."""
        return self._tasks_processed

    @property
    def tasks_failed(self) -> int:
        """Get count of tasks that failed for this worker."""
        return self._tasks_failed


class QueueWorkerPool:
    """
    Pool of per-vendor worker threads.

    Spawns dedicated threads for each vendor based on configured thread counts.
    """

    def __init__(
        self,
        thread_counts: Optional[dict[str, int]] = None,
        db: Optional[NodeDB] = None,
        budgets: Optional[BudgetManager] = None,
        node_id: Optional[str] = None,
    ):
        """
        Initialize the worker pool.

        Args:
            thread_counts: Number of threads per vendor (default: DEFAULT_THREAD_COUNTS)
            db: Database manager
            budgets: Budget manager
            node_id: Node identifier for leasing
        """
        self.thread_counts = thread_counts or DEFAULT_THREAD_COUNTS.copy()
        self.db = db or get_db()
        self.budgets = budgets or get_budget_manager()
        self.node_id = node_id or os.environ.get("EDGE_NODE_ID", os.uname().nodename)

        self._workers: list[VendorWorker] = []
        self._threads: list[threading.Thread] = []
        self._running = False

    def start(self) -> None:
        """Start all worker threads."""
        if self._running:
            logger.warning("QueueWorkerPool already running")
            return

        self._running = True

        for vendor, count in self.thread_counts.items():
            for i in range(count):
                worker = VendorWorker(
                    vendor=vendor,
                    worker_id=i,
                    db=self.db,
                    budgets=self.budgets,
                    node_id=self.node_id,
                )
                thread = threading.Thread(
                    target=worker.run,
                    name=f"worker-{vendor}-{i}",
                    daemon=True,
                )

                self._workers.append(worker)
                self._threads.append(thread)
                thread.start()

        total = sum(self.thread_counts.values())
        logger.info(f"QueueWorkerPool started: {total} worker threads ({self.thread_counts})")

    def stop(self) -> None:
        """Stop all worker threads."""
        if not self._running:
            return

        self._running = False

        # Signal all workers to stop
        for worker in self._workers:
            worker.stop()

        # Wait for threads to finish
        for thread in self._threads:
            thread.join(timeout=10)

        # Count any that didn't finish
        still_running = sum(1 for t in self._threads if t.is_alive())
        if still_running:
            logger.warning(f"{still_running} worker threads still running after timeout")

        self._workers.clear()
        self._threads.clear()

        logger.info("QueueWorkerPool stopped")

    @property
    def is_running(self) -> bool:
        """Check if the pool is running."""
        return self._running

    def get_stats(self) -> dict[str, int]:
        """Get aggregate task counts across all workers."""
        total_processed = 0
        total_failed = 0
        for worker in self._workers:
            total_processed += worker.tasks_processed
            total_failed += worker.tasks_failed
        return {
            "processed": total_processed,
            "failed": total_failed,
        }

    def get_active_counts(self) -> dict[str, int]:
        """Get count of currently active workers per vendor."""
        active: dict[str, int] = {}
        for worker in self._workers:
            if worker.is_running:
                active[worker.vendor] = active.get(worker.vendor, 0) + 1
        return active
