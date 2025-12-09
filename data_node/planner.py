"""
Planner loop for the unified Pi data-node.

Implements:
- Gap detection via audit_for_gaps()
- Forward ingest scheduling (every 15 min check)
- Light audit (every 4h) and heavy audit (02:00 local)
- Task queue population with proper priorities

See updated_node_spec.md §3.2 for planner semantics.
"""

from __future__ import annotations

import os
import threading
import time
from datetime import date, datetime, timedelta
from typing import Generator, Optional

from loguru import logger

from .db import NodeDB, TaskKind, TaskStatus, PartitionStatus, get_db
from .stages import (
    get_current_universe,
    get_date_range,
    check_promotion,
    get_current_stage,
)

# Import calendar for trading day calculations
try:
    from data_layer.reference.calendars import get_trading_days, get_calendar
except ImportError:
    get_trading_days = None
    get_calendar = None
    logger.warning("Calendar module not available, using simple date iteration")


# Planner intervals
FORWARD_CHECK_INTERVAL = 900  # 15 minutes
LIGHT_AUDIT_INTERVAL = 4 * 3600  # 4 hours
HEAVY_AUDIT_HOUR = 2  # 02:00 local time

# Priority levels per spec
PRIORITY_BOOTSTRAP = 0
PRIORITY_GAP = 1
PRIORITY_QC_PROBE = 2
PRIORITY_FORWARD = 5


def _generate_trading_days(start_date: date, end_date: date) -> list[date]:
    """
    Generate trading days between start and end dates.

    Uses exchange_calendars if available, otherwise yields all weekdays.
    """
    if get_trading_days is not None:
        try:
            return get_trading_days(start_date, end_date)
        except Exception as e:
            logger.warning(f"Calendar lookup failed: {e}, falling back to weekdays")

    # Fallback: generate weekdays
    days = []
    current = start_date
    while current <= end_date:
        if current.weekday() < 5:  # Mon-Fri
            days.append(current)
        current += timedelta(days=1)
    return days


def _is_trading_day(dt: date) -> bool:
    """Check if a date is a trading day."""
    if get_calendar is not None:
        try:
            cal = get_calendar("XNYS")
            return cal.is_trading_day(dt)
        except Exception:
            pass
    # Fallback: weekdays
    return dt.weekday() < 5


def audit_for_gaps(
    datasets: list[str],
    db: Optional[NodeDB] = None,
) -> Generator[tuple[str, str, date, PartitionStatus], None, None]:
    """
    Audit partition_status for gaps in data coverage.

    Yields tuples of (dataset, symbol, date, status) for any date that is:
    - Missing from partition_status (RED)
    - Has AMBER or RED status

    Args:
        datasets: List of dataset names to audit
        db: Database instance

    Yields:
        Tuples of (dataset, symbol, date, status) for each gap
    """
    if db is None:
        db = get_db()

    universe = get_current_universe()
    if not universe:
        logger.warning("Empty universe, skipping audit")
        return

    for dataset in datasets:
        start_date, end_date = get_date_range(dataset)
        expected_days = _generate_trading_days(start_date, end_date)

        if not expected_days:
            logger.debug(f"No trading days in range for {dataset}")
            continue

        logger.debug(
            f"Auditing {dataset}: {len(universe)} symbols × "
            f"{len(expected_days)} days ({start_date} to {end_date})"
        )

        # Get existing partition_status for this dataset
        existing = db.get_partition_status_batch(
            table_name=dataset,
            symbols=universe,
            start_date=start_date,
            end_date=end_date,
        )

        # Build lookup: (symbol, date) -> status
        status_lookup: dict[tuple[str, date], PartitionStatus] = {}
        for row in existing:
            key = (row["symbol"], row["dt"])
            status_lookup[key] = PartitionStatus(row["status"])

        # Find gaps
        gaps_found = 0
        for symbol in universe:
            for dt in expected_days:
                key = (symbol, dt)
                status = status_lookup.get(key)

                if status is None:
                    # Missing entirely = RED
                    yield (dataset, symbol, dt, PartitionStatus.RED)
                    gaps_found += 1
                elif status in (PartitionStatus.RED, PartitionStatus.AMBER):
                    yield (dataset, symbol, dt, status)
                    gaps_found += 1

        logger.debug(f"Audit found {gaps_found} gaps in {dataset}")


def upsert_gap_tasks(
    gaps: Generator[tuple[str, str, date, PartitionStatus], None, None],
    db: Optional[NodeDB] = None,
) -> int:
    """
    Create GAP tasks for each detected gap.

    Uses UNIQUE constraint for automatic dedup.

    Args:
        gaps: Generator of (dataset, symbol, date, status) tuples
        db: Database instance

    Returns:
        Number of tasks created
    """
    if db is None:
        db = get_db()

    created = 0
    for dataset, symbol, dt, status in gaps:
        task_id = db.enqueue_task(
            dataset=dataset,
            symbol=symbol,
            start_date=dt,
            end_date=dt,
            kind=TaskKind.GAP,
            priority=PRIORITY_GAP,
            allow_overlap=False,  # Avoid duplicates when BOOTSTRAP/FORWARD already queued
        )
        if task_id:
            created += 1

    return created


def schedule_forward_tasks(
    datasets: list[str],
    db: Optional[NodeDB] = None,
) -> int:
    """
    Schedule FORWARD tasks for today's data.

    Checks if today is a trading day and if data already exists.

    Args:
        datasets: List of datasets to check
        db: Database instance

    Returns:
        Number of tasks created
    """
    if db is None:
        db = get_db()

    today = date.today()

    # Only schedule on trading days
    if not _is_trading_day(today):
        logger.debug(f"{today} is not a trading day, skipping forward ingest")
        return 0

    universe = get_current_universe()
    if not universe:
        return 0

    created = 0

    for dataset in datasets:
        # Get existing status for today
        existing = db.get_partition_status_batch(
            table_name=dataset,
            symbols=universe,
            start_date=today,
            end_date=today,
        )

        # Build set of symbols already GREEN for today
        green_symbols = {
            row["symbol"]
            for row in existing
            if row["status"] == PartitionStatus.GREEN.value
        }

        # Schedule tasks for symbols without GREEN status
        for symbol in universe:
            if symbol in green_symbols:
                continue

            task_id = db.enqueue_task(
                dataset=dataset,
                symbol=symbol,
                start_date=today,
                end_date=today,
                kind=TaskKind.FORWARD,
                priority=PRIORITY_FORWARD,
            )
            if task_id:
                created += 1

    if created > 0:
        logger.info(f"Scheduled {created} FORWARD tasks for {today}")

    return created


def prune_old_failed_tasks(days: int = 30, db: Optional[NodeDB] = None) -> int:
    """
    Prune failed tasks older than N days.

    Args:
        days: Age threshold in days
        db: Database instance

    Returns:
        Number of tasks deleted
    """
    if db is None:
        db = get_db()

    threshold = datetime.utcnow() - timedelta(days=days)
    deleted = db.prune_failed_tasks(before=threshold)

    if deleted > 0:
        logger.info(f"Pruned {deleted} failed tasks older than {days} days")

    return deleted


def audit_amber_partitions(
    datasets: Optional[list[str]] = None,
    db: Optional[NodeDB] = None,
    limit_per_table: int = 1000,
) -> int:
    """
    Find AMBER partitions and create GAP tasks to refetch them.

    AMBER partitions are those that failed row-count validation during QC
    (e.g., incomplete data due to API limits). This function creates GAP
    tasks so the worker pool can refetch them.

    Args:
        datasets: List of datasets to audit (default: equities_eod, equities_minute)
        db: Database instance
        limit_per_table: Maximum partitions to process per table

    Returns:
        Number of GAP tasks created
    """
    if db is None:
        db = get_db()

    if datasets is None:
        datasets = ["equities_eod", "equities_minute"]

    created = 0

    for table in datasets:
        # Get AMBER partitions
        amber_partitions = db.get_partitions_by_status(
            table_name=table,
            status=PartitionStatus.AMBER,
            limit=limit_per_table,
        )

        if not amber_partitions:
            logger.debug(f"No AMBER partitions found for {table}")
            continue

        logger.info(f"Found {len(amber_partitions)} AMBER partitions in {table}")

        for p in amber_partitions:
            # Create GAP task for this partition
            task_id = db.enqueue_task(
                dataset=table,
                symbol=p.symbol,
                start_date=p.dt,
                end_date=p.dt,
                kind=TaskKind.GAP,
                priority=PRIORITY_GAP,
            )

            if task_id:
                created += 1
                logger.debug(f"Created GAP task for AMBER partition: {table}/{p.symbol}/{p.dt}")

    if created > 0:
        logger.info(f"Created {created} GAP tasks for AMBER partitions")

    return created


def audit_red_partitions(
    datasets: Optional[list[str]] = None,
    db: Optional[NodeDB] = None,
    limit_per_table: int = 500,
) -> int:
    """
    Find RED partitions and create high-priority GAP tasks.

    RED partitions are those with complete data loss (0 rows).
    These get higher priority than AMBER.

    Args:
        datasets: List of datasets to audit (default: equities_eod, equities_minute)
        db: Database instance
        limit_per_table: Maximum partitions to process per table

    Returns:
        Number of GAP tasks created
    """
    if db is None:
        db = get_db()

    if datasets is None:
        datasets = ["equities_eod", "equities_minute"]

    created = 0

    for table in datasets:
        # Get RED partitions
        red_partitions = db.get_partitions_by_status(
            table_name=table,
            status=PartitionStatus.RED,
            limit=limit_per_table,
        )

        if not red_partitions:
            continue

        logger.info(f"Found {len(red_partitions)} RED partitions in {table}")

        for p in red_partitions:
            # Create high-priority GAP task
            task_id = db.enqueue_task(
                dataset=table,
                symbol=p.symbol,
                start_date=p.dt,
                end_date=p.dt,
                kind=TaskKind.GAP,
                priority=PRIORITY_BOOTSTRAP,  # Highest priority for complete failures
            )

            if task_id:
                created += 1

    if created > 0:
        logger.info(f"Created {created} high-priority GAP tasks for RED partitions")

    return created


class PlannerLoop:
    """
    Background loop that runs gap audits and schedules tasks.

    Runs:
    - Forward ingest check every 15 minutes
    - Light audit every 4 hours
    - Heavy audit at 02:00 local time
    """

    def __init__(
        self,
        db: Optional[NodeDB] = None,
        datasets: Optional[list[str]] = None,
        forward_interval: int = FORWARD_CHECK_INTERVAL,
        light_interval: int = LIGHT_AUDIT_INTERVAL,
        heavy_hour: int = HEAVY_AUDIT_HOUR,
    ):
        """
        Initialize the planner loop.

        Args:
            db: Database instance
            datasets: List of datasets to manage (default: equities_eod, equities_minute)
            forward_interval: Seconds between forward checks
            light_interval: Seconds between light audits
            heavy_hour: Hour (local time) for heavy audit
        """
        self.db = db or get_db()
        self.datasets = datasets or ["equities_eod", "equities_minute"]
        self.forward_interval = forward_interval
        self.light_interval = light_interval
        self.heavy_hour = heavy_hour

        # Tracking timestamps
        self._last_forward_check: Optional[datetime] = None
        self._last_light_audit: Optional[datetime] = None
        self._last_heavy_audit_date: Optional[date] = None
        self._last_tick: Optional[datetime] = None

        # Loop control
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def run_light_audit(self) -> int:
        """
        Run a light audit for active datasets.

        Detects gaps and enqueues GAP tasks.

        Returns:
            Number of tasks created
        """
        logger.info("Running light audit...")
        gaps = audit_for_gaps(self.datasets, self.db)
        created = upsert_gap_tasks(gaps, self.db)
        logger.info(f"Light audit complete: {created} GAP tasks created")
        self._last_light_audit = datetime.now()
        return created

    def run_heavy_audit(self) -> dict:
        """
        Run a heavy audit with additional maintenance.

        Includes:
        - Full gap audit for all datasets
        - AMBER/RED partition re-queueing
        - Stage promotion check
        - Failed task pruning

        Returns:
            Dict with audit results
        """
        logger.info("Running heavy audit...")

        results = {
            "gaps_created": 0,
            "amber_requeued": 0,
            "red_requeued": 0,
            "promoted": False,
            "pruned": 0,
        }

        # Full gap audit
        gaps = audit_for_gaps(self.datasets, self.db)
        results["gaps_created"] = upsert_gap_tasks(gaps, self.db)

        # Re-queue AMBER partitions (row-count mismatches from QC)
        results["amber_requeued"] = audit_amber_partitions(self.datasets, self.db)

        # Re-queue RED partitions (complete failures)
        results["red_requeued"] = audit_red_partitions(self.datasets, self.db)

        # Check stage promotion
        results["promoted"] = check_promotion(self.db)

        # Prune old failed tasks
        results["pruned"] = prune_old_failed_tasks(days=30, db=self.db)

        logger.info(
            f"Heavy audit complete: {results['gaps_created']} gaps, "
            f"{results['amber_requeued']} AMBER re-queued, "
            f"{results['red_requeued']} RED re-queued, "
            f"promoted={results['promoted']}, pruned={results['pruned']}"
        )

        self._last_heavy_audit_date = date.today()
        return results

    def run_forward_check(self) -> int:
        """
        Check and schedule forward ingest tasks.

        Returns:
            Number of tasks created
        """
        created = schedule_forward_tasks(self.datasets, self.db)
        self._last_forward_check = datetime.now()
        return created

    def should_run_forward(self) -> bool:
        """Check if forward check should run."""
        if self._last_forward_check is None:
            return True

        elapsed = (datetime.now() - self._last_forward_check).total_seconds()
        return elapsed >= self.forward_interval

    def should_run_light_audit(self) -> bool:
        """Check if light audit should run."""
        if self._last_light_audit is None:
            return True

        elapsed = (datetime.now() - self._last_light_audit).total_seconds()
        return elapsed >= self.light_interval

    def should_run_heavy_audit(self) -> bool:
        """Check if heavy audit should run (02:00 local, once per day)."""
        now = datetime.now()

        # Only run at the configured hour
        if now.hour != self.heavy_hour:
            return False

        # Only run once per day
        if self._last_heavy_audit_date == now.date():
            return False

        return True

    def tick(self) -> dict:
        """
        Run one tick of the planner loop.

        Returns:
            Dict with actions taken
        """
        actions = {
            "forward_tasks": 0,
            "light_audit_tasks": 0,
            "heavy_audit": None,
        }

        # Heavy audit takes precedence (runs at 02:00)
        if self.should_run_heavy_audit():
            actions["heavy_audit"] = self.run_heavy_audit()
            # Heavy audit includes everything, so skip light
            self._last_light_audit = datetime.now()
            self._last_forward_check = datetime.now()
            return actions

        # Light audit every 4 hours
        if self.should_run_light_audit():
            actions["light_audit_tasks"] = self.run_light_audit()

        # Forward check every 15 minutes
        if self.should_run_forward():
            actions["forward_tasks"] = self.run_forward_check()

        self._last_tick = datetime.now()
        return actions

    def start(self, threaded: bool = True) -> None:
        """
        Start the planner loop.

        Args:
            threaded: If True, run in background thread
        """
        self._running = True

        if threaded:
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
            logger.info("PlannerLoop started in background thread")
        else:
            self._run_loop()

    def stop(self) -> None:
        """Stop the planner loop."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
            self._thread = None
        logger.info("PlannerLoop stopped")

    def _run_loop(self) -> None:
        """Main planner loop."""
        logger.info("PlannerLoop running")

        # Run initial light audit
        try:
            self.run_light_audit()
        except Exception as e:
            logger.exception(f"Initial audit failed: {e}")

        while self._running:
            try:
                self.tick()
                # Sleep for 60 seconds between checks
                time.sleep(60)

            except KeyboardInterrupt:
                logger.info("PlannerLoop interrupted")
                break
            except Exception as e:
                logger.exception(f"Error in planner loop: {e}")
                time.sleep(60)

        logger.info("PlannerLoop exiting")

    @property
    def is_running(self) -> bool:
        """Check if the loop is running."""
        return self._running

    @property
    def last_tick(self) -> Optional[datetime]:
        """Get the timestamp of the last tick."""
        return self._last_tick

    def get_status(self) -> dict:
        """Get planner status for display."""
        return {
            "running": self._running,
            "last_forward_check": (
                self._last_forward_check.isoformat() if self._last_forward_check else None
            ),
            "last_light_audit": (
                self._last_light_audit.isoformat() if self._last_light_audit else None
            ),
            "last_heavy_audit_date": (
                self._last_heavy_audit_date.isoformat() if self._last_heavy_audit_date else None
            ),
            "current_stage": get_current_stage(),
            "universe_size": len(get_current_universe()),
        }
