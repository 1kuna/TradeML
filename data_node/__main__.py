#!/usr/bin/env python3
"""
Entry point for the unified Pi data-node service.

Starts all four loops and the Rich dashboard:
- QueueWorkerLoop: Processes tasks from backfill_queue
- PlannerLoop: Detects gaps and schedules tasks
- MaintenanceLoop: Runs nightly curation, QC, and export
- Dashboard: Rich live status display

Usage:
  python -m data_node              # Start the node with dashboard
  python -m data_node --no-ui      # Start without dashboard (for systemd)
  python -m data_node --status     # Show current status and exit
  python -m data_node --selfcheck  # Run self-checks and exit

See updated_node_spec.md for architecture details.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

from loguru import logger

# Ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Load environment
from dotenv import load_dotenv
load_dotenv()

from .db import get_db, NodeDB
from .budgets import get_budget_manager, BudgetManager
from .worker import QueueWorkerLoop
from .planner import PlannerLoop
from .maintenance import MaintenanceLoop
from .stages import get_stage_info, get_current_universe
from .ui import NodeStatus, Dashboard, setup_log_handler, print_simple_status


def configure_logging(log_file: bool = True, verbose: bool = False) -> None:
    """Configure loguru logging."""
    logger.remove()

    # Console output
    level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>{name}</cyan> - {message}",
        level=level,
        colorize=True,
    )

    # File output
    if log_file:
        log_dir = Path(os.environ.get("DATA_ROOT", ".")) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "data_node.log"

        logger.add(
            log_path,
            rotation="10 MB",
            retention="7 days",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name} - {message}",
            level="DEBUG",
        )
        logger.info(f"Logging to {log_path}")


def selfcheck() -> bool:
    """
    Run self-checks and report status.

    Returns:
        True if all checks pass
    """
    logger.info("Running self-checks...")
    errors = []
    warnings = []

    # Check DATA_ROOT
    data_root = Path(os.environ.get("DATA_ROOT", "."))
    if not data_root.exists():
        errors.append(f"DATA_ROOT does not exist: {data_root}")
    else:
        logger.info(f"DATA_ROOT: {data_root}")

    # Check database
    try:
        db = get_db()
        stats = db.get_queue_stats()
        logger.info(f"Database OK: {stats['total']} tasks in queue")
    except Exception as e:
        errors.append(f"Database error: {e}")

    # Check budgets config
    try:
        budgets = get_budget_manager()
        status = budgets.get_all_status()
        logger.info(f"Budgets OK: {len(status)} vendors configured")
    except Exception as e:
        warnings.append(f"Budgets warning: {e}")

    # Check API keys
    required_keys = ["ALPACA_API_KEY", "ALPACA_SECRET_KEY"]
    missing_keys = [k for k in required_keys if not os.environ.get(k)]
    if missing_keys:
        errors.append(f"Missing required API keys: {', '.join(missing_keys)}")

    optional_keys = ["FINNHUB_API_KEY", "POLYGON_API_KEY", "FRED_API_KEY"]
    missing_optional = [k for k in optional_keys if not os.environ.get(k)]
    if missing_optional:
        warnings.append(f"Missing optional API keys: {', '.join(missing_optional)}")

    # Check stage config
    try:
        stage_info = get_stage_info()
        logger.info(f"Stage: {stage_info['current_stage']} ({stage_info['name']})")
        logger.info(f"Universe: {stage_info['universe_size']} symbols")
    except Exception as e:
        warnings.append(f"Stage config warning: {e}")

    # Report
    for warning in warnings:
        logger.warning(warning)

    if errors:
        for error in errors:
            logger.error(error)
        logger.error("Self-check FAILED")
        return False

    logger.info("Self-check PASSED")
    return True


def show_status() -> None:
    """Show current node status and exit."""
    status = NodeStatus()

    # Get node info
    status.node_id = os.environ.get("EDGE_NODE_ID", "unknown")
    status.env = os.environ.get("TRADEML_ENV", "local")
    status.data_root = os.environ.get("DATA_ROOT", ".")

    # Get stage info
    try:
        stage_info = get_stage_info()
        status.current_stage = stage_info["current_stage"]
        status.universe_size = stage_info["universe_size"]
    except Exception:
        pass

    # Get queue stats
    try:
        db = get_db()
        stats = db.get_queue_stats()
        status.update_queue_stats(stats)

        # Get coverage
        from .stages import get_date_range
        start, end = get_date_range("equities_eod")
        coverage = db.get_green_coverage(
            table_name="equities_eod",
            start_date=start,
            end_date=end,
        )
        status.green_coverage = coverage
    except Exception:
        pass

    # Get budget status
    try:
        budgets = get_budget_manager()
        for vendor, budget_status in budgets.get_all_status().items():
            status.update_vendor(
                vendor,
                spent_today=budget_status["spent_today"],
                daily_cap=budget_status["soft_daily_cap"],
                tokens_rpm=budget_status["tokens_rpm"],
                hard_rpm=budget_status["hard_rpm"],
            )
    except Exception:
        pass

    print_simple_status(status)


def run_node(
    with_ui: bool = True,
    verbose: bool = False,
) -> None:
    """
    Run the data node with all loops.

    Args:
        with_ui: Whether to show the Rich dashboard
        verbose: Enable verbose logging
    """
    configure_logging(log_file=True, verbose=verbose)

    logger.info("Starting Pi Data-Node...")

    # Run self-check
    if not selfcheck():
        logger.error("Self-check failed, aborting")
        sys.exit(1)

    # Create shared status
    status = NodeStatus()
    status.node_id = os.environ.get("EDGE_NODE_ID", "unknown")
    status.env = os.environ.get("TRADEML_ENV", "local")
    status.data_root = os.environ.get("DATA_ROOT", ".")
    status.started_at = datetime.now()

    # Get stage info
    try:
        stage_info = get_stage_info()
        status.current_stage = stage_info["current_stage"]
        status.universe_size = stage_info["universe_size"]
    except Exception as e:
        logger.warning(f"Could not load stage info: {e}")

    # Set up log handler for dashboard
    if with_ui:
        setup_log_handler(status)

    # Create loops
    db = get_db()
    budgets = get_budget_manager()

    worker_loop = QueueWorkerLoop()
    planner_loop = PlannerLoop(db=db)
    maintenance_loop = MaintenanceLoop(db=db)

    # Register loops in status
    for name in ["Worker", "Planner", "Maintenance"]:
        status.update_loop(name, running=False)

    # Shutdown handler
    shutdown_requested = False

    def handle_shutdown(signum, frame):
        nonlocal shutdown_requested
        if shutdown_requested:
            logger.warning("Force shutdown requested")
            sys.exit(1)
        logger.info("Shutdown requested, stopping loops...")
        shutdown_requested = True

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    # Start loops
    logger.info("Starting loops...")

    worker_loop.start(threaded=True)
    status.update_loop("Worker", running=True)
    logger.info("Worker loop started")

    planner_loop.start(threaded=True)
    status.update_loop("Planner", running=True)
    logger.info("Planner loop started")

    maintenance_loop.start(threaded=True)
    status.update_loop("Maintenance", running=True)
    logger.info("Maintenance loop started")

    # Start dashboard
    dashboard = None
    if with_ui:
        dashboard = Dashboard(status)
        dashboard.start(threaded=True)
        logger.info("Dashboard started")

    # Main loop - update status periodically
    logger.info("Data node running. Press Ctrl-C to stop.")

    try:
        while not shutdown_requested:
            # Update queue stats
            try:
                stats = db.get_queue_stats()
                status.update_queue_stats(stats)
            except Exception:
                pass

            # Update budget status
            try:
                for vendor, budget_status in budgets.get_all_status().items():
                    status.update_vendor(
                        vendor,
                        spent_today=budget_status["spent_today"],
                        daily_cap=budget_status["soft_daily_cap"],
                        tokens_rpm=budget_status["tokens_rpm"],
                        hard_rpm=budget_status["hard_rpm"],
                    )
            except Exception:
                pass

            # Update loop status
            status.update_loop("Worker", running=worker_loop.is_running)
            status.update_loop("Planner", running=planner_loop.is_running)
            status.update_loop("Maintenance", running=maintenance_loop.is_running)

            time.sleep(1)

    except KeyboardInterrupt:
        pass

    # Shutdown
    logger.info("Stopping loops...")

    if dashboard:
        dashboard.stop()

    worker_loop.stop()
    status.update_loop("Worker", running=False)

    planner_loop.stop()
    status.update_loop("Planner", running=False)

    maintenance_loop.stop()
    status.update_loop("Maintenance", running=False)

    # Save budget state
    try:
        budgets.save()
        logger.info("Budget state saved")
    except Exception as e:
        logger.warning(f"Failed to save budget state: {e}")

    logger.info("Data node stopped")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Pi Data-Node Service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--no-ui",
        action="store_true",
        help="Run without Rich dashboard (for systemd/background)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current status and exit",
    )
    parser.add_argument(
        "--selfcheck",
        action="store_true",
        help="Run self-checks and exit",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.status:
        show_status()
        return

    if args.selfcheck:
        configure_logging(log_file=False, verbose=args.verbose)
        success = selfcheck()
        sys.exit(0 if success else 1)

    run_node(with_ui=not args.no_ui, verbose=args.verbose)


if __name__ == "__main__":
    main()
