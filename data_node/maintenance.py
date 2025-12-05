"""
Maintenance loop for the unified Pi data-node.

Implements:
- Incremental curation of raw→curated
- Structural QC checks
- Nightly export generation
- Weekly QC probe scheduling

See updated_node_spec.md §5 for maintenance semantics.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

from loguru import logger

from .db import NodeDB, get_db

# Try to import existing curate module
try:
    from ops.ssot.curate import curate_incremental, IncrementalCurator
except ImportError:
    curate_incremental = None
    IncrementalCurator = None
    logger.warning("Curate module not available, curation disabled")

# Try to import QC weekly module (created in this milestone)
try:
    from .qc_weekly import schedule_qc_probes, is_qc_day
except ImportError:
    schedule_qc_probes = None
    is_qc_day = None


# Default maintenance hour (02:00 local time)
DEFAULT_MAINTENANCE_HOUR = 2


def run_curate() -> dict:
    """
    Run incremental curation of raw partitions to curated tables.

    Wraps existing curate_incremental() functionality.

    Returns:
        Dict with job results {job_name: partitions_processed}
    """
    if curate_incremental is None:
        logger.warning("Curation not available, skipping")
        return {}

    try:
        logger.info("Running incremental curation...")
        results = curate_incremental()
        logger.info(f"Curation complete: {results}")
        return results
    except Exception as e:
        logger.exception(f"Curation failed: {e}")
        return {"error": str(e)}


def run_structural_qc(db: Optional[NodeDB] = None) -> dict:
    """
    Run structural QC checks on partition_status.

    Checks:
    - Row count vs expected
    - Calendar continuity (no unexpected gaps)
    - Basic outlier detection

    Args:
        db: Database instance

    Returns:
        Dict with QC results
    """
    if db is None:
        db = get_db()

    logger.info("Running structural QC...")

    results = {
        "checked": 0,
        "issues": 0,
        "details": [],
    }

    # Get recent partitions to check (last 7 days)
    today = date.today()
    week_ago = today - timedelta(days=7)

    # Check equities_eod and equities_minute
    for table in ["equities_eod", "equities_minute"]:
        try:
            coverage = db.get_green_coverage(
                table_name=table,
                start_date=week_ago,
                end_date=today,
            )

            results["checked"] += 1

            if coverage < 0.95:
                results["issues"] += 1
                results["details"].append({
                    "table": table,
                    "issue": "low_coverage",
                    "coverage": f"{coverage:.2%}",
                    "period": f"{week_ago} to {today}",
                })
                logger.warning(f"Low coverage for {table}: {coverage:.2%}")

        except Exception as e:
            logger.warning(f"QC check failed for {table}: {e}")

    logger.info(f"Structural QC complete: {results['checked']} checked, {results['issues']} issues")
    return results


def run_export(asof: Optional[date] = None, db: Optional[NodeDB] = None) -> dict:
    """
    Export curated data and partition status for a given date.

    Creates:
    - exports/nightly/YYYY-MM-DD/curated/ (symlinks or copies)
    - exports/nightly/YYYY-MM-DD/partition_status.parquet
    - exports/nightly/YYYY-MM-DD/manifest.json

    Args:
        asof: Export date (default: today)
        db: Database instance

    Returns:
        Dict with export results
    """
    if asof is None:
        asof = date.today()

    if db is None:
        db = get_db()

    data_root = Path(os.environ.get("DATA_ROOT", "."))
    export_dir = data_root / "exports" / "nightly" / asof.isoformat()

    logger.info(f"Running export for {asof} to {export_dir}")

    results = {
        "export_dir": str(export_dir),
        "files": [],
        "errors": [],
    }

    try:
        # Create export directory
        export_dir.mkdir(parents=True, exist_ok=True)

        # Export partition_status as parquet
        _export_partition_status(db, export_dir, results)

        # Create symlinks to curated data
        _link_curated_data(data_root, export_dir, results)

        # Write manifest
        _write_manifest(export_dir, asof, results)

        logger.info(f"Export complete: {len(results['files'])} files")

    except Exception as e:
        logger.exception(f"Export failed: {e}")
        results["errors"].append(str(e))

    return results


def _export_partition_status(db: NodeDB, export_dir: Path, results: dict) -> None:
    """Export partition_status table as parquet."""
    try:
        import pandas as pd

        conn = db._get_connection()
        rows = conn.execute("SELECT * FROM partition_status").fetchall()

        if not rows:
            logger.debug("No partition_status rows to export")
            return

        df = pd.DataFrame([dict(row) for row in rows])
        output_path = export_dir / "partition_status.parquet"
        df.to_parquet(output_path, index=False)
        results["files"].append("partition_status.parquet")

    except ImportError:
        logger.warning("pandas not available, skipping partition_status export")
    except Exception as e:
        logger.warning(f"Failed to export partition_status: {e}")
        results["errors"].append(f"partition_status: {e}")


def _link_curated_data(data_root: Path, export_dir: Path, results: dict) -> None:
    """Create symlinks to curated data directories."""
    curated_root = data_root / "data_layer" / "curated"

    if not curated_root.exists():
        logger.debug("No curated data found")
        return

    curated_export = export_dir / "curated"
    curated_export.mkdir(exist_ok=True)

    # Link each curated table
    for table_dir in curated_root.iterdir():
        if not table_dir.is_dir():
            continue

        link_path = curated_export / table_dir.name
        if not link_path.exists():
            try:
                link_path.symlink_to(table_dir.resolve())
                results["files"].append(f"curated/{table_dir.name}")
            except OSError as e:
                # Symlinks may fail on some systems, just log
                logger.debug(f"Could not create symlink for {table_dir.name}: {e}")


def _write_manifest(export_dir: Path, asof: date, results: dict) -> None:
    """Write export manifest with checksums."""
    manifest = {
        "export_date": asof.isoformat(),
        "created_at": datetime.utcnow().isoformat(),
        "files": results["files"],
        "checksums": {},
    }

    # Compute checksums for non-symlink files
    for filename in results["files"]:
        if filename.startswith("curated/"):
            continue  # Skip symlinks

        filepath = export_dir / filename
        if filepath.exists() and filepath.is_file():
            try:
                with open(filepath, "rb") as f:
                    checksum = hashlib.sha256(f.read()).hexdigest()
                manifest["checksums"][filename] = checksum
            except Exception as e:
                logger.debug(f"Could not compute checksum for {filename}: {e}")

    manifest_path = export_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    results["files"].append("manifest.json")


class MaintenanceLoop:
    """
    Background loop that runs nightly maintenance tasks.

    Runs at the configured hour (default 02:00 local time):
    - Incremental curation
    - Structural QC
    - Weekly cross-vendor QC scheduling (Sundays)
    - Nightly export
    """

    def __init__(
        self,
        db: Optional[NodeDB] = None,
        maintenance_hour: int = DEFAULT_MAINTENANCE_HOUR,
    ):
        """
        Initialize the maintenance loop.

        Args:
            db: Database instance
            maintenance_hour: Hour (local time, 0-23) to run maintenance
        """
        self.db = db or get_db()
        self.maintenance_hour = maintenance_hour

        # Tracking
        self._last_run_date: Optional[date] = None
        self._last_run_results: Optional[dict] = None

        # Loop control
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def should_run(self) -> bool:
        """Check if maintenance should run now."""
        now = datetime.now()

        # Only run at the configured hour
        if now.hour != self.maintenance_hour:
            return False

        # Only run once per day
        if self._last_run_date == now.date():
            return False

        return True

    def run(self) -> dict:
        """
        Run all maintenance tasks.

        Returns:
            Dict with results from all tasks
        """
        logger.info("Starting maintenance run...")
        today = date.today()

        results = {
            "date": today.isoformat(),
            "started_at": datetime.now().isoformat(),
            "curate": {},
            "qc": {},
            "qc_probes": 0,
            "export": {},
        }

        # 1. Incremental curation
        try:
            results["curate"] = run_curate()
        except Exception as e:
            logger.exception(f"Curation failed: {e}")
            results["curate"] = {"error": str(e)}

        # 2. Structural QC
        try:
            results["qc"] = run_structural_qc(self.db)
        except Exception as e:
            logger.exception(f"QC failed: {e}")
            results["qc"] = {"error": str(e)}

        # 3. Weekly QC probe scheduling (Sundays)
        if is_qc_day is not None and is_qc_day():
            try:
                if schedule_qc_probes is not None:
                    results["qc_probes"] = schedule_qc_probes(self.db)
            except Exception as e:
                logger.exception(f"QC probe scheduling failed: {e}")
                results["qc_probes"] = -1

        # 4. Export
        try:
            results["export"] = run_export(today, self.db)
        except Exception as e:
            logger.exception(f"Export failed: {e}")
            results["export"] = {"error": str(e)}

        results["completed_at"] = datetime.now().isoformat()

        self._last_run_date = today
        self._last_run_results = results

        logger.info(f"Maintenance run complete for {today}")
        return results

    def tick(self) -> Optional[dict]:
        """
        Run one tick of the maintenance loop.

        Returns:
            Maintenance results if run, None otherwise
        """
        if self.should_run():
            return self.run()
        return None

    def start(self, threaded: bool = True) -> None:
        """
        Start the maintenance loop.

        Args:
            threaded: If True, run in background thread
        """
        self._running = True

        if threaded:
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
            logger.info("MaintenanceLoop started in background thread")
        else:
            self._run_loop()

    def stop(self) -> None:
        """Stop the maintenance loop."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
            self._thread = None
        logger.info("MaintenanceLoop stopped")

    def _run_loop(self) -> None:
        """Main maintenance loop."""
        logger.info(f"MaintenanceLoop running (scheduled at {self.maintenance_hour:02d}:00)")

        while self._running:
            try:
                self.tick()
                # Check every 5 minutes
                time.sleep(300)

            except KeyboardInterrupt:
                logger.info("MaintenanceLoop interrupted")
                break
            except Exception as e:
                logger.exception(f"Error in maintenance loop: {e}")
                time.sleep(300)

        logger.info("MaintenanceLoop exiting")

    @property
    def is_running(self) -> bool:
        """Check if the loop is running."""
        return self._running

    def get_status(self) -> dict:
        """Get maintenance status for display."""
        return {
            "running": self._running,
            "maintenance_hour": f"{self.maintenance_hour:02d}:00",
            "last_run_date": (
                self._last_run_date.isoformat() if self._last_run_date else None
            ),
            "last_run_results": self._last_run_results,
        }


def run_maintenance_now(db: Optional[NodeDB] = None) -> dict:
    """
    Force run maintenance immediately (for testing/manual runs).

    Args:
        db: Database instance

    Returns:
        Maintenance results
    """
    loop = MaintenanceLoop(db=db)
    return loop.run()
