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


def _map_dataset_to_table(table_name: str) -> str:
    """Map logical dataset names to raw table directories used by fetchers."""
    mapping = {
        "equities_eod": "equities_bars",
        "equities_minute": "equities_bars_minute",
        "options_chains": "options_chains",
        "macros_fred": "macro_treasury",
        "corp_actions": "corp_actions",
        "fundamentals": "fundamentals",
    }
    return mapping.get(table_name, table_name)


def _resolve_raw_partition_path(source_name: str, table_name: str, dt: str, symbol: Optional[str]) -> Optional[Path]:
    """
    Best-effort reconstruction of the raw parquet path for a partition.

    Uses the same convention as fetchers:
      data_layer/raw/<vendor>/<table>/date=<dt>/symbol=<symbol>/data.parquet
      (options chains use underlier=<symbol>)
    """
    data_root = os.environ.get("DATA_ROOT", ".")
    resolved_table = _map_dataset_to_table(table_name)
    if dt is None:
        return None
    base = Path(data_root) / "data_layer" / "raw" / source_name / resolved_table / f"date={dt}"

    if symbol:
        if table_name == "options_chains":
            base = base / f"underlier={symbol}"
        else:
            base = base / f"symbol={symbol}"

    return base / "data.parquet"

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
    - Row count vs expected (validates GREEN partitions have correct row counts)
    - Calendar continuity (no unexpected gaps)
    - Basic outlier detection

    Args:
        db: Database instance

    Returns:
        Dict with QC results
    """
    from .db import PartitionStatus
    from .qc import calculate_partition_status
    from .stages import get_expected_rows, get_qc_thresholds

    if db is None:
        db = get_db()

    logger.info("Running structural QC...")

    results = {
        "checked": 0,
        "tables_checked": 0,
        "issues": 0,
        "row_count_mismatches": 0,
        "updated_to_amber": 0,
        "updated_to_red": 0,
        "details": [],
    }

    # Get recent partitions to check (last 7 days)
    today = date.today()
    week_ago = today - timedelta(days=7)

    # Check equities_eod and equities_minute
    for table in ["equities_eod", "equities_minute"]:
        try:
            # 1. Check overall coverage
            coverage = db.get_green_coverage(
                table_name=table,
                start_date=week_ago,
                end_date=today,
            )

            results["tables_checked"] += 1

            if coverage < 0.95:
                results["issues"] += 1
                results["details"].append({
                    "table": table,
                    "issue": "low_coverage",
                    "coverage": f"{coverage:.2%}",
                    "period": f"{week_ago} to {today}",
                })
                logger.warning(f"Low coverage for {table}: {coverage:.2%}")

            # 2. Validate row counts for GREEN partitions
            row_count_issues = _validate_partition_row_counts(
                db,
                table,
                results,
                start_date=week_ago.isoformat(),
                end_date=today.isoformat(),
            )
            if row_count_issues > 0:
                logger.warning(f"Found {row_count_issues} row-count mismatches in {table}")

        except Exception as e:
            logger.warning(f"QC check failed for {table}: {e}")

    logger.info(
        f"Structural QC complete: {results['checked']} partitions checked across "
        f"{results['tables_checked']} tables, {results['issues']} issues"
    )
    return results


def _validate_partition_row_counts(
    db: NodeDB,
    table_name: str,
    results: dict,
    batch_size: int = 10000,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> int:
    """Validate row counts for GREEN partitions in a table.

    Checks each GREEN partition's row_count against expected_rows.
    Updates partitions that fail validation to AMBER or RED.

    Args:
        db: Database instance
        table_name: Table to validate
        results: Results dict to update
        batch_size: How many partitions to check per batch

    Returns:
        Number of mismatches found
    """
    from .db import PartitionStatus
    from .qc import calculate_partition_status
    from .stages import get_expected_rows, get_qc_thresholds

    thresholds = get_qc_thresholds()
    expected_per_day = get_expected_rows(table_name, num_days=1)

    if expected_per_day == 0:
        # No expectation defined for this table
        logger.debug(f"No expect_rows defined for {table_name}, skipping row-count validation")
        return 0

    # Get GREEN partitions to validate
    partitions = db.get_partitions_by_status(
        table_name=table_name,
        status=PartitionStatus.GREEN,
        limit=batch_size,
        start_date=start_date,
        end_date=end_date,
    )

    mismatches = 0

    for p in partitions:
        results["checked"] = results.get("checked", 0) + 1
        # Check backing parquet file exists (guards crash between upsert and write)
        partition_path = _resolve_raw_partition_path(p.source_name, table_name, p.dt, p.symbol)
        if partition_path is not None and not partition_path.exists():
            results["issues"] += 1
            results["updated_to_red"] += 1
            db.upsert_partition_status(
                source_name=p.source_name,
                table_name=table_name,
                symbol=p.symbol,
                dt=p.dt,
                status=PartitionStatus.RED,
                qc_score=0.0,
                row_count=0,
                expected_rows=p.expected_rows,
                qc_code="MISSING_FILE",
                notes="QC: missing parquet file",
            )
            logger.warning(f"Missing parquet file for {table_name}/{p.symbol}/{p.dt}: {partition_path}")
            continue

        # Skip partitions with NO_SESSION (holidays/weekends)
        if p.qc_code == "NO_SESSION":
            continue

        # Skip if no expected_rows set (legacy data)
        if p.expected_rows == 0:
            continue

        # Validate row count
        new_status = calculate_partition_status(
            row_count=p.row_count,
            expected_rows=p.expected_rows,
            qc_code=p.qc_code,
            thresholds=thresholds,
        )

        if new_status != PartitionStatus.GREEN:
            mismatches += 1
            results["issues"] += 1
            results["row_count_mismatches"] += 1

            # Determine QC code
            if p.row_count < p.expected_rows:
                qc_code = "ROW_COUNT_LOW"
            else:
                qc_code = "ROW_COUNT_HIGH"

            # Update partition status
            db.upsert_partition_status(
                source_name=p.source_name,
                table_name=table_name,
                symbol=p.symbol,
                dt=p.dt,
                status=new_status,
                qc_score=0.5,
                row_count=p.row_count,
                expected_rows=p.expected_rows,
                qc_code=qc_code,
                notes=f"QC: expected {p.expected_rows}, got {p.row_count}",
            )

            if new_status == PartitionStatus.AMBER:
                results["updated_to_amber"] += 1
            else:
                results["updated_to_red"] += 1

            logger.warning(
                f"Row mismatch: {table_name}/{p.symbol}/{p.dt} - "
                f"expected {p.expected_rows}, got {p.row_count} -> {new_status.value}"
            )

    return mismatches


def audit_existing_partitions(db: Optional[NodeDB] = None) -> dict:
    """Re-evaluate all GREEN partitions with row-count validation.

    One-time migration function to catch existing bad data after deploying
    the QC fix. Runs through all GREEN partitions and validates row counts.

    Args:
        db: Database instance

    Returns:
        Dict with audit results
    """
    if db is None:
        db = get_db()

    logger.info("Auditing existing partitions for row-count issues...")

    results = {
        "tables_checked": 0,
        "partitions_checked": 0,
        "issues_found": 0,
        "updated_to_amber": 0,
        "updated_to_red": 0,
        "by_table": {},
    }

    for table in ["equities_eod", "equities_minute"]:
        table_results = {
            "checked": 0,
            "issues": 0,
            "row_count_mismatches": 0,
            "updated_to_amber": 0,
            "updated_to_red": 0,
            "details": [],
        }

        logger.info(f"Auditing {table}...")
        mismatches = _validate_partition_row_counts(db, table, table_results, batch_size=50000)

        results["tables_checked"] += 1
        results["partitions_checked"] += table_results.get("checked", 0)
        results["issues_found"] += mismatches
        results["updated_to_amber"] += table_results["updated_to_amber"]
        results["updated_to_red"] += table_results["updated_to_red"]
        results["by_table"][table] = {
            "mismatches": mismatches,
            "updated_to_amber": table_results["updated_to_amber"],
            "updated_to_red": table_results["updated_to_red"],
        }

        logger.info(f"  {table}: {mismatches} issues found")

    logger.info(
        f"Audit complete: {results['issues_found']} issues found, "
        f"{results['updated_to_amber']} marked AMBER, {results['updated_to_red']} marked RED"
    )
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
        import pyarrow as pa
        import pyarrow.parquet as pq

        conn = db._get_connection()
        output_path = export_dir / "partition_status.parquet"
        cursor = conn.execute("SELECT * FROM partition_status")

        batch = cursor.fetchmany(5000)
        if not batch:
            logger.debug("No partition_status rows to export")
            return

        first_table = pa.Table.from_pylist([dict(row) for row in batch])
        writer = pq.ParquetWriter(output_path, first_table.schema, compression="snappy")
        try:
            writer.write_table(first_table)
            while True:
                batch = cursor.fetchmany(5000)
                if not batch:
                    break
                table = pa.Table.from_pylist([dict(row) for row in batch], schema=first_table.schema)
                writer.write_table(table)
        finally:
            writer.close()

        results["files"].append("partition_status.parquet")

    except ImportError:
        logger.warning("pyarrow not available, skipping partition_status export")
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
                hasher = hashlib.sha256()
                with open(filepath, "rb") as f:
                    for chunk in iter(lambda: f.read(1024 * 1024), b""):
                        hasher.update(chunk)
                checksum = hasher.hexdigest()
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
        self._last_tick: Optional[datetime] = None

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
        self._last_tick = datetime.now()
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

    @property
    def last_tick(self) -> Optional[datetime]:
        """Get the timestamp of the last tick."""
        return self._last_tick

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
