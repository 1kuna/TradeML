#!/usr/bin/env python3
"""
Migration script to backfill partition_status from existing parquet files.

This script scans existing data files and populates the partition_status table
so the GAP audit knows what data already exists. Run this ONCE after deploying
the QC fix to avoid re-fetching data that already exists.

Usage:
    cd ~/Desktop/TradeML
    source venv/bin/activate
    python scripts/backfill_partition_status.py

Or with options:
    python scripts/backfill_partition_status.py --dry-run  # Preview only
    python scripts/backfill_partition_status.py --dataset equities_minute  # Single dataset
"""

import argparse
import os
import sys
from datetime import date
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger


def scan_raw_partitions(data_root: str, vendor: str, table: str) -> list[tuple[date, Path]]:
    """
    Scan raw partition directories for a vendor/table combination.

    Returns:
        List of (date, parquet_path) tuples
    """
    base_path = Path(data_root) / "data_layer" / "raw" / vendor / table

    if not base_path.exists():
        logger.warning(f"Path does not exist: {base_path}")
        return []

    partitions = []

    # Look for date=YYYY-MM-DD directories
    for date_dir in base_path.glob("date=*"):
        if not date_dir.is_dir():
            continue

        # Extract date from directory name
        date_str = date_dir.name.replace("date=", "")
        try:
            dt = date.fromisoformat(date_str)
        except ValueError:
            logger.warning(f"Invalid date format in directory: {date_dir.name}")
            continue

        # Find parquet file(s) in the directory
        parquet_files = list(date_dir.glob("*.parquet"))
        if not parquet_files:
            # Check for nested partition structure
            parquet_files = list(date_dir.glob("**/*.parquet"))

        if parquet_files:
            # Use the first parquet file found
            partitions.append((dt, parquet_files[0]))

    return partitions


def count_rows_by_symbol(parquet_path: Path, dataset: str) -> dict[str, dict[date, int]]:
    """
    Read a parquet file and count rows per symbol per date.

    Returns:
        Dict of {symbol: {date: row_count}}
    """
    try:
        import polars as pl
    except ImportError:
        import pandas as pd
        # Fallback to pandas
        df = pd.read_parquet(parquet_path)

        # Determine symbol and date columns
        symbol_col = "symbol" if "symbol" in df.columns else "ticker"
        date_col = "date" if "date" in df.columns else "timestamp"

        if symbol_col not in df.columns:
            logger.warning(f"No symbol column found in {parquet_path}")
            return {}

        # Group and count
        result = {}
        if date_col in df.columns:
            # Convert date column if needed
            if hasattr(df[date_col].iloc[0], 'date'):
                df["_date"] = df[date_col].apply(lambda x: x.date() if hasattr(x, 'date') else x)
            else:
                df["_date"] = pd.to_datetime(df[date_col]).dt.date

            grouped = df.groupby([symbol_col, "_date"]).size()
            for (sym, dt), count in grouped.items():
                if sym not in result:
                    result[sym] = {}
                result[sym][dt] = count
        else:
            # No date column - use file path date
            grouped = df.groupby(symbol_col).size()
            for sym, count in grouped.items():
                result[sym] = {None: count}  # Will use partition date

        return result

    # Polars version (preferred - faster)
    df = pl.read_parquet(parquet_path)

    # Determine symbol and date columns
    symbol_col = "symbol" if "symbol" in df.columns else "ticker"
    date_col = "date" if "date" in df.columns else "timestamp"

    if symbol_col not in df.columns:
        logger.warning(f"No symbol column found in {parquet_path}")
        return {}

    result = {}

    if date_col in df.columns:
        # Extract date if datetime
        if df[date_col].dtype == pl.Datetime:
            df = df.with_columns(pl.col(date_col).dt.date().alias("_date"))
        else:
            df = df.with_columns(pl.col(date_col).cast(pl.Date).alias("_date"))

        # Group and count
        counts = df.group_by([symbol_col, "_date"]).agg(pl.len().alias("count"))

        for row in counts.iter_rows(named=True):
            sym = row[symbol_col]
            dt = row["_date"]
            count = row["count"]

            if sym not in result:
                result[sym] = {}
            result[sym][dt] = count
    else:
        # No date column - count by symbol only
        counts = df.group_by(symbol_col).agg(pl.len().alias("count"))
        for row in counts.iter_rows(named=True):
            result[row[symbol_col]] = {None: row["count"]}

    return result


def get_expected_rows_per_day(dataset: str) -> int:
    """Get expected rows per day for a dataset."""
    from data_node.stages import get_expected_rows
    return get_expected_rows(dataset, num_days=1)


def backfill_partition_status(
    data_root: str,
    datasets: list[str],
    dry_run: bool = False,
) -> dict:
    """
    Backfill partition_status from existing parquet files.

    Args:
        data_root: Path to DATA_ROOT
        datasets: List of datasets to process
        dry_run: If True, only preview changes

    Returns:
        Dict with stats
    """
    from data_node.db import get_db, PartitionStatus
    from data_node.qc import calculate_partition_status
    from data_node.stages import get_qc_thresholds

    db = get_db()
    thresholds = get_qc_thresholds()

    # Map dataset names to table names
    dataset_to_tables = {
        "equities_eod": ["equities_bars"],
        "equities_minute": ["equities_bars_minute"],
    }

    # Map table names to dataset names
    table_to_dataset = {
        "equities_bars": "equities_eod",
        "equities_bars_minute": "equities_minute",
    }

    stats = {
        "files_scanned": 0,
        "partitions_found": 0,
        "partitions_created": 0,
        "partitions_skipped": 0,
        "errors": 0,
    }

    # Known vendors that write data
    vendors = ["alpaca", "massive", "finnhub"]

    for dataset in datasets:
        tables = dataset_to_tables.get(dataset, [dataset])
        expected_per_day = get_expected_rows_per_day(dataset)

        logger.info(f"Processing dataset: {dataset} (expected_rows={expected_per_day})")

        for vendor in vendors:
            for table in tables:
                logger.info(f"  Scanning {vendor}/{table}...")

                partitions = scan_raw_partitions(data_root, vendor, table)

                if not partitions:
                    logger.info(f"    No partitions found")
                    continue

                logger.info(f"    Found {len(partitions)} date partitions")

                for partition_date, parquet_path in partitions:
                    stats["files_scanned"] += 1

                    try:
                        rows_by_symbol = count_rows_by_symbol(parquet_path, dataset)

                        for symbol, date_counts in rows_by_symbol.items():
                            for dt, row_count in date_counts.items():
                                # Use partition date if no date in data
                                actual_date = dt if dt else partition_date

                                # Calculate status based on row count
                                status = calculate_partition_status(
                                    row_count=row_count,
                                    expected_rows=expected_per_day,
                                    qc_code="OK",
                                    thresholds=thresholds,
                                )

                                qc_code = "OK" if status == PartitionStatus.GREEN else "ROW_COUNT_MISMATCH"

                                stats["partitions_found"] += 1

                                if dry_run:
                                    # Just log what would be done
                                    if stats["partitions_found"] <= 20:
                                        logger.info(
                                            f"      [DRY RUN] Would create: "
                                            f"{dataset}/{symbol}/{actual_date} "
                                            f"rows={row_count} status={status.value}"
                                        )
                                    elif stats["partitions_found"] == 21:
                                        logger.info("      ... (truncated)")
                                else:
                                    # Actually insert
                                    db.upsert_partition_status(
                                        source_name=vendor,
                                        table_name=dataset,
                                        symbol=symbol,
                                        dt=actual_date.isoformat() if hasattr(actual_date, 'isoformat') else str(actual_date),
                                        status=status,
                                        qc_score=1.0 if status == PartitionStatus.GREEN else 0.5,
                                        row_count=row_count,
                                        expected_rows=expected_per_day,
                                        qc_code=qc_code,
                                    )
                                    stats["partitions_created"] += 1

                    except Exception as e:
                        logger.error(f"    Error processing {parquet_path}: {e}")
                        stats["errors"] += 1
                        continue

    return stats


def clear_stale_tasks(dry_run: bool = False) -> int:
    """
    Clear PENDING tasks that now have GREEN partition_status.

    After backfilling partition_status, many GAP tasks will be redundant
    because the data already exists. This function marks them as DONE.

    Returns:
        Number of tasks cleared
    """
    from data_node.db import get_db, TaskStatus, PartitionStatus

    db = get_db()
    conn = db._get_connection()

    # Find PENDING tasks where partition is now GREEN
    query = """
        SELECT t.id, t.dataset, t.symbol, t.start_date
        FROM task_queue t
        JOIN partition_status p
            ON t.dataset = p.table_name
            AND t.symbol = p.symbol
            AND t.start_date = p.dt
        WHERE t.status = 'PENDING'
        AND p.status = 'GREEN'
    """

    redundant = conn.execute(query).fetchall()

    if not redundant:
        logger.info("No redundant tasks found")
        return 0

    logger.info(f"Found {len(redundant)} redundant PENDING tasks")

    if dry_run:
        for task_id, dataset, symbol, dt in redundant[:10]:
            logger.info(f"  [DRY RUN] Would mark done: task {task_id} ({dataset}/{symbol}/{dt})")
        if len(redundant) > 10:
            logger.info(f"  ... and {len(redundant) - 10} more")
        return 0

    # Mark them as DONE
    task_ids = [r[0] for r in redundant]

    # Batch update
    for i in range(0, len(task_ids), 500):
        batch = task_ids[i:i+500]
        placeholders = ",".join("?" * len(batch))
        conn.execute(f"""
            UPDATE task_queue
            SET status = 'DONE', completed_at = datetime('now')
            WHERE id IN ({placeholders})
        """, batch)

    conn.commit()
    logger.info(f"Marked {len(task_ids)} redundant tasks as DONE")

    return len(task_ids)


def main():
    parser = argparse.ArgumentParser(
        description="Backfill partition_status from existing parquet files"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without making them",
    )
    parser.add_argument(
        "--dataset",
        choices=["equities_eod", "equities_minute", "all"],
        default="all",
        help="Dataset to process (default: all)",
    )
    parser.add_argument(
        "--clear-tasks",
        action="store_true",
        help="Also clear redundant PENDING tasks after backfilling",
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get("DATA_ROOT", "."),
        help="Path to DATA_ROOT (default: from env or current dir)",
    )

    args = parser.parse_args()

    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")

    # Determine datasets
    if args.dataset == "all":
        datasets = ["equities_eod", "equities_minute"]
    else:
        datasets = [args.dataset]

    logger.info(f"DATA_ROOT: {args.data_root}")
    logger.info(f"Datasets: {datasets}")
    logger.info(f"Dry run: {args.dry_run}")

    # Run backfill
    stats = backfill_partition_status(
        data_root=args.data_root,
        datasets=datasets,
        dry_run=args.dry_run,
    )

    logger.info("=" * 50)
    logger.info("Backfill complete!")
    logger.info(f"  Files scanned: {stats['files_scanned']}")
    logger.info(f"  Partitions found: {stats['partitions_found']}")
    logger.info(f"  Partitions created: {stats['partitions_created']}")
    logger.info(f"  Errors: {stats['errors']}")

    # Optionally clear redundant tasks
    if args.clear_tasks:
        logger.info("")
        logger.info("Clearing redundant tasks...")
        cleared = clear_stale_tasks(dry_run=args.dry_run)
        logger.info(f"  Tasks cleared: {cleared}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
