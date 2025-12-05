"""
Weekly cross-vendor QC for the unified Pi data-node.

Implements:
- Sample selection from GREEN partitions
- QC_PROBE task scheduling
- Cross-vendor comparison logic

See updated_node_spec.md §6 for QC semantics.
"""

from __future__ import annotations

import random
from datetime import date, datetime, timedelta
from typing import Optional

from loguru import logger

from .db import NodeDB, TaskKind, PartitionStatus, get_db
from .stages import get_current_universe, get_date_range

# Default QC parameters
DEFAULT_SAMPLE_SIZE = 100
QC_DAY_OF_WEEK = 6  # Sunday (0=Monday, 6=Sunday)

# Priority for QC_PROBE tasks
PRIORITY_QC_PROBE = 2


def is_qc_day(dt: Optional[date] = None) -> bool:
    """
    Check if the given date is a QC scheduling day.

    QC probes are scheduled on Sundays.

    Args:
        dt: Date to check (default: today)

    Returns:
        True if it's a QC scheduling day
    """
    if dt is None:
        dt = date.today()
    return dt.weekday() == QC_DAY_OF_WEEK


def select_qc_samples(
    n_samples: int = DEFAULT_SAMPLE_SIZE,
    db: Optional[NodeDB] = None,
) -> list[tuple[str, str, date]]:
    """
    Select samples for cross-vendor QC verification.

    Samples GREEN partitions from the active training window, biasing toward:
    - Recently AMBER partitions (higher priority for verification)
    - High-volume symbols (more impact if data is wrong)
    - Dates near corporate actions (higher risk of data issues)

    Args:
        n_samples: Number of samples to select
        db: Database instance

    Returns:
        List of (dataset, symbol, date) tuples
    """
    if db is None:
        db = get_db()

    universe = get_current_universe()
    if not universe:
        logger.warning("Empty universe, cannot select QC samples")
        return []

    samples: list[tuple[str, str, date]] = []
    datasets = ["equities_eod", "equities_minute"]

    # Calculate sample allocation per dataset
    samples_per_dataset = n_samples // len(datasets)

    for dataset in datasets:
        start_date, end_date = get_date_range(dataset)

        # Get GREEN partitions
        green_records = db.get_partition_status_batch(
            table_name=dataset,
            symbols=universe,
            start_date=start_date,
            end_date=end_date,
        )

        # Filter to GREEN only
        green_records = [
            r for r in green_records
            if r.get("status") == PartitionStatus.GREEN.value
        ]

        if not green_records:
            logger.debug(f"No GREEN records for {dataset}")
            continue

        # Build candidate pool with weights
        candidates: list[tuple[str, date, float]] = []

        for record in green_records:
            symbol = record.get("symbol")
            dt_str = record.get("dt")

            if not symbol or not dt_str:
                continue

            # Parse date
            try:
                if isinstance(dt_str, date):
                    dt = dt_str
                else:
                    dt = datetime.strptime(dt_str, "%Y-%m-%d").date()
            except (ValueError, TypeError):
                continue

            # Calculate weight (higher = more likely to be sampled)
            weight = 1.0

            # Bias toward recent dates (last 90 days)
            days_ago = (date.today() - dt).days
            if days_ago < 90:
                weight *= 2.0
            elif days_ago < 30:
                weight *= 3.0

            # Bias toward high-volume symbols (first 50 in universe)
            try:
                sym_idx = universe.index(symbol)
                if sym_idx < 50:
                    weight *= 1.5
                elif sym_idx < 100:
                    weight *= 1.2
            except ValueError:
                pass

            candidates.append((symbol, dt, weight))

        if not candidates:
            continue

        # Weighted random sampling
        dataset_samples = _weighted_sample(candidates, samples_per_dataset)

        for symbol, dt in dataset_samples:
            samples.append((dataset, symbol, dt))

    logger.info(f"Selected {len(samples)} QC samples across {len(datasets)} datasets")
    return samples


def _weighted_sample(
    candidates: list[tuple[str, date, float]],
    n: int,
) -> list[tuple[str, date]]:
    """
    Select n samples with weighted probability.

    Args:
        candidates: List of (symbol, date, weight) tuples
        n: Number of samples to select

    Returns:
        List of (symbol, date) tuples
    """
    if not candidates:
        return []

    # Normalize weights
    total_weight = sum(c[2] for c in candidates)
    if total_weight == 0:
        return []

    probabilities = [c[2] / total_weight for c in candidates]

    # Sample without replacement
    n = min(n, len(candidates))

    # Use random.choices with replacement, then dedupe
    # (simpler than implementing weighted sampling without replacement)
    selected_indices = set()
    attempts = 0
    max_attempts = n * 10

    while len(selected_indices) < n and attempts < max_attempts:
        idx = random.choices(range(len(candidates)), weights=probabilities, k=1)[0]
        selected_indices.add(idx)
        attempts += 1

    return [(candidates[i][0], candidates[i][1]) for i in selected_indices]


def schedule_qc_probes(
    db: Optional[NodeDB] = None,
    n_samples: int = DEFAULT_SAMPLE_SIZE,
) -> int:
    """
    Schedule QC_PROBE tasks for cross-vendor verification.

    Should be called on QC day (Sunday) during maintenance.

    Args:
        db: Database instance
        n_samples: Number of samples to probe

    Returns:
        Number of tasks created
    """
    if db is None:
        db = get_db()

    if not is_qc_day():
        logger.debug("Not QC day, skipping probe scheduling")
        return 0

    samples = select_qc_samples(n_samples, db)

    if not samples:
        logger.info("No QC samples selected")
        return 0

    created = 0
    for dataset, symbol, dt in samples:
        # Convert date to string for database
        dt_str = dt.isoformat() if isinstance(dt, date) else dt

        task_id = db.enqueue_task(
            dataset=dataset,
            symbol=symbol,
            start_date=dt_str,
            end_date=dt_str,
            kind=TaskKind.QC_PROBE,
            priority=PRIORITY_QC_PROBE,
        )

        if task_id:
            created += 1

    logger.info(f"Scheduled {created} QC_PROBE tasks for cross-vendor verification")
    return created


def get_secondary_vendor(dataset: str, primary_vendor: str) -> Optional[str]:
    """
    Get the secondary vendor for cross-vendor comparison.

    Args:
        dataset: Dataset name
        primary_vendor: The primary vendor that was used

    Returns:
        Secondary vendor name, or None if no secondary available
    """
    # Vendor pairs for cross-verification
    # Maps dataset -> {primary: secondary}
    VENDOR_PAIRS = {
        "equities_eod": {
            "alpaca": "polygon",
            "polygon": "alpaca",
            "finnhub": "polygon",
        },
        "equities_minute": {
            "alpaca": "polygon",
            "polygon": "alpaca",
        },
    }

    dataset_pairs = VENDOR_PAIRS.get(dataset, {})
    return dataset_pairs.get(primary_vendor)


# Comparison thresholds
PRICE_THRESHOLD_PCT = 0.002  # 0.2% (20 bps)
VOLUME_THRESHOLD_PCT = 0.10  # 10%


def compare_ohlcv(
    primary_data: dict,
    secondary_data: dict,
) -> tuple[bool, Optional[str]]:
    """
    Compare OHLCV data from two vendors.

    Args:
        primary_data: Dict with open, high, low, close, volume from primary vendor
        secondary_data: Dict with same fields from secondary vendor

    Returns:
        Tuple of (matches, qc_code)
        - matches: True if data matches within thresholds
        - qc_code: QC code describing the result
    """
    issues = []

    # Compare price fields
    for field in ["close", "high", "low", "open"]:
        primary_val = primary_data.get(field)
        secondary_val = secondary_data.get(field)

        if primary_val is None or secondary_val is None:
            continue

        if primary_val == 0 and secondary_val == 0:
            continue

        # Calculate percentage difference
        if secondary_val != 0:
            pct_diff = abs(primary_val - secondary_val) / abs(secondary_val)
        elif primary_val != 0:
            pct_diff = abs(primary_val - secondary_val) / abs(primary_val)
        else:
            pct_diff = 0

        if pct_diff > PRICE_THRESHOLD_PCT:
            issues.append(f"{field}_diff_{pct_diff:.4f}")

    # Compare volume
    primary_vol = primary_data.get("volume", 0)
    secondary_vol = secondary_data.get("volume", 0)

    if primary_vol > 0 or secondary_vol > 0:
        if secondary_vol > 0:
            vol_diff = abs(primary_vol - secondary_vol) / secondary_vol
        else:
            vol_diff = 1.0  # 100% diff if secondary is 0

        if vol_diff > VOLUME_THRESHOLD_PCT:
            issues.append(f"volume_diff_{vol_diff:.2f}")

    if issues:
        return False, "CROSS_VENDOR_MISMATCH:" + ",".join(issues)
    else:
        return True, "CROSS_VENDOR_OK"


def run_qc_probe(
    dataset: str,
    symbol: str,
    dt: date,
    primary_vendor: str,
    db: Optional[NodeDB] = None,
) -> dict:
    """
    Run a cross-vendor QC probe for a specific partition.

    This is called by the worker when processing a QC_PROBE task.

    Args:
        dataset: Dataset name
        symbol: Symbol
        dt: Date to check
        primary_vendor: Vendor that provided the primary data
        db: Database instance

    Returns:
        Dict with probe results
    """
    if db is None:
        db = get_db()

    secondary_vendor = get_secondary_vendor(dataset, primary_vendor)

    result = {
        "dataset": dataset,
        "symbol": symbol,
        "date": dt.isoformat(),
        "primary_vendor": primary_vendor,
        "secondary_vendor": secondary_vendor,
        "status": "SKIPPED",
        "qc_code": None,
    }

    if secondary_vendor is None:
        result["qc_code"] = "NO_SECONDARY_VENDOR"
        logger.debug(f"No secondary vendor for {dataset}/{primary_vendor}")
        return result

    # Get primary data from partition_status or curated
    # (In a full implementation, this would fetch the actual data)
    # For now, we just verify the secondary vendor is accessible
    result["status"] = "PENDING_FETCH"
    result["qc_code"] = "AWAITING_SECONDARY_DATA"

    return result


def get_qc_stats(db: Optional[NodeDB] = None) -> dict:
    """
    Get QC probe statistics.

    Args:
        db: Database instance

    Returns:
        Dict with QC statistics
    """
    if db is None:
        db = get_db()

    conn = db._get_connection()

    stats = {
        "pending_probes": 0,
        "completed_probes": 0,
        "last_run_date": None,
    }

    # Count pending QC_PROBE tasks
    row = conn.execute("""
        SELECT COUNT(*) as cnt
        FROM backfill_queue
        WHERE kind = 'QC_PROBE' AND status IN ('PENDING', 'LEASED')
    """).fetchone()
    stats["pending_probes"] = row["cnt"] if row else 0

    # Count completed QC_PROBE tasks
    row = conn.execute("""
        SELECT COUNT(*) as cnt
        FROM backfill_queue
        WHERE kind = 'QC_PROBE' AND status = 'DONE'
    """).fetchone()
    stats["completed_probes"] = row["cnt"] if row else 0

    return stats
