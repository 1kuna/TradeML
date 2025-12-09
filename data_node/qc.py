"""
QC (Quality Control) functions for the unified Pi data-node.

Implements:
- Partition status calculation based on row count validation
- Row count validation logic with threshold support
- Half-day (early close) detection

See updated_node_spec.md §5 and backfill.yml qc_thresholds for configuration.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from loguru import logger

from .db import PartitionStatus
from .stages import get_expected_rows, get_qc_thresholds


def calculate_partition_status(
    row_count: int,
    expected_rows: int,
    qc_code: Optional[str] = None,
    thresholds: Optional[dict] = None,
) -> PartitionStatus:
    """Calculate partition status based on row count validation.

    Implements the validation logic:
    - NO_SESSION (holiday/weekend): always GREEN
    - No expectation defined (expected_rows=0): always GREEN
    - Missing data (row_count=0): RED
    - Half-day (45-90% of expected, if allow_halfdays): GREEN
    - Below low threshold: AMBER
    - Above high threshold: AMBER (suspicious overfill)
    - Within thresholds: GREEN

    Args:
        row_count: Actual rows fetched for this partition
        expected_rows: Expected rows from config (0 means no validation)
        qc_code: QC code from fetcher (NO_SESSION, OK, etc.)
        thresholds: Override thresholds dict, or use defaults from config

    Returns:
        PartitionStatus (GREEN, AMBER, or RED)
    """
    # Load thresholds if not provided
    if thresholds is None:
        thresholds = get_qc_thresholds()

    # Special cases that are always GREEN
    if qc_code == "NO_SESSION":
        # Holiday/weekend - no data expected
        return PartitionStatus.GREEN

    if expected_rows == 0:
        # No expectation defined - can't validate, assume OK
        return PartitionStatus.GREEN

    # Get thresholds with defaults
    low_ratio = thresholds.get("rows_low_ratio", 0.90)
    high_ratio = thresholds.get("rows_high_ratio", 1.20)
    allow_halfdays = thresholds.get("allow_halfdays", True)

    # Check for missing data
    if row_count == 0:
        return PartitionStatus.RED

    # Calculate ratio
    ratio = row_count / expected_rows

    # Check for half-day (early close) - ~50% of expected
    # Market half-days typically have 195-210 bars vs normal ~390
    if allow_halfdays and 0.45 <= ratio < low_ratio:
        # This looks like a half-day trading session
        logger.debug(
            f"Half-day detected: {row_count}/{expected_rows} = {ratio:.1%}"
        )
        return PartitionStatus.GREEN

    # Check thresholds
    if ratio < low_ratio:
        # Underfilled - needs refetch
        return PartitionStatus.AMBER

    if ratio > high_ratio:
        # Overfilled - suspicious, may be duplicate data
        return PartitionStatus.AMBER

    return PartitionStatus.GREEN


def get_qc_code_for_status(
    status: PartitionStatus,
    row_count: int,
    expected_rows: int,
    original_qc_code: Optional[str] = None,
) -> str:
    """Get an appropriate QC code for a partition status.

    Args:
        status: The calculated partition status
        row_count: Actual rows
        expected_rows: Expected rows
        original_qc_code: Original QC code from fetcher

    Returns:
        QC code string
    """
    if original_qc_code == "NO_SESSION":
        return "NO_SESSION"

    if status == PartitionStatus.GREEN:
        return original_qc_code or "OK"

    if status == PartitionStatus.RED:
        return "NO_DATA"

    # AMBER cases
    if expected_rows > 0 and row_count < expected_rows:
        return "ROW_COUNT_LOW"
    elif expected_rows > 0 and row_count > expected_rows:
        return "ROW_COUNT_HIGH"
    else:
        return "ROW_COUNT_MISMATCH"


def validate_partition_row_count(
    row_count: int,
    expected_rows: int,
    qc_code: Optional[str] = None,
    thresholds: Optional[dict] = None,
) -> tuple[PartitionStatus, str]:
    """Validate a partition's row count and return status + QC code.

    Convenience function that combines calculate_partition_status and
    get_qc_code_for_status.

    Args:
        row_count: Actual rows
        expected_rows: Expected rows (0 = no validation)
        qc_code: Original QC code from fetcher
        thresholds: Override thresholds dict

    Returns:
        Tuple of (PartitionStatus, qc_code)
    """
    status = calculate_partition_status(
        row_count=row_count,
        expected_rows=expected_rows,
        qc_code=qc_code,
        thresholds=thresholds,
    )

    new_qc_code = get_qc_code_for_status(
        status=status,
        row_count=row_count,
        expected_rows=expected_rows,
        original_qc_code=qc_code,
    )

    return status, new_qc_code
