"""Unit tests for partition status persistence (data_layer/qc/)."""

import os
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest


@pytest.fixture
def temp_data_dir(tmp_path, monkeypatch):
    """Set up temporary data directory."""
    data_dir = tmp_path / "data_layer" / "qc"
    data_dir.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    return data_dir


def test_load_partition_status_empty(temp_data_dir):
    """Test loading returns empty DataFrame when no file exists."""
    from data_layer.qc.partition_status import load_partition_status

    result = load_partition_status()

    assert isinstance(result, pd.DataFrame)
    # Should return empty or have expected columns


def test_save_and_load_partition_status(temp_data_dir):
    """Test save/load round-trip."""
    from data_layer.qc.partition_status import save_partition_status, load_partition_status

    # Create test data with expected columns
    test_data = pd.DataFrame({
        "source_name": ["alpaca", "alpaca", "finnhub"],
        "table_name": ["equities_eod", "equities_eod", "options_chains"],
        "symbol": ["AAPL", "MSFT", "SPY"],
        "dt": [date(2024, 1, 15), date(2024, 1, 15), date(2024, 1, 15)],
        "status": ["GREEN", "AMBER", "RED"],
        "row_count": [252, 100, 0],
        "notes": ["", "partial data", "missing"],
    })

    # Save
    save_partition_status(test_data)

    # Load and verify
    loaded = load_partition_status()

    assert len(loaded) == 3
    assert set(loaded["status"]) == {"GREEN", "AMBER", "RED"}
    assert "AAPL" in loaded["symbol"].values


def test_get_status_existing_partition(temp_data_dir):
    """Test getting status for existing partition."""
    from data_layer.qc.partition_status import save_partition_status, get_status, PartitionStatus

    test_data = pd.DataFrame({
        "source_name": ["alpaca"],
        "table_name": ["equities_eod"],
        "symbol": ["AAPL"],
        "dt": [date(2024, 1, 15)],
        "status": ["GREEN"],
        "row_count": [252],
        "notes": [""],
    })
    save_partition_status(test_data)

    status = get_status(
        source_name="alpaca",
        table_name="equities_eod",
        symbol="AAPL",
        dt=date(2024, 1, 15)
    )

    assert status == PartitionStatus.GREEN


def test_get_status_nonexistent_partition(temp_data_dir):
    """Test getting status for non-existent partition."""
    from data_layer.qc.partition_status import get_status

    status = get_status(
        source_name="alpaca",
        table_name="equities_eod",
        symbol="UNKNOWN",
        dt=date(2024, 1, 15)
    )

    assert status is None


def test_get_green_coverage_all_green(temp_data_dir):
    """Test GREEN coverage calculation with all GREEN."""
    from data_layer.qc.partition_status import save_partition_status, get_green_coverage

    start = date(2024, 1, 10)
    end = date(2024, 1, 15)

    test_data = pd.DataFrame({
        "source_name": ["alpaca"] * 3,
        "table_name": ["equities_eod"] * 3,
        "symbol": ["AAPL", "MSFT", "GOOGL"],
        "dt": [date(2024, 1, 15)] * 3,
        "status": ["GREEN", "GREEN", "GREEN"],
        "row_count": [252, 252, 252],
        "notes": [""] * 3,
    })
    save_partition_status(test_data)

    coverage, counts = get_green_coverage("equities_eod", start, end)

    assert coverage == 1.0
    assert counts["GREEN"] == 3


def test_get_green_coverage_mixed(temp_data_dir):
    """Test GREEN coverage calculation with mixed status."""
    from data_layer.qc.partition_status import save_partition_status, get_green_coverage

    start = date(2024, 1, 10)
    end = date(2024, 1, 15)

    test_data = pd.DataFrame({
        "source_name": ["alpaca"] * 4,
        "table_name": ["equities_eod"] * 4,
        "symbol": ["AAPL", "MSFT", "GOOGL", "AMZN"],
        "dt": [date(2024, 1, 15)] * 4,
        "status": ["GREEN", "GREEN", "AMBER", "RED"],
        "row_count": [252, 252, 100, 0],
        "notes": ["", "", "partial", "missing"],
    })
    save_partition_status(test_data)

    coverage, counts = get_green_coverage("equities_eod", start, end)

    assert coverage == 0.5  # 2 GREEN out of 4
    assert counts["GREEN"] == 2
    assert counts["AMBER"] == 1
    assert counts["RED"] == 1


def test_get_green_coverage_with_universe_filter(temp_data_dir):
    """Test GREEN coverage with universe filter."""
    from data_layer.qc.partition_status import save_partition_status, get_green_coverage

    start = date(2024, 1, 10)
    end = date(2024, 1, 15)

    test_data = pd.DataFrame({
        "source_name": ["alpaca"] * 4,
        "table_name": ["equities_eod"] * 4,
        "symbol": ["AAPL", "MSFT", "GOOGL", "AMZN"],
        "dt": [date(2024, 1, 15)] * 4,
        "status": ["GREEN", "RED", "GREEN", "RED"],
        "row_count": [252, 0, 252, 0],
        "notes": ["", "missing", "", "missing"],
    })
    save_partition_status(test_data)

    # Filter to only AAPL and GOOGL (both GREEN)
    coverage, counts = get_green_coverage(
        "equities_eod", start, end,
        universe=["AAPL", "GOOGL"]
    )

    assert coverage == 1.0
    assert counts["GREEN"] == 2


def test_get_green_coverage_unknown_table(temp_data_dir):
    """Test GREEN coverage for unknown table returns 0."""
    from data_layer.qc.partition_status import get_green_coverage

    start = date(2024, 1, 10)
    end = date(2024, 1, 15)

    coverage, counts = get_green_coverage("unknown_table", start, end)

    assert coverage == 0.0


def test_update_partition_status(temp_data_dir):
    """Test updating existing partition status."""
    from data_layer.qc.partition_status import save_partition_status, load_partition_status, update_partition_status

    # Initial data
    initial_data = pd.DataFrame({
        "source_name": ["alpaca"],
        "table_name": ["equities_eod"],
        "symbol": ["AAPL"],
        "dt": [date(2024, 1, 15)],
        "status": ["AMBER"],
        "row_count": [100],
        "notes": ["partial"],
    })
    save_partition_status(initial_data)

    # Update to GREEN
    update_partition_status([{
        "source_name": "alpaca",
        "table_name": "equities_eod",
        "symbol": "AAPL",
        "dt": date(2024, 1, 15),
        "status": "GREEN",
        "row_count": 252,
        "notes": "",
    }])

    # Verify update
    loaded = load_partition_status()
    aapl_row = loaded[loaded["symbol"] == "AAPL"].iloc[0]
    assert aapl_row["status"] == "GREEN"
    assert aapl_row["row_count"] == 252


def test_get_gaps(temp_data_dir):
    """Test getting gaps (RED/AMBER partitions)."""
    from data_layer.qc.partition_status import save_partition_status, get_gaps, PartitionStatus

    start = date(2024, 1, 10)
    end = date(2024, 1, 15)

    test_data = pd.DataFrame({
        "source_name": ["alpaca"] * 4,
        "table_name": ["equities_eod"] * 4,
        "symbol": ["AAPL", "MSFT", "GOOGL", "AMZN"],
        "dt": [date(2024, 1, 15)] * 4,
        "status": ["GREEN", "AMBER", "GREEN", "RED"],
        "row_count": [252, 100, 252, 0],
        "notes": ["", "partial", "", "missing"],
    })
    save_partition_status(test_data)

    gaps = get_gaps("equities_eod", start, end, statuses=[PartitionStatus.RED, PartitionStatus.AMBER])

    # Should return MSFT and AMZN
    assert len(gaps) == 2
    gap_symbols = set(gaps["symbol"])
    assert "MSFT" in gap_symbols
    assert "AMZN" in gap_symbols


def test_partition_status_enum():
    """Test PartitionStatus enum values."""
    from data_layer.qc.partition_status import PartitionStatus

    assert PartitionStatus.GREEN.value == "GREEN"
    assert PartitionStatus.AMBER.value == "AMBER"
    assert PartitionStatus.RED.value == "RED"

    # Test from string
    assert PartitionStatus("GREEN") == PartitionStatus.GREEN
