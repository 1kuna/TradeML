"""
Unit tests for index membership reference data manager.

Tests the logic in data_layer/reference/index_membership.py:
- PIT-safe constituent lookups
- ADD/REMOVE event processing
- Historical membership reconstruction
"""

import pytest
from datetime import date
from pathlib import Path

import pandas as pd

from data_layer.reference.index_membership import IndexMembershipManager


@pytest.fixture
def sample_sp500_events():
    """Create sample S&P 500 membership events for testing."""
    return pd.DataFrame({
        "date": [
            date(2020, 1, 1),   # Initial adds
            date(2020, 1, 1),
            date(2020, 1, 1),
            date(2020, 6, 15),  # Mid-year add
            date(2020, 9, 1),   # Removal
            date(2021, 3, 1),   # Add after removal
        ],
        "index_name": ["SP500"] * 6,
        "symbol": ["AAPL", "MSFT", "GOOGL", "TSLA", "AAPL", "AAPL"],
        "action": ["ADD", "ADD", "ADD", "ADD", "REMOVE", "ADD"],
        "source_name": ["test"] * 6,
    })


@pytest.fixture
def temp_index_dir(sample_sp500_events, tmp_path):
    """Create temporary directory with sample index data."""
    index_dir = tmp_path / "index_membership"
    index_dir.mkdir(parents=True, exist_ok=True)

    # Save S&P 500 data
    sp500_path = index_dir / "sp500.parquet"
    sample_sp500_events.to_parquet(sp500_path, index=False)

    return index_dir


class TestIndexMembershipManager:
    """Tests for IndexMembershipManager class."""

    def test_get_constituents_returns_members_at_date(self, temp_index_dir):
        """Should return correct members at a specific date."""
        manager = IndexMembershipManager(temp_index_dir)

        # At 2020-01-01: AAPL, MSFT, GOOGL added
        members = manager.get_constituents("SP500", date(2020, 1, 1))
        assert members == {"AAPL", "MSFT", "GOOGL"}

        # At 2020-06-15: TSLA added
        members = manager.get_constituents("SP500", date(2020, 6, 15))
        assert members == {"AAPL", "MSFT", "GOOGL", "TSLA"}

    def test_get_constituents_handles_removals(self, temp_index_dir):
        """Should correctly handle REMOVE events."""
        manager = IndexMembershipManager(temp_index_dir)

        # Before AAPL removal (2020-09-01)
        members = manager.get_constituents("SP500", date(2020, 8, 31))
        assert "AAPL" in members

        # After AAPL removal
        members = manager.get_constituents("SP500", date(2020, 9, 1))
        assert "AAPL" not in members
        assert members == {"MSFT", "GOOGL", "TSLA"}

    def test_get_constituents_handles_readd(self, temp_index_dir):
        """Should correctly handle re-adding a removed symbol."""
        manager = IndexMembershipManager(temp_index_dir)

        # AAPL removed on 2020-09-01, re-added on 2021-03-01
        members = manager.get_constituents("SP500", date(2021, 2, 28))
        assert "AAPL" not in members

        members = manager.get_constituents("SP500", date(2021, 3, 1))
        assert "AAPL" in members

    def test_get_constituents_returns_empty_before_any_events(self, temp_index_dir):
        """Should return empty set before any events."""
        manager = IndexMembershipManager(temp_index_dir)

        members = manager.get_constituents("SP500", date(2019, 12, 31))
        assert members == set()

    def test_get_constituents_returns_empty_for_unknown_index(self, temp_index_dir):
        """Should return empty set for unknown index."""
        manager = IndexMembershipManager(temp_index_dir)

        members = manager.get_constituents("UNKNOWN", date(2020, 1, 1))
        assert members == set()

    def test_is_member_returns_correct_boolean(self, temp_index_dir):
        """Should return correct membership status."""
        manager = IndexMembershipManager(temp_index_dir)

        # AAPL is member on 2020-01-01
        assert manager.is_member("SP500", "AAPL", date(2020, 1, 1)) is True

        # TSLA is not member until 2020-06-15
        assert manager.is_member("SP500", "TSLA", date(2020, 6, 14)) is False
        assert manager.is_member("SP500", "TSLA", date(2020, 6, 15)) is True

        # Unknown symbol is not member
        assert manager.is_member("SP500", "UNKNOWN", date(2020, 1, 1)) is False

    def test_filter_by_index_filters_correctly(self, temp_index_dir):
        """Should filter symbols to only index members."""
        manager = IndexMembershipManager(temp_index_dir)

        all_symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "META"]

        # At 2020-03-01: Only AAPL, MSFT, GOOGL are members
        filtered = manager.filter_by_index(all_symbols, "SP500", date(2020, 3, 1))
        assert set(filtered) == {"AAPL", "MSFT", "GOOGL"}

        # At 2020-09-15: AAPL removed, TSLA added
        filtered = manager.filter_by_index(all_symbols, "SP500", date(2020, 9, 15))
        assert set(filtered) == {"MSFT", "GOOGL", "TSLA"}

    def test_get_membership_history(self, temp_index_dir):
        """Should return membership history for a symbol."""
        manager = IndexMembershipManager(temp_index_dir)

        # AAPL has: ADD (2020-01-01), REMOVE (2020-09-01), ADD (2021-03-01)
        history = manager.get_membership_history("SP500", "AAPL")

        assert len(history) == 3
        assert list(history["action"]) == ["ADD", "REMOVE", "ADD"]

    def test_get_membership_history_with_date_filter(self, temp_index_dir):
        """Should filter history by date range."""
        manager = IndexMembershipManager(temp_index_dir)

        # Only events in 2020
        history = manager.get_membership_history(
            "SP500", "AAPL",
            start_date=date(2020, 1, 1),
            end_date=date(2020, 12, 31)
        )

        assert len(history) == 2
        assert list(history["action"]) == ["ADD", "REMOVE"]


class TestIndexMembershipPIT:
    """Tests for PIT (point-in-time) correctness."""

    def test_pit_lookup_is_deterministic(self, temp_index_dir):
        """Same date should always return same result."""
        manager1 = IndexMembershipManager(temp_index_dir)
        manager2 = IndexMembershipManager(temp_index_dir)

        test_date = date(2020, 7, 1)

        assert manager1.get_constituents("SP500", test_date) == manager2.get_constituents("SP500", test_date)

    def test_future_date_returns_latest_known_state(self, temp_index_dir):
        """Future date should return state after all events."""
        manager = IndexMembershipManager(temp_index_dir)

        # Far future date
        members = manager.get_constituents("SP500", date(2030, 1, 1))

        # Should have: MSFT, GOOGL, TSLA, AAPL (re-added)
        assert members == {"AAPL", "MSFT", "GOOGL", "TSLA"}

    def test_events_applied_in_order(self, temp_index_dir):
        """Events should be applied in chronological order."""
        manager = IndexMembershipManager(temp_index_dir)

        # Check state progresses correctly through time
        states = []
        for d in [date(2020, 1, 1), date(2020, 6, 15), date(2020, 9, 1), date(2021, 3, 1)]:
            states.append(manager.get_constituents("SP500", d))

        # Verify progression
        assert len(states[0]) == 3  # Initial adds
        assert len(states[1]) == 4  # +TSLA
        assert len(states[2]) == 3  # -AAPL
        assert len(states[3]) == 4  # +AAPL back


class TestIndexMembershipBulkLoad:
    """Tests for bulk loading functionality."""

    def test_add_event_creates_file(self, tmp_path):
        """add_event should create file if it doesn't exist."""
        manager = IndexMembershipManager(tmp_path / "index_membership")

        manager.add_event("TEST", "AAPL", date(2020, 1, 1), "ADD", "test")

        path = tmp_path / "index_membership" / "test.parquet"
        assert path.exists()

        df = pd.read_parquet(path)
        assert len(df) == 1
        assert df.iloc[0]["symbol"] == "AAPL"

    def test_add_event_appends_to_existing(self, temp_index_dir):
        """add_event should append to existing data."""
        manager = IndexMembershipManager(temp_index_dir)

        initial_count = len(manager.get_all_events("SP500"))
        manager.add_event("SP500", "META", date(2022, 1, 1), "ADD", "test")

        # Invalidate cache and reload
        manager.invalidate_cache()
        new_count = len(manager.get_all_events("SP500"))

        assert new_count == initial_count + 1

    def test_bulk_load_replace(self, tmp_path):
        """bulk_load with replace=True should replace existing data."""
        manager = IndexMembershipManager(tmp_path / "index_membership")

        # First load
        events1 = pd.DataFrame({
            "date": [date(2020, 1, 1)],
            "symbol": ["AAPL"],
            "action": ["ADD"],
        })
        manager.bulk_load("TEST", events1, replace=True)

        # Second load with replace
        events2 = pd.DataFrame({
            "date": [date(2021, 1, 1)],
            "symbol": ["MSFT"],
            "action": ["ADD"],
        })
        manager.bulk_load("TEST", events2, replace=True)

        # Should only have second load
        all_events = manager.get_all_events("TEST")
        assert len(all_events) == 1
        assert all_events.iloc[0]["symbol"] == "MSFT"

    def test_bulk_load_append(self, tmp_path):
        """bulk_load with replace=False should append to existing data."""
        manager = IndexMembershipManager(tmp_path / "index_membership")

        events1 = pd.DataFrame({
            "date": [date(2020, 1, 1)],
            "symbol": ["AAPL"],
            "action": ["ADD"],
        })
        manager.bulk_load("TEST", events1, replace=True)

        events2 = pd.DataFrame({
            "date": [date(2021, 1, 1)],
            "symbol": ["MSFT"],
            "action": ["ADD"],
        })
        manager.bulk_load("TEST", events2, replace=False)

        # Should have both
        all_events = manager.get_all_events("TEST")
        assert len(all_events) == 2
        assert set(all_events["symbol"]) == {"AAPL", "MSFT"}


class TestEmptyIndexHandling:
    """Test handling of missing or empty index data."""

    def test_empty_index_returns_empty_set(self, tmp_path):
        """Missing index file should return empty set."""
        manager = IndexMembershipManager(tmp_path / "missing")

        members = manager.get_constituents("SP500", date(2020, 1, 1))
        assert members == set()

    def test_is_member_returns_false_for_empty_index(self, tmp_path):
        """Missing index should return False for any symbol."""
        manager = IndexMembershipManager(tmp_path / "missing")

        assert manager.is_member("SP500", "AAPL", date(2020, 1, 1)) is False

    def test_filter_returns_empty_for_empty_index(self, tmp_path):
        """Filter should return empty list when index has no data."""
        manager = IndexMembershipManager(tmp_path / "missing")

        filtered = manager.filter_by_index(["AAPL", "MSFT"], "SP500", date(2020, 1, 1))
        assert filtered == []
