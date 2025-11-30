"""
Unit tests for delistings reference data manager.
"""

import pytest
from datetime import date
from pathlib import Path
import tempfile

import pandas as pd

from data_layer.reference.delistings import DelistingsManager


@pytest.fixture
def sample_delistings():
    """Create sample delistings data for testing."""
    return pd.DataFrame({
        "symbol": ["DLIST1", "DLIST2", "DLIST3", "ACTIVE1"],
        "delist_date": [
            date(2020, 1, 15),
            date(2021, 6, 30),
            date(2022, 3, 10),
            date(2025, 12, 31),  # Future date - effectively active
        ],
        "reason": ["M&A", "BANKRUPTCY", "VOLUNTARY", "UNKNOWN"],
        "source_name": ["fmp", "alpha_vantage", "fmp", "fmp"],
        "source_priority": [1, 2, 1, 1],
    })


@pytest.fixture
def temp_delistings_file(sample_delistings, tmp_path):
    """Create temporary parquet file with sample data."""
    path = tmp_path / "delistings" / "data.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    sample_delistings.to_parquet(path, index=False)
    return path


class TestDelistingsManager:
    """Tests for DelistingsManager class."""

    def test_is_delisted_returns_true_for_delisted_symbol(self, temp_delistings_file):
        """Delisted symbol should return True after delist date."""
        manager = DelistingsManager(temp_delistings_file)

        # DLIST1 delisted on 2020-01-15
        assert manager.is_delisted("DLIST1", date(2020, 1, 15)) is True
        assert manager.is_delisted("DLIST1", date(2020, 6, 1)) is True
        assert manager.is_delisted("DLIST1", date(2025, 1, 1)) is True

    def test_is_delisted_returns_false_before_delist_date(self, temp_delistings_file):
        """Symbol should not be considered delisted before its delist date."""
        manager = DelistingsManager(temp_delistings_file)

        # DLIST1 delisted on 2020-01-15 - should be active before that
        assert manager.is_delisted("DLIST1", date(2020, 1, 14)) is False
        assert manager.is_delisted("DLIST1", date(2019, 12, 31)) is False

    def test_is_delisted_returns_false_for_active_symbol(self, temp_delistings_file):
        """Symbol not in delistings should return False."""
        manager = DelistingsManager(temp_delistings_file)

        # Unknown symbol
        assert manager.is_delisted("AAPL", date(2025, 1, 1)) is False

        # ACTIVE1 has future delist date
        assert manager.is_delisted("ACTIVE1", date(2024, 1, 1)) is False

    def test_get_delist_date_returns_correct_date(self, temp_delistings_file):
        """Should return the correct delist date for a symbol."""
        manager = DelistingsManager(temp_delistings_file)

        assert manager.get_delist_date("DLIST1") == date(2020, 1, 15)
        assert manager.get_delist_date("DLIST2") == date(2021, 6, 30)

    def test_get_delist_date_returns_none_for_unknown(self, temp_delistings_file):
        """Should return None for unknown symbols."""
        manager = DelistingsManager(temp_delistings_file)

        assert manager.get_delist_date("UNKNOWN") is None

    def test_get_delisted_symbols_returns_correct_set(self, temp_delistings_file):
        """Should return set of symbols delisted by the given date."""
        manager = DelistingsManager(temp_delistings_file)

        # Before any delistings
        assert manager.get_delisted_symbols(date(2019, 1, 1)) == set()

        # After DLIST1 only
        assert manager.get_delisted_symbols(date(2020, 6, 1)) == {"DLIST1"}

        # After DLIST1 and DLIST2
        assert manager.get_delisted_symbols(date(2022, 1, 1)) == {"DLIST1", "DLIST2"}

        # After all three
        assert manager.get_delisted_symbols(date(2023, 1, 1)) == {"DLIST1", "DLIST2", "DLIST3"}

    def test_get_active_symbols_filters_correctly(self, temp_delistings_file):
        """Should filter out delisted symbols from a list."""
        manager = DelistingsManager(temp_delistings_file)

        all_symbols = ["DLIST1", "DLIST2", "DLIST3", "AAPL", "MSFT"]

        # Before any delistings - all active
        active = manager.get_active_symbols(all_symbols, date(2019, 1, 1))
        assert set(active) == set(all_symbols)

        # After DLIST1 delisted
        active = manager.get_active_symbols(all_symbols, date(2020, 6, 1))
        assert "DLIST1" not in active
        assert set(active) == {"DLIST2", "DLIST3", "AAPL", "MSFT"}

        # After all delistings
        active = manager.get_active_symbols(all_symbols, date(2023, 1, 1))
        assert set(active) == {"AAPL", "MSFT"}

    def test_empty_file_returns_safe_defaults(self, tmp_path):
        """Should handle missing file gracefully."""
        nonexistent = tmp_path / "missing" / "data.parquet"
        manager = DelistingsManager(nonexistent)

        assert manager.is_delisted("AAPL", date(2024, 1, 1)) is False
        assert manager.get_delist_date("AAPL") is None
        assert manager.get_delisted_symbols(date(2024, 1, 1)) == set()
        assert manager.get_active_symbols(["AAPL", "MSFT"], date(2024, 1, 1)) == ["AAPL", "MSFT"]


class TestDelistingsMerge:
    """Tests for merging delistings from multiple sources."""

    def test_merge_prefers_fmp_over_av(self):
        """FMP should have priority over Alpha Vantage for same symbol."""
        fmp_df = pd.DataFrame({
            "symbol": ["TEST1"],
            "delist_date": [date(2020, 1, 1)],
            "reason": ["M&A"],
            "source_name": ["fmp"],
        })

        av_df = pd.DataFrame({
            "symbol": ["TEST1"],
            "delist_date": [date(2020, 2, 1)],  # Different date
            "reason": ["DELISTED"],
            "source_name": ["alpha_vantage"],
        })

        merged = DelistingsManager.merge_sources(fmp_df, av_df)

        # Should have only one entry for TEST1
        assert len(merged) == 1

        # Should use FMP data
        row = merged[merged["symbol"] == "TEST1"].iloc[0]
        assert row["delist_date"] == date(2020, 1, 1)
        assert row["reason"] == "M&A"
        assert row["source_name"] == "fmp"

    def test_merge_combines_unique_symbols(self):
        """Should keep unique symbols from both sources."""
        fmp_df = pd.DataFrame({
            "symbol": ["FMP_ONLY"],
            "delist_date": [date(2020, 1, 1)],
            "reason": ["M&A"],
            "source_name": ["fmp"],
        })

        av_df = pd.DataFrame({
            "symbol": ["AV_ONLY"],
            "delist_date": [date(2021, 1, 1)],
            "reason": ["DELISTED"],
            "source_name": ["alpha_vantage"],
        })

        merged = DelistingsManager.merge_sources(fmp_df, av_df)

        assert len(merged) == 2
        assert set(merged["symbol"]) == {"FMP_ONLY", "AV_ONLY"}

    def test_merge_handles_empty_sources(self):
        """Should handle empty DataFrames gracefully."""
        fmp_df = pd.DataFrame({
            "symbol": ["TEST1"],
            "delist_date": [date(2020, 1, 1)],
            "reason": ["M&A"],
            "source_name": ["fmp"],
        })

        empty_df = pd.DataFrame()

        # FMP only
        merged = DelistingsManager.merge_sources(fmp_df, empty_df)
        assert len(merged) == 1

        # AV only
        merged = DelistingsManager.merge_sources(empty_df, fmp_df)
        assert len(merged) == 1

        # Both empty
        merged = DelistingsManager.merge_sources(empty_df, empty_df)
        assert len(merged) == 0


class TestSurvivorshipBias:
    """Tests specific to survivorship bias elimination."""

    def test_delisted_included_before_delist_date(self, temp_delistings_file):
        """
        For historical training, delisted symbols should be included
        when building universes before their delist date.
        """
        manager = DelistingsManager(temp_delistings_file)

        # DLIST2 delisted on 2021-06-30
        # For training data from 2020, it should be included
        as_of = date(2020, 12, 31)

        # Should NOT be in delisted set
        assert "DLIST2" not in manager.get_delisted_symbols(as_of)

        # Should remain in active list
        symbols = ["DLIST2", "AAPL"]
        active = manager.get_active_symbols(symbols, as_of)
        assert "DLIST2" in active

    def test_pit_consistency_across_dates(self, temp_delistings_file):
        """
        PIT lookups should be consistent - same date should always
        return the same result regardless of when queried.
        """
        manager1 = DelistingsManager(temp_delistings_file)
        manager2 = DelistingsManager(temp_delistings_file)

        test_date = date(2021, 1, 15)

        # Both managers should return identical results
        assert manager1.get_delisted_symbols(test_date) == manager2.get_delisted_symbols(test_date)
        assert manager1.is_delisted("DLIST1", test_date) == manager2.is_delisted("DLIST1", test_date)
