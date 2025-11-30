"""
Unit tests for tick size regime reference data manager.

Tests the logic in data_layer/reference/tick_size_regime.py:
- PIT-safe tick size lookups (Reg NMS, Tick Pilot, sub-penny)
- Regime name resolution
- Tick constraint detection
"""

import pytest
from datetime import date
from pathlib import Path

import pandas as pd

from data_layer.reference.tick_size_regime import TickSizeRegime


@pytest.fixture
def sample_tick_rules():
    """Create sample tick size rules for testing."""
    return pd.DataFrame({
        "effective_date": [
            date(2007, 10, 1),   # Reg NMS market-wide
            date(2016, 10, 3),  # AAPL joins Tick Pilot
            date(2016, 10, 3),  # MSFT joins Tick Pilot
            date(2018, 10, 2),  # AAPL exits Tick Pilot (back to Reg NMS)
            date(2018, 10, 2),  # MSFT exits Tick Pilot (back to Reg NMS)
        ],
        "symbol": [None, "AAPL", "MSFT", "AAPL", "MSFT"],
        "tick_size": [0.01, 0.05, 0.05, 0.01, 0.01],
        "regime_name": ["REG_NMS", "TICK_PILOT", "TICK_PILOT", "REG_NMS", "REG_NMS"],
        "notes": [
            "Reg NMS minimum tick",
            "SEC Tick Pilot",
            "SEC Tick Pilot",
            "Pilot ended",
            "Pilot ended",
        ],
        "source_name": ["sec_reg_nms", "sec_tick_pilot", "sec_tick_pilot", "sec_tick_pilot", "sec_tick_pilot"],
    })


@pytest.fixture
def temp_tick_file(sample_tick_rules, tmp_path):
    """Create temporary parquet file with sample tick size rules."""
    path = tmp_path / "tick_size_regime" / "data.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    sample_tick_rules.to_parquet(path, index=False)
    return path


class TestTickSizeRegime:
    """Tests for TickSizeRegime class."""

    def test_get_tick_size_returns_default_for_standard_symbol(self, temp_tick_file):
        """Non-pilot symbol should return Reg NMS tick size."""
        regime = TickSizeRegime(temp_tick_file)

        # GOOGL is not in Tick Pilot
        tick = regime.get_tick_size("GOOGL", date(2017, 1, 1))
        assert tick == 0.01

    def test_get_tick_size_returns_pilot_tick_during_pilot(self, temp_tick_file):
        """Tick Pilot symbol should return $0.05 during pilot period."""
        regime = TickSizeRegime(temp_tick_file)

        # AAPL in Tick Pilot from 2016-10-03 to 2018-10-02
        tick = regime.get_tick_size("AAPL", date(2017, 6, 1))
        assert tick == 0.05

    def test_get_tick_size_returns_standard_after_pilot_ends(self, temp_tick_file):
        """Symbol should return Reg NMS tick after pilot ends."""
        regime = TickSizeRegime(temp_tick_file)

        # After 2018-10-02, AAPL back to Reg NMS
        tick = regime.get_tick_size("AAPL", date(2019, 1, 1))
        assert tick == 0.01

    def test_get_tick_size_sub_penny_for_low_price(self, temp_tick_file):
        """Price < $1.00 should return sub-penny tick size."""
        regime = TickSizeRegime(temp_tick_file)

        # Any symbol with price < $1.00 gets sub-penny
        tick = regime.get_tick_size("PENNY", date(2020, 1, 1), price=0.50)
        assert tick == 0.0001

    def test_get_tick_size_respects_pit_date(self, temp_tick_file):
        """Tick size should change based on as_of_date."""
        regime = TickSizeRegime(temp_tick_file)

        # Before Tick Pilot
        assert regime.get_tick_size("AAPL", date(2016, 9, 1)) == 0.01

        # During Tick Pilot
        assert regime.get_tick_size("AAPL", date(2017, 1, 1)) == 0.05

        # After Tick Pilot
        assert regime.get_tick_size("AAPL", date(2019, 1, 1)) == 0.01

    def test_get_regime_name_returns_correct_regime(self, temp_tick_file):
        """Should return correct regime name."""
        regime = TickSizeRegime(temp_tick_file)

        assert regime.get_regime_name("AAPL", date(2017, 1, 1)) == "TICK_PILOT"
        assert regime.get_regime_name("GOOGL", date(2017, 1, 1)) == "REG_NMS"


class TestTickConstraint:
    """Tests for tick constraint detection."""

    def test_is_tick_constrained_near_boundary(self, temp_tick_file):
        """Should return True when price is near tick boundary."""
        regime = TickSizeRegime(temp_tick_file)

        # Price of $10.005 is exactly 0.5 ticks from $10.00 and $10.01
        # With threshold_pct=0.5, it should be constrained
        constrained = regime.is_tick_constrained(
            "GOOGL", date(2020, 1, 1), price=10.005, threshold_pct=0.5
        )
        assert constrained is True

    def test_is_tick_constrained_far_from_boundary(self, temp_tick_file):
        """Should return False when price is far from tick boundary."""
        regime = TickSizeRegime(temp_tick_file)

        # Price of $10.00 is exactly on tick boundary
        constrained = regime.is_tick_constrained(
            "GOOGL", date(2020, 1, 1), price=10.00, threshold_pct=0.1
        )
        assert constrained is True  # 0 distance to lower bound

    def test_is_tick_constrained_uses_correct_tick_size(self, temp_tick_file):
        """Constraint check should use appropriate tick size for symbol/date."""
        regime = TickSizeRegime(temp_tick_file)

        # AAPL during Tick Pilot has $0.05 tick
        # Price of $100.025 is 0.5 ticks from $100.00 and $100.05
        constrained = regime.is_tick_constrained(
            "AAPL", date(2017, 1, 1), price=100.025, threshold_pct=0.5
        )
        assert constrained is True


class TestTickPilot:
    """Tests for Tick Pilot program handling."""

    def test_was_tick_pilot_symbol_true_during_pilot(self, temp_tick_file):
        """Should return True for pilot symbols during pilot period."""
        regime = TickSizeRegime(temp_tick_file)

        assert regime.was_tick_pilot_symbol("AAPL", date(2017, 6, 1)) is True
        assert regime.was_tick_pilot_symbol("MSFT", date(2017, 6, 1)) is True

    def test_was_tick_pilot_symbol_false_outside_pilot(self, temp_tick_file):
        """Should return False outside pilot period."""
        regime = TickSizeRegime(temp_tick_file)

        # Before pilot
        assert regime.was_tick_pilot_symbol("AAPL", date(2016, 1, 1)) is False

        # After pilot
        assert regime.was_tick_pilot_symbol("AAPL", date(2019, 1, 1)) is False

    def test_was_tick_pilot_symbol_false_for_non_pilot(self, temp_tick_file):
        """Should return False for symbols not in pilot."""
        regime = TickSizeRegime(temp_tick_file)

        assert regime.was_tick_pilot_symbol("GOOGL", date(2017, 6, 1)) is False

    def test_get_tick_pilot_symbols(self, temp_tick_file):
        """Should return set of all Tick Pilot symbols."""
        regime = TickSizeRegime(temp_tick_file)

        pilot_symbols = regime.get_tick_pilot_symbols()
        assert pilot_symbols == {"AAPL", "MSFT"}


class TestDefaultRules:
    """Tests for default/fallback behavior."""

    def test_missing_file_uses_defaults(self, tmp_path):
        """Should use default Reg NMS rules when file missing."""
        regime = TickSizeRegime(tmp_path / "missing" / "data.parquet")

        # Should default to Reg NMS $0.01
        tick = regime.get_tick_size("AAPL", date(2020, 1, 1))
        assert tick == 0.01

    def test_before_reg_nms_uses_default(self, temp_tick_file):
        """Dates before Reg NMS effective should return default."""
        regime = TickSizeRegime(temp_tick_file)

        # Reg NMS effective 2007-10-01, query before
        tick = regime.get_tick_size("AAPL", date(2005, 1, 1))
        assert tick == 0.01  # Falls back to class default

    def test_initialize_defaults_creates_file(self, tmp_path):
        """initialize_defaults should create base Reg NMS rule."""
        path = tmp_path / "tick_size_regime" / "data.parquet"
        regime = TickSizeRegime(path)

        regime.initialize_defaults()

        assert path.exists()
        df = pd.read_parquet(path)
        assert len(df) == 1
        assert df.iloc[0]["regime_name"] == "REG_NMS"
        assert df.iloc[0]["tick_size"] == 0.01


class TestAddRules:
    """Tests for adding tick size rules."""

    def test_add_rule_creates_new_entry(self, tmp_path):
        """add_rule should create new entry in data file."""
        path = tmp_path / "tick_size_regime" / "data.parquet"
        regime = TickSizeRegime(path)

        regime.add_rule(
            effective_date=date(2020, 1, 1),
            tick_size=0.05,
            regime_name="CUSTOM",
            symbol="TEST",
            notes="Test rule",
        )

        df = pd.read_parquet(path)
        assert len(df) == 1
        assert df.iloc[0]["symbol"] == "TEST"
        assert df.iloc[0]["tick_size"] == 0.05

    def test_bulk_load_tick_pilot_symbols(self, tmp_path):
        """bulk_load_tick_pilot_symbols should add all symbols."""
        path = tmp_path / "tick_size_regime" / "data.parquet"
        regime = TickSizeRegime(path)

        regime.bulk_load_tick_pilot_symbols(["AAPL", "MSFT", "TSLA"])

        df = pd.read_parquet(path)
        assert len(df) == 3
        assert set(df["symbol"]) == {"AAPL", "MSFT", "TSLA"}
        assert all(df["regime_name"] == "TICK_PILOT")
        assert all(df["tick_size"] == 0.05)

    def test_add_rule_appends_to_existing(self, temp_tick_file):
        """add_rule should append to existing data."""
        regime = TickSizeRegime(temp_tick_file)

        initial_count = len(pd.read_parquet(temp_tick_file))

        regime.add_rule(
            effective_date=date(2025, 1, 1),
            tick_size=0.02,
            regime_name="NEW_REGIME",
            symbol=None,  # Market-wide
        )

        regime.invalidate_cache()
        new_count = len(pd.read_parquet(temp_tick_file))
        assert new_count == initial_count + 1


class TestCaching:
    """Tests for caching behavior."""

    def test_cache_invalidation_reloads_data(self, temp_tick_file):
        """invalidate_cache should force reload on next access."""
        regime = TickSizeRegime(temp_tick_file)

        # First access loads data
        _ = regime.get_tick_size("AAPL", date(2017, 1, 1))
        assert regime._cache is not None

        # Invalidate
        regime.invalidate_cache()
        assert regime._cache is None
        assert regime._tick_pilot_symbols is None

        # Next access reloads
        _ = regime.get_tick_size("AAPL", date(2017, 1, 1))
        assert regime._cache is not None


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_exact_pilot_start_date(self, temp_tick_file):
        """Should return pilot tick on exact start date."""
        regime = TickSizeRegime(temp_tick_file)

        # Exactly on pilot start date
        tick = regime.get_tick_size("AAPL", date(2016, 10, 3))
        assert tick == 0.05

    def test_day_before_pilot_start(self, temp_tick_file):
        """Should return standard tick day before pilot."""
        regime = TickSizeRegime(temp_tick_file)

        tick = regime.get_tick_size("AAPL", date(2016, 10, 2))
        assert tick == 0.01

    def test_price_exactly_one_dollar(self, temp_tick_file):
        """Price exactly $1.00 should use standard tick (not sub-penny)."""
        regime = TickSizeRegime(temp_tick_file)

        tick = regime.get_tick_size("PENNY", date(2020, 1, 1), price=1.00)
        assert tick == 0.01

    def test_price_just_under_one_dollar(self, temp_tick_file):
        """Price just under $1.00 should use sub-penny tick."""
        regime = TickSizeRegime(temp_tick_file)

        tick = regime.get_tick_size("PENNY", date(2020, 1, 1), price=0.9999)
        assert tick == 0.0001
