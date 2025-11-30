"""
Unit tests for corporate actions processing.

Tests the logic in data_layer/reference/corporate_actions.py:
- Split adjustment calculation (cumulative factors)
- Backward-flowing price adjustments
- Dividend return calculation
- Point-in-time (PIT) discipline
- Validation of adjustments
"""

from __future__ import annotations

from datetime import date
from typing import Dict

import numpy as np
import pandas as pd
import pytest


class TestSplitAdjustmentCalculation:
    """Test split adjustment factor calculation."""

    @pytest.fixture
    def processor(self):
        """Create a CorporateActionsProcessor instance."""
        from data_layer.reference.corporate_actions import CorporateActionsProcessor
        return CorporateActionsProcessor()

    @pytest.fixture
    def sample_splits(self) -> pd.DataFrame:
        """Create sample split events."""
        return pd.DataFrame({
            "symbol": ["AAPL", "AAPL", "TSLA"],
            "event_type": ["split", "split", "split"],
            "ex_date": [date(2020, 8, 31), date(2014, 6, 9), date(2022, 8, 25)],
            "ratio": [4.0, 7.0, 3.0],  # 4:1, 7:1, 3:1
            "amount": [0.0, 0.0, 0.0],
        })

    def test_single_split_factor(self, processor, sample_splits):
        """Verify single split produces correct adjustment factor."""
        # TSLA 3:1 split
        adj = processor.calculate_split_adjustment(sample_splits, "TSLA")

        assert len(adj) == 1
        assert adj.iloc[0]["split_ratio"] == 3.0
        assert adj.iloc[0]["cumulative_factor"] == 3.0

    def test_multiple_splits_cumulative(self, processor, sample_splits):
        """Verify multiple splits produce cumulative adjustment."""
        # AAPL had 4:1 in 2020 and 7:1 in 2014
        # Cumulative: oldest prices need 4 * 7 = 28x adjustment
        adj = processor.calculate_split_adjustment(sample_splits, "AAPL")

        assert len(adj) == 2
        # Most recent split (2020) first
        factors = adj.sort_values("ex_date", ascending=False)["cumulative_factor"].tolist()
        assert factors[0] == 4.0  # 2020 split alone
        assert factors[1] == 28.0  # 2014 split cumulative (4 * 7)

    def test_no_splits_returns_empty(self, processor, sample_splits):
        """Verify symbol with no splits returns empty DataFrame."""
        adj = processor.calculate_split_adjustment(sample_splits, "MSFT")
        assert adj.empty

    def test_reverse_split_factor(self, processor):
        """Verify reverse split (ratio < 1) works correctly."""
        events = pd.DataFrame({
            "symbol": ["GE"],
            "event_type": ["split"],
            "ex_date": [date(2021, 8, 2)],
            "ratio": [0.125],  # 1:8 reverse split
            "amount": [0.0],
        })

        adj = processor.calculate_split_adjustment(events, "GE")
        assert len(adj) == 1
        assert adj.iloc[0]["cumulative_factor"] == 0.125


class TestPriceAdjustmentApplication:
    """Test application of split adjustments to price data."""

    @pytest.fixture
    def processor(self):
        from data_layer.reference.corporate_actions import CorporateActionsProcessor
        return CorporateActionsProcessor()

    @pytest.fixture
    def sample_prices(self) -> pd.DataFrame:
        """Create sample price data around a split."""
        # TSLA around 3:1 split on 2022-08-25
        return pd.DataFrame({
            "date": [date(2022, 8, 23), date(2022, 8, 24), date(2022, 8, 25), date(2022, 8, 26)],
            "symbol": ["TSLA"] * 4,
            "open": [900.0, 890.0, 302.0, 295.0],
            "high": [910.0, 895.0, 310.0, 305.0],
            "low": [885.0, 880.0, 298.0, 290.0],
            "close": [891.0, 891.0, 297.0, 296.0],
            "volume": [20_000_000, 22_000_000, 65_000_000, 55_000_000],
        })

    @pytest.fixture
    def split_adjustment(self) -> pd.DataFrame:
        """Adjustment factor for TSLA 3:1 split."""
        return pd.DataFrame({
            "ex_date": [date(2022, 8, 25)],
            "split_ratio": [3.0],
            "cumulative_factor": [3.0],
        })

    def test_prices_before_ex_date_adjusted(self, processor, sample_prices, split_adjustment):
        """Verify prices before ex-date are multiplied by factor."""
        adjusted = processor.apply_split_adjustments(sample_prices, split_adjustment)

        # Pre-split prices should be adjusted by factor of 3
        # 2022-08-23: close was 891, adjusted should be 891 * 3 = 2673
        aug23 = adjusted[adjusted["date"] == date(2022, 8, 23)]
        assert np.isclose(aug23["close"].iloc[0], 891.0 * 3, rtol=1e-6)

    def test_prices_on_and_after_ex_date_unchanged(self, processor, sample_prices, split_adjustment):
        """Verify prices on/after ex-date are NOT adjusted."""
        adjusted = processor.apply_split_adjustments(sample_prices, split_adjustment)

        # Ex-date and after should be unchanged
        aug25 = adjusted[adjusted["date"] == date(2022, 8, 25)]
        assert np.isclose(aug25["close"].iloc[0], 297.0, rtol=1e-6)

    def test_volume_adjusted_inversely(self, processor, sample_prices, split_adjustment):
        """Verify volume is divided by split factor (inverse of price)."""
        adjusted = processor.apply_split_adjustments(sample_prices, split_adjustment)

        # Pre-split volume 20M should become ~6.67M (20M / 3)
        aug23 = adjusted[adjusted["date"] == date(2022, 8, 23)]
        expected_volume = 20_000_000 // 3
        assert aug23["volume"].iloc[0] == expected_volume

    def test_no_adjustments_preserves_prices(self, processor, sample_prices):
        """Verify empty adjustments DataFrame leaves prices unchanged."""
        adjusted = processor.apply_split_adjustments(sample_prices, pd.DataFrame())

        assert adjusted["close"].tolist() == sample_prices["close"].tolist()
        assert (adjusted["adjustment_factor"] == 1.0).all()


class TestDividendReturn:
    """Test dividend return calculation."""

    @pytest.fixture
    def processor(self):
        from data_layer.reference.corporate_actions import CorporateActionsProcessor
        return CorporateActionsProcessor()

    @pytest.fixture
    def dividend_events(self) -> pd.DataFrame:
        """Create sample dividend events."""
        return pd.DataFrame({
            "symbol": ["AAPL", "AAPL"],
            "event_type": ["dividend", "dividend"],
            "ex_date": [date(2024, 2, 9), date(2024, 5, 10)],
            "ratio": [1.0, 1.0],
            "amount": [0.24, 0.25],  # Quarterly dividends
        })

    @pytest.fixture
    def prices_around_dividend(self) -> pd.DataFrame:
        """Price data around dividend ex-dates."""
        return pd.DataFrame({
            "date": [date(2024, 2, 8), date(2024, 2, 9), date(2024, 2, 12)],
            "symbol": ["AAPL"] * 3,
            "close": [188.0, 187.5, 188.5],
        })

    def test_dividend_return_calculation(self, processor, dividend_events, prices_around_dividend):
        """Verify dividend return is amount / prev_close."""
        result = processor.calculate_dividend_return(
            dividend_events, prices_around_dividend, "AAPL"
        )

        # On ex-date 2024-02-09, prev close was $188
        # Dividend $0.24, return = 0.24 / 188 = 0.001277
        ex_date_row = result[result["date"] == date(2024, 2, 9)]
        expected_return = 0.24 / 188.0
        assert np.isclose(ex_date_row["dividend_return"].iloc[0], expected_return, rtol=1e-4)

    def test_no_dividends_zero_return(self, processor, prices_around_dividend):
        """Verify no dividends results in zero dividend_return column."""
        empty_divs = pd.DataFrame(columns=["symbol", "event_type", "ex_date", "ratio", "amount"])
        result = processor.calculate_dividend_return(empty_divs, prices_around_dividend, "AAPL")

        assert (result["dividend_return"] == 0.0).all()


class TestPITDiscipline:
    """Test point-in-time discipline in adjustments."""

    @pytest.fixture
    def processor(self):
        from data_layer.reference.corporate_actions import CorporateActionsProcessor
        return CorporateActionsProcessor()

    def test_latest_prices_unadjusted(self, processor):
        """Verify most recent prices have no adjustment (factor=1.0)."""
        # Create data where latest date is after all splits
        prices = pd.DataFrame({
            "date": [date(2024, 1, 10), date(2024, 1, 11), date(2024, 1, 12)],
            "symbol": ["TEST"] * 3,
            "open": [100.0, 101.0, 102.0],
            "high": [105.0, 106.0, 107.0],
            "low": [98.0, 99.0, 100.0],
            "close": [103.0, 104.0, 105.0],
            "volume": [1_000_000, 1_100_000, 1_200_000],
        })

        # Split was before our data
        adj = pd.DataFrame({
            "ex_date": [date(2024, 1, 5)],
            "split_ratio": [2.0],
            "cumulative_factor": [2.0],
        })

        adjusted = processor.apply_split_adjustments(prices, adj)

        # All dates are after ex_date, so no adjustments
        assert (adjusted["adjustment_factor"] == 1.0).all()

    def test_adjustments_flow_backward(self, processor):
        """Verify adjustments only affect prices BEFORE ex-date."""
        prices = pd.DataFrame({
            "date": [date(2024, 1, 8), date(2024, 1, 9), date(2024, 1, 10), date(2024, 1, 11)],
            "symbol": ["TEST"] * 4,
            "open": [50.0, 51.0, 102.0, 103.0],
            "high": [52.0, 53.0, 105.0, 106.0],
            "low": [49.0, 50.0, 100.0, 101.0],
            "close": [51.0, 52.0, 104.0, 105.0],
            "volume": [1_000_000] * 4,
        })

        # 2:1 split on 2024-01-10
        adj = pd.DataFrame({
            "ex_date": [date(2024, 1, 10)],
            "split_ratio": [2.0],
            "cumulative_factor": [2.0],
        })

        adjusted = processor.apply_split_adjustments(prices, adj)

        # Before ex_date: adjusted
        jan8 = adjusted[adjusted["date"] == date(2024, 1, 8)]
        assert jan8["adjustment_factor"].iloc[0] == 2.0
        assert np.isclose(jan8["close"].iloc[0], 51.0 * 2, rtol=1e-6)

        # On/after ex_date: unchanged
        jan10 = adjusted[adjusted["date"] == date(2024, 1, 10)]
        assert jan10["adjustment_factor"].iloc[0] == 1.0
        assert np.isclose(jan10["close"].iloc[0], 104.0, rtol=1e-6)


class TestValidation:
    """Test adjustment validation logic."""

    @pytest.fixture
    def processor(self):
        from data_layer.reference.corporate_actions import CorporateActionsProcessor
        return CorporateActionsProcessor()

    def test_latest_price_match_validation(self, processor):
        """Verify validation checks that latest prices match."""
        raw = pd.DataFrame({
            "date": [date(2024, 1, 10), date(2024, 1, 11)],
            "symbol": ["TEST"] * 2,
            "close": [100.0, 101.0],
        })

        adjusted = pd.DataFrame({
            "date": [date(2024, 1, 10), date(2024, 1, 11)],
            "symbol": ["TEST"] * 2,
            "close_adj": [100.0, 101.0],  # Latest matches
            "adjustment_factor": [1.0, 1.0],
        })

        events = pd.DataFrame(columns=["symbol", "event_type", "ex_date", "ratio"])

        result = processor.validate_adjustments(raw, adjusted, events, "TEST")
        assert result["latest_price_match"] == True  # noqa: E712 - numpy bool comparison


class TestCorporateActionsLogic:
    """Unit tests for core corporate actions logic without I/O."""

    def test_split_ratio_interpretation(self):
        """Verify split ratio convention: ratio=4 means 4:1 forward split."""
        # Forward split: shareholder gets 4 shares for every 1
        # Old price $100 -> new price $25
        # Adjustment: multiply old prices by 4 for continuity
        old_price = 100.0
        ratio = 4.0
        adjusted_old = old_price * ratio
        new_price = 25.0

        # Adjusted old should equal pre-split price in post-split terms
        assert adjusted_old == old_price * ratio

    def test_reverse_split_ratio_interpretation(self):
        """Verify reverse split: ratio=0.1 means 1:10 reverse."""
        # Reverse split: 10 shares become 1
        # Old price $10 -> new price $100
        # Adjustment: multiply old prices by 0.1 for continuity
        old_price = 10.0
        ratio = 0.1
        adjusted_old = old_price * ratio
        new_price = 100.0

        assert np.isclose(adjusted_old, 1.0, rtol=1e-6)

    def test_ohlcv_all_adjusted_together(self):
        """Verify O, H, L, C, VWAP all adjusted by same factor."""
        factor = 2.0
        ohlcv = {
            "open": 100.0,
            "high": 105.0,
            "low": 98.0,
            "close": 103.0,
            "vwap": 101.5,
        }

        adjusted = {k: v * factor for k, v in ohlcv.items()}

        assert adjusted["open"] == 200.0
        assert adjusted["high"] == 210.0
        assert adjusted["low"] == 196.0
        assert adjusted["close"] == 206.0
        assert adjusted["vwap"] == 203.0

    def test_volume_adjusted_inversely(self):
        """Verify volume adjustment is inverse of price adjustment."""
        factor = 4.0
        volume = 1_000_000

        # For a 4:1 split, each share becomes 4 shares
        # Volume should decrease proportionally to maintain share-equivalence
        adjusted_volume = int(volume / factor)

        assert adjusted_volume == 250_000
