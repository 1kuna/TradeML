"""
Unit tests for Combinatorially Purged Cross-Validation (CPCV).

Tests the logic in validation/cpcv/cpcv.py:
- Label time calculation (t0, t1 intervals)
- Purging: Remove training samples with overlapping label periods
- Embargo: Buffer period after test set
- Symbol-aware purging for multi-symbol panels
- Combinatorial split generation
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest


class TestLabelTimeCalculation:
    """Test label time interval calculation."""

    @pytest.fixture
    def cpcv(self):
        """Create a CPCV instance."""
        from validation.cpcv import CPCV
        return CPCV(n_folds=4, embargo_days=5, purge_pct=0.01)

    def test_label_end_time_calculated(self, cpcv):
        """Verify t1 = t0 + horizon_days."""
        labels = pd.DataFrame({
            "date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)],
            "horizon_days": [5, 5, 5],
        })

        label_times = cpcv.get_label_times(labels)

        assert "t0" in label_times.columns
        assert "t1" in label_times.columns
        # First label: t0=Jan 1, horizon=5 -> t1=Jan 6
        assert label_times.iloc[0]["t1"] == date(2024, 1, 6)

    def test_variable_horizons(self, cpcv):
        """Verify different horizons produce correct t1 values."""
        labels = pd.DataFrame({
            "date": [date(2024, 1, 1), date(2024, 1, 1), date(2024, 1, 1)],
            "horizon_days": [1, 5, 10],
        })

        label_times = cpcv.get_label_times(labels)

        assert label_times.iloc[0]["t1"] == date(2024, 1, 2)
        assert label_times.iloc[1]["t1"] == date(2024, 1, 6)
        assert label_times.iloc[2]["t1"] == date(2024, 1, 11)

    def test_symbol_column_preserved(self, cpcv):
        """Verify symbol column is preserved in label times."""
        labels = pd.DataFrame({
            "date": [date(2024, 1, 1), date(2024, 1, 1)],
            "horizon_days": [5, 5],
            "symbol": ["AAPL", "MSFT"],
        })

        label_times = cpcv.get_label_times(labels, symbol_col="symbol")

        assert "symbol" in label_times.columns
        assert label_times.iloc[0]["symbol"] == "AAPL"
        assert label_times.iloc[1]["symbol"] == "MSFT"


class TestPurging:
    """Test purging of overlapping training samples."""

    @pytest.fixture
    def cpcv(self):
        from validation.cpcv import CPCV
        return CPCV(n_folds=4, embargo_days=0, purge_pct=0.01)

    def test_overlapping_samples_purged(self, cpcv):
        """Verify samples with overlapping label periods are removed."""
        # Test: Jan 10-15 (t0=Jan 10, t1=Jan 15)
        # Train candidates with varying end dates:
        # - Jan 1 (t1=Jan 6): No overlap with test Jan 10-15 -> KEEP
        # - Jan 8 (t1=Jan 13): Overlaps test (t1 >= test_t0) -> PURGE
        # - Jan 20 (t1=Jan 25): After test, no overlap -> KEEP
        label_times = pd.DataFrame({
            "t0": [date(2024, 1, 1), date(2024, 1, 8), date(2024, 1, 10), date(2024, 1, 20)],
            "t1": [date(2024, 1, 6), date(2024, 1, 13), date(2024, 1, 15), date(2024, 1, 25)],
        })

        train_indices = np.array([0, 1, 3])  # Jan 1, Jan 8, Jan 20
        test_indices = np.array([2])  # Jan 10

        purged = cpcv.purge_train_set(label_times, train_indices, test_indices)

        # Jan 8 (idx 1) overlaps with test (t1=Jan 13 >= test_t0=Jan 10) - PURGED
        # Jan 1 (idx 0) no overlap (t1=Jan 6 < test_t0=Jan 10) - KEPT
        # Jan 20 (idx 3) no overlap (t0=Jan 20 > test_t1=Jan 15) - KEPT
        assert 1 not in purged  # Overlapping sample purged
        assert 0 in purged  # Non-overlapping kept
        assert 3 in purged  # Non-overlapping kept

    def test_non_overlapping_samples_kept(self, cpcv):
        """Verify non-overlapping samples are retained."""
        label_times = pd.DataFrame({
            "t0": [date(2024, 1, 1), date(2024, 1, 10), date(2024, 1, 20)],
            "t1": [date(2024, 1, 5), date(2024, 1, 15), date(2024, 1, 25)],
        })

        train_indices = np.array([0, 2])  # Jan 1-5 and Jan 20-25
        test_indices = np.array([1])  # Jan 10-15

        purged = cpcv.purge_train_set(label_times, train_indices, test_indices)

        # Neither train sample overlaps with test
        assert len(purged) == 2
        assert 0 in purged
        assert 2 in purged

    def test_empty_train_set_returns_empty(self, cpcv):
        """Verify empty train set returns empty."""
        label_times = pd.DataFrame({
            "t0": [date(2024, 1, 1)],
            "t1": [date(2024, 1, 5)],
        })

        purged = cpcv.purge_train_set(label_times, np.array([]), np.array([0]))

        assert len(purged) == 0


class TestSymbolAwarePurging:
    """Test symbol-aware purging for multi-symbol panels."""

    @pytest.fixture
    def cpcv(self):
        from validation.cpcv import CPCV
        return CPCV(n_folds=4, embargo_days=0)

    def test_same_symbol_overlap_purged(self, cpcv):
        """Verify overlapping samples for SAME symbol are purged."""
        label_times = pd.DataFrame({
            "t0": [date(2024, 1, 1), date(2024, 1, 5), date(2024, 1, 1), date(2024, 1, 5)],
            "t1": [date(2024, 1, 6), date(2024, 1, 10), date(2024, 1, 6), date(2024, 1, 10)],
            "symbol": ["AAPL", "AAPL", "MSFT", "MSFT"],
        })

        # Train: AAPL Jan 1 (idx 0), MSFT Jan 1 (idx 2)
        # Test: AAPL Jan 5 (idx 1)
        train_indices = np.array([0, 2, 3])
        test_indices = np.array([1])

        purged = cpcv.purge_train_set(label_times, train_indices, test_indices)

        # AAPL sample (idx 0) overlaps with AAPL test (idx 1) - should be purged
        # MSFT samples don't overlap with AAPL test - should be kept
        assert 0 not in purged  # AAPL overlap purged
        assert 2 in purged  # MSFT kept
        assert 3 in purged  # MSFT kept

    def test_different_symbol_no_purge(self, cpcv):
        """Verify overlapping time but DIFFERENT symbol is NOT purged."""
        label_times = pd.DataFrame({
            "t0": [date(2024, 1, 1), date(2024, 1, 3)],
            "t1": [date(2024, 1, 5), date(2024, 1, 7)],
            "symbol": ["AAPL", "MSFT"],
        })

        # AAPL Jan 1-5 overlaps with MSFT Jan 3-7 in TIME
        # But different symbols, so no purge
        train_indices = np.array([0])  # AAPL
        test_indices = np.array([1])  # MSFT

        purged = cpcv.purge_train_set(label_times, train_indices, test_indices)

        # AAPL should NOT be purged (different symbol)
        assert 0 in purged


class TestEmbargo:
    """Test embargo period application."""

    @pytest.fixture
    def cpcv_with_embargo(self):
        from validation.cpcv import CPCV
        return CPCV(n_folds=4, embargo_days=5, purge_pct=0.01)

    def test_embargo_removes_samples_after_test(self, cpcv_with_embargo):
        """Verify samples in embargo period after test are removed."""
        # Embargo logic: removes samples where (t0 > test_t1_max) AND (t0 <= embargo_end)
        # Test t1_max = Jan 15, embargo_days = 5 -> embargo_end = Jan 20
        # So samples with t0 in (Jan 15, Jan 20] are embargoed
        label_times = pd.DataFrame({
            "t0": [date(2024, 1, 5), date(2024, 1, 10),   # test samples
                   date(2024, 1, 16), date(2024, 1, 25)],  # train candidates
            "t1": [date(2024, 1, 10), date(2024, 1, 15),
                   date(2024, 1, 21), date(2024, 1, 30)],
        })

        train_indices = np.array([2, 3])  # Jan 16 and Jan 25
        test_indices = np.array([0, 1])  # Test ends Jan 15

        embargoed = cpcv_with_embargo.apply_embargo(
            label_times, train_indices, test_indices
        )

        # Jan 16: t0=Jan 16 > Jan 15 (True) AND t0=Jan 16 <= Jan 20 (True) -> EMBARGOED
        # Jan 25: t0=Jan 25 > Jan 15 (True) AND t0=Jan 25 <= Jan 20 (False) -> KEPT
        assert 2 not in embargoed  # Jan 16 embargoed (within embargo window)
        assert 3 in embargoed  # Jan 25 kept (past embargo window)

    def test_samples_before_test_not_embargoed(self, cpcv_with_embargo):
        """Verify samples BEFORE test set are not affected by embargo."""
        label_times = pd.DataFrame({
            "t0": [date(2024, 1, 1), date(2024, 1, 10)],
            "t1": [date(2024, 1, 5), date(2024, 1, 15)],
        })

        train_indices = np.array([0])  # Jan 1 - before test
        test_indices = np.array([1])  # Jan 10

        embargoed = cpcv_with_embargo.apply_embargo(
            label_times, train_indices, test_indices
        )

        # Jan 1 is before test, should not be embargoed
        assert 0 in embargoed

    def test_zero_embargo_keeps_all(self):
        """Verify zero embargo days keeps all samples."""
        from validation.cpcv import CPCV
        cpcv = CPCV(n_folds=4, embargo_days=0)

        label_times = pd.DataFrame({
            "t0": [date(2024, 1, 5), date(2024, 1, 11)],
            "t1": [date(2024, 1, 10), date(2024, 1, 16)],
        })

        train_indices = np.array([1])
        test_indices = np.array([0])

        embargoed = cpcv.apply_embargo(label_times, train_indices, test_indices)

        # With 0 embargo, sample immediately after test is kept
        assert len(embargoed) == 1


class TestSplitGeneration:
    """Test train/test split generation."""

    @pytest.fixture
    def sample_data(self):
        """Create sample X and labels for split testing."""
        n_samples = 100
        dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(n_samples)]

        X = pd.DataFrame({
            "feature1": np.random.randn(n_samples),
            "feature2": np.random.randn(n_samples),
        })

        labels = pd.DataFrame({
            "date": dates,
            "horizon_days": [5] * n_samples,
            "label": np.random.randint(0, 2, n_samples),
        })

        return X, labels

    def test_correct_number_of_splits(self, sample_data):
        """Verify correct number of splits generated."""
        from validation.cpcv import CPCV
        cpcv = CPCV(n_folds=8, embargo_days=2)

        X, labels = sample_data
        splits = cpcv.split(X, labels)

        # Should have 8 splits for 8 folds
        # (some may be dropped if empty after purge/embargo)
        assert len(splits) <= 8
        assert len(splits) > 0

    def test_train_test_disjoint(self, sample_data):
        """Verify train and test sets are disjoint."""
        from validation.cpcv import CPCV
        cpcv = CPCV(n_folds=4, embargo_days=2)

        X, labels = sample_data
        splits = cpcv.split(X, labels)

        for train_idx, test_idx in splits:
            train_set = set(train_idx)
            test_set = set(test_idx)
            assert train_set.isdisjoint(test_set), "Train and test overlap!"

    def test_test_sets_cover_data(self, sample_data):
        """Verify test sets together cover all data."""
        from validation.cpcv import CPCV
        cpcv = CPCV(n_folds=4, embargo_days=0, purge_pct=0)

        X, labels = sample_data
        splits = cpcv.split(X, labels)

        all_test = set()
        for _, test_idx in splits:
            all_test.update(test_idx)

        # All samples should appear in at least one test set
        assert len(all_test) == len(X)


class TestCombinatorialSplit:
    """Test combinatorial split generation."""

    @pytest.fixture
    def sample_data(self):
        n_samples = 80  # Divisible by 8 folds
        dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(n_samples)]

        X = pd.DataFrame({
            "feature1": np.random.randn(n_samples),
        })

        labels = pd.DataFrame({
            "date": dates,
            "horizon_days": [5] * n_samples,
        })

        return X, labels

    def test_combinatorial_produces_more_splits(self, sample_data):
        """Verify combinatorial produces C(n_folds, n_test_folds) splits."""
        from validation.cpcv import CPCV
        from math import comb

        cpcv = CPCV(n_folds=4, embargo_days=0)
        X, labels = sample_data

        # Standard split
        standard_splits = cpcv.split(X, labels)

        # Combinatorial with 2 test folds: C(4,2) = 6 combinations
        comb_splits = cpcv.combinatorial_split(X, labels, n_test_folds=2)

        assert len(standard_splits) == 4
        # Combinatorial should have C(4,2)=6 splits (some may be reduced by purge/embargo)
        assert len(comb_splits) <= comb(4, 2)
        assert len(comb_splits) > 0

    def test_combinatorial_larger_test_sets(self, sample_data):
        """Verify combinatorial test sets are larger with more test folds."""
        from validation.cpcv import CPCV
        cpcv = CPCV(n_folds=4, embargo_days=0)
        X, labels = sample_data

        comb_splits = cpcv.combinatorial_split(X, labels, n_test_folds=2)

        # Each test set should have 2 folds worth of samples
        for train_idx, test_idx in comb_splits:
            # ~50% test (2/4 folds)
            assert len(test_idx) >= len(X) // 4  # At least 1 fold


class TestIntegration:
    """Integration tests for CPCV workflow."""

    def test_full_workflow_single_symbol(self):
        """Test complete CPCV workflow with single symbol data."""
        from validation.cpcv import CPCV

        np.random.seed(42)
        n_samples = 200

        dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(n_samples)]

        X = pd.DataFrame({
            "feature1": np.random.randn(n_samples),
            "feature2": np.random.randn(n_samples),
        })

        labels = pd.DataFrame({
            "date": dates,
            "horizon_days": [5] * n_samples,
            "label": np.random.randint(0, 2, n_samples),
        })

        cpcv = CPCV(n_folds=5, embargo_days=5)
        splits = cpcv.split(X, labels)

        assert len(splits) > 0

        # Verify each split has valid train/test
        for train_idx, test_idx in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0

    def test_full_workflow_multi_symbol(self):
        """Test complete CPCV workflow with multi-symbol panel."""
        from validation.cpcv import CPCV

        np.random.seed(42)
        symbols = ["AAPL", "MSFT", "GOOGL"]
        dates_per_symbol = 50
        n_samples = len(symbols) * dates_per_symbol

        rows = []
        for sym in symbols:
            for i in range(dates_per_symbol):
                rows.append({
                    "symbol": sym,
                    "date": date(2024, 1, 1) + timedelta(days=i),
                    "feature1": np.random.randn(),
                    "horizon_days": 5,
                })

        df = pd.DataFrame(rows)
        X = df[["feature1"]]
        labels = df[["date", "horizon_days", "symbol"]]

        cpcv = CPCV(n_folds=4, embargo_days=3)
        splits = cpcv.split(X, labels, symbol_col="symbol")

        assert len(splits) > 0

        for train_idx, test_idx in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_sample_per_fold(self):
        """Test with very small dataset (1 sample per fold)."""
        from validation.cpcv import CPCV

        cpcv = CPCV(n_folds=4, embargo_days=0)

        X = pd.DataFrame({"f": [1, 2, 3, 4]})
        labels = pd.DataFrame({
            "date": [date(2024, 1, i+1) for i in range(4)],
            "horizon_days": [1] * 4,
        })

        splits = cpcv.split(X, labels)

        # Should still produce valid splits
        for train_idx, test_idx in splits:
            assert len(test_idx) >= 1

    def test_all_samples_same_date(self):
        """Test when all samples have same date (cross-sectional)."""
        from validation.cpcv import CPCV

        cpcv = CPCV(n_folds=4, embargo_days=5)

        n = 100
        same_date = date(2024, 1, 15)

        X = pd.DataFrame({"f": np.random.randn(n)})
        labels = pd.DataFrame({
            "date": [same_date] * n,
            "horizon_days": [5] * n,
        })

        # Should handle gracefully - all samples have same t0/t1
        splits = cpcv.split(X, labels)
        # May produce valid splits or empty due to heavy purging
        # The key is it shouldn't crash

    def test_long_horizon_heavy_purging(self):
        """Test that long horizons result in more purging."""
        from validation.cpcv import CPCV

        cpcv = CPCV(n_folds=4, embargo_days=0)

        n = 100
        X = pd.DataFrame({"f": np.random.randn(n)})

        # Short horizon (1 day)
        short_labels = pd.DataFrame({
            "date": [date(2024, 1, 1) + timedelta(days=i) for i in range(n)],
            "horizon_days": [1] * n,
        })

        # Long horizon (20 days)
        long_labels = pd.DataFrame({
            "date": [date(2024, 1, 1) + timedelta(days=i) for i in range(n)],
            "horizon_days": [20] * n,
        })

        short_splits = cpcv.split(X, short_labels)
        long_splits = cpcv.split(X, long_labels)

        # Long horizon should result in smaller training sets (more purging)
        for (short_train, _), (long_train, _) in zip(short_splits, long_splits):
            assert len(long_train) <= len(short_train)
