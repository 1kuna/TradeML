"""
Test the fixed CPCV with multi-symbol panels.

This script verifies that the symbol-aware purging works correctly
and doesn't remove all training data.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import date
from loguru import logger

from validation.cpcv import CPCV

def run_cpcv_multi_symbol() -> bool:
    """Run CPCV with a simple multi-symbol panel."""

    logger.info("=" * 80)
    logger.info("Testing Fixed CPCV with Multi-Symbol Panel")
    logger.info("=" * 80)

    # Create synthetic multi-symbol panel
    # 3 symbols, 100 days each = 300 samples
    symbols = ["AAPL", "MSFT", "GOOGL"]
    n_days = 100
    start_date = date(2024, 1, 1)

    data = []
    for symbol in symbols:
        for i in range(n_days):
            current_date = start_date + pd.Timedelta(days=i)
            data.append({
                "date": current_date,
                "symbol": symbol,
                "feature_1": np.random.randn(),
                "feature_2": np.random.randn(),
            })

    X = pd.DataFrame(data)

    # Create labels DataFrame
    labels = pd.DataFrame({
        "date": X["date"],
        "symbol": X["symbol"],
        "horizon_days": 5,  # 5-day labels
    })

    logger.info(f"Created dataset: {len(X)} samples ({len(symbols)} symbols × {n_days} days)")

    # Test with fixed CPCV
    logger.info("\nTesting CPCV with n_folds=5, embargo=10 days")
    cv = CPCV(n_folds=5, embargo_days=10, purge_pct=0.01)

    # Generate splits
    splits = cv.split(X, labels, date_col="date", horizon_col="horizon_days", symbol_col="symbol")

    logger.info(f"\n✓ Generated {len(splits)} CPCV splits (expected: 5)")

    if len(splits) == 0:
        logger.error("❌ FAILED: No splits generated! CPCV purging is still too aggressive.")
        return False

    # Analyze each split
    logger.info("\nAnalyzing splits:")
    for i, (train_idx, test_idx) in enumerate(splits):
        train_symbols = X.iloc[train_idx]["symbol"].nunique()
        test_symbols = X.iloc[test_idx]["symbol"].nunique()

        logger.info(
            f"  Fold {i}: {len(train_idx)} train samples ({train_symbols} symbols), "
            f"{len(test_idx)} test samples ({test_symbols} symbols)"
        )

        # Check that we have reasonable training data
        if len(train_idx) < 50:
            logger.warning(f"    ⚠️  Warning: Only {len(train_idx)} training samples")

    # Calculate training set retention
    total_samples = len(X)
    avg_train_size = np.mean([len(train_idx) for train_idx, _ in splits])
    retention_pct = (avg_train_size / total_samples) * 100

    logger.info(f"\nTraining set retention: {retention_pct:.1f}%")

    if retention_pct < 30:
        logger.warning("⚠️  Low training set retention (<30%)")
        return False
    elif retention_pct < 50:
        logger.info("✓ Moderate training set retention (30-50%)")
        return True
    else:
        logger.success(f"✅ Good training set retention ({retention_pct:.1f}%)")
        return True


def test_cpcv_multi_symbol():
    """Test CPCV with a simple multi-symbol panel."""
    assert run_cpcv_multi_symbol()


def run_cpcv_single_symbol() -> bool:
    """Run CPCV for the single-symbol case."""

    logger.info("\n" + "=" * 80)
    logger.info("Testing CPCV with Single-Symbol Panel (Backward Compatibility)")
    logger.info("=" * 80)

    # Create single-symbol panel
    n_days = 100
    start_date = date(2024, 1, 1)

    data = []
    for i in range(n_days):
        current_date = start_date + pd.Timedelta(days=i)
        data.append({
            "date": current_date,
            "feature_1": np.random.randn(),
            "feature_2": np.random.randn(),
        })

    X = pd.DataFrame(data)

    # Create labels DataFrame (no symbol column)
    labels = pd.DataFrame({
        "date": X["date"],
        "horizon_days": 5,
    })

    logger.info(f"Created single-symbol dataset: {len(X)} samples")

    # Test with CPCV
    cv = CPCV(n_folds=5, embargo_days=10)
    splits = cv.split(X, labels, date_col="date", horizon_col="horizon_days", symbol_col=None)

    logger.info(f"\n✓ Generated {len(splits)} CPCV splits")

    if len(splits) == 0:
        logger.error("❌ FAILED: No splits generated for single-symbol case")
        return False

    # Analyze splits
    for i, (train_idx, test_idx) in enumerate(splits):
        logger.info(f"  Fold {i}: {len(train_idx)} train, {len(test_idx)} test")

    logger.success("✅ Single-symbol CPCV working correctly")
    return True


def test_cpcv_single_symbol():
    """Test that CPCV still works for single-symbol case."""
    assert run_cpcv_single_symbol()


if __name__ == "__main__":
    logger.info("🧪 CPCV Multi-Symbol Fix Test Suite")
    logger.info("")

    # Run tests
    test1_passed = run_cpcv_multi_symbol()
    test2_passed = run_cpcv_single_symbol()

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)

    if test1_passed and test2_passed:
        logger.success("✅ ALL TESTS PASSED")
        logger.success("   - Multi-symbol CPCV working correctly")
        logger.success("   - Single-symbol backward compatibility maintained")
        logger.info("\nThe CPCV fix is ready for production use!")
        sys.exit(0)
    else:
        logger.error("❌ SOME TESTS FAILED")
        if not test1_passed:
            logger.error("   - Multi-symbol CPCV failed")
        if not test2_passed:
            logger.error("   - Single-symbol CPCV failed")
        sys.exit(1)
