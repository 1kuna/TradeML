"""
Simple workflow test for CPCV + PBO + DSR.

Demonstrates the validation suite with a simple synthetic example.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import date, timedelta
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from validation.cpcv import CPCV
from validation.pbo import PBOCalculator
from validation.dsr import DSRCalculator


def main():
    """Run simple validation workflow."""
    logger.info("Testing CPCV + PBO + DSR Validation Workflow")

    # Create simple dataset (single symbol time series)
    np.random.seed(42)
    n_obs = 1000

    dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n_obs)]

    # Features: simple momentum
    prices = 100 + np.cumsum(np.random.randn(n_obs) * 0.5)
    momentum = np.concatenate([[0] * 5, [(prices[i] / prices[i-5] - 1) for i in range(5, n_obs)]])

    features = pd.DataFrame({
        'date': dates,
        'symbol': 'TEST',
        'price': prices,
        'momentum': momentum
    })

    # Labels: 5-day forward return
    forward_returns = np.array([(prices[i+5] / prices[i] - 1) if i < n_obs-5 else 0.0 for i in range(n_obs)])

    labels = pd.DataFrame({
        'date': dates,
        'symbol': 'TEST',
        'horizon_days': 5,
        'forward_return': forward_returns,
        'label': (forward_returns > 0).astype(int)
    })

    # 1. Test CPCV
    print("\n" + "="*60)
    print("1. CPCV SPLITS")
    print("="*60)

    cv = CPCV(n_folds=5, embargo_days=3)
    splits = cv.split(features, labels)

    print(f"Generated {len(splits)} splits")
    if len(splits) > 0:
        for i, (train_idx, test_idx) in enumerate(splits[:3]):
            print(f"  Split {i+1}: train={len(train_idx)}, test={len(test_idx)}")
    else:
        print("  WARNING: No splits generated (likely due to aggressive purging)")

    # 2. Test PBO (simplified - use manual performance matrices)
    print("\n" + "="*60)
    print("2. PBO CALCULATION")
    print("="*60)

    # Simulate 5 strategies tested on 8 trials
    np.random.seed(123)
    n_configs = 5
    n_trials = 8

    # Simulate IS/OOS performance with overfitting
    is_performance = np.random.randn(n_configs, n_trials) * 0.3 + 0.5
    oos_performance = is_performance * 0.6 + np.random.randn(n_configs, n_trials) * 0.4

    pbo_calc = PBOCalculator(n_trials=n_trials)
    pbo_results = pbo_calc.calculate_pbo(is_performance, oos_performance)

    print(f"PBO:                   {pbo_results['pbo']:.2%}")
    print(f"Threshold:             <= 5%")
    print(f"Status:                {'PASS' if pbo_results['pass_threshold'] else 'FAIL'}")
    print(f"Rank-1 OOS Mean:       {pbo_results['rank1_oos_mean']:.4f}")
    print(f"OOS Median (all):      {pbo_results['oos_median_all']:.4f}")

    # 3. Test DSR
    print("\n" + "="*60)
    print("3. DSR CALCULATION")
    print("="*60)

    # Simulate strategy returns
    daily_returns = np.random.randn(252) * 0.01 + 0.0005  # ~12% annual with 16% vol

    dsr_calc = DSRCalculator(annual_factor=252.0)
    dsr_results = dsr_calc.calculate_dsr(daily_returns, n_trials=10)

    print(f"Sharpe Ratio:          {dsr_results['sharpe_ratio']:.2f}")
    print(f"Sharpe Adjustment:     {dsr_results['sharpe_adj']:.2f}")
    print(f"DSR:                   {dsr_results['dsr']:.4f}")
    print(f"P(True SR > 0):        {dsr_results['dsr']*100:.1f}%")
    print(f"Status:                {'PASS' if dsr_results['pass_threshold'] else 'FAIL'}")
    print(f"Skewness:              {dsr_results['skewness']:.2f}")
    print(f"Kurtosis:              {dsr_results['kurtosis']:.2f}")

    # 4. Overall Validation Result
    print("\n" + "="*60)
    print("4. OVERALL VALIDATION RESULT")
    print("="*60)

    overall_pass = pbo_results['pass_threshold'] and dsr_results['pass_threshold']

    print(f"CPCV Splits:           {len(splits)}")
    print(f"PBO:                   {pbo_results['pbo']:.2%} ({'PASS' if pbo_results['pass_threshold'] else 'FAIL'})")
    print(f"DSR:                   {dsr_results['dsr']:.4f} ({'PASS' if dsr_results['pass_threshold'] else 'FAIL'})")
    print(f"\nReady for Production:  {'YES' if overall_pass else 'NO'}")

    if overall_pass:
        print("\nAll validation checks passed!")
        print("Next steps:")
        print("  1. Run shadow trading for >= 4 weeks")
        print("  2. Monitor for drift and degradation")
        print("  3. Promote to production if shadow performs well")
    else:
        print("\nValidation FAILED - strategy needs improvement")
        print("Issues:")
        if not pbo_results['pass_threshold']:
            print(f"  - PBO too high ({pbo_results['pbo']:.2%}) - likely overfit")
        if not dsr_results['pass_threshold']:
            print(f"  - DSR too low ({dsr_results['dsr']:.4f}) - insufficient evidence of skill")

    print("="*60)


if __name__ == "__main__":
    main()
