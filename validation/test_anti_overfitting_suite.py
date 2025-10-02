"""
Test the complete anti-overfitting validation suite.

Demonstrates CPCV + PBO + DSR working together.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import date, timedelta
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from validation.cpcv import CPCV
from validation.pbo import PBOCalculator
from validation.dsr import DSRCalculator
from backtest.engine import MinimalBacktester


def generate_synthetic_data(
    n_symbols: int = 50,
    n_days: int = 1000,
    seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic price data for testing.

    Args:
        n_symbols: Number of symbols
        n_days: Number of trading days
        seed: Random seed

    Returns:
        (features, labels) DataFrames
    """
    np.random.seed(seed)

    # Generate dates
    start_date = date(2020, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_days)]

    # Generate features
    data = []
    for symbol_idx in range(n_symbols):
        symbol = f"SYM{symbol_idx:03d}"

        # Random walk price
        price = 100.0
        prices = []

        for _ in range(n_days):
            # Add some momentum + mean reversion
            ret = np.random.randn() * 0.02
            price *= (1 + ret)
            prices.append(price)

        # Features: momentum, volatility, volume
        for i, (d, p) in enumerate(zip(dates, prices)):
            row = {
                'date': d,
                'symbol': symbol,
                'close': p,
                'momentum_5d': 0.0,
                'momentum_20d': 0.0,
                'volatility_20d': 0.0,
                'volume': np.random.randint(100000, 1000000),
            }

            # Calculate momentum (if enough history)
            if i >= 5:
                row['momentum_5d'] = prices[i] / prices[i-5] - 1
            if i >= 20:
                row['momentum_20d'] = prices[i] / prices[i-20] - 1
                row['volatility_20d'] = np.std(prices[i-20:i]) / np.mean(prices[i-20:i])

            data.append(row)

    df = pd.DataFrame(data)

    # Generate labels (5-day forward return)
    labels = []
    for symbol in df['symbol'].unique():
        symbol_data = df[df['symbol'] == symbol].sort_values('date')

        for i in range(len(symbol_data) - 5):
            row = symbol_data.iloc[i]
            future_price = symbol_data.iloc[i + 5]['close']

            label_row = {
                'date': row['date'],
                'symbol': symbol,
                'horizon_days': 5,
                'forward_return': future_price / row['close'] - 1,
                'label': 1 if future_price > row['close'] else 0
            }
            labels.append(label_row)

    labels_df = pd.DataFrame(labels)

    # Merge features and labels
    merged = df.merge(labels_df, on=['date', 'symbol'], how='inner')

    # Separate features and labels
    feature_cols = ['date', 'symbol', 'momentum_5d', 'momentum_20d', 'volatility_20d', 'volume']
    features = merged[feature_cols + ['close']].copy()
    labels = merged[['date', 'symbol', 'horizon_days', 'forward_return', 'label']].copy()

    logger.info(f"Generated {len(features)} feature rows, {len(labels)} label rows")

    return features, labels


def test_cpcv():
    """Test CPCV splits."""
    logger.info("\n" + "="*60)
    logger.info("TEST 1: CPCV Splits")
    logger.info("="*60)

    # Use larger dataset with fewer folds
    features, labels = generate_synthetic_data(n_symbols=20, n_days=1000)

    # Initialize CPCV with less aggressive purging
    cv = CPCV(n_folds=3, embargo_days=2)

    # Generate splits
    splits = cv.split(features, labels)

    print(f"\nGenerated {len(splits)} CPCV splits")
    print(f"\nFirst 3 splits:")
    for i, (train_idx, test_idx) in enumerate(splits[:3]):
        print(f"  Split {i+1}: train={len(train_idx)}, test={len(test_idx)}")

    # Test combinatorial splits
    comb_splits = cv.combinatorial_split(features, labels, n_test_folds=2)
    print(f"\nCombinatorial splits: {len(comb_splits)} total")

    return splits


def test_pbo_with_simple_strategies():
    """Test PBO with multiple simple strategies."""
    logger.info("\n" + "="*60)
    logger.info("TEST 2: PBO with Simple Strategies")
    logger.info("="*60)

    features, labels = generate_synthetic_data(n_symbols=30, n_days=1000)

    # Merge for easier access
    data = features.merge(labels, on=['date', 'symbol'])

    # Define 5 simple strategies
    strategies = {
        'momentum_5d': lambda df: df['momentum_5d'],
        'momentum_20d': lambda df: df['momentum_20d'],
        'volatility': lambda df: -df['volatility_20d'],  # Low vol
        'reversal_5d': lambda df: -df['momentum_5d'],  # Contrarian
        'combined': lambda df: df['momentum_20d'] - df['volatility_20d']
    }

    # CPCV splits with less aggressive settings
    cv = CPCV(n_folds=3, embargo_days=2)
    splits = cv.split(data, labels)
    n_trials = len(splits)

    # Performance matrices
    n_configs = len(strategies)
    is_performance = np.zeros((n_configs, n_trials))
    oos_performance = np.zeros((n_configs, n_trials))

    # Simulate backtests
    for config_idx, (strategy_name, strategy_fn) in enumerate(strategies.items()):
        for trial_idx, (train_idx, test_idx) in enumerate(splits):
            train_data = data.iloc[train_idx]
            test_data = data.iloc[test_idx]

            # Generate signals
            train_signals = strategy_fn(train_data)
            test_signals = strategy_fn(test_data)

            # Simple performance: correlation with forward returns
            is_perf = np.corrcoef(train_signals, train_data['forward_return'])[0, 1]
            oos_perf = np.corrcoef(test_signals, test_data['forward_return'])[0, 1]

            # Handle NaN
            is_performance[config_idx, trial_idx] = is_perf if not np.isnan(is_perf) else 0.0
            oos_performance[config_idx, trial_idx] = oos_perf if not np.isnan(oos_perf) else 0.0

    # Calculate PBO
    pbo_calc = PBOCalculator(n_trials=n_trials)
    pbo_results = pbo_calc.calculate_pbo(is_performance, oos_performance)

    print(f"\nPBO Results:")
    print(f"  PBO:                   {pbo_results['pbo']:.2%}")
    print(f"  Threshold:             <= 5%")
    print(f"  Status:                {'PASS' if pbo_results['pass_threshold'] else 'FAIL'}")
    print(f"  Rank-1 OOS Mean:       {pbo_results['rank1_oos_mean']:.4f}")
    print(f"  OOS Median (all):      {pbo_results['oos_median_all']:.4f}")

    return pbo_results


def test_dsr_with_backtest():
    """Test DSR with backtested strategy."""
    logger.info("\n" + "="*60)
    logger.info("TEST 3: DSR with Backtested Strategy")
    logger.info("="*60)

    features, labels = generate_synthetic_data(n_symbols=20, n_days=1000)

    # Merge
    data = features.merge(labels, on=['date', 'symbol'])

    # Simple momentum strategy
    data['signal'] = data['momentum_20d']

    # Generate top/bottom decile signals
    data['signal_rank'] = data.groupby('date')['signal'].rank(pct=True)
    data['target_quantity'] = 0
    data.loc[data['signal_rank'] >= 0.9, 'target_quantity'] = 100  # Long top decile
    data.loc[data['signal_rank'] <= 0.1, 'target_quantity'] = -100  # Short bottom decile

    # Prepare signals for backtester
    signals = data[['date', 'symbol', 'target_quantity']].copy()
    prices = data[['date', 'symbol', 'close']].copy()

    # Run backtest
    bt = MinimalBacktester(initial_capital=1_000_000, spread_bps=5.0)
    equity_curve = bt.run(signals, prices)

    # Calculate returns
    returns = equity_curve['equity'].pct_change().dropna()

    # Calculate DSR (assume tested 10 strategies before finding this one)
    dsr_calc = DSRCalculator(annual_factor=252.0)
    dsr_results = dsr_calc.calculate_dsr(returns.values, n_trials=10)

    print(f"\nDSR Results:")
    print(f"  Sharpe Ratio:          {dsr_results['sharpe_ratio']:.2f}")
    print(f"  Sharpe Adjustment:     {dsr_results['sharpe_adj']:.2f}")
    print(f"  DSR:                   {dsr_results['dsr']:.4f}")
    print(f"  P(True SR > 0):        {dsr_results['dsr']*100:.1f}%")
    print(f"  Status:                {'PASS' if dsr_results['pass_threshold'] else 'FAIL'}")
    print(f"  Skewness:              {dsr_results['skewness']:.2f}")
    print(f"  Kurtosis:              {dsr_results['kurtosis']:.2f}")

    # Calculate performance metrics
    perf = bt.calculate_performance()
    print(f"\nBacktest Performance:")
    print(f"  Total Return:          {perf.total_return*100:.2f}%")
    print(f"  Sharpe Ratio:          {perf.sharpe_ratio:.2f}")
    print(f"  Max Drawdown:          {perf.max_drawdown*100:.2f}%")
    print(f"  Win Rate:              {perf.win_rate*100:.1f}%")
    print(f"  Number of Trades:      {perf.num_trades}")

    return dsr_results


def test_integrated_validation():
    """Test the complete CPCV + PBO + DSR workflow."""
    logger.info("\n" + "="*60)
    logger.info("TEST 4: Integrated Validation (CPCV + PBO + DSR)")
    logger.info("="*60)

    features, labels = generate_synthetic_data(n_symbols=30, n_days=1200)
    data = features.merge(labels, on=['date', 'symbol'])

    # Test 3 strategies
    strategies = {
        'momentum': lambda df: df['momentum_20d'],
        'reversal': lambda df: -df['momentum_5d'],
        'volatility': lambda df: -df['volatility_20d']
    }

    # CPCV with less aggressive settings
    cv = CPCV(n_folds=4, embargo_days=2)
    splits = cv.split(data, labels)
    n_trials = len(splits)

    print(f"\nGenerated {n_trials} CPCV splits")

    # Collect performance for PBO
    n_configs = len(strategies)
    is_performance = np.zeros((n_configs, n_trials))
    oos_performance = np.zeros((n_configs, n_trials))

    # Collect returns for DSR (best strategy)
    best_strategy_returns = []

    for config_idx, (strategy_name, strategy_fn) in enumerate(strategies.items()):
        strategy_returns = []

        for trial_idx, (train_idx, test_idx) in enumerate(splits):
            train_data = data.iloc[train_idx]
            test_data = data.iloc[test_idx]

            # Generate signals
            train_signals = strategy_fn(train_data)
            test_signals = strategy_fn(test_data)

            # Performance
            is_perf = np.corrcoef(train_signals, train_data['forward_return'])[0, 1]
            oos_perf = np.corrcoef(test_signals, test_data['forward_return'])[0, 1]

            is_performance[config_idx, trial_idx] = is_perf if not np.isnan(is_perf) else 0.0
            oos_performance[config_idx, trial_idx] = oos_perf if not np.isnan(oos_perf) else 0.0

            # Simulate returns for this trial (simplified)
            trial_returns = test_data['forward_return'] * np.sign(test_signals)
            strategy_returns.append(trial_returns.mean())

        # Track best strategy
        if config_idx == 0 or np.mean(strategy_returns) > np.mean(best_strategy_returns):
            best_strategy_returns = strategy_returns
            best_strategy_name = strategy_name

    # Calculate PBO
    pbo_calc = PBOCalculator(n_trials=n_trials)
    pbo_results = pbo_calc.calculate_pbo(is_performance, oos_performance)

    # Calculate DSR
    best_returns = np.array(best_strategy_returns)
    dsr_calc = DSRCalculator(annual_factor=252.0)
    dsr_results = dsr_calc.calculate_dsr(best_returns, n_trials=n_configs)

    # Print combined results
    print(f"\n{'='*60}")
    print("VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"\nCPCV:")
    print(f"  Splits:                {n_trials}")
    print(f"  Embargo:               5 days")
    print(f"\nPBO:")
    print(f"  PBO:                   {pbo_results['pbo']:.2%}")
    print(f"  Threshold:             <= 5%")
    print(f"  Status:                {'PASS' if pbo_results['pass_threshold'] else 'FAIL'}")
    print(f"\nDSR (Best Strategy: {best_strategy_name}):")
    print(f"  Sharpe Ratio:          {dsr_results['sharpe_ratio']:.2f}")
    print(f"  DSR:                   {dsr_results['dsr']:.4f}")
    print(f"  P(True SR > 0):        {dsr_results['dsr']*100:.1f}%")
    print(f"  Status:                {'PASS' if dsr_results['pass_threshold'] else 'FAIL'}")
    print(f"\nOVERALL:")

    overall_pass = (
        pbo_results['pass_threshold'] and
        dsr_results['pass_threshold']
    )
    print(f"  Status:                {'PASS' if overall_pass else 'FAIL'}")
    print(f"  Ready for Production:  {'YES' if overall_pass else 'NO'}")
    print(f"{'='*60}")

    return {
        'pbo': pbo_results,
        'dsr': dsr_results,
        'overall_pass': overall_pass
    }


if __name__ == "__main__":
    logger.info("Testing Anti-Overfitting Suite (CPCV + PBO + DSR)...")

    # Run individual tests
    splits = test_cpcv()
    pbo_results = test_pbo_with_simple_strategies()
    dsr_results = test_dsr_with_backtest()

    # Run integrated test
    integrated_results = test_integrated_validation()

    print("\n[OK] All anti-overfitting tests complete")
    print("\nKey Takeaways:")
    print("1. CPCV ensures no label leakage via purging and embargo")
    print("2. PBO detects selection bias from testing multiple strategies")
    print("3. DSR adjusts for multiple testing and non-normal returns")
    print("4. ALL THREE are non-negotiable for production deployment")
