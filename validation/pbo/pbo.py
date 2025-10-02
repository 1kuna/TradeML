"""
Probability of Backtest Overfitting (PBO).

Based on "The Probability of Backtest Overfitting" by Bailey et al. (2015).

Key concept:
- Train N configurations on M trials (e.g., CPCV splits)
- Rank configurations by IS performance
- Test OOS performance of top-ranked configurations
- PBO = probability that best IS performer underperforms OOS median

Target: PBO <= 5% for production deployment
"""

from datetime import datetime
from typing import List, Dict, Tuple, Optional, Callable
import pandas as pd
import numpy as np
from loguru import logger
from scipy import stats


class PBOCalculator:
    """
    Calculate Probability of Backtest Overfitting.

    Implementation follows Bailey et al. (2015) methodology:
    1. Run N configurations on M trials (CPCV splits)
    2. For each trial, rank configurations by IS performance
    3. Compute OOS performance of rank-1 IS configuration
    4. PBO = P(OOS_rank1 < median(OOS_all))

    Target: PBO <= 5% (strict threshold for production)
    """

    def __init__(self, n_trials: int = 16, metric_name: str = "sharpe_ratio"):
        """
        Initialize PBO calculator.

        Args:
            n_trials: Number of trials (CPCV splits)
            metric_name: Performance metric to use (sharpe_ratio, total_return, etc.)
        """
        self.n_trials = n_trials
        self.metric_name = metric_name

        logger.info(f"PBO calculator initialized: {n_trials} trials, metric={metric_name}")

    def calculate_pbo(
        self,
        is_performance: np.ndarray,
        oos_performance: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate PBO from IS and OOS performance matrices.

        Args:
            is_performance: (n_configs, n_trials) matrix of IS performance
            oos_performance: (n_configs, n_trials) matrix of OOS performance

        Returns:
            Dict with PBO and related statistics
        """
        n_configs, n_trials = is_performance.shape

        if oos_performance.shape != is_performance.shape:
            raise ValueError(f"IS and OOS shapes must match: {is_performance.shape} vs {oos_performance.shape}")

        if n_trials < self.n_trials:
            logger.warning(f"Only {n_trials} trials provided, expected {self.n_trials}")

        # For each trial, find the rank-1 IS configuration
        rank1_oos_performance = []

        for trial_idx in range(n_trials):
            is_scores = is_performance[:, trial_idx]
            oos_scores = oos_performance[:, trial_idx]

            # Find best IS configuration
            best_is_idx = np.argmax(is_scores)

            # Get its OOS performance
            rank1_oos = oos_scores[best_is_idx]
            rank1_oos_performance.append(rank1_oos)

        rank1_oos_performance = np.array(rank1_oos_performance)

        # Compute OOS median across all configs and trials
        oos_median = np.median(oos_performance)

        # PBO = probability that rank-1 OOS < median OOS
        pbo = np.mean(rank1_oos_performance < oos_median)

        # Compute additional statistics
        results = {
            'pbo': pbo,
            'rank1_oos_mean': np.mean(rank1_oos_performance),
            'rank1_oos_std': np.std(rank1_oos_performance),
            'rank1_oos_median': np.median(rank1_oos_performance),
            'oos_median_all': oos_median,
            'oos_mean_all': np.mean(oos_performance),
            'n_configs': n_configs,
            'n_trials': n_trials,
            'pass_threshold': pbo <= 0.05
        }

        logger.info(
            f"PBO calculated: {pbo:.2%} "
            f"({'PASS' if results['pass_threshold'] else 'FAIL'} <= 5% threshold)"
        )

        return results

    def calculate_pbo_from_backtest_results(
        self,
        backtest_results: List[Dict[str, Dict[str, float]]],
        metric_name: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Calculate PBO from list of backtest results.

        Args:
            backtest_results: List of dicts with structure:
                {
                    'config_0': {'is': 0.5, 'oos': 0.3},
                    'config_1': {'is': 0.6, 'oos': 0.2},
                    ...
                }
            metric_name: Override metric name

        Returns:
            Dict with PBO and statistics
        """
        if metric_name is None:
            metric_name = self.metric_name

        n_trials = len(backtest_results)

        if n_trials == 0:
            raise ValueError("No backtest results provided")

        # Extract configuration names
        config_names = list(backtest_results[0].keys())
        n_configs = len(config_names)

        # Build performance matrices
        is_performance = np.zeros((n_configs, n_trials))
        oos_performance = np.zeros((n_configs, n_trials))

        for trial_idx, trial_results in enumerate(backtest_results):
            for config_idx, config_name in enumerate(config_names):
                if config_name not in trial_results:
                    raise ValueError(f"Config {config_name} missing in trial {trial_idx}")

                config_results = trial_results[config_name]

                if 'is' not in config_results or 'oos' not in config_results:
                    raise ValueError(f"Missing IS/OOS results for {config_name} in trial {trial_idx}")

                is_performance[config_idx, trial_idx] = config_results['is']
                oos_performance[config_idx, trial_idx] = config_results['oos']

        return self.calculate_pbo(is_performance, oos_performance)

    def plot_pbo_distribution(
        self,
        is_performance: np.ndarray,
        oos_performance: np.ndarray,
        output_path: Optional[str] = None
    ):
        """
        Plot PBO distribution (requires matplotlib).

        Args:
            is_performance: (n_configs, n_trials) matrix
            oos_performance: (n_configs, n_trials) matrix
            output_path: Path to save plot (optional)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not installed, skipping plot")
            return

        n_configs, n_trials = is_performance.shape

        # Calculate rank-1 OOS performance
        rank1_oos = []
        for trial_idx in range(n_trials):
            best_is_idx = np.argmax(is_performance[:, trial_idx])
            rank1_oos.append(oos_performance[best_is_idx, trial_idx])

        rank1_oos = np.array(rank1_oos)
        oos_median = np.median(oos_performance)

        # Create plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Left: histogram of rank-1 OOS
        axes[0].hist(rank1_oos, bins=20, alpha=0.7, edgecolor='black')
        axes[0].axvline(oos_median, color='red', linestyle='--',
                       label=f'OOS Median (all): {oos_median:.3f}')
        axes[0].axvline(np.median(rank1_oos), color='blue', linestyle='--',
                       label=f'Rank-1 OOS Median: {np.median(rank1_oos):.3f}')
        axes[0].set_xlabel('OOS Performance')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Rank-1 OOS Distribution')
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Right: IS vs OOS scatter
        for trial_idx in range(n_trials):
            is_scores = is_performance[:, trial_idx]
            oos_scores = oos_performance[:, trial_idx]
            best_is_idx = np.argmax(is_scores)

            axes[1].scatter(is_scores, oos_scores, alpha=0.3, s=20, color='gray')
            axes[1].scatter(is_scores[best_is_idx], oos_scores[best_is_idx],
                          color='red', s=100, marker='*', zorder=10)

        axes[1].plot([is_performance.min(), is_performance.max()],
                    [is_performance.min(), is_performance.max()],
                    'k--', alpha=0.5, label='IS = OOS')
        axes[1].set_xlabel('IS Performance')
        axes[1].set_ylabel('OOS Performance')
        axes[1].set_title('IS vs OOS Performance (red stars = rank-1 IS)')
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"PBO plot saved to {output_path}")
        else:
            plt.show()

        plt.close()


def calculate_pbo(
    is_performance: np.ndarray,
    oos_performance: np.ndarray,
    n_trials: Optional[int] = None
) -> float:
    """
    Convenience function to calculate PBO.

    Args:
        is_performance: (n_configs, n_trials) matrix
        oos_performance: (n_configs, n_trials) matrix
        n_trials: Number of trials (optional, inferred from data)

    Returns:
        PBO value
    """
    if n_trials is None:
        n_trials = is_performance.shape[1]

    calc = PBOCalculator(n_trials=n_trials)
    results = calc.calculate_pbo(is_performance, oos_performance)

    return results['pbo']


def run_pbo_test(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    param_grid: Dict[str, List],
    cv_splitter,
    metric_fn: Callable
) -> Dict[str, float]:
    """
    Run PBO test on model with parameter grid.

    Args:
        model: Sklearn-compatible model
        X_train: Training features (all IS data)
        y_train: Training labels
        X_test: Test features (all OOS data)
        y_test: Test labels
        param_grid: Dict of parameter lists to test
        cv_splitter: Cross-validation splitter (e.g., CPCV)
        metric_fn: Scoring function

    Returns:
        Dict with PBO and statistics
    """
    from sklearn.model_selection import ParameterGrid

    logger.info("Running PBO test...")

    # Generate all parameter combinations
    configs = list(ParameterGrid(param_grid))
    n_configs = len(configs)

    logger.info(f"Testing {n_configs} configurations")

    # Get CV splits
    splits = list(cv_splitter.split(X_train, y_train))
    n_trials = len(splits)

    logger.info(f"Using {n_trials} CV splits")

    # Performance matrices
    is_performance = np.zeros((n_configs, n_trials))
    oos_performance = np.zeros((n_configs, n_trials))

    for config_idx, params in enumerate(configs):
        logger.debug(f"Config {config_idx+1}/{n_configs}: {params}")

        for trial_idx, (train_idx, val_idx) in enumerate(splits):
            # Split data
            X_tr = X_train.iloc[train_idx]
            y_tr = y_train.iloc[train_idx]
            X_val = X_train.iloc[val_idx]
            y_val = y_train.iloc[val_idx]

            # Train model with params
            model.set_params(**params)
            model.fit(X_tr, y_tr)

            # IS performance (on validation fold)
            y_pred_val = model.predict(X_val)
            is_performance[config_idx, trial_idx] = metric_fn(y_val, y_pred_val)

            # OOS performance (on held-out test set)
            y_pred_test = model.predict(X_test)
            oos_performance[config_idx, trial_idx] = metric_fn(y_test, y_pred_test)

    # Calculate PBO
    calc = PBOCalculator(n_trials=n_trials)
    results = calc.calculate_pbo(is_performance, oos_performance)

    logger.info(
        f"PBO Test Complete: PBO={results['pbo']:.2%} "
        f"({'PASS' if results['pass_threshold'] else 'FAIL'})"
    )

    return results


# CLI for testing
if __name__ == "__main__":
    logger.info("Testing PBO calculator...")

    # Simulate backtest results
    # 10 configurations, 16 trials
    np.random.seed(42)

    n_configs = 10
    n_trials = 16

    # Generate correlated IS/OOS performance
    # Higher IS often means lower OOS (overfitting)
    is_performance = np.random.randn(n_configs, n_trials) * 0.5 + 1.0

    # OOS performance is lower and negatively correlated with IS
    oos_performance = np.zeros_like(is_performance)
    for i in range(n_configs):
        for j in range(n_trials):
            # Negative correlation with IS + noise
            oos_performance[i, j] = -0.3 * is_performance[i, j] + np.random.randn() * 0.3 + 0.8

    # Calculate PBO
    calc = PBOCalculator(n_trials=n_trials)
    results = calc.calculate_pbo(is_performance, oos_performance)

    print(f"\n{'='*60}")
    print("PBO TEST RESULTS")
    print(f"{'='*60}")
    print(f"PBO:                   {results['pbo']:.2%}")
    print(f"Threshold:             <= 5%")
    print(f"Status:                {'PASS' if results['pass_threshold'] else 'FAIL'}")
    print(f"\nRank-1 OOS Mean:       {results['rank1_oos_mean']:.4f}")
    print(f"Rank-1 OOS Median:     {results['rank1_oos_median']:.4f}")
    print(f"Rank-1 OOS Std:        {results['rank1_oos_std']:.4f}")
    print(f"\nOOS Median (all):      {results['oos_median_all']:.4f}")
    print(f"OOS Mean (all):        {results['oos_mean_all']:.4f}")
    print(f"\nConfigurations:        {results['n_configs']}")
    print(f"Trials:                {results['n_trials']}")
    print(f"{'='*60}")

    # Test with backtest results format
    print("\n[OK] Testing with backtest results format...")

    backtest_results = []
    for trial_idx in range(n_trials):
        trial_results = {}
        for config_idx in range(n_configs):
            trial_results[f'config_{config_idx}'] = {
                'is': is_performance[config_idx, trial_idx],
                'oos': oos_performance[config_idx, trial_idx]
            }
        backtest_results.append(trial_results)

    results2 = calc.calculate_pbo_from_backtest_results(backtest_results)

    print(f"PBO (from backtest results): {results2['pbo']:.2%}")
    print(f"Match: {abs(results['pbo'] - results2['pbo']) < 1e-10}")

    print("\n[OK] PBO calculator test complete")
