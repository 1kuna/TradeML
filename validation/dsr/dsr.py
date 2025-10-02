"""
Deflated Sharpe Ratio (DSR).

Based on "The Deflated Sharpe Ratio" by Bailey & López de Prado (2014).

Key concept:
- Adjust observed Sharpe ratio for multiple testing and non-normality
- Accounts for number of trials, skewness, and kurtosis
- Returns probability that true Sharpe > 0

Target: DSR > 0 (probability of skill > 50%)
"""

from datetime import datetime
from typing import Optional, Dict, List
import pandas as pd
import numpy as np
from loguru import logger
from scipy import stats


class DSRCalculator:
    """
    Calculate Deflated Sharpe Ratio.

    Implementation follows Bailey & López de Prado (2014):
    1. Estimate Sharpe ratio from returns
    2. Adjust for number of trials (strategies tested)
    3. Adjust for return distribution (skewness, kurtosis)
    4. Return DSR = P(True SR > 0)

    Target: DSR > 0 (probability of skill > 50%)
    Interpretation:
    - DSR > 0: Evidence of skill (true SR likely > 0)
    - DSR <= 0: No evidence of skill (could be luck)
    """

    def __init__(self, annual_factor: float = 252.0):
        """
        Initialize DSR calculator.

        Args:
            annual_factor: Annualization factor (252 for daily, 12 for monthly)
        """
        self.annual_factor = annual_factor

        logger.info(f"DSR calculator initialized (annualization factor: {annual_factor})")

    def calculate_sharpe_ratio(
        self,
        returns: np.ndarray,
        risk_free_rate: float = 0.0
    ) -> float:
        """
        Calculate annualized Sharpe ratio.

        Args:
            returns: Array of returns
            risk_free_rate: Risk-free rate (annualized)

        Returns:
            Annualized Sharpe ratio
        """
        if len(returns) < 2:
            return 0.0

        excess_returns = returns - (risk_free_rate / self.annual_factor)

        mean_return = np.mean(excess_returns)
        std_return = np.std(excess_returns, ddof=1)

        if std_return == 0:
            return 0.0

        sharpe = np.sqrt(self.annual_factor) * mean_return / std_return

        return sharpe

    def calculate_dsr(
        self,
        returns: np.ndarray,
        n_trials: int,
        risk_free_rate: float = 0.0,
        benchmark_sharpe: float = 0.0
    ) -> Dict[str, float]:
        """
        Calculate Deflated Sharpe Ratio.

        Args:
            returns: Array of returns
            n_trials: Number of strategies tested (trials)
            risk_free_rate: Risk-free rate (annualized)
            benchmark_sharpe: Benchmark Sharpe to compare against (default: 0)

        Returns:
            Dict with DSR and related statistics
        """
        n = len(returns)

        if n < 2:
            logger.error("Need at least 2 returns to calculate DSR")
            return {
                'dsr': 0.0,
                'sharpe_ratio': 0.0,
                'pass_threshold': False
            }

        # Calculate observed Sharpe ratio
        sharpe_obs = self.calculate_sharpe_ratio(returns, risk_free_rate)

        # Calculate return moments
        excess_returns = returns - (risk_free_rate / self.annual_factor)

        skewness = stats.skew(excess_returns)
        kurtosis = stats.kurtosis(excess_returns)  # Excess kurtosis

        # Adjust for higher moments
        # Variance of Sharpe ratio estimator
        var_sharpe = (1 + (sharpe_obs**2) / 2 - skewness * sharpe_obs + (kurtosis - 1) / 4 * sharpe_obs**2) / (n - 1)

        if var_sharpe <= 0:
            logger.warning("Negative variance estimate, using simple variance")
            var_sharpe = (1 + sharpe_obs**2 / 2) / (n - 1)

        std_sharpe = np.sqrt(var_sharpe)

        # Adjust for multiple testing
        # Expected maximum Sharpe under null hypothesis
        if n_trials > 1:
            # Euler-Mascheroni constant
            gamma = 0.5772156649

            # Expected maximum of N independent standard normals
            expected_max_z = (1 - gamma) * stats.norm.ppf(1 - 1.0 / n_trials) + gamma * stats.norm.ppf(1 - 1.0 / (n_trials * np.e))

            # Adjust for standard error
            sharpe_adj = std_sharpe * expected_max_z
        else:
            sharpe_adj = 0.0

        # Calculate DSR
        # DSR = P(True SR > benchmark_sharpe | observed data)
        if std_sharpe > 0:
            dsr = stats.norm.cdf((sharpe_obs - sharpe_adj - benchmark_sharpe) / std_sharpe)
        else:
            dsr = 0.0

        results = {
            'dsr': dsr,
            'sharpe_ratio': sharpe_obs,
            'sharpe_adj': sharpe_adj,
            'std_sharpe': std_sharpe,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'n_trials': n_trials,
            'n_observations': n,
            'benchmark_sharpe': benchmark_sharpe,
            'pass_threshold': dsr > 0.5  # DSR > 0.5 means P(True SR > 0) > 50%
        }

        logger.info(
            f"DSR calculated: {dsr:.4f} (SR={sharpe_obs:.2f}, adj={sharpe_adj:.2f}) "
            f"{'PASS' if results['pass_threshold'] else 'FAIL'}"
        )

        return results

    def calculate_dsr_from_equity_curve(
        self,
        equity_curve: pd.Series,
        n_trials: int,
        risk_free_rate: float = 0.0,
        benchmark_sharpe: float = 0.0
    ) -> Dict[str, float]:
        """
        Calculate DSR from equity curve.

        Args:
            equity_curve: Series with equity values over time
            n_trials: Number of strategies tested
            risk_free_rate: Risk-free rate (annualized)
            benchmark_sharpe: Benchmark Sharpe to compare against

        Returns:
            Dict with DSR and statistics
        """
        if len(equity_curve) < 2:
            logger.error("Need at least 2 equity values")
            return {
                'dsr': 0.0,
                'sharpe_ratio': 0.0,
                'pass_threshold': False
            }

        # Calculate returns from equity curve
        returns = equity_curve.pct_change().dropna().values

        return self.calculate_dsr(returns, n_trials, risk_free_rate, benchmark_sharpe)

    def calculate_minimum_track_record_length(
        self,
        target_sharpe: float,
        observed_sharpe: Optional[float] = None,
        returns: Optional[np.ndarray] = None,
        n_trials: int = 1,
        risk_free_rate: float = 0.0,
        confidence: float = 0.95
    ) -> int:
        """
        Calculate minimum track record length (MinTRL).

        MinTRL = minimum number of observations needed to be confident
        that true Sharpe >= target_sharpe.

        Args:
            target_sharpe: Target Sharpe ratio
            observed_sharpe: Observed Sharpe ratio (optional if returns provided)
            returns: Array of returns (optional if observed_sharpe provided)
            n_trials: Number of trials
            risk_free_rate: Risk-free rate
            confidence: Confidence level (default: 95%)

        Returns:
            Minimum number of observations needed
        """
        if observed_sharpe is None and returns is None:
            raise ValueError("Must provide either observed_sharpe or returns")

        if observed_sharpe is None:
            observed_sharpe = self.calculate_sharpe_ratio(returns, risk_free_rate)

        if returns is not None:
            excess_returns = returns - (risk_free_rate / self.annual_factor)
            skewness = stats.skew(excess_returns)
            kurtosis = stats.kurtosis(excess_returns)
        else:
            # Assume normal distribution
            skewness = 0.0
            kurtosis = 0.0

        # Critical z-value for confidence level
        z_alpha = stats.norm.ppf(confidence)

        # Adjust for multiple testing
        if n_trials > 1:
            gamma = 0.5772156649
            expected_max_z = (1 - gamma) * stats.norm.ppf(1 - 1.0 / n_trials) + gamma * stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
        else:
            expected_max_z = 0.0

        # Calculate MinTRL
        # Solve for n in variance formula
        variance_factor = 1 + (observed_sharpe**2) / 2 - skewness * observed_sharpe + (kurtosis - 1) / 4 * observed_sharpe**2

        numerator = variance_factor * (z_alpha / (observed_sharpe - expected_max_z - target_sharpe))**2

        min_trl = int(np.ceil(numerator)) + 1

        logger.info(
            f"MinTRL: {min_trl} observations needed for SR={observed_sharpe:.2f} "
            f"to exceed {target_sharpe:.2f} with {confidence*100:.0f}% confidence"
        )

        return min_trl


def calculate_dsr(
    returns: np.ndarray,
    n_trials: int,
    risk_free_rate: float = 0.0,
    benchmark_sharpe: float = 0.0,
    annual_factor: float = 252.0
) -> float:
    """
    Convenience function to calculate DSR.

    Args:
        returns: Array of returns
        n_trials: Number of strategies tested
        risk_free_rate: Risk-free rate (annualized)
        benchmark_sharpe: Benchmark Sharpe to compare against
        annual_factor: Annualization factor

    Returns:
        DSR value
    """
    calc = DSRCalculator(annual_factor=annual_factor)
    results = calc.calculate_dsr(returns, n_trials, risk_free_rate, benchmark_sharpe)

    return results['dsr']


# CLI for testing
if __name__ == "__main__":
    logger.info("Testing DSR calculator...")

    # Test 1: Good strategy (positive Sharpe, low trials)
    print("\n" + "="*60)
    print("TEST 1: Good Strategy (SR=1.5, 1 trial)")
    print("="*60)

    np.random.seed(42)
    n_obs = 252  # 1 year daily

    # Simulate returns with SR=1.5
    target_sharpe = 1.5
    daily_mean = target_sharpe * 0.15 / np.sqrt(252)  # 15% annual vol
    daily_std = 0.15 / np.sqrt(252)

    returns_good = np.random.normal(daily_mean, daily_std, n_obs)

    calc = DSRCalculator(annual_factor=252.0)
    results_good = calc.calculate_dsr(returns_good, n_trials=1)

    print(f"Sharpe Ratio:          {results_good['sharpe_ratio']:.2f}")
    print(f"DSR:                   {results_good['dsr']:.4f}")
    print(f"P(True SR > 0):        {results_good['dsr']*100:.1f}%")
    print(f"Status:                {'PASS' if results_good['pass_threshold'] else 'FAIL'}")

    # Test 2: Mediocre strategy with multiple trials (overfitting)
    print("\n" + "="*60)
    print("TEST 2: Overfitted Strategy (SR=0.8, 100 trials)")
    print("="*60)

    # Simulate returns with SR=0.8 (but tested 100 strategies)
    target_sharpe = 0.8
    daily_mean = target_sharpe * 0.15 / np.sqrt(252)
    daily_std = 0.15 / np.sqrt(252)

    returns_overfit = np.random.normal(daily_mean, daily_std, n_obs)

    results_overfit = calc.calculate_dsr(returns_overfit, n_trials=100)

    print(f"Sharpe Ratio:          {results_overfit['sharpe_ratio']:.2f}")
    print(f"Sharpe Adjustment:     {results_overfit['sharpe_adj']:.2f}")
    print(f"DSR:                   {results_overfit['dsr']:.4f}")
    print(f"P(True SR > 0):        {results_overfit['dsr']*100:.1f}%")
    print(f"Status:                {'PASS' if results_overfit['pass_threshold'] else 'FAIL'}")

    # Test 3: Strategy with non-normal returns
    print("\n" + "="*60)
    print("TEST 3: Non-Normal Returns (SR=1.0, skew=-0.5, kurtosis=3)")
    print("="*60)

    # Simulate skewed, fat-tailed returns
    returns_nonnormal = np.random.standard_t(df=5, size=n_obs) * 0.01 + 0.001

    results_nonnormal = calc.calculate_dsr(returns_nonnormal, n_trials=10)

    print(f"Sharpe Ratio:          {results_nonnormal['sharpe_ratio']:.2f}")
    print(f"Skewness:              {results_nonnormal['skewness']:.2f}")
    print(f"Kurtosis:              {results_nonnormal['kurtosis']:.2f}")
    print(f"DSR:                   {results_nonnormal['dsr']:.4f}")
    print(f"P(True SR > 0):        {results_nonnormal['dsr']*100:.1f}%")
    print(f"Status:                {'PASS' if results_nonnormal['pass_threshold'] else 'FAIL'}")

    # Test 4: Minimum Track Record Length
    print("\n" + "="*60)
    print("TEST 4: Minimum Track Record Length")
    print("="*60)

    observed_sharpe = 1.2
    target_sharpe = 0.5

    min_trl = calc.calculate_minimum_track_record_length(
        target_sharpe=target_sharpe,
        observed_sharpe=observed_sharpe,
        n_trials=10,
        confidence=0.95
    )

    print(f"Observed Sharpe:       {observed_sharpe:.2f}")
    print(f"Target Sharpe:         {target_sharpe:.2f}")
    print(f"Number of Trials:      10")
    print(f"Confidence:            95%")
    print(f"MinTRL:                {min_trl} observations ({min_trl/252:.1f} years)")

    # Test 5: Compare different trial counts
    print("\n" + "="*60)
    print("TEST 5: Effect of Multiple Testing")
    print("="*60)

    # Fixed returns with SR ~ 1.0
    np.random.seed(123)
    returns_fixed = np.random.normal(0.0005, 0.01, 252)

    print(f"{'Trials':<10} {'Sharpe':<10} {'Adj':<10} {'DSR':<10} {'P(SR>0)':<10}")
    print("-" * 60)

    for n_trials in [1, 5, 10, 50, 100, 500]:
        res = calc.calculate_dsr(returns_fixed, n_trials=n_trials)
        print(f"{n_trials:<10} {res['sharpe_ratio']:<10.2f} {res['sharpe_adj']:<10.2f} "
              f"{res['dsr']:<10.4f} {res['dsr']*100:<10.1f}%")

    print("\n[OK] DSR calculator test complete")
