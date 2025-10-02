"""
SVI (Stochastic Volatility Inspired) surface fitting.

Implements:
- SVI parameterization for IV smiles
- No-arbitrage constraints
- Surface quality checks

References:
- Gatheral, J. (2004). A parsimonious arbitrage-free implied volatility parameterization.
- Gatheral, J., & Jacquier, A. (2014). Arbitrage-free SVI volatility surfaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from loguru import logger


@dataclass
class SVIParams:
    """SVI slice parameters."""

    a: float  # Level parameter
    b: float  # Angle parameter (>= 0)
    rho: float  # Correlation parameter (in [-1, 1])
    m: float  # Translation parameter (ATM location)
    sigma: float  # Scale parameter (>= 0)

    def total_variance(self, k: float) -> float:
        """
        Calculate total variance at log-moneyness k.

        SVI formula:
            w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))

        where w(k) = sigma_BS^2 * T (total variance)
        """
        return self.a + self.b * (self.rho * (k - self.m) + np.sqrt((k - self.m) ** 2 + self.sigma**2))

    def implied_vol(self, k: float, T: float) -> float:
        """
        Calculate implied volatility at log-moneyness k and maturity T.

        Returns
        -------
        float
            Implied volatility (annualized)
        """
        w = self.total_variance(k)
        return np.sqrt(max(w / T, 0)) if T > 0 else 0.0


@dataclass
class SVIQualityMetrics:
    """Quality metrics for SVI fit."""

    rmse: float  # Root mean squared error
    max_error: float  # Maximum absolute error
    has_butterfly_arb: bool  # Butterfly arbitrage check
    has_calendar_arb: bool  # Calendar arbitrage check


class SVICalibrator:
    """
    SVI surface calibrator with no-arbitrage constraints.

    Calibrates SVI parameters to market IV smiles with arbitrage checks.
    """

    def __init__(self, max_iterations: int = 1000, tolerance: float = 1e-6):
        """
        Initialize SVI calibrator.

        Parameters
        ----------
        max_iterations : int
            Maximum optimization iterations
        tolerance : float
            Optimization tolerance
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def fit_slice(
        self,
        strikes: np.ndarray,
        spot: float,
        ivs: np.ndarray,
        T: float,
        weights: Optional[np.ndarray] = None,
    ) -> tuple[SVIParams, SVIQualityMetrics]:
        """
        Fit SVI parameters to a single maturity slice.

        Parameters
        ----------
        strikes : np.ndarray
            Strike prices
        spot : float
            Spot price
        ivs : np.ndarray
            Implied volatilities (annualized)
        T : float
            Time to maturity (years)
        weights : Optional[np.ndarray]
            Weights for each data point (default: equal weights)

        Returns
        -------
        tuple[SVIParams, SVIQualityMetrics]
            Fitted parameters and quality metrics
        """
        if len(strikes) != len(ivs):
            raise ValueError("strikes and ivs must have same length")

        if T <= 0:
            raise ValueError("T must be positive")

        # Convert to log-moneyness
        k = np.log(strikes / spot)

        # Convert IV to total variance
        w_market = ivs**2 * T

        # Initial guess using simple heuristics
        atm_idx = np.argmin(np.abs(k))
        a_init = w_market[atm_idx] * 0.5
        b_init = 0.1
        rho_init = 0.0
        m_init = k[atm_idx]
        sigma_init = 0.1

        x0 = np.array([a_init, b_init, rho_init, m_init, sigma_init])

        # Weights
        if weights is None:
            weights = np.ones_like(k)

        # Objective function: weighted RMSE
        def objective(x):
            a, b, rho, m, sigma = x
            params = SVIParams(a=a, b=b, rho=rho, m=m, sigma=sigma)
            w_model = np.array([params.total_variance(ki) for ki in k])
            residuals = (w_model - w_market) * weights
            return np.sqrt(np.mean(residuals**2))

        # Constraints: b >= 0, -1 <= rho <= 1, sigma >= 0
        bounds = [(None, None), (0, None), (-0.999, 0.999), (None, None), (1e-6, None)]

        # Optimize
        result = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": self.max_iterations, "ftol": self.tolerance},
        )

        if not result.success:
            logger.warning(f"SVI optimization did not converge: {result.message}")

        a, b, rho, m, sigma = result.x
        params = SVIParams(a=a, b=b, rho=rho, m=m, sigma=sigma)

        # Calculate quality metrics
        w_fitted = np.array([params.total_variance(ki) for ki in k])
        errors = w_fitted - w_market
        rmse = np.sqrt(np.mean(errors**2))
        max_error = np.max(np.abs(errors))

        # Arbitrage checks
        has_butterfly_arb = self._check_butterfly_arbitrage(params, k)
        has_calendar_arb = False  # Would need multiple maturities to check

        metrics = SVIQualityMetrics(
            rmse=rmse,
            max_error=max_error,
            has_butterfly_arb=has_butterfly_arb,
            has_calendar_arb=has_calendar_arb,
        )

        return params, metrics

    def _check_butterfly_arbitrage(self, params: SVIParams, k_grid: np.ndarray, n_points: int = 100) -> bool:
        """
        Check for butterfly arbitrage (density must be non-negative).

        Condition: d^2(w)/dk^2 >= 0 everywhere

        Parameters
        ----------
        params : SVIParams
            SVI parameters
        k_grid : np.ndarray
            Grid of log-moneyness points to check
        n_points : int
            Number of points to check

        Returns
        -------
        bool
            True if butterfly arbitrage detected
        """
        # Dense grid for checking
        k_min, k_max = k_grid.min() - 0.5, k_grid.max() + 0.5
        k_check = np.linspace(k_min, k_max, n_points)

        # Numerical second derivative
        dk = k_check[1] - k_check[0]
        w = np.array([params.total_variance(ki) for ki in k_check])
        d2w_dk2 = np.gradient(np.gradient(w, dk), dk)

        # Check if any second derivative is significantly negative
        return np.any(d2w_dk2 < -1e-6)


def fit_svi_slice(
    strikes: np.ndarray,
    spot: float,
    ivs: np.ndarray,
    T: float,
    weights: Optional[np.ndarray] = None,
) -> Dict:
    """
    Convenience function to fit SVI to a single maturity slice.

    Parameters
    ----------
    strikes : np.ndarray
        Strike prices
    spot : float
        Spot price
    ivs : np.ndarray
        Implied volatilities
    T : float
        Time to maturity (years)
    weights : Optional[np.ndarray]
        Observation weights

    Returns
    -------
    Dict
        {
            "params": SVIParams,
            "metrics": SVIQualityMetrics,
            "fit_successful": bool
        }
    """
    calibrator = SVICalibrator()

    try:
        params, metrics = calibrator.fit_slice(strikes, spot, ivs, T, weights)
        return {"params": params, "metrics": metrics, "fit_successful": True}
    except Exception as e:
        logger.error(f"SVI fitting failed: {e}")
        return {"params": None, "metrics": None, "fit_successful": False}
