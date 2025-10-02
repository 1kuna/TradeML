"""
Black-Scholes implied volatility calculator.

Implements:
- Black-Scholes pricing formula
- Newton-Raphson IV solver
- Greeks calculation (Delta, Gamma, Vega, Theta, Rho)
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.optimize import brentq
from dataclasses import dataclass
from typing import Optional, Literal


@dataclass
class Greeks:
    """Option Greeks."""

    delta: float  # Rate of change of option price with respect to underlying
    gamma: float  # Rate of change of delta with respect to underlying
    vega: float  # Rate of change of option price with respect to volatility
    theta: float  # Rate of change of option price with respect to time
    rho: float  # Rate of change of option price with respect to interest rate


class BlackScholesIV:
    """
    Black-Scholes implied volatility and Greeks calculator.

    References:
    - Black, F., & Scholes, M. (1973). The pricing of options and corporate liabilities.
    - Hull, J. C. (2018). Options, Futures, and Other Derivatives.
    """

    @staticmethod
    def d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d1 term in Black-Scholes formula."""
        return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    @staticmethod
    def d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d2 term in Black-Scholes formula."""
        return BlackScholesIV.d1(S, K, T, r, sigma) - sigma * np.sqrt(T)

    @staticmethod
    def call_price(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        """
        Calculate Black-Scholes call option price.

        Parameters
        ----------
        S : float
            Spot price
        K : float
            Strike price
        T : float
            Time to expiration (years)
        r : float
            Risk-free rate (annualized)
        sigma : float
            Volatility (annualized)
        q : float
            Dividend yield (annualized, default 0)

        Returns
        -------
        float
            Call option price
        """
        if T <= 0:
            return max(S - K, 0)

        d1 = BlackScholesIV.d1(S, K, T, r, sigma)
        d2 = BlackScholesIV.d2(S, K, T, r, sigma)

        return S * np.exp(-q * T) * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)

    @staticmethod
    def put_price(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        """
        Calculate Black-Scholes put option price.

        Parameters
        ----------
        S : float
            Spot price
        K : float
            Strike price
        T : float
            Time to expiration (years)
        r : float
            Risk-free rate (annualized)
        sigma : float
            Volatility (annualized)
        q : float
            Dividend yield (annualized, default 0)

        Returns
        -------
        float
            Put option price
        """
        if T <= 0:
            return max(K - S, 0)

        d1 = BlackScholesIV.d1(S, K, T, r, sigma)
        d2 = BlackScholesIV.d2(S, K, T, r, sigma)

        return K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * np.exp(-q * T) * stats.norm.cdf(-d1)

    @staticmethod
    def vega(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        """
        Calculate vega (sensitivity to volatility).

        Returns
        -------
        float
            Vega (in dollars per 1% change in volatility)
        """
        if T <= 0:
            return 0.0

        d1 = BlackScholesIV.d1(S, K, T, r, sigma)
        return S * np.exp(-q * T) * stats.norm.pdf(d1) * np.sqrt(T)

    @staticmethod
    def implied_volatility(
        option_price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        option_type: Literal["call", "put"] = "call",
        q: float = 0.0,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
    ) -> Optional[float]:
        """
        Calculate implied volatility using Brent's method.

        Parameters
        ----------
        option_price : float
            Observed market price of the option
        S : float
            Spot price
        K : float
            Strike price
        T : float
            Time to expiration (years)
        r : float
            Risk-free rate (annualized)
        option_type : Literal["call", "put"]
            Type of option
        q : float
            Dividend yield (annualized)
        max_iterations : int
            Maximum iterations for solver
        tolerance : float
            Convergence tolerance

        Returns
        -------
        Optional[float]
            Implied volatility (annualized), or None if solver fails
        """
        if T <= 0:
            return None

        # Intrinsic value
        if option_type == "call":
            intrinsic = max(S - K, 0)
            pricing_func = BlackScholesIV.call_price
        else:
            intrinsic = max(K - S, 0)
            pricing_func = BlackScholesIV.put_price

        # Check if price is below intrinsic value
        if option_price < intrinsic:
            return None

        # Objective function: difference between model price and market price
        def objective(sigma):
            if sigma <= 0:
                return 1e10
            try:
                model_price = pricing_func(S, K, T, r, sigma, q)
                return model_price - option_price
            except:
                return 1e10

        # Use Brent's method (robust root finder)
        try:
            iv = brentq(objective, 1e-6, 5.0, maxiter=max_iterations, xtol=tolerance)
            return iv if iv > 0 else None
        except:
            return None

    @staticmethod
    def calculate_greeks(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: Literal["call", "put"] = "call",
        q: float = 0.0,
    ) -> Greeks:
        """
        Calculate all Greeks for an option.

        Parameters
        ----------
        S : float
            Spot price
        K : float
            Strike price
        T : float
            Time to expiration (years)
        r : float
            Risk-free rate
        sigma : float
            Volatility
        option_type : Literal["call", "put"]
            Option type
        q : float
            Dividend yield

        Returns
        -------
        Greeks
            All Greeks
        """
        if T <= 0:
            return Greeks(delta=0.0, gamma=0.0, vega=0.0, theta=0.0, rho=0.0)

        d1 = BlackScholesIV.d1(S, K, T, r, sigma)
        d2 = BlackScholesIV.d2(S, K, T, r, sigma)

        # Common terms
        pdf_d1 = stats.norm.pdf(d1)
        cdf_d1 = stats.norm.cdf(d1)
        cdf_d2 = stats.norm.cdf(d2)

        # Delta
        if option_type == "call":
            delta = np.exp(-q * T) * cdf_d1
        else:
            delta = -np.exp(-q * T) * stats.norm.cdf(-d1)

        # Gamma (same for call and put)
        gamma = (np.exp(-q * T) * pdf_d1) / (S * sigma * np.sqrt(T))

        # Vega (same for call and put, in dollars per 1% vol change)
        vega = S * np.exp(-q * T) * pdf_d1 * np.sqrt(T)

        # Theta (time decay)
        if option_type == "call":
            theta = (
                -S * pdf_d1 * sigma * np.exp(-q * T) / (2 * np.sqrt(T))
                + q * S * cdf_d1 * np.exp(-q * T)
                - r * K * np.exp(-r * T) * cdf_d2
            ) / 365.0  # Convert to daily
        else:
            theta = (
                -S * pdf_d1 * sigma * np.exp(-q * T) / (2 * np.sqrt(T))
                - q * S * stats.norm.cdf(-d1) * np.exp(-q * T)
                + r * K * np.exp(-r * T) * stats.norm.cdf(-d2)
            ) / 365.0

        # Rho (interest rate sensitivity)
        if option_type == "call":
            rho = K * T * np.exp(-r * T) * cdf_d2 / 100.0  # Per 1% change in r
        else:
            rho = -K * T * np.exp(-r * T) * stats.norm.cdf(-d2) / 100.0

        return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)


def calculate_iv_from_price(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: Literal["call", "put"] = "call",
    q: float = 0.0,
) -> Optional[float]:
    """
    Convenience function to calculate IV from option price.

    Parameters
    ----------
    price : float
        Market price
    S : float
        Spot price
    K : float
        Strike
    T : float
        Time to expiration (years)
    r : float
        Risk-free rate
    option_type : Literal["call", "put"]
        Option type
    q : float
        Dividend yield

    Returns
    -------
    Optional[float]
        Implied volatility, or None if calculation fails
    """
    return BlackScholesIV.implied_volatility(price, S, K, T, r, option_type, q)
