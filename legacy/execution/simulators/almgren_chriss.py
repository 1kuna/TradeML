"""
Almgren-Chriss optimal execution scheduler.

Implements the optimal trading trajectory that balances:
- Market impact costs (trade faster → higher impact)
- Risk/volatility costs (trade slower → more price risk)

References:
- Almgren, R., & Chriss, N. (2001). Optimal execution of portfolio transactions.
  Journal of Risk, 3, 5-40.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class ACConfig:
    """Configuration for Almgren-Chriss optimal execution."""

    # Risk aversion parameter (lambda)
    risk_aversion: float = 1e-6  # Low risk aversion (focus on minimizing impact)

    # Impact parameters
    eta: float = 0.10  # Temporary impact coefficient
    gamma: float = 0.50  # Permanent impact fraction

    # Volatility estimate (for risk cost)
    volatility: Optional[float] = None  # If None, estimated from historical data

    # Execution constraints
    max_participation_rate: float = 0.10  # Max 10% of ADV per interval
    min_trade_size: float = 1.0  # Min shares to trade in an interval


class AlmgrenChrissScheduler:
    """
    Almgren-Chriss optimal execution scheduler.

    Computes the optimal trading trajectory that minimizes:
        E[Cost] + lambda * Var[Cost]

    where:
    - E[Cost] = market impact costs (temporary + permanent)
    - Var[Cost] = variance of execution cost due to price uncertainty
    - lambda = risk aversion parameter

    The optimal strategy is a linear decay with exponential adjustment
    based on risk aversion.
    """

    def __init__(self, config: Optional[ACConfig] = None):
        """
        Initialize Almgren-Chriss scheduler.

        Parameters
        ----------
        config : Optional[ACConfig]
            Configuration (uses defaults if None)
        """
        self.cfg = config or ACConfig()

    def compute_schedule(
        self,
        total_shares: float,
        adv: float,
        price: float,
        volatility: Optional[float] = None,
        n_intervals: int = 10,
        T: float = 1.0,
    ) -> pd.DataFrame:
        """
        Compute optimal execution schedule.

        Parameters
        ----------
        total_shares : float
            Total shares to execute (positive = buy, negative = sell)
        adv : float
            Average daily volume (shares)
        price : float
            Current price per share
        volatility : Optional[float]
            Price volatility (annualized). If None, uses config default or 0.20.
        n_intervals : int
            Number of trading intervals (e.g., 10 for 10 equal time slices)
        T : float
            Total time horizon in days (e.g., 1.0 = execute over 1 day)

        Returns
        -------
        pd.DataFrame
            Schedule with columns:
            - interval: interval number (0 to n_intervals-1)
            - time: time in days from start
            - shares: shares to trade in this interval
            - remaining_shares: shares remaining after this interval
            - participation_rate: shares / ADV for this interval
            - cost_bps: estimated cost in bps for this interval
        """
        if adv <= 0:
            raise ValueError(f"ADV must be positive, got {adv}")

        if n_intervals <= 0:
            raise ValueError(f"n_intervals must be positive, got {n_intervals}")

        # Use provided volatility or config default
        sigma = volatility if volatility is not None else (self.cfg.volatility or 0.20)

        # Time per interval
        tau = T / n_intervals

        # Impact parameters
        eta = self.cfg.eta
        gamma = self.cfg.gamma
        lambda_risk = self.cfg.risk_aversion

        # Almgren-Chriss parameters
        # kappa = sqrt(lambda * sigma^2 / eta)  (permanent impact parameter)
        # epsilon = temporary impact parameter
        epsilon = eta * (1 - gamma)
        kappa = np.sqrt(lambda_risk * sigma**2 / epsilon) if epsilon > 0 else 0.0

        # Optimal trajectory: exponentially decreasing trade rate
        # n_j = sinh(kappa * (T - j*tau)) / sinh(kappa * T)  (fraction of inventory at time j*tau)
        times = np.arange(n_intervals + 1) * tau
        if kappa * T < 1e-8:
            # Linear trajectory (risk-neutral case)
            inventory_fractions = 1 - times / T
        else:
            inventory_fractions = np.sinh(kappa * (T - times)) / np.sinh(kappa * T)

        # Convert to absolute shares remaining
        inventory = abs(total_shares) * inventory_fractions

        # Shares to trade in each interval
        trades = -np.diff(inventory)  # Negative because inventory decreases

        # Build schedule dataframe
        schedule = []
        for i in range(n_intervals):
            shares_to_trade = trades[i]

            # Apply minimum trade size filter
            if shares_to_trade < self.cfg.min_trade_size:
                continue

            participation_rate = shares_to_trade / adv

            # Cap participation rate
            if participation_rate > self.cfg.max_participation_rate:
                shares_to_trade = self.cfg.max_participation_rate * adv
                participation_rate = self.cfg.max_participation_rate

            # Estimate cost using square-root impact
            impact_bps = eta * np.sqrt(participation_rate) * 10_000

            schedule.append(
                {
                    "interval": i,
                    "time_days": times[i],
                    "shares": shares_to_trade * np.sign(total_shares),  # Restore sign
                    "remaining_shares": inventory[i + 1] * np.sign(total_shares),
                    "participation_rate": participation_rate,
                    "impact_bps": impact_bps,
                }
            )

        df = pd.DataFrame(schedule)

        if df.empty:
            logger.warning(f"No trades scheduled for {total_shares} shares (all below min_trade_size)")
            return df

        # Add cumulative metrics
        df["cumulative_shares"] = df["shares"].cumsum()
        df["cost_usd"] = (df["impact_bps"] / 10_000) * df["shares"].abs() * price

        logger.info(
            f"Scheduled {len(df)} intervals for {abs(total_shares):.0f} shares, "
            f"avg participation: {df['participation_rate'].mean():.2%}, "
            f"total cost: ${df['cost_usd'].sum():.2f}"
        )

        return df

    def simulate_execution(
        self,
        schedule: pd.DataFrame,
        price_data: pd.DataFrame,
        add_noise: bool = False,
        volatility: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Simulate execution following the schedule with realistic fills.

        Parameters
        ----------
        schedule : pd.DataFrame
            Output from compute_schedule()
        price_data : pd.DataFrame
            Historical price data with columns: time, price
        add_noise : bool
            Add price impact noise to simulate realistic execution
        volatility : Optional[float]
            Price volatility for noise generation (if add_noise=True)

        Returns
        -------
        pd.DataFrame
            Execution results with columns: interval, shares, fill_price, slippage_bps
        """
        if schedule.empty:
            return pd.DataFrame()

        sigma = volatility if volatility is not None else (self.cfg.volatility or 0.20)
        results = []

        for _, row in schedule.iterrows():
            interval = row["interval"]
            shares = row["shares"]
            impact_bps = row["impact_bps"]

            # Lookup price at this interval (simplified: use row index)
            if interval < len(price_data):
                base_price = price_data.iloc[interval]["price"]
            else:
                base_price = price_data.iloc[-1]["price"]

            # Apply impact: adverse price movement for buyer/seller
            # Buy: pay impact on top of midprice
            # Sell: receive impact below midprice
            slippage_bps = impact_bps * np.sign(shares)

            # Add noise if requested
            if add_noise:
                noise_bps = np.random.normal(0, sigma * 100) * np.sqrt(1 / 252)  # Daily volatility to bps
                slippage_bps += noise_bps

            fill_price = base_price * (1 + slippage_bps / 10_000)

            results.append(
                {
                    "interval": interval,
                    "shares": shares,
                    "base_price": base_price,
                    "fill_price": fill_price,
                    "slippage_bps": slippage_bps,
                    "cost_usd": (slippage_bps / 10_000) * abs(shares) * base_price,
                }
            )

        return pd.DataFrame(results)

    def estimate_cost_frontier(
        self, total_shares: float, adv: float, price: float, n_intervals_range: List[int]
    ) -> pd.DataFrame:
        """
        Compute cost-time frontier: cost vs urgency tradeoff.

        Parameters
        ----------
        total_shares : float
            Total shares to execute
        adv : float
            Average daily volume
        price : float
            Current price
        n_intervals_range : List[int]
            List of interval counts to evaluate (e.g., [1, 5, 10, 20])

        Returns
        -------
        pd.DataFrame
            Frontier with columns: n_intervals, avg_participation, total_cost_bps, risk_penalty
        """
        results = []

        for n in n_intervals_range:
            schedule = self.compute_schedule(total_shares, adv, price, n_intervals=n)

            if schedule.empty:
                continue

            avg_participation = schedule["participation_rate"].mean()
            total_cost_usd = schedule["cost_usd"].sum()
            notional = abs(total_shares) * price
            total_cost_bps = (total_cost_usd / notional) * 10_000 if notional > 0 else 0.0

            results.append(
                {
                    "n_intervals": n,
                    "avg_participation": avg_participation,
                    "total_cost_usd": total_cost_usd,
                    "total_cost_bps": total_cost_bps,
                }
            )

        return pd.DataFrame(results)
