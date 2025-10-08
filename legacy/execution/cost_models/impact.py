"""
Market impact models for execution cost estimation.

Implements:
- Square-root impact model (Almgren, Chriss, et al.)
- Temporary vs permanent impact decomposition
- Participation rate constraints
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class ImpactConfig:
    """Configuration for market impact model."""

    # Square-root impact parameters (eta * sqrt(participation_rate))
    eta: float = 0.10  # Impact coefficient (10 bps per sqrt(% ADV))
    gamma: float = 0.50  # Permanent impact fraction (50% of total)

    # Participation constraints
    max_participation_rate: float = 0.10  # Max 10% of ADV per period
    min_participation_rate: float = 0.001  # Min 0.1% to avoid division issues


class SquareRootImpact:
    """
    Square-root market impact model.

    Cost formula:
        impact_bps = eta * sqrt(participation_rate)
        where participation_rate = shares_traded / ADV

    Decomposes into:
    - Temporary impact: (1 - gamma) * impact_bps (recovers after trade)
    - Permanent impact: gamma * impact_bps (persists)

    References:
    - Almgren, R., & Chriss, N. (2001). Optimal execution of portfolio transactions.
    - Almgren, R. (2003). Optimal execution with nonlinear impact functions.
    """

    def __init__(self, config: Optional[ImpactConfig] = None):
        """
        Initialize impact model.

        Parameters
        ----------
        config : Optional[ImpactConfig]
            Impact model configuration (uses defaults if None)
        """
        self.cfg = config or ImpactConfig()

    def calculate_impact_bps(self, shares: float, adv: float, price: float) -> dict:
        """
        Calculate market impact in basis points.

        Parameters
        ----------
        shares : float
            Number of shares to trade (signed: positive = buy, negative = sell)
        adv : float
            Average daily volume (in shares)
        price : float
            Current price per share

        Returns
        -------
        dict
            {
                "participation_rate": float,  # shares / ADV
                "impact_bps": float,  # Total impact (temporary + permanent)
                "temporary_bps": float,  # Temporary impact (recovers)
                "permanent_bps": float,  # Permanent impact (persists)
                "impact_cost_usd": float,  # Dollar cost of impact
                "is_capped": bool,  # Whether participation rate was capped
            }
        """
        if adv <= 0:
            raise ValueError(f"ADV must be positive, got {adv}")

        if price <= 0:
            raise ValueError(f"Price must be positive, got {price}")

        # Participation rate (as fraction of ADV)
        participation_rate = abs(shares) / adv

        # Cap participation rate
        is_capped = participation_rate > self.cfg.max_participation_rate
        if is_capped:
            participation_rate = self.cfg.max_participation_rate

        # Floor participation rate to avoid numerical issues
        participation_rate = max(participation_rate, self.cfg.min_participation_rate)

        # Square-root impact formula
        impact_bps = self.cfg.eta * np.sqrt(participation_rate) * 10_000  # Convert to bps

        # Decompose into temporary and permanent
        temporary_bps = (1 - self.cfg.gamma) * impact_bps
        permanent_bps = self.cfg.gamma * impact_bps

        # Dollar cost (adverse price movement)
        impact_cost_usd = (impact_bps / 10_000) * abs(shares) * price

        return {
            "participation_rate": participation_rate,
            "impact_bps": impact_bps,
            "temporary_bps": temporary_bps,
            "permanent_bps": permanent_bps,
            "impact_cost_usd": impact_cost_usd,
            "is_capped": is_capped,
        }

    def calculate_optimal_participation(
        self, shares: float, adv: float, urgency: float = 0.5
    ) -> float:
        """
        Calculate optimal participation rate given urgency.

        Parameters
        ----------
        shares : float
            Total shares to trade
        adv : float
            Average daily volume
        urgency : float
            Urgency parameter (0 = patient, 1 = aggressive)

        Returns
        -------
        float
            Optimal participation rate as fraction of ADV
        """
        # Simple heuristic: scale participation by urgency
        base_participation = abs(shares) / adv
        optimal_participation = base_participation * (0.01 + 0.99 * urgency)

        # Cap at max participation
        return min(optimal_participation, self.cfg.max_participation_rate)

    def calculate_schedule_cost(
        self, shares: float, adv: float, price: float, n_periods: int = 1
    ) -> dict:
        """
        Calculate total cost of executing over multiple periods.

        Parameters
        ----------
        shares : float
            Total shares to trade
        adv : float
            Average daily volume per period
        price : float
            Current price
        n_periods : int
            Number of periods to spread execution over (e.g., 1 day = 1 period)

        Returns
        -------
        dict
            {
                "total_cost_usd": float,  # Total execution cost
                "cost_bps": float,  # Cost as bps of notional
                "shares_per_period": float,  # Shares per period
                "participation_per_period": float,  # Participation rate per period
            }
        """
        shares_per_period = abs(shares) / n_periods
        result = self.calculate_impact_bps(shares_per_period, adv, price)

        total_cost_usd = result["impact_cost_usd"] * n_periods
        notional = abs(shares) * price
        cost_bps = (total_cost_usd / notional) * 10_000 if notional > 0 else 0.0

        return {
            "total_cost_usd": total_cost_usd,
            "cost_bps": cost_bps,
            "shares_per_period": shares_per_period,
            "participation_per_period": result["participation_rate"],
            "n_periods": n_periods,
        }

    def estimate_impact_for_portfolio(
        self, trades: pd.DataFrame, market_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Estimate impact costs for a portfolio of trades.

        Parameters
        ----------
        trades : pd.DataFrame
            Columns: symbol, shares (signed: + = buy, - = sell)
        market_data : pd.DataFrame
            Columns: symbol, price, adv (average daily volume in shares)

        Returns
        -------
        pd.DataFrame
            Trades with additional columns: participation_rate, impact_bps, impact_cost_usd
        """
        # Merge trades with market data
        df = trades.merge(market_data[["symbol", "price", "adv"]], on="symbol", how="left")

        # Calculate impact for each trade
        impacts = df.apply(
            lambda row: pd.Series(self.calculate_impact_bps(row["shares"], row["adv"], row["price"])),
            axis=1,
        )

        return pd.concat([df, impacts], axis=1)
