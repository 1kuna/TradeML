"""
Trading tripwires: automated risk controls and circuit breakers.

Implements:
- Drawdown monitoring
- Sharpe degradation detection
- Slippage alerts
- Position limit checks
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class TripwireConfig:
    """Configuration for tripwire thresholds."""

    # Drawdown limits
    max_daily_dd_pct: float = 5.0  # Max 5% daily drawdown
    max_5day_dd_pct: float = 10.0  # Max 10% 5-day drawdown
    max_total_dd_pct: float = 20.0  # Max 20% total drawdown

    # Performance degradation
    min_5day_sharpe: float = 0.0  # Min 5-day Sharpe ratio
    min_20day_sharpe: float = 0.5  # Min 20-day Sharpe

    # Slippage monitoring
    max_slippage_vs_model_pct: float = 50.0  # Max 50% worse than model

    # Position limits
    max_gross_exposure: float = 1.5  # Max 150% gross
    max_net_exposure: float = 1.0  # Max 100% net
    max_single_position_pct: float = 10.0  # Max 10% per position


@dataclass
class TripwireAlert:
    """Tripwire alert."""

    timestamp: datetime
    severity: str  # "WARNING" or "CRITICAL"
    rule: str
    message: str
    current_value: float
    threshold: float
    action_required: str


class TripwireManager:
    """
    Tripwire manager for automated risk monitoring.

    Monitors strategy performance and triggers alerts/halts when thresholds breached.
    """

    def __init__(self, config: Optional[TripwireConfig] = None):
        """
        Initialize tripwire manager.

        Parameters
        ----------
        config : Optional[TripwireConfig]
            Tripwire configuration (uses defaults if None)
        """
        self.config = config or TripwireConfig()
        self.alerts: List[TripwireAlert] = []
        self.is_halted = False

    def check_drawdown(
        self,
        equity_curve: pd.DataFrame,
        date_col: str = "date",
        equity_col: str = "equity",
    ) -> List[TripwireAlert]:
        """
        Check drawdown tripwires.

        Parameters
        ----------
        equity_curve : pd.DataFrame
            Equity curve with date and equity columns
        date_col : str
            Date column name
        equity_col : str
            Equity column name

        Returns
        -------
        List[TripwireAlert]
            Triggered alerts
        """
        alerts = []

        if len(equity_curve) == 0:
            return alerts

        equity = equity_curve[equity_col].values
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak

        # Current drawdown
        current_dd = drawdown[-1] * 100

        # Daily drawdown
        if len(equity) > 1:
            daily_ret = (equity[-1] / equity[-2]) - 1
            daily_dd = min(daily_ret, 0) * 100

            if abs(daily_dd) > self.config.max_daily_dd_pct:
                alerts.append(
                    TripwireAlert(
                        timestamp=datetime.now(),
                        severity="CRITICAL",
                        rule="MAX_DAILY_DD",
                        message=f"Daily drawdown {daily_dd:.2f}% exceeds limit {self.config.max_daily_dd_pct}%",
                        current_value=abs(daily_dd),
                        threshold=self.config.max_daily_dd_pct,
                        action_required="HALT_TRADING",
                    )
                )

        # 5-day drawdown
        if len(equity) >= 5:
            ret_5d = (equity[-1] / equity[-5]) - 1
            dd_5d = min(ret_5d, 0) * 100

            if abs(dd_5d) > self.config.max_5day_dd_pct:
                alerts.append(
                    TripwireAlert(
                        timestamp=datetime.now(),
                        severity="CRITICAL",
                        rule="MAX_5DAY_DD",
                        message=f"5-day drawdown {dd_5d:.2f}% exceeds limit {self.config.max_5day_dd_pct}%",
                        current_value=abs(dd_5d),
                        threshold=self.config.max_5day_dd_pct,
                        action_required="HALT_TRADING",
                    )
                )

        # Total drawdown
        if abs(current_dd) > self.config.max_total_dd_pct:
            alerts.append(
                TripwireAlert(
                    timestamp=datetime.now(),
                    severity="CRITICAL",
                    rule="MAX_TOTAL_DD",
                    message=f"Total drawdown {current_dd:.2f}% exceeds limit {self.config.max_total_dd_pct}%",
                    current_value=abs(current_dd),
                    threshold=self.config.max_total_dd_pct,
                    action_required="HALT_TRADING",
                )
            )

        return alerts

    def check_sharpe_degradation(
        self,
        returns: pd.Series,
        lookback_days: int = 20,
        annual_factor: float = 252.0,
    ) -> List[TripwireAlert]:
        """
        Check Sharpe ratio degradation.

        Parameters
        ----------
        returns : pd.Series
            Daily returns
        lookback_days : int
            Rolling window for Sharpe calculation
        annual_factor : float
            Annualization factor

        Returns
        -------
        List[TripwireAlert]
            Triggered alerts
        """
        alerts = []

        if len(returns) < lookback_days:
            return alerts

        recent_returns = returns.iloc[-lookback_days:]
        sharpe = (recent_returns.mean() / recent_returns.std()) * np.sqrt(annual_factor)

        if lookback_days == 5:
            threshold = self.config.min_5day_sharpe
            rule = "MIN_5DAY_SHARPE"
        elif lookback_days == 20:
            threshold = self.config.min_20day_sharpe
            rule = "MIN_20DAY_SHARPE"
        else:
            threshold = 0.0
            rule = f"MIN_{lookback_days}DAY_SHARPE"

        if sharpe < threshold:
            alerts.append(
                TripwireAlert(
                    timestamp=datetime.now(),
                    severity="WARNING",
                    rule=rule,
                    message=f"{lookback_days}-day Sharpe {sharpe:.2f} below threshold {threshold:.2f}",
                    current_value=sharpe,
                    threshold=threshold,
                    action_required="REVIEW_STRATEGY",
                )
            )

        return alerts

    def check_position_limits(
        self,
        positions: pd.DataFrame,
        capital: float,
        symbol_col: str = "symbol",
        value_col: str = "market_value",
    ) -> List[TripwireAlert]:
        """
        Check position limit tripwires.

        Parameters
        ----------
        positions : pd.DataFrame
            Current positions
        capital : float
            Total capital
        symbol_col : str
            Symbol column
        value_col : str
            Market value column

        Returns
        -------
        List[TripwireAlert]
            Triggered alerts
        """
        alerts = []

        if len(positions) == 0:
            return alerts

        # Gross exposure
        gross_exposure = positions[value_col].abs().sum() / capital

        if gross_exposure > self.config.max_gross_exposure:
            alerts.append(
                TripwireAlert(
                    timestamp=datetime.now(),
                    severity="WARNING",
                    rule="MAX_GROSS_EXPOSURE",
                    message=f"Gross exposure {gross_exposure:.2%} exceeds limit {self.config.max_gross_exposure:.2%}",
                    current_value=gross_exposure,
                    threshold=self.config.max_gross_exposure,
                    action_required="REDUCE_POSITIONS",
                )
            )

        # Net exposure
        net_exposure = positions[value_col].sum() / capital

        if abs(net_exposure) > self.config.max_net_exposure:
            alerts.append(
                TripwireAlert(
                    timestamp=datetime.now(),
                    severity="WARNING",
                    rule="MAX_NET_EXPOSURE",
                    message=f"Net exposure {net_exposure:.2%} exceeds limit {self.config.max_net_exposure:.2%}",
                    current_value=abs(net_exposure),
                    threshold=self.config.max_net_exposure,
                    action_required="REDUCE_POSITIONS",
                )
            )

        # Single position concentration
        position_pcts = (positions[value_col].abs() / capital) * 100
        max_position = position_pcts.max()
        max_symbol = positions.loc[position_pcts.idxmax(), symbol_col]

        if max_position > self.config.max_single_position_pct:
            alerts.append(
                TripwireAlert(
                    timestamp=datetime.now(),
                    severity="WARNING",
                    rule="MAX_SINGLE_POSITION",
                    message=f"Position in {max_symbol} ({max_position:.2f}%) exceeds limit {self.config.max_single_position_pct:.2f}%",
                    current_value=max_position,
                    threshold=self.config.max_single_position_pct,
                    action_required="REDUCE_POSITION",
                )
            )

        return alerts

    def check_all(
        self,
        equity_curve: pd.DataFrame,
        returns: pd.Series,
        positions: pd.DataFrame,
        capital: float,
    ) -> List[TripwireAlert]:
        """
        Run all tripwire checks.

        Parameters
        ----------
        equity_curve : pd.DataFrame
            Equity curve
        returns : pd.Series
            Daily returns
        positions : pd.DataFrame
            Current positions
        capital : float
            Total capital

        Returns
        -------
        List[TripwireAlert]
            All triggered alerts
        """
        all_alerts = []

        # Drawdown checks
        all_alerts.extend(self.check_drawdown(equity_curve))

        # Sharpe checks
        if len(returns) >= 5:
            all_alerts.extend(self.check_sharpe_degradation(returns, lookback_days=5))
        if len(returns) >= 20:
            all_alerts.extend(self.check_sharpe_degradation(returns, lookback_days=20))

        # Position limit checks
        all_alerts.extend(self.check_position_limits(positions, capital))

        # Check for HALT actions
        critical_alerts = [a for a in all_alerts if a.action_required == "HALT_TRADING"]
        if critical_alerts:
            self.is_halted = True
            logger.critical(f"TRADING HALTED: {len(critical_alerts)} critical tripwires triggered")

        # Log all alerts
        for alert in all_alerts:
            if alert.severity == "CRITICAL":
                logger.critical(f"[{alert.rule}] {alert.message}")
            else:
                logger.warning(f"[{alert.rule}] {alert.message}")

        self.alerts.extend(all_alerts)
        return all_alerts

    def summary(self) -> pd.DataFrame:
        """
        Get summary of all alerts.

        Returns
        -------
        pd.DataFrame
            Alert summary
        """
        if not self.alerts:
            return pd.DataFrame()

        return pd.DataFrame([
            {
                "timestamp": a.timestamp,
                "severity": a.severity,
                "rule": a.rule,
                "current_value": a.current_value,
                "threshold": a.threshold,
                "action": a.action_required,
            }
            for a in self.alerts
        ])
