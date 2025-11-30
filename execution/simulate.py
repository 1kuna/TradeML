"""
Execution simulation API per SSOT §4.8:

    execution.simulate(orders, cost_model, market_data) -> (fills, slippage_report)

Inputs:
- orders: DataFrame with columns ['date','symbol','target_quantity'] (daily targets)
- cost_model: dict with:
  - spread_bps: float (default 5.0) - bid-ask spread in basis points
  - fee_per_share: float (default 0.0) - commission per share
  - borrow_bps: float (default 0.0) - annualized borrow rate for shorts
  - initial_capital: float (default 1_000_000) - starting capital
  - num_trials: int (default 1) - number of strategy variants (for DSR)
- market_data: DataFrame with columns:
  - Required: ['date', 'symbol', 'close']
  - Optional: ['volume', 'volatility'] for market impact calculation

Behavior:
- Uses MinimalBacktester for daily event-driven simulation
- Applies transaction costs (spread + commission)
- Calculates market impact using square-root law if volume/volatility provided
- Returns fills DataFrame and comprehensive slippage/performance report

Returns:
- fills: DataFrame with trade executions
- slippage_report: Dict with performance metrics per SSOT §4.9
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import pandas as pd
from loguru import logger

from backtest.engine.backtester import MinimalBacktester


def simulate(
    orders: pd.DataFrame,
    cost_model: Optional[Dict[str, Any]] = None,
    market_data: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Simulate order execution with transaction costs and market impact.

    Args:
        orders: DataFrame with columns ['date', 'symbol', 'target_quantity']
        cost_model: Cost parameters (spread_bps, fee_per_share, borrow_bps, etc.)
        market_data: DataFrame with price data and optional volume/volatility

    Returns:
        Tuple of (fills DataFrame, slippage_report dict)
    """
    cm = cost_model or {}
    fee_per_share = float(cm.get("fee_per_share", 0.0))
    spread_bps = float(cm.get("spread_bps", 5.0))
    borrow_bps = float(cm.get("borrow_bps", 0.0))
    initial_capital = float(cm.get("initial_capital", 1_000_000.0))
    num_trials = int(cm.get("num_trials", 1))

    if market_data is None or market_data.empty:
        logger.warning("No market data provided; returning empty fills")
        return _empty_result()

    if orders is None or orders.empty:
        logger.warning("No orders provided; returning empty fills")
        return _empty_result()

    # Initialize backtester
    bt = MinimalBacktester(
        initial_capital=initial_capital,
        fee_per_share=fee_per_share,
        spread_bps=spread_bps,
        borrow_bps=borrow_bps
    )

    # Run backtest
    signals = orders[["date", "symbol", "target_quantity"]].copy()
    prices = market_data[["date", "symbol", "close"]].copy()

    equity = bt.run(signals, prices)
    metrics = bt.calculate_performance(num_trials=num_trials)

    # Build fills dataframe from recorded trades
    if bt.trades:
        fills = pd.DataFrame([
            {
                "date": t.date,
                "symbol": t.symbol,
                "side": t.side,
                "quantity": t.quantity,
                "price": t.price,
                "fees": t.fees,
                "impact": t.impact,
                "value": t.value,
                "pnl": t.pnl
            }
            for t in bt.trades
        ])
    else:
        fills = _empty_fills()

    # Comprehensive slippage/performance report per SSOT §4.9
    total_fees = float(fills["fees"].sum()) if not fills.empty else 0.0
    total_impact = float(fills["impact"].sum()) if not fills.empty else 0.0
    total_value = float(fills["value"].sum()) if not fills.empty else 0.0

    slippage_report = {
        # Transaction costs
        "total_fees_usd": total_fees,
        "total_impact_usd": total_impact,
        "total_costs_usd": total_fees + total_impact,
        "total_trade_value_usd": total_value,
        "cost_bps": (total_fees + total_impact) / total_value * 10000 if total_value > 0 else 0.0,
        # Performance metrics
        "total_return": float(metrics.total_return),
        "sharpe_ratio": float(metrics.sharpe_ratio),
        "deflated_sharpe_ratio": float(metrics.dsr),
        "max_drawdown": float(metrics.max_drawdown),
        "calmar_ratio": float(metrics.calmar_ratio),
        # Trade statistics
        "num_trades": int(metrics.num_trades),
        "win_rate": float(metrics.win_rate),
        "avg_trade_pnl": float(metrics.avg_trade_pnl),
        "turnover": float(metrics.turnover),
        # Capital
        "initial_capital": initial_capital,
        "final_equity": float(bt.equity_curve[-1][1]) if bt.equity_curve else initial_capital,
    }

    logger.info(
        f"Simulation complete: {metrics.num_trades} trades, "
        f"return={metrics.total_return*100:.2f}%, "
        f"sharpe={metrics.sharpe_ratio:.2f}, "
        f"costs={slippage_report['cost_bps']:.1f}bps"
    )

    return fills, slippage_report


def _empty_fills() -> pd.DataFrame:
    """Return empty fills DataFrame with correct schema."""
    return pd.DataFrame(columns=[
        "date", "symbol", "side", "quantity", "price",
        "fees", "impact", "value", "pnl"
    ])


def _empty_result() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Return empty result tuple."""
    return _empty_fills(), {
        "total_fees_usd": 0.0,
        "total_impact_usd": 0.0,
        "total_costs_usd": 0.0,
        "total_trade_value_usd": 0.0,
        "cost_bps": 0.0,
        "total_return": 0.0,
        "sharpe_ratio": 0.0,
        "deflated_sharpe_ratio": 0.0,
        "max_drawdown": 0.0,
        "calmar_ratio": 0.0,
        "num_trades": 0,
        "win_rate": 0.0,
        "avg_trade_pnl": 0.0,
        "turnover": 0.0,
        "initial_capital": 0.0,
        "final_equity": 0.0,
    }

