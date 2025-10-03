"""
Minimal execution API per Blueprint:

    execution.simulate(orders, cost_model, market_data) -> (fills, slippage_report)

Inputs:
- orders: DataFrame with columns ['date','symbol','target_quantity'] (daily targets)
- cost_model: dict, supports {'spread_bps': float, 'fee_per_share': float}
- market_data: DataFrame with columns ['date','symbol','close'] (daily closes)

Behavior:
- Uses the existing MinimalBacktester to generate trades and equity curve.
- Returns a fills DataFrame (trades) and a slippage/costs summary.
"""

from __future__ import annotations

from typing import Dict, Tuple
import pandas as pd

from backtest.engine.backtester import MinimalBacktester


def simulate(
    orders: pd.DataFrame,
    cost_model: Dict | None,
    market_data: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict]:
    cm = cost_model or {}
    fee_per_share = float(cm.get("fee_per_share", 0.0))
    spread_bps = float(cm.get("spread_bps", 5.0))

    bt = MinimalBacktester(initial_capital=float(cm.get("initial_capital", 1_000_000.0)), fee_per_share=fee_per_share, spread_bps=spread_bps)
    equity = bt.run(orders[["date", "symbol", "target_quantity"]], market_data[["date", "symbol", "close"]])
    metrics = bt.calculate_performance()

    # Build fills dataframe from recorded trades
    if bt.trades:
        fills = pd.DataFrame([t.__dict__ for t in bt.trades])
    else:
        fills = pd.DataFrame(columns=["date", "symbol", "side", "quantity", "price", "fees", "value", "pnl"])  # empty

    # Basic slippage/costs report
    total_fees = float(fills["fees"].sum()) if not fills.empty else 0.0
    total_value = float(fills["value"].sum()) if not fills.empty else 0.0
    slippage_report = {
        "total_fees_usd": total_fees,
        "total_trade_value_usd": total_value,
        "turnover": float(metrics.turnover),
        "total_return": float(metrics.total_return),
        "sharpe": float(metrics.sharpe_ratio),
        "max_drawdown": float(metrics.max_drawdown),
        "num_trades": int(metrics.num_trades),
    }

    return fills, slippage_report

