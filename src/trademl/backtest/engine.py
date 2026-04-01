"""Deterministic event-driven backtesting."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(slots=True)
class BacktestResult:
    """Backtest outputs."""

    equity_curve: pd.DataFrame
    trade_log: pd.DataFrame
    cost_attribution: pd.DataFrame


def run_backtest(prices: pd.DataFrame, target_weights: pd.DataFrame, cost_model, config: dict) -> BacktestResult:
    """Run a deterministic long-only backtest."""
    capital = float(config.get("initial_capital", 1_000_000.0))
    price_frame = prices.copy()
    price_frame["date"] = pd.to_datetime(price_frame["date"])
    price_pivot = price_frame.pivot(index="date", columns="symbol", values="close").sort_index()
    target_frame = target_weights.copy()
    target_frame["date"] = pd.to_datetime(target_frame["date"])

    positions = {symbol: 0.0 for symbol in price_pivot.columns}
    equity_rows: list[dict[str, float | pd.Timestamp]] = []
    trade_rows: list[dict[str, float | str | pd.Timestamp]] = []
    prior_prices = None

    for current_date, price_row in price_pivot.iterrows():
        if prior_prices is not None:
            for symbol, shares in positions.items():
                capital += shares * (price_row[symbol] - prior_prices[symbol])

        rebalance = target_frame.loc[target_frame["date"] == current_date]
        if not rebalance.empty:
            portfolio_value = capital + sum(positions[symbol] * price_row[symbol] for symbol in positions)
            trade_batch = []
            for symbol in price_pivot.columns:
                target_weight = float(rebalance.loc[rebalance["symbol"] == symbol, "target_weight"].iloc[0]) if symbol in set(rebalance["symbol"]) else 0.0
                desired_shares = 0.0 if price_row[symbol] == 0 else (portfolio_value * target_weight) / price_row[symbol]
                delta_shares = desired_shares - positions[symbol]
                if abs(delta_shares) < 1e-12:
                    continue
                trade_batch.append(
                    {
                        "date": current_date,
                        "symbol": symbol,
                        "shares": delta_shares,
                        "price": price_row[symbol],
                        "trade_value": delta_shares * price_row[symbol],
                    }
                )
                positions[symbol] = desired_shares

            if trade_batch:
                trade_frame = pd.DataFrame(trade_batch)
                costed = cost_model(trade_frame, {"spread_bps": config.get("cost_spread_bps", 5.0), "stress_multiplier": 1.0})
                capital -= float(costed["trade_value"].sum() + costed["cost"].sum())
                trade_rows.extend(costed.to_dict("records"))

        equity = capital + sum(positions[symbol] * price_row[symbol] for symbol in positions)
        equity_rows.append({"date": current_date, "equity": equity})
        prior_prices = price_row

    trade_log = pd.DataFrame(trade_rows)
    equity_curve = pd.DataFrame(equity_rows)
    cost_attribution = trade_log[["date", "symbol", "cost"]] if not trade_log.empty else pd.DataFrame(columns=["date", "symbol", "cost"])
    return BacktestResult(equity_curve=equity_curve, trade_log=trade_log, cost_attribution=cost_attribution)
