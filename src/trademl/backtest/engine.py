"""Deterministic event-driven backtesting."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import pandas as pd
from scipy.stats import spearmanr


@dataclass(slots=True)
class BacktestResult:
    """Backtest outputs."""

    equity_curve: pd.DataFrame
    trade_log: pd.DataFrame
    cost_attribution: pd.DataFrame
    ic_time_series: pd.DataFrame
    decile_returns: pd.DataFrame


def run_backtest(
    prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    cost_model,
    config: dict,
    *,
    corp_actions: pd.DataFrame | None = None,
    prediction_frame: pd.DataFrame | None = None,
) -> BacktestResult:
    """Run a deterministic long-only backtest."""
    capital = float(config.get("initial_capital", 1_000_000.0))
    price_frame = prices.copy()
    price_frame["date"] = pd.to_datetime(price_frame["date"])
    target_frame = target_weights.copy()
    target_frame["date"] = pd.to_datetime(target_frame["date"])
    all_symbols = sorted(set(price_frame["symbol"]).union(set(target_frame["symbol"])))
    price_pivot = price_frame.pivot(index="date", columns="symbol", values="close").reindex(columns=all_symbols).sort_index()
    corp_actions_frame = corp_actions.copy() if corp_actions is not None else pd.DataFrame(columns=["symbol", "event_type", "ex_date", "ratio"])
    if not corp_actions_frame.empty:
        corp_actions_frame["ex_date"] = pd.to_datetime(corp_actions_frame["ex_date"])

    positions = {symbol: 0.0 for symbol in price_pivot.columns}
    last_prices = {symbol: None for symbol in price_pivot.columns}
    equity_rows: list[dict[str, float | pd.Timestamp]] = []
    trade_rows: list[dict[str, float | str | pd.Timestamp]] = []

    for current_date, price_row in price_pivot.iterrows():
        actions_today = corp_actions_frame.loc[corp_actions_frame["ex_date"] == current_date] if not corp_actions_frame.empty else pd.DataFrame()
        if not actions_today.empty:
            for action in actions_today.itertuples(index=False):
                symbol = str(action.symbol)
                if symbol not in positions or positions[symbol] == 0:
                    continue
                action_value = float(action.ratio)
                if action.event_type == "split" and action_value not in {0.0, 1.0}:
                    last_price = last_prices.get(symbol)
                    if last_price is not None:
                        last_prices[symbol] = last_price * action_value
                    old_shares = positions[symbol]
                    positions[symbol] = positions[symbol] / action_value
                    trade_rows.append(
                        {
                            "date": current_date,
                            "symbol": symbol,
                            "shares": positions[symbol] - old_shares,
                            "price": _trade_price(price_row, last_prices, symbol),
                            "trade_value": 0.0,
                            "spread_cost": 0.0,
                            "impact_cost": 0.0,
                            "cost": 0.0,
                            "action": "CORP_ACTION",
                            "event_type": "split",
                        }
                    )
                elif action.event_type == "dividend":
                    cash_credit = positions[symbol] * action_value
                    capital += cash_credit
                    trade_rows.append(
                        {
                            "date": current_date,
                            "symbol": symbol,
                            "shares": 0.0,
                            "price": action_value,
                            "trade_value": cash_credit,
                            "spread_cost": 0.0,
                            "impact_cost": 0.0,
                            "cost": 0.0,
                            "action": "CORP_ACTION",
                            "event_type": "dividend",
                        }
                    )

        for symbol, shares in positions.items():
            if abs(shares) < 1e-12:
                continue
            current_price = _row_price(price_row, symbol)
            previous_price = last_prices.get(symbol)
            if current_price is None:
                continue
            if previous_price is not None:
                capital += shares * (current_price - previous_price)
            last_prices[symbol] = current_price

        for symbol in price_pivot.columns:
            current_price = _row_price(price_row, symbol)
            if current_price is not None:
                last_prices[symbol] = current_price

        rebalance = target_frame.loc[target_frame["date"] == current_date]
        if not rebalance.empty:
            target_by_symbol = rebalance.groupby("symbol")["target_weight"].last().to_dict()
            portfolio_value = capital + sum(
                positions[symbol] * mark_price
                for symbol, shares in positions.items()
                if abs(shares) >= 1e-12 and (mark_price := _trade_price(price_row, last_prices, symbol)) is not None
            )
            trade_batch = []
            for symbol in price_pivot.columns:
                target_weight = float(target_by_symbol.get(symbol, 0.0))
                trade_price = _trade_price(price_row, last_prices, symbol)
                if trade_price is None or trade_price == 0.0:
                    continue
                desired_shares = (portfolio_value * target_weight) / trade_price
                delta_shares = desired_shares - positions[symbol]
                if abs(delta_shares) < 1e-12:
                    continue
                trade_batch.append(
                    {
                        "date": current_date,
                        "symbol": symbol,
                        "shares": delta_shares,
                        "price": trade_price,
                        "trade_value": delta_shares * trade_price,
                        "action": "REBALANCE",
                        "event_type": None,
                    }
                )
                positions[symbol] = desired_shares

            if trade_batch:
                trade_frame = pd.DataFrame(trade_batch)
                costed = cost_model(
                    trade_frame,
                    {
                        "spread_bps": config.get("cost_spread_bps", 5.0),
                        "stress_multiplier": config.get("stress_multiplier", 1.0),
                    },
                )
                capital -= float(costed["trade_value"].sum() + costed["cost"].sum())
                trade_rows.extend(costed.to_dict("records"))

        equity = capital + sum(
            positions[symbol] * mark_price
            for symbol, shares in positions.items()
            if abs(shares) >= 1e-12 and (mark_price := _trade_price(price_row, last_prices, symbol)) is not None
        )
        equity_rows.append({"date": current_date, "equity": equity})

    trade_log = pd.DataFrame(trade_rows)
    equity_curve = pd.DataFrame(equity_rows)
    cost_attribution = trade_log[["date", "symbol", "cost"]] if not trade_log.empty else pd.DataFrame(columns=["date", "symbol", "cost"])
    ic_time_series, decile_returns = _prediction_diagnostics(prediction_frame)
    return BacktestResult(
        equity_curve=equity_curve,
        trade_log=trade_log,
        cost_attribution=cost_attribution,
        ic_time_series=ic_time_series,
        decile_returns=decile_returns,
    )


def _prediction_diagnostics(prediction_frame: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if prediction_frame is None or prediction_frame.empty:
        return (
            pd.DataFrame(columns=["date", "rank_ic"]),
            pd.DataFrame(columns=["date", "bucket", "mean_return"]),
        )

    frame = prediction_frame.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    actual_col = next((column for column in ["label_5d", "actual", "label"] if column in frame.columns), None)
    if actual_col is None:
        return (
            pd.DataFrame(columns=["date", "rank_ic"]),
            pd.DataFrame(columns=["date", "bucket", "mean_return"]),
        )

    ic_rows: list[dict[str, object]] = []
    decile_rows: list[dict[str, object]] = []
    for date_value, group in frame.groupby("date"):
        stat = spearmanr(group["prediction"], group[actual_col], nan_policy="omit").statistic
        ic_rows.append({"date": date_value, "rank_ic": float(stat) if stat == stat else 0.0})
        bucket_count = min(10, len(group))
        if bucket_count < 2:
            continue
        ranked = group.copy()
        ranked["bucket"] = pd.qcut(
            ranked["prediction"].rank(method="first"),
            q=bucket_count,
            labels=False,
            duplicates="drop",
        )
        for bucket, bucket_group in ranked.groupby("bucket"):
            decile_rows.append(
                {
                    "date": date_value,
                    "bucket": int(bucket) + 1,
                    "mean_return": float(bucket_group[actual_col].mean()),
                }
            )

    return pd.DataFrame(ic_rows), pd.DataFrame(decile_rows)


def _row_price(price_row: pd.Series, symbol: str) -> float | None:
    """Return a finite row price when present."""
    value = price_row.get(symbol)
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not isfinite(numeric):
        return None
    return numeric


def _trade_price(price_row: pd.Series, last_prices: dict[str, float | None], symbol: str) -> float | None:
    """Return the current tradeable price or the last observed mark price."""
    current_price = _row_price(price_row, symbol)
    if current_price is not None:
        return current_price
    previous_price = last_prices.get(symbol)
    if previous_price is None or not isfinite(previous_price):
        return None
    return previous_price
