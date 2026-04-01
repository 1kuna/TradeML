"""Deterministic event-driven backtesting."""

from __future__ import annotations

from dataclasses import dataclass

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
    price_pivot = price_frame.pivot(index="date", columns="symbol", values="close").sort_index()
    target_frame = target_weights.copy()
    target_frame["date"] = pd.to_datetime(target_frame["date"])
    corp_actions_frame = corp_actions.copy() if corp_actions is not None else pd.DataFrame(columns=["symbol", "event_type", "ex_date", "ratio"])
    if not corp_actions_frame.empty:
        corp_actions_frame["ex_date"] = pd.to_datetime(corp_actions_frame["ex_date"])

    positions = {symbol: 0.0 for symbol in price_pivot.columns}
    equity_rows: list[dict[str, float | pd.Timestamp]] = []
    trade_rows: list[dict[str, float | str | pd.Timestamp]] = []
    prior_prices = None

    for current_date, price_row in price_pivot.iterrows():
        actions_today = corp_actions_frame.loc[corp_actions_frame["ex_date"] == current_date] if not corp_actions_frame.empty else pd.DataFrame()
        if prior_prices is not None and not actions_today.empty:
            for action in actions_today.itertuples(index=False):
                symbol = str(action.symbol)
                if symbol not in positions or positions[symbol] == 0:
                    continue
                action_value = float(action.ratio)
                if action.event_type == "split" and action_value not in {0.0, 1.0}:
                    prior_prices[symbol] = prior_prices[symbol] * action_value
                    old_shares = positions[symbol]
                    positions[symbol] = positions[symbol] / action_value
                    trade_rows.append(
                        {
                            "date": current_date,
                            "symbol": symbol,
                            "shares": positions[symbol] - old_shares,
                            "price": price_row[symbol],
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
                        "action": "REBALANCE",
                        "event_type": None,
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
