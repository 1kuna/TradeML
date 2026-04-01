"""Cross-sectional Phase 1 equity feature engineering."""

from __future__ import annotations

import exchange_calendars as xcals
import numpy as np
import pandas as pd


def build_features(
    panel: pd.DataFrame,
    config: dict,
    earnings_calendar: pd.DataFrame | None = None,
    *,
    exchange: str = "XNYS",
) -> pd.DataFrame:
    """Build lagged Phase 1 features from adjusted OHLCV bars."""
    frame = panel.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["symbol", "date"]).reset_index(drop=True)

    grouped = frame.groupby("symbol", group_keys=False)
    log_close = np.log(frame["close"].clip(lower=1e-9))
    frame["log_return_1d_raw"] = grouped["close"].transform(lambda s: np.log(s / s.shift(1)))
    market_return = frame.groupby("date")["log_return_1d_raw"].transform("mean")
    dollar_volume = frame["close"] * frame["volume"].fillna(0)

    for window in config.get("price", {}).get("momentum", [5, 20, 60, 126]):
        frame[f"momentum_{window}d"] = grouped["close"].transform(lambda s, w=window: np.log(s / s.shift(w)))

    for window in config.get("price", {}).get("reversal", [1, 5]):
        frame[f"reversal_{window}d"] = grouped["close"].transform(lambda s, w=window: np.log(s / s.shift(w)))

    for window in config.get("price", {}).get("drawdown", [20, 60]):
        frame[f"drawdown_{window}d"] = grouped["close"].transform(lambda s, w=window: s / s.rolling(w).max() - 1.0)

    frame["gap_overnight"] = np.log(frame["open"] / grouped["close"].shift(1))

    for window in config.get("volatility", {}).get("realized", [20, 60]):
        frame[f"realized_vol_{window}d"] = grouped["log_return_1d_raw"].transform(lambda s, w=window: s.rolling(w).std())

    for window in config.get("volatility", {}).get("idiosyncratic", [60]):
        def _idiosyncratic(symbol_returns: pd.Series, w: int = window) -> pd.Series:
            aligned_market = market_return.loc[symbol_returns.index]
            residuals = []
            for idx in range(len(symbol_returns)):
                start = max(0, idx - w + 1)
                y = symbol_returns.iloc[start : idx + 1]
                x = aligned_market.iloc[start : idx + 1]
                if len(y.dropna()) < w:
                    residuals.append(np.nan)
                    continue
                beta = np.cov(y, x)[0, 1] / np.var(x) if np.var(x) else 0.0
                residuals.append((y.iloc[-1] - beta * x.iloc[-1]))
            return pd.Series(residuals, index=symbol_returns.index).rolling(w).std()

        frame[f"idiosyncratic_vol_{window}d"] = grouped["log_return_1d_raw"].transform(_idiosyncratic)

    for window in config.get("liquidity", {}).get("adv_dollar", [20]):
        frame[f"adv_dollar_{window}d"] = dollar_volume.groupby(frame["symbol"]).transform(
            lambda s, w=window: s.rolling(w).mean()
        )

    for window in config.get("liquidity", {}).get("amihud", [20]):
        illiquidity = frame["log_return_1d_raw"].abs() / dollar_volume.replace(0, np.nan)
        frame[f"amihud_{window}d"] = illiquidity.groupby(frame["symbol"]).transform(lambda s, w=window: s.rolling(w).mean())

    if config.get("controls", {}).get("log_price", True):
        frame["log_price"] = log_close

    if earnings_calendar is not None and not earnings_calendar.empty:
        earnings = earnings_calendar.copy()
        earnings["earnings_date"] = pd.to_datetime(earnings["earnings_date"])
        earnings_map = earnings.groupby("symbol")["earnings_date"].apply(list).to_dict()
        calendar = xcals.get_calendar(exchange)
        trading_days = pd.Index(
            calendar.sessions_in_range(frame["date"].min().normalize(), frame["date"].max().normalize())
        )
        trading_positions = {day: idx for idx, day in enumerate(trading_days)}

        def _within_earnings(row: pd.Series) -> bool:
            dates = earnings_map.get(row["symbol"], [])
            current = pd.Timestamp(row["date"])
            current_pos = trading_positions.get(current)
            if current_pos is None:
                return False
            return any(
                (earnings_date in trading_positions)
                and 0 < (trading_positions[earnings_date] - current_pos) <= 5
                for earnings_date in dates
            )

        frame["earnings_within_5d"] = frame.apply(_within_earnings, axis=1)
    else:
        frame["earnings_within_5d"] = False

    feature_cols = [column for column in frame.columns if column not in panel.columns and column != "log_return_1d_raw"]
    for column in feature_cols:
        if column != "earnings_within_5d":
            frame[column] = grouped[column].shift(1)

    keep_columns = ["date", "symbol"] + [column for column in frame.columns if column not in panel.columns and column != "log_return_1d_raw"]
    return frame[keep_columns].sort_values(["date", "symbol"]).reset_index(drop=True)
