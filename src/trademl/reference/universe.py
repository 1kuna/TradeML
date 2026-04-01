"""Universe construction helpers for staged data collection and training."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime

import pandas as pd

from trademl.calendars.exchange import get_trading_days
from trademl.connectors.base import BaseConnector
from trademl.data_node.bootstrap import ALLOWED_STAGE0_EXCHANGES, filter_stage0_assets, select_stage0_universe


@dataclass(slots=True)
class LiquidUniverseBuilder:
    """Build liquid universes from an explicit candidate symbol set."""

    connector: BaseConnector
    exchange: str = "XNYS"
    trailing_sessions: int = 20
    lookback_calendar_days: int = 45

    def build(self, candidates: pd.DataFrame, *, symbol_count: int, as_of_date: str | date | None = None) -> list[str]:
        """Fetch recent bars for candidates and return the top names by trailing ADV."""
        if symbol_count <= 0 or candidates.empty:
            return []
        resolved_as_of = pd.Timestamp(as_of_date or datetime.now(tz=UTC).date()).date()
        sessions = get_trading_days(
            self.exchange,
            resolved_as_of - pd.Timedelta(days=self.lookback_calendar_days),
            resolved_as_of,
        )
        trailing = [day.isoformat() for day in sessions[-self.trailing_sessions :]]
        if not trailing:
            return []
        symbols = candidates["symbol"].astype("string").dropna().str.strip().str.upper().tolist()
        bars = self.connector.fetch("equities_eod", symbols, trailing[0], trailing[-1])
        return select_stage0_universe(
            assets=candidates[["symbol"]].drop_duplicates().reset_index(drop=True),
            bars=bars,
            symbol_count=symbol_count,
            trailing_sessions=self.trailing_sessions,
        )


def active_listing_candidates(
    listing_history: pd.DataFrame,
    *,
    as_of_date: str | date,
    exchanges: set[str] | None = None,
) -> pd.DataFrame:
    """Return active common-stock candidates from the listing history as of a given date."""
    if listing_history.empty:
        return pd.DataFrame(columns=["symbol", "exchange", "asset_type", "status"])
    allowed_exchanges = {exchange.upper() for exchange in (exchanges or ALLOWED_STAGE0_EXCHANGES)}
    as_of = pd.Timestamp(as_of_date).normalize()
    frame = listing_history.copy()
    frame["symbol"] = frame.get("symbol", pd.Series(dtype="string")).astype("string").str.strip().str.upper()
    frame["exchange"] = frame.get("exchange", pd.Series(dtype="string")).astype("string").str.upper()
    frame["asset_type"] = frame.get("asset_type", pd.Series(dtype="string")).astype("string").str.lower()
    frame["ipo_date"] = pd.to_datetime(frame.get("ipo_date"), errors="coerce")
    frame["delist_date"] = pd.to_datetime(frame.get("delist_date"), errors="coerce")
    eligible = frame.loc[
        frame["symbol"].notna()
        & (frame["symbol"] != "")
        & frame["exchange"].isin(allowed_exchanges)
        & (frame["asset_type"] == "common_stock")
        & (frame["ipo_date"].isna() | (frame["ipo_date"] <= as_of))
        & (frame["delist_date"].isna() | (frame["delist_date"] > as_of))
    ].copy()
    return eligible.drop_duplicates(subset=["symbol"]).sort_values("symbol").reset_index(drop=True)


def build_stage1_universe(
    *,
    listing_history: pd.DataFrame,
    connector: BaseConnector,
    symbol_count: int = 500,
    as_of_date: str | date | None = None,
    exchange: str = "XNYS",
) -> list[str]:
    """Build the larger current-active liquid universe for Phase 2 collection expansion."""
    resolved_as_of = pd.Timestamp(as_of_date or datetime.now(tz=UTC).date()).date()
    candidates = active_listing_candidates(listing_history, as_of_date=resolved_as_of)
    if candidates.empty:
        return []
    current_assets = connector.fetch("assets", [], resolved_as_of.isoformat(), resolved_as_of.isoformat())
    tradable_assets = filter_stage0_assets(current_assets)
    if tradable_assets.empty:
        return []
    candidates = (
        candidates.merge(
            tradable_assets[["symbol"]].drop_duplicates(),
            on="symbol",
            how="inner",
        )
        .sort_values("symbol")
        .reset_index(drop=True)
    )
    if candidates.empty:
        return []
    builder = LiquidUniverseBuilder(connector=connector, exchange=exchange)
    return builder.build(candidates, symbol_count=symbol_count, as_of_date=resolved_as_of)


def build_time_varying_universe(
    *,
    listing_history: pd.DataFrame,
    daily_bars: pd.DataFrame,
    rebalance_dates: list[str] | list[pd.Timestamp],
    top_n: int = 500,
    adv_window: int = 20,
) -> pd.DataFrame:
    """Compute a rebalance-date-specific universe from listing eligibility plus trailing ADV."""
    if listing_history.empty or daily_bars.empty or not rebalance_dates:
        return pd.DataFrame(columns=["date", "symbol", "avg_dollar_volume", "rank"])
    bars = daily_bars.copy()
    bars["date"] = pd.to_datetime(bars["date"]).dt.normalize()
    bars["symbol"] = bars["symbol"].astype("string").str.strip().str.upper()
    bars["close"] = pd.to_numeric(bars["close"], errors="coerce")
    bars["volume"] = pd.to_numeric(bars["volume"], errors="coerce")
    bars["dollar_volume"] = bars["close"] * bars["volume"]
    bars = bars.sort_values(["symbol", "date"]).reset_index(drop=True)
    bars["avg_dollar_volume"] = bars.groupby("symbol")["dollar_volume"].transform(
        lambda series: series.rolling(adv_window, min_periods=min(5, adv_window)).mean()
    )

    rows: list[dict[str, object]] = []
    for rebalance_date in [pd.Timestamp(item).normalize() for item in rebalance_dates]:
        eligible = active_listing_candidates(listing_history, as_of_date=rebalance_date)
        if eligible.empty:
            continue
        day_bars = bars.loc[bars["date"] == rebalance_date, ["symbol", "avg_dollar_volume"]].dropna()
        ranked = (
            eligible[["symbol"]]
            .merge(day_bars, on="symbol", how="inner")
            .sort_values(["avg_dollar_volume", "symbol"], ascending=[False, True])
            .head(top_n)
            .reset_index(drop=True)
        )
        if ranked.empty:
            continue
        ranked["date"] = rebalance_date
        ranked["rank"] = range(1, len(ranked) + 1)
        rows.extend(ranked[["date", "symbol", "avg_dollar_volume", "rank"]].to_dict("records"))
    return pd.DataFrame(rows, columns=["date", "symbol", "avg_dollar_volume", "rank"])
