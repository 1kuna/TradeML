"""Shared bootstrap helpers for worker setup and Stage 0 universe seeding."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
import re
from typing import Any, Callable

import pandas as pd

from trademl.calendars.exchange import get_trading_days
from trademl.connectors.base import BaseConnector


DEFAULT_STAGE0_SYMBOL_COUNT = 100
DEFAULT_STAGE0_YEARS = 5
DEFAULT_STAGE0_EXCHANGE = "XNYS"
ALLOWED_STAGE0_EXCHANGES = {"NASDAQ", "NYSE"}
ETF_NAME_TOKENS = {
    "ETF",
    "ETN",
    "ETP",
    "FUND",
    "TRUST",
    "SHARES",
    "BOND",
    "TREASURY",
    "INCOME",
    "ULTRA",
    "BEAR",
    "BULL",
    "INVERSE",
    "LEVERAGED",
    "SPDR",
    "ISHARES",
    "VANGUARD",
    "INVESCO",
    "PROSHARES",
    "DIREXION",
    "WISDOMTREE",
}
ETF_NAME_PATTERN = re.compile(r"\b(?:%s)\b" % "|".join(re.escape(token) for token in sorted(ETF_NAME_TOKENS)))


@dataclass(slots=True)
class Stage0UniverseBuilder:
    """Build the Phase 1 Stage 0 universe from free Alpaca assets plus trailing ADV."""

    connector: BaseConnector
    exchange: str = DEFAULT_STAGE0_EXCHANGE
    trailing_sessions: int = 20
    lookback_calendar_days: int = 45

    def build(self, *, symbol_count: int = DEFAULT_STAGE0_SYMBOL_COUNT, as_of_date: str | date | None = None) -> list[str]:
        """Return the top liquid current-ticker universe for Stage 0."""
        if symbol_count <= 0:
            return []
        resolved_as_of = pd.Timestamp(as_of_date or datetime.now(tz=UTC).date()).date()
        assets = self.connector.fetch("assets", [], resolved_as_of.isoformat(), resolved_as_of.isoformat())
        sessions = get_trading_days(
            self.exchange,
            resolved_as_of - pd.Timedelta(days=self.lookback_calendar_days),
            resolved_as_of,
        )
        trailing = [day.isoformat() for day in sessions[-self.trailing_sessions :]]
        if not trailing:
            raise RuntimeError("unable to compute Stage 0 universe without recent trading sessions")
        start_date = trailing[0]
        end_date = trailing[-1]
        candidate_assets = filter_stage0_assets(assets)
        if candidate_assets.empty:
            raise RuntimeError("Stage 0 universe build found no eligible Alpaca assets")
        symbols = candidate_assets["symbol"].tolist()
        bars = self.connector.fetch("equities_eod", symbols, start_date, end_date)
        universe = select_stage0_universe(
            assets=candidate_assets,
            bars=bars,
            symbol_count=symbol_count,
            trailing_sessions=self.trailing_sessions,
        )
        if len(universe) < symbol_count:
            raise RuntimeError(f"Stage 0 universe build returned {len(universe)} symbols, expected {symbol_count}")
        return universe

    def __call__(self, symbol_count: int) -> list[str]:
        """Callable adapter used by bootstrap flows."""
        return self.build(symbol_count=symbol_count)


def filter_stage0_assets(assets: pd.DataFrame) -> pd.DataFrame:
    """Filter Alpaca assets down to active, tradable current-ticker common stocks."""
    if assets.empty:
        return pd.DataFrame(columns=["symbol", "name", "exchange", "status", "tradable", "asset_class"])
    frame = assets.copy()
    frame["symbol"] = frame.get("symbol", pd.Series(dtype="string")).astype("string").str.strip()
    frame["name"] = frame.get("name", pd.Series(dtype="string")).astype("string").fillna("")
    frame["exchange"] = frame.get("exchange", pd.Series(dtype="string")).astype("string").str.upper()
    frame["status"] = frame.get("status", pd.Series(dtype="string")).astype("string").str.lower()
    frame["asset_class"] = frame.get("asset_class", frame.get("class", pd.Series(dtype="string"))).astype("string").str.lower()
    frame["tradable"] = frame.get("tradable", pd.Series(dtype="bool")).fillna(False).astype(bool)
    filtered = frame.loc[
        frame["symbol"].notna()
        & (frame["symbol"] != "")
        & frame["tradable"]
        & (frame["status"] == "active")
        & (frame["asset_class"] == "us_equity")
        & frame["exchange"].isin(ALLOWED_STAGE0_EXCHANGES)
        & ~frame["name"].str.upper().str.contains(ETF_NAME_PATTERN, na=False)
    ].copy()
    return filtered.drop_duplicates(subset=["symbol"]).sort_values("symbol").reset_index(drop=True)


def select_stage0_universe(
    *,
    assets: pd.DataFrame,
    bars: pd.DataFrame,
    symbol_count: int = DEFAULT_STAGE0_SYMBOL_COUNT,
    trailing_sessions: int = 20,
) -> list[str]:
    """Rank eligible assets by trailing average dollar volume and return the top names."""
    if symbol_count <= 0 or assets.empty:
        return []
    if bars.empty:
        return []
    frame = bars.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["close"] = pd.to_numeric(frame.get("close"), errors="coerce")
    frame["volume"] = pd.to_numeric(frame.get("volume"), errors="coerce")
    frame = frame.dropna(subset=["symbol", "date", "close", "volume"])
    if frame.empty:
        return []
    frame = frame.sort_values(["symbol", "date"]).copy()
    frame["dollar_volume"] = frame["close"] * frame["volume"]
    trailing = frame.groupby("symbol", as_index=False, group_keys=False).tail(trailing_sessions)
    ranked = (
        trailing.groupby("symbol", as_index=False)
        .agg(avg_dollar_volume=("dollar_volume", "mean"), session_count=("date", "nunique"))
        .sort_values(["avg_dollar_volume", "symbol"], ascending=[False, True])
    )
    eligible = assets[["symbol"]].drop_duplicates()
    merged = eligible.merge(ranked, on="symbol", how="inner")
    merged = merged.loc[merged["session_count"] >= max(5, min(trailing_sessions, 10))]
    merged = merged.sort_values(["avg_dollar_volume", "symbol"], ascending=[False, True]).reset_index(drop=True)
    return merged["symbol"].head(symbol_count).tolist()


def resolve_bootstrap_stage(
    local_config: dict[str, Any],
    local_stage: dict[str, Any],
    *,
    universe_builder: Callable[[int], list[str]] | None = None,
) -> tuple[int, list[str], int]:
    """Resolve a usable bootstrap stage from local stage.yml, config defaults, or programmatic build."""
    current = int(local_stage.get("current", local_config.get("stage", {}).get("current", 0)))
    stage_key = f"stage_{current}"
    stage_cfg = local_config.get("stage", {}).get(stage_key, {})
    years = int(local_stage.get("years", stage_cfg.get("eod_years", DEFAULT_STAGE0_YEARS)))

    stage_symbols = [str(symbol) for symbol in local_stage.get("symbols", []) if str(symbol).strip()]
    if stage_symbols:
        return current, stage_symbols, years

    configured_symbols = stage_cfg.get("symbols", [])
    if isinstance(configured_symbols, list) and configured_symbols:
        stage_symbols = [str(symbol) for symbol in configured_symbols if str(symbol).strip()]
        return current, stage_symbols, years

    target_count = int(configured_symbols or stage_cfg.get("symbol_count", DEFAULT_STAGE0_SYMBOL_COUNT))
    if current != 0:
        return current, [], years
    if universe_builder is None:
        return current, [], years
    return current, universe_builder(target_count), years
