from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from trademl.data_node.bootstrap import (
    Stage0UniverseBuilder,
    filter_stage0_assets,
    resolve_bootstrap_stage,
    select_stage0_universe,
)


@dataclass
class FakeConnector:
    assets: pd.DataFrame
    bars: pd.DataFrame

    def fetch(self, dataset: str, symbols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        if dataset == "assets":
            return self.assets.copy()
        if dataset == "equities_eod":
            return self.bars.loc[self.bars["symbol"].isin(symbols)].copy()
        raise ValueError(dataset)


def test_filter_stage0_assets_excludes_non_common_stock_candidates() -> None:
    assets = pd.DataFrame(
        [
            {"symbol": "AAPL", "name": "Apple Inc.", "exchange": "NASDAQ", "status": "active", "tradable": True, "asset_class": "us_equity"},
            {"symbol": "SPY", "name": "SPDR S&P 500 ETF Trust", "exchange": "NYSE", "status": "active", "tradable": True, "asset_class": "us_equity"},
            {"symbol": "OTCM", "name": "OTC Markets", "exchange": "OTC", "status": "active", "tradable": True, "asset_class": "us_equity"},
            {"symbol": "PAUSE", "name": "Paused Corp", "exchange": "NASDAQ", "status": "inactive", "tradable": True, "asset_class": "us_equity"},
            {"symbol": "NOPE", "name": "No Trade Inc.", "exchange": "NASDAQ", "status": "active", "tradable": False, "asset_class": "us_equity"},
        ]
    )

    filtered = filter_stage0_assets(assets)

    assert filtered["symbol"].tolist() == ["AAPL"]


def test_select_stage0_universe_ranks_by_trailing_average_dollar_volume() -> None:
    assets = pd.DataFrame(
        [
            {"symbol": "AAPL"},
            {"symbol": "MSFT"},
            {"symbol": "NVDA"},
        ]
    )
    dates = pd.bdate_range("2026-03-02", periods=20)
    bars = pd.concat(
        [
            pd.DataFrame({"date": dates, "symbol": "AAPL", "close": 100.0, "volume": 10_000}),
            pd.DataFrame({"date": dates, "symbol": "MSFT", "close": 200.0, "volume": 20_000}),
            pd.DataFrame({"date": dates, "symbol": "NVDA", "close": 300.0, "volume": 15_000}),
        ],
        ignore_index=True,
    )

    selected = select_stage0_universe(assets=assets, bars=bars, symbol_count=2, trailing_sessions=20)

    assert selected == ["NVDA", "MSFT"]


def test_stage0_universe_builder_fetches_assets_and_trailing_bars() -> None:
    assets = pd.DataFrame(
        [
            {"symbol": "AAPL", "name": "Apple Inc.", "exchange": "NASDAQ", "status": "active", "tradable": True, "asset_class": "us_equity"},
            {"symbol": "MSFT", "name": "Microsoft Corp", "exchange": "NASDAQ", "status": "active", "tradable": True, "asset_class": "us_equity"},
            {"symbol": "SPY", "name": "SPDR S&P 500 ETF Trust", "exchange": "NYSE", "status": "active", "tradable": True, "asset_class": "us_equity"},
        ]
    )
    dates = pd.bdate_range("2026-03-02", periods=20)
    bars = pd.concat(
        [
            pd.DataFrame({"date": dates, "symbol": "AAPL", "close": 100.0, "volume": 1_000}),
            pd.DataFrame({"date": dates, "symbol": "MSFT", "close": 100.0, "volume": 2_000}),
            pd.DataFrame({"date": dates, "symbol": "SPY", "close": 500.0, "volume": 10_000}),
        ],
        ignore_index=True,
    )
    builder = Stage0UniverseBuilder(connector=FakeConnector(assets=assets, bars=bars))

    selected = builder.build(symbol_count=2, as_of_date="2026-03-31")

    assert selected == ["MSFT", "AAPL"]


def test_resolve_bootstrap_stage_uses_programmatic_builder_for_stage0() -> None:
    local_config = {"stage": {"current": 0, "stage_0": {"symbols": 3, "eod_years": 5}}}
    calls: list[int] = []

    current, symbols, years = resolve_bootstrap_stage(
        local_config,
        {},
        universe_builder=lambda count: calls.append(count) or ["AAA", "BBB", "CCC"],
    )

    assert current == 0
    assert symbols == ["AAA", "BBB", "CCC"]
    assert years == 5
    assert calls == [3]
