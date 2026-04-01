from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from trademl.reference.universe import active_listing_candidates, build_stage1_universe, build_time_varying_universe


@dataclass
class FakeConnector:
    assets: pd.DataFrame
    bars: pd.DataFrame

    def fetch(self, dataset: str, symbols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        if dataset == "assets":
            return self.assets.copy()
        if dataset != "equities_eod":
            raise ValueError(dataset)
        return self.bars.loc[self.bars["symbol"].isin(symbols)].copy()


def test_active_listing_candidates_filters_as_of_date_and_asset_type() -> None:
    listing_history = pd.DataFrame(
        [
            {"symbol": "AAPL", "exchange": "NASDAQ", "asset_type": "common_stock", "ipo_date": "1980-12-12", "delist_date": None},
            {"symbol": "OLD", "exchange": "NASDAQ", "asset_type": "common_stock", "ipo_date": "1980-01-01", "delist_date": "2024-01-05"},
            {"symbol": "ETF1", "exchange": "NYSE", "asset_type": "etf", "ipo_date": "2010-01-01", "delist_date": None},
        ]
    )

    candidates = active_listing_candidates(listing_history, as_of_date="2025-01-02")

    assert candidates["symbol"].tolist() == ["AAPL"]


def test_build_stage1_universe_ranks_active_candidates_by_adv() -> None:
    assets = pd.DataFrame(
        [
            {"symbol": "AAPL", "name": "Apple Inc.", "exchange": "NASDAQ", "status": "active", "tradable": True, "asset_class": "us_equity"},
            {"symbol": "MSFT", "name": "Microsoft Corp", "exchange": "NASDAQ", "status": "active", "tradable": True, "asset_class": "us_equity"},
            {"symbol": "NVDA", "name": "NVIDIA Corp", "exchange": "NASDAQ", "status": "active", "tradable": True, "asset_class": "us_equity"},
        ]
    )
    listing_history = pd.DataFrame(
        [
            {"symbol": "AAPL", "exchange": "NASDAQ", "asset_type": "common_stock", "ipo_date": "1980-12-12", "delist_date": None},
            {"symbol": "MSFT", "exchange": "NASDAQ", "asset_type": "common_stock", "ipo_date": "1986-03-13", "delist_date": None},
            {"symbol": "NVDA", "exchange": "NASDAQ", "asset_type": "common_stock", "ipo_date": "1999-01-22", "delist_date": None},
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

    symbols = build_stage1_universe(
        listing_history=listing_history,
        connector=FakeConnector(assets=assets, bars=bars),
        symbol_count=2,
        as_of_date="2026-03-31",
    )

    assert symbols == ["NVDA", "MSFT"]


def test_build_stage1_universe_intersects_listing_history_with_current_tradable_assets() -> None:
    assets = pd.DataFrame(
        [
            {"symbol": "AAPL", "name": "Apple Inc.", "exchange": "NASDAQ", "status": "active", "tradable": True, "asset_class": "us_equity"},
            {"symbol": "MSFT", "name": "Microsoft Corp", "exchange": "NASDAQ", "status": "active", "tradable": True, "asset_class": "us_equity"},
        ]
    )
    listing_history = pd.DataFrame(
        [
            {"symbol": "-P-HIZ", "exchange": "NASDAQ", "asset_type": "common_stock", "ipo_date": "2023-08-30", "delist_date": None},
            {"symbol": "AAPL", "exchange": "NASDAQ", "asset_type": "common_stock", "ipo_date": "1980-12-12", "delist_date": None},
            {"symbol": "MSFT", "exchange": "NASDAQ", "asset_type": "common_stock", "ipo_date": "1986-03-13", "delist_date": None},
        ]
    )
    dates = pd.bdate_range("2026-03-02", periods=20)
    bars = pd.concat(
        [
            pd.DataFrame({"date": dates, "symbol": "AAPL", "close": 100.0, "volume": 10_000}),
            pd.DataFrame({"date": dates, "symbol": "MSFT", "close": 200.0, "volume": 20_000}),
        ],
        ignore_index=True,
    )

    symbols = build_stage1_universe(
        listing_history=listing_history,
        connector=FakeConnector(assets=assets, bars=bars),
        symbol_count=5,
        as_of_date="2026-03-31",
    )

    assert symbols == ["MSFT", "AAPL"]


def test_build_time_varying_universe_returns_rebalance_specific_membership() -> None:
    listing_history = pd.DataFrame(
        [
            {"symbol": "AAPL", "exchange": "NASDAQ", "asset_type": "common_stock", "ipo_date": "1980-12-12", "delist_date": None},
            {"symbol": "NEW", "exchange": "NASDAQ", "asset_type": "common_stock", "ipo_date": "2026-01-10", "delist_date": None},
        ]
    )
    dates = pd.bdate_range("2026-01-02", periods=30)
    bars = pd.concat(
        [
            pd.DataFrame({"date": dates, "symbol": "AAPL", "close": 100.0, "volume": 10_000}),
            pd.DataFrame({"date": dates[5:], "symbol": "NEW", "close": 50.0, "volume": 30_000}),
        ],
        ignore_index=True,
    )

    universe = build_time_varying_universe(
        listing_history=listing_history,
        daily_bars=bars,
        rebalance_dates=["2026-01-09", "2026-01-30"],
        top_n=2,
    )

    early = universe.loc[universe["date"] == pd.Timestamp("2026-01-09")]
    late = universe.loc[universe["date"] == pd.Timestamp("2026-01-30")]
    assert early["symbol"].tolist() == ["AAPL"]
    assert set(late["symbol"]) == {"AAPL", "NEW"}
