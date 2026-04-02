from __future__ import annotations

from pathlib import Path

import pandas as pd

from trademl.reference.security_master import build_listing_history, build_ticker_changes, rebuild_derived_references


def test_build_listing_history_merges_free_sources() -> None:
    listings = pd.DataFrame(
        [
            {
                "symbol": "AAPL",
                "name": "Apple Inc.",
                "exchange": "NASDAQ",
                "assetType": "Stock",
                "ipoDate": "1980-12-12",
                "delistingDate": None,
                "status": "Active",
            },
            {
                "symbol": "OLD",
                "name": "Old Co",
                "exchange": "NASDAQ",
                "assetType": "Stock",
                "ipoDate": "1980-01-01",
                "delistingDate": "2024-01-05",
                "status": "Delisted",
            },
        ]
    )
    delistings = pd.DataFrame([{"symbol": "OLD", "companyName": "Old Co", "delistedDate": "2024-01-05", "reason": "acquired"}])
    reference_tickers = pd.DataFrame([{"ticker": "AAPL", "name": "Apple Inc.", "primary_exchange": "NASDAQ", "type": "CS", "active": True}])
    tiingo_tickers = pd.DataFrame([{"ticker": "AAPL", "name": "Apple Inc.", "exchangeCode": "NASDAQ", "assetType": "Stock", "startDate": "1980-12-12"}])
    twelve_data_stocks = pd.DataFrame([{"symbol": "AAPL", "name": "Apple Inc.", "exchange": "NASDAQ", "type": "Common Stock"}])

    listing_history = build_listing_history(
        listings=listings,
        delistings=delistings,
        reference_tickers=reference_tickers,
        tiingo_tickers=tiingo_tickers,
        twelve_data_stocks=twelve_data_stocks,
        as_of="2026-04-01",
    )

    aapl = listing_history.loc[listing_history["symbol"] == "AAPL"].iloc[0]
    old = listing_history.loc[listing_history["symbol"] == "OLD"].iloc[0]
    assert aapl["asset_type"] == "common_stock"
    assert "alpha_vantage" in aapl["sources"]
    assert "massive" in aapl["sources"]
    assert "tiingo" in aapl["sources"]
    assert "twelve_data" in aapl["sources"]
    assert old["status"] == "delisted"
    assert old["delist_reason"] == "acquired"


def test_build_ticker_changes_normalizes_symbol_change_history() -> None:
    raw = pd.DataFrame([{"oldSymbol": "FB", "newSymbol": "META", "date": "2022-06-09"}])

    changes = build_ticker_changes(raw, as_of="2026-04-01")

    assert changes.iloc[0]["old_symbol"] == "FB"
    assert changes.iloc[0]["new_symbol"] == "META"
    assert str(changes.iloc[0]["change_date"].date()) == "2022-06-09"


def test_rebuild_derived_references_writes_listing_and_ticker_change_outputs(tmp_path: Path) -> None:
    reference_root = tmp_path / "reference"
    reference_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "symbol": "AAPL",
                "name": "Apple Inc.",
                "exchange": "NASDAQ",
                "assetType": "Stock",
                "ipoDate": "1980-12-12",
                "delistingDate": None,
                "status": "Active",
            }
        ]
    ).to_parquet(reference_root / "listings.parquet", index=False)
    pd.DataFrame([{"ticker": "AAPL", "name": "Apple Inc.", "exchangeCode": "NASDAQ", "assetType": "Stock", "startDate": "1980-12-12"}]).to_parquet(
        reference_root / "tiingo_tickers.parquet",
        index=False,
    )
    pd.DataFrame([{"oldSymbol": "FB", "newSymbol": "META", "date": "2022-06-09"}]).to_parquet(
        reference_root / "symbol_changes.parquet",
        index=False,
    )

    outputs = rebuild_derived_references(reference_root)

    assert reference_root / "listing_history.parquet" in outputs
    assert reference_root / "ticker_changes.parquet" in outputs
    assert (reference_root / "listing_history.parquet").exists()
    assert (reference_root / "ticker_changes.parquet").exists()
