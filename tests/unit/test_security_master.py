from __future__ import annotations

from pathlib import Path

import pandas as pd

import trademl.reference.security_master as security_master
from trademl.reference.security_master import (
    build_listing_history,
    build_sec_companyfacts_fundamentals,
    build_ticker_changes,
    rebuild_derived_references,
)


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


def test_build_sec_companyfacts_fundamentals_normalizes_gzip_payload(tmp_path: Path) -> None:
    facts_path = tmp_path / "companyfacts.json.gz"
    payload = {
        "cik": "320193",
        "facts": {
            "us-gaap": {
                "Assets": {
                    "units": {
                        "USD": [
                            {
                                "end": "2024-09-28",
                                "filed": "2024-11-01",
                                "val": 364980000000,
                                "form": "10-K",
                            }
                        ]
                    }
                }
            }
        },
    }
    import gzip
    import json

    with gzip.open(facts_path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)
    index = pd.DataFrame([{"cik": "0000320193", "facts_path": str(facts_path), "captured_at": "2026-04-01"}])
    tickers = pd.DataFrame([{"cik_str": "320193", "ticker": "AAPL"}])

    fundamentals = build_sec_companyfacts_fundamentals(
        companyfacts_index=index,
        sec_company_tickers=tickers,
    )

    assert fundamentals[["symbol", "metric_date", "metric_name", "metric_value", "source"]].to_dict("records") == [
        {
            "symbol": "AAPL",
            "metric_date": pd.Timestamp("2024-09-28"),
            "metric_name": "us-gaap:Assets:USD",
            "metric_value": "364980000000",
            "source": "sec_edgar_companyfacts",
        }
    ]
    assert fundamentals.iloc[0]["last_verified"] == pd.Timestamp("2024-11-02")


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


def test_rebuild_derived_references_skips_cached_universe_rebuild_when_inputs_unchanged(tmp_path: Path, monkeypatch) -> None:
    reference_root = tmp_path / "data" / "reference"
    raw_bars_root = tmp_path / "data" / "raw" / "equities_bars" / "date=2026-04-01"
    reference_root.mkdir(parents=True, exist_ok=True)
    raw_bars_root.mkdir(parents=True, exist_ok=True)
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
    pd.DataFrame(
        [
            {
                "date": "2026-04-01",
                "symbol": "AAPL",
                "close": 100.0,
                "volume": 1_000_000,
            }
        ]
    ).to_parquet(raw_bars_root / "data.parquet", index=False)

    first_outputs = rebuild_derived_references(reference_root)
    assert (reference_root / ".derived_references_state.json").exists()

    monkeypatch.setattr(
        security_master,
        "build_universe_snapshots",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("unexpected rebuild")),
    )

    second_outputs = rebuild_derived_references(reference_root)

    assert first_outputs == second_outputs
