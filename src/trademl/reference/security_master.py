"""Derived free security-master builders for listing history and ticker changes."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


def build_listing_history(
    *,
    listings: pd.DataFrame,
    alpaca_assets: pd.DataFrame | None = None,
    delistings: pd.DataFrame | None = None,
    reference_tickers: pd.DataFrame | None = None,
    company_profiles: pd.DataFrame | None = None,
    tiingo_tickers: pd.DataFrame | None = None,
    twelve_data_stocks: pd.DataFrame | None = None,
    as_of: str | None = None,
) -> pd.DataFrame:
    """Build a normalized free listing-history table from collected reference sources."""
    frames: list[pd.DataFrame] = []
    verified_at = pd.Timestamp(as_of or datetime.now(tz=UTC))

    if not listings.empty:
        listing_frame = pd.DataFrame(
            {
                "symbol": listings.get("symbol", pd.Series(dtype="string")).astype("string").str.strip().str.upper(),
                "name": listings.get("name", pd.Series(dtype="string")).astype("string"),
                "exchange": listings.get("exchange", pd.Series(dtype="string")).astype("string").str.upper(),
                "asset_type": listings.get("assetType", pd.Series(dtype="string")).map(_normalize_asset_type),
                "ipo_date": _normalize_date_series(listings.get("ipoDate")),
                "delist_date": _normalize_date_series(listings.get("delistingDate")),
                "delist_reason": pd.Series(pd.NA, index=listings.index, dtype="string"),
                "sector": pd.Series(pd.NA, index=listings.index, dtype="string"),
                "industry": pd.Series(pd.NA, index=listings.index, dtype="string"),
                "status": listings.get("status", pd.Series(dtype="string")).astype("string").str.lower(),
                "sources": pd.Series(["alpha_vantage"] * len(listings), dtype="string"),
                "last_verified": pd.Series([verified_at] * len(listings)),
            }
        )
        frames.append(listing_frame)

    if alpaca_assets is not None and not alpaca_assets.empty:
        asset_frame = pd.DataFrame(
            {
                "symbol": alpaca_assets.get("symbol", pd.Series(dtype="string")).astype("string").str.strip().str.upper(),
                "name": alpaca_assets.get("name", pd.Series(dtype="string")).astype("string"),
                "exchange": alpaca_assets.get("exchange", pd.Series(dtype="string")).astype("string").str.upper(),
                "asset_type": alpaca_assets.get("asset_class", pd.Series(dtype="string")).map(_normalize_asset_type),
                "ipo_date": pd.Series(pd.NaT, index=alpaca_assets.index, dtype="datetime64[ns]"),
                "delist_date": pd.Series(pd.NaT, index=alpaca_assets.index, dtype="datetime64[ns]"),
                "delist_reason": pd.Series(pd.NA, index=alpaca_assets.index, dtype="string"),
                "sector": pd.Series(pd.NA, index=alpaca_assets.index, dtype="string"),
                "industry": pd.Series(pd.NA, index=alpaca_assets.index, dtype="string"),
                "status": alpaca_assets.get("status", pd.Series(dtype="string")).astype("string").str.lower(),
                "sources": pd.Series(["alpaca"] * len(alpaca_assets), dtype="string"),
                "last_verified": pd.Series([verified_at] * len(alpaca_assets)),
            }
        )
        frames.append(asset_frame)

    if reference_tickers is not None and not reference_tickers.empty:
        ticker_frame = pd.DataFrame(
            {
                "symbol": reference_tickers.get("ticker", reference_tickers.get("symbol", pd.Series(dtype="string")))
                .astype("string")
                .str.strip()
                .str.upper(),
                "name": reference_tickers.get("name", pd.Series(dtype="string")).astype("string"),
                "exchange": reference_tickers.get(
                    "primary_exchange",
                    reference_tickers.get("exchange", pd.Series(dtype="string")),
                )
                .astype("string")
                .str.upper(),
                "asset_type": reference_tickers.get("type", pd.Series(dtype="string")).map(_normalize_asset_type),
                "ipo_date": _normalize_date_series(reference_tickers.get("list_date")),
                "delist_date": _normalize_date_series(reference_tickers.get("delisted_utc")),
                "delist_reason": pd.Series(pd.NA, index=reference_tickers.index, dtype="string"),
                "sector": pd.Series(pd.NA, index=reference_tickers.index, dtype="string"),
                "industry": pd.Series(pd.NA, index=reference_tickers.index, dtype="string"),
                "status": reference_tickers.get("active", pd.Series(dtype="bool")).map(lambda value: "active" if bool(value) else "delisted"),
                "sources": pd.Series(["massive"] * len(reference_tickers), dtype="string"),
                "last_verified": pd.Series([verified_at] * len(reference_tickers)),
            }
        )
        frames.append(ticker_frame)

    if tiingo_tickers is not None and not tiingo_tickers.empty:
        tiingo_frame = pd.DataFrame(
            {
                "symbol": tiingo_tickers.get("symbol", tiingo_tickers.get("ticker", pd.Series(dtype="string")))
                .astype("string")
                .str.strip()
                .str.upper(),
                "name": tiingo_tickers.get("name", pd.Series(dtype="string")).astype("string"),
                "exchange": tiingo_tickers.get("exchange", tiingo_tickers.get("exchangeCode", pd.Series(dtype="string")))
                .astype("string")
                .str.upper(),
                "asset_type": tiingo_tickers.get("asset_type", tiingo_tickers.get("assetType", pd.Series(dtype="string"))).map(_normalize_asset_type),
                "ipo_date": _normalize_date_series(tiingo_tickers.get("start_date", tiingo_tickers.get("startDate"))),
                "delist_date": _normalize_date_series(tiingo_tickers.get("end_date", tiingo_tickers.get("endDate"))),
                "delist_reason": pd.Series(pd.NA, index=tiingo_tickers.index, dtype="string"),
                "sector": pd.Series(pd.NA, index=tiingo_tickers.index, dtype="string"),
                "industry": pd.Series(pd.NA, index=tiingo_tickers.index, dtype="string"),
                "status": _status_from_delist_dates(tiingo_tickers.get("end_date", tiingo_tickers.get("endDate"))),
                "sources": pd.Series(["tiingo"] * len(tiingo_tickers), dtype="string"),
                "last_verified": pd.Series([verified_at] * len(tiingo_tickers)),
            }
        )
        frames.append(tiingo_frame)

    if twelve_data_stocks is not None and not twelve_data_stocks.empty:
        stock_frame = pd.DataFrame(
            {
                "symbol": twelve_data_stocks.get("symbol", pd.Series(dtype="string")).astype("string").str.strip().str.upper(),
                "name": twelve_data_stocks.get("name", pd.Series(dtype="string")).astype("string"),
                "exchange": twelve_data_stocks.get("exchange", pd.Series(dtype="string")).astype("string").str.upper(),
                "asset_type": twelve_data_stocks.get("type", twelve_data_stocks.get("instrument_type", pd.Series(dtype="string"))).map(_normalize_asset_type),
                "ipo_date": pd.Series(pd.NaT, index=twelve_data_stocks.index, dtype="datetime64[ns]"),
                "delist_date": pd.Series(pd.NaT, index=twelve_data_stocks.index, dtype="datetime64[ns]"),
                "delist_reason": pd.Series(pd.NA, index=twelve_data_stocks.index, dtype="string"),
                "sector": pd.Series(pd.NA, index=twelve_data_stocks.index, dtype="string"),
                "industry": pd.Series(pd.NA, index=twelve_data_stocks.index, dtype="string"),
                "status": pd.Series(["active"] * len(twelve_data_stocks), dtype="string"),
                "sources": pd.Series(["twelve_data"] * len(twelve_data_stocks), dtype="string"),
                "last_verified": pd.Series([verified_at] * len(twelve_data_stocks)),
            }
        )
        frames.append(stock_frame)

    if company_profiles is not None and not company_profiles.empty:
        profile_frame = pd.DataFrame(
            {
                "symbol": company_profiles.get("ticker", company_profiles.get("symbol", pd.Series(dtype="string")))
                .astype("string")
                .str.strip()
                .str.upper(),
                "name": company_profiles.get("name", pd.Series(dtype="string")).astype("string"),
                "exchange": company_profiles.get("exchange", pd.Series(dtype="string")).astype("string").str.upper(),
                "asset_type": pd.Series(pd.NA, index=company_profiles.index, dtype="string"),
                "ipo_date": pd.Series(pd.NaT, index=company_profiles.index, dtype="datetime64[ns]"),
                "delist_date": pd.Series(pd.NaT, index=company_profiles.index, dtype="datetime64[ns]"),
                "delist_reason": pd.Series(pd.NA, index=company_profiles.index, dtype="string"),
                "sector": company_profiles.get("finnhubIndustry", company_profiles.get("sector", pd.Series(dtype="string"))).astype("string"),
                "industry": company_profiles.get("industry", pd.Series(dtype="string")).astype("string"),
                "status": pd.Series(["active"] * len(company_profiles), dtype="string"),
                "sources": pd.Series(["finnhub"] * len(company_profiles), dtype="string"),
                "last_verified": pd.Series([verified_at] * len(company_profiles)),
            }
        )
        frames.append(profile_frame)

    if delistings is not None and not delistings.empty:
        delisting_frame = pd.DataFrame(
            {
                "symbol": delistings.get("symbol", pd.Series(dtype="string")).astype("string").str.strip().str.upper(),
                "name": delistings.get("companyName", delistings.get("name", pd.Series(dtype="string"))).astype("string"),
                "exchange": delistings.get("exchange", pd.Series(dtype="string")).astype("string").str.upper(),
                "asset_type": delistings.get("assetType", pd.Series(dtype="string")).map(_normalize_asset_type),
                "ipo_date": _normalize_date_series(delistings.get("ipoDate")),
                "delist_date": _normalize_date_series(delistings.get("delistedDate", delistings.get("delist_date"))),
                "delist_reason": delistings.get("reason", pd.Series(["unknown"] * len(delistings), dtype="string")).astype("string").fillna("unknown"),
                "sector": delistings.get("sector", pd.Series(dtype="string")).astype("string"),
                "industry": delistings.get("industry", pd.Series(dtype="string")).astype("string"),
                "status": pd.Series(["delisted"] * len(delistings), dtype="string"),
                "sources": pd.Series(["fmp"] * len(delistings), dtype="string"),
                "last_verified": pd.Series([verified_at] * len(delistings)),
            }
        )
        frames.append(delisting_frame)

    if not frames:
        return pd.DataFrame(columns=_listing_history_columns())

    combined = pd.concat(frames, ignore_index=True)
    combined["symbol"] = combined["symbol"].astype("string").str.strip().str.upper()
    combined = combined.dropna(subset=["symbol"])
    combined = combined.loc[combined["symbol"] != ""].copy()
    grouped_rows: list[dict[str, object]] = []
    for symbol, group in combined.groupby("symbol", sort=True):
        source_values = sorted({str(value) for value in group["sources"].dropna().astype("string") if str(value)})
        status = "delisted" if "delisted" in set(group["status"].dropna().astype("string")) else "active"
        asset_type = _first_nonempty(group["asset_type"])
        grouped_rows.append(
            {
                "symbol": symbol,
                "name": _first_nonempty(group["name"]),
                "exchange": _first_nonempty(group["exchange"]),
                "asset_type": asset_type if pd.notna(asset_type) else "common_stock",
                "ipo_date": _min_date(group["ipo_date"]),
                "delist_date": _max_date(group["delist_date"]),
                "delist_reason": _first_nonempty(group["delist_reason"]),
                "sector": _first_nonempty(group["sector"]),
                "industry": _first_nonempty(group["industry"]),
                "status": status,
                "sources": ",".join(source_values),
                "last_verified": verified_at,
            }
        )
    return pd.DataFrame(grouped_rows, columns=_listing_history_columns()).sort_values(["status", "symbol"]).reset_index(drop=True)


def build_ticker_changes(symbol_changes: pd.DataFrame, *, as_of: str | None = None) -> pd.DataFrame:
    """Normalize free symbol-change history into a canonical ticker-change table."""
    if symbol_changes.empty:
        return pd.DataFrame(columns=_ticker_change_columns())
    verified_at = pd.Timestamp(as_of or datetime.now(tz=UTC))
    normalized = pd.DataFrame(
        {
            "old_symbol": symbol_changes.get("oldSymbol", symbol_changes.get("old_symbol", pd.Series(dtype="string")))
            .astype("string")
            .str.strip()
            .str.upper(),
            "new_symbol": symbol_changes.get("newSymbol", symbol_changes.get("new_symbol", pd.Series(dtype="string")))
            .astype("string")
            .str.strip()
            .str.upper(),
            "change_date": _normalize_date_series(symbol_changes.get("date", symbol_changes.get("change_date"))),
            "cik": symbol_changes.get("cik", pd.Series(dtype="string")).astype("string"),
            "reason": symbol_changes.get("reason", pd.Series(["rename"] * len(symbol_changes), dtype="string")).astype("string"),
            "source": symbol_changes.get("source", pd.Series(["fmp"] * len(symbol_changes), dtype="string")).astype("string"),
            "last_verified": pd.Series([verified_at] * len(symbol_changes)),
        }
    )
    normalized = normalized.dropna(subset=["old_symbol", "new_symbol", "change_date"])
    normalized = normalized.loc[(normalized["old_symbol"] != "") & (normalized["new_symbol"] != "")]
    return normalized[_ticker_change_columns()].drop_duplicates().sort_values(["change_date", "old_symbol", "new_symbol"]).reset_index(drop=True)


def rebuild_derived_references(reference_root: Path) -> list[Path]:
    """Rebuild normalized derived reference files from raw collected source files."""
    outputs: list[Path] = []
    listing_inputs = {
        "listings": _read_optional_parquet(reference_root / "listings.parquet"),
        "alpaca_assets": _read_optional_parquet(reference_root / "alpaca_assets.parquet"),
        "delistings": _read_optional_parquet(reference_root / "delistings.parquet"),
        "reference_tickers": _read_optional_parquet(reference_root / "universe.parquet"),
        "company_profiles": _read_optional_parquet(reference_root / "company_profiles.parquet"),
        "tiingo_tickers": _read_optional_parquet(reference_root / "tiingo_tickers.parquet"),
        "twelve_data_stocks": _read_optional_parquet(reference_root / "twelve_data_stocks.parquet"),
    }
    if any(not frame.empty for frame in listing_inputs.values()):
        listing_history = build_listing_history(**listing_inputs)
        output = reference_root / "listing_history.parquet"
        output.parent.mkdir(parents=True, exist_ok=True)
        listing_history.to_parquet(output, index=False)
        outputs.append(output)

    symbol_changes = _read_optional_parquet(reference_root / "symbol_changes.parquet")
    if not symbol_changes.empty:
        output = reference_root / "ticker_changes.parquet"
        output.parent.mkdir(parents=True, exist_ok=True)
        build_ticker_changes(symbol_changes).to_parquet(output, index=False)
        outputs.append(output)
    return outputs


def _read_optional_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _normalize_asset_type(value: object) -> str | pd.NA:
    raw = str(value or "").strip().lower()
    if not raw:
        return pd.NA
    if raw in {"stock", "cs", "common stock", "common_stock", "common"}:
        return "common_stock"
    if "etf" in raw:
        return "etf"
    if "preferred" in raw:
        return "preferred"
    return raw.replace(" ", "_")


def _status_from_delist_dates(values: pd.Series | None) -> pd.Series:
    if values is None:
        return pd.Series(dtype="string")
    parsed = pd.to_datetime(values, errors="coerce")
    return parsed.map(lambda value: "delisted" if pd.notna(value) else "active").astype("string")


def _normalize_date_series(values: pd.Series | None) -> pd.Series:
    if values is None:
        return pd.Series(dtype="datetime64[ns]")
    parsed = pd.to_datetime(values, errors="coerce")
    return parsed.dt.normalize()


def _first_nonempty(series: pd.Series) -> str | pd.NA:
    for value in series.astype("string"):
        if pd.notna(value) and str(value).strip():
            return str(value)
    return pd.NA


def _min_date(series: pd.Series) -> pd.Timestamp | pd.NaT:
    parsed = pd.to_datetime(series, errors="coerce").dropna()
    return parsed.min() if not parsed.empty else pd.NaT


def _max_date(series: pd.Series) -> pd.Timestamp | pd.NaT:
    parsed = pd.to_datetime(series, errors="coerce").dropna()
    return parsed.max() if not parsed.empty else pd.NaT


def _listing_history_columns() -> list[str]:
    return [
        "symbol",
        "name",
        "exchange",
        "asset_type",
        "ipo_date",
        "delist_date",
        "delist_reason",
        "sector",
        "industry",
        "status",
        "sources",
        "last_verified",
    ]


def _ticker_change_columns() -> list[str]:
    return ["old_symbol", "new_symbol", "change_date", "cik", "reason", "source", "last_verified"]
