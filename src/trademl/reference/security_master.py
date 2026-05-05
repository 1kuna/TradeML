"""Derived free security-master builders for Phase 2 reference artifacts."""

from __future__ import annotations

import gzip
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from trademl.reference.universe import build_time_varying_universe


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


def build_security_master(
    *,
    listing_history: pd.DataFrame,
    ticker_changes: pd.DataFrame | None = None,
    sec_company_tickers: pd.DataFrame | None = None,
    as_of: str | None = None,
) -> pd.DataFrame:
    """Build a normalized security master with issuer linkage and rename continuity."""
    if listing_history.empty:
        return pd.DataFrame(columns=_security_master_columns())
    verified_at = pd.Timestamp(as_of or datetime.now(tz=UTC))
    listing = listing_history.copy()
    listing["symbol"] = listing.get("symbol", pd.Series(dtype="string")).astype("string").str.strip().str.upper()
    listing["ipo_date"] = _normalize_date_series(listing.get("ipo_date"))
    listing["delist_date"] = _normalize_date_series(listing.get("delist_date"))

    cik_map: dict[str, str] = {}
    if sec_company_tickers is not None and not sec_company_tickers.empty:
        ticker_col = "ticker" if "ticker" in sec_company_tickers.columns else "symbol"
        cik_col = "cik_str" if "cik_str" in sec_company_tickers.columns else "cik"
        cik_rows = sec_company_tickers[[ticker_col, cik_col]].dropna()
        cik_map = {
            str(row[ticker_col]).strip().upper(): str(row[cik_col]).strip()
            for row in cik_rows.to_dict("records")
            if str(row[ticker_col]).strip() and str(row[cik_col]).strip()
        }

    old_to_new: dict[str, tuple[str, str | None, pd.Timestamp | None]] = {}
    if ticker_changes is not None and not ticker_changes.empty:
        normalized_changes = ticker_changes.copy()
        normalized_changes["old_symbol"] = normalized_changes.get("old_symbol", pd.Series(dtype="string")).astype("string").str.strip().str.upper()
        normalized_changes["new_symbol"] = normalized_changes.get("new_symbol", pd.Series(dtype="string")).astype("string").str.strip().str.upper()
        normalized_changes["change_date"] = _normalize_date_series(normalized_changes.get("change_date"))
        normalized_changes["cik"] = normalized_changes.get("cik", pd.Series(dtype="string")).astype("string")
        for row in normalized_changes.dropna(subset=["old_symbol", "new_symbol"]).to_dict("records"):
            old_to_new[str(row["old_symbol"])] = (
                str(row["new_symbol"]),
                str(row["cik"]).strip() or None,
                row.get("change_date"),
            )

    rows: list[dict[str, object]] = []
    for row in listing.to_dict("records"):
        symbol = str(row.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        chain_target, chain_cik, change_date = old_to_new.get(symbol, (symbol, None, None))
        cik = chain_cik or cik_map.get(symbol) or cik_map.get(chain_target)
        issuer_key = f"cik:{cik}" if cik else f"symbol:{chain_target}"
        rows.append(
            {
                "issuer_key": issuer_key,
                "cik": cik,
                "symbol": symbol,
                "primary_symbol": chain_target,
                "name": row.get("name"),
                "exchange": row.get("exchange"),
                "asset_type": row.get("asset_type"),
                "ipo_date": row.get("ipo_date"),
                "delist_date": row.get("delist_date"),
                "change_date": change_date,
                "status": row.get("status"),
                "sources": row.get("sources"),
                "last_verified": verified_at,
            }
        )
    return pd.DataFrame(rows, columns=_security_master_columns()).sort_values(
        ["issuer_key", "symbol", "ipo_date"],
        na_position="last",
    ).reset_index(drop=True)


def build_earnings_calendar_pit(*, earnings_frames: list[pd.DataFrame], as_of: str | None = None) -> pd.DataFrame:
    """Build a normalized PIT earnings calendar from corroborating free sources."""
    normalized: list[pd.DataFrame] = []
    verified_at = pd.Timestamp(as_of or datetime.now(tz=UTC))
    for frame in earnings_frames:
        if frame is None or frame.empty:
            continue
        source = _first_nonempty(frame.get("source", pd.Series(dtype="string")))
        normalized.append(
            pd.DataFrame(
                {
                    "symbol": frame.get("symbol", frame.get("ticker", pd.Series(dtype="string"))).astype("string").str.strip().str.upper(),
                    "earnings_date": _normalize_date_series(frame.get("date", frame.get("earnings_date"))),
                    "fiscal_period": frame.get("fiscalDateEnding", frame.get("fiscal_period", pd.Series(dtype="string"))).astype("string"),
                    "source": frame.get("source", pd.Series([source or "unknown"] * len(frame), dtype="string")).astype("string"),
                    "last_verified": pd.Series([verified_at] * len(frame)),
                }
            )
        )
    if not normalized:
        return pd.DataFrame(columns=_earnings_calendar_columns())
    combined = pd.concat(normalized, ignore_index=True)
    combined = combined.dropna(subset=["symbol", "earnings_date"])
    if combined.empty:
        return pd.DataFrame(columns=_earnings_calendar_columns())
    grouped = (
        combined.groupby(["symbol", "earnings_date", "fiscal_period"], dropna=False)
        .agg(
            source_count=("source", lambda values: len({str(value) for value in values if str(value)})),
            sources=("source", lambda values: ",".join(sorted({str(value) for value in values if str(value)}))),
            last_verified=("last_verified", "max"),
        )
        .reset_index()
    )
    return grouped[_earnings_calendar_columns()].sort_values(["earnings_date", "symbol"]).reset_index(drop=True)


def build_fundamentals_daily(
    *,
    company_profiles: pd.DataFrame | None = None,
    companyfacts: pd.DataFrame | None = None,
    financial_statements: pd.DataFrame | None = None,
    as_of: str | None = None,
) -> pd.DataFrame:
    """Build a normalized long-form daily fundamentals table from free sources."""
    verified_at = pd.Timestamp(as_of or datetime.now(tz=UTC))
    rows: list[dict[str, object]] = []
    if company_profiles is not None and not company_profiles.empty:
        for item in company_profiles.to_dict("records"):
            symbol = str(item.get("ticker") or item.get("symbol") or "").strip().upper()
            if not symbol:
                continue
            for metric in ("marketCapitalization", "shareOutstanding", "finnhubIndustry", "exchange", "country"):
                if metric not in item or item.get(metric) in (None, ""):
                    continue
                rows.append(
                    {
                        "symbol": symbol,
                        "metric_date": verified_at.normalize(),
                        "metric_name": metric,
                        "metric_value": str(item.get(metric)),
                        "source": "finnhub",
                        "last_verified": verified_at,
                    }
                )
    if financial_statements is not None and not financial_statements.empty:
        for item in financial_statements.to_dict("records"):
            symbol = str(item.get("symbol") or "").strip().upper()
            metric_date = pd.to_datetime(item.get("date"), errors="coerce")
            if not symbol or pd.isna(metric_date):
                continue
            for key, value in item.items():
                if key in {"symbol", "date", "statement_type", "source"} or value in (None, ""):
                    continue
                rows.append(
                    {
                        "symbol": symbol,
                        "metric_date": metric_date.normalize(),
                        "metric_name": f"{item.get('statement_type', 'statement')}:{key}",
                        "metric_value": str(value),
                        "source": str(item.get("source") or "twelve_data"),
                        "last_verified": verified_at,
                    }
                )
    if companyfacts is not None and not companyfacts.empty:
        for item in companyfacts.to_dict("records"):
            symbol = str(item.get("symbol") or item.get("ticker") or "").strip().upper()
            metric_name = str(item.get("metric_name") or item.get("fact") or "").strip()
            metric_value = item.get("metric_value", item.get("value"))
            metric_date = pd.to_datetime(item.get("metric_date", item.get("date")), errors="coerce")
            if not symbol or not metric_name or metric_value in (None, "") or pd.isna(metric_date):
                continue
            rows.append(
                {
                    "symbol": symbol,
                    "metric_date": metric_date.normalize(),
                    "metric_name": metric_name,
                    "metric_value": str(metric_value),
                    "source": str(item.get("source") or "sec_edgar"),
                    "last_verified": verified_at,
                }
            )
    if not rows:
        return pd.DataFrame(columns=_fundamentals_daily_columns())
    frame = pd.DataFrame(rows, columns=_fundamentals_daily_columns()).drop_duplicates(
        subset=["symbol", "metric_date", "metric_name", "source", "metric_value"],
        keep="last",
    )
    return frame.sort_values(["metric_date", "symbol", "metric_name"]).reset_index(drop=True)


def build_sec_companyfacts_fundamentals(
    *,
    companyfacts_index: pd.DataFrame,
    sec_company_tickers: pd.DataFrame | None = None,
    reference_root: Path | None = None,
) -> pd.DataFrame:
    """Normalize streamed SEC companyfacts payloads into PIT-safe long-form fundamentals."""
    if companyfacts_index.empty:
        return pd.DataFrame(columns=_fundamentals_daily_columns())
    ticker_by_cik = _ticker_by_cik(sec_company_tickers)
    rows: list[dict[str, object]] = []
    for item in companyfacts_index.to_dict("records"):
        path = _resolve_companyfacts_path(item, reference_root=reference_root)
        if path is None or not path.exists():
            continue
        try:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError, EOFError):
            continue
        cik = _normalize_cik(item.get("cik") or payload.get("cik"))
        symbol = str(item.get("symbol") or ticker_by_cik.get(cik, "")).strip().upper()
        if not symbol:
            continue
        facts = payload.get("facts") if isinstance(payload, dict) else {}
        if not isinstance(facts, dict):
            continue
        for namespace, namespace_payload in sorted(
            facts.items(), key=lambda pair: str(pair[0])
        ):
            if not isinstance(namespace_payload, dict):
                continue
            for concept, concept_payload in sorted(
                namespace_payload.items(), key=lambda pair: str(pair[0])
            ):
                units = concept_payload.get("units") if isinstance(concept_payload, dict) else {}
                if not isinstance(units, dict):
                    continue
                for unit, values in sorted(units.items(), key=lambda pair: str(pair[0])):
                    if not isinstance(values, list):
                        continue
                    for fact in values:
                        if not isinstance(fact, dict):
                            continue
                        value = fact.get("val")
                        metric_date = pd.to_datetime(fact.get("end"), errors="coerce")
                        filed_date = pd.to_datetime(fact.get("filed"), errors="coerce")
                        if value in (None, "") or pd.isna(metric_date):
                            continue
                        # Companyfacts has filing dates, not acceptance timestamps. Use next day
                        # availability so daily modeling cannot see same-day after-close facts.
                        last_verified = (
                            filed_date.normalize() + pd.Timedelta(days=1)
                            if pd.notna(filed_date)
                            else pd.to_datetime(item.get("captured_at"), errors="coerce")
                        )
                        rows.append(
                            {
                                "symbol": symbol,
                                "metric_date": metric_date.normalize(),
                                "metric_name": f"{namespace}:{concept}:{unit}",
                                "metric_value": str(value),
                                "source": "sec_edgar_companyfacts",
                                "last_verified": last_verified,
                            }
                        )
    if not rows:
        return pd.DataFrame(columns=_fundamentals_daily_columns())
    frame = pd.DataFrame(rows, columns=_fundamentals_daily_columns()).dropna(
        subset=["symbol", "metric_date", "metric_name", "last_verified"]
    )
    frame = frame.drop_duplicates(
        subset=["symbol", "metric_date", "metric_name", "source"],
        keep="last",
    )
    return frame.sort_values(["metric_date", "symbol", "metric_name"]).reset_index(drop=True)


def write_sec_companyfacts_fundamentals(
    *,
    reference_root: Path,
    output: Path | None = None,
    chunk_size: int = 25,
) -> dict[str, object]:
    """Write SEC companyfacts fundamentals with bounded per-chunk memory."""
    index_path = reference_root / "sec_companyfacts.parquet"
    ticker_path = reference_root / "sec_company_tickers.parquet"
    output_path = output or reference_root / "fundamentals_daily.parquet"
    companyfacts_index = _read_optional_parquet(index_path)
    sec_company_tickers = _read_optional_parquet(ticker_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    writer: pq.ParquetWriter | None = None
    rows = 0
    chunks = 0
    try:
        if companyfacts_index.empty:
            empty = pd.DataFrame(columns=_fundamentals_daily_columns())
            empty.to_parquet(tmp_path, index=False)
        else:
            step = max(1, int(chunk_size))
            for start in range(0, len(companyfacts_index), step):
                chunk = companyfacts_index.iloc[start : start + step].copy()
                frame = build_sec_companyfacts_fundamentals(
                    companyfacts_index=chunk,
                    sec_company_tickers=sec_company_tickers,
                    reference_root=reference_root,
                )
                if frame.empty:
                    continue
                table = pa.Table.from_pandas(frame, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(tmp_path, table.schema)
                writer.write_table(table.cast(writer.schema))
                rows += int(len(frame))
                chunks += 1
            if writer is None:
                pd.DataFrame(columns=_fundamentals_daily_columns()).to_parquet(
                    tmp_path,
                    index=False,
                )
        if writer is not None:
            writer.close()
            writer = None
        tmp_path.replace(output_path)
    finally:
        if writer is not None:
            writer.close()
        if tmp_path.exists() and not output_path.exists():
            tmp_path.unlink()
    return {
        "output": str(output_path),
        "rows": rows,
        "chunks": chunks,
        "input_rows": int(len(companyfacts_index)),
    }


def build_sec_filing_index(*, filing_index: pd.DataFrame, sec_company_tickers: pd.DataFrame | None = None) -> pd.DataFrame:
    """Normalize SEC filings into a symbol-aware filing index."""
    if filing_index.empty:
        return pd.DataFrame(columns=_sec_filing_index_columns())
    ticker_by_cik: dict[str, str] = {}
    if sec_company_tickers is not None and not sec_company_tickers.empty:
        ticker_col = "ticker" if "ticker" in sec_company_tickers.columns else "symbol"
        cik_col = "cik_str" if "cik_str" in sec_company_tickers.columns else "cik"
        ticker_by_cik = {
            str(row[cik_col]).strip(): str(row[ticker_col]).strip().upper()
            for row in sec_company_tickers[[ticker_col, cik_col]].dropna().to_dict("records")
            if str(row[cik_col]).strip() and str(row[ticker_col]).strip()
        }
    normalized = pd.DataFrame(
        {
            "cik": filing_index.get("cik", pd.Series(dtype="string")).astype("string").str.strip(),
            "symbol": filing_index.get("symbol", pd.Series(dtype="string")).astype("string").str.strip().str.upper(),
            "form": filing_index.get("form", pd.Series(dtype="string")).astype("string"),
            "filing_date": _normalize_date_series(filing_index.get("filingDate", filing_index.get("filing_date"))),
            "accepted_at": filing_index.get("acceptanceDateTime", filing_index.get("accepted_at", pd.Series(dtype="string"))).astype("string"),
            "accession_number": filing_index.get("accessionNumber", filing_index.get("accession_number", pd.Series(dtype="string"))).astype("string"),
            "source": filing_index.get("source", pd.Series(["sec_edgar"] * len(filing_index), dtype="string")).astype("string"),
        }
    )
    normalized["symbol"] = normalized["symbol"].fillna("")
    normalized.loc[normalized["symbol"] == "", "symbol"] = normalized["cik"].map(ticker_by_cik).fillna("")
    normalized = normalized.dropna(subset=["cik", "filing_date", "form"])
    normalized = normalized.loc[normalized["cik"] != ""].copy()
    return normalized[_sec_filing_index_columns()].drop_duplicates().sort_values(
        ["filing_date", "cik", "form"],
    ).reset_index(drop=True)


def build_macro_vintages(
    *,
    vintagedates: pd.DataFrame | None = None,
    macro_root: Path | None = None,
) -> pd.DataFrame:
    """Build a normalized macro vintage coverage table."""
    if vintagedates is not None and not vintagedates.empty:
        frame = pd.DataFrame(
            {
                "series_id": vintagedates.get("series_id", pd.Series(dtype="string")).astype("string"),
                "vintage_date": _normalize_date_series(vintagedates.get("vintage_date")),
                "source": vintagedates.get("source", pd.Series(["fred"] * len(vintagedates), dtype="string")).astype("string"),
            }
        ).dropna(subset=["series_id", "vintage_date"])
        if not frame.empty:
            return frame[_macro_vintages_columns()].drop_duplicates().sort_values(["series_id", "vintage_date"]).reset_index(drop=True)
    if macro_root is None or not macro_root.exists():
        return pd.DataFrame(columns=_macro_vintages_columns())
    rows: list[dict[str, object]] = []
    for path in sorted(macro_root.glob("series=*/data.parquet")):
        series_id = path.parent.name.partition("=")[2]
        try:
            frame = pd.read_parquet(path, columns=["vintage_date"])
        except Exception:
            continue
        if frame.empty or "vintage_date" not in frame.columns:
            continue
        for vintage_date in pd.to_datetime(frame["vintage_date"], errors="coerce").dropna().dt.normalize().unique():
            rows.append({"series_id": series_id, "vintage_date": vintage_date, "source": "fred"})
    if not rows:
        return pd.DataFrame(columns=_macro_vintages_columns())
    return pd.DataFrame(rows, columns=_macro_vintages_columns()).drop_duplicates().sort_values(["series_id", "vintage_date"]).reset_index(drop=True)


def build_universe_snapshots(
    *,
    listing_history: pd.DataFrame,
    daily_bars: pd.DataFrame,
    top_n: int = 500,
    rebalance_frequency: str = "MS",
) -> pd.DataFrame:
    """Build time-varying universe snapshots from listing history and daily bars."""
    if listing_history.empty or daily_bars.empty:
        return pd.DataFrame(columns=["date", "symbol", "avg_dollar_volume", "rank"])
    bars = daily_bars.copy()
    bars["date"] = pd.to_datetime(bars["date"], errors="coerce").dt.normalize()
    if bars["date"].dropna().empty:
        return pd.DataFrame(columns=["date", "symbol", "avg_dollar_volume", "rank"])
    rebalance_dates = (
        pd.Series(sorted(bars["date"].dropna().unique()))
        .dt.to_period("M")
        .drop_duplicates()
        .dt.to_timestamp(how="start")
        .tolist()
        if rebalance_frequency == "MS"
        else sorted(bars["date"].dropna().unique().tolist())
    )
    return build_time_varying_universe(
        listing_history=listing_history,
        daily_bars=bars,
        rebalance_dates=rebalance_dates,
        top_n=top_n,
    )


def rebuild_derived_references(reference_root: Path) -> list[Path]:
    """Rebuild normalized derived reference files from raw collected source files."""
    state_path = reference_root / ".derived_references_state.json"
    input_fingerprint = _derived_reference_input_fingerprint(reference_root)
    cached_outputs = _cached_derived_reference_outputs(state_path=state_path, input_fingerprint=input_fingerprint)
    if cached_outputs is not None:
        return cached_outputs

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
    else:
        listing_history = pd.DataFrame(columns=_listing_history_columns())

    symbol_changes = _read_optional_parquet(reference_root / "symbol_changes.parquet")
    if not symbol_changes.empty:
        output = reference_root / "ticker_changes.parquet"
        output.parent.mkdir(parents=True, exist_ok=True)
        ticker_changes = build_ticker_changes(symbol_changes)
        ticker_changes.to_parquet(output, index=False)
        outputs.append(output)
    else:
        ticker_changes = _read_optional_parquet(reference_root / "ticker_changes.parquet")

    sec_company_tickers = _read_optional_parquet(reference_root / "sec_company_tickers.parquet")
    if not listing_history.empty:
        output = reference_root / "security_master.parquet"
        output.parent.mkdir(parents=True, exist_ok=True)
        build_security_master(
            listing_history=listing_history,
            ticker_changes=ticker_changes,
            sec_company_tickers=sec_company_tickers,
        ).to_parquet(output, index=False)
        outputs.append(output)

    earnings_frames = [
        _read_optional_parquet(reference_root / "earnings_calendar.parquet"),
        _read_optional_parquet(reference_root / "earnings_calendar_fmp.parquet"),
        _read_optional_parquet(reference_root / "earnings_calendar_twelve_data.parquet"),
    ]
    if any(not frame.empty for frame in earnings_frames):
        output = reference_root / "earnings_calendar_pit.parquet"
        output.parent.mkdir(parents=True, exist_ok=True)
        build_earnings_calendar_pit(earnings_frames=earnings_frames).to_parquet(output, index=False)
        outputs.append(output)

    company_profiles = _read_optional_parquet(reference_root / "company_profiles.parquet")
    companyfacts = _read_optional_parquet(reference_root / "sec_companyfacts.parquet")
    sec_companyfacts_fundamentals = build_sec_companyfacts_fundamentals(
        companyfacts_index=companyfacts,
        sec_company_tickers=sec_company_tickers,
        reference_root=reference_root,
    )
    financial_statements = _read_optional_parquet(reference_root / "financial_statements_twelve_data.parquet")
    if any(not frame.empty for frame in (company_profiles, sec_companyfacts_fundamentals, financial_statements)):
        output = reference_root / "fundamentals_daily.parquet"
        output.parent.mkdir(parents=True, exist_ok=True)
        build_fundamentals_daily(
            company_profiles=company_profiles,
            companyfacts=sec_companyfacts_fundamentals,
            financial_statements=financial_statements,
        ).to_parquet(output, index=False)
        outputs.append(output)

    sec_filings = _read_optional_parquet(reference_root / "sec_filings.parquet")
    if not sec_filings.empty:
        output = reference_root / "sec_filing_index.parquet"
        output.parent.mkdir(parents=True, exist_ok=True)
        build_sec_filing_index(filing_index=sec_filings, sec_company_tickers=sec_company_tickers).to_parquet(output, index=False)
        outputs.append(output)

    macro_root = reference_root.parent / "raw" / "macros_fred"
    macro_vintages = build_macro_vintages(
        vintagedates=_read_optional_parquet(reference_root / "fred_vintagedates.parquet"),
        macro_root=macro_root,
    )
    if not macro_vintages.empty:
        output = reference_root / "macro_vintages.parquet"
        output.parent.mkdir(parents=True, exist_ok=True)
        macro_vintages.to_parquet(output, index=False)
        outputs.append(output)

    raw_bars_root = reference_root.parent / "raw" / "equities_bars"
    if not listing_history.empty and raw_bars_root.exists():
        daily_frames: list[pd.DataFrame] = []
        for path in sorted(raw_bars_root.glob("date=*/data.parquet")):
            try:
                daily_frames.append(pd.read_parquet(path, columns=["date", "symbol", "close", "volume"]))
            except Exception:
                continue
        if daily_frames:
            snapshots = build_universe_snapshots(
                listing_history=listing_history,
                daily_bars=pd.concat(daily_frames, ignore_index=True),
                top_n=500,
            )
            if not snapshots.empty:
                snapshots_root = reference_root / "universe_snapshots"
                snapshots_root.mkdir(parents=True, exist_ok=True)
                for snapshot_date, frame in snapshots.groupby("date", dropna=True):
                    path = snapshots_root / f"date={pd.Timestamp(snapshot_date).strftime('%Y-%m-%d')}" / "data.parquet"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    frame.sort_values(["rank", "symbol"]).to_parquet(path, index=False)
                    outputs.append(path)
    _write_derived_reference_state(
        state_path=state_path,
        input_fingerprint=input_fingerprint,
        outputs=outputs,
    )
    return outputs


def _derived_reference_input_fingerprint(reference_root: Path) -> str:
    """Return a cheap content fingerprint for derived-reference rebuild inputs."""
    candidates: list[Path] = []
    for name in (
        "listings.parquet",
        "alpaca_assets.parquet",
        "delistings.parquet",
        "universe.parquet",
        "company_profiles.parquet",
        "tiingo_tickers.parquet",
        "twelve_data_stocks.parquet",
        "symbol_changes.parquet",
        "ticker_changes.parquet",
        "sec_company_tickers.parquet",
        "earnings_calendar.parquet",
        "earnings_calendar_fmp.parquet",
        "earnings_calendar_twelve_data.parquet",
        "sec_companyfacts.parquet",
        "financial_statements_twelve_data.parquet",
        "sec_filings.parquet",
        "fred_vintagedates.parquet",
    ):
        candidates.append(reference_root / name)
    for pattern in (
        "../raw/macros_fred/series=*/data.parquet",
        "../raw/equities_bars/date=*/data.parquet",
    ):
        candidates.extend(sorted(reference_root.glob(pattern)))
    digest = hashlib.sha1()
    for path in sorted({candidate.resolve() for candidate in candidates if candidate.exists()}):
        stat = path.stat()
        digest.update(str(path).encode("utf-8"))
        digest.update(str(stat.st_size).encode("utf-8"))
        digest.update(str(stat.st_mtime_ns).encode("utf-8"))
    return digest.hexdigest()


def _cached_derived_reference_outputs(*, state_path: Path, input_fingerprint: str) -> list[Path] | None:
    """Return cached derived-reference outputs when the input fingerprint is unchanged."""
    if not state_path.exists():
        return None
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if payload.get("input_fingerprint") != input_fingerprint:
        return None
    outputs = [Path(str(value)) for value in payload.get("outputs", [])]
    if not all(path.exists() for path in outputs):
        return None
    return outputs


def _write_derived_reference_state(*, state_path: Path, input_fingerprint: str, outputs: list[Path]) -> None:
    """Persist the last successful derived-reference rebuild fingerprint."""
    state_path.write_text(
        json.dumps(
            {
                "input_fingerprint": input_fingerprint,
                "outputs": [str(path) for path in outputs],
                "updated_at": datetime.now(tz=UTC).isoformat(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


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


def _security_master_columns() -> list[str]:
    return [
        "issuer_key",
        "cik",
        "symbol",
        "primary_symbol",
        "name",
        "exchange",
        "asset_type",
        "ipo_date",
        "delist_date",
        "change_date",
        "status",
        "sources",
        "last_verified",
    ]


def _earnings_calendar_columns() -> list[str]:
    return ["symbol", "earnings_date", "fiscal_period", "source_count", "sources", "last_verified"]


def _fundamentals_daily_columns() -> list[str]:
    return ["symbol", "metric_date", "metric_name", "metric_value", "source", "last_verified"]


def _ticker_by_cik(sec_company_tickers: pd.DataFrame | None) -> dict[str, str]:
    if sec_company_tickers is None or sec_company_tickers.empty:
        return {}
    cik_col = (
        "cik_str"
        if "cik_str" in sec_company_tickers.columns
        else "cik"
        if "cik" in sec_company_tickers.columns
        else None
    )
    ticker_col = (
        "ticker"
        if "ticker" in sec_company_tickers.columns
        else "symbol"
        if "symbol" in sec_company_tickers.columns
        else None
    )
    if cik_col is None or ticker_col is None:
        return {}
    mapping: dict[str, str] = {}
    for item in sec_company_tickers[[cik_col, ticker_col]].dropna().to_dict("records"):
        cik = _normalize_cik(item.get(cik_col))
        symbol = str(item.get(ticker_col) or "").strip().upper()
        if cik and symbol:
            mapping[cik] = symbol
    return mapping


def _normalize_cik(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    numeric = pd.to_numeric(pd.Series([text]), errors="coerce").iloc[0]
    if pd.notna(numeric):
        return str(int(numeric)).zfill(10)
    return text.zfill(10)


def _resolve_companyfacts_path(item: dict[str, object], *, reference_root: Path | None) -> Path | None:
    relative = str(item.get("facts_relative_path") or "").strip()
    if relative and reference_root is not None:
        return reference_root / relative
    raw_path = str(item.get("facts_path") or "").strip()
    if not raw_path:
        return None
    path = Path(raw_path)
    if path.exists():
        return path
    if reference_root is not None:
        marker = "sec_companyfacts/"
        normalized = raw_path.replace("\\", "/")
        if marker in normalized:
            return reference_root / marker / normalized.split(marker, 1)[1]
    return path


def _sec_filing_index_columns() -> list[str]:
    return ["cik", "symbol", "form", "filing_date", "accepted_at", "accession_number", "source"]


def _macro_vintages_columns() -> list[str]:
    return ["series_id", "vintage_date", "source"]
