"""
Fundamentals curator for equity financial statements.

Fetches income statements, balance sheets, and cash flow statements from FMP,
transforms to standardized schema, computes derived ratios, and writes to
curated Parquet partitions.

PIT (Point-in-Time) discipline:
- Uses filing_date (not period_end) as the knowledge date
- Features are only available AFTER the filing_date
- Prevents look-ahead bias in backtests
"""

from __future__ import annotations

import os
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from loguru import logger


def _safe_float(x) -> Optional[float]:
    """Convert to float, returning None for invalid values."""
    if x is None or pd.isna(x):
        return None
    try:
        return float(x)
    except (ValueError, TypeError):
        return None


def _compute_ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    """Safely compute ratio, returning None if invalid."""
    if numerator is None or denominator is None:
        return None
    if denominator == 0:
        return None
    return numerator / denominator


def _normalize_fmp_income(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Normalize FMP income statement DataFrame to our schema."""
    if df.empty:
        return pd.DataFrame()

    rows = []
    for _, row in df.iterrows():
        # Parse dates
        filing_date = pd.to_datetime(row.get("fillingDate") or row.get("filingDate")).date() \
            if row.get("fillingDate") or row.get("filingDate") else None
        period_end = pd.to_datetime(row.get("date")).date() if row.get("date") else None

        if not filing_date or not period_end:
            continue

        # Determine period type
        period_type = "annual" if row.get("period", "").upper() == "FY" else "quarter"
        fiscal_year = row.get("calendarYear")
        fiscal_quarter = None
        period_str = str(row.get("period", ""))
        if period_str.startswith("Q"):
            try:
                fiscal_quarter = int(period_str[1])
            except ValueError:
                pass

        rows.append({
            "symbol": symbol,
            "filing_date": filing_date,
            "period_end": period_end,
            "period_type": period_type,
            "fiscal_year": int(fiscal_year) if fiscal_year else None,
            "fiscal_quarter": fiscal_quarter,
            "revenue": _safe_float(row.get("revenue")),
            "cost_of_revenue": _safe_float(row.get("costOfRevenue")),
            "gross_profit": _safe_float(row.get("grossProfit")),
            "operating_income": _safe_float(row.get("operatingIncome")),
            "net_income": _safe_float(row.get("netIncome")),
            "ebitda": _safe_float(row.get("ebitda")),
            "eps_basic": _safe_float(row.get("eps")),
            "eps_diluted": _safe_float(row.get("epsdiluted")),
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _normalize_fmp_balance(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Normalize FMP balance sheet DataFrame to our schema."""
    if df.empty:
        return pd.DataFrame()

    rows = []
    for _, row in df.iterrows():
        filing_date = pd.to_datetime(row.get("fillingDate") or row.get("filingDate")).date() \
            if row.get("fillingDate") or row.get("filingDate") else None
        period_end = pd.to_datetime(row.get("date")).date() if row.get("date") else None

        if not filing_date or not period_end:
            continue

        rows.append({
            "filing_date": filing_date,
            "period_end": period_end,
            "total_assets": _safe_float(row.get("totalAssets")),
            "total_liabilities": _safe_float(row.get("totalLiabilities")),
            "total_equity": _safe_float(row.get("totalStockholdersEquity") or row.get("totalEquity")),
            "cash_and_equivalents": _safe_float(row.get("cashAndCashEquivalents")),
            "total_debt": _safe_float(row.get("totalDebt")),
            "working_capital": _safe_float(row.get("netWorkingCapital") or
                                           (_safe_float(row.get("totalCurrentAssets") or 0) -
                                            _safe_float(row.get("totalCurrentLiabilities") or 0))),
            "current_assets": _safe_float(row.get("totalCurrentAssets")),
            "current_liabilities": _safe_float(row.get("totalCurrentLiabilities")),
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _normalize_fmp_cashflow(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Normalize FMP cash flow statement DataFrame to our schema."""
    if df.empty:
        return pd.DataFrame()

    rows = []
    for _, row in df.iterrows():
        filing_date = pd.to_datetime(row.get("fillingDate") or row.get("filingDate")).date() \
            if row.get("fillingDate") or row.get("filingDate") else None
        period_end = pd.to_datetime(row.get("date")).date() if row.get("date") else None

        if not filing_date or not period_end:
            continue

        rows.append({
            "filing_date": filing_date,
            "period_end": period_end,
            "operating_cash_flow": _safe_float(row.get("operatingCashFlow") or row.get("netCashProvidedByOperatingActivities")),
            "capital_expenditure": _safe_float(row.get("capitalExpenditure")),
            "free_cash_flow": _safe_float(row.get("freeCashFlow")),
            "dividends_paid": _safe_float(row.get("dividendsPaid")),
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _compute_derived_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived financial ratios."""
    if df.empty:
        return df

    df = df.copy()

    # Margin ratios
    df["gross_margin"] = df.apply(
        lambda r: _compute_ratio(r.get("gross_profit"), r.get("revenue")), axis=1
    )
    df["operating_margin"] = df.apply(
        lambda r: _compute_ratio(r.get("operating_income"), r.get("revenue")), axis=1
    )
    df["net_margin"] = df.apply(
        lambda r: _compute_ratio(r.get("net_income"), r.get("revenue")), axis=1
    )

    # Return ratios (annualized for quarterly)
    def annualize_factor(row):
        return 4.0 if row.get("period_type") == "quarter" else 1.0

    df["roe"] = df.apply(
        lambda r: _compute_ratio(
            r.get("net_income") * annualize_factor(r) if r.get("net_income") else None,
            r.get("total_equity")
        ), axis=1
    )
    df["roa"] = df.apply(
        lambda r: _compute_ratio(
            r.get("net_income") * annualize_factor(r) if r.get("net_income") else None,
            r.get("total_assets")
        ), axis=1
    )

    # Leverage ratios
    df["debt_to_equity"] = df.apply(
        lambda r: _compute_ratio(r.get("total_debt"), r.get("total_equity")), axis=1
    )
    df["current_ratio"] = df.apply(
        lambda r: _compute_ratio(r.get("current_assets"), r.get("current_liabilities")), axis=1
    )

    return df


def curate_fundamentals(
    symbols: Iterable[str],
    period: str = "quarter",
    limit: int = 20,
    output_dir: str = "data_layer/curated/fundamentals",
) -> int:
    """
    Curate fundamental data for symbols.

    Fetches from FMP, normalizes, computes ratios, and writes partitioned Parquet.

    Args:
        symbols: Symbols to fetch fundamentals for
        period: 'annual' or 'quarter'
        limit: Max filings per symbol
        output_dir: Output directory for curated data

    Returns:
        Number of rows written
    """
    from data_layer.connectors.fmp_connector import FMPConnector, ConnectorError

    api_key = os.getenv("FMP_API_KEY")
    if not api_key:
        logger.warning("FMP_API_KEY not set; skipping fundamentals curation")
        return 0

    try:
        connector = FMPConnector(api_key=api_key)
    except ConnectorError as e:
        logger.error(f"Failed to initialize FMP connector: {e}")
        return 0

    all_data: List[pd.DataFrame] = []
    symbols_list = list(symbols)

    for sym in symbols_list:
        logger.info(f"Fetching fundamentals for {sym}")

        # Fetch all three statement types
        try:
            income_df = connector.fetch_statements(sym, kind="income", period=period, limit=limit)
            balance_df = connector.fetch_statements(sym, kind="balance", period=period, limit=limit)
            cashflow_df = connector.fetch_statements(sym, kind="cashflow", period=period, limit=limit)
        except Exception as e:
            logger.warning(f"Failed to fetch statements for {sym}: {e}")
            continue

        # Normalize each
        income = _normalize_fmp_income(income_df, sym)
        balance = _normalize_fmp_balance(balance_df, sym)
        cashflow = _normalize_fmp_cashflow(cashflow_df, sym)

        if income.empty:
            logger.debug(f"No income data for {sym}")
            continue

        # Merge on filing_date + period_end
        merged = income
        if not balance.empty:
            merged = merged.merge(
                balance,
                on=["filing_date", "period_end"],
                how="left",
                suffixes=("", "_bal")
            )
        if not cashflow.empty:
            merged = merged.merge(
                cashflow,
                on=["filing_date", "period_end"],
                how="left",
                suffixes=("", "_cf")
            )

        # Compute derived ratios
        merged = _compute_derived_ratios(merged)

        # Add metadata
        merged["source_name"] = "fmp"
        merged["ingested_at"] = pd.Timestamp.now(tz="UTC")
        merged["source_uri"] = f"fmp://statements/{sym}/{period}"

        all_data.append(merged)

    if not all_data:
        logger.warning("No fundamentals data fetched")
        return 0

    combined = pd.concat(all_data, ignore_index=True)

    # Select only schema columns (drop intermediate columns)
    schema_cols = [
        "symbol", "filing_date", "period_end", "period_type", "fiscal_year", "fiscal_quarter",
        "revenue", "cost_of_revenue", "gross_profit", "operating_income", "net_income",
        "ebitda", "eps_basic", "eps_diluted",
        "total_assets", "total_liabilities", "total_equity", "cash_and_equivalents",
        "total_debt", "working_capital",
        "operating_cash_flow", "capital_expenditure", "free_cash_flow", "dividends_paid",
        "gross_margin", "operating_margin", "net_margin", "roe", "roa",
        "debt_to_equity", "current_ratio",
        "source_name", "ingested_at", "source_uri"
    ]

    # Keep only columns that exist
    available = [c for c in schema_cols if c in combined.columns]
    combined = combined[available]

    # Write partitioned by symbol
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for sym, grp in combined.groupby("symbol"):
        sym_dir = out_path / f"symbol={sym}"
        sym_dir.mkdir(parents=True, exist_ok=True)
        grp.to_parquet(sym_dir / "data.parquet", index=False)

    logger.info(f"Wrote {len(combined)} fundamentals rows for {len(symbols_list)} symbols to {output_dir}")
    return len(combined)


def load_fundamentals_panel(
    symbols: Iterable[str],
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    base_dir: str = "data_layer/curated/fundamentals",
) -> pd.DataFrame:
    """
    Load curated fundamentals panel for symbols.

    Args:
        symbols: Symbols to load
        start_date: Filter filings on or after this date
        end_date: Filter filings on or before this date
        base_dir: Base directory for curated fundamentals

    Returns:
        Combined DataFrame with all fundamentals
    """
    base_path = Path(base_dir)
    frames: List[pd.DataFrame] = []

    for sym in symbols:
        sym_path = base_path / f"symbol={sym}" / "data.parquet"
        if not sym_path.exists():
            logger.debug(f"No fundamentals for {sym}")
            continue

        try:
            df = pd.read_parquet(sym_path)
            if not df.empty:
                frames.append(df)
        except Exception as e:
            logger.warning(f"Failed to read fundamentals for {sym}: {e}")
            continue

    if not frames:
        return pd.DataFrame()

    panel = pd.concat(frames, ignore_index=True)

    # Convert dates if needed
    if "filing_date" in panel.columns:
        panel["filing_date"] = pd.to_datetime(panel["filing_date"]).dt.date

    # Apply date filters based on filing_date (PIT discipline)
    if start_date:
        panel = panel[panel["filing_date"] >= start_date]
    if end_date:
        panel = panel[panel["filing_date"] <= end_date]

    return panel.sort_values(["symbol", "filing_date"]).reset_index(drop=True)


def get_pit_fundamentals(
    symbols: Iterable[str],
    as_of_date: date,
    base_dir: str = "data_layer/curated/fundamentals",
) -> pd.DataFrame:
    """
    Get point-in-time fundamentals as known on a specific date.

    Returns the most recent filing available for each symbol as of as_of_date.
    This ensures no look-ahead bias.

    Args:
        symbols: Symbols to get fundamentals for
        as_of_date: Date for which to get known fundamentals
        base_dir: Base directory for curated fundamentals

    Returns:
        DataFrame with one row per symbol (most recent filing)
    """
    panel = load_fundamentals_panel(symbols, end_date=as_of_date, base_dir=base_dir)

    if panel.empty:
        return panel

    # Get most recent filing per symbol
    panel = panel.sort_values(["symbol", "filing_date"])
    latest = panel.groupby("symbol").last().reset_index()

    return latest


if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description="Curate fundamentals data")
    parser.add_argument("--symbols", nargs="+", default=["AAPL", "MSFT", "GOOGL"])
    parser.add_argument("--period", choices=["annual", "quarter"], default="quarter")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output", default="data_layer/curated/fundamentals")

    args = parser.parse_args()

    n_rows = curate_fundamentals(
        symbols=args.symbols,
        period=args.period,
        limit=args.limit,
        output_dir=args.output,
    )

    print(f"[OK] Curated {n_rows} fundamentals rows")
