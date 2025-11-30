"""
Parquet schema definitions for equities, options, and reference data.

All schemas include metadata fields for lineage tracking:
- ingested_at: when data was ingested into our system
- source_uri: vendor API endpoint or file path
- source_name: vendor name (e.g., 'alpaca', 'iex', 'finnhub')
"""

import pyarrow as pa
from typing import Dict
from enum import Enum


class DataType(str, Enum):
    """Data types for partitioning and organization."""
    EQUITY_TICKS = "equities_ticks"
    EQUITY_BARS = "equities_bars"
    EQUITY_BARS_ADJ = "equities_bars_adj"
    OPTIONS_NBBO = "options_nbbo"
    OPTIONS_TRADES = "options_trades"
    OPTIONS_IV = "options_iv"
    OPTIONS_SURFACE = "options_surface"
    CORP_ACTIONS = "corp_actions"
    DELISTINGS = "delistings"
    INDEX_MEMBERSHIP = "index_membership"
    TICK_SIZE_REGIME = "tick_size_regime"
    MACRO = "macro"
    CALENDARS = "calendars"
    FUNDAMENTALS = "fundamentals"


# ===== Equities Schemas =====

EQUITY_TICKS_SCHEMA = pa.schema([
    # Core tick data
    ("ts_ns", pa.int64(), False),           # Nanosecond timestamp (UTC)
    ("symbol", pa.string(), False),
    ("price", pa.float64(), False),
    ("size", pa.int32(), False),
    ("side", pa.string(), True),            # 'buy', 'sell', or null
    ("venue", pa.string(), True),           # Exchange/venue code
    ("seq", pa.int64(), True),              # Sequence number from venue

    # Metadata for lineage
    ("ingested_at", pa.timestamp("us"), False),
    ("source_name", pa.string(), False),
    ("source_uri", pa.string(), False),
])


EQUITY_BARS_SCHEMA = pa.schema([
    # Time and identifier
    ("date", pa.date32(), False),
    ("symbol", pa.string(), False),
    ("session_id", pa.string(), False),     # YYYYMMDD format

    # OHLCV
    ("open", pa.float64(), False),
    ("high", pa.float64(), False),
    ("low", pa.float64(), False),
    ("close", pa.float64(), False),
    ("vwap", pa.float64(), True),
    ("volume", pa.int64(), False),

    # Microstructure metrics
    ("nbbo_spread", pa.float64(), True),    # Average spread during bar
    ("trades", pa.int32(), True),           # Number of trades in bar
    ("imbalance", pa.float64(), True),      # Buy volume - sell volume

    # Metadata
    ("ingested_at", pa.timestamp("us"), False),
    ("source_name", pa.string(), False),
    ("source_uri", pa.string(), False),
])


EQUITY_BARS_ADJ_SCHEMA = pa.schema([
    # Adjusted bars (after corporate actions applied)
    ("date", pa.date32(), False),
    ("symbol", pa.string(), False),
    ("session_id", pa.string(), False),

    # Adjusted OHLCV
    ("open_adj", pa.float64(), False),
    ("high_adj", pa.float64(), False),
    ("low_adj", pa.float64(), False),
    ("close_adj", pa.float64(), False),
    ("vwap_adj", pa.float64(), True),
    ("volume_adj", pa.int64(), False),      # Split-adjusted volume

    # Original unadjusted close for reference
    ("close_raw", pa.float64(), False),

    # Adjustment metadata
    ("adjustment_factor", pa.float64(), False),  # Cumulative adjustment ratio
    ("last_adjustment_date", pa.date32(), True), # Most recent corp action

    # Metadata
    ("ingested_at", pa.timestamp("us"), False),
    ("source_name", pa.string(), False),
    ("source_uri", pa.string(), False),
    ("transform_id", pa.string(), False),   # Hash of transformation logic
])


# ===== Options Schemas =====

OPTIONS_NBBO_SCHEMA = pa.schema([
    # Timestamp and identifiers
    ("ts_ns", pa.int64(), False),           # Nanosecond timestamp (UTC)
    ("underlier", pa.string(), False),      # Underlying symbol
    ("expiry", pa.date32(), False),
    ("strike", pa.float64(), False),
    ("cp_flag", pa.string(), False),        # 'C' or 'P'

    # NBBO quote
    ("bid", pa.float64(), True),
    ("ask", pa.float64(), True),
    ("bid_size", pa.int32(), True),
    ("ask_size", pa.int32(), True),
    ("nbbo_mid", pa.float64(), True),       # (bid + ask) / 2

    # Consolidated exchange bitmap (optional)
    ("exchs_bitmap", pa.int32(), True),

    # Metadata
    ("ingested_at", pa.timestamp("us"), False),
    ("source_name", pa.string(), False),
    ("source_uri", pa.string(), False),
])


OPTIONS_TRADES_SCHEMA = pa.schema([
    # Trade data
    ("ts_ns", pa.int64(), False),
    ("underlier", pa.string(), False),
    ("expiry", pa.date32(), False),
    ("strike", pa.float64(), False),
    ("cp_flag", pa.string(), False),

    ("price", pa.float64(), False),
    ("size", pa.int32(), False),
    ("venue", pa.string(), True),

    # Metadata
    ("ingested_at", pa.timestamp("us"), False),
    ("source_name", pa.string(), False),
    ("source_uri", pa.string(), False),
])


OPTIONS_IV_SCHEMA = pa.schema([
    # Implied volatility computed from NBBO mids
    ("date", pa.date32(), False),
    ("underlier", pa.string(), False),
    ("expiry", pa.date32(), False),
    ("strike", pa.float64(), False),
    ("cp_flag", pa.string(), False),

    # IV and Greeks
    ("iv", pa.float64(), True),             # Implied volatility (annualized)
    ("delta", pa.float64(), True),
    ("gamma", pa.float64(), True),
    ("theta", pa.float64(), True),
    ("vega", pa.float64(), True),
    ("rho", pa.float64(), True),

    # Inputs used for computation
    ("underlying_price", pa.float64(), False),
    ("risk_free_rate", pa.float64(), False),
    ("dividend_yield", pa.float64(), True),
    ("time_to_expiry", pa.float64(), False),  # Years
    ("nbbo_mid", pa.float64(), False),
    ("is_itm", pa.bool_(), False),

    # QC flags
    ("iv_valid", pa.bool_(), False),        # IV solved successfully
    ("is_crossed", pa.bool_(), False),      # Bid > ask (invalid quote)

    # Metadata
    ("ingested_at", pa.timestamp("us"), False),
    ("source_name", pa.string(), False),
    ("source_uri", pa.string(), False),
    ("transform_id", pa.string(), False),
])


OPTIONS_SURFACE_SCHEMA = pa.schema([
    # SVI/SSVI surface parameters per expiry
    ("date", pa.date32(), False),
    ("underlier", pa.string(), False),
    ("expiry", pa.date32(), False),

    # SVI parameters (per-expiry slice)
    # Total variance w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))
    ("svi_a", pa.float64(), True),
    ("svi_b", pa.float64(), True),
    ("svi_rho", pa.float64(), True),
    ("svi_m", pa.float64(), True),
    ("svi_sigma", pa.float64(), True),

    # Fit quality metrics
    ("fit_rmse", pa.float64(), True),
    ("num_options_fitted", pa.int32(), False),

    # No-arbitrage flags
    ("butterfly_arb", pa.bool_(), False),
    ("vertical_arb", pa.bool_(), False),
    ("calendar_arb", pa.bool_(), False),

    # Derived surface metrics
    ("atm_iv", pa.float64(), True),         # At-the-money IV
    ("skew_25d", pa.float64(), True),       # 25-delta risk reversal
    ("slope", pa.float64(), True),          # Term structure slope

    # Metadata
    ("ingested_at", pa.timestamp("us"), False),
    ("source_name", pa.string(), False),
    ("source_uri", pa.string(), False),
    ("transform_id", pa.string(), False),
])


# ===== Reference Data Schemas =====

CORP_ACTIONS_SCHEMA = pa.schema([
    ("symbol", pa.string(), False),
    ("event_type", pa.string(), False),     # 'split', 'dividend', 'merger', 'spinoff'

    # Key dates
    ("ex_date", pa.date32(), False),        # Ex-dividend/split date
    ("record_date", pa.date32(), True),
    ("pay_date", pa.date32(), True),

    # Event details
    ("ratio", pa.float64(), True),          # Split ratio (e.g., 2.0 for 2-for-1)
    ("amount", pa.float64(), True),         # Dividend amount per share

    # Metadata
    ("source_name", pa.string(), False),
    ("ingested_at", pa.timestamp("us"), False),
    ("source_uri", pa.string(), False),
])


DELISTINGS_SCHEMA = pa.schema([
    ("symbol", pa.string(), False),
    ("delist_date", pa.date32(), False),
    ("reason", pa.string(), True),          # 'merger', 'bankruptcy', 'voluntary', etc.

    # Metadata
    ("source_name", pa.string(), False),
    ("ingested_at", pa.timestamp("us"), False),
    ("source_uri", pa.string(), False),
])


INDEX_MEMBERSHIP_SCHEMA = pa.schema([
    # Index constituent tracking for PIT universe filtering
    ("date", pa.date32(), False),           # Effective date of the event
    ("index_name", pa.string(), False),     # e.g., "SP500", "R1000", "R2000"
    ("symbol", pa.string(), False),         # Constituent symbol
    ("action", pa.string(), False),         # "ADD" or "REMOVE"

    # Metadata
    ("source_name", pa.string(), False),
    ("ingested_at", pa.timestamp("us"), False),
    ("source_uri", pa.string(), False),
])


TICK_SIZE_REGIME_SCHEMA = pa.schema([
    # Tick size regime tracking for microstructure features
    ("effective_date", pa.date32(), False), # When regime starts
    ("symbol", pa.string(), True),          # NULL for market-wide rules
    ("tick_size", pa.float64(), False),     # Minimum tick increment (e.g., 0.01)
    ("regime_name", pa.string(), False),    # e.g., "REG_NMS", "TICK_PILOT", "STANDARD"
    ("notes", pa.string(), True),           # Additional context

    # Metadata
    ("source_name", pa.string(), False),
    ("ingested_at", pa.timestamp("us"), False),
    ("source_uri", pa.string(), False),
])


MACRO_SCHEMA = pa.schema([
    # Generic macro/rates time series
    ("series_id", pa.string(), False),      # e.g., 'DGS10' for 10Y Treasury
    ("date", pa.date32(), False),
    ("value", pa.float64(), True),

    # For ALFRED vintages (real-time vs revised data)
    ("vintage_date", pa.date32(), True),    # Date this value was published

    # Metadata
    ("source_name", pa.string(), False),
    ("ingested_at", pa.timestamp("us"), False),
    ("source_uri", pa.string(), False),
])


# ===== Fundamentals Schema =====

FUNDAMENTALS_SCHEMA = pa.schema([
    # Identifier and period
    ("symbol", pa.string(), False),
    ("filing_date", pa.date32(), False),        # When the filing was released
    ("period_end", pa.date32(), False),         # End of reporting period
    ("period_type", pa.string(), False),        # 'annual' or 'quarter'
    ("fiscal_year", pa.int32(), False),
    ("fiscal_quarter", pa.int32(), True),       # null for annual

    # Income statement metrics
    ("revenue", pa.float64(), True),
    ("cost_of_revenue", pa.float64(), True),
    ("gross_profit", pa.float64(), True),
    ("operating_income", pa.float64(), True),
    ("net_income", pa.float64(), True),
    ("ebitda", pa.float64(), True),
    ("eps_basic", pa.float64(), True),
    ("eps_diluted", pa.float64(), True),

    # Balance sheet metrics
    ("total_assets", pa.float64(), True),
    ("total_liabilities", pa.float64(), True),
    ("total_equity", pa.float64(), True),
    ("cash_and_equivalents", pa.float64(), True),
    ("total_debt", pa.float64(), True),
    ("working_capital", pa.float64(), True),

    # Cash flow metrics
    ("operating_cash_flow", pa.float64(), True),
    ("capital_expenditure", pa.float64(), True),
    ("free_cash_flow", pa.float64(), True),
    ("dividends_paid", pa.float64(), True),

    # Derived ratios (computed at curation)
    ("gross_margin", pa.float64(), True),       # gross_profit / revenue
    ("operating_margin", pa.float64(), True),   # operating_income / revenue
    ("net_margin", pa.float64(), True),         # net_income / revenue
    ("roe", pa.float64(), True),                # net_income / equity (annualized)
    ("roa", pa.float64(), True),                # net_income / assets (annualized)
    ("debt_to_equity", pa.float64(), True),     # total_debt / equity
    ("current_ratio", pa.float64(), True),      # current_assets / current_liabilities

    # Metadata
    ("source_name", pa.string(), False),
    ("ingested_at", pa.timestamp("us"), False),
    ("source_uri", pa.string(), False),
])


# ===== Schema Registry =====

SCHEMAS: Dict[DataType, pa.Schema] = {
    DataType.EQUITY_TICKS: EQUITY_TICKS_SCHEMA,
    DataType.EQUITY_BARS: EQUITY_BARS_SCHEMA,
    DataType.EQUITY_BARS_ADJ: EQUITY_BARS_ADJ_SCHEMA,
    DataType.OPTIONS_NBBO: OPTIONS_NBBO_SCHEMA,
    DataType.OPTIONS_TRADES: OPTIONS_TRADES_SCHEMA,
    DataType.OPTIONS_IV: OPTIONS_IV_SCHEMA,
    DataType.OPTIONS_SURFACE: OPTIONS_SURFACE_SCHEMA,
    DataType.CORP_ACTIONS: CORP_ACTIONS_SCHEMA,
    DataType.DELISTINGS: DELISTINGS_SCHEMA,
    DataType.INDEX_MEMBERSHIP: INDEX_MEMBERSHIP_SCHEMA,
    DataType.TICK_SIZE_REGIME: TICK_SIZE_REGIME_SCHEMA,
    DataType.MACRO: MACRO_SCHEMA,
    DataType.FUNDAMENTALS: FUNDAMENTALS_SCHEMA,
}


def get_schema(data_type: DataType) -> pa.Schema:
    """
    Get PyArrow schema for a data type.

    Args:
        data_type: DataType enum value

    Returns:
        PyArrow schema

    Raises:
        KeyError: if data_type not found
    """
    return SCHEMAS[data_type]


def validate_dataframe(df, data_type: DataType) -> bool:
    """
    Validate pandas DataFrame against schema.

    Args:
        df: pandas DataFrame to validate
        data_type: Expected data type

    Returns:
        True if valid

    Raises:
        ValueError: if validation fails with details
    """
    import pandas as pd

    schema = get_schema(data_type)
    expected_cols = set(schema.names)
    actual_cols = set(df.columns)

    missing = expected_cols - actual_cols
    extra = actual_cols - expected_cols

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if extra:
        raise ValueError(f"Unexpected columns: {extra}")

    # TODO: Add dtype validation if needed
    return True


if __name__ == "__main__":
    # Schema validation test
    print("Registered schemas:")
    for dt, schema in SCHEMAS.items():
        print(f"\n{dt.value}:")
        print(f"  Fields: {len(schema.names)}")
        print(f"  Columns: {', '.join(schema.names[:5])}...")
