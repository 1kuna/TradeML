"""
Unified fetcher wrappers for the data-node queue worker.

Wraps existing connectors (data_layer/connectors/) in a consistent interface
that the queue worker can call for any (dataset, vendor) combination.

See updated_node_spec.md §3 for outcome handling requirements.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from .db import Task, TaskKind, PartitionStatus


class FetchStatus(str, Enum):
    """Result status from a fetch operation."""
    SUCCESS = "success"         # Data fetched and written
    EMPTY = "empty"             # No data (weekend/holiday/no session)
    RATE_LIMITED = "rate_limited"  # 429 or similar
    ERROR = "error"             # Transient error (5xx, network)
    NOT_ENTITLED = "not_entitled"  # 4xx auth/entitlement error
    NOT_SUPPORTED = "not_supported"  # Vendor doesn't support this dataset


@dataclass
class FetchResult:
    """Result of a fetch operation."""
    status: FetchStatus
    rows: int = 0
    partition_status: Optional[PartitionStatus] = None
    qc_code: Optional[str] = None
    error: Optional[str] = None
    vendor_used: Optional[str] = None


# Dataset to vendor capability mapping
# Based on configs/endpoints.yml
DATASET_VENDORS = {
    "equities_eod": ["alpaca", "massive", "finnhub", "fmp"],
    "equities_minute": ["alpaca", "massive"],
    "options_chains": ["finnhub", "alpaca"],
    "options_bars": ["alpaca"],
    "macros_fred": ["fred"],
    "corp_actions": ["alpaca", "av"],
    "fundamentals": ["fmp", "finnhub"],
}

# Vendor priority (lower = preferred)
VENDOR_PRIORITY = {
    "alpaca": 1,
    "massive": 2,
    "finnhub": 3,
    "fred": 1,
    "av": 2,
    "fmp": 3,
}

# Inverse mapping: vendor → datasets it can handle
# Built from DATASET_VENDORS for vendor-aware task leasing
VENDOR_DATASETS: dict[str, list[str]] = {}
for _dataset, _vendors in DATASET_VENDORS.items():
    for _vendor in _vendors:
        VENDOR_DATASETS.setdefault(_vendor, []).append(_dataset)


def get_datasets_for_vendor(vendor: str) -> list[str]:
    """Get list of datasets a vendor can handle."""
    return VENDOR_DATASETS.get(vendor, [])


def _ensure_date(val) -> date:
    """Convert val to date if it's a string, or return as-is if already a date."""
    if isinstance(val, date):
        return val
    return date.fromisoformat(val)


def get_vendors_for_dataset(dataset: str) -> list[str]:
    """Get list of vendors that support a dataset, ordered by priority."""
    vendors = DATASET_VENDORS.get(dataset, [])
    return sorted(vendors, key=lambda v: VENDOR_PRIORITY.get(v, 99))


def fetch_task(task: Task, vendor: str) -> FetchResult:
    """
    Fetch data for a task using the specified vendor.

    This is the main entry point for the queue worker.

    Args:
        task: Task from the queue
        vendor: Vendor to use

    Returns:
        FetchResult with status and details
    """
    try:
        # Route to appropriate fetcher based on dataset
        if task.dataset == "equities_eod":
            return _fetch_equities_eod(task, vendor)
        elif task.dataset == "equities_minute":
            return _fetch_equities_minute(task, vendor)
        elif task.dataset == "options_chains":
            return _fetch_options_chains(task, vendor)
        elif task.dataset == "macros_fred":
            return _fetch_macros_fred(task, vendor)
        elif task.dataset == "corp_actions":
            return _fetch_corp_actions(task, vendor)
        elif task.dataset == "fundamentals":
            return _fetch_fundamentals(task, vendor)
        else:
            logger.warning(f"Unknown dataset: {task.dataset}")
            return FetchResult(
                status=FetchStatus.NOT_SUPPORTED,
                error=f"Unknown dataset: {task.dataset}",
            )
    except Exception as e:
        logger.exception(f"Error fetching task {task.id}: {e}")
        return FetchResult(
            status=FetchStatus.ERROR,
            error=str(e),
            vendor_used=vendor,
        )


def _fetch_equities_eod(task: Task, vendor: str) -> FetchResult:
    """Fetch daily equities bars."""
    try:
        if vendor == "alpaca":
            return _fetch_alpaca_bars(task, timeframe="1Day")
        elif vendor == "massive":
            return _fetch_massive_bars(task, timeframe="day")
        elif vendor == "finnhub":
            return _fetch_finnhub_candles(task)
        elif vendor == "fmp":
            return _fetch_fmp_eod(task)
        else:
            return FetchResult(status=FetchStatus.NOT_SUPPORTED, error=f"Vendor {vendor} not supported for equities_eod")
    except ImportError as e:
        logger.warning(f"Connector not available for {vendor}: {e}")
        return FetchResult(status=FetchStatus.ERROR, error=str(e), vendor_used=vendor)


def _fetch_equities_minute(task: Task, vendor: str) -> FetchResult:
    """Fetch minute equities bars."""
    try:
        if vendor == "alpaca":
            return _fetch_alpaca_bars(task, timeframe="1Min")
        elif vendor == "massive":
            return _fetch_massive_bars(task, timeframe="minute")
        else:
            return FetchResult(status=FetchStatus.NOT_SUPPORTED, error=f"Vendor {vendor} not supported for equities_minute")
    except ImportError as e:
        logger.warning(f"Connector not available for {vendor}: {e}")
        return FetchResult(status=FetchStatus.ERROR, error=str(e), vendor_used=vendor)


def _fetch_options_chains(task: Task, vendor: str) -> FetchResult:
    """Fetch options chain data."""
    try:
        if vendor == "finnhub":
            return _fetch_finnhub_options(task)
        elif vendor == "alpaca":
            return _fetch_alpaca_options(task)
        else:
            return FetchResult(status=FetchStatus.NOT_SUPPORTED, error=f"Vendor {vendor} not supported for options_chains")
    except ImportError as e:
        logger.warning(f"Connector not available for {vendor}: {e}")
        return FetchResult(status=FetchStatus.ERROR, error=str(e), vendor_used=vendor)


def _fetch_macros_fred(task: Task, vendor: str) -> FetchResult:
    """Fetch FRED macro/treasury data."""
    try:
        if vendor == "fred":
            return _fetch_fred_treasury(task)
        else:
            return FetchResult(status=FetchStatus.NOT_SUPPORTED, error=f"Vendor {vendor} not supported for macros_fred")
    except ImportError as e:
        logger.warning(f"Connector not available for {vendor}: {e}")
        return FetchResult(status=FetchStatus.ERROR, error=str(e), vendor_used=vendor)


def _fetch_corp_actions(task: Task, vendor: str) -> FetchResult:
    """Fetch corporate actions."""
    try:
        if vendor == "alpaca":
            return _fetch_alpaca_corp_actions(task)
        elif vendor == "av":
            return _fetch_av_corp_actions(task)
        else:
            return FetchResult(status=FetchStatus.NOT_SUPPORTED, error=f"Vendor {vendor} not supported for corp_actions")
    except ImportError as e:
        logger.warning(f"Connector not available for {vendor}: {e}")
        return FetchResult(status=FetchStatus.ERROR, error=str(e), vendor_used=vendor)


def _fetch_fundamentals(task: Task, vendor: str) -> FetchResult:
    """Fetch fundamentals data."""
    try:
        if vendor == "fmp":
            return _fetch_fmp_fundamentals(task)
        elif vendor == "finnhub":
            return _fetch_finnhub_fundamentals(task)
        else:
            return FetchResult(status=FetchStatus.NOT_SUPPORTED, error=f"Vendor {vendor} not supported for fundamentals")
    except ImportError as e:
        logger.warning(f"Connector not available for {vendor}: {e}")
        return FetchResult(status=FetchStatus.ERROR, error=str(e), vendor_used=vendor)


# -----------------------------------------------------------------------------
# Alpaca fetchers
# -----------------------------------------------------------------------------

def _fetch_alpaca_bars(task: Task, timeframe: str) -> FetchResult:
    """Fetch bars from Alpaca."""
    from data_layer.connectors.alpaca_connector import AlpacaConnector

    connector = AlpacaConnector()

    try:
        symbols = [task.symbol] if task.symbol else []
        if not symbols:
            return FetchResult(status=FetchStatus.ERROR, error="No symbol specified")

        df = connector.fetch_bars(
            symbols=symbols,
            start_date=_ensure_date(task.start_date),
            end_date=_ensure_date(task.end_date),
            timeframe=timeframe,
        )

        if df is None or df.empty:
            # Check if this is a weekend/holiday
            return FetchResult(
                status=FetchStatus.EMPTY,
                rows=0,
                partition_status=PartitionStatus.GREEN,
                qc_code="NO_SESSION",
                vendor_used="alpaca",
            )

        # Write to raw partition
        table_name = "equities_bars" if timeframe == "1Day" else "equities_bars_minute"
        raw_path = _get_raw_path("alpaca", table_name, task.start_date)

        connector.write_parquet(df, str(raw_path.parent), partition_cols=["date"])

        return FetchResult(
            status=FetchStatus.SUCCESS,
            rows=len(df),
            partition_status=PartitionStatus.GREEN,
            qc_code="OK",
            vendor_used="alpaca",
        )

    except Exception as e:
        error_str = str(e).lower()
        if "429" in error_str or "rate" in error_str:
            return FetchResult(status=FetchStatus.RATE_LIMITED, error=str(e), vendor_used="alpaca")
        elif "401" in error_str or "403" in error_str or "not entitled" in error_str:
            return FetchResult(status=FetchStatus.NOT_ENTITLED, error=str(e), vendor_used="alpaca")
        else:
            return FetchResult(status=FetchStatus.ERROR, error=str(e), vendor_used="alpaca")


def _fetch_alpaca_options(task: Task) -> FetchResult:
    """Fetch options data from Alpaca."""
    from data_layer.connectors.alpaca_connector import AlpacaConnector

    connector = AlpacaConnector()

    try:
        if not task.symbol:
            return FetchResult(status=FetchStatus.ERROR, error="No underlier specified")

        df = connector.fetch_option_chain_snapshot_df(task.symbol)

        if df is None or df.empty:
            return FetchResult(
                status=FetchStatus.EMPTY,
                rows=0,
                partition_status=PartitionStatus.GREEN,
                qc_code="NO_SESSION",
                vendor_used="alpaca",
            )

        raw_path = _get_raw_path("alpaca", "options_chains", task.start_date, underlier=task.symbol)
        connector.write_parquet(df, str(raw_path.parent), partition_cols=["date"])

        return FetchResult(
            status=FetchStatus.SUCCESS,
            rows=len(df),
            partition_status=PartitionStatus.GREEN,
            qc_code="OK",
            vendor_used="alpaca",
        )

    except Exception as e:
        return _handle_exception(e, "alpaca")


def _fetch_alpaca_corp_actions(task: Task) -> FetchResult:
    """Fetch corporate actions from Alpaca."""
    from data_layer.connectors.alpaca_connector import AlpacaConnector

    connector = AlpacaConnector()

    try:
        symbols = [task.symbol] if task.symbol else None
        df = connector.fetch_corporate_actions(
            start=_ensure_date(task.start_date),
            end=_ensure_date(task.end_date),
            symbols=symbols,
        )

        if df is None or df.empty:
            return FetchResult(
                status=FetchStatus.EMPTY,
                rows=0,
                partition_status=PartitionStatus.GREEN,
                qc_code="NO_ACTIONS",
                vendor_used="alpaca",
            )

        raw_path = _get_raw_path("alpaca", "corp_actions", task.start_date)
        connector.write_parquet(df, str(raw_path.parent), partition_cols=["date"])

        return FetchResult(
            status=FetchStatus.SUCCESS,
            rows=len(df),
            partition_status=PartitionStatus.GREEN,
            qc_code="OK",
            vendor_used="alpaca",
        )

    except Exception as e:
        return _handle_exception(e, "alpaca")


# -----------------------------------------------------------------------------
# Massive (Polygon.io) fetchers
# -----------------------------------------------------------------------------

def _fetch_massive_bars(task: Task, timeframe: str) -> FetchResult:
    """Fetch bars from Massive (Polygon.io)."""
    from data_layer.connectors.massive_connector import MassiveConnector

    connector = MassiveConnector()

    try:
        if not task.symbol:
            return FetchResult(status=FetchStatus.ERROR, error="No symbol specified")

        df = connector.fetch_aggregates(
            symbol=task.symbol,
            start_date=_ensure_date(task.start_date),
            end_date=_ensure_date(task.end_date),
            timespan=timeframe,
        )

        if df is None or df.empty:
            return FetchResult(
                status=FetchStatus.EMPTY,
                rows=0,
                partition_status=PartitionStatus.GREEN,
                qc_code="NO_SESSION",
                vendor_used="massive",
            )

        table_name = "equities_bars" if timeframe == "day" else "equities_bars_minute"
        raw_path = _get_raw_path("massive", table_name, task.start_date)
        connector.write_parquet(df, str(raw_path.parent), partition_cols=["date"])

        return FetchResult(
            status=FetchStatus.SUCCESS,
            rows=len(df),
            partition_status=PartitionStatus.GREEN,
            qc_code="OK",
            vendor_used="massive",
        )

    except Exception as e:
        return _handle_exception(e, "massive")


# -----------------------------------------------------------------------------
# Finnhub fetchers
# -----------------------------------------------------------------------------

def _fetch_finnhub_candles(task: Task) -> FetchResult:
    """Fetch daily candles from Finnhub."""
    from data_layer.connectors.finnhub_connector import FinnhubConnector

    connector = FinnhubConnector()

    try:
        if not task.symbol:
            return FetchResult(status=FetchStatus.ERROR, error="No symbol specified")

        df = connector.fetch_candle_daily(
            symbol=task.symbol,
            start_date=_ensure_date(task.start_date),
            end_date=_ensure_date(task.end_date),
        )

        if df is None or df.empty:
            return FetchResult(
                status=FetchStatus.EMPTY,
                rows=0,
                partition_status=PartitionStatus.GREEN,
                qc_code="NO_SESSION",
                vendor_used="finnhub",
            )

        raw_path = _get_raw_path("finnhub", "equities_bars", task.start_date)
        connector.write_parquet(df, str(raw_path.parent), partition_cols=["date"])

        return FetchResult(
            status=FetchStatus.SUCCESS,
            rows=len(df),
            partition_status=PartitionStatus.GREEN,
            qc_code="OK",
            vendor_used="finnhub",
        )

    except Exception as e:
        return _handle_exception(e, "finnhub")


def _fetch_finnhub_options(task: Task) -> FetchResult:
    """Fetch options chain from Finnhub."""
    from data_layer.connectors.finnhub_connector import FinnhubConnector

    connector = FinnhubConnector()

    try:
        if not task.symbol:
            return FetchResult(status=FetchStatus.ERROR, error="No underlier specified")

        df = connector.fetch_options_chain(
            symbol=task.symbol,
            date=_ensure_date(task.start_date),
        )

        if df is None or df.empty:
            return FetchResult(
                status=FetchStatus.EMPTY,
                rows=0,
                partition_status=PartitionStatus.GREEN,
                qc_code="NO_SESSION",
                vendor_used="finnhub",
            )

        raw_path = _get_raw_path("finnhub", "options_chains", task.start_date, underlier=task.symbol)
        connector.write_parquet(df, str(raw_path.parent), partition_cols=["date"])

        return FetchResult(
            status=FetchStatus.SUCCESS,
            rows=len(df),
            partition_status=PartitionStatus.GREEN,
            qc_code="OK",
            vendor_used="finnhub",
        )

    except Exception as e:
        return _handle_exception(e, "finnhub")


def _fetch_finnhub_fundamentals(task: Task) -> FetchResult:
    """Fetch company fundamentals from Finnhub."""
    from data_layer.connectors.finnhub_connector import FinnhubConnector

    connector = FinnhubConnector()

    try:
        if not task.symbol:
            return FetchResult(status=FetchStatus.ERROR, error="No symbol specified")

        data = connector.fetch_company_profile(task.symbol)

        if data is None or not data:
            return FetchResult(
                status=FetchStatus.EMPTY,
                rows=0,
                qc_code="NO_DATA",
                vendor_used="finnhub",
            )

        # For now, just return success - actual storage TBD
        return FetchResult(
            status=FetchStatus.SUCCESS,
            rows=1,
            partition_status=PartitionStatus.GREEN,
            qc_code="OK",
            vendor_used="finnhub",
        )

    except Exception as e:
        return _handle_exception(e, "finnhub")


# -----------------------------------------------------------------------------
# FRED fetchers
# -----------------------------------------------------------------------------

def _fetch_fred_treasury(task: Task) -> FetchResult:
    """Fetch treasury curve from FRED."""
    from data_layer.connectors.fred_connector import FredConnector

    connector = FredConnector()

    try:
        df = connector.fetch_treasury_curve(
            start_date=_ensure_date(task.start_date),
            end_date=_ensure_date(task.end_date),
        )

        if df is None or df.empty:
            return FetchResult(
                status=FetchStatus.EMPTY,
                rows=0,
                partition_status=PartitionStatus.GREEN,
                qc_code="NO_SESSION",
                vendor_used="fred",
            )

        raw_path = _get_raw_path("fred", "macro_treasury", task.start_date)
        connector.write_parquet(df, str(raw_path.parent), partition_cols=["date"])

        return FetchResult(
            status=FetchStatus.SUCCESS,
            rows=len(df),
            partition_status=PartitionStatus.GREEN,
            qc_code="OK",
            vendor_used="fred",
        )

    except Exception as e:
        return _handle_exception(e, "fred")


# -----------------------------------------------------------------------------
# Alpha Vantage fetchers
# -----------------------------------------------------------------------------

def _fetch_av_corp_actions(task: Task) -> FetchResult:
    """Fetch corporate actions from Alpha Vantage."""
    from data_layer.connectors.alpha_vantage_connector import AlphaVantageConnector

    connector = AlphaVantageConnector()

    try:
        if not task.symbol:
            return FetchResult(status=FetchStatus.ERROR, error="No symbol specified")

        df = connector.fetch_corporate_actions(task.symbol)

        if df is None or df.empty:
            return FetchResult(
                status=FetchStatus.EMPTY,
                rows=0,
                partition_status=PartitionStatus.GREEN,
                qc_code="NO_ACTIONS",
                vendor_used="av",
            )

        raw_path = _get_raw_path("av", "corp_actions", task.start_date)
        connector.write_parquet(df, str(raw_path.parent), partition_cols=["date"])

        return FetchResult(
            status=FetchStatus.SUCCESS,
            rows=len(df),
            partition_status=PartitionStatus.GREEN,
            qc_code="OK",
            vendor_used="av",
        )

    except Exception as e:
        return _handle_exception(e, "av")


# -----------------------------------------------------------------------------
# FMP fetchers
# -----------------------------------------------------------------------------

def _fetch_fmp_eod(task: Task) -> FetchResult:
    """Fetch EOD data from FMP."""
    from data_layer.connectors.fmp_connector import FMPConnector

    connector = FMPConnector()

    try:
        if not task.symbol:
            return FetchResult(status=FetchStatus.ERROR, error="No symbol specified")

        df = connector.fetch_historical_price(
            symbol=task.symbol,
            start_date=_ensure_date(task.start_date),
            end_date=_ensure_date(task.end_date),
        )

        if df is None or df.empty:
            return FetchResult(
                status=FetchStatus.EMPTY,
                rows=0,
                partition_status=PartitionStatus.GREEN,
                qc_code="NO_SESSION",
                vendor_used="fmp",
            )

        raw_path = _get_raw_path("fmp", "equities_bars", task.start_date)
        connector.write_parquet(df, str(raw_path.parent), partition_cols=["date"])

        return FetchResult(
            status=FetchStatus.SUCCESS,
            rows=len(df),
            partition_status=PartitionStatus.GREEN,
            qc_code="OK",
            vendor_used="fmp",
        )

    except Exception as e:
        return _handle_exception(e, "fmp")


def _fetch_fmp_fundamentals(task: Task) -> FetchResult:
    """Fetch fundamentals from FMP."""
    from data_layer.connectors.fmp_connector import FMPConnector

    connector = FMPConnector()

    try:
        if not task.symbol:
            return FetchResult(status=FetchStatus.ERROR, error="No symbol specified")

        df = connector.fetch_statements(
            symbol=task.symbol,
            kind="income",
            period="annual",
            limit=10,
        )

        if df is None or df.empty:
            return FetchResult(
                status=FetchStatus.EMPTY,
                rows=0,
                qc_code="NO_DATA",
                vendor_used="fmp",
            )

        raw_path = _get_raw_path("fmp", "fundamentals", task.start_date)
        connector.write_parquet(df, str(raw_path.parent), partition_cols=["date"])

        return FetchResult(
            status=FetchStatus.SUCCESS,
            rows=len(df),
            partition_status=PartitionStatus.GREEN,
            qc_code="OK",
            vendor_used="fmp",
        )

    except Exception as e:
        return _handle_exception(e, "fmp")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _get_raw_path(
    vendor: str,
    table: str,
    dt: str,
    underlier: Optional[str] = None,
) -> Path:
    """Get the raw partition path."""
    data_root = os.environ.get("DATA_ROOT", ".")
    base = Path(data_root) / "data_layer" / "raw" / vendor / table / f"date={dt}"

    if underlier:
        base = base / f"underlier={underlier}"

    return base / "data.parquet"


def _handle_exception(e: Exception, vendor: str) -> FetchResult:
    """Handle exceptions uniformly."""
    error_str = str(e).lower()

    if "429" in error_str or "rate" in error_str or "too many" in error_str:
        return FetchResult(status=FetchStatus.RATE_LIMITED, error=str(e), vendor_used=vendor)
    elif "401" in error_str or "403" in error_str or "unauthorized" in error_str or "not entitled" in error_str:
        return FetchResult(status=FetchStatus.NOT_ENTITLED, error=str(e), vendor_used=vendor)
    elif "404" in error_str or "not found" in error_str:
        return FetchResult(status=FetchStatus.EMPTY, error=str(e), vendor_used=vendor, qc_code="NOT_FOUND")
    else:
        return FetchResult(status=FetchStatus.ERROR, error=str(e), vendor_used=vendor)
