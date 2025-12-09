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

import yaml
from loguru import logger

from .db import Task, TaskKind, PartitionStatus


class FetchStatus(str, Enum):
    """Result status from a fetch operation."""
    SUCCESS = "success"         # Data fetched and written
    PARTIAL = "partial"         # Some days fetched, but fetch failed partway through
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
    rows_by_date: Optional[dict[date, int]] = None  # Per-day row counts for multi-day tasks
    partition_status: Optional[PartitionStatus] = None
    qc_code: Optional[str] = None
    error: Optional[str] = None
    vendor_used: Optional[str] = None
    fetch_params: Optional[dict] = None  # API params used (for selective re-fetch)
    failed_at_date: Optional[str] = None  # For PARTIAL: date where fetch failed
    original_end_date: Optional[str] = None  # For PARTIAL: original task end date


# ---------------------------------------------------------------------------
# Entitlements Config - Source of truth for vendor capabilities
# ---------------------------------------------------------------------------

def _load_entitlements() -> dict:
    """Load vendor entitlements from config file."""
    config_path = Path(__file__).parent.parent / "configs" / "entitlements.yml"
    if not config_path.exists():
        logger.warning(f"Entitlements config not found at {config_path}, using defaults")
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


_ENTITLEMENTS = _load_entitlements()

# Build DATASET_VENDORS from entitlements config
# Maps dataset -> list of vendors that can handle it
DATASET_VENDORS: dict[str, list[str]] = {}
for _vendor, _config in _ENTITLEMENTS.get("vendors", {}).items():
    for _dataset in _config.get("datasets", []):
        DATASET_VENDORS.setdefault(_dataset, []).append(_vendor)

# Vendor priority (lower = preferred)
# FMP removed - free tier has severe endpoint restrictions
VENDOR_PRIORITY = {
    "alpaca": 1,
    "massive": 2,
    "finnhub": 3,
    "fred": 1,
    "av": 2,
}

# Inverse mapping: vendor → datasets it can handle
VENDOR_DATASETS: dict[str, list[str]] = {}
for _vendor, _config in _ENTITLEMENTS.get("vendors", {}).items():
    VENDOR_DATASETS[_vendor] = list(_config.get("datasets", []))

# Excluded datasets per vendor (datasets they should NEVER pick up)
VENDOR_EXCLUDED: dict[str, set[str]] = {}
for _vendor, _config in _ENTITLEMENTS.get("vendors", {}).items():
    VENDOR_EXCLUDED[_vendor] = set(_config.get("excluded", []))


def get_datasets_for_vendor(vendor: str) -> list[str]:
    """Get list of datasets a vendor can handle."""
    return VENDOR_DATASETS.get(vendor, [])


def get_excluded_datasets(vendor: str) -> set[str]:
    """Get set of datasets a vendor is explicitly excluded from."""
    return VENDOR_EXCLUDED.get(vendor, set())


def is_vendor_entitled(vendor: str, dataset: str) -> bool:
    """Check if a vendor is entitled to handle a dataset.

    A vendor is entitled if:
    1. The dataset is in their datasets list, AND
    2. The dataset is NOT in their excluded list
    """
    can_handle = dataset in VENDOR_DATASETS.get(vendor, [])
    is_excluded = dataset in VENDOR_EXCLUDED.get(vendor, set())
    return can_handle and not is_excluded


def _ensure_date(val) -> date:
    """Convert val to date if it's a string, or return as-is if already a date."""
    if val is None:
        raise ValueError("Date value cannot be None")
    if isinstance(val, date):
        return val
    if isinstance(val, str):
        return date.fromisoformat(val)
    raise TypeError(f"Expected date or str, got {type(val).__name__}")


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
            # FMP free tier too limited - reject
            return FetchResult(status=FetchStatus.NOT_SUPPORTED, error="FMP free tier unsupported")
        else:
            return FetchResult(status=FetchStatus.NOT_SUPPORTED, error=f"Vendor {vendor} not supported for equities_eod")
    except ImportError as e:
        logger.warning(f"Connector not available for {vendor}: {e}")
        return FetchResult(status=FetchStatus.ERROR, error=str(e), vendor_used=vendor)


def _fetch_equities_minute(task: Task, vendor: str) -> FetchResult:
    """Fetch minute equities bars.

    For multi-day tasks, chunks into single-day fetches to:
    - Provide per-day progress visibility
    - Avoid loading years of data into memory
    - Enable graceful recovery if a fetch fails partway
    """
    from datetime import timedelta

    start_date = _ensure_date(task.start_date)
    end_date = _ensure_date(task.end_date)

    # For single-day tasks, fetch directly
    if start_date == end_date:
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

    # Multi-day task: chunk into single-day fetches
    logger.info(f"Chunking multi-day minute task: {task.symbol} from {start_date} to {end_date}")

    total_rows = 0
    rows_by_date: dict[date, int] = {}
    days_processed = 0
    days_total = (end_date - start_date).days + 1

    current_date = start_date
    while current_date <= end_date:
        # Create a single-day task copy
        single_day_task = Task(
            id=task.id,
            dataset=task.dataset,
            symbol=task.symbol,
            start_date=current_date.isoformat(),
            end_date=current_date.isoformat(),
            kind=task.kind,
            priority=task.priority,
            status=task.status,
            attempts=task.attempts,
            lease_owner=task.lease_owner,
            lease_expires_at=task.lease_expires_at,
            next_not_before=task.next_not_before,
            last_error=task.last_error,
            created_at=task.created_at,
            updated_at=task.updated_at,
        )

        try:
            if vendor == "alpaca":
                result = _fetch_alpaca_bars(single_day_task, timeframe="1Min")
            elif vendor == "massive":
                result = _fetch_massive_bars(single_day_task, timeframe="minute")
            else:
                return FetchResult(status=FetchStatus.NOT_SUPPORTED, error=f"Vendor {vendor} not supported for equities_minute")

            # Accumulate results
            if result.status == FetchStatus.SUCCESS:
                day_rows = result.rows
                total_rows += day_rows
                rows_by_date[current_date] = day_rows
            elif result.status == FetchStatus.EMPTY:
                # Weekend/holiday - record 0 rows
                rows_by_date[current_date] = 0
            elif result.status in (FetchStatus.RATE_LIMITED, FetchStatus.ERROR):
                # Return partial progress - worker will create follow-up task for remaining days
                logger.warning(f"Chunk fetch failed at {current_date}: {result.error}")
                if rows_by_date:
                    return FetchResult(
                        status=FetchStatus.PARTIAL,
                        rows=total_rows,
                        rows_by_date=rows_by_date,
                        qc_code="PARTIAL",
                        vendor_used=vendor,
                        error=f"Partial fetch, failed at {current_date}: {result.error}",
                        failed_at_date=current_date.isoformat(),  # Track where we failed
                        original_end_date=end_date.isoformat(),    # Track original end date
                    )
                return result
            else:
                # NOT_ENTITLED, NOT_SUPPORTED - propagate immediately
                return result

        except ImportError as e:
            logger.warning(f"Connector not available for {vendor}: {e}")
            return FetchResult(status=FetchStatus.ERROR, error=str(e), vendor_used=vendor)
        except Exception as e:
            logger.exception(f"Error fetching {current_date}: {e}")
            if rows_by_date:
                return FetchResult(
                    status=FetchStatus.PARTIAL,
                    rows=total_rows,
                    rows_by_date=rows_by_date,
                    qc_code="PARTIAL",
                    vendor_used=vendor,
                    error=f"Partial fetch, failed at {current_date}: {e}",
                    failed_at_date=current_date.isoformat(),
                    original_end_date=end_date.isoformat(),
                )
            return FetchResult(status=FetchStatus.ERROR, error=str(e), vendor_used=vendor)

        days_processed += 1
        if days_processed % 30 == 0:
            logger.info(f"Progress: {task.symbol} {days_processed}/{days_total} days, {total_rows} rows")

        current_date += timedelta(days=1)

    logger.info(f"Completed multi-day fetch: {task.symbol} {days_total} days, {total_rows} total rows")

    # Determine fetch_params based on vendor
    if vendor == "alpaca":
        fetch_params = {"feed": "sip", "timeframe": "1Min"}
    elif vendor == "massive":
        fetch_params = {"timespan": "minute"}
    else:
        fetch_params = {}

    return FetchResult(
        status=FetchStatus.SUCCESS,
        rows=total_rows,
        rows_by_date=rows_by_date,
        qc_code="OK",
        vendor_used=vendor,
        fetch_params=fetch_params,
    )


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
            # FMP free tier too limited - reject
            return FetchResult(status=FetchStatus.NOT_SUPPORTED, error="FMP free tier unsupported")
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

    # Track fetch params for lineage
    fetch_params = {"feed": "sip", "timeframe": timeframe}

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
                qc_code="NO_SESSION",
                vendor_used="alpaca",
                fetch_params=fetch_params,
            )

        # Compute per-day row counts for QC validation
        rows_by_date = _compute_rows_by_date(df)

        # Write to raw partition
        table_name = "equities_bars" if timeframe == "1Day" else "equities_bars_minute"
        raw_path = _get_raw_path("alpaca", table_name, task.start_date)

        # Use parent.parent (table dir) so partition_cols creates date=... without nesting
        connector.write_parquet(df, str(raw_path.parent.parent), partition_cols=["date"])

        return FetchResult(
            status=FetchStatus.SUCCESS,
            rows=len(df),
            rows_by_date=rows_by_date,
            qc_code="OK",
            vendor_used="alpaca",
            fetch_params=fetch_params,
        )

    except Exception as e:
        error_str = str(e).lower()
        if "429" in error_str or "rate" in error_str:
            return FetchResult(status=FetchStatus.RATE_LIMITED, error=str(e), vendor_used="alpaca", fetch_params=fetch_params)
        elif "401" in error_str or "403" in error_str or "not entitled" in error_str:
            return FetchResult(status=FetchStatus.NOT_ENTITLED, error=str(e), vendor_used="alpaca", fetch_params=fetch_params)
        else:
            return FetchResult(status=FetchStatus.ERROR, error=str(e), vendor_used="alpaca", fetch_params=fetch_params)


def _fetch_alpaca_options(task: Task) -> FetchResult:
    """Fetch options data from Alpaca (not supported - requires SDK)."""
    return FetchResult(
        status=FetchStatus.NOT_SUPPORTED,
        error="Alpaca options requires SDK (removed - REST only)",
        vendor_used="alpaca",
    )


def _fetch_alpaca_corp_actions(task: Task) -> FetchResult:
    """Fetch corporate actions from Alpaca (not supported - requires SDK)."""
    return FetchResult(
        status=FetchStatus.NOT_SUPPORTED,
        error="Alpaca corporate actions requires SDK (removed - REST only)",
        vendor_used="alpaca",
    )


# -----------------------------------------------------------------------------
# Massive (Polygon.io) fetchers
# -----------------------------------------------------------------------------

def _fetch_massive_bars(task: Task, timeframe: str) -> FetchResult:
    """Fetch bars from Massive (Polygon.io)."""
    from data_layer.connectors.massive_connector import MassiveConnector

    connector = MassiveConnector()

    # Track fetch params for lineage
    fetch_params = {"timespan": timeframe}

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
                qc_code="NO_SESSION",
                vendor_used="massive",
                fetch_params=fetch_params,
            )

        # Compute per-day row counts for QC validation
        rows_by_date = _compute_rows_by_date(df)

        table_name = "equities_bars" if timeframe == "day" else "equities_bars_minute"
        raw_path = _get_raw_path("massive", table_name, task.start_date)
        # Use parent.parent (table dir) so partition_cols creates date=... without nesting
        connector.write_parquet(df, str(raw_path.parent.parent), partition_cols=["date"])

        return FetchResult(
            status=FetchStatus.SUCCESS,
            rows=len(df),
            rows_by_date=rows_by_date,
            qc_code="OK",
            vendor_used="massive",
            fetch_params=fetch_params,
        )

    except Exception as e:
        result = _handle_exception(e, "massive")
        result.fetch_params = fetch_params
        return result


# -----------------------------------------------------------------------------
# Finnhub fetchers
# -----------------------------------------------------------------------------

def _fetch_finnhub_candles(task: Task) -> FetchResult:
    """Fetch daily candles from Finnhub."""
    from data_layer.connectors.finnhub_connector import FinnhubConnector

    connector = FinnhubConnector()

    # Track fetch params for lineage
    fetch_params = {"resolution": "D"}

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
                qc_code="NO_SESSION",
                vendor_used="finnhub",
                fetch_params=fetch_params,
            )

        # Compute per-day row counts for QC validation
        rows_by_date = _compute_rows_by_date(df)

        raw_path = _get_raw_path("finnhub", "equities_bars", task.start_date)
        # Use parent.parent (table dir) so partition_cols creates date=... without nesting
        connector.write_parquet(df, str(raw_path.parent.parent), partition_cols=["date"])

        return FetchResult(
            status=FetchStatus.SUCCESS,
            rows=len(df),
            rows_by_date=rows_by_date,
            qc_code="OK",
            vendor_used="finnhub",
            fetch_params=fetch_params,
        )

    except Exception as e:
        result = _handle_exception(e, "finnhub")
        result.fetch_params = fetch_params
        return result


def _fetch_finnhub_options(task: Task) -> FetchResult:
    """Fetch options chain from Finnhub."""
    from data_layer.connectors.finnhub_connector import FinnhubConnector

    connector = FinnhubConnector()

    # Track fetch params for lineage
    fetch_params = {"endpoint": "option-chain"}

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
                fetch_params=fetch_params,
            )

        raw_path = _get_raw_path("finnhub", "options_chains", task.start_date, underlier=task.symbol)
        # Use parent.parent (table dir) so partition_cols creates date=... without nesting
        connector.write_parquet(df, str(raw_path.parent.parent), partition_cols=["date"])

        return FetchResult(
            status=FetchStatus.SUCCESS,
            rows=len(df),
            partition_status=PartitionStatus.GREEN,
            qc_code="OK",
            vendor_used="finnhub",
            fetch_params=fetch_params,
        )

    except Exception as e:
        result = _handle_exception(e, "finnhub")
        result.fetch_params = fetch_params
        return result


def _fetch_finnhub_fundamentals(task: Task) -> FetchResult:
    """Fetch company fundamentals from Finnhub."""
    from data_layer.connectors.finnhub_connector import FinnhubConnector

    connector = FinnhubConnector()

    # Track fetch params for lineage
    fetch_params = {"endpoint": "profile2"}

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
                fetch_params=fetch_params,
            )

        # TODO: Implement fundamentals storage - for now return EMPTY to avoid phantom GREEN partitions
        # Data was fetched but not persisted, so don't claim SUCCESS
        logger.warning(f"Finnhub fundamentals fetch for {task.symbol}: storage not implemented, returning EMPTY")
        return FetchResult(
            status=FetchStatus.EMPTY,
            rows=0,
            qc_code="STORAGE_NOT_IMPLEMENTED",
            vendor_used="finnhub",
            fetch_params=fetch_params,
        )

    except Exception as e:
        result = _handle_exception(e, "finnhub")
        result.fetch_params = fetch_params
        return result


# -----------------------------------------------------------------------------
# FRED fetchers
# -----------------------------------------------------------------------------

def _fetch_fred_treasury(task: Task) -> FetchResult:
    """Fetch treasury curve from FRED."""
    from data_layer.connectors.fred_connector import FREDConnector

    connector = FREDConnector()

    # Track fetch params for lineage (FRED is simple - no significant config params)
    fetch_params = {"endpoint": "treasury_curve"}

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
                fetch_params=fetch_params,
            )

        raw_path = _get_raw_path("fred", "macro_treasury", task.start_date)
        # Use parent.parent (table dir) so partition_cols creates date=... without nesting
        connector.write_parquet(df, str(raw_path.parent.parent), partition_cols=["date"])

        return FetchResult(
            status=FetchStatus.SUCCESS,
            rows=len(df),
            partition_status=PartitionStatus.GREEN,
            qc_code="OK",
            vendor_used="fred",
            fetch_params=fetch_params,
        )

    except Exception as e:
        result = _handle_exception(e, "fred")
        result.fetch_params = fetch_params
        return result


# -----------------------------------------------------------------------------
# Alpha Vantage fetchers
# -----------------------------------------------------------------------------

def _fetch_av_corp_actions(task: Task) -> FetchResult:
    """Fetch corporate actions from Alpha Vantage."""
    from data_layer.connectors.alpha_vantage_connector import AlphaVantageConnector

    connector = AlphaVantageConnector()

    # Track fetch params for lineage
    fetch_params = {"function": "CORPORATE_ACTIONS"}

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
                fetch_params=fetch_params,
            )

        raw_path = _get_raw_path("av", "corp_actions", task.start_date)
        # Use parent.parent (table dir) so partition_cols creates date=... without nesting
        connector.write_parquet(df, str(raw_path.parent.parent), partition_cols=["date"])

        return FetchResult(
            status=FetchStatus.SUCCESS,
            rows=len(df),
            partition_status=PartitionStatus.GREEN,
            qc_code="OK",
            vendor_used="av",
            fetch_params=fetch_params,
        )

    except Exception as e:
        result = _handle_exception(e, "av")
        result.fetch_params = fetch_params
        return result


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
        # Use parent.parent (table dir) so partition_cols creates date=... without nesting
        connector.write_parquet(df, str(raw_path.parent.parent), partition_cols=["date"])

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
        # Use parent.parent (table dir) so partition_cols creates date=... without nesting
        connector.write_parquet(df, str(raw_path.parent.parent), partition_cols=["date"])

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

def _compute_rows_by_date(df) -> dict[date, int]:
    """Compute per-day row counts from a DataFrame.

    Expects the DataFrame to have a 'date' column with date objects.
    Returns dict mapping date -> row_count for that day.
    """
    if df is None or df.empty:
        return {}

    # Handle different date column formats
    if "date" in df.columns:
        date_col = df["date"]
    elif "timestamp" in df.columns:
        date_col = df["timestamp"].dt.date
    else:
        # Try to find any date-like column
        for col in df.columns:
            if "date" in col.lower() or "time" in col.lower():
                try:
                    import pandas as pd
                    if hasattr(df[col], "dt"):
                        date_col = df[col].dt.date
                        break
                    elif df[col].dtype == object:
                        date_col = pd.to_datetime(df[col]).dt.date
                        break
                except Exception:
                    continue
        else:
            # No date column found
            return {}

    # Group by date and count
    counts = date_col.value_counts().to_dict()

    # Ensure keys are date objects
    result = {}
    for dt, count in counts.items():
        if isinstance(dt, date):
            result[dt] = int(count)
        elif hasattr(dt, "date"):
            result[dt.date()] = int(count)
        else:
            # Try to convert string
            try:
                result[date.fromisoformat(str(dt)[:10])] = int(count)
            except (ValueError, TypeError):
                pass

    return result


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
