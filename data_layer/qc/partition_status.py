"""
Partition Status Ledger Management.

Provides atomic persistence and querying of GREEN/AMBER/RED completeness tracking
per (source, table_name, symbol, dt) partition.

SSOT v2 Section 2.3: Completeness & GREEN/AMBER/RED
- GREEN: partition exists and passes QC
- AMBER: partition exists but QC is marginal
- RED: partition missing
"""

from __future__ import annotations

import os
import io
import shutil
import tempfile
import uuid
from datetime import date, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger


class PartitionStatus(str, Enum):
    """Partition completeness status."""
    GREEN = "GREEN"
    AMBER = "AMBER"
    RED = "RED"


# Canonical schema for partition_status ledger
PARTITION_STATUS_SCHEMA = {
    "source_name": "str",
    "table_name": "str",
    "symbol": "str",  # nullable for non-symbol tables
    "dt": "datetime64[ns]",
    "partition_key": "str",  # nullable alternative key
    "status": "str",  # GREEN, AMBER, RED
    "qc_score": "float64",
    "row_count": "int64",
    "expected_rows": "int64",
    "qc_code": "str",  # OK, SHORT_SESSION, MISSING_ROWS, etc.
    "first_observed_at": "datetime64[ns]",
    "last_observed_at": "datetime64[ns]",
    "notes": "str",
}


def _get_ledger_path() -> Path:
    """Get path to partition_status.parquet ledger."""
    return Path("data_layer/qc/partition_status.parquet")


def _get_ledger_dir() -> Path:
    """Get directory for partition_status ledger."""
    return Path("data_layer/qc")


def _use_db_backend() -> bool:
    """Check if Postgres should be used as the authoritative backend."""
    return os.getenv("PARTITION_STATUS_BACKEND", "").lower() == "postgres"


def _connect_db():
    import psycopg2

    return psycopg2.connect(
        host=os.getenv("PGHOST") or os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("PGPORT") or os.getenv("POSTGRES_PORT", "5432")),
        user=os.getenv("PGUSER") or os.getenv("POSTGRES_USER", "trademl"),
        password=os.getenv("PGPASSWORD") or os.getenv("POSTGRES_PASSWORD", "trademl"),
        dbname=os.getenv("PGDATABASE") or os.getenv("POSTGRES_DB", "trademl"),
    )


def load_partition_status() -> pd.DataFrame:
    """
    Load the partition status ledger from disk.

    Returns:
        DataFrame with partition status or empty DataFrame with correct schema
        if ledger doesn't exist.
    """
    # Prefer Postgres if explicitly enabled
    if _use_db_backend():
        try:
            conn = _connect_db()
            query = """
                SELECT
                    COALESCE(source_name, source) AS source_name,
                    table_name,
                    symbol,
                    dt,
                    partition_key,
                    status,
                    qc_score,
                    row_count,
                    expected_rows,
                    qc_code,
                    first_observed_at,
                    last_observed_at,
                    notes
                FROM partition_status
            """
            df_db = pd.read_sql_query(query, conn)
            conn.close()
            if not df_db.empty:
                df_db["dt"] = pd.to_datetime(df_db["dt"])
                logger.debug(f"Loaded partition_status ledger from DB with {len(df_db)} rows")
                return df_db
        except ImportError:
            logger.debug("psycopg2 not installed; falling back to parquet ledger")
        except Exception as e:
            logger.warning(f"Failed to load partition_status from DB: {e}")

    path = _get_ledger_path()

    if path.exists():
        try:
            df = pd.read_parquet(path)
            logger.debug(f"Loaded partition_status ledger with {len(df)} rows")
            return df
        except Exception as e:
            logger.warning(f"Failed to load partition_status ledger: {e}")

    # Return empty DataFrame with correct schema
    return _create_empty_ledger()


def _create_empty_ledger() -> pd.DataFrame:
    """Create empty DataFrame with correct schema."""
    df = pd.DataFrame({
        "source_name": pd.Series(dtype="str"),
        "table_name": pd.Series(dtype="str"),
        "symbol": pd.Series(dtype="str"),
        "dt": pd.Series(dtype="datetime64[ns]"),
        "partition_key": pd.Series(dtype="str"),
        "status": pd.Series(dtype="str"),
        "qc_score": pd.Series(dtype="float64"),
        "row_count": pd.Series(dtype="int64"),
        "expected_rows": pd.Series(dtype="int64"),
        "qc_code": pd.Series(dtype="str"),
        "first_observed_at": pd.Series(dtype="datetime64[ns]"),
        "last_observed_at": pd.Series(dtype="datetime64[ns]"),
        "notes": pd.Series(dtype="str"),
    })
    return df


def save_partition_status(df: pd.DataFrame) -> None:
    """
    Save partition status ledger with atomic write (.tmp → move).

    Uses tempfile for atomic operation - writes to .tmp file first,
    then atomically renames to final location.

    Args:
        df: DataFrame with partition status to save
    """
    ledger_dir = _get_ledger_dir()
    ledger_path = _get_ledger_path()

    # Ensure directory exists
    ledger_dir.mkdir(parents=True, exist_ok=True)

    # Generate unique temp filename in same directory (for atomic rename)
    tmp_name = f".partition_status_{uuid.uuid4().hex[:8]}.tmp"
    tmp_path = ledger_dir / tmp_name

    try:
        # Write to temp file
        df.to_parquet(tmp_path, index=False)

        # Atomic rename (same filesystem)
        shutil.move(str(tmp_path), str(ledger_path))

        logger.info(f"Saved partition_status ledger ({len(df)} rows) to {ledger_path}")

    except Exception as e:
        # Clean up temp file if it exists
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(f"Failed to save partition_status ledger: {e}") from e

    # Optional DB mirror as authoritative store
    if _use_db_backend():
        try:
            import psycopg2

            conn = _connect_db()
            with conn, conn.cursor() as cur:
                for row in df.to_dict("records"):
                    cur.execute(
                        """
                        INSERT INTO partition_status (
                            source_name, table_name, symbol, dt, partition_key,
                            status, qc_score, row_count, expected_rows,
                            qc_code, first_observed_at, last_observed_at, notes
                        ) VALUES (
                            %(source_name)s, %(table_name)s, %(symbol)s, %(dt)s, %(partition_key)s,
                            %(status)s, %(qc_score)s, %(row_count)s, %(expected_rows)s,
                            %(qc_code)s, %(first_observed_at)s, %(last_observed_at)s, %(notes)s
                        )
                        ON CONFLICT (source_name, table_name, symbol, dt)
                        DO UPDATE SET
                            partition_key = EXCLUDED.partition_key,
                            status = EXCLUDED.status,
                            qc_score = EXCLUDED.qc_score,
                            row_count = EXCLUDED.row_count,
                            expected_rows = EXCLUDED.expected_rows,
                            qc_code = EXCLUDED.qc_code,
                            last_observed_at = EXCLUDED.last_observed_at,
                            notes = EXCLUDED.notes
                        """,
                        row,
                    )
            conn.close()
            logger.info("Saved partition_status ledger to Postgres backend")
        except ImportError:
            logger.debug("psycopg2 not installed; skipped DB mirror for partition_status")
        except Exception as e:
            logger.warning(f"Failed to mirror partition_status to DB: {e}")


def update_partition_status(new_rows: List[dict]) -> pd.DataFrame:
    """
    Merge new status rows with existing ledger using upsert logic.

    For each (source_name, table_name, symbol, dt) key:
    - If exists: update status, qc_score, row_count, last_observed_at, notes
    - If new: insert with first_observed_at = now

    Args:
        new_rows: List of dicts with partition status updates

    Returns:
        Updated DataFrame (also persisted to disk)
    """
    if not new_rows:
        return load_partition_status()

    # Load existing
    existing = load_partition_status()

    # Convert new rows to DataFrame
    now = datetime.utcnow()
    for row in new_rows:
        # Ensure required fields
        if "first_observed_at" not in row or pd.isna(row.get("first_observed_at")):
            row["first_observed_at"] = now
        if "last_observed_at" not in row or pd.isna(row.get("last_observed_at")):
            row["last_observed_at"] = now
        if "qc_code" not in row:
            row["qc_code"] = _derive_qc_code(row)

    new_df = pd.DataFrame(new_rows)

    # Define key columns
    key_cols = ["source_name", "table_name", "symbol", "dt"]

    # Normalize column names (audit.py uses 'source' and 'last_checked')
    if "source" in new_df.columns and "source_name" not in new_df.columns:
        new_df["source_name"] = new_df["source"]
    if "last_checked" in new_df.columns and "last_observed_at" not in new_df.columns:
        new_df["last_observed_at"] = new_df["last_checked"]
    if "rows" in new_df.columns and "row_count" not in new_df.columns:
        new_df["row_count"] = new_df["rows"]

    # Ensure dt is datetime
    if "dt" in new_df.columns:
        new_df["dt"] = pd.to_datetime(new_df["dt"])

    if existing.empty:
        # Just use new rows
        new_df["first_observed_at"] = now
        result = new_df
    else:
        # Ensure dt is datetime in existing
        if "dt" in existing.columns:
            existing["dt"] = pd.to_datetime(existing["dt"])

        # Merge: prefer new values, keep first_observed_at from existing
        # Create key for matching
        def make_key(df):
            return df["source_name"].astype(str) + "|" + df["table_name"].astype(str) + "|" + df["symbol"].fillna("").astype(str) + "|" + df["dt"].astype(str)

        existing["_key"] = make_key(existing)
        new_df["_key"] = make_key(new_df)

        # Find existing keys
        existing_keys = set(existing["_key"])

        # Split new rows into updates and inserts
        updates_mask = new_df["_key"].isin(existing_keys)
        inserts = new_df[~updates_mask].copy()
        updates = new_df[updates_mask].copy()

        # For inserts, set first_observed_at
        inserts["first_observed_at"] = now

        # For updates, get first_observed_at from existing
        if not updates.empty:
            existing_first = existing[["_key", "first_observed_at"]].drop_duplicates("_key")
            updates = updates.merge(
                existing_first.rename(columns={"first_observed_at": "_existing_first"}),
                on="_key",
                how="left"
            )
            updates["first_observed_at"] = updates["_existing_first"].fillna(now)
            updates = updates.drop(columns=["_existing_first"])

        # Remove updated rows from existing
        remaining = existing[~existing["_key"].isin(updates["_key"])]

        # Combine: remaining existing + updates + inserts
        result = pd.concat([remaining, updates, inserts], ignore_index=True)
        result = result.drop(columns=["_key"])

    # Ensure all schema columns exist
    for col, dtype in PARTITION_STATUS_SCHEMA.items():
        if col not in result.columns:
            if dtype == "str":
                result[col] = None
            elif dtype == "float64":
                result[col] = 0.0
            elif dtype == "int64":
                result[col] = 0
            else:
                result[col] = pd.NaT

    # Select and order columns
    result = result[[c for c in PARTITION_STATUS_SCHEMA.keys() if c in result.columns]]

    # Save atomically
    save_partition_status(result)

    return result


def _derive_qc_code(row: dict) -> str:
    """Derive QC code from status and notes."""
    status = row.get("status", "RED")
    notes = row.get("notes", "") or ""

    if status == "GREEN":
        return "OK"
    elif status == "AMBER":
        if "halfday" in notes.lower():
            return "SHORT_SESSION"
        elif "rows_low" in notes.lower():
            return "MISSING_ROWS"
        else:
            return "MARGINAL"
    else:  # RED
        if "missing" in notes.lower():
            return "MISSING"
        else:
            return "MISSING"


def get_status(
    source_name: str,
    table_name: str,
    symbol: Optional[str],
    dt: date
) -> Optional[PartitionStatus]:
    """
    Query status for a specific partition.

    Args:
        source_name: Data source (e.g., 'alpaca', 'finnhub')
        table_name: Table name (e.g., 'equities_eod', 'options_chains')
        symbol: Symbol (None for non-symbol tables)
        dt: Partition date

    Returns:
        PartitionStatus enum or None if not found
    """
    df = load_partition_status()

    if df.empty:
        return None

    mask = (
        (df["source_name"] == source_name) &
        (df["table_name"] == table_name) &
        (pd.to_datetime(df["dt"]).dt.date == dt)
    )

    if symbol is not None:
        mask &= (df["symbol"] == symbol)
    else:
        mask &= df["symbol"].isna()

    matches = df[mask]

    if matches.empty:
        return None

    status_str = matches.iloc[0]["status"]
    return PartitionStatus(status_str)


def get_green_coverage(
    table_name: str,
    start_date: date,
    end_date: date,
    source_name: Optional[str] = None,
    universe: Optional[List[str]] = None
) -> Tuple[float, Dict[str, int]]:
    """
    Compute GREEN coverage fraction for a table over a date window.

    Args:
        table_name: Table to check (e.g., 'equities_eod')
        start_date: Start of window (inclusive)
        end_date: End of window (inclusive)
        source_name: Optional filter by source
        universe: Optional filter by symbols

    Returns:
        Tuple of (green_fraction, status_counts)
        - green_fraction: float in [0, 1]
        - status_counts: dict with GREEN, AMBER, RED counts
    """
    df = load_partition_status()

    if df.empty:
        return 0.0, {"GREEN": 0, "AMBER": 0, "RED": 0}

    # Filter by table
    mask = df["table_name"] == table_name

    # Filter by source if provided
    if source_name:
        mask &= df["source_name"] == source_name

    # Filter by date range
    df_dt = pd.to_datetime(df["dt"]).dt.date
    mask &= (df_dt >= start_date) & (df_dt <= end_date)

    # Filter by universe if provided
    if universe:
        mask &= df["symbol"].isin(universe)

    filtered = df[mask]

    if filtered.empty:
        return 0.0, {"GREEN": 0, "AMBER": 0, "RED": 0}

    status_counts = filtered["status"].value_counts().to_dict()

    # Ensure all keys exist
    for status in ["GREEN", "AMBER", "RED"]:
        if status not in status_counts:
            status_counts[status] = 0

    total = sum(status_counts.values())
    green_count = status_counts.get("GREEN", 0)

    green_fraction = green_count / total if total > 0 else 0.0

    return green_fraction, status_counts


def get_gaps(
    table_name: str,
    start_date: date,
    end_date: date,
    source_name: Optional[str] = None,
    universe: Optional[List[str]] = None,
    statuses: Optional[List[PartitionStatus]] = None
) -> pd.DataFrame:
    """
    Get partitions with specific statuses (gaps) for backfill prioritization.

    Args:
        table_name: Table to check
        start_date: Start of window
        end_date: End of window
        source_name: Optional filter by source
        universe: Optional filter by symbols
        statuses: Statuses to include (default: RED, AMBER)

    Returns:
        DataFrame with gap partitions sorted by date desc
    """
    if statuses is None:
        statuses = [PartitionStatus.RED, PartitionStatus.AMBER]

    df = load_partition_status()

    if df.empty:
        return df

    # Filter
    mask = df["table_name"] == table_name

    if source_name:
        mask &= df["source_name"] == source_name

    df_dt = pd.to_datetime(df["dt"]).dt.date
    mask &= (df_dt >= start_date) & (df_dt <= end_date)

    if universe:
        mask &= df["symbol"].isin(universe)

    status_strs = [s.value for s in statuses]
    mask &= df["status"].isin(status_strs)

    filtered = df[mask].copy()

    if not filtered.empty:
        filtered = filtered.sort_values("dt", ascending=False)

    return filtered


def init_ledger() -> None:
    """
    Initialize empty partition_status ledger if it doesn't exist.

    Call this at system startup to ensure the ledger file exists.
    """
    path = _get_ledger_path()

    if path.exists():
        logger.debug("Partition status ledger already exists")
        return

    empty = _create_empty_ledger()
    save_partition_status(empty)
    logger.info("Initialized empty partition_status ledger")


def generate_coverage_report(
    tables: Optional[List[str]] = None,
    days_back: int = 365
) -> Dict[str, dict]:
    """
    Generate coverage report for monitoring.

    Args:
        tables: Tables to report on (default: all)
        days_back: How far back to look

    Returns:
        Dict with table -> {green_frac, amber_count, red_count, total}
    """
    df = load_partition_status()

    if df.empty:
        return {}

    end_date = date.today()
    start_date = end_date - timedelta(days=days_back)

    # Filter by date range
    df_dt = pd.to_datetime(df["dt"]).dt.date
    mask = (df_dt >= start_date) & (df_dt <= end_date)

    if tables:
        mask &= df["table_name"].isin(tables)

    filtered = df[mask]

    if filtered.empty:
        return {}

    report = {}
    for table_name in filtered["table_name"].unique():
        table_df = filtered[filtered["table_name"] == table_name]
        status_counts = table_df["status"].value_counts().to_dict()

        total = len(table_df)
        green = status_counts.get("GREEN", 0)
        amber = status_counts.get("AMBER", 0)
        red = status_counts.get("RED", 0)

        report[table_name] = {
            "green_fraction": green / total if total > 0 else 0.0,
            "green_count": green,
            "amber_count": amber,
            "red_count": red,
            "total": total,
        }

    return report
