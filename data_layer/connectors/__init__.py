"""Data connectors for free-tier market data sources."""

from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path
from typing import Dict, Optional, Any

from loguru import logger

from .base import BaseConnector, ConnectorError, RateLimitError, DataQualityError
from .alpaca_connector import AlpacaConnector
from .alpha_vantage_connector import AlphaVantageConnector
from .fred_connector import FREDConnector
from .finnhub_connector import FinnhubConnector
from .fmp_connector import FMPConnector
from .massive_connector import MassiveConnector
# Expose EdgeCollector for tests/mocks while keeping lazy import in run_edge_scheduler
# NOTE: EdgeCollector is legacy - new code should use data_node instead
try:
    from legacy.scripts.edge_collector import EdgeCollector  # type: ignore
except Exception:
    EdgeCollector = None  # type: ignore


def run_edge_scheduler(
    asof: Optional[str] = None,
    config_path: str = "configs/edge.yml",
    node_id: Optional[str] = None,
    budgets: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """
    Run the edge scheduler to collect data for the given date.

    This is a DAG-friendly wrapper around the EdgeCollector.

    Args:
        asof: As-of date string (YYYY-MM-DD). Defaults to today.
        config_path: Path to edge collector config.
        node_id: Logical node identity (EDGE_NODE_ID) for leases/logs.
        budgets: Optional per-vendor request budgets to enforce.

    Returns:
        Dict with dataset results keyed by dataset name.
    """
    results: Dict[str, Any] = {}

    if not Path(config_path).exists():
        logger.warning(f"Edge config not found: {config_path}")
        return {"status": "error", "reason": "config_not_found"}

    try:
        # Import EdgeCollector from legacy/scripts
        legacy_path = str(Path(__file__).parent.parent.parent / "legacy" / "scripts")
        if legacy_path not in sys.path:
            sys.path.insert(0, legacy_path)

        from edge_collector import EdgeCollector as _EdgeCollector  # type: ignore

        # Inject asof into environment for the collector to honor
        if asof:
            os.environ["EDGE_ASOF"] = asof
        if node_id:
            os.environ["EDGE_NODE_ID"] = node_id
        if budgets:
            for vendor, limit in budgets.items():
                os.environ[f"NODE_MAX_INFLIGHT_{vendor.upper()}"] = str(limit)

        collector = _EdgeCollector(config_path)
        collector.run()

        # Return basic status - detailed metrics would be in logs
        results = {
            "status": "ok",
            "config": config_path,
            "asof": asof,
            "node_id": node_id or os.getenv("EDGE_NODE_ID"),
            "budgets": budgets or {},
            "datasets": getattr(collector, "_vendor_symbol_cursor", {}),
        }

    except ImportError as e:
        logger.error(f"EdgeCollector import failed: {e}")
        results = {"status": "error", "reason": "import_failed", "error": str(e)}
    except Exception as e:
        logger.exception(f"Edge scheduler failed: {e}")
        results = {"status": "error", "reason": "execution_failed", "error": str(e)}

    return results


def backfill_partition(
    table: str,
    date: str,
    symbol: Optional[str] = None,
    priority: int = 1,
) -> Dict[str, Any]:
    """
    Backfill a specific partition.

    Attempts to enqueue the partition for backfill processing.
    Falls back to direct fetch if queue is unavailable.

    Args:
        table: Table name (e.g., "equities_eod", "equities_minute")
        date: Date string (YYYY-MM-DD)
        symbol: Optional symbol to filter
        priority: Queue priority (higher = sooner)

    Returns:
        Dict with status and any errors.
    """
    results: Dict[str, Any] = {"table": table, "date": date, "symbol": symbol}

    # Try to enqueue via PostgreSQL queue
    try:
        import psycopg2

        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            user=os.getenv("POSTGRES_USER", "trademl"),
            password=os.getenv("POSTGRES_PASSWORD", "trademl"),
            dbname=os.getenv("POSTGRES_DB", "trademl"),
        )

        with conn, conn.cursor() as cur:
            # Check if already in queue
            cur.execute(
                """
                SELECT id FROM backfill_queue
                WHERE table_name = %s
                  AND dt = %s
                  AND (
                        (symbol IS NULL AND %s IS NULL)
                        OR (symbol = %s)
                  )
                LIMIT 1
                """,
                (table, date, symbol, symbol),
            )
            existing = cur.fetchone()

            if existing:
                results["status"] = "already_queued"
                results["queue_id"] = existing[0]
            else:
                # Insert into queue
                cur.execute(
                    """
                    INSERT INTO backfill_queue (source, table_name, symbol, dt, priority, enqueued_at)
                    VALUES (%s, %s, %s, %s, %s, NOW())
                    RETURNING id
                    """,
                    ("dag", table, symbol, date, priority),
                )
                queue_id = cur.fetchone()[0]
                results["status"] = "enqueued"
                results["queue_id"] = queue_id

        conn.close()

    except ImportError:
        logger.debug("psycopg2 not available; falling back to direct, single-partition backfill")
        results["status"] = "queue_unavailable"
        results["fallback"] = "direct_single"

        # Direct backfill requires a symbol to avoid blasting entire universes
        if symbol is None:
            results["error"] = "symbol_required_for_direct_backfill"
            return results

        try:
            import pandas as pd
            from data_layer.connectors.alpaca_connector import AlpacaConnector

            connector = AlpacaConnector(
                api_key=os.getenv("ALPACA_API_KEY"),
                secret_key=os.getenv("ALPACA_SECRET_KEY"),
            )
            asof = pd.to_datetime(date).date()
            timeframe = "1Day" if table == "equities_eod" else "1Min"
            df = connector.fetch_and_transform(
                symbols=[symbol],
                start_date=asof,
                end_date=asof,
                timeframe=timeframe,
                source_uri=f"backfill://{table}/{date}/{symbol}",
            )
            if df.empty:
                results["status"] = "no_data"
                results["rows"] = 0
                return results

            if table == "equities_eod":
                out_root = Path("data_layer/raw/alpaca/equities_bars")
            elif table == "equities_minute":
                out_root = Path("data_layer/raw/alpaca/equities_bars_minute")
            else:
                results["status"] = "unsupported_table"
                return results

            connector.write_parquet(df, out_root, partition_cols=["date"])
            results["status"] = "direct_backfill_written"
            results["rows"] = len(df)
        except Exception as e:
            results["status"] = "error"
            results["error"] = str(e)

    except Exception as e:
        logger.warning(f"Backfill partition failed: {e}")
        results["status"] = "error"
        results["error"] = str(e)

    return results


__all__ = [
    "BaseConnector",
    "ConnectorError",
    "RateLimitError",
    "DataQualityError",
    "AlpacaConnector",
    "AlphaVantageConnector",
    "FREDConnector",
    "FinnhubConnector",
    "FMPConnector",
    "MassiveConnector",
    "run_edge_scheduler",
    "backfill_partition",
]
