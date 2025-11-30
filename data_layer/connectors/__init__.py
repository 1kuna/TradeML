"""Data connectors for free-tier market data sources."""

from __future__ import annotations

import os
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
from .polygon_connector import PolygonConnector


def run_edge_scheduler(asof: Optional[str] = None, config_path: str = "configs/edge.yml") -> Dict[str, Any]:
    """
    Run the edge scheduler to collect data for the given date.

    This is a DAG-friendly wrapper around the EdgeCollector.

    Args:
        asof: As-of date string (YYYY-MM-DD). Defaults to today.
        config_path: Path to edge collector config.

    Returns:
        Dict with dataset results keyed by dataset name.
    """
    results: Dict[str, Any] = {}

    if not Path(config_path).exists():
        logger.warning(f"Edge config not found: {config_path}")
        return {"status": "error", "reason": "config_not_found"}

    try:
        # Import EdgeCollector from scripts
        import sys
        scripts_path = str(Path(__file__).parent.parent.parent / "scripts")
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)

        from edge_collector import EdgeCollector

        collector = EdgeCollector(config_path)
        collector.run()

        # Return basic status - detailed metrics would be in logs
        results = {
            "status": "ok",
            "config": config_path,
            "asof": asof,
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
                WHERE table_name = %s AND dt = %s AND (symbol = %s OR %s IS NULL)
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
        logger.debug("psycopg2 not available; falling back to direct backfill")
        results["status"] = "queue_unavailable"
        results["fallback"] = "direct"

        # Attempt direct backfill
        try:
            from ops.ssot.backfill import backfill_run
            backfill_run(budget={"alpaca": 10, "finnhub": 5})
            results["status"] = "direct_backfill_triggered"
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
    "PolygonConnector",
    "run_edge_scheduler",
    "backfill_partition",
]
