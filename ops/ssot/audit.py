from __future__ import annotations

"""
Audit/Gaps planner for GREEN/AMBER/RED completeness ledger and backfill queue.

Computes per-(source, table, symbol, dt) completeness using:
- expected trading sessions from calendars
- configured expected row counts per table
- actual observed raw partitions

Writes:
- data_layer/qc/partition_status.parquet (canonical ledger) via atomic writes
- DB mirror (partition_status) if DB is reachable
- enqueues RED/AMBER rows into backfill_queue with priority policy

SSOT v2 Section 2.3: Completeness & GREEN/AMBER/RED
"""

import os
import io
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml
from loguru import logger

from data_layer.storage.s3_client import get_s3_client
from data_layer.connectors.polygon_connector import PolygonConnector
from data_layer.qc.partition_status import (
    update_partition_status,
    init_ledger,
    load_partition_status,
    get_green_coverage,
)
from ops.ssot.budget import BudgetManager
from data_layer.reference.calendars import get_trading_days


def _load_universe(path: str = "data_layer/reference/universe_symbols.txt") -> List[str]:
    p = Path(path)
    if not p.exists():
        return ["AAPL", "MSFT", "GOOGL"]
    return [s.strip() for s in p.read_text().splitlines() if s.strip()]


def _save_partition_status(rows: List[dict]):
    """Save partition status using atomic writes via partition_status module."""
    # Initialize ledger if needed
    init_ledger()
    # Use atomic upsert
    update_partition_status(rows)
    logger.info(f"Updated completeness ledger with {len(rows)} rows")


def _db_insert_partition_status(rows: List[dict]):
    # Best-effort DB mirror; silently continue if not reachable
    try:
        import psycopg2

        host = os.getenv("PGHOST") or os.getenv("POSTGRES_HOST", "localhost")
        port = int(os.getenv("PGPORT") or os.getenv("POSTGRES_PORT", "5432"))
        user = os.getenv("PGUSER") or os.getenv("POSTGRES_USER", "trademl")
        password = os.getenv("PGPASSWORD") or os.getenv("POSTGRES_PASSWORD", "trademl")
        dbname = os.getenv("PGDATABASE") or os.getenv("POSTGRES_DB", "trademl")
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            user=os.getenv("POSTGRES_USER", "trademl"),
            password=os.getenv("POSTGRES_PASSWORD", "trademl"),
            dbname=os.getenv("POSTGRES_DB", "trademl"),
        )
        with conn, conn.cursor() as cur:
            for r in rows:
                cur.execute(
                    """
                    INSERT INTO partition_status (source, table_name, symbol, dt, status, rows, expected_rows, qc_score, last_checked, notes)
                    VALUES (%(source)s, %(table_name)s, %(symbol)s, %(dt)s, %(status)s, %(rows)s, %(expected_rows)s, %(qc_score)s, %(last_checked)s, %(notes)s)
                    ON CONFLICT (source, table_name, symbol, dt)
                    DO UPDATE SET status=EXCLUDED.status, rows=EXCLUDED.rows, expected_rows=EXCLUDED.expected_rows, qc_score=EXCLUDED.qc_score, last_checked=EXCLUDED.last_checked, notes=EXCLUDED.notes
                    """,
                    r,
                )
        conn.close()
        logger.info(f"Mirrored {len(rows)} rows into partition_status")
    except Exception as e:
        logger.warning(f"DB mirror skipped: {e}")


def _db_enqueue_backfill(rows: List[dict], priorities: Dict[str, int]):
    try:
        import psycopg2

        host = os.getenv("PGHOST") or os.getenv("POSTGRES_HOST", "localhost")
        port = int(os.getenv("PGPORT") or os.getenv("POSTGRES_PORT", "5432"))
        user = os.getenv("PGUSER") or os.getenv("POSTGRES_USER", "trademl")
        password = os.getenv("PGPASSWORD") or os.getenv("POSTGRES_PASSWORD", "trademl")
        dbname = os.getenv("PGDATABASE") or os.getenv("POSTGRES_DB", "trademl")
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            user=os.getenv("POSTGRES_USER", "trademl"),
            password=os.getenv("POSTGRES_PASSWORD", "trademl"),
            dbname=os.getenv("POSTGRES_DB", "trademl"),
        )
        with conn, conn.cursor() as cur:
            for r in rows:
                prio = priorities.get(r["table_name"], 50)
                cur.execute(
                    """
                    INSERT INTO backfill_queue (source, table_name, symbol, dt, priority, attempts, enqueued_at)
                    VALUES (%s, %s, %s, %s, %s, 0, NOW())
                    ON CONFLICT (source, table_name, symbol, dt) DO NOTHING
                    """,
                    (r["source"], r["table_name"], r["symbol"], r["dt"], prio),
                )
        conn.close()
        logger.info(f"Enqueued {len(rows)} gaps into backfill_queue")
    except Exception as e:
        logger.warning(f"DB enqueue skipped: {e}")


def audit_scan(tables: List[str]) -> None:
    """Compute completeness (GREEN/AMBER/RED) for requested tables and enqueue gaps.

    Currently supports:
      - equities_eod (raw source: alpaca/equities_bars with timeframe 1Day)
    """
    cfg_path = Path("configs/backfill.yml")
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)

    # Optional budget manager (for QC sampling via Polygon)
    bm = None
    try:
        s3 = get_s3_client() if os.getenv("STORAGE_BACKEND", "local").lower() == "s3" else None
        budgets_cfg = {k: int(v.get("requests", 0)) for k, v in (cfg.get("policy", {}).get("daily_api_budget", {}) or {}).items()}
        bm = BudgetManager(initial_limits=budgets_cfg, s3_client=s3)
    except Exception:
        bm = None

    # Universe for symbol-level completeness
    universe = _load_universe()

    # Expected rows per table
    targets = cfg.get("targets", {})
    priorities = {p["table"]: int(p["weight"]) for p in cfg.get("policy", {}).get("priorities", [])}

    storage_backend = os.getenv("STORAGE_BACKEND", "local").lower()
    s3 = None
    if storage_backend == "s3":
        s3 = get_s3_client()

    all_rows: List[dict] = []
    enqueue_rows: List[dict] = []

    for table in tables:
        if table not in targets:
            logger.warning(f"Unknown table in backfill targets: {table}")
            continue

        tcfg = targets[table]
        earliest = pd.to_datetime(tcfg.get("earliest", "2015-01-01")).date()
        expect_rows = int(tcfg.get("expect_rows", tcfg.get("expect_contracts_min", 1)))

        # Where to look for raw data
        if table == "equities_eod":
            # Raw path stores Alpaca bars under raw/<source>/<table>
            raw_prefix = "raw/alpaca/equities_bars"
            source = "alpaca"
        elif table == "equities_minute":
            raw_prefix = "raw/alpaca/equities_bars_minute"
            source = "alpaca"
        elif table == "options_chains":
            raw_prefix = "raw/finnhub/options_chains"
            source = "finnhub"
        else:
            logger.warning(f"Table not yet supported by audit: {table}")
            continue

        # List available date partitions (supports source-first and table-first local layouts)
        if s3:
            objs = s3.list_objects(prefix=f"{raw_prefix}/date=")
            dates = sorted({k["Key"].split("date=")[-1].split("/")[0] for k in objs if "/data.parquet" in k["Key"]})
        else:
            def _local_dates(prefix: str, sub: bool) -> set:
                root = Path("data_layer") / prefix
                if not root.exists():
                    return set()
                if sub:
                    return {p.parent.name.split("=")[-1] for p in root.glob("date=*/underlier=*/data.parquet")}
                return {p.parent.name.split("=")[-1] for p in root.glob("date=*/data.parquet")}

            # source-first primary
            primary = f"{raw_prefix}"
            # table-first alternate
            src, tbl = raw_prefix.split("/")[1:3]
            alternate = f"raw/{tbl}/{src}"
            if table == "options_chains":
                dates = sorted(_local_dates(primary, True) | _local_dates(alternate, True))
            else:
                dates = sorted(_local_dates(primary, False) | _local_dates(alternate, False))

        if not dates:
            logger.warning(f"No raw date partitions found under {raw_prefix}")

        # Compute completeness per date per symbol
        today = date.today()
        all_days = get_trading_days(earliest, today)
        available = {pd.to_datetime(d).date() for d in dates}

        for d in all_days:
            status_map: Dict[str, tuple[str, int]] = {}
            qc_notes: Dict[str, str] = {}
            if d in available:
                # Read partition and group by symbol rows
                if s3:
                    if table == "options_chains":
                        # Fan-in underliers for the date
                        objs = s3.list_objects(prefix=f"{raw_prefix}/date={d.isoformat()}/")
                        frames = []
                        for o in objs:
                            if not o["Key"].endswith("/data.parquet"):
                                continue
                            try:
                                rb, _ = s3.get_object(o["Key"]) 
                                frames.append(pd.read_parquet(io.BytesIO(rb)))
                            except Exception:
                                pass
                        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
                    else:
                        key = f"{raw_prefix}/date={d.isoformat()}/data.parquet"
                        try:
                            raw_bytes, _ = s3.get_object(key)
                            df = pd.read_parquet(io.BytesIO(raw_bytes))
                        except Exception:
                            df = pd.DataFrame()
                else:
                    if table == "options_chains":
                        parts = list((Path("data_layer") / raw_prefix / f"date={d.isoformat()}").glob("underlier=*/data.parquet"))
                        frames = [pd.read_parquet(p) for p in parts if p.exists()]
                        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
                    else:
                        path = Path("data_layer") / raw_prefix / f"date={d.isoformat()}" / "data.parquet"
                        df = pd.read_parquet(path) if path.exists() else pd.DataFrame()

                if not df.empty and "symbol" in df.columns:
                    counts = df.groupby("symbol").size().to_dict()
                    for sym in universe:
                        rows = int(counts.get(sym, 0))
                        # Half-day tolerance for minute bars if configured
                        if table == "equities_minute" and cfg.get("qc_thresholds", {}).get("allow_halfdays", True):
                            half_expect = max(1, int(0.5 * expect_rows))
                            if rows >= expect_rows:
                                st = "GREEN"
                            elif rows >= half_expect:
                                st = "AMBER"
                            else:
                                st = "RED"
                        else:
                            st = "GREEN" if rows >= expect_rows else ("AMBER" if rows > 0 else "RED")
                        status_map[sym] = (st, rows)
                    # QC sampling vs Polygon for today only (limit calls)
                    if table in ("equities_eod", "equities_minute") and d == date.today() and os.getenv("POLYGON_API_KEY"):
                        try:
                            pconn = PolygonConnector(api_key=os.getenv("POLYGON_API_KEY"))
                            # Sample up to 5 symbols
                            sample_syms = [s for s in universe if s in counts][:5]
                            for i, sym in enumerate(sample_syms):
                                if bm and not bm.try_consume("polygon", 1):
                                    break
                                if table == "equities_eod":
                                    pdf = pconn.fetch_aggregates(sym, d, d, timespan="day")
                                    if not pdf.empty and "close" in df.columns:
                                        a_row = df[df["symbol"] == sym]
                                        if not a_row.empty and "close" in a_row.columns:
                                            a_close = float(a_row.iloc[0]["close"])
                                            p_close = float(pdf.iloc[0]["close"])
                                            diff_bps = 10000.0 * (a_close - p_close) / p_close if p_close else 0.0
                                            if abs(diff_bps) >= 20:
                                                qc_notes[sym] = f"poly_close_diff_bps={diff_bps:.1f}"
                                else:  # minute
                                    pdf = pconn.fetch_aggregates(sym, d, d, timespan="minute")
                                    p_rows = len(pdf) if not pdf.empty else 0
                                    a_rows = int(counts.get(sym, 0))
                                    if abs(a_rows - p_rows) >= 10:
                                        qc_notes[sym] = f"poly_min_rows={p_rows}"
                        except Exception:
                            pass
                elif not df.empty and table == "options_chains" and "underlier" in df.columns:
                    counts = df.groupby("underlier").size().to_dict()
                    for sym in universe:
                        rows = int(counts.get(sym, 0))
                        st = "GREEN" if rows >= expect_rows else ("AMBER" if rows > 0 else "RED")
                        status_map[sym] = (st, rows)
                else:
                    for sym in universe:
                        status_map[sym] = ("RED", 0)
            else:
                for sym in universe:
                    status_map[sym] = ("RED", 0)

            # Record rows
            for sym, (st, rows) in status_map.items():
                # Basic QC score & notes (threshold-based)
                status_str, row_count = st, rows
                notes = None
                if status_str == "AMBER":
                    if row_count < expect_rows:
                        notes = f"rows_low:{row_count}/{expect_rows}"
                        if table == "equities_minute" and row_count >= int(0.5 * expect_rows):
                            notes += ",halfday_tolerated"
                elif status_str == "RED":
                    notes = "missing"
                # Attach QC notes from polygon sampling if available
                if sym in qc_notes:
                    notes = (notes + "," if notes else "") + qc_notes[sym]

                row = {
                    "source": source,
                    "table_name": table,
                    "symbol": sym,
                    "dt": d,
                    "status": status_str,
                    "rows": row_count,
                    "expected_rows": expect_rows,
                    "qc_score": 1.0 if status_str == "GREEN" else (0.5 if status_str == "AMBER" else 0.0),
                    "last_checked": datetime.utcnow(),
                    "notes": notes,
                }
                all_rows.append(row)
                if status_str in ("RED", "AMBER"):
                    enqueue_rows.append(row)

    if not all_rows:
        logger.info("No partitions evaluated; audit finished with no rows")
        return

    # Save via atomic writes
    _save_partition_status(all_rows)
    _db_insert_partition_status(all_rows)
    # Enqueue only the most recent X days per symbol to keep queue bounded (simple heuristic)
    if enqueue_rows:
        # Keep last 365 sessions per symbol-table for queue entries
        enq_df = pd.DataFrame(enqueue_rows)
        enq_df = enq_df.sort_values(["symbol", "dt"], ascending=[True, False])
        enq_pruned = enq_df.groupby(["table_name", "symbol"]).head(365)
        _db_enqueue_backfill(enq_pruned.to_dict(orient="records"), priorities)

    logger.info("Audit scan complete")
