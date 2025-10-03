from __future__ import annotations

"""
Backfill subsystem

Two modes:
1) Queue-driven: consume gaps from DB backfill_queue ordered by priority.
2) Window sweep: backward fill by chunks (safety net) for equities_eod.

Idempotent writes to raw/ using partitioned date directories. Uses S3 leases
to avoid concurrent workers stepping on each other.
"""

import os
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import threading

import pandas as pd
import yaml
from loguru import logger

from data_layer.connectors.alpaca_connector import AlpacaConnector
from data_layer.storage.s3_client import get_s3_client
from data_layer.storage.lease_manager import LeaseManager
from ops.ssot.budget import BudgetManager
from data_layer.connectors.finnhub_connector import FinnhubConnector
from data_layer.connectors.polygon_connector import PolygonConnector
from data_layer.connectors.fred_connector import FREDConnector
from utils.concurrency import partition_lock, worker_count
import json


def _load_cfg() -> dict:
    with open("configs/backfill.yml") as f:
        return yaml.safe_load(f)


def _update_backfill_mark(table: str, cursor_date: date):
    """Persist a simple per-table backfill cursor as JSON manifest.

    This is a lightweight marker to reflect how far the backward sweep has progressed.
    """
    path = Path("data_layer/manifests/backfill_marks.json")
    try:
        data = json.loads(path.read_text()) if path.exists() else {}
    except Exception:
        data = {}
    data[str(table)] = {"last_cursor": cursor_date.isoformat()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def _load_universe(path: str = "data_layer/reference/universe_symbols.txt") -> List[str]:
    p = Path(path)
    if not p.exists():
        return ["AAPL", "MSFT", "GOOGL"]
    return [s.strip() for s in p.read_text().splitlines() if s.strip()]


def _append_raw_partition(df: pd.DataFrame, source: str, table: str, ds: date):
    """Append-merge rows into a date partition with basic de-duplication.

    - For S3: use ETag-based conditional writes with retry to avoid races.
    - For local: serialize within-process via partition lock.
    """
    backend = os.getenv("STORAGE_BACKEND", "local").lower()
    part_key = f"raw/{source}/{table}/date={ds.isoformat()}"
    subset_cols = [c for c in ["date", "symbol", "open", "high", "low", "close", "volume"] if c in df.columns]
    if backend == "s3":
        s3 = get_s3_client()
        key = f"{part_key}/data.parquet"
        import io
        # Conditional upsert loop
        attempts = 0
        while attempts < 5:
            attempts += 1
            try:
                try:
                    data, etag = s3.get_object(key)
                    existing = pd.read_parquet(io.BytesIO(data))
                    merged = pd.concat([existing, df], ignore_index=True)
                    if subset_cols:
                        merged = merged.drop_duplicates(subset=subset_cols)
                    buf = io.BytesIO()
                    merged.to_parquet(buf, index=False)
                    s3.put_object(key, buf.getvalue(), if_match=etag)
                    logger.info(f"Upserted {len(df)} rows (total {len(merged)}) to s3://{s3.bucket}/{key}")
                    return
                except Exception:
                    # Assume not found or stale etag; attempt create-if-absent
                    buf = io.BytesIO()
                    df if subset_cols else df
                    df.to_parquet(buf, index=False)
                    s3.put_object(key, buf.getvalue(), if_none_match="*")
                    logger.info(f"Created partition with {len(df)} rows at s3://{s3.bucket}/{key}")
                    return
            except Exception as e:
                # Precondition failed or transient, retry
                if attempts >= 5:
                    logger.warning(f"S3 upsert failed after retries for {key}: {e}")
                    raise
        return
    else:
        # Local source-first layout with in-process lock
        lock = partition_lock(part_key)
        with lock:
            path = Path("data_layer/raw") / source / table / f"date={ds.isoformat()}"
            path.mkdir(parents=True, exist_ok=True)
            out = path / "data.parquet"
            if out.exists():
                existing = pd.read_parquet(out)
                merged = pd.concat([existing, df], ignore_index=True)
                if subset_cols:
                    merged = merged.drop_duplicates(subset=subset_cols)
            else:
                merged = df
            merged.to_parquet(out, index=False)
            logger.info(f"Upserted {len(df)} rows (total {len(merged)}) to {out}")


def _lease_manager_if_s3() -> Optional[LeaseManager]:
    if os.getenv("STORAGE_BACKEND", "local").lower() != "s3":
        return None
    try:
        s3 = get_s3_client()
        return LeaseManager(s3, lease_seconds=180, renew_seconds=60)
    except Exception as e:
        logger.warning(f"Lease manager unavailable: {e}")
        return None


def _db_conn():
    import psycopg2
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        user=os.getenv("POSTGRES_USER", "trademl"),
        password=os.getenv("POSTGRES_PASSWORD", "trademl"),
        dbname=os.getenv("POSTGRES_DB", "trademl"),
    )


def _fetch_queue_batch(limit: int = 100) -> List[Tuple]:
    """Fetch a batch of backfill_queue tasks by priority."""
    try:
        conn = _db_conn()
        with conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, source, table_name, symbol, dt, priority, attempts
                FROM backfill_queue
                ORDER BY priority DESC, enqueued_at NULLS LAST
                LIMIT %s
                """,
                (limit,),
            )
            rows = cur.fetchall()
        conn.close()
        return rows
    except Exception as e:
        logger.warning(f"Queue fetch failed: {e}")
        return []


def _delete_queue_item(item_id: int):
    try:
        conn = _db_conn()
        with conn, conn.cursor() as cur:
            cur.execute("DELETE FROM backfill_queue WHERE id=%s", (item_id,))
        conn.close()
    except Exception as e:
        logger.warning(f"Queue delete failed: {e}")


def _touch_queue_item(item_id: int, err: Optional[str] = None):
    try:
        conn = _db_conn()
        with conn, conn.cursor() as cur:
            if err:
                cur.execute(
                    "UPDATE backfill_queue SET attempts=attempts+1, last_attempt=NOW(), last_err=%s WHERE id=%s",
                    (err[:2000], item_id),
                )
            else:
                cur.execute(
                    "UPDATE backfill_queue SET attempts=attempts+1, last_attempt=NOW() WHERE id=%s",
                    (item_id,),
                )
        conn.close()
    except Exception as e:
        logger.warning(f"Queue update failed: {e}")


def _estimate_requests(num_symbols: int, days: int, batch_size: int = 100) -> int:
    """Rough estimate of API requests for batch fetch (Alpaca bars)."""
    from math import ceil
    return max(1, ceil(num_symbols / batch_size)) * max(1, days)


def _process_queue_equities_eod(rows: List[Tuple], connector: AlpacaConnector, budget: Optional[BudgetManager] = None):
    """Process EOD queue items with bounded concurrency under a single lease."""
    lm = _lease_manager_if_s3()
    lease_name = "backfill.equities_eod"
    if lm and not lm.acquire(lease_name, force=True):
        logger.info("Backfill lease held elsewhere; skipping queue batch")
        return
    try:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _handle_item(item: Tuple):
            (item_id, source, table_name, symbol, dt, priority, attempts) = item
            if table_name != "equities_eod":
                return
            try:
                _touch_queue_item(item_id)  # record attempt
                ds = pd.to_datetime(dt).date()
                # Budget: assume one request per symbol-day
                if budget and not budget.try_consume("alpaca", 1):
                    logger.warning("Budget exhausted for Alpaca (queue item); yielding")
                    _yield_on_budget()
                    return
                df = connector.fetch_bars(symbols=[symbol], start_date=ds, end_date=ds, timeframe="1Day")
                if df.empty:
                    poly_ok = budget.try_consume("polygon", 1) if budget else True
                    if poly_ok:
                        try:
                            pconn = PolygonConnector(api_key=os.getenv("POLYGON_API_KEY"))
                            pdf = pconn.fetch_aggregates(symbol, ds, ds, timespan="day")
                        except Exception:
                            pdf = pd.DataFrame()
                        if not pdf.empty:
                            _append_raw_partition(pdf, source="polygon", table="equities_bars", ds=ds)
                        else:
                            logger.warning(f"No data for {symbol} on {ds}")
                    else:
                        logger.warning("Polygon budget exhausted; skipping EOD fallback")
                else:
                    _append_raw_partition(df[df["symbol"] == symbol], source="alpaca", table="equities_bars", ds=ds)
                _delete_queue_item(item_id)
            except Exception as e:
                logger.exception(f"Backfill item failed (id={item_id}): {e}")
                _touch_queue_item(item_id, err=str(e))

        max_workers = worker_count(default=4)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(_handle_item, r) for r in rows if r[2] == "equities_eod"]
            for _ in as_completed(futs):
                pass
    finally:
        if lm:
            lm.release(lease_name)


def _process_queue_equities_minute(rows: List[Tuple], connector: AlpacaConnector, budget: Optional[BudgetManager] = None):
    """Process minute queue items with bounded concurrency under a single lease."""
    lm = _lease_manager_if_s3()
    lease_name = "backfill.equities_minute"
    if lm and not lm.acquire(lease_name, force=True):
        logger.info("Backfill lease held elsewhere; skipping minute queue batch")
        return
    try:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _handle_item(item: Tuple):
            (item_id, source, table_name, symbol, dt, priority, attempts) = item
            if table_name != "equities_minute":
                return
            try:
                _touch_queue_item(item_id)
                ds = pd.to_datetime(dt).date()
                if budget and not budget.try_consume("alpaca", 1):
                    logger.warning("Budget exhausted for Alpaca (minute); yielding")
                    _yield_on_budget()
                    return
                df = connector.fetch_bars(symbols=[symbol], start_date=ds, end_date=ds, timeframe="1Min")
                if df.empty:
                    poly_ok = budget.try_consume("polygon", 1) if budget else True
                    if poly_ok:
                        try:
                            pconn = PolygonConnector(api_key=os.getenv("POLYGON_API_KEY"))
                            pdf = pconn.fetch_aggregates(symbol, ds, ds, timespan="minute")
                        except Exception:
                            pdf = pd.DataFrame()
                        if not pdf.empty:
                            _append_raw_partition(pdf, source="polygon", table="equities_bars_minute", ds=ds)
                        else:
                            logger.warning(f"No minute data for {symbol} on {ds}")
                    else:
                        logger.warning("Polygon budget exhausted; skipping minute fallback")
                else:
                    _append_raw_partition(df[df["symbol"] == symbol], source="alpaca", table="equities_bars_minute", ds=ds)
                _delete_queue_item(item_id)
            except Exception as e:
                logger.exception(f"Minute backfill item failed (id={item_id}): {e}")
                _touch_queue_item(item_id, err=str(e))

        max_workers = worker_count(default=4)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(_handle_item, r) for r in rows if r[2] == "equities_minute"]
            for _ in as_completed(futs):
                pass
    finally:
        if lm:
            lm.release(lease_name)


def _backfill_window_equities_eod(connector: AlpacaConnector, universe: List[str], earliest: date, chunk_days: int, budget: Optional[BudgetManager] = None):
    # Find last available raw date to determine backward window
    backend = os.getenv("STORAGE_BACKEND", "local").lower()
    if backend == "s3":
        s3 = get_s3_client()
        objs = s3.list_objects(prefix="raw/alpaca/equities_bars/date=")
        have_dates = sorted({o["Key"].split("date=")[-1].split("/")[0] for o in objs if "/data.parquet" in o["Key"]})
    else:
        have_dates = [p.parent.name.split("=")[-1] for p in Path("data_layer/raw/alpaca/equities_bars").glob("date=*/data.parquet")]

    have = {pd.to_datetime(ds).date() for ds in have_dates}

    today = date.today()
    d1 = today
    while d1 >= earliest:
        d0 = max(earliest, d1 - timedelta(days=chunk_days - 1))
        window_days = pd.date_range(d0, d1, freq="D").date
        if all(ds in have for ds in window_days):
            d1 = d0 - timedelta(days=1)
            continue

        # Budget control: estimate requests and shrink window if needed
        est_reqs = _estimate_requests(len(universe), len(window_days))
        if budget and not budget.try_consume("alpaca", est_reqs):
            # Try shrinking chunk to fit at least one request window
            logger.warning(f"Alpaca budget low; shrinking chunk from {chunk_days}d")
            new_chunk = max(1, chunk_days // 2)
            if new_chunk == chunk_days:
                logger.warning("Cannot shrink further; yielding EOD sweep for today")
                _yield_on_budget()
                break
            chunk_days = new_chunk
            continue

        logger.info(f"Backfill sweep {d0} → {d1} for {len(universe)} symbols")
        df = connector.fetch_bars(symbols=universe, start_date=d0, end_date=d1, timeframe="1Day")
        if df.empty:
            d1 = d0 - timedelta(days=1)
            continue
        for ds, sub in df.groupby("date"):
            _append_raw_partition(sub, source="alpaca", table="equities_bars", ds=ds)
        # Update backfill mark to the earliest date we just covered
        _update_backfill_mark("equities_eod", d0)
        d1 = d0 - timedelta(days=1)


def _backfill_window_equities_minute(connector: AlpacaConnector, universe: List[str], earliest: date, chunk_days: int, top_n: int = 50, budget: Optional[BudgetManager] = None):
    # Limit to top N by assumption; universe is prefiltered list
    symbols = universe[:top_n]
    today = date.today()
    d1 = today
    while d1 >= earliest:
        d0 = max(earliest, d1 - timedelta(days=chunk_days - 1))
        window_days = pd.date_range(d0, d1, freq="D").date
        est_reqs = _estimate_requests(len(symbols), len(window_days))
        if budget and not budget.try_consume("alpaca", est_reqs):
            logger.warning(f"Alpaca budget low; shrinking minute chunk from {chunk_days}d")
            new_chunk = max(1, chunk_days // 2)
            if new_chunk == chunk_days:
                logger.warning("Cannot shrink further; yielding minute sweep for today")
                _yield_on_budget()
                break
            chunk_days = new_chunk
            continue

        logger.info(f"Minute backfill sweep {d0} → {d1} for {len(symbols)} symbols (1Min)")
        df = connector.fetch_bars(symbols=symbols, start_date=d0, end_date=d1, timeframe="1Min")
        if df.empty:
            d1 = d0 - timedelta(days=1)
            continue
        for ds, sub in df.groupby("date"):
            _append_raw_partition(sub, source="alpaca", table="equities_bars_minute", ds=ds)
        # Update backfill mark to the earliest date we just covered for minute bars
        try:
            _update_backfill_mark("equities_minute", d0)
        except Exception:
            pass
        d1 = d0 - timedelta(days=1)


def _backfill_options_chains(universe: List[str], budget: Optional[BudgetManager] = None, per_day_limit: int = 10):
    """Exploratory free-tier options chains fetch for today using Finnhub.

    Partitions to raw/finnhub/options_chains/date=YYYY-MM-DD/underlier=SYM/data.parquet
    """
    api_key = os.getenv("FINNHUB_API_KEY")
    if not api_key:
        logger.debug("FINNHUB_API_KEY missing; skipping options_chains backfill")
        return
    conn = FinnhubConnector(api_key=api_key)
    ds = date.today()
    count = 0
    for sym in universe:
        if count >= per_day_limit:
            break
        if budget and not budget.try_consume("finnhub", 1):
            logger.warning("Finnhub budget exhausted; yielding options chains")
            _yield_on_budget()
            break
        try:
            df = conn.fetch_options_chain(sym)
            if df.empty:
                continue
            # Write under raw/finnhub/options_chains/date=.../underlier=...
            backend = os.getenv("STORAGE_BACKEND", "local").lower()
            if backend == "s3":
                s3 = get_s3_client()
                import io
                buf = io.BytesIO()
                df.to_parquet(buf, index=False)
                s3.put_object(f"raw/finnhub/options_chains/date={ds.isoformat()}/underlier={sym}/data.parquet", buf.getvalue())
            else:
                out = Path("data_layer/raw/finnhub/options_chains") / f"date={ds.isoformat()}" / f"underlier={sym}"
                out.mkdir(parents=True, exist_ok=True)
                df.to_parquet(out / "data.parquet", index=False)
            count += 1
        except Exception as e:
            logger.warning(f"Options chain fetch failed for {sym}: {e}")
    # Update backfill mark for options chains for today's date if any were fetched
    try:
        if count > 0:
            _update_backfill_mark("options_chains", ds)
    except Exception:
        pass


def _backfill_macro_treasury(earliest: date, chunk_days: int, budget: Optional[BudgetManager] = None):
    """Backfill macro Treasury curve from FRED by yearly windows.

    Writes to raw/fred/macro_treasury/date=YYYY-MM-DD/data.parquet
    """
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        logger.debug("FRED_API_KEY missing; skipping macro treasury backfill")
    return
    conn = FREDConnector(api_key=api_key)
    today = date.today()
    d1 = today
    while d1 >= earliest:
        d0 = max(earliest, d1 - timedelta(days=chunk_days - 1))
        # Budget: estimate one request per tenor group; fetch_treasury_curve uses multiple series
        if budget and not budget.try_consume("fred", 1):
            logger.warning("FRED budget exhausted; yielding macro backfill")
            _yield_on_budget()
            break
        try:
            df = conn.fetch_treasury_curve(start_date=d0, end_date=d1)
        except Exception as e:
            logger.warning(f"FRED fetch failed {d0}->{d1}: {e}")
            break
        if df.empty:
            d1 = d0 - timedelta(days=1)
            continue
        backend = os.getenv("STORAGE_BACKEND", "local").lower()
        for ds, sub in df.groupby("date"):
            if backend == "s3":
                s3 = get_s3_client()
                import io
                buf = io.BytesIO()
                sub.to_parquet(buf, index=False)
                s3.put_object(f"raw/fred/macro_treasury/date={ds.isoformat()}/data.parquet", buf.getvalue())
            else:
                out = Path("data_layer/raw/macro_treasury/fred") / f"date={ds.isoformat()}"
                out.mkdir(parents=True, exist_ok=True)
                sub.to_parquet(out / "data.parquet", index=False)
        d1 = d0 - timedelta(days=1)


def _backfill_macro_vintages(series: List[str], budget: Optional[BudgetManager] = None, per_day_limit: int = 2):
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        return
    conn = FREDConnector(api_key=api_key)
    out_root = Path("data_layer/raw/macro_vintages/fred")
    out_root.mkdir(parents=True, exist_ok=True)
    done = 0
    for sid in series:
        if budget and not budget.try_consume("fred", 1):
            _yield_on_budget()
            break
        try:
            df = conn.fetch_alfred_vintages(sid)
            if df.empty:
                continue
            p = out_root / f"{sid}.parquet"
            df.to_parquet(p, index=False)
            done += 1
        except Exception as e:
            logger.warning(f"ALFRED fetch failed for {sid}: {e}")
        if done >= per_day_limit:
            break


def _yield_on_budget():
    """Sleep briefly when budget is exhausted, allowing node loop to cycle.

    Controlled via env BACKFILL_BUDGET_SLEEP_SECONDS (default 0 for tests).
    """
    try:
        secs = int(os.getenv("BACKFILL_BUDGET_SLEEP_SECONDS", "0"))
    except Exception:
        secs = 0
    if secs > 0:
        import time
        logger.info(f"Budget depleted; sleeping {secs}s to yield")
        try:
            time.sleep(secs)
        except Exception:
            pass


def backfill_run(budget: Dict[str, int] | None = None) -> None:
    """Fill gaps for supported tables using queue first, then window sweep safety net."""
    cfg = _load_cfg()
    targets = cfg.get("targets", {})
    policy = cfg.get("policy", {})
    budgets_cfg = {k: int(v.get("requests", 0)) for k, v in policy.get("daily_api_budget", {}).items()}

    # Budget manager (S3-aware)
    bm: Optional[BudgetManager] = None
    try:
        s3 = get_s3_client() if os.getenv("STORAGE_BACKEND", "local").lower() == "s3" else None
        bm = BudgetManager(initial_limits=budgets_cfg, s3_client=s3)
    except Exception as e:
        logger.warning(f"Budget manager unavailable: {e}")

    # At minimum we expect equities_eod; others optional
    if "equities_eod" not in targets:
        logger.warning("No equities_eod target in backfill.yml; nothing to do")
        return

    tcfg_eod = targets["equities_eod"]
    earliest_eod = pd.to_datetime(tcfg_eod.get("earliest", "2010-01-01")).date()
    chunk_days_eod = int(tcfg_eod.get("chunk_days", 30))

    universe = _load_universe()
    connector = AlpacaConnector(api_key=os.getenv("ALPACA_API_KEY"), secret_key=os.getenv("ALPACA_SECRET_KEY"))

    # 1) Process queue items first
    rows = _fetch_queue_batch(limit=200)
    if rows:
        _process_queue_equities_eod(rows, connector, budget=bm)
        _process_queue_equities_minute(rows, connector, budget=bm)

    # 2) Window sweeps (safety net)
    _backfill_window_equities_eod(connector, universe, earliest=earliest_eod, chunk_days=chunk_days_eod, budget=bm)

    if "equities_minute" in targets:
        tcfg_m = targets["equities_minute"]
        earliest_m = pd.to_datetime(tcfg_m.get("earliest", "2010-01-01")).date()
        chunk_days_m = int(tcfg_m.get("chunk_days", 5))
        _backfill_window_equities_minute(connector, universe, earliest=earliest_m, chunk_days=chunk_days_m, top_n=50, budget=bm)

    # 3) Options chains exploratory (forward only, small budget)
    if "options_chains" in targets:
        _backfill_options_chains(universe, budget=bm, per_day_limit=10)

    # 4) Macro treasury curve (yearly windows)
    if "macro_treasury" in targets:
        tcfg_mac = targets["macro_treasury"]
        earliest_mac = pd.to_datetime(tcfg_mac.get("earliest", "2000-01-01")).date()
        chunk_days_mac = int(tcfg_mac.get("chunk_days", 365))
        _backfill_macro_treasury(earliest=earliest_mac, chunk_days=chunk_days_mac, budget=bm)

    # 5) Macro ALFRED vintages (optional series list in config under macro_vintages.series)
    if "macro_vintages" in targets:
        series = targets["macro_vintages"].get("series", [])
        if isinstance(series, list) and series:
            _backfill_macro_vintages(series, budget=bm)

    logger.info("Backfill run complete")
