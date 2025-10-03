#!/usr/bin/env python
"""
Edge collector: Distributed data ingestion with resume capability.

Acquires lease, fetches data from vendor APIs, uploads to S3, tracks progress
with bookmarks. Supports graceful shutdown and resume from last checkpoint.
"""

import os
import sys
import signal
import time
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List
import threading

import yaml
import pandas as pd
from dotenv import load_dotenv
from loguru import logger

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_layer.storage.s3_client import get_s3_client
from data_layer.storage.lease_manager import LeaseManager
from data_layer.storage.bookmarks import BookmarkManager
from data_layer.connectors.alpaca_connector import AlpacaConnector
from utils.concurrency import max_inflight_for, worker_count
from concurrent.futures import ThreadPoolExecutor, as_completed
from ops.ssot.budget import BudgetManager


class EdgeCollector:
    """Edge collector with distributed locking and resume capability."""

    def __init__(self, config_path: str):
        """Initialize edge collector from config."""
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # Storage backend
        self.storage_backend = os.getenv("STORAGE_BACKEND", "local")
        logger.info(f"Storage backend: {self.storage_backend}")

        if self.storage_backend == "s3":
            self.s3 = get_s3_client()
            self.lease_mgr = LeaseManager(
                self.s3,
                lease_seconds=self.config.get("locks", {}).get("lease_seconds", 120),
                renew_seconds=self.config.get("locks", {}).get("renew_seconds", 45),
            )
            self.bookmarks = BookmarkManager(self.s3)
        else:
            self.s3 = None
            self.lease_mgr = None
            self.bookmarks = None

        # Connectors
        self.connectors = self._init_connectors()

        # Shutdown flag
        self.shutdown_requested = False
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
        except Exception:
            pass

    def _init_connectors(self):
        """Initialize data connectors from config."""
        connectors = {}

        # Alpaca connector
        if os.getenv("ALPACA_API_KEY"):
            connectors["alpaca"] = AlpacaConnector(
                api_key=os.getenv("ALPACA_API_KEY"),
                secret_key=os.getenv("ALPACA_SECRET_KEY"),
            )
            logger.info("Initialized Alpaca connector")

        # Add other connectors as needed...

        return connectors

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.warning(f"Received signal {signum}, shutting down gracefully...")
        self.shutdown_requested = True
        # Propagate interrupt so orchestrator can exit promptly
        raise KeyboardInterrupt

    def _acquire_lease(self, name: str) -> bool:
        """Acquire lease or exit if already held."""
        if not self.lease_mgr:
            return True  # No locking in local mode

        if self.lease_mgr.acquire(name, force=True):
            logger.info(f"Lease acquired: {name}")
            return True
        else:
            holder = self.lease_mgr.get_holder(name)
            logger.error(f"Lease already held by: {holder}")
            return False

    def _renew_lease_loop(self, name: str, interval: int):
        """Background lease renewal (should run in thread)."""
        while not self.shutdown_requested:
            time.sleep(interval)
            if not self.lease_mgr.renew(name):
                logger.error("Lease renewal failed, exiting")
                self.shutdown_requested = True
                break

    def _upload_to_s3(self, df: pd.DataFrame, source: str, table: str, date: str):
        """Upload dataframe to S3 with partitioning."""
        if not self.s3:
            # Local storage fallback (source-first layout for consistency)
            local_dir = Path(f"data_layer/raw/{source}/{table}/date={date}")
            local_dir.mkdir(parents=True, exist_ok=True)
            local_file = local_dir / "data.parquet"
            df.to_parquet(local_file, index=False)
            logger.info(f"Saved locally: {local_file}")
            return

        # S3 upload
        key = f"raw/{source}/{table}/date={date}/data.parquet"

        # Convert to parquet bytes
        import io
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False)
        data = buffer.getvalue()

        # Upload with temp key first, then rename (atomic-ish)
        temp_key = f"{key}.tmp"
        self.s3.put_object(temp_key, data)

        # Check if final key exists (idempotency)
        if self.s3.object_exists(key):
            logger.debug(f"Object already exists, skipping: {key}")
            self.s3.delete_object(temp_key)
            return

        # Rename temp to final
        # Note: S3 doesn't have atomic rename, so we copy+delete
        self.s3.put_object(key, data)
        self.s3.delete_object(temp_key)

        logger.info(f"Uploaded to S3: {key} ({len(df)} rows, {len(data)} bytes)")

    def _write_manifest(self, source: str, table: str, date: str, row_count: int):
        """Write manifest log for audit trail."""
        if not self.s3:
            return  # Skip in local mode

        manifest_key = f"manifests/{date}/manifest-{source}-{table}.jsonl"
        manifest_entry = {
            "source": source,
            "table": table,
            "date": date,
            "row_count": row_count,
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Append to manifest (read, append, write)
        import json
        try:
            existing_data, etag = self.s3.get_object(manifest_key)
            lines = existing_data.decode('utf-8').strip().split('\n')
        except:
            lines = []

        lines.append(json.dumps(manifest_entry))
        new_data = '\n'.join(lines).encode('utf-8')
        self.s3.put_object(manifest_key, new_data)
        logger.debug(f"Updated manifest: {manifest_key}")

    def _alpaca_fetch_day_parallel(self, connector: AlpacaConnector, symbols: List[str], day_str: str, inflight_override: Optional[int] = None) -> pd.DataFrame:
        """Fetch a single day of bars in parallel by symbol chunks, returning combined DataFrame."""
        # Chunk symbols to match connector's batch size guidance (100)
        BATCH = 100
        chunks = [symbols[i:i + BATCH] for i in range(0, len(symbols), BATCH)]
        inflight = inflight_override if inflight_override is not None else max(1, max_inflight_for("alpaca", default=int(os.getenv("NODE_WORKERS", "4"))))

        def _fetch_chunk(chunk):
            ds = pd.to_datetime(day_str).date()
            return connector.fetch_bars(symbols=chunk, start_date=ds, end_date=ds, timeframe="1Day")

        results = []
        with ThreadPoolExecutor(max_workers=inflight) as ex:
            futs = [ex.submit(_fetch_chunk, ch) for ch in chunks]
            try:
                for fut in as_completed(futs):
                    if self.shutdown_requested:
                        break
                    try:
                        dfp = fut.result()
                        if not dfp.empty:
                            results.append(dfp)
                    except Exception as e:
                        logger.warning(f"Alpaca chunk fetch failed: {e}")
            except KeyboardInterrupt:
                for f in futs:
                    f.cancel()
                raise

        if not results:
            return pd.DataFrame()
        return pd.concat(results, ignore_index=True)

    def collect_alpaca_bars(self):
        """Collect Alpaca equity bars with resume."""
        source = "alpaca"
        table = "equities_bars"
        lease_name = f"edge-{source}-{table}"

        # Acquire lease
        if not self._acquire_lease(lease_name):
            return

        renew_thread = None
        try:
            # Get last bookmark
            last_ts = None
            if self.bookmarks:
                last_ts = self.bookmarks.get_last_timestamp(source, table)

            if last_ts:
                start_date = datetime.fromisoformat(last_ts) + timedelta(days=1)
                logger.info(f"Resuming from bookmark: {last_ts}")
            else:
                # Default: fetch last 30 days
                start_date = datetime.now() - timedelta(days=30)
                logger.info("No bookmark, starting from 30 days ago")

            end_date = datetime.now()

            # Fetch symbols from universe
            universe_file = Path("data_layer/reference/universe_symbols.txt")
            if universe_file.exists():
                with open(universe_file) as f:
                    symbols = [line.strip() for line in f if line.strip()]
            else:
                symbols = ["AAPL", "MSFT", "GOOGL"]  # Fallback

            logger.info(f"Fetching {len(symbols)} symbols from {start_date.date()} to {end_date.date()}")

            connector = self.connectors.get("alpaca")
            if not connector:
                logger.error("Alpaca connector not available")
                return

            # Fetch data day by day for resume granularity
            current_date = start_date.date()
            end = end_date.date()

            # Start lease renewal thread if using leases
            if self.lease_mgr:
                renew_every = int(self.config.get("locks", {}).get("renew_seconds", 45))
                renew_thread = threading.Thread(target=self._renew_lease_loop, args=(lease_name, renew_every), daemon=True)
                renew_thread.start()

            while current_date <= end and not self.shutdown_requested:
                logger.info(f"Fetching data for {current_date}")

                # Parallel fetch bars for all symbols for this day
                df = self._alpaca_fetch_day_parallel(connector, symbols, current_date.isoformat())

                if df.empty:
                    logger.debug(f"No data for {current_date}")
                else:
                    # Upload to S3
                    self._upload_to_s3(df, source, table, current_date.isoformat())

                    # Write manifest
                    self._write_manifest(source, table, current_date.isoformat(), len(df))

                    # Update bookmark
                    if self.bookmarks:
                        self.bookmarks.set(
                            source=source,
                            table=table,
                            last_timestamp=current_date.isoformat(),
                            row_count=len(df),
                        )

                current_date += timedelta(days=1)

            logger.info("Collection complete")

        finally:
            # Release lease
            if self.lease_mgr:
                self.lease_mgr.release(lease_name)
            if renew_thread is not None:
                try:
                    self.shutdown_requested = True
                    renew_thread.join(timeout=2)
                except Exception:
                    pass

    # ---------------- Multi-source fan-out scheduler ----------------

    def _init_budget(self) -> Optional[BudgetManager]:
        policy = self.config.get("policy", {}) if isinstance(self.config, dict) else {}
        limits_cfg = {k: int(v.get("requests", 0)) for k, v in policy.get("daily_api_budget", {}).items()} if policy else {}
        if not limits_cfg:
            return None
        try:
            s3 = self.s3 if self.storage_backend == "s3" else None
            return BudgetManager(initial_limits=limits_cfg, s3_client=s3)
        except Exception:
            return BudgetManager(initial_limits=limits_cfg, s3_client=None)

    def _init_connectors(self):
        """Initialize data connectors from config."""
        connectors = {}

        # Alpaca connector
        if os.getenv("ALPACA_API_KEY"):
            connectors["alpaca"] = AlpacaConnector(
                api_key=os.getenv("ALPACA_API_KEY"),
                secret_key=os.getenv("ALPACA_SECRET_KEY"),
            )
            logger.info("Initialized Alpaca connector")

        # Polygon connector (optional)
        try:
            if os.getenv("POLYGON_API_KEY"):
                from data_layer.connectors.polygon_connector import PolygonConnector
                connectors["polygon"] = PolygonConnector(api_key=os.getenv("POLYGON_API_KEY"))
                logger.info("Initialized Polygon connector")
        except Exception as e:
            logger.warning(f"Polygon connector not available: {e}")

        # Finnhub connector (optional)
        try:
            if os.getenv("FINNHUB_API_KEY"):
                from data_layer.connectors.finnhub_connector import FinnhubConnector
                connectors["finnhub"] = FinnhubConnector(api_key=os.getenv("FINNHUB_API_KEY"))
                logger.info("Initialized Finnhub connector")
        except Exception as e:
            logger.warning(f"Finnhub connector not available: {e}")

        # FRED connector (optional)
        try:
            if os.getenv("FRED_API_KEY"):
                from data_layer.connectors.fred_connector import FREDConnector
                connectors["fred"] = FREDConnector(api_key=os.getenv("FRED_API_KEY"))
                logger.info("Initialized FRED connector")
        except Exception as e:
            logger.warning(f"FRED connector not available: {e}")

        return connectors

    def _symbols_universe(self) -> List[str]:
        universe_file = Path("data_layer/reference/universe_symbols.txt")
        if universe_file.exists():
            with open(universe_file) as f:
                return [line.strip() for line in f if line.strip()]
        return ["AAPL", "MSFT", "GOOGL"]

    def _lease_name_for_task(self, task_name: str) -> str:
        mapping = {
            "alpaca_bars": "edge-alpaca-equities_bars",
            "polygon_bars": "edge-polygon-equities_bars",
            "finnhub_options": "edge-finnhub-options_chains",
            "fred_treasury": "edge-fred-macro_treasury",
        }
        return mapping.get(task_name, f"edge-{task_name}")

    def _acquire_task_lease(self, task_name: str) -> bool:
        if not self.lease_mgr:
            return True
        return self._acquire_lease(self._lease_name_for_task(task_name))

    def _release_task_lease(self, task_name: str):
        if not self.lease_mgr:
            return
        try:
            self.lease_mgr.release(self._lease_name_for_task(task_name))
        except Exception:
            pass

    def _schedule_fanout(self, tasks: List[str]):
        """Fan out across sources with dynamic consolidation as vendors pause.

        Strategy:
        - Build per-task unit generators (small work items)
        - Use a single global threadpool of size NODE_WORKERS
        - Start with one inflight per vendor; when a vendor pauses (rate limit/no work), slots go to others
        """
        # Prepare budget manager
        budget = self._init_budget()

        # Acquire leases per task upfront; start renew threads
        renew_threads = {}
        for t in tasks:
            if not self._acquire_task_lease(t):
                logger.warning(f"Lease held for task {t}; skipping")
                tasks = [x for x in tasks if x != t]
        if self.lease_mgr:
            renew_every = int(self.config.get("locks", {}).get("renew_seconds", 45))
            for t in tasks:
                th = threading.Thread(target=self._renew_lease_loop, args=(self._lease_name_for_task(t), renew_every), daemon=True)
                th.start()
                renew_threads[t] = th

        # Build unit producers
        symbols = self._symbols_universe()
        today = datetime.now().date()

        def alpaca_units():
            if "alpaca" not in self.connectors:
                return
            # Resume window
            last_ts = self.bookmarks.get_last_timestamp("alpaca", "equities_bars") if self.bookmarks else None
            start_date = (datetime.fromisoformat(last_ts).date() + timedelta(days=1)) if last_ts else (today - timedelta(days=30))
            d = start_date
            while d <= today and not self.shutdown_requested:
                # Estimate tokens: ceil(len(symbols)/100)
                from math import ceil
                tokens = max(1, ceil(len(symbols) / 100))
                yield {
                    "vendor": "alpaca",
                    "desc": f"alpaca {d}",
                    "tokens": tokens,
                    "run": lambda day=d: self._run_alpaca_day(symbols, day, budget),
                }
                d += timedelta(days=1)

        def polygon_units():
            if "polygon" not in self.connectors:
                return
            last_ts = self.bookmarks.get_last_timestamp("polygon", "equities_bars") if self.bookmarks else None
            start_date = (datetime.fromisoformat(last_ts).date() + timedelta(days=1)) if last_ts else (today - timedelta(days=7))
            d = start_date
            while d <= today and not self.shutdown_requested:
                tokens = len(symbols)  # ~1 req per symbol-day
                yield {
                    "vendor": "polygon",
                    "desc": f"polygon {d}",
                    "tokens": tokens,
                    "run": lambda day=d, syms=symbols: self._run_polygon_day(syms, day, budget),
                }
                d += timedelta(days=1)

        def finnhub_units():
            if "finnhub" not in self.connectors:
                return
            # options chains best-effort for today on a subset
            per_run = max(1, int(os.getenv("NODE_FINNHUB_UL_PER_UNIT", "5")))
            for i in range(0, min(len(symbols), per_run)):
                sym = symbols[i]
                yield {
                    "vendor": "finnhub",
                    "desc": f"finnhub options {sym}",
                    "tokens": 1,
                    "run": lambda s=sym: self._run_finnhub_options_underlier(s, today, budget),
                }

        def fred_units():
            if "fred" not in self.connectors:
                return
            # One unit: today's curve
            yield {
                "vendor": "fred",
                "desc": f"fred treasury {today}",
                "tokens": 1,
                "run": lambda day=today: self._run_fred_treasury_day(day, budget),
            }

        producers = {}
        for t in tasks:
            if t == "alpaca_bars":
                producers[t] = alpaca_units()
            elif t == "polygon_bars":
                producers[t] = polygon_units()
            elif t == "finnhub_options":
                producers[t] = finnhub_units()
            elif t == "fred_treasury":
                producers[t] = fred_units()

        # Runnable loop
        from itertools import cycle
        task_order = [t for t in tasks if t in producers]
        if not task_order:
            logger.info("No runnable tasks (connectors missing?)")
            # Release leases
            for t in renew_threads:
                try:
                    self.shutdown_requested = True
                    renew_threads[t].join(timeout=1)
                except Exception:
                    pass
            for t in tasks:
                self._release_task_lease(t)
            return

        max_workers = worker_count(default=4)
        logger.info(f"Starting fan-out across tasks: {task_order} with {max_workers} workers")

        vendor_freeze = {}  # vendor -> until_ts
        active = {}
        results_ok = 0
        submitted = 0

        def can_run(vendor: str, tokens: int) -> bool:
            # Check cooldown
            from time import time as _now
            until = vendor_freeze.get(vendor)
            if until and _now() < until:
                return False
            # Check daily budget
            if budget and tokens > 0:
                return budget.try_consume(vendor, tokens)
            return True

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            rr = cycle(task_order)

            # Seed one unit per task if possible
            for t in task_order:
                it = producers.get(t)
                if not it:
                    continue
                try:
                    unit = next(it)
                except StopIteration:
                    continue
                if can_run(unit["vendor"], unit.get("tokens", 1)):
                    fut = ex.submit(unit["run"])
                    active[fut] = (t, unit)
                    submitted += 1

            while active and not self.shutdown_requested:
                # As futures complete, schedule replacements
                done = next(as_completed(list(active.keys())))
                t, unit = active.pop(done)
                status, rows, msg = ("error", 0, "")
                try:
                    res = done.result()
                    if isinstance(res, tuple) and len(res) == 3:
                        status, rows, msg = res
                    elif isinstance(res, int):
                        status, rows, msg = ("ok", res, "")
                    else:
                        status, rows, msg = ("ok", 0, "")
                except Exception as e:
                    emsg = str(e)
                    # crude 429 detection
                    if "429" in emsg or "rate" in emsg.lower():
                        from time import time as _now
                        vendor_freeze[unit["vendor"]] = _now() + 60
                        status = "ratelimited"
                        msg = emsg
                    else:
                        status = "error"
                        msg = emsg

                if status == "ok":
                    results_ok += 1
                elif status == "ratelimited":
                    logger.warning(f"Vendor {unit['vendor']} rate-limited, cooling off")
                elif status == "empty":
                    pass
                elif status == "error":
                    logger.warning(f"Unit error [{unit['desc']}]: {msg}")

                # Attempt to schedule next unit from any task, prefer same task to preserve order
                scheduled = False
                for _ in range(len(task_order)):
                    tn = next(rr)
                    it = producers.get(tn)
                    if not it:
                        continue
                    try:
                        nxt = next(it)
                    except StopIteration:
                        producers[tn] = None
                        continue
                    if can_run(nxt["vendor"], nxt.get("tokens", 1)):
                        fut = ex.submit(nxt["run"])
                        active[fut] = (tn, nxt)
                        submitted += 1
                        scheduled = True
                        break

                if not scheduled and not active:
                    # Try a final pass to pick any remaining units
                    for tn, it in list(producers.items()):
                        if not it:
                            continue
                        try:
                            nxt = next(it)
                        except StopIteration:
                            producers[tn] = None
                            continue
                        if can_run(nxt["vendor"], nxt.get("tokens", 1)):
                            fut = ex.submit(nxt["run"])
                            active[fut] = (tn, nxt)
                            submitted += 1
                            break

        logger.info(f"Fan-out complete. Submitted units: {submitted}, ok: {results_ok}")

        # Cleanup leases
        for t in renew_threads:
            try:
                self.shutdown_requested = True
                renew_threads[t].join(timeout=1)
            except Exception:
                pass
        for t in tasks:
            self._release_task_lease(t)

    # ----- Unit runners for each source -----

    def _run_alpaca_day(self, symbols: List[str], day: datetime.date, budget: Optional[BudgetManager]):
        connector = self.connectors.get("alpaca")
        if not connector:
            return ("empty", 0, "no connector")
        df = self._alpaca_fetch_day_parallel(connector, symbols, day.isoformat(), inflight_override=1)
        if df.empty:
            return ("empty", 0, "no data")
        self._upload_to_s3(df, "alpaca", "equities_bars", day.isoformat())
        self._write_manifest("alpaca", "equities_bars", day.isoformat(), len(df))
        if self.bookmarks:
            self.bookmarks.set("alpaca", "equities_bars", day.isoformat(), len(df))
        return ("ok", int(len(df)), "")

    def _run_polygon_day(self, symbols: List[str], day: datetime.date, budget: Optional[BudgetManager]):
        try:
            from data_layer.connectors.polygon_connector import PolygonConnector
        except Exception:
            return ("empty", 0, "no connector")
        conn: Optional[PolygonConnector] = self.connectors.get("polygon")  # type: ignore
        if not conn:
            return ("empty", 0, "no connector")
        rows = []
        for sym in symbols:
            try:
                df = conn.fetch_aggregates(sym, day, day, timespan="day")
            except Exception as e:
                emsg = str(e)
                if "429" in emsg or "rate" in emsg.lower():
                    return ("ratelimited", 0, emsg)
                df = pd.DataFrame()
            if not df.empty:
                rows.append(df)
        if not rows:
            return ("empty", 0, "no data")
        out = pd.concat(rows, ignore_index=True)
        self._upload_to_s3(out, "polygon", "equities_bars", day.isoformat())
        self._write_manifest("polygon", "equities_bars", day.isoformat(), len(out))
        if self.bookmarks:
            # bookmark advances per day for polygon bars
            self.bookmarks.set("polygon", "equities_bars", day.isoformat(), len(out))
        return ("ok", int(len(out)), "")

    def _run_finnhub_options_underlier(self, symbol: str, day: datetime.date, budget: Optional[BudgetManager]):
        try:
            from data_layer.connectors.finnhub_connector import FinnhubConnector
        except Exception:
            return ("empty", 0, "no connector")
        conn: Optional[FinnhubConnector] = self.connectors.get("finnhub")  # type: ignore
        if not conn:
            return ("empty", 0, "no connector")
        try:
            df = conn.fetch_options_chain(symbol)
        except Exception as e:
            emsg = str(e)
            if "429" in emsg or "rate" in emsg.lower():
                return ("ratelimited", 0, emsg)
            return ("error", 0, emsg)
        if df.empty:
            return ("empty", 0, "no data")
        # Partition by underlier
        if self.s3:
            import io
            buf = io.BytesIO()
            df.to_parquet(buf, index=False)
            self.s3.put_object(f"raw/finnhub/options_chains/date={day.isoformat()}/underlier={symbol}/data.parquet", buf.getvalue())
        else:
            out = Path("data_layer/raw/finnhub/options_chains") / f"date={day.isoformat()}" / f"underlier={symbol}"
            out.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out / "data.parquet", index=False)
        if self.bookmarks:
            self.bookmarks.set("finnhub", "options_chains", day.isoformat(), int(len(df)))
        return ("ok", int(len(df)), "")

    def _run_fred_treasury_day(self, day: datetime.date, budget: Optional[BudgetManager]):
        try:
            from data_layer.connectors.fred_connector import FREDConnector
        except Exception:
            return ("empty", 0, "no connector")
        conn: Optional[FREDConnector] = self.connectors.get("fred")  # type: ignore
        if not conn:
            return ("empty", 0, "no connector")
        try:
            df = conn.fetch_treasury_curve(start_date=day, end_date=day)
        except Exception as e:
            emsg = str(e)
            if "429" in emsg or "rate" in emsg.lower():
                return ("ratelimited", 0, emsg)
            return ("error", 0, emsg)
        if df.empty:
            return ("empty", 0, "no data")
        # Write per date
        if self.s3:
            import io
            buf = io.BytesIO()
            df.to_parquet(buf, index=False)
            self.s3.put_object(f"raw/fred/macro_treasury/date={day.isoformat()}/data.parquet", buf.getvalue())
        else:
            out = Path("data_layer/raw/macro_treasury/fred") / f"date={day.isoformat()}"
            out.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out / "data.parquet", index=False)
        if self.bookmarks:
            self.bookmarks.set("fred", "macro_treasury", day.isoformat(), int(len(df)))
        return ("ok", int(len(df)), "")

    def run(self):
        """Run edge collector."""
        logger.info("Edge collector starting...")

        # Run collection tasks with fan-out scheduler
        tasks = self.config.get("tasks", ["alpaca_bars"])
        self._schedule_fanout(tasks)

        logger.info("Edge collector stopped")


def main():
    parser = argparse.ArgumentParser(description="Edge data collector")
    parser.add_argument(
        "--config",
        default="configs/edge.yml",
        help="Path to config file",
    )
    args = parser.parse_args()

    load_dotenv()
    collector = EdgeCollector(args.config)
    collector.run()


if __name__ == "__main__":
    main()
