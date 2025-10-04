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
from datetime import datetime, timedelta, date
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
from utils.s3_writer import S3Writer
from utils.bad_symbols import BadSymbolCache


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
            # Initialize a background S3 writer to serialize heavy writes
            try:
                self.s3_writer = S3Writer(self.s3)
                self.s3_writer.start()
            except Exception as e:
                logger.warning(f"Failed to start S3 writer: {e}")
                self.s3_writer = None
            # Persistent bad-symbol cache
            try:
                self.bad_symbols = BadSymbolCache(self.s3)
            except Exception as e:
                logger.warning(f"BadSymbolCache init failed (S3): {e}")
                self.bad_symbols = BadSymbolCache(None)
        else:
            self.s3 = None
            self.lease_mgr = None
            self.bookmarks = None
            self.s3_writer = None
            # Local bad-symbol cache
            self.bad_symbols = BadSymbolCache(None)

        # In-memory cache (session) in addition to persisted bad-symbols
        self._vendor_bad_symbols = {"polygon": set(), "alpaca": set(), "finnhub": set(), "fred": set()}

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
        """Background lease renewal (should run in thread).

        More robust behavior: retry on transient failures and attempt re-acquire
        if the lease vanished or expired, before exiting to avoid split-brain.
        """
        failures = 0
        while not self.shutdown_requested:
            time.sleep(interval)
            try:
                if self.lease_mgr.renew(name):
                    failures = 0
                    continue
            except Exception as e:
                logger.warning(f"Lease renew threw: {e}")

            failures += 1
            # Try to re-acquire if lease disappeared/expired
            try:
                holder = self.lease_mgr.get_holder(name)
            except Exception:
                holder = None
            if holder is None:
                try:
                    if self.lease_mgr.acquire(name, force=True):
                        logger.warning("Lease re-acquired after renewal failure")
                        failures = 0
                        continue
                except Exception as e:
                    logger.warning(f"Lease re-acquire failed: {e}")

            if failures >= 2:
                logger.error("Lease renewal failed repeatedly; exiting")
                self.shutdown_requested = True
                break
            # brief backoff before next attempt to avoid tight loop
            try:
                time.sleep(max(1, interval // 3))
            except Exception:
                pass

    def _upload_to_s3(self, df: pd.DataFrame, source: str, table: str, date: str, async_write: bool = False):
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
        if self.s3_writer and async_write:
            fut = self.s3_writer.submit_df_parquet(key, df)
            return fut
        else:
            # Synchronous path (fallback)
            fut = self.s3_writer.submit_df_parquet(key, df) if self.s3_writer else None
            if fut is not None:
                fut.result()
                return
            import io
            buffer = io.BytesIO()
            df.to_parquet(buffer, index=False)
            data = buffer.getvalue()
            temp_key = f"{key}.tmp"
            self.s3.put_object(temp_key, data)
            if self.s3.object_exists(key):
                logger.debug(f"Object already exists, skipping: {key}")
                self.s3.delete_object(temp_key)
                return
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

    def _upload_parquet_key(self, df: pd.DataFrame, key: str, async_write: bool = False):
        if not self.s3:
            # Not used for local; call site should handle local path
            return None
        if self.s3_writer and async_write:
            return self.s3_writer.submit_df_parquet(key, df)
        else:
            fut = self.s3_writer.submit_df_parquet(key, df) if self.s3_writer else None
            if fut is not None:
                fut.result()
                return None
            import io
            buffer = io.BytesIO()
            df.to_parquet(buffer, index=False)
            data = buffer.getvalue()
            temp_key = f"{key}.tmp"
            self.s3.put_object(temp_key, data)
            if self.s3.object_exists(key):
                logger.debug(f"Object already exists, skipping: {key}")
                self.s3.delete_object(temp_key)
                return None
            self.s3.put_object(key, data)
            self.s3.delete_object(temp_key)
            logger.info(f"Uploaded to S3: {key} ({len(df)} rows, {len(data)} bytes)")
            return None

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

    def _should_fetch_eod_for_day(self, vendor: str, day: date) -> bool:
        """Gate EOD/day timeframe fetches for 'today' unless past cutoff or explicitly allowed.

        Env:
          - EOD_FETCH_TODAY=true|false (default false)
          - EOD_TODAY_CUTOFF_UTC_HOUR=int hour (default 22)
        """
        if day != datetime.utcnow().date():
            return True
        allow_today = os.getenv("EOD_FETCH_TODAY", "false").lower() == "true"
        if allow_today:
            cutoff = 22
            try:
                cutoff = int(os.getenv("EOD_TODAY_CUTOFF_UTC_HOUR", "22"))
            except Exception:
                cutoff = 22
            return datetime.utcnow().hour >= cutoff
        return False

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
        renew_threads: dict[str, threading.Thread] = {}
        for t in list(tasks):
            if not self._acquire_task_lease(t):
                logger.warning(f"Lease held for task {t}; skipping")
                tasks = [x for x in tasks if x != t]
        
            if self.lease_mgr:
                renew_every = int(self.config.get("locks", {}).get("renew_seconds", 45))
                for t in tasks:
                    th = threading.Thread(
                        target=self._renew_lease_loop,
                        args=(self._lease_name_for_task(t), renew_every),
                        daemon=True,
                    )
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
                # Gate 'today' until cutoff
                if not self._should_fetch_eod_for_day("alpaca", d):
                    d += timedelta(days=1)
                    continue
                # Estimate tokens: ceil(len(symbols)/100)
                from math import ceil
                # Skip persisted bad symbols for alpaca (best-effort; Alpaca batch drops invalids silently)
                alp_bad = set([s.upper() for s in self._vendor_bad_symbols.get("alpaca", set())]) | self.bad_symbols.vendor_set("alpaca")
                syms = [s for s in symbols if s.upper() not in alp_bad]
                tokens = max(1, ceil(len(syms) / 100))
                yield {
                    "vendor": "alpaca",
                    "desc": f"alpaca {d}",
                    "tokens": tokens,
                    "run": lambda day=d, syms=syms: self._run_alpaca_day(syms, day, budget),
                }
                d += timedelta(days=1)

        def polygon_units():
            if "polygon" not in self.connectors:
                return
            last_ts = self.bookmarks.get_last_timestamp("polygon", "equities_bars") if self.bookmarks else None
            start_date = (datetime.fromisoformat(last_ts).date() + timedelta(days=1)) if last_ts else (today - timedelta(days=7))
            BATCH = max(1, int(os.getenv("NODE_POLYGON_SYMBOLS_PER_UNIT", "10")))
            d = start_date
            while d <= today and not self.shutdown_requested:
                if not self._should_fetch_eod_for_day("polygon", d):
                    d += timedelta(days=1)
                    continue
                # Skip persisted bad symbols
                pg_bad = set([s.upper() for s in self._vendor_bad_symbols.get("polygon", set())]) | self.bad_symbols.vendor_set("polygon")
                val_syms = [s for s in symbols if s.upper() not in pg_bad]
                for i in range(0, len(symbols), BATCH):
                    chunk_all = symbols[i:i+BATCH]
                    chunk = [s for s in chunk_all if s.upper() not in pg_bad]
                    tokens = len(chunk)  # ~1 req per symbol-day
                    yield {
                        "vendor": "polygon",
                        "desc": f"polygon {d} [{i}:{i+len(chunk)}]",
                        "tokens": tokens,
                        "run": lambda day=d, syms=chunk: self._run_polygon_day(syms, day, budget),
                    }
                d += timedelta(days=1)

        def finnhub_units():
            if "finnhub" not in self.connectors:
                return
            # options chains best-effort for today on a subset
            per_run = max(1, int(os.getenv("NODE_FINNHUB_UL_PER_UNIT", "5")))
            fh_bad = set([s.upper() for s in self._vendor_bad_symbols.get("finnhub", set())]) | self.bad_symbols.vendor_set("finnhub")
            picked = 0
            for sym in symbols:
                if picked >= per_run:
                    break
                if sym.upper() in fh_bad:
                    continue
                picked += 1
                yield {
                    "vendor": "finnhub",
                    "desc": f"finnhub options {sym}",
                    "tokens": 1,
                    "run": lambda s=sym: self._run_finnhub_options_underlier(s, today, budget),
                }

        def fred_units():
            if "fred" not in self.connectors:
                return
            # Iterate from bookmark → gated end_day (yesterday before cutoff, else today)
            last_ts = self.bookmarks.get_last_timestamp("fred", "macro_treasury") if self.bookmarks else None
            start_date = (datetime.fromisoformat(last_ts).date() + timedelta(days=1)) if last_ts else (today - timedelta(days=7))
            d = start_date
            while d <= today and not self.shutdown_requested:
                if not self._should_fetch_eod_for_day("fred", d):
                    d += timedelta(days=1)
                    continue
                yield {
                    "vendor": "fred",
                    "desc": f"fred treasury {d}",
                    "tokens": 1,
                    "run": lambda day=d: self._run_fred_treasury_day(day, budget),
                }
                d += timedelta(days=1)

        try:
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
                return

            max_workers = worker_count(default=4)
            logger.info(f"Starting fan-out across tasks: {task_order} with {max_workers} workers")

            vendor_freeze: dict[str, float] = {}  # vendor -> until_ts
            vendor_inflight: dict[str, int] = {}
            active: dict = {}
            results_ok = 0
            submitted = 0

            def vendor_cap(vendor: str) -> int:
                # Default per-vendor inflight: conservative (1), Alpaca allows override
                default_caps = {"alpaca": 1, "polygon": 1, "finnhub": 1, "fred": 1}
                return max(1, max_inflight_for(vendor, default=default_caps.get(vendor, 1)))

            def can_run(vendor: str, tokens: int) -> bool:
                # Check cooldown
                from time import time as _now
                until = vendor_freeze.get(vendor)
                if until and _now() < until:
                    return False
                # Check inflight cap
                if vendor_inflight.get(vendor, 0) >= vendor_cap(vendor):
                    return False
                # Check daily budget
                if budget and tokens > 0:
                    return budget.try_consume(vendor, tokens)
                return True

            # Executor (non-blocking shutdown to avoid hangs on stuck futures)
            ex = ThreadPoolExecutor(max_workers=max_workers)
            rr = cycle(task_order)

            # Seed up to max_workers with RR across tasks, honoring vendor caps
            import random, time as _t
            while len(active) < max_workers and not self.shutdown_requested:
                progressed = False
                for _ in range(len(task_order)):
                    t = next(rr)
                    it = producers.get(t)
                    if not it:
                        continue
                    try:
                        unit = next(it)
                    except StopIteration:
                        producers[t] = None
                        continue
                    v = unit["vendor"]
                    if can_run(v, unit.get("tokens", 1)):
                        fut = ex.submit(unit["run"])
                        active[fut] = (t, unit)
                        vendor_inflight[v] = vendor_inflight.get(v, 0) + 1
                        submitted += 1
                        progressed = True
                        # Small jitter to de-synchronize vendor calls
                        _t.sleep(random.uniform(0.05, 0.15))
                        if len(active) >= max_workers:
                            break
                if not progressed:
                    # Log why nothing was scheduled to aid debugging
                    if not active:
                        try:
                            from time import time as _now
                            caps = {
                                v: f"{vendor_inflight.get(v,0)}/{vendor_cap(v)}"
                                for v in ("alpaca", "polygon", "finnhub", "fred")
                            }
                            freezes = {
                                v: int(max(0, vendor_freeze.get(v, 0) - _now()))
                                for v in ("alpaca", "polygon", "finnhub", "fred")
                            }
                            budgets = {
                                v: (budget.remaining(v) if budget else None)
                                for v in ("alpaca", "polygon", "finnhub", "fred")
                            }
                            logger.warning(
                                f"No units scheduled: caps={caps}, freeze_s={freezes}, budget_remaining={budgets}"
                            )
                        except Exception:
                            pass
                    break

            from concurrent.futures import wait, FIRST_COMPLETED
            max_wait = int(os.getenv("EDGE_SCHEDULER_WAIT_TIMEOUT_SECONDS", "60"))

            while active and not self.shutdown_requested:
                    # As futures complete, schedule replacements (with timeout to avoid deadlock)
                    try:
                        done_set, _ = wait(list(active.keys()), timeout=max_wait, return_when=FIRST_COMPLETED)
                        if not done_set:
                            logger.warning(f"No unit finished within {max_wait}s; breaking scheduler loop (active={len(active)})")
                            break
                        done = next(iter(done_set))
                    except Exception:
                        logger.warning("Scheduler wait failed; breaking loop")
                        break

                    t, unit = active.pop(done)
                    v = unit["vendor"]
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
                    # Decrement inflight
                    if v:
                        vendor_inflight[v] = max(0, vendor_inflight.get(v, 0) - 1)

                    # Attempt to schedule next unit from any task, prefer same task to preserve order
                    scheduled = False
                    import random, time as _t
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
                        v2 = nxt["vendor"]
                        if can_run(v2, nxt.get("tokens", 1)):
                            fut = ex.submit(nxt["run"])
                            active[fut] = (tn, nxt)
                            vendor_inflight[v2] = vendor_inflight.get(v2, 0) + 1
                            submitted += 1
                            scheduled = True
                            _t.sleep(random.uniform(0.03, 0.12))
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
                            v3 = nxt["vendor"]
                            if can_run(v3, nxt.get("tokens", 1)):
                                fut = ex.submit(nxt["run"])
                                active[fut] = (tn, nxt)
                                vendor_inflight[v3] = vendor_inflight.get(v3, 0) + 1
                                submitted += 1
                                break

            logger.info(f"Fan-out complete. Submitted units: {submitted}, ok: {results_ok}")

        finally:
            try:
                # Attempt to stop executor without waiting for potentially stuck futures
                ex.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            # Ensure leases/threads are always cleaned up
            for t, th in list(renew_threads.items()):
                try:
                    self.shutdown_requested = True
                    th.join(timeout=1)
                except Exception:
                    pass
            for t in tasks:
                try:
                    self._release_task_lease(t)
                except Exception:
                    pass

    # ----- Unit runners for each source -----

    def _run_alpaca_day(self, symbols: List[str], day: date, budget: Optional[BudgetManager]):
        connector = self.connectors.get("alpaca")
        if not connector:
            return ("empty", 0, "no connector")
        df = self._alpaca_fetch_day_parallel(connector, symbols, day.isoformat(), inflight_override=1)
        if df.empty:
            return ("empty", 0, "no data")
        fut = self._upload_to_s3(df, "alpaca", "equities_bars", day.isoformat(), async_write=True)
        try:
            # Ensure persisted before manifest/bookmark
            if fut is not None:
                fut.result()
        except Exception as e:
            return ("error", 0, str(e))
        self._write_manifest("alpaca", "equities_bars", day.isoformat(), len(df))
        if self.bookmarks:
            self.bookmarks.set("alpaca", "equities_bars", day.isoformat(), len(df))
        return ("ok", int(len(df)), "")

    def _run_polygon_day(self, symbols: List[str], day: date, budget: Optional[BudgetManager]):
        try:
            from data_layer.connectors.polygon_connector import PolygonConnector
        except Exception:
            return ("empty", 0, "no connector")
        conn: Optional[PolygonConnector] = self.connectors.get("polygon")  # type: ignore
        if not conn:
            return ("empty", 0, "no connector")
        rows = []
        for sym in symbols:
            if sym in self._vendor_bad_symbols.get("polygon", set()):
                continue
            try:
                df = conn.fetch_aggregates(sym, day, day, timespan="day")
            except Exception as e:
                emsg = str(e)
                if "429" in emsg or "rate" in emsg.lower():
                    return ("ratelimited", 0, emsg)
                # Likely invalid/unsupported ticker: suppress further attempts this run
                if any(s in emsg.lower() for s in ["invalid", "not found", "ticker"]):
                    try:
                        self._vendor_bad_symbols.setdefault("polygon", set()).add(sym)
                        try:
                            # two-strike persist within 24h
                            self.bad_symbols.strike("polygon", sym, reason="invalid symbol")
                        except Exception:
                            pass
                    except Exception:
                        pass
                df = pd.DataFrame()
            if not df.empty:
                rows.append(df)
        if not rows:
            return ("empty", 0, "no data")
        out = pd.concat(rows, ignore_index=True)
        fut = self._upload_to_s3(out, "polygon", "equities_bars", day.isoformat(), async_write=True)
        try:
            if fut is not None:
                fut.result()
        except Exception as e:
            return ("error", 0, str(e))
        self._write_manifest("polygon", "equities_bars", day.isoformat(), len(out))
        if self.bookmarks:
            # bookmark advances per day for polygon bars
            self.bookmarks.set("polygon", "equities_bars", day.isoformat(), len(out))
        return ("ok", int(len(out)), "")

    def _run_finnhub_options_underlier(self, symbol: str, day: date, budget: Optional[BudgetManager]):
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
            # Treat symbol not found as invalid
            if any(s in emsg.lower() for s in ["invalid", "not found", "ticker"]):
                try:
                    self._vendor_bad_symbols.setdefault("finnhub", set()).add(symbol)
                    # two-strike persist within 24h
                    self.bad_symbols.strike("finnhub", symbol, reason="invalid symbol")
                except Exception:
                    pass
            return ("error", 0, emsg)
        if df.empty:
            return ("empty", 0, "no data")
        # Partition by underlier
        if self.s3:
            key = f"raw/finnhub/options_chains/date={day.isoformat()}/underlier={symbol}/data.parquet"
            fut = self._upload_parquet_key(df, key, async_write=True)
            try:
                if fut is not None:
                    fut.result()
            except Exception as e:
                return ("error", 0, str(e))
        else:
            out = Path("data_layer/raw/finnhub/options_chains") / f"date={day.isoformat()}" / f"underlier={symbol}"
            out.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out / "data.parquet", index=False)
        if self.bookmarks:
            self.bookmarks.set("finnhub", "options_chains", day.isoformat(), int(len(df)))
        return ("ok", int(len(df)), "")

    def _run_fred_treasury_day(self, day: date, budget: Optional[BudgetManager]):
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
            key = f"raw/fred/macro_treasury/date={day.isoformat()}/data.parquet"
            fut = self._upload_parquet_key(df, key, async_write=True)
            try:
                if fut is not None:
                    fut.result()
            except Exception as e:
                return ("error", 0, str(e))
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
        try:
            self._schedule_fanout(tasks)
        finally:
            # Ensure S3 writer stops cleanly
            try:
                if self.s3_writer:
                    self.s3_writer.stop()
            except Exception:
                pass

        logger.info("Edge collector stopped")


def main():
    parser = argparse.ArgumentParser(description="Edge data collector")
    parser.add_argument(
        "--config",
        default="configs/edge.yml",
        help="Path to config file",
    )
    args = parser.parse_args()

    # Ensure .env overrides any ambient AWS_* vars
    load_dotenv(override=True)
    collector = EdgeCollector(args.config)
    collector.run()


if __name__ == "__main__":
    main()
