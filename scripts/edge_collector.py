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
from typing import Optional, List, Dict
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
from data_layer.connectors.polygon_connector import PolygonConnector
from data_layer.connectors.finnhub_connector import FinnhubConnector
from data_layer.connectors.fred_connector import FREDConnector
from data_layer.connectors.alpha_vantage_connector import AlphaVantageConnector
from data_layer.connectors.fmp_connector import FMPConnector
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

        # Storage backend and data root
        self.storage_backend = os.getenv("STORAGE_BACKEND", "local")
        self.parquet_compression = os.getenv("PARQUET_COMPRESSION", "zstd")
        self.data_root = Path(os.getenv("DATA_ROOT", Path(__file__).resolve().parents[1] / "data"))
        logger.info(f"Storage backend: {self.storage_backend} (data_root={self.data_root})")

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
            # Local bookmarks on SSD
            self.bookmarks = BookmarkManager(
                s3_client=None,
                local_path=os.getenv("BOOKMARKS_PATH", str(self.data_root / "data_layer" / "manifests" / "bookmarks.json")),
            )

        # In-memory cache (session) in addition to persisted bad-symbols
        self._vendor_bad_symbols = {
            "polygon": set(),
            "alpaca": set(),
            "finnhub": set(),
            "fred": set(),
            "av": set(),
            "fmp": set(),
        }

        self._vendor_symbol_cursor: Dict[str, int] = {}

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

        if os.getenv("POLYGON_API_KEY"):
            try:
                connectors["polygon"] = PolygonConnector(api_key=os.getenv("POLYGON_API_KEY"))
                logger.info("Initialized Polygon connector")
            except Exception as e:
                logger.warning(f"Polygon connector init failed: {e}")

        if os.getenv("FINNHUB_API_KEY"):
            try:
                connectors["finnhub"] = FinnhubConnector(api_key=os.getenv("FINNHUB_API_KEY"))
                logger.info("Initialized Finnhub connector")
            except Exception as e:
                logger.warning(f"Finnhub connector init failed: {e}")

        if os.getenv("FRED_API_KEY"):
            try:
                connectors["fred"] = FREDConnector(api_key=os.getenv("FRED_API_KEY"))
                logger.info("Initialized FRED connector")
            except Exception as e:
                logger.warning(f"FRED connector init failed: {e}")

        if os.getenv("ALPHA_VANTAGE_API_KEY"):
            try:
                connectors["av"] = AlphaVantageConnector(api_key=os.getenv("ALPHA_VANTAGE_API_KEY"))
                logger.info("Initialized Alpha Vantage connector")
            except Exception as e:
                logger.warning(f"Alpha Vantage connector init failed: {e}")

        if os.getenv("FMP_API_KEY"):
            try:
                connectors["fmp"] = FMPConnector(api_key=os.getenv("FMP_API_KEY"))
                logger.info("Initialized FMP connector")
            except Exception as e:
                logger.warning(f"FMP connector init failed: {e}")

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
            local_dir = self.data_root / "data_layer" / "raw" / source / table / f"date={date}"
            local_dir.mkdir(parents=True, exist_ok=True)
            local_file = local_dir / "data.parquet"
            try:
                df.to_parquet(local_file, index=False, compression=self.parquet_compression)
            except Exception:
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
        try:
            df.to_parquet(buffer, index=False, compression=self.parquet_compression)
        except Exception:
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
        manifest_key = f"manifests/{date}/manifest-{source}-{table}.jsonl"
        manifest_entry = {
            "source": source,
            "table": table,
            "date": date,
            "row_count": row_count,
            "timestamp": datetime.utcnow().isoformat(),
        }

        if not self.s3:
            # Local append on SSD
            out_path = self.data_root / "data_layer" / manifest_key
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(manifest_entry) + "\n")
            logger.debug(f"Updated manifest (local): {out_path}")
            return

        # Append to manifest (read, append, write)
        import json
        try:
            existing_data, etag = self.s3.get_object(manifest_key)
            lines = existing_data.decode('utf-8').strip().split('\n')
        except Exception:
            lines = []

        lines.append(json.dumps(manifest_entry))
        new_data = '\n'.join(lines).encode('utf-8')
        self.s3.put_object(manifest_key, new_data)
        logger.debug(f"Updated manifest: {manifest_key}")

    def _upload_parquet_key(self, df: pd.DataFrame, key: str, async_write: bool = False):
        """Upload a DataFrame to S3 under an explicit key.

        Returns a Future if async_write and an S3Writer is available; otherwise
        performs a blocking upload and returns None.
        """
        if not self.s3:
            # Not used for local; call site should handle local path
            return None
        if self.s3_writer and async_write:
            return self.s3_writer.submit_df_parquet(key, df)
        if self.s3_writer and not async_write:
            fut = self.s3_writer.submit_df_parquet(key, df)
            fut.result()
            return None
        # Fallback: synchronous upload without S3Writer
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

    def _wait_future_with_logs(self, fut, desc: str, check_seconds: int = 30):
        """Wait on a Future with periodic progress logs to surface stalls."""
        try:
            import time as _time
            import concurrent.futures as _fut
            waited = 0
            while not self.shutdown_requested:
                try:
                    return fut.result(timeout=check_seconds)
                except _fut.TimeoutError:
                    waited += check_seconds
                    logger.warning(f"Waiting on async task: {desc} (waited {waited}s)")
                    continue
        except Exception:
            # Fall back to blocking wait if anything goes wrong with timed waits
            return fut.result()

    def _alpaca_fetch_day_parallel(self, connector: AlpacaConnector, symbols: List[str], day_str: str, inflight_override: Optional[int] = None, timeframe: str = "1Day") -> pd.DataFrame:
        """Fetch a single day of bars (timeframe configurable) in parallel by symbol chunks; return combined DataFrame."""
        # Chunk symbols to match connector's batch size guidance (100)
        BATCH = 100
        chunks = [symbols[i:i + BATCH] for i in range(0, len(symbols), BATCH)]
        inflight = inflight_override if inflight_override is not None else max(1, max_inflight_for("alpaca", default=int(os.getenv("NODE_WORKERS", "4"))))

        def _fetch_chunk(chunk):
            ds = pd.to_datetime(day_str).date()
            return connector.fetch_bars(symbols=chunk, start_date=ds, end_date=ds, timeframe=timeframe)

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
                df = self._alpaca_fetch_day_parallel(connector, symbols, current_date.isoformat(), timeframe="1Day")

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

        # Alpha Vantage connector (optional)
        try:
            if os.getenv("ALPHA_VANTAGE_API_KEY"):
                from data_layer.connectors.alpha_vantage_connector import AlphaVantageConnector
                connectors["av"] = AlphaVantageConnector(api_key=os.getenv("ALPHA_VANTAGE_API_KEY"))
                logger.info("Initialized Alpha Vantage connector")
        except Exception as e:
            logger.warning(f"Alpha Vantage connector not available: {e}")

        # FMP connector (optional)
        try:
            if os.getenv("FMP_API_KEY"):
                from data_layer.connectors.fmp_connector import FMPConnector
                connectors["fmp"] = FMPConnector(api_key=os.getenv("FMP_API_KEY"))
                logger.info("Initialized FMP connector")
        except Exception as e:
            logger.warning(f"FMP connector not available: {e}")

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

    def _rotate_symbols(self, vendor: str, symbols: List[str], count: int) -> List[str]:
        if not symbols or count <= 0:
            return []
        cursor = self._vendor_symbol_cursor.get(vendor, 0)
        picked: List[str] = []
        n = len(symbols)
        for _ in range(min(count, n)):
            picked.append(symbols[cursor % n])
            cursor += 1
        self._vendor_symbol_cursor[vendor] = cursor % n
        return picked

    def _lease_name_for_task(self, task_name: str) -> str:
        mapping = {
            "alpaca_bars": "edge-alpaca-equities_bars",
            "alpaca_minute": "edge-alpaca-equities_bars_minute",
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
        # Work on a copy to safely mutate the list
        tasks = list(tasks)
        acquired: List[str] = []
        for t in tasks:
            if not self._acquire_task_lease(t):
                logger.warning(f"Lease held for task {t}; skipping")
                continue
            acquired.append(t)
        tasks = acquired

        # Start exactly one renew thread per acquired task
        if self.lease_mgr and tasks:
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
            # If only 'today' is in range and it's gated, optionally fall back to yesterday once
            try:
                if start_date == today and not self._should_fetch_eod_for_day("alpaca", today):
                    prev = today - timedelta(days=1)
                    # Avoid redundant fallback if we've already processed yesterday
                    if (not last_ts) or (datetime.fromisoformat(last_ts).date() < prev):
                        start_date = prev
                    else:
                        return  # no units when only gated today remains and yesterday already processed
            except Exception:
                pass
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
            try:
                if start_date == today and not self._should_fetch_eod_for_day("polygon", today):
                    prev = today - timedelta(days=1)
                    if (not last_ts) or (datetime.fromisoformat(last_ts).date() < prev):
                        start_date = prev
                    else:
                        return
            except Exception:
                pass
            # Keep polygon units under scheduler timeout given ~5 rpm free-tier
            # Default to small chunks; override via NODE_POLYGON_SYMBOLS_PER_UNIT
            BATCH = max(1, int(os.getenv("NODE_POLYGON_SYMBOLS_PER_UNIT", "3")))
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

        def alpaca_minute_units():
            if "alpaca" not in self.connectors:
                return
            # Resume from minute bookmark; else start modestly back to keep cycle time bounded
            last_ts = self.bookmarks.get_last_timestamp("alpaca", "equities_bars_minute") if self.bookmarks else None
            start_date = (datetime.fromisoformat(last_ts).date() + timedelta(days=1)) if last_ts else (today - timedelta(days=int(os.getenv("ALPACA_MINUTE_START_DAYS", "7"))))
            d = start_date
            while d <= today and not self.shutdown_requested:
                if not self._should_fetch_eod_for_day("alpaca", d):
                    d += timedelta(days=1)
                    continue
                from math import ceil
                alp_bad = set([s.upper() for s in self._vendor_bad_symbols.get("alpaca", set())]) | self.bad_symbols.vendor_set("alpaca")
                syms = [s for s in symbols if s.upper() not in alp_bad]
                tokens = max(1, ceil(len(syms) / 100))  # ~1 req per 100 symbols per day
                yield {
                    "vendor": "alpaca",
                    "desc": f"alpaca-minute {d}",
                    "tokens": tokens,
                    "run": lambda day=d, syms=syms: self._run_alpaca_minute_day(syms, day, budget),
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
                elif t == "alpaca_minute":
                    producers[t] = alpaca_minute_units()
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
            active_started: dict = {}  # fut -> (start_ts, unit_desc, vendor)
            results_ok = 0
            submitted = 0

            def vendor_cap(vendor: str) -> int:
                # Constrain polygon to 1 to avoid parallel long units under strict RPM
                default_caps = {"alpaca": 2, "polygon": 1, "finnhub": 2, "fred": 2}
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
                        logger.debug(f"Submitting unit: vendor={unit.get('vendor')} desc={unit.get('desc')} tokens={unit.get('tokens',1)}")
                        fut = ex.submit(unit["run"])
                        active[fut] = (t, unit)
                        try:
                            import time as _time
                            active_started[fut] = (_time.time(), unit.get('desc'), unit.get('vendor'))
                        except Exception:
                            pass
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
            # If set to <= 0, wait indefinitely for a unit to finish
            try:
                max_wait_val = int(os.getenv("EDGE_SCHEDULER_WAIT_TIMEOUT_SECONDS", "60"))
            except Exception:
                max_wait_val = 60
            max_wait = None if max_wait_val <= 0 else max_wait_val
            break_on_timeout = os.getenv("EDGE_SCHEDULER_BREAK_ON_TIMEOUT", "false").lower() in ("1","true","yes","on")

            while active and not self.shutdown_requested:
                    # As futures complete, schedule replacements (with timeout to avoid deadlock)
                    try:
                        done_set, _ = wait(list(active.keys()), timeout=max_wait, return_when=FIRST_COMPLETED)
                        if not done_set:
                            # Continue waiting or break (configurable), but always log what is running
                            if isinstance(max_wait, (int, float)):
                                try:
                                    import time as _time
                                    now_ts = _time.time()
                                    durations = []
                                    for fut, (st, desc, vend) in list(active_started.items()):
                                        durations.append(f"{vend}:{desc}={int(now_ts-st)}s")
                                    logger.warning(
                                        f"No unit finished within {max_wait}s; running: {', '.join(durations)} (active={len(active)})"
                                    )
                                except Exception:
                                    logger.warning(f"No unit finished within {max_wait}s; (active={len(active)})")
                            if break_on_timeout:
                                break
                            else:
                                continue
                        done = next(iter(done_set))
                    except Exception:
                        logger.warning("Scheduler wait failed; breaking loop")
                        break

                    t, unit = active.pop(done)
                    try:
                        active_started.pop(done, None)
                    except Exception:
                        pass
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
                                logger.debug(f"Submitting unit: vendor={nxt.get('vendor')} desc={nxt.get('desc')} tokens={nxt.get('tokens',1)}")
                                fut = ex.submit(nxt["run"])
                                active[fut] = (tn, nxt)
                                try:
                                    import time as _time
                                    active_started[fut] = (_time.time(), nxt.get('desc'), nxt.get('vendor'))
                                except Exception:
                                    pass
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
        logger.debug(f"_run_alpaca_day start: {day} symbols={len(symbols)}")
        df = self._alpaca_fetch_day_parallel(connector, symbols, day.isoformat(), inflight_override=1, timeframe="1Day")
        if df.empty:
            return ("empty", 0, "no data")
        fut = self._upload_to_s3(df, "alpaca", "equities_bars", day.isoformat(), async_write=True)
        try:
            # Ensure persisted before manifest/bookmark
            if fut is not None:
                self._wait_future_with_logs(fut, desc=f"alpaca {day} parquet upload")
        except Exception as e:
            return ("error", 0, str(e))
        self._write_manifest("alpaca", "equities_bars", day.isoformat(), len(df))
        if self.bookmarks:
            self.bookmarks.set("alpaca", "equities_bars", day.isoformat(), len(df))
        return ("ok", int(len(df)), "")

    def _run_alpaca_minute_day(self, symbols: List[str], day: date, budget: Optional[BudgetManager]):
        connector = self.connectors.get("alpaca")
        if not connector:
            return ("empty", 0, "no connector")
        logger.debug(f"_run_alpaca_minute_day start: {day} symbols={len(symbols)}")
        df = self._alpaca_fetch_day_parallel(connector, symbols, day.isoformat(), inflight_override=1, timeframe="1Min")
        if df.empty:
            return ("empty", 0, "no data")
        fut = self._upload_to_s3(df, "alpaca", "equities_bars_minute", day.isoformat(), async_write=True)
        try:
            if fut is not None:
                self._wait_future_with_logs(fut, desc=f"alpaca minute {day} parquet upload")
        except Exception as e:
            return ("error", 0, str(e))
        self._write_manifest("alpaca", "equities_bars_minute", day.isoformat(), len(df))
        if self.bookmarks:
            self.bookmarks.set("alpaca", "equities_bars_minute", day.isoformat(), len(df))
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
        logger.debug(f"_run_polygon_day start: {day} symbols={len(symbols)}")
        total = len(symbols)
        import time as _time
        for idx, sym in enumerate(symbols, start=1):
            if sym in self._vendor_bad_symbols.get("polygon", set()):
                continue
            try:
                t0 = _time.time()
                df = conn.fetch_aggregates(sym, day, day, timespan="day")
                logger.debug(f"polygon fetch {day} [{idx}/{total}] {sym} took {(_time.time()-t0):.2f}s rows={0 if df is None else len(df)}")
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
                self._wait_future_with_logs(fut, desc=f"polygon {day} parquet upload")
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
                    self._wait_future_with_logs(fut, desc=f"finnhub options {symbol} {day} parquet upload")
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
                    self._wait_future_with_logs(fut, desc=f"fred treasury {day} parquet upload")
            except Exception as e:
                return ("error", 0, str(e))
        else:
            out = Path("data_layer/raw/macro_treasury/fred") / f"date={day.isoformat()}"
            out.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out / "data.parquet", index=False)
        if self.bookmarks:
            self.bookmarks.set("fred", "macro_treasury", day.isoformat(), int(len(df)))
        return ("ok", int(len(df)), "")

    def _run_alpaca_options_bars_for_underlier(self, underlying: str, day: date, timeframe: str = "1Day", budget: Optional[BudgetManager] = None):
        """Fetch Alpaca option bars for a single underlier by selecting a subset of contracts from chain.

        Persists a single Parquet per date+underlier to:
          raw/alpaca/options_bars/date=YYYY-MM-DD/underlier=UNDERLIER/data.parquet
        """
        conn = self.connectors.get("alpaca")
        if not conn:
            return ("empty", 0, "no connector")
        # Determine feed for chain snapshots (opra|indicative)
        feed = os.getenv("ALPACA_OPTIONS_FEED")
        try:
            limit = max(1, int(os.getenv("ALPACA_OPTIONS_CONTRACTS_PER_UNIT", "50")))
        except Exception:
            limit = 50
        try:
            contracts = conn.fetch_option_chain_symbols(underlying, feed=feed, limit=limit)
        except Exception as e:
            emsg = str(e)
            if "429" in emsg or "rate" in emsg.lower():
                return ("ratelimited", 0, emsg)
            return ("error", 0, emsg)
        if not contracts:
            return ("empty", 0, "no contracts")
        try:
            df = conn.fetch_option_bars(contracts, start_date=day, end_date=day, timeframe=timeframe)
        except Exception as e:
            emsg = str(e)
            if "429" in emsg or "rate" in emsg.lower():
                return ("ratelimited", 0, emsg)
            return ("error", 0, emsg)
        if df.empty:
            return ("empty", 0, "no data")
        # Persist partitioned by date + underlier
        if self.s3:
            key = f"raw/alpaca/options_bars/date={day.isoformat()}/underlier={underlying}/data.parquet"
            fut = self._upload_parquet_key(df, key, async_write=True)
            try:
                if fut is not None:
                    self._wait_future_with_logs(fut, desc=f"alpaca options bars {underlying} {day} parquet upload")
            except Exception as e:
                return ("error", 0, str(e))
        else:
            out = Path("data_layer/raw/alpaca/options_bars") / f"date={day.isoformat()}" / f"underlier={underlying}"
            out.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out / "data.parquet", index=False)
        if self.bookmarks:
            self.bookmarks.set("alpaca", "options_bars", day.isoformat(), int(len(df)))
        return ("ok", int(len(df)), "")

    def _run_alpaca_options_chain_underlier(self, underlying: str, day: date, budget: Optional[BudgetManager] = None):
        """Fetch and persist Alpaca option chain snapshot for an underlier for this day.

        Stores at raw/alpaca/options_chain/date=YYYY-MM-DD/underlier=UNDERLIER/data.parquet
        """
        conn = self.connectors.get("alpaca")
        if not conn:
            return ("empty", 0, "no connector")
        feed = os.getenv("ALPACA_OPTIONS_FEED")
        try:
            limit = max(1, int(os.getenv("ALPACA_OPTIONS_CHAIN_CONTRACTS_LIMIT", "1000")))
        except Exception:
            limit = 1000
        try:
            df = conn.fetch_option_chain_snapshot_df(underlying, feed=feed, limit=limit)
        except Exception as e:
            emsg = str(e)
            if "429" in emsg or "rate" in emsg.lower():
                return ("ratelimited", 0, emsg)
            return ("error", 0, emsg)
        if df.empty:
            return ("empty", 0, "no data")
        if self.s3:
            key = f"raw/alpaca/options_chain/date={day.isoformat()}/underlier={underlying}/data.parquet"
            fut = self._upload_parquet_key(df, key, async_write=True)
            try:
                if fut is not None:
                    self._wait_future_with_logs(fut, desc=f"alpaca options chain {underlying} {day} parquet upload")
            except Exception as e:
                return ("error", 0, str(e))
        else:
            out = Path("data_layer/raw/alpaca/options_chain") / f"date={day.isoformat()}" / f"underlier={underlying}"
            out.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out / "data.parquet", index=False)
        if self.bookmarks:
            self.bookmarks.set("alpaca", "options_chain", day.isoformat(), int(len(df)))
        return ("ok", int(len(df)), "")

    def _run_alpaca_corporate_actions_day(self, day: date, budget: Optional[BudgetManager] = None):
        conn = self.connectors.get("alpaca")
        if not conn:
            return ("empty", 0, "no connector")
        try:
            df = conn.fetch_corporate_actions(day, day, symbols=None)
        except Exception as e:
            emsg = str(e)
            if "429" in emsg or "rate" in emsg.lower():
                return ("ratelimited", 0, emsg)
            return ("error", 0, emsg)
        if df.empty:
            return ("empty", 0, "no data")
        if self.s3:
            key = f"raw/alpaca/corporate_actions/date={day.isoformat()}/data.parquet"
            fut = self._upload_parquet_key(df, key, async_write=True)
            try:
                if fut is not None:
                    self._wait_future_with_logs(fut, desc=f"alpaca corporate actions {day} parquet upload")
            except Exception as e:
                return ("error", 0, str(e))
        else:
            out = Path("data_layer/raw/alpaca/corporate_actions") / f"date={day.isoformat()}"
            out.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out / "data.parquet", index=False)
        if self.bookmarks:
            self.bookmarks.set("alpaca", "corporate_actions", day.isoformat(), int(len(df)))
        return ("ok", int(len(df)), "")

    def _run_finnhub_daily_day(self, symbols: List[str], day: date, budget: Optional[BudgetManager]):
        conn = self.connectors.get("finnhub")
        if not conn:
            return ("empty", 0, "no connector")
        frames = []
        for sym in symbols:
            try:
                df = conn.fetch_candle_daily(sym, day, day)
            except Exception as e:
                emsg = str(e)
                if "429" in emsg or "rate" in emsg.lower():
                    return ("ratelimited", 0, emsg)
                self._vendor_bad_symbols.setdefault("finnhub", set()).add(sym)
                logger.warning(f"Finnhub daily fetch failed for {sym}: {e}")
                continue
            if df.empty:
                continue
            frames.append(df)

        if not frames:
            return ("empty", 0, "no data")

        df_total = pd.concat(frames, ignore_index=True)
        if self.s3:
            key = f"raw/finnhub/equities_bars/date={day.isoformat()}/data.parquet"
            fut = self._upload_parquet_key(df_total, key, async_write=True)
            try:
                if fut is not None:
                    self._wait_future_with_logs(fut, desc=f"finnhub daily {day} parquet upload")
            except Exception as e:
                return ("error", 0, str(e))
        else:
            out = Path("data_layer/raw/finnhub/equities_bars") / f"date={day.isoformat()}"
            out.mkdir(parents=True, exist_ok=True)
            df_total.to_parquet(out / "data.parquet", index=False)
        if self.bookmarks:
            self.bookmarks.set("finnhub", "equities_eod_fn", day.isoformat(), int(len(df_total)))
            # Maintain legacy bookmark key for backward compatibility until old readers migrate
            self.bookmarks.set("finnhub", "equities_bars", day.isoformat(), int(len(df_total)))
        return ("ok", int(len(df_total)), "")

    def _run_alpha_vantage_corp_actions(self, symbols: List[str], budget: Optional[BudgetManager]):
        conn = self.connectors.get("av")
        if not conn:
            return ("empty", 0, "no connector")
        rows_written = 0
        for sym in symbols:
            try:
                df = conn.fetch_corporate_actions(sym)
            except Exception as e:
                emsg = str(e)
                if "429" in emsg or "rate" in emsg.lower():
                    return ("ratelimited", rows_written, emsg)
                logger.warning(f"Alpha Vantage corporate actions failed for {sym}: {e}")
                continue
            if df.empty:
                continue
            if self.s3:
                key = f"raw/alpha_vantage/corp_actions/symbol={sym}/data.parquet"
                try:
                    self._upload_parquet_key(df, key, async_write=False)
                except Exception as e:
                    return ("error", rows_written, str(e))
            else:
                out = Path("data_layer/raw/alpha_vantage/corp_actions") / f"symbol={sym}"
                out.mkdir(parents=True, exist_ok=True)
                df.to_parquet(out / "data.parquet", index=False)
            rows_written += len(df)
        if rows_written == 0:
            return ("empty", 0, "no data")
        return ("ok", rows_written, "")

    def _run_alpha_vantage_options_hist(self, symbol: str, expiry: Optional[str], budget: Optional[BudgetManager]):
        conn = self.connectors.get("av")
        if not conn:
            return ("empty", 0, "no connector")
        try:
            df = conn.fetch_historical_options(symbol, expiry=expiry)
        except Exception as e:
            emsg = str(e)
            if "429" in emsg or "rate" in emsg.lower():
                return ("ratelimited", 0, emsg)
            return ("error", 0, emsg)
        if df.empty:
            return ("empty", 0, "no data")
        total_rows = int(len(df))
        grouped = df.groupby(df["date"])
        for dt, sub in grouped:
            if isinstance(dt, pd.Timestamp):
                dt = dt.date()
            if self.s3:
                key = f"raw/alpha_vantage/options_iv/date={dt.isoformat()}/underlier={symbol}/data.parquet"
                try:
                    self._upload_parquet_key(sub, key, async_write=False)
                except Exception as e:
                    return ("error", total_rows, str(e))
            else:
                out = Path("data_layer/raw/alpha_vantage/options_iv") / f"date={dt.isoformat()}" / f"underlier={symbol}"
                out.mkdir(parents=True, exist_ok=True)
                sub.to_parquet(out / "data.parquet", index=False)
        return ("ok", total_rows, "")

    def _run_fmp_fundamentals(self, symbol: str, period: str, budget: Optional[BudgetManager]):
        conn = self.connectors.get("fmp")
        if not conn:
            return ("empty", 0, "no connector")
        statement_types = ("income", "balance", "cashflow")
        rows_written = 0
        for kind in statement_types:
            try:
                df = conn.fetch_statements(symbol, kind=kind, period=period)
            except Exception as e:
                emsg = str(e)
                if "429" in emsg or "rate" in emsg.lower():
                    return ("ratelimited", rows_written, emsg)
                logger.warning(f"FMP statements fetch failed for {symbol} {kind}: {e}")
                continue
            if df.empty:
                continue
            rows_written += len(df)
            if self.s3:
                key = f"raw/fmp/fundamentals/statement={kind}/period={period}/symbol={symbol}/data.parquet"
                try:
                    self._upload_parquet_key(df, key, async_write=False)
                except Exception as e:
                    return ("error", rows_written, str(e))
            else:
                out = Path("data_layer/raw/fmp/fundamentals") / f"statement={kind}" / f"period={period}" / f"symbol={symbol}"
                out.mkdir(parents=True, exist_ok=True)
                df.to_parquet(out / "data.parquet", index=False)
        if rows_written == 0:
            return ("empty", 0, "no data")
        return ("ok", rows_written, "")

    def _run_fmp_delistings(self, budget: Optional[BudgetManager] = None):
        """Fetch FMP delisted companies list for reference/delistings table."""
        conn = self.connectors.get("fmp")
        if not conn:
            return ("empty", 0, "no connector")

        try:
            df = conn.fetch_delisted_companies()
        except Exception as e:
            emsg = str(e)
            if "429" in emsg or "rate" in emsg.lower():
                return ("ratelimited", 0, emsg)
            logger.warning(f"FMP delistings fetch failed: {e}")
            return ("error", 0, emsg)

        if df.empty:
            return ("empty", 0, "no data")

        rows_written = len(df)

        # Write to reference/delistings location
        if self.s3:
            key = "reference/delistings_fmp.parquet"
            try:
                self._upload_parquet_key(df, key, async_write=False)
            except Exception as e:
                return ("error", rows_written, str(e))
        else:
            out = Path("data_layer/reference")
            out.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out / "delistings_fmp.parquet", index=False)

        # Update bookmark to prevent duplicate fetches today
        today = date.today()
        bookmark_key = f"fmp-delistings-{today.isoformat()}"
        if self.bookmarks:
            self.bookmarks.update_bookmark("fmp", bookmark_key, {"fetched_at": today.isoformat()})

        logger.info(f"FMP delistings: wrote {rows_written} records")
        return ("ok", rows_written, "")

    def run(self, scheduler_mode: Optional[str] = None):
        """Run edge collector."""
        logger.info("Edge collector starting...")

        tasks = self.config.get("tasks", ["alpaca_bars"])
        mode = (scheduler_mode or os.getenv("EDGE_SCHEDULER_MODE", "per_vendor")).lower()
        logger.info(f"Scheduler mode: {mode}")

        try:
            if mode == "per_vendor":
                try:
                    from scripts.scheduler.per_vendor import VendorSupervisor
                except Exception as e:
                    logger.warning(f"Failed to import per-vendor scheduler; falling back to legacy: {e}")
                    self._schedule_fanout(tasks)
                else:
                    sup = VendorSupervisor(self)
                    sup.run(tasks)
            else:
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
    parser.add_argument(
        "--scheduler",
        choices=["legacy", "per_vendor"],
        default=None,
        help="Scheduler mode (overrides EDGE_SCHEDULER_MODE)",
    )
    args = parser.parse_args()

    # Ensure .env overrides any ambient AWS_* vars
    load_dotenv(override=True)
    collector = EdgeCollector(args.config)
    collector.run(scheduler_mode=args.scheduler)


if __name__ == "__main__":
    main()
