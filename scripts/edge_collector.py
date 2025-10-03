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
from typing import Optional

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

    def collect_alpaca_bars(self):
        """Collect Alpaca equity bars with resume."""
        source = "alpaca"
        table = "equities_bars"
        lease_name = f"edge-{source}-{table}"

        # Acquire lease
        if not self._acquire_lease(lease_name):
            return

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

            while current_date <= end and not self.shutdown_requested:
                logger.info(f"Fetching data for {current_date}")

                # Fetch bars for all symbols
                df = connector.fetch_bars(
                    symbols=symbols,
                    start_date=current_date,
                    end_date=current_date,
                    timeframe="1Day",
                )

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

    def run(self):
        """Run edge collector."""
        logger.info("Edge collector starting...")

        # Run collection tasks
        tasks = self.config.get("tasks", ["alpaca_bars"])

        for task in tasks:
            if self.shutdown_requested:
                break

            if task == "alpaca_bars":
                self.collect_alpaca_bars()
            else:
                logger.warning(f"Unknown task: {task}")

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
