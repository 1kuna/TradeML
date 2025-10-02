#!/usr/bin/env python
"""
Curator: Watermark-based processor for moving data from raw/ → curated/.

For each configured job, the curator:
- Discovers new date partitions under raw/<source>/<table>/date=YYYY-MM-DD/
- Applies lightweight transforms (placeholder for now)
- Writes to curated/<table_out>/date=YYYY-MM-DD/data.parquet
- Maintains a watermark per job using BookmarkManager for idempotency
"""

import os
import io
import sys
import json
import argparse
from typing import List, Set
from datetime import datetime, date
from pathlib import Path

import yaml
import pandas as pd
from dotenv import load_dotenv
from loguru import logger

# Ensure repo root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_layer.storage.s3_client import get_s3_client
from data_layer.storage.bookmarks import BookmarkManager


class Curator:
    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.storage_backend = os.getenv("STORAGE_BACKEND", "local")
        logger.info(f"Storage backend: {self.storage_backend}")

        if self.storage_backend == "s3":
            self.s3 = get_s3_client()

            # Use BookmarkManager as a per-job watermark store
            bookmark_key = self.config.get("watermark", {}).get("bookmark_key", "manifests/curator_watermarks.json")
            self.watermarks = BookmarkManager(self.s3, bookmark_key=bookmark_key)
        else:
            self.s3 = None
            self.watermarks = None

    # -------- Helpers --------
    @staticmethod
    def _extract_dates_from_keys(keys: List[str]) -> List[str]:
        dates: Set[str] = set()
        for k in keys:
            # Expect .../date=YYYY-MM-DD/... structure
            parts = k.split('/')
            for p in parts:
                if p.startswith('date=') and len(p) >= 10:
                    ds = p.split('=', 1)[1]
                    # Basic sanity
                    try:
                        datetime.strptime(ds, "%Y-%m-%d")
                        dates.add(ds)
                    except Exception:
                        pass
        return sorted(dates)

    def _list_raw_dates(self, input_prefix: str) -> List[str]:
        if self.s3:
            objs = self.s3.list_objects(prefix=f"{input_prefix}/")
            keys = [o["Key"] for o in objs]
            return self._extract_dates_from_keys(keys)
        else:
            # Local mode: scan filesystem
            root = Path("data_layer") / input_prefix
            if not root.exists():
                return []
            keys = []
            for p in root.rglob("*"):
                if p.is_file():
                    keys.append(str(p.relative_to(Path("data_layer"))))
            return self._extract_dates_from_keys(keys)

    def _read_raw_parquet(self, input_prefix: str, date_str: str) -> pd.DataFrame:
        if self.s3:
            key = f"{input_prefix}/date={date_str}/data.parquet"
            try:
                data, _ = self.s3.get_object(key)
            except Exception as e:
                logger.warning(f"Raw parquet not found for {date_str}: {key} ({e})")
                return pd.DataFrame()
            return pd.read_parquet(io.BytesIO(data))
        else:
            path = Path("data_layer") / input_prefix / f"date={date_str}" / "data.parquet"
            if not path.exists():
                logger.debug(f"Raw parquet not found: {path}")
                return pd.DataFrame()
            return pd.read_parquet(path)

    def _write_curated_parquet(self, output_prefix: str, date_str: str, df: pd.DataFrame):
        if self.s3:
            key = f"{output_prefix}/date={date_str}/data.parquet"
            buf = io.BytesIO()
            df.to_parquet(buf, index=False)
            self.s3.put_object(key, buf.getvalue())
            logger.info(f"Wrote curated to s3://{self.s3.bucket}/{key} ({len(df)} rows)")
        else:
            path = Path("data_layer") / output_prefix / f"date={date_str}"
            path.mkdir(parents=True, exist_ok=True)
            (path / "data.parquet").write_bytes(b"")  # ensure file exists
            df.to_parquet(path / "data.parquet", index=False)
            logger.info(f"Wrote curated to {path}/data.parquet ({len(df)} rows)")

    def _curated_exists(self, output_prefix: str, date_str: str) -> bool:
        if self.s3:
            key = f"{output_prefix}/date={date_str}/data.parquet"
            return self.s3.object_exists(key)
        else:
            path = Path("data_layer") / output_prefix / f"date={date_str}" / "data.parquet"
            return path.exists()

    # -------- Pipeline --------
    def _transform_equities_bars(self, df: pd.DataFrame) -> pd.DataFrame:
        # Placeholder for actual PIT-safe adjustments. For now, pass-through.
        if df.empty:
            return df
        # Ensure stable column order
        cols = [
            "date", "symbol", "open", "high", "low", "close", "vwap",
            "volume", "trades", "nbbo_spread", "imbalance",
            "ingested_at", "source_name", "source_uri",
        ]
        existing = [c for c in cols if c in df.columns]
        return df[existing].copy()

    def process_job(self, job: dict):
        name = job["name"]
        input_prefix = job["input_prefix"].rstrip('/')
        output_prefix = job["output_prefix"].rstrip('/')

        # Discover available dates
        dates = self._list_raw_dates(input_prefix)
        if not dates:
            logger.info(f"No raw dates found under {input_prefix}")
            return

        # Load watermark
        last_ts = None
        if self.watermarks:
            last_ts = self.watermarks.get_last_timestamp("curator", name)
            if last_ts:
                logger.info(f"Resuming {name} from watermark: {last_ts}")

        # Process in order
        for ds in dates:
            if last_ts and ds <= last_ts:
                continue

            if job.get("idempotent", True) and self._curated_exists(output_prefix, ds):
                logger.debug(f"Curated exists for {ds}, skipping")
                # Still advance the watermark
                if self.watermarks:
                    self.watermarks.set("curator", name, ds, row_count=0)
                continue

            raw_df = self._read_raw_parquet(input_prefix, ds)
            if raw_df.empty:
                logger.debug(f"No raw data for {ds}")
                if self.watermarks:
                    self.watermarks.set("curator", name, ds, row_count=0)
                continue

            # Transform stage per job type
            if name.startswith("equities_bars"):
                curated = self._transform_equities_bars(raw_df)
            else:
                curated = raw_df

            self._write_curated_parquet(output_prefix, ds, curated)

            # Update watermark
            if self.watermarks:
                self.watermarks.set("curator", name, ds, row_count=len(curated))

        logger.info(f"Job complete: {name}")

    def run(self):
        logger.info("Curator starting...")
        for job in self.config.get("jobs", []):
            self.process_job(job)
        logger.info("Curator stopped")


def main():
    parser = argparse.ArgumentParser(description="Watermark-based curator")
    parser.add_argument(
        "--config",
        default="configs/curator.yml",
        help="Path to curator config file",
    )
    args = parser.parse_args()

    load_dotenv()
    Curator(args.config).run()


if __name__ == "__main__":
    main()

