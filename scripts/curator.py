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

        # Preload corporate actions (splits + dividends)
        self._corp_actions = self._load_corp_actions()
        self._ca_hash = self._hash_corp_actions(self._corp_actions) if self._corp_actions is not None else None

    # -------- Corporate Actions helpers --------
    @staticmethod
    def _hash_corp_actions(df: pd.DataFrame | None) -> str:
        if df is None or df.empty:
            return ""
        # Stable hash over minimal identifying columns
        keep = [c for c in ["symbol", "event_type", "ex_date", "ratio", "amount"] if c in df.columns]
        if not keep:
            return ""
        d = df[keep].copy()
        if "ratio" in d.columns:
            d["ratio"] = pd.to_numeric(d["ratio"], errors="coerce").fillna(0.0)
        if "amount" in d.columns:
            d["amount"] = pd.to_numeric(d["amount"], errors="coerce").fillna(0.0)
        d = d.sort_values(keep).astype({"symbol": str, "event_type": str})
        return str(pd.util.hash_pandas_object(d, index=False).sum())

    @staticmethod
    def _load_corp_actions() -> pd.DataFrame | None:
        # Load corporate actions (splits + dividends)
        root = Path("data_layer/reference/corp_actions")
        if not root.exists():
            return None
        frames = []
        for p in root.glob("*.parquet"):
            try:
                df = pd.read_parquet(p)
                frames.append(df)
            except Exception:
                pass
        if not frames:
            return None
        ca = pd.concat(frames, ignore_index=True)
        if ca.empty:
            return None
        # Normalize columns
        if "ex_date" in ca.columns:
            ca["ex_date"] = pd.to_datetime(ca["ex_date"]).dt.date
        keep_cols = [c for c in ["symbol", "event_type", "ex_date", "ratio", "amount"] if c in ca.columns]
        return ca[keep_cols].dropna(subset=["symbol", "event_type", "ex_date"])

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
            # Local mode: support source-first and table-first
            roots = [Path("data_layer") / input_prefix]
            try:
                parts = input_prefix.split("/")
                if parts[0] == "raw" and len(parts) >= 3:
                    roots.append(Path("data_layer") / "raw" / parts[2] / parts[1])
            except Exception:
                pass
            keys: List[str] = []
            for root in roots:
                if not root.exists():
                    continue
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

    def _corp_actions_modified_after(self, ts_ns: float) -> bool:
        """Check if any corp_actions parquet modified after given timestamp."""
        root = Path("data_layer/reference/corp_actions")
        if not root.exists():
            return False
        for p in root.glob("*.parquet"):
            try:
                if p.stat().st_mtime_ns > ts_ns:
                    return True
            except Exception:
                continue
        return False

    # -------- Pipeline --------
    def _transform_equities_bars(self, df: pd.DataFrame) -> pd.DataFrame:
        # Corporate-action aware transform: forward cumulative split adjustment
        # Price fields divided by cumulative split ratio up to date; volume multiplied.
        if df.empty:
            return df

        df2 = df.copy()
        df2["date"] = pd.to_datetime(df2["date"]).dt.date

        # Build CA lookup per symbol
        ca = self._corp_actions
        splits = None
        dividends = None
        if ca is not None and not ca.empty and "event_type" in ca.columns:
            splits = ca[ca["event_type"] == "split"] if "ratio" in ca.columns else None
            dividends = ca[ca["event_type"] == "dividend"] if "amount" in ca.columns else None
        def cum_ratio(sym: str, ds) -> float:
            if splits is None or splits.empty:
                return 1.0
            s = splits[splits["symbol"] == sym]
            if s.empty:
                return 1.0
            # Forward cumulative ratio: product of ratios with ex_date <= ds
            r = s[s["ex_date"] <= ds]["ratio"]
            if r.empty:
                return 1.0
            try:
                val = float(pd.to_numeric(r, errors="coerce").fillna(1.0).prod())
                return val if val > 0 else 1.0
            except Exception:
                return 1.0
        def div_cash(sym: str, ds) -> float:
            if dividends is None or dividends.empty:
                return 0.0
            d = dividends[(dividends["symbol"] == sym) & (dividends["ex_date"] == ds)]
            if d.empty:
                return 0.0
            try:
                return float(pd.to_numeric(d["amount"], errors="coerce").fillna(0.0).sum())
            except Exception:
                return 0.0

        out_rows = []
        for (ds, sym), sub in df2.groupby(["date", "symbol"], as_index=False):
            adj_factor = cum_ratio(sym, ds)
            open_adj = sub["open"].iloc[0] / adj_factor if "open" in sub.columns else None
            high_adj = sub["high"].iloc[0] / adj_factor if "high" in sub.columns else None
            low_adj = sub["low"].iloc[0] / adj_factor if "low" in sub.columns else None
            close_adj = sub["close"].iloc[0] / adj_factor if "close" in sub.columns else None
            vwap_adj = sub["vwap"].iloc[0] / adj_factor if "vwap" in sub.columns and pd.notna(sub["vwap"].iloc[0]) else None
            vol_adj = int(sub["volume"].iloc[0] * adj_factor) if "volume" in sub.columns else None

            out_rows.append({
                "date": ds,
                "symbol": sym,
                "session_id": sub.get("session_id", pd.Series([None])).iloc[0],
                "open_adj": open_adj,
                "high_adj": high_adj,
                "low_adj": low_adj,
                "close_adj": close_adj,
                "vwap_adj": vwap_adj,
                "volume_adj": vol_adj,
                "div_cash": div_cash(sym, ds),
                "close_raw": sub["close"].iloc[0] if "close" in sub.columns else None,
                "adjustment_factor": float(adj_factor),
                "last_adjustment_date": None,
                "ingested_at": sub["ingested_at"].iloc[0] if "ingested_at" in sub.columns else pd.Timestamp.utcnow(),
                "source_name": sub["source_name"].iloc[0] if "source_name" in sub.columns else "curator",
                "source_uri": sub["source_uri"].iloc[0] if "source_uri" in sub.columns else "curator://equities_bars_ohlcv",
                "transform_id": "equities_bars_ohlcv_v2",
            })

        curated = pd.DataFrame(out_rows)
        return curated

    def process_job(self, job: dict):
        name = job["name"]
        input_prefix = job["input_prefix"].rstrip('/')
        output_prefix = job["output_prefix"].rstrip('/')

        # Discover available dates
        dates = self._list_raw_dates(input_prefix)
        if not dates:
            logger.info(f"No raw dates found under {input_prefix}")
            return

        # Load watermark and CA hash to decide resume point
        last_ts = None
        ca_hash_changed = False
        ca_min_date = None
        if self.watermarks:
            last_ts = self.watermarks.get_last_timestamp("curator", name)
            if last_ts:
                logger.info(f"Resuming {name} from watermark: {last_ts}")
            last_hash = self.watermarks.get_last_timestamp("curator", f"{name}_ca_hash")
            if (self._ca_hash or "") != (last_hash or ""):
                ca_hash_changed = True
                logger.info("Corporate actions changed; forcing reprocess from earliest split date")
                if self._corp_actions is not None and not self._corp_actions.empty:
                    try:
                        ca_min_date = str(min(self._corp_actions["ex_date"]))
                    except Exception:
                        ca_min_date = None

        # Process in order
        for ds in dates:
            if last_ts and ds <= last_ts:
                # If CA changed and split occurs before/at ds, allow reprocess for impacted window
                if not ca_hash_changed:
                    continue
                if ca_min_date and ds < ca_min_date:
                    continue

            if job.get("idempotent", True) and self._curated_exists(output_prefix, ds):
                # If CA changed and this date is impacted (>= min ex_date), reprocess
                allow_reprocess = False
                if ca_hash_changed and (not ca_min_date or ds >= ca_min_date):
                    allow_reprocess = True
                elif self.watermarks is None:
                    # Local mode: compare mtime
                    cur_path = Path("data_layer") / output_prefix / f"date={ds}" / "data.parquet"
                    try:
                        allow_reprocess = self._corp_actions_modified_after(cur_path.stat().st_mtime_ns)
                    except Exception:
                        allow_reprocess = False
                if not allow_reprocess:
                    logger.debug(f"Curated exists for {ds}, skipping")
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
            elif name.startswith("equities_minute"):
                # Pass-through minute data; ensure minimal columns
                curated = raw_df.copy()
            else:
                curated = raw_df

            self._write_curated_parquet(output_prefix, ds, curated)

            # Update watermark
            if self.watermarks:
                self.watermarks.set("curator", name, ds, row_count=len(curated))
                # Persist CA hash
                if self._ca_hash is not None:
                    self.watermarks.set("curator", f"{name}_ca_hash", self._ca_hash, row_count=0)

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
