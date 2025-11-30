"""
Incremental Curation Engine.

Detects changed raw partitions and rebuilds only affected curated tables.
Uses watermarks and file modification times for efficient change detection.

SSOT v2 Section 2.4: Backfill subsystem
- curate_incremental() detects changed raw partitions and rebuilds affected curated partitions
"""

from __future__ import annotations

import hashlib
import io
import os
import sys
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import yaml
from loguru import logger

# Ensure repo root on path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data_layer.manifests import get_manifest_manager, ManifestManager
from data_layer.reference.corporate_actions import CorporateActionsProcessor


# Dependency graph: which curated tables depend on which raw/reference tables
DEPENDENCY_GRAPH = {
    "curated/equities_ohlcv_adj": {
        "raw_dependencies": ["raw/alpaca/equities_bars"],
        "reference_dependencies": ["reference/corp_actions"],
    },
    "curated/equities_minute": {
        "raw_dependencies": ["raw/alpaca/equities_bars_minute"],
        "reference_dependencies": [],
    },
    "curated/options_iv": {
        "raw_dependencies": ["raw/finnhub/options_chains"],
        "reference_dependencies": [],
    },
    "curated/options_surface": {
        "raw_dependencies": [],  # Depends on curated/options_iv
        "curated_dependencies": ["curated/options_iv"],
        "reference_dependencies": [],
    },
}


class IncrementalCurator:
    """
    Manages incremental curation by tracking changes in raw and reference data.
    """

    def __init__(self, config_path: str = "configs/curator.yml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.manifests = get_manifest_manager()
        self._ca_processor: Optional[CorporateActionsProcessor] = None
        self._ca_hash: Optional[str] = None

        logger.info("IncrementalCurator initialized")

    def _load_corp_actions(self) -> Tuple[Optional[CorporateActionsProcessor], str]:
        """Load corporate actions and compute hash for change detection."""
        root = Path("data_layer/reference/corp_actions")
        if not root.exists():
            return None, ""

        try:
            self._ca_processor = CorporateActionsProcessor()
            # Compute hash over corp_actions parquet files
            hash_parts = []
            for p in sorted(root.glob("*.parquet")):
                stat = p.stat()
                hash_parts.append(f"{p.name}:{stat.st_mtime_ns}:{stat.st_size}")
            self._ca_hash = hashlib.md5("|".join(hash_parts).encode()).hexdigest()
            return self._ca_processor, self._ca_hash
        except Exception as e:
            logger.warning(f"Failed to load corporate actions: {e}")
            return None, ""

    def _get_raw_dates(self, input_prefix: str) -> Dict[str, float]:
        """
        Get available raw dates with their modification times.

        Returns:
            Dict mapping date string to modification timestamp
        """
        dates: Dict[str, float] = {}

        # Support both source-first and table-first layouts
        roots = [Path("data_layer") / input_prefix]
        try:
            parts = input_prefix.split("/")
            if parts[0] == "raw" and len(parts) >= 3:
                roots.append(Path("data_layer") / "raw" / parts[2] / parts[1])
        except Exception:
            pass

        for root in roots:
            if not root.exists():
                continue
            for p in root.glob("date=*/data.parquet"):
                date_str = p.parent.name.split("=")[-1]
                try:
                    # Validate date format
                    datetime.strptime(date_str, "%Y-%m-%d")
                    mtime = p.stat().st_mtime
                    # Keep latest if duplicate
                    if date_str not in dates or mtime > dates[date_str]:
                        dates[date_str] = mtime
                except Exception:
                    continue

        return dates

    def _get_curated_dates(self, output_prefix: str) -> Dict[str, float]:
        """Get curated dates with their modification times."""
        dates: Dict[str, float] = {}
        root = Path("data_layer") / output_prefix

        if not root.exists():
            return dates

        for p in root.glob("date=*/data.parquet"):
            date_str = p.parent.name.split("=")[-1]
            try:
                datetime.strptime(date_str, "%Y-%m-%d")
                dates[date_str] = p.stat().st_mtime
            except Exception:
                continue

        return dates

    def _read_raw_parquet(self, input_prefix: str, date_str: str) -> pd.DataFrame:
        """Read raw parquet for a given date."""
        # Try primary path
        path = Path("data_layer") / input_prefix / f"date={date_str}" / "data.parquet"
        if path.exists():
            return pd.read_parquet(path)

        # Try alternate layout
        try:
            parts = input_prefix.split("/")
            if parts[0] == "raw" and len(parts) >= 3:
                alt_path = Path("data_layer") / "raw" / parts[2] / parts[1] / f"date={date_str}" / "data.parquet"
                if alt_path.exists():
                    return pd.read_parquet(alt_path)
        except Exception:
            pass

        return pd.DataFrame()

    def _write_curated_parquet(
        self,
        output_prefix: str,
        date_str: str,
        df: pd.DataFrame
    ) -> None:
        """Write curated parquet with atomic write."""
        out_dir = Path("data_layer") / output_prefix / f"date={date_str}"
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / "data.parquet"
        tmp_path = out_dir / f".data_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.tmp"

        try:
            df.to_parquet(tmp_path, index=False)
            tmp_path.rename(out_path)
            logger.debug(f"Wrote {len(df)} rows to {out_path}")
        except Exception as e:
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    def _transform_equities_bars(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform raw equities bars to adjusted OHLCV."""
        if df.empty:
            return df

        df2 = df.copy()
        df2["date"] = pd.to_datetime(df2["date"]).dt.date

        # Load corporate actions if not already loaded
        if self._ca_processor is None:
            self._load_corp_actions()

        # Apply adjustments
        out_rows = []
        for (ds, sym), sub in df2.groupby(["date", "symbol"], as_index=False):
            adj_factor = 1.0
            div_cash = 0.0

            if self._ca_processor is not None:
                try:
                    adj_factor = self._ca_processor.get_adjustment_factor(sym, ds)
                    div_cash = self._ca_processor.get_dividend_amount(sym, ds)
                except Exception:
                    pass

            row = {
                "date": ds,
                "symbol": sym,
                "session_id": sub.get("session_id", pd.Series([None])).iloc[0],
                "open_adj": sub["open"].iloc[0] / adj_factor if "open" in sub.columns else None,
                "high_adj": sub["high"].iloc[0] / adj_factor if "high" in sub.columns else None,
                "low_adj": sub["low"].iloc[0] / adj_factor if "low" in sub.columns else None,
                "close_adj": sub["close"].iloc[0] / adj_factor if "close" in sub.columns else None,
                "vwap_adj": sub["vwap"].iloc[0] / adj_factor if "vwap" in sub.columns and pd.notna(sub["vwap"].iloc[0]) else None,
                "volume_adj": int(sub["volume"].iloc[0] * adj_factor) if "volume" in sub.columns else None,
                "div_cash": div_cash,
                "close_raw": sub["close"].iloc[0] if "close" in sub.columns else None,
                "adjustment_factor": float(adj_factor),
                "last_adjustment_date": None,
                "ingested_at": sub["ingested_at"].iloc[0] if "ingested_at" in sub.columns else pd.Timestamp.utcnow(),
                "source_name": sub["source_name"].iloc[0] if "source_name" in sub.columns else "curator",
                "source_uri": sub["source_uri"].iloc[0] if "source_uri" in sub.columns else "curator://equities_ohlcv_adj",
                "transform_id": "equities_ohlcv_adj_v2",
            }
            out_rows.append(row)

        return pd.DataFrame(out_rows) if out_rows else pd.DataFrame()

    def _detect_changes(
        self,
        job: dict,
        ca_hash_changed: bool,
    ) -> Tuple[List[str], List[str]]:
        """
        Detect which dates need to be rebuilt.

        Returns:
            Tuple of (new_dates, rebuild_dates)
            - new_dates: raw dates not yet curated
            - rebuild_dates: curated dates that need rebuild (raw changed or CA changed)
        """
        input_prefix = job["input_prefix"].rstrip("/")
        output_prefix = job["output_prefix"].rstrip("/")

        raw_dates = self._get_raw_dates(input_prefix)
        curated_dates = self._get_curated_dates(output_prefix)

        new_dates = []
        rebuild_dates = []

        for date_str, raw_mtime in raw_dates.items():
            if date_str not in curated_dates:
                # New date
                new_dates.append(date_str)
            else:
                curated_mtime = curated_dates[date_str]
                if raw_mtime > curated_mtime:
                    # Raw was updated after curation
                    rebuild_dates.append(date_str)

        # If corporate actions changed, rebuild all curated dates that might be affected
        if ca_hash_changed and job["name"].startswith("equities_bars"):
            for date_str in curated_dates:
                if date_str not in new_dates and date_str not in rebuild_dates:
                    rebuild_dates.append(date_str)
            logger.info(f"Corporate actions changed, marking {len(rebuild_dates)} dates for rebuild")

        return sorted(new_dates), sorted(rebuild_dates)

    def process_job(self, job: dict, ca_hash_changed: bool) -> int:
        """
        Process a single curation job incrementally.

        Returns:
            Number of partitions processed
        """
        name = job["name"]
        input_prefix = job["input_prefix"].rstrip("/")
        output_prefix = job["output_prefix"].rstrip("/")

        new_dates, rebuild_dates = self._detect_changes(job, ca_hash_changed)

        if not new_dates and not rebuild_dates:
            logger.info(f"Job {name}: no changes detected")
            return 0

        logger.info(f"Job {name}: {len(new_dates)} new, {len(rebuild_dates)} rebuild")

        all_dates = sorted(set(new_dates + rebuild_dates))
        processed = 0

        for date_str in all_dates:
            try:
                raw_df = self._read_raw_parquet(input_prefix, date_str)
                if raw_df.empty:
                    logger.warning(f"No raw data for {date_str}, skipping")
                    continue

                # Transform based on job type
                if name.startswith("equities_bars"):
                    curated_df = self._transform_equities_bars(raw_df)
                elif name.startswith("equities_minute"):
                    # Pass-through for minute data
                    curated_df = raw_df.copy()
                else:
                    curated_df = raw_df

                self._write_curated_parquet(output_prefix, date_str, curated_df)
                processed += 1

                # Update bookmark
                self.manifests.set_bookmark(
                    vendor="curator",
                    dataset=name,
                    last_date=date_str,
                    row_count=len(curated_df),
                    status="success",
                )

            except Exception as e:
                logger.error(f"Failed to process {date_str}: {e}")
                continue

        return processed

    def run(self) -> Dict[str, int]:
        """
        Run incremental curation for all configured jobs.

        Returns:
            Dict mapping job name to number of partitions processed
        """
        logger.info("Starting incremental curation...")

        # Check for corporate actions changes
        _, ca_hash = self._load_corp_actions()

        # Get previous CA hash from bookmark
        prev_bookmark = self.manifests.get_bookmark("curator", "_corp_actions_hash")
        prev_ca_hash = prev_bookmark.metadata.get("hash", "") if prev_bookmark and prev_bookmark.metadata else ""

        ca_hash_changed = ca_hash != prev_ca_hash
        if ca_hash_changed and ca_hash:
            logger.info("Corporate actions have changed since last run")
            # Store new hash
            self.manifests.set_bookmark(
                vendor="curator",
                dataset="_corp_actions_hash",
                last_date=date.today().isoformat(),
                row_count=0,
                status="success",
                metadata={"hash": ca_hash},
            )

        results = {}
        for job in self.config.get("jobs", []):
            try:
                count = self.process_job(job, ca_hash_changed)
                results[job["name"]] = count
            except Exception as e:
                logger.exception(f"Job {job['name']} failed: {e}")
                results[job["name"]] = -1

        logger.info(f"Incremental curation complete: {results}")
        return results


def curate_incremental() -> Dict[str, int]:
    """
    Main entry point for incremental curation.

    Detects changed raw partitions and rebuilds affected curated partitions.
    Uses watermarks for efficient change detection.

    Returns:
        Dict mapping job name to number of partitions processed
    """
    try:
        curator = IncrementalCurator()
        return curator.run()
    except Exception as e:
        logger.exception(f"Incremental curation failed: {e}")
        return {}


if __name__ == "__main__":
    results = curate_incremental()
    print(f"Results: {results}")
