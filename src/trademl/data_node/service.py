"""Main data-node service loop."""

from __future__ import annotations

import contextlib
import logging
import os
import signal
import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import sleep
from zoneinfo import ZoneInfo

import pandas as pd

from trademl.calendars.exchange import is_trading_day
from trademl.connectors.base import BaseConnector, ConnectorError, TemporaryConnectorError
from trademl.data_node.auditor import PartitionAuditor
from trademl.data_node.curator import Curator, CuratorResult
from trademl.data_node.db import DataNodeDB
from trademl.fleet.cluster import ClusterCoordinator, ShardSpec
from trademl.reference.security_master import rebuild_derived_references

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class DataNodePaths:
    """Filesystem paths for the node."""

    root: Path

    @property
    def raw_equities(self) -> Path:
        return self.root / "data" / "raw" / "equities_bars"

    @property
    def curated_equities(self) -> Path:
        return self.root / "data" / "curated" / "equities_ohlcv_adj"

    @property
    def qc_root(self) -> Path:
        return self.root / "data" / "qc"

    @property
    def reference_root(self) -> Path:
        return self.root / "data" / "reference"


class DataNodeService:
    """Collect raw bars, audit completeness, curate data, and sync QC state."""

    def __init__(
        self,
        *,
        db: DataNodeDB,
        connectors: dict[str, BaseConnector],
        auditor: PartitionAuditor,
        curator: Curator,
        paths: DataNodePaths,
        source_name: str = "alpaca",
    ) -> None:
        self.db = db
        self.connectors = connectors
        self.auditor = auditor
        self.curator = curator
        self.paths = paths
        self.source_name = source_name
        self._stop_event = threading.Event()
        self.default_symbols: list[str] = []
        self._collection_history: set[str] = set()
        self._maintenance_history: set[str] = set()
        self._reference_history: set[str] = set()
        self._price_check_history: set[str] = set()

    def stop(self) -> None:
        """Request a graceful shutdown."""
        self._stop_event.set()

    def install_signal_handlers(self) -> None:
        """Install SIGINT and SIGTERM handlers for graceful shutdown."""
        signal.signal(signal.SIGINT, lambda *_: self.stop())
        signal.signal(signal.SIGTERM, lambda *_: self.stop())

    def _write_raw_partition(self, frame: pd.DataFrame) -> list[str]:
        changed_dates: list[str] = []
        for day, day_frame in frame.groupby("date"):
            day_value = pd.Timestamp(day).strftime("%Y-%m-%d")
            partition = self.paths.raw_equities / f"date={day_value}"
            partition.mkdir(parents=True, exist_ok=True)
            day_frame.to_parquet(partition / "data.parquet", index=False)
            changed_dates.append(day_value)
            self.db.update_partition_status(
                source=self.source_name,
                dataset="equities_eod",
                date=day_value,
                status="GREEN",
                row_count=len(day_frame),
                expected_rows=len(day_frame),
                qc_code="OK",
            )
        return changed_dates

    def collect_forward(self, *, trading_date: str, symbols: list[str]) -> list[str]:
        """Fetch and persist the primary daily bars."""
        frame = self.connectors[self.source_name].fetch("equities_eod", symbols, trading_date, trading_date)
        if frame.empty:
            return []
        return self._write_raw_partition(frame)

    def collect_forward_shard(self, *, trading_date: str, symbols: list[str], shard_id: str) -> list[str]:
        """Fetch and persist a shard-specific daily bar slice."""
        if not symbols:
            return []
        frame = self.connectors[self.source_name].fetch("equities_eod", symbols, trading_date, trading_date)
        if frame.empty:
            return []
        return self._write_raw_shard_partition(frame, shard_id=shard_id)

    def process_backfill_queue(self) -> list[str]:
        """Process queued backfill or gap tasks until the queue is drained."""
        changed_dates: list[str] = []
        while task := self.db.lease_next_task():
            connector = self.connectors[self.source_name]
            symbols = [task.symbol] if task.symbol else self.default_symbols
            try:
                frame = connector.fetch(task.dataset, symbols, task.start_date, task.end_date)
            except ConnectorError as exc:
                self.db.mark_task_failed(task.id, str(exc), backoff_minutes=30)
                continue
            if frame.empty:
                self.db.mark_task_failed(task.id, "empty backfill result", backoff_minutes=30)
                continue
            changed_dates.extend(self._write_raw_partition(frame))
            self.db.mark_task_done(task.id)
        return changed_dates

    def collect_reference_data(self, jobs: list[dict[str, object]]) -> list[Path]:
        """Collect reference datasets into parquet files."""
        outputs: list[Path] = []
        self.paths.reference_root.mkdir(parents=True, exist_ok=True)
        failures: list[str] = []
        for job in self._expand_reference_jobs(jobs):
            connector = self.connectors[str(job["source"])]
            try:
                frame = connector.fetch(
                    str(job["dataset"]),
                    list(job.get("symbols", [])),
                    str(job["start_date"]),
                    str(job["end_date"]),
                )
            except ConnectorError as exc:
                failures.append(f"{job['source']}:{job['dataset']}:{job.get('symbols', [])}:{exc}")
                continue
            output = self.paths.reference_root / f"{job['output_name']}.parquet"
            if output.exists():
                existing = pd.read_parquet(output)
                frame = pd.concat([existing, frame], ignore_index=True) if not frame.empty else existing
            if not frame.empty:
                frame = frame.drop_duplicates().reset_index(drop=True)
            frame.to_parquet(output, index=False)
            outputs.append(output)
        outputs.extend(rebuild_derived_references(self.paths.reference_root))
        if failures:
            raise TemporaryConnectorError(f"reference collection incomplete: {' | '.join(failures)}")
        return outputs

    def collect_macro_data(self, series_ids: list[str], start_date: str, end_date: str) -> list[Path]:
        """Collect FRED macro series partitions."""
        connector = self.connectors["fred"]
        frame = connector.fetch("macros_treasury", series_ids, start_date, end_date)
        outputs: list[Path] = []
        for series_id, series_frame in frame.groupby("series_id"):
            partition = self.paths.root / "data" / "raw" / "macros_fred" / f"series={series_id}"
            partition.mkdir(parents=True, exist_ok=True)
            output = partition / "data.parquet"
            series_frame.to_parquet(output, index=False)
            outputs.append(output)
        return outputs

    def run_cross_vendor_price_checks(self, *, trading_date: str, sample_symbols: list[str]) -> Path:
        """Compare the primary source against backup vendors for a sample of symbols."""
        comparisons: list[pd.DataFrame] = []
        primary = self.connectors[self.source_name].fetch("equities_eod", sample_symbols, trading_date, trading_date)
        for vendor in ["massive", "finnhub"]:
            if vendor not in self.connectors:
                continue
            try:
                backup = self.connectors[vendor].fetch("equities_eod", sample_symbols, trading_date, trading_date)
            except Exception:
                continue
            merged = primary[["symbol", "close"]].merge(
                backup[["symbol", "close"]],
                on="symbol",
                how="inner",
                suffixes=("_primary", f"_{vendor}"),
            )
            if not merged.empty:
                merged["vendor"] = vendor
                merged["date"] = trading_date
                comparisons.append(merged)
        self.paths.qc_root.mkdir(parents=True, exist_ok=True)
        output = self.paths.qc_root / f"price_checks_{trading_date}.parquet"
        if comparisons:
            pd.concat(comparisons, ignore_index=True).to_parquet(output, index=False)
        else:
            pd.DataFrame().to_parquet(output, index=False)
        return output

    def curate_dates(
        self,
        corp_actions: pd.DataFrame | None = None,
        *,
        changed_dates: list[str] | None = None,
    ) -> CuratorResult:
        """Rebuild curated partitions from the current raw dataset."""
        raw_files = sorted(self.paths.raw_equities.glob("date=*/data.parquet"))
        raw_frame = pd.concat((pd.read_parquet(path) for path in raw_files), ignore_index=True) if raw_files else pd.DataFrame()
        result = self.curator.write_curated(
            raw_bars=raw_frame,
            corp_actions=corp_actions if corp_actions is not None else self.load_corp_actions_reference(),
            output_root=self.paths.curated_equities,
            changed_dates=changed_dates,
            adjustment_log_path=self.paths.root / "data" / "curated" / "adjustment_log.parquet",
        )
        return result

    def sync_partition_status(self) -> Path:
        """Mirror the local SQLite QC ledger to parquet."""
        self.paths.qc_root.mkdir(parents=True, exist_ok=True)
        output = self.paths.qc_root / "partition_status.parquet"
        rows = [dict(row) for row in self.db.fetch_partition_status()]
        pd.DataFrame(rows).to_parquet(output, index=False)
        return output

    def load_corp_actions_reference(self) -> pd.DataFrame:
        """Load any persisted corp-action reference parquet files into the curator schema."""
        frames: list[pd.DataFrame] = []
        for path in sorted(self.paths.reference_root.glob("*.parquet")):
            frame = pd.read_parquet(path)
            if frame.empty:
                continue
            if path.stem == "corp_actions":
                normalized = frame.copy()
                if "ex_date" in normalized.columns:
                    normalized["ex_date"] = pd.to_datetime(normalized["ex_date"]).dt.date
                frames.append(normalized)
                continue
            if path.stem == "splits":
                normalized = pd.DataFrame(
                    {
                        "symbol": frame["symbol"],
                        "event_type": "split",
                        "ex_date": pd.to_datetime(frame.get("execution_date", frame.get("ex_date", frame.get("exDate")))).dt.date,
                        "ratio": pd.to_numeric(frame.get("ratio"), errors="coerce"),
                        "amount": pd.NA,
                        "source": frame.get("source", pd.Series(["splits"] * len(frame))),
                    }
                )
                if normalized["ratio"].isna().all() and {"split_from", "split_to"}.issubset(frame.columns):
                    normalized["ratio"] = pd.to_numeric(frame["split_from"], errors="coerce") / pd.to_numeric(frame["split_to"], errors="coerce")
                frames.append(normalized.dropna(subset=["ex_date", "ratio"]))
                continue
            if path.stem == "dividends":
                frames.append(
                    pd.DataFrame(
                        {
                            "symbol": frame["symbol"],
                            "event_type": "dividend",
                            "ex_date": pd.to_datetime(frame.get("ex_dividend_date", frame.get("ex_date", frame.get("exDate")))).dt.date,
                            "ratio": pd.NA,
                            "amount": pd.to_numeric(frame.get("cash_amount", frame.get("amount")), errors="coerce"),
                            "source": frame.get("source", pd.Series(["dividends"] * len(frame))),
                        }
                    ).dropna(subset=["ex_date", "amount"])
                )
        if not frames:
            return pd.DataFrame(columns=["symbol", "event_type", "ex_date", "ratio", "amount", "source"])
        combined = pd.concat(frames, ignore_index=True)
        return combined.sort_values(["symbol", "ex_date", "event_type"]).reset_index(drop=True)

    def run_cycle(
        self,
        *,
        trading_date: str,
        symbols: list[str],
        exchange: str,
        audit_start: str,
        audit_end: str,
        corp_actions: pd.DataFrame | None = None,
    ) -> dict[str, object]:
        """Run a single deterministic collection cycle."""
        self.default_symbols = symbols
        forward_dates = self.collect_forward(trading_date=trading_date, symbols=symbols)
        audit_result = self.auditor.audit_range(
            exchange=exchange,
            source=self.source_name,
            dataset="equities_eod",
            start_date=audit_start,
            end_date=audit_end,
            expected_rows=len(symbols),
        )
        backfill_dates = self.process_backfill_queue()
        changed_dates = sorted(set(forward_dates + backfill_dates))
        curated = self.curate_dates(
            corp_actions=corp_actions,
            changed_dates=None if corp_actions is not None and not corp_actions.empty else changed_dates,
        )
        qc_path = self.sync_partition_status()
        return {
            "forward_dates": forward_dates,
            "backfill_dates": backfill_dates,
            "audit_result": audit_result,
            "curated_rows": len(curated.frame),
            "qc_path": qc_path,
        }

    def run_cluster_forever(
        self,
        *,
        coordinator: ClusterCoordinator,
        symbols: list[str],
        exchange: str,
        collection_time_et: str = "16:30",
        maintenance_hour_local: int = 2,
        poll_seconds: float = 60.0,
        audit_lookback_days: int = 7,
        macro_series_ids: list[str] | None = None,
        reference_jobs: list[dict[str, object]] | None = None,
        price_check_symbols: list[str] | None = None,
        now_fn=lambda: datetime.now(tz=UTC),
        sleep_fn=sleep,
    ) -> None:
        """Run the clustered scheduled data-node loop until a stop signal is received."""
        market_tz = ZoneInfo("America/New_York")
        collection_hour, collection_minute = (int(part) for part in collection_time_et.split(":", 1))
        self.default_symbols = symbols

        while not self._stop_event.is_set():
            coordinator.heartbeat_worker()
            current = now_fn()
            if current.tzinfo is None:
                current = current.replace(tzinfo=UTC)
            current_et = current.astimezone(market_tz)
            current_local = current.astimezone()
            trading_date = current_et.date().isoformat()
            owned_shards = coordinator.sync_shard_leases()

            if self._should_run_collection(
                trading_date=trading_date,
                current_et=current_et,
                collection_hour=collection_hour,
                collection_minute=collection_minute,
                exchange=exchange,
            ):
                for shard in owned_shards:
                    self._collect_cluster_shard(trading_date=trading_date, shard=shard)
                self._collection_history.add(trading_date)

                if coordinator.acquire_singleton("audit_curate", trading_date):
                    audit_start = (pd.Timestamp(trading_date) - pd.Timedelta(days=audit_lookback_days)).strftime("%Y-%m-%d")
                    self.auditor.audit_range(
                        exchange=exchange,
                        source=self.source_name,
                        dataset="equities_eod",
                        start_date=audit_start,
                        end_date=trading_date,
                        expected_rows=len(symbols),
                    )
                    self.curate_dates(corp_actions=self.load_corp_actions_reference())
                    self.sync_partition_status()
                    coordinator.mark_singleton_success("audit_curate", trading_date, {"symbol_count": len(symbols)})
                self._run_cluster_auxiliary_tasks(
                    coordinator=coordinator,
                    trading_date=trading_date,
                    current_et=current_et,
                    macro_series_ids=macro_series_ids,
                    reference_jobs=reference_jobs,
                    price_check_symbols=price_check_symbols,
                )

            local_day = current_local.date().isoformat()
            should_run_maintenance = self._should_run_maintenance(
                local_day=local_day,
                current_local=current_local,
                maintenance_hour_local=maintenance_hour_local,
            )
            should_drain_backlog = self.db.has_pending_backfill()
            if should_run_maintenance or should_drain_backlog:
                if coordinator.acquire_singleton("backfill", local_day):
                    changed_dates = self.process_backfill_queue()
                    if changed_dates:
                        self.curate_dates(corp_actions=self.load_corp_actions_reference())
                    self.sync_partition_status()
                    coordinator.mark_singleton_success("backfill", local_day, {"changed_dates": len(changed_dates)})
                    self._maintenance_history.add(local_day)

            # Once EOD bars are fully present, opportunistically fill the slower
            # auxiliary datasets instead of waiting for the next scheduled lane.
            if not self.db.has_pending_backfill():
                anchor_date = self._latest_raw_date()
                if anchor_date:
                    self._run_cluster_auxiliary_tasks(
                        coordinator=coordinator,
                        trading_date=anchor_date,
                        current_et=current_et,
                        macro_series_ids=macro_series_ids,
                        reference_jobs=reference_jobs,
                        price_check_symbols=price_check_symbols,
                    )

            if not self._stop_event.is_set():
                sleep_fn(poll_seconds)

    def run_forever(
        self,
        *,
        symbols: list[str],
        exchange: str,
        collection_time_et: str = "16:30",
        maintenance_hour_local: int = 2,
        poll_seconds: float = 60.0,
        audit_lookback_days: int = 7,
        corp_actions: pd.DataFrame | None = None,
        macro_series_ids: list[str] | None = None,
        reference_jobs: list[dict[str, object]] | None = None,
        price_check_symbols: list[str] | None = None,
        now_fn=lambda: datetime.now(tz=UTC),
        sleep_fn=sleep,
    ) -> None:
        """Run the scheduled data-node loop until a stop signal is received."""
        market_tz = ZoneInfo("America/New_York")
        collection_hour, collection_minute = (int(part) for part in collection_time_et.split(":", 1))

        while not self._stop_event.is_set():
            current = now_fn()
            if current.tzinfo is None:
                current = current.replace(tzinfo=UTC)
            current_et = current.astimezone(market_tz)
            current_local = current.astimezone()
            trading_date = current_et.date()

            if self._should_run_collection(
                trading_date=trading_date.isoformat(),
                current_et=current_et,
                collection_hour=collection_hour,
                collection_minute=collection_minute,
                exchange=exchange,
            ):
                audit_start = (trading_date - pd.Timedelta(days=audit_lookback_days)).strftime("%Y-%m-%d")
                self.run_cycle(
                    trading_date=trading_date.isoformat(),
                    symbols=symbols,
                    exchange=exchange,
                    audit_start=audit_start,
                    audit_end=trading_date.isoformat(),
                    corp_actions=corp_actions,
                )
                if macro_series_ids and "fred" in self.connectors:
                    self.collect_macro_data(macro_series_ids, trading_date.isoformat(), trading_date.isoformat())
                if reference_jobs and self._should_run_reference(current_et):
                    materialized_jobs = [self._materialize_job(job, trading_date.isoformat()) for job in reference_jobs]
                    self.collect_reference_data(materialized_jobs)
                    if any(job.get("output_name") == "corp_actions" for job in materialized_jobs):
                        corp_actions_path = self.paths.root / "data" / "reference" / "corp_actions.parquet"
                        if corp_actions_path.exists():
                            self.curate_dates(corp_actions=pd.read_parquet(corp_actions_path))
                    self._reference_history.add(self._week_key(current_et))
                if price_check_symbols and self._should_run_price_checks(current_et):
                    self.run_cross_vendor_price_checks(
                        trading_date=trading_date.isoformat(),
                        sample_symbols=price_check_symbols,
                    )
                    self._price_check_history.add(self._week_key(current_et))
                self.sync_partition_status()
                self._collection_history.add(trading_date.isoformat())

            local_day = current_local.date().isoformat()
            if self._should_run_maintenance(local_day=local_day, current_local=current_local, maintenance_hour_local=maintenance_hour_local):
                changed_dates = self.process_backfill_queue()
                if changed_dates:
                    self.curate_dates(corp_actions=corp_actions)
                self.sync_partition_status()
                self._maintenance_history.add(local_day)

            if not self._stop_event.is_set():
                sleep_fn(poll_seconds)

    def _should_run_collection(
        self,
        *,
        trading_date: str,
        current_et: datetime,
        collection_hour: int,
        collection_minute: int,
        exchange: str,
    ) -> bool:
        if trading_date in self._collection_history:
            return False
        if not is_trading_day(exchange, trading_date):
            return False
        if current_et.hour < collection_hour:
            return False
        if current_et.hour == collection_hour and current_et.minute < collection_minute:
            return False
        return True

    def _should_run_maintenance(self, *, local_day: str, current_local: datetime, maintenance_hour_local: int) -> bool:
        return local_day not in self._maintenance_history and current_local.hour >= maintenance_hour_local

    @staticmethod
    def _materialize_job(job: dict[str, object], trading_date: str) -> dict[str, object]:
        materialized = dict(job)
        materialized.setdefault("start_date", trading_date)
        materialized.setdefault("end_date", trading_date)
        return materialized

    def _should_run_reference(self, current_et: datetime) -> bool:
        return self._week_key(current_et) not in self._reference_history

    def _should_run_price_checks(self, current_et: datetime) -> bool:
        return self._week_key(current_et) not in self._price_check_history

    @staticmethod
    def _week_key(current_et: datetime) -> str:
        iso = current_et.isocalendar()
        return f"{iso.year}-W{iso.week:02d}"

    def _latest_raw_date(self) -> str | None:
        latest: str | None = None
        for path in self.paths.raw_equities.glob("date=*"):
            _, _, value = path.name.partition("=")
            if value and (latest is None or value > latest):
                latest = value
        return latest

    def _run_cluster_auxiliary_tasks(
        self,
        *,
        coordinator: ClusterCoordinator,
        trading_date: str,
        current_et: datetime,
        macro_series_ids: list[str] | None,
        reference_jobs: list[dict[str, object]] | None,
        price_check_symbols: list[str] | None,
    ) -> None:
        if macro_series_ids and "fred" in self.connectors and coordinator.acquire_singleton("macro", trading_date):
            try:
                self.collect_macro_data(macro_series_ids, trading_date, trading_date)
            except ConnectorError:
                LOGGER.exception("macro collection failed for trading_date=%s", trading_date)
            else:
                coordinator.mark_singleton_success("macro", trading_date, {"series_count": len(macro_series_ids)})

        week_key = self._week_key(current_et)
        if reference_jobs and coordinator.acquire_singleton("reference", week_key):
            materialized_jobs = [self._materialize_job(job, trading_date) for job in reference_jobs]
            try:
                self.collect_reference_data(materialized_jobs)
            except ConnectorError:
                LOGGER.exception("reference collection failed for bucket=%s trading_date=%s", week_key, trading_date)
            else:
                if any(job.get("output_name") == "corp_actions" for job in materialized_jobs):
                    corp_actions_path = self.paths.root / "data" / "reference" / "corp_actions.parquet"
                    if corp_actions_path.exists():
                        self.curate_dates(corp_actions=pd.read_parquet(corp_actions_path))
                coordinator.mark_singleton_success("reference", week_key, {"job_count": len(materialized_jobs)})

        if price_check_symbols and coordinator.acquire_singleton("price_checks", week_key):
            try:
                self.run_cross_vendor_price_checks(trading_date=trading_date, sample_symbols=price_check_symbols)
            except ConnectorError:
                LOGGER.exception("price check collection failed for bucket=%s trading_date=%s", week_key, trading_date)
            else:
                coordinator.mark_singleton_success("price_checks", week_key, {"symbol_count": len(price_check_symbols)})

    @staticmethod
    def _expand_reference_jobs(jobs: list[dict[str, object]]) -> list[dict[str, object]]:
        expanded: list[dict[str, object]] = []
        for job in jobs:
            symbols = list(job.get("symbols", []))
            if len(symbols) <= 1:
                expanded.append(dict(job))
                continue
            for symbol in symbols:
                item = dict(job)
                item["symbols"] = [symbol]
                expanded.append(item)
        return expanded

    def _write_raw_shard_partition(self, frame: pd.DataFrame, *, shard_id: str) -> list[str]:
        changed_dates: list[str] = []
        for day, day_frame in frame.groupby("date"):
            day_value = pd.Timestamp(day).strftime("%Y-%m-%d")
            partition = self.paths.raw_equities / f"date={day_value}"
            shard_root = partition / "shards"
            shard_root.mkdir(parents=True, exist_ok=True)
            shard_path = shard_root / f"{shard_id}.parquet"
            tmp_path = shard_root / f"{shard_id}.{uuid.uuid4().hex}.tmp"
            day_frame.to_parquet(tmp_path, index=False)
            os.replace(tmp_path, shard_path)
            self._merge_raw_shards_for_date(day_value)
            changed_dates.append(day_value)
        return changed_dates

    def _merge_raw_shards_for_date(self, day_value: str) -> Path:
        partition = self.paths.raw_equities / f"date={day_value}"
        shard_root = partition / "shards"
        lock_path = partition / ".merge.lock"
        self._acquire_file_lock(lock_path)
        try:
            frames = [pd.read_parquet(path) for path in sorted(shard_root.glob("*.parquet"))]
            merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            tmp_path = partition / f"data.{uuid.uuid4().hex}.tmp"
            merged.to_parquet(tmp_path, index=False)
            output = partition / "data.parquet"
            os.replace(tmp_path, output)
            return output
        finally:
            with contextlib.suppress(OSError):
                lock_path.unlink()

    @staticmethod
    def _acquire_file_lock(path: Path, *, stale_after_seconds: int = 15) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                return
            except FileExistsError:
                if path.exists() and (datetime.now(tz=UTC).timestamp() - path.stat().st_mtime) > stale_after_seconds:
                    with contextlib.suppress(OSError):
                        path.unlink()
                    continue
                sleep(0.05)

    def _collect_cluster_shard(self, *, trading_date: str, shard: ShardSpec) -> None:
        partition = self.paths.raw_equities / f"date={trading_date}" / "shards" / f"{shard.shard_id}.parquet"
        if partition.exists():
            return
        self.collect_forward_shard(trading_date=trading_date, symbols=shard.symbols, shard_id=shard.shard_id)
