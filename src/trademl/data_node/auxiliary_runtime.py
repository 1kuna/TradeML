"""Auxiliary reference, macro, and QC runtime helpers."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import sleep
from typing import Any, Callable

import numpy as np
import pandas as pd

from trademl.connectors.base import BaseConnector, ConnectorError, PermanentConnectorError
from trademl.data_node.capabilities import auxiliary_capabilities, canonical_qc_capabilities, default_macro_series, vendor_profile
from trademl.data_node.db import DataNodeDB, PlannerTask
from trademl.data_node.planner import plan_auxiliary_tasks
from trademl.fleet.cluster import ClusterCoordinator
from trademl.reference.security_master import rebuild_derived_references

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ReferenceCollectionResult:
    """Outcome of a reference-data collection pass."""

    outputs: list[Path]
    failures: list[str]
    deferred: list[str]


@dataclass(slots=True)
class AuxiliaryExecutionResult:
    """Normalized outcome for planner or ad hoc auxiliary work."""

    outputs: list[str]
    failures: list[str]
    deferred: int
    rows_returned: int
    state: dict[str, object]


class AuxiliaryRuntime:
    """Encapsulate auxiliary collection, reference persistence, and QC work."""

    def __init__(
        self,
        *,
        db: DataNodeDB,
        connectors: dict[str, BaseConnector],
        paths: Any,
        capability_audit_state: dict[str, object],
        worker_id: str,
        default_symbols_getter: Callable[[], list[str]],
        stage_years_getter: Callable[[], int],
        materialize_job_fn: Callable[[dict[str, object], str], dict[str, object]],
        week_key_fn: Callable[[datetime], str],
        curate_dates_fn: Callable[..., Any],
        collect_macro_data_fn: Callable[[list[str], str, str], list[Path]],
        collect_reference_data_fn: Callable[[list[dict[str, object]]], ReferenceCollectionResult],
        run_price_checks_fn: Callable[[str, list[str]], Path],
    ) -> None:
        self.db = db
        self.connectors = connectors
        self.paths = paths
        self.capability_audit_state = capability_audit_state
        self.worker_id = worker_id
        self._default_symbols_getter = default_symbols_getter
        self._stage_years_getter = stage_years_getter
        self._materialize_job_fn = materialize_job_fn
        self._week_key_fn = week_key_fn
        self._curate_dates_fn = curate_dates_fn
        self._collect_macro_data_fn = collect_macro_data_fn
        self._collect_reference_data_fn = collect_reference_data_fn
        self._run_price_checks_fn = run_price_checks_fn

    def collect_reference_data(self, jobs: list[dict[str, object]]) -> ReferenceCollectionResult:
        """Collect reference datasets into parquet files."""
        outputs: list[Path] = []
        self.paths.reference_root.mkdir(parents=True, exist_ok=True)
        failures: list[str] = []
        deferred: dict[str, int] = {}
        expanded_jobs = self._expand_reference_jobs(jobs)
        lane_widths = self._aux_lane_widths(task_kinds={"REFERENCE", "EVENT"})
        vendor_jobs: dict[str, list[dict[str, object]]] = {}
        for job in expanded_jobs:
            vendor_jobs.setdefault(str(job["source"]), []).append(job)

        from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

        with ThreadPoolExecutor(max_workers=max(1, sum(lane_widths.get(vendor, 1) for vendor in vendor_jobs))) as executor:
            futures: dict[object, str] = {}
            positions = {vendor: 0 for vendor in vendor_jobs}
            while True:
                scheduled = False
                for vendor, jobs_for_vendor in vendor_jobs.items():
                    width = max(1, lane_widths.get(vendor, 1))
                    active_for_vendor = sum(1 for active_vendor in futures.values() if active_vendor == vendor)
                    while active_for_vendor < width and positions[vendor] < len(jobs_for_vendor):
                        job = jobs_for_vendor[positions[vendor]]
                        positions[vendor] += 1
                        future = executor.submit(self._execute_auxiliary_job, job)
                        futures[future] = vendor
                        active_for_vendor += 1
                        scheduled = True
                if not futures and not scheduled:
                    break
                if not futures:
                    continue
                done, _pending = wait(list(futures), return_when=FIRST_COMPLETED, timeout=0.1)
                for future in done:
                    vendor = futures.pop(future)
                    result = future.result()
                    for output in result.outputs:
                        outputs.append(Path(output))
                    for failure in result.failures:
                        failures.append(failure)
                    if result.deferred:
                        deferred[vendor] = deferred.get(vendor, 0) + int(result.deferred)
        outputs.extend(rebuild_derived_references(self.paths.reference_root))
        deferred_messages = [f"{vendor}: deferred {count} jobs after budget exhaustion" for vendor, count in sorted(deferred.items())]
        return ReferenceCollectionResult(outputs=outputs, failures=failures, deferred=deferred_messages)

    def collect_macro_data(self, series_ids: list[str], start_date: str, end_date: str) -> list[Path]:
        """Collect FRED macro series partitions."""
        connector = self.connectors["fred"]
        frame = connector.fetch("macros_treasury", series_ids, start_date, end_date)
        outputs: list[Path] = []
        if not frame.empty and "series_id" in frame.columns:
            for series_id, series_frame in frame.groupby("series_id"):
                partition = self.paths.root / "data" / "raw" / "macros_fred" / f"series={series_id}"
                partition.mkdir(parents=True, exist_ok=True)
                output = partition / "data.parquet"
                series_frame.to_parquet(output, index=False)
                outputs.append(output)
        vintages = connector.fetch("vintagedates", series_ids, start_date, end_date)
        if not vintages.empty and {"series_id", "vintage_date"}.issubset(vintages.columns):
            self.paths.reference_root.mkdir(parents=True, exist_ok=True)
            vintage_output = self.paths.reference_root / "fred_vintagedates.parquet"
            if vintage_output.exists():
                existing = pd.read_parquet(vintage_output)
                vintages = pd.concat([existing, vintages], ignore_index=True)
            vintages.sort_values(["series_id", "vintage_date"]).drop_duplicates(["series_id", "vintage_date"]).to_parquet(vintage_output, index=False)
            outputs.append(vintage_output)
        return outputs

    def run_cross_vendor_price_checks(self, *, source_name: str, trading_date: str, sample_symbols: list[str]) -> Path:
        """Compare the primary source against backup vendors for a sample of symbols."""
        comparisons: list[pd.DataFrame] = []
        primary = self.connectors[source_name].fetch("equities_eod", sample_symbols, trading_date, trading_date)
        for capability in canonical_qc_capabilities(
            connectors=self.connectors,
            audit_state=self.capability_audit_state,
        ):
            vendor = capability.vendor
            if vendor == source_name or vendor not in self.connectors:
                continue
            try:
                backup = self.connectors[vendor].fetch("equities_eod", sample_symbols, trading_date, trading_date)
            except Exception as exc:
                LOGGER.warning("price_check_backup_failed vendor=%s trading_date=%s error=%s", vendor, trading_date, exc)
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

    def _process_auxiliary_planner_task(self, task: PlannerTask, vendor: str) -> list[str]:
        """Run a single planner-native auxiliary task."""
        job = {
            "source": vendor,
            "dataset": task.dataset,
            "symbols": list(task.symbols),
            "start_date": task.start_date,
            "end_date": task.end_date,
            "output_name": task.output_name or task.dataset,
            "planner_group": task.planner_group,
            "explode_symbols": False,
            "tier": task.tier,
        }
        result = self._execute_auxiliary_job(
            job,
            task_key=task.task_key,
            task_family=task.task_family,
            planner_group=task.planner_group,
        )
        if result.failures:
            self.db.mark_planner_task_failed(task.task_key, error=" | ".join(result.failures), backoff_minutes=15)
        elif result.deferred:
            self.db.mark_planner_task_partial(task.task_key, error="deferred", backoff_minutes=30)
        elif result.outputs:
            self.db.update_planner_task_progress(
                task_key=task.task_key,
                expected_units=1,
                completed_units=1,
                remaining_units=0,
                state=result.state,
            )
            self.db.mark_planner_task_success(task.task_key)
        else:
            self.db.mark_planner_task_partial(task.task_key, error="empty result", backoff_minutes=15)
        return []

    def _run_cluster_auxiliary_tasks(
        self,
        *,
        coordinator: ClusterCoordinator,
        trading_date: str,
        current_et: datetime,
        macro_series_ids: list[str] | None,
        reference_jobs: list[dict[str, object]] | None,
        price_check_symbols: list[str] | None,
        source_name: str,
    ) -> None:
        planner_managed_aux = self.db.has_pending_planner_tasks(
            task_families=("security_master", "corp_actions", "events_filings", "macro", "supplemental_research")
        )
        planned_macro_series, planned_reference_jobs = self._planned_auxiliary_work(trading_date=trading_date)
        effective_macro_series = planned_macro_series or list(macro_series_ids or [])
        effective_reference_jobs = planned_reference_jobs or list(reference_jobs or [])
        if not planner_managed_aux and effective_macro_series and "fred" in self.connectors and coordinator.acquire_singleton("macro", trading_date):
            try:
                self._collect_macro_data_fn(effective_macro_series, trading_date, trading_date)
            except ConnectorError:
                LOGGER.exception("macro collection failed for trading_date=%s", trading_date)
            else:
                coordinator.mark_singleton_success("macro", trading_date, {"series_count": len(effective_macro_series)})

        week_key = self._week_key_fn(current_et)
        reference_bucket = trading_date if self._core_auxiliary_incomplete() else week_key
        if not planner_managed_aux and effective_reference_jobs and coordinator.acquire_singleton("reference", reference_bucket):
            materialized_jobs = [self._materialize_job_fn(job, trading_date) for job in effective_reference_jobs]
            result = self._collect_reference_data_fn(materialized_jobs)
            if isinstance(result, list):
                result = ReferenceCollectionResult(outputs=result, failures=[], deferred=[])
            if result.failures or result.deferred:
                LOGGER.warning(
                    "reference collection partial for bucket=%s trading_date=%s failures=%s deferred=%s",
                    reference_bucket,
                    trading_date,
                    " | ".join(result.failures) if result.failures else "-",
                    " | ".join(result.deferred) if result.deferred else "-",
                )
            if any(job.get("output_name") == "corp_actions" for job in materialized_jobs):
                corp_actions_path = self.paths.root / "data" / "reference" / "corp_actions.parquet"
                if corp_actions_path.exists():
                    self._curate_dates_fn(corp_actions=pd.read_parquet(corp_actions_path))
            coordinator.mark_singleton_success(
                "reference",
                reference_bucket,
                {
                    "job_count": len(materialized_jobs),
                    "output_count": len(result.outputs),
                    "failure_count": len(result.failures),
                    "deferred_count": len(result.deferred),
                },
            )

        if price_check_symbols and coordinator.acquire_singleton("price_checks", week_key):
            try:
                self._run_price_checks_fn(trading_date, price_check_symbols)
            except ConnectorError:
                LOGGER.exception("price check collection failed for bucket=%s trading_date=%s", week_key, trading_date)
            else:
                coordinator.mark_singleton_success("price_checks", week_key, {"symbol_count": len(price_check_symbols)})

    def _planned_auxiliary_work(self, *, trading_date: str) -> tuple[list[str], list[dict[str, object]]]:
        default_symbols = self._default_symbols_getter()
        if not default_symbols:
            return [], []
        planned = plan_auxiliary_tasks(
            data_root=self.paths.root,
            stage_symbols=default_symbols,
            stage_years=self._stage_years_getter(),
            connectors=self.connectors,
            audit_state=self.capability_audit_state,
            include_research=True,
            current_date=trading_date,
        )
        macro_series: list[str] = []
        reference_jobs: list[dict[str, object]] = []
        for task in planned:
            if task.dataset in {"macros_treasury", "vintagedates"}:
                macro_series.extend(list(task.symbols))
                continue
            if task.task_family != "auxiliary" or not task.preferred_vendors:
                continue
            reference_jobs.append(
                {
                    "source": task.preferred_vendors[0],
                    "dataset": task.dataset,
                    "symbols": list(task.symbols),
                    "output_name": task.output_name or task.dataset,
                    "planner_group": task.planner_group,
                    "explode_symbols": False,
                    "tier": task.tier,
                }
            )
        return sorted(set(macro_series)), reference_jobs

    def _core_auxiliary_incomplete(self) -> bool:
        required_reference = {"listing_history.parquet", "delistings.parquet", "corp_actions.parquet", "sec_filings.parquet", "fred_vintagedates.parquet"}
        available_reference = {path.name for path in self.paths.reference_root.glob("*.parquet")} if self.paths.reference_root.exists() else set()
        if not required_reference.issubset(available_reference):
            return True
        available_macro = {path.name.partition("=")[2] for path in (self.paths.root / "data" / "raw" / "macros_fred").glob("series=*")}
        return not set(default_macro_series()).issubset(available_macro)

    def _aux_lane_widths(self, *, task_kinds: set[str]) -> dict[str, int]:
        widths: dict[str, int] = {}
        canonical_pressure = self.db.has_pending_planner_tasks(task_families=("canonical_bars",)) or self.db.has_pending_backfill()
        for job in auxiliary_capabilities(
            connectors=self.connectors,
            audit_state=self.capability_audit_state,
            include_research=True,
        ):
            if job.task_kind not in task_kinds:
                continue
            profile = vendor_profile(job.vendor)
            if (
                canonical_pressure
                and job.task_kind != "RESEARCH_ONLY"
                and profile is not None
                and profile.saturation_policy in {"canonical_first", "canonical_only"}
            ):
                continue
            width = int(job.lane_width or 1)
            if canonical_pressure and job.task_kind == "RESEARCH_ONLY":
                width = 1
            widths[job.vendor] = max(widths.get(job.vendor, 0), width)
        return widths

    @staticmethod
    def _expand_reference_jobs(jobs: list[dict[str, object]]) -> list[dict[str, object]]:
        expanded: list[dict[str, object]] = []
        for job in jobs:
            symbols = list(job.get("symbols", []))
            if not job.get("explode_symbols", True) or len(symbols) <= 1:
                expanded.append(dict(job))
                continue
            for symbol in symbols:
                item = dict(job)
                item["symbols"] = [symbol]
                expanded.append(item)
        return expanded

    def _execute_auxiliary_job(
        self,
        job: dict[str, object],
        *,
        task_key: str | None = None,
        task_family: str = "auxiliary",
        planner_group: str | None = None,
    ) -> AuxiliaryExecutionResult:
        """Run one auxiliary job with shared attempt, defer, and success semantics."""
        vendor = str(job["source"])
        connector = self.connectors.get(vendor)
        if connector is None:
            return AuxiliaryExecutionResult(
                outputs=[],
                failures=[f"{vendor}:{job['dataset']}:{job.get('symbols', [])}:missing connector"],
                deferred=0,
                rows_returned=0,
                state={},
            )
        budget_manager = getattr(connector, "budget_manager", None)
        if budget_manager is not None and not budget_manager.can_spend(vendor, task_kind="OTHER"):
            return AuxiliaryExecutionResult(outputs=[], failures=[], deferred=1, rows_returned=0, state={})
        effective_task_key = task_key or self._aux_task_key(job)
        leased = self.db.lease_vendor_attempt(
            task_key=effective_task_key,
            task_family=task_family,
            planner_group=str(planner_group or job.get("planner_group", "reference_events_backlog")),
            vendor=vendor,
            lease_owner=self.worker_id,
            payload=job,
        )
        if leased is None:
            return AuxiliaryExecutionResult(outputs=[], failures=[], deferred=0, rows_returned=0, state={})
        if str(job["dataset"]) in {"macros_treasury", "vintagedates"}:
            return self._execute_macro_auxiliary_job(job, task_key=effective_task_key, vendor=vendor)
        try:
            frame = connector.fetch(
                str(job["dataset"]),
                list(job.get("symbols", [])),
                str(job["start_date"]),
                str(job["end_date"]),
            )
        except PermanentConnectorError as exc:
            self.db.mark_vendor_attempt_failed(task_key=effective_task_key, vendor=vendor, error=str(exc), backoff_minutes=0, permanent=True)
            return AuxiliaryExecutionResult(
                outputs=[],
                failures=[f"{vendor}:{job['dataset']}:{job.get('symbols', [])}:{exc}"],
                deferred=0,
                rows_returned=0,
                state={},
            )
        except ConnectorError as exc:
            message = str(exc)
            if "budget exhausted" in message:
                self.db.mark_vendor_attempt_failed(task_key=effective_task_key, vendor=vendor, error=message, backoff_minutes=30)
                return AuxiliaryExecutionResult(outputs=[], failures=[], deferred=1, rows_returned=0, state={})
            self.db.mark_vendor_attempt_failed(task_key=effective_task_key, vendor=vendor, error=message, backoff_minutes=15)
            return AuxiliaryExecutionResult(
                outputs=[],
                failures=[f"{vendor}:{job['dataset']}:{job.get('symbols', [])}:{message}"],
                deferred=0,
                rows_returned=0,
                state={},
            )
        output_name = str(job["output_name"])
        if output_name in {"equities_minute", "ticker_news"}:
            outputs = self._append_partitioned_archive_frame(output_name=output_name, frame=frame)
        else:
            output = self.paths.reference_root / f"{output_name}.parquet"
            self._append_reference_frame(output, frame)
            outputs = [output]
        self.db.mark_vendor_attempt_success(task_key=effective_task_key, vendor=vendor, rows_returned=len(frame))
        return AuxiliaryExecutionResult(
            outputs=[str(path) for path in outputs],
            failures=[],
            deferred=0,
            rows_returned=len(frame),
            state={"outputs": [str(path) for path in outputs]},
        )

    def _execute_macro_auxiliary_job(
        self,
        job: dict[str, object],
        *,
        task_key: str,
        vendor: str,
    ) -> AuxiliaryExecutionResult:
        """Run one macro auxiliary job using the shared outcome contract."""
        try:
            outputs = self._collect_macro_data_fn(list(job.get("symbols", [])), str(job["start_date"]), str(job["end_date"]))
        except ConnectorError as exc:
            self.db.mark_vendor_attempt_failed(task_key=task_key, vendor=vendor, error=str(exc), backoff_minutes=15)
            return AuxiliaryExecutionResult(
                outputs=[],
                failures=[f"{vendor}:{job['dataset']}:{job.get('symbols', [])}:{exc}"],
                deferred=0,
                rows_returned=0,
                state={},
            )
        row_count = len(list(job.get("symbols", [])))
        self.db.mark_vendor_attempt_success(task_key=task_key, vendor=vendor, rows_returned=row_count)
        output_paths = [str(path) for path in outputs]
        return AuxiliaryExecutionResult(
            outputs=output_paths,
            failures=[],
            deferred=0,
            rows_returned=row_count,
            state={"dataset": job["dataset"], "outputs": output_paths},
        )

    def _append_reference_frame(self, output: Path, frame: pd.DataFrame) -> None:
        lock_path = output.with_suffix(".lock")
        self._acquire_file_lock(lock_path)
        try:
            if output.exists():
                existing = pd.read_parquet(output)
                frame = pd.concat([existing, frame], ignore_index=True) if not frame.empty else existing
            if not frame.empty:
                frame = self._deduplicate_reference_frame(frame)
            output.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = output.with_suffix(output.suffix + f".{uuid.uuid4().hex}.tmp")
            frame.to_parquet(tmp_path, index=False)
            os.replace(tmp_path, output)
        finally:
            with contextlib.suppress(OSError):
                lock_path.unlink()

    @classmethod
    def _deduplicate_reference_frame(cls, frame: pd.DataFrame) -> pd.DataFrame:
        """Drop duplicate reference rows, including nested object-valued cells."""
        key_frame = frame.copy()
        for column in key_frame.columns:
            if key_frame[column].dtype == "object":
                key_frame[column] = key_frame[column].map(cls._freeze_reference_value)
        duplicate_mask = key_frame.duplicated(keep="first")
        return frame.loc[~duplicate_mask].reset_index(drop=True)

    @classmethod
    def _freeze_reference_value(cls, value: object) -> object:
        """Convert nested reference payload values into hashable, deterministic keys."""
        if isinstance(value, np.ndarray):
            return tuple(cls._freeze_reference_value(item) for item in value.tolist())
        if isinstance(value, np.generic):
            return cls._freeze_reference_value(value.item())
        if isinstance(value, dict):
            return json.dumps(
                {str(key): cls._freeze_reference_value(inner) for key, inner in sorted(value.items(), key=lambda item: str(item[0]))},
                sort_keys=True,
                separators=(",", ":"),
            )
        if isinstance(value, (list, tuple)):
            return tuple(cls._freeze_reference_value(item) for item in value)
        if isinstance(value, set):
            return tuple(sorted(cls._freeze_reference_value(item) for item in value))
        return value

    @staticmethod
    def _aux_task_key(job: dict[str, object]) -> str:
        symbols = ",".join(list(job.get("symbols", []))) or "__GLOBAL__"
        return "::".join(
            [
                "aux",
                str(job.get("source", "")),
                str(job.get("dataset", "")),
                symbols,
                str(job.get("start_date", "")),
                str(job.get("end_date", "")),
                str(job.get("output_name", "")),
            ]
        )

    def _append_partitioned_archive_frame(self, *, output_name: str, frame: pd.DataFrame) -> list[Path]:
        if frame.empty:
            return []
        archive_root = self.paths.root / "data" / "raw" / output_name
        written: list[Path] = []
        for day_value, day_frame in frame.groupby("date", dropna=True):
            if pd.isna(day_value):
                continue
            output = archive_root / f"date={pd.Timestamp(day_value).strftime('%Y-%m-%d')}" / "data.parquet"
            lock_path = output.with_suffix(".lock")
            self._acquire_file_lock(lock_path)
            try:
                if output.exists():
                    existing = pd.read_parquet(output)
                    combined = pd.concat([existing, day_frame], ignore_index=True)
                else:
                    combined = day_frame.copy()
                combined = self._deduplicate_reference_frame(combined)
                output.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = output.with_suffix(output.suffix + f".{uuid.uuid4().hex}.tmp")
                combined.to_parquet(tmp_path, index=False)
                os.replace(tmp_path, output)
                written.append(output)
            finally:
                with contextlib.suppress(OSError):
                    lock_path.unlink()
        return written

    @staticmethod
    def _acquire_file_lock(path: Path, *, stale_after_seconds: int = 15) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                return
            except FileExistsError:
                if path.exists() and (datetime.now().timestamp() - path.stat().st_mtime) > stale_after_seconds:
                    with contextlib.suppress(OSError):
                        path.unlink()
                    continue
                sleep(0.05)
