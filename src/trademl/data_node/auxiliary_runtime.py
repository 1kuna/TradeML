"""Auxiliary reference, macro, and QC runtime helpers."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from time import perf_counter, sleep
from typing import Any, Callable

import numpy as np
import pandas as pd

from trademl.connectors.base import (
    BaseConnector,
    BudgetBlockedConnectorError,
    ConnectorError,
    PermanentConnectorError,
    RemoteRateLimitConnectorError,
)
from trademl.data_node.budgets import BudgetDecision
from trademl.connectors.sec_edgar import MissingCompanyfactsError
from trademl.data_node.archive_schema import ARCHIVE_SCHEMAS, normalize_archive_frame
from trademl.data_node.capabilities import (
    auxiliary_capabilities,
    canonical_qc_capabilities,
    default_macro_series,
)
from trademl.data_node.db import DataNodeDB, PlannerTask
from trademl.data_node.planner import plan_auxiliary_tasks
from trademl.fleet.cluster import ClusterCoordinator
from trademl.reference.security_master import rebuild_derived_references

LOGGER = logging.getLogger(__name__)

ENTITLEMENT_FAILURE_MARKERS = (
    "403",
    "402",
    "not_entitled",
    "not entitled",
    "not permitted",
    "forbidden",
    "permission to access",
    "subscription",
)


def _looks_like_entitlement_failure(message: str) -> bool:
    """Return whether a connector error means the current key cannot use a lane."""
    text = str(message or "").lower()
    return any(marker in text for marker in ENTITLEMENT_FAILURE_MARKERS)


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
        collect_reference_data_fn: Callable[
            [list[dict[str, object]]], ReferenceCollectionResult
        ],
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

    def collect_reference_data(
        self, jobs: list[dict[str, object]]
    ) -> ReferenceCollectionResult:
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

        with ThreadPoolExecutor(
            max_workers=max(
                1, sum(lane_widths.get(vendor, 1) for vendor in vendor_jobs)
            )
        ) as executor:
            futures: dict[object, str] = {}
            positions = {vendor: 0 for vendor in vendor_jobs}
            while True:
                scheduled = False
                for vendor, jobs_for_vendor in vendor_jobs.items():
                    width = max(1, lane_widths.get(vendor, 1))
                    active_for_vendor = sum(
                        1
                        for active_vendor in futures.values()
                        if active_vendor == vendor
                    )
                    while active_for_vendor < width and positions[vendor] < len(
                        jobs_for_vendor
                    ):
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
                done, _pending = wait(
                    list(futures), return_when=FIRST_COMPLETED, timeout=0.1
                )
                for future in done:
                    vendor = futures.pop(future)
                    result = future.result()
                    for output in result.outputs:
                        outputs.append(Path(output))
                    for failure in result.failures:
                        failures.append(failure)
                    if result.deferred:
                        deferred[vendor] = deferred.get(vendor, 0) + int(
                            result.deferred
                        )
        if outputs:
            outputs.extend(rebuild_derived_references(self.paths.reference_root))
        deferred_messages = [
            f"{vendor}: deferred {count} jobs after budget exhaustion"
            for vendor, count in sorted(deferred.items())
        ]
        return ReferenceCollectionResult(
            outputs=outputs, failures=failures, deferred=deferred_messages
        )

    def collect_macro_data(
        self, series_ids: list[str], start_date: str, end_date: str
    ) -> list[Path]:
        """Collect FRED macro series partitions."""
        connector = self.connectors["fred"]
        frame = connector.fetch("macros_treasury", series_ids, start_date, end_date)
        outputs: list[Path] = []
        if not frame.empty and "series_id" in frame.columns:
            for series_id, series_frame in frame.groupby("series_id"):
                partition = (
                    self.paths.root
                    / "data"
                    / "raw"
                    / "macros_fred"
                    / f"series={series_id}"
                )
                partition.mkdir(parents=True, exist_ok=True)
                output = partition / "data.parquet"
                series_frame.to_parquet(output, index=False)
                outputs.append(output)
        vintages = connector.fetch("vintagedates", series_ids, start_date, end_date)
        if not vintages.empty and {"series_id", "vintage_date"}.issubset(
            vintages.columns
        ):
            self.paths.reference_root.mkdir(parents=True, exist_ok=True)
            vintage_output = self.paths.reference_root / "fred_vintagedates.parquet"
            if vintage_output.exists():
                existing = pd.read_parquet(vintage_output)
                vintages = pd.concat([existing, vintages], ignore_index=True)
            vintages.sort_values(["series_id", "vintage_date"]).drop_duplicates(
                ["series_id", "vintage_date"]
            ).to_parquet(vintage_output, index=False)
            outputs.append(vintage_output)
        return outputs

    def run_cross_vendor_price_checks(
        self, *, source_name: str, trading_date: str, sample_symbols: list[str]
    ) -> Path:
        """Compare the primary source against backup vendors for a sample of symbols."""
        comparisons: list[pd.DataFrame] = []
        primary = self.connectors[source_name].fetch(
            "equities_eod", sample_symbols, trading_date, trading_date
        )
        for capability in canonical_qc_capabilities(
            connectors=self.connectors,
            audit_state=self.capability_audit_state,
        ):
            vendor = capability.vendor
            if vendor == source_name or vendor not in self.connectors:
                continue
            try:
                backup = self.connectors[vendor].fetch(
                    "equities_eod", sample_symbols, trading_date, trading_date
                )
            except Exception as exc:
                LOGGER.warning(
                    "price_check_backup_failed vendor=%s trading_date=%s error=%s",
                    vendor,
                    trading_date,
                    exc,
                )
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

    def _process_auxiliary_planner_task(
        self, task: PlannerTask, vendor: str
    ) -> list[str]:
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
            **task.payload,
        }
        result = self._execute_auxiliary_job(
            job,
            task_key=task.task_key,
            task_family=task.task_family,
            planner_group=task.planner_group,
        )
        if result.failures:
            permanent = bool(result.state.get("permanent_failure"))
            self.db.mark_planner_task_failed(
                task.task_key,
                error=" | ".join(result.failures),
                backoff_minutes=0 if permanent else 15,
                permanent=permanent,
            )
        elif result.deferred:
            self.db.mark_planner_task_partial(
                task.task_key, error="deferred", backoff_minutes=30
            )
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
            self.db.mark_planner_task_partial(
                task.task_key, error="empty result", backoff_minutes=15
            )
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
            task_families=(
                "security_master",
                "corp_actions",
                "events_filings",
                "macro",
                "supplemental_research",
            )
        )
        planned_macro_series, planned_reference_jobs = self._planned_auxiliary_work(
            trading_date=trading_date
        )
        effective_macro_series = planned_macro_series or list(macro_series_ids or [])
        effective_reference_jobs = planned_reference_jobs or list(reference_jobs or [])
        if (
            not planner_managed_aux
            and effective_macro_series
            and "fred" in self.connectors
            and coordinator.acquire_singleton("macro", trading_date)
        ):
            try:
                self._collect_macro_data_fn(
                    effective_macro_series, trading_date, trading_date
                )
            except ConnectorError:
                LOGGER.exception(
                    "macro collection failed for trading_date=%s", trading_date
                )
            else:
                coordinator.mark_singleton_success(
                    "macro", trading_date, {"series_count": len(effective_macro_series)}
                )

        week_key = self._week_key_fn(current_et)
        reference_bucket = (
            trading_date if self._core_auxiliary_incomplete() else week_key
        )
        if (
            not planner_managed_aux
            and effective_reference_jobs
            and coordinator.acquire_singleton("reference", reference_bucket)
        ):
            materialized_jobs = [
                self._materialize_job_fn(job, trading_date)
                for job in effective_reference_jobs
            ]
            result = self._collect_reference_data_fn(materialized_jobs)
            if isinstance(result, list):
                result = ReferenceCollectionResult(
                    outputs=result, failures=[], deferred=[]
                )
            if result.failures or result.deferred:
                LOGGER.warning(
                    "reference collection partial for bucket=%s trading_date=%s failures=%s deferred=%s",
                    reference_bucket,
                    trading_date,
                    " | ".join(result.failures) if result.failures else "-",
                    " | ".join(result.deferred) if result.deferred else "-",
                )
            if any(
                job.get("output_name") == "corp_actions" for job in materialized_jobs
            ):
                corp_actions_path = (
                    self.paths.root / "data" / "reference" / "corp_actions.parquet"
                )
                if corp_actions_path.exists():
                    self._curate_dates_fn(
                        corp_actions=pd.read_parquet(corp_actions_path)
                    )
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

        if price_check_symbols and coordinator.acquire_singleton(
            "price_checks", week_key
        ):
            try:
                self._run_price_checks_fn(trading_date, price_check_symbols)
            except ConnectorError:
                LOGGER.exception(
                    "price check collection failed for bucket=%s trading_date=%s",
                    week_key,
                    trading_date,
                )
            else:
                coordinator.mark_singleton_success(
                    "price_checks", week_key, {"symbol_count": len(price_check_symbols)}
                )

    def _planned_auxiliary_work(
        self, *, trading_date: str
    ) -> tuple[list[str], list[dict[str, object]]]:
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
                    **dict(task.payload or {}),
                }
            )
        return sorted(set(macro_series)), reference_jobs

    def _core_auxiliary_incomplete(self) -> bool:
        required_reference = {
            "listing_history.parquet",
            "delistings.parquet",
            "corp_actions.parquet",
            "sec_filings.parquet",
            "fred_vintagedates.parquet",
        }
        available_reference = (
            {path.name for path in self.paths.reference_root.glob("*.parquet")}
            if self.paths.reference_root.exists()
            else set()
        )
        if not required_reference.issubset(available_reference):
            return True
        available_macro = {
            path.name.partition("=")[2]
            for path in (self.paths.root / "data" / "raw" / "macros_fred").glob(
                "series=*"
            )
        }
        return not set(default_macro_series()).issubset(available_macro)

    def _aux_lane_widths(
        self, *, task_kinds: set[str], canonical_pressure: bool | None = None
    ) -> dict[str, int]:
        widths: dict[str, int] = {}
        _ = canonical_pressure
        for job in auxiliary_capabilities(
            connectors=self.connectors,
            audit_state=self.capability_audit_state,
            include_research=True,
        ):
            if job.task_kind not in task_kinds:
                continue
            width = self._adaptive_vendor_lane_width(
                vendor=job.vendor, base_width=int(job.lane_width or 1)
            )
            widths[job.vendor] = max(widths.get(job.vendor, 0), width)
        if self.db.has_pending_planner_tasks(
            task_families=("events_filings",), datasets=("companyfacts",)
        ):
            widths["sec_edgar"] = min(max(1, widths.get("sec_edgar", 1)), 1)
        return widths

    def _adaptive_vendor_lane_width(self, *, vendor: str, base_width: int) -> int:
        """Scale vendor concurrency from current RPM/headroom instead of static metadata."""
        connector = self.connectors.get(vendor)
        budget_manager = getattr(connector, "budget_manager", None)
        if budget_manager is None:
            return max(1, int(base_width))
        snapshot = budget_manager.snapshot()
        vendor_payload = (snapshot.get("vendors") or {}).get(vendor, {})
        if not isinstance(vendor_payload, dict):
            return max(1, int(base_width))
        telemetry = vendor_payload.get("telemetry") or {}
        window_counts = (
            telemetry.get("window_counts") if isinstance(telemetry, dict) else {}
        ) or {}
        if int(window_counts.get("remote_rate_limits", 0) or 0):
            return 1
        remaining_window = int(vendor_payload.get("remaining_window_requests", 0) or 0)
        remaining_daily = int(vendor_payload.get("remaining_daily_units", 0) or 0)
        rpm = int(vendor_payload.get("rpm", 0) or 0)
        daily_cap = int(vendor_payload.get("daily_cap", 0) or 0)
        if remaining_window <= 0 or remaining_daily <= 0:
            return 1
        if daily_cap and daily_cap <= 1000:
            return max(1, int(base_width))
        target_utilization = float(
            os.getenv("TRADEML_COLLECTION_TARGET_UTILIZATION", "0.98")
        )
        target_requests = max(1, int(rpm * min(1.0, max(0.1, target_utilization))))
        adaptive = max(1, target_requests // 25)
        cap = 8 if vendor == "alpaca" else 4 if vendor == "tiingo" else 1
        return max(1, min(cap, remaining_window, max(int(base_width), adaptive)))

    def _vendor_has_canonical_pressure(self, vendor: str) -> bool:
        """Return whether a vendor is currently needed by canonical work."""
        page = 0
        limit = 256
        while True:
            tasks = self.db.fetch_planner_tasks(
                task_families=("canonical_bars", "canonical_repair"),
                statuses=("PENDING", "PARTIAL", "FAILED", "LEASED"),
                vendor=vendor,
                limit=limit,
                offset=page * limit,
            )
            if not tasks:
                return False
            if any(vendor in task.eligible_vendors for task in tasks):
                return True
            page += 1

    @staticmethod
    def _expand_reference_jobs(
        jobs: list[dict[str, object]]
    ) -> list[dict[str, object]]:
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
                failures=[
                    f"{vendor}:{job['dataset']}:{job.get('symbols', [])}:missing connector"
                ],
                deferred=0,
                rows_returned=0,
                state={},
            )
        if self._storage_watermark_blocks_low_priority(job):
            return AuxiliaryExecutionResult(
                outputs=[],
                failures=[],
                deferred=1,
                rows_returned=0,
                state={"blocked_reason": "storage_watermark"},
            )
        budget_manager = getattr(connector, "budget_manager", None)
        request_units = self._job_credit_cost(job)
        if budget_manager is not None:
            decision = budget_manager.budget_decision(
                vendor, task_kind="OTHER", units=request_units
            )
        else:
            decision = None
        if decision is not None and not decision.allowed:
            self._mark_budget_blocked_lane(
                vendor=vendor, dataset=str(job["dataset"]), decision=decision
            )
            return AuxiliaryExecutionResult(
                outputs=[],
                failures=[],
                deferred=1,
                rows_returned=0,
                state={
                    "blocked_reason": "budget_exhausted",
                    "blocked_dimension": decision.blocked_dimension,
                },
            )
        effective_task_key = task_key or self._aux_task_key(job)
        leased = self.db.lease_vendor_attempt(
            task_key=effective_task_key,
            task_family=task_family,
            planner_group=str(
                planner_group or job.get("planner_group", "reference_events_backlog")
            ),
            vendor=vendor,
            lease_owner=self.worker_id,
            payload=job,
        )
        if leased is None:
            return AuxiliaryExecutionResult(
                outputs=[], failures=[], deferred=0, rows_returned=0, state={}
            )
        if str(job["dataset"]) in {"macros_treasury", "vintagedates"}:
            return self._execute_macro_auxiliary_job(
                job, task_key=effective_task_key, vendor=vendor
            )
        if vendor == "sec_edgar" and str(job["dataset"]) == "companyfacts":
            return self._execute_sec_companyfacts_job(
                job, task_key=effective_task_key, vendor=vendor
            )
        try:
            frame = connector.fetch(
                str(job["dataset"]),
                list(job.get("symbols", [])),
                str(job["start_date"]),
                str(job["end_date"]),
            )
        except PermanentConnectorError as exc:
            message = str(exc)
            blocked_reason = (
                "entitlement_blocked"
                if _looks_like_entitlement_failure(message)
                else "permanent_failure"
            )
            self.db.mark_vendor_attempt_failed(
                task_key=effective_task_key,
                vendor=vendor,
                error=message,
                backoff_minutes=0,
                permanent=True,
            )
            return AuxiliaryExecutionResult(
                outputs=[],
                failures=[
                    f"{vendor}:{job['dataset']}:{job.get('symbols', [])}:{message}"
                ],
                deferred=0,
                rows_returned=0,
                state={"permanent_failure": True, "blocked_reason": blocked_reason},
            )
        except BudgetBlockedConnectorError as exc:
            self._mark_budget_blocked_lane(
                vendor=vendor,
                dataset=str(job["dataset"]),
                reason=str(exc),
                decision=exc.decision,
            )
            backoff_minutes = self._budget_backoff_minutes(exc.decision)
            self.db.mark_vendor_attempt_failed(
                task_key=effective_task_key,
                vendor=vendor,
                error=str(exc),
                backoff_minutes=backoff_minutes,
            )
            return AuxiliaryExecutionResult(
                outputs=[],
                failures=[],
                deferred=1,
                rows_returned=0,
                state={
                    "blocked_reason": "budget_exhausted",
                    "blocked_dimension": exc.decision.blocked_dimension,
                },
            )
        except RemoteRateLimitConnectorError as exc:
            retry_at = datetime.now(tz=UTC) + timedelta(
                seconds=max(1.0, float(exc.retry_after_seconds or 60.0))
            )
            self._mark_remote_rate_limited_lane(
                vendor=vendor,
                dataset=str(job["dataset"]),
                retry_at=retry_at,
                reason=str(exc),
            )
            self.db.mark_vendor_attempt_failed(
                task_key=effective_task_key,
                vendor=vendor,
                error=str(exc),
                backoff_minutes=max(1, int((retry_at - datetime.now(tz=UTC)).total_seconds() // 60)),
            )
            return AuxiliaryExecutionResult(
                outputs=[],
                failures=[],
                deferred=1,
                rows_returned=0,
                state={"blocked_reason": "remote_rate_limit"},
            )
        except ConnectorError as exc:
            message = str(exc)
            if "budget exhausted" in message:
                decision = None
                if budget_manager is not None:
                    decision = budget_manager.budget_decision(
                        vendor, task_kind="OTHER", units=request_units
                    )
                self._mark_budget_blocked_lane(
                    vendor=vendor,
                    dataset=str(job["dataset"]),
                    reason=message,
                    decision=decision,
                )
                self.db.mark_vendor_attempt_failed(
                    task_key=effective_task_key,
                    vendor=vendor,
                    error=message,
                    backoff_minutes=(
                        self._budget_backoff_minutes(decision)
                        if decision is not None
                        else 1
                    ),
                )
                return AuxiliaryExecutionResult(
                    outputs=[],
                    failures=[],
                    deferred=1,
                    rows_returned=0,
                    state={"blocked_reason": "budget_exhausted"},
                )
            self.db.mark_vendor_attempt_failed(
                task_key=effective_task_key,
                vendor=vendor,
                error=message,
                backoff_minutes=15,
            )
            return AuxiliaryExecutionResult(
                outputs=[],
                failures=[
                    f"{vendor}:{job['dataset']}:{job.get('symbols', [])}:{message}"
                ],
                deferred=0,
                rows_returned=0,
                state={},
            )
        output_name = str(job["output_name"])
        if output_name in {"equities_minute", "ticker_news"}:
            outputs = self._append_partitioned_archive_frame(
                output_name=output_name, frame=frame
            )
        else:
            output = self.paths.reference_root / f"{output_name}.parquet"
            self._append_reference_frame(output, frame)
            outputs = [output]
        self.db.mark_vendor_attempt_success(
            task_key=effective_task_key, vendor=vendor, rows_returned=len(frame)
        )
        return AuxiliaryExecutionResult(
            outputs=[str(path) for path in outputs],
            failures=[],
            deferred=0,
            rows_returned=len(frame),
            state={"outputs": [str(path) for path in outputs]},
        )

    @staticmethod
    def _job_credit_cost(job: dict[str, object]) -> int:
        payload_cost = job.get("credit_cost")
        if payload_cost is not None:
            return max(1, int(payload_cost))
        return max(1, int(job.get("request_units", 1) or 1))

    def _mark_budget_blocked_lane(
        self,
        *,
        vendor: str,
        dataset: str,
        reason: str = "budget exhausted",
        decision: BudgetDecision | None = None,
        cooldown_minutes: int = 1,
    ) -> str:
        """Cool down one auxiliary vendor/dataset lane after budget exhaustion."""
        cooldown_dt = (
            decision.next_eligible_at
            if decision is not None and decision.next_eligible_at is not None
            else datetime.now(tz=UTC) + timedelta(minutes=cooldown_minutes)
        )
        cooldown_until = cooldown_dt.isoformat()
        existing = self.db.get_vendor_lane_health(vendor=vendor, dataset=dataset)
        self.db.upsert_vendor_lane_health(
            vendor=vendor,
            dataset=dataset,
            state="BUDGET_BLOCKED",
            cooldown_until=cooldown_until,
            recent_local_budget_blocks=(
                1
                if existing is None
                else int(existing.recent_local_budget_blocks or 0) + 1
            ),
        )
        self.db.defer_planner_tasks_for_vendor_dataset(
            vendor=vendor,
            dataset=dataset,
            next_eligible_at=cooldown_until,
            error=f"{vendor}:{dataset}: {reason}",
            task_families=(
                "security_master",
                "corp_actions",
                "events_filings",
                "macro",
                "supplemental_research",
            ),
        )
        return cooldown_until

    @staticmethod
    def _budget_backoff_minutes(decision: BudgetDecision | None) -> int:
        """Return a bounded planner backoff from a budget decision."""
        if decision is None or decision.next_eligible_at is None:
            return 1
        seconds = max(
            1.0, (decision.next_eligible_at - datetime.now(tz=UTC)).total_seconds()
        )
        return max(1, int((seconds + 59) // 60))

    def _mark_remote_rate_limited_lane(
        self, *, vendor: str, dataset: str, retry_at: datetime, reason: str
    ) -> str:
        """Cool down a lane after an observed remote throttle."""
        cooldown_until = retry_at.isoformat()
        existing = self.db.get_vendor_lane_health(vendor=vendor, dataset=dataset)
        self.db.upsert_vendor_lane_health(
            vendor=vendor,
            dataset=dataset,
            state="COOLDOWN",
            cooldown_until=cooldown_until,
            recent_remote_429s=(
                1 if existing is None else int(existing.recent_remote_429s or 0) + 1
            ),
        )
        self.db.defer_planner_tasks_for_vendor_dataset(
            vendor=vendor,
            dataset=dataset,
            next_eligible_at=cooldown_until,
            error=f"{vendor}:{dataset}: remote rate limit: {reason}",
            task_families=(
                "security_master",
                "corp_actions",
                "events_filings",
                "macro",
                "supplemental_research",
            ),
        )
        return cooldown_until

    def _storage_watermark_blocks_low_priority(self, job: dict[str, object]) -> bool:
        """Return whether raw archive fillers should pause for disk pressure."""
        if str(job.get("retention_class", "")) != "raw_archive":
            return False
        threshold_text = os.getenv(
            "TRADEML_STORAGE_PAUSE_LOW_PRIORITY_PERCENT", ""
        ).strip()
        if not threshold_text:
            return False
        threshold = float(threshold_text)
        if threshold <= 0:
            return False
        usage = shutil.disk_usage(self.paths.root)
        total = max(1, int(usage.total))
        used_percent = (float(usage.used) / float(total)) * 100.0
        return used_percent >= threshold

    def _execute_macro_auxiliary_job(
        self,
        job: dict[str, object],
        *,
        task_key: str,
        vendor: str,
    ) -> AuxiliaryExecutionResult:
        """Run one macro auxiliary job using the shared outcome contract."""
        try:
            outputs = self._collect_macro_data_fn(
                list(job.get("symbols", [])),
                str(job["start_date"]),
                str(job["end_date"]),
            )
        except ConnectorError as exc:
            self.db.mark_vendor_attempt_failed(
                task_key=task_key, vendor=vendor, error=str(exc), backoff_minutes=15
            )
            return AuxiliaryExecutionResult(
                outputs=[],
                failures=[f"{vendor}:{job['dataset']}:{job.get('symbols', [])}:{exc}"],
                deferred=0,
                rows_returned=0,
                state={},
            )
        row_count = len(list(job.get("symbols", [])))
        self.db.mark_vendor_attempt_success(
            task_key=task_key, vendor=vendor, rows_returned=row_count
        )
        output_paths = [str(path) for path in outputs]
        return AuxiliaryExecutionResult(
            outputs=output_paths,
            failures=[],
            deferred=0,
            rows_returned=row_count,
            state={"dataset": job["dataset"], "outputs": output_paths},
        )

    def _execute_sec_companyfacts_job(
        self,
        job: dict[str, object],
        *,
        task_key: str,
        vendor: str,
    ) -> AuxiliaryExecutionResult:
        """Stream SEC companyfacts one CIK at a time to avoid Pi OOM kills."""
        connector = self.connectors[vendor]
        output_paths: list[Path] = []
        index_rows: list[dict[str, object]] = []
        missing_rows: list[dict[str, object]] = []
        rows_returned = 0
        captured_at = datetime.now().isoformat()
        for cik in [str(symbol) for symbol in job.get("symbols", [])]:
            normalized_cik = cik.zfill(10)
            output = (
                self.paths.reference_root
                / "sec_companyfacts"
                / f"cik={normalized_cik}"
                / "companyfacts.json.gz"
            )
            try:
                stream_fn = getattr(connector, "stream_companyfacts_to_gzip")
                metadata = stream_fn(cik=cik, output=output)
            except MissingCompanyfactsError:
                missing_rows.append(
                    {
                        "cik": normalized_cik,
                        "reason": "sec_companyfacts_404",
                        "captured_at": captured_at,
                        "source": "sec_edgar",
                    }
                )
                continue
            except PermanentConnectorError as exc:
                self.db.mark_vendor_attempt_failed(
                    task_key=task_key,
                    vendor=vendor,
                    error=str(exc),
                    backoff_minutes=0,
                    permanent=True,
                )
                return AuxiliaryExecutionResult(
                    outputs=[str(path) for path in output_paths],
                    failures=[f"{vendor}:{job['dataset']}:{cik}:{exc}"],
                    deferred=0,
                    rows_returned=rows_returned,
                    state={"outputs": [str(path) for path in output_paths]},
                )
            except ConnectorError as exc:
                message = str(exc)
                self.db.mark_vendor_attempt_failed(
                    task_key=task_key,
                    vendor=vendor,
                    error=message,
                    backoff_minutes=30 if "budget exhausted" in message else 15,
                )
                return AuxiliaryExecutionResult(
                    outputs=[str(path) for path in output_paths],
                    failures=[] if "budget exhausted" in message else [f"{vendor}:{job['dataset']}:{cik}:{message}"],
                    deferred=1 if "budget exhausted" in message else 0,
                    rows_returned=rows_returned,
                    state={"outputs": [str(path) for path in output_paths]},
                )
            output_paths.append(output)
            index_rows.append(
                {
                    "cik": normalized_cik,
                    "facts_path": str(output),
                    "raw_bytes": int(metadata.get("raw_bytes", 0) or 0),
                    "captured_at": captured_at,
                    "source": "sec_edgar",
                }
            )
            rows_returned += 1
        if index_rows:
            index_path = self.paths.reference_root / "sec_companyfacts.parquet"
            self._append_sec_companyfacts_index(index_path, pd.DataFrame(index_rows))
            output_paths.append(index_path)
        if missing_rows:
            missing_path = (
                self.paths.reference_root / "sec_companyfacts_missing.parquet"
            )
            self._append_sec_companyfacts_missing(
                missing_path, pd.DataFrame(missing_rows)
            )
            output_paths.append(missing_path)
        self.db.mark_vendor_attempt_success(
            task_key=task_key, vendor=vendor, rows_returned=rows_returned
        )
        return AuxiliaryExecutionResult(
            outputs=[str(path) for path in output_paths],
            failures=[],
            deferred=0,
            rows_returned=rows_returned,
            state={
                "dataset": job["dataset"],
                "outputs": [str(path) for path in output_paths],
            },
        )

    def _append_sec_companyfacts_index(self, output: Path, frame: pd.DataFrame) -> None:
        """Append compact SEC companyfacts index rows without loading raw facts."""
        lock_path = output.with_suffix(".lock")
        self._acquire_file_lock(lock_path)
        try:
            if output.exists():
                existing = pd.read_parquet(output)
                frame = pd.concat([existing, frame], ignore_index=True)
            frame = frame.drop_duplicates(subset=["cik"], keep="last")
            output.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = output.with_suffix(output.suffix + f".{uuid.uuid4().hex}.tmp")
            frame.sort_values(["cik"]).to_parquet(tmp_path, index=False)
            os.replace(tmp_path, output)
        finally:
            with contextlib.suppress(OSError):
                lock_path.unlink()

    def _append_sec_companyfacts_missing(
        self, output: Path, frame: pd.DataFrame
    ) -> None:
        """Append SEC companyfacts missing-CIK records."""
        lock_path = output.with_suffix(".lock")
        self._acquire_file_lock(lock_path)
        try:
            if output.exists():
                existing = pd.read_parquet(output)
                frame = pd.concat([existing, frame], ignore_index=True)
            frame = frame.drop_duplicates(subset=["cik", "reason"], keep="last")
            output.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = output.with_suffix(output.suffix + f".{uuid.uuid4().hex}.tmp")
            frame.sort_values(["cik", "reason"]).to_parquet(tmp_path, index=False)
            os.replace(tmp_path, output)
        finally:
            with contextlib.suppress(OSError):
                lock_path.unlink()

    def _append_reference_frame(self, output: Path, frame: pd.DataFrame) -> None:
        lock_path = output.with_suffix(".lock")
        self._acquire_file_lock(lock_path)
        try:
            if output.exists():
                existing = pd.read_parquet(output)
                frame = (
                    pd.concat([existing, frame], ignore_index=True)
                    if not frame.empty
                    else existing
                )
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
                {
                    str(key): cls._freeze_reference_value(inner)
                    for key, inner in sorted(
                        value.items(), key=lambda item: str(item[0])
                    )
                },
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

    def _append_partitioned_archive_frame(
        self, *, output_name: str, frame: pd.DataFrame
    ) -> list[Path]:
        if frame.empty:
            return []
        frame = normalize_archive_frame(output_name, frame)
        archive_root = self.paths.root / "data" / "raw" / output_name
        written: list[Path] = []
        for day_value, day_frame in frame.groupby("date", dropna=True):
            if pd.isna(day_value):
                continue
            started = perf_counter()
            day_key = pd.Timestamp(day_value).strftime("%Y-%m-%d")
            rows_in = int(len(day_frame))
            output = (
                archive_root
                / f"date={day_key}"
                / "data.parquet"
            )
            lock_path = output.with_suffix(".lock")
            self._acquire_file_lock(lock_path)
            try:
                if output.exists():
                    existing = pd.read_parquet(output)
                    combined = pd.concat(
                        [
                            normalize_archive_frame(output_name, existing),
                            normalize_archive_frame(output_name, day_frame),
                        ],
                        ignore_index=True,
                    )
                else:
                    combined = normalize_archive_frame(output_name, day_frame)
                before_dedupe = int(len(combined))
                combined = self._deduplicate_reference_frame(combined)
                duplicates_dropped = max(0, before_dedupe - int(len(combined)))
                output.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = output.with_suffix(
                    output.suffix + f".{uuid.uuid4().hex}.tmp"
                )
                combined.to_parquet(tmp_path, index=False)
                os.replace(tmp_path, output)
                self.db.record_archive_write_telemetry(
                    output_name=output_name,
                    partition_date=day_key,
                    status="success",
                    rows_in=rows_in,
                    rows_written=int(len(combined)),
                    duplicates_dropped=duplicates_dropped,
                    coerced_columns=self._archive_coerced_columns(output_name, combined),
                    schema_mismatch=False,
                    duration_ms=round((perf_counter() - started) * 1000.0, 3),
                )
                written.append(output)
            except Exception as exc:
                self.db.record_archive_write_telemetry(
                    output_name=output_name,
                    partition_date=day_key,
                    status="failed",
                    rows_in=rows_in,
                    rows_written=0,
                    duplicates_dropped=0,
                    coerced_columns=self._archive_coerced_columns(output_name, day_frame),
                    schema_mismatch=True,
                    error=str(exc),
                    duration_ms=round((perf_counter() - started) * 1000.0, 3),
                )
                raise
            finally:
                with contextlib.suppress(OSError):
                    lock_path.unlink()
        return written

    @staticmethod
    def _archive_coerced_columns(output_name: str, frame: pd.DataFrame) -> list[str]:
        schema = ARCHIVE_SCHEMAS.get(output_name)
        if schema is None:
            return []
        expected = set(schema.string_columns) | set(schema.timestamp_columns) | {schema.date_column}
        return sorted(column for column in frame.columns if column in expected)

    @staticmethod
    def _acquire_file_lock(path: Path, *, stale_after_seconds: int = 15) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                return
            except FileExistsError:
                try:
                    mtime = path.stat().st_mtime
                except FileNotFoundError:
                    continue
                if (datetime.now(tz=UTC).timestamp() - mtime) > stale_after_seconds:
                    with contextlib.suppress(OSError):
                        path.unlink()
                    continue
                sleep(0.05)
