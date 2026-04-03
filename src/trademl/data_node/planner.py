"""Coverage planner for canonical, reference, and supplemental tasks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from trademl.data_node.capabilities import (
    VendorCapability,
    auxiliary_capabilities,
    backfill_capabilities,
    default_macro_series,
)
from trademl.data_node.db import BackfillTask, VendorAttempt


@dataclass(slots=True, frozen=True)
class PlannedTask:
    """Planner-emitted deterministic task."""

    task_key: str
    task_family: str
    planner_group: str
    dataset: str
    tier: str
    priority: int
    start_date: str
    end_date: str
    symbols: tuple[str, ...]
    output_name: str | None
    preferred_vendors: tuple[str, ...]
    payload: dict[str, Any]


def canonical_task_key(task: BackfillTask) -> str:
    """Return the deterministic canonical task key."""
    symbol = task.symbol or "__ALL__"
    return f"canonical::{task.dataset}::{symbol}::{task.start_date}::{task.end_date}::{task.kind}"


def choose_vendor_for_canonical_task(
    *,
    task: BackfillTask,
    connectors: dict[str, object],
    audit_state: dict[str, Any] | None,
    attempts: list[VendorAttempt],
    now: datetime | None = None,
) -> str | None:
    """Choose the next eligible vendor for a canonical task."""
    current_time = (now or datetime.now(tz=UTC)).isoformat()
    blocked: set[str] = set()
    for attempt in attempts:
        if attempt.status in {"SUCCESS", "PERMANENT_FAILED"}:
            blocked.add(attempt.vendor)
            continue
        if attempt.status == "LEASED" and attempt.lease_expires_at and attempt.lease_expires_at > current_time:
            blocked.add(attempt.vendor)
            continue
        if attempt.status == "FAILED" and attempt.next_eligible_at and attempt.next_eligible_at > current_time:
            blocked.add(attempt.vendor)
    for capability in backfill_capabilities(dataset=task.dataset, connectors=connectors, audit_state=audit_state):
        if capability.vendor not in blocked:
            return capability.vendor
    return None


def plan_auxiliary_tasks(
    *,
    data_root: Path,
    stage_symbols: list[str],
    stage_years: int,
    connectors: dict[str, object],
    audit_state: dict[str, Any] | None = None,
    include_research: bool = False,
    current_date: str | None = None,
) -> list[PlannedTask]:
    """Plan deterministic auxiliary tasks based on enabled capabilities and current corpus scope."""
    if not stage_symbols:
        return []
    current_ts = pd.Timestamp(current_date or datetime.now(tz=UTC).date().isoformat()).normalize()
    start_date = (current_ts - pd.DateOffset(years=max(1, int(stage_years)))).strftime("%Y-%m-%d")
    end_date = current_ts.strftime("%Y-%m-%d")
    tasks: list[PlannedTask] = []
    company_ticker_map = _load_sec_company_tickers(data_root / "data" / "reference" / "sec_company_tickers.parquet")
    for capability in auxiliary_capabilities(connectors=connectors, audit_state=audit_state, include_research=include_research):
        task_specs = _materialize_capability_tasks(
            capability=capability,
            stage_symbols=stage_symbols,
            company_ticker_map=company_ticker_map,
            start_date=start_date,
            end_date=end_date,
        )
        tasks.extend(task_specs)
    return sorted(tasks, key=lambda task: (task.priority, task.task_key))


def _materialize_capability_tasks(
    *,
    capability: VendorCapability,
    stage_symbols: list[str],
    company_ticker_map: dict[str, str],
    start_date: str,
    end_date: str,
) -> list[PlannedTask]:
    if capability.task_kind == "MACRO":
        return [
            PlannedTask(
                task_key=f"{capability.task_kind.lower()}::{capability.dataset}::{series_id}::{start_date}::{end_date}",
                task_family="macro",
                planner_group=capability.planner_group,
                dataset=capability.dataset,
                tier=capability.tier,
                priority=capability.priority,
                start_date=start_date,
                end_date=end_date,
                symbols=(series_id,),
                output_name=capability.output_name,
                preferred_vendors=(capability.vendor,),
                payload={"capability_id": capability.capability_id, "series_id": series_id},
            )
            for series_id in default_macro_series()
        ]

    if capability.batching_mode == "global":
        return [
            PlannedTask(
                task_key=f"{capability.task_kind.lower()}::{capability.dataset}::{start_date}::{end_date}",
                task_family="auxiliary",
                planner_group=capability.planner_group,
                dataset=capability.dataset,
                tier=capability.tier,
                priority=capability.priority,
                start_date=start_date,
                end_date=end_date,
                symbols=(),
                output_name=capability.output_name,
                preferred_vendors=(capability.vendor,),
                payload={"capability_id": capability.capability_id},
            )
        ]

    if capability.vendor == "sec_edgar" and capability.dataset in {"filing_index", "companyfacts"}:
        symbols = [company_ticker_map[symbol] for symbol in stage_symbols if symbol in company_ticker_map]
    else:
        symbols = list(stage_symbols)
    if not symbols:
        return []
    chunk_size = capability.max_symbols_per_run or len(symbols)
    tasks: list[PlannedTask] = []
    for chunk_index in range(0, len(symbols), chunk_size):
        chunk = tuple(symbols[chunk_index : chunk_index + chunk_size])
        chunk_id = f"{chunk_index // chunk_size:03d}"
        tasks.append(
            PlannedTask(
                task_key=f"{capability.task_kind.lower()}::{capability.dataset}::{chunk_id}::{start_date}::{end_date}",
                task_family="auxiliary",
                planner_group=capability.planner_group,
                dataset=capability.dataset,
                tier=capability.tier,
                priority=capability.priority,
                start_date=start_date,
                end_date=end_date,
                symbols=chunk,
                output_name=capability.output_name,
                preferred_vendors=(capability.vendor,),
                payload={"capability_id": capability.capability_id, "chunk_id": chunk_id},
            )
        )
    return tasks


def training_readiness(
    *,
    raw_green_ratio: float | None,
    has_corp_actions: bool,
    has_listing_history: bool,
    has_delistings: bool,
    has_sec_filings: bool,
    has_macro_vintages: bool,
    macro_series_count: int,
    required_macro_series: int,
) -> dict[str, Any]:
    """Evaluate whether the corpus is ready for Phase 1 training."""
    blockers: list[str] = []
    if raw_green_ratio is None or raw_green_ratio < 0.98:
        blockers.append("canonical_eod_bars")
    if not has_corp_actions:
        blockers.append("corp_actions")
    if not has_listing_history:
        blockers.append("listing_history")
    if not has_delistings:
        blockers.append("delistings")
    if not has_sec_filings:
        blockers.append("sec_filings")
    if not has_macro_vintages:
        blockers.append("macro_vintages")
    if macro_series_count < required_macro_series:
        blockers.append("macro_pack")
    return {"ready": not blockers, "blockers": blockers}


def _load_sec_company_tickers(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    frame = pd.read_parquet(path)
    if frame.empty:
        return {}
    ticker_col = "ticker" if "ticker" in frame.columns else "symbol"
    cik_col = "cik_str" if "cik_str" in frame.columns else "cik"
    return {
        str(row[ticker_col]).strip().upper(): str(row[cik_col]).strip()
        for row in frame[[ticker_col, cik_col]].dropna().to_dict("records")
    }
