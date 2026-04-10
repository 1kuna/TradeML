"""Coverage planner for canonical, reference, and supplemental tasks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from trademl.calendars.exchange import get_trading_days
from trademl.data_node.capabilities import (
    VendorCapability,
    auxiliary_capabilities,
    backfill_capabilities,
    default_macro_series,
)
from trademl.data_node.db import BackfillTask, VendorAttempt
from trademl.data_node.provider_contracts import dataset_contract


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

    @property
    def scope_kind(self) -> str:
        """Return the planner task scope kind."""
        return str(self.payload.get("scope_kind", "generic"))


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
        if not _capability_supports_canonical_task(capability, task):
            continue
        if not _vendor_has_local_budget(
            connector=connectors.get(capability.vendor),
            vendor=capability.vendor,
            dataset=task.dataset,
            symbol_count=1 if task.symbol is not None else int(getattr(capability, "preferred_batch_size", 1) or 1),
        ):
            continue
        if capability.vendor not in blocked:
            return capability.vendor
    return None


def _capability_supports_canonical_task(capability: VendorCapability, task: BackfillTask) -> bool:
    """Return whether a capability can safely execute the leased canonical task shape."""
    if task.symbol is None and capability.batching_mode == "single_symbol":
        return False
    return True


def _vendor_has_local_budget(*, connector: object | None, vendor: str, dataset: str, symbol_count: int) -> bool:
    """Return whether the connector's local budget manager can spend for this task shape now."""
    budget_manager = getattr(connector, "budget_manager", None)
    if budget_manager is None:
        return True
    can_spend = getattr(budget_manager, "can_spend", None)
    if not callable(can_spend):
        return True
    contract = dataset_contract(vendor, dataset)
    request_units = max(1, int(getattr(contract, "request_cost_units", 1) or 1))
    if contract is not None and str(contract.request_cost_basis) == "symbol":
        request_units *= max(1, int(symbol_count))
    return bool(can_spend(vendor, task_kind="FORWARD", units=request_units))


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


def plan_canonical_bar_tasks(
    *,
    data_root: Path | None = None,
    stage_symbols: list[str],
    stage_years: int,
    connectors: dict[str, object],
    audit_state: dict[str, Any] | None = None,
    current_date: str | None = None,
    freeze_report_date: str | None = None,
    symbol_batch_size: int = 1,
    trading_day_chunk_size: int = 20,
) -> list[PlannedTask]:
    """Build atomic symbol-range/date-range canonical bar tasks decoupled from storage partitions."""
    if not stage_symbols or stage_years <= 0:
        return []
    as_of = pd.Timestamp(current_date or datetime.now(tz=UTC).date().isoformat()).normalize()
    nominal_start = (as_of - pd.DateOffset(years=max(1, int(stage_years)))).normalize()
    freeze_end = pd.Timestamp(freeze_report_date).normalize() if freeze_report_date else None
    freeze_start = (
        (freeze_end - pd.DateOffset(years=max(1, int(stage_years)))).normalize()
        if freeze_end is not None
        else None
    )
    start_anchor = min(nominal_start, freeze_start) if freeze_start is not None else nominal_start
    start = start_anchor.date()
    end = as_of.date()
    trading_days = [day.isoformat() for day in get_trading_days("XNYS", start, end)]
    if not trading_days:
        return []
    canonical_vendors = tuple(
        capability.vendor
        for capability in backfill_capabilities(
            dataset="equities_eod",
            connectors=connectors,
            audit_state=audit_state,
        )
    )
    if not canonical_vendors:
        return []

    tasks: list[PlannedTask] = []
    ordered_symbols = list(dict.fromkeys(str(symbol).upper() for symbol in stage_symbols))
    listing_windows = _load_listing_windows(data_root) if data_root is not None else {}
    for symbol_index, symbol in enumerate(ordered_symbols):
        symbol_chunk = (symbol,)
        eligible_days = _eligible_trading_days(
            trading_days=trading_days,
            listing_window=listing_windows.get(symbol),
        )
        if not eligible_days:
            continue
        symbol_chunk_id = f"{symbol_index:05d}"
        for date_index in range(0, len(eligible_days), max(1, trading_day_chunk_size)):
            window = eligible_days[date_index : date_index + max(1, trading_day_chunk_size)]
            if not window:
                continue
            date_chunk_id = f"{date_index // max(1, trading_day_chunk_size):04d}"
            start_date = window[0]
            end_date = window[-1]
            in_freeze_window = (
                freeze_end is not None
                and freeze_start is not None
                and pd.Timestamp(end_date) <= freeze_end
                and pd.Timestamp(start_date) >= freeze_start
            )
            tasks.append(
                PlannedTask(
                    task_key=f"canonical_bars::equities_eod::{symbol_chunk_id}::{date_chunk_id}::{start_date}::{end_date}",
                    task_family="canonical_bars",
                    planner_group="canonical_bars_backlog",
                    dataset="equities_eod",
                    tier="A",
                    priority=5 if in_freeze_window else 10,
                    start_date=start_date,
                    end_date=end_date,
                    symbols=symbol_chunk,
                    output_name="equities_bars",
                    preferred_vendors=_canonical_vendors_for_window(
                        canonical_vendors=canonical_vendors,
                        in_freeze_window=in_freeze_window,
                    ),
                    payload={
                        "scope_kind": "symbol_range",
                        "symbol_chunk_id": symbol_chunk_id,
                        "date_chunk_id": date_chunk_id,
                        "symbol_batch_size": len(symbol_chunk),
                        "trading_days": list(window),
                        "freeze_priority": in_freeze_window,
                    },
                )
            )
    return tasks


def _canonical_vendors_for_window(*, canonical_vendors: tuple[str, ...], in_freeze_window: bool) -> tuple[str, ...]:
    """Return the vendor set to use for a canonical bar task window."""
    critical_path = tuple(
        vendor
        for vendor in canonical_vendors
        if (contract := dataset_contract(vendor, "equities_eod")) is not None and contract.critical_path_allowed
    )
    if in_freeze_window:
        ordered = tuple(vendor for vendor in ("alpaca", "tiingo") if vendor in critical_path)
        return ordered or critical_path or canonical_vendors
    non_critical = tuple(vendor for vendor in canonical_vendors if vendor not in critical_path)
    return critical_path + non_critical if critical_path else canonical_vendors


def plan_coverage_tasks(
    *,
    data_root: Path,
    stage_symbols: list[str],
    stage_years: int,
    connectors: dict[str, object],
    audit_state: dict[str, Any] | None = None,
    include_research: bool = False,
    current_date: str | None = None,
    freeze_report_date: str | None = None,
    symbol_batch_size: int = 1,
    trading_day_chunk_size: int = 20,
) -> list[PlannedTask]:
    """Return the full planner backlog ordered by core-first priority."""
    canonical = plan_canonical_bar_tasks(
        data_root=data_root,
        stage_symbols=stage_symbols,
        stage_years=stage_years,
        connectors=connectors,
        audit_state=audit_state,
        current_date=current_date,
        freeze_report_date=freeze_report_date,
        symbol_batch_size=symbol_batch_size,
        trading_day_chunk_size=trading_day_chunk_size,
    )
    auxiliary = plan_auxiliary_tasks(
        data_root=data_root,
        stage_symbols=stage_symbols,
        stage_years=stage_years,
        connectors=connectors,
        audit_state=audit_state,
        include_research=include_research,
        current_date=current_date,
    )
    return sorted([*canonical, *auxiliary], key=lambda task: (task.priority, task.task_key))


def _load_listing_windows(data_root: Path | None) -> dict[str, tuple[pd.Timestamp | None, pd.Timestamp | None]]:
    """Return symbol-specific IPO/delist windows when listing history is available."""
    if data_root is None:
        return {}
    path = data_root / "data" / "reference" / "listing_history.parquet"
    if not path.exists():
        return {}
    try:
        frame = pd.read_parquet(path, columns=["symbol", "ipo_date", "delist_date"])
    except Exception:
        return {}
    if frame.empty:
        return {}
    frame["symbol"] = frame["symbol"].astype("string").str.strip().str.upper()
    frame["ipo_date"] = pd.to_datetime(frame.get("ipo_date"), errors="coerce").dt.normalize()
    frame["delist_date"] = pd.to_datetime(frame.get("delist_date"), errors="coerce").dt.normalize()
    windows: dict[str, tuple[pd.Timestamp | None, pd.Timestamp | None]] = {}
    for symbol, group in frame.groupby("symbol", dropna=True):
        ipo_dates = group["ipo_date"].dropna()
        delist_dates = group["delist_date"].dropna()
        windows[str(symbol)] = (
            ipo_dates.min() if not ipo_dates.empty else None,
            delist_dates.min() if not delist_dates.empty else None,
        )
    return windows


def _eligible_trading_days(
    *,
    trading_days: list[str],
    listing_window: tuple[pd.Timestamp | None, pd.Timestamp | None] | None,
) -> list[str]:
    """Filter trading days to the symbol's eligible listing window."""
    if not listing_window:
        return list(trading_days)
    ipo_date, delist_date = listing_window
    eligible: list[str] = []
    for trading_day in trading_days:
        trading_ts = pd.Timestamp(trading_day).normalize()
        if ipo_date is not None and trading_ts < ipo_date:
            continue
        if delist_date is not None and trading_ts >= delist_date:
            continue
        eligible.append(trading_day)
    return eligible


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
