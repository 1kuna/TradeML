"""Fleet observability snapshots for current-state health surfaces."""

from __future__ import annotations

import json
import os
import shutil
from time import perf_counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SEVERITY_RANK = {"info": 0, "warning": 1, "critical": 2}


def build_fleet_observability(
    *,
    snapshot: dict[str, Any],
    data_root: Path,
    remote: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build the machine-readable fleet observability snapshot."""
    current = now or datetime.now(tz=UTC)
    policy = _observability_policy(config or {})
    remote = remote or dict(snapshot.get("fleet_remote") or {})
    snapshot = _snapshot_with_remote_pi_observability(snapshot, remote=remote)
    vendors = _vendor_observability(snapshot, policy=policy)
    scheduler = _scheduler_observability(snapshot)
    freshness = _freshness_observability(snapshot, policy=policy, now=current)
    resources = _resource_observability(
        snapshot=snapshot,
        data_root=data_root,
        remote=remote,
        policy=policy,
        now=current,
    )
    archive_schema = _archive_schema_observability(snapshot)
    ingestion_ledger = _ingestion_ledger_observability(snapshot)
    source_availability = _source_availability_observability(data_root=data_root)
    compaction = _compaction_observability(data_root=data_root)
    research = _research_observability(snapshot)
    paper_pnl = _paper_pnl_observability(snapshot)
    collection_saturation = _collection_saturation_observability(vendors)
    payload = {
        "generated_at": current.isoformat(),
        "vendors": vendors,
        "scheduler": scheduler,
        "collection_saturation": collection_saturation,
        "freshness": freshness,
        "resources": resources,
        "archive_schema": archive_schema,
        "ingestion_ledger": ingestion_ledger,
        "source_availability": source_availability,
        "compaction": compaction,
        "research": research,
        "paper_pnl": paper_pnl,
        "issues": [],
        "verdict": "OK",
    }
    issues = collect_observability_issues(payload)
    payload["issues"] = issues
    payload["verdict"] = _verdict_for_issues(issues)
    return payload


def _snapshot_with_remote_pi_observability(snapshot: dict[str, Any], *, remote: dict[str, Any]) -> dict[str, Any]:
    node_observability = dict(((remote.get("pi") or {}).get("node_state") or {}).get("node_observability") or {})
    if not node_observability:
        return snapshot
    merged = dict(snapshot)
    for key in (
        "scheduler_decisions",
        "archive_write_telemetry",
        "lane_health",
        "vendor_attempts_summary",
        "planner_task_summary",
        "budget_summary",
        "ingestion_ledger",
    ):
        if node_observability.get(key):
            merged[key] = node_observability[key]
    if node_observability.get("dataset_coverage"):
        merged["dataset_coverage"] = {**dict(snapshot.get("dataset_coverage") or {}), **dict(node_observability["dataset_coverage"])}
    return merged


def write_fleet_observability(
    *,
    data_root: Path,
    payload: dict[str, Any],
    now: datetime | None = None,
    write_history: bool = True,
) -> dict[str, Any]:
    """Persist the latest fleet observability payload to shared state."""
    current = now or datetime.now(tz=UTC)
    root = data_root / "control" / "cluster" / "state" / "autopilot" / "observability"
    latest = root / "latest.json"
    written: list[str] = []
    root.mkdir(parents=True, exist_ok=True)
    latest.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    written.append(str(latest))
    if write_history:
        history = root / "history" / current.date().isoformat()
        history.mkdir(parents=True, exist_ok=True)
        history_path = history / f"{current.strftime('%H%M%S')}.json"
        history_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
        written.append(str(history_path))
    saturation = payload.get("collection_saturation")
    if saturation:
        saturation_root = data_root / "control" / "cluster" / "state" / "autopilot" / "collection_saturation"
        saturation_root.mkdir(parents=True, exist_ok=True)
        saturation_latest = saturation_root / "latest.json"
        saturation_latest.write_text(json.dumps(saturation, indent=2, sort_keys=True, default=str), encoding="utf-8")
        written.append(str(saturation_latest))
        if write_history:
            saturation_history = saturation_root / "history" / current.date().isoformat()
            saturation_history.mkdir(parents=True, exist_ok=True)
            saturation_path = saturation_history / f"{current.strftime('%H%M%S')}.json"
            saturation_path.write_text(json.dumps(saturation, indent=2, sort_keys=True, default=str), encoding="utf-8")
            written.append(str(saturation_path))
    return {"observability_root": str(root), "latest_path": str(latest), "written": written}


def collect_observability_issues(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return current actionable issues from one observability payload."""
    issues: list[dict[str, Any]] = []
    for row in list((payload.get("vendors") or {}).get("rows") or []):
        if str(row.get("utilization_verdict")) == "underutilized":
            issues.append(
                _issue(
                    "pi",
                    "warning",
                    "vendor_underutilized",
                    f"{row.get('vendor')} has eligible {row.get('dataset')} work but is not spending budget",
                    f"Inspect scheduler decisions for {row.get('vendor')}:{row.get('dataset')}.",
                )
            )
    for row in list((payload.get("freshness") or {}).get("datasets") or []):
        if str(row.get("status")) == "stale":
            issues.append(
                _issue(
                    "data",
                    "warning",
                    "dataset_stale",
                    f"{row.get('dataset')} is stale; latest={row.get('latest_date')}",
                    "Inspect data-node collection and partition manifests.",
                )
            )
    for row in list((payload.get("archive_schema") or {}).get("rows") or []):
        if int(row.get("failures") or 0) > 0 or int(row.get("schema_mismatches") or 0) > 0:
            issues.append(
                _issue(
                    "pi",
                    "warning",
                    "archive_write_unhealthy",
                    f"{row.get('output_name')} archive writes have failures or schema mismatches",
                    "Inspect archive write telemetry and latest task errors.",
                )
            )
    for row in list((payload.get("ingestion_ledger") or {}).get("rows") or []):
        if str(row.get("status")).lower() == "failed":
            issues.append(
                _issue(
                    "pi",
                    "warning",
                    "ingestion_ledger_failure",
                    f"{row.get('dataset')} ingestion failed: {row.get('latest_error') or '-'}",
                    "Inspect the Pi ingestion ledger and archive write telemetry.",
                )
            )
    for dataset, row in dict((payload.get("source_availability") or {}).get("datasets") or {}).items():
        if isinstance(row, dict) and _source_is_actionable(row):
            issues.append(
                _issue(
                    "data",
                    "warning",
                    "source_availability_actionable",
                    f"{dataset} source state is {row.get('state')}: {row.get('reason') or row.get('status') or '-'}",
                    "Inspect source availability and the Pi-to-Mac feature source contract.",
                )
            )
    compaction = dict(payload.get("compaction") or {})
    compaction_failures = int((compaction.get("summary") or {}).get("failures") or 0)
    if compaction_failures > 0:
        issues.append(
            _issue(
                "data",
                "warning",
                "archive_compaction_failed",
                f"{compaction_failures} archive compaction partition(s) failed",
                "Inspect control/cluster/state/data/compaction/latest.json.",
            )
        )
    resources = dict(payload.get("resources") or {})
    if resources.get("nas_status") != "online":
        issues.append(
            _issue(
                "nas",
                "critical",
                "nas_unavailable",
                str(resources.get("nas_reason") or "NAS/shared data root is unavailable"),
                "Verify NAS mount and permissions.",
            )
        )
    if str(resources.get("memory_status")) == "pressure":
        issues.append(
            _issue(
                "system",
                "warning",
                "resource_pressure",
                str(resources.get("memory_reason") or "Memory pressure detected"),
                "Inspect Pi/Mac process RSS and system memory.",
            )
        )
    research = dict(payload.get("research") or {})
    if str(research.get("status") or "").upper() in {"INFRA_BLOCKED", "WAITING_FOR_DATA"}:
        issues.append(
            _issue(
                "mac",
                "critical",
                "research_blocked",
                str(research.get("reason") or "Research autopilot is blocked"),
                "Run research health and inspect the blocker.",
            )
        )
    paper = dict(payload.get("paper_pnl") or {})
    if str(paper.get("status")) == "error":
        issues.append(
            _issue(
                "research",
                "warning",
                "paper_pnl_unavailable",
                str(paper.get("reason") or "Paper PnL summary failed"),
                "Inspect latest paper/shadow artifacts.",
            )
        )
    leaderboard = dict(research.get("feature_family_leaderboard") or {})
    for entry in list(leaderboard.get("entries") or []):
        if not isinstance(entry, dict):
            continue
        if str(entry.get("readiness_status") or "").upper() == "BLOCKED" and _entry_has_actionable_source_blocker(entry):
            issues.append(
                _issue(
                    "research",
                    "warning",
                    "feature_source_blocked",
                    f"{entry.get('feature_version')} feature readiness is blocked: {entry.get('top_rejection_reason') or 'missing source coverage'}",
                    "Inspect feature source contract and Pi-to-Mac NAS dataset coverage.",
                )
            )
        for group, coverage in dict(entry.get("source_coverage") or {}).items():
            if group == "price_liquidity":
                continue
            sources = dict(coverage.get("sources") or {})
            actionable_sources = [
                source for source in sources.values() if isinstance(source, dict) and _source_is_actionable(source)
            ]
            if actionable_sources and all(str(source.get("status")) != "available" for source in sources.values() if isinstance(source, dict)):
                issues.append(
                    _issue(
                        "data",
                        "warning",
                        "feature_source_missing",
                        f"{entry.get('feature_version')} has no available source data for {group}",
                        "Verify the Pi archive output path matches the Mac feature source contract.",
                    )
                )
            for dataset, source in sources.items():
                if not isinstance(source, dict):
                    continue
                if not _source_is_actionable(source):
                    continue
                status = str(source.get("status") or "")
                if status == "missing":
                    issues.append(
                        _issue(
                            "data",
                            "warning",
                            "feature_source_path_missing",
                            f"{entry.get('feature_version')} cannot see {dataset} at the configured source paths",
                            "Verify the Pi-to-Mac data contract and NAS mount path.",
                        )
                    )
                elif status == "invalid_schema":
                    issues.append(
                        _issue(
                            "data",
                            "warning",
                            "feature_source_schema_mismatch",
                            f"{entry.get('feature_version')} source {dataset} is missing required columns: {source.get('missing_required_columns')}",
                            "Inspect the archive schema and feature source contract.",
                        )
                    )
                elif status == "empty":
                    issues.append(
                        _issue(
                            "data",
                            "warning",
                            "feature_source_zero_coverage",
                            f"{entry.get('feature_version')} source {dataset} exists but has zero rows",
                            "Inspect source archive freshness and feature readiness metadata.",
                        )
                    )
    return issues


def _observability_policy(config: dict[str, Any]) -> dict[str, Any]:
    collection = dict(config.get("collection") or {})
    saturation = dict(collection.get("saturation") or {})
    observability = dict(collection.get("observability") or {})
    freshness = dict(collection.get("freshness_policy") or {})
    return {
        "target_utilization": float(saturation.get("target_utilization") or 0.98),
        "vendor_underutilized_after_minutes": int(observability.get("vendor_underutilized_after_minutes") or 10),
        "dataset_stale_after_minutes": int(observability.get("dataset_stale_after_minutes") or 1440),
        "dataset_stale_after_by_dataset": dict(freshness.get("dataset_stale_after_minutes") or freshness.get("datasets") or {}),
        "scheduler_decision_retention_hours": int(observability.get("scheduler_decision_retention_hours") or 24),
        "disk_warn_percent": float((observability.get("resource_warn_thresholds") or {}).get("disk_percent") or 90.0),
    }


def _vendor_observability(snapshot: dict[str, Any], *, policy: dict[str, Any]) -> dict[str, Any]:
    budget_rows = {str(row.get("vendor")): dict(row) for row in list((snapshot.get("budget_summary") or {}).get("rows") or [])}
    throughput_rows = {str(row.get("vendor")): dict(row) for row in list((snapshot.get("vendor_throughput") or {}).get("rows") or [])}
    lane_rows = list((snapshot.get("lane_health") or {}).get("rows") or [])
    scheduler_rows = list((snapshot.get("scheduler_decisions") or {}).get("rows") or [])
    attempt_rows = list((snapshot.get("vendor_attempts_summary") or {}).get("rows") or [])
    archive_rows = list((snapshot.get("archive_write_telemetry") or {}).get("rows") or [])
    scheduler_by_lane: dict[tuple[str, str], dict[str, int]] = {}
    for row in scheduler_rows:
        vendor = str(row.get("vendor") or "")
        dataset = str(row.get("dataset") or "all")
        decision = str(row.get("decision") or "unknown")
        key = (vendor, dataset)
        scheduler_by_lane.setdefault(key, {})[decision] = scheduler_by_lane.setdefault(key, {}).get(decision, 0) + int(row.get("count") or 0)
    attempts_by_lane: dict[tuple[str, str], dict[str, int]] = {}
    for row in attempt_rows:
        key = (str(row.get("vendor") or ""), str(row.get("dataset") or "all"))
        status = str(row.get("status") or "unknown")
        attempts_by_lane.setdefault(key, {})[status] = attempts_by_lane.setdefault(key, {}).get(status, 0) + int(row.get("count") or 0)
    archive_by_dataset: dict[str, dict[str, Any]] = {}
    for row in archive_rows:
        dataset = str(row.get("output_name") or row.get("dataset") or "")
        if not dataset:
            continue
        current = archive_by_dataset.setdefault(dataset, {"rows_in": 0, "rows_written": 0, "writes": 0, "latest_at": None})
        current["rows_in"] = int(current.get("rows_in") or 0) + int(row.get("rows_in") or 0)
        current["rows_written"] = int(current.get("rows_written") or 0) + int(row.get("rows_written") or 0)
        current["writes"] = int(current.get("writes") or 0) + int(row.get("writes") or row.get("count") or 0)
        current["latest_at"] = max([value for value in (current.get("latest_at"), row.get("latest_at") or row.get("last_seen")) if value], default=current.get("latest_at"))
    vendors = sorted(
        set(budget_rows)
        | set(throughput_rows)
        | {str(row.get("vendor")) for row in lane_rows if row.get("vendor")}
        | {vendor for vendor, _dataset in scheduler_by_lane if vendor}
        | {vendor for vendor, _dataset in attempts_by_lane if vendor}
    )
    rows: list[dict[str, Any]] = []
    target_utilization = float(policy["target_utilization"])
    for vendor in vendors:
        budget = budget_rows.get(vendor, {})
        throughput = throughput_rows.get(vendor, {})
        lane_matches = [dict(row) for row in lane_rows if str(row.get("vendor")) == vendor]
        datasets = sorted(
            {str(row.get("dataset") or "unknown") for row in lane_matches}
            | {dataset for lane_vendor, dataset in scheduler_by_lane if lane_vendor == vendor and dataset}
            | {dataset for lane_vendor, dataset in attempts_by_lane if lane_vendor == vendor and dataset}
        ) or ["all"]
        rpm_limit = int(budget.get("rpm_limit") or 0)
        target_rpm = round(float(rpm_limit) * target_utilization, 3) if rpm_limit else 0.0
        for dataset in datasets:
            lane_key = (vendor, dataset)
            lane_scheduler = scheduler_by_lane.get(lane_key, {})
            lane_attempts = attempts_by_lane.get(lane_key, {})
            actual_rpm = _lane_actual_rpm(
                lane_scheduler=lane_scheduler,
                lane_attempts=lane_attempts,
                scheduler_window_minutes=int((snapshot.get("scheduler_decisions") or {}).get("window_minutes") or 15),
                attempts_window_minutes=int((snapshot.get("vendor_attempts_summary") or {}).get("window_minutes") or 60),
                fallback=float(budget.get("outbound_requests_60s") or throughput.get("outbound_requests_per_min") or 0.0),
            )
            lane_specific = [row for row in lane_matches if str(row.get("dataset") or "unknown") == dataset]
            eligible_work = _lane_has_eligible_work(
                snapshot,
                vendor=vendor,
                dataset=dataset,
                budget=budget,
                throughput=throughput,
                scheduler_counts=lane_scheduler,
                attempt_counts=lane_attempts,
            )
            blocked = _vendor_blocker(budget=budget, lane_matches=lane_specific or lane_matches)
            active = actual_rpm > 0.0 or int(lane_scheduler.get("claimed") or 0) > 0 or int(lane_attempts.get("SUCCESS") or 0) > 0
            underutilized = eligible_work and blocked is None and rpm_limit > 0 and not active
            archive = (archive_by_dataset.get(dataset) or archive_by_dataset.get(_archive_output_for_dataset(dataset)) or {}) if active else {}
            rows.append(
                {
                    "vendor": vendor,
                    "dataset": dataset,
                    "eligible_work": eligible_work,
                    "target_utilization": target_utilization,
                    "actual_requests_per_minute": actual_rpm,
                    "target_requests_per_minute": target_rpm,
                    "remaining_minute": max(0, rpm_limit - int(budget.get("rpm_used") or 0)),
                    "remaining_daily": int(budget.get("day_remaining") or 0),
                    "unused_capacity": max(0.0, round(target_rpm - actual_rpm, 3)),
                    "in_flight_lanes": int(lane_scheduler.get("claimed", 0)) + int(lane_attempts.get("LEASED", 0)),
                    "last_outbound_request": archive.get("latest_at") or throughput.get("latest_update") or budget.get("checked_at"),
                    "rows_per_request": _safe_ratio(float(archive.get("rows_in") or throughput.get("rows_per_min") or 0.0), max(actual_rpm, 1.0) if active else actual_rpm),
                    "rows_written_current_window": int(archive.get("rows_written") or archive.get("rows_in") or 0),
                    "current_blocker": blocked,
                    "next_eligible_at": _next_eligible_at(lane_specific or lane_matches),
                    "scheduler_decisions": lane_scheduler,
                    "vendor_attempts": lane_attempts,
                    "utilization_verdict": "underutilized" if underutilized else "blocked" if blocked else "active" if active else "idle",
                }
            )
    return {
        "target_utilization": target_utilization,
        "rows": rows,
        "underutilized_count": sum(1 for row in rows if row["utilization_verdict"] == "underutilized"),
    }


def _scheduler_observability(snapshot: dict[str, Any]) -> dict[str, Any]:
    summary = dict(snapshot.get("scheduler_decisions") or {})
    rows = list(summary.get("rows") or [])
    totals: dict[str, int] = {}
    for row in rows:
        decision = str(row.get("decision") or "unknown")
        totals[decision] = totals.get(decision, 0) + int(row.get("count") or 0)
    return {
        "window_minutes": summary.get("window_minutes"),
        "checked_at": summary.get("checked_at"),
        "totals": totals,
        "rows": rows,
    }


def _lane_actual_rpm(
    *,
    lane_scheduler: dict[str, int],
    lane_attempts: dict[str, int],
    scheduler_window_minutes: int,
    attempts_window_minutes: int,
    fallback: float,
) -> float:
    attempt_count = int(lane_attempts.get("SUCCESS") or 0) + int(lane_attempts.get("FAILED") or 0) + int(lane_attempts.get("LEASED") or 0)
    if attempt_count > 0:
        return round(attempt_count / max(1, attempts_window_minutes), 3)
    request_count = int(lane_scheduler.get("claimed") or 0)
    if request_count > 0:
        return round(request_count / max(1, scheduler_window_minutes), 3)
    return float(fallback or 0.0)


def _collection_saturation_observability(vendors: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for row in list(vendors.get("rows") or []):
        verdict = str(row.get("utilization_verdict") or "idle")
        eligible = bool(row.get("eligible_work"))
        blocker = row.get("current_blocker")
        intentionally_idle = (not eligible) or bool(blocker)
        rows.append(
            {
                "vendor": row.get("vendor"),
                "dataset": row.get("dataset"),
                "eligible_work": eligible,
                "budget_remaining_minute": row.get("remaining_minute"),
                "budget_remaining_daily": row.get("remaining_daily"),
                "calls_spent_current_window": row.get("actual_requests_per_minute"),
                "rows_per_credit": row.get("rows_per_request"),
                "rows_written_current_window": row.get("rows_written_current_window"),
                "blocker": blocker,
                "next_eligible_at": row.get("next_eligible_at"),
                "intentionally_idle": intentionally_idle,
                "unused_capacity": row.get("unused_capacity"),
                "verdict": verdict,
            }
        )
    return {
        "rows": rows,
        "summary": {
            "active": sum(1 for row in rows if row["verdict"] == "active"),
            "underutilized": sum(1 for row in rows if row["verdict"] == "underutilized"),
            "blocked": sum(1 for row in rows if row["verdict"] == "blocked"),
            "intentionally_idle": sum(1 for row in rows if row["intentionally_idle"]),
        },
    }


def _freshness_observability(snapshot: dict[str, Any], *, policy: dict[str, Any], now: datetime) -> dict[str, Any]:
    stale_after = int(policy["dataset_stale_after_minutes"])
    datasets = [
        _freshness_row("raw_equities", snapshot.get("latest_raw_date"), stale_after=_stale_after_for_dataset(policy, "raw_equities", stale_after), now=now),
        _freshness_row("curated_equities", snapshot.get("latest_curated_date"), stale_after=_stale_after_for_dataset(policy, "curated_equities", stale_after), now=now),
    ]
    for name, payload in sorted(dict(snapshot.get("dataset_coverage") or {}).items()):
        if isinstance(payload, dict) and payload.get("latest_date"):
            datasets.append(_freshness_row(str(name), payload.get("latest_date"), stale_after=_stale_after_for_dataset(policy, str(name), stale_after), now=now))
    return {
        "dataset_stale_after_minutes": stale_after,
        "dataset_stale_after_by_dataset": dict(policy.get("dataset_stale_after_by_dataset") or {}),
        "datasets": datasets,
        "stale_count": sum(1 for row in datasets if row["status"] == "stale"),
    }


def _stale_after_for_dataset(policy: dict[str, Any], dataset: str, default: int) -> int:
    raw = dict(policy.get("dataset_stale_after_by_dataset") or {}).get(dataset)
    if isinstance(raw, dict):
        raw = raw.get("stale_after_minutes") or raw.get("expected_cadence_minutes")
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return int(default)


def _resource_observability(
    *,
    snapshot: dict[str, Any],
    data_root: Path,
    remote: dict[str, Any],
    policy: dict[str, Any],
    now: datetime,
) -> dict[str, Any]:
    runtime = dict(snapshot.get("runtime") or {})
    disk = _disk_payload(data_root)
    heartbeat_age = _age_seconds(runtime.get("heartbeat_at"), now=now)
    load_average = list(os.getloadavg()) if hasattr(os, "getloadavg") else []
    nas_online = data_root.exists()
    memory = {**_local_memory_payload(), **dict(snapshot.get("resource_health") or {})}
    return {
        "nas_status": "online" if nas_online else "offline",
        "nas_reason": None if nas_online else f"data root unavailable: {data_root}",
        "disk": disk,
        "disk_status": "pressure" if disk.get("used_percent", 0.0) >= float(policy["disk_warn_percent"]) else "ok",
        "heartbeat_age_seconds": heartbeat_age,
        "pid": runtime.get("pid"),
        "process_running": bool(runtime.get("running")),
        "load_average": load_average,
        "memory_status": memory.get("status", "unknown"),
        "memory_reason": memory.get("reason"),
        "memory": memory,
        "oom_events": memory.get("oom_events", []),
        "remote": remote,
    }


def _archive_schema_observability(snapshot: dict[str, Any]) -> dict[str, Any]:
    summary = dict(snapshot.get("archive_write_telemetry") or {})
    rows = list(summary.get("rows") or [])
    return {
        "window_minutes": summary.get("window_minutes"),
        "checked_at": summary.get("checked_at"),
        "rows": rows,
        "failure_count": sum(int(row.get("failures") or 0) for row in rows),
        "schema_mismatch_count": sum(int(row.get("schema_mismatches") or 0) for row in rows),
    }


def _ingestion_ledger_observability(snapshot: dict[str, Any]) -> dict[str, Any]:
    summary = dict(snapshot.get("ingestion_ledger") or {})
    rows = list(summary.get("rows") or [])
    return {
        "window_minutes": summary.get("window_minutes"),
        "checked_at": summary.get("checked_at"),
        "rows": rows,
        "failure_count": sum(int(row.get("events") or 0) for row in rows if str(row.get("status")) == "failed"),
        "rows_written": sum(int(row.get("rows_written") or 0) for row in rows),
    }


def _source_availability_observability(*, data_root: Path) -> dict[str, Any]:
    path = data_root / "control" / "cluster" / "state" / "data" / "source_availability" / "latest.json"
    payload = _read_json(path)
    if not payload:
        return {"status": "missing", "path": str(path), "datasets": {}, "state_counts": {}}
    return {
        "status": "available",
        "path": str(path),
        "generated_at": payload.get("generated_at"),
        "state_counts": dict(payload.get("state_counts") or {}),
        "datasets": dict(payload.get("datasets") or {}),
    }


def _compaction_observability(*, data_root: Path) -> dict[str, Any]:
    path = data_root / "control" / "cluster" / "state" / "data" / "compaction" / "latest.json"
    payload = _read_json(path)
    if not payload:
        return {"status": "missing", "path": str(path), "rows": [], "summary": {"failures": 0}}
    summary = dict(payload.get("summary") or {})
    return {
        "status": "available",
        "path": str(path),
        "generated_at": payload.get("generated_at"),
        "dry_run": bool(payload.get("dry_run", False)),
        "summary": summary,
        "rows": list(payload.get("rows") or []),
    }


def _research_observability(snapshot: dict[str, Any]) -> dict[str, Any]:
    remote_research = dict(((snapshot.get("fleet_remote") or {}).get("mac") or {}).get("research") or {})
    program = {
        **dict((snapshot.get("health") or {}).get("research_program_summary") or {}),
        **remote_research,
    }
    experiment = dict(snapshot.get("experiment_summary") or {})
    best = dict(program.get("best_candidate_summary") or {})
    progression = dict(program.get("autonomous_progression") or program.get("progression") or {})
    autopsy = dict(program.get("candidate_autopsy") or best.get("candidate_autopsy") or experiment.get("candidate_autopsy") or {})
    paper_smoke = dict(program.get("latest_paper_account_smoke") or program.get("paper_account_smoke") or {})
    modeling = dict(program.get("modeling") or {})
    leaderboard = dict(program.get("feature_family_leaderboard") or {})
    return {
        "status": str(program.get("status") or remote_research.get("status") or "UNKNOWN"),
        "current_experiment_id": program.get("current_experiment_id") or experiment.get("experiment_id") or remote_research.get("current_experiment_id"),
        "supervisor_running": str(program.get("status") or remote_research.get("status") or "").upper() == "RUNNING",
        "completed_runs_24h": int(program.get("completed_runs_24h") or experiment.get("completed_24h") or 0),
        "completed_runs_7d": int(program.get("completed_runs_7d") or experiment.get("completed_7d") or 0),
        "failed_runs_24h": int(program.get("failed_runs_24h") or experiment.get("failed_24h") or 0),
        "infra_blocked_runs_24h": int(program.get("infra_blocked_runs_24h") or 0),
        "top_rejection_reasons": program.get("top_rejection_reasons") or experiment.get("top_rejection_reasons") or [],
        "best_candidate": best.get("best_candidate") or experiment.get("best_candidate"),
        "best_primary_score": best.get("best_primary_score") or experiment.get("best_primary_score"),
        "best_decision_reason": best.get("best_decision_reason") or experiment.get("best_decision_reason"),
        "frontier_lane": program.get("frontier_architecture") or program.get("frontier") or {},
        "architecture_lane": program.get("architecture_lane") or progression.get("current_lane"),
        "complexity_tier": program.get("complexity_tier"),
        "objective_verdict": program.get("objective_verdict") or {},
        "candidate_autopsy": autopsy,
        "diagnostic_mode": program.get("diagnostic_mode") or experiment.get("diagnostic_mode"),
        "follow_up_of_run_id": program.get("follow_up_of_run_id") or experiment.get("follow_up_of_run_id"),
        "progression": progression,
        "pivot_reason": program.get("pivot_reason") or progression.get("pivot_reason"),
        "next_lane": program.get("next_lane") or progression.get("next_lane"),
        "exhausted_lanes": progression.get("exhausted_lanes") or [],
        "last_canary": program.get("last_canary") or {},
        "paper_account_smoke": paper_smoke,
        "modeling": modeling,
        "feature_family_leaderboard": leaderboard,
        "feature_version": modeling.get("feature_version") or experiment.get("feature_version"),
        "label_horizon": modeling.get("current_label_horizon") or experiment.get("label_horizon"),
        "portfolio_profile": modeling.get("current_portfolio_profile") or experiment.get("portfolio_profile"),
        "data_revision": modeling.get("data_revision") or program.get("data_revision") or experiment.get("data_revision"),
        "reason": program.get("wait_reason") or remote_research.get("wait_reason") or best.get("best_decision_reason"),
    }


def _paper_pnl_observability(snapshot: dict[str, Any]) -> dict[str, Any]:
    remote_research = dict(((snapshot.get("fleet_remote") or {}).get("mac") or {}).get("research") or {})
    program = {
        **dict((snapshot.get("health") or {}).get("research_program_summary") or {}),
        **remote_research,
    }
    paper = dict(program.get("latest_paper_outputs") or program.get("paper_outputs") or {})
    shadow = dict(program.get("latest_shadow_paper_outputs") or {})
    pnl = dict(program.get("latest_paper_pnl") or program.get("paper_pnl") or {})
    evidence = dict(program.get("latest_paper_evidence") or program.get("paper_evidence") or {})
    paper_smoke = dict(program.get("latest_paper_account_smoke") or program.get("paper_account_smoke") or {})
    paper_submission = dict(program.get("latest_paper_submission") or program.get("paper_submission") or {})
    source = paper if paper.get("status") == "written" else shadow
    if pnl:
        return {
            "status": str(pnl.get("status") or "available"),
            "date": pnl.get("date") or source.get("date"),
            "path": pnl.get("path"),
            "net_return": pnl.get("net_return"),
            "turnover": pnl.get("turnover"),
            "cost_drag": pnl.get("cost_drag"),
            "max_drawdown": pnl.get("max_drawdown"),
            "benchmark_spy_return": pnl.get("benchmark_spy_return"),
            "source": "paper" if paper.get("status") == "written" else "shadow",
            "paper_order_payloads_path": source.get("paper_order_payloads_path") or source.get("shadow_order_payloads_path"),
            "paper_account_smoke_status": paper_smoke.get("status"),
            "paper_account_smoke_checked_at": paper_smoke.get("checked_at"),
            "paper_submission_status": paper_submission.get("status"),
            "evidence_status": evidence.get("status"),
            "evidence_failures": evidence.get("failures", []),
            "no_live_orders": True,
        }
    if source:
        return {
            "status": "pending_labels",
            "date": source.get("date"),
            "path": source.get("paper_orders_path") or source.get("shadow_orders_path"),
            "source": "shadow" if source.get("non_incumbent") else "paper",
            "paper_order_payloads_path": source.get("paper_order_payloads_path") or source.get("shadow_order_payloads_path"),
            "paper_account_smoke_status": paper_smoke.get("status"),
            "paper_account_smoke_checked_at": paper_smoke.get("checked_at"),
            "paper_submission_status": paper_submission.get("status"),
            "evidence_status": evidence.get("status"),
            "evidence_failures": evidence.get("failures", []),
            "no_live_orders": True,
            "non_incumbent": bool(source.get("non_incumbent")),
        }
    return {
        "status": "pending",
        "reason": "No paper/shadow outputs yet",
        "paper_account_smoke_status": paper_smoke.get("status"),
        "paper_account_smoke_checked_at": paper_smoke.get("checked_at"),
        "paper_submission_status": paper_submission.get("status"),
        "evidence_status": evidence.get("status"),
        "evidence_failures": evidence.get("failures", []),
        "no_live_orders": True,
    }


def _freshness_row(dataset: str, latest_date: Any, *, stale_after: int, now: datetime) -> dict[str, Any]:
    age_minutes: int | None = None
    if latest_date:
        try:
            latest = datetime.fromisoformat(str(latest_date))
        except ValueError:
            latest = datetime.fromisoformat(f"{latest_date}T00:00:00+00:00")
        if latest.tzinfo is None:
            latest = latest.replace(tzinfo=UTC)
        age_minutes = max(0, int((now - latest).total_seconds() // 60))
    status = "unknown" if age_minutes is None else "stale" if age_minutes > stale_after else "fresh"
    return {
        "dataset": dataset,
        "latest_date": str(latest_date) if latest_date else None,
        "age_minutes": age_minutes,
        "expected_cadence_minutes": stale_after,
        "status": status,
    }


def _vendor_has_eligible_work(snapshot: dict[str, Any], *, vendor: str, budget: dict[str, Any], throughput: dict[str, Any]) -> bool:
    if str(budget.get("state")) not in {"available", "no_data"}:
        return False
    planner_summary = dict(snapshot.get("planner_summary") or {})
    progress = dict(planner_summary.get("progress") or {})
    remaining = sum(int(row.get("remaining_units") or 0) for row in progress.values() if isinstance(row, dict))
    if remaining <= 0:
        return False
    if throughput.get("state") in {"backoff", "day_capped", "minute_capped"}:
        return False
    return bool(budget.get("minute_available", True) and budget.get("day_available", True))


def _lane_has_eligible_work(
    snapshot: dict[str, Any],
    *,
    vendor: str,
    dataset: str,
    budget: dict[str, Any],
    throughput: dict[str, Any],
    scheduler_counts: dict[str, int],
    attempt_counts: dict[str, int],
) -> bool:
    if int(scheduler_counts.get("claimed") or 0) > 0 or int(attempt_counts.get("LEASED") or 0) > 0:
        return True
    if str(budget.get("state")) not in {"available", "no_data", ""}:
        return False
    if throughput.get("state") in {"backoff", "day_capped", "minute_capped"}:
        return False
    if not bool(budget.get("minute_available", True) and budget.get("day_available", True)):
        return False
    for row in list((snapshot.get("planner_task_summary") or {}).get("rows") or []):
        if str(row.get("dataset") or "") != dataset:
            continue
        status = str(row.get("status") or "").upper()
        if status in {"PENDING", "LEASED", "PARTIAL"} and int(row.get("count") or 0) > 0:
            return True
    return _vendor_has_eligible_work(snapshot, vendor=vendor, budget=budget, throughput=throughput)


def _archive_output_for_dataset(dataset: str) -> str:
    if dataset in {"stock_trades", "stock_quotes", "option_snapshots", "option_bars", "option_chain_reference"}:
        return "alpaca_market_events"
    if dataset in {"company_news", "news", "stock_news", "press_releases", "news_sentiment"}:
        return "ticker_news"
    return dataset


def _vendor_blocker(*, budget: dict[str, Any], lane_matches: list[dict[str, Any]]) -> str | None:
    states = {str(row.get("state")) for row in lane_matches if row.get("state")}
    if "ENTITLEMENT_BLOCKED" in states:
        return "entitlement_blocked"
    if "AUDIT_FAILED" in states:
        return "audit_failed"
    if "DISABLED" in states:
        return "disabled"
    if "BUDGET_BLOCKED" in states or not bool(budget.get("day_available", True)) or not bool(budget.get("minute_available", True)):
        return str(budget.get("reason") or "budget_blocked")
    if "COOLDOWN" in states:
        return "cooldown"
    return None


def _next_eligible_at(lane_matches: list[dict[str, Any]]) -> str | None:
    values = sorted(str(row.get("cooldown_until")) for row in lane_matches if row.get("cooldown_until"))
    return values[0] if values else None


def _safe_ratio(numerator: float, denominator: float) -> float:
    return round(numerator / denominator, 3) if denominator else 0.0


def _disk_payload(path: Path) -> dict[str, Any]:
    started = perf_counter()
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "used_percent": 0.0,
            "stat_latency_ms": round((perf_counter() - started) * 1000.0, 3),
        }
    usage = shutil.disk_usage(path)
    used_percent = round(((usage.total - usage.free) / usage.total) * 100.0, 2) if usage.total else 0.0
    return {
        "path": str(path),
        "exists": True,
        "total_bytes": usage.total,
        "free_bytes": usage.free,
        "used_percent": used_percent,
        "stat_latency_ms": round((perf_counter() - started) * 1000.0, 3),
    }


def _local_memory_payload() -> dict[str, Any]:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return {"status": "unknown"}
    values: dict[str, int] = {}
    try:
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            key, _, raw_value = line.partition(":")
            parts = raw_value.strip().split()
            if parts:
                values[key] = int(parts[0])
    except (OSError, ValueError):
        return {"status": "unknown"}
    total = int(values.get("MemTotal", 0) or 0)
    available = int(values.get("MemAvailable", 0) or 0)
    available_percent = round((available / total) * 100.0, 2) if total else None
    pressure_text = None
    pressure = Path("/proc/pressure/memory")
    if pressure.exists():
        try:
            pressure_text = pressure.read_text(encoding="utf-8").strip()
        except OSError:
            pressure_text = None
    status = "pressure" if available_percent is not None and available_percent < 10.0 else "ok"
    reason = f"MemAvailable {available_percent}%" if available_percent is not None else None
    return {
        "status": status,
        "reason": reason,
        "mem_total_kb": total,
        "mem_available_kb": available,
        "mem_available_percent": available_percent,
        "pressure": pressure_text,
    }


def _age_seconds(value: Any, *, now: datetime) -> int | None:
    if not value:
        return None
    try:
        stamp = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=UTC)
    return max(0, int((now - stamp).total_seconds()))


def _issue(source: str, severity: str, kind: str, message: str, suggested_action: str) -> dict[str, Any]:
    return {
        "source": source,
        "severity": severity,
        "kind": kind,
        "message": message,
        "suggested_codex_action": suggested_action,
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _source_is_actionable(source: dict[str, Any]) -> bool:
    if bool(source.get("known_unavailable")):
        return False
    state = str(source.get("source_state") or source.get("lifecycle_state") or source.get("state") or "")
    if state in {"ENTITLEMENT_UNAVAILABLE", "DISABLED_BY_POLICY", "AVAILABLE"}:
        return False
    if source.get("actionable") is not None:
        return bool(source.get("actionable"))
    return str(source.get("status") or "") in {"missing", "empty", "invalid_schema"}


def _entry_has_actionable_source_blocker(entry: dict[str, Any]) -> bool:
    for coverage in dict(entry.get("source_coverage") or {}).values():
        if not isinstance(coverage, dict):
            continue
        for source in dict(coverage.get("sources") or {}).values():
            if isinstance(source, dict) and _source_is_actionable(source):
                return True
    return not entry.get("source_coverage")


def _verdict_for_issues(issues: list[dict[str, Any]]) -> str:
    severities = {str(issue.get("severity") or "warning") for issue in issues}
    if "critical" in severities:
        return "BLOCKED"
    if "warning" in severities:
        return "DEGRADED"
    return "OK"
