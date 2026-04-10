"""Experiment planning, launch, status, and comparison helpers."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import yaml

from trademl.data_node.training_control import (
    launch_training_process,
    recommended_training_cutoff,
    resolve_training_target,
    resolve_training_targets,
    shared_training_runtime_path,
    _stage_symbol_count,
    training_status_snapshot,
)


def plan_experiment(
    *,
    spec_path: Path,
    repo_root: Path,
    data_root: Path,
    local_state: Path,
    env_path: Path,
    targets_config_path: Path,
    python_executable: str,
) -> dict[str, Any]:
    """Materialize deterministic run manifests for an experiment spec."""
    spec = _load_spec(spec_path)
    base_config_path = Path(str(spec.get("base_config") or (repo_root / "configs" / "equities_xs.yml"))).expanduser()
    base_config = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    experiment_id = str(spec["experiment_id"])
    phase = int(spec.get("phase", 1))
    available_targets = resolve_training_targets(
        targets_config_path=targets_config_path,
        repo_root=repo_root,
        data_root=data_root,
        local_state=local_state,
        python_executable=python_executable,
    )
    default_target = next((target.name for target in available_targets.values() if target.default), "local")
    target_name = str(spec.get("target") or default_target)
    report_date = _resolve_report_date(spec, data_root=data_root)
    matrix = _realize_matrix(spec.get("matrix") or {})
    max_concurrent = int(spec.get("max_concurrent", 1) or 1)
    local_root = _local_experiment_root(local_state, experiment_id)
    local_root.mkdir(parents=True, exist_ok=True)
    runs_dir = local_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    configs_dir = local_root / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    for row in matrix:
        run_id = _run_id(row)
        runtime_name = f"experiment_{experiment_id}__{run_id}"
        run_config = _apply_overrides(base_config, row.get("config_overrides", {}))
        config_path = configs_dir / f"{run_id}.yml"
        config_path.write_text(yaml.safe_dump(run_config, sort_keys=False), encoding="utf-8")
        target = resolve_training_target(
            target_name=target_name,
            targets_config_path=targets_config_path,
            repo_root=repo_root,
            data_root=data_root,
            local_state=local_state,
            python_executable=python_executable,
        )
        run_manifest = {
            "experiment_id": experiment_id,
            "run_id": run_id,
            "runtime_name": runtime_name,
            "phase": phase,
            "target": target.name,
            "target_kind": target.kind,
            "report_date": report_date,
            "model_suite": row.get("model_suite") or spec.get("model_suite") or ("ridge_only" if phase == 1 else "full"),
            "matrix_values": row["matrix_values"],
            "config_overrides": row.get("config_overrides", {}),
            "config_path": str(config_path),
            "config_hash": hashlib.sha1(config_path.read_bytes()).hexdigest(),
            "local_runtime_path": str(target.local_runtime_root / "training_runs" / target.name / f"{runtime_name}.json"),
            "shared_runtime_path": str(shared_training_runtime_path(data_root=target.data_root, phase=phase, runtime_name=runtime_name)),
            "report_path": str(target.data_root / "reports" / "daily" / f"{report_date}.json"),
            "status": "PLANNED",
            "assessment": {},
        }
        manifest_path = runs_dir / f"{run_id}.json"
        manifest_path.write_text(json.dumps(run_manifest, indent=2, sort_keys=True), encoding="utf-8")
        runs.append(run_manifest)
    summary = {
        "experiment_id": experiment_id,
        "phase": phase,
        "target": target_name,
        "report_date": report_date,
        "spec_path": str(spec_path),
        "base_config_path": str(base_config_path),
        "acceptance": spec.get("acceptance", {}),
        "max_concurrent": max_concurrent,
        "run_count": len(runs),
        "runs": [{"run_id": run["run_id"], "status": run["status"], "model_suite": run["model_suite"], "matrix_values": run["matrix_values"]} for run in runs],
    }
    (local_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def launch_experiment(
    *,
    spec_path: Path,
    repo_root: Path,
    data_root: Path,
    local_state: Path,
    env_path: Path,
    targets_config_path: Path,
    python_executable: str,
) -> dict[str, Any]:
    """Materialize an experiment and launch pending runs up to the concurrency limit."""
    summary = plan_experiment(
        spec_path=spec_path,
        repo_root=repo_root,
        data_root=data_root,
        local_state=local_state,
        env_path=env_path,
        targets_config_path=targets_config_path,
        python_executable=python_executable,
    )
    experiment_id = str(summary["experiment_id"])
    status = experiment_status(
        experiment_id=experiment_id,
        local_state=local_state,
        repo_root=repo_root,
        data_root=data_root,
        targets_config_path=targets_config_path,
        python_executable=python_executable,
    )
    active = int(status["counts"].get("RUNNING", 0) + status["counts"].get("STARTING", 0))
    available_slots = max(0, int(summary.get("max_concurrent", 1)) - active)
    launched: list[dict[str, Any]] = []
    if available_slots <= 0:
        return {**status, "launched": launched}
    for manifest in status["runs"]:
        if available_slots <= 0:
            break
        if manifest["status"] not in {"PLANNED", "FAILED"}:
            continue
        payload = launch_training_process(
            repo_root=repo_root,
            data_root=data_root,
            local_state=local_state,
            env_path=env_path,
            phase=int(manifest["phase"]),
            model_suite=str(manifest["model_suite"]),
            python_executable=python_executable,
            report_date=str(manifest["report_date"]),
            target=str(manifest["target"]),
            targets_config_path=targets_config_path,
            runtime_name=str(manifest["runtime_name"]),
            config_path=Path(str(manifest["config_path"])),
        )
        manifest["status"] = str(payload.get("status", "STARTING")).upper()
        manifest["runtime"] = payload
        _write_run_manifest(local_state=local_state, experiment_id=experiment_id, manifest=manifest)
        launched.append({"run_id": manifest["run_id"], "target": manifest["target"], "status": manifest["status"]})
        available_slots -= 1
    refreshed = experiment_status(
        experiment_id=experiment_id,
        local_state=local_state,
        repo_root=repo_root,
        data_root=data_root,
        targets_config_path=targets_config_path,
        python_executable=python_executable,
    )
    return {**refreshed, "launched": launched}


def experiment_status(
    *,
    experiment_id: str,
    local_state: Path,
    repo_root: Path,
    data_root: Path,
    targets_config_path: Path,
    python_executable: str,
) -> dict[str, Any]:
    """Refresh and return status for all runs in an experiment."""
    local_root = _local_experiment_root(local_state, experiment_id)
    runs: list[dict[str, Any]] = []
    for path in sorted((local_root / "runs").glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        snapshot = training_status_snapshot(
            repo_root=repo_root,
            data_root=data_root,
            local_state=local_state,
            phase=int(manifest["phase"]),
            target=str(manifest["target"]),
            targets_config_path=targets_config_path,
            python_executable=python_executable,
            runtime_name=str(manifest["runtime_name"]),
            tail_lines=20,
        )
        runtime = snapshot.get("runtime") or {}
        status = str(runtime.get("status") or manifest.get("status") or "PLANNED").upper()
        report_path = Path(str(manifest["report_path"]))
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            status = "COMPLETED"
            manifest["assessment"] = report.get("assessment", {})
            manifest["report_preview"] = {
                "coverage": report.get("coverage"),
                "ridge_mean_rank_ic": report.get("ridge", {}).get("mean_rank_ic"),
                "lightgbm_mean_rank_ic": report.get("lightgbm", {}).get("mean_rank_ic"),
            }
        manifest["status"] = status
        manifest["runtime"] = runtime
        manifest["log_tail"] = snapshot.get("log_tail", "")
        _write_run_manifest(local_state=local_state, experiment_id=experiment_id, manifest=manifest)
        runs.append(manifest)
    counts: dict[str, int] = {}
    for run in runs:
        counts[run["status"]] = counts.get(run["status"], 0) + 1
    summary = {
        "experiment_id": experiment_id,
        "counts": counts,
        "run_count": len(runs),
        "runs": runs,
    }
    (local_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def compare_experiment(*, experiment_id: str, local_state: Path) -> dict[str, Any]:
    """Build a deterministic comparison table for completed experiment runs."""
    local_root = _local_experiment_root(local_state, experiment_id)
    rows: list[dict[str, Any]] = []
    for path in sorted((local_root / "runs").glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        report_path = Path(str(manifest.get("report_path")))
        if not report_path.exists():
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        primary_score = report.get("lightgbm", {}).get("mean_rank_ic")
        if primary_score is None:
            primary_score = report.get("ridge", {}).get("mean_rank_ic", 0.0)
        rows.append(
            {
                "run_id": manifest["run_id"],
                "target": manifest["target"],
                "model_suite": manifest["model_suite"],
                "matrix_values": manifest["matrix_values"],
                "coverage": report.get("coverage"),
                "ridge_mean_rank_ic": report.get("ridge", {}).get("mean_rank_ic"),
                "lightgbm_mean_rank_ic": report.get("lightgbm", {}).get("mean_rank_ic"),
                "pbo": report.get("diagnostics", {}).get("pbo"),
                "dsr": report.get("diagnostics", {}).get("dsr"),
                "decision": report.get("assessment", {}).get("decision"),
                "report_path": str(report_path),
                "primary_score": primary_score,
            }
        )
    rows.sort(key=lambda item: (float(item.get("primary_score") or 0.0), str(item["run_id"])), reverse=True)
    best = rows[0] if rows else None
    return {"experiment_id": experiment_id, "rows": rows, "best": best}


def render_experiment_report(*, experiment_id: str, local_state: Path) -> dict[str, Any]:
    """Write experiment comparison JSON and markdown reports."""
    comparison = compare_experiment(experiment_id=experiment_id, local_state=local_state)
    local_root = _local_experiment_root(local_state, experiment_id)
    json_path = local_root / "comparison.json"
    md_path = local_root / "comparison.md"
    json_path.write_text(json.dumps(comparison, indent=2, sort_keys=True), encoding="utf-8")
    md_lines = [f"# Experiment {experiment_id}", ""]
    best = comparison.get("best")
    if best:
        md_lines.extend(
            [
                f"Best run: `{best['run_id']}`",
                "",
            ]
        )
    for row in comparison["rows"]:
        md_lines.extend(
            [
                f"## {row['run_id']}",
                f"- target: {row['target']}",
                f"- model_suite: {row['model_suite']}",
                f"- coverage: {row['coverage']}",
                f"- ridge_mean_rank_ic: {row['ridge_mean_rank_ic']}",
                f"- lightgbm_mean_rank_ic: {row['lightgbm_mean_rank_ic']}",
                f"- pbo: {row['pbo']}",
                f"- dsr: {row['dsr']}",
                f"- decision: {row['decision']}",
                "",
            ]
        )
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    return {"experiment_id": experiment_id, "json_path": str(json_path), "markdown_path": str(md_path), "best": best}


def latest_experiment_summary(*, local_state: Path) -> dict[str, Any]:
    """Return the most recent experiment summary when present."""
    root = local_state / "experiments"
    summaries = sorted(root.glob("*/summary.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not summaries:
        return {}
    return json.loads(summaries[0].read_text(encoding="utf-8"))


def _load_spec(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not payload.get("experiment_id"):
        raise ValueError("experiment spec requires experiment_id")
    return payload


def _resolve_report_date(spec: dict[str, Any], *, data_root: Path) -> str:
    report_date = spec.get("report_date")
    if report_date:
        return str(report_date)
    policy = str(spec.get("report_date_policy") or "phase1_freeze")
    if policy != "phase1_freeze":
        raise ValueError(f"unsupported report_date_policy: {policy}")
    freeze = recommended_training_cutoff(
        data_root=data_root,
        expected_symbol_count=_stage_symbol_count(data_root),
    )
    resolved = freeze.get("date")
    if not resolved:
        raise ValueError("unable to resolve phase1_freeze report_date from current canonical freeze state")
    return str(resolved)


def _realize_matrix(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    if not matrix:
        return [{"matrix_values": {}, "config_overrides": {}}]
    keys = sorted(matrix)
    values = []
    for key in keys:
        raw = matrix.get(key)
        if not isinstance(raw, list) or not raw:
            raise ValueError(f"matrix dimension {key!r} must be a non-empty list")
        values.append(raw)
    rows: list[dict[str, Any]] = []
    for combination in itertools.product(*values):
        matrix_values = {key: value for key, value in zip(keys, combination, strict=True)}
        config_overrides = {key: value for key, value in matrix_values.items() if key != "model_suite"}
        rows.append({"matrix_values": matrix_values, "config_overrides": config_overrides, "model_suite": matrix_values.get("model_suite")})
    return rows


def _apply_overrides(base_config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(json.dumps(base_config))
    for dotted_path, value in overrides.items():
        cursor = payload
        parts = str(dotted_path).split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return payload


def _run_id(matrix_values: dict[str, Any]) -> str:
    serialized = json.dumps(matrix_values, sort_keys=True, default=str)
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()[:10]


def _local_experiment_root(local_state: Path, experiment_id: str) -> Path:
    return local_state / "experiments" / experiment_id


def _write_run_manifest(*, local_state: Path, experiment_id: str, manifest: dict[str, Any]) -> None:
    path = _local_experiment_root(local_state, experiment_id) / "runs" / f"{manifest['run_id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
