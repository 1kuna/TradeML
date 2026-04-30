"""Research architecture and objective registries for autonomous experiments."""

from __future__ import annotations

import contextlib
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ArchitectureEntry:
    """Declarative metadata for one research architecture lane."""

    family: str
    model_suite: str | None
    required_packages: tuple[str, ...]
    complexity_tier: int
    primary_metrics: tuple[str, ...]
    allowed_phases: tuple[int, ...]
    promotable: bool
    implemented: bool
    canary_eligible: bool
    pivot_role: str
    deferred_reason: str | None = None
    config_overrides: dict[str, Any] | None = None

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-serializable registry row."""
        payload = asdict(self)
        payload["required_packages"] = list(self.required_packages)
        payload["primary_metrics"] = list(self.primary_metrics)
        payload["allowed_phases"] = list(self.allowed_phases)
        payload["config_overrides"] = dict(self.config_overrides or {})
        return payload


ARCHITECTURE_REGISTRY: dict[str, ArchitectureEntry] = {
    "linear_baseline": ArchitectureEntry(
        family="linear_baseline",
        model_suite="ridge_only",
        required_packages=(),
        complexity_tier=0,
        primary_metrics=("ridge_mean_rank_ic",),
        allowed_phases=(1, 2),
        promotable=True,
        implemented=True,
        canary_eligible=True,
        pivot_role="sentinel",
        config_overrides={},
    ),
    "tree_challenger": ArchitectureEntry(
        family="tree_challenger",
        model_suite="full",
        required_packages=("lightgbm", "optuna"),
        complexity_tier=1,
        primary_metrics=("lightgbm_mean_rank_ic", "ridge_mean_rank_ic"),
        allowed_phases=(1, 2),
        promotable=True,
        implemented=True,
        canary_eligible=True,
        pivot_role="workhorse_sentinel",
        config_overrides={},
    ),
    "advanced_challenger": ArchitectureEntry(
        family="advanced_challenger",
        model_suite="advanced",
        required_packages=("catboost", "lightgbm", "optuna"),
        complexity_tier=2,
        primary_metrics=("catboost_mean_rank_ic", "lightgbm_mean_rank_ic", "ridge_mean_rank_ic"),
        allowed_phases=(1, 2),
        promotable=True,
        implemented=True,
        canary_eligible=True,
        pivot_role="advanced_primary",
        config_overrides={},
    ),
    "ensemble_meta": ArchitectureEntry(
        family="ensemble_meta",
        model_suite="ensemble",
        required_packages=("lightgbm", "optuna"),
        complexity_tier=3,
        primary_metrics=("ensemble_mean_rank_ic", "lightgbm_mean_rank_ic", "ridge_mean_rank_ic"),
        allowed_phases=(1, 2),
        promotable=True,
        implemented=True,
        canary_eligible=True,
        pivot_role="advanced_failure_pivot",
        config_overrides={},
    ),
    "tabular_deep_challenger": ArchitectureEntry(
        family="tabular_deep_challenger",
        model_suite=None,
        required_packages=("torch",),
        complexity_tier=4,
        primary_metrics=("tabular_deep_mean_rank_ic", "lightgbm_mean_rank_ic", "ridge_mean_rank_ic"),
        allowed_phases=(2,),
        promotable=True,
        implemented=False,
        canary_eligible=False,
        pivot_role="disabled_future",
        deferred_reason="requires a dedicated tabular deep training suite",
        config_overrides={},
    ),
}

MODEL_SUITE_TO_ARCHITECTURE = {
    "ridge_only": "linear_baseline",
    "full": "tree_challenger",
    "advanced": "advanced_challenger",
    "ensemble": "ensemble_meta",
}

OBJECTIVE_REGISTRY: dict[str, dict[str, Any]] = {
    "research_profitability_v1": {
        "primary_metric": "rank_ic",
        "promotion_metrics": [
            "rank_ic",
            "net_return",
            "cost_stress_net_return",
            "pbo",
            "yearly_positivity",
            "drawdown",
            "turnover",
        ],
        "complexity_penalty": {
            "enabled": True,
            "penalty_per_tier": 0.0005,
            "min_complexity_adjusted_improvement": 0.0,
        },
    }
}


def architecture_registry_payload() -> dict[str, dict[str, Any]]:
    """Return all architecture registry rows as JSON-serializable payloads."""
    return {family: entry.to_payload() for family, entry in ARCHITECTURE_REGISTRY.items()}


def objective_registry_payload() -> dict[str, dict[str, Any]]:
    """Return all objective registry rows."""
    return {name: dict(payload) for name, payload in OBJECTIVE_REGISTRY.items()}


def launchable_architecture_presets() -> dict[str, dict[str, Any]]:
    """Return the legacy preset shape for implemented architecture lanes."""
    return {
        family: {"model_suite": entry.model_suite, "config_overrides": dict(entry.config_overrides or {})}
        for family, entry in ARCHITECTURE_REGISTRY.items()
        if entry.implemented and entry.model_suite
    }


def model_suite_dependency_map() -> dict[str, list[str]]:
    """Return required Python packages keyed by implemented model suite."""
    dependencies: dict[str, list[str]] = {}
    for entry in ARCHITECTURE_REGISTRY.values():
        if entry.implemented and entry.model_suite and entry.required_packages:
            current = dependencies.setdefault(entry.model_suite, [])
            current.extend(package for package in entry.required_packages if package not in set(current))
    return dependencies


def resolve_architecture_entry(family: str, *, allow_deferred: bool = False) -> dict[str, Any]:
    """Resolve one architecture family or raise for unsupported/deferred lanes."""
    entry = ARCHITECTURE_REGISTRY.get(str(family))
    if entry is None:
        raise ValueError(f"unsupported architecture_family: {family}")
    if not entry.implemented and not allow_deferred:
        raise ValueError(f"architecture_family {family!r} is deferred: {entry.deferred_reason}")
    return entry.to_payload()


def architecture_family_for_manifest(manifest: dict[str, Any]) -> str:
    """Return the architecture lane represented by a run manifest."""
    matrix_values = dict(manifest.get("matrix_values") or {})
    family = matrix_values.get("architecture_family")
    if family:
        return str(family)
    return MODEL_SUITE_TO_ARCHITECTURE.get(str(manifest.get("model_suite") or ""), "linear_baseline")


def architecture_entry_for_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return registry metadata for a run manifest."""
    return resolve_architecture_entry(architecture_family_for_manifest(manifest))


def primary_score_from_report(*, manifest: dict[str, Any], report: dict[str, Any]) -> float:
    """Return the objective primary rank IC from a report using registry metric order."""
    entry = architecture_entry_for_manifest(manifest)
    for metric in entry["primary_metrics"]:
        value = _metric_from_report(report, str(metric))
        if value is not None:
            return float(value)
    return 0.0


def primary_score_from_preview(*, model_suite: str, preview: dict[str, Any], architecture_family: str | None = None) -> float | None:
    """Return the primary score from a manifest preview using registry metric order."""
    family = architecture_family or MODEL_SUITE_TO_ARCHITECTURE.get(str(model_suite or ""), "linear_baseline")
    entry = resolve_architecture_entry(family)
    for metric in entry["primary_metrics"]:
        value = preview.get(metric)
        if value is not None:
            return float(value)
    return None


def build_objective_verdict(
    *,
    manifest: dict[str, Any],
    primary_score: float,
    survived: bool,
    gate_failures_by_objective: dict[str, list[str]],
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the normalized objective verdict stored with evaluations."""
    entry = architecture_entry_for_manifest(manifest)
    merged = _objective_policy(policy)
    complexity = dict(merged.get("complexity_penalty") or {})
    penalty = float(complexity.get("penalty_per_tier") or 0.0) if bool(complexity.get("enabled", False)) else 0.0
    adjusted = float(primary_score) - int(entry["complexity_tier"]) * penalty
    return {
        "policy": str(merged.get("primary") or "research_profitability_v1"),
        "primary_metric": "rank_ic",
        "primary_score": float(primary_score),
        "complexity_tier": int(entry["complexity_tier"]),
        "complexity_penalty": penalty,
        "complexity_adjusted_score": adjusted,
        "architecture_lane": entry["family"],
        "promotable": bool(entry["promotable"]),
        "passed": bool(survived),
        "gate_failures_by_objective": gate_failures_by_objective,
    }


def complexity_adjusted_score(candidate: dict[str, Any], *, policy: dict[str, Any] | None = None) -> float | None:
    """Return a candidate score adjusted by configured complexity penalty."""
    score = _safe_float(candidate.get("primary_score") or candidate.get("rank_ic"))
    if score is None:
        return None
    merged = _objective_policy(policy)
    complexity = dict(merged.get("complexity_penalty") or {})
    if not bool(complexity.get("enabled", False)):
        return score
    tier = candidate.get("complexity_tier")
    if tier is None and candidate.get("architecture_lane"):
        with contextlib.suppress(ValueError):
            tier = resolve_architecture_entry(str(candidate["architecture_lane"]))["complexity_tier"]
    tier_int = int(tier or 0)
    return score - tier_int * float(complexity.get("penalty_per_tier") or 0.0)


def gate_failures_by_objective(failures: list[str]) -> dict[str, list[str]]:
    """Group flat gate failures into objective dimensions."""
    grouped: dict[str, list[str]] = {}
    for failure in failures:
        key = "predictive"
        if "cost" in failure or "net_return" in failure:
            key = "cost"
        elif (
            "pbo" in failure
            or "placebo" in failure
            or "coverage" in failure
            or "negative_control" in failure
            or "future_news" in failure
            or "feature_ablation" in failure
            or "single_feature" in failure
        ):
            key = "robustness"
        elif "year" in failure:
            key = "stability"
        grouped.setdefault(key, []).append(failure)
    return grouped


def _objective_policy(policy: dict[str, Any] | None) -> dict[str, Any]:
    base = {
        "primary": "research_profitability_v1",
        **OBJECTIVE_REGISTRY["research_profitability_v1"],
    }
    if policy:
        base.update(dict(policy))
        if isinstance(base.get("complexity_penalty"), dict):
            merged_penalty = dict(OBJECTIVE_REGISTRY["research_profitability_v1"]["complexity_penalty"])
            merged_penalty.update(dict(base["complexity_penalty"]))
            base["complexity_penalty"] = merged_penalty
    return base


def _metric_from_report(report: dict[str, Any], metric: str) -> float | None:
    if metric == "ridge_mean_rank_ic":
        value = report.get("ridge", {}).get("mean_rank_ic")
    elif metric == "lightgbm_mean_rank_ic":
        value = report.get("lightgbm", {}).get("mean_rank_ic")
    elif metric == "catboost_mean_rank_ic":
        catboost = report.get("catboost") or {}
        value = None if not isinstance(catboost, dict) or catboost.get("skipped") else catboost.get("mean_rank_ic")
    elif metric == "ensemble_mean_rank_ic":
        ensemble = report.get("ensemble") or {}
        value = None if not isinstance(ensemble, dict) or ensemble.get("skipped") else ensemble.get("mean_rank_ic")
    else:
        value = report.get(metric)
    if value is None:
        return None
    return float(value)


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
