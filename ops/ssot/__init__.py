"""SSOT orchestration primitives: audit, backfill, curation, training gate, routing.

Implements the minimal APIs specified in Architecture_SSOT.md §14.

Lazy-load heavy modules to avoid importing optional ML deps (e.g., sklearn)
when only lightweight utilities are needed (like reference updates).
"""

__all__ = [
    "audit_scan",
    "backfill_run",
    "curate_incremental",
    "train_if_ready",
    "run_cpcv",
    "promote_if_beat_champion",
    "route",
    "route_dataset",
]


def __getattr__(name):
    if name == "audit_scan":
        from .audit import audit_scan
        return audit_scan
    if name == "backfill_run":
        from .backfill import backfill_run
        return backfill_run
    if name == "curate_incremental":
        from .curate import curate_incremental
        return curate_incremental
    if name == "train_if_ready":
        from .train_gate import train_if_ready
        return train_if_ready
    if name == "run_cpcv":
        from .train_gate import run_cpcv
        return run_cpcv
    if name == "promote_if_beat_champion":
        from .train_gate import promote_if_beat_champion
        return promote_if_beat_champion
    if name == "route":
        from .router import route
        return route
    if name == "route_dataset":
        from .router import route_dataset
        return route_dataset
    raise AttributeError(name)
