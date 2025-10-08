"""SSOT orchestration primitives: audit, backfill, curation, training gate, routing.

Implements the minimal APIs specified in Architecture_SSOT.md §14.
"""

from .audit import audit_scan
from .backfill import backfill_run
from .curate import curate_incremental
from .train_gate import train_if_ready, run_cpcv, promote_if_beat_champion
from .router import route, route_dataset

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
