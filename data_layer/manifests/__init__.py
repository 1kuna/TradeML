"""
Manifest management for connector bookmarks and backfill progress tracking.

SSOT v2 Section 2.4: Backfill subsystem

This module provides:
- ConnectorBookmark: Track last fetch date per vendor/dataset
- BackfillMark: Track backfill progress through historical gaps
- ManifestManager: Unified interface for managing manifests
"""

from .manager import (
    ConnectorBookmark,
    BackfillMark,
    ManifestManager,
    get_manifest_manager,
    init_manifests,
)

__all__ = [
    "ConnectorBookmark",
    "BackfillMark",
    "ManifestManager",
    "get_manifest_manager",
    "init_manifests",
]
