"""
Unified Pi Data-Node Service

This package implements a single, queue-based data node that handles:
- Forward ingest (live/continuous data collection)
- Gap detection and backfill
- Bootstrap for new symbols/history windows
- Cross-vendor QC probes
- Curate and export for training

See SSOT_V2.md §5.1 for architecture overview.
"""

__version__ = "0.1.0"
