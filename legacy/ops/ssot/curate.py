from __future__ import annotations

"""Thin wrapper to run the existing curator incrementally (detect new raw)."""

from loguru import logger


def curate_incremental() -> None:
    try:
        from scripts.curator import Curator
        c = Curator("configs/curator.yml")
        c.run()
    except Exception as e:
        logger.exception(f"Curator failed: {e}")

