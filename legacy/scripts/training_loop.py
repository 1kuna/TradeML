#!/usr/bin/env python3
"""
One-click training loop (Windows-friendly):
- Loads env, validates curated data availability and GPU.
- Runs GREEN-gated training for equities_xs in a loop.
- After each train, runs CPCV evaluation and attempts champion promotion.
- Sleeps between iterations (TRAIN_INTERVAL_SECONDS, default 6 hours).

Stop with Ctrl+C.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def main():
    load_dotenv(dotenv_path=REPO_ROOT / ".env", override=False)

    # Import trainer selfchecks
    from scripts.trainer import curated_available, detect_gpu
    from ops.ssot import train_if_ready, run_cpcv, promote_if_beat_champion

    interval = int(os.getenv("TRAIN_INTERVAL_SECONDS", "21600"))  # 6 hours

    # Self-checks (non-fatal warnings if curated not yet ready)
    has_curated = curated_available()
    if not has_curated:
        logger.warning("Curated dataset not found yet — waiting for node to populate.")

    has_gpu, msg = detect_gpu()
    logger.info(msg)
    if not has_gpu:
        logger.error("CUDA GPU required for training loop. Exiting.")
        sys.exit(2)

    logger.info("Starting training loop (Ctrl+C to stop)")
    while True:
        try:
            # 1) Train if GREEN thresholds satisfied
            train_if_ready("equities_xs")
            # Optional: train options_vol if enabled
            try:
                if os.getenv("TRAIN_ENABLE_OPTIONS_VOL", "true").lower() == "true":
                    train_if_ready("options_vol")
            except Exception as e:
                logger.warning(f"options_vol training step failed: {e}")
            # 2) Evaluate CPCV (best-effort)
            try:
                res = run_cpcv("equities_xs")
                logger.info(f"CPCV summary: {res.get('result',{}).get('summary',{}) if isinstance(res, dict) else res}")
            except Exception as e:
                logger.warning(f"CPCV evaluation failed: {e}")
            # 3) Attempt promotion to Champion if bars met
            try:
                promote_if_beat_champion("equities_xs")
            except Exception as e:
                logger.warning(f"Promotion step failed: {e}")
        except Exception as e:
            logger.exception(f"Training iteration failed: {e}")
        finally:
            logger.info(f"Sleeping {interval}s before next iteration")
            try:
                time.sleep(interval)
            except KeyboardInterrupt:
                logger.warning("Interrupted — stopping training loop")
                break


if __name__ == "__main__":
    main()
