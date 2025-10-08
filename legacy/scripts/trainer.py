#!/usr/bin/env python3
"""
Trainer: GPU-required training entrypoint (Windows-friendly).

Goals:
- Require CUDA GPU and fail fast if not available (no CPU fallback).
- Load curated datasets from S3/local, build a simple baseline pipeline, and save artifacts.
- Provide self-checks that validate dataset presence and environment.

This is a scaffold focused on ergonomics; extend with your actual feature/label pipeline.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Tuple

from dotenv import load_dotenv
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def detect_gpu() -> Tuple[bool, str]:
    # Try PyTorch
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return True, "CUDA via PyTorch"
    except Exception:
        pass

    # Try nvidia-smi presence
    try:
        import subprocess

        subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True, "CUDA detected via nvidia-smi"
    except Exception:
        pass

    return False, "No GPU detected (CPU mode)"


def curated_available() -> bool:
    backend = os.getenv("STORAGE_BACKEND", "s3").lower()
    if backend == "s3":
        try:
            from data_layer.storage.s3_client import get_s3_client

            s3 = get_s3_client()
            objs = s3.list_objects(prefix="curated/equities_ohlcv_adj/")
            return len(objs) > 0
        except Exception as e:
            logger.error(f"S3 check failed: {e}")
            return False
    else:
        path = REPO_ROOT / "data_layer/curated/equities_ohlcv_adj"
        return path.exists()


def selfcheck() -> bool:
    ok = True
    has_curated = curated_available()
    if not has_curated:
        logger.error("No curated dataset found under curated/equities_ohlcv_adj. Run node (collector+curator) first.")
        ok = False

    gpu, msg = detect_gpu()
    logger.info(msg)
    if not gpu:
        logger.error("CUDA GPU required. Install NVIDIA drivers + CUDA (WSL2 on Windows recommended) and ensure nvidia-smi works.")
        ok = False

    return ok


def run_training(output_dir: Path):
    # Placeholder: train a simple baseline (CPU/GPU agnostic). Replace with real pipeline.
    import pandas as pd
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_squared_error

    # Minimal example: synthesize a dataset if curated not easily readable here.
    # In real use, load your features/labels from feature_store + labels.
    import numpy as np

    n = 2000
    X = np.random.randn(n, 10)
    y = X @ np.linspace(0.5, -0.5, 10) + 0.1 * np.random.randn(n)

    model = Ridge(alpha=1.0)
    tscv = TimeSeriesSplit(n_splits=5)
    mses = []
    for train_idx, test_idx in tscv.split(X):
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        mses.append(mean_squared_error(y[test_idx], pred))

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.txt").write_text(f"MSE mean={float(np.mean(mses)):.6f}\nMSE std={float(np.std(mses)):.6f}\n")
    logger.info(f"Saved metrics to {output_dir / 'metrics.txt'}")


def main():
    parser = argparse.ArgumentParser(description="GPU-aware trainer (Windows-friendly)")
    parser.add_argument("--selfcheck", action="store_true", help="Validate environment and datasets, then exit")
    parser.add_argument("--output", default=str(REPO_ROOT / "artifacts" / "trainer"))
    args = parser.parse_args()

    load_dotenv(dotenv_path=REPO_ROOT / ".env", override=False)

    if args.selfcheck:
        ok = selfcheck()
        sys.exit(0 if ok else 2)

    if not selfcheck():
        sys.exit(2)

    gpu, msg = detect_gpu()
    logger.info(f"Training start ({msg})")
    run_training(Path(args.output))
    logger.info("Training complete")


if __name__ == "__main__":
    main()
