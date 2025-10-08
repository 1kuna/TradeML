#!/usr/bin/env python3
"""Export trained models to ONNX format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None  # type: ignore

from loguru import logger

from models.equities_xs import (
    export_catboost_onnx,
    export_lgbm_onnx,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export model artifact to ONNX")
    parser.add_argument("--model", default="models/equities_xs/artifacts/latest/model.pkl")
    parser.add_argument("--features", default="models/equities_xs/artifacts/latest/feature_list.json")
    parser.add_argument("--output", default="models/equities_xs/artifacts/latest/model.onnx")
    return parser.parse_args()


def load_feature_list(path: Path) -> list[str]:
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data.get("features", [])
    if isinstance(data, list):
        return data
    raise ValueError("Feature list must be a list or contain 'features'")


def main() -> None:
    if joblib is None:
        raise ImportError("joblib is required to load serialized models")

    args = parse_args()
    model_path = Path(args.model)
    feature_path = Path(args.features)
    output_path = Path(args.output)

    model = joblib.load(model_path)
    features = load_feature_list(feature_path)

    try:
        if hasattr(model, "booster_"):
            export_lgbm_onnx(model, features, output_path)
        elif model.__class__.__module__.startswith("catboost"):
            export_catboost_onnx(model, output_path)
        else:
            raise TypeError("Unsupported model type for ONNX export")
    except Exception as exc:
        logger.error(f"ONNX export failed: {exc}")
        raise

    logger.info(f"Model exported to {output_path}")


if __name__ == "__main__":
    main()
