from __future__ import annotations

"""
Optional MLflow registry integration wrappers.
Gracefully degrade when mlflow is not installed.
"""

from typing import Dict, Optional
from loguru import logger


def _get_mlflow():
    try:
        import mlflow  # type: ignore
        return mlflow
    except Exception:
        return None


def log_run(metrics: Dict, params: Optional[Dict] = None, tags: Optional[Dict] = None) -> None:
    mlflow = _get_mlflow()
    if mlflow is None:
        logger.info("MLflow not installed; skipping registry logging")
        return
    with mlflow.start_run():
        if params:
            mlflow.log_params(params)
        if metrics:
            mlflow.log_metrics(metrics)
        if tags:
            mlflow.set_tags(tags)
        logger.info("Logged run to MLflow")


def register_model(model_name: str, artifacts_path: str, metrics: Optional[Dict] = None, stage: str = "None") -> None:
    mlflow = _get_mlflow()
    if mlflow is None:
        logger.info("MLflow not installed; skipping model registration")
        return
    # Log as artifact and register
    with mlflow.start_run():
        mlflow.log_artifacts(artifacts_path)
        if metrics:
            mlflow.log_metrics(metrics)
        try:
            mlflow.sklearn.log_model(None, model_name)  # placeholder; real impl stores a model object
        except Exception:
            pass
        logger.info(f"Model '{model_name}' processed for registration (placeholder)")

