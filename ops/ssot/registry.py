from __future__ import annotations

"""
MLflow registry integration for model tracking and champion/challenger management.

Provides:
- log_run(): Log training run with params, metrics, and artifacts
- get_champion(): Retrieve current champion model for a family
- promote_to_champion(): Promote a challenger run to champion
- archive_model(): Demote old champions to archived state

Gracefully degrades when mlflow is not installed.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger


def _get_mlflow():
    """Get mlflow module if available."""
    try:
        import mlflow  # type: ignore
        return mlflow
    except ImportError:
        return None


def _get_mlflow_client():
    """Get MLflow tracking client."""
    mlflow = _get_mlflow()
    if mlflow is None:
        return None
    try:
        from mlflow.tracking import MlflowClient
        return MlflowClient()
    except Exception:
        return None


def _ensure_experiment(experiment_name: str) -> Optional[str]:
    """Ensure experiment exists and return its ID."""
    mlflow = _get_mlflow()
    if mlflow is None:
        return None
    try:
        exp = mlflow.get_experiment_by_name(experiment_name)
        if exp is None:
            exp_id = mlflow.create_experiment(experiment_name)
        else:
            exp_id = exp.experiment_id
        return exp_id
    except Exception as e:
        logger.warning(f"Failed to ensure experiment '{experiment_name}': {e}")
        return None


def log_run(
    experiment_name: str,
    params: Dict[str, Any],
    metrics: Dict[str, float],
    artifacts_dir: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
    model_path: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Optional[str]:
    """
    Log a training run to MLflow.

    Args:
        experiment_name: Name of the experiment (e.g., 'equities_xs', 'options_vol')
        params: Training parameters (hyperparameters, config snapshot)
        metrics: Training metrics (sharpe, dsr, pbo, max_dd, etc.)
        artifacts_dir: Optional directory containing artifacts to log
        tags: Optional tags (e.g., {'stage': 'challenger', 'version': '1.0'})
        model_path: Optional path to model file for registration
        model_name: Optional name for registered model

    Returns:
        run_id if successful, None otherwise
    """
    mlflow = _get_mlflow()
    if mlflow is None:
        logger.info("MLflow not installed; skipping run logging")
        return None

    try:
        exp_id = _ensure_experiment(experiment_name)
        if exp_id is None:
            return None

        mlflow.set_experiment(experiment_name)

        with mlflow.start_run() as run:
            # Log parameters
            if params:
                # Flatten nested dicts for MLflow compatibility
                flat_params = {}
                for k, v in params.items():
                    if isinstance(v, dict):
                        for k2, v2 in v.items():
                            flat_params[f"{k}.{k2}"] = str(v2)
                    else:
                        flat_params[k] = str(v)
                mlflow.log_params(flat_params)

            # Log metrics
            if metrics:
                mlflow.log_metrics(metrics)

            # Log artifacts
            if artifacts_dir and Path(artifacts_dir).exists():
                mlflow.log_artifacts(artifacts_dir)

            # Set tags
            default_tags = {"stage": "challenger"}
            if tags:
                default_tags.update(tags)
            mlflow.set_tags(default_tags)

            # Register model if path provided
            if model_path and model_name and Path(model_path).exists():
                try:
                    mlflow.log_artifact(model_path)
                    # Register in model registry
                    model_uri = f"runs:/{run.info.run_id}/{Path(model_path).name}"
                    mlflow.register_model(model_uri, model_name)
                except Exception as e:
                    logger.warning(f"Model registration failed: {e}")

            logger.info(f"Logged run {run.info.run_id} to experiment '{experiment_name}'")
            return run.info.run_id

    except Exception as e:
        logger.error(f"Failed to log run: {e}")
        return None


def get_champion(model_name: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Get the current champion model for a model family.

    Args:
        model_name: Name of the model family (e.g., 'equities_xs')

    Returns:
        Tuple of (run_id, metrics_dict) if champion exists, None otherwise
    """
    mlflow = _get_mlflow()
    client = _get_mlflow_client()
    if mlflow is None or client is None:
        logger.debug("MLflow not available; no champion retrieval")
        return None

    try:
        # Search for runs with stage=Champion tag
        experiment = mlflow.get_experiment_by_name(model_name)
        if experiment is None:
            return None

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.stage = 'Champion'",
            order_by=["start_time DESC"],
            max_results=1,
        )

        if not runs:
            logger.info(f"No champion found for '{model_name}'")
            return None

        champion_run = runs[0]
        metrics = champion_run.data.metrics
        logger.info(f"Found champion for '{model_name}': run_id={champion_run.info.run_id}")
        return (champion_run.info.run_id, metrics)

    except Exception as e:
        logger.warning(f"Failed to get champion for '{model_name}': {e}")
        return None


def get_challenger_runs(
    model_name: str,
    min_sharpe: Optional[float] = None,
    max_pbo: Optional[float] = None,
    limit: int = 10,
) -> List[Tuple[str, Dict[str, float]]]:
    """
    Get challenger runs that meet promotion criteria.

    Args:
        model_name: Name of the model family
        min_sharpe: Minimum Sharpe ratio filter
        max_pbo: Maximum PBO filter
        limit: Maximum number of runs to return

    Returns:
        List of (run_id, metrics_dict) tuples sorted by Sharpe descending
    """
    mlflow = _get_mlflow()
    client = _get_mlflow_client()
    if mlflow is None or client is None:
        return []

    try:
        experiment = mlflow.get_experiment_by_name(model_name)
        if experiment is None:
            return []

        # Search for challenger runs
        filter_parts = ["tags.stage = 'challenger'"]
        if min_sharpe is not None:
            filter_parts.append(f"metrics.sharpe >= {min_sharpe}")
        if max_pbo is not None:
            filter_parts.append(f"metrics.pbo <= {max_pbo}")

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=" AND ".join(filter_parts),
            order_by=["metrics.sharpe DESC"],
            max_results=limit,
        )

        return [(r.info.run_id, r.data.metrics) for r in runs]

    except Exception as e:
        logger.warning(f"Failed to get challengers: {e}")
        return []


def promote_to_champion(model_name: str, run_id: str) -> bool:
    """
    Promote a challenger run to champion status.

    This will:
    1. Archive the current champion (if any)
    2. Set the new run's stage tag to 'Champion'

    Args:
        model_name: Name of the model family
        run_id: Run ID to promote

    Returns:
        True if successful, False otherwise
    """
    mlflow = _get_mlflow()
    client = _get_mlflow_client()
    if mlflow is None or client is None:
        logger.info("MLflow not available; skipping promotion")
        return False

    try:
        # Archive current champion first
        current_champion = get_champion(model_name)
        if current_champion is not None:
            old_run_id, _ = current_champion
            if old_run_id != run_id:
                archive_model(model_name, old_run_id)

        # Promote the new champion
        client.set_tag(run_id, "stage", "Champion")
        logger.info(f"Promoted run {run_id} to Champion for '{model_name}'")
        return True

    except Exception as e:
        logger.error(f"Failed to promote run {run_id}: {e}")
        return False


def archive_model(model_name: str, run_id: str) -> bool:
    """
    Archive a model run (demote from champion/challenger to archived).

    Args:
        model_name: Name of the model family
        run_id: Run ID to archive

    Returns:
        True if successful, False otherwise
    """
    client = _get_mlflow_client()
    if client is None:
        return False

    try:
        client.set_tag(run_id, "stage", "Archived")
        logger.info(f"Archived run {run_id} for '{model_name}'")
        return True
    except Exception as e:
        logger.warning(f"Failed to archive run {run_id}: {e}")
        return False


def get_model_artifact_path(model_name: str, run_id: Optional[str] = None) -> Optional[str]:
    """
    Get the artifact path for a model.

    Args:
        model_name: Name of the model family
        run_id: Specific run ID, or None to get champion

    Returns:
        Artifact URI if found, None otherwise
    """
    mlflow = _get_mlflow()
    client = _get_mlflow_client()
    if mlflow is None or client is None:
        return None

    try:
        if run_id is None:
            champion = get_champion(model_name)
            if champion is None:
                return None
            run_id, _ = champion

        run = client.get_run(run_id)
        return run.info.artifact_uri

    except Exception as e:
        logger.warning(f"Failed to get artifact path: {e}")
        return None


def load_champion_model(model_name: str) -> Optional[Any]:
    """
    Load the champion model for a family.

    Args:
        model_name: Name of the model family

    Returns:
        Loaded model object if found, None otherwise
    """
    mlflow = _get_mlflow()
    if mlflow is None:
        return None

    try:
        champion = get_champion(model_name)
        if champion is None:
            logger.info(f"No champion found for '{model_name}'")
            return None

        run_id, _ = champion
        model_uri = f"runs:/{run_id}/model"

        # Try different model flavors
        for loader in [mlflow.sklearn, mlflow.lightgbm, mlflow.catboost, mlflow.pyfunc]:
            try:
                model = loader.load_model(model_uri)
                logger.info(f"Loaded champion model for '{model_name}' from run {run_id}")
                return model
            except Exception:
                continue

        logger.warning(f"Could not load model from {model_uri}")
        return None

    except Exception as e:
        logger.error(f"Failed to load champion model: {e}")
        return None

