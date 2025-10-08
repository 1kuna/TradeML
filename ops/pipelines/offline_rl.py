"""Offline RL scaffold for sizing and scheduling."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Optional

import numpy as np
import pandas as pd
from loguru import logger

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None  # type: ignore


@dataclass
class OfflineRLConfig:
    dataset_path: str
    artifact_dir: str = "models/rl/artifacts"
    algorithm: Literal["cql", "iql", "bc"] = "cql"
    test_size: float = 0.2
    random_state: int = 42
    update_router: bool = False
    router_cfg_path: str = "configs/router.yml"


def _load_dataset(path: str) -> pd.DataFrame:
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(f"RL dataset not found: {path}")
    if file.suffix == ".csv":
        df = pd.read_csv(file)
    else:
        df = pd.read_parquet(file)
    required = {"reward"}
    if not required.issubset(df.columns):
        raise ValueError(f"RL dataset missing required columns {required}")
    return df


def _split_dataset(df: pd.DataFrame, test_size: float, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_test = max(1, int(len(df) * test_size))
    df_shuffled = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return df_shuffled.iloc[n_test:].reset_index(drop=True), df_shuffled.iloc[:n_test].reset_index(drop=True)


def _default_feature_columns(df: pd.DataFrame) -> list[str]:
    state_cols = [c for c in df.columns if c.startswith("state_")]
    policy_cols = [c for c in df.columns if c.startswith("policy_")]
    feature_cols = state_cols or [c for c in df.columns if c not in {"reward", "action", "prob"}]
    return [c for c in feature_cols if c not in policy_cols]


def _train_sklearn_baseline(X_train: pd.DataFrame, y_train: pd.Series) -> object:
    from sklearn.ensemble import RandomForestRegressor

    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)
    return model


def _offline_policy_evaluation(df: pd.DataFrame) -> Dict[str, float]:
    rewards = df["reward"].to_numpy(dtype=float)
    metrics = {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
    }
    if "prob" in df.columns and "action_value" in df.columns:
        try:
            w = df["action_value"] / (df["prob"] + 1e-6)
            metrics["ipw_reward"] = float(np.mean(w * df["reward"]))
        except Exception as exc:
            logger.debug(f"Failed to compute importance weighting: {exc}")
    return metrics


def run_offline_rl(cfg: OfflineRLConfig) -> Dict:
    df = _load_dataset(cfg.dataset_path)
    if len(df) < 10:
        raise RuntimeError("RL dataset too small; need at least 10 samples.")

    feature_cols = _default_feature_columns(df)
    if not feature_cols:
        raise ValueError("Could not infer feature columns for RL training.")

    train_df, test_df = _split_dataset(df, cfg.test_size, cfg.random_state)
    X_train = train_df[feature_cols]
    y_train = train_df["reward"]
    X_test = test_df[feature_cols]
    y_test = test_df["reward"]

    model = _train_sklearn_baseline(X_train, y_train)
    preds = model.predict(X_test)
    mse = float(np.mean((preds - y_test.values) ** 2))

    artifact_dir = Path(cfg.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path: Optional[Path] = None
    if joblib is not None:
        model_path = artifact_dir / f"policy_{cfg.algorithm}.pkl"
        joblib.dump(model, model_path)
    else:
        logger.warning("joblib not available; skipping policy serialization")

    report = {
        "algorithm": cfg.algorithm,
        "dataset_path": cfg.dataset_path,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "feature_columns": feature_cols,
        "mse": mse,
        "ope": _offline_policy_evaluation(test_df),
        "model_path": str(model_path) if model_path else None,
    }
    (artifact_dir / "rl_summary.json").write_text(json.dumps(report, indent=2))
    logger.info(f"Offline RL training summary: {report}")

    if cfg.update_router and model_path:
        _update_router_policy(cfg.router_cfg_path, model_path)

    return report


def _update_router_policy(router_path: str, model_path: Path) -> None:
    import yaml

    cfg_path = Path(router_path)
    if not cfg_path.exists():
        logger.warning("router.yml not found; skipping RL policy registration")
        return
    with open(cfg_path) as f:
        data = yaml.safe_load(f) or {}
    router = data.setdefault("router", {})
    rl_cfg = router.setdefault("rl_policy", {})
    rl_cfg["enabled"] = True
    rl_cfg["model_path"] = str(model_path)
    with open(cfg_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    logger.info(f"Updated router with RL policy: {model_path}")


class MarketEnv:
    """Toy environment placeholder that mirrors portfolio responses."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        self._idx = 0

    def reset(self) -> Dict[str, float]:
        self._idx = 0
        return self._state()

    def step(self, action: float) -> tuple[Dict[str, float], float, bool, Dict]:
        reward = float(self.df.loc[self._idx, "reward"]) * action
        self._idx += 1
        done = self._idx >= len(self.df)
        return self._state(), reward, done, {}

    def _state(self) -> Dict[str, float]:
        row = self.df.loc[self._idx]
        return {k: float(row[k]) for k in self.df.columns if k.startswith("state_")}


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Run offline RL scaffold")
    parser.add_argument("--config", default="configs/training/rl.yml")
    args = parser.parse_args()

    with open(args.config) as f:
        raw_cfg = yaml.safe_load(f) or {}
    cfg = OfflineRLConfig(**raw_cfg)
    run_offline_rl(cfg)
