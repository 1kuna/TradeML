"""
Lightweight PatchTST-style scaffold for intraday cross-sectional models.

This intentionally avoids a heavy dependency on torch while still providing
an ergonomic API: if torch is available, we fall back to a tiny temporal
convolution network; otherwise we use a scikit-learn gradient boosted tree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

try:  # Optional torch support
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover - CPU baseline is default
    torch = None  # type: ignore
    nn = None  # type: ignore


@dataclass
class PatchConfig:
    context_length: int = 128
    d_model: int = 64
    dropout: float = 0.1
    epochs: int = 10
    lr: float = 1e-3
    batch_size: int = 64


class _TinyTCN(nn.Module):  # pragma: no cover - requires torch
    def __init__(self, input_dim: int, cfg: PatchConfig):
        super().__init__()
        hidden = cfg.d_model
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x).squeeze(-1)
        return self.head(out).squeeze(-1)


def _train_torch_model(X: np.ndarray, y: np.ndarray, cfg: PatchConfig) -> Tuple[object, Dict[str, float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _TinyTCN(input_dim=X.shape[1], cfg=cfg).to(device)
    ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).float())
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(cfg.epochs):
        total_loss = 0.0
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            opt.zero_grad()
            preds = model(batch_X)
            loss = loss_fn(preds, batch_y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(batch_X)
        logger.debug(f"PatchTST epoch {epoch+1}/{cfg.epochs} loss={total_loss / len(ds):.6f}")
    model.eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(X).float().to(device)).cpu().numpy()
    metrics = {
        "rmse": float(np.sqrt(np.mean((preds - y) ** 2))),
    }
    return model, metrics


def _train_sklearn_model(X: np.ndarray, y: np.ndarray) -> Tuple[object, Dict[str, float]]:
    from sklearn.ensemble import GradientBoostingRegressor

    model = GradientBoostingRegressor(random_state=42)
    model.fit(X, y)
    preds = model.predict(X)
    metrics = {
        "rmse": float(np.sqrt(np.mean((preds - y) ** 2))),
    }
    return model, metrics


def train_patchtst(X: pd.DataFrame, y: pd.Series, cfg: Optional[PatchConfig] = None) -> Tuple[object, Dict[str, float]]:
    """
    Train a PatchTST-like model (torch when available, else a tabular baseline).

    Args:
        X: DataFrame of shape [samples, features]; time axis is expected to be
           pre-flattened (e.g., minute features per column).
        y: Series with regression targets (e.g., next-day edge).
        cfg: Optional PatchConfig; ignored if torch is unavailable.
    """
    if X.empty:
        raise ValueError("Intraday training received an empty feature matrix.")
    cfg = cfg or PatchConfig()
    values = X.to_numpy().astype(np.float32)
    target = y.to_numpy().astype(np.float32)

    if torch is not None:
        # Rearrange to [batch, channels, time] expected by conv layers
        # Assume features are already representing patches; treat each column as a channel.
        values_reshaped = values.reshape(values.shape[0], values.shape[1], 1)
        model, metrics = _train_torch_model(values_reshaped, target, cfg)
        backend = "torch"
    else:
        model, metrics = _train_sklearn_model(values, target)
        backend = "sklearn"

    metrics["backend"] = backend
    return model, metrics


def predict_patchtst(model: object, X: pd.DataFrame) -> np.ndarray:
    if torch is not None and isinstance(model, _TinyTCN):
        device = next(model.parameters()).device  # type: ignore[arg-type]
        with torch.no_grad():
            arr = X.to_numpy().astype(np.float32).reshape(X.shape[0], X.shape[1], 1)
            preds = model(torch.from_numpy(arr).to(device)).cpu().numpy()
        return preds
    if hasattr(model, "predict"):
        return np.asarray(model.predict(X))
    raise TypeError("Unsupported intraday model object for prediction.")
