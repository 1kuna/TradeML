"""
Calibration metrics for classifiers: Brier score, log loss, and reliability bins.

Free-friendly utilities for quick integration into reports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass
class CalibrationResult:
    brier: float
    logloss: float
    bins: pd.DataFrame  # columns: prob_bin, n, mean_pred, mean_true


def _safe_logloss(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(y_prob, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def binary_calibration(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> CalibrationResult:
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    if y_true.size == 0:
        return CalibrationResult(brier=float('nan'), logloss=float('nan'), bins=pd.DataFrame(columns=["prob_bin", "n", "mean_pred", "mean_true"]))
    brier = float(np.mean((y_prob - y_true) ** 2))
    logloss = _safe_logloss(y_true, y_prob)
    # Reliability bins
    df = pd.DataFrame({"y": y_true, "p": y_prob})
    df["prob_bin"] = pd.cut(df["p"], bins=np.linspace(0, 1, n_bins + 1), include_lowest=True)
    bins = df.groupby("prob_bin").agg(n=("y", "size"), mean_pred=("p", "mean"), mean_true=("y", "mean")).reset_index()
    return CalibrationResult(brier=brier, logloss=logloss, bins=bins)

