"""Deflated Sharpe ratio helper."""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm


def deflated_sharpe_ratio(returns: np.ndarray, *, num_trials: int = 1) -> float:
    """Compute a lightweight deflated Sharpe ratio approximation."""
    if len(returns) < 2:
        return 0.0
    mean = float(np.mean(returns))
    std = float(np.std(returns, ddof=1))
    if std == 0:
        return 0.0
    sharpe = mean / std * math.sqrt(252)
    expected_max = norm.ppf(1 - 1 / max(num_trials, 2))
    return float(norm.cdf(sharpe - expected_max))
