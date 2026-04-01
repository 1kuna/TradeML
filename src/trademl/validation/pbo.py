"""Probability of backtest overfitting diagnostics."""

from __future__ import annotations

import numpy as np

from trademl.validation.cpcv import CPCVFoldResult


def probability_of_backtest_overfitting(results: list[CPCVFoldResult]) -> float:
    """Estimate PBO from CPCV path rankings."""
    if not results:
        return 0.0
    in_sample = np.array([result.in_sample_score for result in results], dtype=float)
    out_of_sample = np.array([result.out_of_sample_score for result in results], dtype=float)
    if len(results) < 2:
        return float(out_of_sample[0] < in_sample[0])
    chosen = in_sample >= np.median(in_sample)
    oos_percentiles = np.argsort(np.argsort(out_of_sample)).astype(float) / max(1, len(out_of_sample) - 1)
    return float(np.mean(oos_percentiles[chosen] < 0.5))
