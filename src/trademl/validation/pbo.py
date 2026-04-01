"""Probability of backtest overfitting diagnostics."""

from __future__ import annotations

import numpy as np

from trademl.validation.cpcv import CPCVFoldResult


def probability_of_backtest_overfitting(results: list[CPCVFoldResult]) -> float:
    """Estimate PBO from CPCV out-of-fold predictions."""
    if not results:
        return 0.0
    fold_scores = [float(result.oof_predictions["prediction"].corr(result.oof_predictions.iloc[:, 2], method="spearman")) for result in results]
    below_zero = np.mean([score < 0 for score in fold_scores])
    return float(below_zero)
