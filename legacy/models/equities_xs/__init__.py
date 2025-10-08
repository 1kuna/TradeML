"""Cross-sectional equities models (baselines and challengers)."""

from .baselines import (
    train_ridge_regression,
    train_logistic_regression,
)

__all__ = [
    "train_ridge_regression",
    "train_logistic_regression",
]

