"""Equities feature store.

Provides PIT-safe feature engineering utilities for cross-sectional models.
"""

from .features import compute_equity_features  # noqa: F401

__all__ = [
    "compute_equity_features",
]

