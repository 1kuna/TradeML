"""
Data drift detection for production monitoring.

Implements:
- Population Stability Index (PSI)
- KL Divergence
- Feature distribution monitoring
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class DriftResult:
    """Drift detection result."""

    feature: str
    metric: str  # "PSI" or "KL"
    value: float
    threshold: float
    is_drifted: bool


class PSI:
    """
    Population Stability Index (PSI) calculator.

    PSI measures the shift in a variable's distribution between two samples:
        PSI = sum((actual_% - expected_%) * ln(actual_% / expected_%))

    Interpretation:
    - PSI < 0.1: No significant change
    - 0.1 <= PSI < 0.2: Moderate change
    - PSI >= 0.2: Significant change (requires investigation)
    """

    @staticmethod
    def calculate(
        expected: np.ndarray,
        actual: np.ndarray,
        n_bins: int = 10,
        epsilon: float = 1e-10,
    ) -> float:
        """
        Calculate PSI between expected and actual distributions.

        Parameters
        ----------
        expected : np.ndarray
            Baseline distribution (e.g., training data)
        actual : np.ndarray
            Current distribution (e.g., production data)
        n_bins : int
            Number of bins for discretization
        epsilon : float
            Small value to avoid log(0)

        Returns
        -------
        float
            PSI value
        """
        # Create bins based on expected distribution
        _, bins = pd.cut(expected, bins=n_bins, retbins=True, duplicates="drop")

        # Bin both distributions
        expected_binned = pd.cut(expected, bins=bins, include_lowest=True, duplicates="drop")
        actual_binned = pd.cut(actual, bins=bins, include_lowest=True, duplicates="drop")

        # Count frequencies
        expected_counts = expected_binned.value_counts(sort=False)
        actual_counts = actual_binned.value_counts(sort=False)

        # Convert to proportions
        expected_props = (expected_counts / len(expected)).fillna(0)
        actual_props = (actual_counts / len(actual)).fillna(0)

        # Add epsilon to avoid log(0)
        expected_props = expected_props + epsilon
        actual_props = actual_props + epsilon

        # Calculate PSI
        psi = ((actual_props - expected_props) * np.log(actual_props / expected_props)).sum()

        return psi


class KLDivergence:
    """
    Kullback-Leibler Divergence calculator.

    KL(P||Q) = sum(P(x) * log(P(x) / Q(x)))

    Measures how much one probability distribution diverges from another.
    """

    @staticmethod
    def calculate(
        p: np.ndarray,
        q: np.ndarray,
        n_bins: int = 10,
        epsilon: float = 1e-10,
    ) -> float:
        """
        Calculate KL divergence between two distributions.

        Parameters
        ----------
        p : np.ndarray
            Reference distribution
        q : np.ndarray
            Comparison distribution
        n_bins : int
            Number of bins
        epsilon : float
            Small value to avoid division by zero

        Returns
        -------
        float
            KL divergence
        """
        # Create bins
        _, bins = pd.cut(p, bins=n_bins, retbins=True, duplicates="drop")

        # Bin both distributions
        p_binned = pd.cut(p, bins=bins, include_lowest=True, duplicates="drop")
        q_binned = pd.cut(q, bins=bins, include_lowest=True, duplicates="drop")

        # Count frequencies
        p_counts = p_binned.value_counts(sort=False)
        q_counts = q_binned.value_counts(sort=False)

        # Convert to proportions
        p_props = (p_counts / len(p)).fillna(0) + epsilon
        q_props = (q_counts / len(q)).fillna(0) + epsilon

        # Calculate KL divergence
        kl = (p_props * np.log(p_props / q_props)).sum()

        return kl


class DriftDetector:
    """
    Feature drift detector for production monitoring.

    Monitors feature distributions and alerts on significant drift.
    """

    def __init__(
        self,
        psi_threshold: float = 0.2,
        kl_threshold: float = 0.5,
        n_bins: int = 10,
    ):
        """
        Initialize drift detector.

        Parameters
        ----------
        psi_threshold : float
            PSI threshold for drift alert (default: 0.2 = significant change)
        kl_threshold : float
            KL divergence threshold
        n_bins : int
            Number of bins for discretization
        """
        self.psi_threshold = psi_threshold
        self.kl_threshold = kl_threshold
        self.n_bins = n_bins
        self.baseline_stats: Dict[str, pd.Series] = {}

    def set_baseline(self, X: pd.DataFrame) -> None:
        """
        Set baseline feature distributions.

        Parameters
        ----------
        X : pd.DataFrame
            Baseline feature data (e.g., training set)
        """
        self.baseline_stats = {}
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                self.baseline_stats[col] = X[col].dropna()
                logger.info(f"Baseline set for feature '{col}': {len(self.baseline_stats[col])} samples")

    def detect(self, X: pd.DataFrame, metric: str = "PSI") -> List[DriftResult]:
        """
        Detect drift in current data vs baseline.

        Parameters
        ----------
        X : pd.DataFrame
            Current feature data
        metric : str
            "PSI" or "KL"

        Returns
        -------
        List[DriftResult]
            Drift results for each feature
        """
        if not self.baseline_stats:
            raise ValueError("Baseline not set. Call set_baseline() first.")

        results = []

        for col in X.columns:
            if col not in self.baseline_stats:
                logger.warning(f"Feature '{col}' not in baseline, skipping")
                continue

            if not pd.api.types.is_numeric_dtype(X[col]):
                continue

            expected = self.baseline_stats[col].values
            actual = X[col].dropna().values

            if len(actual) == 0:
                logger.warning(f"Feature '{col}' has no data, skipping")
                continue

            try:
                if metric == "PSI":
                    value = PSI.calculate(expected, actual, n_bins=self.n_bins)
                    threshold = self.psi_threshold
                elif metric == "KL":
                    value = KLDivergence.calculate(expected, actual, n_bins=self.n_bins)
                    threshold = self.kl_threshold
                else:
                    raise ValueError(f"Unknown metric: {metric}")

                is_drifted = value > threshold

                results.append(
                    DriftResult(
                        feature=col,
                        metric=metric,
                        value=value,
                        threshold=threshold,
                        is_drifted=is_drifted,
                    )
                )

                if is_drifted:
                    logger.warning(
                        f"Drift detected in '{col}': {metric}={value:.4f} (threshold={threshold})"
                    )

            except Exception as e:
                logger.error(f"Error calculating drift for '{col}': {e}")

        return results

    def summary(self, results: List[DriftResult]) -> pd.DataFrame:
        """
        Create summary DataFrame from drift results.

        Parameters
        ----------
        results : List[DriftResult]
            Drift results

        Returns
        -------
        pd.DataFrame
            Summary with columns: feature, metric, value, threshold, is_drifted
        """
        return pd.DataFrame([
            {
                "feature": r.feature,
                "metric": r.metric,
                "value": r.value,
                "threshold": r.threshold,
                "is_drifted": r.is_drifted,
            }
            for r in results
        ])
