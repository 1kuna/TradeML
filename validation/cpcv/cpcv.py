"""
Combinatorially Purged Cross-Validation (CPCV).

Based on "Advances in Financial Machine Learning" by Marcos López de Prado.

Key features:
1. Purging: Remove samples with overlapping label periods
2. Embargo: Add buffer after test set to prevent leakage
3. Combinatorial: Test all fold combinations
4. Walk-forward awareness: Respect temporal order
"""

from datetime import date, timedelta
from typing import List, Tuple, Dict, Optional
from itertools import combinations
import pandas as pd
import numpy as np
from loguru import logger


class CPCV:
    """
    Combinatorially Purged Cross-Validation with Embargo.

    Prevents data leakage from:
    1. Overlapping labels (purging)
    2. Information flow from future (embargo)
    3. Non-representative train/test splits (combinatorial testing)
    """

    def __init__(
        self,
        n_folds: int = 8,
        embargo_days: int = 10,
        purge_pct: float = 0.01,
    ):
        """
        Initialize CPCV.

        Args:
            n_folds: Number of CV folds
            embargo_days: Days to embargo after test set
            purge_pct: Percentage of samples to purge around test set
        """
        self.n_folds = n_folds
        self.embargo_days = embargo_days
        self.purge_pct = purge_pct

        logger.info(
            f"CPCV initialized: {n_folds} folds, {embargo_days}d embargo, "
            f"{purge_pct*100:.1f}% purge"
        )

    def get_label_times(
        self,
        labels: pd.DataFrame,
        date_col: str = 'date',
        horizon_col: str = 'horizon_days'
    ) -> pd.DataFrame:
        """
        Calculate label end times (when label is determined).

        Args:
            labels: DataFrame with labels
            date_col: Column with label start date
            horizon_col: Column with label horizon in days

        Returns:
            DataFrame with t0 (start) and t1 (end) columns
        """
        label_times = labels[[date_col, horizon_col]].copy()
        label_times.columns = ['t0', 'horizon']

        # Calculate label end time
        label_times['t1'] = label_times.apply(
            lambda row: row['t0'] + timedelta(days=row['horizon']),
            axis=1
        )

        return label_times[['t0', 't1']]

    def purge_train_set(
        self,
        label_times: pd.DataFrame,
        train_indices: np.ndarray,
        test_indices: np.ndarray
    ) -> np.ndarray:
        """
        Remove training samples that overlap with test labels.

        Args:
            label_times: DataFrame with t0, t1 for each sample
            train_indices: Training set indices
            test_indices: Test set indices

        Returns:
            Purged training indices
        """
        test_t0_min = label_times.iloc[test_indices]['t0'].min()
        test_t1_max = label_times.iloc[test_indices]['t1'].max()

        # Find training samples that overlap with test period
        overlap_mask = (
            (label_times.iloc[train_indices]['t1'] >= test_t0_min) &
            (label_times.iloc[train_indices]['t0'] <= test_t1_max)
        )

        # Remove overlapping samples
        purged_train = train_indices[~overlap_mask.values]

        n_purged = len(train_indices) - len(purged_train)
        if n_purged > 0:
            logger.debug(f"Purged {n_purged} training samples (overlap with test)")

        return purged_train

    def apply_embargo(
        self,
        label_times: pd.DataFrame,
        train_indices: np.ndarray,
        test_indices: np.ndarray
    ) -> np.ndarray:
        """
        Apply embargo period after test set.

        Args:
            label_times: DataFrame with t0, t1
            train_indices: Training indices
            test_indices: Test indices

        Returns:
            Embargoed training indices
        """
        test_t1_max = label_times.iloc[test_indices]['t1'].max()
        embargo_end = test_t1_max + timedelta(days=self.embargo_days)

        # Remove training samples in embargo period
        embargo_mask = (
            label_times.iloc[train_indices]['t0'] <= embargo_end
        ) & (
            label_times.iloc[train_indices]['t0'] > test_t1_max
        )

        embargoed_train = train_indices[~embargo_mask.values]

        n_embargoed = len(train_indices) - len(embargoed_train)
        if n_embargoed > 0:
            logger.debug(f"Embargoed {n_embargoed} training samples")

        return embargoed_train

    def split(
        self,
        X: pd.DataFrame,
        labels: pd.DataFrame,
        date_col: str = 'date',
        horizon_col: str = 'horizon_days'
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test splits with purging and embargo.

        Args:
            X: Features DataFrame
            labels: Labels DataFrame with date and horizon
            date_col: Date column name
            horizon_col: Horizon column name

        Returns:
            List of (train_indices, test_indices) tuples
        """
        n_samples = len(X)

        # Calculate label times
        label_times = self.get_label_times(labels, date_col, horizon_col)

        # Create folds based on time
        fold_size = n_samples // self.n_folds
        indices = np.arange(n_samples)

        splits = []

        for fold in range(self.n_folds):
            # Define test set
            test_start = fold * fold_size
            test_end = test_start + fold_size if fold < self.n_folds - 1 else n_samples
            test_indices = indices[test_start:test_end]

            # Define train set (everything else)
            train_indices = np.concatenate([
                indices[:test_start],
                indices[test_end:]
            ])

            # Purge overlapping samples
            train_indices = self.purge_train_set(label_times, train_indices, test_indices)

            # Apply embargo
            train_indices = self.apply_embargo(label_times, train_indices, test_indices)

            if len(train_indices) > 0 and len(test_indices) > 0:
                splits.append((train_indices, test_indices))

        logger.info(f"Generated {len(splits)} CPCV splits")
        return splits

    def combinatorial_split(
        self,
        X: pd.DataFrame,
        labels: pd.DataFrame,
        n_test_folds: int = 2,
        date_col: str = 'date',
        horizon_col: str = 'horizon_days'
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate combinatorial train/test splits.

        Tests all combinations of N folds as test sets.

        Args:
            X: Features
            labels: Labels
            n_test_folds: Number of folds to use as test set
            date_col: Date column
            horizon_col: Horizon column

        Returns:
            List of (train_indices, test_indices) tuples
        """
        n_samples = len(X)
        label_times = self.get_label_times(labels, date_col, horizon_col)

        # Create fold indices
        fold_size = n_samples // self.n_folds
        indices = np.arange(n_samples)

        fold_indices = []
        for fold in range(self.n_folds):
            start = fold * fold_size
            end = start + fold_size if fold < self.n_folds - 1 else n_samples
            fold_indices.append(indices[start:end])

        # Generate all combinations of n_test_folds
        splits = []
        for test_fold_combo in combinations(range(self.n_folds), n_test_folds):
            # Combine test folds
            test_indices = np.concatenate([fold_indices[i] for i in test_fold_combo])

            # Train on remaining folds
            train_fold_combo = [i for i in range(self.n_folds) if i not in test_fold_combo]
            train_indices = np.concatenate([fold_indices[i] for i in train_fold_combo])

            # Purge and embargo
            train_indices = self.purge_train_set(label_times, train_indices, test_indices)
            train_indices = self.apply_embargo(label_times, train_indices, test_indices)

            if len(train_indices) > 0 and len(test_indices) > 0:
                splits.append((train_indices, test_indices))

        logger.info(f"Generated {len(splits)} combinatorial CPCV splits")
        return splits

    def score_splits(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        splits: List[Tuple[np.ndarray, np.ndarray]],
        metric_fn=None
    ) -> Dict[str, float]:
        """
        Score model on all splits.

        Args:
            model: Sklearn-compatible model
            X: Features
            y: Labels
            splits: List of (train_idx, test_idx) tuples
            metric_fn: Scoring function (default: accuracy)

        Returns:
            Dict with metrics
        """
        if metric_fn is None:
            from sklearn.metrics import accuracy_score
            metric_fn = accuracy_score

        scores = []

        for i, (train_idx, test_idx) in enumerate(splits):
            # Train
            model.fit(X.iloc[train_idx], y.iloc[train_idx])

            # Predict
            y_pred = model.predict(X.iloc[test_idx])

            # Score
            score = metric_fn(y.iloc[test_idx], y_pred)
            scores.append(score)

            logger.debug(f"Split {i+1}/{len(splits)}: score={score:.4f}")

        return {
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'min_score': np.min(scores),
            'max_score': np.max(scores),
            'scores': scores
        }


# CLI for testing
if __name__ == "__main__":
    # Simple test
    logger.info("Testing CPCV...")

    # Create dummy data
    np.random.seed(42)
    n_samples = 1000

    dates = pd.date_range('2023-01-01', periods=n_samples, freq='D')
    X = pd.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'date': dates
    })

    labels = pd.DataFrame({
        'date': dates,
        'horizon_days': 5,
        'label': np.random.randint(0, 2, n_samples)
    })

    # Initialize CPCV
    cv = CPCV(n_folds=8, embargo_days=10)

    # Generate splits
    splits = cv.split(X, labels)

    print(f"\n[OK] Generated {len(splits)} splits")

    # Show split sizes
    for i, (train_idx, test_idx) in enumerate(splits[:3]):
        print(f"Split {i+1}: train={len(train_idx)}, test={len(test_idx)}")

    # Test combinatorial
    comb_splits = cv.combinatorial_split(X, labels, n_test_folds=2)
    print(f"\n[OK] Generated {len(comb_splits)} combinatorial splits")
