from __future__ import annotations

import numpy as np

from trademl.models.utils import time_decay_weights


def test_time_decay_weights_match_existing_model_contract() -> None:
    weights = time_decay_weights(4, half_life_days=2)

    np.testing.assert_allclose(weights, np.array([0.5**1.5, 0.5, 0.5**0.5, 1.0]))


def test_time_decay_weights_handles_empty_and_single_row() -> None:
    assert time_decay_weights(0).tolist() == []
    assert time_decay_weights(1).tolist() == [1.0]
