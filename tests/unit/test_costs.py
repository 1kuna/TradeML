from __future__ import annotations

import pytest
import pandas as pd

from trademl.costs.models import apply_costs


def test_cost_model_applies_spread_and_stress() -> None:
    trades = pd.DataFrame([{"trade_value": 10_000.0, "adv": 1_000_000.0}])
    base = apply_costs(trades, {"spread_bps": 5.0, "stress_multiplier": 1.0})
    stressed = apply_costs(trades, {"spread_bps": 5.0, "stress_multiplier": 2.0})

    assert base.iloc[0]["spread_cost"] == pytest.approx(5.0)
    assert stressed.iloc[0]["spread_cost"] == pytest.approx(10.0)
