"""Transaction cost models."""

from __future__ import annotations

import numpy as np
import pandas as pd


def apply_costs(trades: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Apply spread and square-root impact costs to trade notional."""
    frame = trades.copy()
    spread_bps = float(config.get("spread_bps", 5.0))
    impact_coefficient = float(config.get("impact_coefficient", 0.0))
    stress_multiplier = float(config.get("stress_multiplier", 1.0))
    notional = frame["trade_value"].abs()
    frame["spread_cost"] = notional * (spread_bps / 10000.0) * stress_multiplier
    adv = frame.get("adv", pd.Series(np.inf, index=frame.index)).replace(0, np.nan)
    frame["impact_cost"] = notional * impact_coefficient * np.sqrt((notional / adv).fillna(0.0)) * stress_multiplier
    frame["cost"] = frame["spread_cost"] + frame["impact_cost"]
    return frame
