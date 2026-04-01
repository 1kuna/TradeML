"""Universe-relative forward return labels."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_labels(panel: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """Build raw and universe-relative forward returns."""
    frame = panel.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["symbol", "date"]).reset_index(drop=True)
    grouped = frame.groupby("symbol", group_keys=False)
    frame[f"raw_forward_return_{horizon}d"] = grouped["close"].transform(lambda s, h=horizon: np.log(s.shift(-h) / s))
    universe_mean = frame.groupby("date")[f"raw_forward_return_{horizon}d"].transform("mean")
    frame[f"label_{horizon}d"] = frame[f"raw_forward_return_{horizon}d"] - universe_mean
    return frame[["date", "symbol", f"raw_forward_return_{horizon}d", f"label_{horizon}d"]]
