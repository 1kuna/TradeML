from __future__ import annotations

"""
Generate a simple coverage heatmap (date x table) from partition_status.

Output: ops/reports/coverage_YYYY-MM-DD.png
"""

import os
from datetime import date
from pathlib import Path

try:  # pragma: no cover - only when matplotlib is installed
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:  # pragma: no cover
    plt = None
    _HAS_MPL = False
import pandas as pd
from loguru import logger


def coverage_heatmap(ledger_path: str = "data_layer/qc/partition_status.parquet", out_dir: str = "ops/reports") -> str:
    path = Path(ledger_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        logger.warning(f"Ledger not found for coverage heatmap: {path}")
        return ""

    df = pd.read_parquet(path)
    if df.empty:
        logger.warning("Ledger empty; skipping coverage heatmap")
        return ""

    # Pivot date x table_name with GREEN ratio across symbols
    # For each (table, dt): ratio of GREEN statuses among symbols
    grp = df.groupby(["table_name", "dt"]).apply(lambda g: (g["status"] == "GREEN").mean()).reset_index(name="green_ratio")
    if grp.empty:
        return ""

    # Limit to recent 120 days for readability
    grp = grp.sort_values("dt")
    recent_dates = grp["dt"].drop_duplicates().sort_values().tail(120)
    grp = grp[grp["dt"].isin(recent_dates)]

    pivot = grp.pivot(index="table_name", columns="dt", values="green_ratio").fillna(0.0)

    if not _HAS_MPL:
        logger.warning("matplotlib not available; skipping coverage heatmap render")
        return ""

    plt.figure(figsize=(min(18, 1 + len(recent_dates) * 0.15), 2 + 0.5 * len(pivot)))
    im = plt.imshow(pivot.values, aspect="auto", cmap="YlGn", vmin=0, vmax=1)
    plt.colorbar(im, label="GREEN ratio")
    plt.yticks(range(len(pivot.index)), pivot.index)
    cols = pivot.columns.tolist()
    step = max(1, len(cols) // 12)
    idxs = list(range(0, len(cols), step))
    labels = [str(cols[i]) for i in idxs]
    plt.xticks(idxs, labels, rotation=45, ha="right")
    plt.title("Data Coverage (GREEN ratio by table x day)")
    plt.tight_layout()

    out_path = out / f"coverage_{date.today().isoformat()}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info(f"Coverage heatmap written: {out_path}")
    return str(out_path)
