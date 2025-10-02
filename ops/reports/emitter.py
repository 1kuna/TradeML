"""
Reporting emitters for daily outputs.

Implements a minimal `emit_daily` to write equities positions and metrics
to JSON and Markdown in `ops/reports/`.
"""

from __future__ import annotations

import json
import os
from datetime import date as Date, datetime
from typing import Dict, Iterable, List

import pandas as pd
from loguru import logger


def _to_date(d: object) -> str:
    if isinstance(d, datetime):
        return d.date().isoformat()
    if isinstance(d, Date):
        return d.isoformat()
    if isinstance(d, str):
        return d
    raise TypeError(f"Unsupported date type: {type(d)}")


def emit_daily(asof: object, positions: pd.DataFrame, metrics: Dict, strategies: Optional[List[Dict]] = None, out_dir: str = os.path.join("ops", "reports")) -> Dict[str, str]:
    """Write daily positions and metrics to JSON + Markdown.

    positions columns: symbol, target_w, [optional] expected_alpha_bps, conf, tp_bps, sl_bps
    metrics: arbitrary dict (e.g., Sharpe, turnover)
    """
    os.makedirs(out_dir, exist_ok=True)
    asof_s = _to_date(asof)

    payload = {
        "asof": asof_s,
        "universe": positions["symbol"].tolist(),
        "positions": positions.to_dict(orient="records"),
        "metrics": metrics,
        "strategies": strategies or [],
    }

    json_path = os.path.join(out_dir, f"equities_{asof_s}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # Markdown summary
    md_lines = [
        f"# Daily Positions - {asof_s}",
        "",
        "## Metrics",
        *(f"- {k}: {v}" for k, v in metrics.items()),
        "",
        "## Positions (Top 20 by |w|)",
    ]
    pos_md = positions.copy()
    pos_md["absw"] = pos_md["target_w"].abs()
    pos_md = pos_md.sort_values("absw", ascending=False).drop(columns=["absw"]).head(20)
    md_lines.append(pos_md.to_markdown(index=False))

    md_path = os.path.join(out_dir, f"equities_{asof_s}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    logger.info(f"Daily report written: {json_path}, {md_path}")
    return {"json": json_path, "md": md_path}
