"""Signal and validation report emitters."""

from __future__ import annotations

import json
from pathlib import Path


def emit_report(*, report: dict, output_root: Path, report_date: str | None = None) -> tuple[Path, Path]:
    """Write JSON and markdown reports to the expected location."""
    date_key = report_date or "latest"
    daily_root = output_root / "reports" / "daily"
    daily_root.mkdir(parents=True, exist_ok=True)
    json_path = daily_root / f"{date_key}.json"
    md_path = daily_root / f"{date_key}.md"

    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    md_path.write_text(_render_markdown(report), encoding="utf-8")
    return json_path, md_path


def _render_markdown(report: dict) -> str:
    ridge = report.get("ridge", {})
    lgbm = report.get("lightgbm", {})
    diagnostics = report.get("diagnostics", {})
    return "\n".join(
        [
            "# TradeML Report",
            "",
            f"- GREEN coverage: {report.get('coverage', 0):.3f}",
            f"- Ridge mean rank IC: {ridge.get('mean_rank_ic', 0):.4f}",
            f"- LightGBM mean rank IC: {lgbm.get('mean_rank_ic', 0):.4f}",
            f"- Placebo max abs IC: {max((abs(x) for x in diagnostics.get('placebo', [0])), default=0):.4f}",
            f"- Cost-stress net return: {diagnostics.get('cost_stress', {}).get('net_return', 0):.4f}",
        ]
    )
