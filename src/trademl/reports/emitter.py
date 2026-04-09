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
    assessment = report.get("assessment", {})
    sign_flip = diagnostics.get("sign_flip_canary", {})
    ridge_folds = ridge.get("folds", [])
    lightgbm_folds = lgbm.get("folds", [])
    fold_lines = ["## Ridge Folds", ""] + [
        f"- Fold {idx + 1}: IC={fold.get('rank_ic', 0):.4f}, spread={fold.get('decile_spread', 0):.4f}, hit={fold.get('hit_rate', 0):.2%}"
        for idx, fold in enumerate(ridge_folds)
    ]
    if lightgbm_folds:
        fold_lines += ["", "## LightGBM Folds", ""] + [
            f"- Fold {idx + 1}: IC={fold.get('rank_ic', 0):.4f}, spread={fold.get('decile_spread', 0):.4f}, hit={fold.get('hit_rate', 0):.2%}"
            for idx, fold in enumerate(lightgbm_folds)
        ]
    return "\n".join(
        [
            "# TradeML Report",
            "",
            f"- GREEN coverage: {report.get('coverage', 0):.3f}",
            f"- Ridge mean rank IC: {ridge.get('mean_rank_ic', 0):.4f}",
            f"- Ridge alpha: {ridge.get('alpha', 'n/a')}",
            f"- LightGBM mean rank IC: {lgbm.get('mean_rank_ic', 0):.4f}",
            f"- Placebo max abs IC: {max((abs(x) for x in diagnostics.get('placebo', [0])), default=0):.4f}",
            f"- Cost-stress net return: {diagnostics.get('cost_stress', {}).get('net_return', 0):.4f}",
            f"- Sign-flip preferred direction: {sign_flip.get('preferred_direction', 'unknown')}",
            f"- Sign-flip mean rank IC: {sign_flip.get('flipped_mean_rank_ic', 0):.4f}",
            f"- CPCV mean OOS score: {diagnostics.get('cpcv', {}).get('mean_oos_score', 0):.4f}",
            f"- PBO: {diagnostics.get('pbo', 0):.4f}",
            f"- DSR: {diagnostics.get('dsr', 0):.4f}",
            f"- Phase 1 assessment: {assessment.get('decision', 'UNKNOWN')}",
            f"- Assessment rationale: {assessment.get('reason', 'not computed')}",
            "",
            "## Decile Spread Chart Data",
            "",
            json.dumps(ridge.get("decile_chart_data", {}), indent=2),
            "",
            "## Diagnostics",
            "",
            json.dumps(diagnostics, indent=2, default=str),
            "",
            *fold_lines,
        ]
    )
