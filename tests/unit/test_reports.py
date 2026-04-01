from __future__ import annotations

from pathlib import Path

import json

from trademl.reports.emitter import emit_report


def test_report_emitter_writes_json_and_markdown(tmp_path: Path) -> None:
    report = {
        "coverage": 1.0,
        "ridge": {"mean_rank_ic": 0.03},
        "lightgbm": {"mean_rank_ic": 0.04},
        "diagnostics": {"placebo": [0.0], "cost_stress": {"net_return": 0.1}},
    }
    json_path, md_path = emit_report(report=report, output_root=tmp_path, report_date="2026-03-31")

    assert json.loads(json_path.read_text(encoding="utf-8"))["coverage"] == 1.0
    assert "TradeML Report" in md_path.read_text(encoding="utf-8")
