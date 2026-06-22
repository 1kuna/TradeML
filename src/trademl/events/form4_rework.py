"""Bounded diagnostic rework study for the first Form 4 event MVP."""

from __future__ import annotations

from datetime import datetime, timezone
import ast
import json
from pathlib import Path

import pandas as pd

from trademl.events.form4_event_study import (
    FORM4_EVENT_STUDY_VERSION,
    _json_safe,
    _labeled,
    _separation,
    _summary_by_metric,
)


FORM4_REWORK_VERSION = "form4_rework_study_v1"
PRIMARY_HORIZON = 5
PRIMARY_METRIC = f"abret_{PRIMARY_HORIZON}d_net"
REWORK_VARIANTS = (
    "clean_timely_unmixed",
    "officer_director_only",
    "ceo_cfo_only",
    "clustered_buyers",
    "large_purchase_q4",
)


def run_form4_rework_study(*, data_root: Path) -> dict[str, object]:
    """Run the bounded Form 4 rework gate from existing curated artifacts."""
    root = Path(data_root).expanduser()
    primary_labels = _read_required_parquet(
        root
        / "data"
        / "curated"
        / "events"
        / "form4_open_market_buy_labels"
        / "data.parquet"
    )
    controls = _read_control_labels(root)
    baseline = write_form4_baseline_status(root=root)
    packet = build_form4_rework_packet(
        primary_labels=primary_labels,
        controls=controls,
        baseline=baseline["packet"],
    )
    return write_form4_rework_packet(root=root, packet=packet)


def build_form4_rework_packet(
    *,
    primary_labels: pd.DataFrame,
    controls: dict[str, pd.DataFrame],
    baseline: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build the fixed-variant Form 4 rework packet."""
    variant_packets = {
        name: _variant_packet(
            name=name,
            labels=_select_variant(primary_labels, name),
            all_labels=primary_labels,
            controls=controls,
        )
        for name in REWORK_VARIANTS
    }
    passing = [
        name
        for name, packet in variant_packets.items()
        if dict(packet.get("verdict") or {}).get("decision") == "CONTINUE_DIAGNOSTIC"
    ]
    decision = "CONTINUE_DIAGNOSTIC" if passing else "FORM4_KILLED_BASELINE_COMPLETE"
    return _json_safe(
        {
            "version": FORM4_REWORK_VERSION,
            "checked_at": _now_iso(),
            "event_class": "FORM4_OPEN_MARKET_INSIDER_BUY",
            "primary_horizon": PRIMARY_HORIZON,
            "primary_metric": PRIMARY_METRIC,
            "round_trip_cost_bps": 50.0,
            "variant_names": list(REWORK_VARIANTS),
            "baseline": baseline or {},
            "variants": variant_packets,
            "verdict": {
                "decision": decision,
                "status": (
                    "DIAGNOSTIC_CONTINUE"
                    if decision == "CONTINUE_DIAGNOSTIC"
                    else "NOT_PROMOTABLE"
                ),
                "passing_variants": passing,
                "paper_live_allowed": False,
                "paper_live_blocker": (
                    "form4_rework_failed"
                    if decision == "FORM4_KILLED_BASELINE_COMPLETE"
                    else "diagnostic_only_requires_new_hypothesis"
                ),
            },
        }
    )


def write_form4_baseline_status(*, root: Path) -> dict[str, object]:
    """Write a durable status snapshot for the naive Form 4 baseline."""
    root = Path(root)
    event_study_path = (
        root
        / "control"
        / "cluster"
        / "state"
        / "research"
        / "form4_event_study"
        / "latest.json"
    )
    if event_study_path.exists():
        baseline_packet = json.loads(event_study_path.read_text(encoding="utf-8"))
    else:
        baseline_packet = {
            "version": FORM4_EVENT_STUDY_VERSION,
            "verdict": {
                "decision": "BASELINE_MISSING",
                "status": "BLOCKED",
                "failures": ["missing_form4_event_study_packet"],
            },
        }
    verdict = dict(baseline_packet.get("verdict") or {})
    packet = _json_safe(
        {
            "version": "form4_baseline_status_v1",
            "checked_at": _now_iso(),
            "event_class": "FORM4_OPEN_MARKET_INSIDER_BUY",
            "baseline_packet_path": str(event_study_path),
            "baseline_decision": verdict.get("decision"),
            "baseline_status": verdict.get("status"),
            "baseline_failures": list(verdict.get("failures") or []),
            "paper_live_allowed": False,
            "paper_live_blocker": "naive_form4_baseline_not_promotable",
            "packet": baseline_packet,
        }
    )
    target = (
        root
        / "control"
        / "cluster"
        / "state"
        / "research"
        / "form4_baseline_status"
    )
    history = target / "history"
    history.mkdir(parents=True, exist_ok=True)
    latest = target / "latest.json"
    latest.write_text(json.dumps(packet, indent=2, sort_keys=True), encoding="utf-8")
    history_path = history / f"{packet['checked_at'].replace(':', '').replace('+', '_')}.json"
    history_path.write_text(json.dumps(packet, indent=2, sort_keys=True), encoding="utf-8")
    return {"packet_path": str(latest), "history_path": str(history_path), "packet": packet}


def write_form4_rework_packet(
    *, root: Path, packet: dict[str, object]
) -> dict[str, object]:
    """Write the Form 4 rework JSON and Markdown artifacts."""
    root = Path(root)
    target = root / "control" / "cluster" / "state" / "research" / "form4_rework_study"
    history = target / "history"
    history.mkdir(parents=True, exist_ok=True)
    latest = target / "latest.json"
    latest.write_text(json.dumps(packet, indent=2, sort_keys=True), encoding="utf-8")
    history_path = history / f"{packet['checked_at'].replace(':', '').replace('+', '_')}.json"
    history_path.write_text(json.dumps(packet, indent=2, sort_keys=True), encoding="utf-8")
    report_root = root / "reports" / "research" / "form4_rework_study"
    report_root.mkdir(parents=True, exist_ok=True)
    report_path = report_root / "latest.md"
    report_path.write_text(_render_markdown(packet), encoding="utf-8")
    return {
        "packet_path": str(latest),
        "history_path": str(history_path),
        "report_path": str(report_path),
        "packet": packet,
    }


def _variant_packet(
    *,
    name: str,
    labels: pd.DataFrame,
    all_labels: pd.DataFrame,
    controls: dict[str, pd.DataFrame],
) -> dict[str, object]:
    primary = _summary_by_metric(
        _labeled(labels, metric=PRIMARY_METRIC),
        metric=PRIMARY_METRIC,
        bootstrap_iterations=1000,
        bootstrap_seed=29,
    )
    control_summaries = {
        family: _summary_by_metric(
            _labeled(frame, metric=PRIMARY_METRIC),
            metric=PRIMARY_METRIC,
            bootstrap_iterations=1000,
            bootstrap_seed=31,
        )
        for family, frame in sorted(controls.items())
    }
    separation = {
        family: _separation(primary, summary)
        for family, summary in sorted(control_summaries.items())
    }
    verdict = _variant_verdict(primary=primary, separation=separation)
    return _json_safe(
        {
            "variant": name,
            "candidate_count": int(len(labels)),
            "source_candidate_count": int(len(all_labels)),
            "labeled_count": int(primary.get("n") or 0),
            "primary": primary,
            "negative_controls": control_summaries,
            "negative_control_separation": separation,
            "verdict": verdict,
        }
    )


def _variant_verdict(
    *, primary: dict[str, object], separation: dict[str, object]
) -> dict[str, object]:
    failures: list[str] = []
    n = int(primary.get("n") or 0)
    if n < 100:
        failures.append("sample_size<100")
    mean = primary.get("mean")
    if mean is None or float(mean) <= 0.005:
        failures.append("mean_abret_5d_net<=50bps")
    median = primary.get("median")
    if median is None or float(median) <= 0.0:
        failures.append("median_abret_5d_net<=0")
    for family, item in sorted(separation.items()):
        diff = item.get("mean_difference") if isinstance(item, dict) else None
        if diff is None or float(diff) <= 0.0075:
            failures.append(f"{family}_separation<=75bps")
    decision = "CONTINUE_DIAGNOSTIC" if not failures else "REJECTED"
    return {
        "decision": decision,
        "status": "PASS" if decision == "CONTINUE_DIAGNOSTIC" else "NOT_PROMOTABLE",
        "failures": failures,
    }


def _select_variant(labels: pd.DataFrame, name: str) -> pd.DataFrame:
    clean = _clean_timely_unmixed(labels)
    if name == "clean_timely_unmixed":
        return clean
    if name == "officer_director_only":
        return clean[
            _bool_series(clean, "n_officers_buying")
            | _bool_series(clean, "n_directors_buying")
        ].copy()
    if name == "ceo_cfo_only":
        return clean[
            clean.get("ceo_buy", pd.Series(False, index=clean.index)).fillna(False).astype(bool)
            | clean.get("cfo_buy", pd.Series(False, index=clean.index)).fillna(False).astype(bool)
        ].copy()
    if name == "clustered_buyers":
        n_insiders = pd.to_numeric(clean.get("n_insiders_buying"), errors="coerce").fillna(0)
        n_transactions = pd.to_numeric(
            clean.get("eligible_transaction_count"), errors="coerce"
        ).fillna(0)
        return clean[(n_insiders > 1) | (n_transactions > 1)].copy()
    if name == "large_purchase_q4":
        values = pd.to_numeric(clean.get("total_dollar_value_float"), errors="coerce")
        if clean.empty or values.dropna().empty:
            return clean.iloc[0:0].copy()
        threshold = float(values.quantile(0.75))
        return clean[values >= threshold].copy()
    raise ValueError(f"unsupported Form 4 rework variant: {name}")


def _clean_timely_unmixed(labels: pd.DataFrame) -> pd.DataFrame:
    if labels.empty:
        return labels.copy()
    frame = labels.copy()
    mask = frame.get("eligibility_pass", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
    days = pd.to_numeric(frame.get("days_since_transaction"), errors="coerce")
    mask &= days.notna() & (days <= 2)
    disallowed = {
        "mixed_p_and_s",
        "late_report",
        "private_or_unit_purchase",
        "private_or_unit_purchase_flag",
        "ambiguous_or_invalid_ticker",
        "non_common_security_title",
        "missing_price",
        "zero_price",
        "otc_symbol",
    }
    for idx, row in frame.iterrows():
        flags = set(str(item) for item in _list(row.get("source_quality_flags")))
        reasons = set(str(item) for item in _list(row.get("exclusion_reasons")))
        if flags & disallowed or reasons & disallowed:
            mask.loc[idx] = False
    return frame[mask].copy()


def _read_control_labels(root: Path) -> dict[str, pd.DataFrame]:
    base = root / "data" / "curated" / "events" / "form4_control_labels"
    controls: dict[str, pd.DataFrame] = {}
    for family in ("mechanical_acquisition_codes", "sales_placebo"):
        path = base / family / "labels.parquet"
        controls[family] = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    return controls


def _read_required_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing required parquet artifact: {path}")
    return pd.read_parquet(path)


def _bool_series(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(
        frame.get(column, pd.Series(0, index=frame.index)), errors="coerce"
    ).fillna(0)
    return values > 0


def _list(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        converted = value.tolist()
        return converted if isinstance(converted, list) else [converted]
    if isinstance(value, str) and value.startswith("[") and value.endswith("]"):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return []
        return parsed if isinstance(parsed, list) else [parsed]
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass
    return [value]


def _render_markdown(packet: dict[str, object]) -> str:
    verdict = dict(packet.get("verdict") or {})
    lines = [
        "# Form 4 Rework Study",
        "",
        f"- Decision: `{verdict.get('decision')}`",
        f"- Status: `{verdict.get('status')}`",
        f"- Paper/live allowed: `{verdict.get('paper_live_allowed')}`",
        "",
        "## Variants",
    ]
    variants = dict(packet.get("variants") or {})
    for name in REWORK_VARIANTS:
        variant = dict(variants.get(name) or {})
        primary = dict(variant.get("primary") or {})
        variant_verdict = dict(variant.get("verdict") or {})
        lines.extend(
            [
                "",
                f"### {name}",
                f"- Decision: `{variant_verdict.get('decision')}`",
                f"- Candidates: `{variant.get('candidate_count')}`",
                f"- Labeled: `{primary.get('n')}`",
                f"- Mean abnormal net return: `{primary.get('mean')}`",
                f"- Median abnormal net return: `{primary.get('median')}`",
                f"- Hit rate: `{primary.get('hit_rate')}`",
                "- Failures: "
                + (
                    ", ".join(f"`{item}`" for item in variant_verdict.get("failures") or [])
                    if variant_verdict.get("failures")
                    else "None"
                ),
            ]
        )
    return "\n".join(lines) + "\n"


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
