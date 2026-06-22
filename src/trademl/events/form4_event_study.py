"""Form 4 insider-buy event-study and negative-control result packets."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import ast
import json
import math
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd

from trademl.events.form4_labels import (
    Form4LabelConfig,
    build_form4_event_labels_with_market_sources,
)


FORM4_EVENT_STUDY_VERSION = "form4_event_study_v1"
PRIMARY_EVENT_TYPE = "FORM4_OPEN_MARKET_INSIDER_BUY"
MECHANICAL_CONTROL_CODES = ("A", "M", "F")


@dataclass(slots=True, frozen=True)
class Form4EventStudyConfig:
    """Configuration for Form 4 event-study evaluation."""

    primary_horizon: int = 5
    horizons: tuple[int, ...] = (1, 5, 10, 20)
    round_trip_cost_bps: float = 50.0
    min_historical_sample: int = 300
    bootstrap_iterations: int = 1000
    bootstrap_seed: int = 17


def run_form4_event_study(
    *,
    data_root: Path,
    primary_horizon: int = 5,
    horizons: Iterable[int] | None = None,
    round_trip_cost_bps: float = 50.0,
    min_historical_sample: int = 300,
    market_data_roots: Iterable[Path] | None = None,
    source_contract_path: Path | None = None,
) -> dict[str, object]:
    """Write a Form 4 event-study packet from labels and available controls."""
    root = Path(data_root).expanduser()
    config = Form4EventStudyConfig(
        primary_horizon=int(primary_horizon),
        horizons=tuple(sorted(set(int(item) for item in (horizons or (1, 5, 10, 20))))),
        round_trip_cost_bps=float(round_trip_cost_bps),
        min_historical_sample=int(min_historical_sample),
    )
    labels_path = (
        root
        / "data"
        / "curated"
        / "events"
        / "form4_open_market_buy_labels"
        / "data.parquet"
    )
    if not labels_path.exists():
        raise FileNotFoundError(f"missing Form 4 labels parquet: {labels_path}")
    primary_labels = pd.read_parquet(labels_path)
    control_payload = build_and_write_form4_control_labels(
        data_root=root,
        label_config=Form4LabelConfig(
            horizons=config.horizons,
            round_trip_cost_bps=config.round_trip_cost_bps,
        ),
        market_data_roots=market_data_roots,
        source_contract_path=source_contract_path,
    )
    controls = {
        str(item["family"]): pd.read_parquet(str(item["labels_path"]))
        for item in control_payload.get("families", [])
        if item.get("labels_path") and Path(str(item["labels_path"])).exists()
    }
    packet = build_form4_event_study_packet(
        primary_labels=primary_labels,
        controls=controls,
        config=config,
        control_payload=control_payload,
    )
    return write_form4_event_study_packet(root=root, packet=packet)


def build_and_write_form4_control_labels(
    *,
    data_root: Path,
    label_config: Form4LabelConfig,
    market_data_roots: Iterable[Path] | None = None,
    source_contract_path: Path | None = None,
) -> dict[str, object]:
    """Build and persist Form 4 negative-control labels when parsed rows exist."""
    root = Path(data_root)
    control_candidates = build_form4_control_candidates_from_curated(root=root)
    output_root = root / "data" / "curated" / "events" / "form4_control_labels"
    families: list[dict[str, object]] = []
    for family, candidates in sorted(control_candidates.items()):
        family_root = output_root / family
        family_root.mkdir(parents=True, exist_ok=True)
        candidates_path = family_root / "candidates.parquet"
        candidates.to_parquet(candidates_path, index=False)
        if candidates.empty:
            labels = pd.DataFrame()
            source_metadata: dict[str, object] = {}
        else:
            labels, source_metadata = build_form4_event_labels_with_market_sources(
                candidates=candidates,
                data_root=root,
                config=label_config,
                market_data_roots=market_data_roots,
                source_contract_path=source_contract_path,
            )
        labels_path = family_root / "labels.parquet"
        labels.to_parquet(labels_path, index=False)
        families.append(
            {
                "family": family,
                "candidate_count": int(len(candidates)),
                "labeled_count": int((labels.get("label_status") == "LABELED").sum())
                if not labels.empty and "label_status" in labels
                else 0,
                "blocked_count": int((labels.get("label_status") == "BLOCKED").sum())
                if not labels.empty and "label_status" in labels
                else 0,
                "candidates_path": str(candidates_path),
                "labels_path": str(labels_path),
                "source_metadata": _compact_source_metadata(source_metadata),
            }
        )
    payload: dict[str, object] = {
        "version": FORM4_EVENT_STUDY_VERSION,
        "checked_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "families": families,
    }
    target = root / "control" / "cluster" / "state" / "research" / "form4_control_labels"
    target.mkdir(parents=True, exist_ok=True)
    (target / "latest.json").write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return payload


def build_form4_control_candidates_from_curated(*, root: Path) -> dict[str, pd.DataFrame]:
    """Build label-compatible Form 4 control candidates from parsed transaction rows."""
    base = Path(root) / "data" / "curated" / "sec" / "form4"
    submissions_path = base / "submissions" / "data.parquet"
    transactions_path = base / "nonderiv_transactions" / "data.parquet"
    if not submissions_path.exists() or not transactions_path.exists():
        return {
            "mechanical_acquisition_codes": pd.DataFrame(),
            "sales_placebo": pd.DataFrame(),
        }
    submissions = pd.read_parquet(submissions_path)
    transactions = pd.read_parquet(transactions_path)
    by_accession = {
        str(item["accession"]): item for item in submissions.to_dict("records")
    }
    families = {
        "mechanical_acquisition_codes": transactions[
            transactions.get("transaction_code", pd.Series(dtype=object))
            .astype(str)
            .str.upper()
            .isin(MECHANICAL_CONTROL_CODES)
        ],
        "sales_placebo": transactions[
            transactions.get("transaction_code", pd.Series(dtype=object))
            .astype(str)
            .str.upper()
            .eq("S")
        ],
    }
    output: dict[str, pd.DataFrame] = {}
    for family, rows in families.items():
        records: list[dict[str, object]] = []
        if not rows.empty:
            for accession, group in rows.groupby("accession"):
                submission = by_accession.get(str(accession), {})
                records.append(
                    _control_candidate_from_group(
                        family=family,
                        accession=str(accession),
                        submission=submission,
                        rows=group,
                    )
                )
        output[family] = pd.DataFrame(records)
    return output


def build_form4_event_study_packet(
    *,
    primary_labels: pd.DataFrame,
    controls: dict[str, pd.DataFrame],
    config: Form4EventStudyConfig,
    control_payload: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build a machine-readable Form 4 event-study result packet."""
    primary_metric = f"abret_{config.primary_horizon}d_net"
    primary_labeled = _labeled(primary_labels, metric=primary_metric)
    primary_summary = _summary_by_metric(
        primary_labeled,
        metric=primary_metric,
        bootstrap_iterations=config.bootstrap_iterations,
        bootstrap_seed=config.bootstrap_seed,
    )
    by_horizon = {
        str(horizon): {
            "net": _summary_by_metric(
                _labeled(primary_labels, metric=f"ret_{horizon}d_net"),
                metric=f"ret_{horizon}d_net",
                bootstrap_iterations=config.bootstrap_iterations,
                bootstrap_seed=config.bootstrap_seed,
            ),
            "abnormal_net": _summary_by_metric(
                _labeled(primary_labels, metric=f"abret_{horizon}d_net"),
                metric=f"abret_{horizon}d_net",
                bootstrap_iterations=config.bootstrap_iterations,
                bootstrap_seed=config.bootstrap_seed,
            ),
        }
        for horizon in config.horizons
    }
    strength = _strength_bucket_summary(primary_labeled, metric=primary_metric)
    control_summaries: dict[str, object] = {}
    separation: dict[str, object] = {}
    for family, labels in sorted(controls.items()):
        control_labeled = _labeled(labels, metric=primary_metric)
        summary = _summary_by_metric(
            control_labeled,
            metric=primary_metric,
            bootstrap_iterations=config.bootstrap_iterations,
            bootstrap_seed=config.bootstrap_seed,
        )
        control_summaries[family] = summary
        separation[family] = _separation(primary_summary, summary)
    verdict = _event_study_verdict(
        primary_summary=primary_summary,
        strength=strength,
        separation=separation,
        min_sample=config.min_historical_sample,
    )
    return _json_safe(
        {
            "version": FORM4_EVENT_STUDY_VERSION,
            "checked_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "event_class": PRIMARY_EVENT_TYPE,
            "primary_horizon": config.primary_horizon,
            "round_trip_cost_bps": config.round_trip_cost_bps,
            "min_historical_sample": config.min_historical_sample,
            "primary": primary_summary,
            "by_horizon": by_horizon,
            "strength_buckets": strength,
            "negative_controls": control_summaries,
            "negative_control_separation": separation,
            "control_artifacts": control_payload or {},
            "verdict": verdict,
        }
    )


def write_form4_event_study_packet(
    *, root: Path, packet: dict[str, object]
) -> dict[str, object]:
    """Write JSON and Markdown Form 4 event-study artifacts."""
    target = Path(root) / "control" / "cluster" / "state" / "research" / "form4_event_study"
    history = target / "history"
    history.mkdir(parents=True, exist_ok=True)
    latest = target / "latest.json"
    latest.write_text(json.dumps(packet, indent=2, sort_keys=True), encoding="utf-8")
    timestamp = str(packet["checked_at"]).replace(":", "").replace("+", "_")
    history_path = history / f"{timestamp}.json"
    history_path.write_text(json.dumps(packet, indent=2, sort_keys=True), encoding="utf-8")
    report_root = Path(root) / "reports" / "research" / "form4_event_study"
    report_root.mkdir(parents=True, exist_ok=True)
    report_path = report_root / "latest.md"
    report_path.write_text(_render_markdown(packet), encoding="utf-8")
    return {
        "packet_path": str(latest),
        "history_path": str(history_path),
        "report_path": str(report_path),
        "packet": packet,
    }


def _control_candidate_from_group(
    *,
    family: str,
    accession: str,
    submission: dict[str, object],
    rows: pd.DataFrame,
) -> dict[str, object]:
    ticker = _normalize_ticker(
        str(
            submission.get("issuer_trading_symbol_raw")
            or _first_text(rows, "issuer_trading_symbol_raw")
            or ""
        )
    )
    issuer_cik = str(submission.get("issuer_cik") or _first_text(rows, "issuer_cik") or "")
    accepted_at = str(submission.get("accepted_at_utc") or _first_text(rows, "accepted_at_utc") or "")
    flags = {
        str(flag)
        for value in rows.get("source_quality_flags", pd.Series(dtype=object)).tolist()
        for flag in _list(value)
    }
    is_common = not any(flag == "non_common_security_title" for flag in flags)
    document_type = str(submission.get("document_type") or _first_text(rows, "document_type") or "")
    eligibility_pass = bool(
        document_type == "4"
        and accepted_at
        and ticker
        and not _is_invalid_tradable_ticker(ticker)
        and not _is_otc_symbol(ticker)
        and is_common
        and "private_or_unit_purchase_flag" not in flags
        and "late_report" not in flags
    )
    exclusion_reasons: list[str] = []
    if document_type != "4":
        exclusion_reasons.append("amendment")
    if not accepted_at:
        exclusion_reasons.append("missing_accepted_at")
    if not ticker:
        exclusion_reasons.append("missing_ticker")
    elif _is_invalid_tradable_ticker(ticker):
        exclusion_reasons.append("ambiguous_or_invalid_ticker")
    if _is_otc_symbol(ticker):
        exclusion_reasons.append("otc_symbol")
    if not is_common:
        exclusion_reasons.append("non_common_security_title")
    if "private_or_unit_purchase_flag" in flags:
        exclusion_reasons.append("private_or_unit_purchase")
    if "late_report" in flags:
        exclusion_reasons.append("late_report")
    value = _sum_float(rows.get("transaction_value", pd.Series(dtype=object)))
    owner_ciks = sorted(
        {
            str(item)
            for value in rows.get("owner_cik_set", pd.Series(dtype=object)).tolist()
            for item in _list(value)
            if str(item)
        }
    )
    digest = hashlib.sha256(f"{family}:{issuer_cik}:{accession}".encode("utf-8")).hexdigest()[:16]
    return {
        "event_id": f"form4_{family}_{digest}",
        "issuer_cik": issuer_cik,
        "ticker": ticker,
        "primary_security_id": f"{issuer_cik}:{ticker}" if issuer_cik and ticker else "",
        "accessions": [accession],
        "event_type": f"FORM4_CONTROL_{family.upper()}",
        "accepted_at_utc": accepted_at or None,
        "first_seen_at_utc": accepted_at or None,
        "tradable_at_utc": None,
        "n_insiders_buying": len(owner_ciks),
        "n_directors_buying": 0,
        "n_officers_buying": 0,
        "ceo_buy": False,
        "cfo_buy": False,
        "ten_percent_owner_buy": False,
        "total_shares_bought": str(_sum_float(rows.get("transaction_shares", pd.Series(dtype=object)))),
        "total_dollar_value": str(value),
        "total_dollar_value_float": float(value),
        "max_single_purchase_value": str(_max_float(rows.get("transaction_value", pd.Series(dtype=object)))),
        "median_purchase_price": None,
        "purchase_value_to_market_cap": None,
        "purchase_value_to_adv20": None,
        "purchase_value_to_prior_holdings": None,
        "days_since_transaction": None,
        "cluster_7d_purchase_value": str(value),
        "cluster_30d_purchase_value": str(value),
        "prior_90d_net_insider_purchase_value": "0",
        "prior_20d_return": None,
        "prior_60d_return": None,
        "prior_20d_vol": None,
        "prior_20d_adv": None,
        "market_cap_bucket": None,
        "liquidity_bucket": None,
        "news_nearby_flag": False,
        "earnings_nearby_flag": False,
        "offering_nearby_flag": False,
        "mna_nearby_flag": False,
        "eligibility_pass": eligibility_pass,
        "exclusion_reasons": exclusion_reasons,
        "event_strength_score": math.log1p(max(value, 0.0)),
        "parse_confidence": 1.0 if accepted_at and ticker else 0.5,
        "source_quality_score": 1.0,
        "source_quality_flags": sorted(flags),
        "owner_ciks": owner_ciks,
        "owner_names": [],
        "source_transaction_count": int(len(rows)),
        "eligible_transaction_count": int(len(rows)) if eligibility_pass else 0,
        "schema_version": FORM4_EVENT_STUDY_VERSION,
    }


def _summary_by_metric(
    frame: pd.DataFrame,
    *,
    metric: str,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    values = pd.to_numeric(frame.get(metric, pd.Series(dtype=float)), errors="coerce").dropna()
    if values.empty:
        return {
            "metric": metric,
            "n": 0,
            "mean": None,
            "median": None,
            "hit_rate": None,
            "bootstrap_mean_ci_95": [None, None],
            "top5_abs_contribution": None,
        }
    array = values.to_numpy(dtype=float)
    abs_total = float(np.abs(array).sum())
    top5_abs = float(np.sort(np.abs(array))[-5:].sum()) if len(array) else 0.0
    return {
        "metric": metric,
        "n": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "hit_rate": float(np.mean(array > 0)),
        "bootstrap_mean_ci_95": _bootstrap_mean_ci(
            array,
            iterations=bootstrap_iterations,
            seed=bootstrap_seed,
        ),
        "top5_abs_contribution": (top5_abs / abs_total) if abs_total else None,
    }


def _strength_bucket_summary(frame: pd.DataFrame, *, metric: str) -> dict[str, object]:
    if frame.empty or metric not in frame:
        return {"status": "insufficient_sample", "buckets": {}, "q4_minus_q1": None}
    working = frame.copy()
    working[metric] = pd.to_numeric(working[metric], errors="coerce")
    working["event_strength_score"] = pd.to_numeric(
        working.get("event_strength_score"), errors="coerce"
    )
    working = working.dropna(subset=[metric, "event_strength_score"])
    if len(working) < 4 or working["event_strength_score"].nunique() < 4:
        return {"status": "insufficient_sample", "buckets": {}, "q4_minus_q1": None}
    working["strength_bucket"] = pd.qcut(
        working["event_strength_score"],
        q=4,
        labels=["Q1", "Q2", "Q3", "Q4"],
        duplicates="drop",
    )
    buckets = {
        str(bucket): _summary_by_metric(
            group,
            metric=metric,
            bootstrap_iterations=200,
            bootstrap_seed=11,
        )
        for bucket, group in working.groupby("strength_bucket", observed=True)
    }
    q1 = buckets.get("Q1", {}).get("mean")
    q4 = buckets.get("Q4", {}).get("mean")
    return {
        "status": "available" if q1 is not None and q4 is not None else "insufficient_sample",
        "buckets": buckets,
        "q4_minus_q1": float(q4) - float(q1) if q1 is not None and q4 is not None else None,
    }


def _event_study_verdict(
    *,
    primary_summary: dict[str, object],
    strength: dict[str, object],
    separation: dict[str, object],
    min_sample: int,
) -> dict[str, object]:
    n = int(primary_summary.get("n") or 0)
    failures: list[str] = []
    if n < min_sample:
        failures.append(f"sample_size<{min_sample}")
    mean = primary_summary.get("mean")
    median = primary_summary.get("median")
    if mean is not None and float(mean) <= 0.005:
        failures.append("mean_abret_5d_net<=50bps")
    if median is not None and float(median) <= 0.0:
        failures.append("median_abret_5d_net<=0")
    q4_minus_q1 = strength.get("q4_minus_q1")
    if q4_minus_q1 is not None and float(q4_minus_q1) <= 0.0075:
        failures.append("q4_minus_q1<=75bps")
    for family, item in separation.items():
        diff = item.get("mean_difference") if isinstance(item, dict) else None
        if diff is not None and float(diff) <= 0.0075:
            failures.append(f"{family}_separation<=75bps")
    if n == 0:
        decision = "BLOCKED_DATA_COVERAGE"
    elif n < min_sample:
        decision = "COLLECT_MORE_HISTORY"
    elif failures:
        decision = "KILL_OR_REWORK"
    else:
        decision = "CONTINUE"
    return {
        "decision": decision,
        "status": "PASS" if decision == "CONTINUE" else "NOT_PROMOTABLE",
        "failures": failures,
    }


def _separation(primary: dict[str, object], control: dict[str, object]) -> dict[str, object]:
    primary_mean = primary.get("mean")
    control_mean = control.get("mean")
    return {
        "primary_n": primary.get("n"),
        "control_n": control.get("n"),
        "primary_mean": primary_mean,
        "control_mean": control_mean,
        "mean_difference": (
            float(primary_mean) - float(control_mean)
            if primary_mean is not None and control_mean is not None
            else None
        ),
    }


def _labeled(frame: pd.DataFrame, *, metric: str) -> pd.DataFrame:
    if frame.empty or metric not in frame:
        return frame.iloc[0:0].copy()
    status = frame.get("label_status", pd.Series(["LABELED"] * len(frame)))
    return frame[status.astype(str).eq("LABELED") & pd.notna(frame[metric])].copy()


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> list[float | None]:
    if values.size == 0:
        return [None, None]
    if values.size == 1:
        value = float(values[0])
        return [value, value]
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(int(iterations), values.size), replace=True).mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def _render_markdown(packet: dict[str, object]) -> str:
    verdict = dict(packet.get("verdict") or {})
    primary = dict(packet.get("primary") or {})
    lines = [
        "# Form 4 Event Study",
        "",
        f"- Decision: `{verdict.get('decision')}`",
        f"- Status: `{verdict.get('status')}`",
        f"- Primary horizon: `{packet.get('primary_horizon')}d`",
        f"- Labeled events: `{primary.get('n')}`",
        f"- Mean abnormal net return: `{primary.get('mean')}`",
        f"- Median abnormal net return: `{primary.get('median')}`",
        f"- Hit rate: `{primary.get('hit_rate')}`",
        "",
        "## Failures",
    ]
    failures = list(verdict.get("failures") or [])
    lines.extend(f"- `{failure}`" for failure in failures)
    if not failures:
        lines.append("- None")
    return "\n".join(lines) + "\n"


def _first_text(frame: pd.DataFrame, column: str) -> str | None:
    if column not in frame:
        return None
    for value in frame[column].tolist():
        if value is not None and not pd.isna(value) and str(value):
            return str(value)
    return None


def _sum_float(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.sum()) if not numeric.empty else 0.0


def _max_float(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.max()) if not numeric.empty else 0.0


def _is_otc_symbol(ticker: str) -> bool:
    upper = ticker.upper()
    return upper.endswith(".OB") or upper.endswith(".PK") or upper.endswith("Q")


def _is_invalid_tradable_ticker(ticker: str) -> bool:
    normalized = _normalize_ticker(ticker)
    return not bool(re.fullmatch(r"[A-Z][A-Z0-9.-]{0,9}", normalized))


def _normalize_ticker(ticker: str) -> str:
    return re.sub(r"\s+", "", str(ticker or "").strip().upper())


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


def _compact_source_metadata(metadata: dict[str, object]) -> dict[str, object]:
    return _compact_large_lists(metadata)


def _compact_large_lists(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _compact_large_lists(item) for key, item in value.items()}
    if isinstance(value, list):
        compacted = [_compact_large_lists(item) for item in value]
        if len(compacted) > 25:
            return {
                "count": len(compacted),
                "sample": compacted[:10],
                "last": compacted[-1],
            }
        return compacted
    return value


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value
