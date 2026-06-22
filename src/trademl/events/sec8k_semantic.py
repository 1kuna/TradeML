"""SEC 8-K semantic event classification and study pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
import ast
import hashlib
import html
import json
from pathlib import Path
import re
from typing import Iterable

import pandas as pd
import pyarrow.parquet as pq

from trademl.events.form4 import sgml_tag
from trademl.events.form4_event_study import _json_safe, _labeled, _separation, _summary_by_metric
from trademl.events.form4_labels import (
    DEFAULT_HORIZONS,
    Form4LabelConfig,
    build_form4_event_labels_with_market_sources,
)
from trademl.events.sec8k import _timestamp_placebo_candidates
from trademl.events.semantic_classifier import (
    DEFAULT_LMSTUDIO_BASE_URL,
    DEFAULT_SEC_EVENT_MODEL,
    FIELD_KEYS,
    LMStudioSecEventSemanticClient,
    SEC_EVENT_SEMANTIC_SCHEMA_VERSION,
    SecEventSemanticBatchClient,
    blocking_semantic_validation_errors,
    normalize_semantic_payload,
    semantic_batch_exception_is_isolatable,
    validate_semantic_classification,
)


SEC8K_SEMANTIC_CLASSIFICATION_VERSION = "sec8k_semantic_classification_v1"
SEC8K_SEMANTIC_STUDY_VERSION = "sec8k_semantic_study_v1"
SEC8K_SEMANTIC_LABELABILITY_VERSION = "sec8k_semantic_labelability_v1"
SEC8K_SEMANTIC_SCALED_GATE_VERSION = "sec8k_semantic_scaled_gate_v1"
PROMOTABLE_EVENT_TYPES = (
    "DILUTIVE_FINANCING",
    "AUDITOR_TROUBLE",
    "DEBT_DEFAULT_COVENANT_STRESS",
    "CUSTOMER_LOSS",
    "MATERIAL_CONTRACT_AWARD",
)
PRIMARY_HORIZON = 5
SEMANTIC_ROUTING_MODES = ("broad", "targeted")
SEMANTIC_LABELABILITY_MODES = ("all", "prefer-labelable", "labelable-only")
SEMANTIC_SNIPPET_KINDS = ("all", "item_section", "exhibit")
TARGETED_SEC_ITEM_FAMILIES = {
    "4.01": "auditor_change",
    "2.04": "default_or_covenant_stress",
    "3.02": "unregistered_sale_or_financing",
    "1.01": "material_agreement",
}
DEFAULT_TARGETED_SEC_ITEMS = tuple(TARGETED_SEC_ITEM_FAMILIES)
DEFAULT_SEMANTIC_LABELABILITY_HORIZONS = (5,)
DEFAULT_SCALED_GATE_TARGET_ITEMS = ("3.02", "4.01", "2.04")
DEFAULT_SCALED_GATE_FALLBACK_ITEMS = ("1.01",)
DEFAULT_SCALED_GATE_YEARS = (2025, 2024)
DEFAULT_MAC_MINI_LMSTUDIO_BASE_URL = "http://127.0.0.1:1235/v1"
SEC8K_ITEM_EVENT_LIGHT_COLUMNS = (
    "event_id",
    "issuer_cik",
    "ticker",
    "primary_security_id",
    "accession",
    "archive_cik",
    "complete_txt_path",
    "event_type",
    "sec_item_number",
    "accepted_at_utc",
    "first_seen_at_utc",
    "tradable_at_utc",
    "filing_date",
    "primary_document",
    "eligibility_pass",
    "event_strength_score",
    "source_hash",
    "source_path",
)


@dataclass(slots=True, frozen=True)
class Sec8KSemanticSnippet:
    """One stable SEC 8-K text snippet for semantic classification."""

    snippet_id: str
    accession: str
    archive_cik: str
    issuer_cik: str
    ticker: str
    accepted_at_utc: str | None
    filing_date: str | None
    sec_item_number: str | None
    route_family: str
    route_reason: str
    snippet_kind: str
    document_type: str
    filename: str | None
    source_hash: str
    snippet_hash: str
    labelability_status: str | None
    labelability_blockers: list[str]
    text: str

    def to_dict(self) -> dict[str, object]:
        """Return a parquet/JSON-safe representation."""
        return {
            "snippet_id": self.snippet_id,
            "accession": self.accession,
            "archive_cik": self.archive_cik,
            "issuer_cik": self.issuer_cik,
            "ticker": self.ticker,
            "accepted_at_utc": self.accepted_at_utc,
            "filing_date": self.filing_date,
            "sec_item_number": self.sec_item_number,
            "route_family": self.route_family,
            "route_reason": self.route_reason,
            "snippet_kind": self.snippet_kind,
            "document_type": self.document_type,
            "filename": self.filename,
            "source_hash": self.source_hash,
            "snippet_hash": self.snippet_hash,
            "labelability_status": self.labelability_status,
            "labelability_blockers": self.labelability_blockers,
            "text": self.text,
        }


def run_sec_event_semantic_classification(
    *,
    data_root: Path,
    client: SecEventSemanticBatchClient | None = None,
    model: str = DEFAULT_SEC_EVENT_MODEL,
    base_url: str = DEFAULT_LMSTUDIO_BASE_URL,
    timeout_seconds: float = 180.0,
    response_format_mode: str = "json_schema",
    batch_size: int = 4,
    limit: int | None = None,
    max_snippet_chars: int = 4000,
    routing_mode: str = "broad",
    target_items: Iterable[str] | None = None,
    accepted_from: str | None = None,
    accepted_to: str | None = None,
    snippet_kind: str = "all",
    labelability_mode: str = "all",
    resume: bool = False,
    checkpoint_path: Path | None = None,
) -> dict[str, object]:
    """Build SEC 8-K snippets, classify them, and curate semantic candidates."""
    root = Path(data_root).expanduser()
    normalized_routing_mode = _routing_mode(routing_mode)
    normalized_target_items = _target_items(target_items)
    classifier = client or LMStudioSecEventSemanticClient(
        model=model,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
        response_format_mode=response_format_mode,
    )
    snippets, queue_metadata = build_sec8k_semantic_snippet_queue(
        root=root,
        limit=limit,
        max_snippet_chars=max_snippet_chars,
        routing_mode=normalized_routing_mode,
        target_items=normalized_target_items,
        accepted_from=accepted_from,
        accepted_to=accepted_to,
        snippet_kind=snippet_kind,
        labelability_mode=labelability_mode,
    )
    snippet_payload = write_sec8k_semantic_snippets(root=root, snippets=snippets)
    queue_payload = write_sec8k_semantic_queue_report(
        root=root,
        snippets=snippets,
        routing_mode=normalized_routing_mode,
        target_items=normalized_target_items,
        accepted_from=accepted_from,
        accepted_to=accepted_to,
        snippet_kind=snippet_kind,
        labelability_mode=labelability_mode,
        queue_metadata=queue_metadata,
    )
    classifications = classify_sec8k_semantic_snippets(
        snippets=snippets,
        client=classifier,
        batch_size=batch_size,
        root=root,
        resume=resume,
        checkpoint_path=checkpoint_path,
    )
    classification_payload = write_sec8k_semantic_classifications(
        root=root,
        classifications=classifications,
    )
    candidates = build_sec8k_semantic_candidates(
        snippets=pd.DataFrame([snippet.to_dict() for snippet in snippets]),
        classifications=classifications,
    )
    candidate_payload = write_sec8k_semantic_candidates(root=root, candidates=candidates)
    payload = {
        "version": SEC8K_SEMANTIC_CLASSIFICATION_VERSION,
        "schema_version": SEC_EVENT_SEMANTIC_SCHEMA_VERSION,
        "checked_at": _now_iso(),
        "verdict": _classification_verdict(classifications),
        "data_root": str(root),
        "model": getattr(classifier, "model", model),
        "response_format_mode": getattr(
            classifier,
            "response_format_mode",
            response_format_mode,
        ),
        "routing_mode": normalized_routing_mode,
        "target_items": list(normalized_target_items),
        "accepted_from": _date_bound(accepted_from),
        "accepted_to": _date_bound(accepted_to),
        "snippet_kind": _snippet_kind(snippet_kind),
        "labelability_mode": _labelability_mode(labelability_mode),
        "resume": bool(resume),
        "batch_size": int(batch_size),
        "limit": limit,
        "snippet_count": int(len(snippets)),
        "classification_count": int(len(classifications)),
        "reused_classification_count": int(classifications.attrs.get("reused_count", 0)),
        "new_classification_count": int(classifications.attrs.get("classified_count", 0)),
        "early_stop_reason": classifications.attrs.get("early_stop_reason"),
        "promoted_candidate_count": int(len(candidates)),
        "blocking_error_count": int(
            sum(len(_list(value)) for value in classifications.get("errors", pd.Series(dtype=object)).tolist())
        )
        if not classifications.empty
        else 0,
        "warning_count": int(
            sum(len(_list(value)) for value in classifications.get("warnings", pd.Series(dtype=object)).tolist())
        )
        if not classifications.empty
        else 0,
        "artifacts": {
            "snippets": snippet_payload,
            "queue": queue_payload,
            "checkpoint": classifications.attrs.get("checkpoint_payload", {}),
            "classifications": classification_payload,
            "candidates": candidate_payload,
        },
    }
    return write_sec8k_semantic_classification_report(root=root, payload=_json_safe(payload))


def run_sec_event_semantic_labelability_audit(
    *,
    data_root: Path,
    routing_mode: str = "targeted",
    target_items: Iterable[str] | None = None,
    accepted_from: str | None = None,
    accepted_to: str | None = None,
    snippet_kind: str = "item_section",
    horizons: Iterable[int] | None = None,
    round_trip_cost_bps: float = 50.0,
    market_data_roots: Iterable[Path] | None = None,
    source_contract_path: Path | None = None,
) -> dict[str, object]:
    """Audit whether deterministic SEC 8-K candidates can be market-labeled."""
    root = Path(data_root).expanduser()
    queue, source_metadata = build_sec8k_semantic_labelability_queue(
        root=root,
        routing_mode=routing_mode,
        target_items=target_items,
        accepted_from=accepted_from,
        accepted_to=accepted_to,
        snippet_kind=snippet_kind,
        horizons=horizons,
        round_trip_cost_bps=round_trip_cost_bps,
        market_data_roots=market_data_roots,
        source_contract_path=source_contract_path,
    )
    return write_sec8k_semantic_labelability_audit(
        root=root,
        queue=queue,
        source_metadata=source_metadata,
        routing_mode=_routing_mode(routing_mode),
        target_items=_target_items(target_items),
        accepted_from=accepted_from,
        accepted_to=accepted_to,
        snippet_kind=snippet_kind,
        horizons=tuple(sorted(set(int(item) for item in (horizons or DEFAULT_SEMANTIC_LABELABILITY_HORIZONS)))),
        round_trip_cost_bps=round_trip_cost_bps,
    )


def run_sec_event_semantic_scaled_gate(
    *,
    data_root: Path,
    client: SecEventSemanticBatchClient | None = None,
    model: str = DEFAULT_SEC_EVENT_MODEL,
    base_url: str = DEFAULT_MAC_MINI_LMSTUDIO_BASE_URL,
    timeout_seconds: float = 300.0,
    response_format_mode: str = "prompt_json",
    batch_size: int = 1,
    target_items: Iterable[str] | None = None,
    fallback_target_items: Iterable[str] | None = None,
    years: Iterable[int] | None = None,
    max_snippets: int | None = None,
    resume: bool = True,
    primary_horizon: int = PRIMARY_HORIZON,
    min_sample: int = 100,
    min_mean_abret: float = 0.005,
    min_control_separation: float = 0.0075,
    max_top5_abs_contribution: float = 0.35,
) -> dict[str, object]:
    """Run the labelability-first scaled SEC 8-K semantic decision gate."""
    root = Path(data_root).expanduser()
    classifier = client or LMStudioSecEventSemanticClient(
        model=model,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
        response_format_mode=response_format_mode,
    )
    primary_items = _target_items(
        DEFAULT_SCALED_GATE_TARGET_ITEMS if target_items is None else target_items
    )
    fallback_items = (
        _target_items(DEFAULT_SCALED_GATE_FALLBACK_ITEMS)
        if fallback_target_items is None
        else _target_items_exact(fallback_target_items)
    )
    selected_years = tuple(int(year) for year in (years or DEFAULT_SCALED_GATE_YEARS))
    scenarios = _scaled_gate_scenarios(
        years=selected_years,
        primary_items=primary_items,
        fallback_items=fallback_items,
    )
    scenario_payloads: list[dict[str, object]] = []
    final_verdict: dict[str, object] | None = None
    checkpoint_path = (
        root
        / "data"
        / "curated"
        / "events"
        / "sec_event_semantic_classification_checkpoints"
        / "scaled_gate.parquet"
    )
    for scenario in scenarios:
        snippets, queue_metadata = build_sec8k_semantic_snippet_queue(
            root=root,
            limit=max_snippets,
            routing_mode="targeted",
            target_items=scenario["target_items"],
            accepted_from=scenario["accepted_from"],
            accepted_to=scenario["accepted_to"],
            snippet_kind="item_section",
            labelability_mode="labelable-only",
        )
        snippet_payload = write_sec8k_semantic_snippets(root=root, snippets=snippets)
        queue_payload = write_sec8k_semantic_queue_report(
            root=root,
            snippets=snippets,
            routing_mode="targeted",
            target_items=tuple(str(item) for item in scenario["target_items"]),
            accepted_from=str(scenario["accepted_from"]),
            accepted_to=str(scenario["accepted_to"]),
            snippet_kind="item_section",
            labelability_mode="labelable-only",
            queue_metadata=queue_metadata,
        )
        classifications = classify_sec8k_semantic_snippets(
            snippets=snippets,
            client=classifier,
            batch_size=batch_size,
            root=root,
            resume=resume,
            checkpoint_path=checkpoint_path,
            min_promotable_sample=min_sample,
        )
        classification_payload = write_sec8k_semantic_classifications(
            root=root,
            classifications=classifications,
        )
        candidates = build_sec8k_semantic_candidates(
            snippets=pd.DataFrame([snippet.to_dict() for snippet in snippets]),
            classifications=classifications,
        )
        candidate_payload = write_sec8k_semantic_candidates(root=root, candidates=candidates)
        classification_report = write_sec8k_semantic_classification_report(
            root=root,
            payload=_json_safe(
                {
                    "version": SEC8K_SEMANTIC_CLASSIFICATION_VERSION,
                    "schema_version": SEC_EVENT_SEMANTIC_SCHEMA_VERSION,
                    "checked_at": _now_iso(),
                    "verdict": _classification_verdict(classifications),
                    "data_root": str(root),
                    "model": getattr(classifier, "model", model),
                    "response_format_mode": getattr(
                        classifier,
                        "response_format_mode",
                        response_format_mode,
                    ),
                    "routing_mode": "targeted",
                    "target_items": list(scenario["target_items"]),
                    "accepted_from": scenario["accepted_from"],
                    "accepted_to": scenario["accepted_to"],
                    "snippet_kind": "item_section",
                    "labelability_mode": "labelable-only",
                    "scaled_gate_scenario": scenario["name"],
                    "resume": bool(resume),
                    "batch_size": int(batch_size),
                    "limit": max_snippets,
                    "snippet_count": int(len(snippets)),
                    "classification_count": int(len(classifications)),
                    "reused_classification_count": int(classifications.attrs.get("reused_count", 0)),
                    "new_classification_count": int(classifications.attrs.get("classified_count", 0)),
                    "early_stop_reason": classifications.attrs.get("early_stop_reason"),
                    "promoted_candidate_count": int(len(candidates)),
                    "artifacts": {
                        "snippets": snippet_payload,
                        "queue": queue_payload,
                        "checkpoint": classifications.attrs.get("checkpoint_payload", {}),
                        "classifications": classification_payload,
                        "candidates": candidate_payload,
                    },
                }
            ),
        )
        if candidates.empty:
            study_payload = {
                "packet": {
                    "primary": {"n": 0},
                    "verdict": _verdict(
                        "MORE_DATA_REQUIRED",
                        False,
                        ["no_promoted_semantic_candidates"],
                    ),
                }
            }
        else:
            study_payload = run_sec_event_semantic_study(
                data_root=root,
                primary_horizon=primary_horizon,
                horizons=(primary_horizon,),
                min_sample=min_sample,
                min_mean_abret=min_mean_abret,
                min_control_separation=min_control_separation,
                max_top5_abs_contribution=max_top5_abs_contribution,
            )
        study_packet = study_payload.get("packet", {})
        study_verdict = _dict(study_packet.get("verdict"))
        scenario_payload = {
            **scenario,
            "queue_metadata": queue_metadata,
            "snippet_count": int(len(snippets)),
            "classification_count": int(len(classifications)),
            "promoted_candidate_count": int(len(candidates)),
            "early_stop_reason": classifications.attrs.get("early_stop_reason"),
            "labeled_count": int(_dict(study_packet.get("primary")).get("n") or 0),
            "classification_report": classification_report,
            "study_verdict": study_verdict,
        }
        scenario_payloads.append(_json_safe(scenario_payload))
        if _text(study_verdict.get("decision")) in {
            "CONTINUE_SEMANTIC_8K",
            "SEMANTIC_8K_KILLED",
        }:
            final_verdict = study_verdict
            break
    if final_verdict is None:
        final_verdict = _scaled_gate_fallback_verdict(
            scenarios=scenario_payloads,
            min_sample=min_sample,
        )
    packet = _json_safe(
        {
            "version": SEC8K_SEMANTIC_SCALED_GATE_VERSION,
            "checked_at": _now_iso(),
            "data_root": str(root),
            "model": getattr(classifier, "model", model),
            "base_url": base_url,
            "response_format_mode": getattr(
                classifier,
                "response_format_mode",
                response_format_mode,
            ),
            "batch_size": int(batch_size),
            "resume": bool(resume),
            "max_snippets": max_snippets,
            "min_sample": int(min_sample),
            "scenarios": scenario_payloads,
            "verdict": final_verdict,
        }
    )
    return write_sec_event_semantic_scaled_gate_packet(root=root, packet=packet)


def run_sec_event_semantic_study(
    *,
    data_root: Path,
    primary_horizon: int = PRIMARY_HORIZON,
    horizons: Iterable[int] | None = None,
    round_trip_cost_bps: float = 50.0,
    market_data_roots: Iterable[Path] | None = None,
    source_contract_path: Path | None = None,
    min_sample: int = 100,
    min_mean_abret: float = 0.005,
    min_control_separation: float = 0.0075,
    max_top5_abs_contribution: float = 0.35,
) -> dict[str, object]:
    """Label SEC 8-K semantic candidates and write a continue/kill study packet."""
    root = Path(data_root).expanduser()
    candidates_path = root / "data" / "curated" / "events" / "sec_event_semantic_candidates" / "data.parquet"
    if not candidates_path.exists():
        raise FileNotFoundError(f"missing SEC event semantic candidates: {candidates_path}")
    candidates = read_sec8k_item_event_candidates(candidates_path)
    config = Form4LabelConfig(
        horizons=tuple(sorted(set(int(item) for item in (horizons or DEFAULT_HORIZONS)))),
        round_trip_cost_bps=float(round_trip_cost_bps),
    )
    labels, source_metadata = build_form4_event_labels_with_market_sources(
        candidates=candidates,
        data_root=root,
        config=config,
        market_data_roots=market_data_roots,
        source_contract_path=source_contract_path,
    )
    labels_payload = _write_semantic_label_artifact(
        root=root,
        name="sec_event_semantic_labels",
        labels=labels,
        config=config,
        source_metadata=source_metadata,
    )
    placebo = _timestamp_placebo_candidates(candidates)
    placebo_labels, placebo_source_metadata = build_form4_event_labels_with_market_sources(
        candidates=placebo,
        data_root=root,
        config=config,
        market_data_roots=market_data_roots,
        source_contract_path=source_contract_path,
    )
    placebo_payload = _write_semantic_label_artifact(
        root=root,
        name="sec_event_semantic_timestamp_placebo_labels",
        labels=placebo_labels,
        config=config,
        source_metadata=placebo_source_metadata,
    )
    classification_report = _read_optional_json(
        root / "control" / "cluster" / "state" / "research" / "sec_event_semantic_classification" / "latest.json"
    )
    header_packet = _read_optional_json(
        root / "control" / "cluster" / "state" / "research" / "sec_8k_event_study" / "latest.json"
    )
    packet = build_sec_event_semantic_study_packet(
        labels=labels,
        placebo_labels=placebo_labels,
        labels_payload=labels_payload,
        placebo_payload=placebo_payload,
        classification_report=classification_report,
        header_packet=header_packet,
        primary_horizon=primary_horizon,
        round_trip_cost_bps=round_trip_cost_bps,
        min_sample=min_sample,
        min_mean_abret=min_mean_abret,
        min_control_separation=min_control_separation,
        max_top5_abs_contribution=max_top5_abs_contribution,
    )
    return write_sec_event_semantic_study_packet(root=root, packet=packet)


def build_sec8k_semantic_snippets(
    *,
    root: Path,
    limit: int | None = None,
    max_snippet_chars: int = 4000,
    routing_mode: str = "broad",
    target_items: Iterable[str] | None = None,
    accepted_from: str | None = None,
    accepted_to: str | None = None,
    snippet_kind: str = "all",
    labelability_mode: str = "all",
) -> list[Sec8KSemanticSnippet]:
    """Build bounded snippets from existing SEC 8-K raw complete-text archives."""
    snippets, _ = build_sec8k_semantic_snippet_queue(
        root=root,
        limit=limit,
        max_snippet_chars=max_snippet_chars,
        routing_mode=routing_mode,
        target_items=target_items,
        accepted_from=accepted_from,
        accepted_to=accepted_to,
        snippet_kind=snippet_kind,
        labelability_mode=labelability_mode,
    )
    return snippets


def build_sec8k_semantic_snippet_queue(
    *,
    root: Path,
    limit: int | None = None,
    max_snippet_chars: int = 4000,
    routing_mode: str = "broad",
    target_items: Iterable[str] | None = None,
    accepted_from: str | None = None,
    accepted_to: str | None = None,
    snippet_kind: str = "all",
    labelability_mode: str = "all",
) -> tuple[list[Sec8KSemanticSnippet], dict[str, object]]:
    """Build semantic snippets and queue metadata from SEC 8-K raw archives."""
    normalized_routing_mode = _routing_mode(routing_mode)
    normalized_target_items = _target_items(target_items)
    normalized_snippet_kind = _snippet_kind(snippet_kind)
    normalized_labelability_mode = _labelability_mode(labelability_mode)
    candidates_path = root / "data" / "curated" / "events" / "sec_8k_item_events" / "data.parquet"
    if not candidates_path.exists():
        raise FileNotFoundError(f"missing SEC 8-K item candidates: {candidates_path}")
    candidates = read_sec8k_item_event_candidates(candidates_path)
    if candidates.empty:
        return [], {
            "labelability_mode": normalized_labelability_mode,
            "snippet_kind": normalized_snippet_kind,
            "routed_candidate_count": 0,
            "labelable_candidate_count": 0,
            "labelability_status_counts": {},
            "labelability_blocker_counts": {},
        }
    rows = _routed_candidate_rows(
        candidates=candidates,
        routing_mode=normalized_routing_mode,
        target_items=normalized_target_items,
        accepted_from=accepted_from,
        accepted_to=accepted_to,
    )
    queue_metadata: dict[str, object] = {
        "labelability_mode": normalized_labelability_mode,
        "snippet_kind": normalized_snippet_kind,
        "routed_candidate_count": int(len(rows)),
    }
    if normalized_labelability_mode != "all":
        labelability, _ = _labelability_frame_for_rows(
            root=root,
            rows=rows,
            snippet_kind="item_section",
            horizons=DEFAULT_SEMANTIC_LABELABILITY_HORIZONS,
            round_trip_cost_bps=50.0,
            market_data_roots=None,
            source_contract_path=None,
        )
        queue_metadata.update(_labelability_summary(labelability))
        labelability_by_event_id = {
            _text(row.get("event_id")): row for row in labelability.to_dict("records")
        }
        if normalized_labelability_mode == "labelable-only":
            rows = [
                _merge_labelability_row(row, labelability_by_event_id)
                for row in rows
                if _text(labelability_by_event_id.get(_text(row.get("event_id")), {}).get("labelability_status"))
                == "LABELABLE"
            ]
        elif normalized_labelability_mode == "prefer-labelable":
            rows = [
                _merge_labelability_row(row, labelability_by_event_id)
                for row in sorted(
                    rows,
                    key=lambda item: (
                        _text(
                            labelability_by_event_id.get(_text(item.get("event_id")), {}).get(
                                "labelability_status"
                            )
                        )
                        != "LABELABLE",
                        _candidate_event_date(item),
                        _text(item.get("accession")),
                        _normalize_sec_item(item.get("sec_item_number")),
                    ),
                )
            ]
        queue_metadata["post_labelability_candidate_count"] = int(len(rows))
    else:
        queue_metadata.update(
            {
                "labelable_candidate_count": None,
                "labelability_status_counts": {},
                "labelability_blocker_counts": {},
                "post_labelability_candidate_count": int(len(rows)),
            }
        )
    accessions = {
        _text(row.get("accession") or _first(_list(row.get("accessions")))).replace("-", "")
        for row in rows
        if _text(row.get("accession") or _first(_list(row.get("accessions"))))
    }
    complete_text_by_accession = _complete_text_lookup(root=root, accessions=accessions)
    archive_cik_by_accession = _archive_cik_lookup(root=root, accessions=accessions)
    snippets_by_id: dict[str, Sec8KSemanticSnippet] = {}
    snippets: list[Sec8KSemanticSnippet] = []
    for row in rows:
        accession = _text(row.get("accession") or _first(_list(row.get("accessions"))))
        complete_text = complete_text_by_accession.get(accession.replace("-", ""))
        if not accession or not complete_text:
            continue
        context = dict(row)
        context["archive_cik"] = _text(context.get("archive_cik")) or archive_cik_by_accession.get(accession.replace("-", ""), "")
        route_family = _text(context.get("route_family")) or _route_family(
            _text(context.get("sec_item_number")),
            routing_mode=normalized_routing_mode,
        )
        route_reason = _text(context.get("route_reason")) or _route_reason(
            _text(context.get("sec_item_number")),
            routing_mode=normalized_routing_mode,
        )
        source_hash = hashlib.sha256(complete_text.encode("utf-8")).hexdigest()
        item_number = _text(row.get("sec_item_number"))
        item_sections = _item_sections_from_complete_text(complete_text, max_chars=max_snippet_chars)
        section_text = item_sections.get(item_number or "")
        if section_text and normalized_snippet_kind in {"all", "item_section"}:
            snippet = _snippet_from_row(
                row=context,
                accession=accession,
                source_hash=source_hash,
                snippet_kind="item_section",
                document_type="8-K",
                filename=_text(row.get("primary_document")) or "complete.txt",
                sec_item_number=item_number,
                route_family=route_family,
                route_reason=route_reason,
                text=section_text,
            )
            if snippet.snippet_id not in snippets_by_id:
                snippets_by_id[snippet.snippet_id] = snippet
                snippets.append(snippet)
        if normalized_snippet_kind in {"all", "exhibit"}:
            for exhibit in _exhibit_snippets_from_complete_text(
                complete_text,
                max_chars=max_snippet_chars,
            ):
                snippet = _snippet_from_row(
                    row=context,
                    accession=accession,
                    source_hash=source_hash,
                    snippet_kind="exhibit",
                    document_type=exhibit["document_type"],
                    filename=exhibit["filename"],
                    sec_item_number=None,
                    route_family=route_family,
                    route_reason=route_reason,
                    text=exhibit["text"],
                )
                if snippet.snippet_id not in snippets_by_id:
                    snippets_by_id[snippet.snippet_id] = snippet
                    snippets.append(snippet)
    snippets = sorted(
        snippets,
        key=_semantic_labelability_snippet_rank
        if normalized_labelability_mode == "prefer-labelable"
        else _semantic_snippet_rank,
    )
    queue_metadata["snippet_kind_counts"] = _snippet_kind_counts(snippets)
    queue_metadata["snippet_count_before_limit"] = int(len(snippets))
    if limit is not None:
        snippets = snippets[: max(0, int(limit))]
    queue_metadata["snippet_limit"] = limit
    queue_metadata["snippet_count_after_limit"] = int(len(snippets))
    return snippets, queue_metadata


def build_sec8k_semantic_labelability_queue(
    *,
    root: Path,
    routing_mode: str = "targeted",
    target_items: Iterable[str] | None = None,
    accepted_from: str | None = None,
    accepted_to: str | None = None,
    snippet_kind: str = "item_section",
    horizons: Iterable[int] | None = None,
    round_trip_cost_bps: float = 50.0,
    market_data_roots: Iterable[Path] | None = None,
    source_contract_path: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build a market-label feasibility queue for deterministic SEC 8-K rows."""
    normalized_routing_mode = _routing_mode(routing_mode)
    normalized_target_items = _target_items(target_items)
    candidates_path = root / "data" / "curated" / "events" / "sec_8k_item_events" / "data.parquet"
    if not candidates_path.exists():
        raise FileNotFoundError(f"missing SEC 8-K item candidates: {candidates_path}")
    candidates = pd.read_parquet(candidates_path)
    rows = _routed_candidate_rows(
        candidates=candidates,
        routing_mode=normalized_routing_mode,
        target_items=normalized_target_items,
        accepted_from=accepted_from,
        accepted_to=accepted_to,
    )
    return _labelability_frame_for_rows(
        root=root,
        rows=rows,
        snippet_kind=snippet_kind,
        horizons=horizons or DEFAULT_SEMANTIC_LABELABILITY_HORIZONS,
        round_trip_cost_bps=round_trip_cost_bps,
        market_data_roots=market_data_roots,
        source_contract_path=source_contract_path,
    )


def _labelability_frame_for_rows(
    *,
    root: Path,
    rows: list[dict[str, object]],
    snippet_kind: str,
    horizons: Iterable[int],
    round_trip_cost_bps: float,
    market_data_roots: Iterable[Path] | None,
    source_contract_path: Path | None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    normalized_rows = [dict(row, snippet_kind=snippet_kind) for row in rows]
    if not normalized_rows:
        return _empty_labelability_frame(), {"datasets": {}}
    candidates = pd.DataFrame(normalized_rows)
    config = Form4LabelConfig(
        horizons=tuple(sorted(set(int(item) for item in horizons))),
        round_trip_cost_bps=float(round_trip_cost_bps),
    )
    labels, source_metadata = build_form4_event_labels_with_market_sources(
        candidates=candidates,
        data_root=root,
        config=config,
        market_data_roots=market_data_roots,
        source_contract_path=source_contract_path,
    )
    labels_by_event_id = {
        _text(row.get("event_id")): row for row in labels.to_dict("records")
    }
    output_rows: list[dict[str, object]] = []
    for row in normalized_rows:
        event_id = _text(row.get("event_id"))
        label = labels_by_event_id.get(event_id, {})
        output = dict(row)
        blockers = _labelability_blockers(candidate=row, label=label)
        label_status = _text(label.get("label_status"))
        output["tradable_at_utc"] = _text(label.get("tradable_at_utc") or row.get("tradable_at_utc")) or None
        output["label_status"] = label_status or None
        output["labelability_status"] = _labelability_status(label_status, blockers)
        output["labelability_blockers"] = blockers
        output["labelability_horizons"] = list(config.horizons)
        output["round_trip_cost_bps"] = config.round_trip_cost_bps
        output_rows.append(output)
    return pd.DataFrame(output_rows), source_metadata


def _empty_labelability_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "event_id",
            "accession",
            "ticker",
            "accepted_at_utc",
            "sec_item_number",
            "snippet_kind",
            "label_status",
            "labelability_status",
            "labelability_blockers",
            "tradable_at_utc",
        ]
    )


def read_sec8k_item_event_candidates(path: Path) -> pd.DataFrame:
    """Read SEC 8-K item candidates with only columns needed for semantic routing."""
    if not path.exists():
        return pd.DataFrame()
    schema_columns = set(pq.read_schema(path).names)
    selected = [column for column in SEC8K_ITEM_EVENT_LIGHT_COLUMNS if column in schema_columns]
    if not selected:
        return pd.read_parquet(path)
    return pd.read_parquet(path, columns=selected)


def classify_sec8k_semantic_snippets(
    *,
    snippets: list[Sec8KSemanticSnippet],
    client: SecEventSemanticBatchClient,
    batch_size: int = 4,
    root: Path | None = None,
    resume: bool = False,
    checkpoint_path: Path | None = None,
    min_promotable_sample: int | None = None,
) -> pd.DataFrame:
    """Classify snippets in bounded batches and return audit rows."""
    checkpoint = _semantic_checkpoint_path(root=root, checkpoint_path=checkpoint_path)
    checkpoint_rows = _read_semantic_checkpoint(checkpoint) if resume else pd.DataFrame()
    reusable = _reusable_checkpoint_rows(
        snippets=snippets,
        checkpoint_rows=checkpoint_rows,
        model=_text(getattr(client, "model", "")),
    )
    reusable_ids = set(reusable["snippet_id"].astype(str).tolist()) if not reusable.empty else set()
    rows: list[dict[str, object]] = reusable.to_dict("records") if not reusable.empty else []
    pending = [snippet for snippet in snippets if snippet.snippet_id not in reusable_ids]
    early_stop_reason: str | None = None
    min_promotable = int(min_promotable_sample or 0)
    if min_promotable > 0 and _promotable_count(rows) >= min_promotable:
        early_stop_reason = "promotable_sample_gate_reached"
        pending = []
    if min_promotable > 0 and _max_possible_promotable_count(rows=rows, pending=pending) < min_promotable:
        early_stop_reason = "promotable_sample_gate_mathematically_impossible"
        pending = []
    width = max(1, int(batch_size))
    for batch_index, start in enumerate(range(0, len(pending), width)):
        batch = pending[start : start + width]
        batch_rows = _classify_batch_with_isolation(
            batch=batch,
            client=client,
            batch_index=batch_index,
            recovery_warnings=[],
        )
        batch_rows = [_annotate_classification_row(row) for row in batch_rows]
        rows.extend(batch_rows)
        _write_semantic_checkpoint(path=checkpoint, rows=rows)
        remaining = pending[start + width :]
        if min_promotable > 0 and _promotable_count(rows) >= min_promotable:
            early_stop_reason = "promotable_sample_gate_reached"
            break
        if (
            min_promotable > 0
            and _max_possible_promotable_count(rows=rows, pending=remaining) < min_promotable
        ):
            early_stop_reason = "promotable_sample_gate_mathematically_impossible"
            break
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values("snippet_id").reset_index(drop=True)
    new_classified_count = sum(
        1 for row in rows if _text(row.get("snippet_id")) not in reusable_ids
    )
    frame.attrs["reused_count"] = int(len(reusable_ids))
    frame.attrs["classified_count"] = int(new_classified_count)
    frame.attrs["checkpoint_payload"] = {
        "path": str(checkpoint),
        "row_count": int(len(frame)),
        "reused_count": int(len(reusable_ids)),
        "classified_count": int(new_classified_count),
        "early_stop_reason": early_stop_reason,
    }
    frame.attrs["early_stop_reason"] = early_stop_reason
    if pending == []:
        _write_semantic_checkpoint(path=checkpoint, rows=rows)
    return frame


def build_sec8k_semantic_candidates(
    *, snippets: pd.DataFrame, classifications: pd.DataFrame
) -> pd.DataFrame:
    """Promote validated semantic classifications into labelable event candidates."""
    if classifications.empty:
        return pd.DataFrame()
    snippet_lookup = {
        str(row["snippet_id"]): row for row in snippets.to_dict("records")
    } if not snippets.empty else {}
    rows_by_key: dict[tuple[str, str, str, str, str], dict[str, object]] = {}
    snippet_ids_by_key: dict[tuple[str, str, str, str, str], list[str]] = {}
    for row in classifications.sort_values("snippet_id").to_dict("records"):
        snippet = snippet_lookup.get(str(row.get("snippet_id")), {})
        event_type = _text(row.get("event_type"))
        errors = _list(row.get("errors"))
        if event_type not in PROMOTABLE_EVENT_TYPES or errors:
            continue
        ticker = _text(row.get("ticker") or snippet.get("ticker"))
        issuer_cik = _text(row.get("issuer_cik") or snippet.get("issuer_cik"))
        accepted_at = _text(row.get("accepted_at_utc") or snippet.get("accepted_at_utc"))
        exclusions = [str(item) for item in _list(row.get("exclusion_reasons")) if str(item)]
        if not ticker:
            exclusions.append("missing_ticker")
        if not issuer_cik:
            exclusions.append("missing_issuer_cik")
        if not accepted_at:
            exclusions.append("missing_accepted_at")
        accession = _text(row.get("accession"))
        digest = hashlib.sha256(
            f"{accession}:{event_type}:{ticker}:{issuer_cik}:{accepted_at}".encode("utf-8")
        ).hexdigest()[:16]
        key = (accession, event_type, ticker, issuer_cik, accepted_at)
        snippet_id = _text(row.get("snippet_id"))
        snippet_ids_by_key.setdefault(key, [])
        if snippet_id and snippet_id not in snippet_ids_by_key[key]:
            snippet_ids_by_key[key].append(snippet_id)
        candidate = {
            "event_id": f"sec8k_semantic_{digest}",
            "issuer_cik": issuer_cik,
            "ticker": ticker,
            "primary_security_id": f"{issuer_cik}:{ticker}" if issuer_cik and ticker else "",
            "accession": accession,
            "accessions": [accession],
            "event_type": event_type,
            "accepted_at_utc": accepted_at or None,
            "first_seen_at_utc": accepted_at or None,
            "tradable_at_utc": None,
            "eligibility_pass": not exclusions,
            "exclusion_reasons": exclusions,
            "event_strength_score": 1.0,
            "semantic_snippet_id": row.get("snippet_id"),
            "semantic_certainty": row.get("certainty"),
            "semantic_warnings": _list(row.get("warnings")),
            "semantic_evidence": _list(row.get("evidence")),
            "semantic_fields": _dict(row.get("fields")),
            "sec_item_number": row.get("sec_item_number"),
            "route_family": row.get("route_family"),
            "route_reason": row.get("route_reason"),
            "snippet_kind": row.get("snippet_kind"),
            "document_type": row.get("document_type"),
            "filename": row.get("filename"),
            "source_hash": row.get("source_hash"),
            "snippet_hash": row.get("snippet_hash"),
            "parser_version": SEC8K_SEMANTIC_CLASSIFICATION_VERSION,
            "schema_version": SEC8K_SEMANTIC_CLASSIFICATION_VERSION,
        }
        current = rows_by_key.get(key)
        if current is None or _semantic_candidate_rank(candidate) < _semantic_candidate_rank(current):
            rows_by_key[key] = candidate
    rows = []
    for key, candidate in rows_by_key.items():
        snippet_ids = snippet_ids_by_key.get(key, [])
        candidate["semantic_snippet_ids"] = snippet_ids
        candidate["duplicate_semantic_snippet_count"] = max(0, len(snippet_ids) - 1)
        rows.append(candidate)
    return pd.DataFrame(rows).sort_values("event_id").reset_index(drop=True) if rows else pd.DataFrame()


def write_sec8k_semantic_snippets(
    *, root: Path, snippets: list[Sec8KSemanticSnippet]
) -> dict[str, object]:
    """Write SEC 8-K semantic snippets."""
    frame = pd.DataFrame([snippet.to_dict() for snippet in snippets])
    path = root / "data" / "curated" / "events" / "sec_event_semantic_snippets" / "data.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return {"path": str(path), "row_count": int(len(frame))}


def write_sec8k_semantic_queue_report(
    *,
    root: Path,
    snippets: list[Sec8KSemanticSnippet],
    routing_mode: str,
    target_items: tuple[str, ...],
    accepted_from: str | None = None,
    accepted_to: str | None = None,
    snippet_kind: str = "all",
    labelability_mode: str = "all",
    queue_metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    """Write a compact SEC semantic routing/queue report."""
    frame = pd.DataFrame([snippet.to_dict() for snippet in snippets])
    state = root / "control" / "cluster" / "state" / "research" / "sec_event_semantic_queue"
    history = state / "history"
    history.mkdir(parents=True, exist_ok=True)
    payload = _json_safe(
        {
            "version": SEC8K_SEMANTIC_CLASSIFICATION_VERSION,
            "checked_at": _now_iso(),
            "routing_mode": routing_mode,
            "target_items": list(target_items),
            "accepted_from": _date_bound(accepted_from),
            "accepted_to": _date_bound(accepted_to),
            "snippet_kind": _snippet_kind(snippet_kind),
            "labelability_mode": _labelability_mode(labelability_mode),
            "snippet_count": int(len(frame)),
            "accession_count": int(frame["accession"].nunique()) if "accession" in frame else 0,
            "route_family_counts": _value_counts(frame, "route_family"),
            "sec_item_counts": _value_counts(frame, "sec_item_number"),
            "snippet_kind_counts": _value_counts(frame, "snippet_kind"),
            "labelability_status_counts": _value_counts(frame, "labelability_status"),
            "labelability_blocker_counts": _blocker_counts(
                frame.get("labelability_blockers", pd.Series(dtype=object)).tolist()
            )
            if not frame.empty
            else {},
            "queue_metadata": queue_metadata or {},
        }
    )
    latest = state / "latest.json"
    latest.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    history_path = history / f"{payload['checked_at'].replace(':', '').replace('+', '_')}.json"
    history_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return {"report_path": str(latest), "history_path": str(history_path), "payload": payload}


def write_sec8k_semantic_labelability_audit(
    *,
    root: Path,
    queue: pd.DataFrame,
    source_metadata: dict[str, object],
    routing_mode: str,
    target_items: tuple[str, ...],
    accepted_from: str | None,
    accepted_to: str | None,
    snippet_kind: str,
    horizons: tuple[int, ...],
    round_trip_cost_bps: float,
) -> dict[str, object]:
    """Write SEC 8-K semantic labelability queue and report artifacts."""
    path = root / "data" / "curated" / "events" / "sec_event_semantic_labelability_queue" / "data.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    queue.to_parquet(path, index=False)
    state = root / "control" / "cluster" / "state" / "research" / "sec_event_semantic_labelability"
    history = state / "history"
    history.mkdir(parents=True, exist_ok=True)
    payload = _json_safe(
        {
            "version": SEC8K_SEMANTIC_LABELABILITY_VERSION,
            "checked_at": _now_iso(),
            "data_root": str(root),
            "routing_mode": routing_mode,
            "target_items": list(target_items),
            "accepted_from": _date_bound(accepted_from),
            "accepted_to": _date_bound(accepted_to),
            "snippet_kind": snippet_kind,
            "horizons": list(horizons),
            "round_trip_cost_bps": float(round_trip_cost_bps),
            "candidate_count": int(len(queue)),
            "labelable_count": int((queue.get("labelability_status") == "LABELABLE").sum())
            if "labelability_status" in queue
            else 0,
            "status_counts": _value_counts(queue, "labelability_status"),
            "blocker_counts": _blocker_counts(
                queue.get("labelability_blockers", pd.Series(dtype=object)).tolist()
            ),
            "source_metadata": source_metadata,
            "artifact": str(path),
        }
    )
    latest = state / "latest.json"
    latest.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    history_path = history / f"{payload['checked_at'].replace(':', '').replace('+', '_')}.json"
    history_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return {"report_path": str(latest), "history_path": str(history_path), "queue_path": str(path), "payload": payload}


def write_sec8k_semantic_classifications(
    *, root: Path, classifications: pd.DataFrame
) -> dict[str, object]:
    """Write SEC 8-K semantic classification rows."""
    path = root / "data" / "curated" / "events" / "sec_event_semantic_classifications" / "data.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    classifications.to_parquet(path, index=False)
    return {"path": str(path), "row_count": int(len(classifications))}


def write_sec8k_semantic_candidates(
    *, root: Path, candidates: pd.DataFrame
) -> dict[str, object]:
    """Write SEC 8-K semantic candidate rows."""
    path = root / "data" / "curated" / "events" / "sec_event_semantic_candidates" / "data.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_parquet(path, index=False)
    return {"path": str(path), "row_count": int(len(candidates))}


def write_sec8k_semantic_classification_report(
    *, root: Path, payload: dict[str, object]
) -> dict[str, object]:
    """Write SEC 8-K semantic classification report."""
    target = root / "control" / "cluster" / "state" / "research" / "sec_event_semantic_classification"
    history = target / "history"
    history.mkdir(parents=True, exist_ok=True)
    latest = target / "latest.json"
    latest.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    history_path = history / f"{payload['checked_at'].replace(':', '').replace('+', '_')}.json"
    history_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return {"report_path": str(latest), "history_path": str(history_path), "payload": payload}


def build_sec_event_semantic_study_packet(
    *,
    labels: pd.DataFrame,
    placebo_labels: pd.DataFrame,
    labels_payload: dict[str, object],
    placebo_payload: dict[str, object],
    classification_report: dict[str, object] | None,
    header_packet: dict[str, object] | None,
    primary_horizon: int,
    round_trip_cost_bps: float,
    min_sample: int,
    min_mean_abret: float,
    min_control_separation: float,
    max_top5_abs_contribution: float,
) -> dict[str, object]:
    """Build the SEC 8-K semantic event-study packet and verdict."""
    metric = f"abret_{primary_horizon}d_net"
    labeled = _labeled(labels, metric=metric)
    primary = _summary_by_metric(labeled, metric=metric, bootstrap_iterations=1000, bootstrap_seed=101)
    placebo = _summary_by_metric(
        _labeled(placebo_labels, metric=metric),
        metric=metric,
        bootstrap_iterations=1000,
        bootstrap_seed=103,
    )
    by_event_type = {
        str(event_type): _summary_by_metric(
            _labeled(frame, metric=metric),
            metric=metric,
            bootstrap_iterations=1000,
            bootstrap_seed=107,
        )
        for event_type, frame in labels.groupby("event_type", sort=True)
    } if not labels.empty and "event_type" in labels else {}
    separation = _separation(primary, placebo)
    decision = _semantic_decision(
        primary=primary,
        by_event_type=by_event_type,
        separation=separation,
        min_sample=min_sample,
        min_mean_abret=min_mean_abret,
        min_control_separation=min_control_separation,
        max_top5_abs_contribution=max_top5_abs_contribution,
    )
    return _json_safe(
        {
            "version": SEC8K_SEMANTIC_STUDY_VERSION,
            "checked_at": _now_iso(),
            "event_class": "SEC_8K_SEMANTIC_EVENT",
            "primary_horizon": int(primary_horizon),
            "primary_metric": metric,
            "round_trip_cost_bps": float(round_trip_cost_bps),
            "primary": primary,
            "by_event_type": by_event_type,
            "negative_controls": {"timestamp_placebo": placebo},
            "negative_control_separation": {"timestamp_placebo": separation},
            "classification_report": classification_report,
            "header_only_baseline": _header_baseline_summary(header_packet),
            "label_artifacts": labels_payload,
            "timestamp_placebo_artifacts": placebo_payload,
            "verdict": decision,
        }
    )


def write_sec_event_semantic_study_packet(
    *, root: Path, packet: dict[str, object]
) -> dict[str, object]:
    """Write SEC 8-K semantic study JSON and Markdown artifacts."""
    target = root / "control" / "cluster" / "state" / "research" / "sec_event_semantic_study"
    history = target / "history"
    history.mkdir(parents=True, exist_ok=True)
    latest = target / "latest.json"
    latest.write_text(json.dumps(packet, indent=2, sort_keys=True), encoding="utf-8")
    history_path = history / f"{packet['checked_at'].replace(':', '').replace('+', '_')}.json"
    history_path.write_text(json.dumps(packet, indent=2, sort_keys=True), encoding="utf-8")
    report_root = root / "reports" / "research" / "sec_event_semantic_study"
    report_root.mkdir(parents=True, exist_ok=True)
    report_path = report_root / "latest.md"
    report_path.write_text(_render_study_markdown(packet), encoding="utf-8")
    return {
        "packet_path": str(latest),
        "history_path": str(history_path),
        "report_path": str(report_path),
        "packet": packet,
    }


def write_sec_event_semantic_scaled_gate_packet(
    *, root: Path, packet: dict[str, object]
) -> dict[str, object]:
    """Write SEC 8-K semantic scaled-gate JSON artifact."""
    target = root / "control" / "cluster" / "state" / "research" / "sec_event_semantic_scaled_gate"
    history = target / "history"
    history.mkdir(parents=True, exist_ok=True)
    latest = target / "latest.json"
    latest.write_text(json.dumps(packet, indent=2, sort_keys=True), encoding="utf-8")
    history_path = history / f"{packet['checked_at'].replace(':', '').replace('+', '_')}.json"
    history_path.write_text(json.dumps(packet, indent=2, sort_keys=True), encoding="utf-8")
    return {"packet_path": str(latest), "history_path": str(history_path), "packet": packet}


def _write_semantic_label_artifact(
    *,
    root: Path,
    name: str,
    labels: pd.DataFrame,
    config: Form4LabelConfig,
    source_metadata: dict[str, object],
) -> dict[str, object]:
    path = root / "data" / "curated" / "events" / name / "data.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(path, index=False)
    state = root / "control" / "cluster" / "state" / "research" / name
    history = state / "history"
    history.mkdir(parents=True, exist_ok=True)
    status_counts = (
        labels["label_status"].value_counts().sort_index().to_dict()
        if "label_status" in labels
        else {}
    )
    report = _json_safe(
        {
            "version": SEC8K_SEMANTIC_STUDY_VERSION,
            "checked_at": _now_iso(),
            "candidate_count": int(len(labels)),
            "labeled_count": int((labels.get("label_status") == "LABELED").sum())
            if "label_status" in labels
            else 0,
            "blocked_count": int((labels.get("label_status") == "BLOCKED").sum())
            if "label_status" in labels
            else 0,
            "skipped_count": int((labels.get("label_status") == "SKIPPED_INELIGIBLE").sum())
            if "label_status" in labels
            else 0,
            "status_counts": {str(key): int(value) for key, value in status_counts.items()},
            "horizons": list(config.horizons),
            "round_trip_cost_bps": config.round_trip_cost_bps,
            "source_metadata": source_metadata,
            "artifact": str(path),
        }
    )
    latest = state / "latest.json"
    latest.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    history_path = history / f"{report['checked_at'].replace(':', '').replace('+', '_')}.json"
    history_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return {"labels_path": str(path), "report_path": str(latest), "report": report}


def _semantic_decision(
    *,
    primary: dict[str, object],
    by_event_type: dict[str, dict[str, object]],
    separation: dict[str, object],
    min_sample: int,
    min_mean_abret: float,
    min_control_separation: float,
    max_top5_abs_contribution: float,
) -> dict[str, object]:
    n = int(primary.get("n") or 0)
    if n == 0:
        return _verdict("BLOCKED_DATA_COVERAGE", False, ["no_labeled_semantic_events"])
    candidates = {"overall": primary, **by_event_type}
    passing = [
        name
        for name, summary in candidates.items()
        if _passes_semantic_thresholds(
            summary=summary,
            separation=separation,
            min_sample=min_sample,
            min_mean_abret=min_mean_abret,
            min_control_separation=min_control_separation,
            max_top5_abs_contribution=max_top5_abs_contribution,
        )
    ]
    if passing:
        return _verdict("CONTINUE_SEMANTIC_8K", True, [], passing=passing)
    if n < min_sample and not any(int(summary.get("n") or 0) >= min_sample for summary in by_event_type.values()):
        return _verdict("MORE_DATA_REQUIRED", False, ["insufficient_labeled_events"])
    return _verdict("SEMANTIC_8K_KILLED", False, _semantic_failed_gates(primary, separation, min_sample, min_mean_abret, min_control_separation, max_top5_abs_contribution))


def _passes_semantic_thresholds(
    *,
    summary: dict[str, object],
    separation: dict[str, object],
    min_sample: int,
    min_mean_abret: float,
    min_control_separation: float,
    max_top5_abs_contribution: float,
) -> bool:
    return not _semantic_failed_gates(
        summary,
        separation,
        min_sample,
        min_mean_abret,
        min_control_separation,
        max_top5_abs_contribution,
    )


def _semantic_failed_gates(
    summary: dict[str, object],
    separation: dict[str, object],
    min_sample: int,
    min_mean_abret: float,
    min_control_separation: float,
    max_top5_abs_contribution: float,
) -> list[str]:
    failed: list[str] = []
    if int(summary.get("n") or 0) < int(min_sample):
        failed.append("insufficient_labeled_events")
    if _float(summary.get("mean")) is None or float(summary.get("mean") or 0.0) <= float(min_mean_abret):
        failed.append("mean_abret_gate_failed")
    if _float(summary.get("median")) is None or float(summary.get("median") or 0.0) <= 0.0:
        failed.append("median_not_positive")
    if _float(separation.get("mean_difference")) is None or float(separation.get("mean_difference") or 0.0) <= float(min_control_separation):
        failed.append("timestamp_placebo_separation_failed")
    if _float(summary.get("top5_abs_contribution")) is None or float(summary.get("top5_abs_contribution") or 1.0) >= float(max_top5_abs_contribution):
        failed.append("top5_abs_contribution_too_high")
    return failed


def _verdict(
    decision: str,
    move_forward: bool,
    failed_gates: list[str],
    *,
    passing: list[str] | None = None,
) -> dict[str, object]:
    return {
        "decision": decision,
        "move_forward": bool(move_forward),
        "paper_live_allowed": False,
        "paper_live_blocker": "sec_8k_semantic_mvp_not_live_trading_surface",
        "passing_segments": list(passing or []),
        "failed_gates": failed_gates,
    }


def _scaled_gate_scenarios(
    *,
    years: tuple[int, ...],
    primary_items: tuple[str, ...],
    fallback_items: tuple[str, ...],
) -> list[dict[str, object]]:
    scenarios: list[dict[str, object]] = []
    year_window: list[int] = []
    for year in years:
        year_window.append(int(year))
        scenarios.append(
            _scaled_gate_scenario(
                name=f"primary_{min(year_window)}_{max(year_window)}",
                years=tuple(year_window),
                target_items=primary_items,
            )
        )
    all_items = tuple(dict.fromkeys((*primary_items, *fallback_items)))
    if fallback_items:
        scenarios.append(
            _scaled_gate_scenario(
                name=f"fallback_{min(years)}_{max(years)}",
                years=years,
                target_items=all_items,
            )
        )
    return scenarios


def _scaled_gate_scenario(
    *, name: str, years: tuple[int, ...], target_items: tuple[str, ...]
) -> dict[str, object]:
    lower = min(years)
    upper = max(years)
    return {
        "name": name,
        "years": list(years),
        "target_items": list(target_items),
        "accepted_from": f"{lower:04d}-01-01",
        "accepted_to": f"{upper:04d}-12-31",
    }


def _scaled_gate_fallback_verdict(
    *, scenarios: list[dict[str, object]], min_sample: int
) -> dict[str, object]:
    if not scenarios:
        return _verdict("MORE_DATA_REQUIRED", False, ["no_scaled_gate_scenarios"])
    latest = scenarios[-1]
    queue = _dict(latest.get("queue_metadata"))
    target_count = int(queue.get("routed_candidate_count") or 0)
    labelable_count = int(queue.get("labelable_candidate_count") or 0)
    labeled_count = int(latest.get("labeled_count") or 0)
    if target_count >= int(min_sample) and labelable_count < int(min_sample):
        return _verdict(
            "BLOCKED_DATA_COVERAGE",
            False,
            ["insufficient_labelable_market_coverage"],
        )
    if labelable_count >= int(min_sample) and labeled_count < int(min_sample):
        return _verdict(
            "MORE_DATA_REQUIRED",
            False,
            ["insufficient_semantic_event_frequency"],
        )
    return _verdict("MORE_DATA_REQUIRED", False, ["insufficient_target_event_frequency"])


def _classification_verdict(classifications: pd.DataFrame) -> str:
    if classifications.empty:
        return "NO_SNIPPETS"
    failed = int((classifications.get("status") == "FAIL").sum())
    return "PASS" if failed == 0 else "PARTIAL"


def _semantic_candidate_rank(candidate: dict[str, object]) -> tuple[int, int, str, str]:
    snippet_kind_rank = 0 if _text(candidate.get("snippet_kind")) == "item_section" else 1
    warning_count = len(_list(candidate.get("semantic_warnings")))
    return (
        snippet_kind_rank,
        warning_count,
        _text(candidate.get("filename")),
        _text(candidate.get("semantic_snippet_id")),
    )


def _semantic_snippet_rank(snippet: Sec8KSemanticSnippet) -> tuple[int, str, str, str, str]:
    snippet_kind_rank = 0 if snippet.snippet_kind == "item_section" else 1
    return (
        snippet_kind_rank,
        _text(snippet.accepted_at_utc),
        _text(snippet.accession),
        _text(snippet.sec_item_number),
        _text(snippet.filename),
    )


def _semantic_labelability_snippet_rank(
    snippet: Sec8KSemanticSnippet,
) -> tuple[int, int, str, str, str, str]:
    labelability_rank = 0 if snippet.labelability_status == "LABELABLE" else 1
    base = _semantic_snippet_rank(snippet)
    return (labelability_rank, *base)


def _routing_mode(value: str) -> str:
    mode = str(value or "broad").strip().lower()
    if mode not in SEMANTIC_ROUTING_MODES:
        raise ValueError(f"routing_mode must be one of {', '.join(SEMANTIC_ROUTING_MODES)}")
    return mode


def _labelability_mode(value: str) -> str:
    mode = str(value or "all").strip().lower()
    if mode not in SEMANTIC_LABELABILITY_MODES:
        raise ValueError(
            f"labelability_mode must be one of {', '.join(SEMANTIC_LABELABILITY_MODES)}"
        )
    return mode


def _snippet_kind(value: str) -> str:
    kind = str(value or "all").strip().lower()
    if kind not in SEMANTIC_SNIPPET_KINDS:
        raise ValueError(f"snippet_kind must be one of {', '.join(SEMANTIC_SNIPPET_KINDS)}")
    return kind


def _snippet_kind_counts(snippets: list[Sec8KSemanticSnippet]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for snippet in snippets:
        counts[snippet.snippet_kind] = counts.get(snippet.snippet_kind, 0) + 1
    return dict(sorted(counts.items()))


def _target_items(items: Iterable[str] | None) -> tuple[str, ...]:
    selected = tuple(_normalize_sec_item(item) for item in (items or DEFAULT_TARGETED_SEC_ITEMS))
    deduped = tuple(dict.fromkeys(item for item in selected if item))
    return deduped or DEFAULT_TARGETED_SEC_ITEMS


def _target_items_exact(items: Iterable[str]) -> tuple[str, ...]:
    selected = tuple(_normalize_sec_item(item) for item in items)
    return tuple(dict.fromkeys(item for item in selected if item))


def _normalize_sec_item(value: object) -> str:
    text = _text(value)
    if not text:
        return ""
    match = re.search(r"([1-9])\.(\d{1,2})", text)
    if not match:
        return text
    return f"{int(match.group(1))}.{int(match.group(2)):02d}"


def _date_bound(value: object) -> str | None:
    text = _text(value)
    if not text:
        return None
    date_text = _date_text(text)
    if not date_text:
        raise ValueError(f"accepted date bound must be parseable as a date: {text}")
    return date_text


def _date_text(value: object) -> str:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).date().isoformat() if value.tzinfo else value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = _text(value)
    if not text:
        return ""
    if len(text) >= 10 and text[4] == "-" and text[7] == "-":
        year, month, day = text[:4], text[5:7], text[8:10]
        if year.isdigit() and month.isdigit() and day.isdigit():
            return f"{year}-{month}-{day}"
    if len(text) >= 8 and text[:8].isdigit():
        return f"{text[:4]}-{text[4:6]}-{text[6:8]}"
    match = re.search(r"(\d{4})-(\d{2})-(\d{2})", text)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    digits = re.sub(r"\D", "", text)
    if len(digits) >= 8:
        return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"
    return ""


def _candidate_event_date(row: dict[str, object]) -> str:
    for column in (
        "accepted_at_utc",
        "accepted_at",
        "acceptanceDateTime",
        "filing_date",
        "filed_date",
        "filingDate",
    ):
        value = _date_text(row.get(column))
        if value:
            return value
    return ""


def _date_in_range(
    row: dict[str, object],
    *,
    accepted_from: str | None,
    accepted_to: str | None,
) -> bool:
    lower = _date_bound(accepted_from)
    upper = _date_bound(accepted_to)
    if lower is None and upper is None:
        return True
    event_date = _candidate_event_date(row)
    if not event_date:
        return False
    if lower is not None and event_date < lower:
        return False
    if upper is not None and event_date > upper:
        return False
    return True


def _merge_labelability_row(
    row: dict[str, object],
    labelability_by_event_id: dict[str, dict[str, object]],
) -> dict[str, object]:
    labelability = labelability_by_event_id.get(_text(row.get("event_id")), {})
    merged = dict(row)
    merged["labelability_status"] = _text(labelability.get("labelability_status")) or None
    merged["labelability_blockers"] = [
        str(item) for item in _list(labelability.get("labelability_blockers")) if str(item)
    ]
    merged["tradable_at_utc"] = _text(labelability.get("tradable_at_utc") or row.get("tradable_at_utc")) or None
    return merged


def _labelability_blockers(
    *, candidate: dict[str, object], label: dict[str, object]
) -> list[str]:
    status = _text(label.get("label_status"))
    if status == "SKIPPED_INELIGIBLE":
        blockers = _list(candidate.get("exclusion_reasons")) or _list(label.get("label_blockers"))
    else:
        blockers = _list(label.get("label_blockers"))
    if status != "LABELED" and not blockers:
        blockers = [status.lower() if status else "label_not_built"]
    return sorted({str(item) for item in blockers if str(item)})


def _labelability_status(label_status: str, blockers: list[str]) -> str:
    if label_status == "LABELED" and not blockers:
        return "LABELABLE"
    if label_status == "SKIPPED_INELIGIBLE":
        return "SKIPPED_INELIGIBLE"
    return "UNLABELABLE"


def _labelability_summary(frame: pd.DataFrame) -> dict[str, object]:
    return {
        "labelable_candidate_count": int(
            (frame.get("labelability_status") == "LABELABLE").sum()
        )
        if "labelability_status" in frame
        else 0,
        "labelability_status_counts": _value_counts(frame, "labelability_status"),
        "labelability_blocker_counts": _blocker_counts(
            frame.get("labelability_blockers", pd.Series(dtype=object)).tolist()
        ),
    }


def _blocker_counts(values: list[object]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for blockers in values:
        for blocker in _list(blockers):
            text = str(blocker)
            if text:
                counts[text] = counts.get(text, 0) + 1
    return dict(sorted(counts.items()))


def _routed_candidate_rows(
    *,
    candidates: pd.DataFrame,
    routing_mode: str,
    target_items: tuple[str, ...],
    accepted_from: str | None = None,
    accepted_to: str | None = None,
) -> list[dict[str, object]]:
    sort_columns = [
        column
        for column in ("accepted_at_utc", "accession", "sec_item_number")
        if column in candidates
    ]
    frame = candidates.sort_values(sort_columns) if sort_columns else candidates
    rows: list[dict[str, object]] = []
    target_set = set(target_items)
    for row in frame.to_dict("records"):
        if not _date_in_range(row, accepted_from=accepted_from, accepted_to=accepted_to):
            continue
        item_number = _normalize_sec_item(row.get("sec_item_number"))
        if routing_mode == "targeted" and item_number not in target_set:
            continue
        routed = dict(row)
        routed["sec_item_number"] = item_number
        routed["route_family"] = _route_family(item_number, routing_mode=routing_mode)
        routed["route_reason"] = _route_reason(item_number, routing_mode=routing_mode)
        rows.append(routed)
    return rows


def _route_family(item_number: str, *, routing_mode: str) -> str:
    if routing_mode == "targeted":
        return TARGETED_SEC_ITEM_FAMILIES.get(item_number, f"targeted_item_{item_number}")
    return "broad_8k"


def _route_reason(item_number: str, *, routing_mode: str) -> str:
    if routing_mode == "targeted":
        return f"sec_item_{item_number}_targeted_for_semantic_mvp"
    return "broad_header_only_semantic_probe"


def _snippet_from_row(
    *,
    row: dict[str, object],
    accession: str,
    source_hash: str,
    snippet_kind: str,
    document_type: str,
    filename: str | None,
    sec_item_number: str | None,
    route_family: str,
    route_reason: str,
    text: str,
) -> Sec8KSemanticSnippet:
    snippet_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    snippet_id = "sec8k_semantic_" + hashlib.sha256(
        f"{accession}:{snippet_kind}:{sec_item_number}:{document_type}:{filename}:{snippet_hash}".encode("utf-8")
    ).hexdigest()[:20]
    return Sec8KSemanticSnippet(
        snippet_id=snippet_id,
        accession=accession,
        archive_cik=_text(row.get("archive_cik")),
        issuer_cik=_text(row.get("issuer_cik")),
        ticker=_text(row.get("ticker")),
        accepted_at_utc=_text(row.get("accepted_at_utc")) or None,
        filing_date=_text(row.get("filing_date")) or None,
        sec_item_number=sec_item_number or None,
        route_family=route_family,
        route_reason=route_reason,
        snippet_kind=snippet_kind,
        document_type=document_type,
        filename=filename,
        source_hash=source_hash,
        snippet_hash=snippet_hash,
        labelability_status=_text(row.get("labelability_status")) or None,
        labelability_blockers=[
            str(item) for item in _list(row.get("labelability_blockers")) if str(item)
        ],
        text=text,
    )


def _item_sections_from_complete_text(text: str, *, max_chars: int) -> dict[str, str]:
    primary = _primary_document_text(text)
    clean = _clean_text(primary)
    matches = list(re.finditer(r"\bItem\s+([1-9])\.(\d{2})\b", clean, flags=re.IGNORECASE))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        item_number = f"{int(match.group(1))}.{int(match.group(2)):02d}"
        end = matches[index + 1].start() if index + 1 < len(matches) else len(clean)
        section = clean[match.start() : end].strip()
        if section:
            sections.setdefault(item_number, section[:max_chars])
    return sections


def _exhibit_snippets_from_complete_text(text: str, *, max_chars: int) -> list[dict[str, str | None]]:
    snippets: list[dict[str, str | None]] = []
    for block in _document_blocks(text):
        doc_type = _text(sgml_tag(block, "TYPE")).upper()
        if not doc_type.startswith("EX-") or _machine_readable_exhibit_type(doc_type):
            continue
        body = _clean_text(_document_text(block))
        if not body:
            continue
        snippets.append(
            {
                "document_type": doc_type,
                "filename": _text(sgml_tag(block, "FILENAME")) or None,
                "text": body[:max_chars],
            }
        )
    return snippets


def _primary_document_text(text: str) -> str:
    for block in _document_blocks(text):
        if _text(sgml_tag(block, "TYPE")).upper() == "8-K":
            return _document_text(block)
    return text


def _document_text(block: str) -> str:
    match = re.search(r"<TEXT>(.*?)</TEXT>", block, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1) if match else block


def _document_blocks(text: str) -> list[str]:
    return re.findall(r"<DOCUMENT>(.*?)</DOCUMENT>", text, flags=re.IGNORECASE | re.DOTALL)


def _clean_text(text: str) -> str:
    without_tags = re.sub(r"<[^>]+>", " ", text)
    without_entities = html.unescape(
        without_tags.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&quot;", '"')
        .replace("&#160;", " ")
    )
    return re.sub(r"\s+", " ", without_entities).strip()


def _machine_readable_exhibit_type(document_type: str) -> bool:
    upper = document_type.upper()
    return upper.startswith("EX-101") or upper == "EX-104"


def _classify_batch_with_isolation(
    *,
    batch: list[Sec8KSemanticSnippet],
    client: SecEventSemanticBatchClient,
    batch_index: int,
    recovery_warnings: list[str],
) -> list[dict[str, object]]:
    if len(batch) == 1 and callable(getattr(client, "classify", None)):
        return [
            _classify_single_snippet(
                snippet=batch[0],
                client=client,
                batch_index=batch_index,
                recovery_warnings=recovery_warnings,
            )
        ]
    try:
        result = client.classify_batch([(snippet.snippet_id, snippet.text) for snippet in batch])
    except Exception as exc:
        error = f"batch_exception:{type(exc).__name__}:{exc}"
        if len(batch) <= 1 or not semantic_batch_exception_is_isolatable(exc):
            return [
                _classification_failure_row(snippet, batch_index, error)
                for snippet in batch
            ]
        midpoint = len(batch) // 2
        warning = _trim_warning(f"batch_retry_after:{error}")
        return [
            *_classify_batch_with_isolation(
                batch=batch[:midpoint],
                client=client,
                batch_index=batch_index,
                recovery_warnings=[*recovery_warnings, warning],
            ),
            *_classify_batch_with_isolation(
                batch=batch[midpoint:],
                client=client,
                batch_index=batch_index,
                recovery_warnings=[*recovery_warnings, warning],
            ),
        ]
    rows: list[dict[str, object]] = []
    for snippet in batch:
        response = result.results.get(snippet.snippet_id)
        if response is None:
            rows.append(_classification_failure_row(snippet, batch_index, "missing_model_result"))
            continue
        payload, normalization_warnings = normalize_semantic_payload(response.payload)
        errors = validate_semantic_classification(excerpt=snippet.text, payload=payload)
        blocking = blocking_semantic_validation_errors(errors)
        blocking_set = set(blocking)
        warnings = [
            *normalization_warnings,
            *[error for error in errors if error not in blocking_set],
        ]
        rows.append(
            {
                **_snippet_context(snippet),
                "batch_index": batch_index,
                "status": "PASS" if not blocking else "FAIL",
                "errors": blocking,
                "warnings": [*recovery_warnings, *warnings],
                "event_type": payload.get("event_type"),
                "certainty": payload.get("certainty"),
                "is_material_candidate": payload.get("is_material_candidate"),
                "fields": payload.get("fields") or {},
                "evidence": payload.get("evidence") or [],
                "exclusion_reasons": payload.get("exclusion_reasons") or [],
                "short_rationale": payload.get("short_rationale"),
                "content_source": response.content_source,
                "elapsed_ms": round(float(response.elapsed_ms), 3),
                "model": response.model,
            }
        )
    return rows


def _classify_single_snippet(
    *,
    snippet: Sec8KSemanticSnippet,
    client: object,
    batch_index: int,
    recovery_warnings: list[str],
) -> dict[str, object]:
    try:
        response = client.classify(snippet.text)  # type: ignore[attr-defined]
    except Exception as exc:
        return _classification_failure_row(
            snippet,
            batch_index,
            f"single_exception:{type(exc).__name__}:{exc}",
        )
    payload, normalization_warnings = normalize_semantic_payload(response.payload)
    errors = validate_semantic_classification(excerpt=snippet.text, payload=payload)
    blocking = blocking_semantic_validation_errors(errors)
    blocking_set = set(blocking)
    warnings = [
        *normalization_warnings,
        *[error for error in errors if error not in blocking_set],
    ]
    return {
        **_snippet_context(snippet),
        "batch_index": batch_index,
        "status": "PASS" if not blocking else "FAIL",
        "errors": blocking,
        "warnings": [*recovery_warnings, *warnings],
        "event_type": payload.get("event_type"),
        "certainty": payload.get("certainty"),
        "is_material_candidate": payload.get("is_material_candidate"),
        "fields": payload.get("fields") or {},
        "evidence": payload.get("evidence") or [],
        "exclusion_reasons": payload.get("exclusion_reasons") or [],
        "short_rationale": payload.get("short_rationale"),
        "content_source": response.content_source,
        "elapsed_ms": round(float(response.elapsed_ms), 3),
        "model": response.model,
    }


def _trim_warning(value: str, *, max_chars: int = 240) -> str:
    return value if len(value) <= max_chars else value[: max_chars - 3] + "..."


def _semantic_checkpoint_path(*, root: Path | None, checkpoint_path: Path | None) -> Path:
    if checkpoint_path is not None:
        return Path(checkpoint_path).expanduser()
    if root is None:
        return Path("/tmp/trademl_sec_event_semantic_classification_checkpoint.parquet")
    return (
        Path(root).expanduser()
        / "data"
        / "curated"
        / "events"
        / "sec_event_semantic_classification_checkpoints"
        / "data.parquet"
    )


def _read_semantic_checkpoint(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except (OSError, ValueError, pd.errors.ParserError):
        return pd.DataFrame()


def _reusable_checkpoint_rows(
    *,
    snippets: list[Sec8KSemanticSnippet],
    checkpoint_rows: pd.DataFrame,
    model: str,
) -> pd.DataFrame:
    if checkpoint_rows.empty:
        return pd.DataFrame()
    required = {"snippet_id", "snippet_hash", "source_hash", "status", "model", "schema_version"}
    if not required.issubset(set(checkpoint_rows.columns)):
        return pd.DataFrame()
    snippets_by_id = {snippet.snippet_id: snippet for snippet in snippets}
    reusable_rows: list[dict[str, object]] = []
    for row in checkpoint_rows.sort_values("snippet_id").to_dict("records"):
        snippet = snippets_by_id.get(_text(row.get("snippet_id")))
        if snippet is None:
            continue
        if _text(row.get("status")) != "PASS":
            continue
        if _text(row.get("model")) != model:
            continue
        if _text(row.get("schema_version")) != SEC_EVENT_SEMANTIC_SCHEMA_VERSION:
            continue
        if _text(row.get("snippet_hash")) != snippet.snippet_hash:
            continue
        if _text(row.get("source_hash")) != snippet.source_hash:
            continue
        reusable = dict(row)
        warnings = _list(reusable.get("warnings"))
        reusable["warnings"] = [*warnings, "reused_from_semantic_checkpoint"]
        reusable_rows.append(reusable)
    return pd.DataFrame(reusable_rows)


def _max_possible_promotable_count(
    *, rows: list[dict[str, object]], pending: list[Sec8KSemanticSnippet]
) -> int:
    promotable = {
        _text(row.get("snippet_id"))
        for row in rows
        if _text(row.get("status")) == "PASS"
        and _text(row.get("event_type")) in PROMOTABLE_EVENT_TYPES
    }
    pending_ids = {snippet.snippet_id for snippet in pending}
    return len(promotable | pending_ids)


def _promotable_count(rows: list[dict[str, object]]) -> int:
    return sum(
        1
        for row in rows
        if _text(row.get("status")) == "PASS"
        and _text(row.get("event_type")) in PROMOTABLE_EVENT_TYPES
    )


def _annotate_classification_row(row: dict[str, object]) -> dict[str, object]:
    annotated = dict(row)
    annotated["schema_version"] = SEC_EVENT_SEMANTIC_SCHEMA_VERSION
    annotated["classified_at_utc"] = _now_iso()
    return annotated


def _write_semantic_checkpoint(*, path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if path.exists():
        try:
            existing = pd.read_parquet(path)
            frame = pd.concat([existing, frame], ignore_index=True)
        except (OSError, ValueError, pd.errors.ParserError):
            pass
    if not frame.empty:
        frame = (
            frame.sort_values(["snippet_id", "classified_at_utc"])
            .drop_duplicates("snippet_id", keep="last")
            .reset_index(drop=True)
        )
    frame.to_parquet(path, index=False)


def _value_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if frame.empty or column not in frame:
        return {}
    counts = frame[column].fillna("").astype(str).value_counts().sort_index()
    return {str(key): int(value) for key, value in counts.items()}


def _complete_text_lookup(*, root: Path, accessions: set[str] | None = None) -> dict[str, str]:
    lookup: dict[str, str] = {}
    archive_root = root / "data" / "raw" / "sec" / "archives"
    if not archive_root.exists():
        return lookup
    accession_filter = set(accessions or ())
    for path in archive_root.glob("archive_cik=*/accession=*/complete.txt"):
        accession = path.parent.name.removeprefix("accession=")
        if accession_filter and accession not in accession_filter:
            continue
        try:
            lookup[accession] = path.read_text(
                encoding="utf-8",
                errors="replace",
            )
        except OSError:
            continue
    return lookup


def _archive_cik_lookup(*, root: Path, accessions: set[str] | None = None) -> dict[str, str]:
    lookup: dict[str, str] = {}
    archive_root = root / "data" / "raw" / "sec" / "archives"
    if not archive_root.exists():
        return lookup
    accession_filter = set(accessions or ())
    for path in archive_root.glob("archive_cik=*/accession=*/complete.txt"):
        accession = path.parent.name.removeprefix("accession=")
        if accession_filter and accession not in accession_filter:
            continue
        lookup[accession] = path.parent.parent.name.removeprefix("archive_cik=")
    return lookup


def _classification_failure_row(
    snippet: Sec8KSemanticSnippet, batch_index: int, error: str
) -> dict[str, object]:
    return {
        **_snippet_context(snippet),
        "batch_index": batch_index,
        "status": "FAIL",
        "errors": [error],
        "warnings": [],
        "event_type": None,
        "certainty": None,
        "is_material_candidate": None,
        "fields": {key: [] for key in FIELD_KEYS},
        "evidence": [],
        "exclusion_reasons": [],
        "short_rationale": None,
        "content_source": None,
        "elapsed_ms": None,
        "model": None,
    }


def _snippet_context(snippet: Sec8KSemanticSnippet) -> dict[str, object]:
    item = snippet.to_dict()
    item.pop("text", None)
    return item


def _header_baseline_summary(packet: dict[str, object] | None) -> dict[str, object] | None:
    if not packet:
        return None
    return {
        "event_class": packet.get("event_class"),
        "primary": packet.get("primary"),
        "verdict": packet.get("verdict"),
        "checked_at": packet.get("checked_at"),
    }


def _render_study_markdown(packet: dict[str, object]) -> str:
    verdict = dict(packet.get("verdict") or {})
    primary = dict(packet.get("primary") or {})
    lines = [
        "# SEC 8-K Semantic Event Study",
        "",
        f"- Decision: `{verdict.get('decision')}`",
        f"- Move forward: `{verdict.get('move_forward')}`",
        f"- Paper/live allowed: `{verdict.get('paper_live_allowed')}`",
        f"- Primary n: `{primary.get('n')}`",
        f"- Primary mean: `{primary.get('mean')}`",
        f"- Primary median: `{primary.get('median')}`",
        f"- Failed gates: `{', '.join(str(item) for item in verdict.get('failed_gates') or []) or 'none'}`",
        "",
        "## Event Types",
        "",
    ]
    for event_type, summary in sorted(dict(packet.get("by_event_type") or {}).items()):
        lines.append(
            f"- `{event_type}`: n={summary.get('n')}, mean={summary.get('mean')}, median={summary.get('median')}"
        )
    return "\n".join(lines) + "\n"


def _read_optional_json(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _list(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist") and not isinstance(value, dict):
        converted = value.tolist()  # type: ignore[no-untyped-call]
        if converted is value:
            return [value]
        return _list(converted)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                return [text]
            return _list(parsed)
        return [text]
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass
    return [value]


def _dict(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("{") and text.endswith("}"):
            try:
                parsed = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                return {}
            return parsed if isinstance(parsed, dict) else {}
    return {}


def _first(values: list[object]) -> object | None:
    return values[0] if values else None


def _text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _float(value: object) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
