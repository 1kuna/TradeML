"""Deterministic SEC 8-K item-event scaffold and event-study helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import dataclass
import ast
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Iterable

import exchange_calendars as xcals
import pandas as pd

from trademl.events.form4 import normalize_sec_accepted_at, sgml_tag
from trademl.events.form4_event_study import (
    _json_safe,
    _labeled,
    _separation,
    _summary_by_metric,
)
from trademl.events.form4_labels import (
    DEFAULT_HORIZONS,
    Form4LabelConfig,
    build_form4_event_labels_with_market_sources,
)


SEC8K_CANDIDATE_VERSION = "sec8k_item_event_v1"
SEC8K_STUDY_VERSION = "sec8k_event_study_v1"
SEC8K_DECISION_VERSION = "sec8k_decision_v1"
PRIMARY_HORIZON = 5
PRIMARY_METRIC = f"abret_{PRIMARY_HORIZON}d_net"
ITEM_FAMILIES = {
    "1.01": "8K_ITEM_1_01_MATERIAL_AGREEMENT",
    "2.02": "8K_ITEM_2_02_RESULTS_OPERATIONS",
    "2.04": "8K_ITEM_2_04_DEFAULT_COVENANT_STRESS",
    "2.05": "8K_ITEM_2_05_EXIT_DISPOSAL_COSTS",
    "3.02": "8K_ITEM_3_02_UNREGISTERED_SALE_FINANCING",
    "4.01": "8K_ITEM_4_01_AUDITOR_CHANGE",
    "7.01": "8K_ITEM_7_01_REG_FD",
    "8.01": "8K_ITEM_8_01_OTHER",
}
PRIMARY_ALPHA_FAMILIES = (
    "8K_ITEM_1_01_MATERIAL_AGREEMENT",
    "8K_ITEM_2_04_DEFAULT_COVENANT_STRESS",
    "8K_ITEM_2_05_EXIT_DISPOSAL_COSTS",
    "8K_ITEM_3_02_UNREGISTERED_SALE_FINANCING",
    "8K_ITEM_4_01_AUDITOR_CHANGE",
    "8K_ITEM_7_01_REG_FD",
    "8K_ITEM_8_01_OTHER",
)


@dataclass(slots=True, frozen=True)
class Sec8KManifestRow:
    """One SEC index-derived 8-K accession manifest row."""

    archive_cik: str
    form: str
    filed_date: str
    index_filename: str
    accession: str
    accession_no_dashes: str
    discovery_source: str
    index_year: int
    index_quarter: int
    index_file_hash: str
    index_crawled_at: str

    def to_dict(self) -> dict[str, object]:
        """Return a parquet/JSON-safe representation."""
        return {
            "archive_cik": self.archive_cik,
            "form": self.form,
            "filed_date": self.filed_date,
            "index_filename": self.index_filename,
            "accession": self.accession,
            "accession_no_dashes": self.accession_no_dashes,
            "discovery_source": self.discovery_source,
            "index_year": self.index_year,
            "index_quarter": self.index_quarter,
            "index_file_hash": self.index_file_hash,
            "index_crawled_at": self.index_crawled_at,
        }


@dataclass(slots=True, frozen=True)
class Sec8KDocumentInventory:
    """One SGML document block from a SEC 8-K complete submission."""

    document_type: str
    filename: str | None
    description: str | None
    sha256: str
    is_exhibit: bool

    def to_dict(self) -> dict[str, object]:
        """Return a JSON/parquet-safe representation."""
        return {
            "document_type": self.document_type,
            "filename": self.filename,
            "description": self.description,
            "sha256": self.sha256,
            "is_exhibit": self.is_exhibit,
        }


@dataclass(slots=True, frozen=True)
class Sec8KCompleteTextParse:
    """Parsed source metadata from a SEC 8-K complete submission text."""

    accepted_at_raw: str | None
    item_numbers: tuple[str, ...]
    documents: tuple[Sec8KDocumentInventory, ...]
    source_hash: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe representation."""
        return {
            "accepted_at_raw": self.accepted_at_raw,
            "item_numbers": list(self.item_numbers),
            "documents": [document.to_dict() for document in self.documents],
            "source_hash": self.source_hash,
        }


def run_sec8k_candidate_curation(
    *,
    data_root: Path,
    filings_path: Path | None = None,
) -> dict[str, object]:
    """Build SEC 8-K item candidates from existing filing-index artifacts."""
    root = Path(data_root).expanduser()
    path = filings_path or _default_filing_index_path(root)
    if path is None:
        raise FileNotFoundError(
            "missing SEC filing-index artifact; expected data/reference/sec_filing_index.parquet "
            "or data/reference/sec_filings.parquet"
        )
    filings = pd.read_parquet(path)
    candidates = build_sec8k_item_candidates(
        filings=filings,
        source_path=path,
    )
    return write_sec8k_item_candidates(root=root, candidates=candidates)


def run_sec8k_event_study(
    *,
    data_root: Path,
    primary_horizon: int = PRIMARY_HORIZON,
    horizons: Iterable[int] | None = None,
    round_trip_cost_bps: float = 50.0,
    market_data_roots: Iterable[Path] | None = None,
    source_contract_path: Path | None = None,
) -> dict[str, object]:
    """Build SEC 8-K labels, controls, and a first event-study packet."""
    root = Path(data_root).expanduser()
    candidates_path = (
        root / "data" / "curated" / "events" / "sec_8k_item_events" / "data.parquet"
    )
    if not candidates_path.exists():
        raise FileNotFoundError(f"missing SEC 8-K candidate parquet: {candidates_path}")
    candidates = pd.read_parquet(candidates_path)
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
    labels_payload = write_sec8k_labels(
        root=root,
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
    placebo_payload = write_sec8k_placebo_labels(
        root=root,
        labels=placebo_labels,
        config=config,
        source_metadata=placebo_source_metadata,
    )
    packet = build_sec8k_event_study_packet(
        labels=labels,
        placebo_labels=placebo_labels,
        labels_payload=labels_payload,
        placebo_payload=placebo_payload,
        primary_horizon=int(primary_horizon),
        round_trip_cost_bps=float(round_trip_cost_bps),
    )
    return write_sec8k_event_study_packet(root=root, packet=packet)


def run_sec8k_research_decision(
    *,
    data_root: Path,
    min_labeled_events: int = 300,
    min_family_events: int = 100,
    min_mean_abret: float = 0.005,
    min_control_separation: float = 0.0075,
    max_top5_abs_contribution: float = 0.35,
) -> dict[str, object]:
    """Write the continue/kill decision for broad deterministic SEC 8-K item events."""
    root = Path(data_root).expanduser()
    study_path = (
        root
        / "control"
        / "cluster"
        / "state"
        / "research"
        / "sec_8k_event_study"
        / "latest.json"
    )
    if not study_path.exists():
        raise FileNotFoundError(f"missing SEC 8-K event-study packet: {study_path}")
    packet = json.loads(study_path.read_text(encoding="utf-8"))
    ingest = _read_optional_json(
        root / "control" / "cluster" / "state" / "research" / "sec8k_ingest" / "latest.json"
    )
    backfill = _read_optional_json(
        root
        / "control"
        / "cluster"
        / "state"
        / "research"
        / "sec8k_market_backfill"
        / "latest.json"
    )
    decision = build_sec8k_research_decision(
        packet=packet,
        ingest=ingest,
        backfill=backfill,
        min_labeled_events=min_labeled_events,
        min_family_events=min_family_events,
        min_mean_abret=min_mean_abret,
        min_control_separation=min_control_separation,
        max_top5_abs_contribution=max_top5_abs_contribution,
        packet_path=study_path,
    )
    return write_sec8k_research_decision(root=root, decision=decision)


def parse_sec8k_index_manifest(
    text: str,
    *,
    index_year: int,
    index_quarter: int,
    index_crawled_at: str,
    discovery_source: str = "sec_full_index",
) -> list[Sec8KManifestRow]:
    """Parse SEC full-index text into exact Form 8-K manifest rows."""
    index_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    rows: list[Sec8KManifestRow] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("-") or line.lower().startswith("form type"):
            continue
        if "|" in line:
            parts = [part.strip() for part in line.split("|")]
            if len(parts) != 5 or parts[0].lower() == "cik":
                continue
            _, _company_name, form, filed_date, filename = parts
        else:
            match = re.match(
                r"^(?P<form>\S+)\s+(?P<company>.+?)\s{2,}"
                r"(?P<cik>\d{1,10})\s+"
                r"(?P<filed_date>\d{4}-\d{2}-\d{2})\s+"
                r"(?P<filename>edgar/data/\d+/.+?\.txt)\s*$",
                line,
            )
            if match is None:
                continue
            form = match.group("form")
            filed_date = match.group("filed_date")
            filename = match.group("filename")
        if form != "8-K":
            continue
        match = re.search(r"edgar/data/(\d+)/(?:\d+/)?([^/]+)\.txt$", filename)
        if match is None:
            continue
        archive_cik = match.group(1)
        accession = match.group(2)
        rows.append(
            Sec8KManifestRow(
                archive_cik=archive_cik,
                form=form,
                filed_date=filed_date,
                index_filename=filename,
                accession=accession,
                accession_no_dashes=accession.replace("-", ""),
                discovery_source=discovery_source,
                index_year=index_year,
                index_quarter=index_quarter,
                index_file_hash=index_hash,
                index_crawled_at=index_crawled_at,
            )
        )
    return rows


def parse_sec8k_complete_text(text: str) -> Sec8KCompleteTextParse:
    """Parse item headings and exhibit inventory from a complete 8-K SGML text."""
    source_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    accepted = sgml_tag(text, "ACCEPTANCE-DATETIME")
    document_blocks = _document_blocks(text)
    primary_text = _primary_document_text_from_blocks(document_blocks, fallback=text)
    documents: list[Sec8KDocumentInventory] = []
    for block in document_blocks:
        doc_type = (sgml_tag(block, "TYPE") or "").strip().upper()
        filename = _optional_text(sgml_tag(block, "FILENAME"))
        description = _optional_text(sgml_tag(block, "DESCRIPTION"))
        if not doc_type:
            continue
        documents.append(
            Sec8KDocumentInventory(
                document_type=doc_type,
                filename=filename,
                description=description,
                sha256=hashlib.sha256(block.encode("utf-8")).hexdigest(),
                is_exhibit=doc_type.startswith("EX-"),
            )
        )
    return Sec8KCompleteTextParse(
        accepted_at_raw=accepted,
        item_numbers=tuple(sorted(_item_numbers_from_text(primary_text))),
        documents=tuple(documents),
        source_hash=source_hash,
    )


def build_sec8k_item_candidates(
    *,
    filings: pd.DataFrame,
    complete_text_by_accession: dict[str, str] | None = None,
    source_path: Path | None = None,
) -> pd.DataFrame:
    """Build one deterministic candidate row per recognized SEC 8-K item family."""
    complete_text_by_accession = complete_text_by_accession or {}
    rows: list[dict[str, object]] = []
    if filings.empty:
        return pd.DataFrame()
    for filing in filings.sort_values(_sort_columns(filings)).to_dict("records"):
        if str(filing.get("form") or "").upper() != "8-K":
            continue
        accession = _text(
            filing.get("accessionNumber")
            or filing.get("accession")
            or filing.get("accession_number")
        )
        if not accession:
            continue
        complete_text = complete_text_by_accession.get(accession) or complete_text_by_accession.get(
            accession.replace("-", "")
        )
        parsed = parse_sec8k_complete_text(complete_text) if complete_text else None
        source_items = _items_from_filing(filing)
        if parsed is not None:
            source_items = tuple(sorted(set(source_items) | set(parsed.item_numbers)))
        recognized = [item for item in source_items if item in ITEM_FAMILIES]
        item_numbers = recognized or ["UNCLASSIFIED"]
        for item_number in item_numbers:
            rows.append(
                _candidate_row(
                    filing=filing,
                    accession=accession,
                    item_number=item_number,
                    parsed=parsed,
                    source_path=source_path,
                )
            )
    return pd.DataFrame(rows).sort_values("event_id").reset_index(drop=True)


def write_sec8k_item_candidates(
    *, root: Path, candidates: pd.DataFrame
) -> dict[str, object]:
    """Write SEC 8-K item candidates and curation report."""
    root = Path(root)
    target = root / "data" / "curated" / "events" / "sec_8k_item_events"
    target.mkdir(parents=True, exist_ok=True)
    data_path = target / "data.parquet"
    candidates.to_parquet(data_path, index=False)
    report = summarize_sec8k_candidates(candidates=candidates)
    report["artifact"] = str(data_path)
    report_target = root / "control" / "cluster" / "state" / "research" / "sec_8k_item_events"
    history = report_target / "history"
    history.mkdir(parents=True, exist_ok=True)
    latest = report_target / "latest.json"
    latest.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    history_path = history / f"{report['checked_at'].replace(':', '').replace('+', '_')}.json"
    history_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return {"events_path": str(data_path), "report_path": str(latest), "report": report}


def summarize_sec8k_candidates(*, candidates: pd.DataFrame) -> dict[str, object]:
    """Summarize SEC 8-K candidate coverage and exclusions."""
    if candidates.empty:
        family_counts: dict[str, int] = {}
        reason_counts: dict[str, int] = {}
    else:
        family_counts = {
            str(key): int(value)
            for key, value in candidates["event_type"].value_counts().sort_index().items()
        }
        reason_counts = {}
        for value in candidates.get("exclusion_reasons", pd.Series(dtype=object)).tolist():
            for reason in _list(value):
                reason_counts[str(reason)] = reason_counts.get(str(reason), 0) + 1
    return {
        "version": SEC8K_CANDIDATE_VERSION,
        "checked_at": _now_iso(),
        "candidate_count": int(len(candidates)),
        "eligible_count": int(candidates.get("eligibility_pass", pd.Series(dtype=bool)).sum())
        if not candidates.empty
        else 0,
        "family_counts": family_counts,
        "exclusion_reason_counts": dict(sorted(reason_counts.items())),
    }


def write_sec8k_labels(
    *,
    root: Path,
    labels: pd.DataFrame,
    config: Form4LabelConfig,
    source_metadata: dict[str, object],
) -> dict[str, object]:
    """Write SEC 8-K event labels and compact coverage metadata."""
    target = Path(root) / "data" / "curated" / "events" / "sec_8k_item_labels"
    return _write_label_artifact(
        target=target,
        state_target=Path(root) / "control" / "cluster" / "state" / "research" / "sec_8k_item_labels",
        labels=labels,
        config=config,
        source_metadata=source_metadata,
    )


def write_sec8k_placebo_labels(
    *,
    root: Path,
    labels: pd.DataFrame,
    config: Form4LabelConfig,
    source_metadata: dict[str, object],
) -> dict[str, object]:
    """Write SEC 8-K timestamp-placebo labels."""
    target = Path(root) / "data" / "curated" / "events" / "sec_8k_timestamp_placebo_labels"
    return _write_label_artifact(
        target=target,
        state_target=Path(root)
        / "control"
        / "cluster"
        / "state"
        / "research"
        / "sec_8k_timestamp_placebo_labels",
        labels=labels,
        config=config,
        source_metadata=source_metadata,
    )


def build_sec8k_event_study_packet(
    *,
    labels: pd.DataFrame,
    placebo_labels: pd.DataFrame,
    labels_payload: dict[str, object],
    placebo_payload: dict[str, object],
    primary_horizon: int,
    round_trip_cost_bps: float,
) -> dict[str, object]:
    """Build a first SEC 8-K item-event study packet."""
    metric = f"abret_{primary_horizon}d_net"
    by_family: dict[str, object] = {}
    if not labels.empty and "event_type" in labels:
        for family, frame in labels.groupby("event_type", sort=True):
            by_family[str(family)] = _summary_by_metric(
                _labeled(frame, metric=metric),
                metric=metric,
                bootstrap_iterations=1000,
                bootstrap_seed=41,
            )
    primary = _summary_by_metric(
        _labeled(labels, metric=metric),
        metric=metric,
        bootstrap_iterations=1000,
        bootstrap_seed=43,
    )
    placebo = _summary_by_metric(
        _labeled(placebo_labels, metric=metric),
        metric=metric,
        bootstrap_iterations=1000,
        bootstrap_seed=47,
    )
    results_ops = by_family.get("8K_ITEM_2_02_RESULTS_OPERATIONS", {})
    controls = {
        "timestamp_placebo": placebo,
        "item_family_results_operations": results_ops,
    }
    separation = {
        name: _separation(primary, summary if isinstance(summary, dict) else {})
        for name, summary in controls.items()
    }
    decision = "BLOCKED_DATA_COVERAGE" if int(primary.get("n") or 0) == 0 else "DIAGNOSTIC_ONLY"
    return _json_safe(
        {
            "version": SEC8K_STUDY_VERSION,
            "checked_at": _now_iso(),
            "event_class": "SEC_8K_ITEM_EVENT",
            "primary_horizon": int(primary_horizon),
            "primary_metric": metric,
            "round_trip_cost_bps": float(round_trip_cost_bps),
            "primary": primary,
            "by_item_family": by_family,
            "negative_controls": controls,
            "negative_control_separation": separation,
            "label_artifacts": labels_payload,
            "timestamp_placebo_artifacts": placebo_payload,
            "verdict": {
                "decision": decision,
                "status": "PENDING_RESEARCH" if decision == "DIAGNOSTIC_ONLY" else "BLOCKED",
                "paper_live_allowed": False,
                "paper_live_blocker": "sec_8k_mvp_is_diagnostic_only",
            },
        }
    )


def write_sec8k_event_study_packet(
    *, root: Path, packet: dict[str, object]
) -> dict[str, object]:
    """Write SEC 8-K event-study JSON and Markdown artifacts."""
    root = Path(root)
    target = root / "control" / "cluster" / "state" / "research" / "sec_8k_event_study"
    history = target / "history"
    history.mkdir(parents=True, exist_ok=True)
    latest = target / "latest.json"
    latest.write_text(json.dumps(packet, indent=2, sort_keys=True), encoding="utf-8")
    history_path = history / f"{packet['checked_at'].replace(':', '').replace('+', '_')}.json"
    history_path.write_text(json.dumps(packet, indent=2, sort_keys=True), encoding="utf-8")
    report_root = root / "reports" / "research" / "sec_8k_event_study"
    report_root.mkdir(parents=True, exist_ok=True)
    report_path = report_root / "latest.md"
    report_path.write_text(_render_study_markdown(packet), encoding="utf-8")
    return {
        "packet_path": str(latest),
        "history_path": str(history_path),
        "report_path": str(report_path),
        "packet": packet,
    }


def build_sec8k_research_decision(
    *,
    packet: dict[str, object],
    ingest: dict[str, object] | None = None,
    backfill: dict[str, object] | None = None,
    min_labeled_events: int = 300,
    min_family_events: int = 100,
    min_mean_abret: float = 0.005,
    min_control_separation: float = 0.0075,
    max_top5_abs_contribution: float = 0.35,
    packet_path: Path | None = None,
) -> dict[str, object]:
    """Apply deterministic continue/kill gates to the latest SEC 8-K study packet."""
    primary = dict(packet.get("primary") or {})
    controls = dict(packet.get("negative_control_separation") or {})
    timestamp_separation = dict(controls.get("timestamp_placebo") or {})
    primary_n = int(primary.get("n") or 0)
    mean_difference = _optional_float(timestamp_separation.get("mean_difference"))
    study_failed_gates: list[str] = []
    if primary_n < int(min_labeled_events):
        study_failed_gates.append("insufficient_labeled_events")
    if _optional_float(primary.get("mean")) is None or float(primary.get("mean") or 0.0) <= 0:
        study_failed_gates.append("primary_mean_not_positive")
    if _optional_float(primary.get("median")) is None or float(primary.get("median") or 0.0) <= 0:
        study_failed_gates.append("primary_median_not_positive")
    if _optional_float(primary.get("hit_rate")) is None or float(primary.get("hit_rate") or 0.0) <= 0.5:
        study_failed_gates.append("primary_hit_rate_not_above_half")
    if mean_difference is None or mean_difference < float(min_control_separation):
        study_failed_gates.append("timestamp_placebo_separation_failed")

    family_results: dict[str, dict[str, object]] = {}
    passing_families: list[str] = []
    for family in PRIMARY_ALPHA_FAMILIES:
        summary = dict(dict(packet.get("by_item_family") or {}).get(family) or {})
        failed = _family_failed_gates(
            summary=summary,
            min_family_events=min_family_events,
            min_mean_abret=min_mean_abret,
            max_top5_abs_contribution=max_top5_abs_contribution,
        )
        if not failed and not study_failed_gates:
            passing_families.append(family)
        family_results[family] = {
            "n": int(summary.get("n") or 0),
            "mean": summary.get("mean"),
            "median": summary.get("median"),
            "hit_rate": summary.get("hit_rate"),
            "top5_abs_contribution": summary.get("top5_abs_contribution"),
            "bootstrap_mean_ci_95": summary.get("bootstrap_mean_ci_95"),
            "failed_gates": failed,
            "passes": not failed and not study_failed_gates,
        }

    move_forward = bool(passing_families)
    decision = (
        "CONTINUE_NARROW_SEC8K_ITEM_FAMILY"
        if move_forward
        else "BROAD_SEC8K_ITEM_FAMILIES_KILLED"
    )
    return _json_safe(
        {
            "version": SEC8K_DECISION_VERSION,
            "checked_at": _now_iso(),
            "event_class": "SEC_8K_ITEM_EVENT",
            "decision": decision,
            "move_forward": move_forward,
            "paper_live_allowed": False,
            "paper_live_blocker": "sec_8k_decision_not_live_trading_surface",
            "passing_families": passing_families,
            "study_failed_gates": study_failed_gates,
            "family_results": family_results,
            "thresholds": {
                "min_labeled_events": int(min_labeled_events),
                "min_family_events": int(min_family_events),
                "min_mean_abret": float(min_mean_abret),
                "min_control_separation": float(min_control_separation),
                "max_top5_abs_contribution": float(max_top5_abs_contribution),
            },
            "evidence": {
                "packet_path": str(packet_path) if packet_path else None,
                "ingest_verdict": ingest.get("verdict") if ingest else None,
                "manifest_count": ingest.get("manifest_count") if ingest else None,
                "parsed_count": ingest.get("parsed_count") if ingest else None,
                "failed_count": ingest.get("failed_count") if ingest else None,
                "candidate_count": (
                    dict(dict(ingest.get("candidate_artifacts") or {}).get("report") or {}).get(
                        "candidate_count"
                    )
                    if ingest
                    else None
                ),
                "eligible_count": (
                    dict(dict(ingest.get("candidate_artifacts") or {}).get("report") or {}).get(
                        "eligible_count"
                    )
                    if ingest
                    else None
                ),
                "backfill_verdict": backfill.get("verdict") if backfill else None,
                "eligible_primary_candidate_count": (
                    backfill.get("eligible_primary_candidate_count") if backfill else None
                ),
                "minute_rows": backfill.get("minute_rows") if backfill else None,
                "daily_rows": backfill.get("daily_rows") if backfill else None,
                "empty_minute_count": len(backfill.get("empty_minute") or [])
                if backfill
                else None,
                "empty_daily_count": len(backfill.get("empty_daily_symbols") or [])
                if backfill
                else None,
                "primary": primary,
                "timestamp_placebo_separation": timestamp_separation,
            },
            "next_action": (
                "Continue only the named passing family with a narrower hypothesis."
                if move_forward
                else "Do not continue broad deterministic 8-K item-family alpha; keep the SEC/event spine and pivot to a narrower source-first event class or deeper 8-K exhibit/materiality extraction."
            ),
        }
    )


def write_sec8k_research_decision(
    *, root: Path, decision: dict[str, object]
) -> dict[str, object]:
    """Write SEC 8-K research decision JSON and Markdown artifacts."""
    root = Path(root)
    target = root / "control" / "cluster" / "state" / "research" / "sec_8k_decision"
    history = target / "history"
    history.mkdir(parents=True, exist_ok=True)
    latest = target / "latest.json"
    latest.write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")
    history_path = history / f"{decision['checked_at'].replace(':', '').replace('+', '_')}.json"
    history_path.write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")
    report_root = root / "reports" / "research" / "sec_8k_decision"
    report_root.mkdir(parents=True, exist_ok=True)
    report_path = report_root / "latest.md"
    report_path.write_text(_render_decision_markdown(decision), encoding="utf-8")
    return {
        "decision_path": str(latest),
        "history_path": str(history_path),
        "report_path": str(report_path),
        "decision": decision,
    }


def _candidate_row(
    *,
    filing: dict[str, object],
    accession: str,
    item_number: str,
    parsed: Sec8KCompleteTextParse | None,
    source_path: Path | None,
) -> dict[str, object]:
    family = ITEM_FAMILIES.get(item_number, "SEC_8K_UNCLASSIFIED")
    issuer_cik = _text(filing.get("cik") or filing.get("issuer_cik"))
    ticker = _normalize_ticker(
        _text(
            filing.get("ticker")
            or filing.get("symbol")
            or filing.get("issuer_trading_symbol")
            or filing.get("issuer_trading_symbol_raw")
        )
    )
    accepted_raw = _text(
        filing.get("acceptanceDateTime")
        or filing.get("accepted_at_raw")
        or filing.get("accepted_at")
        or filing.get("accepted_at_utc")
    ) or (parsed.accepted_at_raw if parsed is not None else None)
    _, accepted_utc = normalize_sec_accepted_at(accepted_raw)
    item_numbers = [item_number] if item_number != "UNCLASSIFIED" else []
    documents = [document.to_dict() for document in parsed.documents] if parsed else []
    exhibit_inventory = [document for document in documents if document.get("is_exhibit")]
    exhibit_count = len(exhibit_inventory) if parsed is not None else _nonnegative_int(
        filing.get("exhibit_count")
    )
    source_hash = (
        parsed.source_hash
        if parsed is not None
        else (_text(filing.get("source_hash")) or _row_hash(filing))
    )
    exclusions: list[str] = []
    if family == "SEC_8K_UNCLASSIFIED":
        exclusions.append("unsupported_or_missing_item_family")
    if not accepted_utc:
        exclusions.append("missing_accepted_at")
    if not ticker:
        exclusions.append("missing_ticker")
    elif _invalid_ticker(ticker):
        exclusions.append("ambiguous_or_invalid_ticker")
    elif _preferred_like_ticker(ticker):
        exclusions.append("non_common_or_preferred_ticker")
    if not issuer_cik:
        exclusions.append("missing_issuer_cik")
    digest = hashlib.sha256(f"{accession}:{item_number}:{ticker}".encode("utf-8")).hexdigest()[:16]
    return {
        "event_id": f"sec8k_{digest}",
        "issuer_cik": issuer_cik,
        "ticker": ticker,
        "primary_security_id": f"{issuer_cik}:{ticker}" if issuer_cik and ticker else "",
        "accession": accession,
        "accessions": [accession],
        "archive_cik": _text(filing.get("archive_cik")),
        "complete_txt_path": _text(filing.get("complete_txt_path") or filing.get("complete_path")),
        "event_type": family,
        "sec_item_number": item_number,
        "accepted_at_utc": accepted_utc.isoformat() if accepted_utc else None,
        "first_seen_at_utc": accepted_utc.isoformat() if accepted_utc else None,
        "tradable_at_utc": None,
        "filing_date": _text(filing.get("filingDate") or filing.get("filed_date")),
        "report_date": _text(filing.get("reportDate") or filing.get("report_date")),
        "primary_document": _text(filing.get("primaryDocument")),
        "primary_doc_description": _text(filing.get("primaryDocDescription")),
        "item_numbers": item_numbers,
        "document_inventory": documents,
        "exhibit_inventory": exhibit_inventory,
        "exhibit_count": exhibit_count,
        "source_hash": source_hash,
        "source_path": str(source_path) if source_path else None,
        "parser_version": SEC8K_CANDIDATE_VERSION,
        "schema_version": SEC8K_CANDIDATE_VERSION,
        "eligibility_pass": not exclusions,
        "exclusion_reasons": exclusions,
        "event_strength_score": _event_strength(item_number=item_number, exhibit_count=exhibit_count),
    }


def _write_label_artifact(
    *,
    target: Path,
    state_target: Path,
    labels: pd.DataFrame,
    config: Form4LabelConfig,
    source_metadata: dict[str, object],
) -> dict[str, object]:
    target.mkdir(parents=True, exist_ok=True)
    data_path = target / "data.parquet"
    labels.to_parquet(data_path, index=False)
    blocker_counts: dict[str, int] = {}
    for blockers in labels.get("label_blockers", pd.Series(dtype=object)).tolist():
        for blocker in _list(blockers):
            blocker_counts[str(blocker)] = blocker_counts.get(str(blocker), 0) + 1
    status_counts = (
        labels["label_status"].value_counts().sort_index().to_dict()
        if "label_status" in labels
        else {}
    )
    report = _json_safe(
        {
            "version": SEC8K_STUDY_VERSION,
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
            "blocker_counts": dict(sorted(blocker_counts.items())),
            "horizons": list(config.horizons),
            "round_trip_cost_bps": config.round_trip_cost_bps,
            "source_metadata": source_metadata,
            "artifact": str(data_path),
        }
    )
    history = state_target / "history"
    history.mkdir(parents=True, exist_ok=True)
    latest = state_target / "latest.json"
    latest.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    history_path = history / f"{report['checked_at'].replace(':', '').replace('+', '_')}.json"
    history_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return {"labels_path": str(data_path), "report_path": str(latest), "report": report}


def _timestamp_placebo_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    frame = candidates.copy()
    calendar = xcals.get_calendar("XNYS")
    shifted: list[str | None] = []
    for value in frame.get("accepted_at_utc", pd.Series(dtype=object)).tolist():
        shifted.append(_shift_accepted_timestamp(value, calendar=calendar, sessions=-20))
    frame["event_id"] = frame["event_id"].astype(str) + "_timestamp_placebo"
    frame["event_type"] = "SEC_8K_TIMESTAMP_PLACEBO"
    frame["accepted_at_utc"] = shifted
    frame["first_seen_at_utc"] = shifted
    return frame


def _shift_accepted_timestamp(value: object, *, calendar, sessions: int) -> str | None:
    if value is None or pd.isna(value):
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    timestamp = timestamp.tz_convert("UTC")
    session = calendar.date_to_session(timestamp.date(), direction="previous")
    shifted = session
    if sessions < 0:
        for _ in range(abs(int(sessions))):
            shifted = calendar.previous_session(shifted)
    else:
        for _ in range(int(sessions)):
            shifted = calendar.next_session(shifted)
    delta = timestamp - pd.Timestamp(session).tz_localize("UTC")
    return (pd.Timestamp(shifted).tz_localize("UTC") + delta).isoformat()


def _complete_text_lookup(*, root: Path, accessions: set[str]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    if not accessions:
        return lookup
    accession_nodash = {accession.replace("-", "") for accession in accessions}
    archive_root = root / "data" / "raw" / "sec" / "archives"
    if not archive_root.exists():
        return lookup
    for path in archive_root.glob("archive_cik=*/accession=*/complete.txt"):
        accession = path.parent.name.removeprefix("accession=")
        if accession not in accession_nodash:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        lookup[accession] = text
    return lookup


def _default_filing_index_path(root: Path) -> Path | None:
    for path in (
        root / "data" / "reference" / "sec_filing_index.parquet",
        root / "data" / "reference" / "sec_filings.parquet",
        root / "data" / "curated" / "sec" / "filing_index" / "data.parquet",
    ):
        if path.exists():
            return path
    return None


def _items_from_filing(filing: dict[str, object]) -> tuple[str, ...]:
    values: list[object] = []
    for column in ("items", "item", "item_numbers", "sec_items"):
        if filing.get(column) is not None:
            values.extend(_list(filing.get(column)))
    items: set[str] = set()
    for value in values:
        items.update(_item_numbers_from_field(str(value)))
    return tuple(sorted(items))


def _item_numbers_from_text(text: str) -> set[str]:
    return {
        f"{int(match.group(1))}.{int(match.group(2)):02d}"
        for match in re.finditer(r"\bItem\s+([1-9])\.(\d{2})\b", text, flags=re.IGNORECASE)
    }


def _item_numbers_from_field(text: str) -> set[str]:
    return {
        f"{int(match.group(1))}.{int(match.group(2)):02d}"
        for match in re.finditer(r"\b([1-9])\.(\d{2})\b", text)
    }


def _document_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    lower = text.lower()
    start_tag = "<document>"
    end_tag = "</document>"
    cursor = 0
    while True:
        start = lower.find(start_tag, cursor)
        if start < 0:
            break
        body_start = start + len(start_tag)
        end = lower.find(end_tag, body_start)
        if end < 0:
            break
        blocks.append(text[body_start:end])
        cursor = end + len(end_tag)
    return blocks


def _primary_document_text(text: str) -> str:
    return _primary_document_text_from_blocks(_document_blocks(text), fallback=text)


def _primary_document_text_from_blocks(blocks: list[str], *, fallback: str) -> str:
    for block in blocks:
        if _text(sgml_tag(block, "TYPE")).upper() == "8-K":
            return _document_text(block)
    return fallback


def _document_text(block: str) -> str:
    lower = block.lower()
    start_tag = "<text>"
    end_tag = "</text>"
    start = lower.find(start_tag)
    if start < 0:
        return block
    body_start = start + len(start_tag)
    end = lower.find(end_tag, body_start)
    if end < 0:
        return block[body_start:]
    return block[body_start:end]


def _sort_columns(frame: pd.DataFrame) -> list[str]:
    columns = [
        column
        for column in ("filingDate", "acceptanceDateTime", "accessionNumber", "accession")
        if column in frame
    ]
    return columns or list(frame.columns[:1])


def _event_strength(*, item_number: str, exhibit_count: int) -> float:
    weights = {
        "2.04": 4.0,
        "4.01": 4.0,
        "2.05": 3.5,
        "3.02": 3.5,
        "1.01": 3.0,
        "8.01": 2.0,
        "7.01": 1.5,
        "2.02": 0.5,
    }
    return float(weights.get(item_number, 0.0) + math.log1p(max(exhibit_count, 0)))


def _row_hash(row: dict[str, object]) -> str:
    return hashlib.sha256(json.dumps(_json_safe(row), sort_keys=True).encode("utf-8")).hexdigest()


def _normalize_ticker(value: str | None) -> str:
    return re.sub(r"\s+", "", str(value or "").strip().upper())


def _invalid_ticker(value: str) -> bool:
    return not bool(re.fullmatch(r"[A-Z][A-Z0-9.-]{0,9}", _normalize_ticker(value)))


def _preferred_like_ticker(value: str) -> bool:
    ticker = _normalize_ticker(value)
    return "-P" in ticker or ".P" in ticker


def _nonnegative_int(value: object) -> int:
    try:
        if value is None or pd.isna(value):
            return 0
    except (TypeError, ValueError):
        pass
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _optional_text(value: object) -> str | None:
    text = _text(value)
    return text or None


def _text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


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
            return [value]
        return parsed if isinstance(parsed, list) else [parsed]
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass
    return [value]


def _render_study_markdown(packet: dict[str, object]) -> str:
    verdict = dict(packet.get("verdict") or {})
    primary = dict(packet.get("primary") or {})
    lines = [
        "# SEC 8-K Event Study",
        "",
        f"- Decision: `{verdict.get('decision')}`",
        f"- Status: `{verdict.get('status')}`",
        f"- Primary horizon: `{packet.get('primary_horizon')}d`",
        f"- Labeled events: `{primary.get('n')}`",
        f"- Mean abnormal net return: `{primary.get('mean')}`",
        f"- Median abnormal net return: `{primary.get('median')}`",
        f"- Paper/live allowed: `{verdict.get('paper_live_allowed')}`",
        "",
        "## Item Families",
    ]
    for family, summary in sorted(dict(packet.get("by_item_family") or {}).items()):
        item = dict(summary)
        lines.append(f"- `{family}`: n={item.get('n')}, mean={item.get('mean')}, median={item.get('median')}")
    return "\n".join(lines) + "\n"


def _family_failed_gates(
    *,
    summary: dict[str, object],
    min_family_events: int,
    min_mean_abret: float,
    max_top5_abs_contribution: float,
) -> list[str]:
    failed: list[str] = []
    n = int(summary.get("n") or 0)
    mean = _optional_float(summary.get("mean"))
    median = _optional_float(summary.get("median"))
    hit_rate = _optional_float(summary.get("hit_rate"))
    top5 = _optional_float(summary.get("top5_abs_contribution"))
    ci = summary.get("bootstrap_mean_ci_95")
    ci_low = None
    if isinstance(ci, list | tuple) and ci:
        ci_low = _optional_float(ci[0])
    if n < int(min_family_events):
        failed.append("insufficient_family_events")
    if mean is None or mean < float(min_mean_abret):
        failed.append("mean_below_threshold")
    if median is None or median <= 0:
        failed.append("median_not_positive")
    if hit_rate is None or hit_rate <= 0.5:
        failed.append("hit_rate_not_above_half")
    if top5 is None or top5 > float(max_top5_abs_contribution):
        failed.append("top5_contribution_too_high")
    if ci_low is None or ci_low <= 0:
        failed.append("bootstrap_ci_includes_zero")
    return failed


def _render_decision_markdown(decision: dict[str, object]) -> str:
    evidence = dict(decision.get("evidence") or {})
    lines = [
        "# SEC 8-K Research Decision",
        "",
        f"- Decision: `{decision.get('decision')}`",
        f"- Move forward: `{decision.get('move_forward')}`",
        f"- Paper/live allowed: `{decision.get('paper_live_allowed')}`",
        f"- Manifest rows: `{evidence.get('manifest_count')}`",
        f"- Parsed rows: `{evidence.get('parsed_count')}`",
        f"- Candidate count: `{evidence.get('candidate_count')}`",
        f"- Eligible count: `{evidence.get('eligible_count')}`",
        f"- Labeled events: `{dict(evidence.get('primary') or {}).get('n')}`",
        f"- Primary mean: `{dict(evidence.get('primary') or {}).get('mean')}`",
        f"- Primary median: `{dict(evidence.get('primary') or {}).get('median')}`",
        f"- Timestamp-placebo mean difference: `{dict(evidence.get('timestamp_placebo_separation') or {}).get('mean_difference')}`",
        "",
        "## Failed Study Gates",
    ]
    for gate in list(decision.get("study_failed_gates") or []):
        lines.append(f"- `{gate}`")
    lines.extend(["", "## Family Gates"])
    for family, result in sorted(dict(decision.get("family_results") or {}).items()):
        item = dict(result)
        lines.append(
            f"- `{family}`: passes={item.get('passes')}, n={item.get('n')}, "
            f"mean={item.get('mean')}, median={item.get('median')}, "
            f"failed={item.get('failed_gates')}"
        )
    lines.extend(["", "## Next Action", str(decision.get("next_action") or "")])
    return "\n".join(lines) + "\n"


def _read_optional_json(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
