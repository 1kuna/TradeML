from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from trademl.events.semantic_classifier import SecEventSemanticBatchResult, SecEventSemanticResult
from trademl.events.sec8k_semantic import (
    Sec8KSemanticSnippet,
    build_sec8k_semantic_candidates,
    build_sec8k_semantic_snippets,
    build_sec_event_semantic_study_packet,
    classify_sec8k_semantic_snippets,
    run_sec_event_semantic_classification,
    run_sec_event_semantic_labelability_audit,
    run_sec_event_semantic_scaled_gate,
    run_sec_event_semantic_study,
)


def test_sec8k_semantic_snippet_extraction_preserves_source_context(tmp_path: Path) -> None:
    _write_sec8k_semantic_fixture(tmp_path)

    snippets = build_sec8k_semantic_snippets(root=tmp_path)

    assert {snippet.snippet_kind for snippet in snippets} == {"item_section", "exhibit"}
    assert {snippet.document_type for snippet in snippets} == {"8-K", "EX-99.1"}
    assert {snippet.archive_cik for snippet in snippets} == {"320193"}
    assert {snippet.issuer_cik for snippet in snippets} == {"0000320193"}
    assert all(snippet.accepted_at_utc == "2025-04-07T20:30:00+00:00" for snippet in snippets)
    assert any("$145 million contract" in snippet.text for snippet in snippets)
    assert all("&#" not in snippet.text for snippet in snippets)
    assert all(snippet.source_hash for snippet in snippets)
    assert all(snippet.snippet_hash for snippet in snippets)


def test_sec8k_semantic_targeted_routing_uses_sec_items_only(tmp_path: Path) -> None:
    _write_sec8k_multitem_fixture(tmp_path)

    snippets = build_sec8k_semantic_snippets(
        root=tmp_path,
        routing_mode="targeted",
        target_items=("4.01",),
    )

    assert {snippet.sec_item_number for snippet in snippets if snippet.snippet_kind == "item_section"} == {"4.01"}
    assert {snippet.route_family for snippet in snippets} == {"auditor_change"}
    assert all("appointment with no disagreements" in snippet.text or snippet.snippet_kind == "exhibit" for snippet in snippets)
    assert all("sales team update" not in snippet.text.lower() for snippet in snippets)


def test_sec8k_semantic_targeted_routing_defaults_to_high_value_items(tmp_path: Path) -> None:
    _write_sec8k_multitem_fixture(tmp_path)

    snippets = build_sec8k_semantic_snippets(root=tmp_path, routing_mode="targeted")
    item_snippets = [snippet for snippet in snippets if snippet.snippet_kind == "item_section"]

    assert {snippet.sec_item_number for snippet in item_snippets} == {"1.01", "2.04", "3.02", "4.01"}
    assert {snippet.route_family for snippet in item_snippets} == {
        "material_agreement",
        "default_or_covenant_stress",
        "unregistered_sale_or_financing",
        "auditor_change",
    }
    assert all(snippet.sec_item_number != "7.01" for snippet in item_snippets)


def test_sec8k_semantic_targeted_routing_prioritizes_item_sections_before_exhibits(
    tmp_path: Path,
) -> None:
    _write_sec8k_multitem_fixture(tmp_path)

    snippets = build_sec8k_semantic_snippets(
        root=tmp_path,
        routing_mode="targeted",
        target_items=("1.01", "2.04", "3.02", "4.01"),
        limit=4,
    )

    assert len(snippets) == 4
    assert all(snippet.snippet_kind == "item_section" for snippet in snippets)
    assert [snippet.sec_item_number for snippet in snippets] == ["1.01", "2.04", "3.02", "4.01"]


def test_sec8k_semantic_targeted_routing_filters_by_accepted_date(tmp_path: Path) -> None:
    _write_sec8k_date_filter_fixture(tmp_path)

    snippets = build_sec8k_semantic_snippets(
        root=tmp_path,
        routing_mode="targeted",
        target_items=("3.02",),
        accepted_from="2025-04-01",
        accepted_to="2025-04-30",
    )

    assert snippets
    assert {snippet.accession for snippet in snippets} == {"0000320193-25-000011"}
    assert {snippet.sec_item_number for snippet in snippets if snippet.snippet_kind == "item_section"} == {"3.02"}
    assert all(snippet.filing_date == "2025-04-10" for snippet in snippets)


def test_sec8k_semantic_labelability_audit_counts_blockers(tmp_path: Path) -> None:
    _write_sec8k_labelability_fixture(tmp_path)
    _write_labelability_market_fixture(tmp_path)

    payload = run_sec_event_semantic_labelability_audit(
        data_root=tmp_path,
        routing_mode="targeted",
        target_items=("3.02",),
        accepted_from="2025-04-01",
        accepted_to="2025-04-30",
        horizons=(5,),
    )["payload"]

    assert payload["candidate_count"] == 5
    assert payload["labelable_count"] == 1
    assert payload["status_counts"] == {
        "LABELABLE": 1,
        "SKIPPED_INELIGIBLE": 2,
        "UNLABELABLE": 2,
    }
    assert payload["blocker_counts"]["missing_ticker"] == 1
    assert payload["blocker_counts"]["missing_accepted_at"] == 1
    assert payload["blocker_counts"]["missing_entry_minute"] == 1
    assert payload["blocker_counts"]["missing_exit_close_5d"] == 1


def test_sec8k_semantic_labelable_only_skips_unlabelable_llm_rows(tmp_path: Path) -> None:
    _write_sec8k_labelability_fixture(tmp_path)
    _write_labelability_market_fixture(tmp_path)

    payload = run_sec_event_semantic_classification(
        data_root=tmp_path,
        client=_AlwaysDilutiveSemanticClient(),
        routing_mode="targeted",
        target_items=("3.02",),
        accepted_from="2025-04-01",
        accepted_to="2025-04-30",
        labelability_mode="labelable-only",
        batch_size=1,
    )["payload"]

    assert payload["snippet_count"] == 2
    assert payload["promoted_candidate_count"] == 1
    assert payload["artifacts"]["queue"]["payload"]["queue_metadata"]["routed_candidate_count"] == 5
    assert payload["artifacts"]["queue"]["payload"]["queue_metadata"]["labelable_candidate_count"] == 1


def test_sec8k_semantic_snippet_kind_filters_exhibits(tmp_path: Path) -> None:
    _write_sec8k_labelability_fixture(tmp_path)
    _write_labelability_market_fixture(tmp_path)

    snippets = build_sec8k_semantic_snippets(
        root=tmp_path,
        routing_mode="targeted",
        target_items=("3.02",),
        accepted_from="2025-04-01",
        accepted_to="2025-04-30",
        labelability_mode="labelable-only",
        snippet_kind="item_section",
    )

    assert len(snippets) == 1
    assert snippets[0].snippet_kind == "item_section"


def test_sec8k_semantic_prefer_labelable_orders_labelable_first(tmp_path: Path) -> None:
    _write_sec8k_labelability_fixture(tmp_path)
    _write_labelability_market_fixture(tmp_path)

    snippets = build_sec8k_semantic_snippets(
        root=tmp_path,
        routing_mode="targeted",
        target_items=("3.02",),
        accepted_from="2025-04-01",
        accepted_to="2025-04-30",
        labelability_mode="prefer-labelable",
        limit=2,
    )

    assert len(snippets) == 2
    assert snippets[0].ticker == "AAPL"
    assert snippets[0].labelability_status == "LABELABLE"
    assert all(snippet.ticker == "AAPL" for snippet in snippets)
    assert all(snippet.labelability_status == "LABELABLE" for snippet in snippets)


def test_sec8k_semantic_scaled_gate_expands_to_2024_and_preserves_live_block(
    tmp_path: Path,
) -> None:
    _write_sec8k_scaled_gate_fixture(tmp_path)
    _write_scaled_gate_market_fixture(tmp_path)

    payload = run_sec_event_semantic_scaled_gate(
        data_root=tmp_path,
        client=_AlwaysDilutiveSemanticClient(),
        years=(2025, 2024),
        target_items=("3.02",),
        fallback_target_items=(),
        min_sample=1,
        batch_size=1,
        resume=False,
    )["packet"]

    assert [scenario["name"] for scenario in payload["scenarios"]] == [
        "primary_2025_2025",
        "primary_2024_2025",
    ]
    assert payload["scenarios"][-1]["labeled_count"] == 1
    assert payload["verdict"]["decision"] in {"CONTINUE_SEMANTIC_8K", "SEMANTIC_8K_KILLED"}
    assert payload["verdict"]["paper_live_allowed"] is False


def test_sec8k_semantic_scaled_gate_stops_when_sample_impossible(
    tmp_path: Path,
) -> None:
    _write_sec8k_scaled_gate_fixture(tmp_path)
    _write_scaled_gate_market_fixture(tmp_path)
    client = _CountingDilutiveSemanticClient()

    payload = run_sec_event_semantic_scaled_gate(
        data_root=tmp_path,
        client=client,
        years=(2024,),
        target_items=("3.02",),
        fallback_target_items=(),
        min_sample=2,
        batch_size=1,
        resume=False,
    )["packet"]

    assert client.batch_calls == 0
    assert payload["scenarios"][0]["early_stop_reason"] == "promotable_sample_gate_mathematically_impossible"
    assert payload["verdict"]["decision"] == "MORE_DATA_REQUIRED"
    assert payload["verdict"]["paper_live_allowed"] is False


def test_sec8k_semantic_classification_stops_when_promotable_sample_reached(
    tmp_path: Path,
) -> None:
    client = _CountingDilutiveSemanticClient()
    snippets = [_semantic_snippet(index) for index in range(5)]

    classifications = classify_sec8k_semantic_snippets(
        snippets=snippets,
        client=client,
        batch_size=1,
        root=tmp_path,
        min_promotable_sample=2,
    )

    assert client.batch_calls == 2
    assert len(classifications) == 2
    assert classifications.attrs["early_stop_reason"] == "promotable_sample_gate_reached"
    assert classifications.attrs["checkpoint_payload"]["early_stop_reason"] == "promotable_sample_gate_reached"


def test_sec8k_semantic_classification_writes_rows_and_candidates(tmp_path: Path) -> None:
    _write_sec8k_semantic_fixture(tmp_path)

    payload = run_sec_event_semantic_classification(
        data_root=tmp_path,
        client=_FakeBatchSemanticClient(),
        batch_size=2,
    )

    report = payload["payload"]
    assert report["verdict"] == "PASS"
    assert report["snippet_count"] == 2
    assert report["promoted_candidate_count"] == 1
    candidates = pd.read_parquet(
        tmp_path / "data" / "curated" / "events" / "sec_event_semantic_candidates" / "data.parquet"
    )
    assert candidates["event_type"].tolist() == ["MATERIAL_CONTRACT_AWARD"]
    assert candidates.iloc[0]["event_strength_score"] == 1.0
    assert bool(candidates.iloc[0]["eligibility_pass"]) is True


def test_sec8k_semantic_candidate_builder_excludes_routine_and_failed_rows() -> None:
    snippets = pd.DataFrame(
        [
            {
                "snippet_id": "s1",
                "accession": "0000320193-25-000001",
                "issuer_cik": "0000320193",
                "ticker": "AAPL",
                "accepted_at_utc": "2025-04-07T20:30:00+00:00",
            },
            {
                "snippet_id": "s2",
                "accession": "0000320193-25-000001",
                "issuer_cik": "0000320193",
                "ticker": "AAPL",
                "accepted_at_utc": "2025-04-07T20:30:00+00:00",
            },
        ]
    )
    classifications = pd.DataFrame(
        [
            {
                "snippet_id": "s1",
                "accession": "0000320193-25-000001",
                "event_type": "ROUTINE_NO_EVENT",
                "errors": [],
            },
            {
                "snippet_id": "s2",
                "accession": "0000320193-25-000001",
                "event_type": "CUSTOMER_LOSS",
                "errors": ["evidence_quote_not_exact:0"],
            },
        ]
    )

    candidates = build_sec8k_semantic_candidates(
        snippets=snippets,
        classifications=classifications,
    )

    assert candidates.empty


def test_sec8k_semantic_candidate_builder_deduplicates_same_filing_event() -> None:
    snippets = pd.DataFrame(
        [
            {
                "snippet_id": "item",
                "accession": "0000320193-25-000001",
                "issuer_cik": "0000320193",
                "ticker": "AAPL",
                "accepted_at_utc": "2025-04-07T20:30:00+00:00",
            },
            {
                "snippet_id": "exhibit",
                "accession": "0000320193-25-000001",
                "issuer_cik": "0000320193",
                "ticker": "AAPL",
                "accepted_at_utc": "2025-04-07T20:30:00+00:00",
            },
        ]
    )
    classifications = pd.DataFrame(
        [
            {
                "snippet_id": "exhibit",
                "accession": "0000320193-25-000001",
                "issuer_cik": "0000320193",
                "ticker": "AAPL",
                "accepted_at_utc": "2025-04-07T20:30:00+00:00",
                "event_type": "DILUTIVE_FINANCING",
                "errors": [],
                "warnings": ["aux_warning"],
                "evidence": np.array([{"quote": "exact exhibit quote"}], dtype=object),
                "fields": "{'amount': '$10M'}",
                "snippet_kind": "exhibit",
            },
            {
                "snippet_id": "item",
                "accession": "0000320193-25-000001",
                "issuer_cik": "0000320193",
                "ticker": "AAPL",
                "accepted_at_utc": "2025-04-07T20:30:00+00:00",
                "event_type": "DILUTIVE_FINANCING",
                "errors": [],
                "warnings": [],
                "evidence": np.array([{"quote": "exact item quote"}], dtype=object),
                "fields": {"amount": "$10M"},
                "snippet_kind": "item_section",
            },
        ]
    )

    candidates = build_sec8k_semantic_candidates(
        snippets=snippets,
        classifications=classifications,
    )

    assert len(candidates) == 1
    assert candidates.iloc[0]["event_type"] == "DILUTIVE_FINANCING"
    assert candidates.iloc[0]["semantic_snippet_id"] == "item"
    assert candidates.iloc[0]["semantic_evidence"] == [{"quote": "exact item quote"}]
    assert candidates.iloc[0]["semantic_fields"] == {"amount": "$10M"}
    assert candidates.iloc[0]["duplicate_semantic_snippet_count"] == 1
    assert set(candidates.iloc[0]["semantic_snippet_ids"]) == {"item", "exhibit"}


def test_sec8k_semantic_classification_isolates_malformed_batches(tmp_path: Path) -> None:
    _write_sec8k_semantic_fixture(tmp_path)

    payload = run_sec_event_semantic_classification(
        data_root=tmp_path,
        client=_FailWideBatchSemanticClient(),
        batch_size=2,
    )

    report = payload["payload"]
    assert report["verdict"] == "PASS"
    classifications = pd.read_parquet(
        tmp_path
        / "data"
        / "curated"
        / "events"
        / "sec_event_semantic_classifications"
        / "data.parquet"
    )
    assert classifications["status"].tolist() == ["PASS", "PASS"]
    assert any(
        str(warning).startswith("batch_retry_after:batch_exception:ValueError")
        for warnings in classifications["warnings"]
        for warning in warnings
    )


def test_sec8k_semantic_classification_uses_single_prompt_for_singleton_batches(
    tmp_path: Path,
) -> None:
    _write_sec8k_semantic_fixture(tmp_path)
    client = _SinglePreferredSemanticClient()

    payload = run_sec_event_semantic_classification(
        data_root=tmp_path,
        client=client,
        batch_size=1,
    )

    report = payload["payload"]
    assert report["verdict"] == "PASS"
    assert client.single_calls == 2
    assert client.batch_calls == 0


def test_sec8k_semantic_classification_resume_reuses_successful_checkpoint(
    tmp_path: Path,
) -> None:
    _write_sec8k_semantic_fixture(tmp_path)

    first = run_sec_event_semantic_classification(
        data_root=tmp_path,
        client=_SinglePreferredSemanticClient(),
        batch_size=1,
        resume=False,
    )
    assert first["payload"]["new_classification_count"] == 2

    second = run_sec_event_semantic_classification(
        data_root=tmp_path,
        client=_FailIfCalledSemanticClient(),
        batch_size=1,
        resume=True,
    )

    assert second["payload"]["reused_classification_count"] == 2
    assert second["payload"]["new_classification_count"] == 0
    assert second["payload"]["verdict"] == "PASS"
    checkpoint = second["payload"]["artifacts"]["checkpoint"]
    assert Path(str(checkpoint["path"])).exists()


def test_sec8k_semantic_classification_does_not_split_transport_timeouts(
    tmp_path: Path,
) -> None:
    _write_sec8k_semantic_fixture(tmp_path)

    payload = run_sec_event_semantic_classification(
        data_root=tmp_path,
        client=_TimeoutBatchSemanticClient(),
        batch_size=2,
    )

    report = payload["payload"]
    assert report["verdict"] == "PARTIAL"
    classifications = pd.read_parquet(
        tmp_path
        / "data"
        / "curated"
        / "events"
        / "sec_event_semantic_classifications"
        / "data.parquet"
    )
    assert classifications["status"].tolist() == ["FAIL", "FAIL"]
    assert all(
        any(str(error).startswith("batch_exception:ReadTimeout") for error in errors)
        for errors in classifications["errors"]
    )


def test_sec8k_semantic_study_writes_more_data_required_packet(tmp_path: Path) -> None:
    _write_semantic_candidate_fixture(tmp_path)
    _write_market_fixture(tmp_path)

    payload = run_sec_event_semantic_study(
        data_root=tmp_path,
        primary_horizon=5,
        horizons=(5,),
    )

    packet = payload["packet"]
    assert packet["verdict"]["decision"] == "MORE_DATA_REQUIRED"
    assert packet["primary"]["n"] == 1
    assert Path(str(payload["packet_path"])).exists()
    assert (
        tmp_path / "data" / "curated" / "events" / "sec_event_semantic_labels" / "data.parquet"
    ).exists()


def test_sec8k_semantic_study_packet_can_continue_on_passing_segment() -> None:
    labels = pd.DataFrame(
        [
            {
                "event_type": "CUSTOMER_LOSS",
                "label_status": "LABELED",
                "abret_5d_net": 0.02,
            }
            for _ in range(100)
        ]
    )
    placebo = pd.DataFrame(
        [
            {
                "event_type": "SEC_8K_TIMESTAMP_PLACEBO",
                "label_status": "LABELED",
                "abret_5d_net": -0.01,
            }
            for _ in range(100)
        ]
    )

    packet = build_sec_event_semantic_study_packet(
        labels=labels,
        placebo_labels=placebo,
        labels_payload={},
        placebo_payload={},
        classification_report={"verdict": "PASS"},
        header_packet={"primary": {"n": 100, "mean": -0.01}},
        primary_horizon=5,
        round_trip_cost_bps=50.0,
        min_sample=100,
        min_mean_abret=0.005,
        min_control_separation=0.0075,
        max_top5_abs_contribution=0.35,
    )

    assert packet["verdict"]["decision"] == "CONTINUE_SEMANTIC_8K"
    assert packet["verdict"]["move_forward"] is True
    assert "CUSTOMER_LOSS" in packet["verdict"]["passing_segments"]


class _FakeBatchSemanticClient:
    model = "fixture-model"

    def classify_batch(self, snippets: list[tuple[str, str]]) -> SecEventSemanticBatchResult:
        results: dict[str, SecEventSemanticResult] = {}
        for snippet_id, excerpt in snippets:
            if "$145 million contract" in excerpt:
                payload = _payload(
                    event_type="MATERIAL_CONTRACT_AWARD",
                    material_candidate="yes",
                    evidence_quote="$145 million contract",
                    materiality_evidence=["$145 million contract"],
                )
            else:
                payload = _payload(
                    event_type="ROUTINE_NO_EVENT",
                    material_candidate="no",
                    evidence_quote="Routine exhibit text",
                )
            results[snippet_id] = SecEventSemanticResult(
                payload=payload,
                raw_text=json.dumps(payload),
                content_source="content",
                elapsed_ms=1.0,
                model=self.model,
            )
        return SecEventSemanticBatchResult(
            results=results,
            raw_text="{}",
            content_source="content",
            elapsed_ms=1.0,
            model=self.model,
        )


class _FailWideBatchSemanticClient(_FakeBatchSemanticClient):
    def classify_batch(self, snippets: list[tuple[str, str]]) -> SecEventSemanticBatchResult:
        if len(snippets) > 1:
            raise ValueError("malformed batch json")
        return super().classify_batch(snippets)


class _SinglePreferredSemanticClient(_FakeBatchSemanticClient):
    def __init__(self) -> None:
        self.single_calls = 0
        self.batch_calls = 0

    def classify(self, excerpt: str) -> SecEventSemanticResult:
        self.single_calls += 1
        if "$145 million contract" in excerpt:
            payload = _payload(
                event_type="MATERIAL_CONTRACT_AWARD",
                material_candidate="yes",
                evidence_quote="$145 million contract",
                materiality_evidence=["$145 million contract"],
            )
        else:
            payload = _payload(
                event_type="ROUTINE_NO_EVENT",
                material_candidate="no",
                evidence_quote="Routine exhibit text",
            )
        return SecEventSemanticResult(
            payload=payload,
            raw_text=json.dumps(payload),
            content_source="content",
            elapsed_ms=1.0,
            model=self.model,
        )

    def classify_batch(self, snippets: list[tuple[str, str]]) -> SecEventSemanticBatchResult:
        self.batch_calls += 1
        raise AssertionError("singleton batches should use classify()")


class _TimeoutBatchSemanticClient(_FakeBatchSemanticClient):
    model = "timeout-model"

    def classify_batch(self, snippets: list[tuple[str, str]]) -> SecEventSemanticBatchResult:
        raise requests.exceptions.ReadTimeout("semantic model timeout")


class _FailIfCalledSemanticClient(_SinglePreferredSemanticClient):
    def classify(self, excerpt: str) -> SecEventSemanticResult:
        raise AssertionError("resume should reuse checkpoint rows")

    def classify_batch(self, snippets: list[tuple[str, str]]) -> SecEventSemanticBatchResult:
        raise AssertionError("resume should reuse checkpoint rows")


class _AlwaysDilutiveSemanticClient(_FakeBatchSemanticClient):
    model = "dilutive-fixture-model"

    def classify_batch(self, snippets: list[tuple[str, str]]) -> SecEventSemanticBatchResult:
        results: dict[str, SecEventSemanticResult] = {}
        for snippet_id, excerpt in snippets:
            payload = _payload(
                event_type="DILUTIVE_FINANCING",
                material_candidate="yes",
                evidence_quote="private placement",
            )
            results[snippet_id] = SecEventSemanticResult(
                payload=payload,
                raw_text=json.dumps(payload),
                content_source="content",
                elapsed_ms=1.0,
                model=self.model,
            )
        return SecEventSemanticBatchResult(
            results=results,
            raw_text="{}",
            content_source="content",
            elapsed_ms=1.0,
            model=self.model,
        )


class _CountingDilutiveSemanticClient(_AlwaysDilutiveSemanticClient):
    def __init__(self) -> None:
        self.batch_calls = 0

    def classify_batch(self, snippets: list[tuple[str, str]]) -> SecEventSemanticBatchResult:
        self.batch_calls += 1
        return super().classify_batch(snippets)


def _payload(
    *,
    event_type: str,
    material_candidate: str,
    evidence_quote: str,
    materiality_evidence: list[str] | None = None,
) -> dict[str, object]:
    return {
        "event_type": event_type,
        "certainty": "clear",
        "is_material_candidate": material_candidate,
        "fields": {
            "money_amounts": [],
            "share_or_warrant_terms": [],
            "counterparties": [],
            "dates": [],
            "trigger_terms": [],
            "materiality_evidence": list(materiality_evidence or []),
        },
        "evidence": [{"quote": evidence_quote, "supports": event_type}],
        "exclusion_reasons": [],
        "short_rationale": "fixture",
    }


def _write_sec8k_semantic_fixture(root: Path) -> None:
    candidate_path = root / "data" / "curated" / "events" / "sec_8k_item_events" / "data.parquet"
    candidate_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "event_id": "sec8k-test",
                "accession": "0000320193-25-000001",
                "accessions": ["0000320193-25-000001"],
                "issuer_cik": "0000320193",
                "ticker": "AAPL",
                "accepted_at_utc": "2025-04-07T20:30:00+00:00",
                "filing_date": "2025-04-07",
                "sec_item_number": "1.01",
                "primary_document": "form8k.htm",
            }
        ]
    ).to_parquet(candidate_path, index=False)
    raw = (
        root
        / "data"
        / "raw"
        / "sec"
        / "archives"
        / "archive_cik=320193"
        / "accession=000032019325000001"
        / "complete.txt"
    )
    raw.parent.mkdir(parents=True)
    raw.write_text(_complete_text(), encoding="utf-8")


def _write_sec8k_multitem_fixture(root: Path) -> None:
    candidate_path = root / "data" / "curated" / "events" / "sec_8k_item_events" / "data.parquet"
    candidate_path.parent.mkdir(parents=True)
    base = {
        "event_id": "sec8k-multitem",
        "accession": "0000320193-25-000002",
        "accessions": ["0000320193-25-000002"],
        "issuer_cik": "0000320193",
        "ticker": "AAPL",
        "accepted_at_utc": "2025-04-07T20:30:00+00:00",
        "filing_date": "2025-04-07",
        "primary_document": "form8k.htm",
    }
    pd.DataFrame(
        [
            {**base, "event_id": "sec8k-material", "sec_item_number": "1.01"},
            {**base, "event_id": "sec8k-default", "sec_item_number": "2.04"},
            {**base, "event_id": "sec8k-financing", "sec_item_number": "3.02"},
            {**base, "event_id": "sec8k-auditor", "sec_item_number": "4.01"},
            {**base, "event_id": "sec8k-regfd", "sec_item_number": "7.01"},
        ]
    ).to_parquet(candidate_path, index=False)
    raw = (
        root
        / "data"
        / "raw"
        / "sec"
        / "archives"
        / "archive_cik=320193"
        / "accession=000032019325000002"
        / "complete.txt"
    )
    raw.parent.mkdir(parents=True)
    raw.write_text(_multitem_complete_text(), encoding="utf-8")


def _write_sec8k_date_filter_fixture(root: Path) -> None:
    candidate_path = root / "data" / "curated" / "events" / "sec_8k_item_events" / "data.parquet"
    candidate_path.parent.mkdir(parents=True)
    rows = [
        {
            "event_id": "sec8k-jan-financing",
            "accession": "0000320193-25-000010",
            "accessions": ["0000320193-25-000010"],
            "issuer_cik": "0000320193",
            "ticker": "AAPL",
            "accepted_at_utc": "2025-01-15T20:30:00+00:00",
            "filing_date": "2025-01-15",
            "primary_document": "form8k.htm",
            "sec_item_number": "3.02",
        },
        {
            "event_id": "sec8k-apr-financing",
            "accession": "0000320193-25-000011",
            "accessions": ["0000320193-25-000011"],
            "issuer_cik": "0000320193",
            "ticker": "AAPL",
            "accepted_at_utc": "2025-04-10T20:30:00+00:00",
            "filing_date": "2025-04-10",
            "primary_document": "form8k.htm",
            "sec_item_number": "3.02",
        },
    ]
    pd.DataFrame(rows).to_parquet(candidate_path, index=False)
    for accession in ("000032019325000010", "000032019325000011"):
        raw = (
            root
            / "data"
            / "raw"
            / "sec"
            / "archives"
            / "archive_cik=320193"
            / f"accession={accession}"
            / "complete.txt"
        )
        raw.parent.mkdir(parents=True)
        raw.write_text(_multitem_complete_text(), encoding="utf-8")


def _write_sec8k_labelability_fixture(root: Path) -> None:
    candidate_path = root / "data" / "curated" / "events" / "sec_8k_item_events" / "data.parquet"
    candidate_path.parent.mkdir(parents=True)
    base = {
        "accessions": ["0000320193-25-000020"],
        "issuer_cik": "0000320193",
        "accepted_at_utc": "2025-04-07T20:30:00+00:00",
        "filing_date": "2025-04-07",
        "primary_document": "form8k.htm",
        "sec_item_number": "3.02",
        "event_type": "8K_ITEM_3_02_UNREGISTERED_SALE_FINANCING",
        "eligibility_pass": True,
        "exclusion_reasons": [],
    }
    rows = [
        {**base, "event_id": "labelable", "accession": "0000320193-25-000020", "ticker": "AAPL"},
        {**base, "event_id": "missing-minute", "accession": "0000320193-25-000021", "ticker": "MSFT"},
        {**base, "event_id": "missing-exit", "accession": "0000320193-25-000022", "ticker": "GOOG"},
        {
            **base,
            "event_id": "missing-ticker",
            "accession": "0000320193-25-000023",
            "ticker": "",
            "eligibility_pass": False,
            "exclusion_reasons": ["missing_ticker"],
        },
        {
            **base,
            "event_id": "missing-accepted",
            "accession": "0000320193-25-000024",
            "ticker": "TSLA",
            "accepted_at_utc": None,
            "eligibility_pass": False,
            "exclusion_reasons": ["missing_accepted_at"],
        },
    ]
    pd.DataFrame(rows).to_parquet(candidate_path, index=False)
    for accession in range(20, 25):
        raw = (
            root
            / "data"
            / "raw"
            / "sec"
            / "archives"
            / "archive_cik=320193"
            / f"accession=0000320193250000{accession}"
            / "complete.txt"
        )
        raw.parent.mkdir(parents=True)
        raw.write_text(_multitem_complete_text(), encoding="utf-8")


def _write_labelability_market_fixture(root: Path) -> None:
    minute_path = root / "data" / "raw" / "equities_minute" / "date=2025-04-08" / "data.parquet"
    minute_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "timestamp": "2025-04-08T13:35:00+00:00",
                "symbol": symbol,
                "open": 100.0,
                "close": 100.0,
            }
            for symbol in ("AAPL", "GOOG")
        ]
    ).to_parquet(minute_path, index=False)
    daily_path = root / "data" / "raw" / "equities_eod" / "date=2025-04-15" / "data.parquet"
    daily_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {"date": "2025-04-08", "symbol": "AAPL", "close": 100.0},
            {"date": "2025-04-15", "symbol": "AAPL", "close": 103.0},
            {"date": "2025-04-08", "symbol": "GOOG", "close": 100.0},
            {"date": "2025-04-08", "symbol": "IWM", "close": 200.0},
            {"date": "2025-04-15", "symbol": "IWM", "close": 200.0},
        ]
    ).to_parquet(daily_path, index=False)


def _write_sec8k_scaled_gate_fixture(root: Path) -> None:
    candidate_path = root / "data" / "curated" / "events" / "sec_8k_item_events" / "data.parquet"
    candidate_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "event_id": "scaled-2024",
                "accession": "0000320193-24-000030",
                "accessions": ["0000320193-24-000030"],
                "issuer_cik": "0000320193",
                "ticker": "AAPL",
                "accepted_at_utc": "2024-04-08T20:30:00+00:00",
                "filing_date": "2024-04-08",
                "primary_document": "form8k.htm",
                "sec_item_number": "3.02",
                "event_type": "8K_ITEM_3_02_UNREGISTERED_SALE_FINANCING",
                "eligibility_pass": True,
                "exclusion_reasons": [],
            }
        ]
    ).to_parquet(candidate_path, index=False)
    raw = (
        root
        / "data"
        / "raw"
        / "sec"
        / "archives"
        / "archive_cik=320193"
        / "accession=000032019324000030"
        / "complete.txt"
    )
    raw.parent.mkdir(parents=True)
    raw.write_text(_multitem_complete_text(), encoding="utf-8")


def _semantic_snippet(index: int) -> Sec8KSemanticSnippet:
    return Sec8KSemanticSnippet(
        snippet_id=f"snippet-{index}",
        accession=f"0000320193-24-{index:06d}",
        archive_cik="320193",
        issuer_cik="0000320193",
        ticker="AAPL",
        accepted_at_utc="2024-04-08T20:30:00+00:00",
        filing_date="2024-04-08",
        sec_item_number="3.02",
        route_family="unregistered_sale_or_financing",
        route_reason="targeted_item_3.02",
        snippet_kind="item_section",
        document_type="8-K",
        filename="form8k.htm",
        source_hash=f"source-{index}",
        snippet_hash=f"snippet-hash-{index}",
        labelability_status="LABELABLE",
        labelability_blockers=[],
        text="The Company entered into a private placement.",
    )


def _write_scaled_gate_market_fixture(root: Path) -> None:
    minute_path = root / "data" / "raw" / "equities_minute" / "date=2024-04-09" / "data.parquet"
    minute_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "timestamp": "2024-04-09T13:35:00+00:00",
                "symbol": "AAPL",
                "open": 100.0,
                "close": 100.0,
            }
        ]
    ).to_parquet(minute_path, index=False)
    daily_path = root / "data" / "raw" / "equities_eod" / "date=2024-04-16" / "data.parquet"
    daily_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {"date": "2024-04-09", "symbol": "AAPL", "close": 100.0},
            {"date": "2024-04-16", "symbol": "AAPL", "close": 103.0},
            {"date": "2024-04-09", "symbol": "IWM", "close": 200.0},
            {"date": "2024-04-16", "symbol": "IWM", "close": 200.0},
        ]
    ).to_parquet(daily_path, index=False)


def _write_semantic_candidate_fixture(root: Path) -> None:
    path = root / "data" / "curated" / "events" / "sec_event_semantic_candidates" / "data.parquet"
    path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "event_id": "semantic-test",
                "issuer_cik": "0000320193",
                "ticker": "AAPL",
                "primary_security_id": "0000320193:AAPL",
                "accessions": ["0000320193-25-000001"],
                "event_type": "MATERIAL_CONTRACT_AWARD",
                "accepted_at_utc": "2025-04-07T20:30:00+00:00",
                "first_seen_at_utc": "2025-04-07T20:30:00+00:00",
                "tradable_at_utc": None,
                "eligibility_pass": True,
                "exclusion_reasons": [],
                "event_strength_score": 1.0,
            }
        ]
    ).to_parquet(path, index=False)


def _write_market_fixture(root: Path) -> None:
    minute_path = root / "data" / "raw" / "equities_minute" / "date=2025-04-08" / "data.parquet"
    minute_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "timestamp": "2025-04-08T13:35:00+00:00",
                "symbol": "AAPL",
                "open": 100.0,
                "close": 100.0,
            }
        ]
    ).to_parquet(minute_path, index=False)
    daily_path = root / "data" / "raw" / "equities_eod" / "date=2025-04-15" / "data.parquet"
    daily_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {"date": "2025-04-08", "symbol": "AAPL", "close": 100.0},
            {"date": "2025-04-15", "symbol": "AAPL", "close": 103.0},
            {"date": "2025-04-08", "symbol": "IWM", "close": 200.0},
            {"date": "2025-04-15", "symbol": "IWM", "close": 200.0},
        ]
    ).to_parquet(daily_path, index=False)


def _complete_text() -> str:
    return """<SEC-DOCUMENT>
<SEC-HEADER>
<ACCEPTANCE-DATETIME>20250407163000
</SEC-HEADER>
<DOCUMENT>
<TYPE>8-K
<FILENAME>form8k.htm
<TEXT>
Item 1.01 Entry into a Material Definitive Agreement
The Company announced that it was awarded a five-year, $145 million contract for the Company&#8217;s platform.
Item 9.01 Financial Statements and Exhibits
</TEXT>
</DOCUMENT>
<DOCUMENT>
<TYPE>EX-99.1
<FILENAME>ex991.htm
<TEXT>Routine exhibit text.</TEXT>
</DOCUMENT>
<DOCUMENT>
<TYPE>EX-101.SCH
<FILENAME>schema.xsd
<TEXT>00000001 - Document - Cover link:presentationLink</TEXT>
</DOCUMENT>
</SEC-DOCUMENT>
"""


def _multitem_complete_text() -> str:
    return """<SEC-DOCUMENT>
<SEC-HEADER>
<ACCEPTANCE-DATETIME>20250407163000
</SEC-HEADER>
<DOCUMENT>
<TYPE>8-K
<FILENAME>form8k.htm
<TEXT>
Item 1.01 Entry into a Material Definitive Agreement.
The Company entered into a long-term customer agreement.
Item 2.04 Triggering Events That Accelerate or Increase a Direct Financial Obligation.
The Company received a notice of default under its credit facility.
Item 3.02 Unregistered Sales of Equity Securities.
The Company sold unregistered common stock in a private placement.
Item 4.01 Changes in Registrant's Certifying Accountant.
The Company announced an auditor appointment with no disagreements.
Item 7.01 Regulation FD Disclosure.
The Company issued a routine sales team update.
Item 9.01 Financial Statements and Exhibits.
</TEXT>
</DOCUMENT>
<DOCUMENT>
<TYPE>EX-99.1
<FILENAME>ex991.htm
<TEXT>Auditor change exhibit with no disagreements.</TEXT>
</DOCUMENT>
</SEC-DOCUMENT>
"""
