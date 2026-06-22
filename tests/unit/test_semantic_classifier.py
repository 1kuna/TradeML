from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import pytest
import requests

from trademl.events.semantic_classifier import (
    DEFAULT_SEC_EVENT_SEMANTIC_FIXTURES,
    LMStudioSecEventSemanticClient,
    PROMPT_JSON_SINGLE_PREFILL,
    SecEventSemanticBatchResult,
    SecEventSemanticResult,
    batch_response_format,
    blocking_semantic_validation_errors,
    normalize_semantic_payload,
    parse_semantic_batch_model_json,
    parse_semantic_model_json,
    run_sec_event_semantic_fixture_gate,
    validate_semantic_classification,
)


def test_parse_semantic_model_json_tolerates_wrapped_json() -> None:
    assert parse_semantic_model_json('prefix {"event_type": "ROUTINE_NO_EVENT"} suffix') == {
        "event_type": "ROUTINE_NO_EVENT"
    }


def test_parse_semantic_model_json_uses_first_balanced_object() -> None:
    raw = (
        '{"event_type": "ROUTINE_NO_EVENT"}\n\n'
        '{"event_type": "DILUTIVE_FINANCING"}'
    )

    assert parse_semantic_model_json(raw) == {"event_type": "ROUTINE_NO_EVENT"}


def test_parse_semantic_model_json_skips_echoed_input_object() -> None:
    raw = (
        '{"excerpt": "input text"}\n\n'
        '{"event_type": "ROUTINE_NO_EVENT", "certainty": "clear"}'
    )

    assert parse_semantic_model_json(raw) == {
        "event_type": "ROUTINE_NO_EVENT",
        "certainty": "clear",
    }


def test_validator_rejects_numeric_confidence() -> None:
    payload = _valid_payload(
        event_type="ROUTINE_NO_EVENT",
        material_candidate="no",
        evidence_quote="ordinary course of business",
    )
    payload["confidence"] = 0.92

    errors = validate_semantic_classification(
        excerpt="The agreement was made in the ordinary course of business.",
        payload=payload,
    )

    assert "numeric_confidence_key_forbidden:$.confidence" in errors


def test_validator_requires_exact_evidence_and_fields() -> None:
    payload = _valid_payload(
        event_type="DEBT_DEFAULT_COVENANT_STRESS",
        material_candidate="yes",
        evidence_quote="failed to maintain a minimum liquidity covenant",
    )
    payload["fields"]["trigger_terms"] = ["liquidity covenant breach"]

    errors = validate_semantic_classification(
        excerpt="The Company failed to maintain the minimum liquidity covenant.",
        payload=payload,
    )

    assert any(error.startswith("evidence_quote_not_exact:0") for error in errors)
    assert any(
        error.startswith("field_value_not_exact:trigger_terms:0") for error in errors
    )
    blockers = blocking_semantic_validation_errors(errors)
    assert any(error.startswith("evidence_quote_not_exact:0") for error in blockers)
    assert not any(
        error.startswith("field_value_not_exact:trigger_terms:0") for error in blockers
    )


def test_validator_allows_routine_no_event_without_evidence() -> None:
    payload = _valid_payload(
        event_type="ROUTINE_NO_EVENT",
        material_candidate="no",
        evidence_quote="ordinary course of business",
    )
    payload["evidence"] = []

    errors = validate_semantic_classification(
        excerpt="The agreement was made in the ordinary course of business.",
        payload=payload,
    )

    assert "missing_evidence" not in errors
    assert "missing_evidence" not in blocking_semantic_validation_errors(errors)


def test_validator_requires_promotable_event_evidence() -> None:
    payload = _valid_payload(
        event_type="DILUTIVE_FINANCING",
        material_candidate="yes",
        evidence_quote="issuance and sale of 12,000,000 shares",
    )
    payload["evidence"] = []

    errors = validate_semantic_classification(
        excerpt="The Company announced issuance and sale of 12,000,000 shares.",
        payload=payload,
    )

    assert "missing_evidence" in errors
    assert "missing_evidence" in blocking_semantic_validation_errors(errors)


def test_validator_requires_contract_materiality_evidence() -> None:
    payload = _valid_payload(
        event_type="MATERIAL_CONTRACT_AWARD",
        material_candidate="yes",
        evidence_quote="entered into a customer agreement",
    )

    errors = validate_semantic_classification(
        excerpt="The Company entered into a customer agreement.",
        payload=payload,
    )

    assert "missing_materiality_evidence_for_contract_award" in errors


def test_validator_rejects_promotable_event_marked_not_material() -> None:
    payload = _valid_payload(
        event_type="AUDITOR_TROUBLE",
        material_candidate="no",
        evidence_quote="no auditor disagreement",
    )

    errors = validate_semantic_classification(
        excerpt="The Company reported no auditor disagreement.",
        payload=payload,
    )

    assert "promotable_event_not_material_candidate" in errors
    assert "promotable_event_not_material_candidate" in blocking_semantic_validation_errors(errors)


def test_normalizer_demotes_promotable_event_marked_not_material() -> None:
    payload = _valid_payload(
        event_type="AUDITOR_TROUBLE",
        material_candidate="no",
        evidence_quote="no auditor disagreement",
    )

    normalized, warnings = normalize_semantic_payload(payload)

    assert normalized["event_type"] == "ROUTINE_NO_EVENT"
    assert warnings == [
        "normalized_promotable_not_material:AUDITOR_TROUBLE->ROUTINE_NO_EVENT:material=no"
    ]


def test_lmstudio_client_reads_json_from_reasoning_content() -> None:
    session = _FakeSession(
        _valid_payload(
            event_type="CUSTOMER_LOSS",
            material_candidate="yes",
            evidence_quote="will not renew the master supply agreement",
        )
    )
    client = LMStudioSecEventSemanticClient(
        model="qwen3.5-9b-mlx",
        base_url="http://127.0.0.1:1234/v1",
        session=session,
    )

    result = client.classify(
        "MegaRetail notified the Company that it will not renew the master supply agreement."
    )

    assert result.content_source == "reasoning_content"
    assert result.payload["event_type"] == "CUSTOMER_LOSS"
    assert session.calls[0]["url"] == "http://127.0.0.1:1234/v1/chat/completions"
    assert session.calls[0]["json"]["model"] == "qwen3.5-9b-mlx"
    assert "response_format" in session.calls[0]["json"]


def test_lmstudio_client_reads_batch_json_from_reasoning_content() -> None:
    payload = {
        "results": [
            {
                "snippet_id": "s1",
                **_valid_payload(
                    event_type="ROUTINE_NO_EVENT",
                    material_candidate="no",
                    evidence_quote="ordinary course of business",
                ),
            }
        ]
    }
    session = _FakeSession(payload)
    client = LMStudioSecEventSemanticClient(session=session)

    result = client.classify_batch([("s1", "ordinary course of business")])

    assert isinstance(result, SecEventSemanticBatchResult)
    assert result.content_source == "reasoning_content"
    assert result.results["s1"].payload["event_type"] == "ROUTINE_NO_EVENT"
    assert session.calls[0]["json"]["response_format"] == batch_response_format(["s1"])


def test_lmstudio_client_enforces_wall_clock_timeout() -> None:
    client = LMStudioSecEventSemanticClient(
        session=_SlowSession(sleep_seconds=1.0),
        timeout_seconds=0.05,
    )

    with pytest.raises(requests.exceptions.ReadTimeout, match="wall-clock timeout"):
        client.classify_batch([("s1", "ordinary course of business")])


def test_lmstudio_prompt_json_mode_uses_assistant_prefill() -> None:
    payload = _valid_payload(
        event_type="DILUTIVE_FINANCING",
        material_candidate="yes",
        evidence_quote="issuance and sale of 12,000,000 shares",
    )
    session = _FakeSession(payload, content_prefixless=True)
    client = LMStudioSecEventSemanticClient(
        session=session,
        response_format_mode="prompt_json",
    )

    result = client.classify(
        "The Company announced issuance and sale of 12,000,000 shares."
    )

    assert result.payload["event_type"] == "DILUTIVE_FINANCING"
    request = session.calls[0]["json"]
    assert "response_format" not in request
    assert request["messages"][-1] == {
        "role": "assistant",
        "content": PROMPT_JSON_SINGLE_PREFILL,
    }
    assert '"excerpt":' in request["messages"][1]["content"]


def test_batch_parser_rejects_missing_duplicate_and_unexpected_ids() -> None:
    raw = json.dumps(
        {
            "results": [
                {"snippet_id": "s1", **_valid_payload(event_type="ROUTINE_NO_EVENT", material_candidate="no", evidence_quote="x")},
                {"snippet_id": "s1", **_valid_payload(event_type="ROUTINE_NO_EVENT", material_candidate="no", evidence_quote="x")},
                {"snippet_id": "extra", **_valid_payload(event_type="ROUTINE_NO_EVENT", material_candidate="no", evidence_quote="x")},
            ]
        }
    )

    try:
        parse_semantic_batch_model_json(raw, expected_snippet_ids=["s1", "s2"])
    except ValueError as exc:
        text = str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("expected batch id mismatch")

    assert "missing=['s2']" in text
    assert "duplicates=['s1']" in text
    assert "unexpected=['extra']" in text


def test_semantic_fixture_gate_writes_pass_artifacts(tmp_path: Path) -> None:
    client = _FixtureClient()

    result = run_sec_event_semantic_fixture_gate(data_root=tmp_path, client=client)

    payload = result["payload"]
    assert payload["verdict"] == "PASS"
    assert payload["passed"] == len(DEFAULT_SEC_EVENT_SEMANTIC_FIXTURES)
    assert payload["no_numeric_confidence_policy"] is True
    assert Path(str(result["artifact_path"])).exists()
    assert Path(str(result["report_path"])).exists()


def test_semantic_fixture_gate_uses_batch_surface_when_available(tmp_path: Path) -> None:
    client = _FixtureBatchClient()

    result = run_sec_event_semantic_fixture_gate(data_root=tmp_path, client=client)

    assert result["payload"]["verdict"] == "PASS"
    assert client.batch_calls
    assert client.single_calls == 0


def test_semantic_fixture_gate_uses_single_prompt_for_singleton_batches(
    tmp_path: Path,
) -> None:
    client = _FixtureBatchClient()

    result = run_sec_event_semantic_fixture_gate(
        data_root=tmp_path,
        client=client,
        batch_size=1,
    )

    assert result["payload"]["verdict"] == "PASS"
    assert client.batch_calls == []
    assert client.single_calls == len(DEFAULT_SEC_EVENT_SEMANTIC_FIXTURES)


def test_semantic_fixture_gate_does_not_split_transport_timeouts(tmp_path: Path) -> None:
    client = _TimeoutBatchClient()

    result = run_sec_event_semantic_fixture_gate(data_root=tmp_path, client=client)

    assert result["payload"]["verdict"] == "FAIL"
    assert len(client.batch_calls) == 3
    assert result["payload"]["failed"] == len(DEFAULT_SEC_EVENT_SEMANTIC_FIXTURES)


class _FixtureClient:
    model = "fixture-model"

    def classify(self, excerpt: str) -> SecEventSemanticResult:
        fixture = next(
            item for item in DEFAULT_SEC_EVENT_SEMANTIC_FIXTURES if item.excerpt == excerpt
        )
        quote = _first_quote(excerpt)
        materiality = []
        if fixture.expected_event_type == "MATERIAL_CONTRACT_AWARD":
            materiality = ["five-year, $145 million contract"]
        payload = _valid_payload(
            event_type=fixture.expected_event_type,
            material_candidate=fixture.expected_material_candidate,
            evidence_quote=quote,
            materiality_evidence=materiality,
        )
        return SecEventSemanticResult(
            payload=payload,
            raw_text=json.dumps(payload),
            content_source="content",
            elapsed_ms=1.5,
            model=self.model,
        )


class _FixtureBatchClient(_FixtureClient):
    def __init__(self) -> None:
        self.batch_calls: list[list[str]] = []
        self.single_calls = 0

    def classify(self, excerpt: str) -> SecEventSemanticResult:
        self.single_calls += 1
        return super().classify(excerpt)

    def classify_batch(
        self, snippets: list[tuple[str, str]]
    ) -> SecEventSemanticBatchResult:
        self.batch_calls.append([snippet_id for snippet_id, _ in snippets])
        results = {
            snippet_id: _FixtureClient.classify(self, excerpt)
            for snippet_id, excerpt in snippets
        }
        return SecEventSemanticBatchResult(
            results=results,
            raw_text=json.dumps({"results": list(results)}),
            content_source="content",
            elapsed_ms=1.0,
            model=self.model,
        )


class _TimeoutBatchClient:
    model = "timeout-model"

    def __init__(self) -> None:
        self.batch_calls: list[list[str]] = []

    def classify_batch(
        self, snippets: list[tuple[str, str]]
    ) -> SecEventSemanticBatchResult:
        self.batch_calls.append([snippet_id for snippet_id, _ in snippets])
        raise requests.exceptions.ReadTimeout("semantic model timeout")


class _FakeSession:
    def __init__(
        self, payload: dict[str, object], *, content_prefixless: bool = False
    ) -> None:
        self.payload = payload
        self.content_prefixless = content_prefixless
        self.calls: list[dict[str, Any]] = []

    def post(self, url: str, **kwargs: Any) -> "_FakeResponse":
        kwargs["url"] = url
        self.calls.append(kwargs)
        return _FakeResponse(
            self.payload,
            content_prefixless=self.content_prefixless,
        )


class _FakeResponse:
    status_code = 200
    text = "{}"

    def __init__(
        self, payload: dict[str, object], *, content_prefixless: bool = False
    ) -> None:
        self.payload = payload
        self.content_prefixless = content_prefixless

    def json(self) -> dict[str, object]:
        if self.content_prefixless:
            text = json.dumps(self.payload)
            if text.startswith(PROMPT_JSON_SINGLE_PREFILL):
                text = text.removeprefix(PROMPT_JSON_SINGLE_PREFILL)
            else:
                text = text[1:]
            return {
                "choices": [
                    {
                        "message": {
                            "content": text,
                            "reasoning_content": "",
                        }
                    }
                ]
            }
        return {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "reasoning_content": json.dumps(self.payload),
                    }
                }
            ]
        }


class _SlowSession:
    def __init__(self, *, sleep_seconds: float) -> None:
        self.sleep_seconds = sleep_seconds

    def post(self, _url: str, **_kwargs: Any) -> "_FakeResponse":
        time.sleep(self.sleep_seconds)
        return _FakeResponse(
            {
                "results": [
                    {
                        "snippet_id": "s1",
                        **_valid_payload(
                            event_type="ROUTINE_NO_EVENT",
                            material_candidate="no",
                            evidence_quote="ordinary course of business",
                        ),
                    }
                ]
            }
        )


def _valid_payload(
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


def _first_quote(excerpt: str) -> str:
    return excerpt.split(".", 1)[0]
