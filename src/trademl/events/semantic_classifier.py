"""LLM-backed SEC event semantic classification with hard validation gates."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import queue
import threading
import time
from typing import Any, Protocol

import requests


SEC_EVENT_SEMANTIC_SCHEMA_VERSION = "sec_event_semantic_classifier_v1"
SEC_EVENT_SEMANTIC_GATE_VERSION = "sec_event_semantic_fixture_gate_v1"
DEFAULT_LMSTUDIO_BASE_URL = "http://127.0.0.1:1234/v1"
DEFAULT_SEC_EVENT_MODEL = "qwen3.5-9b-mlx"
RESPONSE_FORMAT_MODES = ("json_schema", "prompt_json")
PROMPT_JSON_SINGLE_PREFILL = '{"event_type":'
PROMPT_JSON_BATCH_PREFILL = '{"results":'

SEC_EVENT_TYPES = (
    "DILUTIVE_FINANCING",
    "AUDITOR_TROUBLE",
    "DEBT_DEFAULT_COVENANT_STRESS",
    "CUSTOMER_LOSS",
    "MATERIAL_CONTRACT_AWARD",
    "ROUTINE_NO_EVENT",
    "NEEDS_REVIEW",
)
PROMOTABLE_SEC_EVENT_TYPES = SEC_EVENT_TYPES[:5]
CERTAINTY_VALUES = ("clear", "mixed", "ambiguous", "insufficient")
MATERIAL_CANDIDATE_VALUES = ("yes", "no", "unknown")
FIELD_KEYS = (
    "money_amounts",
    "share_or_warrant_terms",
    "counterparties",
    "dates",
    "trigger_terms",
    "materiality_evidence",
)
PAYLOAD_KEYS = (
    "event_type",
    "certainty",
    "is_material_candidate",
    "fields",
    "evidence",
    "exclusion_reasons",
    "short_rationale",
)
NUMERIC_CONFIDENCE_KEYS = (
    "confidence",
    "confidence_score",
    "probability",
    "probabilities",
    "score",
    "scores",
    "numeric_confidence",
)


SEC_EVENT_CLASSIFICATION_RESPONSE_FORMAT: dict[str, object] = {
    "type": "json_schema",
    "json_schema": {
        "name": "sec_event_semantic_classification",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": list(PAYLOAD_KEYS),
            "properties": {
                "event_type": {"type": "string", "enum": list(SEC_EVENT_TYPES)},
                "certainty": {"type": "string", "enum": list(CERTAINTY_VALUES)},
                "is_material_candidate": {
                    "type": "string",
                    "enum": list(MATERIAL_CANDIDATE_VALUES),
                },
                "fields": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": list(FIELD_KEYS),
                    "properties": {
                        key: {
                            "type": "array",
                            "maxItems": 8,
                            "items": {"type": "string", "minLength": 1, "maxLength": 300},
                        }
                        for key in FIELD_KEYS
                    },
                },
                "evidence": {
                    "type": "array",
                    "maxItems": 3,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["quote", "supports"],
                        "properties": {
                            "quote": {"type": "string", "maxLength": 500},
                            "supports": {"type": "string", "maxLength": 200},
                        },
                    },
                },
                "exclusion_reasons": {
                    "type": "array",
                    "maxItems": 8,
                    "items": {"type": "string", "minLength": 1, "maxLength": 200},
                },
                "short_rationale": {"type": "string", "maxLength": 600},
            },
        },
    },
}


SEC_EVENT_CLASSIFICATION_SYSTEM_PROMPT = """You classify SEC 8-K, exhibit, filing, and public-news excerpts for an event-research pipeline.
Return exactly one JSON object following the supplied schema. event_type is the final label after applying all hard-gate exclusions, not the document topic.
Do not assign numeric confidence, probabilities, or scores. Use categorical certainty only.
Every evidence quote and every fields.* value must be copied exactly from the excerpt. Do not normalize or paraphrase extracted field values.
Use [] for absent array values. Never emit blank strings, placeholders such as "$0", or normalized labels in extracted field arrays.
The final event_type, short_rationale, and exclusion_reasons must agree. If the rationale says an event does not meet a hard gate, the event_type must be ROUTINE_NO_EVENT or NEEDS_REVIEW.
is_material_candidate means the excerpt is an economically meaningful research candidate for its own event type, not only for contract awards. Use yes for clear financing, auditor trouble, default/covenant stress, customer loss, or material contract award events; use no for routine/no-event; use unknown for unresolved review cases.
If event_type is DILUTIVE_FINANCING, AUDITOR_TROUBLE, DEBT_DEFAULT_COVENANT_STRESS, CUSTOMER_LOSS, or MATERIAL_CONTRACT_AWARD, is_material_candidate must be yes. If materiality is no or unknown, choose ROUTINE_NO_EVENT or NEEDS_REVIEW instead.
Never output a promotable event_type with is_material_candidate no or unknown. Never output AUDITOR_TROUBLE for a clean auditor change that says there were no adverse opinions and no disagreements; that final label is ROUTINE_NO_EVENT.
fields.trigger_terms means exact hard-gate trigger snippets only. Do not put generic legal/action words such as "dismissal", "appointed", "entered into", "agreement", or "notified" in trigger_terms unless those words themselves prove the target economic event.

Hard gates:
- MATERIAL_CONTRACT_AWARD means a commercial, customer, agency, or government contract awarded to the issuer that can generate revenue. It requires explicit economic materiality evidence: a disclosed dollar value, multi-year/major agency/customer scope, strategic/material language, or revenue/share impact. Put the exact snippets proving materiality in fields.materiality_evidence. Mergers, acquisitions, divestitures, spin-offs, joint ventures, equity investments, credit agreements, facility amendments, and governance documents are not MATERIAL_CONTRACT_AWARD. A customer agreement with no value and ordinary-course language is ROUTINE_NO_EVENT or NEEDS_REVIEW, never MATERIAL_CONTRACT_AWARD.
- DILUTIVE_FINANCING requires issuance/sale/pricing/convertible/warrant/PIPE/ATM/private-placement style capital-raising evidence. Shareholder rights plans, poison pills, anti-takeover rights distributions, and routine dividend distributions are not DILUTIVE_FINANCING unless the excerpt separately discloses capital-raising issuance or sale terms.
- AUDITOR_TROUBLE requires disagreement, adverse opinion, disclaimer, going concern issue, reportable event, material weakness, unreliability, investigation, or accounting-control problem. Clean auditor change with no disagreements is ROUTINE_NO_EVENT.
- DEBT_DEFAULT_COVENANT_STRESS requires default, covenant breach, acceleration, forbearance, waiver under stress, missed payment, or liquidity covenant failure.
- CUSTOMER_LOSS requires termination, non-renewal, loss, or material reduction of a customer/supplier/commercial relationship.
- NEEDS_REVIEW is correct when semantics are real but materiality or direction is not established.

Examples:
- Clean auditor dismissal plus "no adverse opinion" and "no disagreements" -> ROUTINE_NO_EVENT, is_material_candidate no, fields.trigger_terms [].
- Auditor dismissal plus "disagreements regarding revenue recognition" or "material weaknesses" -> AUDITOR_TROUBLE, is_material_candidate yes, fields.trigger_terms ["disagreements regarding revenue recognition", "material weaknesses"].
- Ordinary-course customer agreement with no disclosed value -> ROUTINE_NO_EVENT, is_material_candidate no.
- Five-year, $145 million contract award -> MATERIAL_CONTRACT_AWARD, is_material_candidate yes.
- Shareholder rights plan or poison pill distribution -> ROUTINE_NO_EVENT or NEEDS_REVIEW, not DILUTIVE_FINANCING.
- Acquisition, divestiture, merger, spin-off, or joint venture -> ROUTINE_NO_EVENT or NEEDS_REVIEW unless it also discloses one of the target event types separately.
"""

SEC_EVENT_PROMPT_JSON_CONTRACT = """Output exactly one JSON object with these keys and JSON types:
- event_type: string enum DILUTIVE_FINANCING, AUDITOR_TROUBLE, DEBT_DEFAULT_COVENANT_STRESS, CUSTOMER_LOSS, MATERIAL_CONTRACT_AWARD, ROUTINE_NO_EVENT, NEEDS_REVIEW.
- certainty: string enum clear, mixed, ambiguous, insufficient. Never use high, medium, or low.
- is_material_candidate: string enum yes, no, unknown. Never use true or false.
- fields: object with money_amounts, share_or_warrant_terms, counterparties, dates, trigger_terms, materiality_evidence. Each value is an array of exact excerpt strings or [].
- evidence: array of objects. Each object has quote:string and supports:string. supports must be exactly the event_type string and is never an array.
- exclusion_reasons: array of strings.
- short_rationale: string.
Do not emit any other keys. Do not emit numeric confidence. Every evidence quote and every fields value must be an exact substring of the excerpt.
"""


@dataclass(slots=True, frozen=True)
class SecEventSemanticFixture:
    """One fixture case for the semantic event classifier gate."""

    fixture_id: str
    expected_event_type: str
    expected_material_candidate: str
    excerpt: str
    description: str


@dataclass(slots=True, frozen=True)
class SecEventSemanticResult:
    """One model response plus transport metadata."""

    payload: dict[str, object]
    raw_text: str
    content_source: str
    elapsed_ms: float
    model: str


@dataclass(slots=True, frozen=True)
class SecEventSemanticBatchResult:
    """One batched model response keyed by snippet id."""

    results: dict[str, SecEventSemanticResult]
    raw_text: str
    content_source: str
    elapsed_ms: float
    model: str


class SecEventSemanticClient(Protocol):
    """Mockable semantic-classification surface."""

    model: str

    def classify(self, excerpt: str) -> SecEventSemanticResult:
        """Classify one SEC/public-event excerpt."""


class SecEventSemanticBatchClient(Protocol):
    """Mockable batched semantic-classification surface."""

    model: str

    def classify_batch(
        self, snippets: list[tuple[str, str]]
    ) -> SecEventSemanticBatchResult:
        """Classify snippets keyed by stable snippet id."""


class LMStudioSecEventSemanticClient:
    """LM Studio OpenAI-compatible semantic classifier client."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_SEC_EVENT_MODEL,
        base_url: str = DEFAULT_LMSTUDIO_BASE_URL,
        timeout_seconds: float = 180.0,
        response_format_mode: str = "json_schema",
        session: Any | None = None,
    ) -> None:
        if response_format_mode not in RESPONSE_FORMAT_MODES:
            raise ValueError(
                "response_format_mode must be one of "
                f"{', '.join(RESPONSE_FORMAT_MODES)}"
            )
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = float(timeout_seconds)
        self.response_format_mode = response_format_mode
        self._external_session = session is not None
        self.session = session or requests.Session()

    def classify(self, excerpt: str) -> SecEventSemanticResult:
        """Classify one SEC/public-event excerpt via LM Studio."""
        started = time.perf_counter()
        response = _post_with_wall_clock_timeout(
            session=self.session,
            url=f"{self.base_url}/chat/completions",
            payload=self._single_request_payload(excerpt),
            timeout_seconds=self.timeout_seconds,
            use_external_session=self._external_session,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if response.status_code != 200:
            raise RuntimeError(
                f"LM Studio semantic classification failed "
                f"status={response.status_code}: {response.text[:500]}"
            )
        body = response.json()
        message = body["choices"][0]["message"]
        raw_text, source = self._raw_message_text(message)
        payload = parse_semantic_model_json(raw_text)
        return SecEventSemanticResult(
            payload=payload,
            raw_text=raw_text,
            content_source=source,
            elapsed_ms=elapsed_ms,
            model=self.model,
        )

    def _single_request_payload(self, excerpt: str) -> dict[str, object]:
        if self.response_format_mode == "prompt_json":
            excerpt_payload = json.dumps({"excerpt": excerpt}, sort_keys=True)
            return {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            SEC_EVENT_CLASSIFICATION_SYSTEM_PROMPT
                            + "\n"
                            + SEC_EVENT_PROMPT_JSON_CONTRACT
                            + "\nContinue the already-started JSON object only. "
                            "No reasoning, markdown, or prose outside JSON."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "Classify this JSON input. Copy evidence quotes and "
                            "fields from input.excerpt exactly. Continue the "
                            "already-started JSON classification object.\n"
                            f"{excerpt_payload}"
                        ),
                    },
                    {"role": "assistant", "content": PROMPT_JSON_SINGLE_PREFILL},
                ],
                "temperature": 0,
                "max_tokens": 2000,
            }
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SEC_EVENT_CLASSIFICATION_SYSTEM_PROMPT},
                {"role": "user", "content": f"Excerpt:\n{excerpt}"},
            ],
            "response_format": SEC_EVENT_CLASSIFICATION_RESPONSE_FORMAT,
            "temperature": 0,
            "max_tokens": 2000,
        }

    def classify_batch(
        self, snippets: list[tuple[str, str]]
    ) -> SecEventSemanticBatchResult:
        """Classify a small batch of SEC/public-event snippets via LM Studio."""
        if not snippets:
            return SecEventSemanticBatchResult(
                results={},
                raw_text="",
                content_source="content",
                elapsed_ms=0.0,
                model=self.model,
            )
        snippet_ids = [snippet_id for snippet_id, _ in snippets]
        started = time.perf_counter()
        response = _post_with_wall_clock_timeout(
            session=self.session,
            url=f"{self.base_url}/chat/completions",
            payload=self._batch_request_payload(snippets=snippets, snippet_ids=snippet_ids),
            timeout_seconds=self.timeout_seconds,
            use_external_session=self._external_session,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if response.status_code != 200:
            raise RuntimeError(
                f"LM Studio semantic batch classification failed "
                f"status={response.status_code}: {response.text[:500]}"
            )
        body = response.json()
        message = body["choices"][0]["message"]
        raw_text, source = self._raw_message_text(message)
        payloads = parse_semantic_batch_model_json(raw_text, expected_snippet_ids=snippet_ids)
        return SecEventSemanticBatchResult(
            results={
                snippet_id: SecEventSemanticResult(
                    payload=payload,
                    raw_text=json.dumps(payload, sort_keys=True),
                    content_source=source,
                    elapsed_ms=elapsed_ms,
                    model=self.model,
                )
                for snippet_id, payload in payloads.items()
            },
            raw_text=raw_text,
            content_source=source,
            elapsed_ms=elapsed_ms,
            model=self.model,
        )

    def _batch_request_payload(
        self, *, snippets: list[tuple[str, str]], snippet_ids: list[str]
    ) -> dict[str, object]:
        snippets_payload = json.dumps(
            {
                "snippets": [
                    {"snippet_id": snippet_id, "excerpt": excerpt}
                    for snippet_id, excerpt in snippets
                ]
            },
            sort_keys=True,
        )
        if self.response_format_mode == "prompt_json":
            return {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            _batch_system_prompt()
                            + "\n"
                            + SEC_EVENT_PROMPT_JSON_CONTRACT
                            + "\nThe top-level object must contain exactly one key, "
                            "results, whose value is an array with exactly one "
                            "classification per input snippet_id. Continue the "
                            "already-started JSON results array only. No reasoning, markdown, "
                            "or prose outside JSON."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "Classify these snippets. Continue the already-started "
                            f"JSON results array.\n{snippets_payload}"
                        ),
                    },
                    {"role": "assistant", "content": PROMPT_JSON_BATCH_PREFILL},
                ],
                "temperature": 0,
                "max_tokens": max(1200, 1200 * len(snippets)),
            }
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _batch_system_prompt()},
                {
                    "role": "user",
                    "content": snippets_payload,
                },
            ],
            "response_format": batch_response_format(snippet_ids),
            "temperature": 0,
            "max_tokens": max(6000, 2000 * len(snippets)),
        }

    def _raw_message_text(self, message: dict[str, object]) -> tuple[str, str]:
        content = str(message.get("content") or "")
        reasoning_content = str(message.get("reasoning_content") or "")
        raw_text = content if content.strip() else reasoning_content
        source = "content" if content.strip() else "reasoning_content"
        if self.response_format_mode == "prompt_json" and content.strip():
            stripped = raw_text.lstrip()
            if not stripped.startswith("{"):
                prefix = (
                    PROMPT_JSON_BATCH_PREFILL
                    if stripped.startswith("[")
                    else PROMPT_JSON_SINGLE_PREFILL
                )
                raw_text = prefix + raw_text
        return raw_text, source


def run_sec_event_semantic_fixture_gate(
    *,
    data_root: Path,
    client: SecEventSemanticClient | None = None,
    fixtures: list[SecEventSemanticFixture] | tuple[SecEventSemanticFixture, ...] | None = None,
    model: str = DEFAULT_SEC_EVENT_MODEL,
    base_url: str = DEFAULT_LMSTUDIO_BASE_URL,
    timeout_seconds: float = 180.0,
    response_format_mode: str = "json_schema",
    batch_size: int = 4,
    limit: int | None = None,
) -> dict[str, object]:
    """Run the semantic classifier fixture gate and write durable artifacts."""
    classifier = client or LMStudioSecEventSemanticClient(
        model=model,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
        response_format_mode=response_format_mode,
    )
    selected = list(fixtures or DEFAULT_SEC_EVENT_SEMANTIC_FIXTURES)
    if limit is not None:
        selected = selected[: max(0, int(limit))]

    checked_at = _now_iso()
    batched_results = (
        _classify_fixture_batch_results(
            classifier=classifier,
            fixtures=selected,
            batch_size=batch_size,
        )
        if callable(getattr(classifier, "classify_batch", None))
        else None
    )
    results: list[dict[str, object]] = []
    for fixture in selected:
        item: dict[str, object] = {
            "fixture_id": fixture.fixture_id,
            "description": fixture.description,
            "expected_event_type": fixture.expected_event_type,
            "expected_material_candidate": fixture.expected_material_candidate,
            "excerpt_sha256": _sha256_text(fixture.excerpt),
            "status": "FAIL",
            "errors": [],
        }
        try:
            if batched_results is None:
                response = classifier.classify(fixture.excerpt)
            else:
                response = batched_results.get(fixture.fixture_id)
                if response is None:
                    raise RuntimeError(f"missing fixture batch result:{fixture.fixture_id}")
                if isinstance(response, Exception):
                    raise response
            payload, normalization_warnings = normalize_semantic_payload(response.payload)
            errors = validate_semantic_classification(
                excerpt=fixture.excerpt,
                payload=payload,
                expected_event_type=fixture.expected_event_type,
                expected_material_candidate=fixture.expected_material_candidate,
            )
            blocking_errors = blocking_semantic_validation_errors(errors)
            blocking_set = set(blocking_errors)
            warnings = [
                *normalization_warnings,
                *[error for error in errors if error not in blocking_set],
            ]
            item.update(
                {
                    "event_type": payload.get("event_type"),
                    "certainty": payload.get("certainty"),
                    "is_material_candidate": payload.get("is_material_candidate"),
                    "content_source": response.content_source,
                    "elapsed_ms": round(float(response.elapsed_ms), 3),
                    "evidence_count": len(payload.get("evidence") or []),
                    "fields": payload.get("fields") or {},
                    "evidence": payload.get("evidence") or [],
                    "exclusion_reasons": payload.get("exclusion_reasons") or [],
                    "short_rationale": payload.get("short_rationale"),
                    "model": response.model,
                    "errors": blocking_errors,
                    "warnings": warnings,
                    "status": "PASS" if not blocking_errors else "FAIL",
                }
            )
        except Exception as exc:  # pragma: no cover - live operational guard
            item["errors"] = [f"exception:{type(exc).__name__}:{exc}"]
        results.append(item)

    failed = sum(1 for item in results if item["status"] != "PASS")
    payload: dict[str, object] = {
        "version": SEC_EVENT_SEMANTIC_GATE_VERSION,
        "schema_version": SEC_EVENT_SEMANTIC_SCHEMA_VERSION,
        "checked_at": checked_at,
        "verdict": "PASS" if failed == 0 else "FAIL",
        "model": getattr(classifier, "model", model),
        "response_format_mode": getattr(
            classifier,
            "response_format_mode",
            response_format_mode,
        ),
        "batch_size": int(batch_size),
        "data_root": str(Path(data_root).expanduser()),
        "fixture_count": len(results),
        "passed": len(results) - failed,
        "failed": failed,
        "warning_count": sum(len(item.get("warnings") or []) for item in results),
        "no_numeric_confidence_policy": True,
        "exact_evidence_required": True,
        "exact_materiality_evidence_required": True,
        "auxiliary_field_exactness_warnings": True,
        "material_contract_requires_materiality_evidence": True,
        "fixtures": results,
    }
    return write_sec_event_semantic_fixture_gate(root=Path(data_root).expanduser(), payload=payload)


def _classify_fixture_batch_results(
    *,
    classifier: Any,
    fixtures: list[SecEventSemanticFixture],
    batch_size: int = 4,
) -> dict[str, SecEventSemanticResult | Exception]:
    results: dict[str, SecEventSemanticResult | Exception] = {}
    width = max(1, int(batch_size))
    for start in range(0, len(fixtures), width):
        batch = fixtures[start : start + width]
        results.update(_classify_fixture_batch_with_isolation(classifier=classifier, fixtures=batch))
    return results


def _classify_fixture_batch_with_isolation(
    *,
    classifier: Any,
    fixtures: list[SecEventSemanticFixture],
) -> dict[str, SecEventSemanticResult | Exception]:
    if len(fixtures) == 1 and callable(getattr(classifier, "classify", None)):
        fixture = fixtures[0]
        try:
            return {fixture.fixture_id: classifier.classify(fixture.excerpt)}
        except Exception as exc:
            return {fixture.fixture_id: exc}
    try:
        result = classifier.classify_batch(
            [(fixture.fixture_id, fixture.excerpt) for fixture in fixtures]
        )
    except Exception as exc:
        if len(fixtures) <= 1 or not semantic_batch_exception_is_isolatable(exc):
            return {fixture.fixture_id: exc for fixture in fixtures}
        midpoint = len(fixtures) // 2
        return {
            **_classify_fixture_batch_with_isolation(
                classifier=classifier,
                fixtures=fixtures[:midpoint],
            ),
            **_classify_fixture_batch_with_isolation(
                classifier=classifier,
                fixtures=fixtures[midpoint:],
            ),
        }
    return {
        fixture.fixture_id: result.results.get(
            fixture.fixture_id,
            RuntimeError(f"missing fixture batch result:{fixture.fixture_id}"),
        )
        for fixture in fixtures
    }


def semantic_batch_exception_is_isolatable(exc: Exception) -> bool:
    """Return true when splitting a batch can plausibly isolate one malformed item."""
    return not isinstance(exc, requests.exceptions.Timeout)


def _post_with_wall_clock_timeout(
    *,
    session: Any,
    url: str,
    payload: dict[str, object],
    timeout_seconds: float,
    use_external_session: bool,
) -> Any:
    """Run one LM Studio POST behind a hard main-thread wall-clock deadline."""
    timeout = float(timeout_seconds)
    if timeout <= 0:
        return session.post(url, json=payload, timeout=timeout_seconds)

    result_queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)

    def worker() -> None:
        try:
            if use_external_session:
                response = session.post(url, json=payload, timeout=timeout_seconds)
            else:
                with requests.Session() as request_session:
                    response = request_session.post(
                        url,
                        json=payload,
                        timeout=timeout_seconds,
                    )
            result_queue.put(("ok", response))
        except BaseException as exc:  # pragma: no cover - exercised via caller paths
            result_queue.put(("error", exc))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise requests.exceptions.ReadTimeout(
            f"LM Studio semantic request exceeded wall-clock timeout "
            f"of {timeout:.1f}s"
        )
    status, value = result_queue.get_nowait()
    if status == "error":
        raise value
    return value


def write_sec_event_semantic_fixture_gate(
    *, root: Path, payload: dict[str, object]
) -> dict[str, object]:
    """Write semantic fixture-gate JSON and Markdown artifacts."""
    target = root / "control" / "cluster" / "state" / "research" / "sec_event_semantic_fixture_gate"
    history = target / "history"
    history.mkdir(parents=True, exist_ok=True)
    latest = target / "latest.json"
    latest.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    history_path = history / f"{str(payload['checked_at']).replace(':', '').replace('+', '_')}.json"
    history_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    report_root = root / "reports" / "research" / "sec_event_semantic_fixture_gate"
    report_root.mkdir(parents=True, exist_ok=True)
    report_path = report_root / "latest.md"
    report_path.write_text(_render_gate_markdown(payload), encoding="utf-8")
    return {
        "artifact_path": str(latest),
        "history_path": str(history_path),
        "report_path": str(report_path),
        "payload": payload,
    }


def parse_semantic_model_json(raw_text: str) -> dict[str, object]:
    """Parse a model JSON payload, tolerating wrapper text around the object."""
    text = raw_text.strip()
    if not text:
        raise ValueError("empty semantic classifier response")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start < 0:
            raise
        parsed = _first_semantic_json_object(text[start:])
        if parsed is None:
            raise
    if not isinstance(parsed, dict):
        raise ValueError("semantic classifier response must be a JSON object")
    return dict(parsed)


def _first_semantic_json_object(text: str) -> dict[str, object] | None:
    first_dict: dict[str, object] | None = None
    for candidate in _balanced_json_objects(text):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        first_dict = first_dict or dict(parsed)
        if "results" in parsed or "event_type" in parsed:
            return dict(parsed)
    return first_dict


def _first_balanced_json_object(text: str) -> str | None:
    return next(iter(_balanced_json_objects(text)), None)


def _balanced_json_objects(text: str) -> list[str]:
    objects: list[str] = []
    depth = 0
    in_string = False
    escaped = False
    start_index: int | None = None
    for index, char in enumerate(text):
        if start_index is None:
            if char == "{":
                start_index = index
                depth = 1
            continue
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                objects.append(text[start_index : index + 1])
                start_index = None
    return objects


def parse_semantic_batch_model_json(
    raw_text: str, *, expected_snippet_ids: list[str]
) -> dict[str, dict[str, object]]:
    """Parse and validate a batched model JSON payload by snippet id."""
    parsed = parse_semantic_model_json(raw_text)
    results = parsed.get("results")
    if not isinstance(results, list):
        raise ValueError("semantic batch response must contain results array")
    expected = list(expected_snippet_ids)
    expected_set = set(expected)
    seen: set[str] = set()
    payloads: dict[str, dict[str, object]] = {}
    duplicates: list[str] = []
    unexpected: list[str] = []
    for index, item in enumerate(results):
        if not isinstance(item, dict):
            raise ValueError(f"semantic batch result is not an object: index={index}")
        snippet_id = item.get("snippet_id")
        if not isinstance(snippet_id, str) or not snippet_id:
            raise ValueError(f"semantic batch result missing snippet_id: index={index}")
        if snippet_id in seen:
            duplicates.append(snippet_id)
            continue
        seen.add(snippet_id)
        if snippet_id not in expected_set:
            unexpected.append(snippet_id)
            continue
        payload = dict(item)
        payload.pop("snippet_id", None)
        payloads[snippet_id] = payload
    missing = sorted(expected_set - seen)
    if duplicates or unexpected or missing:
        raise ValueError(
            "semantic batch id mismatch:"
            f"missing={missing}:duplicates={sorted(duplicates)}:"
            f"unexpected={sorted(unexpected)}"
        )
    return {snippet_id: payloads[snippet_id] for snippet_id in expected}


def batch_response_format(snippet_ids: list[str]) -> dict[str, object]:
    """Return the strict JSON schema for one semantic classification batch."""
    if not snippet_ids:
        raise ValueError("batch response format requires at least one snippet id")
    item_schema = dict(SEC_EVENT_CLASSIFICATION_RESPONSE_FORMAT["json_schema"]["schema"])  # type: ignore[index]
    properties = dict(item_schema["properties"])  # type: ignore[index]
    properties = {
        "snippet_id": {"type": "string", "enum": list(snippet_ids)},
        **properties,
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "sec_event_semantic_classification_batch",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["results"],
                "properties": {
                    "results": {
                        "type": "array",
                        "minItems": len(snippet_ids),
                        "maxItems": len(snippet_ids),
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["snippet_id", *PAYLOAD_KEYS],
                            "properties": properties,
                        },
                    }
                },
            },
        },
    }


def validate_semantic_classification(
    *,
    excerpt: str,
    payload: dict[str, object],
    expected_event_type: str | None = None,
    expected_material_candidate: str | None = None,
) -> list[str]:
    """Return validation errors for one semantic classification payload."""
    errors: list[str] = []
    errors.extend(_validate_top_level_shape(payload))
    errors.extend(_numeric_confidence_errors(payload))
    errors.extend(_validate_evidence(excerpt=excerpt, payload=payload))
    errors.extend(_validate_fields(excerpt=excerpt, payload=payload))
    if expected_event_type is not None and payload.get("event_type") != expected_event_type:
        errors.append(
            f"event_type_mismatch:expected={expected_event_type}:actual={payload.get('event_type')}"
        )
    if (
        expected_material_candidate is not None
        and payload.get("is_material_candidate") != expected_material_candidate
    ):
        errors.append(
            "material_candidate_mismatch:"
            f"expected={expected_material_candidate}:actual={payload.get('is_material_candidate')}"
        )
    if payload.get("event_type") == "MATERIAL_CONTRACT_AWARD":
        fields = payload.get("fields") if isinstance(payload.get("fields"), dict) else {}
        materiality = fields.get("materiality_evidence") if isinstance(fields, dict) else None
        if not isinstance(materiality, list) or not any(str(item).strip() for item in materiality):
            errors.append("missing_materiality_evidence_for_contract_award")
        if payload.get("is_material_candidate") != "yes":
            errors.append("contract_award_not_marked_material_candidate")
    if (
        payload.get("event_type") in PROMOTABLE_SEC_EVENT_TYPES
        and payload.get("is_material_candidate") != "yes"
    ):
        errors.append("promotable_event_not_material_candidate")
    return errors


def normalize_semantic_payload(
    payload: dict[str, object],
) -> tuple[dict[str, object], list[str]]:
    """Normalize model schema contradictions into non-promotable final labels."""
    normalized = dict(payload)
    event_type = normalized.get("event_type")
    material = normalized.get("is_material_candidate")
    warnings: list[str] = []
    if event_type in PROMOTABLE_SEC_EVENT_TYPES and material != "yes":
        replacement = "ROUTINE_NO_EVENT" if material == "no" else "NEEDS_REVIEW"
        normalized["event_type"] = replacement
        warnings.append(
            "normalized_promotable_not_material:"
            f"{event_type}->{replacement}:material={material}"
        )
    return normalized, warnings


def _batch_system_prompt() -> str:
    return (
        SEC_EVENT_CLASSIFICATION_SYSTEM_PROMPT
        + "\nYou will receive JSON with a snippets array. Return exactly one JSON object "
        "with a results array containing one classification per input snippet_id. "
        "Do not omit, add, rename, or duplicate snippet ids. Evidence quotes must be "
        "exact substrings from that specific snippet's excerpt."
    )


def blocking_semantic_validation_errors(errors: list[str]) -> list[str]:
    """Return the subset of validation errors that block fixture-gate passage."""
    blocking_prefixes = (
        "missing_required_keys",
        "unexpected_keys",
        "invalid_event_type",
        "invalid_certainty",
        "invalid_material_candidate",
        "fields_not_object",
        "missing_field_keys",
        "unexpected_field_keys",
        "field_not_string_list",
        "evidence_not_array",
        "missing_evidence",
        "evidence_not_object",
        "evidence_quote_missing",
        "evidence_quote_not_exact",
        "evidence_supports_missing",
        "exclusion_reasons_not_string_list",
        "short_rationale_not_string",
        "numeric_confidence_key_forbidden",
        "event_type_mismatch",
        "material_candidate_mismatch",
        "missing_materiality_evidence_for_contract_award",
        "contract_award_not_marked_material_candidate",
        "promotable_event_not_material_candidate",
        "field_value_not_exact:materiality_evidence",
        "field_value_blank:materiality_evidence",
    )
    return [
        error
        for error in errors
        if any(error.startswith(prefix) for prefix in blocking_prefixes)
    ]


def _validate_top_level_shape(payload: dict[str, object]) -> list[str]:
    errors: list[str] = []
    missing = sorted(set(PAYLOAD_KEYS) - set(payload))
    extra = sorted(set(payload) - set(PAYLOAD_KEYS))
    if missing:
        errors.append(f"missing_required_keys:{','.join(missing)}")
    if extra:
        errors.append(f"unexpected_keys:{','.join(extra)}")
    if payload.get("event_type") not in SEC_EVENT_TYPES:
        errors.append(f"invalid_event_type:{payload.get('event_type')}")
    if payload.get("certainty") not in CERTAINTY_VALUES:
        errors.append(f"invalid_certainty:{payload.get('certainty')}")
    if payload.get("is_material_candidate") not in MATERIAL_CANDIDATE_VALUES:
        errors.append(f"invalid_material_candidate:{payload.get('is_material_candidate')}")
    fields = payload.get("fields")
    if not isinstance(fields, dict):
        errors.append("fields_not_object")
    else:
        missing_fields = sorted(set(FIELD_KEYS) - set(fields))
        extra_fields = sorted(set(fields) - set(FIELD_KEYS))
        if missing_fields:
            errors.append(f"missing_field_keys:{','.join(missing_fields)}")
        if extra_fields:
            errors.append(f"unexpected_field_keys:{','.join(extra_fields)}")
        for key in FIELD_KEYS:
            if key in fields and not _is_string_list(fields[key]):
                errors.append(f"field_not_string_list:{key}")
    if not isinstance(payload.get("evidence"), list):
        errors.append("evidence_not_array")
    if not _is_string_list(payload.get("exclusion_reasons")):
        errors.append("exclusion_reasons_not_string_list")
    if not isinstance(payload.get("short_rationale"), str):
        errors.append("short_rationale_not_string")
    return errors


def _validate_evidence(*, excerpt: str, payload: dict[str, object]) -> list[str]:
    errors: list[str] = []
    evidence = payload.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        if _semantic_evidence_required(payload):
            errors.append("missing_evidence")
        return errors
    for index, item in enumerate(evidence):
        if not isinstance(item, dict):
            errors.append(f"evidence_not_object:{index}")
            continue
        quote = item.get("quote")
        supports = item.get("supports")
        if not isinstance(quote, str) or not quote.strip():
            errors.append(f"evidence_quote_missing:{index}")
        elif quote not in excerpt:
            errors.append(f"evidence_quote_not_exact:{index}:{quote[:80]}")
        if not isinstance(supports, str) or not supports.strip():
            errors.append(f"evidence_supports_missing:{index}")
    return errors


def _semantic_evidence_required(payload: dict[str, object]) -> bool:
    event_type = payload.get("event_type")
    material = payload.get("is_material_candidate")
    return event_type in PROMOTABLE_SEC_EVENT_TYPES or material == "yes"


def _validate_fields(*, excerpt: str, payload: dict[str, object]) -> list[str]:
    errors: list[str] = []
    fields = payload.get("fields")
    if not isinstance(fields, dict):
        return errors
    for key in FIELD_KEYS:
        values = fields.get(key)
        if not isinstance(values, list):
            continue
        for index, value in enumerate(values):
            if not isinstance(value, str) or not value.strip():
                errors.append(f"field_value_blank:{key}:{index}")
            elif value not in excerpt:
                errors.append(f"field_value_not_exact:{key}:{index}:{value[:80]}")
    return errors


def _numeric_confidence_errors(value: object, *, path: str = "$") -> list[str]:
    errors: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if str(key).lower() in NUMERIC_CONFIDENCE_KEYS:
                errors.append(f"numeric_confidence_key_forbidden:{child_path}")
            errors.extend(_numeric_confidence_errors(child, path=child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            errors.extend(_numeric_confidence_errors(child, path=f"{path}[{index}]"))
    return errors


def _is_string_list(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _render_gate_markdown(payload: dict[str, object]) -> str:
    lines = [
        "# SEC Event Semantic Fixture Gate",
        "",
        f"- Verdict: `{payload.get('verdict')}`",
        f"- Model: `{payload.get('model')}`",
        f"- Fixtures: `{payload.get('passed')}/{payload.get('fixture_count')}` passed",
        f"- Warnings: `{payload.get('warning_count')}`",
        "- Numeric confidence: forbidden",
        "- Evidence and materiality evidence: exact-substring required",
        "- Auxiliary fields: warnings when not exact; not downstream feature inputs",
        "",
        "## Fixtures",
        "",
    ]
    for item in payload.get("fixtures", []):
        if not isinstance(item, dict):
            continue
        errors = item.get("errors") or []
        warnings = item.get("warnings") or []
        error_text = ", ".join(str(error) for error in errors) if errors else "none"
        warning_text = (
            ", ".join(str(warning) for warning in warnings) if warnings else "none"
        )
        lines.extend(
            [
                f"### {item.get('fixture_id')}",
                "",
                f"- Status: `{item.get('status')}`",
                f"- Expected: `{item.get('expected_event_type')}`",
                f"- Actual: `{item.get('event_type')}`",
                f"- Material candidate: `{item.get('is_material_candidate')}`",
                f"- Certainty: `{item.get('certainty')}`",
                f"- Errors: {error_text}",
                f"- Warnings: {warning_text}",
                "",
            ]
        )
    return "\n".join(lines)


DEFAULT_SEC_EVENT_SEMANTIC_FIXTURES: tuple[SecEventSemanticFixture, ...] = (
    SecEventSemanticFixture(
        fixture_id="toxic_financing",
        expected_event_type="DILUTIVE_FINANCING",
        expected_material_candidate="yes",
        description="Common-stock and warrant issuance with anti-dilution protection.",
        excerpt=(
            "On March 4, 2025, the Company entered into a Securities Purchase Agreement "
            "with certain institutional investors for the issuance and sale of 12,000,000 "
            "shares of common stock and accompanying Series A warrants at a purchase price "
            "of $0.42 per share. The warrants include full-ratchet anti-dilution protection "
            "if the Company issues securities below the exercise price."
        ),
    ),
    SecEventSemanticFixture(
        fixture_id="auditor_clean_change",
        expected_event_type="ROUTINE_NO_EVENT",
        expected_material_candidate="no",
        description="Auditor change that explicitly says no adverse opinion or disagreements.",
        excerpt=(
            "On May 1, 2025, the Audit Committee approved the dismissal of Smith & Co. "
            "as the Company's independent registered public accounting firm and appointed "
            "Jones LLP. The reports of Smith & Co. did not contain an adverse opinion or "
            "disclaimer of opinion, and there were no disagreements with Smith & Co. on "
            "any matter of accounting principles or practices."
        ),
    ),
    SecEventSemanticFixture(
        fixture_id="auditor_disagreement",
        expected_event_type="AUDITOR_TROUBLE",
        expected_material_candidate="yes",
        description="Auditor dismissal with disagreements and material weaknesses.",
        excerpt=(
            "On May 1, 2025, the Audit Committee dismissed Smith & Co. as the Company's "
            "independent registered public accounting firm. During the two most recent "
            "fiscal years, there were disagreements regarding revenue recognition and "
            "material weaknesses in internal control over financial reporting."
        ),
    ),
    SecEventSemanticFixture(
        fixture_id="debt_default",
        expected_event_type="DEBT_DEFAULT_COVENANT_STRESS",
        expected_material_candidate="yes",
        description="Default notice with liquidity covenant failure and possible acceleration.",
        excerpt=(
            "The Company received a notice of default from its senior lender after failing "
            "to maintain the minimum liquidity covenant required by the Credit Agreement. "
            "The lender reserved all rights and may accelerate the outstanding obligations "
            "if a waiver is not obtained."
        ),
    ),
    SecEventSemanticFixture(
        fixture_id="routine_director",
        expected_event_type="ROUTINE_NO_EVENT",
        expected_material_candidate="no",
        description="Routine director appointment.",
        excerpt=(
            "On June 2, 2025, the Board appointed Jane Doe as a Class II director. Ms. Doe "
            "will receive the Company's standard non-employee director compensation. There "
            "are no arrangements or understandings between Ms. Doe and any other person "
            "pursuant to which she was selected as a director."
        ),
    ),
    SecEventSemanticFixture(
        fixture_id="material_contract_award",
        expected_event_type="MATERIAL_CONTRACT_AWARD",
        expected_material_candidate="yes",
        description="DoD contract award with disclosed value and term.",
        excerpt=(
            "The Company announced that it was awarded a five-year, $145 million contract "
            "by the U.S. Department of Defense to supply ruggedized communications systems. "
            "Work under the contract is expected to begin in the third quarter of 2025."
        ),
    ),
    SecEventSemanticFixture(
        fixture_id="ordinary_contract_no_value",
        expected_event_type="ROUTINE_NO_EVENT",
        expected_material_candidate="no",
        description="Ordinary-course customer agreement with no disclosed value.",
        excerpt=(
            "The Company entered into a customer agreement to provide software services. "
            "The agreement was made in the ordinary course of business and the Company did "
            "not disclose the value of the agreement."
        ),
    ),
    SecEventSemanticFixture(
        fixture_id="customer_loss",
        expected_event_type="CUSTOMER_LOSS",
        expected_material_candidate="yes",
        description="Material customer non-renewal with revenue share.",
        excerpt=(
            "On April 18, 2025, MegaRetail notified the Company that it will not renew the "
            "master supply agreement when the current term expires. Sales to MegaRetail "
            "represented approximately 38% of the Company's revenue for fiscal 2024."
        ),
    ),
    SecEventSemanticFixture(
        fixture_id="shareholder_rights_plan",
        expected_event_type="ROUTINE_NO_EVENT",
        expected_material_candidate="no",
        description="Anti-takeover shareholder rights plan without capital raise.",
        excerpt=(
            "The Board authorized and declared a dividend distribution of one common "
            "share purchase right for each outstanding common share under a shareholder "
            "rights plan intended to protect stockholders from coercive takeover tactics. "
            "The Company did not issue securities for cash or raise new capital."
        ),
    ),
    SecEventSemanticFixture(
        fixture_id="acquisition_not_contract_award",
        expected_event_type="ROUTINE_NO_EVENT",
        expected_material_candidate="no",
        description="Material acquisition with disclosed value but no customer contract award.",
        excerpt=(
            "The Company completed the acquisition of Example Logistics LLC for "
            "$325 million in cash. The transaction expands the Company's footprint "
            "but does not include a customer award, government award, or new revenue "
            "contract with a disclosed customer."
        ),
    ),
)
