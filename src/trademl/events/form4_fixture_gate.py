"""Bounded SEC Form 4 retrieval/parser fixture gate."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Protocol

from trademl.connectors.sec_edgar import SecEdgarConnector
from trademl.data_node.budgets import BudgetManager
from trademl.data_node.provider_contracts import default_vendor_limits
from trademl.events.form4_candidates import (
    build_form4_candidate_events_from_parse_results,
    build_form4_candidate_events_from_curated,
    build_form4_candidate_fixture_gate,
    write_form4_candidate_events,
)
from trademl.events.form4 import (
    Form4ManifestRow,
    Form4ParseResult,
    Form4RetrievalResult,
    parse_form4_ownership_xml,
    write_form4_parse_results,
    write_form4_retrieval_artifacts,
)


FORM4_FIXTURE_GATE_VERSION = "form4_fixture_gate_v1"


class Form4RetrievalClient(Protocol):
    """Small retrieval surface required by the Form 4 fixture gate."""

    def retrieve_form4_ownership_xml(
        self,
        manifest: Form4ManifestRow,
        *,
        primary_document: str | None = None,
        submissions_metadata: dict[str, object] | None = None,
    ) -> Form4RetrievalResult:
        """Retrieve ownership XML and source metadata for one accession."""


@dataclass(slots=True, frozen=True)
class Form4FixtureSpec:
    """One required Form 4 weird-fixture assertion."""

    name: str
    accession: str
    archive_cik: str
    filed_date: str
    form: str = "4"
    description: str = ""
    primary_document: str | None = None
    expected_document_type: str | None = None
    expected_flags: tuple[str, ...] = ()
    forbidden_flags: tuple[str, ...] = ()
    expected_primary_eligible_min: int | None = None
    expected_primary_eligible_max: int | None = None
    expected_candidate_eligible_min: int | None = None
    expected_candidate_eligible_max: int | None = None

    def manifest(self, *, checked_at: str) -> Form4ManifestRow:
        """Return the SEC-index-style manifest row for this fixture."""
        return Form4ManifestRow(
            archive_cik=self.archive_cik,
            form=self.form,
            filed_date=self.filed_date,
            index_filename=(
                "edgar/data/"
                f"{self.archive_cik}/{self.accession.replace('-', '')}/{self.accession}.txt"
            ),
            accession=self.accession,
            accession_no_dashes=self.accession.replace("-", ""),
            discovery_source="form4_fixture_gate",
            index_year=int(self.filed_date[:4]),
            index_quarter=_quarter_for_date(self.filed_date),
            index_file_hash="form4_fixture_gate_static_manifest",
            index_crawled_at=checked_at,
        )


DEFAULT_FORM4_FIXTURES: tuple[Form4FixtureSpec, ...] = (
    Form4FixtureSpec(
        name="amazon_nooyi_weighted_average_buy",
        accession="0001127602-20-013168",
        archive_cik="1018724",
        filed_date="2020-04-03",
        description="Clean common-stock purchase with weighted-average footnotes.",
        expected_primary_eligible_min=1,
        expected_candidate_eligible_min=1,
    ),
    Form4FixtureSpec(
        name="sinclair_class_a_large_buy",
        accession="0001250842-25-000026",
        archive_cik="1971213",
        filed_date="2025-04-07",
        description="Class A common-stock purchase fixture.",
        expected_primary_eligible_min=1,
        expected_candidate_eligible_min=1,
    ),
    Form4FixtureSpec(
        name="tiptree_archive_cik_mismatch_mixed_ps",
        accession="0000769993-15-000534",
        archive_cik="769993",
        filed_date="2015-04-15",
        description="Archive CIK differs from issuer CIK and filing mixes P/S rows.",
        expected_flags=("archive_cik_differs_from_issuer_cik", "mixed_p_and_s"),
        expected_primary_eligible_max=0,
        expected_candidate_eligible_max=0,
    ),
    Form4FixtureSpec(
        name="immediatek_cuban_amendment",
        accession="0001209191-06-060213",
        archive_cik="1084182",
        filed_date="2006-11-15",
        form="4/A",
        description="Amendment with multiple reporting owners/supporting docs.",
        expected_document_type="4/A",
        expected_flags=("amendment",),
        expected_primary_eligible_max=0,
        expected_candidate_eligible_max=0,
    ),
    Form4FixtureSpec(
        name="super_micro_mechanical_amendment",
        accession="0001758554-19-000046",
        archive_cik="1375365",
        filed_date="2019-03-01",
        form="4/A",
        description="Amendment with mechanical M/F rows; no primary buy signal.",
        expected_document_type="4/A",
        expected_flags=("amendment",),
        expected_primary_eligible_max=0,
        expected_candidate_eligible_max=0,
    ),
    Form4FixtureSpec(
        name="archimedes_spac_private_unit_purchase",
        accession="0001437749-25-003569",
        archive_cik="2028516",
        filed_date="2025-02-10",
        description="SPAC/unit/private-language purchase fixture.",
        expected_flags=("private_or_unit_purchase_flag",),
        expected_primary_eligible_max=0,
        expected_candidate_eligible_max=0,
    ),
    Form4FixtureSpec(
        name="bioject_derivative_zero_price",
        accession="0000810084-13-000003",
        archive_cik="1480077",
        filed_date="2013-01-04",
        description="Derivative/warrant zero-price fixture.",
        expected_flags=("derivative_p_present",),
        expected_primary_eligible_max=0,
        expected_candidate_eligible_max=0,
    ),
    Form4FixtureSpec(
        name="eledon_private_placement",
        accession="0001593968-24-000563",
        archive_cik="1824893",
        filed_date="2024-05-08",
        description="Common-stock P row with private-placement/warrant context.",
        expected_flags=("private_or_unit_purchase_flag",),
        expected_primary_eligible_max=0,
        expected_candidate_eligible_max=0,
    ),
    Form4FixtureSpec(
        name="ares_sales_only",
        accession="0001025978-25-000011",
        archive_cik="1176948",
        filed_date="2025-01-14",
        description="Sales-only fixture for negative-control infrastructure.",
        expected_primary_eligible_max=0,
        expected_candidate_eligible_max=0,
    ),
    Form4FixtureSpec(
        name="late_report",
        accession="0001528597-26-000004",
        archive_cik="1569345",
        filed_date="2026-01-02",
        description="Very late filing fixture.",
        expected_flags=("late_report",),
        expected_primary_eligible_max=0,
        expected_candidate_eligible_max=0,
    ),
)


def run_form4_fixture_gate_from_env(
    *,
    data_root: Path,
    limit: int | None = None,
    user_agent: str | None = None,
) -> dict[str, object]:
    """Run the Form 4 fixture gate using SEC connector settings from env."""
    connector = build_sec_form4_connector_from_env(user_agent=user_agent)
    return run_form4_fixture_gate(
        data_root=data_root,
        connector=connector,
        limit=limit,
    )


def build_sec_form4_connector_from_env(
    *, user_agent: str | None = None
) -> SecEdgarConnector:
    """Build the SEC Form 4 retrieval connector from environment settings."""
    resolved_user_agent = (
        user_agent
        or os.getenv("SEC_EDGAR_USER_AGENT")
        or os.getenv("TRADEML_SEC_EDGAR_USER_AGENT")
        or os.getenv("TRADEML_SEC_USER_AGENT")
    )
    if not resolved_user_agent:
        raise RuntimeError(
            "SEC user agent required; set SEC_EDGAR_USER_AGENT or pass --user-agent"
        )
    return SecEdgarConnector(
        base_url=os.getenv("SEC_EDGAR_BASE_URL", "https://data.sec.gov"),
        user_agent=resolved_user_agent,
        budget_manager=BudgetManager(default_vendor_limits()),
    )


def run_form4_fixture_gate(
    *,
    data_root: Path,
    connector: Form4RetrievalClient,
    fixtures: tuple[Form4FixtureSpec, ...] | list[Form4FixtureSpec] | None = None,
    limit: int | None = None,
) -> dict[str, object]:
    """Retrieve, parse, assert, and persist the Form 4 weird-fixture gate."""
    root = Path(data_root).expanduser()
    checked_at = datetime.now(timezone.utc).isoformat()
    selected = list(fixtures or DEFAULT_FORM4_FIXTURES)
    if limit is not None:
        selected = selected[: max(0, int(limit))]

    parse_results: list[Form4ParseResult] = []
    fixture_payloads: list[dict[str, object]] = []
    parse_artifacts: list[str] = []
    candidate_artifacts: dict[str, object] | None = None
    candidate_gate: dict[str, object] | None = None
    for fixture in selected:
        manifest = fixture.manifest(checked_at=checked_at)
        item: dict[str, object] = {
            "name": fixture.name,
            "accession": fixture.accession,
            "archive_cik": fixture.archive_cik,
            "description": fixture.description,
            "status": "FAIL",
            "errors": [],
        }
        errors: list[str] = []
        try:
            retrieval = connector.retrieve_form4_ownership_xml(
                manifest,
                primary_document=fixture.primary_document,
            )
            raw_artifacts = write_form4_retrieval_artifacts(
                root=root,
                manifest=manifest,
                retrieval=retrieval,
            )
            item["raw_artifacts"] = [str(path) for path in raw_artifacts]
            item["retrieval"] = retrieval.metadata.to_dict()
            if retrieval.ownership_xml is None:
                errors.append("missing_ownership_xml")
                item["parse_status"] = "no_primary_xml"
            else:
                raw_xml_path, complete_txt_path = _artifact_paths(raw_artifacts)
                parse_result = parse_form4_ownership_xml(
                    retrieval.ownership_xml,
                    manifest=manifest,
                    retrieval=retrieval.metadata,
                    first_seen_at_utc=datetime.now(timezone.utc),
                    raw_xml_path=raw_xml_path,
                    complete_txt_path=complete_txt_path,
                )
                parse_results.append(parse_result)
                parse_summary = _parse_summary(parse_result)
                item.update(parse_summary)
                errors.extend(_fixture_errors(fixture, parse_result))
        except Exception as exc:  # pragma: no cover - exercised through failure status
            errors.append(f"exception:{type(exc).__name__}:{exc}")
            item["parse_status"] = "exception"
        item["errors"] = errors
        item["status"] = "PASS" if not errors else "FAIL"
        fixture_payloads.append(item)

    if parse_results:
        parse_artifacts = [
            str(path) for path in write_form4_parse_results(root=root, results=parse_results)
        ]
        fixture_candidate_events = build_form4_candidate_events_from_parse_results(
            parse_results
        )
        candidate_events = build_form4_candidate_events_from_curated(root=root)
        candidate_artifacts = write_form4_candidate_events(
            root=root,
            events=candidate_events,
        )
        expectations = _candidate_expectations(selected)
        if expectations:
            candidate_gate = build_form4_candidate_fixture_gate(
                events=fixture_candidate_events,
                expectations=expectations,
            )

    passed = sum(1 for item in fixture_payloads if item["status"] == "PASS")
    failed = len(fixture_payloads) - passed
    if candidate_gate and candidate_gate.get("verdict") != "PASS":
        failed += int(candidate_gate.get("failed") or 0)
    payload: dict[str, object] = {
        "version": FORM4_FIXTURE_GATE_VERSION,
        "checked_at": checked_at,
        "verdict": "PASS" if failed == 0 else "FAIL",
        "data_root": str(root),
        "fixture_count": len(fixture_payloads),
        "passed": passed,
        "failed": failed,
        "parse_artifacts": parse_artifacts,
        "candidate_artifacts": candidate_artifacts,
        "candidate_gate": candidate_gate,
        "fixtures": fixture_payloads,
    }
    _write_gate_artifact(root=root, payload=payload)
    return payload


def _candidate_expectations(
    fixtures: list[Form4FixtureSpec],
) -> dict[str, dict[str, int | None]]:
    expectations: dict[str, dict[str, int | None]] = {}
    for fixture in fixtures:
        if (
            fixture.expected_candidate_eligible_min is not None
            or fixture.expected_candidate_eligible_max is not None
        ):
            expectations[fixture.accession] = {
                "min": fixture.expected_candidate_eligible_min,
                "max": fixture.expected_candidate_eligible_max,
            }
    return expectations


def _parse_summary(result: Form4ParseResult) -> dict[str, object]:
    primary_eligible_count = sum(
        1 for row in result.nonderivative_transactions if row.primary_signal_eligible
    )
    return {
        "parse_status": "ok",
        "document_type": result.document_type,
        "issuer_cik": result.issuer_cik,
        "issuer_symbol": result.issuer_trading_symbol_raw,
        "owner_count": len(result.owners),
        "nonderivative_transaction_count": len(result.nonderivative_transactions),
        "derivative_transaction_count": result.derivative_transaction_count,
        "primary_eligible_count": primary_eligible_count,
        "source_quality_flags": result.source_quality_flags,
    }


def _fixture_errors(
    fixture: Form4FixtureSpec, result: Form4ParseResult
) -> list[str]:
    errors: list[str] = []
    flags = set(result.source_quality_flags)
    missing_flags = sorted(set(fixture.expected_flags) - flags)
    forbidden_flags = sorted(set(fixture.forbidden_flags) & flags)
    if missing_flags:
        errors.append(f"missing_expected_flags:{','.join(missing_flags)}")
    if forbidden_flags:
        errors.append(f"forbidden_flags_present:{','.join(forbidden_flags)}")
    if (
        fixture.expected_document_type is not None
        and result.document_type != fixture.expected_document_type
    ):
        errors.append(
            "document_type_mismatch:"
            f"expected={fixture.expected_document_type}:actual={result.document_type}"
        )
    primary_eligible_count = sum(
        1 for row in result.nonderivative_transactions if row.primary_signal_eligible
    )
    if (
        fixture.expected_primary_eligible_min is not None
        and primary_eligible_count < fixture.expected_primary_eligible_min
    ):
        errors.append(
            "primary_eligible_below_min:"
            f"min={fixture.expected_primary_eligible_min}:actual={primary_eligible_count}"
        )
    if (
        fixture.expected_primary_eligible_max is not None
        and primary_eligible_count > fixture.expected_primary_eligible_max
    ):
        errors.append(
            "primary_eligible_above_max:"
            f"max={fixture.expected_primary_eligible_max}:actual={primary_eligible_count}"
        )
    return errors


def _artifact_paths(paths: list[Path]) -> tuple[str | None, str | None]:
    raw_xml_path = next((str(path) for path in paths if path.name == "primary.xml"), None)
    complete_txt_path = next(
        (str(path) for path in paths if path.name == "complete.txt"), None
    )
    return raw_xml_path, complete_txt_path


def _write_gate_artifact(*, root: Path, payload: dict[str, object]) -> None:
    target = root / "control" / "cluster" / "state" / "research" / "form4_fixture_gate"
    history = target / "history"
    history.mkdir(parents=True, exist_ok=True)
    latest = target / "latest.json"
    latest.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    timestamp = str(payload["checked_at"]).replace(":", "").replace("+", "_")
    (history / f"{timestamp}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _quarter_for_date(filed_date: str) -> int:
    month = int(filed_date[5:7])
    return ((month - 1) // 3) + 1
