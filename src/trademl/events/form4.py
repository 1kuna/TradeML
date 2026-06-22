"""SEC Form 4 ownership filing parsing and raw artifact helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from zoneinfo import ZoneInfo

import pandas as pd


UTC = timezone.utc
EASTERN = ZoneInfo("America/New_York")
PARSER_VERSION = "form4_parser_v1"
SCHEMA_VERSION = "form4_v1"

COMMON_LIKE_ALLOW = (
    "common stock",
    "class a common stock",
    "class b common stock",
    "ordinary shares",
    "ordinary share",
    "common shares",
    "common share",
)
SECURITY_EXCLUDE_TERMS = (
    "warrant",
    "option",
    "restricted stock unit",
    "rsu",
    "preferred",
    "convertible",
    "unit",
    "right to receive",
    "phantom",
    "performance stock unit",
    "depositary share",
    "common units",
    "limited partnership",
)
PRIVATE_PURCHASE_TERMS = (
    "private placement",
    "purchase agreement",
    "subscription agreement",
    "pipe",
    "sponsor",
    "initial public offering",
    "simultaneously with the consummation",
    "warrant",
)


@dataclass(slots=True, frozen=True)
class Form4ManifestRow:
    """One SEC index-derived Form 4 accession manifest row."""

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
class Form4RetrievalMetadata:
    """HTTP/source metadata for a retrieved ownership filing."""

    primary_xml_url: str | None
    primary_xml_http_status: int | None
    primary_xml_sha256: str | None
    complete_txt_url: str | None
    complete_txt_http_status: int | None
    complete_txt_sha256: str | None
    xml_source: str
    accepted_at_raw: str | None
    accepted_at_source: str | None
    accepted_at_et: datetime | None
    accepted_at_utc: datetime | None
    quality_flags: list[str]

    @classmethod
    def from_accepted_raw(
        cls,
        *,
        primary_xml_url: str | None,
        primary_xml_http_status: int | None,
        primary_xml_sha256: str | None,
        complete_txt_url: str | None,
        complete_txt_http_status: int | None,
        complete_txt_sha256: str | None,
        xml_source: str,
        accepted_at_raw: str | None,
        accepted_at_source: str | None,
        quality_flags: list[str],
    ) -> Form4RetrievalMetadata:
        """Build metadata while normalizing SEC accepted timestamps."""
        accepted_at_et, accepted_at_utc = normalize_sec_accepted_at(accepted_at_raw)
        return cls(
            primary_xml_url=primary_xml_url,
            primary_xml_http_status=primary_xml_http_status,
            primary_xml_sha256=primary_xml_sha256,
            complete_txt_url=complete_txt_url,
            complete_txt_http_status=complete_txt_http_status,
            complete_txt_sha256=complete_txt_sha256,
            xml_source=xml_source,
            accepted_at_raw=accepted_at_raw,
            accepted_at_source=accepted_at_source,
            accepted_at_et=accepted_at_et,
            accepted_at_utc=accepted_at_utc,
            quality_flags=list(quality_flags),
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe representation."""
        return {
            "primary_xml_url": self.primary_xml_url,
            "primary_xml_http_status": self.primary_xml_http_status,
            "primary_xml_sha256": self.primary_xml_sha256,
            "complete_txt_url": self.complete_txt_url,
            "complete_txt_http_status": self.complete_txt_http_status,
            "complete_txt_sha256": self.complete_txt_sha256,
            "xml_source": self.xml_source,
            "accepted_at_raw": self.accepted_at_raw,
            "accepted_at_source": self.accepted_at_source,
            "accepted_at_et": (
                self.accepted_at_et.isoformat() if self.accepted_at_et else None
            ),
            "accepted_at_utc": (
                self.accepted_at_utc.isoformat() if self.accepted_at_utc else None
            ),
            "quality_flags": list(self.quality_flags),
        }


@dataclass(slots=True, frozen=True)
class Form4RetrievalResult:
    """Retrieved ownership XML plus optional complete submission text."""

    metadata: Form4RetrievalMetadata
    ownership_xml: str | None
    complete_txt: str | None = None


@dataclass(slots=True, frozen=True)
class Form4Owner:
    """One reporting owner from an ownership filing."""

    owner_cik: str
    owner_name: str
    is_director: bool
    is_officer: bool
    is_ten_percent_owner: bool
    is_other: bool
    officer_title: str | None

    def to_dict(self, *, accession: str) -> dict[str, object]:
        """Return a parquet-safe owner row."""
        return {
            "accession": accession,
            "owner_cik": self.owner_cik,
            "owner_name": self.owner_name,
            "is_director": self.is_director,
            "is_officer": self.is_officer,
            "is_ten_percent_owner": self.is_ten_percent_owner,
            "is_other": self.is_other,
            "officer_title": self.officer_title,
        }


@dataclass(slots=True, frozen=True)
class Form4TransactionRow:
    """One parsed non-derivative Form 4 transaction row."""

    accession: str
    accession_no_dashes: str
    archive_cik: str
    index_filename: str
    manifest_source: str
    form_type: str
    document_type: str
    issuer_cik: str
    issuer_name: str
    issuer_trading_symbol_raw: str
    owner_cik_set: tuple[str, ...]
    reporting_owner_cik: str | None
    reporting_owner_name: str | None
    is_director: bool
    is_officer: bool
    is_ten_percent_owner: bool
    is_other: bool
    officer_title: str | None
    accepted_at_raw: str | None
    accepted_at_source: str | None
    accepted_at_et: datetime | None
    accepted_at_utc: datetime | None
    first_seen_at_utc: datetime | None
    filed_date: str
    period_of_report: str | None
    date_of_original_submission: str | None
    transaction_date: str | None
    transaction_date_raw: str | None
    security_title: str
    security_title_normalized: str
    transaction_code: str
    transaction_form_type: str
    acquired_disposed: str
    transaction_shares: Decimal | None
    transaction_price: Decimal | None
    transaction_value: Decimal | None
    post_transaction_shares: Decimal | None
    direct_or_indirect: str | None
    ownership_nature: str | None
    field_footnote_ids: dict[str, list[str]]
    footnotes_text: str | None
    probably_private_or_unit_purchase: bool
    same_filing_has_sales: bool
    same_owner_same_day_has_sales: bool
    primary_document: str | None
    raw_xml_url: str | None
    raw_xml_path: str | None
    raw_xml_hash: str | None
    complete_txt_url: str | None
    complete_txt_path: str | None
    complete_txt_hash: str | None
    primary_xml_source: str
    parser_version: str
    schema_version: str
    source_quality_flags: list[str]
    primary_signal_eligible: bool

    def to_dict(self) -> dict[str, object]:
        """Return a parquet/JSON-safe representation."""
        payload = {
            "accession": self.accession,
            "accession_no_dashes": self.accession_no_dashes,
            "archive_cik": self.archive_cik,
            "index_filename": self.index_filename,
            "manifest_source": self.manifest_source,
            "form_type": self.form_type,
            "document_type": self.document_type,
            "issuer_cik": self.issuer_cik,
            "issuer_name": self.issuer_name,
            "issuer_trading_symbol_raw": self.issuer_trading_symbol_raw,
            "owner_cik_set": list(self.owner_cik_set),
            "reporting_owner_cik": self.reporting_owner_cik,
            "reporting_owner_name": self.reporting_owner_name,
            "is_director": self.is_director,
            "is_officer": self.is_officer,
            "is_ten_percent_owner": self.is_ten_percent_owner,
            "is_other": self.is_other,
            "officer_title": self.officer_title,
            "accepted_at_raw": self.accepted_at_raw,
            "accepted_at_source": self.accepted_at_source,
            "accepted_at_et": (
                self.accepted_at_et.isoformat() if self.accepted_at_et else None
            ),
            "accepted_at_utc": (
                self.accepted_at_utc.isoformat() if self.accepted_at_utc else None
            ),
            "first_seen_at_utc": (
                self.first_seen_at_utc.isoformat()
                if self.first_seen_at_utc
                else None
            ),
            "filed_date": self.filed_date,
            "period_of_report": self.period_of_report,
            "date_of_original_submission": self.date_of_original_submission,
            "transaction_date": self.transaction_date,
            "transaction_date_raw": self.transaction_date_raw,
            "security_title": self.security_title,
            "security_title_normalized": self.security_title_normalized,
            "transaction_code": self.transaction_code,
            "transaction_form_type": self.transaction_form_type,
            "acquired_disposed": self.acquired_disposed,
            "transaction_shares": _decimal_to_string(self.transaction_shares),
            "transaction_price": _decimal_to_string(self.transaction_price),
            "transaction_value": _decimal_to_string(self.transaction_value),
            "post_transaction_shares": _decimal_to_string(
                self.post_transaction_shares
            ),
            "direct_or_indirect": self.direct_or_indirect,
            "ownership_nature": self.ownership_nature,
            "field_footnote_ids": self.field_footnote_ids,
            "footnotes_text": self.footnotes_text,
            "probably_private_or_unit_purchase": self.probably_private_or_unit_purchase,
            "same_filing_has_sales": self.same_filing_has_sales,
            "same_owner_same_day_has_sales": self.same_owner_same_day_has_sales,
            "primary_document": self.primary_document,
            "raw_xml_url": self.raw_xml_url,
            "raw_xml_path": self.raw_xml_path,
            "raw_xml_hash": self.raw_xml_hash,
            "complete_txt_url": self.complete_txt_url,
            "complete_txt_path": self.complete_txt_path,
            "complete_txt_hash": self.complete_txt_hash,
            "primary_xml_source": self.primary_xml_source,
            "parser_version": self.parser_version,
            "schema_version": self.schema_version,
            "source_quality_flags": list(self.source_quality_flags),
            "primary_signal_eligible": self.primary_signal_eligible,
        }
        return payload


@dataclass(slots=True, frozen=True)
class Form4ParseResult:
    """Document-level Form 4 parse output."""

    accession: str
    archive_cik: str
    issuer_cik: str
    issuer_name: str
    issuer_trading_symbol_raw: str
    document_type: str
    period_of_report: str | None
    date_of_original_submission: str | None
    owners: list[Form4Owner]
    nonderivative_transactions: list[Form4TransactionRow]
    derivative_transaction_count: int
    footnotes: dict[str, str]
    source_quality_flags: list[str]
    accepted_at_raw: str | None = None
    accepted_at_source: str | None = None
    accepted_at_utc: datetime | None = None
    first_seen_at_utc: datetime | None = None


def normalize_sec_accepted_at(raw: str | None) -> tuple[datetime | None, datetime | None]:
    """Normalize a SEC accepted timestamp into ET and UTC."""
    if raw is None:
        return None, None
    text = str(raw).strip()
    if not text:
        return None, None
    if re.fullmatch(r"\d{14}", text):
        accepted_et = datetime.strptime(text, "%Y%m%d%H%M%S").replace(tzinfo=EASTERN)
        return accepted_et, accepted_et.astimezone(UTC)
    normalized = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None, None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=EASTERN)
    accepted_et = parsed.astimezone(EASTERN)
    return accepted_et, parsed.astimezone(UTC)


def parse_form4_index_manifest(
    text: str,
    *,
    index_year: int,
    index_quarter: int,
    index_crawled_at: str,
    discovery_source: str = "sec_full_index",
) -> list[Form4ManifestRow]:
    """Parse SEC full-index text into Form 4/4-A manifest rows."""
    index_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    rows: list[Form4ManifestRow] = []
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
        if form not in {"4", "4/A"}:
            continue
        match = re.search(r"edgar/data/(\d+)/(?:\d+/)?([^/]+)\.txt$", filename)
        if match is None:
            continue
        archive_cik = match.group(1)
        accession = match.group(2)
        rows.append(
            Form4ManifestRow(
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


def extract_primary_ownership_xml_from_complete_txt(txt: str) -> str | None:
    """Extract the primary ownership XML block from a complete SEC submission."""
    for doc in re.findall(r"<DOCUMENT>(.*?)</DOCUMENT>", txt, flags=re.I | re.S):
        doc_type = sgml_tag(doc, "TYPE")
        filename = sgml_tag(doc, "FILENAME")
        if doc_type not in {"4", "4/A"}:
            continue
        xml_match = re.search(r"<XML>(.*?)</XML>", doc, flags=re.I | re.S)
        if xml_match and "<ownershipDocument" in xml_match.group(1):
            return _ownership_xml_fragment(xml_match.group(1))
        if filename and filename.lower().endswith(".xml") and "<ownershipDocument" in doc:
            return _ownership_xml_fragment(doc)
    return None


def sgml_tag(text: str, tag: str) -> str | None:
    """Return a simple SGML tag value from complete submission text."""
    match = re.search(
        rf"<{re.escape(tag)}>\s*([^\n\r<]+)", text, flags=re.IGNORECASE
    )
    if match is None:
        return None
    return match.group(1).strip()


def parse_form4_ownership_xml(
    xml: str,
    *,
    manifest: Form4ManifestRow,
    retrieval: Form4RetrievalMetadata,
    first_seen_at_utc: datetime | None = None,
    raw_xml_path: str | None = None,
    complete_txt_path: str | None = None,
) -> Form4ParseResult:
    """Parse one SEC ownership XML document into normalized Form 4 rows."""
    fragment = _ownership_xml_fragment(xml)
    root = ET.fromstring(fragment)
    document_type = _text(root, ("documentType",)) or manifest.form
    period_of_report = _text(root, ("periodOfReport",))
    date_of_original_submission = _text(root, ("dateOfOriginalSubmission",))
    not_subject = _parse_bool(_text(root, ("notSubjectToSection16",)))
    issuer_cik = _text(root, ("issuer", "issuerCik")) or ""
    issuer_name = _text(root, ("issuer", "issuerName")) or ""
    issuer_symbol = _text(root, ("issuer", "issuerTradingSymbol")) or ""
    remarks = _text(root, ("remarks",)) or ""
    owners = _parse_owners(root)
    owner_cik_set = tuple(sorted(owner.owner_cik for owner in owners if owner.owner_cik))
    owner = owners[0] if owners else None
    footnotes = _parse_footnotes(root)
    footnotes_text = "\n".join(footnotes.values()) if footnotes else None
    derivative_transactions = _children(
        _node(root, ("derivativeTable",)), "derivativeTransaction"
    )
    derivative_transaction_count = len(derivative_transactions)
    derivative_p_present = any(
        (_text(tx, ("transactionCoding", "transactionCode")) or "").upper() == "P"
        for tx in derivative_transactions
    )
    nonderiv_transactions = _children(
        _node(root, ("nonDerivativeTable",)), "nonDerivativeTransaction"
    )
    filing_codes = {
        (_text(tx, ("transactionCoding", "transactionCode")) or "").upper()
        for tx in nonderiv_transactions
    }
    same_filing_has_sales = "S" in filing_codes
    document_flags: list[str] = []
    if document_type == "4/A":
        document_flags.append("amendment")
    if derivative_p_present:
        document_flags.append("derivative_p_present")
    if manifest.archive_cik.lstrip("0") != issuer_cik.lstrip("0"):
        document_flags.append("archive_cik_differs_from_issuer_cik")
    if not_subject:
        document_flags.append("not_subject_to_section16")

    rows: list[Form4TransactionRow] = []
    for tx in nonderiv_transactions:
        row = _parse_nonderivative_transaction(
            tx,
            manifest=manifest,
            retrieval=retrieval,
            document_type=document_type,
            issuer_cik=issuer_cik,
            issuer_name=issuer_name,
            issuer_symbol=issuer_symbol,
            owner=owner,
            owner_cik_set=owner_cik_set,
            period_of_report=period_of_report,
            date_of_original_submission=date_of_original_submission,
            not_subject=not_subject,
            same_filing_has_sales=same_filing_has_sales,
            derivative_p_present=derivative_p_present,
            document_flags=document_flags,
            remarks=remarks,
            footnotes_text=footnotes_text,
            first_seen_at_utc=first_seen_at_utc,
            raw_xml_path=raw_xml_path,
            complete_txt_path=complete_txt_path,
        )
        rows.append(row)

    all_flags = sorted(
        {
            *document_flags,
            *(flag for row in rows for flag in row.source_quality_flags),
        }
    )
    return Form4ParseResult(
        accession=manifest.accession,
        archive_cik=manifest.archive_cik,
        issuer_cik=issuer_cik,
        issuer_name=issuer_name,
        issuer_trading_symbol_raw=issuer_symbol,
        document_type=document_type,
        period_of_report=period_of_report,
        date_of_original_submission=date_of_original_submission,
        owners=owners,
        nonderivative_transactions=rows,
        derivative_transaction_count=derivative_transaction_count,
        footnotes=footnotes,
        source_quality_flags=all_flags,
        accepted_at_raw=retrieval.accepted_at_raw,
        accepted_at_source=retrieval.accepted_at_source,
        accepted_at_utc=retrieval.accepted_at_utc,
        first_seen_at_utc=first_seen_at_utc,
    )


def write_form4_retrieval_artifacts(
    *, root: Path, manifest: Form4ManifestRow, retrieval: Form4RetrievalResult
) -> list[Path]:
    """Write raw Form 4 retrieval artifacts under the NAS-style raw layout."""
    target = (
        Path(root)
        / "data"
        / "raw"
        / "sec"
        / "archives"
        / f"archive_cik={manifest.archive_cik}"
        / f"accession={manifest.accession_no_dashes}"
    )
    target.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    if retrieval.ownership_xml is not None:
        primary = target / "primary.xml"
        primary.write_text(retrieval.ownership_xml, encoding="utf-8")
        paths.append(primary)
    if retrieval.complete_txt is not None:
        complete = target / "complete.txt"
        complete.write_text(retrieval.complete_txt, encoding="utf-8")
        paths.append(complete)
    metadata = target / "metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "manifest": manifest.to_dict(),
                "retrieval": retrieval.metadata.to_dict(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    paths.append(metadata)
    return paths


def write_form4_parse_result(*, root: Path, result: Form4ParseResult) -> list[Path]:
    """Write one curated Form 4 parse output under the NAS-style curated layout."""
    return write_form4_parse_results(root=root, results=[result])


def write_form4_parse_results(
    *, root: Path, results: list[Form4ParseResult]
) -> list[Path]:
    """Write curated Form 4 parse outputs under the NAS-style curated layout."""
    base = Path(root) / "data" / "curated" / "sec" / "form4"
    paths = [
        base / "submissions" / "data.parquet",
        base / "reporting_owners" / "data.parquet",
        base / "nonderiv_transactions" / "data.parquet",
        base / "deriv_transactions" / "data.parquet",
        base / "footnotes" / "data.parquet",
    ]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    _write_merged_parquet(
        paths[0],
        pd.DataFrame(
        [
            {
                "accession": item.accession,
                "archive_cik": item.archive_cik,
                "issuer_cik": item.issuer_cik,
                "issuer_name": item.issuer_name,
                "issuer_trading_symbol_raw": item.issuer_trading_symbol_raw,
                "document_type": item.document_type,
                "period_of_report": item.period_of_report,
                "date_of_original_submission": item.date_of_original_submission,
                "accepted_at_raw": item.accepted_at_raw,
                "accepted_at_source": item.accepted_at_source,
                "accepted_at_utc": (
                    item.accepted_at_utc.isoformat()
                    if item.accepted_at_utc
                    else None
                ),
                "first_seen_at_utc": (
                    item.first_seen_at_utc.isoformat()
                    if item.first_seen_at_utc
                    else None
                ),
                "derivative_transaction_count": item.derivative_transaction_count,
                "source_quality_flags": item.source_quality_flags,
            }
            for item in results
        ]
        ),
        subset=("accession",),
    )
    _write_merged_parquet(
        paths[1],
        pd.DataFrame(
        [
            owner.to_dict(accession=item.accession)
            for item in results
            for owner in item.owners
        ]
        ),
        subset=("accession", "owner_cik", "owner_name"),
    )
    _write_merged_parquet(
        paths[2],
        pd.DataFrame(
        [
            row.to_dict()
            for item in results
            for row in item.nonderivative_transactions
        ]
        ),
        subset=(
            "accession",
            "reporting_owner_cik",
            "transaction_date",
            "security_title",
            "transaction_code",
            "acquired_disposed",
            "transaction_shares",
            "transaction_price",
        ),
    )
    _write_merged_parquet(
        paths[3],
        pd.DataFrame(
        [
            {
                "accession": item.accession,
                "derivative_transaction_count": item.derivative_transaction_count,
            }
            for item in results
        ]
        ),
        subset=("accession",),
    )
    _write_merged_parquet(
        paths[4],
        pd.DataFrame(
        [
            {"accession": item.accession, "footnote_id": key, "text": value}
            for item in results
            for key, value in sorted(item.footnotes.items())
        ]
        ),
        subset=("accession", "footnote_id"),
    )
    return paths


def _parse_nonderivative_transaction(
    tx: ET.Element,
    *,
    manifest: Form4ManifestRow,
    retrieval: Form4RetrievalMetadata,
    document_type: str,
    issuer_cik: str,
    issuer_name: str,
    issuer_symbol: str,
    owner: Form4Owner | None,
    owner_cik_set: tuple[str, ...],
    period_of_report: str | None,
    date_of_original_submission: str | None,
    not_subject: bool,
    same_filing_has_sales: bool,
    derivative_p_present: bool,
    document_flags: list[str],
    remarks: str,
    footnotes_text: str | None,
    first_seen_at_utc: datetime | None,
    raw_xml_path: str | None,
    complete_txt_path: str | None,
) -> Form4TransactionRow:
    security_title = _value_text(tx, ("securityTitle",)) or ""
    transaction_date_raw = _value_text(tx, ("transactionDate",))
    transaction_date = _date_only(transaction_date_raw)
    transaction_form_type = _text(
        tx, ("transactionCoding", "transactionFormType")
    ) or ""
    transaction_code = (
        _text(tx, ("transactionCoding", "transactionCode")) or ""
    ).upper()
    transaction_shares = _decimal(_value_text(tx, ("transactionAmounts", "transactionShares")))
    transaction_price = _decimal(
        _value_text(tx, ("transactionAmounts", "transactionPricePerShare"))
    )
    acquired_disposed = (
        _value_text(tx, ("transactionAmounts", "transactionAcquiredDisposedCode"))
        or ""
    ).upper()
    post_transaction_shares = _decimal(
        _value_text(
            tx,
            (
                "postTransactionAmounts",
                "sharesOwnedFollowingTransaction",
            ),
        )
    )
    direct_or_indirect = _value_text(
        tx, ("ownershipNature", "directOrIndirectOwnership")
    )
    ownership_nature = _value_text(tx, ("ownershipNature", "natureOfOwnership"))
    transaction_value = (
        transaction_shares * transaction_price
        if transaction_shares is not None and transaction_price is not None
        else None
    )
    field_footnote_ids = {
        "security_title": _footnote_ids(_node(tx, ("securityTitle",))),
        "transaction_date": _footnote_ids(_node(tx, ("transactionDate",))),
        "transaction_shares": _footnote_ids(
            _node(tx, ("transactionAmounts", "transactionShares"))
        ),
        "transaction_price": _footnote_ids(
            _node(tx, ("transactionAmounts", "transactionPricePerShare"))
        ),
    }
    security_title_normalized = _normalize_security_title(security_title)
    context_text = " ".join(
        part for part in (security_title, remarks, footnotes_text or "") if part
    )
    private_or_unit = _contains_any(context_text, PRIVATE_PURCHASE_TERMS)
    common_stock = _common_stock_filter_pass(security_title_normalized)
    flags = list(document_flags)
    if same_filing_has_sales:
        flags.append("mixed_p_and_s")
    if derivative_p_present:
        flags.append("derivative_p_present")
    if private_or_unit:
        flags.append("private_or_unit_purchase_flag")
    if transaction_price is None:
        flags.append("missing_price")
    elif transaction_price <= 0:
        flags.append("zero_price")
    if not common_stock:
        flags.append("non_common_security_title")
    late_report = _late_report(manifest.filed_date, transaction_date)
    if late_report:
        flags.append("late_report")
    if transaction_code == "P" and acquired_disposed == "A" and not same_filing_has_sales:
        same_owner_same_day_has_sales = False
    else:
        same_owner_same_day_has_sales = same_filing_has_sales
    primary_signal_eligible = (
        document_type == "4"
        and transaction_form_type == "4"
        and transaction_code == "P"
        and acquired_disposed == "A"
        and transaction_shares is not None
        and transaction_shares > 0
        and transaction_price is not None
        and transaction_price > 0
        and common_stock
        and not private_or_unit
        and not not_subject
        and not same_filing_has_sales
        and not late_report
    )
    return Form4TransactionRow(
        accession=manifest.accession,
        accession_no_dashes=manifest.accession_no_dashes,
        archive_cik=manifest.archive_cik,
        index_filename=manifest.index_filename,
        manifest_source=manifest.discovery_source,
        form_type=manifest.form,
        document_type=document_type,
        issuer_cik=issuer_cik,
        issuer_name=issuer_name,
        issuer_trading_symbol_raw=issuer_symbol,
        owner_cik_set=owner_cik_set,
        reporting_owner_cik=owner.owner_cik if owner else None,
        reporting_owner_name=owner.owner_name if owner else None,
        is_director=owner.is_director if owner else False,
        is_officer=owner.is_officer if owner else False,
        is_ten_percent_owner=owner.is_ten_percent_owner if owner else False,
        is_other=owner.is_other if owner else False,
        officer_title=owner.officer_title if owner else None,
        accepted_at_raw=retrieval.accepted_at_raw,
        accepted_at_source=retrieval.accepted_at_source,
        accepted_at_et=retrieval.accepted_at_et,
        accepted_at_utc=retrieval.accepted_at_utc,
        first_seen_at_utc=first_seen_at_utc,
        filed_date=manifest.filed_date,
        period_of_report=period_of_report,
        date_of_original_submission=date_of_original_submission,
        transaction_date=transaction_date,
        transaction_date_raw=transaction_date_raw,
        security_title=security_title,
        security_title_normalized=security_title_normalized,
        transaction_code=transaction_code,
        transaction_form_type=transaction_form_type,
        acquired_disposed=acquired_disposed,
        transaction_shares=transaction_shares,
        transaction_price=transaction_price,
        transaction_value=transaction_value,
        post_transaction_shares=post_transaction_shares,
        direct_or_indirect=direct_or_indirect,
        ownership_nature=ownership_nature,
        field_footnote_ids=field_footnote_ids,
        footnotes_text=footnotes_text,
        probably_private_or_unit_purchase=private_or_unit,
        same_filing_has_sales=same_filing_has_sales,
        same_owner_same_day_has_sales=same_owner_same_day_has_sales,
        primary_document=None,
        raw_xml_url=retrieval.primary_xml_url,
        raw_xml_path=raw_xml_path,
        raw_xml_hash=retrieval.primary_xml_sha256,
        complete_txt_url=retrieval.complete_txt_url,
        complete_txt_path=complete_txt_path,
        complete_txt_hash=retrieval.complete_txt_sha256,
        primary_xml_source=retrieval.xml_source,
        parser_version=PARSER_VERSION,
        schema_version=SCHEMA_VERSION,
        source_quality_flags=sorted(set(flags)),
        primary_signal_eligible=primary_signal_eligible,
    )


def _write_merged_parquet(
    path: Path,
    frame: pd.DataFrame,
    *,
    subset: tuple[str, ...],
) -> None:
    """Write a parquet frame while preserving prior accessions without duplicates."""
    if path.exists():
        existing = pd.read_parquet(path)
        frame = pd.concat([existing, frame], ignore_index=True)
    existing_subset = [column for column in subset if column in frame.columns]
    if existing_subset:
        frame = frame.drop_duplicates(subset=existing_subset, keep="last")
    else:
        frame = frame.drop_duplicates()
    frame.to_parquet(path, index=False)


def _parse_owners(root: ET.Element) -> list[Form4Owner]:
    owners: list[Form4Owner] = []
    for node in _children(root, "reportingOwner"):
        owners.append(
            Form4Owner(
                owner_cik=_text(
                    node, ("reportingOwnerId", "rptOwnerCik")
                )
                or "",
                owner_name=_text(
                    node, ("reportingOwnerId", "rptOwnerName")
                )
                or "",
                is_director=_parse_bool(
                    _text(
                        node,
                        ("reportingOwnerRelationship", "isDirector"),
                    )
                ),
                is_officer=_parse_bool(
                    _text(node, ("reportingOwnerRelationship", "isOfficer"))
                ),
                is_ten_percent_owner=_parse_bool(
                    _text(
                        node,
                        ("reportingOwnerRelationship", "isTenPercentOwner"),
                    )
                ),
                is_other=_parse_bool(
                    _text(node, ("reportingOwnerRelationship", "isOther"))
                ),
                officer_title=_text(
                    node, ("reportingOwnerRelationship", "officerTitle")
                ),
            )
        )
    return owners


def _parse_footnotes(root: ET.Element) -> dict[str, str]:
    footnotes: dict[str, str] = {}
    for node in root.iter():
        if _local_name(node.tag) != "footnote":
            continue
        footnote_id = node.attrib.get("id")
        if footnote_id:
            footnotes[footnote_id] = " ".join("".join(node.itertext()).split())
    return footnotes


def _ownership_xml_fragment(text: str) -> str:
    start = text.find("<ownershipDocument")
    if start < 0:
        raise ValueError("missing ownershipDocument")
    end_marker = "</ownershipDocument>"
    end = text.find(end_marker, start)
    if end < 0:
        raise ValueError("unterminated ownershipDocument")
    return text[start : end + len(end_marker)]


def _node(node: ET.Element | None, path: tuple[str, ...]) -> ET.Element | None:
    current = node
    for part in path:
        if current is None:
            return None
        current = _first_child(current, part)
    return current


def _text(node: ET.Element | None, path: tuple[str, ...]) -> str | None:
    target = _node(node, path)
    if target is None or target.text is None:
        return None
    value = target.text.strip()
    return value or None


def _value_text(node: ET.Element, path: tuple[str, ...]) -> str | None:
    wrapper = _node(node, path)
    return _text(wrapper, ("value",))


def _children(node: ET.Element | None, name: str) -> list[ET.Element]:
    if node is None:
        return []
    return [child for child in list(node) if _local_name(child.tag) == name]


def _first_child(node: ET.Element, name: str) -> ET.Element | None:
    for child in list(node):
        if _local_name(child.tag) == name:
            return child
    return None


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _footnote_ids(node: ET.Element | None) -> list[str]:
    if node is None:
        return []
    ids: list[str] = []
    for child in node.iter():
        if _local_name(child.tag) == "footnoteId" and child.attrib.get("id"):
            ids.append(str(child.attrib["id"]))
    return ids


def _parse_bool(raw: str | None) -> bool:
    if raw is None:
        return False
    return str(raw).strip().lower() in {"1", "true", "yes"}


def _decimal(raw: str | None) -> Decimal | None:
    if raw is None:
        return None
    text = str(raw).strip().replace(",", "")
    if not text:
        return None
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def _decimal_to_string(value: Decimal | None) -> str | None:
    return str(value) if value is not None else None


def _date_only(raw: str | None) -> str | None:
    if raw is None:
        return None
    match = re.match(r"(\d{4}-\d{2}-\d{2})", raw.strip())
    return match.group(1) if match else raw.strip()[:10]


def _normalize_security_title(title: str) -> str:
    return " ".join(title.lower().replace("-", " ").split())


def _common_stock_filter_pass(normalized_title: str) -> bool:
    if any(term in normalized_title for term in SECURITY_EXCLUDE_TERMS):
        return False
    return any(term in normalized_title for term in COMMON_LIKE_ALLOW)


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    lower = text.lower()
    return any(term in lower for term in terms)


def _late_report(filed_date: str, transaction_date: str | None) -> bool:
    if transaction_date is None:
        return False
    try:
        filed = datetime.fromisoformat(filed_date[:10]).date()
        transacted = datetime.fromisoformat(transaction_date[:10]).date()
    except ValueError:
        return False
    return (filed - transacted).days > 10
