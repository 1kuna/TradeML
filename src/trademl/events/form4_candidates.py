"""SEC Form 4 open-market insider-buy candidate event curation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path
import re

import pandas as pd

from trademl.events.form4 import Form4ParseResult


EVENT_TYPE = "FORM4_OPEN_MARKET_INSIDER_BUY"
EVENT_SCHEMA_VERSION = "form4_candidate_event_v1"


@dataclass(slots=True, frozen=True)
class Form4InsiderPurchaseEvent:
    """One issuer/accession-level Form 4 insider-buy candidate event."""

    event_id: str
    issuer_cik: str
    ticker: str
    primary_security_id: str
    accessions: tuple[str, ...]
    event_type: str
    accepted_at_utc: str | None
    first_seen_at_utc: str | None
    tradable_at_utc: str | None
    n_insiders_buying: int
    n_directors_buying: int
    n_officers_buying: int
    ceo_buy: bool
    cfo_buy: bool
    ten_percent_owner_buy: bool
    total_shares_bought: str
    total_dollar_value: str
    total_dollar_value_float: float
    max_single_purchase_value: str
    median_purchase_price: str | None
    purchase_value_to_market_cap: str | None
    purchase_value_to_adv20: str | None
    purchase_value_to_prior_holdings: str | None
    days_since_transaction: int | None
    cluster_7d_purchase_value: str
    cluster_30d_purchase_value: str
    prior_90d_net_insider_purchase_value: str
    prior_20d_return: float | None
    prior_60d_return: float | None
    prior_20d_vol: float | None
    prior_20d_adv: float | None
    market_cap_bucket: str | None
    liquidity_bucket: str | None
    news_nearby_flag: bool
    earnings_nearby_flag: bool
    offering_nearby_flag: bool
    mna_nearby_flag: bool
    eligibility_pass: bool
    exclusion_reasons: tuple[str, ...]
    event_strength_score: float
    parse_confidence: float
    source_quality_score: float
    source_quality_flags: tuple[str, ...]
    owner_ciks: tuple[str, ...]
    owner_names: tuple[str, ...]
    source_transaction_count: int
    eligible_transaction_count: int
    schema_version: str

    def to_dict(self) -> dict[str, object]:
        """Return a parquet/JSON-safe representation."""
        return {
            "event_id": self.event_id,
            "issuer_cik": self.issuer_cik,
            "ticker": self.ticker,
            "primary_security_id": self.primary_security_id,
            "accessions": list(self.accessions),
            "event_type": self.event_type,
            "accepted_at_utc": self.accepted_at_utc,
            "first_seen_at_utc": self.first_seen_at_utc,
            "tradable_at_utc": self.tradable_at_utc,
            "n_insiders_buying": self.n_insiders_buying,
            "n_directors_buying": self.n_directors_buying,
            "n_officers_buying": self.n_officers_buying,
            "ceo_buy": self.ceo_buy,
            "cfo_buy": self.cfo_buy,
            "ten_percent_owner_buy": self.ten_percent_owner_buy,
            "total_shares_bought": self.total_shares_bought,
            "total_dollar_value": self.total_dollar_value,
            "total_dollar_value_float": self.total_dollar_value_float,
            "max_single_purchase_value": self.max_single_purchase_value,
            "median_purchase_price": self.median_purchase_price,
            "purchase_value_to_market_cap": self.purchase_value_to_market_cap,
            "purchase_value_to_adv20": self.purchase_value_to_adv20,
            "purchase_value_to_prior_holdings": self.purchase_value_to_prior_holdings,
            "days_since_transaction": self.days_since_transaction,
            "cluster_7d_purchase_value": self.cluster_7d_purchase_value,
            "cluster_30d_purchase_value": self.cluster_30d_purchase_value,
            "prior_90d_net_insider_purchase_value": self.prior_90d_net_insider_purchase_value,
            "prior_20d_return": self.prior_20d_return,
            "prior_60d_return": self.prior_60d_return,
            "prior_20d_vol": self.prior_20d_vol,
            "prior_20d_adv": self.prior_20d_adv,
            "market_cap_bucket": self.market_cap_bucket,
            "liquidity_bucket": self.liquidity_bucket,
            "news_nearby_flag": self.news_nearby_flag,
            "earnings_nearby_flag": self.earnings_nearby_flag,
            "offering_nearby_flag": self.offering_nearby_flag,
            "mna_nearby_flag": self.mna_nearby_flag,
            "eligibility_pass": self.eligibility_pass,
            "exclusion_reasons": list(self.exclusion_reasons),
            "event_strength_score": self.event_strength_score,
            "parse_confidence": self.parse_confidence,
            "source_quality_score": self.source_quality_score,
            "source_quality_flags": list(self.source_quality_flags),
            "owner_ciks": list(self.owner_ciks),
            "owner_names": list(self.owner_names),
            "source_transaction_count": self.source_transaction_count,
            "eligible_transaction_count": self.eligible_transaction_count,
            "schema_version": self.schema_version,
        }


def build_form4_candidate_events_from_parse_results(
    results: list[Form4ParseResult],
) -> list[Form4InsiderPurchaseEvent]:
    """Build issuer/accession candidate events from parsed Form 4 documents."""
    submissions = pd.DataFrame(
        [
            {
                "accession": result.accession,
                "issuer_cik": result.issuer_cik,
                "issuer_name": result.issuer_name,
                "issuer_trading_symbol_raw": result.issuer_trading_symbol_raw,
                "document_type": result.document_type,
                "accepted_at_raw": result.accepted_at_raw,
                "accepted_at_source": result.accepted_at_source,
                "accepted_at_utc": (
                    result.accepted_at_utc.isoformat()
                    if result.accepted_at_utc
                    else None
                ),
                "first_seen_at_utc": (
                    result.first_seen_at_utc.isoformat()
                    if result.first_seen_at_utc
                    else None
                ),
                "derivative_transaction_count": result.derivative_transaction_count,
                "source_quality_flags": result.source_quality_flags,
            }
            for result in results
        ]
    )
    transactions = pd.DataFrame(
        [
            row.to_dict()
            for result in results
            for row in result.nonderivative_transactions
        ]
    )
    return build_form4_candidate_events(submissions=submissions, transactions=transactions)


def build_form4_candidate_events_from_curated(
    *, root: Path,
) -> list[Form4InsiderPurchaseEvent]:
    """Build Form 4 candidate events from curated parser parquet outputs."""
    base = Path(root) / "data" / "curated" / "sec" / "form4"
    submissions_path = base / "submissions" / "data.parquet"
    transactions_path = base / "nonderiv_transactions" / "data.parquet"
    if not submissions_path.exists():
        raise FileNotFoundError(f"missing Form 4 submissions parquet: {submissions_path}")
    submissions = pd.read_parquet(submissions_path)
    transactions = (
        pd.read_parquet(transactions_path)
        if transactions_path.exists()
        else pd.DataFrame()
    )
    return build_form4_candidate_events(submissions=submissions, transactions=transactions)


def build_form4_candidate_events(
    *,
    submissions: pd.DataFrame,
    transactions: pd.DataFrame,
) -> list[Form4InsiderPurchaseEvent]:
    """Build deterministic Form 4 issuer/accession candidate event rows."""
    if submissions.empty:
        return []
    if transactions.empty:
        transactions = pd.DataFrame(columns=["accession"])
    events: list[Form4InsiderPurchaseEvent] = []
    for submission in submissions.sort_values("accession").to_dict("records"):
        accession = str(submission["accession"])
        rows = transactions[transactions["accession"] == accession].copy()
        events.append(_candidate_from_group(submission=submission, rows=rows))
    return events


def write_form4_candidate_events(
    *,
    root: Path,
    events: list[Form4InsiderPurchaseEvent],
) -> dict[str, object]:
    """Write candidate events and a compact curation report."""
    root = Path(root)
    target = (
        root
        / "data"
        / "curated"
        / "events"
        / "form4_open_market_buy_candidates"
    )
    target.mkdir(parents=True, exist_ok=True)
    data_path = target / "data.parquet"
    pd.DataFrame([event.to_dict() for event in events]).to_parquet(
        data_path, index=False
    )
    report = summarize_form4_candidate_events(events=events)
    report["artifact"] = str(data_path)
    report_target = (
        root
        / "control"
        / "cluster"
        / "state"
        / "research"
        / "form4_candidate_events"
    )
    history = report_target / "history"
    history.mkdir(parents=True, exist_ok=True)
    latest = report_target / "latest.json"
    latest.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    timestamp = str(report["checked_at"]).replace(":", "").replace("+", "_")
    (history / f"{timestamp}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {"events_path": str(data_path), "report_path": str(latest), "report": report}


def run_form4_candidate_curation(*, data_root: Path) -> dict[str, object]:
    """Build and persist Form 4 candidate events from curated parser outputs."""
    events = build_form4_candidate_events_from_curated(root=data_root)
    return write_form4_candidate_events(root=data_root, events=events)


def summarize_form4_candidate_events(
    *, events: list[Form4InsiderPurchaseEvent]
) -> dict[str, object]:
    """Summarize Form 4 candidate curation output for operator review."""
    eligible = [event for event in events if event.eligibility_pass]
    reason_counts: dict[str, int] = {}
    for event in events:
        for reason in event.exclusion_reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return {
        "version": EVENT_SCHEMA_VERSION,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "candidate_count": len(events),
        "eligible_count": len(eligible),
        "excluded_count": len(events) - len(eligible),
        "exclusion_reason_counts": dict(sorted(reason_counts.items())),
        "total_eligible_dollar_value": _decimal_to_string(
            sum((_decimal(event.total_dollar_value) or Decimal("0")) for event in eligible)
        ),
    }


def build_form4_candidate_fixture_gate(
    *,
    events: list[Form4InsiderPurchaseEvent],
    expectations: dict[str, dict[str, int | None]],
) -> dict[str, object]:
    """Validate candidate event eligibility expectations by accession."""
    events_by_accession = {
        event.accessions[0]: event for event in events if len(event.accessions) == 1
    }
    fixtures: list[dict[str, object]] = []
    for accession, expected in sorted(expectations.items()):
        event = events_by_accession.get(accession)
        eligible_count = 1 if event and event.eligibility_pass else 0
        errors: list[str] = []
        expected_min = expected.get("min")
        expected_max = expected.get("max")
        if expected_min is not None and eligible_count < int(expected_min):
            errors.append(
                f"candidate_eligible_below_min:min={expected_min}:actual={eligible_count}"
            )
        if expected_max is not None and eligible_count > int(expected_max):
            errors.append(
                f"candidate_eligible_above_max:max={expected_max}:actual={eligible_count}"
            )
        fixtures.append(
            {
                "accession": accession,
                "status": "PASS" if not errors else "FAIL",
                "eligible_count": eligible_count,
                "event_id": event.event_id if event else None,
                "exclusion_reasons": list(event.exclusion_reasons) if event else [],
                "errors": errors,
            }
        )
    failed = sum(1 for item in fixtures if item["status"] != "PASS")
    return {
        "verdict": "PASS" if failed == 0 else "FAIL",
        "fixture_count": len(fixtures),
        "passed": len(fixtures) - failed,
        "failed": failed,
        "fixtures": fixtures,
    }


def _candidate_from_group(
    *, submission: dict[str, object], rows: pd.DataFrame
) -> Form4InsiderPurchaseEvent:
    accession = str(submission["accession"])
    issuer_cik = str(submission.get("issuer_cik") or "")
    ticker = _normalize_ticker(str(submission.get("issuer_trading_symbol_raw") or ""))
    source_flags = _submission_flags(submission)
    source_transaction_count = int(len(rows))
    if not rows.empty:
        source_flags.update(
            flag
            for raw_flags in rows.get("source_quality_flags", [])
            for flag in _list(raw_flags)
        )
    eligible_rows = (
        rows[rows["primary_signal_eligible"].fillna(False).astype(bool)].copy()
        if "primary_signal_eligible" in rows
        else rows.iloc[0:0].copy()
    )
    exclusion_reasons = _exclusion_reasons(
        submission=submission,
        rows=rows,
        eligible_rows=eligible_rows,
        source_flags=source_flags,
        ticker=ticker,
    )
    total_shares = _sum_decimal(eligible_rows, "transaction_shares")
    total_value = _sum_decimal(eligible_rows, "transaction_value")
    max_value = _max_decimal(eligible_rows, "transaction_value")
    median_price = _median_decimal(eligible_rows, "transaction_price")
    owner_ciks = _owner_ciks(eligible_rows)
    owner_names = _owner_names(eligible_rows)
    accepted_at = _min_text(eligible_rows, "accepted_at_utc") or _text_value(
        submission.get("accepted_at_utc")
    )
    first_seen = (
        _min_text(eligible_rows, "first_seen_at_utc")
        or _text_value(submission.get("first_seen_at_utc"))
        or accepted_at
    )
    event_id = _event_id(issuer_cik=issuer_cik, accession=accession)
    ceo_buy = _title_contains(eligible_rows, ("ceo", "chief executive"))
    cfo_buy = _title_contains(eligible_rows, ("cfo", "chief financial"))
    n_directors = _unique_role_count(eligible_rows, "is_director")
    n_officers = _unique_role_count(eligible_rows, "is_officer")
    ten_percent_owner_buy = _any_bool(eligible_rows, "is_ten_percent_owner")
    event_strength_score = _event_strength_score(
        total_value=total_value,
        n_insiders=len(owner_ciks),
        ceo_buy=ceo_buy,
        cfo_buy=cfo_buy,
        director_buy=n_directors > 0,
    )
    return Form4InsiderPurchaseEvent(
        event_id=event_id,
        issuer_cik=issuer_cik,
        ticker=ticker,
        primary_security_id=f"{issuer_cik}:{ticker}" if issuer_cik and ticker else "",
        accessions=(accession,),
        event_type=EVENT_TYPE,
        accepted_at_utc=accepted_at,
        first_seen_at_utc=first_seen,
        tradable_at_utc=None,
        n_insiders_buying=len(owner_ciks),
        n_directors_buying=n_directors,
        n_officers_buying=n_officers,
        ceo_buy=ceo_buy,
        cfo_buy=cfo_buy,
        ten_percent_owner_buy=ten_percent_owner_buy,
        total_shares_bought=_decimal_to_string(total_shares),
        total_dollar_value=_decimal_to_string(total_value),
        total_dollar_value_float=float(total_value),
        max_single_purchase_value=_decimal_to_string(max_value),
        median_purchase_price=_decimal_to_string(median_price) if median_price is not None else None,
        purchase_value_to_market_cap=None,
        purchase_value_to_adv20=None,
        purchase_value_to_prior_holdings=None,
        days_since_transaction=_days_since_transaction(eligible_rows),
        cluster_7d_purchase_value=_decimal_to_string(total_value),
        cluster_30d_purchase_value=_decimal_to_string(total_value),
        prior_90d_net_insider_purchase_value="0",
        prior_20d_return=None,
        prior_60d_return=None,
        prior_20d_vol=None,
        prior_20d_adv=None,
        market_cap_bucket=None,
        liquidity_bucket=None,
        news_nearby_flag=False,
        earnings_nearby_flag=False,
        offering_nearby_flag=False,
        mna_nearby_flag=False,
        eligibility_pass=not exclusion_reasons,
        exclusion_reasons=tuple(sorted(exclusion_reasons)),
        event_strength_score=event_strength_score,
        parse_confidence=_parse_confidence(submission=submission, rows=rows),
        source_quality_score=_source_quality_score(source_flags),
        source_quality_flags=tuple(sorted(source_flags)),
        owner_ciks=tuple(owner_ciks),
        owner_names=tuple(owner_names),
        source_transaction_count=source_transaction_count,
        eligible_transaction_count=int(len(eligible_rows)),
        schema_version=EVENT_SCHEMA_VERSION,
    )


def _submission_flags(submission: dict[str, object]) -> set[str]:
    return set(_list(submission.get("source_quality_flags")))


def _exclusion_reasons(
    *,
    submission: dict[str, object],
    rows: pd.DataFrame,
    eligible_rows: pd.DataFrame,
    source_flags: set[str],
    ticker: str,
) -> set[str]:
    reasons: set[str] = set()
    document_type = str(submission.get("document_type") or "")
    if document_type != "4":
        reasons.add("amendment")
    if not ticker:
        reasons.add("missing_ticker")
    elif _is_invalid_tradable_ticker(ticker):
        reasons.add("ambiguous_or_invalid_ticker")
    if _is_otc_symbol(ticker):
        reasons.add("otc_symbol")
    flag_reason_map = {
        "private_or_unit_purchase_flag": "private_or_unit_purchase",
        "mixed_p_and_s": "mixed_p_and_s",
        "late_report": "late_report",
        "not_subject_to_section16": "not_subject_to_section16",
        "missing_price": "missing_price",
        "zero_price": "zero_price",
        "non_common_security_title": "non_common_security_title",
    }
    for flag, reason in flag_reason_map.items():
        if flag in source_flags:
            reasons.add(reason)
    if "derivative_p_present" in source_flags and eligible_rows.empty:
        reasons.add("derivative_p_only_or_contaminated")
    if rows.empty:
        reasons.add("no_nonderivative_transactions")
    if eligible_rows.empty:
        reasons.add("no_strict_open_market_buy_rows")
    if not _has_accepted_at(rows) and not _text_value(submission.get("accepted_at_utc")):
        reasons.add("missing_accepted_at")
    return reasons


def _sum_decimal(rows: pd.DataFrame, column: str) -> Decimal:
    total = Decimal("0")
    if column not in rows:
        return total
    for value in rows[column].tolist():
        total += _decimal(value) or Decimal("0")
    return total


def _max_decimal(rows: pd.DataFrame, column: str) -> Decimal:
    values = [_decimal(value) for value in rows.get(column, pd.Series(dtype=object)).tolist()]
    values = [value for value in values if value is not None]
    return max(values) if values else Decimal("0")


def _median_decimal(rows: pd.DataFrame, column: str) -> Decimal | None:
    values = [_decimal(value) for value in rows.get(column, pd.Series(dtype=object)).tolist()]
    values = sorted(value for value in values if value is not None)
    if not values:
        return None
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return (values[middle - 1] + values[middle]) / Decimal("2")


def _decimal(value: object) -> Decimal | None:
    if value is None or pd.isna(value):
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def _decimal_to_string(value: Decimal) -> str:
    if not value:
        return "0"
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _owner_ciks(rows: pd.DataFrame) -> list[str]:
    owner_ciks: set[str] = set()
    for value in rows.get("owner_cik_set", pd.Series(dtype=object)).tolist():
        owner_ciks.update(str(item) for item in _list(value) if str(item))
    return sorted(owner_ciks)


def _owner_names(rows: pd.DataFrame) -> list[str]:
    names = {
        str(value)
        for value in rows.get("reporting_owner_name", pd.Series(dtype=object)).tolist()
        if value is not None and not pd.isna(value) and str(value)
    }
    return sorted(names)


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
    if pd.isna(value):
        return []
    return [value]


def _min_text(rows: pd.DataFrame, column: str) -> str | None:
    if column not in rows:
        return None
    values = sorted(str(value) for value in rows[column].tolist() if value and not pd.isna(value))
    return values[0] if values else None


def _text_value(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value)
    return text if text else None


def _title_contains(rows: pd.DataFrame, needles: tuple[str, ...]) -> bool:
    if "officer_title" not in rows:
        return False
    for value in rows["officer_title"].tolist():
        text = "" if value is None or pd.isna(value) else str(value).lower()
        if any(needle in text for needle in needles):
            return True
    return False


def _unique_role_count(rows: pd.DataFrame, column: str) -> int:
    if rows.empty or column not in rows:
        return 0
    subset = rows[rows[column].fillna(False).astype(bool)]
    return len(_owner_ciks(subset))


def _any_bool(rows: pd.DataFrame, column: str) -> bool:
    return bool((column in rows) and rows[column].fillna(False).astype(bool).any())


def _event_strength_score(
    *,
    total_value: Decimal,
    n_insiders: int,
    ceo_buy: bool,
    cfo_buy: bool,
    director_buy: bool,
) -> float:
    return round(
        math.log1p(float(total_value))
        + 0.75 * math.log1p(n_insiders)
        + (0.5 if ceo_buy else 0.0)
        + (0.5 if cfo_buy else 0.0)
        + (0.25 if director_buy else 0.0),
        6,
    )


def _parse_confidence(*, submission: dict[str, object], rows: pd.DataFrame) -> float:
    score = 1.0
    if not str(submission.get("issuer_cik") or ""):
        score -= 0.25
    if not str(submission.get("issuer_trading_symbol_raw") or ""):
        score -= 0.25
    if not _has_accepted_at(rows):
        score -= 0.25
    return max(0.0, round(score, 3))


def _source_quality_score(flags: set[str]) -> float:
    penalty = min(0.9, 0.1 * len(flags))
    return round(1.0 - penalty, 3)


def _days_since_transaction(rows: pd.DataFrame) -> int | None:
    accepted = _min_text(rows, "accepted_at_utc")
    if accepted is None or "transaction_date" not in rows:
        return None
    transaction_dates = [
        str(value)[:10]
        for value in rows["transaction_date"].tolist()
        if value is not None and not pd.isna(value)
    ]
    if not transaction_dates:
        return None
    try:
        accepted_date = datetime.fromisoformat(accepted.replace("Z", "+00:00")).date()
        first_transaction = datetime.fromisoformat(min(transaction_dates)).date()
    except ValueError:
        return None
    return (accepted_date - first_transaction).days


def _is_otc_symbol(ticker: str) -> bool:
    upper = ticker.upper()
    return upper.endswith(".OB") or upper.endswith(".PK") or upper.endswith("Q")


def _is_invalid_tradable_ticker(ticker: str) -> bool:
    normalized = _normalize_ticker(ticker)
    return not bool(re.fullmatch(r"[A-Z][A-Z0-9.-]{0,9}", normalized))


def _normalize_ticker(ticker: str) -> str:
    return re.sub(r"\s+", "", str(ticker or "").strip().upper())


def _has_accepted_at(rows: pd.DataFrame) -> bool:
    return _min_text(rows, "accepted_at_utc") is not None


def _event_id(*, issuer_cik: str, accession: str) -> str:
    digest = hashlib.sha256(f"{issuer_cik}:{accession}".encode("utf-8")).hexdigest()[:16]
    return f"form4_buy_{digest}"
