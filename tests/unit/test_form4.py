from __future__ import annotations

from decimal import Decimal
import json
import math
from pathlib import Path

import pandas as pd

from trademl.connectors.base import BudgetBlockedConnectorError, RemoteRateLimitConnectorError
from trademl.data_node.budgets import BudgetDecision
from trademl.events.form4 import (
    Form4ManifestRow,
    Form4RetrievalMetadata,
    Form4RetrievalResult,
    parse_form4_ownership_xml,
    write_form4_parse_result,
    write_form4_parse_results,
    write_form4_retrieval_artifacts,
)
from trademl.events.form4_candidates import (
    build_form4_candidate_events_from_parse_results,
    run_form4_candidate_curation,
)
from trademl.events.form4_fixture_gate import Form4FixtureSpec
from trademl.events.form4_fixture_gate import run_form4_fixture_gate
from trademl.events.form4_ingest import run_form4_ingest
from trademl.events.form4_labels import (
    Form4LabelConfig,
    build_form4_event_labels,
    resolve_form4_tradable_at,
    run_form4_label_curation,
)
from trademl.events.form4_market_backfill import run_form4_market_backfill
from trademl.events.form4_event_study import (
    build_form4_control_candidates_from_curated,
    run_form4_event_study,
)
from trademl.events.form4_rework import run_form4_rework_study


def _manifest(
    *,
    accession: str,
    archive_cik: str = "1018724",
    filed_date: str = "2020-04-03",
) -> Form4ManifestRow:
    return Form4ManifestRow(
        archive_cik=archive_cik,
        form="4",
        filed_date=filed_date,
        index_filename=f"edgar/data/{archive_cik}/{accession}.txt",
        accession=accession,
        accession_no_dashes=accession.replace("-", ""),
        discovery_source="sec_full_index",
        index_year=int(filed_date[:4]),
        index_quarter=2,
        index_file_hash="fixture-index-hash",
        index_crawled_at="2026-05-05T00:00:00Z",
    )


def _retrieval(
    *,
    accepted_at_raw: str = "20200403164253",
    accepted_source: str = "sgml_header",
) -> Form4RetrievalMetadata:
    return Form4RetrievalMetadata.from_accepted_raw(
        primary_xml_url="https://www.sec.gov/Archives/edgar/data/1018724/000112760220013168/primary.xml",
        primary_xml_http_status=200,
        primary_xml_sha256="xml-hash",
        complete_txt_url=None,
        complete_txt_http_status=None,
        complete_txt_sha256=None,
        xml_source="raw_primary",
        accepted_at_raw=accepted_at_raw,
        accepted_at_source=accepted_source,
        quality_flags=[],
    )


def _xml(
    *,
    accession: str,
    document_type: str = "4",
    issuer_cik: str = "1018724",
    issuer_name: str = "Amazon.com, Inc.",
    issuer_symbol: str = "AMZN",
    owners: list[tuple[str, str]] | None = None,
    relationship: str = "<isDirector>1</isDirector><isOfficer>0</isOfficer><isTenPercentOwner>0</isTenPercentOwner><isOther>0</isOther>",
    nonderiv: str = "",
    deriv: str = "",
    footnotes: str = "",
    remarks: str = "",
    period: str = "2020-04-03",
) -> str:
    owner_xml = ""
    for cik, name in owners or [("0001769274", "Indra K. Nooyi")]:
        owner_xml += f"""
  <reportingOwner>
    <reportingOwnerId>
      <rptOwnerCik>{cik}</rptOwnerCik>
      <rptOwnerName>{name}</rptOwnerName>
    </reportingOwnerId>
    <reportingOwnerRelationship>{relationship}</reportingOwnerRelationship>
  </reportingOwner>
"""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<ownershipDocument>
  <schemaVersion>X0508</schemaVersion>
  <documentType>{document_type}</documentType>
  <periodOfReport>{period}</periodOfReport>
  <issuer>
    <issuerCik>{issuer_cik}</issuerCik>
    <issuerName>{issuer_name}</issuerName>
    <issuerTradingSymbol>{issuer_symbol}</issuerTradingSymbol>
  </issuer>
  {owner_xml}
  <nonDerivativeTable>{nonderiv}</nonDerivativeTable>
  <derivativeTable>{deriv}</derivativeTable>
  <footnotes>{footnotes}</footnotes>
  <remarks>{remarks}</remarks>
</ownershipDocument>
<!-- accession {accession} -->
"""


def _nonderiv_tx(
    *,
    code: str = "P",
    acquired: str = "A",
    security: str = "Common Stock",
    date: str = "2020-04-03",
    shares: str = "10",
    price: str = "1922.6925",
    footnote_id: str | None = "F2",
    form_type: str = "4",
) -> str:
    footnote = f'<footnoteId id="{footnote_id}"/>' if footnote_id else ""
    return f"""
    <nonDerivativeTransaction>
      <securityTitle><value>{security}</value></securityTitle>
      <transactionDate><value>{date}</value></transactionDate>
      <transactionCoding>
        <transactionFormType>{form_type}</transactionFormType>
        <transactionCode>{code}</transactionCode>
        <equitySwapInvolved>0</equitySwapInvolved>
      </transactionCoding>
      <transactionAmounts>
        <transactionShares><value>{shares}</value></transactionShares>
        <transactionPricePerShare><value>{price}</value>{footnote}</transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>{acquired}</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
      <postTransactionAmounts><sharesOwnedFollowingTransaction><value>100.5</value></sharesOwnedFollowingTransaction></postTransactionAmounts>
      <ownershipNature>
        <directOrIndirectOwnership><value>D</value></directOrIndirectOwnership>
        <natureOfOwnership><value>Direct</value></natureOfOwnership>
      </ownershipNature>
    </nonDerivativeTransaction>
"""


def _deriv_tx(*, code: str = "P") -> str:
    return f"""
    <derivativeTransaction>
      <securityTitle><value>Warrant</value></securityTitle>
      <transactionCoding><transactionCode>{code}</transactionCode></transactionCoding>
    </derivativeTransaction>
"""


def test_form4_parser_preserves_decimal_footnotes_and_common_buy() -> None:
    manifest = _manifest(accession="0001127602-20-013168")
    result = parse_form4_ownership_xml(
        _xml(
            accession=manifest.accession,
            nonderiv=_nonderiv_tx(),
            footnotes='<footnote id="F2">Weighted average purchase price.</footnote>',
        ),
        manifest=manifest,
        retrieval=_retrieval(),
    )

    row = result.nonderivative_transactions[0]
    assert row.accession == "0001127602-20-013168"
    assert row.transaction_price == Decimal("1922.6925")
    assert row.transaction_value == Decimal("19226.9250")
    assert row.field_footnote_ids["transaction_price"] == ["F2"]
    assert result.footnotes["F2"] == "Weighted average purchase price."
    assert row.owner_cik_set == ("0001769274",)
    assert row.primary_signal_eligible is True
    assert "missing_price" not in row.source_quality_flags


def test_form4_parser_does_not_treat_unitized_stock_fund_as_private_purchase() -> None:
    manifest = _manifest(accession="0001250842-25-000026", archive_cik="1971213")
    result = parse_form4_ownership_xml(
        _xml(
            accession=manifest.accession,
            issuer_cik="1971213",
            issuer_name="Sinclair, Inc.",
            issuer_symbol="SBGI",
            nonderiv=_nonderiv_tx(security="Class A Common Stock", price="14.9807"),
            footnotes='<footnote id="F1">Shares are held in a 401(k) unitized stock fund.</footnote>',
        ),
        manifest=manifest,
        retrieval=_retrieval(),
    )

    row = result.nonderivative_transactions[0]
    assert row.probably_private_or_unit_purchase is False
    assert "private_or_unit_purchase_flag" not in row.source_quality_flags
    assert row.primary_signal_eligible is True


def test_form4_parser_keeps_archive_cik_separate_and_flags_mixed_sales() -> None:
    manifest = _manifest(
        accession="0000769993-15-000534",
        archive_cik="769993",
        filed_date="2015-04-15",
    )
    result = parse_form4_ownership_xml(
        _xml(
            accession=manifest.accession,
            issuer_cik="1393726",
            issuer_name="Tiptree Inc.",
            issuer_symbol="TIPT",
            owners=[
                ("0000895345", "Goldman Sachs Group Inc."),
                ("0000900583", "Goldman Sachs &amp; Co."),
            ],
            nonderiv=(
                _nonderiv_tx(date="2015-01-26-05:00", price="7.3901")
                + _nonderiv_tx(code="S", acquired="D", date="2015-01-26-05:00")
            ),
        ),
        manifest=manifest,
        retrieval=_retrieval(accepted_at_raw="20150415170104"),
    )

    buy_row = result.nonderivative_transactions[0]
    assert buy_row.archive_cik == "769993"
    assert buy_row.issuer_cik == "1393726"
    assert buy_row.transaction_date == "2015-01-26"
    assert buy_row.transaction_date_raw == "2015-01-26-05:00"
    assert buy_row.owner_cik_set == ("0000895345", "0000900583")
    assert buy_row.same_filing_has_sales is True
    assert "archive_cik_differs_from_issuer_cik" in buy_row.source_quality_flags
    assert "mixed_p_and_s" in buy_row.source_quality_flags


def test_form4_parser_parses_amendments_without_primary_signal() -> None:
    manifest = _manifest(
        accession="0001758554-19-000046",
        archive_cik="1375365",
        filed_date="2019-03-01",
    )
    result = parse_form4_ownership_xml(
        _xml(
            accession=manifest.accession,
            document_type="4/A",
            issuer_cik="1375365",
            issuer_name="Super Micro Computer, Inc.",
            issuer_symbol="SMCI",
            nonderiv=(
                _nonderiv_tx(code="M", acquired="A", price="0")
                + _nonderiv_tx(code="F", acquired="D", price="0")
            ),
            deriv=_deriv_tx(code="P"),
        ),
        manifest=manifest,
        retrieval=_retrieval(),
    )

    assert result.document_type == "4/A"
    assert result.derivative_transaction_count == 1
    assert "amendment" in result.source_quality_flags
    assert "derivative_p_present" in result.source_quality_flags
    assert all(
        row.primary_signal_eligible is False
        for row in result.nonderivative_transactions
    )


def test_form4_parser_excludes_private_unit_zero_price_and_late_reports() -> None:
    cases = [
        (
            "0001437749-25-003569",
            "2028516",
            _nonderiv_tx(security="Ordinary Shares", price="10"),
            _deriv_tx(code="P"),
            "private placement sponsor units warrant",
            {"private_or_unit_purchase_flag", "derivative_p_present"},
        ),
        (
            "0000810084-13-000003",
            "1480077",
            _nonderiv_tx(security="Warrant", price="0"),
            "",
            "",
            {"zero_price", "non_common_security_title"},
        ),
        (
            "0001528597-26-000004",
            "1569345",
            _nonderiv_tx(date="2025-01-01", price="5"),
            "",
            "",
            {"late_report"},
        ),
        (
            "0001025978-25-000011",
            "1176948",
            _nonderiv_tx(code="S", acquired="D", price="20"),
            "",
            "",
            set(),
        ),
    ]

    for accession, archive_cik, nonderiv, deriv, remarks, expected_flags in cases:
        result = parse_form4_ownership_xml(
            _xml(
                accession=accession,
                issuer_cik=archive_cik,
                nonderiv=nonderiv,
                deriv=deriv,
                remarks=remarks,
            ),
            manifest=_manifest(
                accession=accession, archive_cik=archive_cik, filed_date="2026-01-10"
            ),
            retrieval=_retrieval(),
        )
        flags = set(result.source_quality_flags) | set(
            result.nonderivative_transactions[0].source_quality_flags
        )
        assert expected_flags <= flags
        assert result.nonderivative_transactions[0].primary_signal_eligible is False


def test_form4_candidate_events_aggregate_clean_purchase() -> None:
    manifest = _manifest(accession="0001127602-20-013168")
    result = parse_form4_ownership_xml(
        _xml(
            accession=manifest.accession,
            relationship=(
                "<isDirector>1</isDirector><isOfficer>1</isOfficer>"
                "<isTenPercentOwner>0</isTenPercentOwner><isOther>0</isOther>"
                "<officerTitle>Chief Executive Officer</officerTitle>"
            ),
            nonderiv=(
                _nonderiv_tx(shares="10", price="100.25")
                + _nonderiv_tx(shares="5", price="101.75")
            ),
        ),
        manifest=manifest,
        retrieval=_retrieval(),
    )

    events = build_form4_candidate_events_from_parse_results([result])

    assert len(events) == 1
    event = events[0]
    assert event.eligibility_pass is True
    assert event.exclusion_reasons == ()
    assert event.event_type == "FORM4_OPEN_MARKET_INSIDER_BUY"
    assert event.ticker == "AMZN"
    assert event.total_shares_bought == "15"
    assert event.total_dollar_value == "1511.25"
    assert event.n_insiders_buying == 1
    assert event.n_directors_buying == 1
    assert event.n_officers_buying == 1
    assert event.ceo_buy is True
    assert event.eligible_transaction_count == 2


def test_form4_candidate_events_explain_exclusions() -> None:
    amendment_manifest = _manifest(
        accession="0001209191-06-060213",
        archive_cik="1084182",
        filed_date="2006-11-17",
    )
    amendment = parse_form4_ownership_xml(
        _xml(
            accession=amendment_manifest.accession,
            document_type="4/A",
            issuer_cik="1084182",
            issuer_symbol="IMKI.OB",
            nonderiv=_nonderiv_tx(price="0"),
        ),
        manifest=amendment_manifest,
        retrieval=_retrieval(),
    )
    derivative_manifest = _manifest(
        accession="0000810084-13-000003",
        archive_cik="1480077",
        filed_date="2013-01-04",
    )
    derivative_only = parse_form4_ownership_xml(
        _xml(
            accession=derivative_manifest.accession,
            issuer_cik="810084",
            issuer_symbol="BJCT",
            deriv=_deriv_tx(code="P"),
        ),
        manifest=derivative_manifest,
        retrieval=_retrieval(),
    )

    events = build_form4_candidate_events_from_parse_results(
        [amendment, derivative_only]
    )
    by_accession = {event.accessions[0]: event for event in events}

    amendment_event = by_accession["0001209191-06-060213"]
    assert amendment_event.eligibility_pass is False
    assert set(amendment_event.exclusion_reasons) >= {
        "amendment",
        "otc_symbol",
        "zero_price",
        "no_strict_open_market_buy_rows",
    }
    derivative_event = by_accession["0000810084-13-000003"]
    assert derivative_event.eligibility_pass is False
    assert set(derivative_event.exclusion_reasons) >= {
        "derivative_p_only_or_contaminated",
        "no_nonderivative_transactions",
        "no_strict_open_market_buy_rows",
    }
    assert "missing_accepted_at" not in derivative_event.exclusion_reasons


def test_form4_candidate_events_exclude_ambiguous_multi_symbol_tickers() -> None:
    result = parse_form4_ownership_xml(
        _xml(
            accession="0001127602-20-013168",
            issuer_symbol="WSO; WSOB",
            nonderiv=_nonderiv_tx(shares="10", price="100"),
        ),
        manifest=_manifest(accession="0001127602-20-013168"),
        retrieval=_retrieval(),
    )

    event = build_form4_candidate_events_from_parse_results([result])[0]

    assert event.eligibility_pass is False
    assert "ambiguous_or_invalid_ticker" in event.exclusion_reasons


def test_form4_raw_and_parsed_artifacts_write_expected_layout(tmp_path: Path) -> None:
    manifest = _manifest(accession="0001250842-25-000026", archive_cik="1971213")
    retrieval = Form4RetrievalResult(
        metadata=_retrieval(accepted_at_raw="20250204180102"),
        ownership_xml="<ownershipDocument />",
        complete_txt="<SEC-DOCUMENT />",
    )

    raw_paths = write_form4_retrieval_artifacts(
        root=tmp_path, manifest=manifest, retrieval=retrieval
    )

    assert (
        tmp_path
        / "data"
        / "raw"
        / "sec"
        / "archives"
        / "archive_cik=1971213"
        / "accession=000125084225000026"
        / "primary.xml"
    ) in raw_paths
    assert (raw_paths[0].parent / "metadata.json").exists()

    parse_result = parse_form4_ownership_xml(
        _xml(accession=manifest.accession, nonderiv=_nonderiv_tx()),
        manifest=manifest,
        retrieval=retrieval.metadata,
    )
    parsed_paths = write_form4_parse_result(root=tmp_path, result=parse_result)

    assert parsed_paths
    transactions = pd.read_parquet(
        tmp_path
        / "data"
        / "curated"
        / "sec"
        / "form4"
        / "nonderiv_transactions"
        / "data.parquet"
    )
    assert transactions.iloc[0]["accession"] == "0001250842-25-000026"


def test_form4_fixture_gate_writes_report_and_combined_curated_outputs(
    tmp_path: Path,
) -> None:
    preexisting_manifest = _manifest(
        accession="0001250842-25-000026",
        archive_cik="1971213",
        filed_date="2025-04-07",
    )
    preexisting = parse_form4_ownership_xml(
        _xml(
            accession=preexisting_manifest.accession,
            issuer_cik="1971213",
            issuer_name="Sinclair, Inc.",
            issuer_symbol="SBGI",
            nonderiv=_nonderiv_tx(
                security="Class A Common Stock",
                date="2025-04-07",
                price="14.9807",
            ),
        ),
        manifest=preexisting_manifest,
        retrieval=_retrieval(),
    )
    write_form4_parse_results(root=tmp_path, results=[preexisting])
    fixtures = [
        Form4FixtureSpec(
            name="clean",
            accession="0001127602-20-013168",
            archive_cik="1018724",
            filed_date="2020-04-03",
            expected_primary_eligible_min=1,
            expected_candidate_eligible_min=1,
        ),
        Form4FixtureSpec(
            name="amendment",
            accession="0001758554-19-000046",
            archive_cik="1375365",
            filed_date="2019-03-01",
            form="4/A",
            expected_document_type="4/A",
            expected_flags=("amendment",),
            expected_primary_eligible_max=0,
            expected_candidate_eligible_max=0,
        ),
    ]
    connector = _FakeForm4Connector(
        {
            "0001127602-20-013168": _xml(
                accession="0001127602-20-013168",
                nonderiv=_nonderiv_tx(),
            ),
            "0001758554-19-000046": _xml(
                accession="0001758554-19-000046",
                document_type="4/A",
                issuer_cik="1375365",
                issuer_name="Super Micro Computer, Inc.",
                issuer_symbol="SMCI",
                nonderiv=_nonderiv_tx(code="M", acquired="A", price="0"),
            ),
        }
    )

    payload = run_form4_fixture_gate(
        data_root=tmp_path,
        connector=connector,
        fixtures=fixtures,
    )

    assert payload["verdict"] == "PASS"
    assert payload["passed"] == 2
    assert payload["candidate_gate"]["verdict"] == "PASS"
    assert payload["candidate_artifacts"]["report"]["eligible_count"] == 2
    latest = (
        tmp_path
        / "control"
        / "cluster"
        / "state"
        / "research"
        / "form4_fixture_gate"
        / "latest.json"
    )
    assert latest.exists()
    persisted = pd.read_parquet(
        tmp_path
        / "data"
        / "curated"
        / "sec"
        / "form4"
        / "submissions"
        / "data.parquet"
    )
    assert set(persisted["accession"]) == {
        "0001250842-25-000026",
        "0001127602-20-013168",
        "0001758554-19-000046",
    }
    candidates = pd.read_parquet(
        tmp_path
        / "data"
        / "curated"
        / "events"
        / "form4_open_market_buy_candidates"
        / "data.parquet"
    )
    assert len(candidates) == 3
    assert set(candidates["eligibility_pass"]) == {False, True}


def test_form4_candidate_curation_rebuilds_from_curated_parse_outputs(
    tmp_path: Path,
) -> None:
    manifest = _manifest(accession="0001127602-20-013168")
    parse_result = parse_form4_ownership_xml(
        _xml(accession=manifest.accession, nonderiv=_nonderiv_tx()),
        manifest=manifest,
        retrieval=_retrieval(),
    )
    write_form4_parse_results(root=tmp_path, results=[parse_result])

    payload = run_form4_candidate_curation(data_root=tmp_path)

    assert payload["report"]["eligible_count"] == 1
    assert Path(str(payload["events_path"])).exists()


def test_form4_tradable_at_uses_publication_time_and_regular_session() -> None:
    assert (
        resolve_form4_tradable_at("2025-04-07T12:00:00+00:00").isoformat()
        == "2025-04-07T13:35:00+00:00"
    )
    assert (
        resolve_form4_tradable_at("2025-04-07T15:00:00+00:00").isoformat()
        == "2025-04-07T15:05:00+00:00"
    )
    assert (
        resolve_form4_tradable_at("2025-04-07T20:30:00+00:00").isoformat()
        == "2025-04-08T13:35:00+00:00"
    )


def test_form4_label_builder_computes_net_and_abnormal_returns() -> None:
    candidates = pd.DataFrame(
        [
            _candidate_row(
                event_id="event-1",
                ticker="AAPL",
                accepted_at_utc="2025-04-07T15:00:00+00:00",
                eligibility_pass=True,
            ),
            _candidate_row(
                event_id="event-2",
                ticker="MSFT",
                accepted_at_utc="2025-04-07T15:00:00+00:00",
                eligibility_pass=False,
            ),
        ]
    )
    minute = pd.DataFrame(
        [
            {
                "timestamp": "2025-04-07T15:05:00+00:00",
                "symbol": "AAPL",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 1000,
            }
        ]
    )
    daily = _daily_label_fixture(
        {
            "AAPL": {
                "2025-04-07": 100.0,
                "2025-04-08": 110.0,
                "2025-04-14": 120.0,
            },
            "IWM": {
                "2025-04-07": 200.0,
                "2025-04-08": 202.0,
                "2025-04-14": 210.0,
            },
        }
    )

    labels = build_form4_event_labels(
        candidates=candidates,
        minute_bars=minute,
        daily_bars=daily,
        config=Form4LabelConfig(horizons=(1, 5), round_trip_cost_bps=50.0),
    )

    labeled = labels.set_index("event_id").loc["event-1"]
    assert labeled["label_status"] == "LABELED"
    assert labeled["tradable_at_utc"] == "2025-04-07T15:05:00+00:00"
    expected_1d = math.log(110.0 / 100.0) - 0.005
    expected_benchmark_1d = math.log(202.0 / 200.0)
    assert abs(labeled["ret_1d_net"] - expected_1d) < 1e-12
    assert abs(labeled["abret_1d_net"] - (expected_1d - expected_benchmark_1d)) < 1e-12
    skipped = labels.set_index("event_id").loc["event-2"]
    assert skipped["label_status"] == "SKIPPED_INELIGIBLE"


def test_form4_label_curation_writes_blocked_report_when_bars_missing(
    tmp_path: Path,
) -> None:
    candidate_path = (
        tmp_path
        / "data"
        / "curated"
        / "events"
        / "form4_open_market_buy_candidates"
        / "data.parquet"
    )
    candidate_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            _candidate_row(
                event_id="event-1",
                ticker="AAPL",
                accepted_at_utc="2025-04-07T15:00:00+00:00",
                eligibility_pass=True,
            )
        ]
    ).to_parquet(candidate_path, index=False)

    payload = run_form4_label_curation(data_root=tmp_path, horizons=(5,))

    assert payload["report"]["blocked_count"] == 1
    assert payload["report"]["blocker_counts"] == {
        "missing_daily_source": 1,
        "missing_minute_source": 1,
    }
    labels = pd.read_parquet(payload["labels_path"])
    assert labels.iloc[0]["label_status"] == "BLOCKED"


def test_form4_label_curation_reads_nas_style_bar_partitions(tmp_path: Path) -> None:
    candidate_path = (
        tmp_path
        / "data"
        / "curated"
        / "events"
        / "form4_open_market_buy_candidates"
        / "data.parquet"
    )
    candidate_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            _candidate_row(
                event_id="event-1",
                ticker="AAPL",
                accepted_at_utc="2025-04-07T15:00:00+00:00",
                eligibility_pass=True,
            )
        ]
    ).to_parquet(candidate_path, index=False)
    minute_path = tmp_path / "data" / "curated" / "equities_minute" / "date=2025-04-07" / "data.parquet"
    minute_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "timestamp": "2025-04-07T15:05:00+00:00",
                "symbol": "AAPL",
                "open": 100.0,
                "close": 100.5,
            }
        ]
    ).to_parquet(minute_path, index=False)
    daily_path = tmp_path / "data" / "curated" / "equities_eod" / "date=2025-04-08" / "data.parquet"
    daily_path.parent.mkdir(parents=True)
    _daily_label_fixture(
        {
            "AAPL": {"2025-04-07": 100.0, "2025-04-08": 110.0},
            "IWM": {"2025-04-07": 200.0, "2025-04-08": 202.0},
        }
    ).to_parquet(daily_path, index=False)

    payload = run_form4_label_curation(data_root=tmp_path, horizons=(1,))

    assert payload["report"]["labeled_count"] == 1
    assert Path(str(payload["labels_path"])).exists()


def test_form4_label_curation_uses_source_contract_and_extra_market_root(
    tmp_path: Path,
) -> None:
    candidate_path = (
        tmp_path
        / "data"
        / "curated"
        / "events"
        / "form4_open_market_buy_candidates"
        / "data.parquet"
    )
    candidate_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            _candidate_row(
                event_id="event-1",
                ticker="AAPL",
                accepted_at_utc="2025-04-07T15:00:00+00:00",
                eligibility_pass=True,
            )
        ]
    ).to_parquet(candidate_path, index=False)
    market_root = tmp_path / "market-root"
    minute_root = market_root / "data" / "raw" / "equities_minute"
    minute_path = minute_root / "date=2025-04-07" / "data.parquet"
    minute_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "vendor_ts": "2025-04-07T15:05:00+00:00",
                "symbol": "AAPL",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
            }
        ]
    ).to_parquet(minute_path, index=False)
    daily_root = market_root / "data" / "curated" / "equities_ohlcv_adj"
    daily_path = daily_root / "date=2025-04-08" / "data.parquet"
    daily_path.parent.mkdir(parents=True)
    _daily_label_fixture(
        {
            "AAPL": {"2025-04-07": 100.0, "2025-04-08": 110.0},
            "SPY": {"2025-04-07": 400.0, "2025-04-08": 404.0},
        }
    ).to_parquet(daily_path, index=False)
    contract_path = (
        tmp_path
        / "control"
        / "cluster"
        / "state"
        / "research"
        / "feature_source_contract"
        / "latest.json"
    )
    contract_path.parent.mkdir(parents=True)
    contract_path.write_text(
        json.dumps(
            {
                "version": "feature_source_contract_v1",
                "datasets": {
                    "equities_minute": {"paths": [str(minute_root)]},
                    "equities_ohlcv_adj": {"paths": [str(daily_root)]},
                },
            }
        ),
        encoding="utf-8",
    )

    payload = run_form4_label_curation(
        data_root=tmp_path,
        horizons=(1,),
        market_data_roots=[market_root],
        source_contract_path=contract_path,
    )

    assert payload["report"]["labeled_count"] == 1
    metadata = payload["report"]["source_metadata"]["datasets"]
    assert metadata["equities_minute"]["loaded_files"] == [str(minute_path)]
    assert metadata["equities_ohlcv_adj"]["loaded_files"] == [str(daily_path)]
    labels = pd.read_parquet(payload["labels_path"])
    assert labels.iloc[0]["entry_price_source"] == "minute_open"


def test_form4_ingest_writes_manifest_parse_outputs_and_candidates(tmp_path: Path) -> None:
    manifest = _manifest(accession="0001127602-20-013168")
    connector = _FakeForm4IngestConnector(
        manifest_rows=[manifest],
        xml_by_accession={
            manifest.accession: _xml(
                accession=manifest.accession,
                nonderiv=_nonderiv_tx(shares="10", price="100"),
            )
        },
    )

    payload = run_form4_ingest(
        data_root=tmp_path,
        connector=connector,
        start_date="2020-04-01",
        end_date="2020-04-30",
    )

    assert payload["verdict"] == "PASS"
    assert payload["parsed_count"] == 1
    assert (
        tmp_path
        / "data"
        / "raw"
        / "sec"
        / "form4_manifest"
        / "year=2020"
        / "qtr=2"
        / "manifest.parquet"
    ).exists()
    candidates = pd.read_parquet(
        tmp_path
        / "data"
        / "curated"
        / "events"
        / "form4_open_market_buy_candidates"
        / "data.parquet"
    )
    assert bool(candidates.iloc[0]["eligibility_pass"]) is True


def test_form4_ingest_retries_remote_rate_limits(tmp_path: Path) -> None:
    manifest = _manifest(accession="0001127602-20-013168")
    connector = _FlakyRateLimitForm4IngestConnector(
        manifest_rows=[manifest],
        xml_by_accession={
            manifest.accession: _xml(
                accession=manifest.accession,
                nonderiv=_nonderiv_tx(shares="10", price="100"),
            )
        },
    )
    sleep_calls: list[float] = []

    payload = run_form4_ingest(
        data_root=tmp_path,
        connector=connector,
        start_date="2020-04-01",
        end_date="2020-04-30",
        max_retrieval_attempts=2,
        rate_limit_pause_seconds=0.25,
        sleep_fn=lambda seconds: sleep_calls.append(seconds),
    )

    assert payload["verdict"] == "PASS"
    assert payload["parsed_count"] == 1
    assert payload["accessions"][0]["retry_events"] == [
        {"attempt": 1, "reason": "remote_rate_limit", "sleep_seconds": 0.25}
    ]
    assert sleep_calls == [0.25]


def test_form4_ingest_reuses_cached_raw_retrieval(tmp_path: Path) -> None:
    manifest = _manifest(accession="0001127602-20-013168")
    xml = _xml(
        accession=manifest.accession,
        nonderiv=_nonderiv_tx(shares="10", price="100"),
    )
    write_form4_retrieval_artifacts(
        root=tmp_path,
        manifest=manifest,
        retrieval=Form4RetrievalResult(
            metadata=_retrieval(),
            ownership_xml=xml,
            complete_txt="<SEC-DOCUMENT />",
        ),
    )
    connector = _CachedOnlyForm4IngestConnector(manifest_rows=[manifest])

    payload = run_form4_ingest(
        data_root=tmp_path,
        connector=connector,
        start_date="2020-04-01",
        end_date="2020-04-30",
    )

    assert payload["verdict"] == "PASS"
    assert payload["parsed_count"] == 1
    assert connector.retrieve_calls == 0
    assert "used_cached_raw_artifact" in payload["accessions"][0]["retrieval"][
        "quality_flags"
    ]


def test_form4_market_backfill_fetches_needed_minute_and_daily_partitions(
    tmp_path: Path,
) -> None:
    candidate_path = (
        tmp_path
        / "data"
        / "curated"
        / "events"
        / "form4_open_market_buy_candidates"
        / "data.parquet"
    )
    candidate_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            _candidate_row(
                event_id="event-1",
                ticker="AAPL",
                accepted_at_utc="2025-04-07T20:30:00+00:00",
                eligibility_pass=True,
            )
        ]
    ).to_parquet(candidate_path, index=False)

    connector = _FakeMarketConnector()
    payload = run_form4_market_backfill(
        data_root=tmp_path,
        connector=connector,
        horizons=(1,),
    )

    assert payload["minute_rows"] == 1
    assert payload["daily_rows"] == 6
    assert payload["minute_symbol_date_request_count"] == 1
    assert connector.calls[0] == (
        "equities_minute",
        ["AAPL"],
        "2025-04-08",
        "2025-04-08",
    )
    assert payload["minute"][0]["written_path_summary"]["count"] == 1
    assert payload["daily"][0]["written_path_summary"]["count"] == 2
    assert (
        tmp_path
        / "data"
        / "raw"
        / "equities_minute"
        / "date=2025-04-08"
        / "data.parquet"
    ).exists()
    assert (
        tmp_path
        / "control"
        / "cluster"
        / "state"
        / "research"
        / "form4_market_backfill"
        / "latest.json"
    ).exists()


def test_form4_market_backfill_splits_daily_symbol_batches(tmp_path: Path) -> None:
    candidate_path = (
        tmp_path
        / "data"
        / "curated"
        / "events"
        / "form4_open_market_buy_candidates"
        / "data.parquet"
    )
    candidate_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            _candidate_row(
                event_id="event-1",
                ticker="AAPL",
                accepted_at_utc="2025-04-07T20:30:00+00:00",
                eligibility_pass=True,
            )
        ]
    ).to_parquet(candidate_path, index=False)
    connector = _FakeMarketConnector()

    payload = run_form4_market_backfill(
        data_root=tmp_path,
        connector=connector,
        horizons=(1,),
        daily_symbol_batch_size=2,
    )

    daily_calls = [call for call in connector.calls if call[0] == "equities_eod"]
    assert daily_calls == [
        ("equities_eod", ["AAPL", "IWM"], "2025-04-08", "2025-04-09"),
        ("equities_eod", ["SPY"], "2025-04-08", "2025-04-09"),
    ]
    assert payload["daily_fetch_batches"] == 2
    assert payload["daily_rows"] == 6


def test_form4_market_backfill_includes_negative_control_candidates(
    tmp_path: Path,
) -> None:
    candidate_path = (
        tmp_path
        / "data"
        / "curated"
        / "events"
        / "form4_open_market_buy_candidates"
        / "data.parquet"
    )
    candidate_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            _candidate_row(
                event_id="event-1",
                ticker="AAPL",
                accepted_at_utc="2025-04-07T20:30:00+00:00",
                eligibility_pass=True,
            )
        ]
    ).to_parquet(candidate_path, index=False)
    sale = parse_form4_ownership_xml(
        _xml(
            accession="0001025978-25-000011",
            issuer_symbol="MSFT",
            nonderiv=_nonderiv_tx(code="S", acquired="D", price="20"),
        ),
        manifest=_manifest(accession="0001025978-25-000011"),
        retrieval=_retrieval(),
    )
    write_form4_parse_results(root=tmp_path, results=[sale])

    payload = run_form4_market_backfill(
        data_root=tmp_path,
        connector=_FakeMarketConnector(),
        horizons=(1,),
    )

    assert payload["include_controls"] is True
    assert payload["primary_candidate_count"] == 1
    assert payload["control_candidate_count"] == 1
    assert payload["backfill_candidate_count"] == 2
    assert payload["candidate_symbols"] == ["AAPL", "MSFT"]


def test_form4_market_backfill_fetches_only_symbols_needed_per_minute_date(
    tmp_path: Path,
) -> None:
    candidate_path = (
        tmp_path
        / "data"
        / "curated"
        / "events"
        / "form4_open_market_buy_candidates"
        / "data.parquet"
    )
    candidate_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            _candidate_row(
                event_id="event-1",
                ticker="AAPL",
                accepted_at_utc="2025-04-07T20:30:00+00:00",
                eligibility_pass=True,
            ),
            _candidate_row(
                event_id="event-2",
                ticker="MSFT",
                accepted_at_utc="2025-04-08T20:30:00+00:00",
                eligibility_pass=True,
            ),
        ]
    ).to_parquet(candidate_path, index=False)
    connector = _FakeMarketConnector()

    payload = run_form4_market_backfill(
        data_root=tmp_path,
        connector=connector,
        horizons=(1,),
        include_controls=False,
    )

    minute_calls = [call for call in connector.calls if call[0] == "equities_minute"]
    assert minute_calls == [
        ("equities_minute", ["AAPL"], "2025-04-08", "2025-04-08"),
        ("equities_minute", ["MSFT"], "2025-04-09", "2025-04-09"),
    ]
    assert payload["minute_rows"] == 2
    assert payload["minute_symbol_date_request_count"] == 2


def test_form4_market_backfill_retries_local_budget_blocks(tmp_path: Path) -> None:
    candidate_path = (
        tmp_path
        / "data"
        / "curated"
        / "events"
        / "form4_open_market_buy_candidates"
        / "data.parquet"
    )
    candidate_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            _candidate_row(
                event_id="event-1",
                ticker="AAPL",
                accepted_at_utc="2025-04-07T20:30:00+00:00",
                eligibility_pass=True,
            )
        ]
    ).to_parquet(candidate_path, index=False)
    connector = _BudgetBlockedMarketConnector()
    sleep_calls: list[float] = []

    payload = run_form4_market_backfill(
        data_root=tmp_path,
        connector=connector,
        horizons=(1,),
        max_fetch_attempts=2,
        rate_limit_pause_seconds=0.25,
        sleep_fn=lambda seconds: sleep_calls.append(seconds),
    )

    assert payload["daily_rows"] == 6
    assert payload["retry_events"] == [
        {
            "attempt": 1,
            "dataset": "equities_eod",
            "start_date": "2025-04-08",
            "end_date": "2025-04-09",
            "symbol_count": 3,
            "reason": "local_budget_block",
            "sleep_seconds": 0.25,
        }
    ]
    assert sleep_calls == [0.25]


def test_form4_event_study_writes_decision_packet(tmp_path: Path) -> None:
    labels_path = (
        tmp_path
        / "data"
        / "curated"
        / "events"
        / "form4_open_market_buy_labels"
        / "data.parquet"
    )
    labels_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                **_candidate_row(
                    event_id=f"event-{index}",
                    ticker=f"T{index}",
                    accepted_at_utc="2025-04-07T20:30:00+00:00",
                    eligibility_pass=True,
                ),
                "label_status": "LABELED",
                "event_strength_score": float(index),
                "ret_5d_net": value,
                "abret_5d_net": value,
            }
            for index, value in enumerate([0.02, 0.03, 0.04, 0.05], start=1)
        ]
    ).to_parquet(labels_path, index=False)

    payload = run_form4_event_study(
        data_root=tmp_path,
        primary_horizon=5,
        horizons=(5,),
        min_historical_sample=300,
    )

    packet = payload["packet"]
    assert packet["primary"]["n"] == 4
    assert packet["verdict"]["decision"] == "COLLECT_MORE_HISTORY"
    assert Path(str(payload["packet_path"])).exists()
    assert Path(str(payload["report_path"])).exists()


def test_form4_rework_study_reuses_existing_artifacts_and_kills_weak_variants(
    tmp_path: Path,
) -> None:
    labels_path = (
        tmp_path
        / "data"
        / "curated"
        / "events"
        / "form4_open_market_buy_labels"
        / "data.parquet"
    )
    labels_path.parent.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    for index in range(8):
        rows.append(
            {
                **_candidate_row(
                    event_id=f"event-{index}",
                    ticker=f"T{index}",
                    accepted_at_utc="2025-04-07T20:30:00+00:00",
                    eligibility_pass=True,
                ),
                "label_status": "LABELED",
                "days_since_transaction": 1,
                "source_quality_flags": [],
                "exclusion_reasons": [],
                "n_officers_buying": 1 if index % 2 == 0 else 0,
                "n_directors_buying": 1 if index % 2 == 1 else 0,
                "ceo_buy": index == 0,
                "cfo_buy": False,
                "n_insiders_buying": 1,
                "eligible_transaction_count": 1,
                "total_dollar_value_float": float(1000 + index),
                "abret_5d_net": -0.01,
            }
        )
    pd.DataFrame(rows).to_parquet(labels_path, index=False)
    for family in ("mechanical_acquisition_codes", "sales_placebo"):
        control_path = (
            tmp_path
            / "data"
            / "curated"
            / "events"
            / "form4_control_labels"
            / family
            / "labels.parquet"
        )
        control_path.parent.mkdir(parents=True)
        pd.DataFrame(
            [
                {
                    **_candidate_row(
                        event_id=f"{family}-{index}",
                        ticker=f"C{index}",
                        accepted_at_utc="2025-04-07T20:30:00+00:00",
                        eligibility_pass=True,
                    ),
                    "label_status": "LABELED",
                    "abret_5d_net": -0.02,
                }
                for index in range(3)
            ]
        ).to_parquet(control_path, index=False)
    baseline_path = (
        tmp_path
        / "control"
        / "cluster"
        / "state"
        / "research"
        / "form4_event_study"
        / "latest.json"
    )
    baseline_path.parent.mkdir(parents=True)
    baseline_path.write_text(
        json.dumps({"verdict": {"decision": "KILL_OR_REWORK", "status": "NOT_PROMOTABLE"}}),
        encoding="utf-8",
    )

    payload = run_form4_rework_study(data_root=tmp_path)

    packet = payload["packet"]
    assert packet["verdict"]["decision"] == "FORM4_KILLED_BASELINE_COMPLETE"
    assert packet["verdict"]["paper_live_allowed"] is False
    assert packet["variants"]["clean_timely_unmixed"]["candidate_count"] == 8
    assert (
        packet["variants"]["clean_timely_unmixed"]["verdict"]["failures"]
        == [
            "sample_size<100",
            "mean_abret_5d_net<=50bps",
            "median_abret_5d_net<=0",
        ]
    )
    assert Path(str(payload["packet_path"])).exists()
    assert (
        tmp_path
        / "control"
        / "cluster"
        / "state"
        / "research"
        / "form4_baseline_status"
        / "latest.json"
    ).exists()


def test_form4_control_candidates_from_curated_include_sales_and_mechanical(
    tmp_path: Path,
) -> None:
    sale = parse_form4_ownership_xml(
        _xml(
            accession="0001025978-25-000011",
            issuer_symbol="ARES",
            nonderiv=_nonderiv_tx(code="S", acquired="D", price="20"),
        ),
        manifest=_manifest(accession="0001025978-25-000011", archive_cik="1176948"),
        retrieval=_retrieval(),
    )
    mechanical = parse_form4_ownership_xml(
        _xml(
            accession="0001758554-19-000046",
            issuer_symbol="SMCI",
            nonderiv=_nonderiv_tx(code="M", acquired="A", price="0"),
        ),
        manifest=_manifest(accession="0001758554-19-000046", archive_cik="1375365"),
        retrieval=_retrieval(),
    )
    write_form4_parse_results(root=tmp_path, results=[sale, mechanical])

    controls = build_form4_control_candidates_from_curated(root=tmp_path)

    assert len(controls["sales_placebo"]) == 1
    assert len(controls["mechanical_acquisition_codes"]) == 1


def test_form4_control_candidates_exclude_ambiguous_multi_symbol_tickers(
    tmp_path: Path,
) -> None:
    sale = parse_form4_ownership_xml(
        _xml(
            accession="0001025978-25-000011",
            issuer_symbol="WSO; WSOB",
            nonderiv=_nonderiv_tx(code="S", acquired="D", price="20"),
        ),
        manifest=_manifest(accession="0001025978-25-000011", archive_cik="1176948"),
        retrieval=_retrieval(),
    )
    write_form4_parse_results(root=tmp_path, results=[sale])

    controls = build_form4_control_candidates_from_curated(root=tmp_path)

    row = controls["sales_placebo"].iloc[0]
    assert bool(row["eligibility_pass"]) is False
    assert "ambiguous_or_invalid_ticker" in row["exclusion_reasons"]


def _candidate_row(
    *,
    event_id: str,
    ticker: str,
    accepted_at_utc: str,
    eligibility_pass: bool,
) -> dict[str, object]:
    return {
        "event_id": event_id,
        "issuer_cik": "0000000000",
        "ticker": ticker,
        "primary_security_id": f"0000000000:{ticker}",
        "accessions": ["0000000000-25-000001"],
        "event_type": "FORM4_OPEN_MARKET_INSIDER_BUY",
        "accepted_at_utc": accepted_at_utc,
        "first_seen_at_utc": accepted_at_utc,
        "tradable_at_utc": None,
        "eligibility_pass": eligibility_pass,
        "exclusion_reasons": [] if eligibility_pass else ["test_ineligible"],
        "total_dollar_value": "100000",
    }


def _daily_label_fixture(values: dict[str, dict[str, float]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for symbol, by_date in values.items():
        for date, close in by_date.items():
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": close,
                    "high": close,
                    "low": close,
                    "close": close,
                    "volume": 1000,
                }
            )
    return pd.DataFrame(rows)


def test_form4_fixture_gate_fails_on_missing_expected_flag(tmp_path: Path) -> None:
    fixture = Form4FixtureSpec(
        name="expected-mixed",
        accession="0000769993-15-000534",
        archive_cik="769993",
        filed_date="2015-04-15",
        expected_flags=("mixed_p_and_s",),
    )
    connector = _FakeForm4Connector(
        {
            "0000769993-15-000534": _xml(
                accession="0000769993-15-000534",
                issuer_cik="769993",
                nonderiv=_nonderiv_tx(),
            ),
        }
    )

    payload = run_form4_fixture_gate(
        data_root=tmp_path,
        connector=connector,
        fixtures=[fixture],
    )

    assert payload["verdict"] == "FAIL"
    assert payload["failed"] == 1
    assert payload["fixtures"][0]["errors"] == ["missing_expected_flags:mixed_p_and_s"]


class _FakeForm4Connector:
    def __init__(self, xml_by_accession: dict[str, str]) -> None:
        self.xml_by_accession = xml_by_accession

    def retrieve_form4_ownership_xml(
        self,
        manifest: Form4ManifestRow,
        *,
        primary_document: str | None = None,
        submissions_metadata: dict[str, object] | None = None,
    ) -> Form4RetrievalResult:
        xml = self.xml_by_accession[manifest.accession]
        return Form4RetrievalResult(
            metadata=_retrieval(),
            ownership_xml=xml,
            complete_txt=f"<SEC-DOCUMENT><ACCEPTANCE-DATETIME>{_retrieval().accepted_at_raw}</SEC-DOCUMENT>",
        )


class _FakeForm4IngestConnector(_FakeForm4Connector):
    def __init__(
        self,
        *,
        manifest_rows: list[Form4ManifestRow],
        xml_by_accession: dict[str, str],
    ) -> None:
        super().__init__(xml_by_accession)
        self.manifest_rows = manifest_rows

    def fetch(
        self,
        dataset: str,
        symbols: list[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        assert dataset == "form4_ownership"
        assert symbols == []
        assert start_date
        assert end_date
        return pd.DataFrame([row.to_dict() for row in self.manifest_rows])


class _FlakyRateLimitForm4IngestConnector(_FakeForm4IngestConnector):
    def __init__(
        self,
        *,
        manifest_rows: list[Form4ManifestRow],
        xml_by_accession: dict[str, str],
    ) -> None:
        super().__init__(manifest_rows=manifest_rows, xml_by_accession=xml_by_accession)
        self.retrieve_calls = 0

    def retrieve_form4_ownership_xml(
        self,
        manifest: Form4ManifestRow,
        *,
        primary_document: str | None = None,
        submissions_metadata: dict[str, object] | None = None,
    ) -> Form4RetrievalResult:
        self.retrieve_calls += 1
        if self.retrieve_calls == 1:
            raise RemoteRateLimitConnectorError("sec_edgar")
        return super().retrieve_form4_ownership_xml(
            manifest,
            primary_document=primary_document,
            submissions_metadata=submissions_metadata,
        )


class _CachedOnlyForm4IngestConnector:
    def __init__(self, *, manifest_rows: list[Form4ManifestRow]) -> None:
        self.manifest_rows = manifest_rows
        self.retrieve_calls = 0

    def fetch(
        self,
        dataset: str,
        symbols: list[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        assert dataset == "form4_ownership"
        assert symbols == []
        assert start_date
        assert end_date
        return pd.DataFrame([row.to_dict() for row in self.manifest_rows])

    def retrieve_form4_ownership_xml(
        self,
        manifest: Form4ManifestRow,
        *,
        primary_document: str | None = None,
        submissions_metadata: dict[str, object] | None = None,
    ) -> Form4RetrievalResult:
        self.retrieve_calls += 1
        raise AssertionError(f"unexpected live retrieval for {manifest.accession}")


class _BudgetBlockedMarketConnector:
    vendor_name = "budget_blocked_market"

    def __init__(self) -> None:
        self.delegate = _FakeMarketConnector()
        self.daily_attempts = 0

    def fetch(
        self,
        dataset: str,
        symbols: list[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        if dataset == "equities_eod":
            self.daily_attempts += 1
            if self.daily_attempts == 1:
                raise BudgetBlockedConnectorError(
                    "alpaca",
                    BudgetDecision(
                        allowed=False,
                        blocked_dimension="rpm",
                        next_eligible_at=None,
                        remaining_minute=0,
                        remaining_daily=10,
                        remaining_non_forward=10,
                        requested_units=1,
                    ),
                )
        return self.delegate.fetch(dataset, symbols, start_date, end_date)


class _FakeMarketConnector:
    vendor_name = "fake_market"

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str], str, str]] = []

    def fetch(
        self,
        dataset: str,
        symbols: list[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        self.calls.append((dataset, list(symbols), start_date, end_date))
        if dataset == "equities_minute":
            return pd.DataFrame(
                [
                    {
                        "date": start_date,
                        "symbol": symbol,
                        "open": 100.0,
                        "high": 101.0,
                        "low": 99.0,
                        "close": 100.5,
                        "vendor_ts": f"{start_date}T13:35:00+00:00",
                    }
                    for symbol in symbols
                ]
            )
        if dataset == "equities_eod":
            rows = []
            for symbol in symbols:
                for date in (start_date, end_date):
                    rows.append(
                        {
                            "date": date,
                            "symbol": symbol,
                            "open": 100.0,
                            "high": 101.0,
                            "low": 99.0,
                            "close": 100.0,
                            "vendor_ts": f"{date}T04:00:00+00:00",
                        }
                    )
            return pd.DataFrame(rows)
        raise ValueError(dataset)
