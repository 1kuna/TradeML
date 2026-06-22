from __future__ import annotations

from pathlib import Path

import pandas as pd

from trademl.connectors.base import PermanentConnectorError
from trademl.events.sec8k import (
    Sec8KManifestRow,
    build_sec8k_research_decision,
    build_sec8k_item_candidates,
    parse_sec8k_index_manifest,
    parse_sec8k_complete_text,
    run_sec8k_candidate_curation,
    run_sec8k_event_study,
    summarize_sec8k_candidates,
)
from trademl.events.sec8k_ingest import run_sec8k_ingest
from trademl.events.sec8k_market_backfill import run_sec8k_market_backfill


def test_sec8k_complete_text_parser_extracts_items_and_exhibits() -> None:
    parsed = parse_sec8k_complete_text(_complete_8k_text())

    assert parsed.accepted_at_raw == "20250407163000"
    assert parsed.item_numbers == ("1.01", "2.02", "2.04", "3.02", "9.01")
    exhibits = [document for document in parsed.documents if document.is_exhibit]
    assert len(exhibits) == 1
    assert exhibits[0].document_type == "EX-99.1"
    assert exhibits[0].filename == "ex991.htm"


def test_sec8k_complete_text_parser_ignores_exhibit_only_item_headings() -> None:
    parsed = parse_sec8k_complete_text(
        """<SEC-DOCUMENT>
<SEC-HEADER><ACCEPTANCE-DATETIME>20250407163000</SEC-HEADER>
<DOCUMENT>
<TYPE>8-K
<FILENAME>form8k.htm
<TEXT>
Item 8.01 Other Events
The primary filing body has only other events.
</TEXT>
</DOCUMENT>
<DOCUMENT>
<TYPE>EX-99.1
<FILENAME>ex991.htm
<TEXT>
Item 2.04 Triggering Events That Accelerate or Increase a Direct Financial Obligation
This exhibit heading must not create a SEC item candidate by itself.
</TEXT>
</DOCUMENT>
</SEC-DOCUMENT>
"""
    )

    assert parsed.item_numbers == ("8.01",)


def test_sec8k_index_manifest_parser_extracts_exact_8k_rows() -> None:
    rows = parse_sec8k_index_manifest(
        """CIK|Company Name|Form Type|Date Filed|Filename
320193|APPLE INC.|8-K|2025-04-07|edgar/data/320193/000032019325000001/0000320193-25-000001.txt
320193|APPLE INC.|8-K/A|2025-04-08|edgar/data/320193/000032019325000002/0000320193-25-000002.txt
320193|APPLE INC.|10-Q|2025-04-09|edgar/data/320193/000032019325000003/0000320193-25-000003.txt
""",
        index_year=2025,
        index_quarter=2,
        index_crawled_at="2026-05-06T00:00:00Z",
    )

    assert len(rows) == 1
    assert rows[0].archive_cik == "320193"
    assert rows[0].accession == "0000320193-25-000001"
    assert rows[0].form == "8-K"


def test_sec8k_candidate_builder_creates_item_events_and_exhibit_inventory() -> None:
    filings = pd.DataFrame(
        [
            {
                "form": "8-K",
                "accessionNumber": "0000320193-25-000001",
                "cik": "0000320193",
                "ticker": "AAPL",
                "filingDate": "2025-04-07",
                "items": "1.01,2.02,2.04,3.02",
            }
        ]
    )

    candidates = build_sec8k_item_candidates(
        filings=filings,
        complete_text_by_accession={"0000320193-25-000001": _complete_8k_text()},
    )

    assert candidates["event_type"].tolist() == [
        "8K_ITEM_1_01_MATERIAL_AGREEMENT",
        "8K_ITEM_2_02_RESULTS_OPERATIONS",
        "8K_ITEM_2_04_DEFAULT_COVENANT_STRESS",
        "8K_ITEM_3_02_UNREGISTERED_SALE_FINANCING",
    ]
    assert candidates["accepted_at_utc"].notna().all()
    assert bool(candidates["eligibility_pass"].all()) is True
    assert candidates.iloc[0]["exhibit_count"] == 1
    assert candidates.iloc[0]["source_hash"]


def test_sec8k_candidate_summary_counts_targeted_semantic_item_families() -> None:
    filings = pd.DataFrame(
        [
            {
                "form": "8-K",
                "accessionNumber": "0000320193-25-000001",
                "cik": "0000320193",
                "ticker": "AAPL",
                "filingDate": "2025-04-07",
                "acceptanceDateTime": "2025-04-07T20:30:00Z",
                "items": "2.04,3.02",
            }
        ]
    )

    payload = build_sec8k_item_candidates(filings=filings)

    assert payload["event_type"].tolist() == [
        "8K_ITEM_2_04_DEFAULT_COVENANT_STRESS",
        "8K_ITEM_3_02_UNREGISTERED_SALE_FINANCING",
    ]
    assert bool(payload["eligibility_pass"].all()) is True
    summary = summarize_sec8k_candidates(candidates=payload)
    assert summary["family_counts"] == {
        "8K_ITEM_2_04_DEFAULT_COVENANT_STRESS": 1,
        "8K_ITEM_3_02_UNREGISTERED_SALE_FINANCING": 1,
    }


def test_sec8k_candidate_curation_reads_existing_filing_artifact(tmp_path: Path) -> None:
    filings_path = tmp_path / "data" / "reference" / "sec_filing_index.parquet"
    filings_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "form": "8-K",
                "accessionNumber": "0000320193-25-000001",
                "cik": "0000320193",
                "ticker": "AAPL",
                "filingDate": "2025-04-07",
                "acceptanceDateTime": "2025-04-07T20:30:00Z",
                "items": "1.01",
            }
        ]
    ).to_parquet(filings_path, index=False)

    payload = run_sec8k_candidate_curation(data_root=tmp_path)

    assert payload["report"]["candidate_count"] == 1
    assert payload["report"]["eligible_count"] == 1
    assert Path(str(payload["events_path"])).exists()


def test_sec8k_candidate_curation_rebuilds_from_persisted_items_without_bulk_raw_lookup(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import trademl.events.sec8k as sec8k

    filings_path = tmp_path / "data" / "reference" / "sec_filing_index.parquet"
    filings_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "form": "8-K",
                "accessionNumber": "0000320193-25-000001",
                "cik": "0000320193",
                "ticker": "AAPL",
                "filingDate": "2025-01-15",
                "acceptanceDateTime": "20250115163000",
                "items": ["3.02"],
                "exhibit_count": 2,
                "source_hash": "source-hash-jan",
                "complete_txt_path": str(tmp_path / "missing-jan-complete.txt"),
            },
            {
                "form": "8-K",
                "accessionNumber": "0000320193-25-000002",
                "cik": "0000320193",
                "ticker": "AAPL",
                "filingDate": "2025-02-15",
                "acceptanceDateTime": "20250215163000",
                "items": ["4.01"],
                "exhibit_count": 4,
                "source_hash": "source-hash-feb",
                "complete_txt_path": str(tmp_path / "missing-feb-complete.txt"),
            },
        ]
    ).to_parquet(filings_path, index=False)

    def fail_bulk_lookup(**_kwargs):
        raise AssertionError("global candidate rebuild must not bulk-read raw complete.txt")

    monkeypatch.setattr(sec8k, "_complete_text_lookup", fail_bulk_lookup)

    payload = run_sec8k_candidate_curation(data_root=tmp_path)

    assert payload["report"]["candidate_count"] == 2
    candidates = pd.read_parquet(
        tmp_path / "data" / "curated" / "events" / "sec_8k_item_events" / "data.parquet"
    )
    assert sorted(candidates["filing_date"].astype(str).unique().tolist()) == [
        "2025-01-15",
        "2025-02-15",
    ]
    assert candidates.set_index("sec_item_number").loc["3.02", "exhibit_count"] == 2
    assert candidates.set_index("sec_item_number").loc["4.01", "source_hash"] == "source-hash-feb"


def test_sec8k_ingest_writes_reference_filing_index_and_candidates(tmp_path: Path) -> None:
    manifest = Sec8KManifestRow(
        archive_cik="320193",
        form="8-K",
        filed_date="2025-04-07",
        index_filename="edgar/data/320193/000032019325000001/0000320193-25-000001.txt",
        accession="0000320193-25-000001",
        accession_no_dashes="000032019325000001",
        discovery_source="sec_full_index",
        index_year=2025,
        index_quarter=2,
        index_file_hash="hash",
        index_crawled_at="2026-05-06T00:00:00Z",
    )
    connector = _FakeSec8KIngestConnector([manifest])

    payload = run_sec8k_ingest(
        data_root=tmp_path,
        connector=connector,
        start_date="2025-04-07",
        end_date="2025-04-07",
        limit=1,
    )

    assert payload["verdict"] == "PASS"
    assert payload["parsed_count"] == 1
    assert (
        tmp_path / "data" / "reference" / "sec_filing_index.parquet"
    ).exists()
    assert payload["candidate_artifacts"]["report"]["candidate_count"] == 4
    candidates = pd.read_parquet(
        tmp_path / "data" / "curated" / "events" / "sec_8k_item_events" / "data.parquet"
    )
    assert set(candidates["ticker"]) == {"AAPL"}


def test_sec8k_event_study_writes_labels_controls_and_packet(tmp_path: Path) -> None:
    candidates_path = (
        tmp_path / "data" / "curated" / "events" / "sec_8k_item_events" / "data.parquet"
    )
    candidates_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "event_id": "sec8k-test",
                "issuer_cik": "0000320193",
                "ticker": "AAPL",
                "primary_security_id": "0000320193:AAPL",
                "accessions": ["0000320193-25-000001"],
                "event_type": "8K_ITEM_1_01_MATERIAL_AGREEMENT",
                "accepted_at_utc": "2025-04-07T20:30:00+00:00",
                "first_seen_at_utc": "2025-04-07T20:30:00+00:00",
                "tradable_at_utc": None,
                "eligibility_pass": True,
                "exclusion_reasons": [],
                "event_strength_score": 3.0,
            }
        ]
    ).to_parquet(candidates_path, index=False)
    _write_market_fixture(tmp_path)

    payload = run_sec8k_event_study(
        data_root=tmp_path,
        primary_horizon=5,
        horizons=(5,),
        round_trip_cost_bps=50.0,
    )

    packet = payload["packet"]
    assert packet["verdict"]["decision"] == "DIAGNOSTIC_ONLY"
    assert packet["primary"]["n"] == 1
    assert Path(str(payload["packet_path"])).exists()
    assert (
        tmp_path
        / "data"
        / "curated"
        / "events"
        / "sec_8k_timestamp_placebo_labels"
        / "data.parquet"
    ).exists()


def test_sec8k_market_backfill_fetches_required_slices(tmp_path: Path) -> None:
    candidates_path = (
        tmp_path / "data" / "curated" / "events" / "sec_8k_item_events" / "data.parquet"
    )
    candidates_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "event_id": "sec8k-test",
                "ticker": "AAPL",
                "accepted_at_utc": "2025-04-07T20:30:00+00:00",
                "eligibility_pass": True,
                "event_type": "8K_ITEM_1_01_MATERIAL_AGREEMENT",
            }
        ]
    ).to_parquet(candidates_path, index=False)
    connector = _FakeSec8KMarketConnector()

    payload = run_sec8k_market_backfill(
        data_root=tmp_path,
        connector=connector,
        horizons=(5,),
        include_timestamp_placebo=False,
        max_fetch_attempts=1,
        daily_symbol_batch_size=2,
    )

    assert payload["verdict"] == "PASS"
    assert payload["candidate_symbols"] == ["AAPL"]
    assert payload["needed_minute_dates"] == ["2025-04-08"]
    assert payload["needed_daily_dates"] == ["2025-04-08", "2025-04-15"]
    assert payload["empty_minute"] == []
    assert payload["empty_daily_symbols"] == []
    assert connector.calls == [
        ("equities_minute", ("AAPL",), "2025-04-08", "2025-04-08"),
        ("equities_eod", ("AAPL", "IWM"), "2025-04-08", "2025-04-15"),
        ("equities_eod", ("SPY",), "2025-04-08", "2025-04-15"),
    ]
    assert (
        tmp_path
        / "control"
        / "cluster"
        / "state"
        / "research"
        / "sec8k_market_backfill"
        / "latest.json"
    ).exists()
    assert (
        tmp_path
        / "data"
        / "raw"
        / "equities_minute"
        / "date=2025-04-08"
        / "data.parquet"
    ).exists()


def test_sec8k_market_backfill_records_invalid_vendor_symbols(tmp_path: Path) -> None:
    candidates_path = (
        tmp_path / "data" / "curated" / "events" / "sec_8k_item_events" / "data.parquet"
    )
    candidates_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "event_id": "sec8k-test",
                "ticker": "BAD-PI",
                "accepted_at_utc": "2025-04-07T20:30:00+00:00",
                "eligibility_pass": True,
                "event_type": "8K_ITEM_1_01_MATERIAL_AGREEMENT",
            }
        ]
    ).to_parquet(candidates_path, index=False)

    payload = run_sec8k_market_backfill(
        data_root=tmp_path,
        connector=_PermanentErrorMarketConnector(),
        horizons=(5,),
        include_timestamp_placebo=False,
        max_fetch_attempts=1,
        sleep_fn=lambda _: None,
    )

    assert payload["verdict"] == "PARTIAL_COVERAGE"
    assert payload["retry_event_count"] == 3
    assert payload["empty_minute"] == [{"date": "2025-04-08", "symbol": "BAD-PI"}]
    assert payload["empty_daily_symbols"] == ["BAD-PI"]


def test_sec8k_market_backfill_filters_target_items_and_dates(tmp_path: Path) -> None:
    candidates_path = (
        tmp_path / "data" / "curated" / "events" / "sec_8k_item_events" / "data.parquet"
    )
    candidates_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "event_id": "target",
                "ticker": "AAPL",
                "accepted_at_utc": "2025-04-07T20:30:00+00:00",
                "eligibility_pass": True,
                "event_type": "8K_ITEM_3_02_UNREGISTERED_SALE_FINANCING",
                "sec_item_number": "3.02",
            },
            {
                "event_id": "wrong-item",
                "ticker": "GOOG",
                "accepted_at_utc": "2025-04-07T20:30:00+00:00",
                "eligibility_pass": True,
                "event_type": "8K_ITEM_4_01_AUDITOR_CHANGE",
                "sec_item_number": "4.01",
            },
            {
                "event_id": "wrong-date",
                "ticker": "MSFT",
                "accepted_at_utc": "2025-03-07T20:30:00+00:00",
                "eligibility_pass": True,
                "event_type": "8K_ITEM_3_02_UNREGISTERED_SALE_FINANCING",
                "sec_item_number": "3.02",
            },
        ]
    ).to_parquet(candidates_path, index=False)
    connector = _FakeSec8KMarketConnector()

    payload = run_sec8k_market_backfill(
        data_root=tmp_path,
        connector=connector,
        horizons=(5,),
        include_timestamp_placebo=False,
        target_items=("3.02",),
        accepted_from="2025-04-01",
        accepted_to="2025-04-30",
        max_fetch_attempts=1,
        daily_symbol_batch_size=2,
    )

    assert payload["candidate_source"] == "sec8k_item_events"
    assert payload["target_items"] == ["3.02"]
    assert payload["candidate_symbols"] == ["AAPL"]
    assert connector.calls == [
        ("equities_minute", ("AAPL",), "2025-04-08", "2025-04-08"),
        ("equities_eod", ("AAPL", "IWM"), "2025-04-08", "2025-04-15"),
        ("equities_eod", ("SPY",), "2025-04-08", "2025-04-15"),
    ]


def test_sec8k_market_backfill_can_use_semantic_candidate_source(tmp_path: Path) -> None:
    candidates_path = (
        tmp_path
        / "data"
        / "curated"
        / "events"
        / "sec_event_semantic_candidates"
        / "data.parquet"
    )
    candidates_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "event_id": "semantic-target",
                "ticker": "AAPL",
                "accepted_at_utc": "2025-04-07T20:30:00+00:00",
                "eligibility_pass": True,
                "event_type": "DILUTIVE_FINANCING",
                "sec_item_number": "3.02",
            }
        ]
    ).to_parquet(candidates_path, index=False)

    payload = run_sec8k_market_backfill(
        data_root=tmp_path,
        connector=_FakeSec8KMarketConnector(),
        horizons=(5,),
        include_timestamp_placebo=False,
        candidate_source="sec_event_semantic_candidates",
        max_fetch_attempts=1,
    )

    assert payload["candidate_source"] == "sec_event_semantic_candidates"
    assert payload["candidate_symbols"] == ["AAPL"]


def test_sec8k_research_decision_kills_weak_broad_item_family() -> None:
    decision = build_sec8k_research_decision(
        packet={
            "primary": {
                "n": 600,
                "mean": -0.014,
                "median": -0.006,
                "hit_rate": 0.46,
                "top5_abs_contribution": 0.18,
            },
            "negative_control_separation": {
                "timestamp_placebo": {
                    "primary_n": 600,
                    "control_n": 580,
                    "mean_difference": -0.044,
                }
            },
            "by_item_family": {
                "8K_ITEM_7_01_REG_FD": {
                    "n": 190,
                    "mean": 0.0058,
                    "median": -0.009,
                    "hit_rate": 0.45,
                    "top5_abs_contribution": 0.39,
                    "bootstrap_mean_ci_95": [-0.03, 0.05],
                }
            },
        },
        ingest={"verdict": "PASS", "manifest_count": 1000, "parsed_count": 1000},
        backfill={"verdict": "PARTIAL_COVERAGE", "minute_rows": 99_450},
    )

    assert decision["decision"] == "BROAD_SEC8K_ITEM_FAMILIES_KILLED"
    assert decision["move_forward"] is False
    assert "timestamp_placebo_separation_failed" in decision["study_failed_gates"]
    assert "median_not_positive" in decision["family_results"]["8K_ITEM_7_01_REG_FD"]["failed_gates"]


def _complete_8k_text() -> str:
    return """<SEC-DOCUMENT>
<SEC-HEADER>
<ACCEPTANCE-DATETIME>20250407163000
</SEC-HEADER>
<DOCUMENT>
<TYPE>8-K
<FILENAME>form8k.htm
<TEXT>
Item 1.01 Entry into a Material Definitive Agreement
Item 2.02 Results of Operations and Financial Condition
Item 2.04 Triggering Events That Accelerate or Increase a Direct Financial Obligation
Item 3.02 Unregistered Sales of Equity Securities
Item 9.01 Financial Statements and Exhibits
</TEXT>
</DOCUMENT>
<DOCUMENT>
<TYPE>EX-99.1
<FILENAME>ex991.htm
<DESCRIPTION>Press release
<TEXT>Exhibit text.</TEXT>
</DOCUMENT>
</SEC-DOCUMENT>
"""


class _FakeSec8KIngestConnector:
    def __init__(self, manifests: list[Sec8KManifestRow]) -> None:
        self.manifests = manifests
        self.retrieval_calls = 0

    def fetch(
        self,
        dataset: str,
        symbols: list[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        if dataset == "sec8k_index":
            return pd.DataFrame([item.to_dict() for item in self.manifests])
        if dataset == "company_tickers":
            return pd.DataFrame(
                [
                    {"cik_str": 320193, "ticker": "AAPL-PB", "title": "Apple Inc."},
                    {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
                ]
            )
        raise AssertionError(f"unexpected dataset: {dataset}")

    def retrieve_complete_submission_text(
        self, *, index_filename: str, endpoint_key: str = "sec8k_complete_txt"
    ) -> tuple[int, str, str]:
        self.retrieval_calls += 1
        assert endpoint_key == "sec8k_complete_txt"
        return 200, _complete_8k_text(), f"https://www.sec.gov/Archives/{index_filename}"


class _FakeSec8KMarketConnector:
    vendor_name = "alpaca"

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[str, ...], str, str]] = []

    def fetch(
        self,
        dataset: str,
        symbols: list[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        self.calls.append((dataset, tuple(symbols), start_date, end_date))
        if dataset == "equities_minute":
            return pd.DataFrame(
                [
                    {
                        "date": start_date,
                        "timestamp": "2025-04-08T13:35:00+00:00",
                        "symbol": "AAPL",
                        "open": 100.0,
                        "close": 100.0,
                    }
                ]
            )
        if dataset == "equities_eod":
            rows = []
            for symbol in symbols:
                rows.extend(
                    [
                        {"date": "2025-04-08", "symbol": symbol, "close": 100.0},
                        {"date": "2025-04-15", "symbol": symbol, "close": 101.0},
                    ]
                )
            return pd.DataFrame(rows)
        raise AssertionError(f"unexpected dataset: {dataset}")


class _PermanentErrorMarketConnector:
    vendor_name = "alpaca"

    def fetch(
        self,
        dataset: str,
        symbols: list[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        if "BAD-PI" in symbols:
            raise PermanentConnectorError("invalid symbol: BAD-PI")
        if dataset == "equities_eod":
            return pd.DataFrame(
                [
                    {"date": start_date, "symbol": symbol, "close": 100.0}
                    for symbol in symbols
                ]
            )
        return pd.DataFrame()


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
