from __future__ import annotations

from pathlib import Path

import pandas as pd

from trademl.events.sec8k_coverage import (
    run_sec8k_coverage_audit,
    run_sec8k_coverage_expand,
    run_sec_event_semantic_coverage_gate,
)


def test_sec8k_coverage_audit_detects_april_only_candidates_and_blockers(
    tmp_path: Path,
) -> None:
    _write_april_only_candidate_fixture(tmp_path)
    _write_labelable_market_fixture(tmp_path)

    payload = run_sec8k_coverage_audit(
        data_root=tmp_path,
        start_date="2025-01-01",
        end_date="2025-04-30",
        target_items=("3.02", "4.01"),
        fallback_target_items=(),
        horizons=(5,),
    )["packet"]

    assert payload["candidate_artifact_range_mismatch"] is True
    assert payload["candidate_missing_months"] == ["2025-01", "2025-02", "2025-03"]
    assert "2025-01" in payload["missing_sec_months"]
    assert payload["labelability_blocker_counts"]["missing_ticker"] == 1
    assert payload["labelability_blocker_counts"]["missing_entry_minute"] == 1
    assert payload["market_label_blocker_count"] == 1
    assert Path(str(payload["monthly_artifact"])).exists()
    assert Path(str(payload["labelability_queue_artifact"])).exists()


def test_sec8k_coverage_audit_flags_range_mismatch_when_multiple_months_have_candidates_but_some_are_missing(
    tmp_path: Path,
) -> None:
    _write_candidate_rows(
        tmp_path,
        [
            {
                "event_id": "jan",
                "accession": "0000320193-25-000001",
                "ticker": "AAPL",
                "accepted_at_utc": "2025-01-15T21:30:00+00:00",
                "filing_date": "2025-01-15",
                "event_type": "8K_ITEM_3_02_UNREGISTERED_SALE_FINANCING",
                "sec_item_number": "3.02",
                "eligibility_pass": True,
                "exclusion_reasons": [],
            },
            {
                "event_id": "mar",
                "accession": "0000320193-25-000003",
                "ticker": "AAPL",
                "accepted_at_utc": "2025-03-15T21:30:00+00:00",
                "filing_date": "2025-03-15",
                "event_type": "8K_ITEM_3_02_UNREGISTERED_SALE_FINANCING",
                "sec_item_number": "3.02",
                "eligibility_pass": True,
                "exclusion_reasons": [],
            },
        ],
    )

    payload = run_sec8k_coverage_audit(
        data_root=tmp_path,
        start_date="2025-01-01",
        end_date="2025-03-31",
        target_items=("3.02",),
        fallback_target_items=(),
        horizons=(5,),
    )["packet"]

    assert payload["candidate_missing_months"] == ["2025-02"]
    assert payload["candidate_artifact_range_mismatch"] is True


def test_sec8k_coverage_audit_treats_missing_manifest_month_as_source_gap_even_with_stray_filing_row(
    tmp_path: Path,
) -> None:
    filing_path = tmp_path / "data" / "reference" / "sec_filing_index.parquet"
    filing_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "form": "8-K",
                "accessionNumber": "0000320193-25-000001",
                "filingDate": "2025-01-02",
                "acceptanceDateTime": "20241231192235",
                "items": [],
                "ticker": "AAPL",
            }
        ]
    ).to_parquet(filing_path, index=False)

    payload = run_sec8k_coverage_audit(
        data_root=tmp_path,
        start_date="2024-12-01",
        end_date="2024-12-31",
        target_items=("3.02",),
        fallback_target_items=(),
        horizons=(5,),
    )["packet"]

    assert payload["missing_sec_months"] == ["2024-12"]
    assert payload["months_requiring_ingest"] == ["2024-12"]
    assert payload["verdict"] == "MISSING_SEC_COVERAGE"


def test_sec8k_coverage_audit_counts_manifest_raw_archive_gaps(
    tmp_path: Path,
) -> None:
    manifest_path = (
        tmp_path / "data" / "curated" / "sec" / "sec8k" / "manifest" / "data.parquet"
    )
    manifest_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "archive_cik": "320193",
                "accession": "0000320193-25-000001",
                "accession_no_dashes": "000032019325000001",
                "filed_date": "2025-02-03",
                "form": "8-K",
                "index_filename": "edgar/data/320193/000032019325000001/0000320193-25-000001.txt",
            }
        ]
    ).to_parquet(manifest_path, index=False)
    filing_path = tmp_path / "data" / "reference" / "sec_filing_index.parquet"
    filing_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "form": "8-K",
                "accessionNumber": "0000320193-25-000001",
                "filingDate": "2025-02-03",
                "acceptanceDateTime": "20250203163000",
                "items": ["3.02"],
                "ticker": "AAPL",
            }
        ]
    ).to_parquet(filing_path, index=False)

    payload = run_sec8k_coverage_audit(
        data_root=tmp_path,
        start_date="2025-02-01",
        end_date="2025-02-28",
        target_items=("3.02",),
        fallback_target_items=(),
        horizons=(5,),
    )["packet"]

    assert payload["raw_archive_gap_months"] == ["2025-02"]
    assert payload["raw_archive_gap_count"] == 1
    assert payload["months_requiring_ingest"] == ["2025-02"]
    assert payload["verdict"] == "MISSING_SEC_COVERAGE"


def test_sec8k_coverage_expand_ingests_only_missing_months_and_rebuilds_global_candidates(
    tmp_path: Path,
) -> None:
    connector = _FakeSec8KConnector()

    payload = run_sec8k_coverage_expand(
        data_root=tmp_path,
        connector=connector,
        start_date="2025-01-01",
        end_date="2025-02-28",
        target_items=("3.02",),
        fallback_target_items=(),
        max_retrieval_attempts=1,
        rate_limit_pause_seconds=0.0,
        sleep_fn=lambda _seconds: None,
    )["packet"]

    assert payload["planned_months"] == ["2025-01", "2025-02"]
    assert connector.fetch_windows == [
        ("sec8k_index", "2025-01-01", "2025-01-31"),
        ("company_tickers", "2025-01-01", "2025-01-31"),
        ("sec8k_index", "2025-02-01", "2025-02-28"),
        ("company_tickers", "2025-02-01", "2025-02-28"),
    ]
    candidates = pd.read_parquet(
        tmp_path / "data" / "curated" / "events" / "sec_8k_item_events" / "data.parquet"
    )
    assert sorted(candidates["filing_date"].astype(str).unique().tolist()) == [
        "2025-01-15",
        "2025-02-15",
    ]
    assert set(candidates["sec_item_number"].astype(str)) == {"3.02"}
    assert payload["after_audit"]["months_requiring_ingest"] == []


def test_sec_event_semantic_coverage_gate_does_not_backfill_missing_ticker_only(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import trademl.events.sec8k_coverage as coverage

    monkeypatch.setattr(
        coverage,
        "run_sec8k_coverage_audit",
        lambda **_kwargs: {"packet": {"months_requiring_ingest": []}},
    )
    monkeypatch.setattr(
        coverage,
        "run_sec8k_candidate_curation",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(
        coverage,
        "run_sec_event_semantic_labelability_audit",
        lambda **_kwargs: {
            "payload": {
                "candidate_count": 150,
                "labelable_count": 20,
                "blocker_counts": {"missing_ticker": 130},
            }
        },
    )
    backfill_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        coverage,
        "run_sec8k_market_backfill_from_env",
        lambda **kwargs: backfill_calls.append(kwargs) or {"verdict": "PASS"},
    )
    monkeypatch.setattr(
        coverage,
        "run_sec_event_semantic_scaled_gate",
        lambda **_kwargs: {"packet": {"verdict": {"decision": "SHOULD_NOT_RUN"}}},
    )

    payload = run_sec_event_semantic_coverage_gate(
        data_root=tmp_path,
        start_date="2025-01-01",
        end_date="2025-12-31",
        min_sample=100,
    )["packet"]

    assert backfill_calls == []
    assert payload["scaled_gate"] is None
    assert payload["verdict"]["decision"] == "BLOCKED_DATA_COVERAGE"
    assert payload["verdict"]["paper_live_allowed"] is False


def test_sec_event_semantic_coverage_gate_blocks_when_source_coverage_remains_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import trademl.events.sec8k_coverage as coverage

    monkeypatch.setattr(
        coverage,
        "run_sec8k_coverage_audit",
        lambda **_kwargs: {"packet": {"months_requiring_ingest": ["2024-01"]}},
    )
    monkeypatch.setattr(coverage, "run_sec8k_candidate_curation", lambda **_kwargs: {})
    monkeypatch.setattr(
        coverage,
        "run_sec8k_coverage_expand",
        lambda **_kwargs: {"packet": {"after_audit": {"months_requiring_ingest": ["2024-01"]}}},
    )
    monkeypatch.setattr(
        coverage,
        "run_sec_event_semantic_labelability_audit",
        lambda **_kwargs: {
            "payload": {"candidate_count": 150, "labelable_count": 120, "blocker_counts": {}}
        },
    )
    monkeypatch.setattr(
        coverage,
        "run_sec_event_semantic_scaled_gate",
        lambda **_kwargs: {"packet": {"verdict": {"decision": "SHOULD_NOT_RUN"}}},
    )

    payload = run_sec_event_semantic_coverage_gate(
        data_root=tmp_path,
        start_date="2024-01-01",
        end_date="2025-12-31",
        min_sample=100,
    )["packet"]

    assert payload["scaled_gate"] is None
    assert payload["verdict"]["decision"] == "BLOCKED_DATA_COVERAGE"
    assert "sec_source_coverage_incomplete_after_expand" in payload["verdict"]["failed_gates"]


def test_sec_event_semantic_coverage_gate_repairs_market_blockers_before_scaled_gate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import trademl.events.sec8k_coverage as coverage

    monkeypatch.setattr(
        coverage,
        "run_sec8k_coverage_audit",
        lambda **_kwargs: {"packet": {"months_requiring_ingest": []}},
    )
    monkeypatch.setattr(coverage, "run_sec8k_candidate_curation", lambda **_kwargs: {})
    labelability_payloads = [
        {"candidate_count": 150, "labelable_count": 20, "blocker_counts": {"missing_entry_minute": 130}},
        {"candidate_count": 180, "labelable_count": 30, "blocker_counts": {"missing_entry_minute": 150}},
        {"candidate_count": 180, "labelable_count": 120, "blocker_counts": {}},
    ]

    def fake_labelability(**_kwargs):
        return {"payload": labelability_payloads.pop(0)}

    monkeypatch.setattr(
        coverage,
        "run_sec_event_semantic_labelability_audit",
        fake_labelability,
    )
    backfill_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        coverage,
        "run_sec8k_market_backfill_from_env",
        lambda **kwargs: backfill_calls.append(kwargs) or {"verdict": "PASS"},
    )
    scaled_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        coverage,
        "run_sec_event_semantic_scaled_gate",
        lambda **kwargs: scaled_calls.append(kwargs)
        or {
            "packet": {
                "verdict": {
                    "decision": "CONTINUE_SEMANTIC_8K",
                    "paper_live_allowed": False,
                }
            }
        },
    )

    payload = run_sec_event_semantic_coverage_gate(
        data_root=tmp_path,
        start_date="2025-01-01",
        end_date="2025-12-31",
        min_sample=100,
    )["packet"]

    assert len(backfill_calls) == 1
    assert backfill_calls[0]["candidate_source"] == "sec8k_item_events"
    assert backfill_calls[0]["target_items"] == ("3.02", "4.01", "2.04", "1.01")
    assert len(scaled_calls) == 1
    assert scaled_calls[0]["target_items"] == ("3.02", "4.01", "2.04")
    assert scaled_calls[0]["fallback_target_items"] == ("1.01",)
    assert payload["verdict"]["decision"] == "CONTINUE_SEMANTIC_8K"
    assert payload["verdict"]["paper_live_allowed"] is False


class _FakeSec8KConnector:
    vendor_name = "sec-fixture"

    def __init__(self) -> None:
        self.fetch_windows: list[tuple[str, str, str]] = []

    def fetch(
        self,
        dataset: str,
        symbols: list[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        self.fetch_windows.append((dataset, start_date, end_date))
        if dataset == "company_tickers":
            return pd.DataFrame([{"cik_str": 320193, "ticker": "AAPL"}])
        month = start_date[:7]
        accession = f"0000320193-{start_date[2:4]}-{month[-2:]}0001"
        return pd.DataFrame(
            [
                {
                    "archive_cik": "320193",
                    "form": "8-K",
                    "filed_date": f"{month}-15",
                    "index_filename": f"edgar/data/320193/{accession}.txt",
                    "accession": accession,
                    "accession_no_dashes": accession.replace("-", ""),
                    "discovery_source": "sec_full_index",
                    "index_year": int(start_date[:4]),
                    "index_quarter": 1,
                    "index_file_hash": "fixture",
                    "index_crawled_at": f"{month}-16T00:00:00+00:00",
                }
            ]
        )

    def retrieve_complete_submission_text(
        self, *, index_filename: str, endpoint_key: str = "sec8k_complete_txt"
    ) -> tuple[int, str, str]:
        accession = Path(index_filename).stem
        month = accession.split("-")[2][:2]
        text = _complete_text(f"2025{month}15163000")
        return 200, text, f"https://www.sec.gov/Archives/{index_filename}"


def _write_april_only_candidate_fixture(root: Path) -> None:
    _write_candidate_rows(
        root,
        [
            {
                "event_id": "labelable",
                "accession": "0000320193-25-000001",
                "ticker": "AAPL",
                "sec_item_number": "3.02",
                "eligibility_pass": True,
                "exclusion_reasons": [],
            },
            {
                "event_id": "missing-minute",
                "accession": "0000320193-25-000002",
                "ticker": "MSFT",
                "sec_item_number": "4.01",
                "eligibility_pass": True,
                "exclusion_reasons": [],
            },
            {
                "event_id": "missing-ticker",
                "accession": "0000320193-25-000003",
                "ticker": "",
                "sec_item_number": "3.02",
                "eligibility_pass": False,
                "exclusion_reasons": ["missing_ticker"],
            },
        ],
    )


def _write_candidate_rows(root: Path, rows: list[dict[str, object]]) -> None:
    path = root / "data" / "curated" / "events" / "sec_8k_item_events" / "data.parquet"
    path.parent.mkdir(parents=True)
    base = {
        "accessions": ["0000320193-25-000001"],
        "issuer_cik": "0000320193",
        "accepted_at_utc": "2025-04-07T20:30:00+00:00",
        "filing_date": "2025-04-07",
        "primary_document": "form8k.htm",
        "event_type": "8K_ITEM_3_02_UNREGISTERED_SALE_FINANCING",
        "eligibility_pass": True,
        "exclusion_reasons": [],
    }
    pd.DataFrame([{**base, **row} for row in rows]).to_parquet(path, index=False)


def _write_labelable_market_fixture(root: Path) -> None:
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


def _complete_text(accepted_at: str) -> str:
    return f"""<SEC-DOCUMENT>
<SEC-HEADER>
<ACCEPTANCE-DATETIME>{accepted_at}
</SEC-HEADER>
<DOCUMENT>
<TYPE>8-K
<FILENAME>form8k.htm
<TEXT>
Item 3.02 Unregistered Sales of Equity Securities.
The Company sold unregistered common stock in a private placement.
Item 9.01 Financial Statements and Exhibits.
</TEXT>
</DOCUMENT>
</SEC-DOCUMENT>
"""
