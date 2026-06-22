# SEC 8-K Item Event MVP

Last updated: 2026-05-06

Parent source of truth:

- [Public Event Research Pivot SSOT](public_event_research_pivot_ssot.md)

## Decision

The deterministic SEC 8-K item-family MVP is killed as an alpha candidate.

This came after the Form 4 baseline and bounded rework gate both failed
promotion criteria. The SEC 8-K implementation remains useful source-first event
infrastructure, but broad item families like Item 1.01, 7.01, and 8.01 are not a
valid paper/live signal and should not receive more alpha-search budget without
a new, narrower documented hypothesis.

Important scope correction: this result kills **header-only deterministic item
classification**, not semantic 8-K research. Future 8-K work must use an LLM or
purpose-built model for exhibit/section semantic classification and structured
extraction. Regex or deterministic matching may only identify stable document
structure such as item boundaries, SGML tags, exhibit tables, timestamps, paths,
and hashes.

## Initial Event Families

The killed first implementation is deterministic and does not use LLM semantic
extraction. It exists as a negative-control baseline for item headers, not as a
template for future semantic event parsing.

Supported item families:

- `8K_ITEM_1_01_MATERIAL_AGREEMENT`;
- `8K_ITEM_2_05_EXIT_DISPOSAL_COSTS`;
- `8K_ITEM_4_01_AUDITOR_CHANGE`;
- `8K_ITEM_7_01_REG_FD`;
- `8K_ITEM_8_01_OTHER`;
- `8K_ITEM_2_02_RESULTS_OPERATIONS` as an earnings/contamination bucket, not the
  primary alpha target.

The scaffold inventories exhibits by document type, filename, description, and
hash. Exhibit semantics are intentionally deferred until the item-level pipeline
proves the source, timestamp, label, and control mechanics.

Any future exhibit/materiality MVP must use model-based semantic extraction into
a strict schema with evidence snippets/source offsets. It must not use regex,
keyword lists, or deterministic heuristics to decide materiality, sentiment,
dilution, contract substance, auditor trouble, covenant/default stress, customer
loss, guidance changes, or any other economic meaning.

## Current Implementation Status

Implemented command surface:

- `trademl research sec8k-ingest`;
- `trademl research sec8k-candidates`;
- `trademl research sec8k-market-backfill`;
- `trademl research sec8k-event-study`;
- `trademl research sec8k-decision`;
- `trademl research sec-event-semantic-gate` for the separate LLM/model
  semantic classifier fixture gate;
- `trademl research sec-event-semantic-classify`;
- `trademl research sec-event-semantic-study`.

Artifacts:

- `data/raw/sec/sec8k_manifest/year=YYYY/qtr=Q/manifest.parquet`;
- `data/curated/sec/sec8k/manifest/data.parquet`;
- `data/raw/sec/archives/archive_cik={archive_cik}/accession={accession_no_dashes}/complete.txt`;
- `data/raw/sec/archives/archive_cik={archive_cik}/accession={accession_no_dashes}/metadata.json`;
- `data/reference/sec_filing_index.parquet`;
- `data/reference/sec_company_tickers.parquet`;
- `data/curated/sec/sec8k/filing_index/data.parquet`;
- `data/curated/events/sec_8k_item_events/data.parquet`;
- `data/curated/events/sec_8k_item_labels/data.parquet`;
- `data/curated/events/sec_8k_timestamp_placebo_labels/data.parquet`;
- `data/curated/events/sec_event_semantic_snippets/data.parquet`;
- `data/curated/events/sec_event_semantic_classifications/data.parquet`;
- `data/curated/events/sec_event_semantic_candidates/data.parquet`;
- `data/curated/events/sec_event_semantic_labels/data.parquet`;
- `data/curated/events/sec_event_semantic_timestamp_placebo_labels/data.parquet`;
- `control/cluster/state/research/sec8k_ingest/latest.json`;
- `control/cluster/state/research/sec8k_market_backfill/latest.json`;
- `control/cluster/state/research/sec_8k_item_events/latest.json`;
- `control/cluster/state/research/sec_8k_item_labels/latest.json`;
- `control/cluster/state/research/sec_8k_timestamp_placebo_labels/latest.json`;
- `control/cluster/state/research/sec_8k_event_study/latest.json`;
- `control/cluster/state/research/sec_8k_decision/latest.json`;
- `control/cluster/state/research/sec_event_semantic_fixture_gate/latest.json`;
- `control/cluster/state/research/sec_event_semantic_classification/latest.json`;
- `control/cluster/state/research/sec_event_semantic_study/latest.json`;
- `reports/research/sec_8k_event_study/latest.md`.
- `reports/research/sec_8k_decision/latest.md`.
- `reports/research/sec_event_semantic_fixture_gate/latest.md`.
- `reports/research/sec_event_semantic_study/latest.md`.

Semantic gate status:

- Default local model candidate: `qwen3.5-9b-mlx` through LM Studio.
- The gate forbids numeric confidence and requires categorical certainty.
- Evidence quotes and materiality-evidence snippets must be exact substrings
  from the excerpt.
- Auxiliary extracted fields are retained for inspection; non-exact auxiliary
  fields are warnings and are not downstream feature inputs.
- `MATERIAL_CONTRACT_AWARD` requires explicit materiality evidence emitted into
  the structured payload, not just a model label.
- Passing the semantic gate only validates the extraction path. It does not
  revive broad deterministic 8-K item families or authorize paper/live use.
- `sec-event-semantic-classify` applies that gate to real archived 8-K item and
  exhibit snippets. `sec-event-semantic-study` then labels only validated
  semantic candidates and writes a continue/rework/kill packet. Header-only
  deterministic 8-K remains a negative-control baseline.

First live SEC/NAS canary:

- Command:
  `trademl research --env-file .env --data-root /Volumes/dev/TradeML sec8k-ingest --start-date 2025-04-07 --end-date 2025-04-07 --limit 20 --max-retrieval-attempts 3 --rate-limit-pause-seconds 10`.
- Result: `PASS`, 20 manifest rows, 20 parsed submissions, 0 failed
  retrievals, 23 deterministic item candidates, 13 eligible candidates.
- Candidate families: 5 `8K_ITEM_1_01_MATERIAL_AGREEMENT`, 7
  `8K_ITEM_7_01_REG_FD`, 5 `8K_ITEM_8_01_OTHER`, and 6
  `SEC_8K_UNCLASSIFIED`.
- Follow-up event-study command:
  `trademl research --env-file .env --data-root /Volumes/dev/TradeML sec8k-event-study --horizon 5 --primary-horizon 5`.
- Initial result: `DIAGNOSTIC_ONLY`; only 1 of 23 candidates labeled because the
  current NAS minute slice only contained `DDD` among the candidate symbols.
- Market backfill command:
  `trademl research --env-file .env --data-root /Volumes/dev/TradeML sec8k-market-backfill --horizon 5 --max-fetch-attempts 3 --rate-limit-pause-seconds 10 --daily-symbol-batch-size 25`.
- Backfill result: `PARTIAL_COVERAGE`; 13 eligible primary candidates and 26
  eligible backfill candidates including timestamp placebo. The backfill wrote
  1,064 minute rows and 297 daily rows, with no retries. Remaining empty daily
  symbols were `ARTNB` and `LBRA`; remaining missing minute symbols after
  label-load were `AIM`, `ARTNB`, and `LBRA`.
- Post-backfill event-study result: `DIAGNOSTIC_ONLY`; 9 labeled, 4 blocked,
  10 skipped ineligible. Primary 5-day abnormal net return was mean
  `-0.018169847182681387` and median `0.008117473668973654` on `n=9`, which is
  diagnostic only and not a tradable result.

Decision slice:

- Commands:
  - `trademl research --env-file .env --data-root /Volumes/dev/TradeML sec8k-ingest --start-date 2025-04-01 --end-date 2025-04-30 --limit 1000 --max-retrieval-attempts 3 --rate-limit-pause-seconds 10`;
  - `trademl research --env-file .env --data-root /Volumes/dev/TradeML sec8k-market-backfill --horizon 5 --max-fetch-attempts 3 --rate-limit-pause-seconds 10 --daily-symbol-batch-size 100`;
  - `trademl research --env-file .env --data-root /Volumes/dev/TradeML sec8k-event-study --horizon 5 --primary-horizon 5`;
  - `trademl research --env-file .env --data-root /Volumes/dev/TradeML sec8k-decision`.
- Ingest result: `PASS`, 1,000 manifest rows, 1,000 parsed submissions, 0
  failed retrievals, 1,151 deterministic item candidates, 708 eligible.
- Backfill result: `PARTIAL_COVERAGE`, 99,450 minute rows and 17,469 daily rows
  written/visible; 223 empty minute symbol-date requests and 31 empty daily
  symbols remained as coverage gaps.
- Event-study result: `DIAGNOSTIC_ONLY`, 600 labeled, 108 blocked, 443 skipped
  ineligible. Primary 5-day abnormal net return was mean
  `-0.014125810557440959`, median `-0.005956147455785862`, hit rate
  `0.4583333333333333`, and bootstrap 95% mean CI
  `[-0.029625197882178395, 0.0033727051232170587]`.
- Timestamp-placebo control: 580 labeled, mean `0.030092210541045782`; real
  events underperformed placebo by `-0.04421802109848674`.
- Item-family notes:
  - `8K_ITEM_1_01_MATERIAL_AGREEMENT`: n=93, mean `-0.05524046269702089`,
    median `-0.020832442854401022`, bootstrap CI fully below zero.
  - `8K_ITEM_7_01_REG_FD`: n=190, mean `0.005789041089902558`, but median
    `-0.009039442461356028`, hit rate `0.4473684210526316`, and bootstrap CI
    includes zero.
  - `8K_ITEM_8_01_OTHER`: n=115, mean `-0.040273823651073924`, median
    `-0.0173920881832752`, bootstrap CI almost entirely below zero.
- Final decision artifact: `BROAD_SEC8K_ITEM_FAMILIES_KILLED`, with
  `move_forward=false`.

Current verdict semantics:

- `DIAGNOSTIC_ONLY` means labels exist and the packet is ready for research
  review.
- `BLOCKED_DATA_COVERAGE` means candidate events could not be labeled from
  available minute/daily market data.
- `BROAD_SEC8K_ITEM_FAMILIES_KILLED` means the broad deterministic item-family
  alpha candidate failed the continue gates and should not be promoted or
  searched further without a narrower hypothesis.
- Paper/live use is always blocked in this MVP.

## Next Gate

Do not continue broad deterministic 8-K item-family alpha. Keep the SEC ingest,
archive, candidate, market-backfill, label, control, and decision machinery as
source-first infrastructure.

The next public-event continuation should be either a narrower 8-K hypothesis
with LLM/model exhibit/materiality extraction and explicit exclusion logic, or a
different source-first event class. Do not add options or paper/live signals from
this deterministic item-family MVP.
