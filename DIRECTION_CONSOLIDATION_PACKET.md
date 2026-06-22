# TradeML Direction Consolidation Packet

Last updated: 2026-05-21

## Purpose

This packet is for an external review of TradeML's direction. It consolidates
the major guidance, redirections, source documents, implementation history, and
known conflicts so Zach can converge the repo onto one final SSOT.

This is not the final SSOT. It is the map of the mess.

## Review Ask For Claude

Please help consolidate TradeML into one single active SSOT.

The repo currently contains:

- an original infrastructure/cross-sectional-ranker SSOT;
- a GPT Pro-driven public-event research pivot;
- postmortems for Form 4 and broad SEC 8-K tests;
- operational handoffs from Pi/Mac/NAS migration and autonomous research work;
- provider/data-node policies;
- agent/coding guardrails from `CLAUDE.md` / `AGENTS.md`;
- stale always-on autopilot docs that conflict with the current runtime freeze.

Your job is to decide:

1. What should survive into the final SSOT.
2. What should become historical/archive-only.
3. What conflicts or stale instructions need to be removed.
4. What the old system was.
5. What the new system is.
6. What exact next gate should happen before anything autonomous restarts.
7. Which implementation surfaces are still useful infrastructure versus dead
   alpha direction.

## Current High-Level Truth

TradeML is paused while direction is being consolidated.

- Raspberry Pi collector was stopped and disabled.
- Mac mini research LaunchAgent was booted out/disabled.
- NAS mount plumbing was left alone because it is not an autonomous research
  loop.
- No paper/live trading is authorized.
- No broad autonomous research loop should restart until the next gate is chosen
  and explicitly tied to one accepted SSOT.

Current best working summary:

> TradeML should probably keep its PIT data/storage/research infrastructure, but
> its alpha direction should shift away from generic broad daily ranker/autopilot
> search and toward narrow, pre-registered, source-first public-event hypotheses
> with strict evidence, controls, and kill criteria.

That statement is provisional. The point of this packet is to decide whether it
is correct and how to encode it cleanly.

## User Redirections And Preferences To Preserve

These are directionally important because they changed how the repo should be
interpreted:

- Zach wants one single SSOT, not a pile of overlapping strategy docs.
- Zach wants clear boundaries between:
  - the old system;
  - the new proposed system;
  - what was tested;
  - what remains viable;
  - what is killed or historical.
- Zach does not want agents to keep mutating `HANDOFF.md` as a running log.
  Handoff-style artifacts are useful only when intentionally created for review
  or pickup.
- Zach wants runtime truth, not inherited plan theater. If a process, transfer,
  artifact, model, or result is claimed, verify it from live state/logs/files.
- Zach strongly rejected regex/keyword rules for narrative/economic semantic
  extraction.
- Deterministic parsing is allowed for stable source structure: SEC accession
  metadata, SGML/XML tags, item-section boundaries, timestamps, paths, hashes,
  and schema-backed source fields.
- Narrative/economic meaning must come from an LLM or purpose-built model into a
  strict schema, then be validated deterministically against exact evidence
  spans/source offsets.
- No model should allocate capital directly or explain away failed gates.
- No live trading or live Alpaca endpoint use is authorized.
- Paper/shadow artifacts are diagnostics only until a future explicit promotion
  plan says otherwise.
- SQLite stays local to the machine that owns it; parquet/raw/curated artifacts
  go to the NAS.
- Canonical storage should be under the NAS `dev` share:
  - local Mac: `/Volumes/dev/TradeML`;
  - Mac mini: `/Users/openclaw/atlas_mounts/dev/TradeML`;
  - Raspberry Pi: `/mnt/dev/TradeML`.
- The old local fake-NAS / `atlas_mounts` Mac path should not be revived as a
  local data root.

## Current Repo Shape

Project Markdown files found, excluding `.venv` and `.pytest_cache`:

- `AGENTS.md`
- `CLAUDE.md`
- `DEV_GUIDE.md`
- `HANDOFF.md`
- `README.md`
- `SSOT.md`
- `docs/AUTOPILOT_ROADMAP.md`
- `docs/Data_Sourcing_Playbook.md`
- `docs/FUTURE_DATA_VENDOR_MAP_2026-04-10.md`
- `docs/Provider_Doc_Audit.md`
- `docs/Provider_Role_Matrix.md`
- `docs/provider_contract_audit_2026-04-09.md`
- `docs/research/PATH_FORWARD_SSOT.md`
- `docs/research/autonomous_research_architecture_policy.md`
- `docs/research/form4_insider_purchase_event_mvp_direction.md`
- `docs/research/gpt_pro_architecture_recommendation_2026-04-29.md`
- `docs/research/paper_trading_api.md`
- `docs/research/public_event_research_pivot_ssot.md`
- `docs/research/sec_8k_item_event_mvp_direction.md`
- `docs/research/session_handoff_2026-05-04.md`
- `docs/research/source_prompts/2026-05-05_gpt55_cleaned_public_event_foundry_prompt.md`
- `docs/research/source_prompts/2026-05-05_gpt_pro_form4_historical_retrieval_path.md`
- `docs/research/source_prompts/2026-05-05_gpt_pro_form4_mvp.md`
- `docs/research/source_prompts/2026-05-05_gpt_pro_original_brutal_take.md`
- `docs/research/source_prompts/README.md`

Non-project Markdown exists under `.venv/` and `.pytest_cache/`; ignore it for
direction review.

The worktree is dirty. Direction review should account for uncommitted docs and
implementation work rather than assuming `main` is clean/current.

Notable dirty/untracked areas observed during the audit:

- root guidance/docs: `AGENTS.md`, `CLAUDE.md`, `SSOT.md`, `HANDOFF.md`;
- data-node/provider/config surfaces: `.env.template`, `configs/node.yml`,
  `src/trademl/data_node/*`, `src/trademl/connectors/sec_edgar.py`,
  `src/trademl/data_node/provider_contracts.py`;
- dashboard/fleet/autopilot surfaces:
  `src/trademl/dashboard/controller.py`, `src/trademl/fleet/autopilot.py`;
- Mac/NAS operational scripts: `ops/install_macos_nas_launchagent.sh`,
  `ops/macos_mount_trademl_nas.sh`,
  `ops/consolidate_trademl_storage.sh`,
  `ops/consolidate_trademl_sec_parallel_resume.sh`;
- event research implementation: `src/trademl/events/`,
  `tests/unit/test_form4.py`, `tests/unit/test_sec8k.py`,
  `tests/unit/test_sec8k_coverage.py`, `tests/unit/test_sec8k_semantic.py`,
  `tests/unit/test_semantic_classifier.py`;
- research docs and source prompts under `docs/research/`.

## Document Authority Map

### Must Read For Direction Consolidation

#### `docs/research/PATH_FORWARD_SSOT.md`

Current draft active research-direction document.

What it says:

- runtime is frozen;
- `SSOT.md` is infrastructure contract;
- this doc owns research direction/restart criteria;
- Form 4 naive insider-buy MVP is killed;
- broad deterministic 8-K item families are killed;
- semantic 8-K extraction is mechanically viable but too sparse to prove alpha;
- next restart must be a bounded offline research gate;
- paper/live remains blocked.

Review concern:

- It was written quickly as a consolidation draft. Treat it as the current best
  hypothesis, not as unquestionable truth.

#### `SSOT.md`

Original repo-wide engineering/infrastructure SSOT.

What it says:

- TradeML is an autonomous equities research system;
- Pi collects market data to NAS;
- training hosts run cross-sectional models;
- Phase 1 focuses on free-tier daily bars, Ridge baseline, LightGBM challenger,
  walk-forward validation, costs always on;
- later phases add cleaner security master, more features, paper trading,
  options, meta-stackers, and scale;
- PIT discipline, raw immutability, realistic costs, deterministic validation,
  and local SQLite are non-negotiable.

Review concern:

- It still describes the older generic daily ranker project. It now has a top
  note pointing at `PATH_FORWARD_SSOT.md` for research direction, but a final
  one-SSOT rewrite needs to decide whether to merge, archive, or rewrite the old
  phases.

#### `docs/research/public_event_research_pivot_ssot.md`

Historical pivot SSOT and source-first policy reference.

What it says:

- pivot away from generic architecture search toward a source-first public-event
  research foundry;
- keep Pi/Mac/NAS infrastructure;
- search for narrow, lawful, public-information event classes where data is
  annoying/fragmented/entity-linkage-heavy and capacity is small;
- LLMs extract/classify narrative semantics into strict schemas;
- deterministic code validates evidence and bookkeeping;
- broad generic ranker becomes sentinel/comparator, not main entry engine;
- defines event-pipeline objects, source backlog, validation rules, non-goals.

Review concern:

- It has historical language around first MVPs that can be misread as current
  active work. It should become background rationale or be merged into final
  SSOT.

#### `HANDOFF.md`

Operational/status bundle from 2026-05-12.

What it says:

- captures the pivot, no-regex semantic extraction, storage migration, Mac/Pi
  path truth, Form 4 and SEC 8-K outcomes, semantic 8-K commands, feature
  blockers, and useful operational commands;
- explains canonical NAS paths and old local/fake-NAS cleanup;
- records stale Mac mini and Pi runtime details before the later runtime freeze.

Review concern:

- Useful for what actually happened, but stale after 2026-05-20. Do not let it
  override the runtime freeze or next-gate policy.

### Strong Supporting Evidence

#### `docs/research/form4_insider_purchase_event_mvp_direction.md`

Postmortem plus reusable Form 4 source-pipeline specification.

What it says:

- Form 4 open-market insider purchases were chosen as the smallest first
  event-pipeline MVP;
- the first strict MVP and bounded rework were implemented/evaluated;
- final verdict: `FORM4_KILLED_BASELINE_COMPLETE`;
- no variant passed all continuation gates;
- Form 4 remains useful only as source-first event-pipeline benchmark
  infrastructure unless a new documented hypothesis changes the event definition
  or controls.

Review concern:

- Top-level “first MVP should be Form 4” language is historical. The result is
  killed, not current direction.

#### `docs/research/sec_8k_item_event_mvp_direction.md`

Postmortem plus reusable SEC/event mechanics reference.

What it says:

- deterministic broad 8-K item-family alpha is killed;
- broad header/item classification is a negative-control baseline only;
- future 8-K work must use LLM/model semantic extraction for exhibit/materiality
  and structured economic meaning;
- SEC ingest/archive/candidate/backfill/label/control/decision machinery remains
  useful;
- semantic 8-K result is not paper/live ready.

Review concern:

- Do not read this as permission to continue broad item-family search.

#### `docs/research/source_prompts/README.md`

Index for raw GPT Pro / GPT-5.5 source prompts.

What it says:

- raw GPT prompts are source artifacts, not polished strategy docs;
- use them to check whether active SSOT captured original intent;
- points to the brutal take, cleaned public-event foundry prompt, Form 4 MVP,
  and Form 4 retrieval path.

Review concern:

- Include the README in review to understand provenance. Only read full raw
  prompts if checking lost intent.

#### `docs/research/source_prompts/2026-05-05_gpt_pro_original_brutal_take.md`

Raw GPT Pro critique that drove the pivot.

Key direction:

- original architecture was too broad to falsify;
- avoid a unified financial world model;
- LLMs should produce auditable structured event objects, not trades;
- entity graph is useful plumbing, not GNN-first alpha;
- small-trader edge is more plausible in annoying, low-capacity,
  entity-linkage-heavy public data;
- concrete loop: hypothesis, data collection, features, labels, leakage checks,
  backtesting, statistical correction, paper trading, live micro-sizing only
  later, kill criteria.

Review concern:

- Source/provenance, not current instruction.

#### `docs/research/source_prompts/2026-05-05_gpt55_cleaned_public_event_foundry_prompt.md`

Cleaned implementation-oriented version of the GPT Pro pivot.

Key direction:

- build a public-event foundry;
- prioritize source records, entity table, event table, market data tables;
- proposes hypothesis families such as federal contract materiality drift, SEC
  exhibit novelty drift, PEAD/revisions, and later earnings options implied-move
  mispricing.

Review concern:

- Strong source material for final SSOT language, but not itself active.

#### `docs/research/source_prompts/2026-05-05_gpt_pro_form4_mvp.md`

Raw GPT Pro tactical Form 4 MVP recommendation.

Key direction:

- Form 4 was proposed as the smallest clean test of source-first event
  infrastructure;
- gives schema, labels, controls, pass/kill criteria, and implementation
  sequence.

Review concern:

- Historical. The actual Form 4 MVP later failed.

#### `docs/research/source_prompts/2026-05-05_gpt_pro_form4_historical_retrieval_path.md`

Raw GPT Pro retrieval/parser details for Form 4.

Key direction:

- use SEC full-index manifest, archive CIK, raw XML, `.txt` fallback, amendment
  handling, parser edge cases, fixture requirements, and source-quality
  telemetry.

Review concern:

- Valuable implementation detail if Form 4 parser/storage is reused. Not
  current alpha direction.

### Old System / Autonomy / Provider Guidance

#### `docs/AUTOPILOT_ROADMAP.md`

Old north-star operating model for always-on Pi and Mac loops.

What it says:

- Pi runs continuously and saturates quota lanes;
- Mac mini runs perpetual supervised research;
- LaunchAgent ownership and runbook;
- manual promotion gates.

Review concern:

- Conflicts with current runtime freeze and offline-next-gate requirement. Useful
  only as old-system context.

#### `docs/research/autonomous_research_architecture_policy.md`

Control-plane policy for the old Mac mini research loop.

What it says:

- supervised rank-return architecture priority:
  Ridge/rank sentinel, LightGBM workhorse, CatBoost/advanced challenger,
  ensemble/meta, deferred tabular deep;
- no RL/GNN/sequence/foundation forecaster as current alpha engine;
- promotion via rank IC, all-year positivity, cost-positive backtest,
  placebo/PBO, drawdown, turnover, and complexity penalty;
- includes no-regex semantic extraction policy.

Review concern:

- Partly stale: generic daily ranker is no longer active alpha direction.
  Semantic extraction guardrail still aligns.

#### `docs/Data_Sourcing_Playbook.md`

Vendor/data collection policy.

What it says:

- vendor roles for Alpaca, Tiingo, Massive/Polygon, Twelve Data, Finnhub, Alpha
  Vantage, FRED/ALFRED, FMP, SEC EDGAR;
- PIT rules;
- live-smoke notes;
- canonical bar and QC policy.

Review concern:

- Still useful for infrastructure if Pi/data node survives. Not alpha direction.

#### `docs/Provider_Role_Matrix.md`

Docs-backed operating policy for data collection.

What it says:

- canonical bars: Alpaca primary, Tiingo deep history, Twelve Data supplemental,
  Massive independent QC, Finnhub disabled for canonical bars;
- security-master/corp-actions/events vendor roles;
- runtime rules.

Review concern:

- Keep if final SSOT preserves data-node infrastructure.

#### `docs/FUTURE_DATA_VENDOR_MAP_2026-04-10.md`

Future raw archive/vendor map.

What it says:

- bank rolling-window and supplemental raw data without contaminating Phase 1;
- minute/news/event archives are research-only until PIT-safe curation exists.

Review concern:

- “Collect now” posture is constrained by runtime freeze.

#### `docs/Provider_Doc_Audit.md`

Provider capability/backlog audit.

What it says:

- vendor-by-vendor capability notes and possible upgrades.

Review concern:

- Low-level source note, not direction.

#### `docs/provider_contract_audit_2026-04-09.md`

Source note for provider contract implementation.

What it says:

- docs-backed contract decisions in `provider_contracts.py`: pagination, batch
  sizes, permanent failure semantics, critical-path eligibility.

Review concern:

- Useful only if reviewing data-node implementation.

#### `docs/research/paper_trading_api.md`

Paper trading guardrail.

What it says:

- Alpaca paper trading is default paper-money target;
- paper endpoint only;
- order submission disabled by default;
- no live Alpaca endpoint is used by research autopilot.

Review concern:

- Keep as guardrail if paper/shadow survives.

### Agent / Coding Guidance

#### `CLAUDE.md`

Coding-agent instruction file.

Useful guidance to absorb:

- `SSOT.md` is canonical engineering spec;
- follow `DEV_GUIDE.md` phase-by-phase unless direction is superseded;
- test-first where practical;
- dependency injection for external dependencies;
- no future leakage;
- deterministic outputs;
- rank normalization, not z-scores;
- universe-relative labels;
- walk-forward purge;
- costs always on;
- SQLite local to Pi;
- log everything;
- no regex/keyword semantic extraction for economic meaning;
- no S3/MinIO, Docker orchestration, Prefect/Airflow;
- no premature options/SVI/intraday, Kelly sizing, vol targeting, champion
  automation, market cap feature, or days-to-earnings predictive feature.

Review concern:

- It should not be sent as a standalone strategic doc. Absorb useful guardrails
  into the final SSOT or new agent instructions.
- It does not mention the current `PATH_FORWARD_SSOT.md` override.

#### `AGENTS.md`

Nearly duplicate agent instruction file.

Review concern:

- Contains stale pointer to `docs/SSOT_V3.md`, which is absent.
- Do not use as source of direction except to confirm duplicate guardrails.

#### `DEV_GUIDE.md`

Old implementation guide.

What it says:

- build original system phase-by-phase: scaffold, calendars/utilities,
  connectors, data node, features/labels, validation, models/portfolio, reports.

Review concern:

- Useful implementation sequencing for old infrastructure.
- Does not know about public-event foundry, killed MVPs, runtime freeze, or
  current next-gate policy.

#### `README.md`

Operator overview/quickstart.

Review concern:

- Useful orientation only. It does not establish current direction.

## What Was Actually Tested

### Form 4

Tested:

- naive strict open-market common-stock insider-buy event rule;
- bounded rework variants:
  - `clean_timely_unmixed`;
  - `officer_director_only`;
  - `ceo_cfo_only`;
  - `clustered_buyers`;
  - `large_purchase_q4`.

Verdict:

- `FORM4_KILLED_BASELINE_COMPLETE`;
- not promotable;
- no paper/live.

Important nuance:

- This kills the tested Form 4 baseline, not every possible insider-trading
  hypothesis forever.
- Continuing Form 4 would require a new documented hypothesis and fresh gate.

### Broad Deterministic SEC 8-K Item Families

Tested:

- deterministic broad item families such as Item 1.01, 7.01, 8.01, 4.01;
- April 2025 1,000-filing slice;
- market backfill, event labels, timestamp placebo, event study, decision
  packet.

Verdict:

- `BROAD_SEC8K_ITEM_FAMILIES_KILLED`;
- broad header/item alpha is not promotable;
- no paper/live.

Important nuance:

- This kills header-only deterministic item-family alpha.
- It does not kill semantic 8-K research using LLM/model extraction with strict
  evidence validation.

### Semantic SEC 8-K

Tested:

- schema-driven semantic classifier mechanics;
- targeted semantic route;
- labelability and scaled gate checks;
- event study on promoted semantic candidates.

Current state:

- classifier path mechanically passes;
- latest semantic study is `MORE_DATA_REQUIRED`;
- primary labeled sample was too small (`n=6` in the latest summarized packet);
- scaled gate failed on insufficient semantic event frequency;
- no paper/live.

Important nuance:

- Positive-looking tiny-sample returns are not evidence of edge.
- The next useful work is frequency/labelability-first hypothesis selection,
  not another always-on broad research loop.

## Old System Boundary

The old system was:

- Pi always-on collector;
- NAS as central parquet store;
- Mac mini perpetual research loop;
- generic cross-sectional daily ranker as primary alpha target;
- Ridge/LightGBM/CatBoost/ensemble challenger progression;
- feature-family canaries;
- automated research/control-plane progression;
- paper/shadow diagnostics but no live trading.

Useful old-system infrastructure:

- PIT raw/curated data discipline;
- NAS storage layout;
- local SQLite machine state;
- provider capability/contract machinery;
- SEC ingest/archive;
- event labels and controls;
- event-study reports and decision packets;
- dashboard/fleet observability;
- paper/shadow guardrails.

Likely stale old-system direction:

- generic daily ranker as primary alpha engine;
- always-on autonomous research loop before the next direction gate;
- broad architecture search over rank-return features;
- broad deterministic 8-K item-family alpha;
- naive Form 4 insider-buy alpha;
- LaunchAgent/autopilot docs that assume the system should be running now.

## New System Boundary

The new proposed system is:

- public-event research foundry;
- narrow hypothesis one at a time;
- source-first, PIT, append-only archive;
- event schemas and entity resolution;
- semantic extraction with LLM/model only where narrative meaning is needed;
- deterministic validation of exact evidence, offsets, timestamps, allowed
  schema, controls, and labels;
- frequency/labelability gate before expensive model work;
- event study with negative controls and timestamp placebo;
- continue/rework/kill packet;
- paper/shadow only after a later explicit shadow-validation gate.

Open question for Claude:

- Should this fully replace the old original SSOT, or should it become the
  research layer on top of the old infrastructure SSOT?

## Candidate Final SSOT Shape

A single final SSOT should probably include:

1. Current status and runtime freeze.
2. What TradeML is now.
3. What TradeML is not.
4. Old system components to keep.
5. Old system components to retire/archive.
6. Current research thesis.
7. Semantic extraction and evidence policy.
8. Data/storage/runtime invariants.
9. Provider/data-node boundaries.
10. Event hypothesis workflow.
11. Validation, controls, cost, and kill criteria.
12. Paper/live prohibition and future promotion gates.
13. Exact next gate.
14. Document authority/archival plan.

## Proposed Minimal Review Packet

For direction consolidation, send Claude:

1. `DIRECTION_CONSOLIDATION_PACKET.md` (this file)
2. `docs/research/PATH_FORWARD_SSOT.md`
3. `SSOT.md`
4. `docs/research/public_event_research_pivot_ssot.md`
5. `HANDOFF.md`
6. `docs/research/source_prompts/README.md`

If Claude needs original GPT Pro source material, add:

7. `docs/research/source_prompts/2026-05-05_gpt_pro_original_brutal_take.md`
8. `docs/research/source_prompts/2026-05-05_gpt55_cleaned_public_event_foundry_prompt.md`

If Claude needs proof/postmortem details, add:

9. `docs/research/form4_insider_purchase_event_mvp_direction.md`
10. `docs/research/sec_8k_item_event_mvp_direction.md`

If Claude reviews data-node/provider survival, add:

11. `docs/Provider_Role_Matrix.md`
12. `docs/Data_Sourcing_Playbook.md`

If Claude reviews paper/shadow boundaries, add:

13. `docs/research/paper_trading_api.md`

Do not send as primary direction:

- `CLAUDE.md` or `AGENTS.md` as standalone strategic docs;
- `README.md`;
- `DEV_GUIDE.md` unless implementation phasing is under review;
- `docs/AUTOPILOT_ROADMAP.md` unless old autonomous-loop context is needed;
- `.venv/` or `.pytest_cache/` Markdown.

## Questions Claude Should Answer

1. Should `PATH_FORWARD_SSOT.md` become the final SSOT, or should it be merged
   into a rewritten root `SSOT.md`?
2. Should the final SSOT preserve the old Phase 1/2/3 structure at all?
3. Which old infrastructure is genuinely still needed for the public-event
   foundry?
4. Which docs should be archived or marked historical?
5. Should the generic daily ranker remain as sentinel only, or be removed from
   active implementation plans?
6. What exact next public-event hypothesis should be frequency-tested first?
7. What minimum evidence would justify restarting Pi collection or Mac mini
   research?
8. What should the single document authority hierarchy be after consolidation?
9. How should agent instructions be rewritten so future agents stop drifting
   between old and new systems?
10. What evidence from Form 4 / 8-K is strong enough to encode as final verdict
    versus merely provisional?
