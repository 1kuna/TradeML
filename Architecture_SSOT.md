Single Source of Truth (SSOT): Backfill, Regime Windows, and Model Architecture

Purpose
A complete, implementation‑ready specification that unifies our backfill subsystem, regime‑aware training policy, and multi‑model architecture with routing and promotion. This SSOT supersedes previous drafts where conflicts exist. All configs, tables, and CLI contracts here are canonical.

⸻

0) Core Decisions (take these as law)
	1.	Two sweeps, always‑on:
	•	Forward sweep (today → future) ingests daily deltas with bookmarks.
	•	Backward sweep (today → past) continuously backfills history in rate‑limit‑aware chunks until coverage targets are GREEN.
	2.	Train only on GREEN: Each model declares table dependencies and GREEN thresholds over its train window. If unmet, the trainer refuses to run.
	3.	Regime‑bounded training: Collect broadly; fit narrowly using modern regime masks + time‑decay. Older eras are for stress testing and priors, not for model fitting.
	4.	Specialists > monolith: Maintain separate expert models (daily equities, intraday equities, options vol). Use a router/meta‑blender gated by regime flags and data completeness. Do not train an all‑in monolith.
	5.	Stable curated schemas: Vendor changes (free → prosumer) occur beneath the curated layer with no changes to research code.
	6.	PIT discipline first: Raw, immutable payloads with arrival timestamps; curated/adjusted views downstream with full lineage.

⸻

1) System Layout (paths, services, versioning)

/repo
  /infra                 # docker-compose: Postgres, MinIO (S3), scheduler
  /data_layer
    raw/                 # immutable Parquet, partitioned by date/symbol/source
    curated/             # adjusted, leakage‑safe tables; schema is stable
    reference/           # corp_actions, delistings, calendars, index_membership, tick_size_regime
    manifests/           # bookmarks.json (forward), backfill_marks.json (backward)
    qc/                  # partition_status.parquet (GREEN/AMBER/RED + metrics)
  /feature_store         # equities/, options/
  /labeling              # horizon returns, triple‑barrier
  /validation            # CPCV, PBO, DSR utilities
  /models                # equities_xs/, intraday_xs/, options_vol/
  /ops                   # orchestration, CLI, monitoring, reports

	•	Storage: Parquet on MinIO/S3; bucket versioning enabled.
	•	Metadata: Postgres for run logs, partition_status, backfill_queue.
	•	Orchestration: Prefect/Airflow; nightly DAG defined in §7.
	•	Registry: MLflow for models + lineage (champion/challenger).

⸻

2) Data Completeness Model (GREEN/AMBER/RED)

Completeness is evaluated per (source, table, symbol, dt).
	•	GREEN: partition exists, QC passes (row bounds, calendars match, checksums).
	•	AMBER: partition exists but marginal (partial day, under‑rows, vendor outage, low QC).
	•	RED: missing partition.

Artifacts
	•	qc/partition_status.parquet — canonical completeness ledger (also mirrored in DB for SQL queries).
	•	manifests/bookmarks.json — forward sweep per‑worker watermarks.
	•	manifests/backfill_marks.json — backward sweep per‑table cursors.

⸻

3) DB Schema (minimal, sufficient)

-- Completeness/QC — one row per (source, table, symbol, dt)
CREATE TABLE IF NOT EXISTS partition_status (
  source TEXT,
  table_name TEXT,
  symbol TEXT,
  dt DATE,
  status TEXT CHECK (status IN ('GREEN','AMBER','RED')),
  rows INT,
  expected_rows INT,
  qc_score REAL,
  last_checked TIMESTAMP,
  notes TEXT,
  PRIMARY KEY (source, table_name, symbol, dt)
);

-- Backfill queue ledger
CREATE TABLE IF NOT EXISTS backfill_queue (
  id SERIAL PRIMARY KEY,
  source TEXT,
  table_name TEXT,
  symbol TEXT,
  dt DATE,
  priority INT,
  attempts INT DEFAULT 0,
  enqueued_at TIMESTAMP,
  last_attempt TIMESTAMP,
  last_err TEXT,
  UNIQUE (source, table_name, symbol, dt)
);


⸻

4) Backfill Subsystem (control flow & rules)

4.1 Nightly control flow (two sweeps)
	1.	Audit / Planner
	•	Compare expected vs actual per partition using exchange calendars and table‑specific row expectations.
	•	Write/merge partition_status; (re)populate backfill_queue with RED/AMBER, prioritized by: (a) reference tables first (corp_actions, delistings), (b) proximity to today, (c) strategic weight (EOD > minute > options chains, etc.).
	2.	Backfill Worker (ROLE=edge)
	•	Lease per‑table lock (e.g., s3://.../locks/backfill.<table>.lock).
	•	Consume tasks in priority order; fetch in chunked windows honoring per‑vendor token buckets.
	•	Write to raw/.../.tmp → checksum/ETag → atomic move; update manifests; mark GREEN/AMBER with reason; advance backfill_mark.
	3.	Curator
	•	Detect new raw/ partitions; rebuild only affected curated/ partitions (incremental); recompute derived artifacts (rolling features, SVI surfaces) just for impacted dates; refresh completeness.

4.2 Idempotency, dedupe, and overlap safety
	•	Never overwrite existing partitions; only append missing ones.
	•	Closed‑interval windows (e.g., [t0, t1)) for deterministic replays.
	•	Concurrent workers coordinate via locks + ETag preconditions.

4.3 Rate‑limit governance
	•	Per‑vendor token buckets read from backfill.yml budgets.
	•	On budget pressure, shrink chunk size, sleep, or yield the lease.

4.4 Priorities (what to fill first)
	1.	Corporate actions & delistings (fix survivorship bias).
	2.	Equities EOD OHLCV (cheap, broad coverage).
	3.	Minute bars (top‑liquidity names first).
	4.	Options chains (underliers we actually research/trade).
	5.	Macro/ALFRED vintages (fetch full series early; deltas daily).

4.5 Acceptance tests for backfill
	1.	Synthetic gaps → audit flags RED → backfill fills → AMBER→GREEN flips.
	2.	Rate‑limit storm → worker shrinks chunks and eventually completes.
	3.	Dual workers → leases prevent duplication; ETags catch races.
	4.	Derived rebuild → only affected curated dates change; hashes stable elsewhere.
	5.	Late corp actions → OHLCV adjustments localized and deterministic.

⸻

5) Regime & Horizon Policy (collect vs train)

Key idea: Don’t chase the “oldest possible.” Respect structural breaks (decimalization, Reg NMS, LULD, Tick Size Pilot, impending half‑penny quoting). Use recent high‑fidelity eras for fitting; keep older periods for stress.

5.1 Canonical horizons (encode in backfill.yml)

horizons:
  equities_eod:    { collect_earliest: '1990-01-01', train_window_years: 15 }
  equities_minute: { collect_earliest: '2010-01-01', train_window_years: 10 }
  equities_ticks:  { collect_earliest: '2012-06-01', train_window_years: 4 }  # post-LULD
  options_chains:  { collect_earliest: '2012-01-01', train_window_years: 10 }
  options_nbbo:    { collect_earliest: '2010-01-01', train_window_years: 8 }

5.2 Regime masks for training configs

training:
  sample_weight: time_decay(half_life_months: 18)
  regime_masks:
    - { name: 'pre-2012',  use_for: 'stress_only' }
    - { name: '2012-2016', use_for: 'fit_and_validate' }
    - { name: '2016-2024', use_for: 'fit_and_validate' }
    - { name: '2025+',     use_for: 'monitor_only' }


⸻

6) Backfill Configuration (canonical backfill.yml)

scheduler:
  timezone: America/New_York

policy:
  max_concurrency: 3
  daily_api_budget:
    alpaca:  { requests: 5000 }
    fred:    { requests: 2000 }
    sec:     { requests: 1000 }
    av:      { requests: 500 }
    opra:    { requests: 20000 }   # when enabled

  priorities:
    - { table: corp_actions,     weight: 100 }
    - { table: delistings,       weight: 100 }
    - { table: equities_eod,     weight: 90 }
    - { table: equities_minute,  weight: 80 }
    - { table: options_chains,   weight: 70 }
    - { table: options_nbbo,     weight: 60 }

targets:
  equities_eod:    { earliest: '2000-01-01', chunk_days: 30, expect_rows: 1 }
  equities_minute: { earliest: '2010-01-01', chunk_days: 5,  expect_rows: 390 }
  options_chains:  { earliest: '2015-01-01', chunk_days: 1,  expect_contracts_min: 50 }

qc_thresholds:
  green_min_ratio: 0.98   # share of GREEN partitions required over train window
  rows_low_ratio:  0.90
  rows_high_ratio: 1.20
  allow_halfdays:  true

Note: Remove legacy IEX free‑tier plans; use Alpaca for minute/EOD dev; upgrade to Databento/Polygon for consolidated TAQ/OPRA when research requires. Keep curated schemas unchanged.

⸻

7) Orchestration (nightly DAG)

1) ingest.forward()                 # today’s deltas
2) audit.scan()                     # compute partition_status; enqueue gaps
3) backfill.run(budget=from YAML)   # fill hot gaps; advance marks
4) curate.incremental()             # rebuild curated for new raw partitions
5) qc.refresh()                     # recompute GREEN/AMBER/RED
6) train.if_ready(equities_xs)
7) train.if_ready(options_vol)
8) evaluate.cpcv_and_shadow()       # CPCV + DSR + PBO; shadow vs champion
9) promote.if_beat_champion()       # registry swap
10) report.emit_daily()             # blotter + scorecard

CLI targets

ata-audit gaps --tables equities_eod equities_minute options_chains
ata-backfill plan --max 500 --since 2020-01-01
ROLE=edge ata-backfill run --budget vendor=alpaca:3000,sec:500
ata-audit recheck --status AMBER --tables equities_minute


⸻

8) Model Set, Router, and Blending

8.1 Specialist models
	•	Equities_xs (daily cross‑sectional)
	•	Inputs: curated EOD OHLCV, reference events.
	•	Window: 10–15y with time‑decay.
	•	Algos: Ridge/Logit → LightGBM (monotone where appropriate).
	•	Validation: CPCV + PBO + DSR.
	•	Intraday_xs (minute) (optional; only when GREEN coverage is sufficient)
	•	Inputs: minute bars; simple microstructure features.
	•	Window: 5–10y modern era.
	•	Options_vol (IV surface dynamics)
	•	Inputs: options chains → NBBO mids → IV → SVI parameters + QC.
	•	Window: ~7–10y.
	•	Algo: GBDT baseline; shallow NN only if it beats.

8.2 Router / Meta‑Blender

Router inputs: (a) regime flags, (b) completeness vector (per table GREEN/AMBER/RED), (c) liquidity tier.

Routing rules (minimal viable):
	•	If required table for a sleeve is not GREEN within the sleeve’s window → skip training/inference for that sleeve/date.
	•	For inference today: if options data for a name/date is not GREEN → suppress options signals for that underlier today.

Meta‑blending: Fit a stacking regressor/classifier over OOS folds only where multiple signals are present. Else, fall back to the available expert. Keep CPCV consistency (no leakage across folds).

Router pseudocode

def route(asof, symbol, ctx):
    regime = ctx.regime_flags(asof)
    comp   = ctx.completeness(symbol, asof)

    signals = {}
    if comp.satisfies('equities_eod', window=years(15)):
        signals['equities_xs'] = predict_equities_xs(symbol, asof, regime)

    if comp.satisfies('equities_minute', window=years(5)):
        signals['intraday_xs'] = predict_intraday_xs(symbol, asof, regime)

    if comp.satisfies('options_chains', window=years(7)):
        signals['options_vol'] = predict_options_vol(symbol, asof, regime)

    return blend(signals, regime)


⸻

9) Training Gating, Retrain Triggers, and Promotion

9.1 GREEN‑gated training (hard rules)

Model	Requires GREEN on…	Threshold over train window
Equities_xs	equities_eod, corp_actions, delistings	≥ 98% of partitions GREEN (10–15y)
Intraday_xs	equities_minute, calendars	≥ 98% GREEN (5–10y)
Options_vol	options_chains (or NBBO), IV/Surface QC rate	≥ 95–98% GREEN (7–10y) + surface QC ≥ X%

The trainer refuses to run if thresholds are not met.

9.2 Retrain cadence and event triggers
	•	Cadence: Equities_xs weekly; Options_vol weekly; Intraday_xs bi‑weekly.
	•	Event triggers:
	1.	Coverage delta: GREEN share ↑ ≥ 1% since last train.
	2.	Drift: feature PSI/KL, or calibration (Brier/logloss) breaches.
	3.	Champion underperformance: rolling DSR below floor for N weeks.

9.3 Promotion (champion ↔ challenger)
	•	Compute CPCV metrics + DSR/PBO; run multi‑week shadow. Promote only if net OOS exceeds champion under identical costs/impact and DSR/PBO bars are met.

⸻

10) Validation & Anti‑Overfitting
	•	CPCV with purging and embargo; OOS windows ≥ 6 months (daily sleeves).
	•	Time‑decay sample weights; regime masks to avoid mixing incompatible eras.
	•	Report PBO and DSR for every candidate; log multiplicity.
	•	Maintain a dark hold‑out untouched until final sign‑off.

⸻

11) Data Sourcing Policy (free → prosumer)
	•	Equities minute/EOD: Alpaca for dev; Databento/Polygon for consolidated TAQ (upgrade when needed). Avoid vendor pre‑adjusted OHLCV unless we log the algorithm; apply our own corporate‑action logic.
	•	Options: Finnhub chains for exploration; OPRA NBBO via Databento or Cboe/Polygon for research‑grade IV/Greeks; compute IV from NBBO mids; fit daily SVI/SSVI with no‑arb checks.
	•	Corporate actions & listings: Alpha Vantage LISTING_STATUS, splits/divs; reconcile with SEC EDGAR.
	•	Delistings & index membership: persist as reference; ensure universes include delisted names.
	•	Macro: FRED + ALFRED vintages (revision‑aware labels/features).
	•	Tick size regime: maintain per‑date mapping (e.g., half‑penny program) and feed into spread/impact models.

Vendor changes must be drop‑in below curated schemas. All raw files carry ingested_at, source_uri, vendor_ts, api_version.

⸻

12) Security, PIT, and Calendars
	•	Persist public release timestamps for fundamentals/news; lag features to first public availability.
	•	Calendars: generate sessions (open/close, half‑days, DST) with unit tests; strict half‑day handling.
	•	Index reconstitutions: rebuild per date; never leak today’s membership backwards.
	•	Borrow realism: haircuts or caps if borrow unknown; ingest indicative borrow fees when available.

⸻

13) Config Templates (copy/paste)

13.1 training/equities_xs.yml

dependencies: [equities_eod, corp_actions, delistings]
green_threshold: { window_years: 15, min_ratio: 0.98 }
cpcv: { folds: 8, embargo_days: 10 }
time_decay: { half_life_months: 18 }
regime_masks:
  - { name: 'pre-2012',  use_for: stress_only }
  - { name: '2012-2016', use_for: fit_and_validate }
  - { name: '2016-2024', use_for: fit_and_validate }
  - { name: '2025+',     use_for: monitor_only }
promotion_bars: { DSR_min: 0.0, PBO_max: 0.05, net_sharpe_min: 1.0 }

13.2 training/options_vol.yml

dependencies: [options_chains]
min_surface_qc_rate: 0.95
green_threshold: { window_years: 10, min_ratio: 0.95 }
cpcv: { folds: 8, embargo_days: 10 }
time_decay: { half_life_months: 18 }
regime_masks:
  - { name: '2012-2016', use_for: fit_and_validate }
  - { name: '2016-2024', use_for: fit_and_validate }
  - { name: '2025+',     use_for: monitor_only }

13.3 Router config stub

router:
  fallback: best_available
  features: [regime_flags, completeness_vector, liquidity_tier]
  stacker:
    enabled: true
    model: lightgbm
    folds: 8


⸻

14) Minimal APIs (must implement)

def audit_scan(tables: list[str]) -> None: ...

def backfill_run(budget: dict) -> None: ...

def curate_incremental() -> None: ...

# GREEN-gated dispatcher

def train_if_ready(model_name: str) -> None: ...

# Validation/Governance

def run_cpcv(model_name: str) -> dict: ...

def promote_if_beat_champion(model_name: str) -> None: ...


⸻

15) Monitoring & Dashboards
	•	Coverage heatmap (date × dataset) with GREEN/AMBER/RED ratios by train window.
	•	Data drift: PSI/KL on key features; alerting thresholds.
	•	Calibration: Brier/logloss for classifiers; reliability plots.
	•	Performance stability: rolling IR, DSR trend; PBO history.
	•	Execution realism: modeled vs realized impact; slippage distributions.

⸻

16) Rollout Order (do these next)
	1.	Create partition_status, backfill_queue; ship ata-audit and ata-backfill CLIs.
	2.	Remove legacy IEX free‑tier dependencies from default plans; pin Alpaca for dev minute/EOD; stage Databento/Polygon connectors.
	3.	Encode regime windows + time‑decay in training/*.yml; add per‑model GREEN thresholds.
	4.	Implement the router (availability + regime rules). Add CPCV‑consistent stacking once at least two signals are GREEN.

⸻

17) Appendix: Acceptance Tests (end‑to‑end)
	•	Data ops → model ops: delete a week of minute bars for 50 symbols → audit marks RED → backfill fills → curate rebuilds → GREEN ≥ threshold → train_if_ready(intraday_xs) unblocks.
	•	Regime mask sanity: flipping to include pre‑2012 in fitting should degrade OOS CPCV metrics (expected).
	•	Vendor swap: switch minute from Alpaca→Polygon under stable curated schema; OOS metrics remain comparable; coverage improves.
	•	Promotion gate: challenger beats champion net with better DSR/PBO → auto‑promotion and new shadow cycle starts.

⸻

Final Checklist (copy into your PR template)
	•	partition_status + backfill_queue migrations applied.
	•	backfill.yml committed with budgets, horizons, priorities.
	•	CLIs ata-audit, ata-backfill, train, evaluate, promote wired into DAG.
	•	Regime masks + time‑decay present in training configs.
	•	GREEN thresholds enforced; trainer exits non‑zero if unmet.
	•	Router routes strictly by completeness + regime; stacker respects CPCV.
	•	Coverage heatmap in dashboard; DSR/PBO reporting in evaluation outputs.
	•	Vendor change below curated layer introduces no research‑code diffs.