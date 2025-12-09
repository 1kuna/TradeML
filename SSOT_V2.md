# TradeML — Unified Single Source of Truth (SSOT) v2

**Purpose.** This document is the canonical, implementation‑ready specification for the TradeML system. It unifies and supersedes:

- Architecture & model carve‑out plans
- Backfill / GREEN‑gating / regime policies
- Data‑sourcing playbook & free→prosumer vendor plan
- Edge scheduler redesign & multi‑provider pipeline upgrade
- Blueprint, progress, quick‑ref, and status docs

All future code, configs, and design decisions MUST conform to this SSOT unless explicitly updated in a new version of this document.

---

## 0. Precedence, Scope, and Invariants

### 0.1 What this SSOT governs

- Data layer (connectors, schemas, PIT discipline, backfill, QC)
- Feature and label definitions for all models
- Model stack (equities_xs, options_vol, intraday_xs, meta‑stacker, offline‑RL layer)
- Routing & portfolio rules (GREEN‑gated sleeves, meta‑blending, risk)
- Orchestration (node loop, nightly DAG, edge scheduler)
- Validation & promotion (CPCV, PBO, DSR, champion↔challenger)
- Minimal APIs, CLIs, and file layout contracts

### 0.2 Canonical precedence

In case of conflict between docs or code comments, precedence is:

1. **This SSOT v2**
2. Architecture_SSOT (backfill, regimes, GREEN‑gating) and Architecture Carve‑Out (model stack)
3. Data_Sourcing_Playbook + Data_Sources_Detail (connectors and free‑tier limits)
4. EDGE Scheduler & Multi‑Provider Pipeline docs
5. TradeML Blueprint (long‑form rationale, not implementation detail)
6. STATUS / PROGRESS / QUICK_REFERENCE (current code, examples)

### 0.3 Non‑negotiable invariants

- Two‑sweep data ops (forward ingest + continuous backfill) never stop.
- Models only train on windows where dependencies are GREEN.
- Raw payloads are immutable and PIT‑correct; all adjustments are downstream and reproducible.
- No new production modeling code is added under `/legacy/`; that tree is frozen.
- Every model is governed by CPCV, PBO, DSR, and a dark holdout.
- Champion promotion is automated and rule‑driven, not manual.

---

## 1. System Layout & Repo Contracts

### 1.1 Repo tree

```text
/repo
  /infra                  # Docker, DB, MinIO, MLflow, Redis
  /data_layer
    connectors/           # Alpaca, Massive (Polygon.io), Finnhub, AV, FMP, FRED...
    raw/                  # immutable vendor payloads
    curated/              # adjusted, PIT‑safe tables
    reference/            # corp_actions, delistings, calendars, universe...
    manifests/            # bookmarks, backfill_marks
    qc/                   # partition_status + QC artifacts
  /feature_store
    equities/
    options/
    intraday/
  /labeling
    horizon/
    triple_barrier/
  /validation
    cpcv/
    pbo.py
    dsr.py
    calibration.py
  /models
    equities_xs/
    intraday_xs/
    options_vol/
    meta/                 # stacker
  /portfolio
    build.py
  /execution
    cost_models/
    simulators/
    brokers/
  /backtest
    engine/
  /ops
    ssot/
      train_gate.py
      router.py
      audit.py
      backfill.py
      reference.py
    pipelines/
      equities_xs.py
      intraday_xs.py
      options_vol.py
      stacker_train.py
      offline_rl.py
    monitoring/
      drift.py
      coverage.py
      tripwires.py
    reports/
      emitter.py
  /configs
    backfill.yml
    curator.yml
    endpoints.yml
    edge.yml
    training/
      equities_xs.yml
      intraday_xs.yml
      options_vol.yml
      stacker.yml
      rl.yml
    router.yml
  /scripts
    node.py
    edge_collector.py
    scheduler/
      per_vendor.py
      producers.py
    curator.py
    ata.py
    build_universe.py
    fetch_equities_data.py
    test_phase2_pipeline.py
    test_cpcv_fix.py
    train_equities.py
    score_daily.py
    export_onnx.py
    paper_trade_alpaca.py
    windows/
      training_run.bat
  /legacy/
    ...   # frozen old modeling stack
```

**Rule:** the `/legacy` tree is read‑only for new work. New code under `/models`, `/feature_store`, `/ops/pipelines`, `/portfolio`, `/backtest`, `/validation`, and `/ops/ssot` defines the current architecture.

### 1.2 Services & storage

- **Storage (current)**: Local-first on external SSD, Parquet + zstd, partitioned by date/symbol or logical keys. All ingest artifacts, manifests/bookmarks, QC ledgers, and logs live on the SSD so it can be moved to the workstation for training/eval without extra services.
- **Future addition**: Optional remote/object-store replication (MinIO/S3) for multi-node or cloud sync. Not required for the baseline flow.
- Metadata/control: SQLite/DuckDB on the SSD for manifests/bookmarks/partition_status; Postgres remains optional for richer ops metrics when available.
- Orchestration: Pi node loop + optional Prefect/Airflow.
- Registry: MLflow for experiments, champion/challenger registry, and artifacts (reachable from workstation; optional on Pi).

### 1.3 Time & calendar conventions

- All timestamps stored in raw and curated tables MUST be normalized to **UTC** at ingest.
- Daily "date" fields (e.g., the `date` column in `raw/equities_bars`) represent the **trading day in the primary listing exchange's local time** (US/Eastern for US‑listed equities).
- Intraday features and backtests operate on UTC timestamps but rely on `reference/calendars` to interpret sessions, half‑days, and holidays. No ad‑hoc daylight‑saving logic is allowed in feature/label code; only the calendar is authoritative.
- Vendor payloads that arrive in local time MUST be normalized to UTC and, where relevant, retain the original timezone or exchange identifier in metadata.
- All cross‑asset joins (macro, fundamentals, options, equities) MUST be done via explicit time‑lag logic and calendars, not by assuming same‑day availability.

### 1.4 Runtime configuration & environment

Runtime config is layered as:

1. Base YAML configs under `/configs` (checked into git).
2. Optional environment‑specific overrides (e.g., `/configs/env/dev/*.yml`).
3. Environment variables (the highest precedence) set via `.env` or the process environment.

Key environment variables include:

- **Core infra**
  - `TRADEML_ENV` — one of `local`, `dev`, or `prod`.
  - `DB_URI` — Postgres connection string.
  - `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY` — object store config.
  - `MLFLOW_TRACKING_URI` — MLflow tracking server.
  - `REDIS_URL` — Redis instance for locks/queues if used.
- **Node identity & scheduler**
  - `EDGE_NODE_ID` — logical node name used for leases and logging.
  - `EDGE_SCHEDULER_MODE` — MUST default to `per_vendor` in production.
  - `NODE_MAX_INFLIGHT_DEFAULT` and `NODE_MAX_INFLIGHT_<VENDOR>` — concurrency caps per node and vendor.
  - `VENDOR_FREEZE_SECONDS_<VENDOR>` — backoff durations applied when 429s or vendor errors occur.
- **Backfill & safety**
  - `BACKFILL_MAX_DAYS_PER_TASK` — upper bound on days per backfill task.
  - `BACKFILL_MIN_GREEN_FRAC` — optional global override for minimum GREEN fraction.
  - `TRAINING_ENABLE_AUTO_PROMOTE` — gate for whether `promote_if_beat_champion` can auto‑promote; MUST be `false` in `local` unless explicitly overridden.

Environment profiles:

- **Pi / edge node** — conservative budgets and concurrency (`NODE_MAX_INFLIGHT_DEFAULT` low; vendor budgets small); full node loop enabled.
- **Workstation** — higher concurrency and budgets, used for heavy backfills and research runs.
- **CI** — training disabled, backfill disabled; only unit/integration tests and static analysis run.

---

## 2. Data Layer SSOT

### 2.1 Datasets & canonical schemas

Under `/data_layer` we standardize these core tables:

**Raw:**

- `raw/equities_bars` — daily/minute bars from Alpaca / Finnhub / Massive
  - Columns: `date`, `symbol`, `open`, `high`, `low`, `close`, `vwap`, `volume`, `nbbo_spread`, `trades`, `session_id`, metadata (`ingested_at`, `source_name`, `source_uri`, `vendor_ts`, `api_version`).
- `raw/equities_ticks` — tick TAQ (future; from Databento/Massive when upgraded)
  - Columns: `ts_ns`, `symbol`, `price`, `size`, `side`, `venue`, `seq`, metadata.
- `raw/options_nbbo` — OPRA or vendor equivalent
  - Columns: `ts_ns`, `underlier`, `expiry`, `strike`, `cp_flag`, `bid`, `ask`, `bid_size`, `ask_size`, `nbbo_mid`, `exchs_bitmap`, metadata.
- `raw/macros_fred` — macro time series + vintages
  - Columns: `series_id`, `observation_date`, `value`, `vintage_date`, metadata.
- `raw/fundamentals_*` — fundamental statements from FMP/AV (income, balance, cashflow).

**Reference:**

- `reference/corp_actions` — `symbol`, `event_type`, `ex_date`, `record_date`, `pay_date`, `ratio`, `source_priority`.
- `reference/delistings` — `symbol`, `delist_date`, `reason`, `source_priority`.
- `reference/calendars` — exchange sessions, early closes, holidays.
- `reference/index_membership` — date‑stamped index constituents.
- `reference/tick_size_regime` — tick size regime by symbol×date.
- `reference/universe` — PIT universe with eligibility flags.

**Curated:**

- `curated/equities_ohlcv_adj` — our own CA‑adjusted price series used by research.
- `curated/options_iv` — per‑contract IV & Greeks from NBBO mids.
- `curated/options_svi_surface` — per‑underlier, per‑date SVI/SSVI parameters and QC metrics.

All raw tables are **append‑only**; curated tables are recomputed deterministically and can be rebuilt from raw + reference.

### 2.2 Vendors & connector policy

We operate a **free→prosumer** vendor stack:

- Equities minute & EOD: Alpaca now; Databento/Massive later.
- Options chains & IV: Finnhub now (exploratory); OPRA via Databento/Massive/Cboe later.
- Corporate actions & listings: Alpha Vantage (LISTING_STATUS, splits/divs), FMP (delistings), SEC/EDGAR for authoritative dates.
- Macro & rates: FRED/ALFRED + Treasury data.

All connectors:

- Implement retry with exponential backoff and jitter.
- Respect per‑vendor budgets defined in `configs/backfill.yml` and `configs/edge.yml`.
- Write raw payloads to staging → checksum → atomic move into `raw/`.
- Attach metadata (`ingested_at`, `source_uri`, `vendor_ts`, `api_version`).

### 2.3 Completeness & GREEN/AMBER/RED

Completeness is tracked per `(source, table_name, symbol, dt)` and persisted to `qc/partition_status.parquet` and Postgres.

- **GREEN:** partition exists and passes QC (row counts in bounds, calendars aligned, checksums ok).
- **AMBER:** partition exists but QC is marginal (short session, vendor outage, row counts outside preferred but within tolerable bounds).
- **RED:** partition missing.

QC thresholds and expectations (rows per day, half‑days allowed, etc.) are configured in `configs/backfill.yml` and table‑specific QC modules.

### 2.4 Backfill subsystem

On the Pi, all ingest work flows through a **unified ingest queue** (`backfill_queue`) that handles forward ingest, gap‑filling, bootstrap, and QC probes in a single priority‑ordered system.

**Task kinds** (`kind` field):
- `BOOTSTRAP` — initial historical data fetch for new symbols or expanded windows
- `GAP` — fill RED/AMBER partitions detected by audit
- `FORWARD` — collect today's deltas (live/continuous ingest)
- `QC_PROBE` — weekly cross‑vendor spot‑checks

**Priority order** (lower = more important):
- `0` — `BOOTSTRAP` inside current stage window
- `1` — `GAP` inside current stage window
- `2` — `QC_PROBE` (weekly cross‑vendor checks)
- `3` — `BOOTSTRAP` outside stage window (ancient history)
- `5` — `FORWARD` (live ingest)

Tasks are processed FIFO within each priority lane, ordered by `(priority, created_at)`.

**Core control flow:**

1. `audit_scan()` compares expected vs actual partitions, updates `partition_status`, and upserts `GAP` tasks into `backfill_queue` for RED/AMBER partitions within the current stage window.
2. Queue worker leases tasks in priority order, enforcing per‑vendor budgets (token buckets + daily caps with 85/90/100% slices by kind). Completed tasks update raw tables and `partition_status`.
3. `curate_incremental()` detects changed raw partitions and rebuilds only affected curated partitions plus dependent derived artifacts.
4. `qc.refresh()` recomputes GREEN/AMBER/RED and coverage metrics.

**Budget slices by kind:**
- 0–85% of daily cap: `BOOTSTRAP` + `GAP` can spend freely
- 85–90%: only `QC_PROBE` + `FORWARD`
- 90–100%: only `FORWARD` (guaranteed allocation for live ingest)

**Idempotency rules:**

- Partitions are written to `.tmp` paths, verified, then moved into place.
- Leases with TTL prevent parallel workers from duplicating work; expired leases revert to PENDING.
- Task deduplication via `UNIQUE(dataset, symbol, start_date, end_date, kind)` constraint.
- Backfill windows use closed intervals `[t0, t1)` to keep replays deterministic.

### 2.5 Per‑vendor edge scheduler

The edge scheduler uses **per‑vendor executors** so slow vendors cannot block others.

- `scripts/scheduler/per_vendor.py` defines `VendorSupervisor` and `VendorRunner`.
- Each vendor has its own executor, lease, budget manager, and freeze logic.
- Work units are produced by `scripts/scheduler/producers.py`, with functions like:
  - `alpaca_bars_units()`
  - `alpaca_minute_units()`
  - `massive_bars_units()`
  - `finnhub_options_units()`
  - `finnhub_daily_units()`
  - `fred_treasury_units()`
  - `av_corp_actions_units()`
  - `av_options_hist_units()`
  - `fmp_fundamentals_units()`

Each unit is a dict: `{vendor, desc, tokens, run}` where `tokens` feeds vendor budgets.

Caps and freeze behavior are configured by env (`NODE_MAX_INFLIGHT_*`, `VENDOR_FREEZE_SECONDS_*`) and `edge.yml`.

### 2.6 Capability‑aware dataset routing

We keep connector variety but present **one stable curated schema** to research.

- `configs/endpoints.yml` defines, per vendor, which datasets they support, rough rpm, and whether coverage is unique.
- `ops/ssot/router.py` exposes `route(dataset, asof, universe)` returning an ordered list of providers to use.
- Producers consult `route()` before emitting work units, preferring unique or high‑limit providers.
- Vendor changes happen **below** the curated layer; schemas in `curated/` and `reference/` remain stable.

### 2.7 PIT discipline & universe construction

PIT rules:

- Every feature and label uses only data that was publicly available at that time.
- Macro and fundamentals must use vintages (ALFRED) or release timestamps; features are lagged accordingly.
- Index membership, corp_actions, and delistings are recorded with effective dates and applied to universes and prices without peeking.

Universe construction:

- Begin from all US common stocks above a liquidity floor (e.g., 60‑day ADV threshold), including delisted.
- Exclude non‑common share classes unless explicitly needed (ETFs, preferreds, etc.).
- Persist `reference/universe.csv` with PIT eligibility flags.

### 2.8 Metadata & control tables

Several Postgres tables act as control planes for data quality and orchestration. Their schemas are part of the SSOT.

**partition_status**

Tracks completeness and QC for each logical partition.

- `id` (UUID, primary key)
- `source_name` (text) — vendor or pipeline (e.g., `alpaca`, `finnhub`, `curator`)
- `table_name` (text) — e.g., `raw/equities_bars`, `curated/equities_ohlcv_adj`
- `symbol` (text, nullable) — NULL for non‑symbol tables (macro, fundamentals)
- `dt` (date, nullable) — primary partition date when applicable
- `partition_key` (text, nullable) — alternative key for non‑date partitions (e.g., `series_id=vix`)
- `status` (enum: `GREEN`, `AMBER`, `RED`)
- `qc_score` (double precision) — optional [0, 1] QC score
- `row_count` (integer)
- `expected_rows` (integer, nullable)
- `qc_code` (text, nullable) — short code describing the QC outcome (e.g., `OK`, `SHORT_SESSION`, `MISSING_ROWS`, `CHECKSUM_MISMATCH`)
- `first_observed_at` (timestamptz)
- `last_observed_at` (timestamptz)

**backfill_queue**

Represents tasks in the unified ingest queue. On the Pi, this is stored in SQLite (`data_layer/control/node.sqlite`); optionally mirrored to Postgres for workstation visibility.

- `id` (INTEGER, primary key, autoincrement)
- `dataset` (text) — logical dataset name, e.g., `equities_eod`, `equities_minute`, `options_chains`, `macros_fred`
- `symbol` (text, nullable)
- `start_date` (date)
- `end_date` (date)
- `kind` (text) — task type: `BOOTSTRAP`, `GAP`, `FORWARD`, or `QC_PROBE`
- `priority` (integer) — smaller numbers processed first (see §2.4 for priority semantics)
- `status` (enum: `PENDING`, `LEASED`, `DONE`, `FAILED`)
- `attempts` (integer, default 0)
- `lease_owner` (text, nullable) — node ID holding the current lease
- `lease_expires_at` (timestamptz, nullable) — when the lease expires; expired leases revert to PENDING
- `next_not_before` (timestamptz, nullable) — used for exponential backoff after failures
- `last_error` (text, nullable)
- `created_at` (timestamptz)
- `updated_at` (timestamptz)

**Constraints:**
- `UNIQUE(dataset, symbol, start_date, end_date, kind)` — prevents duplicate tasks for the same work unit

**Index:**
- `idx_backfill_queue_status ON (status, priority, created_at)` — efficient lease_next_task queries

**pipeline_runs** (optional but recommended)

High‑level record of pipeline executions (ingest, curate, train, evaluate, promote).

- `id` (UUID, primary key)
- `pipeline_name` (text) — e.g., `equities_xs_train`, `options_vol_backfill`
- `params_json` (jsonb) — config snapshot
- `status` (enum: `RUNNING`, `FAILED`, `SUCCEEDED`, `CANCELLED`)
- `started_at` (timestamptz)
- `finished_at` (timestamptz, nullable)
- `metrics_json` (jsonb, nullable)
- `artifact_uri` (text, nullable) — link to MLflow or object‑store artifacts

MLflow remains the canonical registry for model artifacts; `pipeline_runs` is optimized for operational visibility and querying.

### 2.9 Vendor & endpoint appendix

The exact vendor limits and endpoint URLs are maintained in `configs/endpoints.yml` and vendor‑specific configs, but the SSOT defines the **shape** of this registry and typical defaults.

Each entry in the capabilities registry has the form:

- `dataset` — logical dataset name (`equities_bars`, `options_nbbo`, `macros_fred`, `corp_actions`, `fundamentals`)
- `vendor` — `alpaca`, `massive`, `finnhub`, `fred`, `alfred`, `alpha_vantage`, `fmp`, etc.
- `endpoint` — path or function name for the vendor API
- `hard_rpm` — hard requests‑per‑minute limit used for budgeting
- `soft_daily_cap` — approximate daily request ceiling
- `burst_limit` — short‑term burst allowance, if known
- `priority` — integer used by `route(dataset, ...)` to pick preferred providers
- `notes` — free‑text caveats (e.g., "AV listing status resets daily at 00:00 UTC", "FRED series limited to 120 requests/min")

Indicative examples (actual numbers live in configs):

- `equities_bars`:
  - `alpaca` → `/v2/stocks/bars`, `hard_rpm` ~ 200, primary free source for US equities EOD/minute.
  - `finnhub` → `/stock/candle`, backup EOD/minute with smaller caps.
- `options_chain` / `options_nbbo`:
  - `finnhub` → options chain endpoints, free exploratory data.
  - `databento` / `massive` → SIP/OPRA feed when upgraded, with higher limits and better quality.
- `corp_actions`:
  - `alpha_vantage` → `LISTING_STATUS`, splits/dividends functions.
  - `fmp` → delistings and company metadata.
- `macros_fred` / `macros_alfred`:
  - `fred` → `/series/observations`.
  - `alfred` → `/series/observations` with vintage dimension.

The capabilities registry is the single source of truth for:

- How many calls a given dataset may make to a given vendor.
- Which vendors are considered primary vs backup for a dataset.
- Which endpoints must be updated when vendor documentation changes.

In case of conflict between the registry and ad‑hoc code, the registry and this SSOT win.

---

## 3. Features & Labels SSOT

### 3.1 Equities_xs — features

Goal: daily cross‑sectional features per `(date, symbol)`.

Minimum feature set (equities_xs `v1`):

- Price‑based
  - Multi‑horizon momentum: 5, 20, 60, 126‑day log returns.
  - Gap statistics (overnight, open‑to‑close) and recent drawdowns.
- Volatility
  - 20 and 60‑day realized volatility.
  - Rolling downside volatility.
- Liquidity
  - 20‑day ADV.
  - Turnover, Amihud illiquidity.
- Seasonality & calendar
  - Day‑of‑week and month‑of‑year encoded as sin/cos.
  - Distance to earnings.
- Risk / cross‑sectional context
  - Size proxies (market cap).
  - Sector/industry dummies.

Implementation:

- `feature_store/equities/features.py` exposes:
  - `build_features(panel, cfg) -> DataFrame` returning standardized `feat_*` columns plus `symbol`, `asof`.
- All features are computed PIT‑safe via lagging and rolling windows.

### 3.2 Options_vol — features

Per underlier‑date, using options IV surfaces:

- Surface shape
  - ATM IV level; skew; curvature.
  - Term‑structure slope (short vs long tenor IV).
- Relative level
  - IV rank / percentile vs trailing distribution.
- Structure/context
  - Time to expiry; moneyness buckets.
  - Distance to earnings and macro events.
- Activity
  - Volume, open interest, changes, where available.

Implementation:

- `feature_store/options/iv.py` computes IV from NBBO mids with robust filters.
- `feature_store/options/svi.py` fits SVI/SSVI surfaces and emits per‑expiry parameters plus `no_arb_flags` and `fit_rmse`.

### 3.3 Intraday_xs — features (optional sleeve)

On a liquid subset with minute bars & TOB/LOB summaries:

- VWAP dislocation vs last trade.
- Order‑flow imbalance and signed volume.
- Short‑horizon realized volatility and microstructure noise proxies.

Implementation:

- `feature_store/intraday/features.py` produces per‑minute/aggregated daily features.

### 3.4 Labels

All labels live under `/labeling` and are configurable via YAML.

- `labeling/horizon/build.py`:
  - Forward k‑day return labels, e.g., 5‑day or 20‑day.
- `labeling/triple_barrier/build.py`:
  - Triple‑barrier labels with profit‑take and stop‑loss scaled by volatility and a max holding horizon.

Options‑specific labels:

- Delta‑hedged PnL over a horizon (vol sleeve).
- Probability that a candidate spread or structure (vertical, calendar, etc.) achieves positive PnL.

Labels must:

- Be PIT‑safe (no peeking at future volatility or events).
- Avoid overlap with test folds in CPCV (handled by purging and embargo).

---

## 4. Model Stack & Routing SSOT

### 4.1 Specialist models

We maintain separate expert models and a meta‑blender:

- `equities_xs`: daily cross‑sectional stock selection.
- `intraday_xs`: intraday sleeve on minute features (optional).
- `options_vol`: options IV surface / vol‑edge sleeve.
- `meta/stacker`: stacking regressor/classifier over sleeve outputs.
- `offline_rl`: optional RL policies for sizing & execution schedule.

### 4.2 Equities_xs

- Algorithm: LightGBM or CatBoost regression on forward returns, with support for switching to classification if needed.
- Training window: 10–15 years of modern data, as defined in training config.
- Weighting: time‑decay with ~18‑month half‑life and optional regime masks (pre‑2012 mostly stress only; 2012–2016 and 2016+ fit/validate).
- Validation: CPCV with 8 folds, 10‑day embargo, OOS windows ≥6 months.
- Artifacts:
  - `model.pkl` (native), `model.onnx` (runtime inference), `feature_list.json`, `training_cfg.json`, OOF predictions.

### 4.3 Options_vol

- Algorithm: GBDT (LightGBM/XGBoost) or similar, on surface features described above.
- Training window: 7–10 years of options data.
- Dependencies: GREEN options chains/nbbo, options IV surfaces with acceptable QC rate.
- Targets: delta‑hedged PnL or edge probabilities.

### 4.4 Intraday_xs (optional)

- Algorithm: PatchTST or Mamba, targeting daily scores from minute‑frequency sequences.
- Context length: 256–512 observations; patch size 16–32.
- Export: TorchScript/ONNX for inference.

### 4.5 Meta‑blender (stacker)

- Trains on OOF predictions from available sleeves.
- Enforces CPCV consistency: stacker sees only OOS fold predictions.
- Implementation: `models/meta/stacker.py` with:
  - `train_stacker(oof_df, y, cfg)`
  - `load_stacker(path)`
  - `stack_scores(df_scores, model_or_weights)`

### 4.6 Router

`ops/ssot/router.py` is the single routing authority at prediction time.

Inputs:

- Regime flags (from training config).
- Data completeness (GREEN/AMBER/RED per required table & window).
- Liquidity tier.

Behavior:

1. Determine which sleeves are eligible for a given `(asof, symbol)` based on GREEN thresholds and dependencies.
2. Compute each eligible sleeve’s signal.
3. If multiple signals exist, pass them to the stacker for blending.
4. If only one sleeve is available, fall back to its raw signal.

Router configuration lives in `configs/router.yml`, which names the active stacker artifact, sleeve weights, and fallbacks.

### 4.7 Portfolio & execution policies

Portfolio construction (`portfolio/build.py`):

- Convert aggregated scores to expected returns / risk‑scaled z‑scores.
- Apply per‑name caps (e.g., 5%) and gross exposure cap (e.g., 1.0–1.5).
- Target annualized volatility (e.g., 10–12%) via volatility targeting.
- Use fractional Kelly (0.2–0.5×) based on conservative edge estimates.
- Enforce turnover constraints and shorting realism (borrow fees or short caps where borrow unknown).

Execution:

- Cost model: fee + effective spread + square‑root law impact.
- Simulator: Almgren–Chriss for schedule optimization and cost forecasts.
- Brokers: Alpaca client for paper/live, with TWAP participation caps.

### 4.8 Offline RL (optional layer)

Scope: RL modulates sizes/schedules; signals remain supervised.

- Env: `MarketEnv` wraps backtester: state includes factors, realized vol, drawdown, cost stats; actions are weight/participation buckets.
- Algorithms: CQL or IQL, with optional Decision Transformer as challenger.
- OPE: doubly‑robust estimators compare RL policies to baseline.
- Integration: RL outputs multiplicative adjustments to portfolio weights or schedule choices; must not break risk caps.

### 4.9 Execution & backtest data contracts

Backtest and execution components must agree on core data shapes.

**Bar schema (backtest engine)**

Minimum fields for equity bars:

- `dt` (date or timestamptz UTC) — bar end time
- `symbol` (text)
- `open`, `high`, `low`, `close` (double precision)
- `vwap` (double precision, nullable)
- `volume` (bigint)
- `nbbo_spread` (double precision, nullable)
- `turnover` (double precision, nullable)
- `session_id` (text, nullable) — matches `reference/calendars` session

**Tick schema (optional)**

- `ts` (timestamptz UTC)
- `symbol` (text)
- `price` (double precision)
- `size` (bigint)
- `side` (enum: `BUY`, `SELL`, `UNKNOWN`)
- `venue` (text, nullable)
- `seq` (bigint, nullable)

**Order schema**

Orders passed into simulators and broker adapters MUST include:

- `order_id` (UUID or unique text)
- `ts` (timestamptz UTC) — time the order was generated
- `symbol` (text)
- `side` (enum: `BUY`, `SELL`)
- `qty` (double precision or bigint, depending on instrument)
- `order_type` (enum: `MKT`, `LMT`)
- `limit_price` (double precision, nullable for market orders)
- `tif` (enum: `DAY`, `IOC`, `FOK`, `GTC`)
- `strategy_tag` (text) — e.g., `equities_xs`, `options_vol`

**Fill schema**

- `fill_id` (UUID or unique text)
- `order_id` (foreign key into orders)
- `ts` (timestamptz UTC)
- `symbol` (text)
- `side` (enum: `BUY`, `SELL`)
- `qty` (double precision or bigint)
- `price` (double precision)
- `fee` (double precision, nullable)
- `liquidity_flag` (enum: `MAKER`, `TAKER`, `UNKNOWN`)

Simulators under `execution/simulators/` and broker adapters under `execution/brokers/` must both conform to these schemas so that backtests, paper trading, and live trading can share the same portfolio and order‑generation logic.

**Daily report JSON / MD contract**

`ops/reports/emitter.py` produces a JSON + Markdown pair for each trading day. The JSON MUST include at least:

- `asof` (date)
- `pnl` (double precision) — daily and cumulative
- `gross_exposure`, `net_exposure` (double precision)
- `turnover` (double precision)
- `positions` — array of `{symbol, weight, side, strategy_tag}`
- `strategies` — per‑strategy metrics (Sharpe, turnover, contribution)
- `risk_metrics` — e.g., realized vol, max drawdown to date, VaR/ES estimates
- `data_health` — GREEN/AMBER/RED coverage summaries for key datasets

The Markdown report renders a human‑friendly view over the same JSON, and must not introduce ad‑hoc metrics not present in the JSON payload.

---

## 5. Orchestration & Node Loop SSOT

### 5.1 Unified Pi Data‑Node Service

The Pi runs a **unified data‑node service** (`python -m data_node`) that handles all data collection, audit, backfill, curation, and QC. Training, evaluation, and promotion remain on the workstation.

**Four concurrent loops:**

1. **ForwardIngestLoop** (every 15 min):
   - Check bookmarks/manifests to see if today's data is incomplete
   - Enqueue `FORWARD` tasks into `backfill_queue` (priority=5)

2. **PlannerLoop**:
   - **Light run (every 4h):** `audit_scan()` on active datasets; upsert `GAP` tasks for RED/AMBER partitions
   - **Heavy run (02:00 local):** full audit, stage gating (Stage 0→1 promotion), purge old FAILED tasks

3. **QueueWorkerLoop** (continuous):
   - Lease next task by `(priority, created_at)` respecting vendor budgets
   - Call dataset fetcher → write raw parquet → update partition_status
   - Mark task DONE or FAILED with backoff

4. **MaintenanceLoop** (02:00 local):
   - `curate_incremental()` for all raw partitions touched since last maintenance
   - Structural QC (`qc.refresh`) for those partitions
   - Weekly: enqueue `QC_PROBE` tasks for cross‑vendor spot‑checks
   - Export curated tables + QC ledger to `exports/nightly/YYYY-MM-DD/`

### 5.2 Workstation Nightly DAG

Training and model governance run on the workstation, consuming Pi exports:

1. Pull `exports/nightly/YYYY-MM-DD/` from Pi (or shared storage)
2. `train_if_ready('equities_xs')` — only if GREEN thresholds are satisfied
3. `train_if_ready('options_vol')` and `train_if_ready('intraday_xs')` (where enabled)
4. `evaluate.cpcv_and_shadow()` — recompute CPCV metrics and update challenger stats
5. `promote_if_beat_champion()` — apply promotion rules
6. `report.emit_daily()` — render MD + JSON blotters and coverage/drift plots

### 5.3 Launchers

- **Pi**: `python -m data_node` or `scripts/pi_data_node_wizard.py` for guided setup
- **Workstation**: `ops/dags/nightly.py` with training/eval/promote enabled; `scripts/windows/training_run.bat` for one‑click runs

All production automation should be expressed via these APIs/CLIs, not bespoke scripts.

---

## 6. Validation, Governance, and Promotion SSOT

### 6.1 GREEN‑gated training

Every model declares in `configs/training/<model>.yml`:

- `dependencies`: list of tables it requires.
- `green_threshold`: window length and minimum GREEN ratio.
- Optional QC requirements (e.g., minimum surface QC rate for options).

`train_if_ready(name)` checks these conditions via `partition_status` and refuses to train if unmet.

Canonical default thresholds for v2 (can be overridden explicitly in configs, but these are the baseline assumptions):

- **equities_xs**
  - Lookback window: 15 years of `curated/equities_ohlcv_adj` and associated reference data.
  - Minimum GREEN fraction: 0.98 on `curated/equities_ohlcv_adj` over the window.
  - Minimum GREEN fraction: 0.99 on `reference/corp_actions` and `reference/delistings` over the same window.
- **options_vol**
  - Lookback window: 7 years of options data.
  - Minimum GREEN fraction: 0.95 on `raw/options_nbbo` over the window.
  - Minimum acceptable surface QC rate (e.g., `no_arb_flags` OK and `fit_rmse` within tolerance) on at least 0.90 of sessions.
- **intraday_xs** (if enabled)
  - Lookback window: 3 years of minute bars for the intraday universe.
  - Minimum GREEN fraction: 0.97 on intraday minute bar tables.

When in doubt, configs SHOULD default to these thresholds rather than invent new ones ad‑hoc.

### 6.2 CPCV, PBO, DSR

Validation stack:

- CPCV with symbol‑aware purging and embargo (default 8 folds, 10‑day embargo).
- DSR calculation for Sharpe significance under non‑Gaussian returns.
- PBO estimating probability of backtest overfitting, accounting for multiplicity.

Results are logged for every candidate model and used in promotion decisions.

### 6.3 Go/no‑go bars

Equities_xs sleeve must satisfy, on OOS, net of costs:

- Sharpe ≥ 1.0
- Max drawdown ≤ 20%
- DSR > 0
- PBO ≤ 5%

Options_vol sleeve must satisfy equivalent bars on both delta‑hedged PnL and total PnL for its strategies, with Vega/Gamma caps respected.

### 6.4 Champion–Challenger & shadow trading

- MLflow registry holds champion and challengers for each model family.
- New training runs log metrics, artifacts, and metadata (including experiment count) to MLflow.
- `evaluate.cpcv_and_shadow()` computes CPCV metrics and runs challengers in shadow backtests.
- `promote_if_beat_champion()` promotes a challenger only if:
  - GREEN thresholds met.
  - CPCV metrics beat champion under same cost model.
  - PBO/DSR and risk metrics are acceptable.
  - Shadow trading over N weeks shows consistent improvement.

### 6.5 Monitoring & tripwires

Monitoring includes:

- Coverage heatmaps (date × dataset GREEN/AMBER/RED ratios).
- Feature drift metrics (PSI, KL) versus training distribution.
- Forecast calibration plots (Brier/log‑loss, reliability curves).
- Performance stability (rolling IR, DSR trend, PnL distribution).
- Execution realism (modeled vs realized impact; slippage distributions).

Tripwires:

- Soft de‑risk bands when drawdown exceeds thresholds.
- Hard kill when drawdown, slippage, or data issues breach critical limits.

---

## 7. Minimal APIs & CLIs (Contracts)

These functions are considered public, stable contracts:

```python
# ops/ssot/audit.py
def audit_scan(tables: list[str]) -> None: ...

# ops/ssot/backfill.py
def backfill_run(budget: dict | None = None) -> None: ...

# ops/ssot/curate.py
def curate_incremental() -> None: ...

# ops/ssot/train_gate.py
def train_if_ready(model_name: str) -> None: ...

def run_cpcv(model_name: str) -> dict: ...

def promote_if_beat_champion(model_name: str) -> None: ...

# feature_store/equities/features.py
def compute_equity_features(date, universe) -> "pd.DataFrame": ...

# labeling/triple_barrier/build.py
def triple_barrier_labels(...) -> "pd.DataFrame": ...

# portfolio/build.py
def build(scores_df, risk_cfg) -> dict: ...

# execution/simulators
def simulate(orders, cost_model, market_data) -> tuple: ...

# feature_store/options/svi.py
def fit_svi_surface(date, underlier) -> dict: ...

# ops/reports/emitter.py
def emit_daily(asof, positions, strategies, metrics) -> tuple[str, str]: ...
```

CLIs (via `scripts/ata.py` and other scripts):

- `ata-audit gaps --tables ...`
- `ata-backfill run --budget ...`
- `ata-train equities_xs` (and other models)
- `ata-evaluate model_name`
- `ata-promote model_name`
- `python scripts/test_cpcv_fix.py`
- `python scripts/test_phase2_pipeline.py`

Any new automation or notebooks should use these APIs/CLIs.

### 7.2 Testing expectations and invariants

All new pipelines, connectors, and model variants MUST ship with tests that enforce key invariants. At minimum:

- **Connectors & ingest**
  - Unit tests for each vendor connector that validate: pagination, rate‑limit handling, normalization and UTC conversion, and mapping from vendor payload → raw schema.
  - Round‑trip tests on small synthetic payloads to ensure idempotent writes (no duplicate partitions when rerun).
- **Backfill & audit**
  - Tests for `audit_scan` and `backfill_run` that simulate missing partitions, ensure RED → backfill tasks → GREEN transitions, and confirm idempotency under repeated runs.
  - Tests that verify exponential backoff and `next_not_before` handling in `backfill_queue`.
- **Features & labels**
  - Tests that confirm PIT safety: shifting input data forward or backward in time must not change labels/features outside the allowed window.
  - Schema tests that assert all configured `feat_*` and label columns exist and have non‑null coverage above thresholds on sample data.
- **Training, CPCV, and promotion**
  - Regression tests similar to `test_cpcv_fix.py` that validate multi‑symbol CPCV purging and embargo behavior.
  - Tests for `train_if_ready` and `promote_if_beat_champion` that simulate GREEN/RED regimes and verify that models are only trained/promoted when thresholds and go/no‑go bars are satisfied.
- **Routing, portfolio, and execution**
  - Tests for `ops/ssot/router.py` verifying sleeve enable/disable logic based on GREEN/AMBER/RED and regime masks.
  - Portfolio tests that check position caps, gross/net exposure limits, and turnover constraints.
  - Simulator tests that validate `simulate()` respects the order and fill schemas and produces consistent slippage metrics.

CI SHOULD run these tests on every change to SSOT‑governed modules. A change that breaks SSOT assumptions (e.g., new dependency, altered schema) MUST include an explicit SSOT version bump and a corresponding update to this document.

---

## 8. Operational Status & Gaps (v2 Baseline)

### 8.1 Implemented per SSOT

- Node loop with forward ingest → audit → backfill → curate → audit refresh.
- partition_status and backfill_queue schemas and logic.
- Backfill for equities EOD & minute (Alpaca), partial options and macro.
- GREEN‑gated training for equities_xs.
- CPCV, PBO, DSR primitives, and fixed multi‑symbol CPCV purging.
- Universe expansion to 152 symbols with systematic selection.
- IV, Greeks, and SVI surface infrastructure; coverage and drift monitoring.

### 8.2 Remaining to fully match SSOT v2

- Complete multi‑provider ingest (Finnhub daily, AV historical options, FMP fundamentals) and fully wire them through edge scheduler and backfill.
- Expand backfill & QC depth for all new tables (macro vintages, options_nbbo, fundamentals).
- Full corporate‑action adjustment chain and localized rebuilds for late events.
- End‑to‑end intraday_xs and options_vol pipelines with router + stacker integration.
- MLflow‑backed champion–challenger automation and multi‑week shadow trading.
- Optional offline RL pipeline integrated as a post‑alpha layer.
- Daily automated scoring & reports that use the **champion** models only.

---

## 9. Acceptance Criteria for "SSOT‑Complete"

TradeML is considered fully aligned with this SSOT v2 when:

1. **Data coverage**
   - Last 2–5 years: ≥98% GREEN coverage for equities EOD and minute across the trading universe.
   - Options chains for chosen underliers: ≥95% GREEN coverage.
   - Macro/fundamental/futures series needed by models have stable GREEN coverage.

2. **Model performance & governance**
   - Equities_xs champion meets all go/no‑go bars on a realistic universe.
   - Options_vol and (if enabled) intraday_xs meet equivalent bars in their domains.
   - Champion–challenger and shadow trading flows operate automatically with clear promotion history and rollbacks.

3. **Routing & portfolio**
   - Router/stacker can switch between champion configs via config changes only.
   - Portfolio and execution models produce consistent, capacity‑aware, cost‑realistic trades.

4. **Operations**
   - Node loop runs indefinitely (Pi + workstation) without manual babysitting.
   - Daily blotters, coverage heatmaps, and drift dashboards are produced reliably.
   - Tripwires and monitoring provide safe‑by‑default behavior.

Once these conditions are validated, this SSOT v2 becomes the reference for any future refactors, vendor upgrades, and model extensions. Any changes to these guarantees must be made via a new SSOT version, reviewed alongside config changes and code PRs.

---

## 10. Wizardized Operations (Edge + Workstation)

We standardize on interactive CLI wizards as the primary UX for all operational flows. Each wizard must be single-command start (`python <wizard>.py`), fully guided, resume-safe, and loud on unexpected failures.

**Edge (Pi) data‑node wizard** (`scripts/pi_data_node_wizard.py`)

Responsibilities:
- Detect environment (Pi vs Mac) and locate external SSD
- Prompt for storage root (default: `/mnt/.../data_layer`)
- Collect API keys (Alpaca, Finnhub, Alpha Vantage, FRED, FMP, Massive)
- Generate/patch `.env` with `EDGE_NODE_ID`, `TRADEML_ENV=local`, conservative `NODE_MAX_INFLIGHT_*` defaults
- Initialize SQLite control DB (`data_layer/control/node.sqlite`) with `backfill_queue` + `partition_status` tables
- Initialize stage config (`data_layer/control/stage.yml`) with Stage 0
- Seed `BOOTSTRAP` tasks for Stage 0 universe (100 symbols × 5y EOD + 1y minute)
- Configure maintenance time (default: 02:00 local)
- Optionally start `python -m data_node` with Rich dashboard

Storage:
- SQLite `node.sqlite` on SSD for queue and partition_status mirror
- Parquet + zstd for raw/curated data
- All control data (queue, manifests, partition_status, logs) co‑located on SSD for portability

Stage‑gated expansion:
- **Stage 0** (initial): 100 symbols, 5y EOD, 1y minute
- **Stage 1** (target): 500 symbols, 15y EOD, 5y minute
- Auto‑promote from Stage 0→1 when GREEN coverage ≥0.98 on equities EOD/minute

Resume: wizard reads SSD state (`stage.yml`, queue, manifests/bookmarks) and continues from last offsets automatically.

UX: prints actionable next steps; fails loudly on schema/gating issues (rate‑limit backoff handled silently by the queue worker).

**Workstation wizards (train/eval/inference/paper trade)**
- Single entrypoints for training, evaluation, shadow/paper trading, and inference serving; reuse the same patterns: venv/bootstrap, env/config selection, artifact discovery (champion/challenger), GREEN gating checks, CPCV/DSR/PBO runs, and promotion decisions.
- Data access: must work offline from the SSD copy (Parquet + zstd) or Pi exports. Future addition: optional remote/object-store sync for multi-node/cloud use.
- Output: consistent log capture to file from start of run; expose links for MLflow UI, reports, or any web dashboards started by the wizard.

**Wizard invariants**
- One-command launch, zero hidden prerequisites; all required services are started or instructions are printed.
- Timezone-aware logging (UTC) from process start; logs are persisted even if not tailed.
- No silent degradation: schema/version mismatches, missing GREEN dependencies, or promotion gates trigger loud, blocking errors with remediation hints.
- State stored with the data so plugging the SSD into another machine preserves progress and resumes deterministically.
