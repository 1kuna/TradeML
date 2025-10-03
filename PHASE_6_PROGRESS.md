# Phase 6 Progress — Autonomy & Free‑Only Enhancements

This document tracks what’s completed and what remains from the original “missing” list, while preserving the 1‑click Pi node and Windows trainer flows and using free‑tier data only.

## Completed In This Phase

- Backfill marks extended beyond EOD
  - `equities_minute` mark updated during sweep: `ops/ssot/backfill.py:307`
  - `options_chains` mark written when fetched: `ops/ssot/backfill.py:353`
- Curator now dividend‑aware with localized rebuilds
  - Loads splits + dividends, adds `div_cash`, reprocesses impacted dates on CA hash changes: `scripts/curator.py`
- Node autonomy expanded (free‑only)
  - Options IV build + SVI fit (best‑effort): `scripts/node.py:206`
  - Options strategies emit and next‑day delta‑hedged PnL evaluation: `scripts/node.py:239`, `ops/ssot/options_eval.py`
  - Reference approximations persisted daily: ADV‑based index membership and tick‑size regime: `ops/reference/index_membership.py`, `ops/reference/tick_size.py`
- Windows trainer remains 1‑click and optional options gate
  - Optional `options_vol` train gate run: `scripts/training_loop.py:51`
- Router/meta‑blender configurability
  - Reads YAML weights and supports config‑based linear weights when `stacker.enabled: true`: `ops/ssot/router.py`, `configs/router.yml`
- Calibration metrics (equities_xs)
  - Brier/log‑loss computed and included in metrics: `validation/calibration.py`, `ops/pipelines/equities_xs.py`
- Backtester borrow fees
  - Daily borrow cost on shorts via `borrow_bps`: `backtest/engine/backtester.py`
- Promotion policy: optional multi‑week shadow
  - Configurable `promotion_policy.require_shadow_days`, defaults to 20: `configs/training/equities_xs.yml`, `ops/ssot/train_gate.py`

- Parallel ingestion on RPi (workers)
  - Bounded thread pools for IO-heavy paths with conservative defaults: `utils/concurrency.py`
  - Edge collector now fans out across sources (Alpaca/Polygon/Finnhub/FRED) with a global scheduler; starts 1 unit per source and reallocates workers when a vendor cools down (rate limits) or runs out of work. Alpaca uses per-day fetch; bookmarks update only after persist: `scripts/edge_collector.py`
  - Backfill queue items processed in parallel under S3 lease; per-partition ETag retries prevent races: `ops/ssot/backfill.py`
  - Env knobs: `NODE_WORKERS` (global), `NODE_MAX_INFLIGHT_ALPACA` (per-source)

## Still Missing / Outstanding

- Options pipeline end‑to‑end (training → CPCV → promotion)
  - Current `options_vol` is a scaffold (fits SVI from curated IV); no CPCV training or promotion path yet.
- Duplicated IV/SVI paths
  - Curated builder (`ops/ssot/options.py`) and pipeline scaffold (`ops/pipelines/options_vol.py`) both exist; unify or make one authoritative.
- CPCV‑consistent meta‑blender (stacking)
  - Router supports manual linear weights; no learned stacker (GBDT/linear) trained via CPCV yet.
- Calibration reliability plots and MLflow logging
  - Numeric Brier/log‑loss is present; reliability curves/PNGs and registry logging not wired.
- Tick‑size regime and index membership from official sources
  - Free approximations shipped; vendor/official lists not integrated (by design in free‑only mode).
- Options backtesting
  - Diagnostic delta‑hedged PnL exists (`backtest/engine/options_pnl.py`), but not a full mark‑to‑market options backtester with daily hedging and Greeks.
  - HTB/borrow availability tiers not modeled.
- Orchestration via Prefect/Airflow DAGs
  - Pi node loop still the orchestrator; no DAG definitions yet.
- Acceptance tests
  - Present: budget shrink unit test, S3 lease integration test.
  - Missing: e2e tests for rate‑limit storms and dual‑worker/ETag races.

## Next Steps (Proposed)

- Learned meta‑blender
  - Train linear/GBDT stacker via CPCV on out‑of‑fold predictions; persist weights; make router load them when `stacker.enabled`.
- Reliability plots
  - Emit reliability bins to PNG; include in daily report and (optionally) log to MLflow.
- Options pipeline E2E
  - Unify IV/SVI path; add minimal options model with CPCV + diagnostics; record to MLflow; wire promotion gates.
- Options backtester (free‑friendly)
  - Daily mark‑to‑market with SVI surfaces; delta rebalancing costs; simple borrow rules; remain source‑agnostic.
- Orchestration (DAGs)
  - Define a Prefect or Airflow DAG for audit → backfill → curate → refresh → reports → training → promotion; keep Pi node as a thin runner for the DAG.
- Acceptance tests
  - Add e2e tests that simulate rate‑limit shrink and dual‑worker object contention with S3 ETags/leases.

## What “Orchestration via Prefect/Airflow DAGs” Means

- Orchestration: a scheduler coordinates tasks, their dependencies, retries, and state (success/failure) across machines.
- DAG (Directed Acyclic Graph): tasks as nodes; edges encode ordering/dependencies (e.g., audit → backfill → curate → audit refresh → reports → training → promotion).
- Prefect
  - Python‑native “flows” and “tasks”, easy local dev, retries/caching, UI/observability; can run fully locally (keeps 1‑click viable).
- Airflow
  - Heavier‑weight scheduler with rich operators and a strong UI; often used with a persistent metastore and workers.
- For this repo
  - We’d model SSOT steps as a DAG; schedules or event triggers kick runs; retries/backoff handle vendor rate limits; leases/ETags keep idempotency. A single command (or the Pi node) can launch the flow locally for 1‑click operation.

## Polygon — Future‑Critical Items (Keep Lean, High‑ROI)

- Centralized rate governance
  - Single Polygon budget across backfill/audit/reference with hard rpm smoothing, 429 Retry‑After honoring, and jitter. Avoid bursts; pause tasks when error rate > X%.
- Two‑year window guarantees
  - Enforce start_date clamps in all call sites (contracts, aggregates), with unit tests. Prevents wasted calls and vendor 4xx.
- Incremental paging with bookmarks
  - Persist `next_cursor` (tickers/contracts) in a manifest to fetch deltas only; de‑dup on key fields. Cuts daily request footprint materially.
- QC thresholds and fallback policy
  - Tune clear thresholds (e.g., |close diff| > 10–20 bps, minute row delta > 10) to trigger notes or fallback fetches; log counters to spot degradation early.
- Targeted options sampling
  - Sample only top‑ADV underliers and nearest expiries; cap N contracts/day; wire into IV diagnostics. Keeps usage impactful without burning budget.
- Observability & locks
  - Emit daily Polygon usage/error stats (requests, 429s, merges) to S3 JSON; gate high‑cost tasks with S3 leases to avoid multi‑worker spikes.
