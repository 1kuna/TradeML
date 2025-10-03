# TradeML SSOT Progress

This document tracks implementation progress against Architecture_SSOT.md. It summarizes what’s shipped, what’s pending, and the near‑term plan to close the gaps.

## Summary
- Hands‑off Raspberry Pi node now runs the complete SSOT “nightly” loop continuously (forward ingest → audit → backfill → curate → audit refresh), with retries and no manual steps.
- GREEN‑gated training for `equities_xs` is available with a one‑click Windows launcher and a continuous loop.
- Core SSOT data structures (completeness ledger and backfill queue), configs, and minimal APIs are implemented.

## Completed (mapped to SSOT)

- 1) System Layout & Orchestration
  - Node loop integrates “nightly DAG” steps and runs continuously on the Pi.
    - scripts/node.py, scripts/pi_node.sh
  - One‑click Windows training loop.
    - scripts/windows/training_run.bat, scripts/training_loop.py

- 2) Data Completeness Model (GREEN/AMBER/RED)
  - Computation and ledger persisted to `data_layer/qc/partition_status.parquet` and mirrored to Postgres.
    - ops/ssot/audit.py
  - Calendar‑aware coverage checks using exchange calendars.
    - data_layer/reference/calendars.py

- 3) DB Schema
  - Added `partition_status` and `backfill_queue` tables with indexes and comments.
    - infra/init-db/01-init-schema.sql

- 4) Backfill Subsystem
  - Queue‑driven backfill for equities EOD (Alpaca) consuming `backfill_queue` by priority.
  - S3 lease to prevent concurrent workers; idempotent upsert of raw partitions.
  - Safety‑net backward window sweep when queue is empty.
    - Token‑bucket style daily API budgets (persisted manifest) honored when sweeping/queueing; window shrinks under pressure.
    - Extended to `equities_minute` (1Min) for top‑liquidity subset; exploratory `options_chains` (Finnhub) forward fill.
    - ops/ssot/backfill.py, ops/ssot/budget.py, data_layer/storage/lease_manager.py

- 6) Backfill Configuration
  - Canonical config: budgets, priorities, targets, QC thresholds, horizons.
    - configs/backfill.yml

- 7) Orchestration (nightly DAG → Pi node loop)
  - Forward ingest (edge) → audit → backfill → curate → audit(refresh) → sleep.
    - scripts/node.py, scripts/edge_collector.py, scripts/curator.py

- 8) Model Set (initial)
  - `equities_xs` pipeline present; GREEN‑gated training wrapper.
    - ops/ssot/train_gate.py, ops/pipelines/equities_xs.py

- 9) Training Gating
  - Enforced GREEN threshold from configs for equities_xs.
    - configs/training/equities_xs.yml

- 10) Validation
  - CPCV/PBO/DSR primitives implemented and used by equities pipeline.
    - validation/cpcv, validation/pbo, validation/dsr, ops/pipelines/equities_xs.py

- 11) Data Sourcing (partial)
  - Alpaca equities (bars) connector working.
  - Free‑tier delistings updater (Alpha Vantage LISTING_STATUS) integrated into node loop.
    - data_layer/connectors/alpaca_connector.py, ops/ssot/reference.py

- 12) Calendars & PIT (partial)
  - Exchange calendar helper and PIT metadata in connectors; curated loader supports date‑partitions.
    - data_layer/reference/calendars.py, data_layer/curated/loaders.py

- 13) Monitoring & Dashboards (seed)
  - Coverage heatmap (date × table) emitted nightly from `partition_status`.
    - ops/monitoring/coverage.py (writes `ops/reports/coverage_YYYY-MM-DD.png`)

- 14) Minimal APIs
  - Implemented: `audit_scan`, `backfill_run`, `curate_incremental`, `train_if_ready`, `run_cpcv` (stub wrapper), `promote_if_beat_champion` (stub).
    - ops/ssot/*, scripts/ata.py

- Rollout/Tooling extras
  - SSOT CLI for manual operations (optional): scripts/ata.py
  - GitHub issue filed for alerting integration (Discord/Slack/Email) via gh CLI (Issue #1).

## In Progress / Remaining

- Backfill enhancements
  - Fine‑tune budget heuristics (cross‑cycle persistence, multi‑worker fairness) and add sleeps/yields instead of hard skip when depleted.
  - Extend to: `options_nbbo` when enabled, and macro vintages (FRED/ALFRED) with cadence‑aware planners.
  - Add acceptance tests for dual workers/ETags and rate‑limit storms.

- Completeness/QC depth
  - Add QC thresholds and reason codes (rows_low/high, half‑days, checksums), AMBER rationales.
  - Persist QC artifacts and align audit to compute GREEN ratios over train window by dependency.

- Curation
  - Corporate‑action adjustments: implemented split‑day localized adjustment; extend to full back‑adjustment chain and late‑CA localized rebuilds.
  - Recompute derived artifacts (rolling features, options surfaces) only for impacted dates (incremental dags & hashes).

- Models & Router
  - Implement `intraday_xs` and `options_vol` pipelines (NBBO→IV→SVI with QC) and route by availability & regime masks.
  - Add CPCV‑consistent meta‑blender (stacking) when multiple signals exist; fallback policies.

- Training governance & registry
  - Retrain cadence + triggers (coverage delta/drift/champion underperformance).
  - MLflow registry integration; champion–challenger, multi‑week shadow trading.
  - Implement `promote_if_beat_champion` per SSOT bars.

- Monitoring & dashboards
  - Add drift (PSI/KL) and calibration (Brier/logloss) plots; DSR/PBO surfacing in reports.
  - Drift (PSI/KL) and calibration (Brier/logloss) plots; DSR/PBO surfacing in reports.
  - Execution realism dashboards (impact vs realized).

- Data sourcing & reference
  - Corporate actions ingestion (splits/dividends) PIT‑safe; SEC/EDGAR reconciliation.
  - Index membership, tick‑size regime; macro vintages.

## Next Steps (priority order)
1) Backfill: add graceful sleep/yield when budget depleted; add macro/FRED target and planner; acceptance tests for rate‑limit storms.
2) Curation: implement full back‑adjustment for splits/dividends across history with localized rebuilds triggered by new CA; add unit tests.
3) Options: extend `options_chains` planner (historical backfill in small windows), and sketch NBBO→IV path; scaffold `options_vol` pipeline.
4) Training governance: integrate MLflow registry (local server) and record champion/challenger metadata + shadow logs; implement `promote_if_beat_champion` policy.
5) Monitoring: add drift snapshot to reports and wire into node; add minimal dashboard JSON for coverage/drift.

## One‑Click Launchers
- Pi node: `bash scripts/run_node.sh`
- Windows training: `scripts\windows\training_run.bat`

## References
- Architecture: Architecture_SSOT.md
- Key implementations (paths):
  - Node: scripts/node.py, scripts/pi_node.sh
  - Audit: ops/ssot/audit.py
  - Backfill: ops/ssot/backfill.py
  - Curator: scripts/curator.py, ops/ssot/curate.py
  - Reference updater: ops/ssot/reference.py
  - Training gate: ops/ssot/train_gate.py
  - Configs: configs/backfill.yml, configs/training/
  - DB schema: infra/init-db/01-init-schema.sql
