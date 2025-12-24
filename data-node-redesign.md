# Data Node Redesign (Pi-first)

## Goals
- Make the Raspberry Pi a self-contained **data node**: continuous ingest, periodic audit/backfill/curate/QC, and nightly export of ready-to-train datasets.
- Replace single-vendor backfills with a **queue-based dispatcher** (FIFO by design, with scoring on top) so multiple vendors can fill gaps in parallel, honoring priorities, budgets, and entitlements.
- Keep training/evaluation on the training machine; the Pi only ships curated, QC’d artifacts.
- Collapse the current split between the node loop and nightly DAG into a **single, unified data-node service** whose detailed scheduling/structure can be redesigned by an LLM, subject to these constraints.

## Current State (as of Dec 2025)

This section is intentionally descriptive so a reader/LLM can see what exists today before proposing a replacement.
### Bootstrap & Node
- `bootstrap.sh` + `rpi_wizard.py` set up the venv, .env, data/log symlinks, and start the **node loop** (`scripts/node.py`).
- The node loop runs the **edge collector** (`scripts/edge_collector.py`), which:
  - Initializes connectors (alpaca, massive, fred, etc.).
  - Schedules ingest tasks (per-vendor schedulers) for equities bars (day/minute), options bars, macro, etc.
  - Writes raw parquet to `data_layer/raw/<vendor>/<table>/date=YYYY-MM-DD/symbol=SYMBOL/data.parquet`
    (options use `underlier=SYMBOL`; macro/series datasets remain date-only).
  - Writes manifests to `data_layer/manifests/<vendor>/<table>/manifest-*.jsonl`.
  - Updates bookmarks to resume from the last collected date.
- Ingest cadence is frequent (interval loop). Backfill logic is vendor-first and static (no cross-vendor queue).

### Nightly DAG (separate today)
- `ops/dags/nightly.py` orchestrates: ingest.forward → audit.scan → backfill.run → curate.incremental → qc.refresh → train/eval/promote/report.
- It is **not** run automatically by the Pi bootstrap; typically runs on a scheduler (could be elsewhere).
- Backfill is budgeted, but assignment is static (per-vendor loops), not a shared queue.
- QC ledger lives at `data_layer/qc/partition_status.parquet` (optionally mirrored to DB).

### Pain Points
- Nightly is detached from the Pi by default; QC/curation/backfill don’t run unless manually scheduled.
- Single-vendor backfill leaves other vendors idle and gaps unfilled when the primary vendor is saturated or missing data.
- Training is bundled in nightly flags; needs explicit disabling on the Pi.

## Scope and Non-Goals
**In scope**
- Redesign of the **Pi data node** as a unified service that:
  - Continuously ingests raw data from multiple vendors.
  - Detects and tracks gaps.
  - Fills gaps via a vendor-agnostic queue.
  - Curates and QC-checks data into a consistent, “ready for training” format.
  - Exports daily artifacts for a separate training box.

**Out of scope**
- Model architectures, training/evaluation pipelines, and promotion logic on the training machine.
- UX/reporting layers beyond basic logs/metrics/artifacts.
- Vendor-specific bug details (e.g., transient SDK issues) — those are implementation details, not architectural drivers.

The intention is that this redesign **replaces both**:
- The current edge/node ingest loop, and
- The current nightly DAG orchestration for ingest/audit/backfill/curate/QC,
with a single, Pi-resident data-node process (or tightly coupled set of processes) whose design the LLM is free to reshape within the constraints above.

## Data Model & Scale Assumptions

At a high level:
- **Equities bars (daily)**: `alpaca: equities_bars` — OHLCV per symbol per trading day.
- **Equities bars (minute)**: `alpaca: equities_bars_minute` — OHLCV per symbol per minute for regular sessions.
- **Options bars**: `alpaca: options_bars` — OHLCV per option contract per day/minute (entitlement-dependent).
- **Macro / Treasuries**: `fred: macro_treasury` — daily yields/curves for multiple tenors.
- Additional datasets may exist (e.g., fundamentals, corporate actions), but the core architecture should handle **(vendor, dataset, symbol, dt)** generically.

Scale assumptions (can be refined):
- Symbol universe: configurable (e.g., a few hundred to a few thousand symbols).
- History: up to 20+ years of daily data; potentially multiple years of minute data where feasible.
- Per-day volume: for minute bars, thousands of rows per symbol per day; for daily, 1 row per symbol per day.

These assumptions matter for:
- Pi resource sizing (RAM, disk, CPU).
- Feasibility of running QC/curation on-box.
- How aggressive backfills can be without overwhelming the node.

## Environment & Constraints

- **Pi hardware** (approximate assumptions):
  - ARM CPU, limited cores.
  - 4–8 GB RAM.
  - External SSD recommended (bootstrap/wizard encourages this).
  - Home/office network, subject to vendor API rate limits.
- **Training machine**:
  - More CPU/RAM; suitable for training and evaluation.
  - Accesses Pi exports over LAN or similar.

Constraints:
- Pi should **not** run heavy model training; it focuses on ingest + data health/organization/QC.
- Design must tolerate intermittent connectivity and reboots (idempotent, restart-safe).
- Vendor rate limits and entitlements must be respected (per-vendor budgets).

## Target Model (Pi as unified data node)
### Responsibilities on the Pi
- **Continuous ingest**: same edge collector behavior.
- **Nightly data management** (no training/eval):
  - Audit → gap detection.
  - Queue-driven backfill (multi-vendor).
  - Curate incremental.
  - QC refresh (update `qc/partition_status.parquet`).
- **Nightly export**: copy curated/QC artifacts to an `exports/` folder (e.g., `exports/nightly/YYYY-MM-DD/`) for LAN transfer to the training machine.

### Keep off the Pi
- Training, evaluation, promotion, and heavy model artifacts; those run on the training PC after pulling exports.

## Proposed Queue-Based Backfill/Audit Design

This is the **preferred direction** for how gaps should be filled, but the exact implementation details (e.g., how it’s embedded into a unified service) are left to the LLM to design.
### Core Concepts
- **Gap inventory**: audit produces tasks like `{dataset, symbol, dt, expected_rows, priority}`.
- **FIFO task queue (original idea)**:
  - Gaps are enqueued in roughly FIFO order as they are discovered (e.g., oldest missing days first, then newer).
  - Each task describes *what* needs to be filled (e.g., “equities options for AAPL on 2004-01-15”).
  - Vendors/workers **pull** tasks when free, rather than a central scheduler pushing everything to one vendor.
- **Vendor capabilities & priority**:
  - A registry defines which datasets each vendor can fill (equities, options, macro, etc.), plus:
    - Entitlements and known coverage.
    - Relative preference/quality (e.g., Alpaca preferred over Massive for equities, etc.).
    - Per-vendor budgets (max RPM, daily call budgets).
  - This captures “dibs” semantics: higher-priority vendor gets first claim, but others can pick up tasks when the preferred vendor is busy or not applicable.
- **Scoring/dispatch on top of FIFO**:
  - For each queued gap, eligible vendors are ranked by score (e.g., coverage × priority ÷ expected cost/latency).
  - The highest-score vendor that has available budget claims the work; if it is saturated or fails, the next vendor may claim it.
- **Per-vendor budgets/locks**:
  - Token buckets for RPM/requests-per-interval.
  - Leases/locks to avoid two workers doing the same task concurrently.
- **Outcome handling**:
  - Success: write raw + manifest + bookmark.
  - Empty/Weekend: mark benign and suppress requeues.
  - Not entitled/4xx: demote vendor for that dataset/symbol/date.
  - Rate limited: requeue with backoff.
  - Error: requeue with bounded retries, then mark stalled.
- **Idempotence**: task key `(dataset, symbol, dt, vendor)`; safe to retry.

### Data Structures
- `gap_tasks` queue (priority/FIFO) persisted locally (e.g., sqlite/jsonl).
- `vendor_capabilities.yaml/json`: datasets, feeds, priority, budgets.
- `leases` table: `(task_id, vendor, lease_expires_at)`.

### Workers
- **Ingest worker** (existing edge collector or its successor):
  - Continues to pull latest data for “today” / near-real-time ingest.
  - May also opportunistically dequeue high-priority “today” gaps from the queue.
- **Backfill worker(s)** (currently in nightly, future: unified with ingest):
  - Drain `gap_tasks` within per-run budget, honoring vendor budgets.
  - Can be scheduled (e.g., nightly) or run as a long-lived background worker as part of the unified data node.

### Export
- After nightly completes curate + QC:
  - Materialize curated partitions and `qc/partition_status.parquet` into `exports/nightly/YYYY-MM-DD/`.
  - Optional checksum/manifest for the export bundle.

## Transitional Plan (to be refined or replaced)

These steps describe a pragmatic path from “today” to the unified design. An LLM is free to propose a cleaner end-state that subsumes these if appropriate.

1. **Pi nightly config (short term)**:
   - Run `ops/dags/nightly.py` daily on the Pi with training/eval/promote disabled; ingest/audit/backfill/curate/QC enabled.
   - This reuses existing code paths while we validate Pi resource usage for audit/backfill/curate/QC.
2. **Exports**:
   - Add an export step to nightly to copy curated + QC outputs into `exports/nightly/<asof>/` for the training box.
3. **Queue MVP**:
   - Produce gaps from audit.
   - Persist gaps to a queue (sqlite/jsonl).
   - Implement a dispatcher that scores vendors and issues tasks on top of FIFO semantics.
   - Modify backfill to pull from the queue instead of static vendor-first loops.
4. **Capabilities registry**:
   - Define per-vendor dataset coverage, priority, and budgets in a single config file.
5. **Observability**:
   - Log queue depth, vendor throughput, success/empty/error rates; surface in logs/metrics.
6. **Hard disable training on Pi**:
   - Default nightly flags on Pi set training/eval/promote to false; training is explicitly a separate concern on the training machine.

## How “Done” Looks Nightly (Pi)
- Ingest: current-day deltas collected.
- Audit: gaps computed and queued.
- Backfill: dequeued tasks until budgets spent or queue empty.
- Curate: incremental rebuild for touched partitions.
- QC: `qc/partition_status.parquet` updated.
- Export: `exports/nightly/YYYY-MM-DD/` contains curated tables + QC ledger, ready for pull to training machine.

## Open Questions / Options
- Queue store choice: sqlite vs jsonl + in-memory. (SQLite recommended for durability/concurrency.)
- Dispatch granularity: symbol-day tasks vs day-only tasks for multi-symbol pulls (e.g., alpaca multi-bars). Likely per-symbol-day to maximize vendor parallelism, but allow batching by vendor when executing.
- Backfill depth/budget: per-run caps and lookback windows to prevent long Pi runs; tune via env.
- Export format: parquet + manifest + checksums; optionally tarball for easy transfer. 

## QC Scope (to define and expand)
Current: QC pipeline exists (`data_layer/qc/partition_status.parquet`), but coverage/content need to be defined for the Pi-first flow.

Proposed baseline checks (Pi):
- **Completeness**: expected row counts per (dataset, symbol, day) vs manifest/bookmark; flag SHORT_SESSION, MISSING_ROWS.
- **Continuity**: gaps in trading days per symbol; ensure weekends/holidays are treated as benign empties.
- **Schema/Types**: column presence and dtypes (e.g., open/high/low/close/volume/trade_count).
- **Adjustments sanity**: detect large discontinuities that suggest missing split/dividend adjustments; ensure adjustment mode is consistent (raw/split/dividend/all).
- **Cross-vendor spot checks (optional)**: small sample price/volume comparisons when multiple vendors cover the same symbol/day.
- **Checksum/hash**: parquet checksum or deterministic digest per partition for integrity.
- **Reject/Quarantine rules**: configurable thresholds to mark RED/AMBER and hold partitions out of exports.

Add-ons to consider:
- **Corporate actions alignment**: verify splits/dividends applied match CA feed for equities.
- **Session timing**: per-symbol minute-bar session length matches expected market hours (detect half-days).
- **Outlier detection**: simple z-score/robust thresholds on returns/volume to flag anomalies.

We should codify the initial QC checklist and thresholds, then iterate with metrics/alerts.

## Unified Data-Node Service (What the LLM Should Design)

The long-term goal is for the Pi to run a **single, coherent data-node service** that merges the current node loop and nightly DAG responsibilities. Within the constraints above, the LLM reviewing this document is expected to:

- Propose a concrete architecture for a unified service that:
  - Schedules/executes ingest, audit, backfill, curate, QC, and export at appropriate cadences.
  - Uses the FIFO + scoring queue for gap-filling across vendors.
  - Is restart-safe and idempotent on the Pi.
- Decide how to:
  - Organize the internal scheduler (e.g., cron-like, event-driven, or explicit loops).
  - Partition responsibilities between long-lived workers vs scheduled batch tasks.
  - Minimize complexity while preserving clarity and debuggability.
- Treat this design as a **replacement** for:
  - The current `scripts/node.py` loop, and
  - The ingest/audit/backfill/curate/QC portions of `ops/dags/nightly.py`
  (training/eval/promote remain off-Pi).

The FIFO queue concept and vendor-claim semantics described above are the preferred backbone; the LLM is free to refine the details (scoring, backoff, worker layout) to arrive at a single, cohesive data-node architecture.
