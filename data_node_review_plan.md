# Data Node Review: Findings, Recommendations, and Change Plan

Assumptions
- Scope includes `data_node/` plus `data_layer/connectors/` and related utilities that run on the Pi.
- Vendor/API limits are treated as “verified in-code” unless we explicitly re-check with Context7 or the public internet before implementation changes.

## Findings

### Critical
1) Budget limiter can block forever and double-counts spend; per-request gating always uses `FORWARD`, bypassing BOOTSTRAP/GAP/QC slices. This can stall workers indefinitely and exhaust daily budgets early.
   - `data_layer/connectors/base.py:100`, `data_layer/connectors/base.py:156`, `data_node/budgets.py:286`
2) Raw storage partitions are date-only; tasks are per-symbol. Each symbol’s fetch can overwrite another symbol’s file for the same date, and QC file checks are symbol-blind.
   - `data_node/fetchers.py:425`, `data_node/fetchers.py:688`, `data_node/maintenance.py:21`

### High
3) WorkerPool threads are non-daemon and `stop()` clears references even if threads are still alive. Shutdown can hang if any worker is stuck in fetch or IO.
   - `data_node/worker.py:1320`, `data_node/worker.py:1333`
4) VendorWorker treats NOT_ENTITLED/NOT_SUPPORTED as transient errors; tasks can spin forever on vendors that can never serve them.
   - `data_node/worker.py:1238`
5) Export loads entire `partition_status` into pandas; this can OOM/freeze the Pi on large histories.
   - `data_node/maintenance.py:399`

### Medium
6) Budget exhaustion immediately releases tasks with no backoff, creating hot lease/release loops and DB churn.
   - `data_node/worker.py:883`
7) Gap audit pulls full symbol×date ranges into memory and does nested loops; this is O(n_symbols×n_days) and can stall audits.
   - `data_node/planner.py:125`
8) QC_PROBE tasks are scheduled but never executed by the worker. Cross-vendor QC is a no-op.
   - `data_node/qc_weekly.py:340`, `data_node/maintenance.py:561`
9) Freshness defer applies to all datasets; it can delay non-market datasets and older ranges unnecessarily.
   - `data_node/worker.py:866`
10) Structural QC row-count validation does not apply the “last 7 days” window and `results["checked"]` is never incremented; metrics are misleading.
   - `data_node/maintenance.py:169`, `data_node/maintenance.py:201`

### Low
11) QC sampling weights are inverted; the <30-day boost is unreachable, so recency is underweighted.
   - `data_node/qc_weekly.py:124`
12) Production uses WorkerPool/VendorWorker, but tests mostly exercise QueueWorker; the production path isn’t covered.
   - `tests/integration/test_worker_e2e.py:8`
13) Connector sessions are created per task without explicit close; over long uptime this can accumulate sockets/FDs depending on GC.
   - `data_node/fetchers.py:476`, `data_layer/connectors/base.py:39`
14) Manifest checksum reads full files into memory; minor overhead for large exports.
   - `data_node/maintenance.py:458`

## Recommendations (Priority Order)
1) Separate RPM gating from daily spend. Remove double-spend. Pass real TaskKind to daily cap checks. Avoid indefinite blocking by deferring tasks when daily caps are hit.
2) Fix raw storage layout to partition by `date` + `symbol` (and `underlier` where applicable), and update QC file path resolution.
3) Unify QueueWorker/VendorWorker result handling, including NOT_ENTITLED/NOT_SUPPORTED backoff and vendor ineligibility cooldowns.
4) Make shutdown deterministic with graceful stop deadline, and avoid clearing thread refs until threads finish; add a hard-stop path if needed.
5) Make maintenance and audits resource-safe (chunk exports, windowed QC checks, incremental gap audits).
6) Implement QC_PROBE execution end-to-end with persistent results and automated GAP requeue on mismatches.
7) Fix QC sampling weights and add tests for WorkerPool/QC_PROBE paths.

## Full Change Plan

1) Budget/rate-limit refactor
   - Add a BudgetManager RPM-only token function (e.g., `acquire_rpm_token(vendor)` or `try_acquire_rpm(vendor)`) that does not touch daily spend.
   - Change `BaseConnector._rate_limit()` to use RPM token acquisition only. If tokens aren’t available, sleep based on RPM without touching daily caps.
   - Record daily spend in one place after each request, with the correct TaskKind. Eliminate the current double spend (try_spend + spend).
   - Ensure daily cap enforcement uses actual TaskKind slices (BOOTSTRAP/GAP/QC/FORWARD).
   - Add logging to distinguish “RPM throttled” vs “daily cap reached.”

2) Budget-aware task deferral
   - When daily caps are hit, defer tasks using `next_not_before` (e.g., 15–60 min) instead of immediate release.
   - Avoid hot lease/release loops under daily cap exhaustion.

3) Raw storage layout fix
   - Change `_get_raw_path` and all fetcher writes to partition by `date` and `symbol` (and `underlier` for options). Example: `.../date=YYYY-MM-DD/symbol=XYZ/data.parquet`.
   - Ensure DataFrames include the partition columns so `write_parquet(partition_cols=["date", "symbol"])` works correctly.
   - Update `_resolve_raw_partition_path` to include symbol and underlier.
   - Update any curation readers that assume date-only partitions.

4) Data migration/backfill plan
   - Decide whether to migrate existing date-only raw data or re-fetch.
   - If migrating, write a one-time script to move/merge `date=.../data.parquet` into `date=.../symbol=.../data.parquet` by reading and re-partitioning.
   - If re-fetching, mark the date-only partitions as invalid (RED) to force GAP tasks.

5) Worker behavior alignment
   - Factor QueueWorker/VendorWorker result handling into a shared helper.
   - Implement NOT_ENTITLED/NOT_SUPPORTED vendor cooldowns and switch to alternate vendors where possible.
   - For QC_PROBE tasks, route to a dedicated handler (see step 9).

6) Shutdown correctness
   - Add a graceful shutdown deadline (e.g., 30–60s). After deadline, force exit or mark threads as daemon on next run.
   - Only clear worker/thread lists after confirmed joins.
   - Add logging for threads still alive at shutdown.

7) Maintenance scaling
   - Stream `partition_status` export in chunks (fetchmany + incremental parquet writer) to avoid loading entire table in memory.
   - Apply rolling windows (e.g., last 7–30 days) to structural QC validation and record accurate “checked” counts.
   - Compute checksums in streaming chunks rather than full-file reads.

8) Planner scaling
   - Replace full history audit with incremental window audits on the Pi.
   - Cache trading day lists and avoid nested loops over entire history.
   - Optionally move full history audit to a larger machine or schedule monthly with smaller windows.

9) QC_PROBE end-to-end
   - When worker receives QC_PROBE task: fetch secondary vendor data, load primary data, compare OHLCV, and store results in a new `qc_probes` table (or `partition_status` notes + `qc_results` table).
   - On mismatch, mark partition AMBER and enqueue GAP task.
   - Log every probe outcome for traceability.

10) QC sampling fixes
   - Fix recency weighting so <30 days gets higher weight than <90 days.
   - Add tests for weighted sampling distributions if needed.

11) Tests
   - Add integration tests for WorkerPool/VendorWorker behaviors, budget gating, and QC_PROBE.
   - Add tests for raw storage partition layout and QC file existence checks.

12) Docs and verification
   - Update docs/specs if partition layout or QC behavior changes.
   - Add/extend operational notes for Pi memory constraints and export/audit schedules.

## Dependencies / External Checks
- If budget limits, rate limits, or vendor capabilities need updating, verify via Context7 or current vendor docs before code changes.
