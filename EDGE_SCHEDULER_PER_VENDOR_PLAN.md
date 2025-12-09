# Per‑Vendor Executors: Edge Collector Scheduler Redesign

This document specifies a redesign of the edge collector scheduler to run each vendor/source with its own executor, queues, and leases so slow vendors never block progress on faster ones. It is intended as a standalone handoff plan with implementation details, acceptance criteria, and rollout steps.

## Goals

- Decouple vendors so a slow vendor (e.g., Massive) cannot stall Alpaca/Finnhub/FRED work.
- Continuously feed work for each vendor up to its configured cap and budget, independent of other vendors’ completions.
- Preserve existing guarantees: S3 leases, idempotent writes, bookmarks, budgets, and graceful shutdown.
- Provide clear diagnostics (unit start/finish, durations, in‑flight, budgets, freezes).
- Minimize invasive changes to connectors and storage layers.

## Non‑Goals

- No change to connector APIs or data schemas.
- No change to LeaseManager semantics or S3Writer implementation beyond usage.
- No switch to async/await; remain thread‑based.
- No process isolation in this phase (kept as future option).

## Current Pain Points (Context)

- Single global executor with “seed then replace on completion” causes idle threads to wait when the remaining units are long‑running, vendor‑capped tasks.
- Slow Massive day units (~12–25s per request, 10 symbols per unit) often exceed the 60s scheduler guard, leading to resets.
- Fast vendors (FRED/Finnhub, Alpaca minute) finish early, then appear idle.

## High‑Level Design

- Supervisor coordinates independent VendorRunners.
- One VendorRunner per vendor (alpaca, massive, finnhub, fred). Each has:
  - Its own work generator (UnitProducer)
  - Its own bounded ThreadPoolExecutor sized by vendor cap
  - Its own S3 lease key and renew thread
  - Its own budget+freeze logic and in‑flight tracking
- All VendorRunners run concurrently. The supervisor aggregates status/logs, propagates shutdown, and waits for runners to idle/finish.
- Unit lifecycle per vendor:
  1) Generate next unit(s) while under cap and budget
  2) Submit to vendor executor
  3) On completion, record metrics, update bookmarks/manifests, schedule next
  4) If rate‑limited or budget‑exhausted, set freeze and keep runner alive (no submission until unfreezed)

## Components & Responsibilities

- VendorSupervisor (new):
  - Reads `configs/edge.yml` tasks list.
  - Creates one VendorRunner per vendor present in tasks.
  - Starts/stops runners, collates progress, handles SIGINT/SIGTERM.
  - Ensures S3Writer is started once and shared.

- VendorRunner (new):
  - Fields: `vendor_name`, `tasks` (subtasks for that vendor), `executor`, `lease_name`, `lease_mgr`, `budget_mgr`, `bookmarks`, `s3_writer`, `connectors`, `pacer`.
  - Loop:
    - Acquire lease (per vendor). Start renew thread.
    - While not shutdown:
      - Fill up executor slots: generate and submit units while `inflight < cap` and `budget.try_consume(tokens)` and not frozen.
      - Harvest completions with short wait (e.g., 1–2s); on timeout, loop back to attempt more submissions (keeps feeding without global completion coupling).
      - Periodically emit heartbeat (in‑flight, submitted, ok, ratelimited, errors, freezes left, budget left).
    - On shutdown: cancel futures, release lease, join renew thread.

- UnitProducer (per vendor):
  - Alpaca: `alpaca_bars` (1Day), `alpaca_minute` (1Min)
  - Massive: `massive_bars` (day; chunking by env)
  - Finnhub: `finnhub_options` (per underlier)
  - FRED: `fred_treasury` (per day)
  - For each subtask, respects bookmarks and EOD gating for “today”.

- Budget/Pacing Integration:
  - Before submission, call `budget.try_consume(vendor, tokens)`; if false, set runner freeze or reduce unit size.
  - `RequestPacer` still used inside connectors; runner freezes handle vendor‑level cooldowns (e.g., 60s after 429).

- Leases:
  - One lease per vendor runner: `locks/edge-<vendor>-<group>.lock` (e.g., `edge-alpaca-collector`, `edge-massive-collector`).
  - If a specific subtask requires uniqueness (e.g., options chains), keep current task‑level lease names and acquire them transiently around the unit if needed. Otherwise, per‑vendor lease is sufficient to prevent duplicate collection across devices.

## Work Generation & Granularity

- Alpaca Day Bars (`alpaca_bars`): per‑day unit with full universe, batch fetch (chunks of 100) → 1 S3 write per day.
- Alpaca Minute (`alpaca_minute`): per‑day unit with full universe, batch fetch (chunks of 100) → 1 S3 write per day. Start from configurable days back (e.g., 7/30/365) independent of day bars.
- Massive Day Bars (`massive_bars`): per‑day, chunked symbol lists to bound wall‑clock duration. Default `NODE_MASSIVE_SYMBOLS_PER_UNIT=3`, cap inflight=1.
- Finnhub Options (`finnhub_options`): per‑underlier unit; limit underliers per cycle via env; continue round‑robin across universe.
- FRED Treasury (`fred_treasury`): per‑day unit; fetch full curve in one unit.

## Concurrency & Caps

- Caps configured via env (with safe defaults):
  - `NODE_MAX_INFLIGHT_ALPACA` (default 2)
  - `NODE_MAX_INFLIGHT_MASSIVE` (default 1)
  - `NODE_MAX_INFLIGHT_FINNHUB` (default 2)
  - `NODE_MAX_INFLIGHT_FRED` (default 2)
- Each runner owns its executor sized to cap.
- Global worker limit from `NODE_WORKERS` no longer constrains per‑vendor threads; use it only as a soft ceiling if desired (optional future work).

## Freezes & Backoff

- On unit error containing `429` or explicit rate‑limit detection → set `freeze_until = now + 60s` (configurable per vendor by env).
- On budget depletion → freeze until next period or sleep (`BudgetManager` already persisted).
- Runner loop checks freeze before submitting new units; still harvests completions.

## Logging & Diagnostics

- Per unit submission: `vendor`, `desc`, `tokens`.
- Per unit completion: `status` (ok/empty/ratelimited/error), `rows`, `elapsed_ms`.
- Runner heartbeat every ~30s: `inflight`, `cap`, `submitted`, `completed_ok`, `errors`, `ratelimited`, `freeze_remaining_s`, `budget_remaining`.
- S3/HTTP “Begin …” logs retained; async parquet waits use periodic messages.

## Configuration Changes

- `configs/edge.yml`:
  - `tasks:` may include `alpaca_minute` in addition to existing tasks.
  - Optional `vendors:` block to override caps/freezes (or use env only).
- Env (new/used):
  - `NODE_MAX_INFLIGHT_ALPACA|MASSIVE|FINNHUB|FRED`
  - `NODE_MASSIVE_SYMBOLS_PER_UNIT` (default 3)
  - `ALPACA_MINUTE_START_DAYS` (default 7)
  - `VENDOR_FREEZE_SECONDS_<VENDOR>` (optional, default 60)

## Failure Handling

- S3 timeouts configured via boto3 `Config` (already in repo) prevent indefinite hangs.
- On connector exceptions, return `(error, 0, msg)`; runner increments error counters and continues.
- On repeated failures, consider exponential increase of freeze (optional enhancement).

## Graceful Shutdown

- Supervisor installs SIGINT/SIGTERM.
- On shutdown:
  - Signal runners to stop; stop feeding; wait up to N seconds for inflight futures; cancel remaining; release leases.
  - Stop S3Writer last.

## Migration & Rollout

- Feature flag: `EDGE_SCHEDULER_MODE=per_vendor|legacy` (default `legacy` for safety; plan to switch default after verification).
- Phase 1 (dev): Implement runners and supervisor; parity with current tasks; enable flag locally.
- Phase 2 (staging/PI): Enable per‑vendor; observe logs/metrics for a day; tune caps/chunk sizes.
- Phase 3 (default): Flip default to `per_vendor`; keep `legacy` path for rollback for one release.

## Implementation Steps (Code‑Level)

1) Create module `scripts/scheduler/per_vendor.py`:
   - `class VendorRunner`
     - ctor(args): vendor, subtasks, connectors, s3, lease_mgr, bookmarks, budget_mgr, s3_writer
     - methods: `start()`, `stop()`, `_renew_lease_loop()`, `_feed_loop()`, `_harvest_loop()`, `_submit(unit)`
   - `class VendorSupervisor`
     - ctor(edge_config, env)
     - `run(tasks)`: build runners by vendor; manage lifecycle; aggregate exit status

2) Extract per‑vendor UnitProducers from `scripts/edge_collector.py`:
   - `alpaca_bars_units()`, `alpaca_minute_units()`, `massive_bars_units()`, `finnhub_options_units()`, `fred_treasury_units()`
   - Each yields dicts: `{vendor, desc, tokens, run}`
   - Move to `scripts/scheduler/producers.py` or keep in `edge_collector.py` behind thin wrappers used by runners

3) Refactor `scripts/edge_collector.py`:
   - Add `--scheduler=legacy|per_vendor` arg or read `EDGE_SCHEDULER_MODE`
   - If `per_vendor`: instantiate `VendorSupervisor` and pass wiring (s3, lease_mgr, bookmarks, connectors, budget_mgr, s3_writer)
   - Retain legacy `_schedule_fanout` path for fallback

4) Leases:
   - Define per‑vendor lease names in `_lease_name_for_vendor(vendor)`:
     - alpaca: `edge-alpaca-collector`
     - massive: `edge-massive-collector`
     - finnhub: `edge-finnhub-collector`
     - fred: `edge-fred-collector`
   - Optional: for tasks requiring exclusivity (e.g., options chains), also acquire transient task‑level lease in unit runner, reuse current naming

5) Caps & Freezes:
   - Helper: `vendor_cap(vendor) -> int` from env with defaults
   - Helper: `vendor_freeze_seconds(vendor) -> int` from env (default 60)
   - Runner maintains `inflight` map; submission path checks `inflight < cap` and not frozen

6) Budget Integration:
   - Before submit, estimate `tokens` for unit and call `budget.try_consume(vendor, tokens)`
   - If false: set budget freeze (e.g., until next period reported by `BudgetManager` if available; otherwise fixed sleep)

7) Logging & Metrics:
   - Add structured logger fields (vendor, desc, unit_id) on submit/complete
   - Heartbeat every 30s per runner
   - Maintain counters: `submitted`, `completed_ok`, `errors`, `ratelimited`

8) Shutdown:
   - Supervisor `stop()` sets `shutdown_requested` on each runner; waits with timeout; cancels futures if necessary; joins renew threads; releases leases

9) Configuration & Docs:
   - Update `configs/edge.yml` to document `alpaca_minute` task and vendor caps
   - Document env knobs in `.env.template`
   - Add README snippet for enabling per‑vendor scheduler

## Testing Plan

- Unit: VendorRunner submission and harvest logic with fake unit functions; verify cap, freeze, and budget behavior.
- Unit: Lease acquire/renew/release lifecycle; simulate missing lock and re‑acquire.
- Integration (local):
  - Run with `STORAGE_BACKEND=local` and stub connectors returning delayed futures to simulate long Massive; ensure Alpaca continues submitting.
  - Run with S3/MinIO and verify leases, uploads, and bookmarks update independently per vendor.
- Failure Injection:
  - Simulate S3 timeouts (already guarded) and ensure unit reports error without deadlock.
  - Simulate 429 from connectors; verify per‑vendor freeze and continued work on others.

## Acceptance Criteria

- Slow Massive never prevents Alpaca minute/day, Finnhub, or FRED from continuing to submit and complete units.
- Per‑vendor caps and budgets are enforced; logs show independent runner progress and heartbeats.
- Graceful shutdown releases all leases and stops S3Writer cleanly.
- No regressions in data layout and manifests; bookmarks advance as before.

## Rollback Plan

- Keep legacy `_schedule_fanout` and config flag `EDGE_SCHEDULER_MODE=legacy`.
- Single commit revert of the supervisor/runners import + flag use restores legacy behavior.

## Future Improvements

- Massive grouped endpoint for daily aggregates to reduce per‑symbol latency.
- Optional process isolation: run Massive in a separate process/container using the same S3 leases.
- Prometheus/StatsD metrics export for runner counters and latencies.
- Per‑vendor dynamic chunk sizing based on observed unit durations and timeout budget.

## Estimation

- Implementation: 1–2 days
- Validation on Pi/minio: 0.5–1 day
- Observability tuning and docs: 0.5 day

---

Handoff Checklist:
- Implement per‑vendor supervisor and runners per above steps.
- Wire existing producers; verify unit outputs and bookmarks.
- Add env defaults and docs.
- Test locally (with sleep‑simulated delays) and against MinIO.
- Enable in staging with `EDGE_SCHEDULER_MODE=per_vendor` and caps tuned.
