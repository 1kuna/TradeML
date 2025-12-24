Here’s the “final” spec you can hand to the coding agent.

I’ll first lock in the decisions (including budgets), then give a numbered implementation plan with concrete tasks and file locations.

---

## 0. Locked‑in decisions

1. **Single physical queue** (`backfill_queue`), used for *all* ingest work on the Pi:
   - `kind ∈ {BOOTSTRAP, GAP, FORWARD, QC_PROBE}`.
   - FIFO **within** each priority lane: tasks ordered by `(priority, created_at)`.

2. **Priority order (lower = more important)**:
   - `0` – `BOOTSTRAP` inside current stage window (Stage 0 / Stage 1).
   - `1` – `GAP` inside current stage window.
   - `2` – `QC_PROBE` (weekly cross‑vendor spot‑checks).
   - `3` – `BOOTSTRAP` outside stage window (ancient history).
   - `5` – `FORWARD` (live / continuous ingest).

3. **Gated universe/history**:
   - **Stage 0** (initial): 100 symbols, 5y EOD, 1y minute.
   - **Stage 1** (target): 500 symbols, 15y EOD, 5y minute.
   - Auto‑promote from Stage 0→1 once GREEN coverage ≥ 0.98 on equities EOD/minute over Stage 0 window (100‑symbol universe).

4. **Holidays/weekends**:
   - For non‑trading days, write `partition_status` rows as:
     - `status='GREEN'`, `expected_rows=0`, `row_count=0`, `qc_code='NO_SESSION'`.  
   - No gap tasks for those dates, ever.

5. **Cross‑vendor QC**:
   - Runs **weekly**, not nightly.
   - Implemented as `QC_PROBE` tasks in the same queue.
   - Priority below `BOOTSTRAP`/`GAP`, above `FORWARD`.
   - Budget‑capped (see §1.2) so it never owns the vendor entirely.

6. **Maintenance window**:
   - Heavy maintenance (full audit, stage gating, curate/QC, export) at **02:00 local** on the Pi.

7. **SSOT strategy**:
   - **Update SSOT_V2 *now*** so it matches this Pi + queue design and becomes the canonical spec.  
   - Data‑node‑redesign doc becomes “implementation zoom‑in” that defers to SSOT_V2 on conflicts.

8. **Budgets YAML**:
   - Keep existing `configs/backfill.yml` / `configs/endpoints.yml` as the source of truth for daily budgets and RPM caps; extend (don’t overwrite) where vendors are missing.

9. **Pi responsibilities**:
   - Pi runs **only** ingest/audit/backfill/curate/QC/export; training/eval/promotion stay on the workstation per carve‑out.

10. **Live UI**:
    - Implement with `rich` (not `curses`) for a single‑window dashboard of queue + workers.

---

## 1. Budgets (workers + queue)

### 1.1 Per‑vendor rate caps (logical defaults)

Agent must **read existing values** from `configs/backfill.yml` and `configs/endpoints.yml` first; only add or adjust where they are missing or obviously inconsistent. Use these as defaults when adding missing entries, derived from the free‑tier table in the endpoints/limits doc.

Proposed defaults (soft caps):

- **Alpaca**
  - `hard_rpm = 150` (≈75% of ~200/min guideline).
  - `soft_daily_cap = 10_000` requests/day.
- **Alpha Vantage (AV)**
  - `hard_rpm = 4` (below 5/min).  
  - `soft_daily_cap = 400` (below 500/day).
- **FRED**
  - `hard_rpm = 80` (below 120/min).  
  - `soft_daily_cap = 5_000`.
- **Finnhub**
  - `hard_rpm = 50` (below 60/min overall).  
  - `soft_daily_cap = 10_000` (well under combined minute caps).
- **FMP**
  - `hard_rpm = 3` (gives breathing room vs 250/day).  
  - `soft_daily_cap = 200` (below 250/day).
- **Massive (Polygon.io)**
  - `hard_rpm = 4` (below 5/min).  
  - `soft_daily_cap = 300`.

Where `backfill.yml` already defines numbers (e.g. Alpaca/FRED/AV), keep them; only add Finnhub/FMP/Massive entries based on the values above.

### 1.2 Per‑kind daily budget fractions

For each vendor `v` with daily cap `B_v`:

- Allow **backfill** (`BOOTSTRAP + GAP`) tasks while `spent_total_v < 0.85 * B_v`.
- Allow **QC_PROBE** tasks while `spent_total_v < 0.90 * B_v`.
- Allow **FORWARD** tasks while `spent_total_v < 1.00 * B_v`.

This implements:

- 0–85%: `BOOTSTRAP+GAP` can spend freely.
- 85–90%: only `QC_PROBE` + `FORWARD`.
- 90–100%: only `FORWARD`.

Result: gap/backfill are dominant; cross‑vendor QC happens before live; live is **guaranteed** the last 10% of calls so it can’t be starved.

### 1.3 Concurrency (workers)

Use the existing `NODE_MAX_INFLIGHT_*` env vars described in SSOT_V2.

On the Pi:

- `NODE_MAX_INFLIGHT_DEFAULT = 3`
- Vendor‑specific caps:
  - `ALPACA = 2`, `FINNHUB = 2`
  - `FRED = 1`, `ALPHAVANTAGE = 1`, `FMP = 1`, `MASSIVE = 1`

Agent should wire these into the new queue worker so at most `NODE_MAX_INFLIGHT_VENDOR` tasks per vendor are leased at once, and no more than `NODE_MAX_INFLIGHT_DEFAULT` overall.

---

## 2. Queue + control schema (Pi, SQLite)

On the SSD, create `data_layer/control/node.sqlite` with:

### 2.1 `backfill_queue` (generic ingest queue)

Physical table name **remains** `backfill_queue` for SSOT compatibility, but now covers all ingest work.

```sql
CREATE TABLE IF NOT EXISTS backfill_queue (
  id               INTEGER PRIMARY KEY AUTOINCREMENT,
  dataset          TEXT NOT NULL,
  symbol           TEXT,
  start_date       DATE NOT NULL,
  end_date         DATE NOT NULL,
  kind             TEXT NOT NULL,   -- BOOTSTRAP | GAP | FORWARD | QC_PROBE
  priority         INTEGER NOT NULL,
  status           TEXT NOT NULL CHECK (status IN ('PENDING','LEASED','DONE','FAILED')),
  attempts         INTEGER NOT NULL DEFAULT 0,
  lease_owner      TEXT,
  lease_expires_at TIMESTAMP,
  next_not_before  TIMESTAMP,
  last_error       TEXT,
  created_at       TIMESTAMP NOT NULL,
  updated_at       TIMESTAMP NOT NULL,
  UNIQUE(dataset, symbol, start_date, end_date, kind)
);

CREATE INDEX IF NOT EXISTS idx_backfill_queue_status
  ON backfill_queue(status, priority, created_at);
```

### 2.2 `partition_status` mirror

Mirror the Postgres schema in SSOT_V2, but store a copy in SQLite (plus the Parquet ledger at `data_layer/qc/partition_status.parquet`).

- For non‑trading days, insert `status='GREEN', expected_rows=0, row_count=0, qc_code='NO_SESSION'`.

Agent must expose a small helper module `data_node/db.py` with transactional helpers:

- `enqueue_task(dataset, symbol, start_date, end_date, kind, priority)`
- `lease_next_task(now, global_budget_state) -> Task | None`
- `mark_task_done(task_id)`
- `mark_task_failed(task_id, error, backoff_until=None)`

Leasing must set `status='LEASED'` + `lease_owner` + `lease_expires_at` in a single transaction; expired leases are treated as PENDING in `lease_next_task`.

---

## 3. Dispatcher algorithm (queue worker)

Core rule: always pick **oldest PENDING task in the highest‑priority lane** that fits vendor budgets.

Pseudocode:

```python
def lease_next_task(now):
    # 1) Find the next eligible task (priority + FIFO)
    task = conn.execute("""
      SELECT *
      FROM backfill_queue
      WHERE status = 'PENDING'
        AND (next_not_before IS NULL OR next_not_before <= ?)
      ORDER BY priority ASC, created_at ASC
      LIMIT 1
    """, (now,)).fetchone()

    if not task:
        return None

    # 2) Choose vendor for this dataset
    vendors = capabilities.for_dataset(task.dataset)   # reads configs/endpoints.yml

    # Filter vendors by entitlements and per-vendor budgets
    eligible = [v for v in vendors if budgets.can_spend(v, task.kind)]

    if not eligible:
        # no vendor can take this now; set next_not_before and bail
        backoff = now + timedelta(minutes=5)
        conn.execute("""
          UPDATE backfill_queue
          SET next_not_before=?, updated_at=?
          WHERE id=?
        """, (backoff, now, task.id))
        return None

    vendor = pick_best_vendor(eligible)  # min (vendor_priority, current_load)

    # 3) Lease the task
    lease_ttl = compute_ttl(task)        # e.g. 2–10 minutes based on window size
    conn.execute("""
      UPDATE backfill_queue
      SET status='LEASED',
          lease_owner=?,
          lease_expires_at=?,
          updated_at=?
      WHERE id=? AND status='PENDING'
    """, (NODE_ID, now + lease_ttl, now, task.id))

    # Re-read row to confirm lease (avoid race)
    # ...

    return task, vendor
```

On completion:

- **Success**: write raw parquet to `data_layer/raw/<vendor>/<table>/date=YYYY-MM-DD/symbol=SYMBOL/data.parquet`
  (options use `underlier=SYMBOL`; macro/series datasets remain date-only) via existing connector code, update manifests, update `partition_status` to GREEN/AMBER, increment vendor budget counters, mark task DONE.
- **Holiday/weekend** (empty): mark partition GREEN with `qc_code='NO_SESSION'`, task DONE.
- **429/5xx**: increment `attempts`, set `next_not_before` with exponential backoff, clear lease, leave `status='PENDING'`.
- **Hard 4xx / entitlement**: mark vendor as ineligible for `(dataset,symbol)` in a local cache or capability override; after N attempts, mark task FAILED and surface in UI.

Vendor budgets:

- `budgets.can_spend(v, kind)` must enforce:
  - `spent_total_v < soft_daily_cap[v]`
  - 0–85/90/100% slices by `kind` as in §1.2
  - per‑minute `hard_rpm` via token bucket (sliding window).

---

## 4. Gap detection vs “known backlog”

`audit_scan()` should:

1. Use table‑specific expectations from `configs/backfill.yml` (row counts, earliest dates, etc.).
2. For each `(dataset,symbol,dt)`:
   - Determine expected rows (0 for weekends/holidays).
   - Compare to existing raw partitions (date+symbol/underlier as above; macro/series date-only); write `partition_status` as GREEN/AMBER/RED.
3. For RED/AMBER partitions within the **current stage window**, *upsert* a `GAP` task into `backfill_queue` with:
   - `kind='GAP'`
   - `start_date=end_date=dt` (or a small window)
   - `priority=1`
   - Use `INSERT OR IGNORE` to avoid duplicates thanks to the `UNIQUE(...)` constraint.

This matches what you described: gaps exist in `partition_status` as soon as they are real, but audit never spams duplicate queue entries; the presence of a queued/leased task *is* the resolution plan.

---

## 5. Stage gating

Keep a small control table or YAML on SSD, e.g. `data_layer/control/stage.yml`:

```yaml
current_stage: 0
stages:
  0:
    name: "bootstrap_small"
    universe_size: 100
    equities_eod_years: 5
    equities_minute_years: 1
  1:
    name: "full"
    universe_size: 500
    equities_eod_years: 15
    equities_minute_years: 5
```

Agent responsibilities:

1. Wizard initializes `current_stage=0` and seeds `BOOTSTRAP` tasks accordingly (100 symbols × 5y EOD + 1y minute).  
2. At 02:00, `PlannerLoop`:
   - Computes GREEN fraction for Stage 0 using `partition_status` (≥0.98 threshold).
   - If met, update `current_stage=1` and enqueue Stage‑1 `BOOTSTRAP` tasks:
     - Extend history for Stage‑0 symbols.
     - Add remaining symbols up to 500.

No user input is needed for this; wizard just tells you what stage you’re in.

---

## 6. QC policies (Pi)

### 6.1 Structural QC (always‑on)

When `curate_incremental()` runs (nightly 02:00) on the Pi:

- For each affected partition:
  - Row‑count vs expected (consider half‑days if allowed).
  - Calendar continuity (no unexpected gaps; weekends/holidays as NO_SESSION).
  - Schema and dtype checks vs SSOT table schemas.
  - Basic outlier checks on returns/volume; set `qc_code` like `RET_OUTLIER`, `VOL_SPIKE`.
  - Checksum/hash per parquet partition for corruption detection.

Update `qc/partition_status.parquet` and SQLite mirror accordingly.

### 6.2 Weekly cross‑vendor QC

Once per week (e.g. Sunday 03:00):

1. Sample a “large” subset of `(dataset,symbol,dt)` in the active train window with GREEN status, biased toward:
   - Recently AMBER partitions,
   - High‑volume names,
   - Dates near corporate actions.

2. For each sample, enqueue a `QC_PROBE` task with `priority=2`:
   - Task’s fetcher:
     - Reads curated values (primary vendor).
     - Fetches same bars from secondary vendor(s) based on `configs/endpoints.yml` and the sourcing playbook mapping.
     - Computes relative diffs on close/high/low and volume.

3. If deviation above thresholds (e.g. 0.1–0.2% on prices, large volume ratios), set `qc_code='CROSS_VENDOR_MISMATCH'`, keep partition AMBER and surface in UI.

Rate‑limit usage:

- `QC_PROBE` tasks are subject to the 85/90/100% budget slices; because they also only run weekly, practical daily load remains moderate.

---

## 7. Unified Pi data‑node loops

Agent must replace `scripts/node.py` + Pi use of `ops/dags/nightly.py` with a unified `data_node.py`, **without** changing the workstation training flows defined in SSOT_V2 / carve‑out.

Four loops (threads or async tasks):

1. **ForwardIngestLoop**
   - Every N minutes:
     - Look at bookmarks/manifests (existing edge collectors) to see if today’s EOD/minute data is incomplete.
     - Enqueue `FORWARD` tasks into `backfill_queue` (kind=FORWARD, priority=5) rather than calling connectors directly.

2. **PlannerLoop**
   - Light run: every 4 hours:
     - `audit_scan()` on active datasets/universe; upsert `GAP` tasks for RED/AMBER partitions in current stage window.
   - Heavy run: once at 02:00:
     - Full `audit_scan()` across datasets.
     - Stage gating (Stage 0→1) and enqueuing new `BOOTSTRAP` tasks.
     - Optionally purge or downgrade very old FAILED tasks.

3. **QueueWorkerLoop**
   - Continuous loop using dispatcher algorithm (§3):
     - Lease next task (with budgets & leases).
     - Call dataset fetcher → raw → manifest → partition_status.
     - On success/failure, adjust budgets and update DB.

4. **MaintenanceLoop (02:00)**
   - `curate_incremental()` for all raw partitions touched since last maintenance.
   - Structural QC (`qc.refresh`) for those partitions.
   - On weekly QC day, enqueue `QC_PROBE` tasks.
   - Export curated tables + QC ledger to `exports/nightly/YYYY-MM-DD/`.

All loops share a `NodeStatus` object updated in memory and surfaced to the UI.

---

## 8. Wizard & `rich` UI

### 8.1 Pi data‑node wizard

Implement `scripts/pi_data_node_wizard.py` that:

1. **Detects environment & storage**:
   - Locate external SSD; propose default storage root `/mnt/.../data_layer`.
   - Set `EDGE_NODE_ID`, `TRADEML_ENV=local`, and conservative scheduler defaults (from SSOT_V2 §1.4).

2. **Collects API keys**:
   - Alpaca, Finnhub, Alpha Vantage, FRED, FMP, Massive.
   - Writes them into `.env` / config files exactly where existing connectors expect them.

3. **Initializes control DB & stage**:
   - Create `node.sqlite` with `backfill_queue` + `partition_status`.
   - Write `stage.yml` with Stage 0 config.

4. **Seeds Stage‑0 bootstrap**:
   - Build initial 100‑symbol universe (using existing universe builder or a simple static file).
   - Enqueue `BOOTSTRAP` tasks for 5y EOD + 1y minute for those symbols.

5. **Schedule config**:
   - Ask (or default) 02:00 maintenance time.
   - Write schedule config to YAML (or `.env`) so `data_node.py` can read it.

6. **Summary & launch**:
   - Print summary of vendors enabled, budgets, universe, stage, and maintenance time.
   - Offer to start `data_node.py` immediately, which spawns the `rich` dashboard.

### 8.2 `rich` dashboard

Create `data_node/ui.py` using `rich.live`:

- **Top status line**:
  - `NODE <id> | ENV <TRADEML_ENV> | ROOT <path> | QUEUE: PENDING <n_pend> FAILED <n_fail>`
- **Table of loops**:
  - Rows: ForwardIngest, Planner, Worker, Maintenance.
  - Columns: `status` (spinner/idle), `last_run`, `current_task` (dataset, kind, symbol/date range), `vendor`, `budget_used`.
- **Vendor budgets panel**:
  - Per vendor: `spent_today / soft_daily_cap` and current RPM as percentages (read from budgets manager).
- **Log tail**:
  - Last ~20 lines from `data_node.log`.

All loops update `NodeStatus` via a thread‑safe object; the dashboard refreshes every ~0.5–1.0 seconds.

---

## 9. SSOT_V2 updates (agent tasks)

The agent must treat SSOT_V2 as source of truth and patch it to match this design.

Concretely:

1. **§2.4 Backfill subsystem**:
   - Clarify that `backfill_queue` is the **unified ingest queue** on the Pi, not just for backfill.  
   - Mention `kind ∈ {BOOTSTRAP,GAP,FORWARD,QC_PROBE}` and that tasks are prioritized by `(priority, created_at)`.

2. **§2.8 Metadata & control tables**:
   - Extend the documented `backfill_queue` schema with `kind`, `lease_owner`, `lease_expires_at`.  
   - Note the uniqueness on `(dataset,symbol,start_date,end_date,kind)` for dedupe.

3. **§5 Orchestration & Node Loop SSOT**:
   - Replace “Pi node + nightly DAG” wording with **“unified Pi data‑node service”** that runs:
     - forward ingest → audit → queue‑driven backfill → curate → QC → nightly export *only*, with training/eval/promotion explicitly off‑Pi.

4. **§10 Wizardized Operations (Edge)**:
   - Update the Edge wizard description to:
     - Mention `node.sqlite` on SSD.
     - Mention queue‑based backfill/ingest instead of per‑vendor scheduler.
     - Mention stage‑gated expansion (100→500 symbols; 5y/1y→15y/5y).

5. **Cross‑reference Data_Sources_Detail & Data_Sourcing_Playbook**:
   - In the “Vendor & endpoint appendix”, ensure example limits align with the endpoints/limits doc’s limits table and the playbook’s coverage map.

6. **Architecture_SSOT alignment**:
   - Ensure backfill horizons and daily budgets in `backfill.yml` examples are consistent with the gated stage design (Stage 0 and Stage 1 windows are subsets of the `collect_earliest` horizons).

---

## 10. Concrete implementation checklist (for the agent)

**Step 0 — Wire SSOT & configs**

- [ ] Update `SSOT_V2.md` as in §9.  
- [ ] Verify `configs/backfill.yml` and `configs/endpoints.yml` contain limits for all vendors; add Finnhub/FMP/Massive entries following §1.1 and the free‑tier tables.

**Step 1 — Control DB & queue**

- [ ] Create `data_layer/control/node.sqlite` on SSD.  
- [ ] Implement `backfill_queue` and `partition_status` SQLite schema as in §§2.1–2.2.  
- [ ] Implement `data_node/db.py` with transaction‑safe helpers and leasing logic.

**Step 2 — Budget manager**

- [ ] Implement `data_node/budgets.py` that:
  - Loads per‑vendor `hard_rpm` and `soft_daily_cap` from configs.  
  - Maintains token buckets for RPM and daily counts.
  - Implements `can_spend(vendor, kind)` using 85/90/100% slices (§1.2).

**Step 3 — Fetcher wrappers**

- [ ] For each dataset (`equities_eod`, `equities_minute`, `options`, `macros_fred`, corp actions, etc.), wrap existing connector code (under `data_layer/connectors/`) in pure functions `fetch_<dataset>(vendor, symbol, start_date, end_date, kind)` that:
  - Call the vendor API respecting budgets and retries (per playbook).  
  - Write raw parquet + manifests exactly as existing code does (date+symbol/underlier partitioning; macro/series date-only).

**Step 4 — Queue worker**

- [ ] Implement `data_node/worker.py` with the dispatcher algorithm in §3, using:
  - `lease_next_task()`
  - `fetch_<dataset>()`
  - `partition_status` updater
  - Budget manager

**Step 5 — Planner & audit**

- [ ] Factor out `audit_scan()` from `ops/ssot/audit.py` to be callable on the Pi, writing both Parquet and SQLite `partition_status`.  
- [ ] Implement `PlannerLoop` that runs every 4h + heavily at 02:00, enqueuing `GAP` tasks and handling stage gating as in §§4–5.

**Step 6 — Maintenance pipeline**

- [ ] Wrap `curate_incremental()` + QC refresh from existing nightly DAG into `data_node/maintenance.py`.  
- [ ] Implement export step to `exports/nightly/YYYY-MM-DD/` with:
  - Curated tables needed for training.
  - `qc/partition_status.parquet`.
  - Optional manifest/checksum of exported files.

**Step 7 — Stage gating**

- [ ] Implement `stage.yml` and a helper that:
  - Computes GREEN fraction for Stage 0.
  - Triggers Stage 1 expansion when thresholds met, seeding new `BOOTSTRAP` tasks.

**Step 8 — Weekly cross‑vendor QC**

- [ ] Implement `qc_weekly_planner()` that:
  - Selects sample partitions.
  - Enqueues `QC_PROBE` tasks.
- [ ] Implement `fetch_qc_probe()` using secondary vendors from `endpoints.yml` and playbook mapping.

**Step 9 — Wizard**

- [ ] Implement `scripts/pi_data_node_wizard.py` per §8.1:
  - Environment detection, SSD path, `.env` creation, API keys, node id.
  - Control DB init + Stage 0 seeding.
  - Scheduled maintenance time and config.
  - Optionally starting `data_node.py` at the end.

**Step 10 — `rich` UI**

- [ ] Implement `data_node/ui.py` with:
  - Shared `NodeStatus` object.
  - Live dashboard as in §8.2.
- [ ] `data_node/__main__.py` should:
  - Start all four loops.
  - Start the `rich` dashboard.
  - Handle graceful shutdown (Ctrl‑C → all loops stop, leases eventually expire, restart is safe).

**Step 11 — Kill Pi nightly / keep workstation nightly**

- [ ] Remove/disable running `ops/dags/nightly.py` on the Pi; Pi uses `data_node.py` only.  
- [ ] Ensure workstation still uses SSOT‑defined nightly DAG for training/eval, consuming `exports/nightly` from the Pi.

**Step 12 — Tests**

- [ ] Add tests mirroring SSOT backfill acceptance tests (insert synthetic gaps → RED → `GAP` tasks → worker fills → GREEN).  
- [ ] Test budget enforcement: simulate large queue and ensure per‑kind slices behave as intended.
- [ ] Test restart safety: kill worker mid‑task, confirm leases expire and tasks re‑run cleanly.

---

This is enough detail for the agent to implement without improvising: queue semantics, budgets, QC cadence, stage gating, wizard, UI, and SSOT edits are all fixed.
