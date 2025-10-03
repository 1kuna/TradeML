Short answer: the original blueprint implied resumable ingestion, but it didn’t spell out a formal backfill & gap-repair subsystem. You’re right—we should add it. Below is a drop-in blueprint update your team (or LLM) can implement without guessing.

Backfill & Data Quality — Drop-in Module

Why

Two concurrent “sweeps”:
	•	Forward sweep (today → future): regular ingestion with bookmarks.
	•	Backward sweep (today → past): backfill workers that detect gaps and pull history in rate-limit–friendly chunks until coverage goals are met.

Scope (what we backfill)
	•	Equities: EOD OHLCV, minute bars, (optional) single-venue ticks (IEX) for microstructure protos.
	•	Options: daily chains + NBBO mids (if/when you have OPRA), IV surfaces (SVI) derived from mids.
	•	Reference: corporate actions & delistings (full history first), index membership (as available), macro/rates (FRED/ALFRED vintages), SEC filing indices.

Data completeness model

Define completeness per (source, table, symbol, date), graded:
	•	GREEN: partition exists and QC passes (row counts in bounds, calendars match, checksums recorded).
	•	AMBER: partition exists but QC is marginal (low rows, partial day, vendor outage).
	•	RED: partition missing.

Completeness is stored in partition_status and drives backfill priorities.

⸻

Storage & state (MinIO/Tailscale stays the same)

s3://ata/raw/<source>/<table>/date=YYYY-MM-DD/...parquet
s3://ata/curated/<table>/date=YYYY-MM-DD/...parquet
s3://ata/manifests/bookmarks.json        # forward sweep watermarks (per worker)
s3://ata/manifests/backfill_marks.json   # backward sweep watermarks (per worker)
s3://ata/qc/partition_status.parquet     # GREEN/AMBER/RED + metrics
s3://ata/locks/<worker>.lock             # lease files

Bucket versioning enabled (already in your MinIO drop-in).

⸻

New DB tables (SQLite or Postgres)

-- completeness / QC (one row per partition)
CREATE TABLE partition_status (
  source TEXT, table_name TEXT, symbol TEXT, dt DATE,
  status TEXT CHECK(status IN ('GREEN','AMBER','RED')),
  rows INT, expected_rows INT, qc_score REAL,
  last_checked TIMESTAMP, notes TEXT,
  PRIMARY KEY (source, table_name, symbol, dt)
);

-- backfill queue ledger
CREATE TABLE backfill_queue (
  id INTEGER PRIMARY KEY,
  source TEXT, table_name TEXT, symbol TEXT, dt DATE,
  priority INT, attempts INT DEFAULT 0,
  enqueued_at TIMESTAMP, last_attempt TIMESTAMP, last_err TEXT,
  UNIQUE (source, table_name, symbol, dt)
);


⸻

Backfill config (new backfill.yml)

scheduler:
  timezone: America/New_York
policy:
  max_concurrency: 3
  daily_api_budget:
    iex:   { requests: 8000 }
    alpaca:{ requests: 5000 }
    fred:  { requests: 2000 }
    sec:   { requests: 1000 }
    av:    { requests: 500 }
    opra:  { requests: 20000 }  # when enabled
  priorities:
    - { table: corp_actions,   weight: 100 }   # fix survivorship first
    - { table: delistings,     weight: 100 }
    - { table: equities_eod,   weight: 90 }
    - { table: equities_minute,weight: 80 }
    - { table: options_chains, weight: 70 }
    - { table: options_nbbo,   weight: 60 }

targets:
  equities_eod:
    earliest: "2000-01-01"
    chunk_days: 30
    expect_rows: 1             # per symbol per date
  equities_minute:
    earliest: "2010-01-01"
    chunk_days: 5
    expect_rows: 390           # RTH bars; adjust with calendar
  iex_ticks:
    earliest: "recent"         # depends on vendor window
    chunk_days: 1
    expect_rows_min: 1
  options_chains:
    earliest: "2015-01-01"
    chunk_days: 1
    expect_contracts_min: 50

qc_thresholds:
  rows_low_ratio: 0.9
  rows_high_ratio: 1.2
  allow_halfdays: true


⸻

How it runs (control flow)

1) Nightly audit (on any device)
	•	Scan calendars and raw/ to compute expected vs. actual rows.
	•	Write/merge partition_status.parquet.
	•	Populate/refresh backfill_queue for RED and AMBER partitions.

2) Backfill worker (ROLE=edge; runs on Pi by default)
	•	Acquire lease locks/backfill.<table>.lock.
	•	Pull a batch from backfill_queue ordered by priority, recency, and “gappiness” (fill latest holes first, then extend further back).
	•	For each task:
	1.	Read backfill_mark to avoid overlapping windows across devices.
	2.	Fetch vendor data in chunk_days windows with rate-limit aware pacing.
	3.	Write append-only to raw/...tmp, verify ETag, move to final key.
	4.	Insert/append manifest rows; mark partition GREEN/AMBER depending on QC.
	5.	Advance backfill_mark and re-enqueue if still incomplete.

3) Curator (ROLE=curator; PC/Mac)
	•	On start (and hourly), list new raw/ partitions; build curated/ for those dates.
	•	Recompute derived artifacts affected by backfill (e.g., rolling features, SVI surfaces).
	•	Update a curator watermark and refresh partition_status for those partitions.

⸻

Idempotency & dedupe
	•	Same rules as forward ingestion: write to temp → verify checksum/ETag → move → only then update marks.
	•	Vendor windows are closed intervals (e.g., [t0, t1)); backfill workers don’t write over existing keys—only add missing ones.
	•	If a window returns empty or too few rows, tag AMBER with a reason (vendor outage, holiday, half-day) and schedule a low-frequency recheck.

⸻

Rate-limit governance
	•	Each worker maintains a per-vendor token bucket per backfill.yml.
	•	If budget is low, the worker:
	•	shrinks chunk size,
	•	sleeps to next window, or
	•	yields the lease (lets another table progress).

⸻

Vendor-specific notes (free-first reality)
	•	IEX HIST/DEEP: typically T+1 and limited history on free; backfill up to vendor window; store coverage metadata so models don’t assume earlier IEX ticks exist.
	•	Alpaca bars: straightforward for minute; be strict with exchange calendars (holidays/halfdays).
	•	FRED/ALFRED: ask for full series history immediately (cheap), then delta-poll daily; store vintage date for PIT correctness.
	•	SEC EDGAR: backfill filing index by day; rate-limit heavily; store filing_time for PIT alignment.
	•	Alpha Vantage: backfill LISTING_STATUS and corporate actions; reconcile with SEC.
	•	Options: if you only have chains (no NBBO), record as low-fidelity in partition_status so the curator can skip SVI on those dates or mark derived outputs AMBER.

⸻

CLI & Make targets

# audit and plan
ata-audit gaps --tables equities_eod equities_minute options_chains
ata-backfill plan --max 500 --since 2020-01-01

# run until budget exhausted (Pi)
ROLE=edge STORAGE_BACKEND=s3 ata-backfill run --budget vendor=alpaca:3000,sec:500

# recheck AMBER partitions only
ata-audit recheck --status AMBER --tables equities_minute


⸻

Acceptance tests
	1.	Synthetic holes: delete a day of minute bars for 10 symbols → ata-audit gaps marks RED; backfill fills it; status flips GREEN.
	2.	Rate-limit simulation: force low budgets → worker scales down chunk size and eventually completes.
	3.	Overlap safety: run two backfill workers on different machines → leases prevent double work; ETag preconditions catch races.
	4.	Derived rebuild: backfill a past week → curator recomputes rolling features and SVI surfaces only for affected dates; no stale artifacts remain.
	5.	Corporate-action backfill first: with splits/divs added late, curated OHLCV adjusts deterministically and hashes change only for affected dates.

⸻

What changes in the repo (diff summary)
	•	New: backfill.yml, ata-audit, ata-backfill (simple Python CLIs).
	•	New tables/files: partition_status, backfill_queue, backfill_marks.json.
	•	Edge collector gains a backfill worker mode (same binary; --mode backfill).
	•	Curator watches raw/ and incrementally rebuilds curated/ for backfilled dates.
	•	Dashboards add a coverage heatmap (date × dataset) and QC counters.

⸻

Rollout order (do this first)
	1.	Backfill corporate actions/delistings (entire available history).
	2.	Backfill EOD OHLCV for your universe (broadest coverage, cheapest).
	3.	Backfill minute bars for top-1000 names (last N years).
	4.	Backfill options chains for top underliers (daily snapshots).
	5.	FRED/ALFRED + SEC full history first, then delta updates.

⸻

Policy: training only after GREEN thresholds
	•	Define per-table GREEN thresholds (e.g., ≥98% partitions GREEN in last 2 years).
	•	Models train only on GREEN partitions; AMBER allowed only if you explicitly override and log why.
	•	Every experiment report includes the coverage heatmap for its data slice.