# TradeML Handoff

Last updated: 2026-06-22  
Repo: `/Users/zach/Documents/Git/TradeML`  
Branch checked: `main`

This replaces the previous root handoff. Treat that earlier file as discarded
draft context, not authority.

## Source Of Truth

Use these documents in this order:

1. `docs/research/PATH_FORWARD_SSOT.md` owns current research direction,
   killed alpha paths, and restart criteria.
2. `SSOT.md` remains the engineering and infrastructure contract: PIT
   discipline, storage layout, validation discipline, cost assumptions,
   local-SQLite rule, and implementation invariants.
3. `DIRECTION_CONSOLIDATION_PACKET.md` is a review map for collapsing the docs
   into one final SSOT. It is not final authority.
4. `DEV_GUIDE.md`, `docs/AUTOPILOT_ROADMAP.md`, and older handoff/session docs
   describe useful old machinery but can conflict with the current runtime
   freeze.

Important doc caution: root `AGENTS.md` still points at absent
`docs/SSOT_V3.md`. Root `CLAUDE.md` and live `SSOT.md` point at `SSOT.md`.
Until Zach asks for doc cleanup, prefer the live `SSOT.md` plus
`PATH_FORWARD_SSOT.md`.

## Current Direction

TradeML is paused as an autonomous system. The current active research shape is:

> Shared PIT infrastructure plus narrow, pre-registered, source-first
> public-event hypotheses with strict evidence, controls, and kill criteria.

The old broad daily ranker is not the active alpha path. Keep it only as a
sentinel baseline, leakage detector, market-context control, and comparison
surface.

The SEC/event spine is still valuable infrastructure:

- SEC raw archive and manifest ingest;
- source-structure parsing;
- candidate generation;
- market backfill;
- first-tradable-time labeling;
- timestamp placebo controls;
- event-study and decision packets;
- schema/evidence validation for model-extracted semantics.

## Current Stop Rules

Do not restart or broaden any runtime before the next gate is accepted.

- Do not restart the Raspberry Pi collector.
- Do not restart the Mac mini perpetual research loop.
- Do not run the old broad SEC 8-K item loop.
- Do not promote any current result to paper/live.
- Do not use live Alpaca endpoints for trading.
- Do not use regex, keyword lists, or header-only deterministic rules to decide
  economic meaning in narrative sources.

Deterministic parsing is allowed for stable source structure: SEC accession
metadata, SGML/XML tags, item boundaries, timestamps, paths, hashes, schemas,
source offsets, and validation. Narrative economic meaning must come from an
LLM or purpose-built model into a strict schema, then be validated
deterministically against exact evidence spans/source locators.

## What Is Killed

Form 4 naive insider-buy alpha is killed.

- Verdict: `FORM4_KILLED_BASELINE_COMPLETE`.
- The strict open-market common-stock buy rule failed promotion gates.
- The bounded rework variants also failed; the best-looking `ceo_cfo_only`
  slice had only 47 labeled events and failed the `n >= 100` minimum.
- Keep Form 4 as source-first event-pipeline benchmark infrastructure only,
  unless a new documented hypothesis changes the event definition, data policy,
  or controls.

Broad deterministic SEC 8-K item-family alpha is killed.

- Verdict: `BROAD_SEC8K_ITEM_FAMILIES_KILLED`.
- April 2025 1,000-filing slice parsed 1,000 submissions and produced 600
  labeled events.
- Primary 5-day abnormal net return mean was about `-1.41%`; median about
  `-0.60%`.
- Timestamp placebo outperformed real events by about `4.42%`.
- Header-only item parsing may remain a negative control and source-routing aid
  only.

## What Is Not Killed

Semantic SEC/event extraction is mechanically viable but not alpha-proven.

- Latest summarized semantic 8-K state: `MORE_DATA_REQUIRED`.
- Primary labeled sample was `n=6`, far below a tradable evidence bar.
- Scaled gate failed for insufficient semantic event frequency.
- Tiny positive-looking returns are not evidence of edge; they are a warning to
  fix hypothesis frequency and controls before more model/runtime spend.

## Next Gate

The next valid work is offline and bounded:

1. Pick exactly one narrow public-event hypothesis.
2. Pre-register the mechanism: who is mispricing what, why it should persist,
   who is on the other side, expected holding period, capacity, costs, and kill
   criteria.
3. Run source inventory and labelability before any LLM classification.
4. Prove a plausible path to at least 100 labeled primary events after
   eligibility, liquidity, and first-tradable-time filters.
5. If frequency or labelability fails, kill or re-scope before model spend.
6. If frequency passes, run semantic extraction into a strict schema with exact
   evidence validation.
7. Run event study with timestamp placebo and at least one semantic negative
   control.
8. Produce a decision packet: `CONTINUE`, `REWORK`, or `KILL`.

Candidate families worth frequency-testing, per `PATH_FORWARD_SSOT.md`:

- material customer or supplier contract awards;
- dilutive financing or toxic security issuance;
- covenant/default/liquidity stress;
- auditor trouble with stronger historical coverage;
- customer loss or major contract termination;
- regulatory/legal action with explicit company impact.

Pick one family for the next MVP. Do not run a menu of hypotheses through the
old autonomous loop.

## Storage And Runtime Cautions

Canonical storage remains the NAS `dev` share:

- local Mac: `/Volumes/dev/TradeML`;
- Mac mini: `/Users/openclaw/atlas_mounts/dev/TradeML`;
- Raspberry Pi: `/mnt/dev/TradeML`.

SQLite stays local to the owning machine. Parquet/raw/curated artifacts go to
the NAS. Do not recreate the old local fake-NAS root under
`/Users/zach/atlas_mounts`.

This handoff did not verify live Pi, Mac mini, NAS, launchd, systemd, or remote
process state. The task was explicitly docs/local-checkout only. Any future
runtime claim should be checked from live state before action.

The codebase still contains old control surfaces for the perpetual research
program, dashboard start/stop controls, Pi service management, and autopilot
health checks. Their existence is not authorization to use them during the
current freeze.

## Recent Local History

Latest relevant commits on `main` at handoff rewrite time:

- `1434ce3 Document TradeML public event handoff`
- `f48a3f1 refactor: retire legacy backfill execution`
- `83155bf feat: maximize pi data collection`
- `4b9e07f Update canonical SSOT`
- `591cb67 Advance stage collection and canonicalize SSOT`

The currently committed public-event implementation includes:

- `src/trademl/events/` Form 4, SEC 8-K, semantic-classifier, coverage, label,
  and event-study modules;
- SEC EDGAR connector expansion;
- CLI surfaces under `trademl research ...`;
- tests covering Form 4, SEC 8-K, semantic classification, provider contracts,
  capabilities, connectors, CLI, and script integration;
- NAS `dev` path updates in config and ops scripts.

## Validation Status

Local validation performed for this handoff:

- `git status --short` before editing: clean.
- No services were started. No Pi, Mac mini, NAS, launchd, systemd, or remote
  runtime checks were performed.
- Targeted local event/research suite passed:

  ```bash
  python -m pytest tests/unit/test_form4.py tests/unit/test_sec8k.py tests/unit/test_sec8k_coverage.py tests/unit/test_sec8k_semantic.py tests/unit/test_semantic_classifier.py tests/unit/test_connectors.py tests/unit/test_cli.py tests/unit/test_capabilities.py tests/unit/test_provider_contracts.py tests/integration/test_scripts.py -q
  ```

  Result: `175 passed in 50.31s`.

## Exact Next Steps

1. Decide whether to consolidate `SSOT.md`, `PATH_FORWARD_SSOT.md`, and the
   direction packet into one final SSOT. That is the repo-level cleanup Zach
   asked for in the packet, but it was out of scope for this handoff-only task.
2. Fix the stale `AGENTS.md` canonical-spec pointer when doc cleanup is in
   scope.
3. Choose one narrow public-event hypothesis for the offline frequency and
   labelability gate.
4. Write the pre-registration and source-inventory/labelability test before
   adding model extraction or restarting any runtime.
5. Only after the gate passes, update the Mac mini program state and restart
   commands so they run the accepted gate, not stale `perpetual-macmini`
   research loops.
