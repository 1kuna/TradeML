# TradeML Path Forward SSOT

Last updated: 2026-05-20

## Status

This is the active source of truth for TradeML's research direction and restart
criteria.

The repo-wide `SSOT.md` remains the infrastructure and engineering contract:
storage layout, PIT rules, validation discipline, cost assumptions, data-node
shape, and implementation invariants. This document owns the current research
decision: what alpha path is alive, what has been killed, and what must be
proven before the Raspberry Pi collector or Mac mini research loop are restarted.

Raw GPT Pro / GPT-5.5 responses are source artifacts, not active instructions.
Child MVP docs are postmortems unless this document explicitly reactivates them.

## Runtime Freeze

As of 2026-05-20, autonomous TradeML runtime is intentionally paused while the
research direction is consolidated.

- Raspberry Pi `trademl-node.service`: stopped and disabled.
- Mac mini `com.trademl.research.perpetual-macmini`: booted out if loaded and
  disabled.
- Mac mini `com.trademl.mount-nas`: left running because it is mount plumbing,
  not an autonomous research or collection loop.

Do not restart the Pi collector or Mac mini research loop until this document's
next gate is accepted and the restart command is tied to that gate.

## Current Verdict

The GPT Pro critique was directionally correct: the original broad architecture
was too easy to turn into an impressive but unfalsifiable research machine. The
path forward is not a unified financial world model, not generic options AI, and
not more autonomous search over broad daily features.

TradeML's viable direction is:

> Shared PIT infrastructure plus narrow, pre-registered, source-first public
> event hypotheses with strict evidence, controls, and kill criteria.

The current SEC/event infrastructure is useful. The tested alpha candidates are
not yet viable.

## What Is Killed

### Generic Daily Ranker As Primary Alpha

The generic daily cross-sectional ranker is not the active alpha direction. It
can remain as a sentinel baseline, leakage detector, market-context control, and
comparison surface, but it should not drive paper/live entries.

### Form 4 Naive Insider-Buy MVP

Verdict: `FORM4_KILLED_BASELINE_COMPLETE`.

Evidence:

- First strict Form 4 open-market common-stock buy rule was not promotable.
- Bounded rework variants were tested and none passed all continuation gates.
- The best-looking `ceo_cfo_only` slice had only 47 labeled events and failed
  the minimum sample gate.

Decision:

- Do not spend more automated search budget on Form 4 unless a new documented
  hypothesis changes the event definition, data revision policy, or controls.
- Keep Form 4 only as source-first event-pipeline benchmark infrastructure.

### Broad Deterministic 8-K Item Families

Verdict: `BROAD_SEC8K_ITEM_FAMILIES_KILLED`.

Evidence from the April 2025 1,000-filing slice:

- 1,000 manifest rows parsed successfully.
- 1,151 deterministic item candidates, 708 eligible.
- 600 labeled events.
- 5-day abnormal net return mean: `-0.014125810557440959`.
- 5-day abnormal net return median: `-0.005956147455785862`.
- Timestamp placebo mean difference: `-0.04421802109848674`.
- `move_forward=false`; `paper_live_allowed=false`.

Decision:

- Do not continue broad header-only item-family alpha.
- Header-only or deterministic item parsing may remain a negative control and
  source-routing aid only.

## What Is Not Killed

### SEC/Event Spine

Keep the SEC and event-research infrastructure:

- SEC ingest and raw archive storage;
- candidate generation;
- first-tradable-time labeling;
- market backfill;
- timestamp placebo controls;
- event-study and decision packets;
- strict report/state artifacts.

This infrastructure is the main reusable asset from the last iteration.

### Semantic 8-K Extraction Path

The semantic path is mechanically viable but not alpha-viable yet.

Current semantic 8-K state:

- Classifier path verdict: `PASS` for the extraction/classification mechanics.
- Latest semantic study verdict: `MORE_DATA_REQUIRED`.
- Primary labeled sample: `n=6`.
- Primary 5-day abnormal net return mean: `0.036617823940397705`.
- Primary median: `0.004295468611636561`.
- Failed gate: `insufficient_labeled_events`.
- Scaled gate failed: `insufficient_semantic_event_frequency`.
- `move_forward=false`; `paper_live_allowed=false`.

Interpretation:

- The extraction machinery can produce schema-validated semantic event objects.
- The currently targeted semantic queue is far too sparse to make a trading
  decision.
- Positive-looking `n=6` results are not evidence of edge. They are a warning to
  improve candidate frequency and controls before spending more model/runtime
  budget.

## Active Research Shape

TradeML should operate as a public-event research foundry:

```text
public source record
  -> append-only PIT raw archive
  -> source-structure parser
  -> LLM or purpose-built semantic extractor
  -> deterministic schema/evidence validator
  -> entity-linked tradable instrument
  -> first-tradable-time label
  -> event study with controls
  -> continue / rework / kill packet
  -> only then paper/shadow consideration
```

Model policy:

- Use deterministic code for source structure, timestamps, paths, hashes,
  stable SEC fields, schemas, validation, controls, and audit trails.
- Use an LLM or purpose-built model for narrative economic meaning.
- Do not use regex, keyword lists, or header-only rules to decide sentiment,
  materiality, dilution, contract substance, auditor trouble, covenant/default
  stress, customer loss, guidance changes, or any other economic meaning.
- LLM outputs must include strict schema fields, exact evidence snippets, and
  source offsets or equivalent source locators before downstream use.

## Next Gate

The next restart must be a bounded offline research gate, not an autonomous
always-on loop.

Gate: choose one narrow public-event hypothesis and prove that it has enough
PIT source coverage, labelability, and semantic frequency before running another
alpha study.

Required before any restart:

1. Pre-register exactly one hypothesis.
2. Define who is mispricing what, why the mispricing might persist, who is on
   the other side, expected holding period, capacity, costs, and what kills it.
3. Run a source inventory and labelability pass before LLM classification.
4. Require a plausible path to at least 100 labeled primary events after
   eligibility, liquidity, and first-tradable-time filters.
5. If frequency fails, kill or re-scope before model spend.
6. If frequency passes, run semantic extraction into a strict schema with exact
   evidence validation.
7. Run event study with timestamp placebo and at least one semantic negative
   control.
8. Produce a decision packet with `CONTINUE`, `REWORK`, or `KILL`.

Paper/live remains blocked until a later shadow-validation gate passes. No
current result authorizes paper/live trading.

## Recommended Immediate Hypothesis Search

Do not restart the old broad SEC 8-K item loop. The immediate offline work is to
build a candidate-frequency ledger over the existing SEC/event archive for a
small menu of semantic event hypotheses, then pick one.

Candidate families worth frequency-testing:

- material customer or supplier contract awards;
- dilutive financing or toxic security issuance;
- covenant/default/liquidity stress;
- auditor trouble with stronger historical coverage;
- customer loss or major contract termination;
- regulatory/legal action with explicit company impact.

The output should be a table with:

- candidate family;
- source documents and stable routing rule;
- candidate count before semantics;
- labelable count after ticker, liquidity, and market-data filters;
- expected semantic positive rate;
- minimum viable historical window;
- required model/schema;
- known confounders and controls;
- proceed / reject recommendation.

Pick only one family for the next MVP.

## Document Roles

- `docs/research/PATH_FORWARD_SSOT.md`: active research direction and restart
  criteria.
- `SSOT.md`: engineering/infrastructure invariants.
- `docs/research/public_event_research_pivot_ssot.md`: historical pivot detail;
  subordinate to this document for current next steps.
- `docs/research/form4_insider_purchase_event_mvp_direction.md`: Form 4
  postmortem and reusable source-pipeline notes.
- `docs/research/sec_8k_item_event_mvp_direction.md`: 8-K item/semantic
  postmortem and reusable SEC/event mechanics.
- `docs/research/source_prompts/`: raw GPT Pro / GPT-5.5 source archive.

## Restart Commands

Only run these after the next gate is accepted.

Raspberry Pi collector:

```bash
ssh trademl-pi-lan 'systemctl --user enable --now trademl-node.service'
```

Mac mini research loop:

```bash
ssh trademl-mac-lan 'launchctl enable gui/$(id -u)/com.trademl.research.perpetual-macmini && launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.trademl.research.perpetual-macmini.plist'
```

Before restart, update the Mac mini program state so it runs the accepted gate,
not the stale `perpetual-macmini-p1-f1160` research loop.
