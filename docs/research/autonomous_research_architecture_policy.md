# Autonomous Research Architecture Policy

This policy translates the GPT Pro architecture recommendation into the repo-native Mac Mini research loop. It is a control-plane policy, not a new feature pipeline.

## Primary Framing

TradeML optimizes supervised cross-sectional alpha prediction plus cost-aware portfolio validation. The research loop should predict point-in-time residual/rank returns, validate with walk-forward and backtest gates, emit paper/shadow signals, and promote only research incumbents. It must not use live order placement or promote live-trading models.

## Architecture Priority

1. `linear_baseline`: Ridge/rank sentinel. It is the leakage detector and minimum viable incumbent baseline.
2. `tree_challenger`: LightGBM workhorse. It should remain a cheap sentinel/workhorse in every frontier epoch.
3. `advanced_challenger`: CatBoost-first challenger with LightGBM/Ridge fallback metrics.
4. `ensemble_meta`: implemented rank-averaged Ridge/LightGBM meta lane. It becomes a pivot lane after advanced-first epochs fail to produce a promotable candidate.
5. `tabular_deep_challenger`: deferred logical lane until a dedicated tabular deep training suite exists.

RL, sequence transformers, GNNs, and foundation forecasters are disabled future lanes. They are not alpha engines for the current autonomous path.

## Objective Policy

The canonical objective is `research_profitability_v1`.

Promotion evidence is ranked by:

- primary rank IC, using the architecture registry metric order;
- all-year positivity and assessment `GO`;
- cost-positive backtest and cost-stress net return;
- PBO and placebo robustness;
- drawdown and turnover when available;
- improvement over incumbent after complexity penalty.

The simpler model wins when a more complex model is statistically indistinguishable. A complex model must beat the incumbent after the configured complexity penalty, not just tie it on raw rank IC.

## Autonomous Progression

The frontier lane stays advanced-first, but every epoch carries one tree sentinel and one linear sentinel unless budget or dependency preflight blocks them. When an epoch exhausts without a promotable candidate, the supervisor pivots to a new epoch, window, or validated data/feature lane rather than waiting or repeating stale broad cartesian searches.

Raw minute/news/text archives remain excluded from training until a separate point-in-time feature curation pass makes them modeling-ready.

Narrative news, filing, exhibit, IR, legal, regulatory, and contract text must
not be semantically classified with regex or keyword heuristics. Deterministic
parsing may segment stable source structure and preserve provenance; semantic
event extraction must use an LLM or purpose-built model into a strict schema,
with deterministic validation of evidence spans/source offsets, supported
numeric fields, timestamps, categorical certainty/ambiguity, and allowed event
classes. Numeric model confidence scores are not accepted as extraction truth.
