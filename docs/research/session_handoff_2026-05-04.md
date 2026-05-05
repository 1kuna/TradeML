# TradeML Session Handoff - 2026-05-04

This handoff summarizes the current TradeML state after the recent Pi collection, Mac Mini research, observability, feature-canary, and autonomy-hardening work. It is intended for the next Codex/operator session to pick up without relying on chat memory.

## Current Repo State

- Repo: `/Users/zach/Documents/Git/TradeML`
- Branch: `main`
- Worktree at handoff creation: clean
- Latest commit observed: `1f56eb7 fix: ground fleet observability in pi telemetry`
- Recent important commits:
  - `1f56eb7 fix: ground fleet observability in pi telemetry`
  - `4fbe3b1 fix: expose feature leaderboard in remote health`
  - `2831ceb fix: preserve duplicate feature canary evidence`
  - `affcf3c fix: finalize completed feature canary entries`
  - `d9cb56d fix: record all blocked feature canaries`
  - `aad4cf4 perf: vectorize SEC feature readiness build`
  - `d58df4f feat: gate feature canaries on source readiness`
  - `f078ce0 fix: pause research during feature canary batches`

## Remote Hosts

Do not commit credentials. Zach has supplied credentials in chat before, but this handoff intentionally records only host identities.

- Raspberry Pi data node: Tailscale `100.76.4.69`, local/LAN IP previously `192.168.68.89`
- Mac Mini research host: Tailscale `100.102.98.14`, local/LAN IP previously `192.168.68.70`
- NAS: LAN IP previously `192.168.68.54`

## North Star

TradeML is being shaped into an autonomous research and data-collection system:

- Raspberry Pi continuously collects as much free-plan market data as possible into NAS-backed parquet archives while keeping SQLite local to the Pi.
- Mac Mini continuously runs supervised, PIT-safe research and promotion checks against curated/modeling artifacts.
- Dashboard/fleet observability should answer the high-level questions: systems online, data collecting, research running, paper/shadow performance, and whether Codex should inspect.
- No live trading, live Alpaca endpoint usage, or live model promotion is allowed in the current phase.

## Major Work Completed This Session

### Pi Data Collection

- Moved the data node toward planner-native, multi-lane collection rather than old legacy backfill behavior.
- Added/expanded budget-aware scheduling, lane health, stale lease recovery, and scheduler decision telemetry.
- Added handling for entitlement/permanent failures so paid/unavailable lanes do not wedge hot loops.
- Added/paralleled Alpaca lane expansion work for stocks/news/crypto/options-style archive surfaces, guarded by entitlement audits.
- Added/fixed archive schema normalization and telemetry for parquet append behavior.
- Added deployment provenance and Pi-side observability surfaces so fleet health can read live node truth instead of stale controller state.

### Mac Mini Research

- Hardened the perpetual research control plane:
  - preflight gates,
  - stale-run sweeping,
  - incumbent registry behavior,
  - strict promotion gates,
  - paper/shadow output boundaries,
  - no-live-orders policy.
- Pivoted architecture search to controlled advanced-first frontier behavior.
- Added architecture and objective registries for Ridge/linear sentinels, LightGBM/tree challengers, CatBoost/advanced challengers, and ensemble/meta lanes.
- Added autonomous progression state so repeated non-promotable lanes pivot rather than wait forever.
- Added strong-rejected candidate autopsy/follow-up behavior.
- Added GPT Pro architecture source artifact at `docs/research/gpt_pro_architecture_recommendation_2026-04-29.md`.
- Added paper trading API notes at `docs/research/paper_trading_api.md`.

### Feature Factory And Research Inputs

- Added versioned modeling feature/label factory flow:
  - `price_liquidity_v1`
  - `sec_filing_events_v1`
  - `news_event_aggregates_v1`
  - `minute_daily_aggregates_v1`
  - `multi_source_daily_v1`
- Added source adapters/readiness gates so feature canaries block when required source coverage is genuinely missing instead of creating fake successful artifacts.
- Added feature canary orchestration and backend-only feature-family leaderboard.
- Added checks around PIT metadata, label maturity, coverage status, and duplicate canary evidence.

### Fleet Observability

- Added a fleet observability snapshot and issue-bucket-oriented current-state layer.
- Added Pi remote SQLite telemetry summarization for scheduler decisions, archive writes, lane health, attempts, planner task status, and deployed/schema provenance.
- Updated observability to distinguish current blockers from historical terminal failures.
- Updated current-state logic so missing incumbent or immature paper/PnL evidence is pending, not degraded.
- Added watchdog-oriented issue cleanup/current-state truth.

## Latest Live Mac Mini Research Status

Last checked on 2026-05-04 from this repo via SSH to the Mac Mini.

Mac research is running.

Key observed state:

- Program: `perpetual-macmini`
- Current experiment: `perpetual-macmini-p1-f547`
- Experiment status: `RUNNING`
- Queue counts at check time: `COMPLETED: 1`, `RUNNING: 1`, `PLANNED: 2`
- Active run id: `dcda2fdad4`
- Active run lane: `ensemble`
- Active report date: `2026-03-09`
- Frontier completed runs: `2665`
- Frontier search epoch: `36`
- Shortlist count: `0`
- Incumbent: none
- Paper account smoke: `ok`
- Latest paper/shadow PnL: approximately `-0.13%`

Best historical candidate still known from research status:

- Experiment: `perpetual-macmini-p1-f244`
- Run: `ae6b12e3df`
- Candidate/lane: CatBoost advanced
- Rank IC / primary score: about `0.0771`
- Not shortlisted because `assessment.decision != GO` and not all yearly IC values were positive.

Recent experiments show signal but no promotable model:

- Several recent candidates were in the `0.062` to `0.069` IC range.
- These were rejected primarily for robustness/stability, especially yearly IC positivity.
- Current experiment `p1-f547` was weaker so far, with best completed score about `0.0094`, classified `weak_rejected`.

Interpretation: the system is finding meaningful signal, but the strict gates are correctly preventing unstable models from becoming incumbents.

## Current Known Blockers

### 1. Multi-Source Feature Coverage Mismatch

Research status still reported feature-label preflight as not OK for multi-source feature versions:

```text
news_events has no usable source-backed feature coverage;
minute_daily has no usable source-backed feature coverage
```

Earlier live evidence suggested:

- Pi has source archives under its NAS mount:
  - `/mnt/trademl/data/raw/ticker_news`
  - `/mnt/trademl/data/raw/equities_minute`
- Mac was looking under:
  - `/Users/openclaw/atlas_mounts/nas/data/raw/ticker_news`
  - `/Users/openclaw/atlas_mounts/nas/data/raw/equities_minute`
- Those Mac paths were missing at the time of the earlier check.

Likely diagnosis: the Pi and Mac do not currently see the same NAS path/layout for these datasets, or the source contract still maps to a path that is not actually visible from the Mac.

Why it matters: price-only research has found unstable-but-interesting signal. The next meaningful improvement is getting `ticker_news` and `equities_minute` source coverage visible to the Mac feature factory so the new PIT-safe feature versions can be canaried.

### 2. No Promotable Incumbent Yet

This is not an infra problem and should not be treated as degraded. It means the strict gates are doing their job. There is signal, but no candidate has passed prediction, robustness, cost/backtest, and paper evidence gates.

### 3. Paper/Shadow Evidence Still Immature

Paper smoke works, and paper/shadow artifacts exist, but latest observed paper/shadow PnL was negative at about `-0.13%`. Pending/immature evidence should remain a pending state, not degraded.

## Suggested Next Steps

### First Priority: Fix Mac-Visible Source Contract

Goal: make Mac feature readiness reflect the actual Pi-collected `ticker_news` and `equities_minute` archives.

Concrete checks:

1. On the Pi, confirm current archive paths and row growth:
   - `ticker_news`
   - `equities_minute`
2. On the Mac, confirm whether the same NAS share is mounted and where those directories appear.
3. Inspect the shared source contract artifact:
   - `control/cluster/state/research/feature_source_contract/latest.json`
4. Update the source contract or NAS mount mapping so Mac feature factory can read the actual archives.
5. Re-run feature readiness/build for:
   - `news_event_aggregates_v1`
   - `minute_daily_aggregates_v1`
   - `multi_source_daily_v1`
6. Re-run the bounded feature canary batch at the primary `5d` horizon.

### Second Priority: Re-Check The Running Mac Experiment

The active experiment `perpetual-macmini-p1-f547` had one running ensemble run at the last check. Next session should refresh:

```bash
cd /Users/zach/Documents/Git/TradeML
sshpass -e ssh openclaw@100.102.98.14 \
  'cd /Users/openclaw/TradeML && .venv/bin/python -m trademl.cli research \
  --data-root /Users/openclaw/atlas_mounts/nas \
  --local-state /Users/openclaw/TradeML/control \
  --env-file /Users/openclaw/TradeML/.env \
  status --program-id perpetual-macmini'
```

Set `SSHPASS` in the local shell instead of embedding secrets in commands.

### Third Priority: Let The System Run After Source Fix

Once Mac can see source-backed news/minute coverage:

1. Build feature versions.
2. Run bounded feature canaries.
3. Confirm leaderboard writes:
   - `control/cluster/state/research/feature_family_leaderboard/latest.json`
4. Confirm no false degraded dashboard state.
5. Resume/confirm exactly one Mac LaunchAgent-owned perpetual supervisor.
6. Let the fleet run for a real soak window.

## Useful Commands

Run fleet observability from local repo:

```bash
cd /Users/zach/Documents/Git/TradeML
.venv/bin/python -m trademl.cli fleet observability \
  --data-root /path/to/nas \
  --pi 100.76.4.69 \
  --mac 100.102.98.14 \
  --json
```

Run/watch fleet watchdog:

```bash
cd /Users/zach/Documents/Git/TradeML
.venv/bin/python -m trademl.cli fleet watchdog \
  --data-root /path/to/nas \
  --pi 100.76.4.69 \
  --mac 100.102.98.14 \
  --heal
```

Run Mac research status remotely:

```bash
sshpass -e ssh openclaw@100.102.98.14 \
  'cd /Users/openclaw/TradeML && .venv/bin/python -m trademl.cli research \
  --data-root /Users/openclaw/atlas_mounts/nas \
  --local-state /Users/openclaw/TradeML/control \
  --env-file /Users/openclaw/TradeML/.env \
  status --program-id perpetual-macmini'
```

Check Pi service remotely:

```bash
sshpass -e ssh zach@100.76.4.69 \
  'systemctl --user status trademl-node.service --no-pager'
```

Inspect Pi logs:

```bash
sshpass -e ssh zach@100.76.4.69 \
  'journalctl --user -u trademl-node.service -n 200 --no-pager'
```

## Do Not Do

- Do not submit live broker orders.
- Do not use Alpaca live trading endpoints.
- Do not promote a model to live trading.
- Do not treat missing incumbent as degraded.
- Do not treat immature paper labels/PnL as degraded.
- Do not put SQLite on NAS.
- Do not let raw news/minute/text enter training until curated PIT feature artifacts pass readiness and leakage checks.

## Bottom Line

The system is running and finding non-trivial signal, but no model is promotable yet. The most valuable next move is not loosening gates; it is fixing the Mac-visible data contract for Pi-collected `ticker_news` and `equities_minute`, then rerunning the new feature-version canaries. That is the path most likely to turn the current unstable IC signal into something robust enough to pass promotion gates.
