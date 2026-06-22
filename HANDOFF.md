# TradeML Handoff

Last updated: 2026-06-22  
Repo: `/Users/zach/Documents/Git/TradeML`  
Branch: `main`  
Current pause point: direction docs plus the grouped SEC/event and NAS-root
checkpoints in the latest local `main` history.

## Current Pause Point - 2026-06-22

TradeML is ready to resume from a committed handoff state once this docs
checkpoint is pushed. The runtime direction is frozen: do not restart the
Raspberry Pi collector, Mac mini research loop, or any autonomous research
supervisor until `docs/research/PATH_FORWARD_SSOT.md` has an accepted next
offline gate.

Grouped checkpoint commits immediately before this handoff:

- `92baa50 Add SEC public event research pipeline`
- `8b2d88a Point TradeML nodes at dev NAS root`

This final handoff/direction commit records the active research direction and
the review packet needed to consolidate the docs into one final SSOT.

### What Changed

- Added source-first SEC public-event infrastructure for Form 4 and 8-K work
  under `src/trademl/events/`.
- Added CLI commands and tests for Form 4 ingestion/candidate/label/study
  workflows and SEC 8-K deterministic, semantic, coverage, and decision gates.
- Expanded SEC EDGAR connector/capability/provider-contract support for event
  datasets.
- Pointed Mac/Pi/default node paths at the canonical NAS `dev` share:
  local Mac `/Volumes/dev/TradeML`, Pi `/mnt/dev/TradeML`, Mac mini
  `/Users/openclaw/atlas_mounts/dev/TradeML`.
- Added operational consolidation scripts for TradeML storage and SEC parallel
  resume.
- Added `docs/research/PATH_FORWARD_SSOT.md`, source prompt archive, and the
  direction consolidation packet.

### Active Direction

Current alpha direction is not the old broad daily ranker or always-on
autopilot loop. The active shape is:

> Shared PIT infrastructure plus narrow, pre-registered, source-first
> public-event hypotheses with strict evidence, controls, and kill criteria.

Form 4 naive insider-buy work is killed as an alpha path. Broad deterministic
8-K item-family work is killed as an alpha path. Semantic 8-K extraction is
mechanically viable but currently too sparse to justify paper/live or more
autonomous runtime spend.

### Next Gate

Pick exactly one narrow public-event hypothesis and run an offline source
inventory/labelability/frequency gate before any LLM classification or restart.
The gate must show a plausible path to at least 100 labeled primary events after
eligibility, liquidity, and first-tradable-time filters.

Paper/live remains blocked.

### Validation

Run on 2026-06-22:

```bash
python -m pytest tests/unit/test_form4.py tests/unit/test_sec8k.py tests/unit/test_sec8k_coverage.py tests/unit/test_sec8k_semantic.py tests/unit/test_semantic_classifier.py tests/unit/test_connectors.py tests/unit/test_cli.py tests/unit/test_capabilities.py tests/unit/test_provider_contracts.py tests/integration/test_scripts.py -q
```

Result: `175 passed`.

This is a fresh replacement for the previous root `HANDOFF.md`. The prior handoff
was read first; useful older context was preserved here, especially the north
star, Pi/Mac architecture, weak-rejection diagnostics, source-contract warnings,
and operational commands. Anything below marked "current" was checked in this
thread on 2026-05-08.

## Critical Context

TradeML has pivoted away from a broad "generic market prediction" or
"world-model trader" idea. The project direction is now:

> Build a source-first, public-information event research system that is hard to
> fool: point-in-time data, append-only source records, auditable semantic event
> extraction, realistic labels/costs, controls/placebos, execution logging, and
> hard kill criteria.

The system is still research-only. There are no live orders, no live model
promotion, and no autonomous capital allocation. Paper/shadow artifacts are for
operational validation and diagnostics only unless a future explicit promotion
plan changes that.

## Non-Negotiable User Preference

Zach strongly objected to regex/keyword rules for economic semantic extraction.
This is now a project rule in `AGENTS.md` and `CLAUDE.md`:

- Deterministic parsing is allowed for stable source structure only:
  SEC accession metadata, SGML/XML tags, item-section boundaries, timestamps,
  paths, hashes, filenames, archive CIKs, etc.
- Narrative/economic meaning must be classified or extracted by an LLM or
  purpose-built model into a strict schema, then validated deterministically
  against exact evidence quotes/spans/source offsets.
- Do not use regex or keyword heuristics for sentiment, materiality, dilution,
  contract awards, auditor trouble, customer loss, covenant/default stress,
  guidance changes, or any other economic event meaning.
- Do not ask an LLM for numeric confidence. Use categorical outputs, evidence
  requirements, schema validation, and deterministic blocking/warning rules.

If a future task involves public-event/news/filing semantics, use the local
model path by default, not regex.

## Canonical Storage Decision

Zach wants all projects under the NAS `dev` folder. Treat this as canonical:

- Canonical TradeML NAS path on this Mac: `/Volumes/dev/TradeML`
- Canonical Mac mini path: `/Users/openclaw/atlas_mounts/dev/TradeML`
- Canonical Pi path: `/mnt/dev/TradeML`
- The old `/Volumes/TradeML` share and local fake-NAS data have been
  consolidated into `/Volumes/dev/TradeML`.
- `/Users/zach/atlas_mounts` was removed entirely per Zach's request; do not
  recreate an `atlas_mounts` symlink on the Mac.

### Why This Matters

There was about 184 GB of SEC raw data under:

`/Users/zach/atlas_mounts/nas/data/raw/sec`

That path is local APFS, not a real NAS mount. It inflated local disk usage and
must not be used as the long-term data root. Future commands should use
`/Volumes/dev/TradeML` locally.

## Current Storage Transfer Status

Current as of 2026-05-11:

- The storage transfer is complete.
- Canonical local data root is `/Volumes/dev/TradeML`.
- `/Users/zach/atlas_mounts` is absent.
- The failed partial SEC tar temp was removed from the NAS archive.
- Storage consolidation logs/states were moved to
  `/Volumes/dev/TradeML/archive/maintenance/storage_consolidation_2026-05-08_09`.
- NAS active root is organized as:
  `archive/`, `control/`, `data/`, `experiments/`, `logs/`, `models/`,
  `reports/`.
- Remaining NAS trash, if any, is outside the active tree under
  `/Volumes/dev/#recycle/TradeML/control/cluster/leases`; it is many tiny
  recycle-bin lock copies and should be cleared from the NAS UI/server side,
  not over SMB.

### Monitor Commands

```bash
/opt/homebrew/bin/tmux ls
cat /Volumes/dev/TradeML/control/maintenance/storage_consolidation_tmux_20260508.state
tail -f /Volumes/dev/TradeML/control/maintenance/storage_consolidation_tmux_20260508.log
ps -axo pid,ppid,stat,etime,command | grep -E '[c]onsolidate_trademl_storage|[r]sync.*TradeML|[r]sync.*atlas_mounts'
```

Attach if needed:

```bash
/opt/homebrew/bin/tmux attach -t trademl_storage
```

Do not casually kill this job. It is the active cleanup path for the old NAS
share and the 184 GB local SEC tree.

### Why tmux Is Used

Attempts to run the copy as `nohup` from Codex were cleaned up when the shell
session ended. Attempts to run it with `launchd` could read `/Volumes/dev` but
failed to write to `/Volumes/dev/TradeML` with `Operation not permitted`. The
tmux session is the working durable approach because it runs in the interactive
user context that can write to the SMB mount.

### Archived Local Junk Already Handled

These local paths were previously archived into
`/Volumes/dev/TradeML/archive/local_mac_storage/2026-05-08/` and removed from
the Mac:

- `/Users/zach/mnt`
- `/Users/zach/trademl` except useful `.env` key names

The old `/Users/zach/trademl/.env` contained remote-Mac related keys already
represented in the repo `.env`; no unique secret values needed to be copied.
Do not print `.env` values in chat or logs.

There is/was a failed partial tar attempt:

`/Volumes/dev/TradeML/archive/local_mac_storage/2026-05-08/Users_zach_atlas_mounts_nas_data_raw_sec.tar.tmp`

The active consolidation script excludes/ignores that tmp artifact. The
important source is the actual local tree at
`/Users/zach/atlas_mounts/nas/data/raw/sec`.

## Worker Pointer Status

### Mac Mini

Working SSH command:

```bash
ssh -i /Users/zach/.ssh/trademl_ed25519 openclaw@openclaws-mac-mini.local ...
```

Aliases like `trademl-mac` / `mac-mini-trademl` may be stale. Use the explicit
command above if in doubt.

Current verified Mac mini state:

- Repo: `/Users/openclaw/TradeML`
- `.env`:
  - `NAS_MOUNT=/Users/openclaw/atlas_mounts/dev/TradeML`
  - `NAS_SHARE=//192.168.68.54/dev`
- `configs/node.yml`:
  - `node.nas_mount: /Users/openclaw/atlas_mounts/dev/TradeML`
  - `node.nas_share: //192.168.68.54/dev`
  - `training_targets.workstation-remote.data_root: /Users/openclaw/atlas_mounts/dev/TradeML`
- Mounted share:
  - `//kuna@192.168.68.54/dev on /Users/openclaw/atlas_mounts/dev`
- `ops/macos_mount_trademl_nas.sh` and the LaunchAgent now mount SMB share
  `dev` at `/Users/openclaw/atlas_mounts/dev` and verify
  `TradeML/data/raw/ticker_news` plus `TradeML/data/raw/equities_minute`.
- Active research state was migrated on 2026-05-12 from the old local Mac mini
  root `/Users/openclaw/atlas_mounts/nas` to
  `/Users/openclaw/atlas_mounts/dev/TradeML`.
- Migration backup/report:
  `/Users/openclaw/TradeML/control/maintenance/nas_home_migration_20260512T133554Z/`.
- Full old-root archive/removal job was started on 2026-05-12 after Zach
  correctly pushed back on leaving the Mac mini disk bloated. Active job:
  `/Users/openclaw/TradeML/control/maintenance/macmini_old_nas_archive_20260512T141419Z_parallel/`.
  Destination:
  `/Users/openclaw/atlas_mounts/dev/TradeML/archive/macmini_local_old_nas/20260512T141419Z_parallel/`.
  The job splits the old root into tar chunks, writes up to four chunks in
  parallel, verifies each tar by entry count, and only then removes
  `/Users/openclaw/atlas_mounts/nas`.
- At last check the archive/removal job was still `archiving`; the local old
  root still existed. Do not start another archive strategy unless this job has
  failed or been intentionally stopped.
- The active Mac mini research files checked after migration had zero live
  references to `/Users/openclaw/atlas_mounts/nas`:
  `program_state.json`, `experiment_supervisors/perpetual-macmini-p1-f1160.json`,
  and the `perpetual-macmini-p1-f1160` run manifests.
- The LaunchAgent plist is path-correct but intentionally unloaded for now:
  launchd's background session wedged in Python `open()` while creating/appending
  the NAS progression-event JSONL. The same NAS path is writable from the
  interactive SSH user session.
- Current live Mac mini research runner is detached from the SSH user session,
  not launchd:
  PID `76043`, command `python -m trademl.cli research --data-root
  /Users/openclaw/atlas_mounts/dev/TradeML ... start --program
  configs/research/perpetual_macmini.yml --poll-seconds 60`.
- Recovered experiment supervisor:
  `perpetual-macmini-p1-f1160`, PID `76049`, reached `COMPLETED` after draining
  the remaining planned runs.
- Recovered runs `10f4ae761e`, `1a0dd818ef`, and `2ee070c1e9` all failed
  quickly because the canonical NAS root is missing
  `data/curated/modeling/features/news_event_aggregates_v1`; this is a feature
  readiness/artifact issue, not an old-path issue.

### Raspberry Pi

Working SSH alias:

```bash
ssh trademl-pi ...
```

Current verified Pi state:

- Hostname: `servpi`
- New mount:
  - `//192.168.68.54/dev` mounted at `/mnt/dev`
- Canonical TradeML data root on Pi:
  - `/mnt/dev/TradeML`
- `/etc/fstab` has the `//192.168.68.54/dev /mnt/dev cifs ...` entry.
- User service:
  - `systemctl --user is-active trademl-node.service` -> `active`
- Runtime process:
  - `/home/zach/TradeML/.venv/bin/python -m trademl.data_node --config /home/zach/trademl-node/node.yml --root /home/zach/trademl-node --env-file /home/zach/trademl-node/.env`

Important root cause fixed:

The Pi service materializes `/home/zach/trademl-node/.env` and
`/home/zach/trademl-node/node.yml` from the shared NAS cluster manifest on
startup. We initially edited the runtime files directly, but the service rewrote
them back to the old share because the manifest still had old values.

Fixed manifest:

`/Volumes/dev/TradeML/control/cluster/manifest.yml`

Now contains:

- `nas_share: //192.168.68.54/dev`
- `nas_mount: /mnt/dev/TradeML`
- Mac training data root:
  `/Users/openclaw/atlas_mounts/dev/TradeML`

After updating the manifest and restarting `trademl-node.service`, the Pi files
stayed correct:

- `/home/zach/trademl-node/.env`
  - `NAS_MOUNT=/mnt/dev/TradeML`
  - `NAS_SHARE=//192.168.68.54/dev`
- `/home/zach/trademl-node/node.yml`
  - `nas_mount: /mnt/dev/TradeML`
  - `nas_share: //192.168.68.54/dev`
  - `data_root: /Users/openclaw/atlas_mounts/dev/TradeML`

Check command:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=5 trademl-pi '
  systemctl --user is-active trademl-node.service
  findmnt -no SOURCE,TARGET,FSTYPE /mnt/dev
  grep -nE "^(NAS_MOUNT|NAS_SHARE)=" /home/zach/trademl-node/.env /home/zach/TradeML/.env
  grep -nE "nas_mount:|nas_share:|data_root:" /home/zach/trademl-node/node.yml /home/zach/trademl-node/configs/node.yml /home/zach/TradeML/configs/node.yml
'
```

## Repo Code/Config Changes Made For Storage

Files touched for the canonical dev path:

- `.env.template`
- `configs/node.yml`
- `SSOT.md`
- `README.md`
- `docs/AUTOPILOT_ROADMAP.md`
- `docs/research/session_handoff_2026-05-04.md`
- `src/scripts/pi_data_node_wizard.py`
- `src/trademl/data_node/__main__.py`
- `src/trademl/dashboard/controller.py`
- `src/trademl/fleet/autopilot.py`
- `ops/macos_mount_trademl_nas.sh`
- `ops/install_macos_nas_launchagent.sh`
- `tests/integration/test_scripts.py`

New operational helper:

- `ops/consolidate_trademl_storage.sh`

Validation run after these changes:

```bash
bash -n ops/consolidate_trademl_storage.sh
git diff --check
ruff check src/scripts/pi_data_node_wizard.py src/trademl/data_node/__main__.py src/trademl/dashboard/controller.py src/trademl/fleet/autopilot.py tests/integration/test_scripts.py
PYTHONPATH=src pytest tests/integration/test_scripts.py::test_pi_wizard_initializes_state tests/unit/test_dashboard.py::test_persist_node_settings_updates_config_env_stage_and_fstab -q
```

Results:

- `ruff`: passed
- `git diff --check`: passed
- focused pytest: `2 passed`

Full unit tests were not rerun after the storage changes because the active task
was path migration and cleanup, not a full code release.

## Dirty Worktree Warning

The worktree is intentionally dirty and contains a lot of prior Form 4 / 8-K
implementation work plus the storage migration changes. Do not revert unrelated
changes.

At last check, `git status --short` included:

- Modified:
  - `.env.template`
  - `AGENTS.md`
  - `CLAUDE.md`
  - `SSOT.md`
  - `configs/node.yml`
  - `docs/AUTOPILOT_ROADMAP.md`
  - `docs/research/autonomous_research_architecture_policy.md`
  - `docs/research/session_handoff_2026-05-04.md`
  - `ops/install_macos_nas_launchagent.sh`
  - `ops/macos_mount_trademl_nas.sh`
  - `src/scripts/pi_data_node_wizard.py`
  - `src/trademl/cli.py`
  - `src/trademl/connectors/sec_edgar.py`
  - `src/trademl/dashboard/controller.py`
  - `src/trademl/data_node/__main__.py`
  - `src/trademl/data_node/capabilities.py`
  - `src/trademl/data_node/provider_contracts.py`
  - `src/trademl/fleet/autopilot.py`
  - `tests/integration/test_scripts.py`
  - `tests/unit/test_capabilities.py`
  - `tests/unit/test_cli.py`
  - `tests/unit/test_connectors.py`
  - `tests/unit/test_provider_contracts.py`
- Untracked:
  - `HANDOFF.md`
  - `docs/research/form4_insider_purchase_event_mvp_direction.md`
  - `docs/research/public_event_research_pivot_ssot.md`
  - `docs/research/sec_8k_item_event_mvp_direction.md`
  - `docs/research/source_prompts/`
  - `ops/consolidate_trademl_storage.sh`
  - `src/trademl/events/`
  - `tests/unit/test_form4.py`
  - `tests/unit/test_sec8k.py`
  - `tests/unit/test_sec8k_coverage.py`
  - `tests/unit/test_sec8k_semantic.py`
  - `tests/unit/test_semantic_classifier.py`

If asked to commit, stage intentionally by path. Do not blindly `git add .`
unless Zach explicitly asks for all pending changes.

## Prompt / Direction Documents

The following repo docs were created or updated so future sessions do not lose
the pivot context:

- `docs/research/public_event_research_pivot_ssot.md`
  - Source-of-truth pivot doc for public-event research foundry direction.
  - Points to the Form 4 MVP and later 8-K semantic work.
- `docs/research/form4_insider_purchase_event_mvp_direction.md`
  - Form 4 event MVP details, current outcome, and why it is not a live/paper
    signal.
- `docs/research/sec_8k_item_event_mvp_direction.md`
  - 8-K item/header-only baseline direction and later semantic pivot notes.
- `docs/research/source_prompts/`
  - Copies of the original prompts/responses from GPT Pro/GPT 5.5/user context.
  - This was created because Zach was worried important source-feed details were
    missed on the first pass.

The "source-first feeds" to preserve long-term include:

- SEC EDGAR submissions, filings, company facts, RSS
- Company IR RSS / press-release pages
- FTC RSS and relevant endpoints
- FDA RSS and openFDA
- ClinicalTrials.gov API
- SAM.gov opportunities API
- USAspending API
- Federal Register API
- CourtListener and RECAP where usable
- USPTO / PatentsView
- Official agency/company X accounts only if API access and terms allow
- General news APIs only as supplementary awareness/contamination layer

## Public-Event Pivot Summary

GPT Pro/GPT 5.5 argued that the previous broad "institutional AI lab" direction
was too broad to falsify. The accepted pivot:

- Do not try to beat market makers on SPY/QQQ/NVDA/AAPL/liquid macro headlines.
- Do not build a generic "predict tomorrow's price" model.
- Build a boring machine that prevents self-deception.
- Focus on narrow, low-capacity public-information edges where the data is
  annoying, fragmented, entity-linkage-heavy, document-heavy, or delayed in
  market awareness.
- Use simple equities first. Options come later only after event infrastructure
  and execution modeling are proven.
- Keep the generic ranker as a sentinel/comparator, not the entry engine.

Research tracks, in rough order:

1. Public-event drift in undercovered listed equities.
2. PEAD/revisions in undercovered names.
3. Earnings options IV mispricing later, with defined-risk structures only.

Core infrastructure goals:

- append-only raw records
- source hash
- source/vendor/ingestion/first-seen timestamps
- parser/schema versions
- provider capability metadata
- entity resolution
- normalized event store
- point-in-time research datasets
- boring backtester with costs, controls, and outlier tests
- experiment registry tracking every attempt
- execution logger before live trading
- deterministic risk gates

## Form 4 Work

### Why Form 4 Was Chosen

GPT Pro recommended Form 4 open-market insider purchases as the first MVP
because it is structured, public, timestamped, long-only, cheap to parse, easy to
control, and tests the whole event-research stack without LLM hallucination or
options-fill fantasy.

Strict event definition:

- SEC Form 4
- non-derivative transaction
- transaction code `P`
- acquired/disposed `A`
- common stock / ordinary shares only
- positive shares
- positive price
- issuer maps to listed common equity
- exclude OTC, derivatives, units, warrants, preferred, private placement/SPAC
  sponsor language, amendments from the primary sample, mixed P/S in strict
  bucket, late reports where relevant

### Form 4 Retrieval/Parser Foundation

Implemented direction:

- SEC full-index/daily-index manifest discovery for `4` and `4/A`
- Keep `archive_cik`, `issuer_cik`, and owner CIKs separate
- Build archive URLs from index filename/archive CIK, not issuer CIK or accession
  prefix
- Raw primary XML fetch with complete `.txt` SGML extraction fallback
- Accepted timestamp precedence:
  submissions JSON -> SGML header -> accession index
- Namespace-agnostic XML parser
- Decimal shares/prices/values
- Field-level footnote references
- Multiple reporting owners as sorted owner set
- `4/A` parsed but excluded from primary signal
- Quality flags for archive/issuer mismatch, amendments, mixed P/S, derivative P,
  private/unit/SPAC/PIPE/warrant language, missing/zero price, non-common title,
  OTC/late reports, supporting docs, etc.

Fixture gate covered the weird cases GPT Pro listed:

- Amazon / Indra Nooyi `0001127602-20-013168`
- Sinclair / David Smith `0001250842-25-000026`
- Tiptree / Goldman `0000769993-15-000534`
- Immediatek / Mark Cuban `0001209191-06-060213`
- Super Micro / Sara Liu `0001758554-19-000046`
- Archimedes SPAC `0001437749-25-003569`
- Bioject / Logomasini `0000810084-13-000003`
- Eledon private-placement style `0001593968-24-000563`
- Ares sales-only `0001025978-25-000011`
- very late filing `0001528597-26-000004`

### Form 4 Outcome

The Form 4 baseline/rework was not promotable. Treat Form 4 as:

- useful ingestion/parser/label/control infrastructure
- useful fixture gate and source-quality benchmark
- not a paper/live signal
- not worth continued alpha search unless a new documented hypothesis reopens it

The intended status artifact/verdict language:

- `KILL_OR_REWORK` for the naive baseline
- `FORM4_KILLED_BASELINE_COMPLETE` if bounded rework gates fail

The bounded rework variants implemented/planned:

- `clean_timely_unmixed`
- `officer_director_only`
- `ceo_cfo_only`
- `clustered_buyers`
- `large_purchase_q4`

Continue Form 4 only if a variant has enough sample and passes mean/median/control
separation gates. Otherwise leave it killed.

## SEC 8-K Work

### Header/Item-Only Baseline

After Form 4, the next MVP shifted to SEC 8-K item/exhibit events.

Deterministic parsing is allowed for structure only:

- SGML blocks
- accession metadata
- accepted timestamps
- item headings
- filenames
- hashes
- exhibit inventory

Initial item families:

- `1.01` material agreement
- `2.02` results/operations, treated as earnings/contamination bucket
- `2.04` default/covenant stress
- `2.05` exit/disposal costs
- `3.02` unregistered sale/dilutive financing
- `4.01` auditor change
- `7.01` Reg FD
- `8.01` other

The deterministic header/item-only 8-K baseline is killed as an alpha candidate.
It remains a negative-control/baseline and structural spine.

### Semantic 8-K MVP

The accepted semantic plan:

- Command: `trademl research sec-event-semantic-classify`
- Read existing SEC 8-K raw `complete.txt` and curated filing/candidate artifacts
- Build bounded snippets from item sections and exhibits
- Classify via LM Studio using model `qwen3.5-9b-mlx`
- Mac mini endpoint through SSH tunnel:
  `http://127.0.0.1:1235/v1`
- Default response mode: `prompt_json`
- Initial batch size: `1` for reliability; earlier plan had `4`, but reliability
  won for targeted canaries
- Resume/checkpoint successful rows
- No numeric confidence
- No regex semantic classification

Validated/promoted semantic event types:

- `AUDITOR_TROUBLE`
- `DEBT_DEFAULT_COVENANT_STRESS`
- `DILUTIVE_FINANCING`
- `CUSTOMER_LOSS`
- `MATERIAL_CONTRACT_AWARD`

Candidate rows must match the existing labeler/event-study contract:

- stable `event_id`
- `issuer_cik`
- `ticker`
- `primary_security_id`
- `accessions`
- `event_type`
- `accepted_at_utc`
- `first_seen_at_utc`
- `tradable_at_utc`
- `eligibility_pass`
- `exclusion_reasons`
- fixed diagnostic `event_strength_score=1.0`

The model path was proven enough to continue: the Mac mini LM Studio route worked
and promoted candidates in canaries. The blocker became labelability/coverage,
not model intelligence.

### Targeted Semantic Event Gate

Target items were expanded because the April 2025 spine initially missed
important structural coverage:

- `2.04` -> default/covenant stress
- `3.02` -> unregistered/dilutive financing

Target routing priority:

1. `4.01` auditor trouble
2. `2.04` default/covenant stress
3. `3.02` dilutive/unregistered financing
4. `1.01` material agreement, lower priority/noisier

Commands added/planned in CLI:

- `sec8k-ingest`
- `sec8k-candidates`
- `sec8k-market-backfill`
- `sec8k-event-study`
- `sec8k-decision`
- `sec8k-coverage-audit`
- `sec8k-coverage-expand`
- `sec-event-semantic-classify`
- `sec-event-semantic-labelability-audit`
- `sec-event-semantic-study`
- `sec-event-semantic-scaled-gate`
- `sec-event-semantic-coverage-gate`

### Labelability-First Gate

The next implementation direction was:

1. Audit SEC 8-K coverage for 2024-2025 target items before spending LLM time.
2. Expand missing SEC coverage month by month.
3. Rebuild global `sec_8k_item_events` from combined filing index, not latest
   ingest slice.
4. Run labelability audit before LLM:
   - ticker present
   - accepted timestamp present
   - entry minute available
   - exit daily available
   - benchmark data available
5. Backfill only market-data blockers for target/date rows.
6. Run semantic classification with `labelable-only`.
7. Run semantic study and decide.

Terminal decisions:

- `CONTINUE_SEMANTIC_8K` if adequate sample passes return, median, placebo
  separation, and top-5 contribution gates.
- `SEMANTIC_8K_KILLED` if adequate labeled semantic sample exists and fails.
- `BLOCKED_DATA_COVERAGE` if enough target rows exist but ticker/market data
  cannot produce enough labels after repair.
- `MORE_DATA_REQUIRED_TRUE_COVERAGE` if full 2024-2025 coverage cannot produce
  enough promotable/labeled semantic events.

Current status:

- Implementation exists in the dirty worktree.
- 2026-05-11 operator decision after storage completed: do not continue treating
  the current SEC 8-K semantic lane as the viability proof. The official
  semantic artifact is still `MORE_DATA_REQUIRED`, not `SEMANTIC_8K_KILLED`,
  because an adequate promoted semantic sample never materialized. For product
  direction, this is a no-go/parked alpha lane unless a new documented
  hypothesis changes the event universe or sample requirements.
- Evidence:
  - Broad deterministic/header 8-K item families were already killed as alpha.
  - Latest semantic study promoted only 6 labeled events and failed
    `insufficient_labeled_events`.
  - Latest labelability artifact found 7,215 targeted deterministic candidates
    but only 336 labelable target-item snippets.
  - The scaled-gate checkpoint had 299 classified snippets, with only 51
    promotable semantic event types; the persisted scaled gate verdict was
    `MORE_DATA_REQUIRED` with `insufficient_semantic_event_frequency`.
- A live rerun of `sec-event-semantic-coverage-gate` against
  `/Volumes/dev/TradeML` on 2026-05-11 was stopped after more than 10 minutes:
  it was spending time in per-file `stat`/directory enumeration over SMB and
  had not written a new state packet. Fix that performance bug before any
  future full NAS gate run.

## LLM vs Embedding Decision

Use an LLM for schema-bound semantic classification/extraction when the output
must include:

- event type
- materiality category
- exact evidence quote
- reason text
- validation-friendly fields
- distinction among economic event classes

Use embeddings for:

- retrieval
- clustering
- deduplication
- candidate shortlist
- similarity search

Embeddings alone are not enough for the initial event schema because they do not
produce auditable event objects with exact evidence and controlled labels.

## Research Autopilot / Weak Rejection Diagnostics

The previous research lane remains relevant but is no longer the primary alpha
direction while event-research infrastructure is being built.

Important committed work at HEAD `458eb09`:

- Added `weak_rejection_plateau` diagnostic mode.
- Added configurable weak plateau policy in
  `configs/research/perpetual_macmini.yml`.
- Planner launches bounded diagnostic families before generic frontier churn when
  weak rejection signatures repeat.
- Rejection counts aggregate across frontier history.
- Diagnostic metadata persists through specs/manifests/summaries/program
  state/research health/review packets/progression audit.
- Research audit writes `diagnostic_state` and only surfaces Codex issues if the
  plateau is not being handled.

Validation from that work:

```bash
python -m pytest tests/unit/test_research.py tests/unit/test_research_audit.py tests/unit/test_experiments.py -q
ruff check src tests
```

Result at the time:

- `128 passed`
- ruff passed

Older live research facts from the prior handoff may now be stale:

- Mac mini research was running `perpetual-macmini`
- A high-IC advanced candidate around `0.077` existed but failed strict gates
- Rejections were dominated by fragility/stability failures
- Missing incumbent and immature paper labels should not be considered degraded

If this matters, verify current Mac state before acting:

```bash
ssh -i /Users/zach/.ssh/trademl_ed25519 openclaw@openclaws-mac-mini.local '
  cd /Users/openclaw/TradeML &&
  .venv/bin/python -m trademl.cli research \
    --data-root /Users/openclaw/atlas_mounts/dev/TradeML \
    --local-state /Users/openclaw/TradeML/control \
    --env-file /Users/openclaw/TradeML/.env \
    health --program-id perpetual-macmini
'
```

## Fleet / Source Contract Caveat

This was revisited on 2026-05-11 after the NAS migration.

Current source-contract status:

- Fixed the persisted feature source contract and source availability artifacts
  under `/Volumes/dev/TradeML/control/cluster/state/...`; the active artifacts
  no longer point at `/mnt/trademl`, `/Volumes/TradeML`, or
  `/Users/zach/atlas_mounts`.
- Latest fleet data-quality audit against `/Volumes/dev/TradeML` found 7 OK
  rows, 2 warnings, and 0 critical failures.
- OK: `equities_minute`, `ticker_news`, `stock_trades`, `stock_quotes`,
  `sec_filings`, `equities_ohlcv_adj`, `macros_fred`.
- Still missing/warning: `sec_companyfacts` and `fundamentals_daily` at their
  canonical paths under `/Volumes/dev/TradeML/data/reference/`.
- The old local audit claim that the Pi was offline was a false local-snapshot
  result. When rerun with `--pi-host trademl-pi`, the Pi resolved as online and
  the stale `node_offline` issue disappeared.

Mac mini research status:

- Active state migration is complete as of 2026-05-12.
- Backup/report:
  `/Users/openclaw/TradeML/control/maintenance/nas_home_migration_20260512T133554Z/`.
- The migration copied the active/referenced old-root artifacts into
  `/Users/openclaw/atlas_mounts/dev/TradeML` without overwriting existing files:
  30 unique referenced paths checked, 76 files copied, 30,344,723 bytes copied,
  1,261 existing files skipped, 0 errors.
- The old local Mac mini root still exists at
  `/Users/openclaw/atlas_mounts/nas` and is about 32 GB, but it now has a live
  archive-then-remove job. The job will remove the local root after all NAS tar
  chunks verify.
- Earlier broad full-root `rsync` and single-tar attempts were stopped because
  SMB metadata/small-write behavior was too slow. The active job is the parallel
  tar-chunk job under
  `/Users/openclaw/TradeML/control/maintenance/macmini_old_nas_archive_20260512T141419Z_parallel/`.
- LaunchAgent is not the current runner. It is path-correct but unsafe right
  now because the launchd background process wedged in kernel `open()` on the
  NAS progression JSONL. Keep it unloaded unless you are explicitly fixing the
  launchd/SMB write mechanism.
- The current runner was started with `research start --detach` from the SSH
  user session. At last check:
  - research supervisor PID `76043`, status `INFRA_BLOCKED`
  - wait reason: missing modeling artifact root
    `/Users/openclaw/atlas_mounts/dev/TradeML/data/curated/modeling/features/news_event_aggregates_v1`
  - experiment supervisor `perpetual-macmini-p1-f1160`, PID `76049`, status
    `COMPLETED`
  - recovered runs `10f4ae761e`, `1a0dd818ef`, and `2ee070c1e9` failed on
    missing
    `data/curated/modeling/features/news_event_aggregates_v1`
  - launchd status: service not loaded

Older audits showed a mismatch between direct worker truth and local audit truth:

- Pi was collecting raw data.
- Mac mini could see some NAS data.
- Local Mac audit/source-contract paths sometimes looked at the wrong or
  incomplete root and produced `source_unavailable` warnings.

After the storage migration, this should be revisited. The source-contract/audit
layer should prove:

Pi collection -> canonical dev NAS -> Mac feature factory -> event canaries.

Expected canonical source paths after migration:

- `/Volumes/dev/TradeML/data/raw/ticker_news`
- `/Volumes/dev/TradeML/data/raw/equities_minute`
- `/Volumes/dev/TradeML/data/raw/sec`
- `/Volumes/dev/TradeML/data/curated/equities_ohlcv_adj`
- `/Volumes/dev/TradeML/data/reference/sec_companyfacts`
- `/Volumes/dev/TradeML/control/cluster/state/...`

Known free-plan unavailable/zero-coverage sources should be informational, not
degraded warnings.

## Useful Commands

### Check Storage / NAS Shape

```bash
df -h /System/Volumes/Data /Volumes/dev
test ! -e /Users/zach/atlas_mounts && echo atlas_mounts_absent
ls -la /Volumes/dev/TradeML
```

### Check Mac Mini Mount / Config

```bash
ssh -i /Users/zach/.ssh/trademl_ed25519 openclaw@openclaws-mac-mini.local '
  cd /Users/openclaw/TradeML
  grep -nE "^(NAS_MOUNT|NAS_SHARE)=" .env
  grep -nE "nas_mount:|nas_share:|data_root:" configs/node.yml
  mount | grep -E "atlas_mounts/dev"
'
```

### Check Mac Mini Research Runner

Use the explicit SSH key. Do not infer Mac mini state from local Mac files.

```bash
ssh -i /Users/zach/.ssh/trademl_ed25519 \
  -o BatchMode=yes -o ConnectTimeout=8 \
  openclaw@openclaws-mac-mini.local '
    cd /Users/openclaw/TradeML
    .venv/bin/python -m trademl.cli research \
      --data-root /Users/openclaw/atlas_mounts/dev/TradeML \
      --local-state /Users/openclaw/TradeML/control \
      --env-file .env \
      status --program-id perpetual-macmini
    ps -axo pid,ppid,stat,etime,%cpu,%mem,command |
      grep -E "[t]rademl.cli research|[t]raining_job.py|[e]xperiment_supervisor" || true
    launchctl print gui/$(id -u)/com.trademl.research.perpetual-macmini 2>&1 |
      sed -n "1,20p"
  '
```

### Check Mac Mini Old-Root Archive Job

Avoid `du` or broad `ls` on the active NAS archive directory while the archive
writers are running; those calls can block in SMB kernel I/O. Use the local
state file instead:

```bash
ssh -i /Users/zach/.ssh/trademl_ed25519 \
  -o BatchMode=yes -o ConnectTimeout=8 \
  openclaw@openclaws-mac-mini.local '
    python3 - <<'"'"'PY'"'"'
import json
from pathlib import Path
state = Path("/Users/openclaw/TradeML/control/maintenance/macmini_old_nas_archive_20260512T141419Z_parallel/state.json")
payload = json.loads(state.read_text())
print({k: payload.get(k) for k in ["status", "heartbeat_at", "completed_at", "exit_code"]})
for name, chunk in payload.get("chunks", {}).items():
    print(name, chunk.get("status"), chunk.get("archive_bytes"), chunk.get("actual_entries"), chunk.get("expected_entries"))
PY
    ps -axo pid,ppid,stat,etime,%cpu,%mem,command |
      grep -E "[r]un_parallel_archive|[t]ar -cf - -C /Users/openclaw/atlas_mounts/nas|[d]d of=.*/20260512T141419Z_parallel|[t]ar -tf .*/20260512T141419Z_parallel" || true
    [ -e /Users/openclaw/atlas_mounts/nas ] && echo old_root_exists || echo old_root_absent
  '
```

### Check Pi Service / Config

```bash
ssh -o BatchMode=yes -o ConnectTimeout=5 trademl-pi '
  systemctl --user is-active trademl-node.service
  findmnt -no SOURCE,TARGET,FSTYPE /mnt/dev
  grep -nE "^(NAS_MOUNT|NAS_SHARE)=" /home/zach/trademl-node/.env /home/zach/TradeML/.env
  grep -nE "nas_mount:|nas_share:|data_root:" /home/zach/trademl-node/node.yml /home/zach/trademl-node/configs/node.yml /home/zach/TradeML/configs/node.yml
'
```

### Semantic 8-K Coverage Gate Status

The canonical local data root is now:

```bash
PYTHONPATH=src python -m trademl.cli research \
  --env-file .env \
  --data-root /Volumes/dev/TradeML \
  sec-event-semantic-coverage-gate
```

Do not run this again as the default next step. The current SEC 8-K semantic
lane is parked/no-go as the viability proof. Reopen only after changing the
hypothesis or fixing the gate so it uses curated parquet/index artifacts instead
of expensive SMB filesystem enumeration.

If manually running pieces:

```bash
PYTHONPATH=src python -m trademl.cli research --env-file .env --data-root /Volumes/dev/TradeML sec8k-coverage-audit
PYTHONPATH=src python -m trademl.cli research --env-file .env --data-root /Volumes/dev/TradeML sec8k-coverage-expand
PYTHONPATH=src python -m trademl.cli research --env-file .env --data-root /Volumes/dev/TradeML sec-event-semantic-labelability-audit
PYTHONPATH=src python -m trademl.cli research --env-file .env --data-root /Volumes/dev/TradeML sec-event-semantic-scaled-gate
```

For Mac mini LM Studio:

```bash
ssh -f -N -L 127.0.0.1:1235:127.0.0.1:1235 \
  -i /Users/zach/.ssh/trademl_ed25519 \
  openclaw@openclaws-mac-mini.local
```

Default classifier runtime:

- model: `qwen3.5-9b-mlx`
- base URL: `http://127.0.0.1:1235/v1`
- response mode: `prompt_json`
- batch size: `1`
- `--resume`

## Next Steps

Do these in order:

1. Resolve the current Mac mini research `INFRA_BLOCKED` state. The blocker is
   missing modeling artifact root
   `/Users/openclaw/atlas_mounts/dev/TradeML/data/curated/modeling/features/news_event_aggregates_v1`.
   The expected active root is `/Users/openclaw/atlas_mounts/dev/TradeML`;
   launchd should remain unloaded unless the SMB background-session issue is
   being fixed.
2. Rebuild/refresh the missing NAS-backed feature artifacts, especially
   `data/curated/modeling/features/news_event_aggregates_v1`, before retrying
   any recovered family members that require news-event aggregate features.
3. Check the active old-root archive job. When it reaches `complete`, confirm
   `/Users/openclaw/atlas_mounts/nas` is absent and the Mac mini free space has
   increased. If it fails, inspect the chunk status and `archive.log` in
   `/Users/openclaw/TradeML/control/maintenance/macmini_old_nas_archive_20260512T141419Z_parallel/`.
4. Rebuild/refresh the NAS-backed modeling registry and feature-readiness
   artifacts from `/Volumes/dev/TradeML` / `/Users/openclaw/atlas_mounts/dev/TradeML`
   so `ticker_news` and `equities_minute` are visible to the feature factory.
5. Decide the next public-event research hypothesis outside the current SEC 8-K
   lane, or explicitly reopen SEC 8-K with a new sample universe and a fixed
   NAS-safe gate.
6. If reopening SEC 8-K, first fix `sec-event-semantic-coverage-gate` so it does
   not scan/stat market-data paths row-by-row over SMB.

Do not resume Form 4 alpha work unless a new documented hypothesis reopens it.
Do not run broad news/filing semantic extraction with regex. Do not promote any
paper/live behavior from Form 4 or 8-K without a separate explicit promotion
plan.
