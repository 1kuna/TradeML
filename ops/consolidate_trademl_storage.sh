#!/usr/bin/env bash

# Consolidate old TradeML NAS/share artifacts into the canonical dev/TradeML tree.
# This is an operational maintenance helper: it logs each copy step and only
# removes the local fake-NAS tree after every local copy step succeeds.

OLD_ROOT="${OLD_ROOT:-/Volumes/TradeML}"
CANONICAL_ROOT="${CANONICAL_ROOT:-/Volumes/dev/TradeML}"
LOCAL_FAKE_NAS="${LOCAL_FAKE_NAS:-/Users/zach/atlas_mounts/nas}"
MAINTENANCE_DIR="${MAINTENANCE_DIR:-$CANONICAL_ROOT/control/maintenance}"
LOG_FILE="${LOG_FILE:-$MAINTENANCE_DIR/storage_consolidation_$(date -u +%Y%m%dT%H%M%SZ).log}"
STATE_FILE="${STATE_FILE:-$MAINTENANCE_DIR/storage_consolidation.state}"

mkdir -p "$MAINTENANCE_DIR"
exec >>"$LOG_FILE" 2>&1

log_step() {
  printf "%s %s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

write_state() {
  printf "%s\n" "$*" >"$STATE_FILE"
  log_step "$*"
}

run_rsync_dir() {
  local src="$1"
  local dst="$2"
  local label="$3"
  if [[ ! -d "$src" ]]; then
    log_step "skip_missing label=$label src=$src"
    return 0
  fi
  write_state "step=$label"
  mkdir -p "$dst"
  rsync -a --exclude=".DS_Store" "$src"/ "$dst"/
  local rc=$?
  log_step "rsync_exit label=$label rc=$rc"
  return "$rc"
}

run_rsync_file() {
  local src="$1"
  local dst="$2"
  local label="$3"
  if [[ ! -f "$src" ]]; then
    log_step "skip_missing label=$label src=$src"
    return 0
  fi
  write_state "step=$label"
  mkdir -p "$dst"
  rsync -a --exclude=".DS_Store" "$src" "$dst"/
  local rc=$?
  log_step "rsync_exit label=$label rc=$rc"
  return "$rc"
}

write_state "step=start"
mkdir -p "$CANONICAL_ROOT/control" "$CANONICAL_ROOT/data"

# Copy stable old-share artifacts without walking live lease/trash trees.
run_rsync_file "$OLD_ROOT/control/node.sqlite" "$CANONICAL_ROOT/control" old_control_node_sqlite || true
run_rsync_dir "$OLD_ROOT/control/cluster/workers" "$CANONICAL_ROOT/control/cluster/workers" old_control_cluster_workers || true
run_rsync_dir "$OLD_ROOT/control/cluster/events" "$CANONICAL_ROOT/control/cluster/events" old_control_cluster_events || true
run_rsync_dir "$OLD_ROOT/control/cluster/shards" "$CANONICAL_ROOT/control/cluster/shards" old_control_cluster_shards || true

old_data_dirs=(
  curated/equities_ohlcv_adj
  qc
  raw/alpaca_market_events
  raw/alpaca_snapshots
  raw/crypto_bars
  raw/equities_bars
  raw/equities_minute
  raw/macros_fred
  raw/ticker_news
  reference/sec_companyfacts
)
for rel in "${old_data_dirs[@]}"; do
  run_rsync_dir "$OLD_ROOT/data/$rel" "$CANONICAL_ROOT/data/$rel" "old_data_${rel//\//_}" || true
done

for rel in experiments logs models reports; do
  run_rsync_dir "$OLD_ROOT/$rel" "$CANONICAL_ROOT/$rel" "old_misc_$rel" || true
done

# Copy local fake-NAS artifacts. If any of these fail, keep the local tree.
local_ok=1
local_dirs=(
  control/cluster/state
  reports/research
  data/curated
  data/reference
  data/raw/equities_eod
  data/raw/equities_minute
  data/raw/sec
)
for rel in "${local_dirs[@]}"; do
  run_rsync_dir "$LOCAL_FAKE_NAS/$rel" "$CANONICAL_ROOT/$rel" "local_${rel//\//_}" || local_ok=0
done

if [[ "$local_ok" == "1" && -d "$LOCAL_FAKE_NAS" && ! -L "$LOCAL_FAKE_NAS" ]]; then
  write_state "step=replace_local_fake_nas_with_symlink"
  rm -rf "$LOCAL_FAKE_NAS"
  mkdir -p "$(dirname "$LOCAL_FAKE_NAS")"
  ln -s "$CANONICAL_ROOT" "$LOCAL_FAKE_NAS"
else
  log_step "local_fake_nas_not_removed local_ok=$local_ok"
fi

write_state "step=verify"
verify_paths=(
  "$CANONICAL_ROOT/data/raw/ticker_news"
  "$CANONICAL_ROOT/data/raw/equities_minute"
  "$CANONICAL_ROOT/data/raw/sec"
  "$CANONICAL_ROOT/control/cluster/state/last_success.json"
  "$CANONICAL_ROOT/control/node.sqlite"
  "$LOCAL_FAKE_NAS"
)
for path in "${verify_paths[@]}"; do
  if [[ -e "$path" ]]; then
    log_step "present $path"
  else
    log_step "missing $path"
  fi
done

write_state "step=complete"
