#!/usr/bin/env bash

# Resume the local fake-NAS SEC copy with parallel archive_cik shards.
#
# The original maintenance script copies data/raw/sec as one huge rsync tree.
# Over SMB that can spend hours on per-file metadata round trips. This helper
# keeps the same source/destination contract, but splits the independent
# archive_cik directories across multiple rsync workers. Already-matching files
# are skipped by rsync's normal size+mtime check, so this is safe to run after a
# partial single-worker copy.

set -u

CANONICAL_ROOT="${CANONICAL_ROOT:-/Volumes/dev/TradeML}"
LOCAL_FAKE_NAS="${LOCAL_FAKE_NAS:-/Users/zach/atlas_mounts/nas}"
JOBS="${JOBS:-8}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
ARCHIVE_MIN_NAME="${ARCHIVE_MIN_NAME:-}"

SRC_SEC="$LOCAL_FAKE_NAS/data/raw/sec"
DST_SEC="$CANONICAL_ROOT/data/raw/sec"
MAINTENANCE_DIR="${MAINTENANCE_DIR:-$CANONICAL_ROOT/control/maintenance}"
LOG_FILE="${LOG_FILE:-$MAINTENANCE_DIR/storage_consolidation_parallel_${RUN_ID}.log}"
STATE_FILE="${STATE_FILE:-$MAINTENANCE_DIR/storage_consolidation_parallel_${RUN_ID}.state}"
WORK_DIR="${WORK_DIR:-/tmp/trademl_sec_parallel_${RUN_ID}}"

mkdir -p "$MAINTENANCE_DIR" "$WORK_DIR"
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
  mkdir -p "$dst"
  rsync -rt --exclude=".DS_Store" "$src"/ "$dst"/
  local rc=$?
  log_step "rsync_exit label=$label rc=$rc"
  return "$rc"
}

copy_archive_shard() {
  local shard="$1"
  local list_file="$2"
  local count=0
  local failures=0
  local src_dir rel dst_dir rc

  log_step "worker_start shard=$shard list=$list_file"
  while IFS= read -r src_dir; do
    [[ -n "$src_dir" ]] || continue
    rel="${src_dir#"$SRC_SEC"/}"
    dst_dir="$DST_SEC/$rel"
    mkdir -p "$dst_dir"
    rsync -rt --exclude=".DS_Store" "$src_dir"/ "$dst_dir"/
    rc=$?
    count=$((count + 1))
    if [[ "$rc" != "0" ]]; then
      failures=$((failures + 1))
      log_step "worker_rsync_failure shard=$shard rel=$rel rc=$rc"
    elif (( count % 100 == 0 )); then
      log_step "worker_progress shard=$shard dirs=$count failures=$failures last=$rel"
    fi
  done <"$list_file"

  log_step "worker_done shard=$shard dirs=$count failures=$failures"
  if [[ "$failures" == "0" ]]; then
    return 0
  fi
  return 1
}

if [[ "$JOBS" -lt 1 ]]; then
  echo "JOBS must be >= 1" >&2
  exit 2
fi

if [[ ! -d "$SRC_SEC" ]]; then
  echo "missing source SEC tree: $SRC_SEC" >&2
  exit 2
fi

write_state "step=start jobs=$JOBS"
mkdir -p "$DST_SEC"

# Re-copy the small non-archive SEC folders first. These are cheap and make the
# resume helper self-contained after the original script is stopped mid-step.
for src in "$SRC_SEC"/*; do
  [[ -d "$src" ]] || continue
  [[ "$(basename "$src")" == "archives" ]] && continue
  write_state "step=copy_sec_sidecar name=$(basename "$src")"
  run_rsync_dir "$src" "$DST_SEC/$(basename "$src")" "sidecar_$(basename "$src")" || exit 1
done

write_state "step=build_archive_shards min=${ARCHIVE_MIN_NAME:-none}"
if [[ -n "$ARCHIVE_MIN_NAME" ]]; then
  find "$SRC_SEC/archives" -maxdepth 1 -mindepth 1 -type d -name 'archive_cik=*' -print |
    sort |
    while IFS= read -r src_dir; do
      base="$(basename "$src_dir")"
      [[ "$base" < "$ARCHIVE_MIN_NAME" ]] && continue
      printf "%s\n" "$src_dir"
    done >"$WORK_DIR/archive_dirs.all"
else
  find "$SRC_SEC/archives" -maxdepth 1 -mindepth 1 -type d -name 'archive_cik=*' -print | sort >"$WORK_DIR/archive_dirs.all"
fi

total_dirs="$(wc -l <"$WORK_DIR/archive_dirs.all" | tr -d ' ')"
if [[ "$total_dirs" == "0" ]]; then
  echo "no archive_cik directories found under $SRC_SEC/archives" >&2
  exit 2
fi

shard=0
while IFS= read -r src_dir; do
  printf "%s\n" "$src_dir" >>"$WORK_DIR/archive_dirs.$shard"
  shard=$(((shard + 1) % JOBS))
done <"$WORK_DIR/archive_dirs.all"

write_state "step=copy_archives_parallel total_dirs=$total_dirs jobs=$JOBS"
mkdir -p "$DST_SEC/archives"

pids=""
for shard in $(seq 0 $((JOBS - 1))); do
  list_file="$WORK_DIR/archive_dirs.$shard"
  : >"$WORK_DIR/worker.$shard.rc"
  if [[ ! -s "$list_file" ]]; then
    printf "0\n" >"$WORK_DIR/worker.$shard.rc"
    continue
  fi
  (
    copy_archive_shard "$shard" "$list_file"
    printf "%s\n" "$?" >"$WORK_DIR/worker.$shard.rc"
  ) &
  pids="$pids $!"
done

for pid in $pids; do
  wait "$pid"
done

failures=0
for shard in $(seq 0 $((JOBS - 1))); do
  rc="$(cat "$WORK_DIR/worker.$shard.rc" 2>/dev/null || echo 1)"
  if [[ "$rc" != "0" ]]; then
    failures=$((failures + 1))
  fi
done

if [[ "$failures" != "0" ]]; then
  write_state "step=failed worker_failures=$failures"
  exit 1
fi

write_state "step=replace_local_fake_nas_with_symlink"
backup="${LOCAL_FAKE_NAS}.migrated_${RUN_ID}"
if [[ -L "$LOCAL_FAKE_NAS" ]]; then
  log_step "already_symlink path=$LOCAL_FAKE_NAS target=$(readlink "$LOCAL_FAKE_NAS")"
elif [[ -e "$backup" ]]; then
  echo "backup path already exists: $backup" >&2
  exit 1
else
  mv "$LOCAL_FAKE_NAS" "$backup"
  mkdir -p "$(dirname "$LOCAL_FAKE_NAS")"
  ln -s "$CANONICAL_ROOT" "$LOCAL_FAKE_NAS"
  log_step "symlink_created path=$LOCAL_FAKE_NAS target=$CANONICAL_ROOT backup=$backup"
  write_state "step=remove_migrated_local_tree backup=$backup"
  rm -rf "$backup"
fi

write_state "step=verify"
for path in \
  "$DST_SEC/archives" \
  "$DST_SEC/form4_manifest" \
  "$DST_SEC/sec8k_manifest" \
  "$LOCAL_FAKE_NAS"; do
  if [[ -e "$path" ]]; then
    log_step "present $path"
  else
    log_step "missing $path"
    exit 1
  fi
done

write_state "step=complete"
