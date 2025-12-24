#!/usr/bin/env bash
set -euo pipefail

# Reset or edit edge collector bookmarks stored in S3/MinIO
# Uses `mc` (MinIO client) and system Python only (no repo deps).
#
# Usage examples:
#   bash scripts/tools/reset_bookmarks.sh --list
#   bash scripts/tools/reset_bookmarks.sh --vendor alpaca --table equities_bars_minute
#   bash scripts/tools/reset_bookmarks.sh --vendor alpaca            # remove all alpaca:* bookmarks
#   bash scripts/tools/reset_bookmarks.sh --alpaca-all               # remove alpaca day+minute bookmarks
#   bash scripts/tools/reset_bookmarks.sh --massive-day              # remove massive:equities_bars
#   bash scripts/tools/reset_bookmarks.sh --fred-treasury            # remove fred:macro_treasury
#
# Requires:
#   - `mc` configured with alias `minio` (scripts/pi_node.sh does this automatically)

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_DIR"

BUCKET=${S3_BUCKET:-}
if [[ -z "$BUCKET" && -f .env ]]; then
  BUCKET=$(grep -E '^S3_BUCKET=' .env | tail -n1 | cut -d= -f2- || true)
fi
BUCKET=${BUCKET:-ata}

OBJ_KEY="manifests/bookmarks.json"
TMP_JSON="$(mktemp -t bookmarks.XXXXXX.json)"
trap 'rm -f "$TMP_JSON"' EXIT

need_mc() {
  if ! command -v mc >/dev/null 2>&1; then
    echo "[ERROR] 'mc' (MinIO client) not found. Install or run scripts/pi_node.sh ensure_mc_alias first." >&2
    exit 2
  fi
}

download() {
  if mc ls "minio/${BUCKET}/${OBJ_KEY}" >/dev/null 2>&1; then
    mc cp "minio/${BUCKET}/${OBJ_KEY}" "$TMP_JSON" >/dev/null
  else
    echo '{}' > "$TMP_JSON"
  fi
}

upload() {
  mc cp "$TMP_JSON" "minio/${BUCKET}/${OBJ_KEY}" >/dev/null
  echo "✓ Updated s3://${BUCKET}/${OBJ_KEY}"
}

list_keys() {
  need_mc; download
  python - "$TMP_JSON" <<'PY'
import json,sys
p=sys.argv[1]
with open(p,'r',encoding='utf-8') as f:
    data=json.load(f)
for k in sorted(data.keys()):
    v=data[k]
    print(f"{k} -> {v.get('last_timestamp','')} rows={v.get('last_row_count','')}")
PY
}

delete_key() {
  local vendor="$1" table="${2:-}"
  need_mc; download
  python - "$TMP_JSON" "$vendor" "$table" <<'PY'
import json,sys
path,vendor,table=sys.argv[1:4]
with open(path,'r',encoding='utf-8') as f:
    data=json.load(f)
to_del=[]
if table:
    to_del=[f"{vendor}:{table}"]
else:
    pref=f"{vendor}:"
    to_del=[k for k in data.keys() if k.startswith(pref)]
for k in to_del:
    if k in data:
        data.pop(k,None)
with open(path,'w',encoding='utf-8') as f:
    json.dump(data,f,indent=2)
print("Deleted:",", ".join(to_del) if to_del else "<none>")
PY
  upload
}

usage() {
  cat <<USAGE
Usage:
  $0 --list
  $0 --vendor <alpaca|massive|finnhub|fred> [--table <table_name>]
  Shortcuts:
    --alpaca-all       # remove alpaca day+minute bookmarks
    --alpaca-minute    # remove alpaca:equities_bars_minute
    --alpaca-day       # remove alpaca:equities_bars
    --massive-day      # remove massive:equities_bars
    --fred-treasury    # remove fred:macro_treasury
USAGE
}

if [[ $# -eq 0 ]]; then usage; exit 1; fi

VENDOR=""; TABLE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --list) list_keys; exit 0 ;;
    --vendor) VENDOR="$2"; shift 2 ;;
    --table) TABLE="$2"; shift 2 ;;
    --alpaca-all) VENDOR="alpaca"; TABLE=""; shift ;;
    --alpaca-minute) VENDOR="alpaca"; TABLE="equities_bars_minute"; shift ;;
    --alpaca-day) VENDOR="alpaca"; TABLE="equities_bars"; shift ;;
    --massive-day) VENDOR="massive"; TABLE="equities_bars"; shift ;;
    --fred-treasury) VENDOR="fred"; TABLE="macro_treasury"; shift ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -n "$VENDOR" ]]; then
  delete_key "$VENDOR" "$TABLE"
else
  usage; exit 1
fi
