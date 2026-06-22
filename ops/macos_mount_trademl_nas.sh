#!/usr/bin/env bash
set -euo pipefail

mount_point="${TRADEML_NAS_MOUNT_POINT:-$HOME/atlas_mounts/dev}"
server="${TRADEML_NAS_SERVER:-192.168.68.54}"
share="${TRADEML_NAS_SHARE:-dev}"
user="${TRADEML_NAS_USER:-kuna}"
keychain_service="${TRADEML_NAS_KEYCHAIN_SERVICE:-trademl-nas-smb}"
required_paths="${TRADEML_NAS_REQUIRED_PATHS:-TradeML/data/raw/ticker_news:TradeML/data/raw/equities_minute}"

log() {
  printf '%s %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"
}

is_mounted() {
  mount | grep -F " on ${mount_point} " >/dev/null 2>&1
}

verify_required_paths() {
  local missing=0
  local old_ifs="$IFS"
  IFS=':'
  for relative_path in $required_paths; do
    if [[ -n "$relative_path" && ! -e "${mount_point}/${relative_path}" ]]; then
      log "missing required NAS path: ${mount_point}/${relative_path}"
      missing=1
    fi
  done
  IFS="$old_ifs"
  return "$missing"
}

urlencode() {
  python3 - "$1" <<'PY'
from __future__ import annotations

import sys
from urllib.parse import quote

print(quote(sys.argv[1], safe=""))
PY
}

password_from_keychain() {
  security find-generic-password -a "$user" -s "$keychain_service" -w 2>/dev/null
}

mkdir -p "$mount_point"

if is_mounted; then
  verify_required_paths
  log "TradeML NAS already mounted at ${mount_point}"
  exit 0
fi

password="${TRADEML_NAS_PASSWORD:-}"
if [[ -z "$password" ]]; then
  password="$(password_from_keychain || true)"
fi
if [[ -z "$password" ]]; then
  log "missing NAS password: add a Keychain item service=${keychain_service} account=${user}"
  exit 2
fi

encoded_user="$(urlencode "$user")"
encoded_password="$(urlencode "$password")"
encoded_share="$(urlencode "$share")"

log "mounting TradeML NAS share //${user}@${server}/${share} at ${mount_point}"
/sbin/mount_smbfs "//${encoded_user}:${encoded_password}@${server}/${encoded_share}" "$mount_point"
unset password encoded_password

verify_required_paths
log "TradeML NAS mounted at ${mount_point}"
