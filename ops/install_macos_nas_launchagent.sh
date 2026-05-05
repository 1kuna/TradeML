#!/usr/bin/env bash
set -euo pipefail

label="${TRADEML_NAS_LAUNCHD_LABEL:-com.trademl.mount-nas}"
repo_root="${TRADEML_REPO_ROOT:-$HOME/TradeML}"
mount_script="${TRADEML_NAS_MOUNT_SCRIPT:-$repo_root/ops/macos_mount_trademl_nas.sh}"
mount_point="${TRADEML_NAS_MOUNT_POINT:-$HOME/atlas_mounts/trademl_nas}"
server="${TRADEML_NAS_SERVER:-192.168.68.54}"
share="${TRADEML_NAS_SHARE:-TradeML}"
user="${TRADEML_NAS_USER:-kuna}"
keychain_service="${TRADEML_NAS_KEYCHAIN_SERVICE:-trademl-nas-smb}"
required_paths="${TRADEML_NAS_REQUIRED_PATHS:-data/raw/ticker_news:data/raw/equities_minute}"
interval="${TRADEML_NAS_MOUNT_INTERVAL_SECONDS:-300}"
plist_path="${TRADEML_NAS_PLIST_PATH:-$HOME/Library/LaunchAgents/${label}.plist}"
log_dir="${TRADEML_NAS_LOG_DIR:-$repo_root/logs}"

log() {
  printf '%s %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"
}

xml_escape() {
  python3 - "$1" <<'PY'
from __future__ import annotations

import html
import sys

print(html.escape(sys.argv[1], quote=True))
PY
}

if [[ ! -x "$mount_script" ]]; then
  log "mount script is missing or not executable: $mount_script"
  exit 2
fi

if [[ -n "${TRADEML_NAS_PASSWORD:-}" ]]; then
  security add-generic-password -a "$user" -s "$keychain_service" -w "$TRADEML_NAS_PASSWORD" -U >/dev/null
  log "stored NAS password in Keychain service=${keychain_service} account=${user}"
fi

mkdir -p "$(dirname "$plist_path")" "$log_dir" "$mount_point"

cat >"$plist_path" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>$(xml_escape "$label")</string>
  <key>ProgramArguments</key>
  <array>
    <string>$(xml_escape "$mount_script")</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>TRADEML_NAS_MOUNT_POINT</key>
    <string>$(xml_escape "$mount_point")</string>
    <key>TRADEML_NAS_SERVER</key>
    <string>$(xml_escape "$server")</string>
    <key>TRADEML_NAS_SHARE</key>
    <string>$(xml_escape "$share")</string>
    <key>TRADEML_NAS_USER</key>
    <string>$(xml_escape "$user")</string>
    <key>TRADEML_NAS_KEYCHAIN_SERVICE</key>
    <string>$(xml_escape "$keychain_service")</string>
    <key>TRADEML_NAS_REQUIRED_PATHS</key>
    <string>$(xml_escape "$required_paths")</string>
    <key>PATH</key>
    <string>/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
  </dict>
  <key>RunAtLoad</key>
  <true/>
  <key>StartInterval</key>
  <integer>$(xml_escape "$interval")</integer>
  <key>StandardOutPath</key>
  <string>$(xml_escape "$log_dir/mount-trademl-nas.out.log")</string>
  <key>StandardErrorPath</key>
  <string>$(xml_escape "$log_dir/mount-trademl-nas.err.log")</string>
</dict>
</plist>
PLIST

plutil -lint "$plist_path" >/dev/null

uid="$(id -u)"
launchctl bootout "gui/$uid" "$plist_path" >/dev/null 2>&1 || true
launchctl bootstrap "gui/$uid" "$plist_path"
launchctl kickstart -k "gui/$uid/$label"

log "installed and kicked LaunchAgent ${label} from ${plist_path}"
