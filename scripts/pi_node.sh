#!/usr/bin/env bash
set -euo pipefail

# All-in-one Pi Node script
# - Installs prerequisites (Tailscale, Docker)
# - Launches MinIO and provisions app credentials
# - Ensures .env is present and S3 vars are set
# - Creates a Python venv and installs minimal dependencies (node runs in venv)
# - Runs a resilient loop that resumes on restart
#
# Usage:
#   bash scripts/pi_node.sh up            # bootstrap + selfcheck + run loop
#   bash scripts/pi_node.sh selfcheck     # only run checks
#   bash scripts/pi_node.sh install       # install Docker/Compose + venv tools
#
# Notes:
# - 1-click: `up` handles everything (Docker/Compose install, MinIO, venv, run loop).
# - MinIO startup is guarded: if S3_ENDPOINT is remote or compose-managed MinIO is detected, local MinIO won't start.

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_DIR"

# Per-repo temp dir for safe, atomic edits (avoids sed* litter)
TMP_DIR="$REPO_DIR/.tmp"
mkdir -p "$TMP_DIR" 2>/dev/null || true
# PID directory for background workers
PID_DIR="${PID_DIR:-$REPO_DIR/run}"
mkdir -p "$PID_DIR" 2>/dev/null || true
# Cleanup temp dir on exit/interrupt
cleanup_tmp() { rm -rf "$TMP_DIR" 2>/dev/null || true; }
trap cleanup_tmp EXIT INT TERM

MIN_PY_DEPS=(boto3 pandas pyarrow pyyaml python-dotenv loguru alpaca-py requests exchange-calendars)

ensure_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    return 1
  fi
}

# Normalize CRLF to LF for given files (in-place via temp file)
normalize_lf() {
  for f in "$@"; do
    [[ -f "$f" ]] || continue
    local tf
    tf=$(mktemp -p "$TMP_DIR" eol.XXXXXX)
    # Strip trailing CR if present
    awk '{ sub(/\r$/, ""); print }' "$f" > "$tf" && mv "$tf" "$f" || rm -f "$tf"
  done
}

# Upsert key=value into a .env-style file without in-place sed
# Modes: always (replace if exists), if-empty (only add if absent)
set_env_kv() {
  local file="$1" key="$2" value="$3" mode="${4:-always}"
  # Validate key name
  if ! printf '%s' "$key" | grep -qE '^[A-Za-z_][A-Za-z0-9_]*$'; then
    return 0
  fi
  if [[ "$mode" == "if-empty" ]] && grep -q "^$key=" "$file" 2>/dev/null; then
    return 0
  fi
  if grep -q "^$key=" "$file" 2>/dev/null; then
    local tf
    tf=$(mktemp -p "$TMP_DIR" env.XXXXXX)
    awk -v k="$key" -v v="$value" 'BEGIN{r=0} $0 ~ ("^"k"="){print k"="v; r=1; next} {print} END{exit (r?0:0)}' "$file" > "$tf" && mv "$tf" "$file" || rm -f "$tf"
  else
    printf '%s=%s\n' "$key" "$value" >> "$file"
  fi
}

get_minio_root_creds() {
  # Populate MINIO_ROOT_USER and MINIO_ROOT_PASSWORD into env safely
  # Preference order: existing env -> docker inspect -> saved file -> defaults
  if [[ -n "${MINIO_ROOT_USER:-}" && -n "${MINIO_ROOT_PASSWORD:-}" ]]; then
    export MINIO_ROOT_USER MINIO_ROOT_PASSWORD
    return 0
  fi
  if sudo docker ps -a --format '{{.Names}}' 2>/dev/null | grep -q '^minio$'; then
    local envs
    envs=$(sudo docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' minio 2>/dev/null || true)
    local u p
    u=$(printf "%s" "$envs" | grep '^MINIO_ROOT_USER=' | head -n1 | cut -d= -f2-)
    p=$(printf "%s" "$envs" | grep '^MINIO_ROOT_PASSWORD=' | head -n1 | cut -d= -f2-)
    if [[ -n "$u" && -n "$p" ]]; then
      export MINIO_ROOT_USER="$u" MINIO_ROOT_PASSWORD="$p"
      return 0
    fi
  fi
  if [[ -f .minio_root.env ]]; then
    # shellcheck disable=SC1091
    source .minio_root.env
    if [[ -n "${MINIO_ROOT_USER:-}" && -n "${MINIO_ROOT_PASSWORD:-}" ]]; then
      export MINIO_ROOT_USER MINIO_ROOT_PASSWORD
      return 0
    fi
  fi
  export MINIO_ROOT_USER="rootadmin" MINIO_ROOT_PASSWORD="rootadmin"
}

is_local_host() {
  # Returns 0 if host looks local
  case "$1" in
    127.0.0.1|localhost|::1|0.0.0.0) return 0 ;;
    *) return 1 ;;
  esac
}

should_start_local_minio() {
  # Decide whether to launch local MinIO container.
  # Skip if:
  #  - USE_COMPOSE_MINIO=true (prefer compose-managed service)
  #  - S3_ENDPOINT points to a non-local host
  #  - a compose-managed MinIO (trademl_minio) is already running
  if [[ "${USE_COMPOSE_MINIO:-false}" == "true" ]]; then
    return 1
  fi

  # Extract S3_ENDPOINT from environment or .env (if present)
  local ep
  ep=${S3_ENDPOINT:-}
  if [[ -z "$ep" && -f .env ]]; then
    ep=$(grep -E '^S3_ENDPOINT=' .env | tail -n1 | cut -d= -f2- || true)
  fi

  if [[ -n "$ep" ]]; then
    # Parse host from URL
    local host
    host=$(printf "%s" "$ep" | sed -E 's#^[a-zA-Z]+://([^:/]+).*$#\1#')
    if [[ -n "$host" ]] && ! is_local_host "$host"; then
      return 1
    fi
  fi

  # If compose-managed MinIO is running, skip
  if sudo docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^trademl_minio$'; then
    return 1
  fi

  return 0
}

# venv-only; no Docker/MinIO provisioning here. Keep function name for compatibility.
install_prereqs() {
  echo "[1/5] Ensuring Docker is installed..."
  if ! ensure_command docker; then
    # Try official convenience script (works well on RPi/Debian)
    if ! ensure_command curl; then
      sudo apt-get update -y || true
      sudo apt-get install -y curl || true
    fi
    if command -v curl >/dev/null 2>&1; then
      curl -fsSL https://get.docker.com | sh || true
    fi
    # Fallback to apt docker.io if needed
    if ! command -v docker >/dev/null 2>&1; then
      sudo apt-get update -y || true
      sudo apt-get install -y docker.io || true
    fi
    if ! command -v docker >/dev/null 2>&1; then
      echo "[ERROR] Docker installation failed. Install Docker manually and rerun." >&2
      exit 2
    fi
  fi

  echo "[2/5] Ensuring Docker Compose plugin..."
  if ! docker compose version >/dev/null 2>&1; then
    echo "docker compose plugin missing; installing via Docker official repo..."
    # Install prerequisites for Docker apt repo
    sudo apt-get update -y || true
    sudo apt-get install -y ca-certificates curl gnupg || true
    sudo install -m 0755 -d /etc/apt/keyrings || true
    if [ ! -f /etc/apt/keyrings/docker.gpg ]; then
      curl -fsSL https://download.docker.com/linux/debian/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg || true
      sudo chmod a+r /etc/apt/keyrings/docker.gpg || true
    fi
    codename=$(. /etc/os-release && echo "$VERSION_CODENAME")
    arch=$(dpkg --print-architecture)
    echo "deb [arch=${arch} signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian ${codename} stable" | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
    sudo apt-get update -y || true
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin || true
  fi

  echo "[3/5] Ensuring openssl (for credentials) and Python venv tools..."
  sudo apt-get update -y || true
  sudo apt-get install -y openssl python3-venv python3-pip || true

  echo "[4/5] (Optional) Installing Tailscale..."
  if ! ensure_command tailscale; then
    if command -v curl >/dev/null 2>&1; then
      curl -fsSL https://tailscale.com/install.sh | sh || true
    fi
  fi

  echo "[5/5] Ensuring MinIO is running (docker), unless remote or compose-managed..."
  if should_start_local_minio; then
    if ! sudo docker ps --format '{{.Names}}' | grep -q '^minio$'; then
      sudo mkdir -p /srv/minio
      export MINIO_ROOT_USER=${MINIO_ROOT_USER:-minioadmin}
      export MINIO_ROOT_PASSWORD=${MINIO_ROOT_PASSWORD:-minioadmin}
      # Persist root creds for future provisioning
      printf "MINIO_ROOT_USER=%s\nMINIO_ROOT_PASSWORD=%s\n" "$MINIO_ROOT_USER" "$MINIO_ROOT_PASSWORD" > .minio_root.env
      chmod 600 .minio_root.env || true
      sudo docker rm -f minio 2>/dev/null || true
      sudo docker run -d --name minio --restart=always \
        -e MINIO_ROOT_USER="$MINIO_ROOT_USER" \
        -e MINIO_ROOT_PASSWORD="$MINIO_ROOT_PASSWORD" \
        -p 9000:9000 -p 9001:9001 \
        -v /srv/minio:/data \
        minio/minio server /data --console-address ":9001"
    fi

    echo "Waiting for MinIO to become healthy..."
    if wait_for_minio; then
      echo "Provisioning MinIO app user..."
      export OUTPUT_ENV=.env.s3
      # Point provisioning to local endpoint explicitly
      get_minio_root_creds
      MINIO_ENDPOINT=127.0.0.1:9000 MINIO_SCHEME=http \
      MINIO_ROOT_USER="${MINIO_ROOT_USER:-rootadmin}" MINIO_ROOT_PASSWORD="${MINIO_ROOT_PASSWORD:-rootadmin}" \
      bash ./scripts/provision_minio.sh || true
    else
      echo "MinIO did not become healthy in time; skipping provisioning." >&2
    fi
  else
    echo "Skipping local MinIO startup (remote S3 endpoint or compose-managed MinIO detected)."
  fi
}

ensure_mc_alias() {
  if ! command -v mc >/dev/null 2>&1; then
    echo "mc not installed; skipping alias setup"
    return 0
  fi
  # Load from env or .env
  local ep key secret bucket
  ep=${S3_ENDPOINT:-}
  key=${AWS_ACCESS_KEY_ID:-}
  secret=${AWS_SECRET_ACCESS_KEY:-}
  bucket=${S3_BUCKET:-}
  if [[ -z "$ep" && -f .env ]]; then ep=$(grep -E '^S3_ENDPOINT=' .env | tail -n1 | cut -d= -f2- || true); fi
  if [[ -z "$key" && -f .env ]]; then key=$(grep -E '^AWS_ACCESS_KEY_ID=' .env | tail -n1 | cut -d= -f2- || true); fi
  if [[ -z "$secret" && -f .env ]]; then secret=$(grep -E '^AWS_SECRET_ACCESS_KEY=' .env | tail -n1 | cut -d= -f2- || true); fi
  if [[ -z "$bucket" && -f .env ]]; then bucket=$(grep -E '^S3_BUCKET=' .env | tail -n1 | cut -d= -f2- || true); fi
  if [[ -z "$ep" || -z "$key" || -z "$secret" ]]; then
    echo "Missing S3 env vars; skipping mc alias"
    return 0
  fi
  # Set/refresh alias and verify
  mc alias set minio "$ep" "$key" "$secret" --api S3v4 >/dev/null 2>&1 || true
  if ! mc ls minio/ >/dev/null 2>&1; then
    echo "Resetting mc alias credentials for minio..."
    mc alias rm minio >/dev/null 2>&1 || true
    mc alias set minio "$ep" "$key" "$secret" --api S3v4 >/dev/null 2>&1 || true
  fi
}

prune_locks() {
  if ! command -v mc >/dev/null 2>&1; then
    return 0
  fi
  local bucket
  bucket=${S3_BUCKET:-}
  if [[ -z "$bucket" && -f .env ]]; then bucket=$(grep -E '^S3_BUCKET=' .env | tail -n1 | cut -d= -f2- || true); fi
  [[ -z "$bucket" ]] && return 0
  # Remove locks prefix safely; it is ephemeral
  if mc ls "minio/$bucket/locks/" >/dev/null 2>&1; then
    echo "Clearing S3 locks/ prefix before start..."
    mc rm -r --force "minio/$bucket/locks/" >/dev/null 2>&1 || true
  fi
}

ensure_env() {
  if [[ ! -f .env ]]; then
    cp .env.template .env
  fi
  # Keep a pre-merge backup to allow revert if validation fails
  cp .env .env.premerge 2>/dev/null || true
  # Sync .env with .env.template (non-destructive; preserves existing values)
  if command -v python3 >/dev/null 2>&1; then
    python3 scripts/sync_env.py --no-backup || true
  fi
  # Merge S3 snippet if present
  if [[ -f .env.s3 ]] && [[ "${S3_ENV_MERGE_MODE:-if-empty}" == "legacy" ]]; then
    # Replace or append keys
    while IFS='=' read -r k v; do
      [[ -z "$k" ]] && continue
      set_env_kv ".env" "$k" "$v" "always"
    done < .env.s3
    echo "✓ Updated .env with S3 credentials"
  fi

  # Controlled S3 creds merge (never|if-empty|always), default if-empty
  if [[ -f .env.s3 ]] && [[ "${S3_ENV_MERGE_MODE:-if-empty}" != "legacy" ]]; then
    mode=${S3_ENV_MERGE_MODE:-if-empty}
    normalize_lf .env.s3 .env
    while IFS='=' read -r k v; do
      [[ -z "$k" ]] && continue
      if ! printf '%s' "$k" | grep -qE '^[A-Za-z_][A-Za-z0-9_]*$'; then
        continue
      fi
      # Skip applying empty values to avoid wiping populated keys
      if [[ -z "$v" ]]; then
        continue
      fi
      case "$mode" in
        never)
          ;;
        if-empty|*)
          set_env_kv ".env" "$k" "$v" "if-empty"
          ;;
        always)
          set_env_kv ".env" "$k" "$v" "always"
          ;;
      esac
    done < .env.s3
    echo "S3 creds merge ($mode) applied"
  fi

  # Validate merged S3 creds (optional, default on)
  if [[ "${S3_ENV_VALIDATE:-true}" == "true" ]]; then
    # Export current .env so ensure_mc_alias uses the right values
    set -a; source .env; set +a
    ensure_mc_alias || true
    if command -v mc >/dev/null 2>&1; then
      if ! mc ls "minio/${S3_BUCKET:-}/" >/dev/null 2>&1; then
        echo "[WARN] S3 creds validation failed; attempting forced merge from .env.s3"
        if [[ -f .env.s3 ]]; then
          # Force merge .env.s3 values (overwrite)
          while IFS='=' read -r k v; do
            [[ -z "$k" ]] && continue
            # Skip empty values to avoid clobbering keys with blanks
            if [[ -z "$v" ]]; then
              continue
            fi
            set_env_kv ".env" "$k" "$v" "always"
          done < .env.s3
          # Re-export and try again
          set -a; source .env; set +a
          ensure_mc_alias || true
          if mc ls "minio/${S3_BUCKET:-}/" >/dev/null 2>&1; then
            echo "✓ Forced merge succeeded; using updated S3 credentials"
          else
            echo "[WARN] Forced merge did not validate; reverting .env to pre-merge"
            if [[ -f .env.premerge ]]; then
              mv .env.premerge .env
            fi
            set -a; source .env; set +a
          fi
        else
          echo "[WARN] No .env.s3 present to force-merge; reverting to pre-merge if available"
          if [[ -f .env.premerge ]]; then
            mv .env.premerge .env
          fi
          set -a; source .env; set +a
        fi
      fi
    fi
  fi

  # Force required toggles
  set_env_kv ".env" "STORAGE_BACKEND" "s3" "always"
  if ! grep -q "^ROLE=" .env; then echo "ROLE=edge" >> .env; fi
  # Ensure per-vendor scheduler is enabled out of the box
  set_env_kv ".env" "EDGE_SCHEDULER_MODE" "per_vendor" "always"

  # Cleanup pre-merge backup if still present (successful merge)
  rm -f .env.premerge 2>/dev/null || true

  # Concurrency defaults (increase workers and vendor inflight caps)
  set_env_kv ".env" "NODE_WORKERS" "${NODE_WORKERS:-6}" "always"
  set_env_kv ".env" "NODE_MAX_INFLIGHT_POLYGON" "${NODE_MAX_INFLIGHT_POLYGON:-2}" "always"
  set_env_kv ".env" "NODE_MAX_INFLIGHT_FINNHUB" "${NODE_MAX_INFLIGHT_FINNHUB:-2}" "always"
  set_env_kv ".env" "NODE_MAX_INFLIGHT_FRED" "${NODE_MAX_INFLIGHT_FRED:-2}" "always"
  # Sensible default rate-limit cooldowns (if not explicitly set)
  set_env_kv ".env" "NODE_VENDOR_FREEZE_SECONDS_ALPACA" "${NODE_VENDOR_FREEZE_SECONDS_ALPACA:-60}" "if-empty"
  set_env_kv ".env" "NODE_VENDOR_FREEZE_SECONDS_POLYGON" "${NODE_VENDOR_FREEZE_SECONDS_POLYGON:-60}" "if-empty"
  set_env_kv ".env" "NODE_VENDOR_FREEZE_SECONDS_FINNHUB" "${NODE_VENDOR_FREEZE_SECONDS_FINNHUB:-60}" "if-empty"
  set_env_kv ".env" "NODE_VENDOR_FREEZE_SECONDS_FRED" "${NODE_VENDOR_FREEZE_SECONDS_FRED:-60}" "if-empty"
  # Backfill horizons (defaults guided by SSOT/Playbook)
  set_env_kv ".env" "ALPACA_DAY_START_DAYS" "${ALPACA_DAY_START_DAYS:-5475}" "if-empty"    # ~15y
  set_env_kv ".env" "ALPACA_MINUTE_START_DAYS" "${ALPACA_MINUTE_START_DAYS:-730}" "if-empty" # ~2y
  set_env_kv ".env" "POLYGON_DAY_START_DAYS" "${POLYGON_DAY_START_DAYS:-3650}" "if-empty"   # ~10y
  set_env_kv ".env" "FRED_TREASURY_START_DAYS" "${FRED_TREASURY_START_DAYS:-18250}" "if-empty" # ~50y
}

ensure_python() {
  if [[ ! -d venv ]]; then
    if ! python3 -m venv venv; then
      echo "[ERROR] Failed to create venv. On Debian/RPi: sudo apt-get install python3-venv" >&2
      exit 2
    fi
  fi
  source venv/bin/activate
  python -m pip install --upgrade pip
  echo "Installing minimal Python dependencies..."
  python -m pip install "${MIN_PY_DEPS[@]}"
}

run_loop() {
  source venv/bin/activate
  # Avoid ambient AWS_* overriding .env inside Python
  unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN AWS_PROFILE
  python scripts/node.py --interval "${RUN_INTERVAL_SECONDS:-900}"
}

# ---------- Worker lifecycle helpers ----------

pid_file_for() {
  local vendor="$1"
  echo "$PID_DIR/${vendor}.pid"
}

is_pid_running() {
  local pid="$1"
  if [[ -z "$pid" ]]; then return 1; fi
  if kill -0 "$pid" >/dev/null 2>&1; then return 0; fi
  if command -v ps >/dev/null 2>&1 && ps -p "$pid" >/dev/null 2>&1; then return 0; fi
  return 1
}

stop_vendor() {
  local vendor="$1"
  local pf; pf=$(pid_file_for "$vendor")
  if [[ ! -f "$pf" ]]; then
    return 0
  fi
  local pid; pid=$(cat "$pf" 2>/dev/null || true)
  if [[ -z "$pid" ]]; then
    rm -f "$pf" 2>/dev/null || true
    return 0
  fi
  if is_pid_running "$pid"; then
    echo "Stopping $vendor (pid $pid)..."
    kill -TERM "$pid" 2>/dev/null || true
    # wait up to 20s for graceful shutdown
    local i
    for i in $(seq 1 20); do
      if ! is_pid_running "$pid"; then break; fi
      sleep 1
    done
    if is_pid_running "$pid"; then
      echo "$vendor did not exit in time; force killing..."
      kill -KILL "$pid" 2>/dev/null || true
    fi
  fi
  rm -f "$pf" 2>/dev/null || true
}

stop_all_vendors() {
  # Stop known vendors first
  local vendors=(polygon alpaca finnhub fred)
  local v
  for v in "${vendors[@]}"; do
    stop_vendor "$v"
  done
  # Clean up any stray pid files
  if compgen -G "$PID_DIR/*.pid" >/dev/null; then
    for pf in "$PID_DIR"/*.pid; do
      local base; base=$(basename "$pf" .pid)
      stop_vendor "$base"
    done
  fi
}

wait_for_minio() {
  local url=${1:-http://127.0.0.1:9000/minio/health/live}
  local attempts=${2:-30}
  local delay=${3:-2}
  for i in $(seq 1 "$attempts"); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep "$delay"
  done
  return 1
}

cmd=${1:-up}
case "$cmd" in
  install)
    install_prereqs
    ensure_env
    ensure_python
    ;;
  selfcheck)
    install_prereqs
    ensure_env
    ensure_python
    ensure_mc_alias
    # Avoid ambient AWS_* overriding .env inside Python
    unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN AWS_PROFILE
    python scripts/node.py --selfcheck
    ;;
  up)
    install_prereqs
    ensure_env
    ensure_python
    ensure_mc_alias
    prune_locks
    python scripts/node.py --selfcheck
    echo "Starting per-vendor workers..."
    # Logs directory
    log_dir="${LOG_DIR:-./logs}"
    mkdir -p "$log_dir"
    
    start_vendor() {
      vendor="$1"; conc="$2"; rpm="$3"; budget="$4"
      # Stop existing vendor if running
      stop_vendor "$vendor" || true
      # Export vendor-specific env for this process only
      echo "Launching $vendor worker (conc=$conc rpm=$rpm budget=$budget)"
      nohup env \
        VENDOR="$vendor" \
        CONCURRENCY="$conc" \
        RPM_LIMIT="$rpm" \
        BUDGET="$budget" \
        python -m trademl.workers.vendor_worker \
          >> "$log_dir/${vendor}.log" 2>&1 &
      pid=$!
      echo "$vendor started (pid $pid)"
      echo "$pid" > "$(pid_file_for "$vendor")"
    }

    # Example defaults; tune via editing below or exporting env before calling
    start_vendor polygon 1 5   1000
    start_vendor alpaca  8 200 0
    start_vendor finnhub 4 60  0
    start_vendor fred    2 60  0
    echo "✓ Workers launched. Tail logs with: tail -f -n 100 $log_dir/*.log"
    ;;
  down)
    echo "Stopping vendor workers..."
    stop_all_vendors
    echo "✓ All workers stopped"
    ;;
  status)
    echo "Vendor worker status:"
    for pf in "$PID_DIR"/*.pid; do
      [[ -e "$pf" ]] || continue
      v=$(basename "$pf" .pid)
      pid=$(cat "$pf" 2>/dev/null || true)
      if is_pid_running "$pid"; then
        echo "- $v: running (pid $pid)"
      else
        echo "- $v: not running (stale pid $pid); cleaning up"
        rm -f "$pf" 2>/dev/null || true
      fi
    done
    ;;
  *)
    echo "Unknown command: $cmd" >&2
    echo "Usage: bash scripts/pi_node.sh [install|selfcheck|up|down|status]" >&2
    exit 2
    ;;
esac
