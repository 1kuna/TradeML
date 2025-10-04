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

MIN_PY_DEPS=(boto3 pandas pyarrow pyyaml python-dotenv loguru alpaca-py requests)

ensure_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    return 1
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
      if grep -q "^$k=" .env; then
        sed -i.bak "s|^$k=.*|$k=$v|" .env
      else
        echo "$k=$v" >> .env
      fi
    done < .env.s3
    echo "✓ Updated .env with S3 credentials"
  fi

  # Controlled S3 creds merge (never|if-empty|always), default if-empty
  if [[ -f .env.s3 ]] && [[ "${S3_ENV_MERGE_MODE:-if-empty}" != "legacy" ]]; then
    mode=${S3_ENV_MERGE_MODE:-if-empty}
    sed -i 's/\r$//' .env.s3 .env 2>/dev/null || true
    while IFS='=' read -r k v; do
      [[ -z "$k" ]] && continue
      if ! printf '%s' "$k" | grep -qE '^[A-Za-z_][A-Za-z0-9_]*$'; then
        continue
      fi
      case "$mode" in
        never)
          ;;
        if-empty|*)
          if ! grep -q "^$k=" .env; then
            echo "$k=$v" >> .env
          fi
          ;;
        always)
          if grep -q "^$k=" .env; then
            sed -i.bak "s|^$k=.*|$k=$v|" .env
          else
            echo "$k=$v" >> .env
          fi
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
        echo "[WARN] S3 creds validation failed; reverting .env to pre-merge"
        if [[ -f .env.premerge ]]; then
          mv .env.premerge .env
        fi
        # Re-export reverted env
        set -a; source .env; set +a
      fi
    fi
  fi

  # Force required toggles
  sed -i.bak "s|^STORAGE_BACKEND=.*|STORAGE_BACKEND=s3|" .env || true
  if ! grep -q "^ROLE=" .env; then echo "ROLE=edge" >> .env; fi

  # Cleanup pre-merge backup if still present (successful merge)
  rm -f .env.premerge 2>/dev/null || true
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
    python scripts/node.py --selfcheck
    ;;
  up)
    install_prereqs
    ensure_env
    ensure_python
    ensure_mc_alias
    prune_locks
    python scripts/node.py --selfcheck
    echo "Starting Pi Node loop..."
    run_loop
    ;;
  *)
    echo "Unknown command: $cmd" >&2
    echo "Usage: bash scripts/pi_node.sh [install|selfcheck|up]" >&2
    exit 2
    ;;
esac
