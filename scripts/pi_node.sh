#!/usr/bin/env bash
set -euo pipefail

# All-in-one Pi Node script
# - Installs prerequisites (Tailscale, Docker)
# - Launches MinIO and provisions app credentials
# - Ensures .env is present and S3 vars are set
# - Creates a Python venv and installs minimal dependencies
# - Runs a resilient loop that resumes on restart
#
# Usage:
#   bash scripts/pi_node.sh up            # bootstrap + selfcheck + run loop
#   bash scripts/pi_node.sh selfcheck     # only run checks
#   bash scripts/pi_node.sh install       # install prereqs only

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_DIR"

MIN_PY_DEPS=(boto3 pandas pyarrow pyyaml python-dotenv loguru alpaca-py requests)

ensure_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    return 1
  fi
}

install_prereqs() {
  echo "[1/4] Installing Docker & tools..."
  if ! ensure_command docker; then
    sudo apt-get update -y
    sudo apt-get install -y docker.io docker-compose-plugin curl openssl python3-venv python3-pip
  fi

  echo "[2/4] (Optional) Installing Tailscale..."
  if ! ensure_command tailscale; then
    curl -fsSL https://tailscale.com/install.sh | sh || true
  fi

  echo "[3/4] Ensuring MinIO is running..."
  if ! sudo docker ps --format '{{.Names}}' | grep -q '^minio$'; then
    sudo mkdir -p /srv/minio
    export MINIO_ROOT_USER=${MINIO_ROOT_USER:-rootadmin}
    export MINIO_ROOT_PASSWORD=${MINIO_ROOT_PASSWORD:-$(openssl rand -hex 24)}
    sudo docker rm -f minio 2>/dev/null || true
    sudo docker run -d --name minio --restart=always \
      -e MINIO_ROOT_USER="$MINIO_ROOT_USER" \
      -e MINIO_ROOT_PASSWORD="$MINIO_ROOT_PASSWORD" \
      -p 9000:9000 -p 9001:9001 \
      -v /srv/minio:/data \
      minio/minio server /data --console-address ":9001"
  fi

  echo "[4/4] Provisioning MinIO app user..."
  export OUTPUT_ENV=.env.s3
  bash ./scripts/provision_minio.sh || true
}

ensure_env() {
  if [[ ! -f .env ]]; then
    cp .env.template .env
  fi
  # Merge S3 snippet if present
  if [[ -f .env.s3 ]]; then
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

  # Force required toggles
  sed -i.bak "s|^STORAGE_BACKEND=.*|STORAGE_BACKEND=s3|" .env || true
  if ! grep -q "^ROLE=" .env; then echo "ROLE=edge" >> .env; fi
}

ensure_python() {
  if [[ ! -d venv ]]; then
    python3 -m venv venv
  fi
  source venv/bin/activate
  pip install --upgrade pip
  echo "Installing minimal Python dependencies..."
  pip install "${MIN_PY_DEPS[@]}"
}

run_loop() {
  source venv/bin/activate
  python scripts/node.py --interval "${RUN_INTERVAL_SECONDS:-900}"
}

cmd=${1:-up}
case "$cmd" in
  install)
    install_prereqs
    ensure_env
    ensure_python
    ;;
  selfcheck)
    ensure_env
    ensure_python
    python scripts/node.py --selfcheck
    ;;
  up)
    ensure_env
    install_prereqs
    ensure_python
    python scripts/node.py --selfcheck
    echo "Starting Pi Node loop..."
    run_loop
    ;;
  *)
    echo "Unknown command: $cmd" >&2
    echo "Usage: bash scripts/pi_node.sh [install|selfcheck|once|up]" >&2
    exit 2
    ;;
esac
