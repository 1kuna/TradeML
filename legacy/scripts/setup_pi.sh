#!/usr/bin/env bash
set -euo pipefail

# Automated Raspberry Pi provisioning: Tailscale + Docker + MinIO
# Usage: TS_AUTHKEY=tskey-auth-XXXX TS_HOSTNAME=minio bash scripts/setup_pi.sh

echo "[1/5] Installing Tailscale..."
curl -fsSL https://tailscale.com/install.sh | sh

echo "[2/5] Enrolling in Tailscale..."
if [[ -z "${TS_AUTHKEY:-}" ]]; then
  echo "ERROR: TS_AUTHKEY env var is required" >&2
  exit 1
fi
TS_HOSTNAME=${TS_HOSTNAME:-minio}
sudo tailscale up --authkey "$TS_AUTHKEY" --hostname "$TS_HOSTNAME" --ssh || true
tailscale status || true

echo "[3/5] Installing Docker & Compose plugin..."
sudo apt-get update -y
sudo apt-get install -y docker.io docker-compose-plugin curl openssl

echo "[4/5] Launching MinIO container..."
sudo mkdir -p /srv/minio && sudo chown -R "$USER" /srv/minio

export MINIO_ROOT_USER=${MINIO_ROOT_USER:-rootadmin}
export MINIO_ROOT_PASSWORD=${MINIO_ROOT_PASSWORD:-$(openssl rand -hex 24)}
echo "MinIO root user: $MINIO_ROOT_USER"
echo "MinIO root password: $MINIO_ROOT_PASSWORD"

sudo docker rm -f minio 2>/dev/null || true
sudo docker run -d --name minio --restart=always \
  -e MINIO_ROOT_USER="$MINIO_ROOT_USER" \
  -e MINIO_ROOT_PASSWORD="$MINIO_ROOT_PASSWORD" \
  -p 9000:9000 -p 9001:9001 \
  -v /srv/minio:/data \
  minio/minio server /data --console-address ":9001"

echo "[5/5] Provisioning bucket and app user..."
bash ./scripts/provision_minio.sh

echo "\n✓ Raspberry Pi setup complete"
echo "- MinIO API:     http://$TS_HOSTNAME:9000"
echo "- MinIO Console: http://$TS_HOSTNAME:9001"
echo "Update your .env with the printed AWS_ACCESS_KEY_ID/SECRET from provision_minio.sh"
