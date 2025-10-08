#!/usr/bin/env bash
set -euo pipefail

# Provision MinIO: create bucket, enable versioning, create app user & policy

MINIO_ENDPOINT=${MINIO_ENDPOINT:-127.0.0.1:9000}
MINIO_SCHEME=${MINIO_SCHEME:-http}
MINIO_ROOT_USER=${MINIO_ROOT_USER:-minioadmin}
MINIO_ROOT_PASSWORD=${MINIO_ROOT_PASSWORD:-minioadmin}
BUCKET=${S3_BUCKET:-ata}

echo "Using endpoint: $MINIO_SCHEME://$MINIO_ENDPOINT"

# Download mc client if not present
if ! command -v mc >/dev/null 2>&1; then
  echo "Installing MinIO client (mc)..."
  OS=$(uname -s | tr '[:upper:]' '[:lower:]')
  ARCH=$(uname -m)
  case "$ARCH" in
    aarch64|arm64) MC_ARCH=arm64 ;;
    x86_64|amd64) MC_ARCH=amd64 ;;
    armv7l|armv6l) MC_ARCH=arm ;;
    *) MC_ARCH=amd64 ;;
  esac
  if [ "$OS" = "darwin" ]; then
    curl -fsSL https://dl.min.io/client/mc/release/darwin-amd64/mc -o mc
  else
    curl -fsSL https://dl.min.io/client/mc/release/linux-${MC_ARCH}/mc -o mc
  fi
  chmod +x mc
  sudo mv mc /usr/local/bin/mc
fi

echo "Configuring mc alias..."
mc alias set local "$MINIO_SCHEME://$MINIO_ENDPOINT" "$MINIO_ROOT_USER" "$MINIO_ROOT_PASSWORD"

echo "Creating bucket $BUCKET (if not exists) and enabling versioning..."
mc mb --ignore-existing local/"$BUCKET"
mc version enable local/"$BUCKET"

APP_USER=${APP_USER:-ata}
APP_PASS=${APP_PASS:-$(openssl rand -hex 24)}

echo "Creating app user: $APP_USER"
mc admin user add local "$APP_USER" "$APP_PASS" || true

POLICY_FILE=$(mktemp)
cat > "$POLICY_FILE" <<'JSON'
{
  "Version": "2012-10-17",
  "Statement": [
    { "Effect": "Allow", "Action": ["s3:ListBucket"], "Resource": ["arn:aws:s3:::*"] },
    { "Effect": "Allow", "Action": ["s3:GetObject","s3:PutObject","s3:DeleteObject","s3:ListBucketMultipartUploads","s3:AbortMultipartUpload"],
      "Resource": ["arn:aws:s3:::*/*"] }
  ]
}
JSON

echo "Creating and attaching policy..."
# Newer mc uses 'create' and 'attach'; fall back to legacy if needed
if mc admin policy create local ata-policy "$POLICY_FILE" 2>/dev/null; then
  mc admin policy attach local ata-policy --user "$APP_USER"
else
  mc admin policy add local ata-policy "$POLICY_FILE" || true
  mc admin policy set local ata-policy user="$APP_USER"
fi
rm -f "$POLICY_FILE"

if [[ -n "${OUTPUT_ENV:-}" ]]; then
  echo "Writing S3 env snippet to $OUTPUT_ENV"
  cat > "$OUTPUT_ENV" <<EOF
AWS_ACCESS_KEY_ID=$APP_USER
AWS_SECRET_ACCESS_KEY=$APP_PASS
S3_ENDPOINT=$MINIO_SCHEME://$MINIO_ENDPOINT
S3_BUCKET=$BUCKET
S3_REGION=us-east-1
S3_FORCE_PATH_STYLE=true
EOF
fi

echo "\nExport these in your .env on every device:"
echo "AWS_ACCESS_KEY_ID=$APP_USER"
echo "AWS_SECRET_ACCESS_KEY=$APP_PASS"
echo "S3_ENDPOINT=$MINIO_SCHEME://$MINIO_ENDPOINT"
echo "S3_BUCKET=$BUCKET"
echo "S3_REGION=us-east-1"
echo "S3_FORCE_PATH_STYLE=true"

echo "\n✓ MinIO provisioning complete"
