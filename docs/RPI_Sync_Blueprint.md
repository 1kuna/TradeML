MinIO + Tailscale — Universal S3 Backend (Drop‑in for Blueprint)

This drop‑in replaces Syncthing with a private, always‑on S3 endpoint hosted on your Raspberry Pi and reachable from every device via Tailscale (WireGuard). It plugs directly into the existing blueprint’s STORAGE_BACKEND=s3 path, so the same code runs on Pi, Windows, and macOS with only .env changes.

⸻

0) Goals
	•	One bucket, one set of credentials → every device can read/write s3://ata/{raw,curated,manifests,locks}.
	•	Works on LAN or remote without port‑forwarding/NAT.
	•	TLS optional: traffic is already encrypted inside the Tailscale tunnel.
	•	No code branches; behavior flips via env/compose.

⸻

1) Network & Names
	•	Tailscale creates a private tailnet. Each node gets a stable name, e.g., pi.tailnet.ts.net (MagicDNS).
	•	We will expose MinIO at http://pi.tailnet.ts.net:9000 (API) and http://pi.tailnet.ts.net:9001 (console).

If you prefer short names, set your Pi’s Tailscale hostname to minio so the endpoint is http://minio:9000 within the tailnet.

⸻

2) Secrets & .env (shared across devices)

Create a repo‑root .env file not committed to git (or use a secrets manager). These are the only fields you change per device:

# Role / backend
ROLE=edge                  # edge | curator
STORAGE_BACKEND=s3         # s3 is now the default

# S3 (MinIO on Pi via Tailscale)
S3_ENDPOINT=http://pi.tailnet.ts.net:9000
S3_BUCKET=ata
S3_REGION=us-east-1        # MinIO ignores region but SDKs require one
S3_FORCE_PATH_STYLE=true   # important for MinIO
AWS_ACCESS_KEY_ID=REPLACE_ME
AWS_SECRET_ACCESS_KEY=REPLACE_ME

# (Optional) local paths if you ever flip to local backend
DATA_ROOT=/srv/ata-data

# Tailscale (used only on the Pi for auto-enroll)
TS_AUTHKEY=tskey-auth-XXXXXXXXXXXXXXXXXXXXXXXXXXXX
TS_HOSTNAME=minio         # how the node shows up in MagicDNS


⸻

3) Raspberry Pi Setup (LLM can run these verbatim)

3.1 Install Tailscale and enroll the Pi

curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up --authkey=$TS_AUTHKEY --hostname=$TS_HOSTNAME --ssh

	•	--ssh lets you SSH over Tailscale without opening LAN ports.
	•	Confirm: tailscale status shows the device and MagicDNS name.

3.2 Install Docker and run MinIO

sudo apt-get update && sudo apt-get install -y docker.io docker-compose-plugin
sudo mkdir -p /srv/minio && sudo chown -R $USER /srv/minio

# Strong root creds for the MinIO console (not your app keys)
export MINIO_ROOT_USER=rootadmin
export MINIO_ROOT_PASSWORD=$(openssl rand -hex 24)

docker run -d --name minio --restart=always \
  -e MINIO_ROOT_USER=$MINIO_ROOT_USER \
  -e MINIO_ROOT_PASSWORD=$MINIO_ROOT_PASSWORD \
  -p 9000:9000 -p 9001:9001 \
  -v /srv/minio:/data \
  minio/minio server /data --console-address ":9001"

3.3 Provision bucket, app user, and policy (with MinIO Client mc)

# Download mc (MinIO client) binary
curl -fsSL https://dl.min.io/client/mc/release/linux-amd64/mc -o mc && chmod +x mc
./mc alias set local http://127.0.0.1:9000 $MINIO_ROOT_USER $MINIO_ROOT_PASSWORD

# Create bucket
./mc mb local/ata
./mc version enable local/ata           # enable bucket versioning

# Create an app user for the pipeline and print keys
APP_USER=ata
APP_PASS=$(openssl rand -hex 24)
./mc admin user add local $APP_USER $APP_PASS

# Least-privilege policy for our app user
cat > policy-ata.json <<'JSON'
{
  "Version": "2012-10-17",
  "Statement": [
    { "Effect": "Allow", "Action": ["s3:ListBucket"], "Resource": ["arn:aws:s3:::ata"] },
    { "Effect": "Allow", "Action": ["s3:GetObject","s3:PutObject","s3:DeleteObject","s3:ListBucketMultipartUploads","s3:AbortMultipartUpload"],
      "Resource": ["arn:aws:s3:::ata/*"] }
  ]
}
JSON
./mc admin policy add local ata-policy policy-ata.json
./mc admin policy set local ata-policy user=$APP_USER

echo "AWS_ACCESS_KEY_ID=$APP_USER"
echo "AWS_SECRET_ACCESS_KEY=$APP_PASS"

	•	Put those two values into .env on every device.

Optional hardening: ./mc ilm add local/ata --expiry-days 3650 (lifecycle), SSE-KMS config, and audit logging can be added later; not required for MVP.

⸻

4) Repo Diffs (code stays the same; infra files new)

4.1 Compose: keep profiles and add an optional dev MinIO

# docker-compose.yml (excerpt)
services:
  edge-collector:
    image: ghcr.io/yourorg/ata:latest
    env_file: .env
    command: ["python","edge_collector.py","--config","/etc/ata/edge.yml"]
    profiles: ["edge"]

  curator:
    image: ghcr.io/yourorg/ata:latest
    env_file: .env
    command: ["python","curator.py","--since-last-watermark"]
    profiles: ["curator"]

  # Optional local dev S3 on non-Pi devices
  dev-minio:
    image: minio/minio
    command: server /data --console-address :9001
    environment:
      - MINIO_ROOT_USER=${AWS_ACCESS_KEY_ID}
      - MINIO_ROOT_PASSWORD=${AWS_SECRET_ACCESS_KEY}
    ports: ["9000:9000","9001:9001"]
    volumes: ["dev-minio:/data"]
    profiles: ["dev-s3"]

volumes:
  dev-minio: {}

4.2 Storage config (already supported by the blueprint)

# config fragment (edge.yml / curator.yml)
storage:
  backend: ${STORAGE_BACKEND}          # s3
  s3:
    endpoint: ${S3_ENDPOINT}           # http://pi.tailnet.ts.net:9000
    bucket: ${S3_BUCKET}
    region: ${S3_REGION}
    access_key: ${AWS_ACCESS_KEY_ID}
    secret_key: ${AWS_SECRET_ACCESS_KEY}
    force_path_style: ${S3_FORCE_PATH_STYLE}
locks:
  lease_seconds: 120
  renew_seconds: 45

SDK hint: set S3_FORCE_PATH_STYLE=true and use the endpoint override. For boto3: Config(signature_version='s3v4'), endpoint_url=..., config=Config(s3={'addressing_style':'path'}).

⸻

5) LLM Tasks (exact items to implement)
	1.	S3 client wrapper used by edge collector and curator:
	•	Accepts env/config above; supports put_object, get_object, list_objects, conditional put (If-Match/If-None-Match with ETag), and multipart uploads.
	•	Helper: lease_acquire(name, ttl), lease_renew(name), lease_release(name) using locks/<name>.lock JSON with {holder, expires_at}.
	2.	Bookmarks & manifests stored in S3:
	•	s3://ata/manifests/bookmarks.json (or sqlite file) with ETag preconditions to prevent clobbering.
	•	s3://ata/manifests/<date>/manifest-<source>-<shard>.jsonl append‑style logs for auditing.
	3.	Edge collector resume:
	•	Read bookmark → ingest window → upload to raw/<source>/<table>/date=YYYY-MM-DD/... (tmp key → final key after ETag verify) → append manifest → advance bookmark (ETag precondition).
	4.	Curator watermark:
	•	List raw partitions; skip ones already present in curated/ (by key or ETag table) → write curated Parquet → update curator_watermark.json atomically.
	5.	Env & CLI:
	•	Respect .env for S3 settings and role (ROLE=edge|curator).
	•	Provide make targets: make edge-up, make curator-up, make dev-s3.
	6.	Smoke tests:
	•	tests/test_s3_lease.py (acquire/renew/steal after expiry).
	•	tests/test_resume.py (kill mid‑window; next run resumes).
	•	tests/test_curate_once.py (idempotent curate).

⸻

6) Device Setup (Windows/macOS)

Install Tailscale (GUI installer), sign in once. Confirm you can ping pi.tailnet.ts.net.

Set .env on each device with the same S3 keys and endpoint. Then run:

# Collector (if you also collect here)
ROLE=edge STORAGE_BACKEND=s3 docker compose --profile edge up -d
# Curator + training
ROLE=curator STORAGE_BACKEND=s3 docker compose --profile curator up -d

Locking ensures only one active collector per worker.

⸻

7) Security & Hardening
	•	Least privilege: the ata app user has access only to ata bucket. Use separate users for dev vs prod if desired.
	•	Versioning enabled for accidental deletes/overwrites. Add lifecycle rules later to prune old parts.
	•	Encryption: traffic is encrypted by Tailscale; you may still enable SSE-S3 (MinIO KMS) later.
	•	Backups: periodically mc mirror local/ata file:///backups/ata-$(date +%F) or sync to cold storage.

⸻

8) Operations & Health
	•	MinIO console at http://pi.tailnet.ts.net:9001 for object browser, metrics, alerts.
	•	Add a nightly health task that writes logs/edge_health.json to S3 with rows/day, last vendor contact, and error counts.

⸻

9) Acceptance Tests (must pass)
	1.	Universal access: from Windows/macOS, aws s3 ls --endpoint-url $S3_ENDPOINT s3://ata/ lists the bucket.
	2.	Lease behavior: start collectors on two machines → only one acquires the lease; kill it → the other takes over after TTL.
	3.	Power cut: unplug Pi mid‑upload → on reboot, collector resumes without corruption; ETag checks prevent duplicates.
	4.	Curate at scale: leave Pi collecting 48h; curator spins up and processes all new partitions exactly once.

⸻

10) Runbook
	•	Pi first‑time: enroll Tailscale → run MinIO container → create bucket + app user via mc → paste keys into .env on all devices.
	•	Daily use: Pi keeps collecting; any PC/Mac can curate/train by starting the curator profile.
	•	Recovery: if Pi is replaced, re‑enroll Tailscale with same hostname, restore /srv/minio data dir, and restart container.

⸻

TL;DR for the LLM agent
	1.	Provision Tailscale + MinIO on the Pi; create ata bucket and ata user; return keys.
	2.	Write .env and compose profiles; implement S3 client, leases, bookmarks, and curator watermark logic.
	3.	Ship smoke tests and make targets; validate acceptance tests.
	4.	Document the MagicDNS endpoint in the project README and default configs.