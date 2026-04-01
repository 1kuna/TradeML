#!/usr/bin/env sh
set -eu

REPO_ROOT=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
VENV_PATH="${REPO_ROOT}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"
BIN_DIR="${HOME}/.local/bin"
WRAPPER_PATH="${BIN_DIR}/trademl"

"${PYTHON_BIN}" -m venv "${VENV_PATH}"
"${VENV_PATH}/bin/pip" install --upgrade pip
"${VENV_PATH}/bin/pip" install -e "${REPO_ROOT}[dev,dashboard]"

mkdir -p "${BIN_DIR}"
cat > "${WRAPPER_PATH}" <<EOF
#!/usr/bin/env sh
exec "${VENV_PATH}/bin/trademl" "\$@"
EOF
chmod +x "${WRAPPER_PATH}"

echo "TradeML dashboard installed."
echo "Wrapper: ${WRAPPER_PATH}"
echo "Run: trademl dashboard"
