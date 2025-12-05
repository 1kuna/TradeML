import json
import os
import subprocess
from pathlib import Path


def test_wizard_auto_dry_run(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    venv_path = tmp_path / "venv"
    env = os.environ.copy()
    env.update(
        {
            "RPI_WIZARD_AUTO": "1",
            "RPI_WIZARD_SKIP_SYMLINKS": "1",
            "DATA_ROOT": str(data_root),
            "EDGE_NODE_ID": "test-node",
            "RPI_WIZARD_VENV": str(venv_path),
            "RPI_WIZARD_ENV_PATH": str(tmp_path / ".env"),
        }
    )
    repo_root = Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        ["python", "rpi_wizard.py", "--dry-run", "--fresh"],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    state_path = data_root / "trademl_state" / "rpi_wizard_state.json"
    assert state_path.exists(), "Wizard did not persist state in auto mode"
    data = json.loads(state_path.read_text())
    assert data.get("edge_node_id") == "test-node"
    assert data.get("data_root") == str(data_root)
