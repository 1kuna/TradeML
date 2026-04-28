from __future__ import annotations

import os
import subprocess
from pathlib import Path


def test_install_systemd_wrapper_passes_node_options_before_subcommand(tmp_path: Path) -> None:
    home = tmp_path / "home"
    bin_dir = home / ".local" / "bin"
    bin_dir.mkdir(parents=True)
    argv_path = tmp_path / "argv.txt"
    fake_trademl = bin_dir / "trademl"
    fake_trademl.write_text(
        "#!/usr/bin/env sh\n"
        "printf '%s\\n' \"$@\" > \"$TRADEML_ARGV_CAPTURE\"\n",
        encoding="utf-8",
    )
    fake_trademl.chmod(0o755)
    workspace = tmp_path / "worker"
    service_path = tmp_path / "trademl-node.service"
    env = {**os.environ, "HOME": str(home), "TRADEML_ARGV_CAPTURE": str(argv_path)}

    result = subprocess.run(
        ["sh", "install_systemd.sh", str(service_path), str(workspace)],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    argv = argv_path.read_text(encoding="utf-8").splitlines()
    assert argv == [
        "node",
        "--workspace-root",
        str(workspace),
        "--config",
        str(workspace / "node.yml"),
        "--env-file",
        str(workspace / ".env"),
        "install-service",
        "--service-path",
        str(service_path),
    ]
