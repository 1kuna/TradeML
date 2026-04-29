from __future__ import annotations

import subprocess
from typing import Any

from trademl.fleet import remote


def test_run_password_ssh_redacts_password_env(monkeypatch) -> None:
    observed: dict[str, Any] = {}
    monkeypatch.setenv("TRADEML_TEST_PASSWORD", "secret-value")

    def fake_run(command, **kwargs):  # noqa: ANN001
        observed["command"] = command
        observed["env"] = kwargs["env"]
        return subprocess.CompletedProcess(command, 0, "active\n", "")

    monkeypatch.setattr(remote.subprocess, "run", fake_run)

    result = remote.run_password_ssh(
        {
            "host": "100.0.0.1",
            "user": "zach",
            "password_env": "TRADEML_TEST_PASSWORD",
        },
        "systemctl --user is-active trademl-node.service",
    )

    payload = result.to_dict()
    assert observed["command"][:3] == ["sshpass", "-e", "ssh"]
    assert observed["env"]["SSHPASS"] == "secret-value"
    assert "secret-value" not in " ".join(payload["command"])
    assert payload["stdout"] == "active\n"


def test_run_password_ssh_reports_timeout(monkeypatch) -> None:
    def timeout_run(*_args, **_kwargs):  # noqa: ANN002, ANN003
        raise subprocess.TimeoutExpired(cmd="ssh", timeout=30)

    monkeypatch.setattr(remote.subprocess, "run", timeout_run)

    result = remote.run_password_ssh({"host": "100.0.0.1", "user": "zach"}, "date")

    assert result.returncode == 124
    assert result.timed_out is True
    assert "timed out" in result.stderr
