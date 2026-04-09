from __future__ import annotations

from trademl.pi_remote import (
    RemoteTarget,
    dashboard_snapshot_script,
    python_here_doc,
    restart_node_script,
    wrap_remote_command,
)


def test_remote_target_from_env_defaults(monkeypatch) -> None:
    monkeypatch.setenv("TRADEML_PI_PASSWORD", "secret")

    target = RemoteTarget.from_env()

    assert target.host == "100.76.4.69"
    assert target.user == "zach"
    assert target.password == "secret"
    assert target.port == 22


def test_wrap_remote_command_applies_cwd_and_env() -> None:
    wrapped = wrap_remote_command("python -m example", cwd="/tmp/work", env={"PYTHONPATH": "src", "A": "1"})

    assert wrapped.startswith("cd /tmp/work && env ")
    assert "A=1" in wrapped
    assert "PYTHONPATH=src" in wrapped
    assert wrapped.endswith("python -m example")


def test_python_here_doc_renders_delimited_script() -> None:
    command = python_here_doc("print('ok')", python_executable="python3")

    assert command.startswith("python3 - <<'PY'\n")
    assert "print('ok')" in command
    assert command.endswith("\nPY")


def test_remote_snapshot_and_restart_scripts_embed_expected_calls() -> None:
    snapshot = dashboard_snapshot_script(
        workspace_root="/home/zach/trademl-node",
        config_path="/home/zach/trademl-node/node.yml",
        env_path="/home/zach/trademl-node/.env",
    )
    restart = restart_node_script(
        workspace_root="/home/zach/trademl-node",
        config_path="/home/zach/trademl-node/node.yml",
        env_path="/home/zach/trademl-node/.env",
    )

    assert "collect_dashboard_snapshot" in snapshot
    assert "resolve_node_settings" in snapshot
    assert "/home/zach/trademl-node/node.yml" in snapshot
    assert "restart_node" in restart
    assert "/home/zach/trademl-node/.env" in restart
