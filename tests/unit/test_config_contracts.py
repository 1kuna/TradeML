from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_shipped_macmini_research_runs_on_the_macmini_not_self_ssh() -> None:
    config = yaml.safe_load((REPO_ROOT / "configs" / "research" / "perpetual_macmini.yml").read_text(encoding="utf-8"))

    assert config["target"] == "local"


def test_shipped_workstation_remote_uses_current_tailscale_target_without_baked_host_key() -> None:
    config = yaml.safe_load((REPO_ROOT / "configs" / "node.yml").read_text(encoding="utf-8"))
    target = config["training_targets"]["workstation-remote"]

    assert target["host"] == "100.102.98.14"
    assert "identity_file" not in target
    assert target["password_env"] == "TRADEML_WORKSTATION_REMOTE_PASSWORD"
