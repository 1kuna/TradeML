import io
import logging
import os
import subprocess

import rpi_wizard


def _logger_with_stream(name: str):
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    logger.addHandler(handler)
    return logger, stream, handler


def _cleanup_logger(logger: logging.Logger, handler: logging.Handler):
    logger.removeHandler(handler)
    handler.close()


def test_setup_logging_recovers_broken_log_dir(tmp_path):
    log_dir = tmp_path / "logs"
    broken_target = tmp_path / "missing" / "logs"
    log_dir.symlink_to(broken_target)

    logger, log_file = rpi_wizard.setup_logging(log_dir)
    logger.info("hello")
    assert log_dir.is_dir()
    assert not log_dir.is_symlink()
    assert log_file.exists()

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def test_ensure_venv_rebuilds_invalid_python(tmp_path):
    venv_path = tmp_path / "venv"
    bin_dir = venv_path / "bin"
    bin_dir.mkdir(parents=True)
    fake_python = bin_dir / "python"
    fake_python.write_text("#!/bin/sh\nexit 1\n")
    fake_python.chmod(0o755)

    logger, stream, handler = _logger_with_stream("test_rpi_wizard_venv")
    rebuilt_path = rpi_wizard.ensure_venv(venv_path, logger)
    assert rebuilt_path == venv_path

    subprocess.run([str(venv_path / "bin" / "python"), "-c", "print('ok')"], check=True)
    logs = stream.getvalue()
    assert "Rebuilding venv" in logs or "Created venv" in logs
    _cleanup_logger(logger, handler)


def test_upsert_env_deduplicates(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "FOO=1\nBAR=2\nFOO=old\n# comment\nBAZ=3\n"
    )
    logger, stream, handler = _logger_with_stream("test_rpi_wizard_env")
    rpi_wizard.upsert_env(env_path, {"FOO": "new", "BAR": "2", "QUX": "4"}, logger)
    contents = env_path.read_text().strip().splitlines()
    assert contents[0] == "FOO=new"
    assert "BAR=2" in contents
    assert "BAZ=3" in contents
    assert "QUX=4" in contents
    assert all(contents.count(line) == 1 for line in contents)
    _cleanup_logger(logger, handler)


def test_node_resolves_bad_data_root(monkeypatch, tmp_path):
    bad_root = "/Users/not-here/data"
    env = os.environ.copy()
    env["DATA_ROOT"] = bad_root
    # We can't import scripts.node directly because loguru writes during import; instead run resolve helper
    from importlib import reload
    import scripts.node as node

    reload(node)
    with monkeypatch.context() as m:
        m.setenv("DATA_ROOT", bad_root)
        m.setattr(node.sys, "platform", "linux")
        data_root, warnings = node._resolve_data_root()
    assert data_root.exists()
    assert data_root.is_dir()
    assert any("macOS-style" in w for w in warnings)


def test_sanitize_prev_state_drops_incompatible_paths(monkeypatch):
    logger, stream, handler = _logger_with_stream("test_prev_state")
    prev = {
        "data_root": "/Users/bad/path",
        "venv_path": "/private/tmp/venv",
        "edge_node_id": "prev-node",
    }
    with monkeypatch.context() as m:
        m.setattr(rpi_wizard.sys, "platform", "linux")
        sanitized = rpi_wizard._sanitize_prev_state(prev, logger)
    assert "data_root" not in sanitized
    assert "venv_path" not in sanitized
    assert sanitized.get("edge_node_id") == "prev-node"
    logs = stream.getvalue()
    assert "Ignoring incompatible path" in logs
    _cleanup_logger(logger, handler)
