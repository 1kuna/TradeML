import os
from datetime import date
from pathlib import Path

from scripts.edge_collector import EdgeCollector


def test_edge_collector_local_manifest(tmp_path, monkeypatch):
    env = os.environ.copy()
    env.update(
        {
            "STORAGE_BACKEND": "local",
            "DATA_ROOT": str(tmp_path / "data"),
            "PARQUET_COMPRESSION": "zstd",
        }
    )
    # Ensure env is visible to EdgeCollector
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("DATA_ROOT", env["DATA_ROOT"])
    monkeypatch.setenv("PARQUET_COMPRESSION", "zstd")

    config_path = Path(__file__).resolve().parents[1] / "configs" / "edge.yml"
    collector = EdgeCollector(config_path=str(config_path))
    dstr = date.today().isoformat()
    collector._write_manifest("alpaca", "equities_bars", dstr, row_count=123)
    manifest_path = Path(env["DATA_ROOT"]) / "data_layer" / "manifests" / dstr / f"manifest-alpaca-equities_bars.jsonl"
    assert manifest_path.exists()
    content = manifest_path.read_text().strip().splitlines()
    assert content, "Manifest file was created but empty"
