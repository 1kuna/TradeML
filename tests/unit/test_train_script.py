from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


def test_train_module_import_does_not_require_catboost_for_baseline_paths() -> None:
    module_path = Path(__file__).resolve().parents[2] / "src" / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("train_under_test", module_path)
    assert spec is not None
    assert spec.loader is not None

    poisoned_catboost = types.ModuleType("trademl.models.catboost")
    prior = sys.modules.get("trademl.models.catboost")
    sys.modules["trademl.models.catboost"] = poisoned_catboost
    try:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if prior is None:
            sys.modules.pop("trademl.models.catboost", None)
        else:
            sys.modules["trademl.models.catboost"] = prior

    assert callable(module.run_training)

