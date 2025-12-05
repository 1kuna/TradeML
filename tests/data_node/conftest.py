"""
Pytest fixtures for data_node tests.
"""

# IMPORTANT: Path setup must be at the very top, before any other imports
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Now safe to import other modules
import os
import tempfile

import pytest


@pytest.fixture
def temp_data_root(tmp_path):
    """Create a temporary data root directory."""
    data_root = tmp_path / "data_layer"
    data_root.mkdir(parents=True)

    # Create reference directory with mock universe
    ref_dir = data_root / "reference"
    ref_dir.mkdir()

    universe = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "V", "JNJ"]
    (ref_dir / "universe_symbols.txt").write_text("\n".join(universe))

    # Set environment
    old_root = os.environ.get("DATA_ROOT")
    os.environ["DATA_ROOT"] = str(tmp_path)

    yield tmp_path

    # Restore
    if old_root:
        os.environ["DATA_ROOT"] = old_root
    else:
        os.environ.pop("DATA_ROOT", None)


@pytest.fixture
def node_db(temp_data_root):
    """Create a fresh NodeDB instance."""
    from data_node.db import NodeDB, reset_db

    # Reset singleton
    reset_db()

    db = NodeDB()
    db.init_db()

    yield db

    db.close()
    reset_db()


@pytest.fixture
def budget_manager(temp_data_root):
    """Create a fresh BudgetManager instance with test-friendly config.

    Uses small daily caps (100) for fast tests, and very high RPM (10000)
    to ensure daily budget slices (not RPM) are the limiting factor.
    """
    import yaml
    from data_node.budgets import BudgetManager, reset_budget_manager

    # Reset singleton
    reset_budget_manager()

    # Create test config with rpm=daily_cap for simplicity:
    # - Slice tests: RPM matches daily_cap so never hits RPM limit first
    # - RPM tests: can_spend returns False (either RPM or daily limit)
    test_config = {
        "policy": {
            "rate_limits": {
                "alpaca": {"hard_rpm": 200, "soft_daily_cap": 200},
                "finnhub": {"hard_rpm": 200, "soft_daily_cap": 200},
                "av": {"hard_rpm": 200, "soft_daily_cap": 200},
                "fred": {"hard_rpm": 200, "soft_daily_cap": 200},
                "fmp": {"hard_rpm": 200, "soft_daily_cap": 200},
                "polygon": {"hard_rpm": 200, "soft_daily_cap": 200},
            }
        }
    }

    config_path = temp_data_root / "test_backfill.yml"
    with open(config_path, "w") as f:
        yaml.dump(test_config, f)

    manager = BudgetManager(config_path=config_path)

    yield manager

    reset_budget_manager()


@pytest.fixture
def budget_manager_rpm(temp_data_root):
    """BudgetManager fixture optimized for RPM tests.

    Uses daily_cap >> rpm so RPM can be tested in isolation.
    """
    import yaml
    from data_node.budgets import BudgetManager, reset_budget_manager

    reset_budget_manager()

    # Low RPM, high daily_cap for testing RPM refills without hitting daily limit
    test_config = {
        "policy": {
            "rate_limits": {
                "alpaca": {"hard_rpm": 60, "soft_daily_cap": 10000},
                "finnhub": {"hard_rpm": 60, "soft_daily_cap": 10000},
            }
        }
    }

    config_path = temp_data_root / "test_rpm_backfill.yml"
    with open(config_path, "w") as f:
        yaml.dump(test_config, f)

    manager = BudgetManager(config_path=config_path)
    yield manager
    reset_budget_manager()


@pytest.fixture
def mock_universe_path(temp_data_root):
    """Patch the universe path for stages module."""
    import data_node.stages as stages_mod

    ref_dir = temp_data_root / "data_layer" / "reference"
    original = stages_mod._get_universe_path
    stages_mod._get_universe_path = lambda: ref_dir / "universe_symbols.txt"

    yield ref_dir / "universe_symbols.txt"

    stages_mod._get_universe_path = original
