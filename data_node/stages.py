"""
Stage gating for the unified Pi data-node.

Implements:
- Stage configuration (universe size, history depth)
- Stage 0→1 promotion based on GREEN coverage threshold
- Bootstrap task seeding for new stages

See updated_node_spec.md §4 for stage semantics.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import yaml
from loguru import logger

from .db import NodeDB, TaskKind, get_db


@dataclass
class StageDefinition:
    """Definition of a single stage."""
    name: str
    universe_size: int
    equities_eod_years: int
    equities_minute_years: int
    green_threshold: float = 0.98


@dataclass
class StageConfig:
    """Current stage configuration state."""
    current_stage: int
    promoted_at: Optional[datetime]
    stages: dict[int, StageDefinition]


# Default stage definitions per spec §4
DEFAULT_STAGES = {
    0: StageDefinition(
        name="bootstrap_small",
        universe_size=100,
        equities_eod_years=5,
        equities_minute_years=1,
        green_threshold=0.98,
    ),
    1: StageDefinition(
        name="full",
        universe_size=500,
        equities_eod_years=15,
        equities_minute_years=5,
        green_threshold=0.98,
    ),
}


def _get_stage_config_path() -> Path:
    """Get path to stage.yml config file."""
    data_root = os.environ.get("DATA_ROOT", ".")
    return Path(data_root) / "data_layer" / "control" / "stage.yml"


def _get_universe_path() -> Path:
    """Get path to universe symbols file."""
    return Path("data_layer/reference/universe_symbols.txt")


def load_stage_config() -> StageConfig:
    """
    Load stage configuration from stage.yml.

    Creates default Stage 0 config if file doesn't exist.

    Returns:
        StageConfig with current stage and definitions
    """
    config_path = _get_stage_config_path()

    if not config_path.exists():
        logger.info("No stage.yml found, initializing with Stage 0")
        config = StageConfig(
            current_stage=0,
            promoted_at=None,
            stages=DEFAULT_STAGES.copy(),
        )
        save_stage_config(config)
        return config

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    # Parse stage definitions
    stages = {}
    for stage_num, stage_data in raw.get("stages", {}).items():
        stages[int(stage_num)] = StageDefinition(
            name=stage_data.get("name", f"stage_{stage_num}"),
            universe_size=stage_data.get("universe_size", 100),
            equities_eod_years=stage_data.get("equities_eod_years", 5),
            equities_minute_years=stage_data.get("equities_minute_years", 1),
            green_threshold=stage_data.get("green_threshold", 0.98),
        )

    # Use defaults for any missing stages
    for stage_num, default_def in DEFAULT_STAGES.items():
        if stage_num not in stages:
            stages[stage_num] = default_def

    # Parse promoted_at timestamp
    promoted_at = None
    if raw.get("promoted_at"):
        try:
            promoted_at = datetime.fromisoformat(raw["promoted_at"])
        except (ValueError, TypeError):
            pass

    return StageConfig(
        current_stage=raw.get("current_stage", 0),
        promoted_at=promoted_at,
        stages=stages,
    )


def save_stage_config(config: StageConfig) -> None:
    """
    Save stage configuration to stage.yml.

    Uses atomic write to prevent corruption.
    """
    config_path = _get_stage_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Build YAML structure
    data = {
        "current_stage": config.current_stage,
        "promoted_at": config.promoted_at.isoformat() if config.promoted_at else None,
        "stages": {},
    }

    for stage_num, stage_def in config.stages.items():
        data["stages"][stage_num] = {
            "name": stage_def.name,
            "universe_size": stage_def.universe_size,
            "equities_eod_years": stage_def.equities_eod_years,
            "equities_minute_years": stage_def.equities_minute_years,
            "green_threshold": stage_def.green_threshold,
        }

    # Atomic write
    tmp_path = config_path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    tmp_path.rename(config_path)

    logger.debug(f"Saved stage config to {config_path}")


def load_universe_symbols() -> list[str]:
    """
    Load all symbols from universe_symbols.txt.

    Returns:
        List of symbols in priority order
    """
    universe_path = _get_universe_path()

    if not universe_path.exists():
        logger.warning(f"Universe file not found: {universe_path}")
        return []

    symbols = []
    with open(universe_path) as f:
        for line in f:
            sym = line.strip()
            if sym and not sym.startswith("#"):
                symbols.append(sym)

    return symbols


def get_current_universe() -> list[str]:
    """
    Get the current universe based on current stage.

    Returns:
        List of symbols (first N from universe_symbols.txt based on stage)
    """
    config = load_stage_config()
    stage_def = config.stages.get(config.current_stage)

    if stage_def is None:
        logger.warning(f"Unknown stage {config.current_stage}, using Stage 0 defaults")
        stage_def = DEFAULT_STAGES[0]

    all_symbols = load_universe_symbols()
    return all_symbols[:stage_def.universe_size]


def get_date_range(dataset: str) -> tuple[date, date]:
    """
    Get the date range for a dataset based on current stage.

    Args:
        dataset: Dataset name (equities_eod, equities_minute, etc.)

    Returns:
        Tuple of (start_date, end_date) for the dataset
    """
    config = load_stage_config()
    stage_def = config.stages.get(config.current_stage)

    if stage_def is None:
        stage_def = DEFAULT_STAGES[0]

    today = date.today()

    if dataset == "equities_eod":
        years = stage_def.equities_eod_years
    elif dataset == "equities_minute":
        years = stage_def.equities_minute_years
    else:
        # Default to EOD range for other datasets
        years = stage_def.equities_eod_years

    start_date = today - timedelta(days=years * 365)
    return (start_date, today)


def get_extended_date_range(
    dataset: str,
    new_stage: int,
    old_stage: int,
) -> tuple[date, date]:
    """
    Get the extended date range when promoting to a new stage.

    Returns the date range for history that wasn't covered in the old stage.

    Args:
        dataset: Dataset name
        new_stage: Stage being promoted to
        old_stage: Previous stage

    Returns:
        Tuple of (extension_start, extension_end) for additional history
    """
    config = load_stage_config()

    old_def = config.stages.get(old_stage, DEFAULT_STAGES.get(old_stage))
    new_def = config.stages.get(new_stage, DEFAULT_STAGES.get(new_stage))

    if old_def is None or new_def is None:
        raise ValueError(f"Invalid stage transition: {old_stage} -> {new_stage}")

    today = date.today()

    if dataset == "equities_eod":
        old_years = old_def.equities_eod_years
        new_years = new_def.equities_eod_years
    elif dataset == "equities_minute":
        old_years = old_def.equities_minute_years
        new_years = new_def.equities_minute_years
    else:
        old_years = old_def.equities_eod_years
        new_years = new_def.equities_eod_years

    # Extension is from new_start to old_start
    new_start = today - timedelta(days=new_years * 365)
    old_start = today - timedelta(days=old_years * 365)

    return (new_start, old_start - timedelta(days=1))


def check_promotion(db: Optional[NodeDB] = None) -> bool:
    """
    Check if current stage should be promoted to next stage.

    Promotion criteria:
    - Both equities_eod and equities_minute have GREEN coverage >= threshold
    - Next stage exists

    Args:
        db: Database instance (default: get_db())

    Returns:
        True if promotion occurred
    """
    if db is None:
        db = get_db()

    config = load_stage_config()
    current_stage = config.current_stage
    next_stage = current_stage + 1

    # Check if next stage exists
    if next_stage not in config.stages:
        logger.debug(f"No Stage {next_stage} defined, cannot promote")
        return False

    current_def = config.stages[current_stage]

    # Get current universe
    universe = get_current_universe()

    # Check coverage for both datasets
    start_eod, end_eod = get_date_range("equities_eod")
    start_min, end_min = get_date_range("equities_minute")

    coverage_eod = db.get_green_coverage(
        table_name="equities_eod",
        symbols=universe,
        start_date=start_eod,
        end_date=end_eod,
    )

    coverage_min = db.get_green_coverage(
        table_name="equities_minute",
        symbols=universe,
        start_date=start_min,
        end_date=end_min,
    )

    logger.info(
        f"Stage {current_stage} coverage check: "
        f"equities_eod={coverage_eod:.2%}, "
        f"equities_minute={coverage_min:.2%}, "
        f"threshold={current_def.green_threshold:.2%}"
    )

    # Check threshold
    if coverage_eod < current_def.green_threshold:
        logger.debug(f"equities_eod coverage {coverage_eod:.2%} below threshold")
        return False

    if coverage_min < current_def.green_threshold:
        logger.debug(f"equities_minute coverage {coverage_min:.2%} below threshold")
        return False

    # Promote!
    logger.info(f"Promoting from Stage {current_stage} to Stage {next_stage}")

    config.current_stage = next_stage
    config.promoted_at = datetime.utcnow()
    save_stage_config(config)

    # Seed bootstrap tasks for new stage
    seed_bootstrap_tasks(next_stage, current_stage, db)

    return True


def seed_bootstrap_tasks(
    stage: int,
    previous_stage: Optional[int] = None,
    db: Optional[NodeDB] = None,
) -> int:
    """
    Seed BOOTSTRAP tasks for a new stage.

    For initial stage (no previous), seeds all symbols × full date range.
    For stage promotion, seeds:
    - Extended history for existing symbols
    - Full history for new symbols

    Args:
        stage: Stage to seed for
        previous_stage: Previous stage (None for initial bootstrap)
        db: Database instance

    Returns:
        Number of tasks created
    """
    if db is None:
        db = get_db()

    config = load_stage_config()
    stage_def = config.stages.get(stage)

    if stage_def is None:
        logger.error(f"Unknown stage {stage}, cannot seed")
        return 0

    # Get universe for this stage
    all_symbols = load_universe_symbols()
    stage_symbols = all_symbols[:stage_def.universe_size]

    # Datasets to bootstrap
    datasets = ["equities_eod", "equities_minute"]

    tasks_created = 0

    for dataset in datasets:
        if previous_stage is None:
            # Initial bootstrap - full range for all symbols
            start_date, end_date = get_date_range(dataset)
            symbols_to_seed = stage_symbols

            logger.info(
                f"Seeding initial {dataset} bootstrap: "
                f"{len(symbols_to_seed)} symbols × {start_date} to {end_date}"
            )

            for symbol in symbols_to_seed:
                task_id = db.enqueue_task(
                    dataset=dataset,
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    kind=TaskKind.BOOTSTRAP,
                    priority=0,  # Highest priority for initial bootstrap
                )
                if task_id:
                    tasks_created += 1
        else:
            # Stage promotion
            prev_def = config.stages.get(previous_stage, DEFAULT_STAGES.get(previous_stage))
            if prev_def is None:
                continue

            prev_symbols = all_symbols[:prev_def.universe_size]
            new_symbols = [s for s in stage_symbols if s not in prev_symbols]

            # Extended history for existing symbols
            try:
                ext_start, ext_end = get_extended_date_range(dataset, stage, previous_stage)
                if ext_start < ext_end:
                    logger.info(
                        f"Seeding {dataset} history extension: "
                        f"{len(prev_symbols)} symbols × {ext_start} to {ext_end}"
                    )

                    for symbol in prev_symbols:
                        task_id = db.enqueue_task(
                            dataset=dataset,
                            symbol=symbol,
                            start_date=ext_start,
                            end_date=ext_end,
                            kind=TaskKind.BOOTSTRAP,
                            priority=1,
                        )
                        if task_id:
                            tasks_created += 1
            except ValueError:
                pass

            # Full history for new symbols
            if new_symbols:
                start_date, end_date = get_date_range(dataset)
                logger.info(
                    f"Seeding {dataset} for new symbols: "
                    f"{len(new_symbols)} symbols × {start_date} to {end_date}"
                )

                for symbol in new_symbols:
                    task_id = db.enqueue_task(
                        dataset=dataset,
                        symbol=symbol,
                        start_date=start_date,
                        end_date=end_date,
                        kind=TaskKind.BOOTSTRAP,
                        priority=1,
                    )
                    if task_id:
                        tasks_created += 1

    logger.info(f"Seeded {tasks_created} BOOTSTRAP tasks for Stage {stage}")
    return tasks_created


def get_current_stage() -> int:
    """Get the current stage number."""
    return load_stage_config().current_stage


def get_stage_info() -> dict:
    """
    Get info about current stage for status display.

    Returns:
        Dict with stage info
    """
    config = load_stage_config()
    stage_def = config.stages.get(config.current_stage, DEFAULT_STAGES[0])

    return {
        "current_stage": config.current_stage,
        "name": stage_def.name,
        "universe_size": stage_def.universe_size,
        "equities_eod_years": stage_def.equities_eod_years,
        "equities_minute_years": stage_def.equities_minute_years,
        "green_threshold": stage_def.green_threshold,
        "promoted_at": config.promoted_at.isoformat() if config.promoted_at else None,
    }
