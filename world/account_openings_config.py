"""
Configuration resolver and procedural scheduler for genuine account openings.
Loads assumption ranges from config/account_openings.yaml and schedules synthetic
civilian person and merchant shop openings across the simulation calendar.
"""

from pathlib import Path
from typing import Any

import numpy as np

from core.config_validator import load_yaml
from world.outside_events import OutsideEvent
from world.run_identity import REPO_ROOT

DEFAULT_ACCOUNT_OPENINGS_PATH: Path = REPO_ROOT / "config" / "account_openings.yaml"


def load_account_openings_config(config_path: Path | None = None) -> dict[str, Any]:
    """
    Loads and parses the account openings YAML configuration file.

    Retrieves calibrated assumption bounds for civilian account creation rates
    and shop opening shares across training, test, and stress evaluation splits.
    """
    path = config_path if config_path is not None else DEFAULT_ACCOUNT_OPENINGS_PATH
    if not path.exists():
        raise FileNotFoundError(f"Account openings config not found: {path}")
    return load_yaml(path)


def resolve_opening_rate_range(
    split: str = "training", config: dict[str, Any] | None = None
) -> tuple[float, float]:
    """
    Extracts the lower and upper bounds for genuine account openings per 1000 people per 30 days.

    Inspects the specified configuration split and returns the rate interval as floating-point numbers.
    """
    cfg = config if config is not None else load_account_openings_config()
    params = cfg.get("parameters", {})
    entry = params.get("genuine_account_openings_per_1000_people_per_30_days", {})
    if split not in entry:
        raise ValueError(f"Split '{split}' not found for genuine_account_openings_per_1000_people_per_30_days")
    bounds = entry[split]
    return float(bounds[0]), float(bounds[1])


def resolve_shop_share_range(
    split: str = "training", config: dict[str, Any] | None = None
) -> tuple[float, float]:
    """
    Extracts the lower and upper bounds for the proportion of mid-run openings that are commercial shops.

    Inspects the specified configuration split and returns the shop share fraction bounds.
    """
    cfg = config if config is not None else load_account_openings_config()
    params = cfg.get("parameters", {})
    entry = params.get("share_of_openings_that_are_shops", {})
    if split not in entry:
        raise ValueError(f"Split '{split}' not found for share_of_openings_that_are_shops")
    bounds = entry[split]
    return float(bounds[0]), float(bounds[1])


def schedule_genuine_account_openings(
    n_people: int,
    n_days: int,
    split: str,
    rng: np.random.Generator,
    config: dict[str, Any] | None = None,
) -> tuple[OutsideEvent, ...]:
    """
    Procedurally schedules genuine outside open_account events using random stream 8.

    Samples the opening rate and shop fraction within calibrated split intervals, scales the
    monthly count to the population and duration, and positions events across simulation minutes.
    """
    min_rate, max_rate = resolve_opening_rate_range(split, config)
    rate = float(rng.uniform(min_rate, max_rate))

    min_share, max_share = resolve_shop_share_range(split, config)
    shop_share = float(rng.uniform(min_share, max_share))

    expected_openings = (rate / 1000.0) * n_people * (n_days / 30.0)
    total_count = int(round(expected_openings))

    total_minutes = n_days * 1440
    events: list[OutsideEvent] = []
    for _ in range(total_count):
        minute = int(rng.integers(0, total_minutes))
        is_shop = float(rng.uniform(0.0, 1.0)) < shop_share
        owner_kind = "shop" if is_shop else "person"
        events.append(
            OutsideEvent(
                minute=minute,
                kind="open_account",
                owner_kind=owner_kind,
                opening_stream="genuine_account_openings",
            )
        )

    # Sort scheduled openings chronologically by minute.
    events.sort(key=lambda ev: ev.minute)
    return tuple(events)
