"""
WHAT: Config loader and integer drawer for background realism parameters.
WHY: Loads festival, calendar, and diurnal rhythm intervals from config/background_realism.yaml,
     allowing existing config_reader functions to be reused seamlessly without duplication.
ISSUE IDs: R2 (background realism configuration).
TERM: background realism: The empirical and assumed parameter intervals controlling calendar and diurnal cycles.
"""

from pathlib import Path
from typing import Any

import numpy as np

from core.config_validator import load_yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BACKGROUND_REALISM_PATH = REPO_ROOT / "config" / "background_realism.yaml"

EXPECTED_BACKGROUND_KEYS = [
    "festival_peak_day_in_world",
    "festival_uplift_fraction",
    "festival_rise_days",
    "festival_fade_days",
    "other_festival_count",
    "other_festival_uplift_fraction",
    "other_festival_rise_fade_days",
    "month_start_uplift_fraction",
    "month_start_easing_days",
    "weekend_difference_fraction",
    "night_activity_floor",
    "daily_peak_width_hours",
    "peak_hour_salaried_employee",
    "peak_hour_student",
    "peak_hour_retiree",
    "peak_hour_small_shopkeeper",
    "peak_hour_gig_worker",
]


def read_background_realism(config_path: Path | str | None = None) -> dict[str, Any]:
    """
    WHAT: Reads config/background_realism.yaml and returns the dictionary containing all parameters.
    WHY: Supplies pre-registered background rhythm intervals to world simulation generators.
    """
    path = Path(config_path) if config_path is not None else DEFAULT_BACKGROUND_REALISM_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at path: {path}")

    # Load yaml document using the shared core validator loader.
    data = load_yaml(path)
    return data


def draw_whole_number(
    entry: dict[str, Any],
    rng: np.random.Generator,
    range_name: str = "training",
    param_name: str = "unnamed_parameter",
) -> int:
    """
    WHAT: Draws one integer uniformly at random within the named interval [low, high] of an entry.
    WHY: Samples discrete counts (such as festival days or easing durations) without leaving bounds.
    """
    if range_name not in entry:
        raise ValueError(
            f"Missing range split '{range_name}' in config parameter '{param_name}'"
        )

    bounds = entry[range_name]
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        raise ValueError(
            f"Malformed range for '{param_name}' in split '{range_name}': expected 2 elements, got {bounds}"
        )

    low = int(bounds[0])
    high = int(bounds[1])
    if low > high:
        raise ValueError(
            f"Malformed range bounds for '{param_name}': min ({low}) cannot exceed max ({high})"
        )

    # Draw integer uniformly within the inclusive bounds [low, high].
    return int(rng.integers(low, high, endpoint=True))
