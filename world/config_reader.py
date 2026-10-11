"""
WHAT: Config loader and bounded random number drawer for world generation.
WHY: Reads pre-registered parameter intervals from config/world_ranges.yaml and draws values uniformly inside them.
TERM: parameter range: A lower and upper boundary [min, max] defining allowed values for a simulation setting.
"""

from pathlib import Path
from typing import Any
import numpy as np

from core.config_validator import load_yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WORLD_RANGES_PATH = REPO_ROOT / "config" / "world_ranges.yaml"


def read_world_ranges(config_path: Path | str | None = None) -> dict[str, Any]:
    """
    WHAT: Reads the world ranges YAML file and returns the full dictionary.
    WHY: Provides the master catalog of research-backed and assumption ranges for world generation.
    """
    path = Path(config_path) if config_path is not None else DEFAULT_WORLD_RANGES_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at path: {path}")
    data = load_yaml(path)
    return data


def get_parameter_entry(ranges_dict: dict[str, Any], param_name: str) -> dict[str, Any]:
    """
    WHAT: Retrieves the parameter block for a given parameter name.
    WHY: Validates that the requested parameter exists, raising a clear error if missing or misspelled.
    """
    parameters = ranges_dict.get("parameters", {})
    if param_name not in parameters:
        raise ValueError(f"Unknown config parameter key: '{param_name}'")
    return parameters[param_name]


def draw_in_range(
    entry: dict[str, Any],
    rng: np.random.Generator,
    range_name: str = "training",
    param_name: str = "unnamed_parameter",
) -> float:
    """
    WHAT: Draws one floating point number uniformly at random inside the named range of an entry.
    WHY: All stochastic simulation parameters must vary across runs within their allowed empirical bands.
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

    low = float(bounds[0])
    high = float(bounds[1])
    if low > high:
        raise ValueError(
            f"Malformed range bounds for '{param_name}': min ({low}) cannot exceed max ({high})"
        )

    # Draw uniformly between the lower and upper bounds.
    return float(rng.uniform(low, high))
