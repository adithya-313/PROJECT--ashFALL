"""
WHAT: Config loader and bounded random number drawer for world generation.
It parses configuration files containing research-backed statistical intervals and extracts entries.
It also provides safe bounded uniform sampling utilities used across all world initialization routines.

WHY: Simulation parameters must be grounded in explicit empirical assumptions and research literature.
Centralizing configuration loading avoids hardcoded magic numbers scattered across domain files.
Controlled bounded draws ensure reproducible experiments across training, test, and stress evaluation splits.

TERM: parameter range: A lower and upper boundary [min, max] defining allowed values for a simulation setting.
      For example, a town having between 3 and 6 regions.
TERM: configuration split: An evaluation category partition (such as training, test, or stress) that groups
      parameter intervals. For example, testing a model under wider stress bounds.
TERM: uniform distribution: A random sampling where every value between the minimum and maximum has an equal
      likelihood of being chosen. For example, spinning an evenly balanced spinner.
"""

from pathlib import Path
from typing import Any

import numpy as np

from core.config_validator import load_yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WORLD_RANGES_PATH = REPO_ROOT / "config" / "world_ranges.yaml"


def read_world_ranges(config_path: Path | str | None = None) -> dict[str, Any]:
    """
    WHAT: Reads the world ranges YAML configuration file and parses it into a nested Python dictionary.
    If no custom path is provided, it defaults to config/world_ranges.yaml located at the repository root.
    It verifies file presence on disk, raising FileNotFoundError if the path cannot be found.

    WHY: Provides the master catalog of research-backed and assumption ranges for synthetic world generation.
    Loading from an explicit YAML schema ensures that experimental boundaries remain transparent and auditable.
    A single entry point guarantees all simulation modules access identical baseline distributions.

    Args:
        config_path: Optional filesystem path or string pointing to the YAML configuration file.

    Returns:
        Nested dictionary containing parameter keys, metadata annotations, and split interval boundaries.

    Example:
        # Load the default configuration and verify top-level sections exist:
        # cfg = read_world_ranges()
        # assert "parameters" in cfg
    """
    # Resolve default YAML configuration path if no custom path was provided.
    path = Path(config_path) if config_path is not None else DEFAULT_WORLD_RANGES_PATH
    # Verify that the target configuration file exists before attempting to load it.
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at path: {path}")
    # Parse the YAML file into structured Python dictionaries using the safe loader.
    data = load_yaml(path)
    return data


def get_parameter_entry(ranges_dict: dict[str, Any], param_name: str) -> dict[str, Any]:
    """
    WHAT: Retrieves the specific parameter configuration block for a given parameter name.
    It extracts the requested dictionary from the top-level 'parameters' mapping within the ranges data.
    If the specified parameter name is not present in the configuration, it raises a descriptive ValueError.

    WHY: Downstream world generation routines require calibrated bounds to initialize agent distributions.
    Early validation prevents subtle missing-key crashes deep inside stochastic simulation loops.
    Explicit error messages immediately notify developers of configuration key misspellings.

    Args:
        ranges_dict: Master dictionary loaded from the world ranges YAML configuration.
        param_name: String identifier representing the target parameter entry to retrieve.

    Returns:
        Dictionary containing metadata, unit types, and split intervals for the named parameter.

    Example:
        # Retrieve the region count configuration block:
        # cfg = read_world_ranges()
        # entry = get_parameter_entry(cfg, "number_of_regions")
        # assert entry["training"] == [3, 6]
    """
    # Extract the top-level parameters dictionary or default to an empty mapping.
    parameters = ranges_dict.get("parameters", {})
    # Verify that the requested parameter exists in the dictionary.
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
    WHAT: Draws one floating point number uniformly at random inside the named range split of an entry.
    It unpacks the two-element lower and upper boundary list for the requested evaluation split.
    The value is sampled continuously between the boundaries using the provided random generator.

    WHY: Stochastic simulation parameters must vary across runs within their allowed empirical bands.
    Using explicit NumPy random generators guarantees deterministic repeatability across experiment runs.
    Validating interval ordering protects against malformed configuration data producing inverted samples.

    Args:
        entry: Parameter configuration dictionary containing split interval lists.
        rng: Initialized NumPy random generator for reproducible sampling.
        range_name: String name of the interval split to sample from, defaulting to 'training'.
        param_name: String label used in diagnostic error messages when validation fails.

    Returns:
        Floating-point number sampled uniformly between the lower and upper bounds.

    Example:
        # Sample uniformly from a [3.0, 6.0] interval using a fixed seed:
        # entry = {"training": [3.0, 6.0]}
        # val = draw_in_range(entry, np.random.default_rng(42), range_name="training")
        # assert round(val, 4) == 5.3219
    """
    # Ensure the requested evaluation split exists in this parameter entry.
    if range_name not in entry:
        raise ValueError(
            f"Missing range split '{range_name}' in config parameter '{param_name}'"
        )

    # Validate that the bounds specification is a two-element list or tuple.
    bounds = entry[range_name]
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        raise ValueError(
            f"Malformed range for '{param_name}' in split '{range_name}': expected 2 elements, got {bounds}"
        )

    # Extract lower and upper bounds as floating-point values and verify valid ordering.
    low = float(bounds[0])
    high = float(bounds[1])
    if low > high:
        raise ValueError(
            f"Malformed range bounds for '{param_name}': min ({low}) cannot exceed max ({high})"
        )

    # Draw uniformly between the validated lower and upper bounds.
    return float(rng.uniform(low, high))
