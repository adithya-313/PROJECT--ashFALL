"""
WHAT: Container and loader for all four simulation configuration dictionaries.
It bundles world_ranges, background_realism, failure_realism, and device_realism into an immutable object.
It provides a single function to read the baseline configurations from disk using standard readers.

WHY: Allows simulation components to receive configuration objects directly instead of reading disk.
Passing configuration explicitly enables testing, layering, and reproducible runs with modified parameters.
Freezing the container guarantees loaded configuration dictionaries are not accidentally modified.

TERM: world configs: The complete set of four configuration files governing a simulation run.
      For example, holding ranges for economy, background rhythms, failures, and devices.
TERM: configuration name: The canonical identifier for one of the four domain configuration files.
      For example, 'world_ranges' or 'background_realism'.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from world.background_config import read_background_realism
from world.config_reader import read_world_ranges

CONFIG_NAMES: tuple[str, ...] = (
    "world_ranges",
    "background_realism",
    "failure_realism",
    "device_realism",
)


@dataclass(frozen=True)
class WorldConfigs:
    """
    WHAT: Holds the four domain configuration dictionaries required to build and simulate a world.
    It encapsulates world ranges, background realism, failure realism, and device realism.
    Fast mode and builders accept this structure to allow layered configuration overrides.

    WHY: Decouples simulation execution from hardcoded filesystem configuration paths.
    Freezing the container prevents accidental modification of loaded configuration data during a run.
    It provides a single typed object to pass between builders, planners, and runners.

    Args:
        world_ranges: Dictionary loaded from config/world_ranges.yaml.
        background_realism: Dictionary loaded from config/background_realism.yaml.
        failure_realism: Dictionary loaded from config/failure_realism.yaml.
        device_realism: Dictionary loaded from config/device_realism.yaml.

    Returns:
        WorldConfigs instance holding all four domain configuration dictionaries.

    Example:
        >>> from world.world_configs import read_world_configs
        >>> configs = read_world_configs()
        >>> len(configs.world_ranges["parameters"])
        79
        >>> len(configs.background_realism["parameters"])
        17
    """

    world_ranges: dict[str, Any]
    background_realism: dict[str, Any]
    failure_realism: dict[str, Any]
    device_realism: dict[str, Any]


def read_world_configs() -> WorldConfigs:
    """
    WHAT: Reads the four standard configuration files from disk into an immutable WorldConfigs container.
    It loads world ranges, background realism, failure realism, and device realism using their standard readers.
    The resulting structure contains all parameters required for a complete simulation run.

    WHY: Provides a single authoritative entry point for loading the complete set of simulation configurations.
    Standardizing file paths and reader invocations prevents inconsistent configuration loading across modules.
    It serves as the baseline configuration state before any custom layers are applied.

    Args:
        None.

    Returns:
        WorldConfigs instance containing the four loaded configuration dictionaries.

    Example:
        >>> from world.world_configs import read_world_configs, CONFIG_NAMES
        >>> configs = read_world_configs()
        >>> len(CONFIG_NAMES)
        4
        >>> sorted(configs.failure_realism["parameters"].keys())[:2]
        ['bank_incident_daily_chance', 'bank_incident_duration_hours']
    """
    # 1. Read master world ranges from config/world_ranges.yaml via standard reader.
    world_ranges = read_world_ranges()

    # 2. Read background temporal rhythm settings from config/background_realism.yaml.
    background_realism = read_background_realism()

    # 3. Read failure and incident settings from config/failure_realism.yaml.
    failure_realism = read_background_realism("config/failure_realism.yaml")

    # 4. Read device lifecycle settings from config/device_realism.yaml.
    device_realism = read_background_realism(Path("config/device_realism.yaml"))

    # 5. Package all four configuration dictionaries into an immutable container.
    return WorldConfigs(
        world_ranges=world_ranges,
        background_realism=background_realism,
        failure_realism=failure_realism,
        device_realism=device_realism,
    )
