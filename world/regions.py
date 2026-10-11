"""
WHAT: Geographical region count selection and uniform region assignment.
WHY: Assigns every citizen and shop to one of several municipal sectors to support regional fraud dispersion.
ISSUE IDs: E3 (regions).
TERM: region: A discrete geographic sub-district within the simulated town (indexed 0 to n_regions - 1).
"""

from typing import Any

import numpy as np


def draw_number_of_regions(
    entry: dict[str, Any],
    rng: np.random.Generator,
    range_name: str = "training",
) -> int:
    """
    WHAT: Draws the total count of municipal regions in the world as an integer from the configured range.
    WHY: Determines the geographic partition count for the world based on the number_of_regions parameter.
    """
    if range_name not in entry:
        raise ValueError(f"Missing range split '{range_name}' for number_of_regions")

    bounds = entry[range_name]
    low = int(bounds[0])
    high = int(bounds[1])

    # Draw an integer uniformly between low and high (inclusive).
    return int(rng.integers(low, high, endpoint=True))


def assign_region(n_regions: int, rng: np.random.Generator) -> int:
    """
    WHAT: Assigns a region index between 0 and n_regions - 1 with equal probability.
    WHY: DECISION: Regions are drawn with equal chance because the config holds no unequal population shares.
         (Victim and mule clustering by geography will be introduced in Level 12.)
    """
    if n_regions <= 0:
        raise ValueError(f"Region count must be positive, got {n_regions}")

    # Draw a uniform random index from 0 to n_regions - 1.
    return int(rng.integers(0, n_regions))
