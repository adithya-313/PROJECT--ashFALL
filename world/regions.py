"""
WHAT: Geographical region count selection and uniform region assignment.
It draws the total number of municipal sectors from configured parameter intervals and maps entities.
Every person and shop in the simulation is assigned to a specific region to enable spatial tracking.

WHY: Real-world financial systems and fraud networks exhibit localized geographic patterns and regional crime syndicates.
Partitioning simulated entities into municipal sectors provides the spatial foundation needed for regional fraud modeling.
Uniform assignment establishes an unbiased baseline prior to introducing localized demographic clustering in Level 12.

ISSUE IDs: E3 (regions).
TERM: region: A discrete municipal district or geographic zone within the simulated town (indexed 0 to n_regions - 1). For example, dividing a city into North, South, Central, and East sectors.
TERM: discrete uniform distribution: A random selection where every integer outcome in a given range has equal probability. For example, rolling a fair 6-sided die.
"""

from typing import Any

import numpy as np


def draw_number_of_regions(
    entry: dict[str, Any],
    rng: np.random.Generator,
    range_name: str = "training",
) -> int:
    """
    WHAT: Samples the total count of municipal regions from the configured parameter interval.
    It reads the lower and upper integer limits for the requested split and samples an integer.
    The resulting count determines into how many geographic partitions the town is divided.

    WHY: Dynamic region counts prevent the simulation from overfitting to a fixed town topology.
    Testing under varying region counts verifies that fraud patterns generalize across city layouts.
    Configured intervals keep the synthetic town scale within realistic municipal boundaries.

    Args:
        entry: Parameter configuration dictionary containing split interval bounds.
        rng: Initialized NumPy random generator ensuring reproducible sampling.
        range_name: Name of the interval split to sample from, defaulting to 'training'.

    Returns:
        Integer count of municipal regions sampled within the configured interval.

    Example:
        # Sample the region count from a [3, 6] training interval:
        # entry = {"training": [3, 6]}
        # n = draw_number_of_regions(entry, np.random.default_rng(42))
        # assert n == 3
    """
    # Verify that the requested split exists in the number of regions configuration entry.
    if range_name not in entry:
        raise ValueError(f"Missing range split '{range_name}' for number_of_regions")

    bounds = entry[range_name]
    # Extract lower and upper integer bounds for the municipal sector count.
    low = int(bounds[0])
    high = int(bounds[1])

    # Draw an integer uniformly between low and high (inclusive).
    return int(rng.integers(low, high, endpoint=True))


def assign_region(n_regions: int, rng: np.random.Generator) -> int:
    """
    WHAT: Assigns a region index between 0 and n_regions - 1 with equal probability.
    It uses the provided random generator to draw an integer index uniformly across available zones.
    The resulting index anchors a person or merchant to a specific municipal district.

    WHY: All simulated participants must reside within a designated municipal territory for spatial tracking.
    In the absence of neighborhood density data, equal probability provides an unbiased neutral baseline.
    This clean separation allows future simulation phases to introduce non-uniform victim and mule clustering.

    Args:
        n_regions: Total count of municipal regions available in the simulated world.
        rng: Initialized NumPy random generator ensuring deterministic sampling.

    Returns:
        Integer index representing the assigned region, ranging from 0 to n_regions - 1.

    Example:
        # Assign a region index among 5 available sectors using seed 42:
        # reg = assign_region(5, np.random.default_rng(42))
        # assert reg == 3
    """
    # Validate that the world contains at least one valid municipal region.
    if n_regions <= 0:
        raise ValueError(f"Region count must be positive, got {n_regions}")

    # Draw a uniform random index from 0 to n_regions - 1.
    return int(rng.integers(0, n_regions))

