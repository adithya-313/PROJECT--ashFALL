"""
WHAT: Device hardware and operating system profile modeling for simulated individuals.
WHY: Simulates client device characteristics (platform, OS version generation, and hardware fingerprint)
     so fraud rules can examine device patterns without falsely penalizing legitimate OS updates.
TERM: device fingerprint: A synthetic pseudo-hardware identifier uniquely distinguishing a client phone.
TERM: OS generation: An integer rating (1 to OS_GENERATIONS) representing older versus newer operating system releases.
"""

from dataclasses import dataclass

import numpy as np

from world.placeholders import ANDROID_SHARE, OS_GENERATIONS


@dataclass
class Device:
    """
    WHAT: Represents a physical mobile phone associated with a person.
    WHY: Contains platform and operating system version details used for device velocity and change checks.
    """

    device_id: str
    platform: str
    os_generation: int
    fingerprint: str


def create_device(device_id: str, rng: np.random.Generator) -> Device:
    """
    WHAT: Synthesizes a new Device instance with platform, generation, and unique fingerprint.
    WHY: Assigns each simulated citizen an authentic mobile profile. DECISION: Platform has no effect
         on any downstream behavior other than device reporting, mirroring real-world device distribution.
    """
    # Platform chosen based on market share (ANDROID_SHARE = 94% Android, 6% iOS).
    is_android = float(rng.uniform(0.0, 1.0)) < ANDROID_SHARE
    platform = "android" if is_android else "ios"

    # OS generation chosen uniformly between 1 (oldest) and OS_GENERATIONS (newest).
    os_generation = int(rng.integers(1, OS_GENERATIONS, endpoint=True))

    # Build unique fingerprint using platform, OS generation, and random hex token.
    hex_token = f"{int(rng.integers(0, 0xFFFFFFFF, endpoint=True)):08x}"
    fingerprint = f"fp_{platform}_v{os_generation}_{hex_token}"

    return Device(
        device_id=device_id,
        platform=platform,
        os_generation=os_generation,
        fingerprint=fingerprint,
    )
