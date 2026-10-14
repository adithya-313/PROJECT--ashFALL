"""
WHAT: Device hardware and operating system profile modeling for simulated individuals.
WHY: Simulates client device characteristics (platform, OS version generation, and hardware fingerprint)
     so fraud rules can examine device patterns without falsely penalizing legitimate OS updates.
TERM: device fingerprint: A synthetic pseudo-hardware identifier uniquely distinguishing a client phone,
      e.g. like a digital serial number or MAC address that helps a bank verify it is really your phone.
TERM: OS generation: An integer representing the release age of the phone operating system,
      e.g. generation 1 represents an older OS like Android 10, while generation 5 represents a modern OS like Android 14.
TERM: mobile platform: The primary operating system family powering a handset,
      e.g. Android vs iOS running on a consumer smartphone.
"""

from dataclasses import dataclass

import numpy as np

from world.placeholders import ANDROID_SHARE, OS_GENERATIONS


@dataclass
class Device:
    """
    WHAT: Represents a physical mobile phone associated with a citizen.
    It encapsulates the device identifier, operating system platform, OS generation rating, and unique hardware fingerprint.
    WHY: Risk engines inspect device characteristics to detect account takeover and sudden device switching.
    Tracking stable device attributes enables testing genuine software upgrades against anomalous hardware replacements.
    Args:
        device_id: Unique string identifier for the device, such as 'dev_person_000000'.
        platform: The operating system family of the phone, either 'android' or 'ios'.
        os_generation: Integer generation rating from 1 to 5 indicating system currency.
        fingerprint: Unique synthetic hardware hash string distinguishing this physical handset.
    Example:
        dev = Device("dev_0", "android", 3, "fp_android_v3_705a5661")
        assert dev.platform == "android" and dev.os_generation == 3
    """

    device_id: str
    platform: str
    os_generation: int
    fingerprint: str


def create_device(device_id: str, rng: np.random.Generator) -> Device:
    """WHAT: Synthesizes a new Device instance with platform, OS generation, and fingerprint.
    It samples the platform using real-world market shares (94% Android, 6% iOS).
    It then picks an OS generation and computes a distinct hardware token string.
    WHY: Every citizen in the world needs an authentic mobile phone profile to execute payments.
    Generating diverse device profiles allows fraud systems to evaluate device velocity and upgrades.
    Args: device_id: Unique string identifier; rng: Random generator for reproducible draws.
    Returns: Device instance configured with platform, version generation, and fingerprint string.
    Example:
        dev = create_device("dev_0", np.random.default_rng(42))
        assert dev.platform == "android" and dev.fingerprint == "fp_android_v3_705a5661"
    """
    # Compare a uniform random draw against empirical market share to pick Android vs iOS.
    is_android = float(rng.uniform(0.0, 1.0)) < ANDROID_SHARE
    platform = "android" if is_android else "ios"

    # Select an integer OS release generation uniformly between oldest and newest.
    os_generation = int(rng.integers(1, OS_GENERATIONS, endpoint=True))

    # Generate a random 32-bit hex token and assemble the structured device fingerprint.
    hex_token = f"{int(rng.integers(0, 0xFFFFFFFF, endpoint=True)):08x}"
    fingerprint = f"fp_{platform}_v{os_generation}_{hex_token}"

    # Package all drawn device properties into the Device dataclass container.
    return Device(
        device_id=device_id,
        platform=platform,
        os_generation=os_generation,
        fingerprint=fingerprint,
    )
