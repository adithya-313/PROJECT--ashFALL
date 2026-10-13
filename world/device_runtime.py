"""
WHAT: Device runtime transition engine applying hardware events and stamping payment telemetry.
WHY: Applies phone replacements, app reinstalls, and OS updates to simulated citizen profiles (issue R5).
TERM: device event: a change happening to a phone such as a replacement or update,
      e.g. buying a new mobile phone or installing an operating system patch.
TERM: fingerprint: a code that identifies a specific phone,
      e.g. a unique serial number so servers know which phone sent a payment.
TERM: known devices: the list of phone fingerprints a person has already used before,
      e.g. remembering your old phone so only a brand-new phone looks unfamiliar.
TERM: device stamp: a note attached to a payment showing sender phone details,
      e.g. recording the phone serial number and whether it was used for the first time.
TERM: new device: a phone fingerprint seen for the first time for a person,
      e.g. sending a payment from a newly bought handset.
"""

from dataclasses import dataclass
from typing import Any

from world.device_events import DeviceEvent


@dataclass(frozen=True)
class DeviceStamp:
    """
    WHAT: Telemetry snapshot recording sender hardware fingerprint and novelty per transaction.
    WHY: Supplies Level 14 feed records and enables fraud rules to test new device velocity.
    """

    fingerprint: str | None
    os_generation: int | None
    is_new_device: bool


NO_DEVICE_STAMP = DeviceStamp(None, None, False)


def apply_device_event(person: Any, event: DeviceEvent) -> None:
    """
    WHAT: Applies an individual lifecycle event to update a simulated person's phone profile.
    WHY: Keeps physical device state aligned with the scheduled timeline in chronological order.
         Validates starting hardware invariants before applying mutations to catch drift early.
    Args: person: Person entity; event: Validated DeviceEvent record specifying transitions.
    Returns: None. Modifies the target Person device in place.
    Example:
        # apply_device_event(person, event); assert person.device.fingerprint == event.fingerprint_after
    """
    # 1. Verify target person ID matches event person ID.
    if person.person_id != event.person_id:
        raise RuntimeError(f"Person mismatch: expected {person.person_id}, got {event.person_id}")
    # 2. Verify fingerprint and OS generation match before-event hardware state.
    if person.device.fingerprint != event.fingerprint_before:
        raise RuntimeError(f"FP mismatch: expected {person.device.fingerprint}, got {event.fingerprint_before}")
    if person.device.os_generation != event.os_generation_before:
        raise RuntimeError(f"OS mismatch: expected {person.device.os_generation}, got {event.os_generation_before}")
    # 3. Update operating system generation and hardware fingerprint.
    person.device.fingerprint = event.fingerprint_after
    person.device.os_generation = event.os_generation_after
    # 4. For physical phone replacements, increment device identifier suffix.
    if event.kind == "phone_change":
        orig_id = getattr(person, "_orig_device_id", person.device.device_id)
        person._orig_device_id = orig_id
        n = getattr(person, "_phone_changes", 0) + 1
        person._phone_changes = n
        person.device.device_id = f"{orig_id}_p{n}"


def initial_known_fingerprints(people: list[Any]) -> dict[str, set[str]]:
    """
    WHAT: Constructs a lookup mapping each individual to their starting device fingerprint.
    WHY: Establishes the baseline history of recognized hardware before calendar transactions occur.
         Ensures initial payments on original hardware are never falsely flagged as new devices.
    Args:
        people: List of Person instances across the simulated world.
    Returns:
        Dictionary mapping person identifier strings to sets of known fingerprint strings.
    Example:
        # known = initial_known_fingerprints(world.people)
        # assert person.device.fingerprint in known[person.person_id]
    """
    # Initialize dictionary recording starting hardware fingerprints per citizen.
    known: dict[str, set[str]] = {}
    # Register the initial device fingerprint for every citizen in the town.
    for p in people:
        known[p.person_id] = {p.device.fingerprint}
    return known


def stamp_sender_device(person_or_none: Any, known: dict[str, set[str]]) -> DeviceStamp:
    """
    WHAT: Produces a device stamp documenting sender hardware and checks whether the device is new.
    WHY: Implements rule R5 invariant where OS updates keep existing fingerprints and never trigger new device alerts.
         Institutional shop senders without personal devices receive an empty sentinel stamp.
    Args:
        person_or_none: Person instance initiating payment, or None if payment originates from a shop.
        known: Mutable dictionary mapping person identifiers to sets of observed fingerprints.
    Returns:
        DeviceStamp documenting sender device fingerprint, OS generation, and new device flag.
    Example:
        # stamp = stamp_sender_device(person, known)
        # assert stamp.fingerprint == person.device.fingerprint
    """
    # 1. Commercial shop senders do not possess personal mobile devices; return sentinel stamp.
    if person_or_none is None:
        return NO_DEVICE_STAMP
    # 2. Extract current hardware telemetry for the person paying.
    p_id = person_or_none.person_id
    current_fp = person_or_none.device.fingerprint
    # 3. Check if current fingerprint has been seen; OS updates keep old fingerprint so is_new is False.
    is_new = current_fp not in known[p_id]
    # 4. Add current fingerprint to known set so subsequent payments from this device are not marked new.
    known[p_id].add(current_fp)
    return DeviceStamp(current_fp, person_or_none.device.os_generation, is_new)
