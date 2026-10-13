"""
WHAT: Canonical cryptographic hash utilities for simulation audit runs and baseline verification.
WHY: Guarantees reproducible, collision-resistant digests across simulation ledgers, events, and devices.
TERM: hash: A fixed-length 64-character hexadecimal SHA-256 digest calculated over serialized domain objects.
TERM: golden value: A pre-calculated baseline hash or count saved from an audited run to prevent regressions.
"""

import hashlib
from typing import Any


def hash_ledger_entries(ledgers: list[Any]) -> str:
    """
    WHAT: Serializes double-entry bookkeeping ledger logs across all accounts into a SHA-256 digest.
    WHY: Verifies bit-for-bit accounting determinism for simulation routines across monthly lifecycles.
    Args: ledgers: Sequence of bank Ledger containers holding transaction journal entries.
    Returns: A 64-character hexadecimal SHA-256 digest string uniquely representing journal entries.
    Example:
        # digest = hash_ledger_entries(world.ledgers)
        # assert len(digest) == 64 and isinstance(digest, str)
    """
    # 1. Format every journal record across all bank ledgers into pipe-delimited records.
    records: list[str] = [
        f"{e.entry_id}|{e.minute}|{e.account_id}|{e.amount_paise}|{e.kind}|{e.transfer_id}"
        for ldg in ledgers
        for e in ldg.entries
    ]
    # 2. Join lines with standard newline separators and compute canonical SHA-256 digest.
    payload = "\n".join(records).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def hash_payment_events(events: list[Any]) -> str:
    """
    WHAT: Serializes payment events into colon-delimited lines and computes a SHA-256 event digest.
    WHY: Implements the exact canonical event list hashing specification matching _hash_event_list.
    Args: events: Sequence of PaymentEvent instances recording simulated transaction attempts.
    Returns: A 64-character hexadecimal SHA-256 digest string capturing the entire event sequence.
    Example:
        # digest = hash_payment_events(simulation_results.events)
        # assert len(digest) == 64 and isinstance(digest, str)
    """
    # 1. Serialize each payment event into canonical colon-separated event attributes.
    lines: list[str] = [
        f"{ev.minute}:{ev.from_account}:{ev.to_account}:{ev.amount_paise}:"
        f"{ev.from_bank_index}:{ev.to_bank_index}:{ev.status}:{ev.reason}:{ev.failure_domain}"
        for ev in events
    ]
    # 2. Join serialized lines with newline characters and compute the SHA-256 hexadecimal hash.
    payload = "\n".join(lines).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def hash_device_stamps(stamps: list[Any]) -> str:
    """
    WHAT: Serializes per-transaction hardware telemetry stamps and calculates a SHA-256 digest.
    WHY: Verifies exact device fingerprint, OS generation, and novelty detection during runs.
    Args: stamps: List of DeviceStamp records representing telemetry for each executed transaction.
    Returns: A 64-character hexadecimal SHA-256 digest string representing sender device states.
    Example:
        # digest = hash_device_stamps(result.event_sender_devices)
        # assert len(digest) == 64 and isinstance(digest, str)
    """
    # 1. Format each telemetry stamp into pipe-separated fingerprint, OS generation, and novelty.
    lines: list[str] = [f"{s.fingerprint}|{s.os_generation}|{s.is_new_device}" for s in stamps]
    # 2. Join formatted stamps using newline delimiters and calculate cryptographic SHA-256 digest.
    payload = "\n".join(lines).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def hash_people_devices(people: list[Any]) -> str:
    """
    WHAT: Serializes person identifier and current hardware device state into a SHA-256 digest.
    WHY: Verifies citizen hardware ownership and OS generation alignment across population sets.
    Args: people: Sequence of Person instances containing assigned hardware devices.
    Returns: A 64-character hexadecimal SHA-256 digest string representing population devices.
    Example:
        # digest = hash_people_devices(world.people)
        # assert len(digest) == 64 and isinstance(digest, str)
    """
    # 1. Serialize citizen IDs alongside handset ID, hardware fingerprint, and OS version.
    lines: list[str] = [
        f"{p.person_id}|{p.device.device_id}|{p.device.fingerprint}|{p.device.os_generation}"
        for p in people
    ]
    # 2. Join lines with newline delimiters and compute SHA-256 cryptographic hexadecimal digest.
    payload = "\n".join(lines).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
