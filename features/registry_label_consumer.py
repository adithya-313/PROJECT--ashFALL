# File: features/registry_label_consumer.py
# Purpose: Level 16d-1 - bank-side registry label consumer: turns national registry events into labels.
"""
WHAT: Turns a registry.events message into registry labels for the banks it concerns. A pure matching function
(no database, no Kafka) decides which of a bank's own accounts an event matches; a thin wrapper loads the bank's own
identifiers, matches, and writes LabelRow(source="registry").

WHY: A bank may only use what it can see: its own account ids, the devices its accounts used, and the KYC phone on its
account_opened events. A registry hit on one of those becomes a label at the world time the bank processes the event.
Matches are not filtered for recycled or shared identifiers (issue N1): the innocent current holder of a recycled
phone is labelled, which is the intended wrong label.

TERM: DPIP participant: a bank enrolled in the registry scheme; a non-participant never reads registry events.
TERM: reported account: the account the reporting bank filed the report about; its hash is the event's
      linked_account_hash (or the identifier hash itself for an account item).
TERM: skip rule: the reporting bank does not label the reported account itself (it already knows it); a different
      account at the reporting bank that shares the phone or device is labelled.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from features.label_store import LabelRow, insert_labels
from nationalsim.keyed_hash import compute_identifier_hash

__all__ = [
    "OwnIdentifiers",
    "build_own_identifiers",
    "match_registry_event",
    "registry_label_rows",
    "consume_registry_events",
]

REGISTRY_SOURCE = "registry"
POLICE_BULK = "police_bulk"
HOLD_REQUEST = "hold_request"


@dataclass(frozen=True)
class OwnIdentifiers:
    """
    WHAT: One bank's own hashed identifiers: account hash per account, and which of its accounts used each device
    hash and phone hash.
    WHY: The registry only holds hashes, so matching is a dictionary lookup on the bank's own hashes.
    Args:
        bank_id: The bank, e.g. "bank_0".
        participant: True when the bank is a DPIP participant.
        account_hash_of: account id -> keyed hash of "<bank_index>:<account id>".
        accounts_by_account_hash: account hash -> account id.
        accounts_by_device_hash: device hash -> accounts that used that device.
        accounts_by_phone_hash: phone hash -> accounts whose KYC phone it is.
    Example:
        OwnIdentifiers("bank_0", True, {"a": "h"}, {"h": "a"}, {}, {})
    """

    bank_id: str
    participant: bool
    account_hash_of: Mapping[str, str] = field(default_factory=dict)
    accounts_by_account_hash: Mapping[str, str] = field(default_factory=dict)
    accounts_by_device_hash: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    accounts_by_phone_hash: Mapping[str, tuple[str, ...]] = field(default_factory=dict)


def build_own_identifiers(
    bank_id: str,
    bank_index: int,
    participant: bool,
    accounts: Iterable[str],
    fingerprints: Mapping[str, Iterable[str]],
    phones: Mapping[str, str],
    key: str,
    run_id: str,
) -> OwnIdentifiers:
    """
    WHAT: Hashes a bank's own identifiers the way a registry reporter hashes them (account namespaced by the bank
    index; device and phone global) and indexes them for lookup.
    WHY: One place defines the bank-side hashes, so tests and the wrapper agree.
    Args:
        bank_id: The bank id.
        bank_index: The bank index used in account hashes.
        participant: Whether the bank is a DPIP participant.
        accounts: All of the bank's own account ids.
        fingerprints: account id -> device fingerprints that account used.
        phones: account id -> KYC phone (accounts without one are absent).
        key: The DPIP key.
        run_id: The run id.
    Returns:
        The indexed OwnIdentifiers.
    Example:
        build_own_identifiers("bank_0", 0, True, ["a"], {}, {"a": "sim-1"}, key, "r1")
    """
    account_hash_of = {
        a: compute_identifier_hash(key, run_id, "account", a, bank_index=bank_index) for a in sorted(set(accounts))
    }
    by_device: dict[str, list[str]] = {}
    for account in sorted(fingerprints):
        for fingerprint in sorted(set(fingerprints[account])):
            by_device.setdefault(compute_identifier_hash(key, run_id, "device", fingerprint), []).append(account)
    by_phone: dict[str, list[str]] = {}
    for account in sorted(phones):
        by_phone.setdefault(compute_identifier_hash(key, run_id, "phone", phones[account]), []).append(account)
    return OwnIdentifiers(
        bank_id=bank_id,
        participant=participant,
        account_hash_of=account_hash_of,
        accounts_by_account_hash={h: a for a, h in account_hash_of.items()},
        accounts_by_device_hash={h: tuple(v) for h, v in by_device.items()},
        accounts_by_phone_hash={h: tuple(v) for h, v in by_phone.items()},
    )


def match_registry_event(event: Mapping[str, Any], own: OwnIdentifiers) -> list[str]:
    """
    WHAT: Returns the bank's own accounts a registry event concerns, sorted. A non-participant gets nothing. The event's
    identifier hash is looked up among the bank's own account, device or phone hashes by identifier type. The
    reporting bank skips only the reported account itself (its own hash equals the event's linked_account_hash);
    a different account at the reporting bank that shares the phone or device is returned. Nothing else is filtered.
    WHY: The skip rule must never hide a real shared-identifier match, and a pre-world seed (no linked account) is
    never skipped. A police_bulk event ignores DPIP participation and matches only the holding bank's own
    account hash.
    Args:
        event: The registry.events payload (identifier_hash, identifier_type, bank_id, linked_account_hash).
        own: The bank's own identifiers.
    Returns:
        Sorted account ids to label (possibly empty).
    Example:
        match_registry_event({"identifier_hash": h, "identifier_type": "phone", "bank_id": "bank_1"}, own)
    """
    identifier_hash = str(event["identifier_hash"])
    kind = str(event["identifier_type"])
    if event.get("event_type") in (POLICE_BULK, HOLD_REQUEST):
        # A police crackdown or a 1930 hold request goes to the holding bank whether or not it takes part in DPIP;
        # only its own hash matches.
        found_bulk = own.accounts_by_account_hash.get(identifier_hash) if kind == "account" else None
        return [found_bulk] if found_bulk is not None else []
    if not own.participant:
        return []
    if kind == "account":
        found = own.accounts_by_account_hash.get(identifier_hash)
        matched = [found] if found is not None else []
    elif kind == "device":
        matched = list(own.accounts_by_device_hash.get(identifier_hash, ()))
    elif kind == "phone":
        matched = list(own.accounts_by_phone_hash.get(identifier_hash, ()))
    else:
        raise ValueError(f"Unknown registry identifier_type {kind!r}")
    if own.bank_id == event.get("bank_id"):
        reported = event.get("linked_account_hash") or (identifier_hash if kind == "account" else None)
        matched = [a for a in matched if own.account_hash_of.get(a) != reported]
    return sorted(set(matched))


def registry_label_rows(
    event: Mapping[str, Any], accounts: Sequence[str], own: OwnIdentifiers, run_id: str, arrival_time: datetime
) -> list[LabelRow]:
    """
    WHAT: Builds one registry LabelRow per matched account.
    WHY: label_id is run-scoped and deterministic: "<report_id>:<identifier_type>"; the primary key adds bank and
    account, so each bank labels each account once per report item.
    Args:
        event: The registry.events payload.
        accounts: Accounts returned by match_registry_event.
        own: The bank's own identifiers.
        run_id: The run id.
        arrival_time: Aware world time the bank processed the event.
    Returns:
        LabelRow list (verdict fraud, source registry, no payment ids).
    Example:
        registry_label_rows(event, ["acc_1"], own, "r1", now)
    """
    label_id = f"{event['report_id']}:{event['identifier_type']}"
    if event.get("event_type") == HOLD_REQUEST:
        return [
            LabelRow(
                run_id=run_id, bank_id=own.bank_id, label_id=str(event["label_id"]), source="complaint",
                account_id=account, payment_ids=tuple(event.get("payment_ids") or ()), verdict="fraud",
                arrival_time=arrival_time, claimed_type=event.get("claimed_type"), via=HOLD_REQUEST,
            )
            for account in accounts
        ]
    if event.get("event_type") == POLICE_BULK:
        return [
            LabelRow(
                run_id=run_id, bank_id=own.bank_id, label_id=label_id, source=POLICE_BULK, account_id=account,
                payment_ids=(), verdict="fraud", arrival_time=arrival_time, region=event.get("region"),
                batch_id=event.get("batch_id"), via=None,
            )
            for account in accounts
        ]
    return [
        LabelRow(
            run_id=run_id, bank_id=own.bank_id, label_id=label_id, source=REGISTRY_SOURCE, account_id=account,
            payment_ids=(), verdict="fraud", arrival_time=arrival_time, via="registry",
        )
        for account in accounts
    ]


async def _load_own_identifiers(
    conn: Any, bank_id: str, bank_index: int, participant: bool, key: str, run_id: str
) -> OwnIdentifiers:
    """Loads a bank's own accounts, device fingerprints and KYC phones from its workspace tables and hashes them."""
    account_rows = await conn.fetch(
        "SELECT DISTINCT account_id FROM bank_account_events WHERE bank_id = $1 AND kind = 'account_opened'", bank_id
    )
    phone_rows = await conn.fetch(
        "SELECT account_id, phone FROM bank_account_events "
        "WHERE bank_id = $1 AND kind = 'account_opened' AND phone IS NOT NULL",
        bank_id,
    )
    debit_rows = await conn.fetch(
        "SELECT DISTINCT account_id, sender_device_fingerprint AS fp FROM transactions "
        "WHERE bank_id = $1 AND side = 'debit' AND sender_device_fingerprint IS NOT NULL",
        bank_id,
    )
    event_rows = await conn.fetch(
        "SELECT account_id, details FROM bank_account_events WHERE bank_id = $1 AND kind = 'device_binding_change'",
        bank_id,
    )
    fingerprints: dict[str, set[str]] = {}
    for row in debit_rows:
        fingerprints.setdefault(row["account_id"], set()).add(row["fp"])
    for row in event_rows:
        details = row["details"] if isinstance(row["details"], dict) else json.loads(row["details"])
        fp = details.get("fingerprint_after")
        if fp:
            fingerprints.setdefault(row["account_id"], set()).add(fp)
    return build_own_identifiers(
        bank_id, bank_index, participant, [r["account_id"] for r in account_rows], fingerprints,
        {r["account_id"]: r["phone"] for r in phone_rows}, key, run_id,
    )


async def consume_registry_events(
    conn: Any,
    events: Sequence[Mapping[str, Any]],
    banks: Mapping[str, bool],
    key: str,
    run_id: str,
    arrival_time: datetime,
) -> int:
    """
    WHAT: Processes a batch of registry events for every bank: loads the bank's own identifiers, matches each event,
    and inserts registry labels stamped with the given world time. Returns the number of labels inserted.
    WHY: The replay drain hands each batch of registry messages here before committing offsets, so a decision never
    reads labels still in flight.
    Args:
        conn: Workspace database connection.
        events: Decoded registry.events payloads (the envelope's data), in topic order.
        banks: bank id -> DPIP participation.
        key: The DPIP key.
        run_id: The run id.
        arrival_time: Aware world time of processing (the replay world clock).
    Returns:
        Count of labels actually inserted (duplicates are ignored by the primary key).
    Example:
        inserted = await consume_registry_events(conn, events, {"bank_0": True}, key, "r1", now)
    """
    if not events:
        return 0
    inserted = 0
    for bank_id in sorted(banks):
        bank_index = int(bank_id.removeprefix("bank_"))
        own = await _load_own_identifiers(conn, bank_id, bank_index, banks[bank_id], key, run_id)
        rows: list[LabelRow] = []
        for event in events:
            if event.get("run_id") != run_id:
                continue
            rows.extend(registry_label_rows(event, match_registry_event(event, own), own, run_id, arrival_time))
        inserted += await insert_labels(conn, rows)
    return inserted
