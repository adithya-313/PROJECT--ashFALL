"""
WHAT: Device synthesis, round-robin assignment, truth control spans, and telemetry events for gang devices (Level 12b B-2b).
Provides functions to provision external hardware devices, assign them cyclically across acquired mule accounts,
generate forensic ground truth DeviceControlSpan records, and synthesize client-side GangDeviceEvent records.

WHY: Criminal syndicates operate separate hardware profiles (such as rented phones, emulators, or remote tooling)
rather than transacting from victim handsets. Modeling gang-controlled devices establishes realistic device velocity,
hardware sharing patterns, and compromise telemetry without mutating Level 11 citizen device states.

TERM: gang device: a dedicated external phone or client hardware instance provisioned and controlled by a criminal syndicate.
TERM: device control span: a forensic ground truth interval recording gang manipulation over an account from a specific device.
TERM: gang device event: a synthetic telemetry event representing device acquisition or unauthorized account takeover operations.
TERM: round-robin assignment: cyclical allocation mapping sequential mule accounts across the pool of gang devices.
"""

from dataclasses import dataclass
from typing import Sequence
import numpy as np

from world.devices import Device, create_device
from world.tactic_config import get_gang_device_prefix
from world.truth_records import DeviceControlSpan


@dataclass(frozen=True)
class GangDeviceEvent:
    """
    WHAT: Represents a client-side telemetry event emitted during external device acquisition or account compromise.
    It encapsulates simulation minute timestamp, device identifier, optional controlled account identifier,
    event classification kind, client operating system platform, OS generation rating, and hardware fingerprint.

    WHY: Decision E4 requires external device operations to produce observable telemetry in the device-event stream
    so downstream fraud engines can detect account takeover indicators such as sudden SIM swap or emulator activation.

    Args:
        minute: Simulation minute timestamp when the event transpired.
        device_id: Neutral hardware identifier string for the external device (e.g., 'dev_ext_000001').
        account_id: Optional account identifier affected by the event, or None for device-level acquisition.
        kind: Classification kind (device_acquired, sim_swap, linked_number_switch, emulator_enabled, remote_access_enabled).
        platform: Operating system platform of the device ('android' or 'ios').
        os_generation: Integer generation rating indicating operating system currency.
        fingerprint: Unique synthetic hardware token string distinguishing the handset.

    Returns:
        An initialized immutable GangDeviceEvent record.

    Example:
        # event = GangDeviceEvent(100, "dev_ext_000001", None, "device_acquired", "android", 3, "fp_android_v3_705a5661")
        # assert event.kind == "device_acquired" and event.account_id is None
    """

    minute: int
    device_id: str
    account_id: str | None
    kind: str
    platform: str
    os_generation: int
    fingerprint: str


def make_gang_devices(
    count: int,
    rng: np.random.Generator,
    start_index: int = 1,
    prefix: str | None = None,
) -> list[Device]:
    """
    WHAT: Synthesizes a list of distinct external gang devices using Level 11 device generation routines.
    It generates neutral identifiers formatted with a configured prefix and six zero-padded decimal digits,
    drawing platform distribution and hardware fingerprints deterministically via create_device.

    WHY: Decision E2 requires gang device identifiers to remain neutral without leaking illicit roles
    to the payment engine, while reusing Level 11 hardware profiles to maintain identical statistical distributions.

    Args:
        count: Number of gang devices to synthesize.
        rng: Dedicated random number generator stream (gang_devices child 6).
        start_index: Starting integer offset for formatting zero-padded device identifiers.
        prefix: Optional identifier prefix string (defaults to get_gang_device_prefix()).

    Returns:
        List of initialized Device instances.

    Example:
        # rng = np.random.default_rng(42)
        # devices = make_gang_devices(2, rng, start_index=1)
        # assert len(devices) == 2 and devices[0].device_id == "dev_ext_000001"
    """
    # 1. Resolve neutral device identifier prefix from configuration.
    device_prefix = prefix if prefix is not None else get_gang_device_prefix()

    # 2. Synthesize requested number of distinct devices with formatted identifiers.
    devices: list[Device] = []
    for index in range(count):
        device_id = f"{device_prefix}{start_index + index:06d}"
        device = create_device(device_id, rng)
        devices.append(device)

    return devices


def assign_devices_round_robin(
    accounts: Sequence[str],
    devices: Sequence[Device],
) -> dict[str, Device]:
    """
    WHAT: Assigns acquired citizen mule accounts across the pool of available gang devices in round-robin order.
    It iterates across accounts and assigns each to the device at index (account_index % device_count),
    enabling a single device to control multiple mule accounts as observed in real-world laundering rings.

    WHY: Decision E1 specifies that syndicates pool multiple mule accounts onto shared physical devices,
    producing multi-account hardware clusters that downstream forensic analysis can detect.

    Args:
        accounts: Ordered sequence of citizen account identifiers acquired by the gang.
        devices: Sequence of synthesized gang Device instances available for assignment.

    Returns:
        Dictionary mapping each account identifier string to its assigned Device instance.

    Raises:
        ValueError: If accounts are provided but the device sequence is empty.

    Example:
        # mapping = assign_devices_round_robin(["acc_1", "acc_2", "acc_3"], devices[:2])
        # assert mapping["acc_1"].device_id == devices[0].device_id
        # assert mapping["acc_3"].device_id == devices[0].device_id
    """
    # 1. Reject empty device pools when accounts require control assignment.
    if not devices and accounts:
        raise ValueError("Cannot assign accounts to an empty gang device sequence.")

    # 2. Allocate devices cyclically across accounts in round-robin order.
    account_to_device: dict[str, Device] = {}
    device_count = len(devices)
    for index, account_id in enumerate(accounts):
        assigned_device = devices[index % device_count]
        account_to_device[account_id] = assigned_device

    return account_to_device


def build_device_control_spans(
    gang_id: str,
    account_to_device: dict[str, Device],
    take_control_minute: int,
    sim_swap: bool,
    linked_number_switch: bool,
    emulator: bool,
    remote_access: bool,
) -> list[DeviceControlSpan]:
    """
    WHAT: Constructs forensic ground truth DeviceControlSpan records linking each controlled account to its gang device.
    It records the start minute, unending validity (valid_to is None), and configured compromise flags.

    WHY: Decision E3 creates ground truth spans for evaluation frameworks to benchmark device clustering
    and compromise indicator detection without exposing truth facts to the simulation runtime.

    Args:
        gang_id: String identifier of the controlling gang syndicate.
        account_to_device: Mapping from account identifiers to assigned gang Device instances.
        take_control_minute: Simulation minute timestamp when the takeover block executed.
        sim_swap: Boolean indicating unauthorized SIM reissuance.
        linked_number_switch: Boolean indicating alteration of bank linked mobile phone number.
        emulator: Boolean indicating transaction execution within a virtual emulated environment.
        remote_access: Boolean indicating presence of remote screen control overlay software.

    Returns:
        List of immutable DeviceControlSpan records.

    Example:
        # spans = build_device_control_spans("gang_a", mapping, 100, True, False, False, True)
        # assert len(spans) == len(mapping) and spans[0].valid_to is None
    """
    # 1. Construct one DeviceControlSpan per controlled account.
    spans: list[DeviceControlSpan] = []
    for account_id, device in account_to_device.items():
        span = DeviceControlSpan(
            device_id=device.device_id,
            account_id=account_id,
            gang_id=gang_id,
            valid_from=take_control_minute,
            valid_to=None,
            sim_swap=sim_swap,
            linked_number_switch=linked_number_switch,
            emulator=emulator,
            remote_access=remote_access,
        )
        spans.append(span)

    return spans


def build_gang_device_events(
    devices: Sequence[Device],
    account_to_device: dict[str, Device],
    take_control_minute: int,
    sim_swap: bool,
    linked_number_switch: bool,
    emulator: bool,
    remote_access: bool,
) -> list[GangDeviceEvent]:
    """
    WHAT: Generates client-side synthetic GangDeviceEvent telemetry records for device takeover operations.
    For each device, it emits one device_acquired event; then for each controlled account, it emits
    one event for each configured compromise flag that evaluates to True at the take_control minute.

    WHY: Decision E4 models observable device telemetry to simulate client event feeds for downstream
    detection systems, ensuring telemetry contains no truth labels, roles, or gang identifiers.

    Args:
        devices: Sequence of external gang devices provisioned by the syndicate.
        account_to_device: Mapping from account identifiers to assigned Device instances.
        take_control_minute: Simulation minute timestamp when takeover executed.
        sim_swap: Boolean indicating SIM swap status.
        linked_number_switch: Boolean indicating registered telephone change.
        emulator: Boolean indicating virtual emulator execution.
        remote_access: Boolean indicating remote control session.

    Returns:
        List of immutable GangDeviceEvent records chronologically positioned at the take_control minute.

    Example:
        # events = build_gang_device_events(devices[:1], {"acc_1": devices[0]}, 100, True, False, True, False)
        # assert len(events) == 3  # device_acquired, sim_swap, emulator_enabled
    """
    # 1. Emit one device_acquired event for each provisioned physical or virtual device.
    events: list[GangDeviceEvent] = [
        GangDeviceEvent(take_control_minute, d.device_id, None, "device_acquired", d.platform, d.os_generation, d.fingerprint)
        for d in devices
    ]

    # 2. Emit specific compromise flag events for each account linked to a device.
    flags = [
        (sim_swap, "sim_swap"),
        (linked_number_switch, "linked_number_switch"),
        (emulator, "emulator_enabled"),
        (remote_access, "remote_access_enabled"),
    ]
    for account_id, device in account_to_device.items():
        for is_active, kind in flags:
            if is_active:
                events.append(GangDeviceEvent(take_control_minute, device.device_id, account_id, kind, device.platform, device.os_generation, device.fingerprint))

    return events
