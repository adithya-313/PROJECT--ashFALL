"""
WHAT: Unit tests for gang device creation, round-robin assignment, truth control spans, and telemetry (world/gang_devices.py).
Asserts that device IDs match neutral patterns, fingerprints follow Level 11 formats, accounts are assigned round-robin,
spans reflect configured compromise flags, telemetry counts match device and active flag totals, identical seeds reproduce
devices, and retire/rotate parameters are strictly refused per Decision E6.

WHY: Criminal syndicates operate external shared devices and compromise mechanisms (SIM swap, emulators, remote access).
Comprehensive testing ensures hardware identifiers, ground truth spans, and client telemetry remain consistent,
deterministic, and architecturally separated from citizen device state per Decisions E1 through E8.

TERM: gang device: a dedicated external phone or client hardware instance provisioned and controlled by a criminal syndicate.
TERM: device control span: a forensic ground truth interval recording gang manipulation over an account from a specific device.
TERM: gang device event: a synthetic telemetry event representing device acquisition or unauthorized account takeover operations.
TERM: round-robin assignment: cyclical allocation mapping sequential mule accounts across the pool of gang devices.
"""

import re
import numpy as np
import pytest

from world.devices import create_device
from world.gang_config import BlockStep, FixtureSource, GangBudget, GangConfig, validate_gang_config
from world.gang_devices import (
    assign_devices_round_robin,
    build_device_control_spans,
    build_gang_device_events,
    make_gang_devices,
)

GENEROUS_BUDGET = GangBudget(money_paise=50_000_000, time_minutes=50_000, max_risk=0.99)
DEFAULT_SOURCE = FixtureSource("acc_collector_01", 0, 50_000_000, 100)


def test_gang_device_ids_match_neutral_pattern() -> None:
    """
    WHAT: Asserts that synthesized gang device identifiers strictly adhere to the neutral regex pattern '^dev_ext_\\d{6}$'.
    It generates 5 gang devices and validates each identifier against the compiled regular expression.

    WHY: Decision E2 requires device identifiers to be neutral so that the ID string itself never reveals
    the fraudulent role or syndicate affiliation to downstream simulated bank fraud rules.

    Args:
        None.

    Returns:
        None. Fails if any device identifier deviates from the '^dev_ext_\\d{6}$' format.

    Example:
        # devices = make_gang_devices(2, rng)
        # assert devices[0].device_id == "dev_ext_000001"
    """
    # 1. Synthesize gang devices using independent RNG stream.
    rng = np.random.default_rng(101)
    devices = make_gang_devices(5, rng)

    # 2. Assert all device identifiers match neutral regex format.
    pattern = re.compile(r"^dev_ext_\d{6}$")
    for device in devices:
        assert pattern.match(device.device_id) is not None, f"Device ID '{device.device_id}' does not match pattern"


def test_gang_device_fingerprints_match_level_11_format() -> None:
    """
    WHAT: Asserts that gang device fingerprints match the structural format produced by Level 11's create_device.
    It inspects a reference device created by create_device to extract platform prefix and token length without typing constants.

    WHY: Decision E2 dictates that external devices reuse Level 11 hardware profiles to maintain statistical realism
    without introducing detectable formatting differences between legitimate and fraudulent devices.

    Args:
        None.

    Returns:
        None. Fails if gang device fingerprint format diverges from Level 11 device synthesis.

    Example:
        # gang_dev = make_gang_devices(1, rng)[0]
        # assert gang_dev.fingerprint.startswith("fp_" + gang_dev.platform + "_v")
    """
    # 1. Generate reference device from Level 11 routine and gang device from gang routine.
    rng = np.random.default_rng(202)
    sample_ref = create_device("probe_dev", rng)
    gang_device = make_gang_devices(1, rng)[0]

    # 2. Verify fingerprint structural components (fp, platform, version, token) match reference design.
    ref_parts = sample_ref.fingerprint.split("_")
    gang_parts = gang_device.fingerprint.split("_")
    assert gang_parts[0] == ref_parts[0]  # "fp"
    assert gang_parts[1] in ("android", "ios")
    assert gang_parts[2].startswith("v") and gang_parts[2][1:].isdigit()
    assert len(gang_parts[3]) == len(ref_parts[3])  # token hex length


def test_round_robin_device_assignment() -> None:
    """
    WHAT: Asserts that acquired citizen mule accounts are assigned across gang devices in strict cyclical round-robin order.
    It provisions 2 devices and 5 accounts, then verifies that accounts 0, 2, 4 map to device 0 and accounts 1, 3 map to device 1.

    WHY: Decision E1 models shared hardware devices where multiple mule accounts are operated from the same handset,
    reflecting real-world mule ring operations while maintaining deterministic account-to-device mapping.

    Args:
        None.

    Returns:
        None. Fails if account assignment deviates from cyclical round-robin sequencing.

    Example:
        # mapping = assign_devices_round_robin(["m1", "m2", "m3"], devices[:2])
        # assert mapping["m1"] == devices[0] and mapping["m3"] == devices[0]
    """
    # 1. Synthesize 2 devices and define 5 sequential account identifiers.
    rng = np.random.default_rng(303)
    devices = make_gang_devices(2, rng)
    accounts = [f"acc_mule_{idx:02d}" for idx in range(5)]

    # 2. Execute round-robin assignment across accounts.
    assignment = assign_devices_round_robin(accounts, devices)

    # 3. Assert cyclical assignment across the device sequence.
    assert len(assignment) == 5
    assert assignment[accounts[0]].device_id == devices[0].device_id
    assert assignment[accounts[1]].device_id == devices[1].device_id
    assert assignment[accounts[2]].device_id == devices[0].device_id
    assert assignment[accounts[3]].device_id == devices[1].device_id
    assert assignment[accounts[4]].device_id == devices[0].device_id


def test_spans_carry_configured_flags() -> None:
    """
    WHAT: Asserts that DeviceControlSpan truth records carry exact compromise flags, gang identifier, and valid_to None.
    It builds control spans with sim_swap=True and emulator=True, then validates all fields across the generated records.

    WHY: Decision E3 creates ground truth control spans for downstream evaluation without leaking compromise facts
    into the reactive runtime, ensuring evaluation benchmarks measure true device attribution accurately.

    Args:
        None.

    Returns:
        None. Fails if control spans contain incorrect flags, invalid validity minutes, or mismatched IDs.

    Example:
        # spans = build_device_control_spans("gang_a", mapping, 100, True, False, True, False)
        # assert spans[0].sim_swap is True and spans[0].valid_to is None
    """
    # 1. Create mapping of 2 accounts to 1 device.
    rng = np.random.default_rng(404)
    devices = make_gang_devices(1, rng)
    mapping = {"acc_mule_01": devices[0], "acc_mule_02": devices[0]}

    # 2. Build ground truth spans with active sim_swap and emulator flags.
    spans = build_device_control_spans(
        "gang_control_test", mapping, 120, sim_swap=True, linked_number_switch=False,
        emulator=True, remote_access=False
    )

    # 3. Assert span attributes match configured flags and unending duration.
    assert len(spans) == 2
    for span in spans:
        assert span.gang_id == "gang_control_test"
        assert span.valid_from == 120
        assert span.valid_to is None
        assert span.sim_swap is True
        assert span.linked_number_switch is False
        assert span.emulator is True
        assert span.remote_access is False
        assert span.device_id == devices[0].device_id


def test_telemetry_event_counts_and_structure() -> None:
    """
    WHAT: Asserts that GangDeviceEvent telemetry records contain exactly one device_acquired event per device
    plus exactly one compromise event per active True flag for each controlled account at the take_control minute.

    WHY: Decision E4 models observable client telemetry feeds, verifying that event counts and kind labels
    accurately reflect device provisioning and account takeover signals without leaking gang identity.

    Args:
        None.

    Returns:
        None. Fails if event counts, kinds, or timestamps diverge from expected takeover telemetry.

    Example:
        # events = build_gang_device_events(devices[:2], mapping_3_accounts, 150, True, False, True, False)
        # assert len(events) == 2 + 3 * 2  # 8 events total
    """
    # 1. Prepare 2 devices and 3 accounts mapping to them.
    rng = np.random.default_rng(505)
    devices = make_gang_devices(2, rng)
    accounts = ["acc_mule_01", "acc_mule_02", "acc_mule_03"]
    mapping = assign_devices_round_robin(accounts, devices)

    # 2. Emit telemetry events with sim_swap=True and remote_access=True.
    events = build_gang_device_events(
        devices, mapping, 150, sim_swap=True, linked_number_switch=False,
        emulator=False, remote_access=True
    )

    # 3. Verify event count: 2 device_acquired + (3 accounts * 2 true flags) = 8 events.
    assert len(events) == 8
    acquired_events = [e for e in events if e.kind == "device_acquired"]
    assert len(acquired_events) == 2
    assert all(e.account_id is None for e in acquired_events)

    # 4. Verify account-level compromise events.
    sim_events = [e for e in events if e.kind == "sim_swap"]
    remote_events = [e for e in events if e.kind == "remote_access_enabled"]
    assert len(sim_events) == 3 and len(remote_events) == 3
    assert all(e.minute == 150 for e in events)


def test_identical_seed_produces_identical_devices() -> None:
    """
    WHAT: Asserts that provisioning gang devices with identical random number seeds yields identical device profiles.
    It generates 3 devices from seed 777 twice and verifies pairwise equality across identifiers and fingerprints.

    WHY: Simulation reproducibility is a core engineering principle; gang device synthesis must remain strictly deterministic.

    Args:
        None.

    Returns:
        None. Fails if two runs with the same seed generate divergent hardware profiles.

    Example:
        # dev_a = make_gang_devices(2, np.random.default_rng(777))
        # dev_b = make_gang_devices(2, np.random.default_rng(777))
        # assert dev_a[0].fingerprint == dev_b[0].fingerprint
    """
    # 1. Generate two device lists with identical seeds.
    devices_first = make_gang_devices(3, np.random.default_rng(777))
    devices_second = make_gang_devices(3, np.random.default_rng(777))

    # 2. Assert pairwise equivalence of device properties.
    for dev_a, dev_b in zip(devices_first, devices_second):
        assert dev_a.device_id == dev_b.device_id
        assert dev_a.platform == dev_b.platform
        assert dev_a.os_generation == dev_b.os_generation
        assert dev_a.fingerprint == dev_b.fingerprint


def test_take_control_refuses_retire_parameter() -> None:
    """
    WHAT: Asserts that configuring unknown device lifecycle parameters like 'retire_after_days' raises ValueError.
    It builds a gang configuration naming 'retire_after_days' in take_control and verifies immediate rejection.

    WHY: Decision E6 establishes that training devices are not retired on schedule; unsupported retirement
    parameters must be rejected cleanly to enforce the dividing line between training and held-out tactics.

    Args:
        None.

    Returns:
        None. Fails if retire_after_days is accepted by the gang configuration validator.

    Example:
        # validate_gang_config(config_with_retire) raises ValueError with 'retire_after_days'.
    """
    # 1. Assemble gang configuration containing forbidden retire_after_days parameter.
    steps = (
        BlockStep("acquire", (("count", 2),)),
        BlockStep("take_control", (("devices", 1), ("retire_after_days", 10))),
    )
    config = GangConfig("gang_retire", 1, GENEROUS_BUDGET, DEFAULT_SOURCE, steps)

    # 2. Assert validation refusal naming the unrecognized parameter.
    with pytest.raises(ValueError, match="unknown parameter 'retire_after_days'"):
        validate_gang_config(config)


def test_take_control_ordering_and_boolean_validation() -> None:
    """
    WHAT: Asserts that take_control must follow acquire and rejects non-boolean compromise parameters.
    It tests take_control before acquire, non-boolean sim_swap strings, and out-of-range device counts.

    WHY: Enforces structural integrity of gang configurations, ensuring tactics follow valid operational flows.

    Args:
        None.

    Returns:
        None. Fails if invalid ordering or malformed parameter types are accepted.

    Example:
        # validate_gang_config(take_control_before_acquire) raises ValueError.
    """
    # 1. Assert take_control before acquire is rejected.
    bad_order_steps = (BlockStep("take_control", (("devices", 1),)), BlockStep("acquire", (("count", 2),)))
    with pytest.raises(ValueError, match="cannot take control before acquiring accounts"):
        validate_gang_config(GangConfig("gang_order", 1, GENEROUS_BUDGET, DEFAULT_SOURCE, bad_order_steps))

    # 2. Assert non-boolean flag value is rejected.
    bad_bool_steps = (
        BlockStep("acquire", (("count", 2),)),
        BlockStep("take_control", (("devices", 1), ("sim_swap", "yes"))),
    )
    with pytest.raises(ValueError, match="must be boolean"):
        validate_gang_config(GangConfig("gang_bool", 1, GENEROUS_BUDGET, DEFAULT_SOURCE, bad_bool_steps))

    # 3. Assert out-of-range device count is rejected.
    bad_device_steps = (
        BlockStep("acquire", (("count", 2),)),
        BlockStep("take_control", (("devices", 10),)),
    )
    with pytest.raises(ValueError, match="devices must be an integer in \\[1, 5\\]"):
        validate_gang_config(GangConfig("gang_devices_out", 1, GENEROUS_BUDGET, DEFAULT_SOURCE, bad_device_steps))
