"""
WHAT: Tests for the interactive runner's fraud switch in its off position (the default). With the switch off the
runner must give Level 11's world byte for byte, add only in-memory ids, and refuse any fraud input instead of
ignoring it.

WHY: Ground rule 13 requires a switch with a safe fallback, and rule 27 requires Level 11's output to stay frozen.
These tests check both against the frozen anchors in config/frozen_anchors.yaml, not against numbers typed here.

TERM: fraud switch: RunnerSettings.fraud_enabled; off (False) is the default and means "run Level 11 exactly".
TERM: frozen anchor: a value Level 11 pinned that must never move, e.g. the 64-character hash of the fraud-off events.
TERM: event lines: the 14-field JSON text Level 11 writes for each payment event, one line per event.
TERM: planned payment id: the name shared by all pieces of one planned payment, e.g. "<run id>-p-000000".
TERM: payment id: the name of one event, e.g. "<run id>-000042" for the 43rd event.
TERM: device event: a phone change, app reinstall or operating-system update applied to a person during the run.
"""

import hashlib
import re

import pytest

from core.config_validator import load_yaml
from world.calendar_runner import PlannedPayment
from world.fast_mode import FastRunSettings, event_lines, run_fast_world
from world.interactive_records import FraudInputs, PaymentView, RunnerSettings
from world.interactive_runner import SWITCH_OFF_MESSAGE, interactive_fast_output, run_interactive_world
from world.outside_events import OutsideEvent
from world.run_hashes import hash_ledger_entries, hash_people_devices
from world.run_identity import REPO_ROOT

# Expected values come from the frozen-anchor config, never from literals in this file.
FROZEN = load_yaml(REPO_ROOT / "config" / "frozen_anchors.yaml")["frozen_anchors"]["fraud_off_events"]
TINY = FastRunSettings(
    seed=FROZEN["seed"], n_people=FROZEN["n_people"], n_shops=FROZEN["n_shops"],
    n_days=FROZEN["n_days"], range_name=FROZEN["range_name"],
)
# A fixed version string, so these tests never depend on git being installed.
TEST_CODE_VERSION = "test-version"


def test_fraud_off_reproduces_the_frozen_events_hash() -> None:
    """
    WHAT: Runs the tiny world through run_interactive_world with every default (switch off) and hashes its event lines
    with Level 11's own event_lines. The hash must equal the frozen sha256 from config/frozen_anchors.yaml.

    WHY: This is test T1: the fallback must be Level 11 byte for byte. Using Level 11's own serialiser means the check
    cannot pass by accident through a copied formatter.

    Args:
        None.

    Returns:
        None. The test fails if one byte of the 2,915 event lines differs.

    Example:
        The hash of the joined lines starts with 24fc0e25009d7946, the frozen value.
    """
    # Run with the switch off and serialise with Level 11's own function.
    output = run_interactive_world(TINY, code_version=TEST_CODE_VERSION)
    lines = event_lines(interactive_fast_output(output))
    assert hashlib.sha256("".join(lines).encode("utf-8")).hexdigest() == FROZEN["sha256"]


def test_fraud_off_equals_level_11_run_everywhere() -> None:
    """
    WHAT: Runs the tiny world once with Level 11's run_fast_world and once through the runner with the switch off,
    then compares the full Level 11 result (events, counts, purposes, device stamps, device counts), the ledger hash
    and the people's device hash.

    WHY: The runner prepares the world itself (it repeats run_fast_world's first eight steps so it can keep the device
    schedule). This test proves that preparation is identical to Level 11's in every output, not only the events.

    Args:
        None.

    Returns:
        None. The test fails if any compared output differs.

    Example:
        Both runs give 2,915 events and identical ledger hashes.
    """
    # Level 11's own run and the runner's fallback run.
    level_11 = run_fast_world(TINY)
    runner = run_interactive_world(TINY, code_version=TEST_CODE_VERSION)
    # The Level 11-shaped result, the bank ledgers and every person's phone must all match.
    assert runner.result.run_result == level_11.result
    assert hash_ledger_entries(runner.world.ledgers) == hash_ledger_entries(level_11.world.ledgers)
    assert hash_people_devices(runner.world.people) == hash_people_devices(level_11.world.people)


def test_fraud_off_adds_ids_and_device_events_in_memory() -> None:
    """
    WHAT: Checks the extra, in-memory parts of a fraud-off result: one well-formed payment id per event, one planned
    payment id per event (all pieces of a split payment share one), every source "planned", and every scheduled device
    event listed as applied (Level 11 applies the rest after its last payment).

    WHY: Later parts read these ids even when fraud is off. They must line up with the events one to one, or a later
    lookup would attach an id to the wrong event.

    Args:
        None.

    Returns:
        None. The test fails if a list is the wrong length, an id is malformed or repeated, or the device list differs.

    Example:
        2,915 events give 2,915 payment ids such as "<16 hex>-000000", and 32 device events are applied.
    """
    # Run with the switch off.
    output = run_interactive_world(TINY, code_version=TEST_CODE_VERSION)
    result = output.result
    event_count = len(result.run_result.events)
    # Payment ids: one per event, all different, all the right shape.
    assert len(result.payment_ids) == event_count and len(set(result.payment_ids)) == event_count
    assert all(re.fullmatch(r"[0-9a-f]{16}-\d{6}", payment_id) for payment_id in result.payment_ids)
    # Planned payment ids and sources line up with the events too.
    assert len(result.planned_payment_ids) == event_count
    assert result.event_sources == ["planned"] * event_count
    # Every scheduled device event was applied, matching Level 11's own count.
    assert result.applied_device_events == list(output.device_events)
    assert result.run_result.device_counts["events_applied"] == len(output.device_events)


@pytest.mark.parametrize("which", ["payment", "outside_event", "hook"])
def test_fraud_inputs_with_switch_off_are_refused(which: str) -> None:
    """
    WHAT: Tries three ways of giving fraud input while the switch is off: a fraud payment, an outside event, and a hook
    that is not allow_all. Each must raise ValueError("fraud inputs need fraud_enabled=True").

    WHY: Silently ignoring these would let someone believe a lien or fraud payment was applied. Refusing them is the
    switch's promise that "off" really means Level 11 and nothing else.

    Args:
        which: Which kind of fraud input this case supplies.

    Returns:
        None. The test fails if a run starts instead of raising.

    Example:
        run_interactive_world(TINY, fraud_inputs=FraudInputs(payments=(payment,))) raises the message.
    """
    # Build exactly one kind of fraud input for this case.
    payment = PlannedPayment(60, "acc_person_000001", "acc_shop_0001", 1000, 0, "everyday")
    inputs = {
        "payment": FraudInputs(payments=(payment,)),
        "outside_event": FraudInputs(outside_events=(OutsideEvent(60, "debit_hold", "acc_person_000001"),)),
        "hook": FraudInputs(),
    }[which]

    def decline_everything(view: PaymentView) -> str:
        """
        WHAT: A hook that declines every payment; any hook other than allow_all counts as a fraud input.

        WHY: The switch-off check must catch a custom hook as well as payments and events, because a hook alone can
        change the run.

        Args:
            view: The PaymentView; not used.

        Returns:
            Always "decline".

        Example:
            decline_everything(PaymentView("id", 1, "a", "b", 100)) returns "decline".
        """
        return "decline"

    hook = decline_everything if which == "hook" else None
    # With the switch off, every one of these must be refused before anything runs.
    with pytest.raises(ValueError, match=SWITCH_OFF_MESSAGE):
        if hook is None:
            run_interactive_world(TINY, RunnerSettings(), inputs, code_version=TEST_CODE_VERSION)
        else:
            run_interactive_world(TINY, RunnerSettings(), inputs, hook, code_version=TEST_CODE_VERSION)
