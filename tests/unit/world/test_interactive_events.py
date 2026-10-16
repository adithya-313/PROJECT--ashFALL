"""
WHAT: Tests for events-v2 lines (world/interactive_events.py): every line holds Level 11's 14 fields plus payment_id
and planned_payment_id, no field names anything about fraud or truth, and with no fraud the 14 shared fields match
Level 11's own lines exactly.

WHY: Banks and models will read these lines. If a field hinted at which events are fraud, every later result would be
contaminated; if the 14 shared fields drifted, the new lines would no longer describe the same events as Level 11's.

TERM: events-v1: Level 11's 14-field event line.
TERM: events-v2: events-v1 plus payment_id and planned_payment_id, 16 fields.
TERM: truth: which events are fraud, which accounts are mules, which gang did what; known to the world only.
TERM: payment id: the name of one event; planned payment id: the name shared by the pieces of one payment.
"""

import json

from core.config_validator import load_yaml
from world.fast_mode import FastRunSettings, event_lines, run_fast_world
from world.interactive_events import EVENTS_V2_FORMAT, interactive_event_lines
from world.interactive_records import FraudInputs, RunnerSettings
from world.interactive_runner import run_interactive_world
from world.outside_events import OutsideEvent
from world.run_identity import REPO_ROOT

# The tiny world from the frozen-anchor config.
FROZEN = load_yaml(REPO_ROOT / "config" / "frozen_anchors.yaml")["frozen_anchors"]["fraud_off_events"]
TINY = FastRunSettings(FROZEN["seed"], FROZEN["n_people"], FROZEN["n_shops"], FROZEN["n_days"], FROZEN["range_name"])
# Words no field name may contain (rules 15 and 29).
FORBIDDEN_WORDS = ("source", "fraud", "truth", "role", "gang")
NEW_KEYS = {"payment_id", "planned_payment_id"}


def test_lines_have_v1_fields_plus_two_ids_and_nothing_revealing() -> None:
    """
    WHAT: Builds events-v2 lines for a fraud-on run with a debit hold active (so some lines are funds_on_hold failures)
    and checks every line has exactly 16 keys: Level 11's 14 plus payment_id and planned_payment_id, with no key
    containing "source", "fraud", "truth", "role" or "gang".

    WHY: Checking a run where the runner actually intervened makes sure intervention never adds a revealing field.

    Args:
        None.

    Returns:
        None. The test fails on a missing, extra or revealing key.

    Example:
        Every one of the 2,915 lines has 16 keys, and the format name is "events-v2".
    """
    # A run where the first person account is held from minute 0 to day 5.
    hold = OutsideEvent(0, "debit_hold", "acc_person_000001", encumbrance_id="enc-test")
    release = OutsideEvent(5 * 1440, "release", encumbrance_id="enc-test")
    output = run_interactive_world(TINY, RunnerSettings(fraud_enabled=True),
                                   FraudInputs(outside_events=(hold, release)), code_version="test-version")
    v1_keys = set(json.loads(event_lines(run_fast_world(TINY))[0]))
    assert EVENTS_V2_FORMAT == "events-v2" and len(v1_keys) == 14
    for line in interactive_event_lines(output):
        keys = set(json.loads(line))
        assert keys == v1_keys | NEW_KEYS
        assert not any(word in key for key in keys for word in FORBIDDEN_WORDS)


def test_empty_plan_v1_part_matches_level_11_exactly() -> None:
    """
    WHAT: With the switch on, no fraud and the allow-all bank, removes payment_id and planned_payment_id from each
    events-v2 line, writes the rest the same compact way, and compares it with Level 11's event line at the same
    position. All 2,915 must match character for character.

    WHY: The 14 shared fields must be Level 11's exactly, so anything that reads events-v1 can read the shared part of
    events-v2 and see the same world.

    Args:
        None.

    Returns:
        None. The test fails if any line's shared part differs or the line counts differ.

    Example:
        Line 1's shared part equals Level 11's line 1: '{"amount_paise":19192,"failure_domain":null,...}'.
    """
    # Both sets of lines for the same world.
    output = run_interactive_world(TINY, RunnerSettings(fraud_enabled=True), code_version="test-version")
    v2_lines = interactive_event_lines(output)
    v1_lines = event_lines(run_fast_world(TINY))
    assert len(v2_lines) == len(v1_lines)
    for v2_line, v1_line in zip(v2_lines, v1_lines):
        # Strip the two ids and write the rest exactly as events-v1 is written.
        record = json.loads(v2_line)
        for key in NEW_KEYS:
            del record[key]
        assert json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n" == v1_line
