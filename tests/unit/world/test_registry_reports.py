"""
Tests for registry reports (world/registry_reports.py, RegistryReport in world/label_records.py and
build_registry_reports in world/label_builder.py).
Covers that the record matches the written contract, that wrong registry entries come only from false complaints,
that the victim complaint share and the bank reporting share are independent, the reporting share and delays, the
first-complaint rule for confirmation, and the refusal of unknown mule-side roles.

TERM: wrong registry entry: a report about an account that holds no mule-side role at its confirmed minute.
TERM: false complaint: a complaint whose payment ids are all non-fraud events.
"""

import copy
import dataclasses
import json
import math
import re
from typing import Any

import numpy as np
import pytest

import world.label_builder as label_builder
from showcase.level_12_support_b import get_showcase_b_settings
from showcase.level_12_support_c import load_showcase_c_fixture
from world.config_reader import read_world_ranges
from world.interactive_records import InteractiveWorldOutput, RunnerSettings
from world.interactive_runner import run_interactive_world
from world.label_builder import build_label_events, build_registry_reports
from world.label_records import (
    REGISTRY_REPORT_KIND,
    LabelEvent,
    RegistryReport,
    load_labels_config,
    resolve_registry_delay_bounds_minutes,
    validate_labels_config,
)
from world.registry_reports import confirm_accounts, draw_bank_registry_report_share, plan_registry_reports
from world.run_identity import REPO_ROOT

CONTRACT_PATH = REPO_ROOT / "docs" / "interfaces" / "national_systems.md"
CONTRACT_HEADING = "## Level 12e: registry-report events emitted by the world"
SPLIT = "training"
SHARE_NAME = "share_reported_to_shared_database"
VICTIM_SHARE_NAME = "victim_complaint_share"
FALSE_RATE_NAME = "false_complaints_per_100000_genuine_payments"


@pytest.fixture(scope="module")
def labels_config() -> dict[str, Any]:
    """Loads and validates the labels config once for the module."""
    return load_labels_config()


@pytest.fixture(scope="module")
def cashout_output() -> InteractiveWorldOutput:
    """Runs the Part C victim-to-cash-out fixture gang once, shared by the world-level tests."""
    settings, code_version = get_showcase_b_settings()
    return run_interactive_world(
        settings, RunnerSettings(fraud_enabled=True), gang_configs=(load_showcase_c_fixture(),),
        code_version=code_version,
    )


def with_ranges_forced(output: InteractiveWorldOutput, forced: dict[str, float]) -> InteractiveWorldOutput:
    """
    Returns the output with each named world_ranges entry set to exactly [value, value] for its split, in a copy of
    the effective configuration. The run itself is untouched; only label planning reads these entries.
    """
    forced_config = copy.deepcopy(output.effective_config)
    split = forced_config["settings"]["range_name"]
    parameters = forced_config["world_configs"]["world_ranges"]["parameters"]
    for name, value in forced.items():
        parameters[name][split] = [value, value]
    return dataclasses.replace(output, effective_config=forced_config)


def make_complaint(label_id: str, minute: int, account: str = "acc_x") -> LabelEvent:
    """Builds a numbered complaint about `account` arriving at `minute`."""
    return LabelEvent(label_id, "complaint", minute, "acc_c", 0, account, 1, (f"p_{label_id}",), "investment")


def test_registry_report_fields_match_the_written_contract() -> None:
    """Checks the RegistryReport field names and order equal the JSON example in national_systems.md."""
    text = CONTRACT_PATH.read_text(encoding="utf-8")
    section = text[text.index(CONTRACT_HEADING):]
    match = re.search(r"```json\n(.*?)```", section, flags=re.DOTALL)
    assert match is not None
    example = json.loads(match.group(1))

    assert list(example.keys()) == [field.name for field in dataclasses.fields(RegistryReport)]
    assert example["kind"] == REGISTRY_REPORT_KIND
    assert isinstance(example["complaint_ids"], list)


def test_wrong_registry_entries_come_only_from_false_complaints(
    cashout_output: InteractiveWorldOutput, labels_config: dict[str, Any]
) -> None:
    """
    Checks every report about an account with no mule-side role at its confirmed minute names a false complaint.

    The victim share and the bank share are forced to 1.0 and the false rate to 20,000 per 100,000, so the world has
    both right and wrong entries. The test also checks each confirmed minute is the first complaint's arrival.
    """
    forced = with_ranges_forced(cashout_output, {VICTIM_SHARE_NAME: 1.0, SHARE_NAME: 1.0, FALSE_RATE_NAME: 20_000.0})
    labels = build_label_events(forced)
    reports = build_registry_reports(forced, labels)
    result = forced.result
    fraud_ids = {
        result.payment_ids[position] for position, source in enumerate(result.event_sources) if source == "fraud"
    }
    false_label_ids = {
        label.label_id for label in labels if all(payment_id not in fraud_ids for payment_id in label.payment_ids)
    }
    mule_side_roles = labels_config["mule_side_roles"]

    def holds_mule_side_role(account_id: str, minute: int) -> bool:
        """True when a mule-side role span of the account covers the minute (half-open span)."""
        for span in result.truth.roles:
            if span.account_id != account_id or span.role not in mule_side_roles:
                continue
            if span.valid_from <= minute and (span.valid_to is None or minute < span.valid_to):
                return True
        return False

    wrong = [report for report in reports if not holds_mule_side_role(report.account_id, report.confirmed_minute)]
    right = [report for report in reports if holds_mule_side_role(report.account_id, report.confirmed_minute)]
    assert len(wrong) > 0
    assert len(right) > 0
    for report in wrong:
        assert any(complaint_id in false_label_ids for complaint_id in report.complaint_ids)

    first_arrival: dict[str, int] = {}
    for label in labels:
        known = first_arrival.get(label.reported_account)
        if known is None or label.arrival_minute < known:
            first_arrival[label.reported_account] = label.arrival_minute
    for report in reports:
        assert report.confirmed_minute == first_arrival[report.account_id]


def test_victim_share_and_bank_share_are_independent(
    cashout_output: InteractiveWorldOutput, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Checks changing only the victim complaint share leaves the bank share and the reported-or-not draws unchanged,
    and changing only the bank share leaves the complaints and the drawn world settings unchanged.

    The draws are compared by recording, inside the real builder call, the share and the first uniforms the
    registry stream is about to give, not just counts of reports.
    """
    recorded: list[tuple[float, list[float], int]] = []
    real_planner = label_builder.plan_registry_reports

    def recording_planner(confirmed: Any, holders: Any, ranges: Any, split: Any, config: Any, rng: Any,
                          run_id: Any, digits: Any) -> Any:
        """Records the share and first draws the registry stream would give, then calls the real planner."""
        probe = np.random.Generator(np.random.PCG64())
        probe.bit_generator.state = copy.deepcopy(rng.bit_generator.state)
        share = draw_bank_registry_report_share(ranges, split, probe, config)
        recorded.append((share, probe.random(20).tolist(), len(confirmed)))
        return real_planner(confirmed, holders, ranges, split, config, rng, run_id, digits)

    monkeypatch.setattr(label_builder, "plan_registry_reports", recording_planner)
    victim_one = with_ranges_forced(cashout_output, {VICTIM_SHARE_NAME: 1.0, FALSE_RATE_NAME: 20_000.0})
    victim_low = with_ranges_forced(cashout_output, {VICTIM_SHARE_NAME: 0.2, FALSE_RATE_NAME: 20_000.0})
    labels_one, labels_low = build_label_events(victim_one), build_label_events(victim_low)
    build_registry_reports(victim_one, labels_one)
    build_registry_reports(victim_low, labels_low)

    (share_one, draws_one, count_one), (share_low, draws_low, count_low) = recorded
    assert labels_one != labels_low
    assert share_one == share_low
    compared = min(count_one, count_low, 20)
    assert compared > 0
    assert draws_one[:compared] == draws_low[:compared]

    # The reverse: only the bank share range changes, so the complaints and the drawn settings stay the same.
    bank_zero = with_ranges_forced(cashout_output, {SHARE_NAME: 0.0, FALSE_RATE_NAME: 20_000.0})
    bank_one = with_ranges_forced(cashout_output, {SHARE_NAME: 1.0, FALSE_RATE_NAME: 20_000.0})
    labels_zero, labels_full = build_label_events(bank_zero), build_label_events(bank_one)
    assert labels_zero == labels_full
    assert len(labels_zero) > 0
    assert len(build_registry_reports(bank_zero, labels_zero)) == 0
    assert len(build_registry_reports(bank_one, labels_full)) > 0


def test_observed_reporting_share_matches_the_drawn_share(labels_config: dict[str, Any]) -> None:
    """Checks, on a large synthetic confirmed list, the reported share is within tolerance of the drawn share."""
    sample_size = 5000
    assert sample_size >= labels_config["rate_check_min_sample"]
    world_ranges = read_world_ranges()
    confirmed = [(f"acc_{index}", index % 3, 1000 + index, (f"r-L{index:06d}",)) for index in range(sample_size)]

    reports = plan_registry_reports(
        confirmed, {}, world_ranges, SPLIT, labels_config, np.random.default_rng(71), "r", 6
    )

    drawn_share = draw_bank_registry_report_share(world_ranges, SPLIT, np.random.default_rng(71), labels_config)
    standard_error = math.sqrt(drawn_share * (1 - drawn_share) / sample_size)
    assert abs(len(reports) / sample_size - drawn_share) <= labels_config["rate_check_standard_errors"] * standard_error


def test_report_delays_are_inside_the_bounds_and_added_to_the_confirmed_minute(labels_config: dict[str, Any]) -> None:
    """Checks every report arrives confirmed_minute plus a delay inside the configured bounds, with a fingerprint."""
    world_ranges = read_world_ranges()
    low, high = resolve_registry_delay_bounds_minutes(SPLIT, labels_config)
    confirmed = [(f"acc_{index}", 0, 500 + index, (f"r-L{index:06d}",)) for index in range(3000)]
    fingerprints = {"acc_0": "fp_zero"}

    reports = plan_registry_reports(
        confirmed, fingerprints, world_ranges, SPLIT, labels_config, np.random.default_rng(81), "r", 6
    )

    assert len(reports) > 0
    for report in reports:
        assert low <= report.reported_minute - report.confirmed_minute <= high
        assert report.report_id.startswith("r-R")
    assert [report.report_id for report in reports] == [f"r-R{position:06d}" for position in range(1, len(reports) + 1)]
    by_account = {report.account_id: report for report in reports}
    if "acc_0" in by_account:
        assert by_account["acc_0"].device_fingerprint == "fp_zero"
    assert all(report.device_fingerprint is None for report in reports if report.account_id != "acc_0")


def test_confirmation_uses_the_first_complaint_and_only_ids_at_that_minute() -> None:
    """Checks confirm_accounts takes the earliest arrival, whatever the input order, and keeps ids up to it."""
    later = make_complaint("r-L000002", 20)
    first = make_complaint("r-L000001", 10)
    tied = make_complaint("r-L000003", 10)
    other = make_complaint("r-L000004", 5, account="acc_y")

    confirmed = confirm_accounts([later, tied, other, first])

    by_account = {item[0]: item for item in confirmed}
    assert by_account["acc_x"][2] == 10
    assert by_account["acc_x"][3] == ("r-L000001", "r-L000003")
    assert by_account["acc_y"][2] == 5
    assert [item[0] for item in confirmed] == ["acc_y", "acc_x"]


def test_unknown_mule_side_role_and_bad_delay_pairs_are_refused(labels_config: dict[str, Any]) -> None:
    """Checks validate_labels_config refuses a role missing from truth_store and malformed delay pairs."""
    unknown_role = copy.deepcopy(labels_config)
    unknown_role["mule_side_roles"] = ["mule", "kingpin"]
    with pytest.raises(ValueError, match="kingpin"):
        validate_labels_config(unknown_role)

    empty_roles = copy.deepcopy(labels_config)
    empty_roles["mule_side_roles"] = []
    with pytest.raises(ValueError, match="non-empty"):
        validate_labels_config(empty_roles)

    inverted = copy.deepcopy(labels_config)
    inverted["registry_report_delay_minutes"]["test"] = [500, 100]
    with pytest.raises(ValueError, match="low <= high"):
        validate_labels_config(inverted)

    negative = copy.deepcopy(labels_config)
    negative["registry_report_delay_minutes"]["stress"] = [-1, 100]
    with pytest.raises(ValueError, match="low <= high"):
        validate_labels_config(negative)

