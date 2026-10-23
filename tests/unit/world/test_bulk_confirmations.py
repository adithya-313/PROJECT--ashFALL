"""
Tests for bulk confirmations (world/bulk_confirmations.py, BulkConfirmation in world/label_records.py and
build_bulk_confirmations in world/label_builder.py).
Covers that, with the labelled short-lag fixture layer, a 12c scam world gets crackdowns whose accounts are mule-side
in truth and sit in the crackdown's region, that a crackdown asking for more accounts than exist records both
numbers, that a world without the fixture can have no crackdown, that the same seed repeats exactly, and that
accounts opened mid-run are never swept and are counted.

TERM: crackdown: one bulk confirmation event with a cut-off minute, a region, a requested size and a delay.
TERM: fixture layer: showcase/fixtures/level_12/bulk_short_lag.yaml, a labelled non-real range set (issue 12-12).
"""

import copy
import dataclasses
from typing import Any

import pytest

from showcase.level_12_support_b import get_showcase_b_settings
from showcase.level_12_support_c import load_showcase_c_fixture
from world.interactive_records import InteractiveWorldOutput, RunnerSettings
from world.interactive_runner import run_interactive_world
from world.label_builder import build_bulk_confirmations
from world.label_records import BulkConfirmation, load_labels_config
from world.layers import Layer, read_layer
from world.run_identity import REPO_ROOT

FIXTURE_LAYER_PATH = REPO_ROOT / "showcase" / "fixtures" / "level_12" / "bulk_short_lag.yaml"
RATE_NAME = "bulk_confirmation_events_per_year"
SIZE_NAME = "bulk_confirmation_size_accounts"


@pytest.fixture(scope="module")
def labels_config() -> dict[str, Any]:
    """Loads and validates the labels config once for the module."""
    return load_labels_config()


@pytest.fixture(scope="module")
def scam_output() -> InteractiveWorldOutput:
    """Runs the Part C victim-to-cash-out fixture gang once (61 days, seed 101), shared by all tests."""
    settings, code_version = get_showcase_b_settings()
    return run_interactive_world(
        settings, RunnerSettings(fraud_enabled=True), gang_configs=(load_showcase_c_fixture(),),
        code_version=code_version,
    )


def with_ranges(output: InteractiveWorldOutput, changes: dict[str, list[float]]) -> InteractiveWorldOutput:
    """
    Returns the output with each named world_ranges entry set to the given pair for the run's split, in a copy of
    the effective configuration. The run itself is untouched; only label planning reads these entries.
    """
    new_config = copy.deepcopy(output.effective_config)
    split = new_config["settings"]["range_name"]
    parameters = new_config["world_configs"]["world_ranges"]["parameters"]
    for name, pair in changes.items():
        parameters[name][split] = list(pair)
    return dataclasses.replace(output, effective_config=new_config)


def with_fixture_layer(output: InteractiveWorldOutput) -> InteractiveWorldOutput:
    """Applies the training pairs of the bulk_short_lag layer file to a copy of the effective configuration."""
    layer: Layer = read_layer(FIXTURE_LAYER_PATH)
    pairs = {name: ranges["training"] for name, ranges in layer.changes["world_ranges"].items()}
    return with_ranges(output, pairs)


def region_of_account(output: InteractiveWorldOutput) -> dict[str, int]:
    """Maps every person and shop account of the world to its region."""
    regions = {person.account_id: person.region for person in output.world.people}
    regions.update({shop.account_id: shop.region for shop in output.world.shops})
    return regions


def mule_side_starts(output: InteractiveWorldOutput, labels_config: dict[str, Any]) -> dict[str, list[int]]:
    """Maps each account to the start minutes of its mule-side role spans in truth."""
    starts: dict[str, list[int]] = {}
    for span in output.result.truth.roles:
        if span.role in labels_config["mule_side_roles"]:
            starts.setdefault(span.account_id, []).append(span.valid_from)
    return starts


def test_fixture_layer_gives_batches_with_mule_side_accounts_in_their_region(
    scam_output: InteractiveWorldOutput, labels_config: dict[str, Any]
) -> None:
    """
    Checks the fixture layer gives at least one batch, and every confirmed account is mule-side in truth with a
    role start at or before the cut-off and sits in the batch's region.

    With the fixture values (50 crackdowns a year, 3 to 5 days of lag) the seed-101 world gets 9 crackdowns, of
    which 2 hold the one mule-side account that has a region.
    """
    layered = with_fixture_layer(scam_output)

    result = build_bulk_confirmations(layered)

    assert len(result.confirmations) >= 1
    regions = region_of_account(layered)
    starts = mule_side_starts(layered, labels_config)
    confirmed_accounts = 0
    for batch in result.confirmations:
        assert isinstance(batch, BulkConfirmation)
        for account_id in batch.account_ids:
            confirmed_accounts += 1
            assert account_id in starts
            assert min(starts[account_id]) <= batch.cutoff_minute
            assert regions[account_id] == batch.region
    assert confirmed_accounts > 0


def test_requested_size_above_available_records_both_numbers(
    scam_output: InteractiveWorldOutput, labels_config: dict[str, Any]
) -> None:
    """Checks a crackdown asking for 10,000 accounts confirms fewer, and requested and confirmed are both recorded."""
    layered = with_ranges(with_fixture_layer(scam_output), {SIZE_NAME: [10_000, 10_000]})
    regions = region_of_account(layered)
    starts = mule_side_starts(layered, labels_config)

    result = build_bulk_confirmations(layered)

    assert len(result.confirmations) >= 1
    for batch in result.confirmations:
        assert batch.requested_size == 10_000
        eligible = [
            account_id for account_id, starts_list in starts.items()
            if account_id in regions and regions[account_id] == batch.region and min(starts_list) <= batch.cutoff_minute
        ]
        assert len(batch.account_ids) == len(eligible)
        assert len(batch.account_ids) < batch.requested_size
    assert any(len(batch.account_ids) > 0 for batch in result.confirmations)


def test_training_world_without_the_fixture_can_have_no_crackdown(scam_output: InteractiveWorldOutput) -> None:
    """Checks the unmodified training ranges give a valid (possibly empty) result, and a zero rate gives none."""
    result = build_bulk_confirmations(scam_output)
    assert all(isinstance(batch, BulkConfirmation) for batch in result.confirmations)

    no_crackdowns = build_bulk_confirmations(with_ranges(scam_output, {RATE_NAME: [0, 0]}))
    assert no_crackdowns.confirmations == ()


def test_same_seed_gives_identical_batches(scam_output: InteractiveWorldOutput) -> None:
    """Checks building twice from the same world and configuration gives equal batches, ids included."""
    layered = with_fixture_layer(scam_output)

    first = build_bulk_confirmations(layered)
    second = build_bulk_confirmations(layered)

    assert first == second
    assert len(first.confirmations) > 0
    assert [batch.batch_id for batch in first.confirmations][0].endswith("-B000001")


def test_accounts_opened_mid_run_are_never_swept_and_are_counted(
    scam_output: InteractiveWorldOutput, labels_config: dict[str, Any]
) -> None:
    """Checks no opened account is in a batch, and the excluded count equals the mule-side accounts with no region."""
    layered = with_fixture_layer(scam_output)
    regions = region_of_account(layered)
    starts = mule_side_starts(layered, labels_config)
    opened_ids = {opened[1] for opened in layered.result.opened_accounts}
    expected_without_region = [account_id for account_id in starts if account_id not in regions]
    assert len(expected_without_region) > 0

    result = build_bulk_confirmations(layered)

    swept = {account_id for batch in result.confirmations for account_id in batch.account_ids}
    assert not swept & set(expected_without_region)
    assert not swept & opened_ids
    assert len(result.mule_side_accounts_without_region) == len(expected_without_region)
    assert set(result.mule_side_accounts_without_region) == set(expected_without_region)
