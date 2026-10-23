"""
Tests for the complaint planners (world/complaints.py) and the label records and config readers
(world/label_records.py). Every test uses large synthetic inputs and derives its expected numbers from
config/labels.yaml and config/world_ranges.yaml, never from typed-in values.
Covers the complaint share, delays by scam type, stacking, false-complaint totals and the shop-to-person ratio,
draw counts that do not depend on settings, refusals, label numbering and the shape of a label event.

TERM: standard error: the typical size of chance error in a measured share, mean or count.
TERM: tolerance: the configured number of standard errors a measurement may differ from its expected value.
"""

import copy
import dataclasses
import math
from typing import Any

import numpy as np
import pytest

from world.complaints import assign_label_ids, plan_false_complaints, plan_true_complaints
from world.config_reader import get_parameter_entry, read_world_ranges
from world.device_schedule import MINUTES_PER_DAY
from world.label_records import (
    GenuinePayment,
    LabelEvent,
    LabelWorldSettings,
    VictimLoss,
    draw_label_world_settings,
    load_labels_config,
    resolve_delay_bounds_minutes,
    validate_labels_config,
)
from world.scam_config import resolve_remainder_scam_type
from world.scam_planner import SCAM_INVESTMENT, SCAM_JOB

SPLIT = "training"
LAST_PAID_MINUTE = 10_000
FORBIDDEN_FIELD_WORDS = ("false", "true", "gang", "role", "story", "truth", "mule", "victim", "fraud")


@pytest.fixture(scope="module")
def config() -> dict[str, Any]:
    """Loads and validates the labels config once for the module."""
    return load_labels_config()


@pytest.fixture(scope="module")
def world_ranges() -> dict[str, Any]:
    """Reads the unlayered world ranges from disk once for the module."""
    return read_world_ranges()


def make_losses(count: int, scam_type: str, mule: str = "acc_mule") -> list[VictimLoss]:
    """Builds `count` synthetic losses of one scam type, one distinct victim each, all paying the same mule."""
    return [
        VictimLoss(f"acc_v{index}", index % 3, mule, 2, scam_type, (f"p{index}",), LAST_PAID_MINUTE)
        for index in range(count)
    ]


def make_payments(count: int, shop_fraction: float) -> list[GenuinePayment]:
    """Builds `count` synthetic genuine payments; the first `shop_fraction` of them go to a shop account."""
    shop_total = int(count * shop_fraction)
    payments: list[GenuinePayment] = []
    for index in range(count):
        is_shop = index < shop_total
        payee = "acc_shop_0" if is_shop else "acc_person_0"
        payments.append(GenuinePayment(f"g{index}", index % 5000, f"acc_p{index % 997}", 0, payee, 1, is_shop))
    return payments


def forced_settings(share: float = 1.0, rate: float = 1.0, weight: float = 5.0) -> LabelWorldSettings:
    """Builds settings directly, so a test can force the complaint share without drawing it."""
    return LabelWorldSettings(share, rate, weight)


def test_victim_complaint_share_matches_the_drawn_share(world_ranges: dict[str, Any], config: dict[str, Any]) -> None:
    """Checks the share of complaining victims is within the configured standard errors of the drawn share."""
    settings = draw_label_world_settings(world_ranges, SPLIT, np.random.default_rng(11), config)
    sample_size = 5000
    assert sample_size >= config["rate_check_min_sample"]

    events = plan_true_complaints(
        make_losses(sample_size, SCAM_INVESTMENT), settings, world_ranges, SPLIT, np.random.default_rng(12), config
    )

    share = settings.victim_complaint_share
    standard_error = math.sqrt(share * (1 - share) / sample_size)
    assert abs(len(events) / sample_size - share) <= config["rate_check_standard_errors"] * standard_error


@pytest.mark.parametrize("scam_type", [SCAM_INVESTMENT, SCAM_JOB, resolve_remainder_scam_type()])
def test_delays_stay_in_bounds_and_average_the_midpoint(
    scam_type: str, world_ranges: dict[str, Any], config: dict[str, Any]
) -> None:
    """Checks, per scam type with everyone complaining, each delay is inside its bounds and the mean is central."""
    sample_size = 3000
    low, high = resolve_delay_bounds_minutes(world_ranges, SPLIT, scam_type, config)

    events = plan_true_complaints(
        make_losses(sample_size, scam_type), forced_settings(share=1.0), world_ranges, SPLIT,
        np.random.default_rng(21), config,
    )

    assert len(events) == sample_size
    delays = [event.arrival_minute - LAST_PAID_MINUTE for event in events]
    assert all(low <= delay <= high for delay in delays)
    # A whole-minute uniform delay has variance ((high - low + 1)^2 - 1) / 12.
    standard_error = math.sqrt(((high - low + 1) ** 2 - 1) / 12) / math.sqrt(sample_size)
    midpoint = (low + high) / 2
    assert abs(sum(delays) / sample_size - midpoint) <= config["rate_check_standard_errors"] * standard_error


def test_investment_delays_are_longer_than_unauthorised_debit_delays(
    world_ranges: dict[str, Any], config: dict[str, Any]
) -> None:
    """Checks, on the training ranges, even the shortest investment delay exceeds the longest debit delay."""
    investment_low, _ = resolve_delay_bounds_minutes(world_ranges, SPLIT, SCAM_INVESTMENT, config)
    _, debit_high = resolve_delay_bounds_minutes(world_ranges, SPLIT, resolve_remainder_scam_type(), config)
    assert investment_low > debit_high


def test_day_ranges_are_converted_to_whole_minutes(world_ranges: dict[str, Any], config: dict[str, Any]) -> None:
    """Checks the investment bounds equal the configured day range times minutes per day, derived in the test."""
    reference = config["complaint_delay_by_scam_type"][SCAM_INVESTMENT]["world_ranges_reference"]
    day_low, day_high = get_parameter_entry(world_ranges, reference)[SPLIT]
    assert resolve_delay_bounds_minutes(world_ranges, SPLIT, SCAM_INVESTMENT, config) == (
        round(day_low * MINUTES_PER_DAY), round(day_high * MINUTES_PER_DAY),
    )


def test_ten_losses_to_one_mule_give_ten_complaints_about_that_mule(
    world_ranges: dict[str, Any], config: dict[str, Any]
) -> None:
    """Checks stacking: with share 1.0, ten victims of one mule give ten complaints, all reporting that mule."""
    events = plan_true_complaints(
        make_losses(10, SCAM_JOB, mule="acc_stacked"), forced_settings(share=1.0), world_ranges, SPLIT,
        np.random.default_rng(31), config,
    )
    assert len(events) == 10
    assert {event.reported_account for event in events} == {"acc_stacked"}
    assert len({event.complainant_account for event in events}) == 10


def test_false_complaints_match_the_expected_total_and_the_merchant_ratio(
    world_ranges: dict[str, Any], config: dict[str, Any]
) -> None:
    """
    Checks the false-complaint total against the expected total and the shop/person rate ratio against the weight.

    The training range tops out at 2 per 100,000, so 200,000 payments would give only a few events and a ratio
    test with no power. The rate is therefore the top of the stress range, and the weight is the top of the
    training range, both read from config.
    """
    payment_count = 600_000
    rate_name = config["false_complaint_rate_per_100000"]["world_ranges_reference"]
    weight_name = config["false_complaint_merchant_weight"]["world_ranges_reference"]
    rate = get_parameter_entry(world_ranges, rate_name)["stress"][1]
    weight = get_parameter_entry(world_ranges, weight_name)[SPLIT][1]
    settings = forced_settings(share=0.5, rate=rate, weight=weight)
    payments = make_payments(payment_count, shop_fraction=0.5)
    shop_count = sum(1 for payment in payments if payment.payee_is_shop)
    person_count = payment_count - shop_count
    tolerance = config["rate_check_standard_errors"]

    events = plan_false_complaints(payments, settings, world_ranges, SPLIT, np.random.default_rng(41), config)

    # (a) The total is Poisson-like, so its standard error is the square root of the expected total.
    expected_total = rate * payment_count / 100_000
    assert abs(len(events) - expected_total) <= tolerance * math.sqrt(expected_total)
    # (b) Delta method: for R = shop_events / person_events, Var(R) / R^2 is about 1/E_shop + 1/E_person, where E
    # are the expected event counts of each group. The observed ratio must be within tolerance standard errors of
    # the weight, with R equal to the weight in expectation.
    shop_ids = {payment.payment_id for payment in payments if payment.payee_is_shop}
    observed_shop = sum(1 for event in events if event.payment_ids[0] in shop_ids)
    observed_person = len(events) - observed_shop
    assert observed_person > 0
    person_chance = expected_total / (person_count + weight * shop_count)
    expected_person = person_chance * person_count
    expected_shop = weight * person_chance * shop_count
    relative_error = math.sqrt(1 / expected_shop + 1 / expected_person)
    observed_ratio_of_rates = (observed_shop / shop_count) / (observed_person / person_count)
    assert abs(observed_ratio_of_rates - weight) <= tolerance * relative_error * weight


def test_draw_count_does_not_depend_on_settings(world_ranges: dict[str, Any], config: dict[str, Any]) -> None:
    """Checks two different settings leave each generator in the same state, so draw counts match."""
    losses = make_losses(500, SCAM_INVESTMENT)
    payments = make_payments(1000, shop_fraction=0.5)
    settings_a = forced_settings(share=0.0, rate=0.5, weight=2.0)
    settings_b = forced_settings(share=1.0, rate=5.0, weight=9.0)

    for planner, items in ((plan_true_complaints, losses), (plan_false_complaints, payments)):
        rng_a, rng_b = np.random.default_rng(51), np.random.default_rng(51)
        planner(items, settings_a, world_ranges, SPLIT, rng_a, config)  # type: ignore[operator]
        planner(items, settings_b, world_ranges, SPLIT, rng_b, config)  # type: ignore[operator]
        assert rng_a.random() == rng_b.random()


def test_settings_draw_uses_exactly_three_draws(world_ranges: dict[str, Any], config: dict[str, Any]) -> None:
    """Checks draw_label_world_settings leaves the generator exactly three uniform draws further on."""
    rng = np.random.default_rng(61)
    draw_label_world_settings(world_ranges, SPLIT, rng, config)
    reference = np.random.default_rng(61)
    reference.random(3)
    assert rng.random() == reference.random()


def test_refusals(world_ranges: dict[str, Any], config: dict[str, Any]) -> None:
    """Checks a chance above 1, an unknown scam type, a key mismatch and `reason` as a scam type are all refused."""
    too_many = forced_settings(rate=1_000_000.0)
    with pytest.raises(ValueError, match="cannot exceed 1"):
        plan_false_complaints(make_payments(10, 0.5), too_many, world_ranges, SPLIT, np.random.default_rng(1), config)

    mystery_loss = [VictimLoss("acc_v", 0, "acc_m", 1, "mystery", ("p1",), 5)]
    with pytest.raises(ValueError, match="Unknown scam type"):
        plan_true_complaints(mystery_loss, forced_settings(), world_ranges, SPLIT, np.random.default_rng(1), config)

    missing_job = copy.deepcopy(config)
    del missing_job["complaint_delay_by_scam_type"][SCAM_JOB]
    with pytest.raises(ValueError, match="must equal the scam types"):
        validate_labels_config(missing_job)
    extra_type = copy.deepcopy(config)
    extra_type["complaint_delay_by_scam_type"]["romance"] = copy.deepcopy(
        config["complaint_delay_by_scam_type"][SCAM_JOB]
    )
    with pytest.raises(ValueError, match="must equal the scam types"):
        validate_labels_config(extra_type)

    # `reason` documents the block: it passes validation but is not a scam type with a delay range.
    validate_labels_config(config)
    with pytest.raises(ValueError, match="Unknown scam type"):
        resolve_delay_bounds_minutes(world_ranges, SPLIT, "reason", config)


def test_assign_label_ids_is_deterministic_sorted_and_refuses_ties() -> None:
    """Checks ids follow the sort order, repeat exactly, are zero-padded and exact ties are refused."""
    def event(minute: int, account: str, payment: str) -> LabelEvent:
        """Builds a minimal event for numbering tests."""
        return LabelEvent("", "complaint", minute, account, 0, "acc_r", 1, (payment,), "investment")

    events = [event(30, "acc_b", "p3"), event(10, "acc_z", "p1"), event(10, "acc_a", "p2")]
    numbered = assign_label_ids(events, "run1", 4)

    assert [item.label_id for item in numbered] == ["run1-L0001", "run1-L0002", "run1-L0003"]
    assert [item.payment_ids[0] for item in numbered] == ["p2", "p1", "p3"]
    assert assign_label_ids(list(reversed(events)), "run1", 4) == numbered
    assert events[0].label_id == ""
    with pytest.raises(ValueError, match="tie"):
        assign_label_ids([event(10, "acc_a", "p1"), event(10, "acc_a", "p1")], "run1", 4)


def test_label_event_has_no_field_that_reveals_truth() -> None:
    """Checks no LabelEvent field name contains a word that would tell whether the complaint is true."""
    names = [field.name for field in dataclasses.fields(LabelEvent)]
    for name in names:
        assert not any(word in name for word in FORBIDDEN_FIELD_WORDS), name
