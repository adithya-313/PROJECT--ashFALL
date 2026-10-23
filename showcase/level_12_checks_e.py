"""
WHAT: Implementation of Level 12 Showcase Part E checks (Checks 21 through 25).
Shows complaints arriving against the truth in a scam world, complaint delays by scam type, a simulated analyst whose
accuracy falls as the queue grows, one bulk-label crackdown that confirms mule accounts of one region late, and the
complaint share measured on a large synthetic batch.

WHY: Verifies Level 12e: labels built after the run from the finished result, an analyst that is a pure function,
and crackdown confirmations, each measured from real code on real or clearly labelled synthetic inputs
(amendments A-020, A-021 and A-022).

TERM: complaint: a label event in which a victim (true) or an ordinary payer (false) says a payment was fraud.
TERM: queue load: open cases divided by the analyst team's daily capacity in cases.
TERM: crackdown: one bulk confirmation of mule-side accounts of one region, arriving after a lag.
TERM: synthetic input: made-up records fed to a real function, so a statistical check has enough cases.
"""

import math
from statistics import median
from typing import Any

from showcase.framework import CheckOutcome
from showcase.level_12_support_e import fraud_story_by_payment_id, get_bulk_run, get_complaints_run
from world.analyst import (
    AnalystParams,
    decide_case,
    expected_accuracy,
    load_analyst_config,
)
from world.complaints import plan_true_complaints
from world.config_reader import read_world_ranges
from world.device_schedule import MINUTES_PER_DAY
from world.interactive_loop import build_component_streams
from world.label_builder import collect_victim_losses
from world.label_records import (
    LabelWorldSettings,
    VictimLoss,
    draw_label_world_settings,
    load_labels_config,
    resolve_delay_bounds_minutes,
)
from world.scam_config import resolve_remainder_scam_type
from world.scam_planner import SCAM_INVESTMENT, SCAM_JOB
from world.truth_records import SCAM_STEP_PAYOUT_BACK, SCAM_STEP_VICTIM_PAYMENT

SPLIT = "training"
SYNTHETIC_LOSSES_PER_TYPE = 1000
COMPLAINT_SHARE_SAMPLE = 5000
ANALYST_CASES_PER_LOAD = 5000
ANALYST_LOADS = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
ANALYST_EVIDENCE_STRENGTH = 0.5


def _share_tolerance(share: float, sample_size: int, standard_errors: float) -> float:
    """Returns the allowed distance between an observed and an expected share: standard_errors binomial errors."""
    return standard_errors * math.sqrt(share * (1.0 - share) / sample_size)


async def run_complaints_against_truth_check(workspace: Any) -> CheckOutcome:
    """
    WHAT: Evaluates the complaints of a scam world judged against the in-memory truth. Every complaint that names a
    gang payment must name only successful victim_payment events of the complainant to the reported account, and no
    payout_back id may appear on any complaint. The complaint share is not judged here: a world of 17 victims is too
    small for that, and check 25 proves it on a large batch.
    WHY: Proves Check 21: complaints are about real losses only, and a payout back is never complained about.
    Args:
        workspace: Optional execution workspace supplied by the showcase framework.
    Returns:
        CheckOutcome with the complaint count, the victim count and a delay chart.
    Example:
        outcome = await run_complaints_against_truth_check(None)
        assert "all naming real victim payments" in outcome.key_number
    """
    output, labels, _drawn = get_complaints_run()
    stories = fraud_story_by_payment_id(output)
    events_by_id = {
        output.result.payment_ids[position]: event for position, event in enumerate(output.result.run_result.events)
    }
    losses = collect_victim_losses(output)
    loss_by_payments = {loss.payment_ids: loss for loss in losses}

    for label in labels:
        for payment_id in label.payment_ids:
            if payment_id in stories:
                step = stories[payment_id].scam_step
                assert step != SCAM_STEP_PAYOUT_BACK, f"Payout back {payment_id} was complained about"
    true_labels = [label for label in labels if label.payment_ids[0] in stories]
    assert true_labels, "Expected at least one true complaint without forcing the complaint share"
    for label in true_labels:
        assert label.payment_ids in loss_by_payments, f"Complaint {label.label_id} does not match any victim loss"
        for payment_id in label.payment_ids:
            event = events_by_id[payment_id]
            assert stories[payment_id].scam_step == SCAM_STEP_VICTIM_PAYMENT, f"{payment_id} is not a victim payment"
            assert event.status == "success", f"{payment_id} did not succeed"
            assert event.from_account == label.complainant_account, f"{payment_id} was not paid by the complainant"
            assert event.to_account == label.reported_account, f"{payment_id} did not pay the reported account"

    loss_count = len(losses)
    points = [(0.0, 0.0)]
    delays_days = sorted(
        (label.arrival_minute - loss_by_payments[label.payment_ids].last_paid_minute) / MINUTES_PER_DAY
        for label in true_labels
    )
    for position, delay_days in enumerate(delays_days, start=1):
        points.append((delay_days, 100.0 * position / loss_count))
    chart = {
        "type": "timeline", "title": "Victims who have complained, by days after their last payment",
        "x_label": "days after the victim's last payment", "y_label": "cumulative % of victims who complained",
        "points": points, "shaded": [],
    }
    key_number = f"{len(true_labels)} complaints from {loss_count} victims, all naming real victim payments"
    evidence = {"complained": len(true_labels), "victim_losses": loss_count, "labels_total": len(labels)}
    return CheckOutcome(key_number=key_number, evidence=evidence, chart=chart)


async def run_complaint_delay_by_type_check(workspace: Any) -> CheckOutcome:
    """
    WHAT: Runs the real plan_true_complaints on 1,000 synthetic victim losses per scam type, with the complaint share
    forced to 1.0 by building the settings here. It asserts every delay is inside its type's bounds, that the
    configured investment minimum is above the configured unauthorised-debit maximum, and that the same holds for the
    drawn delays. The inputs are synthetic: made-up losses, not a simulated world.
    WHY: Proves Check 22: complaint delays follow the scam type, from minutes for a debit to days for an investment.
    Args:
        workspace: Optional execution workspace supplied by the showcase framework.
    Returns:
        CheckOutcome with the median delay per type and a delay chart with one band per type.
    Example:
        outcome = await run_complaint_delay_by_type_check(None)
        assert "median" in outcome.key_number
    """
    world_ranges = read_world_ranges()
    labels_config = load_labels_config()
    settings = LabelWorldSettings(victim_complaint_share=1.0, false_complaint_rate_per_100000=1.0,
                                  false_complaint_merchant_weight=2.0)
    scam_types = [SCAM_INVESTMENT, SCAM_JOB, resolve_remainder_scam_type()]
    last_paid_minute = 10_000
    delays_by_type: dict[str, list[int]] = {}
    for type_number, scam_type in enumerate(scam_types):
        losses = [
            VictimLoss(f"acc_v{index}", 0, "acc_m", 1, scam_type, (f"p{index}",), last_paid_minute)
            for index in range(SYNTHETIC_LOSSES_PER_TYPE)
        ]
        rng = build_component_streams(220 + type_number, {"check_22": 0})["check_22"]
        events = plan_true_complaints(losses, settings, world_ranges, SPLIT, rng, labels_config)
        assert len(events) == SYNTHETIC_LOSSES_PER_TYPE, "Share 1.0 must make every loss complain"
        delays_by_type[scam_type] = sorted(event.arrival_minute - last_paid_minute for event in events)
        low, high = resolve_delay_bounds_minutes(world_ranges, SPLIT, scam_type, labels_config)
        smallest, largest = delays_by_type[scam_type][0], delays_by_type[scam_type][-1]
        assert low <= smallest and largest <= high, f"{scam_type} delay out of bounds"

    debit_type = resolve_remainder_scam_type()
    investment_low = resolve_delay_bounds_minutes(world_ranges, SPLIT, SCAM_INVESTMENT, labels_config)[0]
    debit_high = resolve_delay_bounds_minutes(world_ranges, SPLIT, debit_type, labels_config)[1]
    assert investment_low > debit_high, "Investment delays must start after unauthorised-debit delays end"
    assert min(delays_by_type[SCAM_INVESTMENT]) > max(delays_by_type[debit_type]), "Drawn delays overlap across types"

    points: list[tuple[float, float]] = []
    shaded: list[tuple[float, float, str]] = []
    for scam_type in scam_types:
        start_index = len(points)
        points.extend((float(start_index + offset), delay / MINUTES_PER_DAY)
                      for offset, delay in enumerate(delays_by_type[scam_type]))
        shaded.append((float(start_index), float(len(points) - 1), scam_type))
    chart = {
        "type": "timeline", "title": "Complaint delay by scam type (synthetic losses)",
        "x_label": "complaint (sorted by scam type)", "y_label": "delay in days", "points": points, "shaded": shaded,
    }
    medians = {scam_type: median(delays) for scam_type, delays in delays_by_type.items()}
    parts = [f"{scam_type}: {medians[scam_type]:.0f} min ({medians[scam_type] / MINUTES_PER_DAY:.2f} days)"
             for scam_type in scam_types]
    key_number = "median delay, synthetic losses: " + "; ".join(parts)
    evidence = {
        "inputs": f"synthetic: {SYNTHETIC_LOSSES_PER_TYPE} made-up victim losses per scam type, share forced to 1.0",
        "median_minutes": medians, "investment_min_minutes": min(delays_by_type[SCAM_INVESTMENT]),
        "debit_max_minutes": max(delays_by_type[debit_type]),
    }
    return CheckOutcome(key_number=key_number, evidence=evidence, chart=chart)


def _midpoint(world_ranges: dict[str, Any], name: str) -> float:
    """Returns the middle of the training range of a world_ranges entry."""
    low, high = world_ranges["parameters"][name][SPLIT]
    return (float(low) + float(high)) / 2.0


async def run_analyst_queue_check(workspace: Any) -> CheckOutcome:
    """
    WHAT: Builds analyst parameters at the training midpoints of the configuration, then for each queue load from
    0.5 to 6 computes the expected accuracy and measures it over 5,000 seeded decisions on cases of medium evidence
    strength with no model score. It asserts expected accuracy never rises with load, that it is lower at load 6 than at
    load 1, and that each measured accuracy is within the configured standard errors of the expected one.
    WHY: Proves Check 23: the simulated analyst gets worse as the queue outgrows the team, in the way A-022 states.
    Args:
        workspace: Optional execution workspace supplied by the showcase framework.
    Returns:
        CheckOutcome with the measured accuracy at load 1 and load 6 and a load-versus-accuracy chart.
    Example:
        outcome = await run_analyst_queue_check(None)
        assert "at load 6" in outcome.key_number
    """
    world_ranges = read_world_ranges()
    analyst_config = load_analyst_config()
    standard_errors = load_labels_config()["rate_check_standard_errors"]
    minutes_low, minutes_high = analyst_config["minutes_per_case"][SPLIT]
    analysts_low, analysts_high = analyst_config["analysts_on_shift"][SPLIT]
    def reference(key: str) -> str:
        """Returns the world_ranges entry name the analyst config uses for one accuracy setting."""
        return str(analyst_config[key]["world_ranges_reference"])

    params = AnalystParams(
        accuracy_base=_midpoint(world_ranges, reference("accuracy_base")),
        drop_weak_evidence=_midpoint(world_ranges, reference("accuracy_drop_when_evidence_weak")),
        drop_overload=_midpoint(world_ranges, reference("accuracy_drop_when_queue_overloaded")),
        follows_score_share=_midpoint(world_ranges, reference("follows_model_score_share")),
        minutes_per_case=int(round((minutes_low + minutes_high) / 2)),
        analysts_on_shift=int(round((analysts_low + analysts_high) / 2)),
        shift_minutes=analyst_config["shift_minutes"]["value"],
        accuracy_floor=float(analyst_config["accuracy_floor"]["value"]),
        score_threshold=float(analyst_config["score_threshold"]["value"]),
    )

    rng = build_component_streams(23, {"check_23": 0})["check_23"]
    expected_by_load: list[float] = []
    measured_by_load: list[float] = []
    for load in ANALYST_LOADS:
        expected = expected_accuracy(params, ANALYST_EVIDENCE_STRENGTH, load)[0]
        draws = rng.random((ANALYST_CASES_PER_LOAD, 2))
        correct = 0
        for case_index in range(ANALYST_CASES_PER_LOAD):
            is_mule = case_index % 2 == 0
            outcome = decide_case(
                is_mule, ANALYST_EVIDENCE_STRENGTH, load, None, params,
                float(draws[case_index, 0]), float(draws[case_index, 1]),
            )
            if outcome.verdict_is_mule == is_mule:
                correct += 1
        measured = correct / ANALYST_CASES_PER_LOAD
        tolerance = _share_tolerance(expected, ANALYST_CASES_PER_LOAD, standard_errors)
        assert abs(measured - expected) <= tolerance, f"Measured accuracy {measured} is off {expected} at load {load}"
        expected_by_load.append(expected)
        measured_by_load.append(measured)

    never_rises = all(later <= earlier for earlier, later in zip(expected_by_load, expected_by_load[1:]))
    assert never_rises, "Accuracy rose with load"
    at_one = expected_by_load[ANALYST_LOADS.index(1.0)]
    at_six = expected_by_load[ANALYST_LOADS.index(6.0)]
    assert at_six < at_one, "Accuracy at load 6 must be below accuracy at load 1"

    measured_one = measured_by_load[ANALYST_LOADS.index(1.0)]
    measured_six = measured_by_load[ANALYST_LOADS.index(6.0)]
    chart = {
        "type": "timeline", "title": "Measured analyst accuracy by queue load",
        "x_label": "queue load (open cases / daily capacity)", "y_label": "share of verdicts correct",
        "points": list(zip(ANALYST_LOADS, measured_by_load)), "shaded": [],
    }
    key_number = (
        f"accuracy {measured_one:.3f} at load 1 -> {measured_six:.3f} at load 6 (measured, "
        f"{ANALYST_CASES_PER_LOAD} cases per load)"
    )
    evidence = {
        "loads": ANALYST_LOADS, "expected": expected_by_load, "measured": measured_by_load,
        "evidence_strength": ANALYST_EVIDENCE_STRENGTH, "cases_per_load": ANALYST_CASES_PER_LOAD,
        "daily_capacity_cases": params.analysts_on_shift * params.shift_minutes / params.minutes_per_case,
    }
    return CheckOutcome(key_number=key_number, evidence=evidence, chart=chart)


async def run_bulk_batch_check(workspace: Any) -> CheckOutcome:
    """
    WHAT: Takes the first bulk-confirmation batch of the 12c scam world (under the labelled short-lag fixture) that
    confirmed at least one account, and for every confirmed account asserts that it is mule-side in truth with a role
    start at or before the cut-off, that it lies in the batch's region, and that the batch arrives after the last fraud
    payment the account sent or received at or before the cut-off (an account with no such payment fails the check).
    It also asserts the lag between cut-off and arrival is inside the fixture's lag bounds. The fixture is labelled, not
    a real range.
    WHY: Proves Check 24: a crackdown confirms mules of one region, and its confirmation arrives late relative to
    the fraud it covers. A rented mule's owner keeps paying normally, so "late" is measured against fraud payments.
    Args:
        workspace: Optional execution workspace supplied by the showcase framework.
    Returns:
        CheckOutcome with the batch, its lag and a cumulative confirmed-accounts chart with a lag band.
    Example:
        outcome = await run_bulk_batch_check(None)
        assert "after the cut-off" in outcome.key_number
    """
    output, result, (lag_low, lag_high) = get_bulk_run()
    labels_config = load_labels_config()
    regions = {person.account_id: person.region for person in output.world.people}
    regions.update({shop.account_id: shop.region for shop in output.world.shops})
    role_starts: dict[str, list[int]] = {}
    for span in output.result.truth.roles:
        if span.role in labels_config["mule_side_roles"]:
            role_starts.setdefault(span.account_id, []).append(span.valid_from)
    fraud_minutes_by_account: dict[str, list[int]] = {}
    for position, event in enumerate(output.result.run_result.events):
        if output.result.event_sources[position] == "fraud":
            fraud_minutes_by_account.setdefault(event.from_account, []).append(event.minute)
            fraud_minutes_by_account.setdefault(event.to_account, []).append(event.minute)

    def fraud_minutes_up_to(account_id: str, cutoff_minute: int) -> list[int]:
        """Returns the minutes of the account's fraud events at or before the cut-off, in time order."""
        minutes = fraud_minutes_by_account[account_id] if account_id in fraud_minutes_by_account else []
        return sorted(minute for minute in minutes if minute <= cutoff_minute)

    # A mule role starts at warm-up, so a batch can confirm a mule before any fraud money moved. That is a correct
    # label but not a late one, so such a batch is skipped here, counted, and never used to pass the check.
    batches_with_accounts = [item for item in result.confirmations if item.account_ids]
    batch = None
    skipped_without_prior_fraud = 0
    for candidate in batches_with_accounts:
        if all(fraud_minutes_up_to(account_id, candidate.cutoff_minute) for account_id in candidate.account_ids):
            batch = candidate
            break
        skipped_without_prior_fraud += 1
    assert batch is not None, (
        "no bulk batch confirmed an account after fraud money moved "
        f"({len(result.confirmations)} batches, {len(batches_with_accounts)} with accounts, "
        f"{skipped_without_prior_fraud} skipped for an account with no fraud event before the cut-off)"
    )

    last_fraud_minutes: list[int] = []
    for account_id in batch.account_ids:
        assert account_id in role_starts, f"{account_id} holds no mule-side role in truth"
        assert min(role_starts[account_id]) <= batch.cutoff_minute, f"{account_id} role began after the cut-off"
        assert regions[account_id] == batch.region, f"{account_id} is not in region {batch.region}"
        last_fraud = fraud_minutes_up_to(account_id, batch.cutoff_minute)[-1]
        arrival = batch.arrival_minute
        assert arrival > last_fraud, f"Batch arrives at {arrival}, not after fraud at {last_fraud}"
        last_fraud_minutes.append(last_fraud)
    lag_minutes = batch.arrival_minute - batch.cutoff_minute
    assert lag_low <= lag_minutes <= lag_high, f"Lag {lag_minutes} is outside the fixture bounds {lag_low}..{lag_high}"

    points = [(0.0, 0.0)]
    confirmed_so_far = 0
    for item in sorted(result.confirmations, key=lambda candidate: candidate.arrival_minute):
        confirmed_so_far += len(item.account_ids)
        points.append((item.arrival_minute / MINUTES_PER_DAY, float(confirmed_so_far)))
    chart = {
        "type": "timeline", "title": "Accounts confirmed by bulk batches (labelled fixture, not a real range)",
        "x_label": "day of the world", "y_label": "accounts confirmed so far", "points": points,
        "shaded": [(batch.cutoff_minute / MINUTES_PER_DAY, batch.arrival_minute / MINUTES_PER_DAY, "lag")],
    }
    days_after_cutoff = lag_minutes / MINUTES_PER_DAY
    days_after_fraud = (batch.arrival_minute - max(last_fraud_minutes)) / MINUTES_PER_DAY
    key_number = (
        f"batch {batch.batch_id.split('-')[-1]} confirmed {len(batch.account_ids)} "
        f"account{'' if len(batch.account_ids) == 1 else 's'} in region {batch.region}, "
        f"{days_after_cutoff:.1f} days after the cut-off and {days_after_fraud:.1f} days after the last fraud "
        "payment it covers"
    )
    evidence = {
        "fixture": "bulk_short_lag.yaml is a labelled fixture, not a real range (issue 12-12)",
        "batch_id": batch.batch_id, "region": batch.region, "cutoff_minute": batch.cutoff_minute,
        "arrival_minute": batch.arrival_minute, "requested_size": batch.requested_size,
        "account_ids": list(batch.account_ids), "last_fraud_minutes": last_fraud_minutes,
        "lag_bounds_minutes": [lag_low, lag_high], "batches_total": len(result.confirmations),
        "batches_with_accounts": len(batches_with_accounts),
        "batches_skipped_no_fraud_before_cutoff": skipped_without_prior_fraud,
        "mule_side_accounts_without_region": len(result.mule_side_accounts_without_region),
    }
    return CheckOutcome(key_number=key_number, evidence=evidence, chart=chart)


async def run_complaint_share_large_batch_check(workspace: Any) -> CheckOutcome:
    """
    WHAT: Draws the world's complaint settings from the training ranges with draw_label_world_settings (seeded
    generators from the world helper), passes 5,000 synthetic victim losses of the three scam types to the real
    plan_true_complaints, and asserts the sample is at least the configured minimum and that the observed share of
    complaining victims is within the configured number of standard errors of the drawn share. The inputs are
    synthetic: made-up losses, not a simulated world.
    WHY: Proves Check 25: the complaint share follows its drawn value. A 17-victim world is too small to show it
    (check 21), so the comparison is made on a batch large enough for the tolerance to mean something.
    Args:
        workspace: Optional execution workspace supplied by the showcase framework.
    Returns:
        CheckOutcome with the observed share, the drawn share and the tolerance.
    Example:
        outcome = await run_complaint_share_large_batch_check(None)
        assert "vs drawn share" in outcome.key_number
    """
    world_ranges = read_world_ranges()
    labels_config = load_labels_config()
    streams = build_component_streams(250, {"check_25_settings": 0, "check_25_complaints": 1})
    drawn = draw_label_world_settings(world_ranges, SPLIT, streams["check_25_settings"], labels_config)

    scam_types = [SCAM_INVESTMENT, SCAM_JOB, resolve_remainder_scam_type()]
    losses = [
        VictimLoss(f"acc_v{index}", 0, "acc_m", 1, scam_types[index % len(scam_types)], (f"p{index}",), 10_000)
        for index in range(COMPLAINT_SHARE_SAMPLE)
    ]
    events = plan_true_complaints(losses, drawn, world_ranges, SPLIT, streams["check_25_complaints"], labels_config)

    min_sample = labels_config["rate_check_min_sample"]
    assert COMPLAINT_SHARE_SAMPLE >= min_sample, f"Sample {COMPLAINT_SHARE_SAMPLE} is under the minimum {min_sample}"
    observed = len(events) / COMPLAINT_SHARE_SAMPLE
    tolerance = _share_tolerance(
        drawn.victim_complaint_share, COMPLAINT_SHARE_SAMPLE, labels_config["rate_check_standard_errors"]
    )
    assert abs(observed - drawn.victim_complaint_share) <= tolerance, (
        f"Observed share {observed:.4f} is off the drawn share {drawn.victim_complaint_share:.4f} "
        f"by more than {tolerance:.4f}"
    )

    key_number = (
        f"observed share {observed:.3f} vs drawn share {drawn.victim_complaint_share:.3f} on "
        f"{COMPLAINT_SHARE_SAMPLE:,} victims (tolerance {tolerance:.3f})"
    )
    evidence = {
        "inputs": f"synthetic: {COMPLAINT_SHARE_SAMPLE} made-up victim losses across the three scam types",
        "observed_share": observed, "drawn_share": drawn.victim_complaint_share, "tolerance": tolerance,
        "sample_size": COMPLAINT_SHARE_SAMPLE, "min_sample": min_sample, "complaints": len(events),
    }
    return CheckOutcome(key_number=key_number, evidence=evidence)
