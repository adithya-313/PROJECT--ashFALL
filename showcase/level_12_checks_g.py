"""
WHAT: Implementation of Level 12 Showcase Part G checks (Checks 31 through 37).
Evaluates test world gang taxonomy tagging under truth-v2, slow-drip piece and dwell timing dispersion,
device rotation periodic generation timetables, deferred fake_routine_life status, rented mule routine credits,
training simulation held-out option refusal, validator mechanism enforcement, and exhaustive taxonomic property checks.

WHY: Verifies Level 12g (amendments A-025 and Level 12g brief): held-out tactic family execution and isolation.
Confirms that unseen tactics execute exclusively in test and stress partitions, while training partitions remain pure.

TERM: held-out family: an evasion tactic reserved exclusively for test and stress evaluation worlds.
TERM: slow_drip: moving money in small sub-₹2,000 pieces spread over days with multi-hour holding intervals.
TERM: device_rotation: periodic swapping of controlling mobile devices for each mule account across generations.
TERM: truth-v2: ground-truth directory format recording manifest format 'truth-v2' and gang taxonomic classifications.
"""

from typing import Any

from showcase.framework import CheckOutcome
from showcase.level_12_support_g import (
    inspect_device_rotation_world,
    inspect_rented_mule_routine,
    inspect_slow_drip_world,
    inspect_test_world_truth,
    verify_held_out_mechanism_validator,
    verify_held_out_property_distribution,
    verify_training_held_out_refusal,
)


async def run_tagged_gangs_check(workspace: Any) -> CheckOutcome:
    """
    WHAT: Runs the first test world, writes truth-v2 artifacts, and verifies that gangs are classified as
    seen or held-out. Asserts at least one seen gang, exactly one held-out gang, and manifest format 'truth-v2'.

    WHY: Proves Check 31: evaluation benchmarks receive cryptographically validated gang taxonomies under truth-v2.

    Args:
        workspace: Optional execution workspace supplied by the showcase framework.

    Returns:
        CheckOutcome reporting seen and held-out gang counts and the active held-out tactic family.

    Example:
        outcome = await run_tagged_gangs_check(None)
        assert "1 held-out" in outcome.key_number
    """
    info = inspect_test_world_truth(workspace)
    assert info["seen_count"] >= 1, f"expected >=1 seen gang, got {info['seen_count']}"
    assert info["held_out_count"] == 1, f"expected 1 held-out gang, got {info['held_out_count']}"
    assert info["manifest_format"] == "truth-v2", f"manifest format '{info['manifest_format']}' != truth-v2"

    key_number = f"{info['total_gangs']} gangs: {info['seen_count']} seen, 1 held-out ({info['held_out_family']})"
    evidence = {
        "world_id": info["world_id"], "seed": info["seed"], "run_id": info["run_id"],
        "manifest_format": info["manifest_format"], "seen_count": info["seen_count"],
        "held_out_count": info["held_out_count"], "held_out_family": info["held_out_family"],
    }
    return CheckOutcome(key_number=key_number, evidence=evidence)


async def run_slow_drip_check(workspace: Any) -> CheckOutcome:
    """
    WHAT: Inspects slow-drip route payments in the first slow_drip test world, checking piece limits and holding waits.
    Constructs a timeline chart plotting cumulative pieces leaving layer 1 against elapsed simulation days.

    WHY: Proves Check 32: slow_drip moves money in sub-₹2,000 pieces spread over days with holding dwell intervals.

    Args:
        workspace: Optional execution workspace supplied by the showcase framework.

    Returns:
        CheckOutcome with piece count, largest INR, holding minutes, dispersion days, and timeline chart.

    Example:
        outcome = await run_slow_drip_check(None)
        assert "pieces" in outcome.key_number
    """
    res = inspect_slow_drip_world()
    assert res["largest_inr"] < 2000.0, f"largest piece Rs {res['largest_inr']} >= 2,000"
    assert res["drip_hold_minutes"] > 60, f"hold minutes {res['drip_hold_minutes']} <= 60"
    if res["pieces"] > 1:
        assert res["first_hop_span_days"] >= 3.0, f"span {res['first_hop_span_days']:.1f} days < 3 days"

    chart = {
        "type": "timeline", "title": "Slow drip: cumulative pieces leaving layer 1",
        "x_label": "Days since first route payment", "y_label": "Cumulative pieces",
        "points": res["points"],
    }
    key_number = (
        f"{res['pieces']} pieces, largest Rs {res['largest_inr']:,.0f}, "
        f"every hold >= {res['drip_hold_minutes']} min, spread over {res['first_hop_span_days']:.1f} days"
    )
    evidence = {
        "world_id": res["world_id"], "gang_id": res["gang_id"], "pieces": res["pieces"],
        "largest_inr": res["largest_inr"], "drip_hold_minutes": res["drip_hold_minutes"],
        "first_hop_span_days": round(res["first_hop_span_days"], 2),
    }
    return CheckOutcome(key_number=key_number, evidence=evidence, chart=chart)


async def run_device_rotation_check(workspace: Any) -> CheckOutcome:
    """
    WHAT: Evaluates device rotation generation timetables and verifies strict one-account device exclusivity.
    Constructs a timeline chart plotting distinct devices used over time with shaded rotation period bands.

    WHY: Proves Check 33: device_rotation swaps mobile handsets periodically on deterministic generation boundaries.

    Args:
        workspace: Optional execution workspace supplied by the showcase framework.

    Returns:
        CheckOutcome with mule count, generations, total devices, exclusivity proof, and timeline chart.

    Example:
        outcome = await run_device_rotation_check(None)
        assert "generations" in outcome.key_number
    """
    res = inspect_device_rotation_world()
    expected_devices = res["mules"] * res["generations"]
    assert res["devices"] == expected_devices, f"devices {res['devices']} != {expected_devices}"
    assert res["span_changes_exact"], "rotation span change was not on generation period boundary"
    # Verify strict device exclusivity: at no minute does any device control more than one account.
    one_account_per_device = res["one_account_per_device"]
    assert one_account_per_device, "device controlled multiple accounts simultaneously"

    chart = {
        "type": "timeline", "title": "Device rotation timetable across generations",
        "x_label": "Days since take_control", "y_label": "Distinct devices used",
        "points": res["points"], "shaded": res["shaded"],
    }
    key_number = (
        f"{res['mules']} mules x {res['generations']} generations = {res['devices']} devices; "
        "most accounts on one device at once: 1"
    )
    evidence = {
        "world_id": res["world_id"], "gang_id": res["gang_id"], "mules": res["mules"],
        "generations": res["generations"], "devices": res["devices"], "period_days": res["period_days"],
    }
    return CheckOutcome(key_number=key_number, evidence=evidence, chart=chart)


async def run_fake_routine_life_deferred_check(workspace: Any) -> CheckOutcome:
    """
    WHAT: Verifies fake_routine_life deferred status under A-025 and audits rented mule routine credits in
    test-b00-0000. Asserts that incoming non-gang routine credits originate exclusively from genuine non-gang citizens.

    WHY: Proves Check 34: deferred evasion mechanics leave legitimate economic transactions intact without gang leakage.

    Args:
        workspace: Optional execution workspace supplied by the showcase framework.

    Returns:
        CheckOutcome reporting deferred status string, audited mule account identifier, and routine credit counts.

    Example:
        outcome = await run_fake_routine_life_deferred_check(None)
        assert "DEFERRED (A-025)" in outcome.key_number
    """
    res = inspect_rented_mule_routine()
    assert res["deferred_status"] == "DEFERRED (A-025)", f"status '{res['deferred_status']}' != DEFERRED (A-025)"
    assert res["gang_senders"] == 0, f"rented mule received {res['gang_senders']} credits from gang accounts"

    key_number = (
        f"fake_routine_life {res['deferred_status']}; "
        f"rented mule: {res['routine_credits']} routine credits, 0 from gang accounts"
    )
    evidence = {
        "world_id": res["world_id"], "mule_account": res["mule_account"],
        "routine_credits": res["routine_credits"], "gang_senders": res["gang_senders"],
        "deferred_status": res["deferred_status"],
    }
    return CheckOutcome(key_number=key_number, evidence=evidence)


async def run_training_held_out_refusal_check(workspace: Any) -> CheckOutcome:
    """
    WHAT: Attempts to execute an interactive training world with a gang carrying held-out drip options and captures
    refusal. Verifies that the runner refuses execution with ValueError naming the option, slow_drip, and training.

    WHY: Proves Check 35: training simulations strictly forbid held-out evasion mechanics to prevent data leakage.

    Args:
        workspace: Optional execution workspace supplied by the showcase framework.

    Returns:
        CheckOutcome reporting the first 100 characters of the refusal message.

    Example:
        outcome = await run_training_held_out_refusal_check(None)
        assert "slow_drip" in outcome.key_number
    """
    refusal = verify_training_held_out_refusal()
    assert "drip_piece_max_inr" in refusal, "refusal missing option name 'drip_piece_max_inr'"
    assert "slow_drip" in refusal, "refusal missing family name 'slow_drip'"
    assert "training" in refusal, "refusal missing split 'training'"

    key_number = refusal[:100]
    evidence = {"error_type": "ValueError", "message": refusal, "length": len(refusal)}
    return CheckOutcome(key_number=key_number, evidence=evidence)


async def run_held_out_validator_check(workspace: Any) -> CheckOutcome:
    """
    WHAT: Mutates evaluation configuration by activating fake_routine_life without block options and checks refusal.
    Verifies that check_held_out_mechanisms raises ValueError citing missing implementing block options.

    WHY: Proves Check 36: evaluation policies reject active held-out family declarations lacking tactical mechanics.

    Args:
        workspace: Optional execution workspace supplied by the showcase framework.

    Returns:
        CheckOutcome with the first 100 characters of the validator refusal message.

    Example:
        outcome = await run_held_out_validator_check(None)
        assert "has no implementing block option" in outcome.key_number
    """
    refusal = verify_held_out_mechanism_validator()
    assert "has no implementing block option" in refusal, "refusal missing 'has no implementing block option'"

    key_number = refusal[:100]
    evidence = {"error_type": "ValueError", "message": refusal, "length": len(refusal)}
    return CheckOutcome(key_number=key_number, evidence=evidence)


async def run_held_out_property_distribution_check(workspace: Any) -> CheckOutcome:
    """
    WHAT: Evaluates taxonomic gang distribution invariants across all 680 planned worlds in the Latin hypercube sampler.
    Duplicated from tests/unit/world/test_held_out_property.py because that module does not expose a reusable helper.

    WHY: Proves Check 37: 440 training/validation worlds have 0 held-out gangs; 240 test/stress worlds each have 1.

    Args:
        workspace: Optional execution workspace supplied by the showcase framework.

    Returns:
        CheckOutcome with exact summary sentence of held-out distribution across world partitions.

    Example:
        outcome = await run_held_out_property_distribution_check(None)
        assert "N=440" in outcome.key_number
    """
    counts = verify_held_out_property_distribution()
    assert counts["training_validation"] == 440, f"training/validation {counts['training_validation']} != 440"
    assert counts["tv_held_out"] == 0, f"training/validation held-out {counts['tv_held_out']} != 0"
    assert counts["test_stress"] == 240, f"test/stress {counts['test_stress']} != 240"
    assert counts["ts_held_out"] == 240, f"test/stress held-out {counts['ts_held_out']} != 240"
    assert (
        counts["families"]["device_rotation"] == 129
    ), f"device_rotation {counts['families']['device_rotation']} != 129"
    assert counts["families"]["slow_drip"] == 111, f"slow_drip {counts['families']['slow_drip']} != 111"

    key_number = (
        "N=440 training/validation: 0 held-out gangs; "
        "N=240 test/stress: 240 with one (device_rotation 129, slow_drip 111)"
    )
    evidence = {
        "training_validation": counts["training_validation"], "tv_held_out": counts["tv_held_out"],
        "test_stress": counts["test_stress"], "ts_held_out": counts["ts_held_out"],
        "families": counts["families"],
    }
    return CheckOutcome(key_number=key_number, evidence=evidence)
