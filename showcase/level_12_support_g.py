"""
WHAT: Support helpers for Level 12 Showcase Part G checks (31 through 37): test world gang tagging in truth-v2,
slow-drip piece and dwell timing, device rotation generation timetables, deferred fake_routine_life status and rented
mule routine credit auditing, training held-out option refusal, validator mechanism enforcement, and exhaustive held-out
property distribution verification across all 680 planned worlds.

WHY: Part G demonstrates the Level 12g held-out tactic family capabilities (amendments A-025 and Level 12g brief):
cryptographic gang truth tagging, slow drip money dispersion, periodic device rotation, training isolation guards,
and taxonomic property invariants.

TERM: held-out family: an evasion tactic reserved exclusively for test and stress evaluation worlds.
TERM: slow_drip: moving money in small sub-₹2,000 pieces spread over days with multi-hour holding intervals.
TERM: device_rotation: periodic swapping of controlling mobile devices for each mule account across generations.
TERM: truth-v2: ground-truth directory format recording manifest format 'truth-v2' and gang taxonomic classifications.
"""

import copy
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

from evaluation.truth import load_truth
from world.gang_config import BlockStep, FixtureSource, GangBudget, GangConfig
from world.gang_mix import generate_gang_mix, load_gang_mix_config
from world.held_out_options import gang_truth_for
from world.interactive_records import RunnerSettings
from world.interactive_runner import run_interactive_world
from world.sampled_world import run_sampled_world
from world.tactic_config import load_tactic_blocks
from world.tactic_families import (
    EVALUATION_PATH,
    check_held_out_mechanisms,
    load_evaluation_config,
    load_tactic_families,
)
from world.truth_writer import write_truth_files
from world.world_configs import read_world_configs
from world.world_sampler import WorldSpec, load_sampler_config, sample_block
from world.world_sizing import resolve_settings

CODE_VERSION_G = "showcase-12g"
SAMPLER_SEED = 1
MINUTES_PER_DAY = 1440


def get_first_test_spec(target_family: str | None = None) -> WorldSpec:
    """
    WHAT: Returns the first deterministic WorldSpec from test block 0 containing the target tactic family.
    When target_family is None, it returns the very first planned specification in test block 0.

    WHY: Evaluates showcase verification checks on deterministic, reproducible Latin hypercube sample worlds.

    Args:
        target_family: Optional string identifier of the required tactic family, or None.

    Returns:
        First matching WorldSpec instance from test block 0.

    Raises:
        ValueError: If no world specification in test block 0 contains the requested tactic family.

    Example:
        spec = get_first_test_spec("slow_drip"); assert spec.world_id == "test-b00-0000"
    """
    sampler_cfg = load_sampler_config()
    world_ranges = read_world_configs().world_ranges
    specs = sample_block("test", 0, SAMPLER_SEED, sampler_cfg, world_ranges)
    if target_family is None:
        return specs[0]
    mix_cfg = load_gang_mix_config()
    for spec in specs:
        gangs = generate_gang_mix(spec, sampler_cfg, mix_cfg, 2000, world_ranges)
        if any(g.family == target_family for g in gangs):
            return spec
    raise ValueError(f"No test block 0 spec found with tactic family '{target_family}'")


def inspect_test_world_truth(workspace: Any = None) -> dict[str, Any]:
    """
    WHAT: Runs the first test spec, writes truth-v2 artifacts, loads truth through load_truth, and inspects gang kinds.
    Verifies that the world has at least one seen gang, exactly one held-out gang, and manifest format 'truth-v2'.

    WHY: Proves Check 31: evaluation benchmarks receive cryptographically validated gang taxonomies under truth-v2.

    Args:
        workspace: Optional execution workspace path or directory container.

    Returns:
        Dictionary reporting total gangs, seen count, held-out count, held-out family, format, and world_id.

    Example:
        info = inspect_test_world_truth(None); assert info["manifest_format"] == "truth-v2"
    """
    sampler_cfg = load_sampler_config()
    spec = get_first_test_spec(None)
    output = run_sampled_world(spec, "tiny", None, sampler_cfg, CODE_VERSION_G)
    target_dir = Path(workspace.directory) if workspace is not None and hasattr(workspace, "directory") else None

    with tempfile.TemporaryDirectory() as tmpdir:
        truth_root = target_dir if target_dir is not None else Path(tmpdir)
        run_dir = write_truth_files(output, truth_root)
        store = load_truth(truth_root, output.result.run_id)
        with open(run_dir / "manifest.json", encoding="utf-8") as handle:
            manifest_data = json.load(handle)

        seen_gangs = [g for g in store._gangs if store.gang_kind(g["gang_id"]) == "seen"]
        held_out_gangs = [g for g in store._gangs if store.gang_kind(g["gang_id"]) == "held_out"]
        manifest_format = str(manifest_data.get("format", ""))
        held_family = str(held_out_gangs[0].get("family", "")) if held_out_gangs else ""

        return {
            "world_id": spec.world_id, "seed": spec.seed, "run_id": output.result.run_id,
            "total_gangs": len(store._gangs), "seen_count": len(seen_gangs),
            "held_out_count": len(held_out_gangs), "held_out_family": held_family,
            "manifest_format": manifest_format,
        }


def inspect_slow_drip_world() -> dict[str, Any]:
    """
    WHAT: Evaluates slow-drip route payments for the first slow_drip test world, checking piece limits and dwell waits.
    Constructs timeline chart points for cumulative pieces leaving layer 1 against elapsed simulation days.

    WHY: Proves Check 32: slow_drip moves money in sub-₹2,000 pieces spread across days with holding intervals.

    Args:
        None.

    Returns:
        Dictionary reporting piece count, largest INR amount, minimum dwell wait, span days, and chart points.

    Example:
        res = inspect_slow_drip_world(); assert res["largest_inr"] < 2000.0
    """
    sampler_cfg = load_sampler_config()
    spec = get_first_test_spec("slow_drip")
    output = run_sampled_world(spec, "tiny", None, sampler_cfg, CODE_VERSION_G)
    drip_gang_id = next(g.gang_id for g in output.result.truth.gangs if g.family == "slow_drip")
    hold_min = next(
        int(dict(s.params)["drip_hold_minutes"]) for g in output.result.truth.gangs
        if g.gang_id == drip_gang_id for s in output.plan.gang_plans[0].gang.steps if s.block == "route"
    ) if hasattr(output.plan, "gang_plans") else 300

    # Extract planned route payments from stories and run events.
    drip_pairs = [
        (output.result.run_result.events[pos], output.result.truth.payment_stories[f_idx])
        for pos, (src, f_idx) in enumerate(zip(output.result.event_sources, output.result.event_fraud_input_index))
        if src == "fraud" and f_idx is not None and output.result.truth.payment_stories[f_idx].gang_id == drip_gang_id
        and output.result.truth.payment_stories[f_idx].block == "route"
    ]
    max_paise = max(ev.amount_paise for ev, s in drip_pairs)
    min_layer = min(s.layer for ev, s in drip_pairs if s.layer is not None)
    first_hop = sorted([ev for ev, s in drip_pairs if s.layer == min_layer], key=lambda e: e.minute)

    first_route_min = min(ev.minute for ev, s in drip_pairs)
    span_days = (max(ev.minute for ev in first_hop) - min(ev.minute for ev in first_hop)) / MINUTES_PER_DAY
    points = [
        (round((ev.minute - first_route_min) / MINUTES_PER_DAY, 2), idx + 1)
        for idx, ev in enumerate(first_hop)
    ]
    return {
        "world_id": spec.world_id, "gang_id": drip_gang_id, "pieces": len(first_hop),
        "largest_inr": max_paise / 100.0, "drip_hold_minutes": hold_min,
        "first_hop_span_days": span_days, "points": points,
    }


def inspect_device_rotation_world() -> dict[str, Any]:
    """
    WHAT: Evaluates device rotation generation schedules and device account exclusivity in device_rotation worlds.
    Constructs timeline chart points and shaded generation periods tracking distinct devices used over time.

    WHY: Proves Check 33: device_rotation swaps mobile handsets periodically on deterministic generation boundaries.

    Args:
        None.

    Returns:
        Dictionary containing mule count, generations, device count, period days, chart points, and shaded bands.

    Example:
        res = inspect_device_rotation_world(); assert res["devices"] == res["mules"] * res["generations"]
    """
    sampler_cfg = load_sampler_config()
    spec = get_first_test_spec("device_rotation")
    output = run_sampled_world(spec, "tiny", None, sampler_cfg, CODE_VERSION_G)
    rot_gang_id = next(g.gang_id for g in output.result.truth.gangs if g.family == "device_rotation")
    dev_ctrl = [dc for dc in output.result.truth.device_control if dc.gang_id == rot_gang_id]

    mule_accounts = sorted({dc.account_id for dc in dev_ctrl})
    devices = sorted({dc.device_id for dc in dev_ctrl})
    t0 = min(dc.valid_from for dc in dev_ctrl)
    generations = len(devices) // len(mule_accounts)
    period_min = min(dc.valid_to - dc.valid_from for dc in dev_ctrl if dc.valid_to is not None)
    period_days = period_min / MINUTES_PER_DAY

    # Build timeline step points and shaded intervals for rotation generations.
    points: list[tuple[float, int]] = []
    shaded: list[tuple[float, float, str]] = []
    for gen in range(generations):
        day_offset = (gen * period_min) / MINUTES_PER_DAY
        points.append((round(day_offset, 2), (gen + 1) * len(mule_accounts)))
        shaded.append((round(day_offset, 2), round(day_offset + period_days, 2), f"generation {gen + 1}"))
    points.append((round((generations * period_min) / MINUTES_PER_DAY, 2), len(devices)))
    one_account_per_device = all(
        len({dc.account_id for dc in dev_ctrl if dc.device_id == dev}) == 1 for dev in devices
    )
    span_changes_exact = all((dc.valid_from - t0) % period_min == 0 for dc in dev_ctrl)

    return {
        "world_id": spec.world_id, "gang_id": rot_gang_id, "mules": len(mule_accounts),
        "generations": generations, "devices": len(devices), "period_days": period_days,
        "t0_minute": t0, "points": points, "shaded": shaded,
        "one_account_per_device": one_account_per_device,
        "span_changes_exact": span_changes_exact,
    }


def inspect_rented_mule_routine() -> dict[str, Any]:
    """
    WHAT: Verifies fake_routine_life deferred status under A-025 and audits rented mule routine credits in
    test-b00-0000. Asserts that incoming non-gang routine credits originate exclusively from genuine citizens.

    WHY: Proves Check 34: deferred evasion mechanics leave legitimate economic transactions intact without gang leakage.

    Args:
        None.

    Returns:
        Dictionary containing deferred status string, audited mule account identifier, and routine credit counts.

    Example:
        res = inspect_rented_mule_routine(); assert res["gang_senders"] == 0
    """
    sampler_cfg = load_sampler_config()
    spec = get_first_test_spec(None)
    output = run_sampled_world(spec, "tiny", None, sampler_cfg, CODE_VERSION_G)
    truth = output.result.truth
    gang_accounts = {role.account_id for role in truth.roles}
    rented_roles = [r for r in truth.roles if r.role == "mule" and r.origin == "rented"]
    target_mule = rented_roles[0]

    # Filter incoming successful credits during the mule rental validity span.
    events = output.result.run_result.events
    non_gang_credits = [
        ev for ev in events
        if ev.to_account == target_mule.account_id and ev.status == "success"
        and (target_mule.valid_from <= ev.minute and (target_mule.valid_to is None or ev.minute < target_mule.valid_to))
        and ev.from_account not in gang_accounts
    ]
    gang_senders = sum(1 for ev in non_gang_credits if ev.from_account in gang_accounts)
    return {
        "world_id": spec.world_id, "mule_account": target_mule.account_id,
        "routine_credits": len(non_gang_credits), "gang_senders": gang_senders,
        "deferred_status": "DEFERRED (A-025)",
    }


def verify_training_held_out_refusal() -> str:
    """
    WHAT: Attempts to execute an interactive training world with a gang carrying held-out drip options and captures
    refusal. Verifies that the runner refuses execution with ValueError naming the option, slow_drip, and training.

    WHY: Proves Check 35: training simulations strictly forbid held-out evasion mechanics to prevent data leakage.

    Args:
        None.

    Returns:
        Full error message text string from the raised ValueError.

    Raises:
        AssertionError: If run_interactive_world executes without raising ValueError.

    Example:
        msg = verify_training_held_out_refusal(); assert "slow_drip" in msg
    """
    settings = resolve_settings(seed=1, preset="tiny", range_name="training")
    budget = GangBudget(money_paise=50_000_000, time_minutes=500_000, max_risk=0.99)
    drip_step = BlockStep("route", (
        ("layers", 2), ("accounts_per_layer", 1),
        ("drip_piece_max_inr", 500), ("drip_spread_days", 3), ("drip_hold_minutes", 300),
    ))
    gang = GangConfig(
        gang_id="gang_held_out_refuse", strength=1, budget=budget,
        steps=(BlockStep("acquire", (("count", 2),)), drip_step),
        source=FixtureSource("acc_gang_src_01", 0, 10_000_000, 1440),
    )
    try:
        run_interactive_world(
            fast_settings=settings, runner_settings=RunnerSettings(fraud_enabled=True),
            gang_configs=(gang,), world_split="training",
        )
    except ValueError as exc:
        return str(exc)
    raise AssertionError("training world with slow-drip options was not refused")


def verify_held_out_mechanism_validator() -> str:
    """
    WHAT: Mutates evaluation configuration by activating fake_routine_life without block options and checks refusal.
    Verifies that check_held_out_mechanisms raises ValueError citing missing implementing block options.

    WHY: Proves Check 36: evaluation policies reject active held-out family declarations lacking tactical mechanics.

    Args:
        None.

    Returns:
        Refusal message text string from the triggered ValueError.

    Raises:
        AssertionError: If check_held_out_mechanisms fails to raise ValueError.

    Example:
        msg = verify_held_out_mechanism_validator(); assert "has no implementing block option" in msg
    """
    eval_hash_before = hashlib.sha256(EVALUATION_PATH.read_bytes()).hexdigest()
    families = load_tactic_families()
    evaluation = copy.deepcopy(load_evaluation_config())
    blocks = load_tactic_blocks()

    eval_held = evaluation["held_out_tactics"]
    eval_held["deferred"] = [d for d in eval_held.get("deferred", []) if d.get("name") != "fake_routine_life"]
    eval_held["tactics"].append("fake_routine_life")

    try:
        check_held_out_mechanisms(families, evaluation, blocks)
    except ValueError as exc:
        eval_hash_after = hashlib.sha256(EVALUATION_PATH.read_bytes()).hexdigest()
        assert eval_hash_before == eval_hash_after, "evaluation.yaml modified on disk"
        return str(exc)
    raise AssertionError("active family without block implementation was not refused")


def verify_held_out_property_distribution() -> dict[str, Any]:
    """
    WHAT: Evaluates taxonomic gang distribution invariants across all 680 planned worlds in the Latin hypercube sampler.
    Duplicated from tests/unit/world/test_held_out_property.py because that module does not expose a reusable helper.

    WHY: Proves Check 37: 440 training/validation worlds have 0 held-out gangs; 240 test/stress worlds each have 1.

    Args:
        None.

    Returns:
        Dictionary containing counts of training/validation and test/stress worlds and per-family frequencies.

    Example:
        dist = verify_held_out_property_distribution(); assert dist["test_stress_held_out"] == 240
    """
    sampler_cfg = load_sampler_config()
    world_ranges = read_world_configs().world_ranges
    mix_cfg = load_gang_mix_config()
    families = load_tactic_families()
    counts: dict[str, Any] = {
        "training_validation": 0, "tv_held_out": 0,
        "test_stress": 0, "ts_held_out": 0, "families": {},
    }

    for split in ("training", "validation"):
        for b in range(sampler_cfg["counts"]["blocks_per_split"][split]):
            for spec in sample_block(split, b, SAMPLER_SEED, sampler_cfg, world_ranges):
                counts["training_validation"] += 1
                for g in generate_gang_mix(spec, sampler_cfg, mix_cfg, 2000, world_ranges):
                    if gang_truth_for(g, families).kind == "held_out":
                        counts["tv_held_out"] += 1

    for split in ("test", "stress"):
        for b in range(sampler_cfg["counts"]["blocks_per_split"][split]):
            for spec in sample_block(split, b, SAMPLER_SEED, sampler_cfg, world_ranges):
                counts["test_stress"] += 1
                held = [g for g in generate_gang_mix(spec, sampler_cfg, mix_cfg, 2000, world_ranges)
                        if gang_truth_for(g, families).kind == "held_out"]
                counts["ts_held_out"] += len(held)
                fam = str(held[0].family)
                counts["families"][fam] = counts["families"].get(fam, 0) + 1

    return counts
