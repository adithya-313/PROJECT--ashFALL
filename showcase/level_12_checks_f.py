"""
WHAT: Implementation of Level 12 Showcase Part F checks (Checks 26 through 30). Shows an example pattern file that
validates and previews while an over-budget one is refused, an example custom building block running inside a gang, a
coverage plot of sampled worlds across two ranges, validation worlds listed apart from training worlds, and a test-split
world running on the training ranges.

WHY: Verifies Level 12f (amendment A-023): the many-worlds sampler, pattern files, local plugins and the world invariant
checks, each measured from real code on the real example files and sampled blocks.

TERM: pattern file: a YAML description of one gang in the tactic language, checked and previewed before use.
TERM: plugin: a local Python file that adds one tactic block that may only spend time and money.
TERM: stratum: one of n equal slices of a range; a Latin hypercube block puts one world in each.
TERM: split: which group a world belongs to (training, validation, test, stress); it decides the tactics allowed.
"""

import tempfile
from pathlib import Path
from typing import Any

from showcase.framework import CheckOutcome
from showcase.level_12_support_f import (
    BASIC_PATTERN,
    COOL_OFF_PLUGIN,
    OVER_BUDGET_PATTERN,
    coverage_of_block,
    example_pattern_names,
    plan_split_blocks,
    run_plugin_world,
    run_test_split_world,
    seed_range,
    validate_example_patterns,
)
from world.world_sampler import check_seeds_disjoint


async def run_pattern_file_check(workspace: Any) -> CheckOutcome:
    """
    WHAT: Validates the basic example pattern file for real (loads it and previews it) and validates the over-budget
    example, which must be refused with a PatternFileError that starts with the file name and "over budget:".
    WHY: Proves Check 26: a good pattern passes with a preview and a pattern that cannot stay inside its budget is
    refused with a plain reason. The check fails if either half fails.
    Args:
        workspace: Optional execution workspace supplied by the showcase framework.
    Returns:
        CheckOutcome with the preview sentence as the key number.
    Example:
        outcome = await run_pattern_file_check(None)
        assert "ESTIMATE" in outcome.key_number
    """
    preview, refusal = validate_example_patterns()
    assert preview.mule_accounts >= 1, "the basic example previewed with no mule accounts"
    evidence = {
        "valid_file": str(BASIC_PATTERN.name), "refused_file": str(OVER_BUDGET_PATTERN.name),
        "refusal": refusal, "names": example_pattern_names(), "victims": preview.victims,
        "payments_planned": preview.payments_planned, "span_days": preview.span_days,
    }
    return CheckOutcome(key_number=preview.summary_text(), evidence=evidence)


async def run_plugin_block_check(workspace: Any) -> CheckOutcome:
    """
    WHAT: Registers the cool_off plugin, plans a gang with acquire, cool_off, scam and route, runs one small world with
    it, and compares the run's id with the id of the same world planned without the plugin step.
    WHY: Proves Check 27: a local custom block runs inside a gang, its file hash is recorded in the run's effective
    configuration, and changing the plan (here, dropping the step) changes the run id.
    Args:
        workspace: Optional execution workspace supplied by the showcase framework.
    Returns:
        CheckOutcome with the days waited and the recorded plugin hash.
    Example:
        outcome = await run_plugin_block_check(None)
        assert "plugin_sha256" in outcome.key_number
    """
    result = run_plugin_world()
    assert 1 <= result["days_waited"] <= 7, f"cool_off waited {result['days_waited']} days, outside its range [1, 7]"
    assert result["recorded_sha256"] == result["file_sha256"], "the run did not record the plugin file's hash"
    assert result["run_id"] == result["expected_run_id"], "the run id differs from the id built from its configuration"
    assert result["run_id"] != result["run_id_without_plugin"], "the plugin step did not change the run id"
    key_number = f"waited {result['days_waited']} days; plugin_sha256 {result['recorded_sha256'][:16]}..."
    evidence = {
        "plugin_file": str(COOL_OFF_PLUGIN.name), "days_waited": result["days_waited"],
        "plugin_sha256": result["recorded_sha256"], "run_id": result["run_id"],
        "run_id_without_plugin_step": result["run_id_without_plugin"],
    }
    return CheckOutcome(key_number=key_number, evidence=evidence)


async def run_coverage_plot_check(workspace: Any) -> CheckOutcome:
    """
    WHAT: Plans training block 0 and plots victim_complaint_share against share_reported_to_shared_database, both
    continuous world-scope settings. It counts the strata (equal slices of each range) the block's worlds fall in on
    each axis; both must equal the block size. The evidence also lists, for every integer setting, its distinct
    values and the whole numbers its range holds. Plan only; no world is run.
    WHY: Proves Check 28: a block spreads its worlds across the ranges instead of clumping them. Integer settings are
    stratified before rounding, so their note is evidence only and does not fail the check.
    Args:
        workspace: Optional execution workspace supplied by the showcase framework.
    Returns:
        CheckOutcome with the strata counts, a scatter chart and the integer-setting evidence.
    Example:
        outcome = await run_coverage_plot_check(None)
        assert outcome.key_number == "40 of 40 strata hit on x, 40 of 40 on y"
    """
    coverage = coverage_of_block()
    size = coverage["block_size"]
    assert coverage["x_strata_hit"] == size, f"x hit {coverage['x_strata_hit']} of {size} strata"
    assert coverage["y_strata_hit"] == size, f"y hit {coverage['y_strata_hit']} of {size} strata"
    chart = {
        "type": "scatter", "title": "Training block 0: where its worlds sit in two ranges",
        "x_label": "victim_complaint_share", "y_label": "share_reported_to_shared_database",
        "series": [{"name": "training block 0", "points": coverage["points"]}],
    }
    key_number = f"{coverage['x_strata_hit']} of {size} strata hit on x, {coverage['y_strata_hit']} of {size} on y"
    evidence = {
        "block_size": size, "integer_settings": coverage["integer_settings"], "integer_note": coverage["note"],
    }
    return CheckOutcome(key_number=key_number, evidence=evidence, chart=chart)


async def run_validation_listed_apart_check(workspace: Any) -> CheckOutcome:
    """
    WHAT: Plans training block 0 and validation block 0, writes both manifests into a temporary folder, reads them
    back, and runs check_seeds_disjoint on what was read (it also refuses the reserved seed 101).
    WHY: Proves Check 29: validation worlds are listed apart from training worlds, with seeds that cannot overlap, and a
    manifest on disk gives back the same worlds.
    Args:
        workspace: Optional execution workspace supplied by the showcase framework.
    Returns:
        CheckOutcome with the counts of seeds and the seed ranges and provenance hashes as evidence.
    Example:
        outcome = await run_validation_listed_apart_check(None)
        assert "0 shared" in outcome.key_number
    """
    with tempfile.TemporaryDirectory(prefix="ashfall_12f_") as folder:
        planned = plan_split_blocks("validation", Path(folder))
    training, validation = planned["training"], planned["validation"]
    check_seeds_disjoint({"training": training, "validation": validation}, planned["reserved"])
    shared = {spec.seed for spec in training} & {spec.seed for spec in validation}
    assert not shared, f"{len(shared)} seeds are shared"
    key_number = (
        f"{len(training)} training seeds, {len(validation)} validation seeds, {len(shared)} shared, "
        f"{planned['reserved'][0]} not used"
    )
    evidence = {
        "training_seeds": seed_range(training), "validation_seeds": seed_range(validation),
        "training_provenance": planned["training_provenance"],
        "validation_provenance": planned["validation_provenance"],
    }
    return CheckOutcome(key_number=key_number, evidence=evidence)


async def run_test_world_on_training_ranges_check(workspace: Any) -> CheckOutcome:
    """
    WHAT: Plans the first world of test block 0, runs it with run_sampled_world (generated gang mix) and checks it with
    check_world. It must have read the training range column, recorded the split "test", and have no invariant problem.
    WHY: Proves Check 30: a test world draws its numbers from the training ranges (report Section 6) yet stays a test
    world, so the 12d family guard still treats it as one.
    Args:
        workspace: Optional execution workspace supplied by the showcase framework.
    Returns:
        CheckOutcome with the range column, recorded split, gang count and invariant problem count.
    Example:
        outcome = await run_test_world_on_training_ranges_check(None)
        assert "0 invariant problems" in outcome.key_number
    """
    output, spec, gang_count, problems = run_test_split_world()
    recorded_split = output.effective_config["world_split"]
    column = output.effective_config["settings"]["range_name"]
    assert column == "training", f"the test world read the '{column}' range column"
    assert recorded_split == "test", f"the recorded world_split is '{recorded_split}'"
    assert not problems, f"{len(problems)} invariant problems: {problems[:3]}"
    key_number = (
        f"range column {column}, world_split {recorded_split}, {gang_count} gangs, {len(problems)} invariant problems"
    )
    evidence = {"world_id": spec.world_id, "seed": spec.seed, "run_id": output.result.run_id, "events": len(
        output.result.run_result.events
    )}
    return CheckOutcome(key_number=key_number, evidence=evidence)
