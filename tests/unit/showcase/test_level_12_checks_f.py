"""
WHAT: Unit tests for the Level 12f showcase checks (26 through 30): part f is exactly five checks in the brief's order
and kinds, the over-budget half of check 26 raises, check 28 reports 40 and 40 strata with a scatter chart of 40 points,
and check 29 shows no shared seed. The world-running checks 27 and 30 are not run here; the showcase run covers them.

WHY: Guarantees the part f registration and its cheap checks hold in isolation, so a change to the sampler, the pattern
files or the manifests shows up here before a full showcase run.

TERM: part: One letter naming a slice of a level that is built and reviewed on its own, e.g. "f" for Level 12f.
TERM: scatter chart: a chart of points with an x and a y value, drawn by showcase/reporting.py as inline SVG.
"""

import asyncio

import pytest

from showcase import level_12
from showcase.framework import validate_checks
from showcase.level_12_checks_f import run_coverage_plot_check, run_validation_listed_apart_check
from showcase.level_12_support_f import OVER_BUDGET_PATTERN
from world.pattern_files import PatternFileError, validate_pattern_file
from world.world_sampler import load_sampler_config


def test_part_f_checks_match_the_brief() -> None:
    """Part f is five checks named 26 to 30 in order, of kinds edge, edge, normal, consistency, edge."""
    all_checks = level_12.get_checks(quick=True)
    validate_checks(all_checks)
    assert len(all_checks) == 37
    part_f = [check for check in all_checks if check.part == "f"]
    assert [check.name.split(".")[0] for check in part_f] == ["26", "27", "28", "29", "30"]
    assert [check.kind for check in part_f] == ["edge", "edge", "normal", "consistency", "edge"]
    assert asyncio.run(level_12.find_missing_preconditions()) == []


def test_check_26_over_budget_half_raises() -> None:
    """The over-budget example is refused with a PatternFileError starting with the file name and 'over budget:'."""
    with pytest.raises(PatternFileError, match=f"{OVER_BUDGET_PATTERN.name}: over budget:"):
        validate_pattern_file(OVER_BUDGET_PATTERN)


def test_check_28_hits_every_stratum_on_both_continuous_axes_and_draws_one_point_per_world() -> None:
    """The coverage check says 40 of 40 and 40 of 40, its scatter has 40 points, and the integer note is evidence."""
    block_size = load_sampler_config()["block_size"]["value"]
    outcome = asyncio.run(run_coverage_plot_check(None))
    assert outcome.key_number == f"{block_size} of {block_size} strata hit on x, {block_size} of {block_size} on y"
    assert outcome.chart is not None
    assert outcome.chart["type"] == "scatter"
    assert len(outcome.chart["series"]) == 1
    assert len(outcome.chart["series"][0]["points"]) == block_size
    integers = outcome.evidence["integer_settings"]
    assert set(integers) == {
        "number_of_regions", "gang_count", "route_layers", "dwell_minutes", "devices_per_gang",
        "victims_per_receiving_mule",
    }
    for counts in integers.values():
        assert 1 <= counts["distinct_values"] <= block_size
        assert counts["integers_in_range"] >= 1
    assert "rounding to whole numbers can merge neighbouring strata" in outcome.evidence["integer_note"]


def test_check_29_shows_no_shared_seed_and_not_seed_101() -> None:
    """The check passes with 40 training seeds, 40 validation seeds, 0 shared, and the reserved seed 101 unused."""
    outcome = asyncio.run(run_validation_listed_apart_check(None))
    assert outcome.key_number == "40 training seeds, 40 validation seeds, 0 shared, 101 not used"
    assert outcome.evidence["training_seeds"] == "1000000 to 1000039"
    assert outcome.evidence["validation_seeds"] == "2000000 to 2000039"
