"""
WHAT: Implementation of Level 12 Showcase Part H checks (Checks 38 through 41).
Evaluates the edge-case scenario library across runnable, later-level, and deferred boundary cases;
executes the trivial-world single-signal separability check across live gated worlds, live report-only
worlds, and recorded 680-world summary metrics; inspects fraud-on world cards for base rates and lower-bound
inflation factors; and validates outside dataset sanity reporting against PaySim and IBM AML-Data.

WHY: Verifies Level 12h (C12): edge-case boundary stability, non-trivial feature detection thresholds,
empirical base rate scaling, and external benchmark plausibility comparisons.

TERM: edge case: A specific boundary scenario with defined configuration overrides, gang tactics, and expectations.
TERM: trivial world: A synthetic simulation where a single feature separates illicit accounts at AUPRC >= 0.90.
TERM: fraud inflation factor: Ratio of simulation complaint density over empirical real-world complaint rate.
TERM: sanity report: Non-tuning plausibility comparison between simulated distributions and external reference datasets.
"""

from typing import Any

from showcase.framework import CheckOutcome
from showcase.level_12_support_h import (
    inspect_edge_case_library,
    inspect_fraud_world_card_base_rate,
    inspect_sanity_report_check,
    inspect_trivial_world_live_and_recorded,
)


async def run_edge_cases_check(workspace: Any) -> CheckOutcome:
    """
    WHAT: Evaluates every case in scenarios/edge_cases/ through world.edge_cases, asserting expected behavior.
    Builds an evidence table listing number, name, status, expected behavior, and result.

    WHY: Proves Check 38: verifies that all runnable edge cases pass and non-runnable cases record their level
    or amendment.

    Args:
        workspace: Optional execution workspace supplied by the showcase framework.

    Returns:
        CheckOutcome reporting total cases, runnable passes, later levels, deferred cases, and full evidence table.

    Example:
        outcome = await run_edge_cases_check(None)
        assert "runnable PASS" in outcome.key_number
    """
    info = inspect_edge_case_library()
    assert info["runnable_pass"] > 0, "must have runnable passing cases"
    assert info["total_cases"] == info["runnable_pass"] + info["later_level_count"] + info["deferred_count"]
    for row in info["table"]:
        if row["status"] == "runnable":
            assert "PASS" in row["result"], f"runnable case {row['name']} must pass"
        elif row["status"] == "later_level":
            assert "Level" in row["result"], f"later_level case {row['name']} must show level"
        elif row["status"] == "deferred":
            assert "deferred" in row["result"], f"deferred case {row['name']} must show amendment"

    return CheckOutcome(key_number=info["key_number"], evidence={"cases": info["table"]})


async def run_trivial_world_check(workspace: Any) -> CheckOutcome:
    """
    WHAT: Runs trivial_world.check_world live on training and validation worlds and report-only test worlds,
    and cross-checks the recorded 680-world summary metrics.

    WHY: Proves Check 39: verifies that live gated worlds stay below the configured threshold and recorded
    summary shows 0 gated failures.

    Args:
        workspace: Optional execution workspace supplied by the showcase framework.

    Returns:
        CheckOutcome reporting live best AP scores against threshold and recorded gated failure counts.

    Example:
        outcome = await run_trivial_world_check(None)
        assert "0 failed" in outcome.key_number
    """
    info = inspect_trivial_world_live_and_recorded()
    live_gated = info["evidence"]["live_gated"]
    assert live_gated["training"]["passed"], "live training world must pass trivial world check"
    assert live_gated["validation"]["passed"], "live validation world must pass trivial world check"
    recorded = info["evidence"]["recorded, 680 worlds"]
    assert recorded["gated_failed"] == 0, "recorded summary must show 0 gated failures"

    return CheckOutcome(key_number=info["key_number"], evidence=info["evidence"])


async def run_fraud_world_card_check(workspace: Any) -> CheckOutcome:
    """
    WHAT: Evaluates build_fraud_world_card on the 12e showcase scam world, measuring base rate and inflation factor.

    WHY: Proves Check 40: verifies that fraud world cards report fraud share, complaints, rate per 100k,
    real-world ratio and source, and lower-bound inflation factor.

    Args:
        workspace: Optional execution workspace supplied by the showcase framework.

    Returns:
        CheckOutcome reporting fraud inflation factor and full base rate evidence attributes.

    Example:
        outcome = await run_fraud_world_card_check(None)
        assert "factor" in outcome.key_number
    """
    info = inspect_fraud_world_card_base_rate()
    evidence = info["evidence"]
    assert evidence["factor"] > 1.0, f"factor {evidence['factor']} must be > 1.0"
    assert evidence["complaints"] >= 0, "complaints count must be non-negative"
    assert evidence["real_world_ratio"] > 0, "real-world ratio must be positive"

    return CheckOutcome(key_number=info["key_number"], evidence=evidence)


async def run_sanity_report_check(workspace: Any) -> CheckOutcome:
    """
    WHAT: Executes world_genuine_stats live on 2 training worlds and validates the recorded PaySim and IBM tables.
    Includes the Level 12h brief map table in the evidence dictionary.

    WHY: Proves Check 41: verifies plausibility comparison reporting, external licensing metadata, and maps
    the brief's 17 showcase items to check numbers.

    Args:
        workspace: Optional execution workspace supplied by the showcase framework.

    Returns:
        CheckOutcome reporting genuine payment count and verification of PaySim and IBM references.

    Example:
        outcome = await run_sanity_report_check(None)
        assert "PaySim" in outcome.key_number
    """
    info = inspect_sanity_report_check()
    assert info["evidence"]["live_sample"]["genuine_count"] > 0, "must find genuine payments"
    assert "Level 12h brief items -> checks" in info["evidence"], "must include brief items mapping"

    return CheckOutcome(key_number=info["key_number"], evidence=info["evidence"])
