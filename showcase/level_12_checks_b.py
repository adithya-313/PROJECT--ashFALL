"""
WHAT: Implementation of Level 12 Showcase Part B checks (Checks 6 through 10).
Evaluates multi-hop gang laundering routing, runtime and static truth isolation guards,
budget overrun refusal, deterministic multi-artifact reproducibility, and temporal truth properties.

WHY: Verifies the Level 12b reactive engine capabilities: tactical planning from plain configs,
criminal infrastructure simulation (gang devices and roles), and cryptographic isolation of ground truth.

TERM: Level 12: World engine II supporting reactive outside events and interactive fraud simulation.
TERM: hidden truth: ground-truth records describing illicit roles, device controls, and payment stories.
TERM: mule depth: distance in payment hops from the initial compromised source to an illicit account.
TERM: payment story: forensic narrative attributing a fraud payment to a gang, tactic block, and layer.
TERM: temporal truth: account and device roles resolve dynamically according to simulation time intervals.
"""

import tempfile
from pathlib import Path
from typing import Any

from core.truth_import_check import find_truth_violations
from evaluation.truth import load_truth
from showcase.framework import CheckOutcome
from showcase.level_12_support_b import (
    REPO_ROOT,
    compare_reproducible_runs,
    load_showcase_b_fixture,
    run_showcase_b_world,
    verify_gang_money_rules,
    verify_runtime_truth_caller_guard,
    verify_static_truth_import_guard,
    verify_temporal_mule_transition,
)
from world.gang_config import GangBudget, GangConfig
from world.gang_planner import GangBudgetExceeded


async def run_gang_two_layers_check(workspace: Any) -> CheckOutcome:
    """
    WHAT: Runs the frozen-anchor world with the two-layer fixture gang, writes hidden truth files,
    loads truth via load_truth, and verifies mule depths {1, 2}, route stories, collector role, and money conservation.

    WHY: Proves Check 6: a configured gang executes multi-hop laundering routing, attaches forensic
    payment stories to all fraud events, and preserves double-entry accounting integrity.

    Args:
        workspace: Optional execution workspace supplied by the showcase framework.

    Returns:
        CheckOutcome reporting mule counts, depth distribution, and route payment totals.

    Example:
        # outcome = asyncio.run(run_gang_two_layers_check(None))
        # assert "4 mules at depths 1-2" in outcome.key_number
    """
    # 1. Execute world with fixture gang and load hidden truth from temporary directory.
    gang, _ = load_showcase_b_fixture()
    with tempfile.TemporaryDirectory() as temp_dir:
        truth_root = Path(temp_dir)
        output, run_dir = run_showcase_b_world(gang_config=gang, truth_root=truth_root)
        assert run_dir is not None
        store = load_truth(truth_root, output.result.run_id)

        # 2. Verify mule depths are exactly {1, 2} and collector role is established.
        mule_roles = [r for r in store._roles if r["role"] == "mule"]
        mule_depths = {r["depth"] for r in mule_roles}
        assert mule_depths == {1, 2}, f"Expected depths {{1, 2}}, got {mule_depths}"
        collector_role = store.role_at(gang.source.account_id, gang.source.start_minute)
        assert collector_role == "collector", f"Collector role is {collector_role}"

        # 3. Assert all fraud events possess forensic payment stories with block 'route'.
        fraud_positions = [idx for idx, src in enumerate(output.result.event_sources) if src == "fraud"]
        for pos in fraud_positions:
            payment_id = output.result.payment_ids[pos]
            assert store.story_of(payment_id)["block"] == "route"

        # 4. Verify double-entry money conservation rules accounting for fixture opening balance.
        verify_gang_money_rules(output, gang.source.opening_balance_paise)

        key_number = f"{len(mule_roles)} mules at depths 1-2, {len(fraud_positions)} route payments, all with stories"
        evidence = {
            "roles_sample": store._roles[:3],
            "stories_sample": list(store._payment_stories.values())[:3],
            "mule_count": len(mule_roles),
            "route_payments": len(fraud_positions),
        }
        return CheckOutcome(key_number=key_number, evidence=evidence)


async def run_truth_isolation_negative_check(workspace: Any) -> CheckOutcome:
    """
    WHAT: Verifies triple-layer truth isolation: runtime PermissionError when loaded from 'api.caller',
    static AST detection of forbidden imports in synthetic 'api/service.py', and zero violations in real repo.

    WHY: Proves Check 7: operational banking packages are strictly prohibited from inspecting simulation
    ground truth at runtime and at lint time, protecting models from synthetic label leakage.

    Args:
        workspace: Optional execution workspace supplied by the showcase framework.

    Returns:
        CheckOutcome reporting runtime refusal, static detection count, and real repo clean status.

    Example:
        # outcome = asyncio.run(run_truth_isolation_negative_check(None))
        # assert "runtime PermissionError" in outcome.key_number
    """
    # 1. Run world and create valid truth directory in temporary directory.
    gang, _ = load_showcase_b_fixture()
    with tempfile.TemporaryDirectory() as temp_dir:
        truth_root = Path(temp_dir)
        output, run_dir = run_showcase_b_world(gang_config=gang, truth_root=truth_root)
        assert run_dir is not None

        # 2. Verify runtime caller guard raises PermissionError when invoked by operational module.
        runtime_msg = verify_runtime_truth_caller_guard(truth_root, output.result.run_id, truth_root)

        # 3. Verify static checker detects unauthorized import in a synthetic operational tree.
        static_violations = verify_static_truth_import_guard(truth_root)
        assert len(static_violations) == 1, f"Expected 1 static violation, got {static_violations}"

        # 4. Verify static scan on the real repository finds zero unauthorized imports or env usages.
        real_violations = find_truth_violations(REPO_ROOT)
        assert not real_violations, f"Real repo violations found: {real_violations}"

        key_number = (
            f"runtime PermissionError from api.caller; static scan caught {len(static_violations)}/1 in api; "
            f"real repo 0 violations"
        )
        evidence = {
            "runtime_message": runtime_msg,
            "static_violations": static_violations,
            "real_violations_count": len(real_violations),
        }
        return CheckOutcome(key_number=key_number, evidence=evidence)


async def run_over_budget_gang_negative_check(workspace: Any) -> CheckOutcome:
    """
    WHAT: Plans a gang whose money budget equals the deliberately insufficient over_budget_money_paise.
    Asserts GangBudgetExceeded is raised naming money, planned sum, and block, and verifies nothing ran.

    WHY: Proves Check 8: gangs cannot execute campaigns that exceed resource boundaries,
    and budget refusals fail fast before creating truth artifacts or executing world payments.

    Args:
        workspace: Optional execution workspace supplied by the showcase framework.

    Returns:
        CheckOutcome reporting the shortened budget refusal diagnostic and zero-run verification.

    Example:
        # outcome = asyncio.run(run_over_budget_gang_negative_check(None))
        # assert "refused" in outcome.key_number
    """
    # 1. Construct over-budget gang configuration with low money ceiling from fixture.
    gang, over_budget_money_paise = load_showcase_b_fixture()
    tight_budget = GangBudget(
        money_paise=over_budget_money_paise,
        time_minutes=gang.budget.time_minutes,
        max_risk=gang.budget.max_risk,
    )
    over_gang = GangConfig(
        gang_id=gang.gang_id,
        strength=gang.strength,
        budget=tight_budget,
        source=gang.source,
        steps=gang.steps,
    )

    # 2. Attempt execution in temporary directory and capture GangBudgetExceeded.
    with tempfile.TemporaryDirectory() as temp_dir:
        truth_root = Path(temp_dir)
        try:
            run_showcase_b_world(gang_config=over_gang, truth_root=truth_root)
        except GangBudgetExceeded as error:
            message = str(error)
            assert "money" in message
            assert "exceeds limit" in message
            assert "route" in message
            # 3. Assert nothing was written to the truth directory.
            assert not any(truth_root.iterdir()), "Truth folder was created despite budget refusal"
        else:
            raise AssertionError("GangBudgetExceeded was not raised for over-budget gang.")

        key_number = f"refused: {message}"
        evidence = {"exception_message": message, "truth_folder_empty": True}
        return CheckOutcome(key_number=key_number, evidence=evidence)


async def run_gang_consistency_check(workspace: Any) -> CheckOutcome:
    """
    WHAT: Runs the fixture gang simulation twice under the same seed and compares events-v2 digest,
    manifest, roles, device control, payment stories, applied gang device events, and merged stream.

    WHY: Proves Check 9: simulation execution, device telemetry generation, and cryptographic truth
    serialization are completely deterministic and bit-for-bit repeatable.

    Args:
        workspace: Optional execution workspace supplied by the showcase framework.

    Returns:
        CheckOutcome reporting 7/7 identical comparisons across independent executions.

    Example:
        # outcome = asyncio.run(run_gang_consistency_check(None))
        # assert "7/7 matches" in outcome.key_number
    """
    # 1. Execute two independent simulation runs with truth stored in isolated temporary directories.
    gang, _ = load_showcase_b_fixture()
    with tempfile.TemporaryDirectory() as td1, tempfile.TemporaryDirectory() as td2:
        root1, root2 = Path(td1), Path(td2)
        out1, dir1 = run_showcase_b_world(gang_config=gang, truth_root=root1)
        out2, dir2 = run_showcase_b_world(gang_config=gang, truth_root=root2)
        assert dir1 is not None and dir2 is not None

        # 2. Evaluate equality across events-v2, 4 truth files, gang device events, and device stream.
        matches, total = compare_reproducible_runs(dir1, out1, dir2, out2)
        assert matches == total, f"Expected {total}/{total} matches, got {matches}/{total}"

        key_number = f"{matches}/{total} matches (events-v2, 4 truth files, device events, unified stream)"
        evidence = {"matches": matches, "total": total, "parity_ratio": f"{matches}/{total}"}
        return CheckOutcome(key_number=key_number, evidence=evidence)


async def run_temporal_truth_edge_check(workspace: Any) -> CheckOutcome:
    """
    WHAT: Evaluates temporal transitions of a leased mule account: role is 'genuine' before recruitment,
    transitions to 'mule' at valid_from, and controllers_of returns the gang device with sim_swap and remote_access.

    WHY: Proves Check 10: truth is temporal and accounts are genuine prior to recruitment without retroactive
    contamination, while gang devices take control at the takeover timestamp.

    Args:
        workspace: Optional execution workspace supplied by the showcase framework.

    Returns:
        CheckOutcome reporting temporal role transition and device takeover verification.

    Example:
        # outcome = asyncio.run(run_temporal_truth_edge_check(None))
        # assert "genuine at min" in outcome.key_number
    """
    # 1. Execute world with fixture gang and load hidden truth store.
    gang, _ = load_showcase_b_fixture()
    with tempfile.TemporaryDirectory() as temp_dir:
        truth_root = Path(temp_dir)
        output, run_dir = run_showcase_b_world(gang_config=gang, truth_root=truth_root)
        assert run_dir is not None
        store = load_truth(truth_root, output.result.run_id)

        # 2. Verify temporal transitions and device takeover telemetry via helper.
        key_number, evidence = verify_temporal_mule_transition(store)
        return CheckOutcome(key_number=key_number, evidence=evidence)
