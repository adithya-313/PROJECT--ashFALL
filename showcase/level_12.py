"""
WHAT: Entry point for Level 12 showcase ("World engine II: a world that can react (built: part 12a of 12a-12g)").
Orchestrates automated verification checks for Level 12a covering engine consistency, split-world equivalence,
lien debit blocking, held payment isolation, and negative issue register schema validation.

WHY: Provides a single deterministic command verifying the Level 12 reactive world engine completely offline.
Proves that interactive outside events, bank decision hooks, and hold/resume lifecycles function correctly
without regressing Level 11 double-entry accounting or deterministic simulation repeatability.

TERM: Level 12: World engine II supporting reactive outside events and interactive fraud simulation.
TERM: check: An automated verification routine evaluating an individual simulation property against strict assertions.
TERM: precondition: A mandatory file or environment dependency required to exist before running showcase checks.
TERM: part: One letter naming a slice of a level that is built and reviewed on its own, e.g. "a" for Level 12a.
"""

from pathlib import Path

from showcase.framework import CheckDefinition
from showcase.level_12_checks import (
    run_consistency_check,
    run_held_payment_isolation_check,
    run_lien_blocks_debit_check,
    run_split_world_check,
    run_v2_register_negative_check,
)

LEVEL_NUMBER = 12
LEVEL_TITLE = "World engine II: a world that can react (built: part 12a of 12a-12g)"
NEEDS_DATABASE = False

WHAT_YOU_SAW = (
    "Level 12a builds a reactive world engine: outside events (liens, debit holds, releases, payment resumes), "
    "an interactive runner that can hold and resume payments while preserving true origin and child random draws, "
    "bit-for-bit equivalence with Level 11 under allow-all execution, and v2 issue tracking."
)

REPO_ROOT = Path(__file__).resolve().parent.parent
REQUIRED_FILES = [
    REPO_ROOT / "config" / "showcase_level_12.yaml",
    REPO_ROOT / "config" / "issues_v2.yaml",
    REPO_ROOT / "config" / "frozen_anchors.yaml",
    REPO_ROOT / "scripts" / "golden_values.yaml",
]


async def find_missing_preconditions() -> list[str]:
    """
    WHAT: Evaluates filesystem prerequisites and returns descriptive strings for any missing dependencies.
    It inspects paths for foundational Level 12 configuration files, issue registers, and frozen anchors.
    An empty list return indicates all required files exist and showcase execution may proceed safely.

    WHY: Prevents cryptic missing-file crashes by failing fast with clear diagnostic messages.
    Ensures that essential domain configurations and validation assets are present before execution.

    Args:
        None.

    Returns:
        List of plain-English diagnostic error strings identifying any missing required files.

    Example:
        missing = asyncio.run(find_missing_preconditions()); assert len(missing) == 0
    """
    # Verify each mandatory filesystem artifact required for Level 12 showcase execution.
    return [
        f"{path.name} is missing - it is part of the Level 12 files"
        for path in REQUIRED_FILES
        if not path.exists()
    ]


def _get_checks_1_to_3() -> list[CheckDefinition]:
    """
    WHAT: Constructs check definition descriptors for Level 12a checks 1 through 3.
    It builds CheckDefinition objects for engine consistency, split-world equivalence, and lien debit blocking.
    Each definition includes the check title, category kind, proving statement, runner callable, and reference files.

    WHY: Decomposing check construction into modular subroutines keeps all functions under 30 code lines.
    Separates baseline consistency and execution equivalence checks from targeted isolation checks.
    Provides clear diagnostic references for troubleshooting and test suite orchestration.

    Returns:
        List of CheckDefinition containers representing checks 1 through 3.

    Example:
        checks = _get_checks_1_to_3(); assert len(checks) == 3
    """
    check_1 = CheckDefinition(
        "1. fraud-off engine consistency with Level 11", "consistency",
        "The fraud-off world engine bit-for-bit matches frozen anchors and golden fingerprints.",
        run_consistency_check, "Baseline random streams or event serialization logic drifted.",
        "config/frozen_anchors.yaml, world/fast_mode.py", part="a",
    )
    check_2 = CheckDefinition(
        "2. split-world equivalence across fraud switch", "normal",
        "The allow-all interactive runner reproduces Level 11 on the split-world benchmark with identical IDs.",
        run_split_world_check, "Payment splitting or child stream handling diverged between engine modes.",
        "world/interactive_runner.py, config/showcase_level_12.yaml", part="a",
    )
    check_3 = CheckDefinition(
        "3. lien blocks outgoing debits with funds_on_hold", "edge",
        "A targeted outside lien encumbrance blocks 30 debits as funds_on_hold while money rules hold.",
        run_lien_blocks_debit_check, "Encumbrance book failed to track liens or failed to block outgoing transfers.",
        "world/encumbrance.py, world/interactive_loop.py", part="a",
    )
    return [check_1, check_2, check_3]


def _get_checks_4_to_5() -> list[CheckDefinition]:
    """
    WHAT: Constructs check definition descriptors for Level 12a checks 4 and 5.
    It builds CheckDefinition objects for compliance payment hold isolation and negative issue register validation.
    Each definition includes the check title, category kind, proving statement, runner callable, and reference files.

    WHY: Decomposing check construction into modular subroutines keeps all functions under 30 code lines.
    Separates edge-case payment freezing and register negative assertions into a focused builder.
    Ensures thorough validation coverage across isolation and schema refusal requirements.

    Returns:
        List of CheckDefinition containers representing checks 4 and 5.

    Example:
        checks = _get_checks_4_to_5(); assert len(checks) == 2
    """
    check_4 = CheckDefinition(
        "4. held payment isolation and execution freeze", "edge",
        "A compliance-held payment produces zero events and ledger entries while keeping other events identical.",
        run_held_payment_isolation_check, "Bank hold hook leaked events or executed transfers prematurely.",
        "world/interactive_loop.py, world/interactive_records.py", part="a",
    )
    check_5 = CheckDefinition(
        "5. refusal of invalid v2 issue register", "negative",
        "An invalid v2 issue register missing evidence is safely rejected while real files stay unchanged.",
        run_v2_register_negative_check, "Schema validation failed to enforce mandatory evidence requirements.",
        "core/issue_register_v2.py, config/issues_v2.yaml", part="a",
    )
    return [check_4, check_5]


def get_checks(quick: bool = False) -> list[CheckDefinition]:
    """
    WHAT: Constructs the ordered sequence of 5 Level 12a showcase verification checks.
    All checks belong to part 'a' and span consistency, normal, edge, and negative check kinds.

    WHY: Supplies the showcase execution engine with the authoritative test suite for Level 12a.
    Enforces required coverage distribution across normal, edge, and negative categories while
    tagging each check with part='a' for selective execution.

    Args:
        quick: Boolean flag indicating whether quick execution mode is requested.

    Returns:
        List of CheckDefinition instances configuring Level 12a showcase evaluations.

    Example:
        checks = get_checks(quick=True); assert len(checks) == 5
    """
    # Concatenate checks from helper builders to construct complete Level 12a suite.
    checks: list[CheckDefinition] = []
    checks.extend(_get_checks_1_to_3())
    checks.extend(_get_checks_4_to_5())
    return checks

