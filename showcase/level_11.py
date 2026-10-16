"""
WHAT: Entry point for the Level 11 showcase ("A simulated India of everyday UPI payments").
It orchestrates 16 automated checks covering pre-registration amendments, issue register tracking, fast-mode simulation,
empirical grounding, seed divergence, bank outage partial degradation, OS update waves, switch outages, wedding bursts,
banking asymmetry, bit-for-bit repeatability, capacity limits, double-entry ledgers, live replay, and golden baselines.

WHY: Provides a single deterministic command verifying the entire Level 11 world engine completely offline.
Executing checks against canonical seeds guarantees that simulation mechanics behave reproducibly and match empirical
data.
Exposing both quick and full evaluation modes allows rapid local validation within 1 minute or exhaustive audits within
5.

TERM: Level 11: The municipal world simulation engine generating realistic citizens, commercial shops, and UPI payment
      lifecycles.
TERM: fast mode: A streamlined simulation execution pipeline running an entire world directly in memory without delays.
TERM: check: An automated verification routine evaluating an individual simulation property against strict assertions.
TERM: precondition: A mandatory file or environment dependency required to exist before running showcase checks.
"""

from pathlib import Path

from showcase.framework import CheckDefinition
from showcase.level_11_checks import (
    run_amendments_check,
    run_bank_outage_check,
    run_grounding_check,
    run_issue_register_check,
    run_os_wave_check,
    run_seed_divergence_check,
    run_tiny_world_check,
)
from showcase.level_11_checks_more import (
    run_bank_asymmetry_check,
    run_golden_fingerprints_check,
    run_layer_isolation_check,
    run_ledger_invariants_check,
    run_live_replay_check,
    run_oversized_world_check,
    run_repeatability_check,
    run_switch_vs_registry_check,
    run_wedding_episode_check,
)

# The framework looks up write_custom_page on this module; the alias marks it as a deliberate re-export.
from showcase.level_11_page import write_custom_page as write_custom_page

LEVEL_NUMBER = 11
LEVEL_TITLE = "A simulated India of everyday UPI payments"
NEEDS_DATABASE = False

WHAT_YOU_SAW = (
    "Level 11 builds a simulated world of everyday UPI payments: people, shops, three different banks, "
    "festivals, outages, phone changes and life events such as a wedding collection. Every setting carries "
    "a source label, stress tests are labelled layers, the bank ledgers balance, and the same seed gives "
    "byte-identical output."
)

REPO_ROOT = Path(__file__).resolve().parent.parent
REQUIRED_FILES = [
    REPO_ROOT / "config" / "world_ranges.yaml",
    REPO_ROOT / "config" / "background_realism.yaml",
    REPO_ROOT / "config" / "failure_realism.yaml",
    REPO_ROOT / "config" / "device_realism.yaml",
    REPO_ROOT / "config" / "issues.yaml",
    REPO_ROOT / "scripts" / "golden_values.yaml",
    REPO_ROOT / "showcase" / "templates" / "level_11_page.html",
]


async def find_missing_preconditions() -> list[str]:
    """
    WHAT: Evaluates filesystem prerequisites and returns descriptive strings for any missing dependencies.
    It inspects paths for foundational configuration files, golden baselines, and HTML report templates.
    An empty list return indicates all required files exist and showcase execution may proceed safely.

    WHY: Prevents cryptic missing-file crashes by failing fast with clear diagnostic messages.
    Ensures that essential domain configurations and report rendering assets are present before execution.
    Conforms to the showcase framework specification for precondition discovery.

    Returns:
        List of plain-English diagnostic error strings identifying any missing required files.

    Example:
        >>> import asyncio
        >>> missing = asyncio.run(find_missing_preconditions())
        >>> isinstance(missing, list) and len(missing) == 0
        True
    """
    return [
        f"{path.name} is missing - it is part of the Level 11 files"
        for path in REQUIRED_FILES
        if not path.exists()
    ]


def _get_checks_1_to_4() -> list[CheckDefinition]:
    """
    WHAT: Constructs check definition descriptors for checks 1 through 4.
    It builds CheckDefinition objects for amendments, issues, tiny world simulation, and grounding table audits.
    Each definition includes check name, category kind, proving statement, runner callable, and troubleshooting links.

    WHY: Decomposing check construction into modular subroutines keeps all functions under 30 code lines.
    Separates foundational research and configuration validation from downstream operational checks.
    Provides clear diagnostic references for each check's failure troubleshooting paths.

    Returns:
        List of CheckDefinition containers representing checks 1 through 4.

    Example:
        >>> checks = _get_checks_1_to_4()
        >>> len(checks) == 4 and checks[0].name.startswith("1.")
        True
    """
    c1 = CheckDefinition(
        "1. pre-registration amendments and proof", "consistency",
        "Amendments to evaluation rules are recorded openly, and no new-engine file predates the pre-registration.",
        run_amendments_check, "An amendment was lost or a new file predated evaluation rules.",
        "docs/research/amendments.md, config/evaluation.yaml"
    )
    c2 = CheckDefinition(
        "2. issue register status counts and copy rejection", "negative",
        "The 31 research issues show 28 built, 2 planned and 1 deferred; broken copies without mandatory fields are "
        "refused.",
        run_issue_register_check, "An issue was modified or lost from config/issues.yaml.",
        "config/issues.yaml, core/issue_register.py"
    )
    c3 = CheckDefinition(
        "3. tiny world fast-mode simulation", "normal",
        "A tiny world of 60 people and 12 shops over 61 days runs end to end, and after removing weekend, month-start "
        "and fixed-date payments, festival days carry the extra payments the plan sets.",
        run_tiny_world_check, "World building failed, or festival multipliers no longer reach calendar-scaled "
        "payments.",
        "showcase/level_11_festival.py, world/festivals.py, world/rhythms.py, world/calendar_runner.py"
    )
    c4 = CheckDefinition(
        "4. empirical grounding table", "normal",
        "Every parameter across background, failure and device realism is catalogued with an epistemic label and "
        "source citation.",
        run_grounding_check, "A parameter lost its label, source citation or assumption rationale.",
        "world/grounding.py, config/background_realism.yaml"
    )
    return [c1, c2, c3, c4]


def _get_checks_5_to_8() -> list[CheckDefinition]:
    """
    WHAT: Constructs check definition descriptors for checks 5 through 8.
    It builds CheckDefinition objects for seed divergence, bank outage degradation, OS update waves, and switch outages.
    Each definition includes check name, category kind, proving statement, runner callable, and troubleshooting links.

    WHY: Modular helper ensuring the master check list assembly conforms to the 30-line code ceiling.
    Groups simulation divergence, hardware update waves, and infrastructure failure scenario checks together.
    Supplies the runner with clear diagnostic references for each scenario check.

    Returns:
        List of CheckDefinition containers representing checks 5 through 8.

    Example:
        >>> checks = _get_checks_5_to_8()
        >>> len(checks) == 4 and checks[0].name.startswith("5.")
        True
    """
    c5 = CheckDefinition(
        "5. independent seed divergence", "edge",
        "Two different seeds place festivals, outages and OS update waves on different days.",
        run_seed_divergence_check, "Random generator streams collapsed or became deterministic across seeds.",
        "world/world_plan.py, world/calendar_runner.py"
    )
    c6 = CheckDefinition(
        "6. bank downtime partial degradation", "edge",
        "Inside the first long bank outage, payments touching the outage bank partly succeed and partly fail with "
        "bank downtime, while payments between the other banks have no bank downtime failures.",
        run_bank_outage_check, "Bank downtime stopped all outage-bank payments, hit no payment, or reached other "
        "banks.",
        "scenarios/stress/bank-outage-long.yaml, world/failures.py"
    )
    c7 = CheckDefinition(
        "7. os update wave adoption curve and device stability", "edge",
        "With the OS update wave layer, more phones reach the newest OS and in sharper daily jumps than in the "
        "base run, and new-device flags stay within four Poisson standard deviations of the base (issue R5).",
        run_os_wave_check, "OS updates raised new-device flags, or the layer no longer speeds up OS adoption.",
        "scenarios/stress/os-update-wave.yaml, world/device_runtime.py"
    )
    c8 = CheckDefinition(
        "8. switch outage vs registry outage failure domains", "negative",
        "A central switch outage fails all payments inside the window, while deferred registry outages cause zero "
        "event drift.",
        run_switch_vs_registry_check, "Central switch routing failed to fail or deferred scenario altered events.",
        "scenarios/stress/switch-outage.yaml, scenarios/stress/registry-outage.yaml"
    )
    return [c5, c6, c7, c8]


def _get_checks_9_to_12() -> list[CheckDefinition]:
    """
    WHAT: Constructs check definition descriptors for checks 9 through 12.
    It builds CheckDefinition objects for wedding inflows, banking asymmetry, repeatability, and capacity sizing
    refusal.
    Each definition includes check name, category kind, proving statement, runner callable, and troubleshooting links.

    WHY: Modular helper ensuring the master check list assembly conforms to the 30-line code ceiling.
    Groups social burst episodes, demographic banking heterogeneity, and capacity safety checks together.
    Supplies the runner with clear diagnostic references for each domain check.

    Returns:
        List of CheckDefinition containers representing checks 9 through 12.

    Example:
        >>> checks = _get_checks_9_to_12()
        >>> len(checks) == 4 and checks[0].name.startswith("9.")
        True
    """
    c9 = CheckDefinition(
        "9. wedding collection episode money surge", "edge",
        "An injected wedding collection episode, planned with the engine's episode random stream, brings money "
        "into the host account only on days inside its window.",
        run_wedding_episode_check, "Episode payment planning or calendar dispatch failed.",
        "world/episode_payments.py, world/episodes.py"
    )
    c10 = CheckDefinition(
        "10. bank asymmetry in share and weak kyc", "normal",
        "The three banks differ in customer share, weak-KYC share and hold-response delay.",
        run_bank_asymmetry_check, "Bank parameters became uniform or failed to reflect institutional categories.",
        "world/banks.py, config/world_ranges.yaml"
    )
    c11 = CheckDefinition(
        "11. deterministic bit-for-bit repeatability", "consistency",
        "Identical seeds produce bit-for-bit identical payment event files matching golden SHA-256 24fc0e25...",
        run_repeatability_check, "Simulation state drifted or unseeded random draws entered the pipeline.",
        "world/fast_mode.py, world/run_hashes.py"
    )
    c12 = CheckDefinition(
        "12. refusal of oversized world", "negative",
        "Proposed worlds exceeding host memory or runtime limits are safely refused with actionable capacity guidance.",
        run_oversized_world_check, "The capacity sizing decision engine accepted an oversized world.",
        "world/world_sizing.py, world/machine_limits.py"
    )
    return [c9, c10, c11, c12]


def _get_checks_13_to_16() -> list[CheckDefinition]:
    """
    WHAT: Constructs check definition descriptors for checks 13 through 16.
    It builds CheckDefinition objects for double-entry ledgers, live replay, layer isolation, and golden fingerprints.
    Each definition includes check name, category kind, proving statement, runner callable, and troubleshooting links.

    WHY: Modular helper ensuring the master check list assembly conforms to the 30-line code ceiling.
    All four checks run in both quick and full mode: together all 16 checks fit the 1-minute quick budget,
    and a NOT RUN check would fail the quick run.

    Returns:
        List of CheckDefinition containers representing checks 13 through 16.

    Example:
        >>> checks = _get_checks_13_to_16()
        >>> len(checks) == 4 and checks[0].name.startswith("13.")
        True
    """
    c13 = CheckDefinition(
        "13. double-entry ledger invariants", "consistency",
        "Double-entry ledgers maintain non-negative balances, entry consistency, and total currency conservation.",
        run_ledger_invariants_check, "A transaction minted currency, allowed negative balance, or corrupted logs.",
        "world/checks.py, world/ledger.py"
    )
    c14 = CheckDefinition(
        "14. live replay event stream fidelity", "consistency",
        "Live streaming mode replays fast-mode events across loopback HTTP with byte-identical payload fidelity.",
        run_live_replay_check, "Live mode altered payloads or failed loopback replay.",
        "world/live_mode.py, tests/unit/world/test_world_live_mode.py"
    )
    c15 = CheckDefinition(
        "15. stress scenario parameter isolation", "normal",
        "Every stress scenario layer modifies only its named parameters, leaving unmentioned domain settings "
        "identical.",
        run_layer_isolation_check, "A stress layer mutated unmentioned parameter blocks.",
        "scenarios/stress/README.md, world/layers.py"
    )
    c16 = CheckDefinition(
        "16. golden simulation fingerprints", "consistency",
        "The golden fingerprints FP, G1, G2 and G3 recompute in this run and match the stored values.",
        run_golden_fingerprints_check, "World topology, monthly routine, or calendar hashes drifted.",
        "scripts/verify_world.py, scripts/golden_values.yaml"
    )
    return [c13, c14, c15, c16]


def get_checks(quick: bool) -> list[CheckDefinition]:
    """
    WHAT: Returns the ordered list of all 16 Level 11 showcase checks configured for the designated execution mode.
    Quick and full mode run the same 16 checks; the framework still passes the quick flag,
    which this level accepts but does not need because all checks fit the quick budget.

    WHY: Supplies the showcase execution engine with the authoritative test suite for Level 11.
    Combining modular sublists ensures the master dispatcher satisfies the 30-line code ceiling.
    Validating checks against coverage standards guarantees an appropriate balance of normal, edge, and negative checks.

    Args:
        quick: True in quick mode; the same 16 checks run either way.

    Returns:
        List of all 16 CheckDefinition instances in sequential execution order.

    Example:
        >>> checks = get_checks(quick=True)
        >>> len(checks) == 16 and checks[0].name.startswith("1.")
        True
    """
    all_checks: list[CheckDefinition] = []
    all_checks.extend(_get_checks_1_to_4())
    all_checks.extend(_get_checks_5_to_8())
    all_checks.extend(_get_checks_9_to_12())
    all_checks.extend(_get_checks_13_to_16())
    return all_checks
