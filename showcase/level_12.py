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
from showcase.level_12_checks_b import (
    run_gang_consistency_check,
    run_gang_two_layers_check,
    run_over_budget_gang_negative_check,
    run_temporal_truth_edge_check,
    run_truth_isolation_negative_check,
)
from showcase.level_12_checks_c import (
    run_genuine_shop_downstream_check,
    run_mule_lifecycle_check,
    run_rented_mule_normal_life_check,
    run_victim_payment_outage_check,
    run_victim_to_cashout_check,
)
from showcase.level_12_checks_d import (
    run_account_market_check,
    run_counts_by_family_check,
    run_held_out_in_training_refusal_check,
    run_late_lien_observation_check,
    run_training_world_families_check,
)
from showcase.level_12_checks_e import (
    run_analyst_queue_check,
    run_bulk_batch_check,
    run_complaint_delay_by_type_check,
    run_complaint_share_large_batch_check,
    run_complaints_against_truth_check,
)
from showcase.level_12_checks_f import (
    run_coverage_plot_check,
    run_pattern_file_check,
    run_plugin_block_check,
    run_test_world_on_training_ranges_check,
    run_validation_listed_apart_check,
)
from showcase.level_12_checks_g import (
    run_device_rotation_check,
    run_fake_routine_life_deferred_check,
    run_held_out_property_distribution_check,
    run_held_out_validator_check,
    run_slow_drip_check,
    run_tagged_gangs_check,
    run_training_held_out_refusal_check,
)
from showcase.level_12_checks_h import (
    run_edge_cases_check,
    run_fraud_world_card_check,
    run_sanity_report_check,
    run_trivial_world_check,
)

LEVEL_NUMBER = 12
LEVEL_TITLE = "World engine II: a world that can react (built: parts 12a-12f of 12a-12g)"
NEEDS_DATABASE = False

WHAT_YOU_SAW = (
    "Level 12a builds a reactive world engine: outside events (liens, debit holds, releases, payment resumes), "
    "an interactive runner that can hold and resume payments while preserving true origin and child random draws, "
    "bit-for-bit equivalence with Level 11 under allow-all execution, and v2 issue tracking. "
    "Level 12b adds hidden truth isolation, a tactic language with budget enforcement, and gang device telemetry. "
    "Level 12c adds victim money routing, mule lifecycle transitions from warm-up to burned, "
    "rented owner activity, and cash-out channels. "
    "Level 12d adds an account market where gangs buy burned accounts from each other, a delayed lien feed "
    "that gangs react to only after they learn of it, and tactic families that keep held-out tactics out of "
    "training worlds. "
    "Level 12e adds labels: complaints about payments that a bank could receive, with delays by scam type, "
    "bulk confirmations by police crackdowns, and a simulated analyst whose accuracy falls as the queue grows. "
    "Level 12f adds a many-worlds sampler (Latin hypercube blocks per split, disjoint seed blocks, test worlds on the "
    "training ranges as Section 6 requires), a per-world gang mix, pattern files with a budget check and preview, "
    "local plugins that may only spend time and money, and world invariant checks used by a sweep of 80 generated "
    "world runs."
)

REPO_ROOT = Path(__file__).resolve().parent.parent
REQUIRED_FILES = [
    REPO_ROOT / "config" / "showcase_level_12.yaml",
    REPO_ROOT / "config" / "issues_v2.yaml",
    REPO_ROOT / "config" / "frozen_anchors.yaml",
    REPO_ROOT / "config" / "base_rate.yaml",
    REPO_ROOT / "config" / "sanity_report.yaml",
    REPO_ROOT / "reports" / "trivial_world" / "summary.md",
    REPO_ROOT / "reports" / "sanity" / "summary.md",
    REPO_ROOT / "scripts" / "golden_values.yaml",
    REPO_ROOT / "showcase" / "fixtures" / "level_12" / "gang_two_layers.yaml",
    REPO_ROOT / "showcase" / "fixtures" / "level_12" / "gang_victims_cashout.yaml",
    REPO_ROOT / "showcase" / "fixtures" / "level_12" / "gangs_market_two.yaml",
    REPO_ROOT / "showcase" / "fixtures" / "level_12" / "gangs_families_training.yaml",
    REPO_ROOT / "showcase" / "fixtures" / "level_12" / "gang_held_out_in_training.yaml",
    REPO_ROOT / "showcase" / "fixtures" / "level_12" / "gang_complaints.yaml",
    REPO_ROOT / "showcase" / "fixtures" / "level_12" / "bulk_short_lag.yaml",
    REPO_ROOT / "patterns" / "examples" / "scam_ring_basic.yaml",
    REPO_ROOT / "patterns" / "examples" / "scam_ring_over_budget.yaml",
    REPO_ROOT / "patterns" / "plugins" / "cool_off.py",
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


def _get_checks_6_to_8() -> list[CheckDefinition]:
    """
    WHAT: Constructs check definition descriptors for Level 12b checks 6 through 8.
    It builds CheckDefinition objects for multi-hop gang routing, truth isolation guards,
    and budget overrun refusal.

    WHY: Decomposing check construction into modular subroutines keeps all functions under 30 code lines.
    Separates routing verification and negative isolation guards into a focused builder.

    Returns:
        List of CheckDefinition containers representing checks 6 through 8.

    Example:
        checks = _get_checks_6_to_8(); assert len(checks) == 3
    """
    check_6 = CheckDefinition(
        "6. a gang moves fixture money through two layers", "normal",
        "A gang routes fixture money through two layers of mules with complete forensic payment stories.",
        run_gang_two_layers_check, "Gang planning or routing failed to construct mule chains or attach stories.",
        "world/gang_planner.py, showcase/fixtures/level_12/gang_two_layers.yaml", part="b",
    )
    check_7 = CheckDefinition(
        "7. ashFALL code cannot open the truth", "negative",
        "Operational ashFALL packages are strictly prevented from reading hidden truth at runtime and build time.",
        run_truth_isolation_negative_check,
        "Caller inspection or static AST import guards failed to block truth access.",
        "evaluation/truth.py, core/truth_import_check.py", part="b",
    )
    check_8 = CheckDefinition(
        "8. an over-budget gang is refused with a plain reason", "negative",
        "A gang exceeding its configured spending limits is halted with an informative reason before execution.",
        run_over_budget_gang_negative_check,
        "Budget tracking failed to detect expenditure overruns or failed to raise GangBudgetExceeded.",
        "world/gang_planner.py, world/interactive_runner.py", part="b",
    )
    return [check_6, check_7, check_8]


def _get_checks_9_to_10() -> list[CheckDefinition]:
    """
    WHAT: Constructs check definition descriptors for Level 12b checks 9 and 10.
    It builds CheckDefinition objects for multi-artifact deterministic reproducibility
    and temporal truth role transitions.

    WHY: Decomposing check construction into modular subroutines keeps all functions under 30 code lines.
    Separates multi-artifact consistency assertions and temporal boundary checks into a focused builder.

    Returns:
        List of CheckDefinition containers representing checks 9 and 10.

    Example:
        checks = _get_checks_9_to_10(); assert len(checks) == 2
    """
    check_9 = CheckDefinition(
        "9. the same seed gives the same gang activity and truth", "consistency",
        "Identical seeds produce bit-for-bit identical event streams, device streams, and truth files.",
        run_gang_consistency_check, "Nondeterministic random stream calls or unsorted dictionary serialization.",
        "world/interactive_runner.py, world/truth_writer.py", part="b",
    )
    check_10 = CheckDefinition(
        "10. truth is temporal: a rented account is genuine before its rental", "edge",
        "Rented mule accounts remain genuine before recruitment, and gang devices take control at the takeover minute.",
        run_temporal_truth_edge_check,
        "Temporal role span evaluation leaked mule status retroactively or device control times drifted.",
        "evaluation/truth.py, world/gang_planner.py", part="b",
    )
    return [check_9, check_10]


def _get_checks_11_to_13() -> list[CheckDefinition]:
    """
    WHAT: Constructs check definition descriptors for Level 12c checks 11 through 13.
    It builds CheckDefinition objects for victim-to-cash-out laundering flow,
    rented mule normal routine preservation, and temporal mule lifecycle progression.

    WHY: Decomposing check construction into modular subroutines keeps all functions under 30 code lines.
    Separates victim routing, owner routine co-existence, and temporal lifecycle transitions into a focused builder.

    Returns:
        List of CheckDefinition containers representing checks 11 through 13.

    Example:
        checks = _get_checks_11_to_13(); assert len(checks) == 3
    """
    check_11 = CheckDefinition(
        "11. victim to mules to cash-out for one gang", "normal",
        "A scam gang converts victim fraud into layered mule forwards and diverse cash-out extractions.",
        run_victim_to_cashout_check,
        "Victim payment generation, forwarding schedule, or cash-out routing failed.",
        "world/scam_planner.py, world/cash_out.py", part="c",
    )
    check_12 = CheckDefinition(
        "12. a rented mule keeps its normal life", "edge",
        "A rented mule account continues normal financial routine alongside paused outgoing owner transfers.",
        run_rented_mule_normal_life_check,
        "Rented owner activity planner paused all activity or failed to preserve genuine transactions.",
        "world/rented_owner_activity.py, world/interactive_loop.py", part="c",
    )
    check_13 = CheckDefinition(
        "13. one mule from warm-up to burned", "edge",
        "A mule account transitions deterministically through warm_up, active, and burned lifecycle phases.",
        run_mule_lifecycle_check,
        "Mule lifecycle phase calculation drifted or failed to track transition minute boundaries.",
        "evaluation/truth.py, world/gang_planner.py", part="c",
    )
    return [check_11, check_12, check_13]


def _get_checks_14_to_15() -> list[CheckDefinition]:
    """
    WHAT: Constructs check definition descriptors for Level 12c checks 14 and 15.
    It builds CheckDefinition objects for downstream genuine merchant payments and outage forward suppression.

    WHY: Decomposing check construction into modular subroutines keeps all functions under 30 code lines.
    Separates downstream genuine merchant interaction and negative bank downtime handling into a focused builder.

    Returns:
        List of CheckDefinition containers representing checks 14 and 15.

    Example:
        checks = _get_checks_14_to_15(); assert len(checks) == 2
    """
    check_14 = CheckDefinition(
        "14. a genuine shop receives mule money and stays genuine", "edge",
        "A genuine retail merchant receives illicit mule money through cash-out while staying genuine in truth.",
        run_genuine_shop_downstream_check,
        "Downstream merchant shop was incorrectly assigned an illicit role span in truth.",
        "world/cash_out.py, evaluation/truth.py", part="c",
    )
    check_15 = CheckDefinition(
        "15. a victim payment fails in an outage and nothing is forwarded", "negative",
        "Bank downtime halts victim money inflows cleanly, causing dependent forwarding transfers to be skipped.",
        run_victim_payment_outage_check,
        "Outage decision failed to decline victim payment or forward skipping logic failed.",
        "world/interactive_loop.py, world/failures.py", part="c",
    )
    return [check_14, check_15]


def _get_checks_16_to_20() -> list[CheckDefinition]:
    """
    WHAT: Constructs check definition descriptors for Level 12d checks 16 through 20.
    It builds CheckDefinition objects for the account market, tactic families in a training world, counts by
    family, a lien observed after the feed delay, and the refusal of a held-out family in a training world.

    WHY: Decomposing check construction into modular subroutines keeps all functions under 30 code lines.
    Separates the market, family and reaction checks of part d into a focused builder.

    Returns:
        List of CheckDefinition containers representing checks 16 through 20.

    Example:
        checks = _get_checks_16_to_20(); assert len(checks) == 5
    """
    check_16 = CheckDefinition(
        "16. two gangs share accounts bought from the market", "edge",
        "A burned rented mule of one gang is resold to a later gang and the truth owner changes exactly at the resale.",
        run_account_market_check,
        "Resale matching, plan rewriting or truth span closing drifted from the resale minute.",
        "world/account_market.py, showcase/fixtures/level_12/gangs_market_two.yaml", part="d",
    )
    check_17 = CheckDefinition(
        "17. a training world contains no held-out family", "edge",
        "A training world built from gangs of different training families contains no pre-registered held-out family.",
        run_training_world_families_check,
        "Family validation let a held-out family into a training world or the held-out list drifted.",
        "world/tactic_families.py, config/tactic_families.yaml, config/evaluation.yaml", part="d",
    )
    check_18 = CheckDefinition(
        "18. a gang world with counts by family", "normal",
        "Fraud payments and gang accounts can be counted per tactic family and every family produced activity.",
        run_counts_by_family_check,
        "A gang family produced no payments or accounts, or gang ids no longer map to families.",
        "world/gang_config.py, showcase/fixtures/level_12/gangs_families_training.yaml", part="d",
    )
    check_19 = CheckDefinition(
        "19. a lien is observed late; money flows until observed, then stops", "edge",
        "A gang keeps moving money through a liened mule until the feed delay passes, and not after.",
        run_late_lien_observation_check,
        "The observation feed delivered the lien early or late, or the stop rule did not cancel later flows.",
        "world/gang_feed.py, world/gang_reactions.py", part="d",
    )
    check_20 = CheckDefinition(
        "20. a training world file that names a held-out family is refused", "negative",
        "A gang naming a held-out family in a training world is refused with a plain reason before any planning.",
        run_held_out_in_training_refusal_check,
        "The family check was skipped or ran after planning began.",
        "world/tactic_families.py, world/interactive_runner.py", part="d",
    )
    return [check_16, check_17, check_18, check_19, check_20]


def _get_checks_21_to_25() -> list[CheckDefinition]:
    """
    WHAT: Constructs check definition descriptors for Level 12e checks 21 through 25.
    It builds CheckDefinition objects for complaints judged against the truth, complaint delays by scam type, the
    analyst's accuracy against queue load, one late bulk-label batch for a region, and the complaint share on a
    large synthetic batch.

    WHY: Decomposing check construction into modular subroutines keeps all functions under 30 code lines.
    Separates the label and analyst checks of part e into a focused builder.

    Returns:
        List of CheckDefinition containers representing checks 21 through 25.

    Example:
        checks = _get_checks_21_to_25(); assert len(checks) == 5
    """
    check_21 = CheckDefinition(
        "21. complaints arriving against the truth", "normal",
        "Complaints in a scam world name only successful victim payments of the complainant, and a payout back is "
        "never complained about. The complaint share is proven by check 25, not here.",
        run_complaints_against_truth_check,
        "Label building named a failed payment, a payout back or another account.",
        "world/label_builder.py, world/complaints.py, showcase/fixtures/level_12/gang_complaints.yaml", part="e",
    )
    check_22 = CheckDefinition(
        "22. complaint delay by scam type", "edge",
        "Complaint delays stay inside each scam type's range and investment complaints come later than debit "
        "complaints. The inputs are synthetic: 1,000 made-up victim losses per scam type, run through the real "
        "planner.",
        run_complaint_delay_by_type_check,
        "Delay bounds were read from the wrong entry, converted wrongly from days, or drawn outside the range.",
        "world/complaints.py, world/label_records.py, config/labels.yaml", part="e",
    )
    check_23 = CheckDefinition(
        "23. analyst accuracy falls as the queue grows", "edge",
        "The simulated analyst's accuracy never rises with queue load and is lower at load 6 than at load 1, "
        "and measured accuracy matches the formula.",
        run_analyst_queue_check,
        "The overload term was dropped or the accuracy formula drifted from amendment A-022.",
        "world/analyst.py, config/analyst.yaml", part="e",
    )
    check_24 = CheckDefinition(
        "24. a bulk-label batch arrives late for one region", "edge",
        "A crackdown batch confirms only mule-side accounts of one region, within the lag bounds, after the last "
        "fraud payment it covers. The fixture values are labelled, not a real range.",
        run_bulk_batch_check,
        "Bulk planning ignored the region or cut-off, or the lag or arrival drifted from the fixture bounds.",
        "world/bulk_confirmations.py, showcase/fixtures/level_12/bulk_short_lag.yaml", part="e",
    )
    check_25 = CheckDefinition(
        "25. complaint share on a large batch", "normal",
        "The share of victims who complain follows the share the world drew, measured on 5,000 victims. The inputs "
        "are synthetic: made-up victim losses run through the real planner.",
        run_complaint_share_large_batch_check,
        "The complaint draw was compared with the wrong value or the share drifted from its drawn value.",
        "world/complaints.py, world/label_records.py, config/labels.yaml", part="e",
    )
    return [check_21, check_22, check_23, check_24, check_25]


def _get_checks_26_to_30() -> list[CheckDefinition]:
    """
    WHAT: Constructs check definition descriptors for Level 12f checks 26 through 30.
    It builds CheckDefinition objects for the example pattern files, the example plugin block, the coverage plot of a
    sampled block, validation worlds listed apart from training worlds, and a test-split world on the training ranges.

    WHY: Decomposing check construction into modular subroutines keeps all functions under 30 code lines.
    Separates the sampler, pattern and plugin checks of part f into a focused builder.

    Returns:
        List of CheckDefinition containers representing checks 26 through 30.

    Example:
        checks = _get_checks_26_to_30(); assert len(checks) == 5
    """
    check_26 = CheckDefinition(
        "26. an example pattern file validates and previews; an over-budget version is refused", "edge",
        "The example pattern file loads, previews and names its mule accounts, and the same ring with a budget of 100 "
        "paise is refused with a plain reason that starts with 'over budget:'.",
        run_pattern_file_check,
        "The pattern loader, the budget check or the preview changed, or an example file was edited.",
        "world/pattern_files.py, patterns/examples/scam_ring_basic.yaml, patterns/examples/scam_ring_over_budget.yaml",
        part="f",
    )
    check_27 = CheckDefinition(
        "27. the example custom building block runs", "edge",
        "The cool_off plugin plans inside a gang and runs in a small world, its file hash is recorded in the run's "
        "effective configuration, and the run id differs from the same world planned without the plugin step.",
        run_plugin_block_check,
        "The plugin registry, the planner hook or the gang's plain form stopped carrying the plugin hash.",
        "world/tactic_plugins.py, patterns/plugins/cool_off.py, world/gang_config.py", part="f",
    )
    check_28 = CheckDefinition(
        "28. a coverage plot of sampled worlds spread across two ranges", "normal",
        "Training block 0 puts exactly one world in each of the 40 equal slices of the victim complaint share "
        "range and of the registry report share range (40 and 40 on the two continuous axes). Integer settings "
        "are listed as evidence only.",
        run_coverage_plot_check,
        "The hypercube stopped giving one point per stratum, or a range was changed under the plot.",
        "world/world_sampler.py, config/world_sampler.yaml", part="f",
    )
    check_29 = CheckDefinition(
        "29. validation worlds are listed apart from training worlds, with disjoint seeds", "consistency",
        "Training and validation blocks are written to separate manifests, read back, and share no seed; the reserved "
        "Level 11 seed 101 is in neither.",
        run_validation_listed_apart_check,
        "A split was planned from another split's seed block, or a manifest changed on the way to disk.",
        "world/world_sampler.py, config/world_sampler.yaml", part="f",
    )
    check_30 = CheckDefinition(
        "30. a test-split world runs on the training ranges and is still a test world", "edge",
        "A test-split world reads the training range column, records world_split 'test', and every world invariant "
        "holds after it runs with a generated gang mix.",
        run_test_world_on_training_ranges_check,
        "The split and range column were tied together again, or a gang-opened account broke a world invariant.",
        "world/sampled_world.py, world/world_invariants.py, world/interactive_runner.py", part="f",
    )
    return [check_26, check_27, check_28, check_29, check_30]


def _get_checks_31_to_37() -> list[CheckDefinition]:
    """
    WHAT: Constructs check definition descriptors for Level 12g checks 31 through 37.
    It builds CheckDefinition objects for truth-v2 gang taxonomy tags, slow-drip piece and dwell timing dispersion,
    device rotation generation timetables, deferred fake_routine_life rented mule routine credits, training held-out
    refusal, evaluation config validator enforcement, and exhaustive taxonomic property checks across all 680 worlds.

    WHY: Decomposing check construction into modular subroutines keeps all functions under 30 code lines.
    Separates the held-out tactic family verification checks of part g into a focused builder.

    Returns:
        List of CheckDefinition containers representing checks 31 through 37.

    Example:
        checks = _get_checks_31_to_37(); assert len(checks) == 7
    """
    check_31 = CheckDefinition(
        "31. a test world holds one seen gang and one held-out gang, each tagged in the truth", "normal",
        "A test world generated by the sampler tags its gangs as seen or held_out in truth-v2 with at least one "
        "seen and one held-out gang.",
        run_tagged_gangs_check,
        "Gang truth tagging failed or manifest format drifted from truth-v2.",
        "world/truth_writer.py, evaluation/truth.py, world/truth_records.py", part="g",
    )
    check_32 = CheckDefinition(
        "32. slow_drip moves money in small pieces held for hours, spread over days", "edge",
        "The slow_drip evasion family splits funds into pieces under Rs 2,000, holds each piece for hours, "
        "and disperses routes over multiple days.",
        run_slow_drip_check,
        "Slow-drip payment piece caps, hold durations, or multi-day dispersal spans were violated.",
        "world/slow_drip.py, world/gang_planner.py, config/tactic_blocks.yaml", part="g",
    )
    check_33 = CheckDefinition(
        "33. device_rotation swaps each mule's phone on a timetable", "edge",
        "The device_rotation evasion family rotates mule mobile devices periodically according to generation "
        "timetables with exclusive single-account control.",
        run_device_rotation_check,
        "Device rotation timetables drifted or a device controlled multiple accounts simultaneously.",
        "world/gang_devices.py, world/interactive_runner.py, config/tactic_blocks.yaml", part="g",
    )
    check_34 = CheckDefinition(
        "34. fake_routine_life is deferred (A-025); a rented mule's routine credits come from genuine payers", "edge",
        "fake_routine_life is marked DEFERRED (A-025), and a rented mule in a test world receives legitimate "
        "routine credits exclusively from genuine citizens.",
        run_fake_routine_life_deferred_check,
        "fake_routine_life status changed or gang accounts contaminated rented mule routine credits.",
        "world/tactic_families.py, world/interactive_runner.py, config/tactic_families.yaml", part="g",
    )
    check_35 = CheckDefinition(
        "35. a training world given a held-out option is refused with a plain reason", "negative",
        "An interactive training world given a gang with held-out options is refused with a plain reason "
        "naming the option, family, and split.",
        run_training_held_out_refusal_check,
        "The held-out guard failed to intercept held-out options in a training world.",
        "world/held_out_options.py, world/interactive_loop.py, config/tactic_blocks.yaml", part="g",
    )
    check_36 = CheckDefinition(
        "36. the validator rejects a held-out family name with no implementing block", "negative",
        "Evaluation configuration validation rejects a held-out family without an implementing block "
        "with a plain error.",
        run_held_out_validator_check,
        "check_held_out_mechanisms failed to reject an unimplemented active held-out family.",
        "core/config_validator.py, config/evaluation.yaml, config/tactic_blocks.yaml", part="g",
    )
    check_37 = CheckDefinition(
        "37. no held-out gang in any training or validation world; one in every test and stress world", "consistency",
        "Across all 680 sampled worlds, training and validation worlds contain zero held-out gangs, while test "
        "and stress worlds contain exactly one.",
        run_held_out_property_distribution_check,
        "Held-out gang replacement logic leaked into training/validation or failed in test/stress.",
        "world/gang_mix.py, world/world_sampler.py, config/gang_mix.yaml", part="g",
    )
    return [check_31, check_32, check_33, check_34, check_35, check_36, check_37]


def _get_checks_38_to_41() -> list[CheckDefinition]:
    """
    WHAT: Constructs check definition descriptors for Level 12h checks 38 through 41.
    It builds CheckDefinition objects for the edge-case scenario library, trivial-world single-signal
    separability check across live and recorded worlds, fraud-on world card base rate and inflation factor,
    and outside dataset sanity reporting against PaySim and IBM AML-Data.

    WHY: Decomposing check construction into modular subroutines keeps all functions under 30 code lines.
    Separates the final verification and benchmark checks of part h into a focused builder.

    Returns:
        List of CheckDefinition containers representing checks 38 through 41.

    Example:
        checks = _get_checks_38_to_41(); assert len(checks) == 4
    """
    check_38 = CheckDefinition(
        "38. the edge-case library is listed, and each runnable case shows its expected behaviour", "edge",
        "The edge-case library is evaluated across all cases in scenarios/edge_cases; every runnable case passes "
        "and non-runnable cases record their active level or deferred amendment.",
        run_edge_cases_check,
        "An edge case failed its expectation assertion or an inactive case lacked level/amendment metadata.",
        "world/edge_cases.py, scenarios/edge_cases", part="h",
    )
    check_39 = CheckDefinition(
        "39. the trivial-world check: the best single signal stays below the threshold", "consistency",
        "trivial_world.check_world run live on training and validation blocks stays below threshold, and recorded "
        "summary metrics across 680 worlds report zero gated failures.",
        run_trivial_world_check,
        "A live gated world exceeded the single-signal threshold or the recorded summary contained failures.",
        "evaluation/trivial_world.py, config/evaluation.yaml, reports/trivial_world/summary.md", part="h",
    )
    check_40 = CheckDefinition(
        "40. a fraud-on world card carries its base rate and inflation factor", "normal",
        "A fraud-enabled world card built by build_fraud_world_card reports fraud share, complaint count, complaint "
        "rate per 100,000 payments, and a lower-bound fraud inflation factor.",
        run_fraud_world_card_check,
        "Base rate calculation failed or the inflation factor did not exceed the baseline lower bound.",
        "world/world_card.py, world/base_rate.py, config/base_rate.yaml", part="h",
    )
    check_41 = CheckDefinition(
        "41. the sanity report: what matched outside data and what did not, recorded as-is", "edge",
        "world_genuine_stats run live on training worlds compares against recorded PaySim and IBM tables, verifying "
        "licensing and plausibility declarations, and mapping 17 brief items to checks.",
        run_sanity_report_check,
        "Plausibility declaration missing, dataset references corrupted, or live genuine payments failed to sample.",
        "evaluation/sanity_report.py, config/sanity_report.yaml, reports/sanity/summary.md", part="h",
    )
    return [check_38, check_39, check_40, check_41]


def get_checks(quick: bool = False) -> list[CheckDefinition]:
    """
    WHAT: Constructs the ordered sequence of 41 Level 12 showcase verification checks across parts a to h.
    All checks span consistency, normal, edge, and negative check kinds across parts 'a' to 'h'.

    WHY: Supplies the showcase execution engine with the authoritative test suite for Level 12.
    Enforces required coverage distribution across normal, edge, and negative categories while
    tagging each check with part='a', 'b', 'c', 'd', 'e', 'f', 'g' or 'h' for selective execution.

    Args:
        quick: Boolean flag indicating whether quick execution mode is requested.

    Returns:
        List of CheckDefinition instances configuring Level 12 showcase evaluations.

    Example:
        checks = get_checks(quick=True); assert len(checks) == 41
    """
    # Concatenate checks from helper builders to construct complete Level 12 suite.
    checks: list[CheckDefinition] = []
    checks.extend(_get_checks_1_to_3())
    checks.extend(_get_checks_4_to_5())
    checks.extend(_get_checks_6_to_8())
    checks.extend(_get_checks_9_to_10())
    checks.extend(_get_checks_11_to_13())
    checks.extend(_get_checks_14_to_15())
    checks.extend(_get_checks_16_to_20())
    checks.extend(_get_checks_21_to_25())
    checks.extend(_get_checks_26_to_30())
    checks.extend(_get_checks_31_to_37())
    checks.extend(_get_checks_38_to_41())
    return checks



