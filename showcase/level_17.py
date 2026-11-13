# File: showcase/level_17.py
# Purpose: Level 17 showcase: the rules baseline, its look-alike false triggers and the OS-update wave.
"""
WHAT: Level 17 turns the typologies into a rules baseline. This showcase runs the committed rules
table on constructed mule cases and on two genuine look-alikes, runs the OS-update-wave stress
scenario through the real interactive world and bank feed to show that an OS update alone never
looks like a new device, and shows the two guards that keep a guess or a leaky model out of results.

WHY: The rules are the bar the models must beat, so their firing, their reason codes and their
known false triggers have to be shown with real calls, not described.
"""

from __future__ import annotations

from typing import Any

from showcase.framework import CheckDefinition

LEVEL_NUMBER = 17
LEVEL_TITLE = "Rules baseline, look-alikes and the OS-update wave"
NEEDS_DATABASE = False

WHAT_YOU_SAW = (
    "Level 17 ships five bank-visible rules: fast pass-through, a large credit on a new account, "
    "many payers in one day, near-zero balance after an outflow, and a new device with money moving. "
    "The four constructed mule cases fire all five rules with reason codes RB1..RB5; a busy tea stall "
    "fires RB3 and is counted as a known false trigger rather than hidden; a gig worker fires nothing. "
    "The OS-update-wave stress scenario is run through the interactive world and the bank feed: accounts "
    "whose only device event is an OS update never raise new_device_r5, while a real phone change does. "
    "The committed rules config is untuned on purpose: the evaluation guard refuses it, and the anomaly "
    "detector refuses a test split, so neither a guess nor a model that saw test rows can reach a result."
)


def find_missing_preconditions() -> list[str]:
    """Checks the committed Level 17 fixtures and scikit-learn are present before any check runs."""
    from showcase import level_17_checks as c

    return c.find_missing_preconditions()


def get_checks(quick: bool = False) -> list[CheckDefinition]:
    """Returns the Level 17 checks in the architect's order."""
    from showcase import level_17_checks as c

    def d(name: str, kind: str, proves: str, run: Any, cause: str, where: str) -> CheckDefinition:
        """One check definition."""
        return CheckDefinition(name=name, kind=kind, proves=proves, run=run, likely_cause=cause, where_to_look=where)

    return [
        d(
            "17-1 each rule fires on its constructed case with its reason code",
            "normal",
            "The committed rules config and engine fire all five rules on the four constructed mule cases, "
            "each with the reason code RB1..RB5 the decision path reports.",
            c.check_17_1_rules_fire_on_constructed_cases,
            "A rule condition, its thresholds, or its reason code changed.",
            "config/rules_baseline.yaml, decisions/rules_engine.py",
        ),
        d(
            "17-2 tea stall and gig worker: known false trigger counted, real worker quiet",
            "edge",
            "A busy tea stall fires exactly RB3 and is counted as one known false trigger; a gig worker fires nothing.",
            c.check_17_2_tea_stall_and_gig_worker,
            "A rule threshold or the look-alike feature values changed.",
            "config/rules_baseline.yaml, tests/unit/decisions/test_rules_engine.py",
        ),
        d(
            "17-3 OS-update wave: OS updates stay quiet, phone changes flag",
            "edge",
            "In the OS-update-wave stress scenario, accounts whose only device event is an OS update never "
            "raise new_device_r5 and rule_5 never fires on them, while a real phone change raises "
            "new_device_r5 and lets rule_5 fire once money moves.",
            c.check_17_3_os_update_wave,
            "OS updates are treated as device changes, or phone changes stopped reaching the bank feed.",
            "world/bank_feed.py, features/m1_control_region.py, scenarios/stress/os-update-wave.yaml",
        ),
        d(
            "17-6 untuned rules and a test-split detector are refused",
            "negative",
            "require_tuned refuses the committed all-untuned rules config and train_anomaly_detector "
            'refuses split="test"; the check passes only because both calls raise.',
            c.check_17_6_untuned_and_split_guards,
            "The untuned guard or the detector's split check was removed or weakened.",
            "evaluation/level17_evaluation.py, decisions/anomaly_detector.py",
        ),
    ]
