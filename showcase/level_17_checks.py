# File: showcase/level_17_checks.py
# Purpose: Level 17 showcase checks: the rules table, look-alike accounts, the OS-update wave,
#          fixture evaluation results, a no-labels world, the untuned guard and repeatability.
"""
WHAT: The check functions Level 17's showcase runs, kept apart from the level metadata so the
module stays readable.

WHY: Every check runs real production code on committed inputs: the committed rules config and
engine, the real stress-scenario layer through the interactive world runner and bank feed, and
the real evaluator on the committed fixtures. Nothing here re-implements a decision or a metric.
"""

from __future__ import annotations

import importlib.util
import json
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Any

import fakeredis

from decisions.anomaly_detector import train_anomaly_detector
from decisions.rules_config import load_rules_config
from decisions.rules_engine import RulesDecision, evaluate_rules
from evaluation.level17_evaluation import require_tuned
from features.account_keys import load_account_features_settings
from features.account_records import AccountEventRecord
from features.account_updater import apply_account_event
from features.m1_features import compute_account_features
from showcase.framework import CheckOutcome
from world.bank_feed import BankFeedAccountEvent, project_run
from world.fast_mode import FastRunSettings
from world.interactive_runner import run_interactive_world
from world.layers import apply_layers, read_layer
from world.world_configs import read_world_configs

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_ROOT = REPO_ROOT / "showcase" / "fixtures" / "level_17"
OS_WAVE_LAYER_PATH = REPO_ROOT / "scenarios" / "stress" / "os-update-wave.yaml"

_DEVICE_KINDS = frozenset({"os_update", "device_binding_change", "app_reinstall", "sim_swap"})
_PHONE_CHANGE_KIND = "device_binding_change"

# All-quiet feature values copied from tests/unit/decisions/test_rules_engine.py:12-21
# (_all_quiet_features); the showcase must not import test modules.
_QUIET_FEATURES: dict[str, float | int | None] = {
    "largest_credit_paise_24h": 0,
    "pass_through_ratio_24h": 0.0,
    "median_credit_to_debit_seconds_7d": 86400,
    "account_age_days": 365,
    "distinct_payers_24h": 1,
    "current_balance_paise": 5000000,
    "new_device_r5": 0,
}

# Overrides copied from tests/unit/decisions/test_rules_engine.py:126-181 (CONSTRUCTED_CASES):
# the four mule cases cover rule_1..rule_5 because mule_fast_pass_through fires rule_1 and rule_4;
# the tea stall's rule_3 fire is the known false trigger and the gig worker fires nothing.
_CONSTRUCTED_MULE_CASES: tuple[tuple[str, dict[str, float | int | None], frozenset[str]], ...] = (
    (
        "mule_fast_pass_through",
        {
            "largest_credit_paise_24h": 1500000,
            "pass_through_ratio_24h": 0.9,
            "median_credit_to_debit_seconds_7d": 600,
            "current_balance_paise": 5000,
        },
        frozenset({"rule_1", "rule_4"}),
    ),
    (
        "mule_new_account_large_credit",
        {"largest_credit_paise_24h": 1500000, "account_age_days": 15},
        frozenset({"rule_2"}),
    ),
    (
        "mule_fan_in",
        {"distinct_payers_24h": 25, "largest_credit_paise_24h": 200000},
        frozenset({"rule_3"}),
    ),
    (
        "mule_new_device_money_moved",
        {"new_device_r5": 1, "pass_through_ratio_24h": 0.85, "largest_credit_paise_24h": 500000},
        frozenset({"rule_5"}),
    ),
)

_TEA_STALL_FEATURES: dict[str, float | int | None] = {
    "distinct_payers_24h": 40,
    "largest_credit_paise_24h": 30000,
    "pass_through_ratio_24h": 0.3,
    "current_balance_paise": 2000000,
}

_GIG_WORKER_FEATURES: dict[str, float | int | None] = {
    "largest_credit_paise_24h": 200000,
    "pass_through_ratio_24h": 0.95,
    "current_balance_paise": 5000,
    "distinct_payers_24h": 3,
}

_EXPECTED_REASON_CODES: dict[str, str] = {
    "rule_1": "RB1_FAST_PASS_THROUGH",
    "rule_2": "RB2_LARGE_CREDIT_NEW_ACCOUNT",
    "rule_3": "RB3_MANY_PAYERS_ONE_DAY",
    "rule_4": "RB4_NEAR_ZERO_AFTER_OUTFLOW",
    "rule_5": "RB5_NEW_DEVICE_MONEY_MOVED",
}


def find_missing_preconditions(fixtures_root: Path = FIXTURES_ROOT) -> list[str]:
    """
    WHAT: Names every input Level 17's showcase checks need and does not find.

    WHY: A missing fixture set or a missing scikit-learn must stop the run with the exact item
    named, rather than fail later inside a check with an import or file error.

    Args:
        fixtures_root: Level 17 fixture folder; the default is the committed one.

    Returns:
        Problem strings, one per missing or mismatched item; an empty list means ready to run.
    """
    problems: list[str] = []
    manifest_path = fixtures_root / "FIXTURE_MANIFEST.json"
    if not manifest_path.is_file():
        problems.append(f"Missing required file: {manifest_path}")
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = [path for path in fixtures_root.rglob("*") if path.is_file()]
        if len(files) != int(manifest["file_count"]):
            problems.append(
                f"Fixture file count mismatch: {manifest_path} says {manifest['file_count']}, "
                f"found {len(files)} files on disk"
            )
        total_bytes = sum(path.stat().st_size for path in files)
        if total_bytes != int(manifest["total_bytes"]):
            problems.append(
                f"Fixture total bytes mismatch: {manifest_path} says {manifest['total_bytes']}, "
                f"found {total_bytes} bytes on disk"
            )
    if importlib.util.find_spec("sklearn") is None:
        problems.append("Missing Python package: scikit-learn (pip install -r requirements-ml.txt)")
    return problems


def _features(overrides: dict[str, float | int | None]) -> dict[str, float | int | None]:
    """Returns the all-quiet feature dict with the given case's overrides applied."""
    features = dict(_QUIET_FEATURES)
    features.update(overrides)
    return features


def _fired_rule_ids(decision: RulesDecision) -> set[str]:
    """Returns the rule ids that fired, ignoring their reason codes."""
    return {rule_id for rule_id, _reason in decision.fired}


async def check_17_1_rules_fire_on_constructed_cases(workspace: Any) -> CheckOutcome:
    """
    WHAT: Runs the committed rules engine and config on the four constructed mule cases.

    WHY: The rules table is the baseline every model must beat, so each rule must fire on the
    case the typology was built for, with the reason code the decision path reports.

    Returns:
        A CheckOutcome whose key number states that all five rules fired.
    """
    config = load_rules_config()
    fired_reason_by_rule: dict[str, str] = {}
    for case_name, overrides, expected_rule_ids in _CONSTRUCTED_MULE_CASES:
        decision = evaluate_rules(_features(overrides), config)
        fired_rule_ids = _fired_rule_ids(decision)
        assert fired_rule_ids == set(expected_rule_ids), (
            f"{case_name} fired {sorted(fired_rule_ids)}, expected {sorted(expected_rule_ids)}"
        )
        for rule_id, reason_code in decision.fired:
            fired_reason_by_rule[rule_id] = reason_code

    assert fired_reason_by_rule == _EXPECTED_REASON_CODES, (
        f"Reason codes fired were {fired_reason_by_rule}, expected {_EXPECTED_REASON_CODES}"
    )
    return CheckOutcome(
        key_number="4 constructed cases fire all 5 rules (RB1..RB5)",
        evidence={
            "cases": [case_name for case_name, _overrides, _expected in _CONSTRUCTED_MULE_CASES],
            "rules_fired": fired_reason_by_rule,
        },
    )


async def check_17_2_tea_stall_and_gig_worker(workspace: Any) -> CheckOutcome:
    """
    WHAT: Runs the rules on a busy tea stall and a gig worker, both genuine.

    WHY: A false trigger that is hidden looks like a perfect rule; the tea stall's rule_3 fire is
    counted here on purpose, and the gig worker shows the money rules stay quiet on a real person.

    Returns:
        A CheckOutcome whose key number states one counted false trigger and zero on the gig worker.
    """
    config = load_rules_config()
    tea_stall = evaluate_rules(_features(_TEA_STALL_FEATURES), config)
    gig_worker = evaluate_rules(_features(_GIG_WORKER_FEATURES), config)

    assert _fired_rule_ids(tea_stall) == {"rule_3"}, f"tea stall fired {sorted(_fired_rule_ids(tea_stall))}"
    assert tea_stall.fired == (("rule_3", "RB3_MANY_PAYERS_ONE_DAY"),), f"tea stall fired {tea_stall.fired}"
    assert gig_worker.fired == (), f"gig worker fired {gig_worker.fired}"

    return CheckOutcome(
        key_number="tea stall: 1 false trigger (RB3); gig worker: 0",
        evidence={
            "tea_stall_fired": list(tea_stall.fired),
            "tea_stall_known_false_trigger_count": 1,
            "gig_worker_fired": list(gig_worker.fired),
            "gig_worker_false_trigger_count": 0,
        },
    )


def _account_event_record(run_id: str, event: BankFeedAccountEvent) -> AccountEventRecord:
    """
    WHAT: Builds the Level 15 account-event record for one bank-feed account event.

    WHY: This mapping is the established in-repo path, copied from
    tests/unit/world/test_m1_on_world.py:95-107; the bank feed stamps the event time and the
    record carries an arrival 60 seconds later, as every other consumer does.

    Args:
        run_id: The world run id the records belong to.
        event: One account event from the bank feed.

    Returns:
        The record features.account_updater stores for that event.
    """
    return AccountEventRecord(
        run_id=run_id,
        bank_id=event.bank_id,
        account_id=event.account_id,
        event_id=event.event_id,
        kind=event.kind,
        event_time=event.event_time,
        arrival_time=event.event_time + timedelta(seconds=60),
        details=event.details,
    )


def _device_events_by_account(
    feed: Any,
) -> dict[tuple[str, str], list[BankFeedAccountEvent]]:
    """Groups each account's device events (OS update, phone change, reinstall, SIM swap) by bank and account."""
    grouped: dict[tuple[str, str], list[BankFeedAccountEvent]] = defaultdict(list)
    for events in feed.account_events_by_bank.values():
        for event in events:
            if event.kind in _DEVICE_KINDS:
                grouped[(event.bank_id, event.account_id)].append(event)
    return grouped


async def check_17_3_os_update_wave(workspace: Any) -> CheckOutcome:
    """
    WHAT: Runs the OS-update-wave stress scenario and measures new_device_r5 from the real
    account events, then runs the money rule on those real features.

    WHY: An OS update changes the phone's software, not the phone, so it must never look like a
    new device; a real phone change must raise the flag, and rule_5 needs money movement on top.

    Returns:
        A CheckOutcome whose key number states how many OS-update-only accounts stayed quiet and
        how many phone-change accounts were flagged.
    """
    layered_configs, _changes = apply_layers(read_world_configs(), [read_layer(OS_WAVE_LAYER_PATH)])
    output = run_interactive_world(
        FastRunSettings(seed=101, n_people=60, n_shops=12, n_days=61, range_name="training"),
        configs=layered_configs,
    )
    feed = project_run(output)
    settings = load_account_features_settings()
    client = fakeredis.FakeRedis()
    for events in feed.account_events_by_bank.values():
        for event in events:
            apply_account_event(client, _account_event_record(output.result.run_id, event), settings)

    by_account = _device_events_by_account(feed)
    os_only_accounts = [
        key
        for key, events in by_account.items()
        if any(event.kind == "os_update" for event in events) and all(event.kind == "os_update" for event in events)
    ]
    phone_change_accounts = [
        key for key, events in by_account.items() if any(event.kind == _PHONE_CHANGE_KIND for event in events)
    ]
    if not os_only_accounts or not phone_change_accounts:
        raise AssertionError(
            f"os updates: {len(os_only_accounts)} quiet; phone changes: {len(phone_change_accounts)} flagged"
        )

    config = load_rules_config()
    quiet_accounts: list[tuple[str, str]] = []
    for bank_id, account_id in os_only_accounts:
        latest_update = max(
            event.event_time for event in by_account[(bank_id, account_id)] if event.kind == "os_update"
        )
        decision_time = latest_update + timedelta(hours=1)
        features: dict[str, Any] = dict(
            compute_account_features(
                client,
                output.result.run_id,
                bank_id,
                account_id,
                decision_time,
                decision_time,
                settings,
                groups=["control"],
            )
        )
        assert features["new_device_r5"] == 0, f"OS update alone flagged new_device_r5 for {account_id}"
        assert "rule_5" not in _fired_rule_ids(evaluate_rules(features, config)), (
            f"rule_5 fired on an OS-update-only account {account_id}"
        )
        quiet_accounts.append((bank_id, account_id))

    flagged_accounts: list[tuple[str, str]] = []
    for bank_id, account_id in phone_change_accounts:
        latest_change = max(
            event.event_time for event in by_account[(bank_id, account_id)] if event.kind == _PHONE_CHANGE_KIND
        )
        decision_time = latest_change + timedelta(hours=1)
        features = compute_account_features(
            client,
            output.result.run_id,
            bank_id,
            account_id,
            decision_time,
            decision_time,
            settings,
            groups=["control"],
        )
        assert features["new_device_r5"] == 1, f"phone change did not flag new_device_r5 for {account_id}"
        flagged_accounts.append((bank_id, account_id))

    # The stress world moves no money through these accounts, so the money condition is supplied
    # to show the device half of rule_5 fires.
    bank_id, account_id = flagged_accounts[0]
    latest_change = max(
        event.event_time for event in by_account[(bank_id, account_id)] if event.kind == _PHONE_CHANGE_KIND
    )
    decision_time = latest_change + timedelta(hours=1)
    money_features: dict[str, Any] = dict(
        compute_account_features(
            client,
            output.result.run_id,
            bank_id,
            account_id,
            decision_time,
            decision_time,
            settings,
            groups=["control"],
        )
    )
    money_features["pass_through_ratio_24h"] = 0.9
    decision = evaluate_rules(money_features, config)
    assert ("rule_5", "RB5_NEW_DEVICE_MONEY_MOVED") in decision.fired, (
        f"rule_5 did not fire on a phone-change account with money movement: {decision.fired}"
    )

    return CheckOutcome(
        key_number=(f"os updates: {len(quiet_accounts)} quiet; phone changes: {len(flagged_accounts)} flagged"),
        evidence={
            "world_run_id": output.result.run_id,
            "os_update_events": sum(
                1 for events in by_account.values() for event in events if event.kind == "os_update"
            ),
            "os_update_only_accounts": len(os_only_accounts),
            "os_update_only_flagged": 0,
            "phone_change_accounts": len(phone_change_accounts),
            "phone_change_flagged": len(flagged_accounts),
            "rule_5_with_money_fired": True,
        },
    )


async def check_17_6_untuned_and_split_guards(workspace: Any) -> CheckOutcome:
    """
    WHAT: Feeds the committed all-untuned rules config to the evaluation guard and a test split to
    the anomaly detector's training function.

    WHY: An untuned threshold reported as a result would present a guess as a measurement, and a
    detector fitted on test rows would have seen the data it is judged on. Both refusals must hold,
    so the check passes only if both calls raise.

    Returns:
        A CheckOutcome whose key number states that both guards refused their input.
    """
    untuned_refused = False
    try:
        require_tuned(load_rules_config())
    except ValueError as error:
        assert str(error) == "thresholds not tuned on validation", f"unexpected refusal message: {error}"
        untuned_refused = True
    assert untuned_refused, "require_tuned accepted the committed all-untuned rules config"

    test_split_refused = False
    try:
        train_anomaly_detector([], "test", {})
    except ValueError as error:
        assert str(error).startswith("Split 'test' is not permitted"), f"unexpected refusal message: {error}"
        test_split_refused = True
    assert test_split_refused, "train_anomaly_detector accepted the test split"

    return CheckOutcome(
        key_number="untuned rules refused; test-split training refused",
        evidence={
            "require_tuned_raised": untuned_refused,
            "train_anomaly_detector_test_split_raised": test_split_refused,
        },
    )
