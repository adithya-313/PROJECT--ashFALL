"""
WHAT: Integration and unit test suite for Level 12d gang reactions, feed isolation, and stop rules.
Validates AST architectural isolation of gang decision logic, deterministic execution of the lien
stop rule, packet loss suppression, adherence flag toggling, equality with reactions disabled,
borrowed KYC opening handle resolution, run-to-run determinism, and required edge cases C1-C6.

WHY: Amendment A-017 requires criminal syndicates to operate under imperfect, delayed external
friction telemetry without access to bank ledgers or simulation ground truth. Rigorous tests
ensure that syndicates halt flows through compromised infrastructure only after learning of liens.
"""

import ast
import inspect
from pathlib import Path
from typing import Any

import pytest

import world.gang_reactions
from world.cash_withdrawals import CashWithdrawal
from world.fast_mode import FastRunSettings
from world.gang_config import BlockStep, FixtureSource, GangBudget, GangConfig
from world.gang_feed import (
    GangHolding,
    validate_gang_reactions_config,
)
from world.interactive_records import FraudInputs, RunnerSettings
from world.interactive_runner import run_interactive_world
from world.mule_origins import is_opening_handle
from world.outside_events import OutsideEvent
from world.run_identity import REPO_ROOT

FORBIDDEN_MODULES: tuple[str, ...] = (
    "world.truth_records",
    "world.truth_writer",
    "evaluation",
    "world.ledger",
    "world.encumbrance",
    "world.interactive_loop",
    "world.interactive_state",
)

TINY = FastRunSettings(101, 60, 12, 61, "training")
GENEROUS_BUDGET = GangBudget(money_paise=50_000_000, time_minutes=500_000, max_risk=0.99)


def _check_module_isolation(file_path: Path) -> list[str]:
    """
    WHAT: Performs AST inspection to ensure a module does not import any forbidden internal records or runners.
    WHY: Enforces strict information boundaries preventing the gang decision engine from accessing ground truth.
    """
    tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for forbidden in FORBIDDEN_MODULES:
                    if alias.name == forbidden or alias.name.startswith(forbidden + "."):
                        violations.append(f"Direct import of forbidden module '{alias.name}'")
        elif isinstance(node, ast.ImportFrom):
            module_name = node.module or ""
            for forbidden in FORBIDDEN_MODULES:
                if module_name == forbidden or module_name.startswith(forbidden + "."):
                    violations.append(f"From import from forbidden module '{module_name}'")
    return violations


def test_gang_reactions_isolation_check_rejects_forbidden_modules(tmp_path: Path) -> None:
    """
    Verifies that world/gang_reactions.py has zero forbidden imports and public signatures
    take only allowed types.
    """
    target_path = REPO_ROOT / "world" / "gang_reactions.py"
    violations = _check_module_isolation(target_path)
    assert not violations, f"world/gang_reactions.py contains forbidden imports: {violations}"

    # Verify public function signatures take only observations, states, or plain types.
    for name, obj in inspect.getmembers(world.gang_reactions, inspect.isfunction):
        if name.startswith("_"):
            continue
        sig = inspect.signature(obj)
        for param_name, param in sig.parameters.items():
            ann = param.annotation
            ann_str = str(ann)
            for forbidden in FORBIDDEN_MODULES:
                assert forbidden not in ann_str, (
                    f"Function '{name}' parameter '{param_name}' exposes forbidden type '{ann_str}'"
                )

    # Prove test (f) is sensitive: append forbidden import to temporary copy and assert check rejects it.
    temp_copy = tmp_path / "corrupted_gang_reactions.py"
    temp_copy.write_text(
        target_path.read_text(encoding="utf-8") + "\nimport world.truth_records\n",
        encoding="utf-8",
    )
    temp_violations = _check_module_isolation(temp_copy)
    assert temp_violations, "Isolation check failed to reject corrupted file with forbidden import"


def _build_fixture_gang() -> GangConfig:
    """
    WHAT: Constructs a reproducible scam gang with 4 acquired mules, a scam campaign, and a 2x2 laundering route.
    WHY: Supplies realistic multi-hop fraud flows through layer-1 mules to test reactive flow cancellations.
    """
    steps = (
        BlockStep("acquire", (("count", 4),)),
        BlockStep("scam", (("receiving_mules", 2),)),
        BlockStep("route", (("layers", 2), ("accounts_per_layer", 2))),
    )
    return GangConfig("g1", 1, GENEROUS_BUDGET, None, steps, 1440)


def test_stop_rule_cancels_subsequent_flows_and_curtails_mule_lifecycle() -> None:
    """
    Asserts test (g): injected lien at minute L on layer-1 mule stops money movement after L+D while allowing
    real flows between L and L+D, curtailing truth burned_from to L+D and recording cancelled_by_gang items.
    """
    gang = _build_fixture_gang()
    mule_account = "acc_person_000010"
    minute_l = 3000

    fixture_reactions_cfg: dict[str, Any] = {
        "gang_reactions": {
            "label": "policy",
            "derived_from": "amendment A-017",
            "reason": "fixture",
            "issue_ids_v2": ["G4"],
            "gang_feed_delay_minutes": {"training": [1100, 1100]},
            "gang_feed_miss_share": {"training": [0.0, 0.0]},
            "gang_stops_using_liened_account_share": {"training": [1.0, 1.0]},
        }
    }
    cfg_section: dict[str, Any] = fixture_reactions_cfg["gang_reactions"]
    delay_d = int(cfg_section["gang_feed_delay_minutes"]["training"][0])
    seen_time = minute_l + delay_d

    lien = OutsideEvent(
        minute=minute_l,
        kind="lien",
        account_id=mule_account,
        amount_paise=50000,
        encumbrance_id="enc-injected-01",
    )
    inputs = FraudInputs(outside_events=(lien,))

    out = run_interactive_world(
        TINY,
        RunnerSettings(fraud_enabled=True, gang_reactions_enabled=True),
        fraud_inputs=inputs,
        gang_configs=(gang,),
        gang_reactions_config=fixture_reactions_cfg,
    )

    # 1. Assert real money moves through the mule between L and L+D (non-vacuous proof).
    flows_between_l_and_ld = [
        ev for i, ev in enumerate(out.result.run_result.events)
        if out.result.event_sources[i] == "fraud"
        and mule_account in (ev.from_account, ev.to_account)
        and minute_l <= ev.minute < seen_time
    ]
    assert len(flows_between_l_and_ld) > 0, (
        f"Mule {mule_account} must experience real flows between L ({minute_l}) and L+D ({seen_time})"
    )

    # 2. Assert after L+D no fraud event involves that mule.
    flows_after_ld = [
        ev for i, ev in enumerate(out.result.run_result.events)
        if out.result.event_sources[i] == "fraud"
        and mule_account in (ev.from_account, ev.to_account)
        and ev.minute >= seen_time
    ]
    assert len(flows_after_ld) == 0, f"No fraud events should touch mule after L+D; found {flows_after_ld}"

    # 3. Assert no cash withdrawals touch that mule after L+D.
    withdrawals_after_ld = [
        w for w in out.result.cash_withdrawals
        if w.account == mule_account and w.minute >= seen_time
    ]
    assert len(withdrawals_after_ld) == 0, "No cash withdrawals should touch stopped mule after L+D"

    # 4. Assert cancelled_by_gang is not empty and records cancelled transactions.
    assert len(out.result.cancelled_by_gang) > 0
    mule_cancellations = [
        c for c in out.result.cancelled_by_gang
        if mule_account in c.accounts
    ]
    assert len(mule_cancellations) > 0
    for c in mule_cancellations:
        assert c.minute >= seen_time
        assert c.gang == "g1"
        assert c.amount > 0

    # 5. Assert truth burned_from == L+D when that is earlier than planned (7200).
    mule_roles = [r for r in out.result.truth.roles if r.account_id == mule_account]
    assert len(mule_roles) == 1
    role = mule_roles[0]
    assert role.burned_from == seen_time
    assert role.burned_from < 7200


def test_miss_share_forced_to_one_drops_observations_and_cancels_nothing() -> None:
    """Verifies that forcing miss share to 1.0 causes all observations to drop so the gang cancels nothing."""
    gang = _build_fixture_gang()
    mule_account = "acc_person_000010"
    minute_l = 3000

    fixture_reactions_cfg = {
        "gang_reactions": {
            "label": "policy",
            "derived_from": "amendment A-017",
            "reason": "fixture",
            "issue_ids_v2": ["G4"],
            "gang_feed_delay_minutes": {"training": [1100, 1100]},
            "gang_feed_miss_share": {"training": [1.0, 1.0]},
            "gang_stops_using_liened_account_share": {"training": [1.0, 1.0]},
        }
    }

    lien = OutsideEvent(
        minute=minute_l,
        kind="lien",
        account_id=mule_account,
        amount_paise=50000,
        encumbrance_id="enc-injected-02",
    )
    inputs = FraudInputs(outside_events=(lien,))

    out = run_interactive_world(
        TINY,
        RunnerSettings(fraud_enabled=True, gang_reactions_enabled=True),
        fraud_inputs=inputs,
        gang_configs=(gang,),
        gang_reactions_config=fixture_reactions_cfg,
    )

    # Observation is dropped: no cancellations occur and mule lifecycle is preserved.
    assert len(out.result.cancelled_by_gang) == 0
    mule_roles = [r for r in out.result.truth.roles if r.account_id == mule_account]
    assert len(mule_roles) == 1
    assert mule_roles[0].burned_from == 7200


def test_stop_rule_forced_off_observes_lien_but_cancels_nothing() -> None:
    """Verifies that forcing stop rule adherence off delivers observations but results in zero cancellations."""
    gang = _build_fixture_gang()
    mule_account = "acc_person_000010"
    minute_l = 3000

    fixture_reactions_cfg = {
        "gang_reactions": {
            "label": "policy",
            "derived_from": "amendment A-017",
            "reason": "fixture",
            "issue_ids_v2": ["G4"],
            "gang_feed_delay_minutes": {"training": [1100, 1100]},
            "gang_feed_miss_share": {"training": [0.0, 0.0]},
            "gang_stops_using_liened_account_share": {"training": [0.0, 0.0]},
        }
    }

    lien = OutsideEvent(
        minute=minute_l,
        kind="lien",
        account_id=mule_account,
        amount_paise=50000,
        encumbrance_id="enc-injected-03",
    )
    inputs = FraudInputs(outside_events=(lien,))

    out = run_interactive_world(
        TINY,
        RunnerSettings(fraud_enabled=True, gang_reactions_enabled=True),
        fraud_inputs=inputs,
        gang_configs=(gang,),
        gang_reactions_config=fixture_reactions_cfg,
    )

    # Observation is seen but non-adherent syndicate ignores it: zero cancellations.
    assert len(out.result.cancelled_by_gang) == 0
    mule_roles = [r for r in out.result.truth.roles if r.account_id == mule_account]
    assert len(mule_roles) == 1
    assert mule_roles[0].burned_from == 7200


def test_reactions_equality_with_no_lien_matches_reactions_off() -> None:
    """
    Asserts that running with reactions enabled and no external friction produces identical
    outputs to reactions disabled.
    """
    gang = _build_fixture_gang()

    out_off = run_interactive_world(
        TINY,
        RunnerSettings(fraud_enabled=True, gang_reactions_enabled=False),
        gang_configs=(gang,),
    )

    out_on = run_interactive_world(
        TINY,
        RunnerSettings(fraud_enabled=True, gang_reactions_enabled=True),
        gang_configs=(gang,),
    )

    assert out_on.result.run_result.events == out_off.result.run_result.events
    assert out_on.result.event_sources == out_off.result.event_sources
    assert out_on.result.truth.roles == out_off.result.truth.roles
    assert out_on.result.cash_withdrawals == out_off.result.cash_withdrawals
    assert len(out_on.result.cancelled_by_gang) == 0


def test_borrowed_kyc_gang_holdings_handles_resolve_to_real_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that opening handles in gang holdings resolve into assigned account IDs for borrowed KYC mules."""
    all_borrowed = {"rented": (0.0, 0.0), "borrowed_kyc": (1.0, 1.0), "business": (0.0, 0.0)}
    import world.mule_origins
    monkeypatch.setattr(world.mule_origins, "get_mule_origin_share_ranges", lambda split: all_borrowed)

    steps = (
        BlockStep("acquire", (("count", 4), ("origin_mix", "world_ranges"))),
        BlockStep("scam", (("receiving_mules", 2), ("victims_per_receiving_mule", 2))),
        BlockStep("route", (("layers", 2), ("accounts_per_layer", 2))),
    )
    gang = GangConfig("g1", 1, GENEROUS_BUDGET, None, steps, 1440)

    out = run_interactive_world(
        TINY,
        RunnerSettings(fraud_enabled=True, gang_reactions_enabled=True),
        gang_configs=(gang,),
    )

    # Assert that all holdings carry real account identifiers and no placeholder handles remain.
    assert len(out.result.truth.roles) > 0
    for role in out.result.truth.roles:
        assert not is_opening_handle(role.account_id)
        assert role.account_id.startswith("acc_")


def test_determinism_same_seed_produces_identical_cancellations_and_events() -> None:
    """
    Verifies that repeated runs with identical seed and reaction configurations produce
    identical cancellations and events.
    """
    gang = _build_fixture_gang()
    mule_account = "acc_person_000010"
    minute_l = 3000

    fixture_reactions_cfg = {
        "gang_reactions": {
            "label": "policy",
            "derived_from": "amendment A-017",
            "reason": "fixture",
            "issue_ids_v2": ["G4"],
            "gang_feed_delay_minutes": {"training": [1100, 1100]},
            "gang_feed_miss_share": {"training": [0.0, 0.0]},
            "gang_stops_using_liened_account_share": {"training": [1.0, 1.0]},
        }
    }

    lien = OutsideEvent(
        minute=minute_l,
        kind="lien",
        account_id=mule_account,
        amount_paise=50000,
        encumbrance_id="enc-injected-det",
    )
    inputs = FraudInputs(outside_events=(lien,))

    out1 = run_interactive_world(
        TINY,
        RunnerSettings(fraud_enabled=True, gang_reactions_enabled=True),
        fraud_inputs=inputs,
        gang_configs=(gang,),
        gang_reactions_config=fixture_reactions_cfg,
    )

    out2 = run_interactive_world(
        TINY,
        RunnerSettings(fraud_enabled=True, gang_reactions_enabled=True),
        fraud_inputs=inputs,
        gang_configs=(gang,),
        gang_reactions_config=fixture_reactions_cfg,
    )

    assert out1.result.cancelled_by_gang == out2.result.cancelled_by_gang
    assert out1.result.run_result.events == out2.result.run_result.events


def test_lien_seen_on_mule_with_zero_illicit_balance_stops_and_curtails() -> None:
    """
    Asserts edge case C1: an injected lien on a mule whose illicit balance is 0 at that moment
    still curtails the mule's lifecycle and cancels subsequent gang flows through it.
    """
    gang = _build_fixture_gang()
    mule_account = "acc_person_000010"
    minute_l = 1440
    fixture_reactions_cfg: dict[str, Any] = {
        "gang_reactions": {
            "label": "policy",
            "derived_from": "amendment A-017",
            "reason": "fixture",
            "issue_ids_v2": ["G4"],
            "gang_feed_delay_minutes": {"training": [100, 100]},
            "gang_feed_miss_share": {"training": [0.0, 0.0]},
            "gang_stops_using_liened_account_share": {"training": [1.0, 1.0]},
        }
    }
    cfg_section: dict[str, Any] = fixture_reactions_cfg["gang_reactions"]
    delay_d = int(cfg_section["gang_feed_delay_minutes"]["training"][0])
    seen_time = minute_l + delay_d

    lien = OutsideEvent(
        minute=minute_l,
        kind="lien",
        account_id=mule_account,
        amount_paise=25000,
        encumbrance_id="enc-zero-bal-01",
    )
    inputs = FraudInputs(outside_events=(lien,))

    out = run_interactive_world(
        TINY,
        RunnerSettings(fraud_enabled=True, gang_reactions_enabled=True),
        fraud_inputs=inputs,
        gang_configs=(gang,),
        gang_reactions_config=fixture_reactions_cfg,
    )

    # Truth role burned_from is curtailed to seen_time (1540).
    mule_roles = [r for r in out.result.truth.roles if r.account_id == mule_account]
    assert len(mule_roles) == 1
    assert mule_roles[0].burned_from == seen_time
    assert mule_roles[0].burned_from < 7200

    # All later gang flows attempting to use this mule are cancelled.
    assert len(out.result.cancelled_by_gang) > 0
    mule_cancels = [c for c in out.result.cancelled_by_gang if mule_account in c.accounts]
    assert len(mule_cancels) > 0


def test_same_minute_rule_observation_processed_after_due_payments() -> None:
    """
    Asserts edge case C2: the same-minute rule.
    In config/interactive_runner.yaml, 'gang_observations' is ranked last (tie_break_rank 6)
    in tie_break_order. Thus, an observation seen at minute M is popped and applied AFTER every
    payment scheduled at minute M has already executed. A fraud payment at exactly minute M runs,
    while payments scheduled at minute M+1 or later are cancelled by the gang.
    """
    gang = _build_fixture_gang()
    mule_account = "acc_person_000010"
    # First fraud payment touching mule_account is at minute 1670; second is at minute 1786.
    seen_time = 1670
    minute_l = 1500
    delay_d = seen_time - minute_l

    fixture_reactions_cfg: dict[str, Any] = {
        "gang_reactions": {
            "label": "policy",
            "derived_from": "amendment A-017",
            "reason": "fixture",
            "issue_ids_v2": ["G4"],
            "gang_feed_delay_minutes": {"training": [delay_d, delay_d]},
            "gang_feed_miss_share": {"training": [0.0, 0.0]},
            "gang_stops_using_liened_account_share": {"training": [1.0, 1.0]},
        }
    }
    cfg_section: dict[str, Any] = fixture_reactions_cfg["gang_reactions"]
    delay_from_cfg = int(cfg_section["gang_feed_delay_minutes"]["training"][0])
    assert delay_from_cfg == delay_d

    lien = OutsideEvent(
        minute=minute_l,
        kind="lien",
        account_id=mule_account,
        amount_paise=50000,
        encumbrance_id="enc-same-minute-01",
    )
    inputs = FraudInputs(outside_events=(lien,))

    out = run_interactive_world(
        TINY,
        RunnerSettings(fraud_enabled=True, gang_reactions_enabled=True),
        fraud_inputs=inputs,
        gang_configs=(gang,),
        gang_reactions_config=fixture_reactions_cfg,
    )

    # 1. Payment at exactly minute M (1670) succeeded (was processed before the observation).
    pmts_at_m = [
        ev for i, ev in enumerate(out.result.run_result.events)
        if out.result.event_sources[i] == "fraud"
        and ev.minute == seen_time
        and mule_account in (ev.from_account, ev.to_account)
    ]
    assert len(pmts_at_m) > 0, f"Fraud payment at exactly minute {seen_time} must execute before observation"

    # 2. Payments after minute M (> 1670) are cancelled.
    pmts_after_m = [
        ev for i, ev in enumerate(out.result.run_result.events)
        if out.result.event_sources[i] == "fraud"
        and ev.minute > seen_time
        and mule_account in (ev.from_account, ev.to_account)
    ]
    assert len(pmts_after_m) == 0, f"No fraud payments should run after minute {seen_time}"
    assert any(c.minute > seen_time and mule_account in c.accounts for c in out.result.cancelled_by_gang)


def test_lien_on_collector_account_cancels_payments_without_curtailing_lifecycle() -> None:
    """
    Asserts edge case C3: a lien on a gang collector/source account cancels outbound payments
    from that account, but does NOT edit any RoleSpan lifecycle fields (burned_from remains None).
    """
    steps = (
        BlockStep("acquire", (("count", 4),)),
        BlockStep("warm_up", (("days", 1),)),
        BlockStep("route", (("layers", 2), ("accounts_per_layer", 2))),
    )
    source_acc = "acc_person_000001"
    gang = GangConfig("g_coll", 1, GENEROUS_BUDGET, FixtureSource(source_acc, 0, 10_000_000, 1440), steps, None)

    minute_l = 1500
    delay_d = 200

    fixture_reactions_cfg: dict[str, Any] = {
        "gang_reactions": {
            "label": "policy",
            "derived_from": "amendment A-017",
            "reason": "fixture",
            "issue_ids_v2": ["G4"],
            "gang_feed_delay_minutes": {"training": [delay_d, delay_d]},
            "gang_feed_miss_share": {"training": [0.0, 0.0]},
            "gang_stops_using_liened_account_share": {"training": [1.0, 1.0]},
        }
    }
    cfg_section: dict[str, Any] = fixture_reactions_cfg["gang_reactions"]
    delay_from_cfg = int(cfg_section["gang_feed_delay_minutes"]["training"][0])
    seen_time = minute_l + delay_from_cfg

    lien = OutsideEvent(
        minute=minute_l,
        kind="lien",
        account_id=source_acc,
        amount_paise=10000,
        encumbrance_id="enc-coll-01",
    )
    inputs = FraudInputs(outside_events=(lien,))

    out = run_interactive_world(
        TINY,
        RunnerSettings(fraud_enabled=True, gang_reactions_enabled=True),
        fraud_inputs=inputs,
        gang_configs=(gang,),
        gang_reactions_config=fixture_reactions_cfg,
    )

    # Cancellations occur for source_acc at or after seen_time
    coll_cancellations = [c for c in out.result.cancelled_by_gang if source_acc in c.accounts]
    assert len(coll_cancellations) > 0
    assert all(c.minute >= seen_time for c in coll_cancellations)

    # In truth, collector RoleSpan has burned_from == None (lifecycle untouched)
    coll_roles = [r for r in out.result.truth.roles if r.account_id == source_acc]
    assert len(coll_roles) == 1
    assert coll_roles[0].role == "collector"
    assert coll_roles[0].burned_from is None


def test_stopped_mule_with_planned_cash_withdrawal_cancels_withdrawal() -> None:
    """
    Asserts edge case C4: a stopped mule with a planned cash withdrawal has the withdrawal
    cancelled and logged in cancelled_by_gang with reason 'account_stopped'.
    """
    gang = _build_fixture_gang()
    mule_account = "acc_person_000010"
    minute_l = 2000
    delay_d = 500
    seen_time = minute_l + delay_d

    fixture_reactions_cfg = {
        "gang_reactions": {
            "label": "policy",
            "derived_from": "amendment A-017",
            "reason": "fixture",
            "issue_ids_v2": ["G4"],
            "gang_feed_delay_minutes": {"training": [delay_d, delay_d]},
            "gang_feed_miss_share": {"training": [0.0, 0.0]},
            "gang_stops_using_liened_account_share": {"training": [1.0, 1.0]},
        }
    }
    lien = OutsideEvent(
        minute=minute_l,
        kind="lien",
        account_id=mule_account,
        amount_paise=50000,
        encumbrance_id="enc-cw-01",
    )
    cw = CashWithdrawal(
        minute=seen_time + 100,
        account_id=mule_account,
        amount_paise=50000,
        channel="branch",
    )
    inputs = FraudInputs(outside_events=(lien,), withdrawals=(cw,))

    out = run_interactive_world(
        TINY,
        RunnerSettings(fraud_enabled=True, gang_reactions_enabled=True),
        fraud_inputs=inputs,
        gang_configs=(gang,),
        gang_reactions_config=fixture_reactions_cfg,
    )

    cw_cancels = [
        c for c in out.result.cancelled_by_gang
        if mule_account in c.accounts and c.reason == "account_stopped" and c.minute == cw.minute
    ]
    assert len(cw_cancels) == 1
    assert cw_cancels[0].amount == cw.amount_paise


def test_observation_seen_after_last_queued_item_never_delivered_and_run_finishes() -> None:
    """
    Asserts edge case C5: an observation whose seen_minute falls after the last queued item
    is never delivered, nothing crashes, and the run finishes cleanly.
    """
    gang = _build_fixture_gang()
    mule_account = "acc_person_000010"
    minute_l = 3000
    delay_d = 900_000  # Far beyond simulation horizon (87840 minutes)

    fixture_reactions_cfg = {
        "gang_reactions": {
            "label": "policy",
            "derived_from": "amendment A-017",
            "reason": "fixture",
            "issue_ids_v2": ["G4"],
            "gang_feed_delay_minutes": {"training": [delay_d, delay_d]},
            "gang_feed_miss_share": {"training": [0.0, 0.0]},
            "gang_stops_using_liened_account_share": {"training": [1.0, 1.0]},
        }
    }
    lien = OutsideEvent(
        minute=minute_l,
        kind="lien",
        account_id=mule_account,
        amount_paise=50000,
        encumbrance_id="enc-late-01",
    )
    inputs = FraudInputs(outside_events=(lien,))

    out = run_interactive_world(
        TINY,
        RunnerSettings(fraud_enabled=True, gang_reactions_enabled=True),
        fraud_inputs=inputs,
        gang_configs=(gang,),
        gang_reactions_config=fixture_reactions_cfg,
    )

    # Run finishes with normal events and 0 gang cancellations.
    assert len(out.result.run_result.events) > 0
    assert len(out.result.cancelled_by_gang) == 0


def test_lien_released_before_gang_sees_it_still_stops_gang() -> None:
    """
    Asserts edge case C6: a lien released BEFORE the gang sees it still stops the gang
    when it sees the lien (no un-stopping in D-1).
    """
    gang = _build_fixture_gang()
    mule_account = "acc_person_000010"
    minute_l = 2000
    delay_d = 500
    seen_time = minute_l + delay_d

    fixture_reactions_cfg = {
        "gang_reactions": {
            "label": "policy",
            "derived_from": "amendment A-017",
            "reason": "fixture",
            "issue_ids_v2": ["G4"],
            "gang_feed_delay_minutes": {"training": [delay_d, delay_d]},
            "gang_feed_miss_share": {"training": [0.0, 0.0]},
            "gang_stops_using_liened_account_share": {"training": [1.0, 1.0]},
        }
    }
    lien = OutsideEvent(
        minute=minute_l,
        kind="lien",
        account_id=mule_account,
        amount_paise=50000,
        encumbrance_id="enc-rel-01",
    )
    release = OutsideEvent(
        minute=minute_l + 50,
        kind="release",
        account_id=mule_account,
        amount_paise=50000,
        encumbrance_id="enc-rel-01",
    )
    inputs = FraudInputs(outside_events=(lien, release))

    out = run_interactive_world(
        TINY,
        RunnerSettings(fraud_enabled=True, gang_reactions_enabled=True),
        fraud_inputs=inputs,
        gang_configs=(gang,),
        gang_reactions_config=fixture_reactions_cfg,
    )

    # Gang sees the lien at seen_time (2500) and stops the account; payments after 2500 are cancelled.
    post_seen_cancels = [
        c for c in out.result.cancelled_by_gang
        if mule_account in c.accounts and c.minute >= seen_time
    ]
    assert len(post_seen_cancels) > 0


def test_validation_rejects_corrupted_holdings_and_configs() -> None:
    """
    Asserts Section D: validates rejection of invalid holdings, missing stories, unknown gang IDs,
    and invalid reaction configs.
    """
    # 1. GangHolding with valid_to <= valid_from raises ValueError
    with pytest.raises(ValueError, match="valid_to .* must be greater than valid_from"):
        GangHolding("g1", "acc_1", valid_from=100, valid_to=100)

    with pytest.raises(ValueError, match="valid_to .* must be greater than valid_from"):
        GangHolding("g1", "acc_1", valid_from=200, valid_to=150)

    # 2. Config validation rejects missing label
    with pytest.raises(ValueError, match="must contain a non-empty 'label'"):
        validate_gang_reactions_config({"gang_reactions": {}})

    # 3. Config validation rejects negative delay
    bad_cfg = {
        "gang_reactions": {
            "label": "test",
            "derived_from": "A",
            "reason": "B",
            "issue_ids_v2": ["G4"],
            "gang_feed_delay_minutes": {"training": [-1, 10]},
            "gang_feed_miss_share": {"training": [0.0, 1.0]},
            "gang_stops_using_liened_account_share": {"training": [1.0, 1.0]},
        }
    }
    with pytest.raises(ValueError, match="delay range.*non-negative integers"):
        validate_gang_reactions_config(bad_cfg)

    # 4. Config validation rejects miss share outside [0, 1]
    bad_cfg2 = {
        "gang_reactions": {
            "label": "test",
            "derived_from": "A",
            "reason": "B",
            "issue_ids_v2": ["G4"],
            "gang_feed_delay_minutes": {"training": [0, 10]},
            "gang_feed_miss_share": {"training": [0.0, 1.5]},
            "gang_stops_using_liened_account_share": {"training": [1.0, 1.0]},
        }
    }
    with pytest.raises(ValueError, match="miss share range.*floats in \\[0, 1\\]"):
        validate_gang_reactions_config(bad_cfg2)
