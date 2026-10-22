"""
WHAT: Integration and unit test suite for Level 12d gang reactions, feed isolation, and stop rules.
Validates AST architectural isolation of gang decision logic, deterministic execution of the lien
stop rule, packet loss suppression, adherence flag toggling, equality with reactions disabled,
borrowed KYC opening handle resolution, and run-to-run determinism.

WHY: Amendment A-017 requires criminal syndicates to operate under imperfect, delayed external
friction telemetry without access to bank ledgers or simulation ground truth. Rigorous tests
ensure that syndicates halt flows through compromised infrastructure only after learning of liens.
"""

import ast
import inspect
from pathlib import Path
from typing import Any
import pytest
import numpy as np

import world.gang_reactions
from world.gang_config import BlockStep, GangBudget, GangConfig
from world.gang_feed import GangObservation
from world.gang_reactions import GangReactionState
from world.interactive_records import FraudInputs, RunnerSettings
from world.interactive_runner import run_interactive_world
from world.fast_mode import FastRunSettings
from world.mule_origins import get_mule_origin_share_ranges, is_opening_handle
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
    """Verifies that world/gang_reactions.py has zero forbidden imports and public signatures take only allowed types."""
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
    delay_d = 1100
    seen_time = minute_l + delay_d

    # Labelled injected lien on layer-1 mule at minute L.
    lien = OutsideEvent(
        minute=minute_l,
        kind="lien",
        account_id=mule_account,
        amount_paise=50000,
        encumbrance_id="enc-injected-01",
    )
    inputs = FraudInputs(outside_events=(lien,))

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
    delay_d = 1100

    lien = OutsideEvent(
        minute=minute_l,
        kind="lien",
        account_id=mule_account,
        amount_paise=50000,
        encumbrance_id="enc-injected-02",
    )
    inputs = FraudInputs(outside_events=(lien,))

    fixture_reactions_cfg = {
        "gang_reactions": {
            "label": "policy",
            "derived_from": "amendment A-017",
            "reason": "fixture",
            "issue_ids_v2": ["G4"],
            "gang_feed_delay_minutes": {"training": [delay_d, delay_d]},
            "gang_feed_miss_share": {"training": [1.0, 1.0]},
            "gang_stops_using_liened_account_share": {"training": [1.0, 1.0]},
        }
    }

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
    delay_d = 1100

    lien = OutsideEvent(
        minute=minute_l,
        kind="lien",
        account_id=mule_account,
        amount_paise=50000,
        encumbrance_id="enc-injected-03",
    )
    inputs = FraudInputs(outside_events=(lien,))

    fixture_reactions_cfg = {
        "gang_reactions": {
            "label": "policy",
            "derived_from": "amendment A-017",
            "reason": "fixture",
            "issue_ids_v2": ["G4"],
            "gang_feed_delay_minutes": {"training": [delay_d, delay_d]},
            "gang_feed_miss_share": {"training": [0.0, 0.0]},
            "gang_stops_using_liened_account_share": {"training": [0.0, 0.0]},
        }
    }

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
    """Asserts that running with reactions enabled and no external friction produces identical outputs to reactions disabled."""
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
    """Verifies that repeated runs with identical seed and reaction configurations produce identical cancellations and events."""
    gang = _build_fixture_gang()
    mule_account = "acc_person_000010"
    minute_l = 3000
    delay_d = 1100

    lien = OutsideEvent(
        minute=minute_l,
        kind="lien",
        account_id=mule_account,
        amount_paise=50000,
        encumbrance_id="enc-injected-det",
    )
    inputs = FraudInputs(outside_events=(lien,))

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
