"""
WHAT: Unit and integration tests for the account market resale planner and gang planning integration (Level 12d, G5).
Verifies configuration loading and validation, resale share resolution and drawing, offer filtering, resale planning,
and comprehensive GangPlan identity substitution across all account-bearing dataclass fields.

WHY: Police cases document syndicates acquiring accounts from common suppliers who resell burned kits.
Guarantees deterministic resale behavior, constant random draw streams, and strict non-overlapping temporal role spans.
"""

import hashlib
from dataclasses import asdict
from typing import Any

import numpy as np
import pytest

from core.config_validator import load_yaml
from evaluation.truth import TruthStore
from world.account_market import (
    apply_resale,
    draw_resale_share,
    find_resale_offers,
    load_account_market_config,
    plan_resales,
    resolve_resale_share_range,
    validate_account_market_config,
)
from world.calendar_runner import PlannedPayment
from world.cash_withdrawals import CashWithdrawal
from world.fast_mode import FastRunSettings, event_lines
from world.gang_config import BlockStep, GangBudget, GangConfig
from world.gang_devices import GangDeviceEvent
from world.gang_planner import BlockCost, GangPlan
from world.interactive_records import FraudInputs, IllicitMoneyRule, Resale, RunnerSettings
from world.interactive_runner import interactive_fast_output, run_interactive_world
from world.outside_events import OutsideEvent
from world.run_identity import REPO_ROOT
from world.truth_records import (
    DeviceControlSpan,
    PaymentStory,
    RoleSpan,
    TruthFacts,
    validate_no_overlapping_role_spans,
)


def test_account_market_config_validation_success() -> None:
    """
    WHAT: Verifies that load_account_market_config and validate_account_market_config pass on valid configs.
    WHY: Enforces that policy configurations conform strictly to labelled block requirements.
    """
    cfg = load_account_market_config()
    assert "account_market" in cfg
    assert cfg["account_market"]["label"] == "policy"
    assert "G5" in cfg["account_market"]["issue_ids_v2"]

    fixture_cfg: dict[str, Any] = {
        "account_market": {
            "label": "fixture",
            "derived_from": "amendment A-017",
            "reason": "Test fixture configuration.",
            "issue_ids_v2": ["G5"],
            "share_of_accounts_sold_to_several_gangs": {
                "training": [0.1, 0.4],
                "test": [0.05, 0.5],
            },
        }
    }
    validate_account_market_config(fixture_cfg)


def test_account_market_config_validation_failures() -> None:
    """
    WHAT: Tests that validate_account_market_config raises plain ValueErrors on invalid structures or bounds.
    WHY: Invalid or malformed configuration must be rejected immediately rather than silently defaulting.
    """
    with pytest.raises(ValueError, match="must be a dictionary"):
        validate_account_market_config(["not", "a", "dict"])  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="non-empty 'label'"):
        validate_account_market_config({"account_market": {"label": ""}})

    with pytest.raises(ValueError, match="missing required key"):
        validate_account_market_config({"account_market": {"label": "policy"}})

    bad_bounds_cfg: dict[str, Any] = {
        "account_market": {
            "label": "fixture",
            "derived_from": "amendment A-017",
            "reason": "Invalid bounds fixture.",
            "issue_ids_v2": ["G5"],
            "share_of_accounts_sold_to_several_gangs": {
                "training": [0.5, 0.2],
            },
        }
    }
    with pytest.raises(ValueError, match="lower <= upper"):
        validate_account_market_config(bad_bounds_cfg)

    negative_bounds_cfg: dict[str, Any] = {
        "account_market": {
            "label": "fixture",
            "derived_from": "amendment A-017",
            "reason": "Negative bounds fixture.",
            "issue_ids_v2": ["G5"],
            "share_of_accounts_sold_to_several_gangs": {
                "training": [-0.1, 0.2],
            },
        }
    }
    with pytest.raises(ValueError, match="floats in \\[0, 1\\]"):
        validate_account_market_config(negative_bounds_cfg)


def test_resolve_and_draw_resale_share_always_draws() -> None:
    """
    WHAT: Asserts that draw_resale_share always takes a draw regardless of whether the range is degenerate.
    WHY: Ground rule G5 requires stream position invariance across degenerate and non-degenerate bounds.
    """
    low, high = resolve_resale_share_range("training")
    assert 0.0 <= low <= high <= 1.0

    degen_cfg = {
        "account_market": {
            "share_of_accounts_sold_to_several_gangs": {"training": [0.25, 0.25]}
        }
    }
    non_degen_cfg = {
        "account_market": {
            "share_of_accounts_sold_to_several_gangs": {"training": [0.10, 0.50]}
        }
    }
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)

    val1 = draw_resale_share("training", rng1, config=degen_cfg)
    val2 = draw_resale_share("training", rng2, config=non_degen_cfg)

    assert val1 == 0.25
    assert 0.10 <= val2 <= 0.50
    # Both RNGs must have advanced by exactly 1 draw, so the next draw from each is identical.
    assert rng1.random() == rng2.random()


def test_find_resale_offers_picks_only_rented_burned_open_ended_mules() -> None:
    """
    WHAT: Validates that find_resale_offers filters earlier role spans to only rented, burned, open-ended mules.
    WHY: Excludes unburned accounts, borrowed/business mules, non-mules, and closed spans from resale offers.
    """
    eligible1 = RoleSpan("acc_01", "mule", "gang_1", 1, 100, None, origin="rented", burned_from=300)
    eligible2 = RoleSpan("acc_02", "mule", "gang_1", 2, 100, None, origin="rented", burned_from=400)
    unburned = RoleSpan("acc_03", "mule", "gang_1", 1, 100, None, origin="rented", burned_from=None)
    closed = RoleSpan("acc_04", "mule", "gang_1", 1, 100, 500, origin="rented", burned_from=350)
    borrowed = RoleSpan("acc_05", "mule", "gang_1", 1, 100, None, origin="borrowed_kyc", burned_from=300)
    business = RoleSpan("acc_06", "mule", "gang_1", 1, 100, None, origin="business", burned_from=300)
    collector = RoleSpan("acc_07", "collector", "gang_1", None, 100, None, origin=None, burned_from=300)
    cash_out = RoleSpan("acc_08", "cash_out", "gang_1", None, 100, None, origin=None, burned_from=300)

    spans = [eligible1, unburned, eligible2, closed, borrowed, business, collector, cash_out]
    offers = find_resale_offers(spans)

    assert offers == [eligible1, eligible2]


def test_plan_resales_stream_position_and_offer_constraints() -> None:
    """
    WHAT: Asserts that plan_resales draws exactly once per rented mule and enforces offer burned_from limits.
    WHY: Each rented mule must advance the stream position regardless of offer availability or acceptance.
    """
    offer1 = RoleSpan("acc_old_1", "mule", "gang_1", 1, 100, None, origin="rented", burned_from=500)
    offer2 = RoleSpan("acc_old_2", "mule", "gang_1", 1, 100, None, origin="rented", burned_from=800)
    offers = [offer1, offer2]

    # New gang has 3 rented mules, 1 borrowed mule, and 1 cash_out.
    mule_a = RoleSpan("acc_new_a", "mule", "gang_2", 1, 600, None, origin="rented")
    mule_b = RoleSpan("acc_new_b", "mule", "gang_2", 1, 700, None, origin="rented")
    mule_c = RoleSpan("acc_new_c", "mule", "gang_2", 1, 900, None, origin="rented")
    mule_borrowed = RoleSpan("acc_new_d", "mule", "gang_2", 1, 600, None, origin="borrowed_kyc")
    cash_out = RoleSpan("acc_new_e", "cash_out", "gang_2", None, 600, None, origin=None)

    new_roles = [mule_a, mule_borrowed, mule_b, cash_out, mule_c]

    # With share = 1.0, every draw < share is True.
    rng_test = np.random.default_rng(123)
    rng_control = np.random.default_rng(123)

    resales = plan_resales(new_roles, offers, share=1.0, rng=rng_test)

    # 3 rented mules -> exactly 3 uniform draws taken.
    _ = [rng_control.uniform(0.0, 1.0) for _ in range(3)]
    assert rng_test.random() == rng_control.random()

    # mule_a (valid_from=600): picks offer1 (burned_from=500 <= 600)
    # mule_b (valid_from=700): offer2 has burned_from=800 > 700, so cannot be used yet!
    # mule_c (valid_from=900): picks offer2 (burned_from=800 <= 900)
    assert len(resales) == 2
    assert resales[0] == Resale(
        new_gang="gang_2",
        new_account="acc_new_a",
        sold_account="acc_old_1",
        previous_gang="gang_1",
        resale_minute=600,
    )
    assert resales[1] == Resale(
        new_gang="gang_2",
        new_account="acc_new_c",
        sold_account="acc_old_2",
        previous_gang="gang_1",
        resale_minute=900,
    )

    # Verify no offer is used twice:
    sold_accounts = [r.sold_account for r in resales]
    assert len(sold_accounts) == len(set(sold_accounts))


def test_apply_resale_leaves_no_trace_of_new_account() -> None:
    """
    WHAT: Tests that apply_resale replaces new_account with sold_account across every account-bearing field.
    WHY: Scanning every field ensures no synthetic placeholder escapes substitution in the resulting GangPlan.
    """
    old_acc = "acc_placeholder_001"
    sold_acc = "acc_real_001"
    other_acc = "acc_other_002"

    p1 = PlannedPayment(
        minute=100,
        from_account=old_acc,
        to_account=other_acc,
        amount_paise=1000,
        creation_index=0,
        purpose="fraud",
    )
    p2 = PlannedPayment(
        minute=200,
        from_account=other_acc,
        to_account=old_acc,
        amount_paise=2000,
        creation_index=1,
        purpose="fraud",
    )
    role_span = RoleSpan(
        account_id=old_acc,
        role="mule",
        gang_id="gang_2",
        depth=1,
        valid_from=100,
        valid_to=None,
        origin="rented",
    )
    dev_span = DeviceControlSpan("dev_01", old_acc, "gang_2", 100, None, False, False, False, False)
    story = PaymentStory("gang_2", "route", 1, "Story without account id")
    truth = TruthFacts(roles=(role_span,), device_control=(dev_span,), payment_stories=(story,))
    dev_event = GangDeviceEvent(
        minute=100,
        device_id="dev_01",
        account_id=old_acc,
        kind="device_acquired",
        platform="android",
        os_generation=12,
        fingerprint="fp",
    )
    money_rule = IllicitMoneyRule(
        input_index=0,
        spends_illicit=True,
        credits_illicit=True,
        requires_credit_from=old_acc,
    )
    outside_event = OutsideEvent(minute=100, kind="lien", account_id=old_acc, amount_paise=5000)
    withdrawal = CashWithdrawal(minute=150, account_id=old_acc, amount_paise=3000, channel="atm")
    cost = BlockCost(money_paise=100, time_minutes=50, risk=0.01)

    plan = GangPlan(
        payments=(p1, p2),
        truth=truth,
        costs=(cost,),
        totals=cost,
        gang_device_events=(dev_event,),
        money_rules=(money_rule,),
        outside_events=(outside_event,),
        withdrawals=(withdrawal,),
    )

    resale = Resale(
        new_gang="gang_2",
        new_account=old_acc,
        sold_account=sold_acc,
        previous_gang="gang_1",
        resale_minute=100,
    )

    new_plan = apply_resale(plan, resale)

    # 1. Planned payments
    assert new_plan.payments[0].from_account == sold_acc
    assert new_plan.payments[0].to_account == other_acc
    assert new_plan.payments[1].from_account == other_acc
    assert new_plan.payments[1].to_account == sold_acc

    # 2. Roles
    assert new_plan.truth.roles[0].account_id == sold_acc

    # 3. Device control
    assert new_plan.truth.device_control[0].account_id == sold_acc

    # 4. Gang device events
    assert new_plan.gang_device_events[0].account_id == sold_acc

    # 5. Money rules
    assert new_plan.money_rules[0].requires_credit_from == sold_acc

    # 6. Outside events
    assert new_plan.outside_events[0].account_id == sold_acc

    # 7. Cash withdrawals
    assert new_plan.withdrawals[0].account_id == sold_acc

    # Exhaustive scan: old_acc must not appear in any attribute or nested object in new_plan.
    scan_queue: list[Any] = [new_plan]
    visited_ids: set[int] = set()
    while scan_queue:
        current = scan_queue.pop()
        current_id = id(current)
        if current_id in visited_ids:
            continue
        visited_ids.add(current_id)
        if isinstance(current, str):
            assert old_acc not in current, f"Found lingering {old_acc} in string {current}"
        elif isinstance(current, (list, tuple)):
            scan_queue.extend(current)
        elif hasattr(current, "__dict__"):
            scan_queue.extend(current.__dict__.values())


FROZEN_EVENTS = load_yaml(REPO_ROOT / "config" / "frozen_anchors.yaml")["frozen_anchors"]["fraud_off_events"]
TINY = FastRunSettings(
    seed=FROZEN_EVENTS["seed"],
    n_people=FROZEN_EVENTS["n_people"],
    n_shops=FROZEN_EVENTS["n_shops"],
    n_days=FROZEN_EVENTS["n_days"],
    range_name=FROZEN_EVENTS["range_name"],
)
GENEROUS_BUDGET = GangBudget(money_paise=50_000_000, time_minutes=500_000, max_risk=0.99)

FIXTURE_MARKET_SHARE_1: dict[str, Any] = {
    "account_market": {
        "label": "fixture",
        "derived_from": "amendment A-017",
        "reason": "Test fixture full resale share.",
        "issue_ids_v2": ["G5"],
        "share_of_accounts_sold_to_several_gangs": {
            "training": [1.0, 1.0],
            "test": [1.0, 1.0],
        },
    }
}

FIXTURE_MARKET_SHARE_0: dict[str, Any] = {
    "account_market": {
        "label": "fixture",
        "derived_from": "amendment A-017",
        "reason": "Test fixture zero resale share.",
        "issue_ids_v2": ["G5"],
        "share_of_accounts_sold_to_several_gangs": {
            "training": [0.0, 0.0],
            "test": [0.0, 0.0],
        },
    }
}

FIXTURE_REACTIONS: dict[str, Any] = {
    "gang_reactions": {
        "label": "fixture",
        "derived_from": "amendment A-017",
        "reason": "Test fixture observation reactions.",
        "issue_ids_v2": ["G4"],
        "gang_feed_delay_minutes": {"training": [10, 10]},
        "gang_feed_miss_share": {"training": [0.0, 0.0]},
        "gang_stops_using_liened_account_share": {"training": [1.0, 1.0]},
        "gang_rotates_to_spare_account_share": {"training": [0.0, 0.0]},
        "gang_splits_amounts_share": {"training": [0.0, 0.0]},
        "gang_split_threshold_inr": {"training": [25000, 25000]},
    }
}


def _build_two_scam_gangs() -> tuple[GangConfig, GangConfig]:
    """
    WHAT: Constructs two sequential scam syndicates using rented mule recruitment.
    WHY: Supplies burned rented kits from gang 1 eligible for secondary market resale to gang 2.
    """
    steps_1 = (
        BlockStep("acquire", (("count", 4), ("origin_mix", "all_rented"))),
        BlockStep("scam", (("receiving_mules", 2),)),
        BlockStep("route", (("layers", 1), ("accounts_per_layer", 2))),
    )
    steps_2 = (
        BlockStep("acquire", (("count", 4), ("origin_mix", "all_rented"))),
        BlockStep("scam", (("receiving_mules", 2),)),
        BlockStep("route", (("layers", 1), ("accounts_per_layer", 2))),
    )
    g1 = GangConfig("g1", 1, GENEROUS_BUDGET, None, steps_1, 1440)
    g2 = GangConfig("g2", 2, GENEROUS_BUDGET, None, steps_2, 15840)
    return g1, g2


def test_two_scam_gangs_resale_share_1_moves_money_and_transfers_truth_owner() -> None:
    """
    WHAT: Asserts that with resale share 1.0, burned rented accounts are resold from gang 1 to gang 2.
    Verifies that evaluation truth owner transitions exactly at resale_minute, role spans do not overlap,
    and both gangs successfully route illicit funds through the resold account in their active windows.
    WHY: Proves end-to-end integration of secondary account market with non-overlapping temporal ownership.
    """
    g1, g2 = _build_two_scam_gangs()
    out = run_interactive_world(
        TINY,
        RunnerSettings(fraud_enabled=True, account_market_enabled=True),
        gang_configs=(g1, g2),
        account_market_config=FIXTURE_MARKET_SHARE_1,
    )
    res = out.result
    assert len(res.account_resales) >= 1
    validate_no_overlapping_role_spans(res.truth.roles)

    store = TruthStore(
        [asdict(r) for r in res.truth.roles],
        [asdict(d) for d in res.truth.device_control],
        {},
    )

    resold_with_both_moves: list[str] = []
    for r in res.account_resales:
        sold = r.sold_account
        rmin = r.resale_minute

        # Evaluation truth owner changes exactly at resale_minute:
        assert store.owner_at(sold, rmin - 1) == r.previous_gang
        assert store.owner_at(sold, rmin) == r.new_gang
        assert store.owner_at(sold, rmin + 1) == r.new_gang

        # Spans don't overlap:
        g1_spans = [s for s in res.truth.roles if s.account_id == sold and s.gang_id == r.previous_gang]
        g2_spans = [s for s in res.truth.roles if s.account_id == sold and s.gang_id == r.new_gang]
        assert len(g1_spans) == 1
        assert len(g2_spans) == 1
        assert g1_spans[0].valid_to == rmin
        assert g2_spans[0].valid_from == rmin

        # Both gangs move money through that account in their own windows:
        g1_pmts = [
            e for e, s in zip(res.run_result.events, res.event_sources)
            if s == "fraud" and (e.from_account == sold or e.to_account == sold)
            and e.minute < rmin and e.status == "success"
        ]
        g2_pmts = [
            e for e, s in zip(res.run_result.events, res.event_sources)
            if s == "fraud" and (e.from_account == sold or e.to_account == sold)
            and e.minute >= rmin and e.status == "success"
        ]
        if (
            len(g1_pmts) > 0
            and sum(e.amount_paise for e in g1_pmts) > 0
            and len(g2_pmts) > 0
            and sum(e.amount_paise for e in g2_pmts) > 0
        ):
            resold_with_both_moves.append(sold)

    assert len(resold_with_both_moves) >= 1


def test_resale_share_zero_matches_market_off() -> None:
    """
    WHAT: Tests that setting resale share to [0.0, 0.0] produces events, truth, and withdrawals
    identical to running with account market disabled.
    WHY: Verifies that zero resale share introduces zero behavioral deviation from the unaugmented baseline.
    """
    g1, g2 = _build_two_scam_gangs()
    out_zero = run_interactive_world(
        TINY,
        RunnerSettings(fraud_enabled=True, account_market_enabled=True),
        gang_configs=(g1, g2),
        account_market_config=FIXTURE_MARKET_SHARE_0,
    )
    out_off = run_interactive_world(
        TINY,
        RunnerSettings(fraud_enabled=True, account_market_enabled=False),
        gang_configs=(g1, g2),
    )
    assert out_zero.result.run_result.events == out_off.result.run_result.events
    assert out_zero.result.truth == out_off.result.truth
    assert out_zero.result.cash_withdrawals == out_off.result.cash_withdrawals
    assert len(out_zero.result.account_resales) == 0


def test_market_off_byte_identical_to_anchor() -> None:
    """
    WHAT: Verifies a fraud-on, two-gang, reactions-on world with market OFF reproduces the exact 5ada8cf hash.
    WHY: Proves adding the market switch and secondary resale machinery preserves pre-existing simulation outputs.
    """
    # Event hash pinned at commit 5ada8cf (2,930 events).
    expected_5ada8cf_hash = "1c7d05dde08a97409f0fbe980cb0e88b9def4b88e779138fd2dc6c2e3565f29d"
    steps_1 = (
        BlockStep("acquire", (("count", 4), ("origin_mix", "all_rented"))),
        BlockStep("scam", (("receiving_mules", 2),)),
        BlockStep("route", (("layers", 1), ("accounts_per_layer", 2))),
    )
    steps_2 = (
        BlockStep("acquire", (("count", 4), ("origin_mix", "all_rented"))),
        BlockStep("scam", (("receiving_mules", 2),)),
        BlockStep("route", (("layers", 1), ("accounts_per_layer", 2))),
    )
    g1 = GangConfig("g1", 1, GENEROUS_BUDGET, None, steps_1, 1440)
    g2 = GangConfig("g2", 2, GENEROUS_BUDGET, None, steps_2, 15840)
    out = run_interactive_world(
        TINY,
        RunnerSettings(fraud_enabled=True, gang_reactions_enabled=True, account_market_enabled=False),
        gang_configs=(g1, g2),
        code_version="5ada8cf",
    )
    lines = event_lines(interactive_fast_output(out))
    digest = hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()
    assert len(out.result.run_result.events) == 2930
    assert digest == expected_5ada8cf_hash


def test_no_seller_activity_on_sold_account_at_or_after_resale_minute() -> None:
    """
    WHAT: Verifies no payment, withdrawal or device event of seller gang occurs on sold accounts at or after resale.
    WHY: Ground rule G5 terminates the seller gang's operational authority upon secondary market transfer.
    """
    g1, g2 = _build_two_scam_gangs()
    out = run_interactive_world(
        TINY,
        RunnerSettings(fraud_enabled=True, account_market_enabled=True),
        gang_configs=(g1, g2),
        account_market_config=FIXTURE_MARKET_SHARE_1,
    )
    res = out.result
    assert len(res.account_resales) >= 1

    for r in res.account_resales:
        sold = r.sold_account
        rmin = r.resale_minute
        seller = r.previous_gang

        # Match payment events to gangs through event_fraud_input_index -> truth.payment_stories[index].gang_id:
        seller_pmts_before = [
            e for e, idx in zip(res.run_result.events, res.event_fraud_input_index)
            if idx is not None
            and res.truth.payment_stories[idx].gang_id == seller
            and (e.from_account == sold or e.to_account == sold)
            and e.minute < rmin
        ]
        seller_pmts_at_or_after = [
            e for e, idx in zip(res.run_result.events, res.event_fraud_input_index)
            if idx is not None
            and res.truth.payment_stories[idx].gang_id == seller
            and (e.from_account == sold or e.to_account == sold)
            and e.minute >= rmin
        ]
        assert len(seller_pmts_before) > 0, f"Expected seller gang '{seller}' activity before resale on {sold}"
        assert len(seller_pmts_at_or_after) == 0, (
            f"Found unexpected seller payments on {sold} at or after {rmin}: {seller_pmts_at_or_after}"
        )

        # Withdrawals planned by seller on sold account at or after resale_minute:
        seller_w_at_or_after = [
            w for w in res.cash_withdrawals
            if w.account == sold and w.minute >= rmin
        ]
        assert len(seller_w_at_or_after) == 0, (
            f"Found unexpected seller withdrawals on {sold} at or after {rmin}: {seller_w_at_or_after}"
        )

        # Applied gang device events for sold account at or after resale_minute:
        seller_d_at_or_after = [
            d for d in res.applied_gang_device_events
            if d.account_id == sold and d.minute >= rmin
        ]
        assert len(seller_d_at_or_after) == 0, (
            f"Found unexpected seller device events on {sold} at or after {rmin}: {seller_d_at_or_after}"
        )


def test_resale_determinism() -> None:
    """
    WHAT: Confirms that two runs with the same master seed produce identical account_resales and event logs.
    WHY: Demonstrates reproducibility and determinism of secondary account resale planning.
    """
    g1, g2 = _build_two_scam_gangs()
    out1 = run_interactive_world(
        TINY,
        RunnerSettings(fraud_enabled=True, account_market_enabled=True),
        gang_configs=(g1, g2),
        account_market_config=FIXTURE_MARKET_SHARE_1,
    )
    out2 = run_interactive_world(
        TINY,
        RunnerSettings(fraud_enabled=True, account_market_enabled=True),
        gang_configs=(g1, g2),
        account_market_config=FIXTURE_MARKET_SHARE_1,
    )
    assert out1.result.account_resales == out2.result.account_resales
    assert out1.result.run_result.events == out2.result.run_result.events


def test_children_0_to_9_unchanged_when_child_16_used() -> None:
    """
    WHAT: Tests that spawning 17 children leaves the bit-generator states of children 0 through 9 unchanged.
    WHY: Prevents secondary market planning on stream 16 from shifting any antecedent simulation draws.
    """
    seed = 101
    c10 = np.random.SeedSequence(seed).spawn(10)
    c17 = np.random.SeedSequence(seed).spawn(17)
    for i in range(10):
        assert c10[i].state == c17[i].state
        rng10 = np.random.default_rng(c10[i])
        rng17 = np.random.default_rng(c17[i])
        assert rng10.integers(0, 1_000_000, size=50).tolist() == rng17.integers(0, 1_000_000, size=50).tolist()


def test_no_eligible_offer_takes_draw_per_mule() -> None:
    """
    WHAT: Tests that when earlier gang offers have burned_from after the new gang's span starts, no resales occur
    while exactly one draw is still consumed on the RNG stream for each of the new gang's rented mules.
    WHY: Enforces offer temporal viability and stream position invariance regardless of offer availability.
    """
    offer = RoleSpan("acc_old", "mule", "g1", 1, 5000, None, origin="rented", burned_from=6000)
    mule_a = RoleSpan("acc_new_a", "mule", "g2", 1, 1440, None, origin="rented")
    mule_b = RoleSpan("acc_new_b", "mule", "g2", 1, 1440, None, origin="rented")

    rng_test = np.random.default_rng(42)
    rng_control = np.random.default_rng(42)

    resales = plan_resales([mule_a, mule_b], [offer], share=1.0, rng=rng_test)
    assert len(resales) == 0

    _ = [rng_control.uniform(0.0, 1.0) for _ in range(2)]
    assert rng_test.random() == rng_control.random()


def test_resold_account_liened_while_held_by_gang_2() -> None:
    """
    WHAT: Asserts that an outside lien event placed on a resold account during gang 2's ownership window
    is delivered solely to gang 2, while gang 1 receives no telemetry and remains completely unaffected.
    WHY: Validates operational isolation and dynamic holding lookup for resold infrastructure.
    """
    g1, g2 = _build_two_scam_gangs()
    lien = OutsideEvent(
        minute=16000,
        kind="lien",
        account_id="acc_person_000006",
        amount_paise=50000,
        encumbrance_id="enc-g2-lien",
    )
    inputs = FraudInputs(outside_events=(lien,))

    out = run_interactive_world(
        TINY,
        RunnerSettings(fraud_enabled=True, gang_reactions_enabled=True, account_market_enabled=True),
        fraud_inputs=inputs,
        gang_configs=(g1, g2),
        account_market_config=FIXTURE_MARKET_SHARE_1,
        gang_reactions_config=FIXTURE_REACTIONS,
    )
    res = out.result
    assert len(res.account_resales) >= 1
    assert len(res.cancelled_by_gang) > 0
    assert all(c.gang == "g2" for c in res.cancelled_by_gang)
    assert any("acc_person_000006" in c.accounts for c in res.cancelled_by_gang)


def test_fixture_role_in_truth_roles_does_not_become_gang_holding() -> None:
    """
    WHAT: Asserts that an outside fixture RoleSpan passed in via fraud_inputs.truth.roles does not become
    a gang holding, ensuring that subsequent liens on that fixture account emit no gang feed observations
    and cause zero gang cancellations on subsequent transfers.
    WHY: Proves gang holdings are constructed solely from gang plan roles, preventing external fixture roles
    from leaking into syndicate infrastructure and triggering spurious reactions.
    """
    g1, g2 = _build_two_scam_gangs()
    fixture_acc = "acc_person_000001"
    recipient_acc = "acc_person_000003"

    fixture_role = RoleSpan(
        account_id=fixture_acc,
        role="mule",
        gang_id="g1",
        depth=1,
        valid_from=1,
        valid_to=None,
        origin="rented",
    )
    lien = OutsideEvent(
        minute=2000,
        kind="lien",
        account_id=fixture_acc,
        amount_paise=50000,
        encumbrance_id="enc-fixture-guard",
    )
    pmt = PlannedPayment(
        minute=4000,
        from_account=fixture_acc,
        to_account=recipient_acc,
        amount_paise=1000,
        creation_index=0,
        purpose="everyday",
    )
    story = PaymentStory(
        gang_id="g1",
        block="scam",
        layer=1,
        description="Scam transfer",
    )
    inputs = FraudInputs(
        payments=(pmt,),
        outside_events=(lien,),
        truth=TruthFacts(
            roles=(fixture_role,),
            payment_stories=(story,),
        ),
    )

    out = run_interactive_world(
        TINY,
        RunnerSettings(fraud_enabled=True, gang_reactions_enabled=True, account_market_enabled=True),
        fraud_inputs=inputs,
        gang_configs=(g1, g2),
        account_market_config=FIXTURE_MARKET_SHARE_1,
        gang_reactions_config=FIXTURE_REACTIONS,
    )
    res = out.result

    # 1. Prove the lien really was placed in the encumbrance log.
    lien_entries = [
        entry for entry in res.encumbrance_log
        if entry[1] == "place_lien" and entry[2] == "enc-fixture-guard"
    ]
    assert len(lien_entries) == 1, f"Expected placed lien in encumbrance_log, got {res.encumbrance_log}"

    # 2. Assert no gang observes it: no feed item was processed, so burned_from remains None.
    matching_roles = [r for r in res.truth.roles if r.account_id == fixture_acc]
    assert len(matching_roles) == 1
    assert matching_roles[0].burned_from is None, (
        f"Fixture role was unexpectedly curtailed to {matching_roles[0].burned_from}"
    )

    # 3. Assert no cancellation on the fixture account.
    fixture_cancellations = [
        c for c in res.cancelled_by_gang
        if fixture_acc in c.accounts
    ]
    assert len(fixture_cancellations) == 0, (
        f"Fixture account unexpectedly suffered gang cancellations: {fixture_cancellations}"
    )

    # 4. Prove the payment from the fixture account actually executed (non-vacuous proof).
    executed_pmts = [
        e for e in res.run_result.events
        if e.from_account == fixture_acc and e.to_account == recipient_acc and e.minute == 4000
    ]
    assert len(executed_pmts) == 1, f"Expected payment from {fixture_acc} to execute"


