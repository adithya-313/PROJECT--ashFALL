"""
Tests for mule origins and gang-opened mule accounts (world/mule_origins.py, world/opening_handles.py and the acquire
step of world/gang_planner.py). Covers the origin draw and its config, the handles a plan uses, the id replay that gives
each handle its real id in queue order, the refusals, and whole worlds where borrowed-KYC and business mules are opened
mid-run next to genuine openings.

TERM: mule origin: rented, borrowed_kyc or business; stored in truth on the mule's role span.
TERM: handle: a placeholder account name "new:<gang_id>:<n>" for an account the gang opens during the run.
TERM: opening replay: counting the open_account events in queue order to find each opening's id before the run.
TERM: genuine opening: an ordinary person or shop account opened mid-run by the account openings switch.
"""

import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

import world.mule_origins as mule_origins
import world.opening_handles as opening_handles
from showcase.level_12_support_b import get_showcase_b_settings
from world.build_world import World, build_world
from world.calendar_runner import PlannedPayment
from world.checks import check_balances_equal_entry_sums, check_money_conserved, check_no_negative_balances
from world.gang_config import BlockStep, GangBudget, GangConfig, gang_config_to_plain, validate_gang_config
from world.gang_planner import plan_gang
from world.interactive_records import FraudInputs, RunnerSettings
from world.interactive_runner import prepare_world, run_interactive_world
from world.mule_origins import (
    build_opening_event,
    draw_mule_origins,
    draw_origin_shares,
    is_opening_handle,
    make_opening_handle,
)
from world.opening_handles import replay_openings, resolve_opening_handles
from world.outside_events import OutsideEvent, validate_outside_event
from world.tactic_config import (
    TACTIC_BLOCKS_PATH,
    WORLD_RANGES_PATH,
    get_mule_origin_share_ranges,
    get_new_account_owner_kind,
    get_origin_mix_options,
)
from world.truth_writer import write_truth_files

BUDGET = GangBudget(money_paise=50_000_000, time_minutes=500_000, max_risk=0.99)
START_MINUTE = 1440
ALL_RENTED = {"rented": (1.0, 1.0), "borrowed_kyc": (0.0, 0.0), "business": (0.0, 0.0)}
ALL_BORROWED = {"rented": (0.0, 0.0), "borrowed_kyc": (1.0, 1.0), "business": (0.0, 0.0)}
ALL_BUSINESS = {"rented": (0.0, 0.0), "borrowed_kyc": (0.0, 0.0), "business": (1.0, 1.0)}
NEW_ONLY = {"rented": (0.0, 0.0), "borrowed_kyc": (1.0, 1.0), "business": (1.0, 1.0)}


def make_origin_gang(origin_mix: str | None, acquire_count: int = 4, layers: int = 2) -> GangConfig:
    """Builds a scam gang that acquires mules with the given origin_mix (None leaves the parameter out)."""
    acquire_params: tuple[tuple[str, Any], ...] = (("count", acquire_count),)
    if origin_mix is not None:
        acquire_params += (("origin_mix", origin_mix),)
    steps = (
        BlockStep("acquire", acquire_params),
        BlockStep("scam", (("receiving_mules", 2), ("victims_per_receiving_mule", 2))),
        BlockStep("route", (("layers", layers), ("accounts_per_layer", 2))),
    )
    return GangConfig("g1", 1, BUDGET, None, steps, START_MINUTE)


def use_shares(monkeypatch: pytest.MonkeyPatch, shares: dict[str, tuple[float, float]]) -> None:
    """Makes the origin draw read the given share ranges instead of the config files."""
    monkeypatch.setattr(mule_origins, "get_mule_origin_share_ranges", lambda split: shares)


@pytest.fixture(scope="module")
def origin_world() -> World:
    """Builds one reproducible world of 60 people and 12 shops, shared by the planning tests of this module."""
    return build_world(seed=101, n_people=60, n_shops=12)


def run_origin_gang(gang: GangConfig, runner_settings: RunnerSettings | None = None, **kwargs: Any) -> Any:
    """Runs the seed-101 showcase world with fraud on and the given gang."""
    settings, code_version = get_showcase_b_settings()
    return run_interactive_world(
        fast_settings=settings, runner_settings=runner_settings or RunnerSettings(fraud_enabled=True),
        gang_configs=(gang,), code_version=code_version, **kwargs,
    )


# ---- config and draws -----------------------------------------------------------------------------------------


def test_origin_config_values_are_read_from_the_files() -> None:
    """Checks the share ranges, owner kinds and origin_mix options come from the committed config files."""
    assert get_mule_origin_share_ranges("training") == {
        "rented": (0.2, 0.5), "borrowed_kyc": (0.1, 0.4), "business": (0.05, 0.3),
    }
    assert get_new_account_owner_kind("rented") is None
    assert get_new_account_owner_kind("borrowed_kyc") == "person"
    assert get_new_account_owner_kind("business") == "shop"
    assert get_origin_mix_options() == (["all_rented", "world_ranges"], "all_rented")


def test_origin_config_follows_narrowed_files(tmp_path: Path) -> None:
    """Narrows the world ranges and the block config and checks each resolver returns the narrowed values."""
    ranges = yaml.safe_load(WORLD_RANGES_PATH.read_text(encoding="utf-8"))
    ranges["parameters"]["mule_origin_share_business"]["training"] = [0.7, 0.9]
    ranges_path = tmp_path / "world_ranges.yaml"
    ranges_path.write_text(yaml.safe_dump(ranges), encoding="utf-8")
    blocks = yaml.safe_load(TACTIC_BLOCKS_PATH.read_text(encoding="utf-8"))
    blocks["mule_origin_shares"]["business"]["new_account_owner_kind"] = "person"
    blocks["tactic_blocks"]["blocks"]["acquire"]["parameters"]["origin_mix"]["allowed_values"] = ["world_ranges"]
    blocks["tactic_blocks"]["blocks"]["acquire"]["parameters"]["origin_mix"]["default"] = "world_ranges"
    blocks_path = tmp_path / "tactic_blocks.yaml"
    blocks_path.write_text(yaml.safe_dump(blocks), encoding="utf-8")

    assert get_mule_origin_share_ranges("training", blocks_path, ranges_path)["business"] == (0.7, 0.9)
    assert get_new_account_owner_kind("business", blocks_path) == "person"
    assert get_origin_mix_options(blocks_path) == (["world_ranges"], "world_ranges")

    blocks["tactic_blocks"]["blocks"]["acquire"]["parameters"]["origin_mix"]["default"] = "other"
    blocks_path.write_text(yaml.safe_dump(blocks), encoding="utf-8")
    with pytest.raises(ValueError, match="default origin_mix 'other'"):
        get_origin_mix_options(blocks_path)


def test_origin_config_refusals(tmp_path: Path) -> None:
    """Checks a missing split, an unknown origin and a bad owner kind each raise one plain sentence."""
    with pytest.raises(ValueError, match="no range for split 'nope'"):
        get_mule_origin_share_ranges("nope")
    with pytest.raises(ValueError, match="Unknown mule origin 'stolen'"):
        get_new_account_owner_kind("stolen")
    blocks = yaml.safe_load(TACTIC_BLOCKS_PATH.read_text(encoding="utf-8"))
    blocks["mule_origin_shares"]["business"]["new_account_owner_kind"] = "bank"
    blocks_path = tmp_path / "tactic_blocks.yaml"
    blocks_path.write_text(yaml.safe_dump(blocks), encoding="utf-8")
    with pytest.raises(ValueError, match="must be person, shop or null"):
        get_new_account_owner_kind("business", blocks_path)


def test_shares_are_normalised_to_sum_to_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """Checks drawn shares are divided by their sum, so ranges that do not sum to 1 still give chances."""
    use_shares(monkeypatch, {"rented": (2.0, 2.0), "borrowed_kyc": (1.0, 1.0), "business": (1.0, 1.0)})

    shares = draw_origin_shares("training", np.random.default_rng(1))

    assert shares == {"rented": 0.5, "borrowed_kyc": 0.25, "business": 0.25}


def test_zero_total_share_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """Checks that shares drawn as all zero cannot be normalised and raise a plain sentence."""
    use_shares(monkeypatch, {"rented": (0.0, 0.0), "borrowed_kyc": (0.0, 0.0), "business": (0.0, 0.0)})

    with pytest.raises(ValueError, match="sum to 0"):
        draw_origin_shares("training", np.random.default_rng(1))


@pytest.mark.parametrize(
    ("shares", "expected"),
    [(ALL_RENTED, "rented"), (ALL_BORROWED, "borrowed_kyc"), (ALL_BUSINESS, "business")],
)
def test_narrowed_shares_give_one_origin_to_every_mule(
    monkeypatch: pytest.MonkeyPatch, shares: dict[str, tuple[float, float]], expected: str
) -> None:
    """Narrows the shares so one origin has the whole chance and checks every mule gets that origin."""
    use_shares(monkeypatch, shares)

    assert draw_mule_origins(6, "training", np.random.default_rng(3)) == [expected] * 6


def test_real_ranges_give_a_mix_of_all_three_origins() -> None:
    """Checks that with the committed ranges a large draw contains all three origins."""
    assert set(draw_mule_origins(300, "training", np.random.default_rng(3))) == {"rented", "borrowed_kyc", "business"}


def test_handles_and_opening_requests() -> None:
    """Checks handle naming, handle recognition, and that the opening request follows the origin's owner kind."""
    assert make_opening_handle("g1", 2) == "new:g1:2"
    assert is_opening_handle("new:g1:2") and not is_opening_handle("acc_person_000001")

    person_event = build_opening_event("g1", 0, "borrowed_kyc", 100)
    shop_event = build_opening_event("g1", 1, "business", 100)

    assert (person_event.owner_kind, person_event.opening_handle) == ("person", "new:g1:0")
    assert (shop_event.owner_kind, shop_event.opening_handle) == ("shop", "new:g1:1")
    assert person_event.opening_stream == "gang_account_openings" and person_event.kind == "open_account"
    with pytest.raises(ValueError, match="needs no opening"):
        build_opening_event("g1", 2, "rented", 100)


# ---- planning -------------------------------------------------------------------------------------------------


def test_omitted_origin_mix_keeps_plain_form_and_records_no_origin(origin_world: World) -> None:
    """Checks that leaving origin_mix out adds nothing to the plain dict, plans no openings, and records no origin."""
    gang = make_origin_gang(None)

    plan = plan_gang(gang, origin_world, np.random.default_rng(9))

    assert "origin_mix" not in str(gang_config_to_plain(gang))
    assert plan.outside_events == ()
    assert {role.origin for role in plan.truth.roles} == {None}


def test_all_rented_uses_existing_accounts_and_marks_them_rented(origin_world: World) -> None:
    """Checks origin_mix all_rented picks only existing people, plans no openings, and records origin rented."""
    plan = plan_gang(make_origin_gang("all_rented"), origin_world, np.random.default_rng(9))

    mule_roles = [role for role in plan.truth.roles if role.role == "mule"]
    assert len(mule_roles) == 4
    assert {role.origin for role in mule_roles} == {"rented"}
    assert not any(is_opening_handle(role.account_id) for role in mule_roles)
    assert plan.outside_events == ()


def test_world_ranges_with_new_accounts_plans_handles_and_openings(
    monkeypatch: pytest.MonkeyPatch, origin_world: World
) -> None:
    """Checks borrowed and business mules are handles with matching open_account requests at the acquire minute."""
    use_shares(monkeypatch, NEW_ONLY)

    plan = plan_gang(make_origin_gang("world_ranges"), origin_world, np.random.default_rng(9))

    mule_roles = [role for role in plan.truth.roles if role.role == "mule"]
    assert all(is_opening_handle(role.account_id) for role in mule_roles)
    assert len(plan.outside_events) == 4
    assert {event.minute for event in plan.outside_events} == {START_MINUTE}
    assert [event.opening_handle for event in plan.outside_events] == [f"new:g1:{n}" for n in range(4)]
    origin_of = {role.account_id: role.origin for role in mule_roles}
    for event in plan.outside_events:
        expected_origin = "borrowed_kyc" if event.owner_kind == "person" else "business"
        assert origin_of[str(event.opening_handle)] == expected_origin


def test_world_ranges_with_all_rented_plans_no_openings(monkeypatch: pytest.MonkeyPatch, origin_world: World) -> None:
    """Checks shares of 1, 0, 0 give four existing-account mules, all rented, and no openings."""
    use_shares(monkeypatch, ALL_RENTED)

    plan = plan_gang(make_origin_gang("world_ranges"), origin_world, np.random.default_rng(9))

    assert plan.outside_events == ()
    assert {role.origin for role in plan.truth.roles if role.role == "mule"} == {"rented"}


def test_invalid_origin_mix_is_refused() -> None:
    """Checks gang validation refuses an origin_mix that config does not allow."""
    with pytest.raises(ValueError, match="origin_mix must be one of all_rented, world_ranges"):
        validate_gang_config(make_origin_gang("random"))


# ---- id replay and handle resolution --------------------------------------------------------------------------


def person_open(minute: int, handle: str | None = None) -> OutsideEvent:
    """Builds a person open_account event, genuine when handle is None and a gang opening otherwise."""
    stream = "genuine_account_openings" if handle is None else "gang_account_openings"
    return OutsideEvent(minute, "open_account", owner_kind="person", opening_stream=stream, opening_handle=handle)


def shop_open(minute: int, handle: str) -> OutsideEvent:
    """Builds a gang shop open_account event with a handle."""
    return OutsideEvent(
        minute, "open_account", owner_kind="shop", opening_stream="gang_account_openings", opening_handle=handle
    )


def test_replay_follows_queue_order_minute_then_list_position() -> None:
    """Checks ids follow minute order, ties follow list position, and persons and shops count separately."""
    events = (person_open(500, "new:g:0"), person_open(100), person_open(500), shop_open(300, "new:g:1"))

    ids, minutes = replay_openings(events, 60, 12)

    assert ids == {"new:g:1": "acc_shop_0012", "new:g:0": "acc_person_000061"}
    assert minutes == {"new:g:1": 300, "new:g:0": 500}


def test_replay_refuses_a_repeated_handle() -> None:
    """Checks two openings with the same handle are refused."""
    with pytest.raises(ValueError, match="Two open_account events carry the handle 'new:g:0'"):
        replay_openings((person_open(100, "new:g:0"), person_open(200, "new:g:0")), 60, 12)


def test_resolve_replaces_every_stored_account_id_and_leaves_handleless_inputs_alone() -> None:
    """Checks payments, roles, device spans, device events, seeds and outside events lose their handles."""
    from world.gang_devices import GangDeviceEvent
    from world.interactive_records import IllicitSeed
    from world.truth_records import DeviceControlSpan, RoleSpan, TruthFacts

    handle = "new:g:0"
    inputs = FraudInputs(
        payments=(PlannedPayment(200, handle, "acc_person_000001", 500, 0, "everyday"),),
        outside_events=(person_open(100, handle), OutsideEvent(150, "debit_hold", account_id=handle)),
        truth=TruthFacts(
            roles=(RoleSpan(handle, "mule", "g", 1, 100, None, "borrowed_kyc"),),
            device_control=(DeviceControlSpan("dev", handle, "g", 100, None, False, False, False, False),),
        ),
        gang_device_events=(GangDeviceEvent(100, "dev", handle, "device_acquired", "android", 3, "fp"),),
        illicit_seeds=(IllicitSeed(handle, 100),),
    )

    resolved, assigned = resolve_opening_handles(inputs, 60, 12)

    assert assigned == {handle: "acc_person_000060"}
    assert resolved.payments[0].from_account == "acc_person_000060"
    assert resolved.truth.roles[0].account_id == "acc_person_000060"
    assert resolved.truth.roles[0].origin == "borrowed_kyc"
    assert resolved.truth.device_control[0].account_id == "acc_person_000060"
    assert resolved.gang_device_events[0].account_id == "acc_person_000060"
    assert resolved.illicit_seeds[0].account_id == "acc_person_000060"
    assert resolved.outside_events[1].account_id == "acc_person_000060"
    assert resolved.outside_events[0].opening_handle == handle
    untouched = FraudInputs(outside_events=(person_open(100),))
    assert resolve_opening_handles(untouched, 60, 12) == (untouched, {})


def test_payment_before_the_opening_is_refused_and_unknown_handle_is_refused() -> None:
    """Checks a payment dated before its handle's opening, and a handle that nothing opens, are both refused."""
    early = FraudInputs(
        payments=(PlannedPayment(99, "new:g:0", "acc_person_000001", 500, 0, "everyday"),),
        outside_events=(person_open(100, "new:g:0"),),
    )
    with pytest.raises(ValueError, match="before it is opened at minute 100"):
        resolve_opening_handles(early, 60, 12)

    orphan = FraudInputs(
        payments=(PlannedPayment(200, "new:g:9", "acc_person_000001", 500, 0, "everyday"),),
        outside_events=(person_open(100, "new:g:0"),),
    )
    with pytest.raises(ValueError, match="no open_account event opens it"):
        resolve_opening_handles(orphan, 60, 12)


def test_runner_refuses_an_early_payment_before_anything_runs() -> None:
    """Runs the whole runner with a payment before its handle's opening and checks the same refusal comes out."""
    settings, code_version = get_showcase_b_settings()
    early = FraudInputs(
        payments=(PlannedPayment(99, "new:g:0", "acc_person_000001", 500, 0, "everyday"),),
        outside_events=(person_open(100, "new:g:0"),),
    )

    with pytest.raises(ValueError, match="before it is opened"):
        run_interactive_world(
            settings, RunnerSettings(fraud_enabled=True), fraud_inputs=early, code_version=code_version
        )


def test_apply_time_id_mismatch_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Makes the replay assign a wrong id and checks open_account's real id is caught when the event is applied."""
    monkeypatch.setattr(opening_handles, "person_account_id", lambda index: f"acc_person_9{index:05d}")
    settings, code_version = get_showcase_b_settings()
    inputs = FraudInputs(outside_events=(person_open(100, "new:g:0"),))

    with pytest.raises(RuntimeError, match="produced 'acc_person_000060' but 'acc_person_900060' was assigned"):
        run_interactive_world(
            settings, RunnerSettings(fraud_enabled=True), fraud_inputs=inputs, code_version=code_version
        )


def test_handle_rules_on_outside_events() -> None:
    """Checks a genuine opening with a handle, and any other event kind with a handle, are refused."""
    with pytest.raises(ValueError, match="genuine open_account event must not carry an opening_handle"):
        validate_outside_event(
            OutsideEvent(1, "open_account", owner_kind="person", opening_stream="genuine_account_openings",
                         opening_handle="new:g:0")
        )
    with pytest.raises(ValueError, match="only an open_account event may carry an opening_handle"):
        validate_outside_event(OutsideEvent(1, "debit_hold", account_id="acc_a", opening_handle="new:g:0"))
    validate_outside_event(person_open(1, "new:g:0"))


# ---- whole worlds ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mixed_output() -> Any:
    """Runs a gang whose four mules are all new accounts (borrowed-KYC and business) beside genuine openings."""
    patcher = pytest.MonkeyPatch()
    use_shares(patcher, NEW_ONLY)
    try:
        genuine_same_minute = FraudInputs(outside_events=(person_open(START_MINUTE),))
        output = run_origin_gang(
            make_origin_gang("world_ranges"), RunnerSettings(fraud_enabled=True, account_openings_enabled=True),
            fraud_inputs=genuine_same_minute,
        )
    finally:
        patcher.undo()
    return output


def test_new_mules_get_normal_format_ids_in_time_order_with_genuine_openings(mixed_output: Any) -> None:
    """Checks mule ids use the standard formats, count on from the genuine openings in time order, and are mules."""
    opened = mixed_output.result.opened_accounts
    minutes = [row[0] for row in opened]
    assert minutes == sorted(minutes)

    person_ids = [row[1] for row in opened if row[2] == "person"]
    shop_ids = [row[1] for row in opened if row[2] == "shop"]
    assert person_ids == [f"acc_person_{60 + n:06d}" for n in range(len(person_ids))]
    assert shop_ids == [f"acc_shop_{12 + n:04d}" for n in range(len(shop_ids))]
    assert all(re.fullmatch(r"acc_person_\d{6}|acc_shop_\d{4}", row[1]) for row in opened)

    gang_rows = [row for row in opened if row[3] == "gang_account_openings"]
    assert len(gang_rows) == 4 and {row[0] for row in gang_rows} == {START_MINUTE}
    first_at_start = next(row for row in opened if row[0] == START_MINUTE)
    assert first_at_start[3] == "genuine_account_openings"
    mule_ids = {role.account_id for role in mixed_output.result.truth.roles if role.role == "mule"}
    assert mule_ids == {row[1] for row in gang_rows}
    assert {role.origin for role in mixed_output.result.truth.roles if role.role == "mule"} == {
        "borrowed_kyc", "business"
    }


def test_no_handle_survives_anywhere_in_the_output(mixed_output: Any, tmp_path: Path) -> None:
    """Scans the whole result, the opened log and every written truth file for a leftover handle."""
    assert "new:" not in repr(mixed_output.result)
    run_dir = write_truth_files(mixed_output, tmp_path)
    for truth_file in run_dir.iterdir():
        assert "new:" not in truth_file.read_text(encoding="utf-8"), truth_file.name


def test_money_is_conserved_including_mule_opening_balances(mixed_output: Any) -> None:
    """Checks ledgers balance and total money equals the start plus the opening balances of every opened account."""
    settings, _ = get_showcase_b_settings()
    fresh_world = prepare_world(settings, None).world
    opening_sum = sum(ledger.balance_of(account) for ledger in fresh_world.ledgers for account in ledger.accounts)
    ledgers = mixed_output.world.ledgers
    opened_balance_total = sum(row[5] for row in mixed_output.result.opened_accounts)

    assert mixed_output.result.money_created_paise == opened_balance_total
    assert check_no_negative_balances(ledgers) == []
    assert check_balances_equal_entry_sums(ledgers) == []
    assert check_money_conserved(ledgers, opening_sum + mixed_output.result.money_created_paise) == []


def test_new_mules_receive_and_forward_victim_money(mixed_output: Any) -> None:
    """Checks the new accounts really take part: layer 1 mules receive victim payments that succeeded."""
    result = mixed_output.result
    mule_ids = {role.account_id for role in result.truth.roles if role.role == "mule"}
    received = [
        event for event, source in zip(result.run_result.events, result.event_sources)
        if source == "fraud" and event.to_account in mule_ids and event.status == "success"
    ]
    assert received


def test_wrong_assignment_order_would_be_caught_at_apply_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reverses the replay's tie order and checks a same-minute genuine and gang opening then disagree at apply."""
    real_replay = opening_handles.replay_openings

    def reversed_replay(events: Any, next_person: int, next_shop: int) -> Any:
        """Replays the openings with list order reversed, which is not the order the queue applies them in."""
        return real_replay(tuple(reversed(events)), next_person, next_shop)

    monkeypatch.setattr(opening_handles, "replay_openings", reversed_replay)
    settings, code_version = get_showcase_b_settings()
    inputs = FraudInputs(outside_events=(person_open(100), person_open(100, "new:g:0")))

    with pytest.raises(RuntimeError, match="was assigned before the run"):
        run_interactive_world(
            settings, RunnerSettings(fraud_enabled=True), fraud_inputs=inputs, code_version=code_version
        )
