"""
WHAT: The Level 12a interactive runner. With the fraud switch off it hands the world to Level 11's run_world
unchanged. With it on, it runs the same world through an ordered queue where a bank hook can hold or decline a
payment, outside events can place liens and debit holds, and fraud payments can be added. This module holds the
switch, the world preparation and the fraud-off path; the fraud-on loop lives in world/interactive_loop.py.

WHY (a) genuine payments keep their random numbers. Level 11 decides every payment's failures with one stream of
random numbers, the run stream, read in a fixed order: payment 1's pieces first, then payment 2's, and so on. If
fraud made one genuine payment skip its draws, every later genuine payment would read numbers meant for another
and the whole genuine world would shift. Example from the 60-person world with seed 101: the first payment (minute 1,
Rs 191.92 at a shop) reads 0.1040 and the second reads 0.8907. A payment fails in the background only when its number
is below 0.0123 (wrong PIN 0.0020 + network drop 0.0024 + technical decline 0.0079), so both succeed. If a bank held
the first payment and it skipped its draw, the second payment would read 0.1040 instead of 0.8907; across 2,915
payments such shifts change which ones fail. So a genuine payment always reads its draws at its planned place, even
when the bank holds or declines it; fraud and resumed payments read from their own streams (children 4 and 5) and
can never move a genuine draw.

WHY (b) private Level 11 helpers. Planning and running a payment piece live in calendar_runner.py's private
helpers (names starting with "_"). Copying them would let two copies drift apart; editing Level 11 is not allowed.
So this module calls them directly. What guards that choice: tests compare the fraud-on runner with an empty plan
against run_world on every output (RunResult, event hash, ledger hash, device hashes), the frozen-anchor test pins
Level 11's own output, and a signature test fails the moment a helper's parameters change.

TERM: fraud switch: RunnerSettings.fraud_enabled; off (the default) means Level 11 exactly.
TERM: run stream: the random numbers Level 11 uses to plan payments and decide their failures (child 1 of the seed).
TERM: child: one numbered random stream split from the seed; children 0-3 belong to Level 11, 4 and 5 to this runner.
TERM: draw: one random number taken from a stream.
TERM: planned payment: one payment the world plans before running, e.g. a salary, rent, a shop purchase.
TERM: piece (leg): one part of a payment split at the per-payment limit; a Rs 1,53,065 payment against a Rs 61,226
      limit becomes three pieces.
TERM: private helper: a function whose name starts with "_", meant for use inside its own file.
TERM: effective configuration: the settings plus the four config files a world was built from, as plain data.
TERM: run id: a 16-character fingerprint of seed, effective configuration and code version.
"""

import copy
import os
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np

from world.amounts import split_into_legal_payments
from world.background_config import read_background_realism
from world.build_world import World, build_world
from world.calendar_runner import (
    PlannedPayment,
    _collect_all_planned_payments,
    _merge_planned_payments,
    run_world,
)
from world.config_reader import read_world_ranges
from world.device_events import DeviceEvent
from world.device_schedule import draw_device_event_schedule
from world.episode_payments import plan_episode_payments
from world.episodes import Episode, draw_episode_schedule
from world.fast_mode import FastRunSettings
from world.gang_config import GangConfig, gang_config_to_plain
from world.gang_feed import build_gang_holdings_from_roles
from world.gang_planner import plan_gang
from world.interactive_loop import run_fraud_on
from world.interactive_records import (
    FraudInputs,
    HookDecision,
    IllicitMoneyRule,
    IllicitSeed,
    InteractiveRunResult,
    InteractiveWorldOutput,
    PaymentView,
    Resale,
    RunnerSettings,
    allow_all,
)
from world.interactive_state import start_runner_state
from world.payments import find_ledger_for_account
from world.run_identity import REPO_ROOT, find_code_version, load_runner_config, make_payment_id, make_run_id
from world.tactic_families import (
    check_family_lists_agree,
    load_evaluation_config,
    load_tactic_families,
    validate_gang_families,
)
from world.truth_records import TruthFacts
from world.world_configs import WorldConfigs, read_world_configs
from world.world_plan import WorldPlan, draw_world_plan, spawn_life_streams, spawn_world_streams

# The message for fraud inputs given while the switch is off; ignoring them silently would hide a mistake.
SWITCH_OFF_MESSAGE: str = "fraud inputs need fraud_enabled=True"


@dataclass(frozen=True)
class PreparedWorld:
    """
    WHAT: A world ready to run: built entities, the calendar plan, the run stream, the device schedule, the episodes
    and their planned payments. It is the state just before Level 11's step 9 (run_world).

    WHY: Both switch positions need exactly the same preparation, and the fraud-on side needs the device schedule
    that Level 11's FastRunOutput does not keep. Building it once here keeps the two sides identical.

    Args:
        world: The freshly built World.
        plan: The WorldPlan.
        run_rng: The run stream (child 1), not yet used.
        device_events: The device schedule.
        episodes: The drawn life episodes.
        episode_payments: The episodes' planned payments.
        episode_report: Counters from planning those payments.

    Returns:
        A frozen PreparedWorld.

    Example:
        prepare_world(FastRunSettings(101, 60, 12, 61, "training"), None).world.seed is 101.
    """

    world: World
    plan: WorldPlan
    run_rng: np.random.Generator
    device_events: tuple[DeviceEvent, ...]
    episodes: list[Episode]
    episode_payments: tuple[PlannedPayment, ...]
    episode_report: dict[str, Any]


def prepare_world(fast_settings: FastRunSettings, configs: WorldConfigs | None) -> PreparedWorld:
    """
    WHAT: Repeats steps 1 to 8 of Level 11's run_fast_world in the same order with the same arguments: build the world,
    split the seed into the plan, run, device and episode streams, draw the plan, the device schedule, the episodes
    and the episodes' payments. It stops before running any payment.

    WHY: run_fast_world runs everything in one go and keeps no device schedule. Repeating its first eight steps here
    gives both switch positions the same starting point; the fraud-off test proves the result still matches Level 11
    byte for byte.

    Args:
        fast_settings: Seed, people, shops, days and range name, e.g. FastRunSettings(101, 60, 12, 61, "training").
        configs: Layered configs, or None to read the four config files as Level 11 does.

    Returns:
        The PreparedWorld.

    Example:
        prepare_world(FastRunSettings(101, 60, 12, 61, "training"), None) has 60 people and 32 device events.
    """
    # 1-2. Build people, shops, banks and ledgers from the world ranges.
    ranges = configs.world_ranges if configs is not None else read_world_ranges()
    world = build_world(fast_settings.seed, fast_settings.n_people, fast_settings.n_shops, ranges=ranges)
    # 3-4. Split the seed into Level 11's four streams (children 0 to 3).
    plan_rng, run_rng = spawn_world_streams(fast_settings.seed)
    device_rng, episode_rng = spawn_life_streams(fast_settings.seed)
    # 5. Draw the calendar, festivals, incidents and failure rates.
    background = configs.background_realism if configs is not None else None
    failure = configs.failure_realism if configs is not None else None
    plan = draw_world_plan(
        plan_rng, fast_settings.n_days, fast_settings.range_name, background_config=background, failure_config=failure
    )
    # 6. Draw the device schedule; the same file Level 11 reads, found from the repo root.
    device_path = REPO_ROOT / "config" / "device_realism.yaml"
    device_config = configs.device_realism if configs is not None else read_background_realism(device_path)
    device_events = draw_device_event_schedule(device_config, device_rng, world.people, plan.festivals, plan.n_days)
    # 7-8. Draw life episodes and plan their payments on the episode stream.
    range_name = fast_settings.range_name
    episodes = draw_episode_schedule(ranges, episode_rng, world.people, plan.n_days, range_name=range_name)
    episode_plan = plan_episode_payments(world, plan, episodes, ranges, episode_rng, range_name=range_name)
    return PreparedWorld(
        world, plan, run_rng, tuple(device_events), episodes, tuple(episode_plan.payments), episode_plan.report
    )


def check_switch(settings: RunnerSettings, fraud_inputs: FraudInputs, hook: Callable[[PaymentView], HookDecision],
                 state: Any) -> None:
    """
    WHAT: Refuses fraud inputs, a custom hook or mid-run account openings when the fraud switch is off, with the plain
    message "fraud inputs need fraud_enabled=True" or "account openings need fraud_enabled=True".

    WHY: With the switch off the runner falls back to Level 11, which cannot use any of these. Accepting them and
    quietly ignoring them would let someone believe a lien or a fraud payment was applied when it was not.

    Args:
        settings: The fraud switch.
        fraud_inputs: The fraud payments and outside events.
        hook: The bank hook.
        state: A prepared runner state, or None.

    Returns:
        None. It raises ValueError when the inputs need the switch on.

    Example:
        check_switch(RunnerSettings(), FraudInputs(payments=(payment,)), allow_all, None) raises the message.
    """
    if not settings.fraud_enabled and settings.account_openings_enabled:
        raise ValueError("account openings need fraud_enabled=True")
    # With the switch on, every input is allowed.
    if settings.fraud_enabled:
        return
    # With it off, only the empty defaults are allowed.
    if fraud_inputs != FraudInputs() or hook is not allow_all or state is not None:
        raise ValueError(SWITCH_OFF_MESSAGE)


def planned_payment_ids_by_event(
    world: Any, plan: WorldPlan, run_rng_copy: np.random.Generator,
    episode_payments: Sequence[PlannedPayment], run_id: str, digits: int,
) -> list[str]:
    """
    WHAT: Works out, before Level 11 runs, which planned payment each future event belongs to. It plans the payments
    exactly as run_world will (on a copy of the run stream, so the real one is untouched), splits each at its bank's
    per-payment limit, and gives every piece its payment's planned payment id.

    WHY: Level 11's result lists events but not which payment each came from. The fraud-off path must not change
    run_world, so it predicts the grouping from the same plan instead. Planning reads no balances, so the copy gives
    the identical list (checked: 2,915 of 2,915 payments equal in the seed-101 world).

    Args:
        world: The prepared World, before any payment ran.
        plan: The WorldPlan.
        run_rng_copy: A copy of the run stream; it is used up here.
        episode_payments: The episodes' planned payments.
        run_id: The run id, e.g. "66db9aa439b092ee".
        digits: The padded width of the sequence number, from config (6).

    Returns:
        One planned payment id per future event, e.g. 2,915 ids for the seed-101 world, the first
        "<run id>-p-000000".

    Example:
        For a payment of 15,306,555 paise against a 6,122,622 limit, three events share one planned payment id.
    """
    # Plan exactly as run_world does: routine payments on the run stream, then episode payments merged in.
    planned = _merge_planned_payments(_collect_all_planned_payments(world, plan, run_rng_copy), episode_payments)
    planned_ids: list[str] = []
    # Every piece of payment number n gets that payment's id.
    for sequence, payment in enumerate(planned):
        sender_ledger = find_ledger_for_account(world.ledgers, payment.from_account)
        if sender_ledger is None:
            raise RuntimeError(f"account {payment.from_account} has no bank, so its pieces cannot be counted")
        piece_count = len(split_into_legal_payments(payment.amount_paise, sender_ledger.upi_limit_paise))
        planned_ids.extend([make_payment_id(f"{run_id}-p", sequence, digits)] * piece_count)
    return planned_ids


def run_fraud_off(
    world: Any, plan: WorldPlan, rng: np.random.Generator, device_events: Sequence[DeviceEvent],
    episode_payments: Sequence[PlannedPayment], run_id: str, runner_config: dict[str, Any],
) -> InteractiveRunResult:
    """
    WHAT: The fallback: runs Level 11's run_world unchanged and wraps its result. It adds payment ids from the event
    positions, the planned payment id of each event, "planned" as every event's source, and the device events (all
    of them are applied, because run_world applies the rest after its last payment).

    WHY: Ground rule 13: the switch off must be Level 11 exactly. The extra ids exist in memory only; nothing Level 11
    writes or hashes changes.

    Args:
        world: The prepared World.
        plan: The WorldPlan.
        rng: The run stream, unused so far.
        device_events: The device schedule.
        episode_payments: The episodes' planned payments.
        run_id: The run id.
        runner_config: The interactive_runner config block.

    Returns:
        The InteractiveRunResult; its run_result is exactly run_world's.

    Example:
        For the seed-101 tiny world: 2,915 events, 32 device events applied, 2,915 payment ids.
    """
    # Predict each event's planned payment from a copy of the run stream, before run_world uses the real one.
    digits = runner_config["position_digits"]
    planned_ids = planned_payment_ids_by_event(world, plan, copy.deepcopy(rng), episode_payments, run_id, digits)
    # Level 11, untouched.
    result = run_world(world, plan, rng, device_events=device_events, episode_payments=episode_payments)
    # The prediction must cover every event exactly; a mismatch means Level 11 changed how it plans or splits.
    if len(planned_ids) != len(result.events):
        raise RuntimeError(f"predicted {len(planned_ids)} events but Level 11 produced {len(result.events)}")
    payment_ids = [make_payment_id(run_id, position, digits) for position in range(len(result.events))]
    return InteractiveRunResult(
        run_result=result, payment_ids=payment_ids, planned_payment_ids=planned_ids,
        event_sources=["planned"] * len(result.events), events_resumed=[False] * len(result.events),
        applied_device_events=list(device_events),
        event_fraud_input_index=[None] * len(result.events), run_id=run_id,
        truth=TruthFacts(),
    )


def run_interactive(
    world: Any, plan: WorldPlan, rng: np.random.Generator,
    device_events: Sequence[DeviceEvent] = (), episode_payments: Sequence[PlannedPayment] = (), *,
    settings: RunnerSettings = RunnerSettings(), fraud_inputs: FraudInputs = FraudInputs(),
    hook: Callable[[PaymentView], HookDecision] = allow_all, run_id: str,
    runner_config: dict[str, Any] | None = None, state: Any = None,
    account_openings_config: dict[str, Any] | None = None,
    gang_reactions_config: dict[str, Any] | None = None,
) -> InteractiveRunResult:
    """
    WHAT: Runs one prepared world. With the switch off it falls back to Level 11's run_world; with it on it runs the
    fraud-on loop. Fraud inputs, a custom hook or a prepared state with the switch off raise ValueError.

    WHY: One entry point with one switch is the rule-13 shape: callers flip a setting and never choose between two
    code paths themselves.

    Args:
        world: The prepared World.
        plan: The WorldPlan.
        rng: The run stream (child 1), unused so far.
        device_events: The device schedule.
        episode_payments: The episodes' planned payments.
        settings: The fraud switch.
        fraud_inputs: Fraud payments and outside events (switch on only).
        hook: The bank hook (switch on only); allow_all by default.
        run_id: The run id.
        runner_config: The interactive_runner config block; None reads config/interactive_runner.yaml.
        state: A prepared runner state with extra registered accounts (switch on only), or None.
        account_openings_config: Optional account openings configuration dictionary.
        gang_reactions_config: Optional gang reactions configuration dictionary.

    Returns:
        The InteractiveRunResult.

    Example:
        run_interactive(world, plan, run_rng, run_id="66db9aa439b092ee") on the seed-101 world gives 2,915 events.
    """
    # Refuse inputs that only make sense with fraud on.
    check_switch(settings, fraud_inputs, hook, state)
    config = runner_config if runner_config is not None else load_runner_config()
    if not settings.fraud_enabled:
        return run_fraud_off(world, plan, rng, device_events, episode_payments, run_id, config)
    # Fraud on: the runner's own lookup tables, built the way run_world builds them unless the caller prepared them.
    runner_state = state if state is not None else start_runner_state(world)
    return run_fraud_on(
        world, plan, rng, device_events, episode_payments, fraud_inputs=fraud_inputs, hook=hook, run_id=run_id,
        runner_config=config, state=runner_state, settings=settings,
        account_openings_config=account_openings_config,
        gang_reactions_config=gang_reactions_config,
    )


def effective_config_for(
    fast_settings: FastRunSettings, configs: WorldConfigs | None,
    gang_configs: tuple[GangConfig, ...] = (),
    *,
    runner_settings: RunnerSettings = RunnerSettings(),
    account_openings_config: dict[str, Any] | None = None,
    gang_reactions_config: dict[str, Any] | None = None,
    account_market_config: dict[str, Any] | None = None,
    world_split: str | None = None,
) -> dict[str, Any]:
    """
    WHAT: Builds the effective configuration as plain data: the run settings plus the four config dictionaries the
    world is built from (the given layered configs, or the four files as read from disk), plus gang configurations,
    account openings configurations, and gang reactions configurations if enabled.

    WHY: The run id must change when any setting changes, including one value deep inside a config file or a gang
    tactical configuration. Folding the full configuration into the id guarantees that per Decision D5 and D7.

    Args:
        fast_settings: The run settings.
        configs: Layered configs, or None for the files on disk.
        gang_configs: Sequence of GangConfig records, serialized to plain dictionaries only when non-empty.
        runner_settings: RunnerSettings governing feature flags like account_openings_enabled
            and gang_reactions_enabled.
        account_openings_config: Optional account openings configuration dictionary.
        gang_reactions_config: Optional gang reactions configuration dictionary.
        account_market_config: Optional account market configuration dictionary.
        world_split: The world's split, written to the result as "world_split" only when it is not None.

    Returns:
        {"settings": {...}, "world_configs": {...}} and optionally "gang_configs",
        account openings, and gang reactions fields.

    Example:
        effective_config_for(FastRunSettings(101, 60, 12, 61, "training"), None)["settings"]["seed"] is 101.
    """
    # Plain dictionaries serialise the same way on every machine.
    world_configs = configs if configs is not None else read_world_configs()
    result: dict[str, Any] = {"settings": asdict(fast_settings), "world_configs": asdict(world_configs)}
    if gang_configs:
        result["gang_configs"] = [gang_config_to_plain(gc) for gc in gang_configs]
    if runner_settings.account_openings_enabled:
        from world.account_openings_config import load_account_openings_config

        openings_cfg = (
            account_openings_config
            if account_openings_config is not None
            else load_account_openings_config()
        )
        result["account_openings_enabled"] = True
        result["account_openings_config"] = openings_cfg
    if runner_settings.gang_reactions_enabled:
        from world.gang_feed import load_gang_reactions_config, validate_gang_reactions_config

        reactions_cfg = (
            gang_reactions_config
            if gang_reactions_config is not None
            else load_gang_reactions_config()
        )
        validate_gang_reactions_config(reactions_cfg)
        result["gang_reactions_enabled"] = True
        result["gang_reactions_config"] = reactions_cfg
    if runner_settings.account_market_enabled:
        from world.account_market import load_account_market_config, validate_account_market_config

        market_cfg = (
            account_market_config
            if account_market_config is not None
            else load_account_market_config()
        )
        validate_account_market_config(market_cfg)
        result["account_market_enabled"] = True
        result["account_market_config"] = market_cfg
    # Written only when given, like "family", so every run id made without a world_split stays as it was.
    if world_split is not None:
        result["world_split"] = world_split
    return result


def family_guard_split(world_split: str | None, range_name: str) -> str:
    """Returns the name the 12d family guard is given: the world's split when one is given, else its range name."""
    # A world's split decides which tactics it may hold; its range column decides its numbers. Section 6 makes
    # these differ for test worlds (test split, training column), so the guard gets the split when there is one.
    return world_split if world_split is not None else range_name


def _check_split_and_range(world_split: str, range_name: str) -> None:
    """Accepts exactly the split and range column pairs of Section 6 and refuses every other pairing by name."""
    allowed = {"training": "training", "validation": "training", "test": "training", "stress": "stress"}
    if world_split not in allowed:
        raise ValueError(f"world_split '{world_split}' must be training, validation, test or stress")
    if allowed[world_split] != range_name:
        raise ValueError(
            f"world_split '{world_split}' cannot use the {range_name} range column; it must use {allowed[world_split]}"
        )


def _apply_gang_configs(
    prepared: PreparedWorld, gang_configs: tuple[GangConfig, ...], state: Any,
    fraud_inputs: FraudInputs, seed: int, runner_config: dict[str, Any],
    *,
    account_market_enabled: bool = False,
    account_market_config: dict[str, Any] | None = None,
    held_out_split: str | None = None,
) -> tuple[Any, FraudInputs]:
    """
    WHAT: Registers fixture source accounts in runner state, plans each gang campaign using the gang_planning
    stream (child 7), and appends scheduled payments and truth facts to FraudInputs.

    WHY: Decision D5 connects gang tactical planning to reactive runner execution, ensuring criminal accounts
    and scheduled multi-hop fraud payments enter the simulation loop deterministically.

    Args:
        prepared: The PreparedWorld instance.
        gang_configs: Tuple of GangConfig records.
        state: RunnerState instance or None.
        fraud_inputs: Existing FraudInputs to extend.
        seed: Master simulation integer seed.
        runner_config: Interactive runner configuration dictionary.
        account_market_enabled: Boolean flag indicating if secondary account market resales are active.
        account_market_config: Optional account market configuration dictionary.

    Returns:
        Tuple of (updated runner state, combined FraudInputs).

    Example:
        # state, inputs = _apply_gang_configs(prepared, gang_configs, state, inputs, 101, config)
    """
    active_state = state if state is not None else start_runner_state(prepared.world)
    seeds = list(fraud_inputs.illicit_seeds)
    for g in gang_configs:
        # A gang whose money comes from a scam step has no fixture source to register or seed.
        if g.source is None:
            continue
        if g.source.account_id not in active_state.bank_map:
            active_state.register_account(
                g.source.account_id, g.source.bank_index,
                opening_balance_paise=g.source.opening_balance_paise, minute=g.source.start_minute,
            )
        seeds.append(IllicitSeed(account_id=g.source.account_id, amount_paise=g.source.opening_balance_paise))

    children = np.random.SeedSequence(seed).spawn(10)
    dev_rng, plan_rng = np.random.default_rng(children[6]), np.random.default_rng(children[7])
    reserved = {g.source.account_id for g in gang_configs if g.source is not None}
    pmts = list(fraud_inputs.payments)
    roles = list(fraud_inputs.truth.roles)
    dev_ctrl = list(fraud_inputs.truth.device_control)
    stories = list(fraud_inputs.truth.payment_stories)
    g_dev_events = list(fraud_inputs.gang_device_events)
    rules = list(fraud_inputs.money_rules)
    opening_events = list(fraud_inputs.outside_events)
    withdrawals = list(fraud_inputs.withdrawals)

    resale_share: float | None = None
    market_rng: np.random.Generator | None = None
    all_resales: list[Resale] = []
    if account_market_enabled:
        from world.account_market import (
            apply_resale,
            draw_resale_share,
            find_resale_offers,
            plan_resales,
        )

        children_17 = np.random.SeedSequence(seed).spawn(17)
        market_rng = np.random.default_rng(children_17[16])
        resale_share = draw_resale_share(prepared.plan.range_name, market_rng, config=account_market_config)

    next_device_index = 1
    for k, g in enumerate(gang_configs):
        plan = plan_gang(
            g, prepared.world, plan_rng, reserved_accounts=reserved, gang_devices_rng=dev_rng,
            world_split=held_out_split,
            device_start_index=next_device_index,
        )
        # Advance device index by number of devices this gang made: unique device labels per world
        dev_acquired_count = sum(1 for e in plan.gang_device_events if e.kind == "device_acquired")
        next_device_index += dev_acquired_count

        if k >= 1 and account_market_enabled and resale_share is not None and market_rng is not None:
            offers = find_resale_offers(roles)
            gang_resales = plan_resales(plan.truth.roles, offers, resale_share, market_rng)
            for r in gang_resales:
                plan = apply_resale(plan, r)
                # Close the previous gang's span: valid_to = resale_minute (keep its lifecycle fields valid).
                for idx, s in enumerate(roles):
                    if s.gang_id == r.previous_gang and s.account_id == r.sold_account and s.valid_to is None:
                        roles[idx] = replace(s, valid_to=r.resale_minute)
                        break
                for idx, d in enumerate(dev_ctrl):
                    if d.gang_id == r.previous_gang and d.account_id == r.sold_account and d.valid_to is None:
                        dev_ctrl[idx] = replace(d, valid_to=r.resale_minute)
                all_resales.append(r)

        # Mules, victims and collectors of this gang are unavailable to later gangs; a person cannot hold two
        # overlapping roles in the truth.
        reserved.update(r.account_id for r in plan.truth.roles)

        # Refuse gang-planned route, scam, and cash-out payments without an illicit money rule.
        plan_rules_by_index = {rule.input_index: rule for rule in plan.money_rules}
        for i, _ in enumerate(plan.payments):
            story = plan.truth.payment_stories[i]
            if story.block in ("route", "scam", "cash_out") and i not in plan_rules_by_index:
                raise ValueError(f"Gang '{g.gang_id}' {story.block} payment at index {i} has no illicit money rule.")

        base_offset = len(pmts)
        for rule in plan.money_rules:
            rules.append(
                IllicitMoneyRule(
                    input_index=base_offset + rule.input_index,
                    spends_illicit=rule.spends_illicit,
                    credits_illicit=rule.credits_illicit,
                    requires_credit_from=rule.requires_credit_from,
                )
            )

        pmts.extend(plan.payments)
        roles.extend(plan.truth.roles)
        dev_ctrl.extend(plan.truth.device_control)
        stories.extend(plan.truth.payment_stories)
        g_dev_events.extend(plan.gang_device_events)
        opening_events.extend(plan.outside_events)
        withdrawals.extend(plan.withdrawals)

    gang_plan_roles = roles[len(fraud_inputs.truth.roles):]
    holdings = list(fraud_inputs.gang_holdings) + list(build_gang_holdings_from_roles(gang_plan_roles))

    new_truth = TruthFacts(roles=tuple(roles), device_control=tuple(dev_ctrl), payment_stories=tuple(stories))
    return active_state, FraudInputs(
        payments=tuple(pmts), outside_events=tuple(opening_events), truth=new_truth,
        gang_device_events=tuple(g_dev_events),
        money_rules=tuple(rules), illicit_seeds=tuple(seeds),
        withdrawals=tuple(withdrawals),
        gang_holdings=tuple(holdings),
        account_resales=tuple(all_resales),
    )


def run_interactive_world(
    fast_settings: FastRunSettings, runner_settings: RunnerSettings = RunnerSettings(),
    fraud_inputs: FraudInputs = FraudInputs(), hook: Callable[[PaymentView], HookDecision] = allow_all,
    configs: WorldConfigs | None = None, code_version: str | None = None, state_factory: Any = None,
    gang_configs: tuple[GangConfig, ...] = (),
    account_openings_config: dict[str, Any] | None = None,
    gang_reactions_config: dict[str, Any] | None = None,
    account_market_config: dict[str, Any] | None = None,
    *,
    world_split: str | None = None,
) -> InteractiveWorldOutput:
    """
    WHAT: Builds and runs a whole world: prepares it as Level 11 does, names the run (seed, effective configuration,
    code version), runs it through run_interactive, and returns everything including the device schedule.

    WHY: This is the one call later parts use. Passing the code version in lets a container state it; leaving it out
    asks find_code_version, which falls back to "unknown" rather than failing.

    Args:
        fast_settings: Seed, people, shops, days, range name.
        runner_settings: The fraud switch; off by default.
        fraud_inputs: Fraud payments and outside events (switch on only).
        hook: The bank hook; allow_all by default.
        configs: Layered configs, or None for the files on disk.
        code_version: The version to record, or None to look it up.
        state_factory: A function making the runner state from the prepared world (switch on only), or None.
        gang_configs: Gang configurations planned on child 7 and executed when fraud switch is on.
        account_openings_config: Optional account openings configuration dictionary.
        gang_reactions_config: Optional gang reactions configuration dictionary.
        account_market_config: Optional account market configuration dictionary.
        world_split: The split the world belongs to (training, validation, test or stress), or None to use
            fast_settings.range_name for both jobs as before. Recorded in the effective configuration only when given.

    Returns:
        The InteractiveWorldOutput.

    Raises:
        ValueError: For an unknown world_split, or a split paired with the wrong range column (training or validation
            with stress, stress with training).

    Example:
        run_interactive_world(FastRunSettings(101, 60, 12, 61, "training"), code_version="d41b176") gives 2,915
        events whose event lines hash to 24fc0e25009d7946...
    """
    if world_split is not None:
        _check_split_and_range(world_split, fast_settings.range_name)
    if not runner_settings.fraud_enabled and runner_settings.account_openings_enabled:
        raise ValueError("account openings need fraud_enabled=True")
    if not runner_settings.fraud_enabled and runner_settings.gang_reactions_enabled:
        raise ValueError("gang reactions need fraud_enabled=True")
    if not runner_settings.fraud_enabled and runner_settings.account_market_enabled:
        raise ValueError("account market needs fraud_enabled=True")
    if runner_settings.gang_reactions_enabled and not gang_configs:
        raise ValueError("gang reactions need at least one gang")
    if not runner_settings.fraud_enabled and gang_configs:
        raise ValueError(SWITCH_OFF_MESSAGE)
    if any(g.family is not None for g in gang_configs):
        families = load_tactic_families()
        eval_cfg = load_evaluation_config()
        check_family_lists_agree(families, eval_cfg)
        validate_gang_families(
            gang_configs, family_guard_split(world_split, fast_settings.range_name), families=families
        )
    runner_config = load_runner_config()
    effective_config = effective_config_for(
        fast_settings, configs, gang_configs,
        runner_settings=runner_settings, account_openings_config=account_openings_config,
        gang_reactions_config=gang_reactions_config,
        account_market_config=account_market_config,
        world_split=world_split,
    )
    version = code_version if code_version is not None else find_code_version(
        os.environ, REPO_ROOT, runner_config["code_version_git_timeout_seconds"]
    )
    run_id = make_run_id(fast_settings.seed, effective_config, version, runner_config["run_id_hex_length"])
    prepared = prepare_world(fast_settings, configs)
    state = state_factory(prepared) if state_factory is not None else None
    if gang_configs:
        state, fraud_inputs = _apply_gang_configs(
            prepared, gang_configs, state, fraud_inputs, fast_settings.seed, runner_config,
            account_market_enabled=runner_settings.account_market_enabled,
            account_market_config=account_market_config,
            held_out_split=family_guard_split(world_split, fast_settings.range_name),
        )
    result = run_interactive(
        prepared.world, prepared.plan, prepared.run_rng, prepared.device_events, prepared.episode_payments,
        settings=runner_settings, fraud_inputs=fraud_inputs, hook=hook, run_id=run_id,
        runner_config=runner_config, state=state, account_openings_config=account_openings_config,
        gang_reactions_config=gang_reactions_config,
    )
    return InteractiveWorldOutput(
        prepared.world, prepared.plan, prepared.episodes, prepared.episode_report, prepared.device_events,
        result, effective_config, version,
    )


def interactive_fast_output(output: InteractiveWorldOutput) -> Any:
    """
    WHAT: Wraps an interactive world's Level 11-shaped parts in Level 11's FastRunOutput record, so Level 11's own
    event_lines and hash functions can read it unchanged.

    WHY: Comparing the runner with Level 11 must use Level 11's own serialisation, not a copy of it. Handing it the
    record it expects is the simplest way.

    Args:
        output: An InteractiveWorldOutput.

    Returns:
        A FastRunOutput with the same world, plan, run result, episodes and episode report.

    Example:
        event_lines(interactive_fast_output(output)) gives the same 2,915 lines as event_lines(run_fast_world(...)).
    """
    # Imported here: fast_mode is only needed by readers of this helper.
    from world.fast_mode import FastRunOutput

    return FastRunOutput(
        world=output.world, plan=output.plan, result=output.result.run_result,
        episodes=output.episodes, episode_report=output.episode_report,
    )


def device_event_stream(result: InteractiveRunResult) -> list[dict[str, Any]]:
    """
    WHAT: Merges applied Level 11 device events and gang device events into a single chronological stream of plain
    dictionaries, ordering by minute, then Level 11 events first, then arrival order.

    WHY: Downstream detection engines and verification checks read one unified device event stream without needing
    to know internal simulator structures. The stream contains no gang identities, roles, or ground truth labels.

    Args:
        result: The InteractiveRunResult from an interactive world execution.

    Returns:
        List of plain dictionaries representing device lifecycle events, each containing a 'kind' attribute.

    Example:
        # In a run with 32 Level 11 events and 0 gang events:
        stream = device_event_stream(output.result)
        # len(stream) == 32 and "kind" in stream[0]
    """
    # 1. Tag each event with minute, source priority (0 for Level 11, 1 for gang), and original position.
    tagged_events: list[tuple[int, int, int, dict[str, Any]]] = [
        (e.minute, 0, idx, asdict(e)) for idx, e in enumerate(result.applied_device_events)
    ]
    tagged_events.extend(
        (e.minute, 1, idx, asdict(e)) for idx, e in enumerate(result.applied_gang_device_events)
    )
    # 2. Sort chronologically by minute, breaking ties with Level 11 first then original order.
    tagged_events.sort(key=lambda item: (item[0], item[1], item[2]))
    # 3. Strip sorting tags and return plain dictionary records.
    return [record for _, _, _, record in tagged_events]

