"""
WHAT: Tactical planning engine translating gang configurations into payments, hidden truth, and costs (Level 12b B-2a).
Produces immutable GangPlan instances containing scheduled fraud transfers, ground truth records, and resource costs.

WHY: Fraud simulation requires turning high-level tactical intent into concrete chronological events.
Simulating resource constraints through money fees, campaign time, and cumulative detection risks guarantees
that simulated syndicates operate within calibrated operational budgets per Decisions D7 and D8.

TERM: gang plan: complete campaign output containing planned payments, truth facts, block costs, and total expenditures.
TERM: block cost: three-dimensional expenditure of a single block step in money (paise), simulated time, and risk.
TERM: gang budget: maximum permitted campaign expenditure across money, time, and detection risk.
TERM: supplier fee: percentage fee paid to account suppliers based on money moved through acquired mule accounts.
"""

from dataclasses import dataclass
from typing import Any, Sequence
import numpy as np

from world.build_world import World
from world.calendar_runner import PlannedPayment
from world.gang_config import BlockStep, GangBudget, GangConfig, validate_gang_config
from world.gang_devices import (
    GangDeviceEvent,
    assign_devices_round_robin,
    build_device_control_spans,
    build_gang_device_events,
    make_gang_devices,
)
from world.tactic_blocks import (
    acquire_mule_accounts,
    advance_warmup_clock,
    calculate_send_minute,
    draw_hop_dwell_minutes,
    draw_route_stream_amount,
)
from world.tactic_config import (
    get_block_risk_range,
    get_block_time_range,
    get_disguise_jitter_range,
    get_gang_payment_purpose,
    get_supplier_fee_range,
)
from world.truth_records import DeviceControlSpan, PaymentStory, RoleSpan, TruthFacts

MINUTES_PER_DAY: int = 1440


class GangBudgetExceeded(ValueError):
    """
    WHAT: Exception raised when a gang campaign exceeds its allocated spending limits in money, time, or risk.

    WHY: Decision D8 requires strict enforcement of gang budgets; campaigns exceeding limits must be refused.

    Args:
        message: Diagnostic string identifying the breached resource, planned total, limit, and causal block.

    Returns:
        An initialized GangBudgetExceeded instance.

    Example:
        # raise GangBudgetExceeded("Gang 'gang_a' budget exceeded for money: planned 100 exceeds limit 50 at block 'route'.")
    """


@dataclass(frozen=True)
class BlockCost:
    """
    WHAT: Represents the resource expenditure incurred by executing a single tactic building block step.

    WHY: Decision D8 tracks multi-dimensional costs across transaction supplier fees, execution time, and risk.

    Args:
        money_paise: Expenditure in integer paise paid to account suppliers or operational intermediaries.
        time_minutes: Duration in simulated minutes elapsed during the execution of this step.
        risk: Probability of detection per use for this step within the interval [0.0, 1.0].

    Returns:
        An initialized immutable BlockCost record.

    Example:
        # cost = BlockCost(money_paise=40_000, time_minutes=120, risk=0.015)
    """

    money_paise: int
    time_minutes: int
    risk: float


@dataclass(frozen=True)
class GangPlan:
    """
    WHAT: Complete tactical campaign schedule containing scheduled payments, truth records, and resource costs.

    WHY: Encapsulates all simulation artifacts generated from a GangConfig specification for runner execution.

    Args:
        payments: Chronologically ordered tuple of scheduled PlannedPayment instances ready for runner injection.
        truth: Hidden truth container holding account roles, device compromises, and forensic stories.
        costs: Sequence of BlockCost records corresponding one-to-one with the gang's executed steps.
        totals: Aggregated BlockCost capturing cumulative money fees, campaign time, and combined risk.

    Returns:
        An initialized immutable GangPlan record.

    Example:
        # plan = plan_gang(gang_config, world, rng)
        # assert len(plan.payments) > 0 and plan.totals.money_paise <= gang_config.budget.money_paise
    """

    payments: tuple[PlannedPayment, ...]
    truth: TruthFacts
    costs: tuple[BlockCost, ...]
    totals: BlockCost
    gang_device_events: tuple[GangDeviceEvent, ...] = ()


def _check_running_budget(
    gang_id: str, budget: GangBudget, running_money: int, running_time: int, running_risk: float,
    block_name: str
) -> None:
    """
    WHAT: Asserts that cumulative campaign resource expenditures do not breach the configured GangBudget.
    If running money, time, or risk exceeds the respective budget limit, it raises a GangBudgetExceeded exception
    naming the resource, the planned total, the configured limit, and the tactical block that caused the breach.

    WHY: Decision D8 requires strict enforcement of budget limits; campaigns that exceed resource boundaries
    must be halted immediately with informative diagnostics before any payments are injected into the runner.

    Args:
        gang_id: Unique string identifier of the criminal gang being planned.
        budget: Configured GangBudget containing money, time, and risk ceilings.
        running_money: Cumulative transaction fees in paise incurred up to the current block.
        running_time: Cumulative elapsed simulated minutes spent up to the current block.
        running_risk: Cumulative detection probability accumulated up to the current block.
        block_name: Identifier string of the block whose execution triggered the budget check.

    Returns:
        None. Raises GangBudgetExceeded if any resource ceiling is breached.

    Example:
        # _check_running_budget("gang_a", budget, 500, 100, 0.01, "route")
    """
    if running_money > budget.money_paise:
        raise GangBudgetExceeded(
            f"Gang '{gang_id}' budget exceeded for money: "
            f"planned {running_money} paise exceeds limit {budget.money_paise} paise at block '{block_name}'."
        )
    if running_time > budget.time_minutes:
        raise GangBudgetExceeded(
            f"Gang '{gang_id}' budget exceeded for time: "
            f"planned {running_time} minutes exceeds limit {budget.time_minutes} minutes at block '{block_name}'."
        )
    if running_risk > budget.max_risk + 1e-9:
        raise GangBudgetExceeded(
            f"Gang '{gang_id}' budget exceeded for risk: "
            f"planned {running_risk:.6f} exceeds limit {budget.max_risk:.6f} at block '{block_name}'."
        )


def _build_route_roles(
    config: GangConfig, routed_accounts: list[str], layers: int, accounts_per_layer: int,
    acquisition_minute: int
) -> list[RoleSpan]:
    """
    WHAT: Constructs RoleSpan assertions for the fixture collector account and all routed citizen mule accounts.
    The collector receives role 'collector' at depth None valid from the fixture start minute; each intermediate
    and terminal mule account receives role 'mule' at its specific layer depth valid from acquisition minute.

    WHY: Ground Rule 15 and Decision D7 require explicit forensic attribution of accounts to criminal roles,
    layer depths, and temporal validity intervals to enable downstream audit and detection benchmarking.

    Args:
        config: Full GangConfig specification providing gang identifier and fixture source details.
        routed_accounts: Sequence of citizen account identifiers assigned to routing layers.
        layers: Integer number of hops in the laundering chain.
        accounts_per_layer: Integer number of parallel accounts operating in each layer.
        acquisition_minute: Simulation minute timestamp when mule accounts were acquired.

    Returns:
        List of initialized RoleSpan instances.

    Example:
        # roles = _build_route_roles(config, ["acc_m1", "acc_m2"], 2, 1, 100)
        # assert len(roles) == 3 and roles[0].role == "collector"
    """
    roles: list[RoleSpan] = [
        RoleSpan(config.source.account_id, "collector", config.gang_id, None, config.source.start_minute, None)
    ]
    for l in range(layers):
        layer_num = l + 1
        for acct in routed_accounts[l * accounts_per_layer : (l + 1) * accounts_per_layer]:
            roles.append(RoleSpan(acct, "mule", config.gang_id, layer_num, acquisition_minute, None))
    return roles


def _plan_single_hop(
    gang_id: str, sender: str, receiver: str, amount_paise: int, send_minute: int, layer: int
) -> tuple[PlannedPayment, PaymentStory]:
    """
    WHAT: Builds a single PlannedPayment transfer and its corresponding forensic PaymentStory narrative.
    It packages the sender, receiver, paise amount, dispatch minute, and purpose into a PlannedPayment record,
    and constructs a companion PaymentStory associating the transfer with the executing gang, block, and layer.

    WHY: Decision D7 requires every injected fraud transfer to carry forensic narrative metadata (PaymentStory)
    identifying the executing syndicate and tactical hop position for downstream detection evaluation.

    Args:
        gang_id: Identifier string of the criminal gang executing the payment.
        sender: Source account identifier string dispatching the funds.
        receiver: Destination account identifier string receiving the funds.
        amount_paise: Integer transfer amount in paise.
        send_minute: Simulation minute timestamp when the payment is dispatched.
        layer: Integer layer hop depth of the receiving account in the laundering chain.

    Returns:
        Tuple of (PlannedPayment, PaymentStory).

    Example:
        # pmt, story = _plan_single_hop("gang_a", "acc_1", "acc_2", 100000, 100, 1)
        # assert pmt.amount_paise == 100000 and story.layer == 1
    """
    purpose = get_gang_payment_purpose()
    payment = PlannedPayment(
        minute=send_minute, from_account=sender, to_account=receiver, amount_paise=amount_paise,
        creation_index=0, purpose=purpose
    )
    story = PaymentStory(
        gang_id=gang_id, block="route", layer=layer, description=f"Gang '{gang_id}' route to layer {layer}"
    )
    return payment, story


def _plan_route_stream(
    gang_id: str, source_account: str, stream_accounts: list[str], stream_amount: int,
    start_minute: int, dwell_param: int | None, has_disguise: bool, jitter_param: int | None,
    fixed_dwell: int | None, rng: np.random.Generator
) -> tuple[list[PlannedPayment], list[PaymentStory], int]:
    """
    WHAT: Generates planned payment transfers and forensic stories along a single multi-hop laundering stream.
    Transfers propagate from the fixture collector account to Layer 1, then forward sequentially through each
    downstream layer. Intermediate mule accounts forward the exact amount received, respecting dwell windows.

    WHY: Decision D7 models layered money movement as an unbroken forwarding chain where intermediate mules
    forward received funds within realistic holding windows, with optional timing disguise perturbations.

    Args:
        gang_id: String identifier of the gang orchestrating the laundering route.
        source_account: Identifier string of the fixture collector account funding the route.
        stream_accounts: List of account identifier strings assigned to this stream from Layer 1 to Layer N.
        stream_amount: Integer transfer amount in paise routed along this stream.
        start_minute: Simulation minute timestamp when the initial transfer is dispatched from the source.
        dwell_param: Optional configured dwell duration in minutes.
        has_disguise: Boolean indicating whether timing disguise is active.
        jitter_param: Optional maximum jitter window in minutes.
        fixed_dwell: Fixed dwell duration drawn once for strength 1 campaigns, or None.
        rng: Gang-specific random number generator stream.

    Returns:
        Tuple containing list of PlannedPayment records, list of PaymentStory records, and max elapsed dwell.

    Example:
        # payments, stories, elapsed = _plan_route_stream("gang_a", "src", ["m1", "m2"], 500000, 100, ...)
    """
    payments: list[PlannedPayment] = []
    stories: list[PaymentStory] = []
    current_sender = source_account
    receive_minute = start_minute
    total_dwell = 0

    for hop_idx, receiver in enumerate(stream_accounts):
        layer = hop_idx + 1
        if hop_idx == 0:
            send_minute = start_minute
        else:
            hop_dwell = fixed_dwell if fixed_dwell is not None else draw_hop_dwell_minutes(dwell_param, rng)
            send_minute = calculate_send_minute(receive_minute, hop_dwell, has_disguise, jitter_param, rng)
            total_dwell += hop_dwell

        pmt, story = _plan_single_hop(gang_id, current_sender, receiver, stream_amount, send_minute, layer)
        payments.append(pmt)
        stories.append(story)
        current_sender = receiver
        receive_minute = send_minute

    return payments, stories, total_dwell


def _execute_route_step(
    config: GangConfig, step: BlockStep, acquired_accounts: list[str], acquisition_minute: int,
    current_minute: int, rng: np.random.Generator
) -> tuple[list[PlannedPayment], list[RoleSpan], list[PaymentStory], BlockCost]:
    """
    WHAT: Orchestrates multi-hop laundering transfers across layered mule accounts, assigning forensic roles and costs.
    It partitions acquired accounts across layers, draws transfer amounts and dwell times per Decision D7,
    dispatches parallel laundering streams, verifies collector opening balances, and assesses fee and risk costs.

    WHY: Decision D7 establishes layered routing as the core mechanism for concealing money provenance,
    connecting compromised collector funds to intermediate mules and terminal cash-out accounts.

    Args:
        config: Full GangConfig specification providing budget, source, strength, and step itinerary.
        step: BlockStep record for the route tactical block containing routing parameters.
        acquired_accounts: Sequence of citizen account identifiers previously acquired by the gang.
        acquisition_minute: Simulation minute timestamp when mule accounts were acquired.
        current_minute: Current simulation minute timestamp of the gang campaign.
        rng: Gang-specific random number generator stream.

    Returns:
        Tuple of (planned payments list, role spans list, payment stories list, and route BlockCost).

    Example:
        # payments, roles, stories, cost = _execute_route_step(config, step, mules, 100, 100, rng)
    """
    params = step.params_dict
    layers, accounts_per_layer = params.get("layers", 1), params.get("accounts_per_layer", 1)
    needed = layers * accounts_per_layer
    if len(acquired_accounts) < needed:
        raise ValueError(f"Gang '{config.gang_id}' needs {needed} accounts for routing but has {len(acquired_accounts)}.")

    disguise_step = next((s for s in config.steps if s.block == "disguise"), None)
    has_disguise = disguise_step is not None
    jitter = disguise_step.params_dict.get("jitter_minutes") if disguise_step else None
    fix_dwell = draw_hop_dwell_minutes(params.get("dwell_minutes"), rng) if config.strength == 1 else None
    fix_amt = draw_route_stream_amount(config.strength, params.get("amount_inr"), rng) if config.strength == 1 else None

    routed = acquired_accounts[:needed]
    roles = _build_route_roles(config, routed, layers, accounts_per_layer, acquisition_minute)
    pmts, stories, total_paise, max_dwell = _plan_all_route_streams(
        config, routed, layers, accounts_per_layer, fix_amt, fix_dwell, current_minute, params,
        has_disguise, jitter, rng
    )

    if total_paise > config.source.opening_balance_paise:
        raise ValueError(f"Gang '{config.gang_id}' routes {total_paise} paise exceeding source balance.")

    cost = _calculate_route_costs(total_paise, max_dwell, rng)
    return pmts, roles, stories, cost


def _plan_all_route_streams(
    config: GangConfig, routed: list[str], layers: int, accounts_per_layer: int,
    fix_amt: int | None, fix_dwell: int | None, current_minute: int, params: dict[str, Any],
    has_disguise: bool, jitter: int | None, rng: np.random.Generator
) -> tuple[list[PlannedPayment], list[PaymentStory], int, int]:
    """
    WHAT: Iterates across parallel laundering streams to synthesize scheduled transfers and forensic stories.
    It slices accounts for each stream, determines per-stream transfer sums, dispatches transfers, and tracks dwell.

    WHY: Isolates multi-stream iteration from route step validation and accounting logic.

    Args:
        config: Full GangConfig specification providing gang and fixture source identifiers.
        routed: Flat list of account identifiers allocated across all layers.
        layers: Integer number of hops in the laundering chain.
        accounts_per_layer: Number of parallel streams.
        fix_amt: Fixed transfer amount in paise for strength 1 campaigns, or None.
        fix_dwell: Fixed dwell duration in minutes for strength 1 campaigns, or None.
        current_minute: Simulation minute timestamp when routing initiates.
        params: Dictionary of parameters configured for the route block.
        has_disguise: Boolean indicating whether timing camouflage is active.
        jitter: Optional maximum jitter window in minutes.
        rng: Gang-specific random number generator stream.

    Returns:
        Tuple of (payments list, stories list, total paise routed, and maximum dwell minutes).

    Example:
        # pmts, stories, paise, dwell = _plan_all_route_streams(config, routed, 2, 1, ...)
    """
    pmts: list[PlannedPayment] = []
    stories: list[PaymentStory] = []
    total_paise, max_dwell = 0, 0

    for k in range(accounts_per_layer):
        accts = [routed[l * accounts_per_layer + k] for l in range(layers)]
        amt = fix_amt if fix_amt is not None else draw_route_stream_amount(config.strength, params.get("amount_inr"), rng)
        total_paise += amt
        s_pmts, s_strs, s_dwell = _plan_route_stream(
            config.gang_id, config.source.account_id, accts, amt, current_minute,
            params.get("dwell_minutes"), has_disguise, jitter, fix_dwell, rng
        )
        pmts.extend(s_pmts)
        stories.extend(s_strs)
        max_dwell = max(max_dwell, s_dwell)

    return pmts, stories, total_paise, max_dwell


def _calculate_route_costs(total_paise: int, max_dwell: int, rng: np.random.Generator) -> BlockCost:
    """
    WHAT: Computes the operational transaction fee and detection risk costs for a route execution step.
    It samples supplier fee percentage within [0.03, 0.06] from tactic costs and samples route detection risk.

    WHY: Decision D8 isolates financial transaction fee evaluation and detection risk sampling from transfer dispatch.

    Args:
        total_paise: Total money amount routed through acquired mule accounts in integer paise.
        max_dwell: Maximum dwell duration in simulated minutes observed across parallel streams.
        rng: Gang-specific random number generator stream.

    Returns:
        BlockCost record containing fee money in paise, dwell time in minutes, and route risk probability.

    Example:
        # cost = _calculate_route_costs(1_000_000, 30, rng)
        # assert cost.money_paise > 0 and cost.time_minutes == 30
    """
    min_fee, max_fee = get_supplier_fee_range()
    fee_share = float(rng.uniform(min_fee, max_fee))
    money_cost = int(round(fee_share * total_paise))
    min_risk, max_risk = get_block_risk_range("route")
    route_risk = float(rng.uniform(min_risk, max_risk))
    return BlockCost(money_cost, max_dwell, route_risk)


def _plan_acquire_step(
    world: World, config: GangConfig, step: BlockStep, clock_minute: int,
    reserved_accounts: set[str] | None, rng: np.random.Generator
) -> tuple[list[str], int, BlockCost]:
    """
    WHAT: Executes the account acquisition building block step, recruiting citizen accounts and assessing costs.
    It pulls citizen accounts from the world while excluding reserved fixtures, records the acquisition timestamp,
    and samples procedural time and risk expenditures from calibrated policy ranges.

    WHY: Decision D7 requires gangs to recruit authentic citizen accounts before attempting laundering routes,
    incurring realistic operational time and detection risk as specified in Decision D8.

    Args:
        world: Simulated World instance containing people and shops.
        config: GangConfig specification identifying the gang and source account.
        step: BlockStep for the acquire block containing recruitment count.
        clock_minute: Current simulation minute timestamp of the gang campaign.
        reserved_accounts: Set of account identifiers reserved by prior gangs.
        rng: Gang-specific random number generator stream.

    Returns:
        Tuple of (list of acquired account IDs, acquisition minute timestamp, and acquire BlockCost).

    Example:
        # accts, minute, cost = _plan_acquire_step(world, config, step, 100, None, rng)
    """
    count = step.params_dict.get("count", 1)
    excluded = {config.source.account_id} | (reserved_accounts or set())
    acquired = acquire_mule_accounts(world, count, rng, excluded)
    min_time, max_time = get_block_time_range("acquire")
    time_cost = int(rng.integers(min_time, max_time + 1))
    min_risk, max_risk = get_block_risk_range("acquire")
    risk_cost = float(rng.uniform(min_risk, max_risk))
    return acquired, clock_minute, BlockCost(0, time_cost, risk_cost)


def _plan_warmup_step(
    step: BlockStep, clock_minute: int, rng: np.random.Generator
) -> tuple[int, BlockCost]:
    """
    WHAT: Executes the account seasoning building block step, advancing the gang clock and evaluating costs.
    It advances the campaign simulation clock forward by the configured seasoning days and samples procedural risk.

    WHY: Decision D7 models account seasoning (warm_up) as an elapsed time barrier where newly leased
    accounts remain dormant to evade heuristic new-account detection before active routing begins.

    Args:
        step: BlockStep for the warm_up block containing seasoning days.
        clock_minute: Current simulation minute timestamp of the gang campaign.
        rng: Gang-specific random number generator stream.

    Returns:
        Tuple of (new simulation minute timestamp, and warm_up BlockCost).

    Example:
        # new_min, cost = _plan_warmup_step(step, 100, rng)
    """
    days = step.params_dict.get("days", 1)
    new_clock = advance_warmup_clock(clock_minute, days)
    min_risk, max_risk = get_block_risk_range("warm_up")
    risk_cost = float(rng.uniform(min_risk, max_risk))
    return new_clock, BlockCost(0, days * MINUTES_PER_DAY, risk_cost)


def _plan_disguise_step(
    step: BlockStep, rng: np.random.Generator
) -> BlockCost:
    """
    WHAT: Evaluates resource costs incurred by executing the disguise timing perturbation building block.
    It calculates simulated execution time from configured jitter parameters and samples detection risk.

    WHY: Decision D7 models timing disguise as an intentional evasion mechanism; assessing resource costs
    ensures sophisticated camouflaged campaigns pay appropriate operational time and risk fees.

    Args:
        step: BlockStep for the disguise block containing jitter parameters.
        rng: Gang-specific random number generator stream.

    Returns:
        BlockCost record representing disguise resource expenditure.

    Example:
        # cost = _plan_disguise_step(step, rng)
        # assert cost.money_paise == 0 and cost.time_minutes >= 0
    """
    jitter = step.params_dict.get("jitter_minutes")
    min_jitter, max_jitter = get_disguise_jitter_range()
    time_cost = jitter if jitter is not None else int(rng.integers(min_jitter, max_jitter + 1))
    min_risk, max_risk = get_block_risk_range("disguise")
    risk_cost = float(rng.uniform(min_risk, max_risk))
    return BlockCost(0, time_cost, risk_cost)


def _plan_take_control_step(
    config: GangConfig, step: BlockStep, acquired_accounts: list[str], clock_minute: int,
    gang_devices_rng: np.random.Generator, rng: np.random.Generator
) -> tuple[list[DeviceControlSpan], list[GangDeviceEvent], BlockCost]:
    """
    WHAT: Synthesizes external gang devices, round-robin account assignments, truth spans, and telemetry events.
    It provisions devices using child 6 gang_devices stream, assigns them cyclically across acquired mule accounts,
    records ground truth DeviceControlSpan records, and creates client-side GangDeviceEvent takeover telemetry.

    WHY: Decision E1, E2, E3, and E4 model criminal device control infrastructure, establishing observable
    compromise signals and ground truth device groupings for evaluation benchmarks without leaking truth labels.

    Args:
        config: Full GangConfig specification providing gang identity.
        step: BlockStep for the take_control block containing device count and compromise flags.
        acquired_accounts: Sequence of citizen account identifiers previously acquired by the gang.
        clock_minute: Current simulation minute timestamp of the gang campaign.
        gang_devices_rng: Dedicated random number generator stream for device creation (child 6).
        rng: Gang-specific random number generator stream for time and risk expenditures.

    Returns:
        Tuple of (list of DeviceControlSpan records, list of GangDeviceEvent records, and take_control BlockCost).

    Example:
        # spans, events, cost = _plan_take_control_step(config, step, mules, 100, dev_rng, rng)
    """
    # 1. Extract device count and boolean compromise flags from parameters.
    devices_count = step.params_dict.get("devices", 1)
    sim_swap = bool(step.params_dict.get("sim_swap", False))
    linked_number_switch = bool(step.params_dict.get("linked_number_switch", False))
    emulator = bool(step.params_dict.get("emulator", False))
    remote_access = bool(step.params_dict.get("remote_access", False))

    # 2. Provision devices on child stream 6 and allocate across mules in round-robin order.
    devices = make_gang_devices(devices_count, gang_devices_rng)
    account_to_device = assign_devices_round_robin(acquired_accounts, devices)

    # 3. Build forensic ground truth spans and synthetic client telemetry events.
    spans = build_device_control_spans(
        config.gang_id, account_to_device, clock_minute, sim_swap, linked_number_switch, emulator, remote_access
    )
    events = build_gang_device_events(
        devices, account_to_device, clock_minute, sim_swap, linked_number_switch, emulator, remote_access
    )

    # 4. Sample execution duration and risk from calibrated take_control ranges.
    min_time, max_time = get_block_time_range("take_control")
    time_cost = int(rng.integers(min_time, max_time + 1))
    min_risk, max_risk = get_block_risk_range("take_control")
    risk_cost = float(rng.uniform(min_risk, max_risk))
    cost = BlockCost(0, time_cost, risk_cost)

    return spans, events, cost


def _plan_campaign_steps(
    config: GangConfig, world: World, rng: np.random.Generator,
    reserved_accounts: set[str] | None, dev_rng: np.random.Generator
) -> tuple[Any, ...]:
    """
    WHAT: Iterates through configured block steps, dispatching procedural handlers and accumulating expenditures.
    It tracks clock minutes, recruited mules, truth spans, telemetry events, and checks running budget limits.

    WHY: Separates multi-block campaign step dispatch from top-level plan assembly and validation.

    Args:
        config: Immutable GangConfig defining gang itinerary and limits.
        world: Simulated World instance containing people and shops.
        rng: Master random generator stream for gang planning.
        reserved_accounts: Set of account identifiers reserved by prior gangs.
        dev_rng: Dedicated random generator stream for gang devices.

    Returns:
        Tuple of (payments, roles, spans, stories, costs, money, time, prob_safe, events).

    Example:
        # res = _plan_campaign_steps(config, world, rng, None, dev_rng)
    """
    clock_min, acq_min = config.source.start_minute, config.source.start_minute
    mules: list[str] = []
    device_spans: list[DeviceControlSpan] = []
    device_events: list[GangDeviceEvent] = []
    pmts, roles, stories, costs = [], [], [], []
    money, time, prob_safe = 0, 0, 1.0

    for step in config.steps:
        if step.block == "acquire":
            mules, acq_min, cost = _plan_acquire_step(world, config, step, clock_min, reserved_accounts, rng)
        elif step.block == "take_control":
            d_spans, d_events, cost = _plan_take_control_step(config, step, mules, clock_min, dev_rng, rng)
            device_spans.extend(d_spans); device_events.extend(d_events)
        elif step.block == "warm_up":
            clock_min, cost = _plan_warmup_step(step, clock_min, rng)
        elif step.block == "route":
            r_pmts, r_roles, r_stories, cost = _execute_route_step(config, step, mules, acq_min, clock_min, rng)
            pmts.extend(r_pmts); roles.extend(r_roles); stories.extend(r_stories)
        elif step.block == "disguise":
            cost = _plan_disguise_step(step, rng)
        else:
            raise ValueError(f"Unrecognized block '{step.block}'.")

        costs.append(cost)
        money += cost.money_paise; time += cost.time_minutes; prob_safe *= (1.0 - cost.risk)
        _check_running_budget(config.gang_id, config.budget, money, time, 1.0 - prob_safe, step.block)

    return pmts, roles, device_spans, stories, costs, money, time, prob_safe, device_events


def plan_gang(
    config: GangConfig, world: World, rng: np.random.Generator,
    reserved_accounts: set[str] | None = None,
    gang_devices_rng: np.random.Generator | None = None,
) -> GangPlan:
    """
    WHAT: Synthesizes a complete tactical campaign plan from a GangConfig specification and simulated World state.
    It validates structural rules, recruits authentic citizen mule accounts, simulates seasoning dormancy,
    provisions gang devices and compromise telemetry, schedules layered transfers, and enforces budget constraints.

    WHY: Ground Rule 15 and Decisions D7 and D8 establish procedural gang generation as the bridge between
    high-level fraud typologies and low-level reactive world execution, guaranteeing economic and temporal consistency.

    Args:
        config: Immutable GangConfig defining gang identity, strength, budget, fixture source, and steps.
        world: Simulated World instance containing citizen accounts, commercial shops, and ledgers.
        rng: Master random number generator stream assigned to this gang's planning process.
        reserved_accounts: Optional set of account identifiers reserved by prior gangs or system fixtures.
        gang_devices_rng: Optional dedicated random number generator stream for device creation (child 6).

    Returns:
        An initialized GangPlan containing planned transfers, truth assertions, step costs, and total expenditures.

    Example:
        # plan = plan_gang(gang_config, world, gang_rng)
        # assert len(plan.payments) > 0 and plan.totals.money_paise <= gang_config.budget.money_paise
    """
    validate_gang_config(config)
    dev_rng = gang_devices_rng if gang_devices_rng is not None else rng
    pmts, roles, spans, stories, costs, money, time, safe, events = _plan_campaign_steps(
        config, world, rng, reserved_accounts, dev_rng
    )
    return _build_gang_plan(pmts, roles, spans, stories, costs, money, time, safe, events)


def _build_gang_plan(
    pmts: list[PlannedPayment], roles: list[RoleSpan], device_spans: list[DeviceControlSpan],
    stories: list[PaymentStory], costs: list[BlockCost], money: int, time: int, prob_safe: float,
    gang_device_events: list[GangDeviceEvent] | None = None,
) -> GangPlan:
    """
    WHAT: Packages planned payments, truth facts, block costs, and total expenditures into an immutable GangPlan.
    It indexes planned payments sequentially from zero and groups role spans, device spans, and stories into TruthFacts.

    WHY: Standardizes final plan construction to guarantee consistent container indexing and immutable records.

    Args:
        pmts: List of PlannedPayment records scheduled during campaign planning.
        roles: List of RoleSpan records identifying account role assignments.
        device_spans: List of DeviceControlSpan records identifying gang device control.
        stories: List of PaymentStory records documenting transaction narratives.
        costs: List of BlockCost records for individual executed blocks.
        money: Total cumulative transaction fees in paise.
        time: Total cumulative elapsed campaign minutes.
        prob_safe: Cumulative probability of evading detection across all blocks.
        gang_device_events: Optional list of GangDeviceEvent records emitted during device takeover.

    Returns:
        Immutable GangPlan record ready for runner execution.

    Example:
        # plan = _build_gang_plan(pmts, roles, [], stories, costs, 500, 100, 0.98)
        # assert len(plan.payments) == len(pmts)
    """
    indexed = tuple(
        PlannedPayment(p.minute, p.from_account, p.to_account, p.amount_paise, idx, p.purpose)
        for idx, p in enumerate(pmts)
    )
    truth = TruthFacts(roles=tuple(roles), device_control=tuple(device_spans), payment_stories=tuple(stories))
    events = tuple(gang_device_events or [])
    return GangPlan(indexed, truth, tuple(costs), BlockCost(money, time, 1.0 - prob_safe), events)
