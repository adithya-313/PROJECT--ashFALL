"""
WHAT: Plans the scam step of a gang (Level 12c C-2): picks victims from the ordinary people of the world, gives each
victim a scam type and a payment schedule, and turns the schedules into planned payments, illicit money rules,
payment stories and victim role spans. It also reports, for each receiving mule, when its last victim payment
arrives and how much illicit money it is planned to hold afterwards.

WHY: Victims are the gang's money source. Everything a victim pays and every payout back is decided here from
config/scam_schedules.yaml, so the route can later forward exactly what the victims paid and nothing else.

TERM: receiving mule: an acquired mule account that victims pay.
TERM: victim: an ordinary person, never a mule or a fixture source, who pays a receiving mule because of a scam.
TERM: scam type: investment, job or unauthorised_debit; it decides the shape of the payment schedule.
TERM: payout back: a small payment from the receiving mule back to the victim, after the first payment and before
      the last, to build trust for later payments.
TERM: inflow: what a receiving mule is planned to hold after all victim payments and payouts: payments minus payouts.
TERM: paise: Rs 1 = 100 paise.
"""

from dataclasses import dataclass

import numpy as np

from world.build_world import World
from world.calendar_runner import PlannedPayment
from world.gang_config import BlockStep, GangConfig
from world.interactive_records import IllicitMoneyRule
from world.money import rupees_to_paise
from world.scam_config import (
    resolve_debit_amount_inr,
    resolve_debit_payment_count,
    resolve_debit_pays_back,
    resolve_debit_spread_minutes,
    resolve_first_payment_amount_inr,
    resolve_investment_or_job_share,
    resolve_investment_share_of_investment_or_job,
    resolve_min_minutes_between_victim_payments,
    resolve_payment_escalation_multiple,
    resolve_payment_spread_days,
    resolve_payments_per_victim,
    resolve_payout_back_share,
    resolve_remainder_scam_type,
)
from world.tactic_config import get_block_risk_range, get_gang_payment_purpose, get_scam_victims_per_mule_range
from world.truth_records import PaymentStory, RoleSpan

MINUTES_PER_DAY: int = 1440
SCAM_INVESTMENT: str = "investment"
SCAM_JOB: str = "job"


@dataclass(frozen=True)
class ScamPayment:
    """
    WHAT: One planned money movement of one victim's scam: a victim payment to the receiving mule or the payout back.

    WHY: Keeps a schedule as plain data, so schedule rules can be tested before any payment object exists.

    Args:
        minute: Simulation minute at which the payment is dispatched.
        amount_paise: Planned amount in paise, at least 1.
        is_payout: True for the payout from the mule back to the victim, False for a victim payment.

    Returns:
        An initialized immutable ScamPayment.

    Example:
        ScamPayment(1500, 250000, False) is a victim payment of Rs 2,500 at minute 1500.
    """

    minute: int
    amount_paise: int
    is_payout: bool


@dataclass(frozen=True)
class VictimSchedule:
    """
    WHAT: Everything one victim does in a scam: who the victim is, which receiving mule is paid, the scam type, and
    the payments in minute order (victim payments plus at most one payout back).

    WHY: One record per victim is the unit from which payments, stories, rules and roles are all built.

    Args:
        victim_account: The victim's account id.
        receiving_mule: The receiving mule's account id.
        scam_type: investment, job, or the remainder type from config.
        payments: Payments in minute order; the first one is always a victim payment.

    Returns:
        An initialized immutable VictimSchedule.

    Example:
        VictimSchedule("acc_person_000007", "acc_person_000003", "job", (ScamPayment(1440, 300000, False),))
    """

    victim_account: str
    receiving_mule: str
    scam_type: str
    payments: tuple[ScamPayment, ...]


@dataclass(frozen=True)
class MuleInflow:
    """
    WHAT: What the route needs to know about one receiving mule: the minute its last victim payment is planned and
    the money it is planned to hold afterwards (victim payments minus payouts back).

    WHY: The route forwards exactly this amount, starting a dwell after the last victim payment.

    Args:
        last_victim_payment_minute: Minute of the mule's latest planned victim payment.
        net_paise: Planned victim payments minus planned payouts, in paise.

    Returns:
        An initialized immutable MuleInflow.

    Example:
        MuleInflow(last_victim_payment_minute=9000, net_paise=450000)
    """

    last_victim_payment_minute: int
    net_paise: int


@dataclass(frozen=True)
class ScamPlan:
    """
    WHAT: The planned output of one scam step: payments, aligned stories and money rules, victim role spans, each
    receiving mule's inflow, and the step's duration and risk.

    WHY: The gang planner appends these to its own lists; rule input indexes are local to this plan's payments and
    the planner shifts them by the number of payments planned before.

    Args:
        payments: Planned payments in victim order, then minute order.
        stories: One PaymentStory per payment, same order.
        rules: One IllicitMoneyRule per payment, same order, with input_index equal to the payment's position here.
        roles: One victim RoleSpan per victim.
        mule_inflows: Receiving mule account id -> MuleInflow.
        span_minutes: Minutes from the step start to the latest payment.
        risk: Detection risk of the step, drawn from the scam risk range.

    Returns:
        An initialized immutable ScamPlan.

    Example:
        plan = plan_scam_step(world, config, step, mules, 1440, None, rng)
        assert len(plan.payments) == len(plan.stories) == len(plan.rules)
    """

    payments: tuple[PlannedPayment, ...]
    stories: tuple[PaymentStory, ...]
    rules: tuple[IllicitMoneyRule, ...]
    roles: tuple[RoleSpan, ...]
    mule_inflows: dict[str, MuleInflow]
    span_minutes: int
    risk: float


@dataclass(frozen=True)
class TypeChances:
    """
    WHAT: The two chances, drawn once per scam step, that decide every victim's scam type.

    WHY: Drawing the chances once gives the step one mix of scam types; each victim then draws against that mix.

    Args:
        investment_or_job: Chance that a victim's scam is investment or job.
        investment_within: Chance, given investment or job, that it is investment.

    Returns:
        An initialized immutable TypeChances.

    Example:
        TypeChances(investment_or_job=0.7, investment_within=0.5)
    """

    investment_or_job: float
    investment_within: float


def draw_type_chances(split: str, rng: np.random.Generator) -> TypeChances:
    """
    WHAT: Draws the two scam-type chances uniformly from their configured ranges for the split.

    WHY: The ranges are config values; the step needs one concrete pair to classify its victims.

    Args:
        split: Split name, e.g. "training".
        rng: The gang planning stream (child 7).

    Returns:
        The drawn TypeChances. Two numbers are read from rng, investment_or_job first.

    Example:
        draw_type_chances("training", np.random.default_rng(7)).investment_or_job is between 0.55 and 0.8
    """
    low_job, high_job = resolve_investment_or_job_share(split)
    low_inv, high_inv = resolve_investment_share_of_investment_or_job(split)
    investment_or_job = float(rng.uniform(low_job, high_job))
    investment_within = float(rng.uniform(low_inv, high_inv))
    return TypeChances(investment_or_job, investment_within)


def draw_scam_type(chances: TypeChances, rng: np.random.Generator) -> str:
    """
    WHAT: Draws one victim's scam type: with the investment-or-job chance it is investment or job (split by the
    second chance), otherwise it is the remainder type from config.

    WHY: Reading the second number only for investment-or-job victims keeps the draw order easy to follow.

    Args:
        chances: The step's TypeChances.
        rng: The gang planning stream (child 7).

    Returns:
        "investment", "job", or the remainder type. One number is read, a second for investment or job.

    Example:
        draw_scam_type(TypeChances(1.0, 1.0), rng) == "investment"
    """
    first_draw = float(rng.uniform(0.0, 1.0))
    if first_draw >= chances.investment_or_job:
        return resolve_remainder_scam_type()
    second_draw = float(rng.uniform(0.0, 1.0))
    if second_draw < chances.investment_within:
        return SCAM_INVESTMENT
    return SCAM_JOB


def choose_victims(world: World, count: int, excluded: set[str], rng: np.random.Generator) -> list[str]:
    """
    WHAT: Picks count distinct ordinary people as victims: person accounts only, never an excluded account.
    Candidates are sorted before the draw so the same seed gives the same victims on every platform.

    WHY: Victims must be ordinary people; mules, fixture sources and accounts reserved by other gangs are excluded.

    Args:
        world: The World whose people are the candidates.
        count: How many victims to pick.
        excluded: Account ids that must not be picked.
        rng: The gang planning stream (child 7).

    Returns:
        The chosen account ids, in draw order.

    Raises:
        ValueError: If fewer than count candidates remain.

    Example:
        choose_victims(world, 3, {"acc_person_000001"}, rng) returns three other person accounts.
    """
    eligible = sorted(person.account_id for person in world.people if person.account_id not in excluded)
    if len(eligible) < count:
        raise ValueError(f"Cannot pick {count} scam victims; only {len(eligible)} eligible people are available.")
    chosen_positions = rng.choice(len(eligible), size=count, replace=False)
    return [eligible[int(position)] for position in chosen_positions]


def _draw_offsets_after_first(
    count_after_first: int, spread_minutes: int, minimum_gap: int, rng: np.random.Generator
) -> list[int]:
    """
    WHAT: Draws the minute offsets, from the first payment, of the payments after the first. Offsets are sorted
    and then pushed apart so consecutive payments are at least minimum_gap minutes apart.

    WHY: The gap guarantees room for a payout strictly between the first and the last payment.

    Args:
        count_after_first: How many payments follow the first.
        spread_minutes: Largest raw offset to draw.
        minimum_gap: Smallest allowed distance between consecutive payments, in minutes (0 for no rule).
        rng: The gang planning stream (child 7).

    Returns:
        count_after_first offsets in increasing order; the first payment itself is at offset 0 and not listed.

    Example:
        With spread 0 and gap 2, three later payments get offsets [2, 4, 6].
    """
    if count_after_first == 0:
        return []
    raw_offsets = sorted(int(value) for value in rng.integers(0, spread_minutes + 1, size=count_after_first))
    offsets: list[int] = []
    previous_offset = 0
    for raw_offset in raw_offsets:
        offset = max(raw_offset, previous_offset + minimum_gap)
        offsets.append(offset)
        previous_offset = offset
    return offsets


def plan_payout_back(payments: list[ScamPayment], split: str, rng: np.random.Generator) -> ScamPayment | None:
    """
    WHAT: Plans the one payout back, if the schedule leaves room for it: at a minute strictly after the first
    payment and strictly before the last, for a drawn share of the victim payments made before that minute,
    rounded down to whole paise.

    WHY: Payouts build trust for the later payments. With one payment, or payments closer than two minutes, there
    is no minute strictly between them, so there is no payout. A share that rounds to zero paise is also no payout.

    Args:
        payments: The victim's payments in minute order, victim payments only.
        split: Split name.
        rng: The gang planning stream (child 7).

    Returns:
        The payout as a ScamPayment with is_payout True, or None. When there is room, reads the minute and then the
        share from rng.

    Example:
        Payments of Rs 1,000 at minute 0 and Rs 2,000 at minute 500 give a payout between minutes 1 and 499.
    """
    first_minute = payments[0].minute
    last_minute = payments[-1].minute
    if last_minute - first_minute < 2:
        return None
    payout_minute = int(rng.integers(first_minute + 1, last_minute))
    low_share, high_share = resolve_payout_back_share(split)
    share = float(rng.uniform(low_share, high_share))
    paid_so_far = sum(payment.amount_paise for payment in payments if payment.minute < payout_minute)
    payout_paise = int(share * paid_so_far)
    if payout_paise <= 0:
        return None
    return ScamPayment(payout_minute, payout_paise, True)


def plan_investment_or_job_payments(first_minute: int, split: str, rng: np.random.Generator) -> list[ScamPayment]:
    """
    WHAT: Plans the payments of an investment or job victim. The count and the spread in days are drawn; the first
    payment is at first_minute for a drawn amount; each later payment is the previous amount times one drawn
    escalation multiple, rounded down to whole paise; later payments sit at sorted drawn offsets inside the spread,
    pushed apart by the minimum gap. A payout back is added when there is room.

    WHY: These scams escalate over days and pay a little back to keep the victim paying.

    Args:
        first_minute: Minute of the victim's first payment.
        split: Split name.
        rng: The gang planning stream (child 7).

    Returns:
        The payments in minute order, victim payments plus the payout back if any.

    Example:
        A victim with 3 payments, first Rs 3,000 and multiple 1.5, pays 300000, 450000 and 675000 paise.
    """
    low_count, high_count = resolve_payments_per_victim(split)
    low_days, high_days = resolve_payment_spread_days(split)
    low_amount, high_amount = resolve_first_payment_amount_inr(split)
    low_multiple, high_multiple = resolve_payment_escalation_multiple(split)

    payment_count = int(rng.integers(low_count, high_count + 1))
    spread_days = int(rng.integers(low_days, high_days + 1))
    first_amount_paise = rupees_to_paise(float(rng.uniform(low_amount, high_amount)))
    escalation_multiple = float(rng.uniform(low_multiple, high_multiple))

    minimum_gap = resolve_min_minutes_between_victim_payments()
    offsets = _draw_offsets_after_first(payment_count - 1, spread_days * MINUTES_PER_DAY, minimum_gap, rng)

    payments = [ScamPayment(first_minute, first_amount_paise, False)]
    amount_paise = first_amount_paise
    for offset in offsets:
        amount_paise = int(amount_paise * escalation_multiple)
        payments.append(ScamPayment(first_minute + offset, amount_paise, False))

    payout = plan_payout_back(payments, split, rng)
    if payout is not None:
        payments.append(payout)
        payments.sort(key=lambda payment: payment.minute)
    return payments


def plan_unauthorised_debit_payments(first_minute: int, split: str, rng: np.random.Generator) -> list[ScamPayment]:
    """
    WHAT: Plans the payments of an unauthorised-debit victim. The count and the spread in minutes are drawn; the
    first debit is at first_minute; later debits sit at sorted drawn offsets inside the spread; every debit has its
    own drawn amount. A payout back is planned only if config says debit victims get one.

    WHY: After an account takeover the debits follow within minutes, with no escalation story.

    Args:
        first_minute: Minute of the victim's first debit.
        split: Split name.
        rng: The gang planning stream (child 7).

    Returns:
        The payments in minute order, debits plus the payout back if config allows one and there is room.

    Example:
        A victim with 2 debits and spread 30 pays at first_minute and at first_minute plus 0 to 30 minutes.
    """
    low_count, high_count = resolve_debit_payment_count(split)
    low_spread, high_spread = resolve_debit_spread_minutes(split)
    low_amount, high_amount = resolve_debit_amount_inr(split)

    payment_count = int(rng.integers(low_count, high_count + 1))
    spread_minutes = int(rng.integers(low_spread, high_spread + 1))
    offsets = _draw_offsets_after_first(payment_count - 1, spread_minutes, 0, rng)

    payments: list[ScamPayment] = []
    for offset in [0] + offsets:
        amount_paise = rupees_to_paise(float(rng.uniform(low_amount, high_amount)))
        payments.append(ScamPayment(first_minute + offset, amount_paise, False))

    if resolve_debit_pays_back():
        payout = plan_payout_back(payments, split, rng)
        if payout is not None:
            payments.append(payout)
            payments.sort(key=lambda payment: payment.minute)
    return payments


def plan_victim_schedule(
    victim: str, mule: str, chances: TypeChances, first_minute: int, split: str, rng: np.random.Generator
) -> VictimSchedule:
    """
    WHAT: Draws one victim's scam type and plans the matching payment schedule.

    WHY: The type decides which schedule shape applies; both shapes live in their own functions.

    Args:
        victim: The victim's account id.
        mule: The receiving mule's account id.
        chances: The step's TypeChances.
        first_minute: Minute of the victim's first payment.
        split: Split name.
        rng: The gang planning stream (child 7).

    Returns:
        The VictimSchedule.

    Example:
        plan_victim_schedule("acc_person_000007", "acc_person_000003", chances, 1440, "training", rng).scam_type
    """
    scam_type = draw_scam_type(chances, rng)
    if scam_type in (SCAM_INVESTMENT, SCAM_JOB):
        payments = plan_investment_or_job_payments(first_minute, split, rng)
    else:
        payments = plan_unauthorised_debit_payments(first_minute, split, rng)
    return VictimSchedule(victim, mule, scam_type, tuple(payments))


def _build_schedule_records(
    gang_id: str, schedule: VictimSchedule, purpose: str
) -> tuple[list[PlannedPayment], list[PaymentStory], list[IllicitMoneyRule]]:
    """
    WHAT: Turns one victim's schedule into planned payments with aligned stories and illicit money rules. Victim
    payments run victim to mule and only credit illicit money; the payout runs mule to victim and only spends it.
    The rule input_index is the position inside the returned lists; the caller shifts it.

    WHY: The rules are what let the runner forward only illicit money that arrived; the stories carry the scam type
    because the payment purpose cannot.

    Args:
        gang_id: The gang's id for story text.
        schedule: The victim's VictimSchedule.
        purpose: The payment purpose from config.

    Returns:
        (payments, stories, rules), equal in length and in the same order as schedule.payments.

    Example:
        A schedule of two payments and a payout gives three payments, three stories, three rules.
    """
    payments: list[PlannedPayment] = []
    stories: list[PaymentStory] = []
    rules: list[IllicitMoneyRule] = []
    victim_payment_total = sum(1 for payment in schedule.payments if not payment.is_payout)
    victim_payment_number = 0
    for position, scam_payment in enumerate(schedule.payments):
        if scam_payment.is_payout:
            sender, receiver = schedule.receiving_mule, schedule.victim_account
            description = f"Gang '{gang_id}' {schedule.scam_type} scam payout back"
            rule = IllicitMoneyRule(input_index=position, spends_illicit=True, credits_illicit=False)
        else:
            victim_payment_number += 1
            sender, receiver = schedule.victim_account, schedule.receiving_mule
            description = (
                f"Gang '{gang_id}' {schedule.scam_type} scam payment {victim_payment_number} of {victim_payment_total}"
            )
            rule = IllicitMoneyRule(input_index=position, spends_illicit=False, credits_illicit=True)
        payments.append(
            PlannedPayment(scam_payment.minute, sender, receiver, scam_payment.amount_paise, 0, purpose)
        )
        stories.append(PaymentStory(gang_id=gang_id, block="scam", layer=None, description=description))
        rules.append(rule)
    return payments, stories, rules


def _compute_mule_inflows(schedules: list[VictimSchedule]) -> dict[str, MuleInflow]:
    """
    WHAT: For each receiving mule, finds the minute of its latest planned victim payment and the planned money it
    holds afterwards: all its victims' payments minus all payouts back.

    WHY: The route forwards this amount and starts counting its dwell from that minute.

    Args:
        schedules: Every victim's VictimSchedule.

    Returns:
        Mule account id -> MuleInflow.

    Raises:
        ValueError: If a mule would hold no money after payouts, so there would be nothing to forward.

    Example:
        Payments of 300000 and 450000 and a payout of 100000 give net_paise 650000.
    """
    last_minutes: dict[str, int] = {}
    net_amounts: dict[str, int] = {}
    for schedule in schedules:
        mule = schedule.receiving_mule
        for payment in schedule.payments:
            if payment.is_payout:
                net_amounts[mule] = net_amounts.get(mule, 0) - payment.amount_paise
            else:
                net_amounts[mule] = net_amounts.get(mule, 0) + payment.amount_paise
                last_minutes[mule] = max(last_minutes.get(mule, 0), payment.minute)
    inflows: dict[str, MuleInflow] = {}
    for mule, net_paise in net_amounts.items():
        if net_paise <= 0:
            raise ValueError(f"Receiving mule '{mule}' would hold {net_paise} paise after payouts; nothing to forward.")
        inflows[mule] = MuleInflow(last_minutes[mule], net_paise)
    return inflows


def _draw_victim_counts(step: BlockStep, receiving_count: int, rng: np.random.Generator) -> list[int]:
    """
    WHAT: Returns how many victims each receiving mule has: the step's explicit victims_per_receiving_mule for every
    mule if it is given, otherwise one uniform draw per mule from the configured victims range.

    WHY: Lets a config pin the count while the default follows world_ranges.

    Args:
        step: The scam BlockStep.
        receiving_count: Number of receiving mules.
        rng: The gang planning stream (child 7).

    Returns:
        One count per receiving mule, each at least 1.

    Raises:
        ValueError: If a count is below 1.

    Example:
        With victims_per_receiving_mule 3 and 2 mules the result is [3, 3] and nothing is read from rng.
    """
    explicit = step.params_dict.get("victims_per_receiving_mule")
    if explicit is not None:
        return [int(explicit)] * receiving_count
    low_victims, high_victims = get_scam_victims_per_mule_range()
    counts = [int(rng.integers(low_victims, high_victims + 1)) for _ in range(receiving_count)]
    if min(counts) < 1:
        raise ValueError(f"Every receiving mule needs at least 1 victim, drew {min(counts)}.")
    return counts


def plan_scam_step(
    world: World, config: GangConfig, step: BlockStep, mules: list[str], start_minute: int,
    reserved_accounts: set[str] | None, rng: np.random.Generator
) -> ScamPlan:
    """
    WHAT: Plans a whole scam step. The first receiving_mules acquired mules receive victim money. Draw order on rng:
    the two scam-type chances, one victim count per mule (unless pinned), the victims in one draw, then for each
    victim in mule order its scam type and schedule, and last the step's risk.

    WHY: Victims are the gang's only money source when it has no fixture source, so this step creates the money
    the route may forward.

    Args:
        world: The World whose people are candidate victims; its drawn range_name selects the config split.
        config: The gang's GangConfig (id and optional fixture source).
        step: The scam BlockStep.
        mules: Accounts acquired by the gang so far.
        start_minute: Minute at which every victim makes the first payment.
        reserved_accounts: Accounts reserved by earlier gangs, never victims.
        rng: The gang planning stream (child 7).

    Returns:
        The ScamPlan.

    Raises:
        ValueError: If fewer mules were acquired than receiving_mules, or too few eligible people exist.
        KeyError: If the world did not record its range_name.

    Example:
        plan = plan_scam_step(world, config, scam_step, mules, 1440, None, rng)
        assert {role.role for role in plan.roles} == {"victim"}
    """
    receiving_count = int(step.params_dict["receiving_mules"])
    if receiving_count > len(mules):
        raise ValueError(
            f"Gang '{config.gang_id}' scam needs {receiving_count} receiving mules but acquired {len(mules)}."
        )
    split = str(world.parameters_drawn["range_name"])
    receiving_mules = mules[:receiving_count]

    chances = draw_type_chances(split, rng)
    victim_counts = _draw_victim_counts(step, receiving_count, rng)
    excluded = set(mules) | (reserved_accounts or set())
    if config.source is not None:
        excluded.add(config.source.account_id)
    victims = choose_victims(world, sum(victim_counts), excluded, rng)

    schedules: list[VictimSchedule] = []
    next_victim = 0
    for mule, victim_count in zip(receiving_mules, victim_counts):
        for victim in victims[next_victim:next_victim + victim_count]:
            schedules.append(plan_victim_schedule(victim, mule, chances, start_minute, split, rng))
        next_victim += victim_count

    return _assemble_scam_plan(config.gang_id, schedules, start_minute, rng)


def _assemble_scam_plan(
    gang_id: str, schedules: list[VictimSchedule], start_minute: int, rng: np.random.Generator
) -> ScamPlan:
    """
    WHAT: Builds the final ScamPlan from the victims' schedules: payments, stories and rules with the rule indexes
    made global to the plan, one victim role per victim from the victim's first payment, the mule inflows, the span
    of the step in minutes, and a risk drawn from the scam risk range.

    WHY: Keeps schedule drawing separate from the bookkeeping that turns schedules into runner inputs.

    Args:
        gang_id: The gang's id.
        schedules: Every victim's VictimSchedule.
        start_minute: Minute at which the step starts.
        rng: The gang planning stream (child 7); one number is read, for the risk.

    Returns:
        The ScamPlan.

    Example:
        Two victims with three and one payments give four payments, four stories, four rules and two roles.
    """
    purpose = get_gang_payment_purpose()
    payments: list[PlannedPayment] = []
    stories: list[PaymentStory] = []
    rules: list[IllicitMoneyRule] = []
    roles: list[RoleSpan] = []
    for schedule in schedules:
        victim_payments, victim_stories, victim_rules = _build_schedule_records(gang_id, schedule, purpose)
        offset = len(payments)
        for rule in victim_rules:
            rules.append(IllicitMoneyRule(offset + rule.input_index, rule.spends_illicit, rule.credits_illicit))
        payments.extend(victim_payments)
        stories.extend(victim_stories)
        roles.append(RoleSpan(schedule.victim_account, "victim", gang_id, None, schedule.payments[0].minute, None))

    latest_minute = max(payment.minute for payment in payments)
    low_risk, high_risk = get_block_risk_range("scam")
    risk = float(rng.uniform(low_risk, high_risk))
    return ScamPlan(
        tuple(payments), tuple(stories), tuple(rules), tuple(roles),
        _compute_mule_inflows(schedules), latest_minute - start_minute, risk,
    )
