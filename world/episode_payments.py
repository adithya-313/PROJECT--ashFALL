"""
WHAT: Scheduled episode payment planning and financial pattern generation.
WHY: Translates ordinary citizens' temporary burst episodes into realistic candidate payments (issue A1).
TERM: episode: a bounded period of high-velocity financial activity for an ordinary citizen,
      e.g. collecting wedding contributions or organizing a college festival.
TERM: collector: the ordinary person the episode belongs to, who receives money or coordinates the group,
      e.g. a festival treasurer or a relative handling family utility bills.
TERM: payer: a person sending money to the collector during an episode,
      e.g. an attendee contributing wedding funds or a kitty member paying monthly dues.
TERM: inflow: a payment received by the collector during an episode,
      e.g. incoming gifts or festival entry fees paid by attendees.
TERM: outflow: a payment sent out by the collector following the episode,
      e.g. paying catering merchants or disbursing a kitty prize to a winner.
TERM: kitty round: a contribution cycle in a chit fund or kitty where members contribute money,
      e.g. round 1 of 3 monthly collections across all participating members.
TERM: cap: a ceiling limiting the number of distinct payers to the number of available people in the world,
      e.g. capping a 100-payer request at 59 available citizens in a 60-person town.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from world.background_config import draw_whole_number
from world.calendar_runner import PlannedPayment, persona_hour_probabilities
from world.config_reader import draw_in_range, get_parameter_entry
from world.episodes import Episode
from world.world_plan import WorldPlan

# Purpose labels for planned episode transactions.
EPISODE_IN_PURPOSE: str = "episode_in"
EPISODE_OUT_PURPOSE: str = "episode_out"

# Allowed destination merchant kinds for wedding and medical collection funds.
WEDDING_MEDICAL_SHOP_KINDS: tuple[str, ...] = ("restaurant", "grocery", "utility")

# Allowed destination merchant kinds for college fest treasurer expenditures.
COLLEGE_FEST_SHOP_KINDS: tuple[str, ...] = ("restaurant", "grocery")

# Maximum number of contributing relatives in bill-assistance episodes.
RELATIVES_MAX: int = 2

# Allowed destination merchant kinds for relative bill assistance payments.
BILL_ASSISTANCE_SHOP_KINDS: tuple[str, ...] = ("utility",)


@dataclass(frozen=True)
class EpisodeSettings:
    """
    WHAT: Holds the six sampled financial parameters governing an individual episode's payment flows.
    WHY: Packages drawn configuration parameters so episode flow generators receive immutable numeric rates.
    """

    contribution_multiple: float
    kitty_rounds: int
    sale_multiple: float
    outflow_share: float
    outflow_delay_days: int
    outflow_payment_count: int


@dataclass(frozen=True)
class EpisodePlanResult:
    """
    WHAT: Encapsulates scheduled episode transactions and planning audit counters.
    WHY: Delivers planned candidate payment transfers to calendar runner alongside diagnostic metrics.
    """

    payments: list[PlannedPayment]
    report: dict[str, int]


def _get_config_entry(ranges: dict[str, Any], key: str) -> dict[str, Any]:
    """
    WHAT: Retrieves parameter dictionary block from raw or nested config dictionary container.
    WHY: Supports both master world ranges configuration files and lightweight test dictionary fixtures.
    Args: ranges: Configuration dictionary containing parameter blocks; key: Parameter key name.
    Returns: Parameter block dictionary specifying distributions across training, test, and stress splits.
    Example:
        # entry = _get_config_entry(ranges, "episode_outflow_delay_days")
        # assert "training" in entry
    """
    # Look for top-level parameters dictionary key from standard world ranges file.
    if "parameters" in ranges:
        # Resolve parameter distribution block from nested parameters dictionary.
        return get_parameter_entry(ranges, key)
    # Check fallback dictionary structure used in unit test fixture dictionaries.
    if key in ranges:
        # Return parameter distribution block directly from flat mock dictionary.
        return ranges[key]
    # Raise error if parameter identifier is absent from both dictionary structures.
    raise ValueError(f"Unknown config parameter key: '{key}'")


def draw_episode_settings(
    ranges: dict[str, Any], rng: np.random.Generator, range_name: str = "training"
) -> EpisodeSettings:
    """
    WHAT: Samples the six financial parameters for an episode from world ranges in a fixed draw order.
    WHY: Drawing all six in fixed order ensures random stream consumption is constant across episode kinds.
         This prevents one episode's kind from perturbing the random stream of subsequent episodes.
    Args: ranges: World ranges dictionary container; rng: Episode stream; range_name: Split name.
    Returns: EpisodeSettings holding sampled contribution, rounds, sale, and outflow parameters.
    Example:
        # settings = draw_episode_settings(ranges, rng, "training"); assert settings.kitty_rounds >= 1
    """
    # Draw continuous multiplier scaling each contributor's median payment size.
    c_m = draw_in_range(_get_config_entry(ranges, "episode_contribution_multiple_of_payer_median"), rng, range_name)
    # Draw discrete count of payment rounds in a kitty or chit fund episode.
    k_r = draw_whole_number(_get_config_entry(ranges, "episode_kitty_contribution_rounds"), rng, range_name)
    # Draw continuous multiplier determining the single large sale transaction amount.
    s_m = draw_in_range(_get_config_entry(ranges, "episode_sale_multiple_of_collector_median"), rng, range_name)
    # Draw continuous fraction of aggregate inflows that must be disbursed downstream.
    o_s = draw_in_range(_get_config_entry(ranges, "episode_outflow_share_of_inflow"), rng, range_name)
    # Draw discrete count of days between episode completion and outflow disbursement.
    o_d = draw_whole_number(_get_config_entry(ranges, "episode_outflow_delay_days"), rng, range_name)
    # Draw discrete count of separate transactions used to disburse the outbound funds.
    o_c = draw_whole_number(_get_config_entry(ranges, "episode_outflow_payment_count"), rng, range_name)
    return EpisodeSettings(c_m, k_r, s_m, o_s, o_d, o_c)


def _draw_inflow_time(
    day: int, payer: Any, episode: Episode, probs: dict[str, np.ndarray], rng: np.random.Generator
) -> int:
    """
    WHAT: Determines transaction minute for an inflow using payer diurnal rhythms bounded by the episode.
    WHY: Emulates realistic waking-hour payer activity while ensuring transactions fall strictly inside the episode.
    Args: day: Calendar day index; payer: Person; episode: Episode bounds; probs: Diurnal weights; rng: RNG.
    Returns: Integer simulation minute strictly within [episode.start_minute, episode.end_minute).
    Example:
        # minute = _draw_inflow_time(day, payer, episode, probs, rng)
        # assert episode.start_minute <= minute < episode.end_minute
    """
    # Sample transaction hour from diurnal distribution to mimic human waking hours.
    hour = int(rng.choice(24, p=probs[payer.persona_name]))
    # Pick minute within selected hour uniformly at random across sixty minutes.
    minute_in_hour = int(rng.integers(0, 60))
    raw_minute = day * 1440 + hour * 60 + minute_in_hour
    # Clamp minute to strictly reside within the half-open episode window [start, end).
    return max(episode.start_minute, min(episode.end_minute - 1, raw_minute))


def _select_payers(
    candidates: list[Any], needed: int, rng: np.random.Generator, report: dict[str, int]
) -> list[Any]:
    """
    WHAT: Selects contributing payers from available citizens without replacement, applying caps if needed.
    WHY: Enforces that payer count cannot exceed the simulated town population and audits every cap event.
    Args: candidates: Eligible citizen entities; needed: Requested payer count; rng: RNG stream; report: Audit counters.
    Returns: List of Person instances chosen to contribute to the episode.
    Example:
        # payers = _select_payers(candidates, 50, rng, report); assert len(payers) <= len(candidates)
    """
    available = len(candidates)
    # Check if the requested number of payers exceeds the available citizen count.
    if needed > available:
        # Increment audit counter tracking episodes capped by world population.
        report["payer_caps"] += 1
        # Return all available citizens when demand exceeds available supply.
        return list(candidates)
    # Sample required number of payer indices uniformly without replacement.
    chosen_indices = rng.choice(available, size=needed, replace=False)
    # Map sampled numeric indices back to citizen entity records.
    return [candidates[int(i)] for i in chosen_indices]


def _build_inflow_payment(
    payer: Any, collector: Any, episode: Episode, amount: int, day: int,
    probs: dict[str, np.ndarray], rng: np.random.Generator, idx: int
) -> PlannedPayment:
    """
    WHAT: Creates a single planned inflow transaction record stamped with diurnal timing.
    WHY: Unifies creation of candidate inflow payments across collection, kitty, and assistance episodes.
    Args: payer: Sender; collector: Recipient; episode: Window; amount: Paise; day: Day;
        probs: Rhythm; rng: RNG; idx: Counter.
    Returns: PlannedPayment instance representing the scheduled incoming transfer.
    Example:
        # pmt = _build_inflow_payment(p, c, ep, 1000, 2, probs, rng, 0); assert pmt.amount_paise == 1000
    """
    # Sample payment minute from payer diurnal distribution clamped to episode bounds.
    minute = _draw_inflow_time(day, payer, episode, probs, rng)
    # Instantiate immutable PlannedPayment marked with episode_in purpose tag.
    return PlannedPayment(minute, payer.account_id, collector.account_id, amount, idx, EPISODE_IN_PURPOSE)


def _plan_collection_or_fest(
    collector: Any, candidates: list[Any], episode: Episode, settings: EpisodeSettings,
    probs: dict[str, np.ndarray], rng: np.random.Generator, report: dict[str, int], start_idx: int
) -> tuple[list[PlannedPayment], int]:
    """
    WHAT: Plans single-inflow payments from selected payers for collection or festival treasurer episodes.
    WHY: Models group fundraising where each contributor sends a single gift or fee during the festival window.
    Args: collector: Recipient; candidates: Citizens; episode: Specification; settings: Rates;
        probs: Rhythm; rng: RNG; report: Audit; start_idx: Idx.
    Returns: Tuple of (planned payment records, updated sequence counter index).
    Example:
        # pmts, idx = _plan_collection_or_fest(collector, candidates, ep, settings, probs, rng, report, 0)
        # assert len(pmts) >= 0
    """
    payments, idx = [], start_idx
    payers = _select_payers(candidates, episode.payer_count, rng, report)
    start_day = episode.start_minute // 1440
    length_days = max(1, (episode.end_minute - episode.start_minute) // 1440)
    # Plan a single contribution payment for each selected contributing citizen.
    for p in payers:
        day = int(rng.integers(start_day, start_day + length_days))
        amt = max(1, int(round(settings.contribution_multiple * p.amount_profile.median_paise)))
        # Build candidate payment transfer stamped with diurnal timestamp.
        payments.append(_build_inflow_payment(p, collector, episode, amt, day, probs, rng, idx))
        idx += 1
        report["inflows_planned"] += 1
    return payments, idx


def _plan_kitty_inflows(
    collector: Any, candidates: list[Any], episode: Episode, settings: EpisodeSettings,
    probs: dict[str, np.ndarray], rng: np.random.Generator, report: dict[str, int], start_idx: int
) -> tuple[list[PlannedPayment], list[Any], int]:
    """
    WHAT: Plans recurring contribution inflows across multiple rounds for chit fund or kitty episodes.
    WHY: Emulates rotating savings groups where each member contributes across sequential scheduled rounds.
    Args: collector: Manager; candidates: Citizens; episode: Specification; settings: Rates;
        probs: Rhythm; rng: RNG; report: Audit; start_idx: Idx.
    Returns: Tuple of (planned payment records, list of member Persons, updated sequence counter index).
    Example:
        # pmts, members, idx = _plan_kitty_inflows(collector, candidates, ep, settings, probs, rng, report, 0)
        # assert len(pmts) == len(members) * settings.kitty_rounds
    """
    payments, idx = [], start_idx
    members = _select_payers(candidates, episode.payer_count, rng, report)
    start_day = episode.start_minute // 1440
    length_days = max(1, (episode.end_minute - episode.start_minute) // 1440)
    # Iterate across each contribution round of the rotating chit fund.
    for r in range(settings.kitty_rounds):
        round_day = start_day + (r * length_days) // settings.kitty_rounds
        for m in members:
            amt = max(1, int(round(settings.contribution_multiple * m.amount_profile.median_paise)))
            # Build planned candidate payment record for this round contribution.
            payments.append(_build_inflow_payment(m, collector, episode, amt, round_day, probs, rng, idx))
            idx += 1
            report["inflows_planned"] += 1
    return payments, members, idx


def _plan_sale_inflows(
    collector: Any, candidates: list[Any], episode: Episode, settings: EpisodeSettings,
    probs: dict[str, np.ndarray], rng: np.random.Generator, report: dict[str, int], start_idx: int
) -> tuple[list[PlannedPayment], Any | None, int]:
    """
    WHAT: Plans single large asset purchase payment for sell-then-repay episodes, verifying affordability.
    WHY: Models asset sales where the buyer must have sufficient opening liquidity to afford the purchase.
    Args: collector: Seller; candidates: Citizens; episode: Specification; settings: Rates;
        probs: Rhythm; rng: RNG; report: Audit; start_idx: Idx.
    Returns: Tuple of (planned payments list, buyer Person or None, updated sequence counter index).
    Example:
        # pmts, buyer, idx = _plan_sale_inflows(collector, candidates, ep, settings, probs, rng, report, 0)
        # assert buyer is None or len(pmts) == 1
    """
    sale_amt = max(1, int(round(settings.sale_multiple * collector.amount_profile.median_paise)))
    eligible = [p for p in candidates if p.opening_balance_paise >= sale_amt]
    # Check if any candidate has sufficient opening liquidity to afford the purchase.
    if not eligible:
        report["sales_skipped_no_buyer"] += 1
        return [], None, start_idx
    # Select one qualified buyer and draw transaction timestamp.
    buyer = eligible[int(rng.integers(0, len(eligible)))]
    start_day = episode.start_minute // 1440
    length_days = max(1, (episode.end_minute - episode.start_minute) // 1440)
    day = int(rng.integers(start_day, start_day + length_days))
    # Build candidate payment transfer from buyer to seller.
    pmt = _build_inflow_payment(buyer, collector, episode, sale_amt, day, probs, rng, start_idx)
    report["inflows_planned"] += 1
    return [pmt], buyer, start_idx + 1


def _sample_relatives(candidates: list[Any], rng: np.random.Generator) -> list[Any]:
    """
    WHAT: Uniformly selects 1 or 2 relatives from eligible citizen candidates.
    WHY: Implements family remittance rule capping contributing relatives at RELATIVES_MAX.
    Args: candidates: List of citizen entities; rng: Random number generator stream.
    Returns: Sampled subset of citizen entities acting as contributing family members.
    Example:
        # rels = _sample_relatives(candidates, rng); assert len(rels) in (0, 1, 2)
    """
    max_rel = min(RELATIVES_MAX, len(candidates))
    # Return empty list if no candidate citizens exist in town.
    if max_rel < 1:
        return []
    # Sample count of assisting relatives uniformly between 1 and max_rel.
    n_rel = int(rng.integers(1, max_rel, endpoint=True))
    chosen_indices = rng.choice(len(candidates), size=n_rel, replace=False)
    # Map sampled relative indices to citizen entity instances.
    return [candidates[int(i)] for i in chosen_indices]


def _plan_relatives_inflows(
    collector: Any, candidates: list[Any], episode: Episode, settings: EpisodeSettings,
    probs: dict[str, np.ndarray], rng: np.random.Generator, report: dict[str, int], start_idx: int
) -> tuple[list[PlannedPayment], int]:
    """
    WHAT: Plans 1 to 2 contributing family members sending assistance funds to settle household utility bills.
    WHY: Replicates informal family remittances where relatives transfer money for collective bill payments.
    Args: collector: Recipient; candidates: Citizens; episode: Specification; settings: Rates;
        probs: Rhythm; rng: RNG; report: Audit; start_idx: Idx.
    Returns: Tuple of (planned payments list, updated sequence counter index).
    Example:
        # pmts, idx = _plan_relatives_inflows(collector, candidates, ep, settings, probs, rng, report, 0)
        # assert len(pmts) in (1, 2)
    """
    payments, idx = [], start_idx
    relatives = _sample_relatives(candidates, rng)
    start_day = episode.start_minute // 1440
    length_days = max(1, (episode.end_minute - episode.start_minute) // 1440)
    # Generate single assistance transfer for each sampled contributing relative.
    for r in relatives:
        day = int(rng.integers(start_day, start_day + length_days))
        amt = max(1, int(round(settings.contribution_multiple * r.amount_profile.median_paise)))
        # Build candidate payment transfer stamped with diurnal timestamp.
        payments.append(_build_inflow_payment(r, collector, episode, amt, day, probs, rng, idx))
        idx += 1
        report["inflows_planned"] += 1
    return payments, idx


def _get_shop_recipients(world: Any, allowed_kinds: tuple[str, ...]) -> list[str]:
    """
    WHAT: Collects account identifiers of registered merchants matching specified commercial categories.
    WHY: Ensures outbound payments route only to permitted shop categories and raises error if missing.
    Args: world: World simulation container holding commercial merchants; allowed_kinds: Permitted shop categories.
    Returns: List of shop account identifier strings matching the criteria.
    Example:
        # accs = _get_shop_recipients(world, ("restaurant", "grocery")); assert len(accs) > 0
    """
    # Filter commercial merchants in the world to permitted category kinds.
    shops = [s.account_id for s in world.shops if s.kind in allowed_kinds]
    if not shops:
        # Halt execution if simulation world topology lacks required merchant types.
        raise RuntimeError(f"Required shop kinds {allowed_kinds} do not exist in world")
    # Return list of matching merchant account identifiers.
    return shops


def _calculate_outflow_parts(total: int, count: int) -> list[int]:
    """
    WHAT: Splits total outflow paise into equal parts with remainder allocated to final part.
    WHY: Satisfies requirement that remainder paise are concentrated into the last payment installment.
    Args: total: Aggregate outflow amount in paise; count: Desired number of payment installments.
    Returns: List of integer paise amounts summing exactly to the total.
    Example:
        # parts = _calculate_outflow_parts(100, 3); assert parts == [33, 33, 34]
    """
    k = max(1, count)
    base = total // k
    rem = total % k
    # Return single total payment if only one installment is requested.
    if k == 1:
        return [total]
    # Return base installments with remainder paise in final part.
    return [base] * (k - 1) + [base + rem]


def _build_outflow_payments(
    collector: Any, recipients: list[str], parts: list[int], minute: int,
    start_idx: int, report: dict[str, int], rng: np.random.Generator
) -> tuple[list[PlannedPayment], int]:
    """
    WHAT: Emits PlannedPayment records for each installment part and updates audit counters.
    WHY: Separates transaction emission loop from temporal horizon gating logic.
    Args: collector: Payer; recipients: Target accounts; parts: Installments; minute: Timestamp;
        start_idx: Counter; report: Audit; rng: RNG.
    Returns: Tuple of (planned payment items list, updated creation index counter).
    Example:
        # pmts, idx = _build_outflow_payments(c, ['acc1'], [100], 100, 0, rep, rng); assert len(pmts) == 1
    """
    payments, idx = [], start_idx
    # Emit planned candidate payment transfer for each disbursement installment.
    for part in parts:
        rec_acc = recipients[int(rng.integers(0, len(recipients)))]
        # Build candidate payment transfer tagged with episode_out purpose.
        payments.append(PlannedPayment(minute, collector.account_id, rec_acc, part, idx, EPISODE_OUT_PURPOSE))
        idx += 1
        report["outflows_planned"] += 1
    return payments, idx


def _plan_outflows(
    collector: Any, recipients: list[str], total_inflow: int, settings: EpisodeSettings,
    probs: dict[str, np.ndarray], rng: np.random.Generator, report: dict[str, int],
    start_idx: int, n_days: int, episode: Episode, is_single_recipient: bool = False
) -> tuple[list[PlannedPayment], int]:
    """
    WHAT: Plans outbound disbursement payments from the collector to designated recipients or commercial shops.
    WHY: Disburses aggregated funds downstream according to configured outflow shares and settlement delays.
    Args: collector: Disburser; recipients: Targets; total_inflow: Total paise; settings: Rates;
        probs: Rhythm; rng: RNG; report: Audit; start_idx: Idx; n_days: Horizon; episode: Bounds;
        is_single_recipient: Single recipient flag.
    Returns: Tuple of (planned outflow payments, updated sequence counter index).
    Example:
        # pmts, idx = _plan_outflows(col, accs, 10000, settings, probs, rng, report, 0, 61, ep)
        # assert len(pmts) >= 0
    """
    outflow_total = int(round(settings.outflow_share * total_inflow))
    if outflow_total <= 0:
        return [], start_idx
    k = 1 if is_single_recipient else settings.outflow_payment_count
    parts = _calculate_outflow_parts(outflow_total, k)
    # Sample disbursement hour from collector diurnal activity profile.
    hour = int(rng.choice(24, p=probs[collector.persona_name]))
    outflow_minute = episode.end_minute + int(round(settings.outflow_delay_days * 1440)) + hour * 60
    # Drop disbursements that exceed simulation calendar horizon.
    if outflow_minute >= n_days * 1440:
        report["outflows_beyond_horizon"] += len(parts)
        return [], start_idx
    # Emit candidate payment transfers for valid within-horizon disbursements.
    return _build_outflow_payments(collector, recipients, parts, outflow_minute, start_idx, report, rng)


def _dispatch_peer_inflows(
    world: Any, collector: Any, candidates: list[Any], episode: Episode, settings: EpisodeSettings,
    probs: dict[str, np.ndarray], rng: np.random.Generator, report: dict[str, int], start_idx: int
) -> tuple[list[PlannedPayment], list[str], int]:
    """
    WHAT: Generates inflows and peer recipient accounts for chit funds and informal asset sales.
    WHY: Handles peer-to-peer episode types where recipient is a specific citizen rather than a commercial shop.
    Args: world: Container; collector: Host; candidates: Citizens; episode: Specification;
        settings: Rates; probs: Rhythm; rng: RNG; report: Audit; start_idx: Idx.
    Returns: Tuple of (inflow payments list, recipient accounts list, updated sequence index).
    Example:
        # inf, rec, idx = _dispatch_peer_inflows(w, col, cand, ep, setts, probs, rng, rep, 0)
        # assert isinstance(inf, list)
    """
    # Handle rotating savings chit fund or kitty collection inflows.
    if episode.kind == "chit_fund_or_kitty":
        inflows, members, idx = _plan_kitty_inflows(
            collector, candidates, episode, settings, probs, rng, report, start_idx
        )
        winner = members[int(rng.integers(0, len(members)))]
        return inflows, [winner.account_id], idx
    # Plan single large asset sale inflow checking prospective buyer liquidity.
    inflows, buyer, idx = _plan_sale_inflows(collector, candidates, episode, settings, probs, rng, report, start_idx)
    if buyer is None:
        return [], [], idx
    # Select one informal lender uniformly to receive loan repayment outflow.
    lenders = [p for p in candidates if p.person_id != buyer.person_id]
    lender = lenders[int(rng.integers(0, len(lenders)))]
    return inflows, [lender.account_id], idx


def _dispatch_episode_inflows(
    world: Any, collector: Any, candidates: list[Any], episode: Episode, settings: EpisodeSettings,
    probs: dict[str, np.ndarray], rng: np.random.Generator, report: dict[str, int], start_idx: int
) -> tuple[list[PlannedPayment], list[str], bool, int]:
    """
    WHAT: Routes inflow and recipient account generation according to episode category.
    WHY: Keeps category-specific branching cleanly isolated to preserve modular function bounds.
    Args: world: Container; collector: Host; candidates: Citizens; episode: Specification;
        settings: Rates; probs: Rhythm; rng: RNG; report: Audit; start_idx: Idx.
    Returns: Tuple of (inflow payments list, recipient accounts list, single recipient flag, updated index).
    Example:
        # inf, rec, single, idx = _dispatch_episode_inflows(w, col, cand, ep, setts, probs, rng, rep, 0)
        # assert isinstance(inf, list)
    """
    # Plan peer inflows for kitty or asset sale types.
    if episode.kind in ("chit_fund_or_kitty", "sell_then_repay_loan"):
        inflows, recipients, idx = _dispatch_peer_inflows(
            world, collector, candidates, episode, settings, probs, rng, report, start_idx
        )
        return inflows, recipients, True, idx
    # Handle wedding, medical collection, or college festival fundraising inflows.
    if episode.kind in ("wedding_or_medical_collection", "college_fest_treasurer"):
        is_wedding = episode.kind == "wedding_or_medical_collection"
        kinds = WEDDING_MEDICAL_SHOP_KINDS if is_wedding else COLLEGE_FEST_SHOP_KINDS
        inflows, idx = _plan_collection_or_fest(collector, candidates, episode, settings, probs, rng, report, start_idx)
        return inflows, _get_shop_recipients(world, kinds), False, idx
    # Handle family utility bill assistance remittances from relatives.
    inflows, idx = _plan_relatives_inflows(collector, candidates, episode, settings, probs, rng, report, start_idx)
    return inflows, _get_shop_recipients(world, BILL_ASSISTANCE_SHOP_KINDS), False, idx


def _plan_single_episode(
    world: Any, plan: WorldPlan, episode: Episode, ranges: dict[str, Any], rng: np.random.Generator,
    probs: dict[str, np.ndarray], range_name: str, report: dict[str, int], start_idx: int,
    p_by_id: dict[str, Any]
) -> tuple[list[PlannedPayment], int]:
    """
    WHAT: Orchestrates inflow generation and downstream outflow disbursement for one scheduled episode.
    WHY: Coordinates episode lifecycle stages while delegating detailed logic to modular helpers.
    Args: world: World; plan: Plan; episode: Episode; ranges: Ranges; rng: RNG; probs: Rhythm;
        range_name: Split; report: Audit; start_idx: Idx; p_by_id: Persons.
    Returns: Tuple of (planned payments list, updated sequence counter index).
    Example:
        # pmts, idx = _plan_single_episode(world, plan, ep, ranges, rng, probs, "training", rep, 0, p_map)
        # assert len(pmts) >= 0
    """
    settings = draw_episode_settings(ranges, rng, range_name)
    collector, candidates = p_by_id[episode.person_id], [p for p in world.people if p.person_id != episode.person_id]
    # Plan candidate inflows and determine downstream disbursement targets.
    inflows, recipients, is_single, idx = _dispatch_episode_inflows(
        world, collector, candidates, episode, settings, probs, rng, report, start_idx
    )
    if not inflows:
        return [], idx
    # Record planned episode and disburse collected funds downstream.
    report["episodes_planned"] += 1
    total_in = sum(p.amount_paise for p in inflows)
    outflows, idx = _plan_outflows(
        collector, recipients, total_in, settings, probs, rng, report, idx, plan.n_days, episode, is_single
    )
    return inflows + outflows, idx


def plan_episode_payments(
    world: Any, plan: WorldPlan, episodes: list[Episode], ranges: dict[str, Any],
    rng: np.random.Generator, range_name: str = "training"
) -> EpisodePlanResult:
    """
    WHAT: Generates all planned inflow and outflow candidate payments for scheduled episodes.
    WHY: Translates discrete lifestyle episodes into chronological transaction records for the calendar runner.
    Args: world: Container; plan: WorldPlan; episodes: Episodes; ranges: Ranges; rng: RNG; range_name: Split.
    Returns: EpisodePlanResult containing list of PlannedPayment records and aggregate diagnostic counters.
    Example:
        # res = plan_episode_payments(world, plan, episodes, ranges, ep_rng, "training")
        # assert len(res.payments) >= 0
    """
    report = {
        "episodes_planned": 0, "payer_caps": 0, "sales_skipped_no_buyer": 0,
        "outflows_beyond_horizon": 0, "inflows_planned": 0, "outflows_planned": 0,
    }
    payments, p_by_id, probs, idx = [], {p.person_id: p for p in world.people}, persona_hour_probabilities(plan), 0
    # Process each scheduled episode sequentially in chronological schedule order.
    for ep in episodes:
        new_pmts, idx = _plan_single_episode(world, plan, ep, ranges, rng, probs, range_name, report, idx, p_by_id)
        # Append newly generated candidate payments to aggregate output container.
        payments.extend(new_pmts)
    # Return immutable result object packaging payments list and report.
    return EpisodePlanResult(payments=payments, report=report)
