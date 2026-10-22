"""
WHAT: Cash-out channel distribution and planning module (Level 12c C-4, D3).
Distributes illicit funds from terminal-layer mule accounts across four channels:
crypto collectors (gang-opened, borrowed-KYC, role 'cash_out'), ATM withdrawals
(spread across days <= card limit), branch withdrawals (drawn delay hours),
and genuine goods-selling merchant shops (grocery, restaurant, no role span).

WHY: Models realistic extraction of criminal proceeds leaving the banking system or
diverting into ordinary consumer purchases (issues G2 and A2).

ISSUE IDs: G2 (cash-out channels), A2 (genuine shops downstream of mules).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core.config_validator import load_yaml
from world.build_world import World
from world.calendar_runner import PlannedPayment
from world.cash_withdrawals import CHANNEL_ATM, CHANNEL_BRANCH, CashWithdrawal
from world.interactive_records import IllicitMoneyRule
from world.mule_origins import build_opening_event, make_opening_handle
from world.outside_events import OutsideEvent
from world.run_identity import REPO_ROOT
from world.tactic_blocks import calculate_send_minute, draw_hop_dwell_minutes
from world.tactic_config import (
    get_block_risk_range,
    get_gang_payment_purpose,
    load_world_ranges,
)
from world.truth_records import PaymentStory, RoleSpan

CASH_OUT_CONFIG_PATH: Path = REPO_ROOT / "config" / "cash_out.yaml"


@dataclass(frozen=True)
class CashOutPlan:
    """
    WHAT: Output of planning the cash-out step for all terminal mules of a gang.
    WHY: Bundles payments, truth role spans, stories, withdrawals, and opening events.
    """

    payments: list[PlannedPayment]
    roles: list[RoleSpan]
    stories: list[PaymentStory]
    withdrawals: list[CashWithdrawal]
    outside_events: list[OutsideEvent]
    money_rules: list[IllicitMoneyRule]
    time_cost: int
    risk_cost: float


def load_cash_out_config(path: Path | None = None) -> dict[str, Any]:
    """
    WHAT: Loads config/cash_out.yaml dictionary.
    WHY: Provides declarative configuration for channel shares, shop kinds, and withdrawal delays.
    """
    resolved = path if path is not None else CASH_OUT_CONFIG_PATH
    return load_yaml(resolved)


def get_cash_out_channel_share_ranges(
    split: str,
    config_path: Path | None = None,
    world_ranges_path: Path | None = None,
) -> dict[str, tuple[float, float]]:
    """
    WHAT: Resolves the parameter share ranges for cash-out channels from config/world_ranges.yaml.
    WHY: Keeps numerical ranges single-sourced in world_ranges.yaml by reference.
    """
    cash_out_cfg = load_cash_out_config(config_path)["cash_out"]
    channel_shares_cfg = cash_out_cfg["channel_shares"]
    world_params = load_world_ranges(world_ranges_path)["parameters"]

    ranges: dict[str, tuple[float, float]] = {}
    for ch in ("crypto", "atm", "branch", "genuine_shop"):
        setting = channel_shares_cfg[ch]
        ref = setting["world_ranges_reference"]
        if ref not in world_params:
            raise ValueError(f"Channel '{ch}' references unknown parameter '{ref}'.")
        bounds = world_params[ref][split]
        ranges[ch] = (float(bounds[0]), float(bounds[1]))

    return ranges


def draw_cash_out_shares(
    split: str,
    rng: np.random.Generator,
    config_path: Path | None = None,
    world_ranges_path: Path | None = None,
) -> dict[str, float]:
    """
    WHAT: Draws channel shares per Decision D3:
    Draws genuine_shop_share, then splits the rest across crypto/atm/branch by their
    drawn shares divided by their sum (independent ranges policy).
    WHY: Preserves genuine consumer shopping proportion while allocating remaining
    proceeds across criminal extraction channels, guaranteeing shares sum to 1.0.
    """
    ranges = get_cash_out_channel_share_ranges(split, config_path, world_ranges_path)
    shop_low, shop_high = ranges["genuine_shop"]
    genuine_shop_share = float(rng.uniform(shop_low, shop_high)) if shop_low < shop_high else shop_low

    rest = max(0.0, 1.0 - genuine_shop_share)

    c_low, c_high = ranges["crypto"]
    a_low, a_high = ranges["atm"]
    b_low, b_high = ranges["branch"]

    raw_crypto = float(rng.uniform(c_low, c_high)) if c_low < c_high else c_low
    raw_atm = float(rng.uniform(a_low, a_high)) if a_low < a_high else a_low
    raw_branch = float(rng.uniform(b_low, b_high)) if b_low < b_high else b_low

    other_sum = raw_crypto + raw_atm + raw_branch
    if rest <= 0.0:
        return {
            "genuine_shop": 1.0,
            "crypto": 0.0,
            "atm": 0.0,
            "branch": 0.0,
        }

    if other_sum <= 0.0:
        raise ValueError("Sum of other cash-out channel shares is zero; cannot normalize.")

    crypto_share = rest * (raw_crypto / other_sum)
    atm_share = rest * (raw_atm / other_sum)
    branch_share = rest * (raw_branch / other_sum)

    return {
        "crypto": crypto_share,
        "atm": atm_share,
        "branch": branch_share,
        "genuine_shop": genuine_shop_share,
    }


def draw_atm_daily_limit_paise(
    split: str,
    rng: np.random.Generator,
    world_ranges_path: Path | None = None,
) -> int:
    """
    WHAT: Draws the ATM card daily limit in paise from config/world_ranges.yaml.
    WHY: Enforces realistic per-card withdrawal ceilings during cash extraction.
    """
    world_params = load_world_ranges(world_ranges_path)["parameters"]
    bounds = world_params["atm_daily_limit_inr"][split]
    inr_limit = int(rng.integers(int(bounds[0]), int(bounds[1]) + 1))
    return inr_limit * 100


def draw_branch_delay_minutes(
    split: str,
    rng: np.random.Generator,
    config_path: Path | None = None,
) -> int:
    """
    WHAT: Draws the over-the-counter branch cash withdrawal delay in minutes.
    WHY: Branch withdrawals are slower than ATMs due to branch hours and teller interaction.
    """
    cash_out_cfg = load_cash_out_config(config_path)["cash_out"]
    bounds = cash_out_cfg["branch_withdrawal_delay_hours"][split]
    hours = float(rng.uniform(float(bounds[0]), float(bounds[1])))
    return int(round(hours * 60))


def get_genuine_shop_kinds(config_path: Path | None = None) -> tuple[str, ...]:
    """
    WHAT: Returns allowed merchant kinds for genuine goods-selling shops downstream of mules.
    WHY: Mules buy retail consumer goods (grocery, restaurant); utilities and employers are excluded.
    """
    cash_out_cfg = load_cash_out_config(config_path)["cash_out"]
    return tuple(cash_out_cfg["genuine_shop_kinds"]["kinds"])


def plan_cash_out_step(
    world: World,
    gang_id: str,
    strength: int,
    terminal_mules: list[tuple[str, int, int]],
    split: str,
    layers: int,
    opening_start_index: int,
    has_disguise: bool,
    jitter: int | None,
    fixed_dwell: int | None,
    dwell_param: Any,
    rng: np.random.Generator,
    config_path: Path | None = None,
    world_ranges_path: Path | None = None,
) -> CashOutPlan:
    """
    WHAT: Plans the complete cash-out step for terminal layer mules.
    Distributes funds into crypto collector, ATM withdrawals, branch withdrawals, and genuine shops.
    """
    payments: list[PlannedPayment] = []
    roles: list[RoleSpan] = []
    stories: list[PaymentStory] = []
    withdrawals: list[CashWithdrawal] = []
    outside_events: list[OutsideEvent] = []
    money_rules: list[IllicitMoneyRule] = []

    opening_seq = opening_start_index
    genuine_kinds = get_genuine_shop_kinds(config_path)
    eligible_shops = [s for s in world.shops if s.kind in genuine_kinds]
    if not eligible_shops:
        eligible_shops = list(world.shops)

    max_delay = 0

    # One crypto collector account per gang
    crypto_handle: str | None = None
    crypto_arrival_minutes: list[int] = []
    crypto_send_minutes: list[int] = []

    channel_order = ["crypto", "atm", "branch", "genuine_shop"]

    for mule_acct, inflow_paise, arrival_minute in terminal_mules:
        if inflow_paise <= 0:
            continue

        shares = draw_cash_out_shares(split, rng, config_path, world_ranges_path)
        # Split across channels by shares, rounded down to whole paise
        raw_amounts = {ch: int(inflow_paise * shares[ch]) for ch in channel_order}
        rem = inflow_paise - sum(raw_amounts.values())
        # Remainder to the largest channel
        largest_ch = max(channel_order, key=lambda ch: (shares[ch], raw_amounts[ch]))
        raw_amounts[largest_ch] += rem

        # A channel amount under Rs 1 (100 paise) is dropped
        for ch in channel_order:
            if raw_amounts[ch] < 100:
                raw_amounts[ch] = 0

        amt_crypto = raw_amounts["crypto"]
        amt_atm = raw_amounts["atm"]
        amt_branch = raw_amounts["branch"]
        amt_shop = raw_amounts["genuine_shop"]

        hop_dwell = fixed_dwell if fixed_dwell is not None else draw_hop_dwell_minutes(dwell_param, rng)
        send_min = calculate_send_minute(arrival_minute, hop_dwell, has_disguise, jitter, rng)

        # Channel 1: Crypto collector
        if amt_crypto > 0:
            if crypto_handle is None:
                crypto_handle = make_opening_handle(gang_id, opening_seq)
                crypto_opening_event = build_opening_event(gang_id, opening_seq, "borrowed_kyc", arrival_minute)
                opening_seq += 1
                outside_events.append(crypto_opening_event)

            crypto_arrival_minutes.append(arrival_minute)
            crypto_send_minutes.append(send_min)

            pmt = PlannedPayment(
                from_account=mule_acct,
                to_account=crypto_handle,
                amount_paise=amt_crypto,
                minute=send_min,
                creation_index=0,
                purpose=get_gang_payment_purpose(),
            )
            payments.append(pmt)
            money_rules.append(
                IllicitMoneyRule(input_index=len(payments) - 1, spends_illicit=True, credits_illicit=True)
            )
            stories.append(
                PaymentStory(
                    gang_id=gang_id,
                    block="cash_out",
                    layer=layers + 1,
                    description=f"Gang '{gang_id}' cash-out via crypto collector",
                )
            )
            max_delay = max(max_delay, send_min - arrival_minute)

        # Channel 2: ATM withdrawals
        if amt_atm > 0:
            daily_limit = draw_atm_daily_limit_paise(split, rng, world_ranges_path)
            rem_atm = amt_atm
            day_offset = 0
            while rem_atm > 0:
                chunk = min(daily_limit, rem_atm)
                atm_min = send_min + day_offset * 1440
                withdrawals.append(
                    CashWithdrawal(
                        minute=atm_min,
                        account_id=mule_acct,
                        amount_paise=chunk,
                        channel=CHANNEL_ATM,
                        card_daily_limit_paise=daily_limit,
                    )
                )
                rem_atm -= chunk
                day_offset += 1
                max_delay = max(max_delay, atm_min - arrival_minute)

        # Channel 3: Branch withdrawals
        if amt_branch > 0:
            branch_delay = draw_branch_delay_minutes(split, rng, config_path)
            branch_min = send_min + branch_delay
            withdrawals.append(
                CashWithdrawal(
                    minute=branch_min,
                    account_id=mule_acct,
                    amount_paise=amt_branch,
                    channel=CHANNEL_BRANCH,
                )
            )
            max_delay = max(max_delay, branch_min - arrival_minute)

        # Channel 4: Genuine merchant shop
        if amt_shop > 0:
            shop_idx = int(rng.integers(0, len(eligible_shops)))
            chosen_shop = eligible_shops[shop_idx]
            pmt = PlannedPayment(
                from_account=mule_acct,
                to_account=chosen_shop.account_id,
                amount_paise=amt_shop,
                minute=send_min,
                creation_index=0,
                purpose=get_gang_payment_purpose(),
            )
            payments.append(pmt)
            # Stolen money exits to genuine economy: spends_illicit=True, credits_illicit=False
            money_rules.append(
                IllicitMoneyRule(input_index=len(payments) - 1, spends_illicit=True, credits_illicit=False)
            )
            stories.append(
                PaymentStory(
                    gang_id=gang_id,
                    block="cash_out",
                    layer=layers + 1,
                    description=f"Gang '{gang_id}' cash-out via genuine retail merchant",
                )
            )
            max_delay = max(max_delay, send_min - arrival_minute)

    # Finalize crypto collector RoleSpan if any crypto payments were planned
    if crypto_handle is not None and crypto_send_minutes:
        roles.append(
            RoleSpan(
                account_id=crypto_handle,
                role="cash_out",
                gang_id=gang_id,
                depth=None,
                valid_from=min(crypto_arrival_minutes),
                valid_to=None,
                origin=None,
                active_from=None,
                burned_from=None,
            )
        )

    min_risk, max_risk = get_block_risk_range("cash_out")
    risk_cost = float(rng.uniform(min_risk, max_risk))

    return CashOutPlan(
        payments=payments,
        roles=roles,
        stories=stories,
        withdrawals=withdrawals,
        outside_events=outside_events,
        money_rules=money_rules,
        time_cost=max_delay,
        risk_cost=risk_cost,
    )
