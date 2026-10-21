"""
WHAT: Procedural helper module for the split_below_threshold tactic building block.
Splits planned payments exceeding a configured or drawn threshold into equal pieces below it at the
same simulated minute, providing each piece with its own IllicitMoneyRule and PaymentStory ("split piece i of n").
Refuses splits that breach the slow_drip dividing line in training worlds while permitting them in test/stress.

WHY: Criminal syndicates structure large transfers below regulatory or rule-based reporting thresholds to evade
detection (amendment A-016). Testing models this as an explicit tactical building block that transforms downstream
routing or cash-out payments without shifting overall balance conservation.

TERM: threshold_inr: Maximum transfer ceiling in INR above which payments are partitioned into smaller legs.
TERM: split piece i of n: Forensic story notation identifying each equal leg of a partitioned payment.
TERM: slow_drip dividing line: Held-out boundary (2,000 INR) below which training gangs may not partition transfers.
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from world.interactive_records import IllicitMoneyRule, PlannedPayment
from world.tactic_config import get_block_definition, resolve_dividing_line
from world.truth_records import PaymentStory


def get_threshold_inr_range(
    split: str = "training",
    config_path: Path | None = None,
) -> tuple[int, int]:
    """
    WHAT: Resolves the [min, max] threshold in INR for the split_below_threshold tactic block.

    WHY: Reads calibrated bounds from config/tactic_blocks.yaml across training, test, and stress splits.

    Args:
        split: Range name identifier ('training', 'test', or 'stress').
        config_path: Optional path override for tactic_blocks.yaml.

    Returns:
        Two-element integer tuple (min_threshold_inr, max_threshold_inr).

    Example:
        # min_th, max_th = get_threshold_inr_range("training")
        # assert min_th == 10000 and max_th == 50000
    """
    definition = get_block_definition("split_below_threshold", config_path)
    param = definition["parameters"]["threshold_inr"]
    if split == "stress" and "stress" in param:
        bounds = param["stress"]
    elif split in param:
        bounds = param[split]
    elif "range" in param:
        bounds = param["range"]
    else:
        raise KeyError("No valid threshold range found in split_below_threshold config.")
    return int(bounds[0]), int(bounds[1])


def draw_threshold_paise(
    split: str,
    threshold_param: int | float | None,
    rng: np.random.Generator,
    config_path: Path | None = None,
) -> int:
    """
    WHAT: Resolves or samples the split threshold in integer paise.

    WHY: Gangs configure explicit threshold parameters or draw stochastic ceilings within split boundaries.

    Args:
        split: Range name identifier ('training', 'test', or 'stress').
        threshold_param: Explicit configured threshold in INR, or None to sample.
        rng: Gang random number generator stream.
        config_path: Optional path override for tactic_blocks.yaml.

    Returns:
        Integer threshold amount in paise.

    Example:
        # th_paise = draw_threshold_paise("training", 25000, rng)
        # assert th_paise == 2500000
    """
    if threshold_param is not None:
        threshold_inr = float(threshold_param)
    else:
        min_th, max_th = get_threshold_inr_range(split, config_path)
        threshold_inr = float(rng.integers(min_th, max_th + 1))
    return int(round(threshold_inr * 100))


def apply_split_below_threshold(
    payments: Sequence[PlannedPayment],
    rules: Sequence[IllicitMoneyRule],
    stories: Sequence[PaymentStory],
    threshold_paise: int,
    gang_id: str,
    split: str = "training",
    config_path: Path | None = None,
    world_ranges_path: Path | None = None,
) -> tuple[list[PlannedPayment], list[IllicitMoneyRule], list[PaymentStory]]:
    """
    WHAT: Partitions any payment strictly greater than threshold_paise into equal pieces below threshold_paise.
    Each resulting piece retains the payment's minute, sender, receiver, and purpose, but receives its own
    IllicitMoneyRule and PaymentStory ("split piece i of n"). Refuses splits below slow_drip dividing line in training.

    WHY: Decision D2 and amendment A-016 require structured evasion below reporting thresholds while preventing
    leakage into the held-out slow_drip tactic family (transfers below 2,000 INR) during training.

    Args:
        payments: Sequence of PlannedPayment records scheduled by a route or cash_out step.
        rules: Sequence of IllicitMoneyRule records corresponding 1-to-1 with payments.
        stories: Sequence of PaymentStory records corresponding 1-to-1 with payments.
        threshold_paise: Maximum payment ceiling in integer paise.
        gang_id: Gang identifier for error diagnostics.
        split: World range split ('training', 'test', or 'stress').
        config_path: Optional path override for tactic_blocks.yaml.
        world_ranges_path: Optional path override for world_ranges.yaml.

    Returns:
        Tuple of (new_payments, new_rules, new_stories).

    Raises:
        ValueError: If split is 'training' and any resulting piece is below the slow_drip dividing line.

    Example:
        # p, r, s = apply_split_below_threshold(pmts, rules, stories, 2500000, "gang_1", "training")
    """
    slow_drip_min_inr = float(resolve_dividing_line("slow_drip_max_amount_inr", config_path, world_ranges_path))
    slow_drip_min_paise = int(round(slow_drip_min_inr * 100))

    new_payments: list[PlannedPayment] = []
    new_rules: list[IllicitMoneyRule] = []
    new_stories: list[PaymentStory] = []

    for p, r, s in zip(payments, rules, stories):
        if p.amount_paise <= threshold_paise:
            new_rules.append(
                IllicitMoneyRule(
                    input_index=len(new_payments),
                    spends_illicit=r.spends_illicit,
                    credits_illicit=r.credits_illicit,
                    requires_credit_from=r.requires_credit_from,
                )
            )
            new_payments.append(p)
            new_stories.append(s)
            continue

        n = (p.amount_paise // threshold_paise) + 1
        base_piece = p.amount_paise // n
        remainder = p.amount_paise % n
        piece_amounts = [base_piece + (1 if k < remainder else 0) for k in range(n)]

        min_piece = min(piece_amounts)
        if split == "training" and min_piece < slow_drip_min_paise:
            raise ValueError(
                f"Gang '{gang_id}' split piece amount {min_piece / 100:.2f} INR is below "
                f"training slow drip dividing line of {slow_drip_min_inr:.0f} INR."
            )

        for k in range(1, n + 1):
            piece_amt = piece_amounts[k - 1]
            piece_payment = PlannedPayment(
                minute=p.minute,
                from_account=p.from_account,
                to_account=p.to_account,
                amount_paise=piece_amt,
                creation_index=0,
                purpose=p.purpose,
            )
            piece_story = PaymentStory(
                gang_id=s.gang_id,
                block=s.block,
                layer=s.layer,
                description=f"split piece {k} of {n}",
            )
            piece_rule = IllicitMoneyRule(
                input_index=len(new_payments),
                spends_illicit=r.spends_illicit,
                credits_illicit=r.credits_illicit,
                requires_credit_from=r.requires_credit_from,
            )
            new_payments.append(piece_payment)
            new_rules.append(piece_rule)
            new_stories.append(piece_story)

    return new_payments, new_rules, new_stories
