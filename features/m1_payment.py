# File: features/m1_payment.py
# Purpose: Level 15b M1 payment-level feature calculation functions.
# Issue: Level 15b M1 payment-level features (plan Task 1, R8, G1, R7).
"""
WHAT: Pure and state-backed functions computing M1 payment-level features for a proposed transaction:
  - first_payment_to_payee: True when the payer has never previously transferred to this payee.
  - amount_vs_payer_median_ratio: Ratio of proposed amount to payer's historical 7d median successful debit.
  - amount_vs_payer_max_ratio: Ratio of proposed amount to payer's historical 7d max successful debit.
  - payee_same_bank_risk: 1.0 if same-bank payee is confirmed fraud, 0.5 if complained, 0.0 if normal;
    None if other bank.
  - hour_ist: Indian Standard Time (IST) hour of event_time (0 to 23).
  - payee_type: Candidate payee classification ('person' or 'merchant').
  - customer_failed_attempts_24h: Count of payer's failed attempts in last 24h with failure_domain == 'customer'.

WHY: Evaluates real-time risk at the exact payment evaluation moment (decision_time = proposed.event_time)
without reading the wall clock or suffering future-leakage. Technical failure domains (sender_bank, npci_central)
never count against customer failed attempt velocity (Issue R7).
"""

import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from core.config import config
from features.account_keys import (
    AccountFeaturesSettings,
    get_default_settings,
)
from features.account_labels import (
    complained_accounts_visible,
    labelled_accounts_visible,
)
from features.account_reader import read_account_snapshot
from features.account_summaries import read_counterparty_days
from features.feature_set import load_feature_set_m1
from features.registry_screening import RegistryScreener

IST = timezone(timedelta(hours=5, minutes=30))


@dataclass(frozen=True)
class ProposedPayment:
    """
    Immutable representation of a proposed UPI payment evaluated by commercial bank fraud screening.

    Attributes:
        run_id: 16-hex simulation run identifier.
        bank_id: Tenant commercial bank evaluating the debit leg.
        payer_account_id: Customer initiating the transfer.
        payee_bank_id: Receiving bank identifier.
        payee_vpa: Receiving account identifier / VPA.
        payee_type: Classification string ('person' or 'merchant').
        amount_paise: Transaction amount in integer paise (> 0).
        event_time: Timezone-aware timestamp of the proposed transaction.
    """

    run_id: str
    bank_id: str
    payer_account_id: str
    payee_bank_id: str
    payee_vpa: str
    payee_type: str
    amount_paise: int
    event_time: datetime

    def __post_init__(self) -> None:
        if not self.run_id or not isinstance(self.run_id, str):
            raise ValueError(f"Invalid run_id: {self.run_id!r}; must be non-empty string")
        if not self.bank_id or not isinstance(self.bank_id, str):
            raise ValueError(f"Invalid bank_id: {self.bank_id!r}; must be non-empty string")
        if not self.payer_account_id or not isinstance(self.payer_account_id, str):
            raise ValueError(f"Invalid payer_account_id: {self.payer_account_id!r}; must be non-empty string")
        if not self.payee_bank_id or not isinstance(self.payee_bank_id, str):
            raise ValueError(f"Invalid payee_bank_id: {self.payee_bank_id!r}; must be non-empty string")
        if not self.payee_vpa or not isinstance(self.payee_vpa, str):
            raise ValueError(f"Invalid payee_vpa: {self.payee_vpa!r}; must be non-empty string")
        if self.payee_type not in ("person", "merchant"):
            raise ValueError(f"Invalid payee_type: {self.payee_type!r}; must be 'person' or 'merchant'")
        if not isinstance(self.amount_paise, int) or self.amount_paise <= 0:
            raise ValueError(f"Invalid amount_paise: {self.amount_paise!r}; must be positive integer")
        if not isinstance(self.event_time, datetime) or self.event_time.tzinfo is None:
            raise ValueError(f"Invalid event_time: {self.event_time!r}; must be timezone-aware datetime")


def compute_payment_features(
    client: Any,
    proposed: ProposedPayment,
    arrival_cutoff: datetime,
    settings: AccountFeaturesSettings | None = None,
    screener: RegistryScreener | None = None,
    dpip_key: str | None = None,
) -> dict[str, int | float | str | bool | None]:
    """
    PURPOSE:
        Computes all M1 payment_level group features for a proposed transaction.
        The decision time is strictly proposed.event_time. Never reads the wall clock.

    INPUTS:
        client: Redis client.
        proposed: ProposedPayment instance.
        arrival_cutoff: Upper bound for ingestion watermark visibility.
        settings: Optional AccountFeaturesSettings.

    RETURN VALUE:
        Dictionary mapping payment_level feature names to computed values.
    """
    cfg = settings if settings is not None else get_default_settings()
    decision_time = proposed.event_time

    # 1. first_payment_to_payee: True when payee_bank_id:payee_vpa never appears
    # among payer's debit counterparties in closed days or visible 7-day raw window.
    target_payee = f"{proposed.payee_bank_id}:{proposed.payee_vpa}"
    closed_debits = read_counterparty_days(
        client,
        proposed.run_id,
        proposed.bank_id,
        proposed.payer_account_id,
        "debit",
        decision_time,
        arrival_cutoff,
        cfg,
    )
    has_closed_history = target_payee in closed_debits and len(closed_debits[target_payee]) > 0

    snapshot = read_account_snapshot(
        client,
        proposed.run_id,
        proposed.bank_id,
        proposed.payer_account_id,
        decision_time,
        arrival_cutoff,
        cfg,
    )
    has_raw_history = False
    for r in snapshot.side_records:
        if r.side == "debit" and r.status == "success" and r.counterparty_vpa:
            if f"{r.counterparty_bank_id}:{r.counterparty_vpa}" == target_payee:
                has_raw_history = True
                break

    first_payment_to_payee = not (has_closed_history or has_raw_history)

    # 2. amount_vs_payer_median_ratio and amount_vs_payer_max_ratio:
    # proposed amount divided by median / max of payer's successful debit amounts in 7-day raw window.
    successful_debits = [
        r.amount_paise
        for r in snapshot.side_records
        if r.side == "debit" and r.status == "success"
    ]
    if successful_debits:
        median_debit = statistics.median(successful_debits)
        max_debit = max(successful_debits)
        amount_vs_payer_median_ratio = float(proposed.amount_paise / median_debit)
        amount_vs_payer_max_ratio = float(proposed.amount_paise / max_debit)
    else:
        amount_vs_payer_median_ratio = None
        amount_vs_payer_max_ratio = None

    # 3. payee_same_bank_risk: None if payee_bank_id != this bank;
    # else 1.0 if payee is labelled confirmed fraud, 0.5 if complained account, else 0.0.
    if proposed.payee_bank_id != proposed.bank_id:
        payee_same_bank_risk = None
    else:
        payee_account = proposed.payee_vpa
        labelled = labelled_accounts_visible(
            client, proposed.run_id, proposed.bank_id, decision_time, arrival_cutoff
        )
        complained = complained_accounts_visible(
            client, proposed.run_id, proposed.bank_id, decision_time, arrival_cutoff
        )
        if payee_account in labelled:
            payee_same_bank_risk = 1.0
        elif payee_account in complained:
            payee_same_bank_risk = 0.5
        else:
            payee_same_bank_risk = 0.0

    # 4. hour_ist: 0-23 IST hour of event_time
    ist_dt = proposed.event_time.astimezone(IST)
    hour_ist = ist_dt.hour

    # 5. payee_type: proposed payee_type string
    payee_type = proposed.payee_type

    # 6. customer_failed_attempts_24h: payer's failed sides in 24h where failure_domain == 'customer'.
    # Issue R7: technical failures (e.g. sender_bank, npci_central) never count against customer velocity.
    cutoff_24h = decision_time - timedelta(seconds=86400)
    customer_failed_attempts_24h = 0
    for r in snapshot.side_records:
        if (
            r.side == "debit"
            and r.status == "failed"
            and r.failure_domain == "customer"
            and cutoff_24h < r.event_time <= decision_time
        ):
            customer_failed_attempts_24h += 1

    # 7. payee_registry_match: True / False if screened against national registry; None if unknown or no screener.
    payee_registry_match: bool | None = None
    if screener is not None:
        resolved_key = dpip_key
        if resolved_key is None and config.dpip_participant_key is not None:
            resolved_key = config.dpip_participant_key.get_secret_value()

        if resolved_key:
            from features.m1_intelligence import get_bank_index
            from nationalsim.keyed_hash import compute_identifier_hash

            payee_bank_idx = get_bank_index(proposed.payee_bank_id)
            payee_hash = compute_identifier_hash(
                key=resolved_key,
                run_id=proposed.run_id,
                identifier_type="account",
                raw_value=proposed.payee_vpa,
                bank_index=payee_bank_idx,
            )
            outcome = screener.screen(payee_hash, "account", as_of=decision_time)
            if outcome.status == "match":
                payee_registry_match = True
            elif outcome.status == "no_match":
                payee_registry_match = False
            else:
                payee_registry_match = None

    result: dict[str, int | float | str | bool | None] = {
        "payee_registry_match": payee_registry_match,
        "first_payment_to_payee": first_payment_to_payee,
        "amount_vs_payer_median_ratio": amount_vs_payer_median_ratio,
        "amount_vs_payer_max_ratio": amount_vs_payer_max_ratio,
        "payee_same_bank_risk": payee_same_bank_risk,
        "hour_ist": hour_ist,
        "payee_type": payee_type,
        "customer_failed_attempts_24h": customer_failed_attempts_24h,
    }

    # Schema drift check against payment_level group in feature_set_m1.yaml
    feature_set = load_feature_set_m1()
    expected_names = set(f.name for f in feature_set.features_by_group("payment_level"))
    actual_names = set(result.keys())
    if actual_names != expected_names:
        missing = expected_names - actual_names
        unexpected = actual_names - expected_names
        raise ValueError(
            f"Payment feature drift detected against schema. "
            f"Missing features: {sorted(missing)}, Unexpected features: {sorted(unexpected)}"
        )

    return result
