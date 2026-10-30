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
    extract_run_id,
    get_default_settings,
)
from features.account_labels import (
    complained_accounts_visible,
    labelled_accounts_visible,
)
from features.account_reader import read_account_snapshot
from features.account_summaries import read_counterparty_days
from features.feature_set import load_feature_set_m1
from features.m1_features import load_m1_config
from features.registry_screening import RegistryScreener

__all__ = [
    "ProposedPayment",
    "compute_payment_features",
    "compute_payment_level_features",
    "compute_payer_only_payment_features",
]

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


def _extract_prop(proposal: Any, *keys: str) -> Any:
    """Extract property from dict or model object across alternate key names."""
    for k in keys:
        if isinstance(proposal, dict):
            if k in proposal:
                return proposal[k]
        elif hasattr(proposal, k):
            val = getattr(proposal, k)
            if val is not None:
                return val
    return None


def compute_payer_only_payment_features(
    redis_client: Any,
    proposal: Any,
    decision_time: datetime,
    settings: AccountFeaturesSettings | None = None,
    screener: RegistryScreener | None = None,
    dpip_key: str | None = None,
    arrival_cutoff: datetime | None = None,
) -> dict[str, Any]:
    """
    WHAT: Computes payer-only and proposal-level payment features for a proposed transaction.

    WHY: At decision evaluation time, the bank evaluates the transaction strictly from the
    payer's perspective without requiring private payee profile state. Missing optional fields
    (e.g. payee_type, receiver_vpa, channel) cleanly return None without crashing.
    """
    cfg = settings if settings is not None else get_default_settings()
    cutoff = arrival_cutoff if arrival_cutoff is not None else decision_time

    # 1. amount_paise
    raw_amount = _extract_prop(proposal, "amount_paise")
    amount_paise: int | None = int(raw_amount) if raw_amount is not None else None

    # 2. payment_channel
    payment_channel = _extract_prop(proposal, "channel", "payment_channel")

    # 3. payer_device_fingerprint
    payer_device_fingerprint = _extract_prop(
        proposal, "sender_device_fingerprint", "payer_device_fingerprint", "device_fingerprint"
    )

    # 4. payer_os_generation
    raw_os = _extract_prop(proposal, "sender_os_generation", "payer_os_generation", "os_generation")
    payer_os_generation: int | None = int(raw_os) if raw_os is not None else None

    # 5. round_amount (multiple of ₹100 = 10000 paise)
    round_amount: bool | None = (amount_paise % 10000 == 0) if amount_paise is not None else None

    # 6. structured_amount (in regulatory structuring band)
    structured_amount: bool | None = None
    if amount_paise is not None:
        m1_cfg = load_m1_config()
        is_structured = False
        for threshold in m1_cfg.reporting_thresholds_paise:
            lower_bound = int(threshold * (1.0 - m1_cfg.below_threshold_band_fraction))
            if lower_bound <= amount_paise < threshold:
                is_structured = True
                break
        structured_amount = is_structured

    # Payer account identification
    payer_account_id = _extract_prop(proposal, "payer_account_id", "sender_account_id")
    bank_id = _extract_prop(proposal, "bank_id")
    run_id = _extract_prop(proposal, "run_id")
    if not run_id:
        payment_id = _extract_prop(proposal, "payment_id")
        if payment_id:
            try:
                run_id = extract_run_id(payment_id)
            except ValueError:
                run_id = None

    # Read snapshot if payer state is available
    snapshot = None
    if redis_client is not None and payer_account_id and bank_id and run_id:
        snapshot = read_account_snapshot(
            redis_client,
            run_id,
            bank_id,
            payer_account_id,
            decision_time,
            cutoff,
            cfg,
        )

    # 7. same_amount_count_24h
    same_amount_count_24h: int | None = None
    if snapshot is not None and amount_paise is not None:
        cutoff_24h = decision_time - timedelta(seconds=86400)
        same_amount_count_24h = sum(
            1
            for r in snapshot.side_records
            if r.side == "debit"
            and r.status == "success"
            and r.amount_paise == amount_paise
            and cutoff_24h < r.event_time <= decision_time
        )

    # 8. payee_type (optional, produces None if missing or None)
    payee_type = _extract_prop(proposal, "payee_type")

    # 9. payee_registry_match (None if payee_vpa is None or no screener)
    payee_vpa = _extract_prop(proposal, "payee_vpa", "receiver_vpa")
    payee_bank_id = _extract_prop(proposal, "payee_bank_id", "receiver_bank_id")
    payee_registry_match: bool | None = None
    if (
        screener is not None
        and payee_vpa is not None
        and payee_bank_id is not None
        and run_id is not None
    ):
        resolved_key = dpip_key
        if resolved_key is None and config.dpip_participant_key is not None:
            resolved_key = config.dpip_participant_key.get_secret_value()

        if resolved_key:
            from features.m1_intelligence import get_bank_index
            from nationalsim.keyed_hash import compute_identifier_hash

            payee_bank_idx = get_bank_index(payee_bank_id)
            payee_hash = compute_identifier_hash(
                key=resolved_key,
                run_id=run_id,
                identifier_type="account",
                raw_value=payee_vpa,
                bank_index=payee_bank_idx,
            )
            outcome = screener.screen(payee_hash, "account", as_of=decision_time)
            if outcome.status == "match":
                payee_registry_match = True
            elif outcome.status == "no_match":
                payee_registry_match = False
            else:
                payee_registry_match = None

    # Canonical M1 payment_level features
    first_payment_to_payee: bool | None = None
    if (
        redis_client is not None
        and payer_account_id
        and bank_id
        and run_id
        and payee_bank_id
        and payee_vpa
    ):
        target_payee = f"{payee_bank_id}:{payee_vpa}"
        closed_debits = read_counterparty_days(
            redis_client,
            run_id,
            bank_id,
            payer_account_id,
            "debit",
            decision_time,
            cutoff,
            cfg,
        )
        has_closed = target_payee in closed_debits and len(closed_debits[target_payee]) > 0
        has_raw = False
        if snapshot is not None:
            for r in snapshot.side_records:
                if r.side == "debit" and r.status == "success" and r.counterparty_vpa:
                    if f"{r.counterparty_bank_id}:{r.counterparty_vpa}" == target_payee:
                        has_raw = True
                        break
        first_payment_to_payee = not (has_closed or has_raw)

    amount_vs_payer_median_ratio: float | None = None
    amount_vs_payer_max_ratio: float | None = None
    if snapshot is not None and amount_paise is not None:
        successful_debits = [
            r.amount_paise
            for r in snapshot.side_records
            if r.side == "debit" and r.status == "success"
        ]
        if successful_debits:
            amount_vs_payer_median_ratio = float(
                amount_paise / statistics.median(successful_debits)
            )
            amount_vs_payer_max_ratio = float(amount_paise / max(successful_debits))

    payee_same_bank_risk: float | None = None
    if payee_bank_id is not None and bank_id is not None:
        if payee_bank_id != bank_id:
            payee_same_bank_risk = None
        elif payee_vpa is not None and redis_client is not None and run_id is not None:
            labelled = labelled_accounts_visible(
                redis_client, run_id, bank_id, decision_time, cutoff
            )
            complained = complained_accounts_visible(
                redis_client, run_id, bank_id, decision_time, cutoff
            )
            if payee_vpa in labelled:
                payee_same_bank_risk = 1.0
            elif payee_vpa in complained:
                payee_same_bank_risk = 0.5
            else:
                payee_same_bank_risk = 0.0

    ist_dt = decision_time.astimezone(IST)
    hour_ist = ist_dt.hour

    customer_failed_attempts_24h = 0
    if snapshot is not None:
        cutoff_24h = decision_time - timedelta(seconds=86400)
        for r in snapshot.side_records:
            if (
                r.side == "debit"
                and r.status == "failed"
                and r.failure_domain == "customer"
                and cutoff_24h < r.event_time <= decision_time
            ):
                customer_failed_attempts_24h += 1

    return {
        "amount_paise": amount_paise,
        "payment_channel": payment_channel,
        "payer_device_fingerprint": payer_device_fingerprint,
        "payer_os_generation": payer_os_generation,
        "round_amount": round_amount,
        "structured_amount": structured_amount,
        "same_amount_count_24h": same_amount_count_24h,
        "payee_type": payee_type,
        "payee_registry_match": payee_registry_match,
        "first_payment_to_payee": first_payment_to_payee,
        "amount_vs_payer_median_ratio": amount_vs_payer_median_ratio,
        "amount_vs_payer_max_ratio": amount_vs_payer_max_ratio,
        "payee_same_bank_risk": payee_same_bank_risk,
        "hour_ist": hour_ist,
        "customer_failed_attempts_24h": customer_failed_attempts_24h,
    }


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
    payer_features = compute_payer_only_payment_features(
        client,
        proposed,
        proposed.event_time,
        settings=settings,
        screener=screener,
        dpip_key=dpip_key,
        arrival_cutoff=arrival_cutoff,
    )


    result: dict[str, int | float | str | bool | None] = {
        "payee_registry_match": payer_features["payee_registry_match"],
        "first_payment_to_payee": payer_features["first_payment_to_payee"],
        "amount_vs_payer_median_ratio": payer_features["amount_vs_payer_median_ratio"],
        "amount_vs_payer_max_ratio": payer_features["amount_vs_payer_max_ratio"],
        "payee_same_bank_risk": payer_features["payee_same_bank_risk"],
        "hour_ist": payer_features["hour_ist"],
        "payee_type": payer_features["payee_type"],
        "customer_failed_attempts_24h": payer_features["customer_failed_attempts_24h"],
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


compute_payment_level_features = compute_payment_features
