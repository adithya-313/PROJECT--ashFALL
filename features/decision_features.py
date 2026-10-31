# File: features/decision_features.py
# Purpose: Level 15c decision-time feature computation and validation.
# Issue: Level 15c watermark and feature log (plan Task 1, D4).
"""
WHAT: Computes decision-time feature vectors and builds FeatureLogRow instances
for shadow logging during /v1/decisions evaluation.

WHY: Enforces Section 22 evaluation correctness: decision features are computed
only when the bank's ingestion watermark is present. If unwatermarked, status is
marked 'no_watermark' without guessing or falling back to wall-clock time.
Decision time is strictly request.event_time; decided_at is clock.now_ist().
"""

from datetime import datetime, timezone
from typing import Any

from api.v1.models import DecisionRequest
from core.event_clock import Clock
from features.account_keys import (
    AccountFeaturesSettings,
    extract_run_id,
    get_default_settings,
)
from features.feature_log import (
    FeatureLogRow,
    canonical_feature_names,
)
from features.ingest_watermark import get_bank_watermark
from features.m1_features import compute_account_features
from features.m1_payment import (
    ProposedPayment,
    compute_payee_vpa_payment_features,
    compute_payer_only_payment_features,
    compute_payment_features,
)
from features.registry_screening import (
    LiveRegistryScreener,
    RecordingRegistryScreener,
    RegistryScreener,
    ScreenOutcome,
)

__all__ = ["compute_decision_row"]


def _format_registry_answers(answers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Formats in-memory recording answers into clean, JSON-serializable dictionaries."""
    formatted: list[dict[str, Any]] = []
    for a in answers:
        outcome = a["outcome"]
        outcome_dict: dict[str, Any] | str
        if isinstance(outcome, ScreenOutcome):
            outcome_dict = {
                "status": outcome.status,
                "first_visible_at": (
                    outcome.first_visible_at.isoformat() if outcome.first_visible_at is not None else None
                ),
            }
        else:
            outcome_dict = str(outcome)

        as_of = a["as_of"]
        received_at = a["received_at"]
        formatted.append(
            {
                "identifier_hash": a["identifier_hash"],
                "identifier_type": a["identifier_type"],
                "as_of": as_of.isoformat() if isinstance(as_of, datetime) else None,
                "outcome": outcome_dict,
                "received_at": (received_at.isoformat() if isinstance(received_at, datetime) else str(received_at)),
            }
        )
    return formatted


def compute_decision_row(
    redis_client: Any,
    bank_id: str,
    request: DecisionRequest,
    clock: Clock,
    settings: AccountFeaturesSettings | None = None,
    screener: RegistryScreener | None = None,
    dpip_key: str | None = None,
) -> FeatureLogRow:
    """
    WHAT: Computes all canonical M1 features as-of request.event_time for a DecisionRequest.

    WHY: Decouples real-time payment decisions from training feature vector collection.
    Guarantees strict Section 22 correctness via watermark verification.

    Args:
        redis_client: Redis client instance.
        bank_id: Tenant commercial bank identifier (required non-empty str).
        request: Validated DecisionRequest instance.
        clock: Injected Clock protocol instance (required).
        settings: Optional AccountFeaturesSettings.
        screener: Optional national RegistryScreener.
        dpip_key: Optional explicit DPIP HMAC key.

    Returns:
        FeatureLogRow ready for partitioned persistence in feature_log table.
    """
    if not isinstance(bank_id, str) or not bank_id:
        raise ValueError(f"Invalid bank_id: {bank_id!r}; must be non-empty string")

    payment_id = request.payment_id
    decision_time = request.event_time
    decided_at = clock.now_ist()
    run_id = extract_run_id(payment_id)
    account_id = request.sender_account_id

    canonical_names = canonical_feature_names()
    all_none_features: dict[str, Any] = {name: None for name in canonical_names}

    # 1. Watermark check: read bank watermark across all sources
    watermark: datetime | None = None
    if redis_client is not None:
        watermark = get_bank_watermark(redis_client, bank_id)

    # Condition for no_watermark is strictly watermark is None (F3)
    if watermark is None:
        return FeatureLogRow(
            bank_id=bank_id,
            payment_id=payment_id,
            run_id=run_id,
            account_id=account_id,
            decision_time=decision_time,
            decided_at=decided_at,
            arrival_cutoff=None,
            watermark_lag_ms=None,
            status="no_watermark",
            feature_values=all_none_features,
            missing_request_fields=[],
            registry_answers=[],
        )

    # G1: arrival_cutoff is watermark when watermark is not None
    arrival_cutoff = watermark

    # G4: lag is measured against clock.now_ist() (decided_at), not decision_time
    decided_at_utc = decided_at.astimezone(timezone.utc)
    watermark_utc = watermark.astimezone(timezone.utc)
    watermark_lag_ms = int((decided_at_utc - watermark_utc).total_seconds() * 1000)

    # 2. Incomplete request check (F4 & G5)
    missing: list[str] = []
    if not request.receiver_vpa:
        missing.append("receiver_vpa")
    if request.payee_type is None or request.payee_type not in ("person", "merchant"):
        missing.append("payee_type")

    recording_screener: RecordingRegistryScreener | None = None
    if screener is not None:
        recording_screener = RecordingRegistryScreener(LiveRegistryScreener(screener), clock=clock)

    cfg = settings if settings is not None else get_default_settings()

    account_groups = [
        "inflow",
        "outflow",
        "balance_pattern",
        "routine_life",
        "lifecycle",
        "control",
        "region",
        "network",
        "intelligence",
    ]

    # Account features are computed in every case except no_watermark
    account_feats = compute_account_features(
        client=redis_client,
        run_id=run_id,
        bank_id=bank_id,
        account_id=account_id,
        decision_time=decision_time,
        arrival_cutoff=arrival_cutoff,
        settings=cfg,
        groups=account_groups,
        screener=recording_screener,
        dpip_key=dpip_key,
    )

    payment_feats: dict[str, Any]
    missing_request_fields: list[str]
    if missing:
        # Compute payer-only features; payee-VPA features computed if receiver_vpa present (G5)
        status = "incomplete_request"
        missing_request_fields = list(missing)
        payer_only = compute_payer_only_payment_features(
            client=redis_client,
            run_id=run_id,
            bank_id=bank_id,
            payer_account_id=account_id,
            amount_paise=request.amount_paise,
            event_time=decision_time,
            arrival_cutoff=arrival_cutoff,
            settings=cfg,
        )
        first_payment_to_payee: Any = None
        payee_same_bank_risk: Any = None
        payee_registry_match: Any = None

        if request.receiver_vpa:
            vpa_feats = compute_payee_vpa_payment_features(
                client=redis_client,
                run_id=run_id,
                bank_id=bank_id,
                payer_account_id=account_id,
                payee_bank_id=request.receiver_bank_id,
                payee_vpa=request.receiver_vpa,
                event_time=decision_time,
                arrival_cutoff=arrival_cutoff,
                settings=cfg,
                screener=recording_screener,
                dpip_key=dpip_key,
            )
            first_payment_to_payee = vpa_feats["first_payment_to_payee"]
            payee_same_bank_risk = vpa_feats["payee_same_bank_risk"]
            payee_registry_match = vpa_feats["payee_registry_match"]

        payee_type = request.payee_type if request.payee_type in ("person", "merchant") else None
        payment_feats = {
            "amount_vs_payer_median_ratio": payer_only["amount_vs_payer_median_ratio"],
            "amount_vs_payer_max_ratio": payer_only["amount_vs_payer_max_ratio"],
            "customer_failed_attempts_24h": payer_only["customer_failed_attempts_24h"],
            "hour_ist": payer_only["hour_ist"],
            "payee_type": payee_type,
            "first_payment_to_payee": first_payment_to_payee,
            "payee_same_bank_risk": payee_same_bank_risk,
            "payee_registry_match": payee_registry_match,
        }
    else:
        status = "ok"
        missing_request_fields = []
        assert request.receiver_vpa is not None
        assert request.payee_type in ("person", "merchant")
        proposed = ProposedPayment(
            run_id=run_id,
            bank_id=bank_id,
            payer_account_id=account_id,
            payee_bank_id=request.receiver_bank_id,
            payee_vpa=request.receiver_vpa,
            payee_type=request.payee_type,
            amount_paise=request.amount_paise,
            event_time=decision_time,
        )
        payment_feats = compute_payment_features(
            client=redis_client,
            proposed=proposed,
            arrival_cutoff=arrival_cutoff,
            settings=cfg,
            screener=recording_screener,
            dpip_key=dpip_key,
        )

    features_dict: dict[str, Any] = {}
    for name in canonical_names:
        if name in account_feats:
            features_dict[name] = account_feats[name]
        elif name in payment_feats:
            features_dict[name] = payment_feats[name]
        else:
            features_dict[name] = None

    registry_answers = (
        _format_registry_answers(recording_screener.answers)
        if (recording_screener is not None and recording_screener.answers)
        else []
    )

    return FeatureLogRow(
        bank_id=bank_id,
        payment_id=payment_id,
        run_id=run_id,
        account_id=account_id,
        decision_time=decision_time,
        decided_at=decided_at,
        arrival_cutoff=arrival_cutoff,
        watermark_lag_ms=watermark_lag_ms,
        status=status,
        feature_values=features_dict,
        missing_request_fields=missing_request_fields,
        registry_answers=registry_answers,
    )
