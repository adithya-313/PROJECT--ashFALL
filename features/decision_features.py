# File: features/decision_features.py
# Purpose: Level 15c decision-time feature computation and validation.
# Issue: Level 15c watermark and feature log (plan Task 1, D4).
"""
WHAT: Computes decision-time feature vectors and builds FeatureLogRow instances
for shadow logging during /v1/decisions evaluation.

WHY: Enforces Section 22 evaluation correctness: decision features are computed
only when the bank's ingestion watermark is >= decision_time. If unwatermarked,
status is marked 'no_watermark' without guessing or falling back to wall-clock time.
"""

import logging
from datetime import datetime, timezone
from typing import Any

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
from features.m1_payment import compute_payer_only_payment_features
from features.registry_screening import RegistryScreener

logger = logging.getLogger(__name__)

__all__ = ["SystemClock", "compute_decision_row"]


class SystemClock:
    """System clock providing UTC and IST timezone-aware datetimes."""

    def now_utc(self) -> datetime:
        return datetime.now(timezone.utc)

    def now_ist(self) -> datetime:
        return datetime.now(timezone.utc)


def compute_decision_row(
    redis_client: Any,
    request: Any,
    clock: Any = None,
    settings: AccountFeaturesSettings | None = None,
    screener: RegistryScreener | None = None,
    bank_id: str | None = None,
) -> FeatureLogRow:
    """
    WHAT: Computes all canonical M1 features as-of decision_time for a DecisionRequest.

    WHY: Decouples real-time payment decisions from training feature vector collection.
    Guarantees strict Section 22 correctness via watermark verification.
    """
    computed_at = datetime.now(timezone.utc)
    clk = clock if clock is not None else SystemClock()
    if hasattr(clk, "now_utc"):
        decision_time: datetime = clk.now_utc()
    elif hasattr(clk, "now_ist"):
        decision_time = clk.now_ist().astimezone(timezone.utc)
    elif callable(clk):
        decision_time = clk()
    else:
        decision_time = datetime.now(timezone.utc)

    if decision_time.tzinfo is None:
        decision_time = decision_time.replace(tzinfo=timezone.utc)

    resolved_bank_id = bank_id
    if resolved_bank_id is None:
        resolved_bank_id = getattr(request, "bank_id", None)
    if resolved_bank_id is None:
        resolved_bank_id = "bank_0"

    payment_id = getattr(request, "payment_id", "")
    decision_id = f"{payment_id}-dec"

    canonical_names = canonical_feature_names()
    all_none_features: dict[str, Any] = {name: None for name in canonical_names}

    # 1. Watermark check: read bank watermark across assigned Kafka partitions [0]
    watermark: datetime | None = None
    if redis_client is not None:
        try:
            watermark = get_bank_watermark(redis_client, resolved_bank_id, assigned_partitions=[0])
        except Exception as exc:
            logger.warning("Failed to read watermark for bank %s: %s", resolved_bank_id, exc)
            watermark = None

    if watermark is None or watermark < decision_time:
        return FeatureLogRow(
            bank_id=resolved_bank_id,
            payment_id=payment_id,
            decision_id=decision_id,
            decision_time=decision_time,
            arrival_watermark=watermark,
            status="no_watermark",
            feature_values=dict(all_none_features),
            computed_at=computed_at,
        )

    # 2. Incomplete request check
    sender_account_id = getattr(request, "sender_account_id", None)
    amount_paise = getattr(request, "amount_paise", None)
    event_time = getattr(request, "event_time", None)

    if not payment_id or not sender_account_id or amount_paise is None or amount_paise <= 0 or event_time is None:
        return FeatureLogRow(
            bank_id=resolved_bank_id,
            payment_id=payment_id,
            decision_id=decision_id,
            decision_time=decision_time,
            arrival_watermark=watermark,
            status="incomplete_request",
            feature_values=dict(all_none_features),
            computed_at=computed_at,
        )

    # 3. Compute account and payment features
    try:
        run_id = extract_run_id(payment_id)
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
        account_feats = compute_account_features(
            client=redis_client,
            run_id=run_id,
            bank_id=resolved_bank_id,
            account_id=sender_account_id,
            decision_time=decision_time,
            arrival_cutoff=watermark,
            settings=cfg,
            groups=account_groups,
            screener=screener,
        )

        proposal_dict = {
            "run_id": run_id,
            "bank_id": resolved_bank_id,
            "payer_account_id": sender_account_id,
            "payee_bank_id": getattr(request, "receiver_bank_id", None),
            "payee_vpa": getattr(request, "receiver_vpa", None),
            "amount_paise": amount_paise,
            "channel": getattr(request, "channel", None),
            "payee_type": getattr(request, "payee_type", None),
            "event_time": event_time,
        }
        payment_feats = compute_payer_only_payment_features(
            redis_client=redis_client,
            proposal=proposal_dict,
            decision_time=decision_time,
            settings=cfg,
            screener=screener,
            arrival_cutoff=watermark,
        )

        features_dict: dict[str, Any] = {}
        ordered_values: list[float | None] = []
        for name in canonical_names:
            if name in account_feats:
                val = account_feats[name]
            elif name in payment_feats:
                val = payment_feats[name]
            else:
                val = None
            features_dict[name] = val

            if val is None:
                ordered_values.append(None)
            elif isinstance(val, bool):
                ordered_values.append(1.0 if val else 0.0)
            elif isinstance(val, (int, float)):
                ordered_values.append(float(val))
            elif name == "payee_type":
                ordered_values.append(0.0 if val == "person" else 1.0 if val == "merchant" else None)
            elif name == "account_type":
                ordered_values.append(0.0 if val == "savings" else 1.0 if val == "current" else None)
            else:
                ordered_values.append(None)

        return FeatureLogRow(
            bank_id=resolved_bank_id,
            payment_id=payment_id,
            decision_id=decision_id,
            decision_time=decision_time,
            arrival_watermark=watermark,
            status="ok",
            feature_values=features_dict,
            feature_values_ordered=ordered_values,
            computed_at=computed_at,
        )
    except Exception as exc:
        logger.error(
            "Feature computation failed for payment %s on bank %s: %s",
            payment_id,
            resolved_bank_id,
            exc,
        )
        return FeatureLogRow(
            bank_id=resolved_bank_id,
            payment_id=payment_id,
            decision_id=decision_id,
            decision_time=decision_time,
            arrival_watermark=watermark,
            status="error",
            feature_values={"error": str(exc)},
            computed_at=computed_at,
        )
