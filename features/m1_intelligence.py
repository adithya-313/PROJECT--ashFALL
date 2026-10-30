# File: features/m1_intelligence.py
# Purpose: Level 15b M1 intelligence group feature calculation functions.
# Issue: Level 15b M1 features (plan Task 1, G1, R8, architect rules 1, 2, 3, decision 9c).
"""
WHAT: Pure and state-backed functions computing M1 intelligence group features:
  - registry_match_count: Number of this account's identifiers (its own account hash plus
    each visible lifetime device fingerprint) that have an active fraud match on the national registry.
  - registry_match_age_days: Whole days between the earliest first_visible_at of any matching
    identifier and the decision time.

WHY:
  - Connects bank transaction decisions to national shared fraud intelligence (DPIP).
  - Enforces Rule 3 (honest degradation): if any screen returns 'unknown' or the bank-side DPIP key
    is not configured, returns None (unknown) for both features. Never returns a partial or false count.
  - Preserves tenant isolation (A6): account hashes are namespaced with the bank index loaded from config.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from core.config import config
from features.account_keys import (
    AccountFeaturesSettings,
    get_default_settings,
)
from features.account_summaries import read_lifetime
from features.registry_screening import RegistryScreener
from nationalsim.keyed_hash import compute_identifier_hash


def load_bank_ids_by_index(config_path: Path | None = None) -> dict[int, str]:
    """
    WHAT: Loads the bank_ids_by_index mapping from config/bank_feed.yaml.
    WHY: Design rule: commercial banks must map bank_id to bank_index via config, never string parsing.
    """
    root_dir = Path(__file__).resolve().parent.parent
    path = config_path if config_path is not None else root_dir / "config" / "bank_feed.yaml"
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    mapping = data["bank_feed"]["bank_ids_by_index"]["mapping"]
    return {int(k): str(v) for k, v in mapping.items()}


def get_bank_index(bank_id: str, config_path: Path | None = None) -> int:
    """Resolve integer simulation bank index for a bank identifier string."""
    mapping = load_bank_ids_by_index(config_path)
    index_by_id = {v: k for k, v in mapping.items()}
    if bank_id not in index_by_id:
        raise ValueError(f"Unknown bank_id {bank_id!r} not found in bank_ids_by_index mapping")
    return index_by_id[bank_id]


def compute_intelligence_features(
    client: Any,
    run_id: str,
    bank_id: str,
    account_id: str,
    decision_time: datetime,
    arrival_cutoff: datetime,
    settings: AccountFeaturesSettings | None = None,
    screener: RegistryScreener | None = None,
    dpip_key: str | None = None,
) -> dict[str, int | None]:
    """
    PURPOSE:
        Computes M1 intelligence group features (registry_match_count, registry_match_age_days)
        as of the decision timestamp.

    INPUTS:
        client: Redis client.
        run_id: Simulation run identifier.
        bank_id: Tenant commercial bank identifier.
        account_id: Customer account identifier.
        decision_time: Historical evaluation decision timestamp.
        arrival_cutoff: Storage watermark cutoff.
        settings: Optional AccountFeaturesSettings.
        screener: Optional RegistryScreener instance.
        dpip_key: Optional explicit DPIP HMAC key; defaults to config.dpip_participant_key.

    RETURNS:
        Dictionary mapping feature names to int counts / ages, or None if unknown.
    """
    cfg = settings if settings is not None else get_default_settings()

    # If no screener provided or no DPIP key available, features are unknown (Rule 3)
    if screener is None:
        return {
            "registry_match_count": None,
            "registry_match_age_days": None,
        }

    resolved_key: str | None = dpip_key
    if resolved_key is None and config.dpip_participant_key is not None:
        resolved_key = config.dpip_participant_key.get_secret_value()

    if not resolved_key:
        return {
            "registry_match_count": None,
            "registry_match_age_days": None,
        }

    bank_idx = get_bank_index(bank_id)

    # 1. Screen the account identifier
    account_hash = compute_identifier_hash(
        key=resolved_key,
        run_id=run_id,
        identifier_type="account",
        raw_value=account_id,
        bank_index=bank_idx,
    )
    outcome_acc = screener.screen(account_hash, "account", as_of=decision_time)
    if outcome_acc.status == "unknown":
        return {
            "registry_match_count": None,
            "registry_match_age_days": None,
        }

    # 2. Screen each visible lifetime device fingerprint
    lifetime = read_lifetime(client, run_id, bank_id, account_id, decision_time, arrival_cutoff, cfg)
    screened_outcomes = [outcome_acc]

    for fp in sorted(lifetime.fingerprints):
        fp_hash = compute_identifier_hash(
            key=resolved_key,
            run_id=run_id,
            identifier_type="device",
            raw_value=fp,
            bank_index=None,
        )
        outcome_fp = screener.screen(fp_hash, "device", as_of=decision_time)
        if outcome_fp.status == "unknown":
            return {
                "registry_match_count": None,
                "registry_match_age_days": None,
            }
        screened_outcomes.append(outcome_fp)

    # 3. Calculate match count and earliest first_visible_at
    matches = [o for o in screened_outcomes if o.status == "match"]
    match_count = len(matches)

    if match_count == 0:
        return {
            "registry_match_count": 0,
            "registry_match_age_days": None,
        }

    visible_times = [m.first_visible_at for m in matches if m.first_visible_at is not None]
    if not visible_times:
        return {
            "registry_match_count": match_count,
            "registry_match_age_days": None,
        }

    earliest_visible = min(visible_times)
    # Convert both to UTC to calculate elapsed seconds safely
    dt_utc = decision_time.astimezone(timezone.utc)
    ev_utc = earliest_visible.astimezone(timezone.utc)
    delta_seconds = (dt_utc - ev_utc).total_seconds()
    age_days = max(0, int(delta_seconds // 86400))

    return {
        "registry_match_count": match_count,
        "registry_match_age_days": age_days,
    }

