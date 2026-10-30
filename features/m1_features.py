# File: features/m1_features.py
# Purpose: Shared entry point, schema drift enforcement, and feature vector extraction for M1 live mule features.
# Issue: Level 15b M1 features (plan Task 1, G1, G2, R8, R5, R7).
"""
WHAT: Provides the shared high-level feature extraction entry point compute_account_features(),
orchestrating point-in-time snapshot loading, closed-day summary queries, and dispatching to
group-specific pure feature calculation functions.

WHY: Guarantees strict architectural separation between Redis I/O and behavioral calculations.
Enforces that every returned FeatureVector strictly matches the canonical data/schemas/feature_set_m1.yaml
catalog, raising ValueError on any drift, unknown groups, or missing values.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from features.account_keys import (
    AccountFeaturesSettings,
    get_default_settings,
)
from features.account_reader import read_account_snapshot
from features.account_summaries import read_counterparty_days
from features.feature_set import load_feature_set_m1
from features.m1_inflow_outflow import (
    compute_below_threshold_credit_count_7d,
    compute_cash_out_channels_7d,
    compute_distinct_payers,
    compute_first_time_payer_share_24h,
    compute_median_credit_to_debit_seconds_7d,
    compute_other_bank_outflow_share_7d,
    compute_pass_through_ratio,
    compute_small_credit_count,
    compute_top_receiver_share_7d,
)
from features.registry_screening import RegistryScreener

FeatureVector = dict[str, int | float | str | bool | None]

# Set of functional groups supported by the calculation engine in Level 15b.
SUPPORTED_GROUPS: frozenset[str] = frozenset(
    {
        "inflow",
        "outflow",
        "balance_pattern",
        "routine_life",
        "lifecycle",
        "control",
        "region",
        "network",
        "intelligence",
    }
)


@dataclass(frozen=True)
class M1FeaturesConfig:
    """
    Configuration parameters for M1 feature calculations loaded from config/account_features.yaml.
    """

    small_credit_max_paise: int
    reporting_thresholds_paise: tuple[int, ...]
    below_threshold_band_fraction: float
    near_zero_balance_max_paise: int


def load_m1_config(config_path: Path | None = None) -> M1FeaturesConfig:
    """
    WHAT: Loads behavioral thresholds from config/account_features.yaml without hardcoded numbers.
    WHY: Code honesty rule: thresholds must come from configuration, never typed literals.
    """
    root_dir = Path(__file__).resolve().parent.parent
    if config_path is None:
        config_path = root_dir / "config" / "account_features.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)

    block = raw_cfg["account_features"]
    raw_thresholds = block["reporting_thresholds_paise"]["value"]

    return M1FeaturesConfig(
        small_credit_max_paise=int(block["small_credit_max_paise"]["value"]),
        reporting_thresholds_paise=tuple(int(v) for v in raw_thresholds),
        below_threshold_band_fraction=float(block["below_threshold_band_fraction"]["value"]),
        near_zero_balance_max_paise=int(block["near_zero_balance_max_paise"]["value"]),
    )


def compute_account_features(
    client: Any,
    run_id: str,
    bank_id: str,
    account_id: str,
    decision_time: datetime,
    arrival_cutoff: datetime,
    settings: AccountFeaturesSettings | None = None,
    groups: Sequence[str] = ("inflow", "outflow", "balance_pattern"),
    config: M1FeaturesConfig | None = None,
    screener: RegistryScreener | None = None,
    dpip_key: str | None = None,
) -> FeatureVector:
    """
    PURPOSE:
        Computes the FeatureVector for the requested functional groups using Redis account state.

    INPUTS:
        client: Redis or FakeRedis client.
        run_id: Simulation run identifier.
        bank_id: Tenant commercial bank identifier.
        account_id: Customer account identifier.
        decision_time: Scoring evaluation timestamp (upper bound on event_time).
        arrival_cutoff: Ingestion watermark storage cutoff timestamp (upper bound on arrival_time).
        settings: Optional AccountFeaturesSettings.
        groups: Sequence of group names to compute (e.g. ('inflow', 'outflow', 'balance_pattern')).
        config: Optional pre-loaded M1FeaturesConfig.

    KEY LOGIC:
        1. Validates that every requested group is in SUPPORTED_GROUPS. An unknown group raises ValueError.
        2. Queries the point-in-time AccountSnapshot exactly once.
        3. If 'inflow' is requested, queries closed counterparty credit days once.
        4. Invokes group pure functions to compute feature values.
        5. Validates returned feature keys against data/schemas/feature_set_m1.yaml; any drift
           between schema catalog and computed keys raises ValueError.

    RETURN:
        FeatureVector: Dictionary mapping feature name to computed value or None ('empty').
    """
    for g in groups:
        if g not in SUPPORTED_GROUPS:
            raise ValueError(f"Unknown feature group {g!r}; supported groups are {sorted(SUPPORTED_GROUPS)}")

    cfg_settings = settings if settings is not None else get_default_settings()
    m1_cfg = config if config is not None else load_m1_config()

    # Read account snapshot once
    snapshot = read_account_snapshot(
        client=client,
        run_id=run_id,
        bank_id=bank_id,
        account_id=account_id,
        decision_time=decision_time,
        arrival_cutoff=arrival_cutoff,
        settings=cfg_settings,
    )

    result: FeatureVector = {}

    if "inflow" in groups:
        # Read closed credit counterparty days for first_time_payer_share_24h
        closed_cp_days = read_counterparty_days(
            client=client,
            run_id=run_id,
            bank_id=bank_id,
            account_id=account_id,
            direction="credit",
            decision_time=decision_time,
            arrival_cutoff=arrival_cutoff,
            settings=cfg_settings,
        )

        result["distinct_payers_1h"] = compute_distinct_payers(snapshot.side_records, decision_time, 3600)
        result["distinct_payers_24h"] = compute_distinct_payers(snapshot.side_records, decision_time, 86400)
        result["distinct_payers_7d"] = compute_distinct_payers(snapshot.side_records, decision_time, 604800)
        result["first_time_payer_share_24h"] = compute_first_time_payer_share_24h(
            snapshot.side_records, closed_cp_days, decision_time
        )
        result["small_credit_count_24h"] = compute_small_credit_count(
            snapshot.side_records, decision_time, 86400, m1_cfg.small_credit_max_paise
        )
        result["small_credit_count_7d"] = compute_small_credit_count(
            snapshot.side_records, decision_time, 604800, m1_cfg.small_credit_max_paise
        )
        result["below_threshold_credit_count_7d"] = compute_below_threshold_credit_count_7d(
            snapshot.side_records,
            decision_time,
            m1_cfg.reporting_thresholds_paise,
            m1_cfg.below_threshold_band_fraction,
        )

    if "outflow" in groups:
        result["pass_through_ratio_24h"] = compute_pass_through_ratio(
            snapshot.side_records, snapshot.withdrawals, decision_time, 86400
        )
        result["pass_through_ratio_7d"] = compute_pass_through_ratio(
            snapshot.side_records, snapshot.withdrawals, decision_time, 604800
        )
        result["median_credit_to_debit_seconds_7d"] = compute_median_credit_to_debit_seconds_7d(
            snapshot.side_records, snapshot.withdrawals, decision_time
        )
        result["top_receiver_share_7d"] = compute_top_receiver_share_7d(snapshot.side_records, decision_time)
        result["other_bank_outflow_share_7d"] = compute_other_bank_outflow_share_7d(
            snapshot.side_records, bank_id, decision_time
        )
        cash_channels = compute_cash_out_channels_7d(snapshot.side_records, snapshot.withdrawals, decision_time)
        result.update(cash_channels)

    if "balance_pattern" in groups:
        from features.m1_balance import (
            compute_current_balance_paise,
            compute_min_balance_paise_7d,
            compute_near_zero_cycles_7d,
        )

        result["current_balance_paise"] = compute_current_balance_paise(
            snapshot.side_records, snapshot.withdrawals, decision_time
        )
        result["min_balance_paise_7d"] = compute_min_balance_paise_7d(
            snapshot.side_records, snapshot.withdrawals, decision_time
        )
        result["near_zero_cycles_7d"] = compute_near_zero_cycles_7d(
            snapshot.side_records,
            snapshot.withdrawals,
            decision_time,
            m1_cfg.near_zero_balance_max_paise,
        )

    if "routine_life" in groups:
        from features.m1_routine_lifecycle import compute_routine_life_features

        routine_feats = compute_routine_life_features(
            client=client,
            run_id=run_id,
            bank_id=bank_id,
            account_id=account_id,
            decision_time=decision_time,
            arrival_cutoff=arrival_cutoff,
            settings=cfg_settings,
        )
        result.update(routine_feats)

    if "lifecycle" in groups:
        from features.m1_routine_lifecycle import compute_lifecycle_features

        lifecycle_feats = compute_lifecycle_features(
            client=client,
            run_id=run_id,
            bank_id=bank_id,
            account_id=account_id,
            snapshot=snapshot,
            decision_time=decision_time,
            arrival_cutoff=arrival_cutoff,
            settings=cfg_settings,
        )
        result.update(lifecycle_feats)

    if "control" in groups:
        from features.m1_control_region import compute_control_features

        control_feats = compute_control_features(
            client=client,
            run_id=run_id,
            bank_id=bank_id,
            account_id=account_id,
            snapshot=snapshot,
            decision_time=decision_time,
            arrival_cutoff=arrival_cutoff,
            settings=cfg_settings,
        )
        result.update(control_feats)

    if "region" in groups:
        from features.m1_control_region import compute_region_features

        region_feats = compute_region_features(
            client=client,
            run_id=run_id,
            bank_id=bank_id,
            account_id=account_id,
            snapshot=snapshot,
            decision_time=decision_time,
            arrival_cutoff=arrival_cutoff,
            settings=cfg_settings,
        )
        result.update(region_feats)

    if "network" in groups:
        from features.m1_network import compute_network_features

        network_feats = compute_network_features(
            client=client,
            run_id=run_id,
            bank_id=bank_id,
            account_id=account_id,
            snapshot=snapshot,
            decision_time=decision_time,
            arrival_cutoff=arrival_cutoff,
            settings=cfg_settings,
        )
        result.update(network_feats)

    if "intelligence" in groups:
        from features.m1_intelligence import compute_intelligence_features

        intel_feats = compute_intelligence_features(
            client=client,
            run_id=run_id,
            bank_id=bank_id,
            account_id=account_id,
            decision_time=decision_time,
            arrival_cutoff=arrival_cutoff,
            settings=cfg_settings,
            screener=screener,
            dpip_key=dpip_key,
        )
        result.update(intel_feats)

    # Schema drift check: guarantee returned keys exactly match canonical schema for requested groups
    feature_set = load_feature_set_m1()
    expected_names: set[str] = set()
    for g in groups:
        expected_names.update(f.name for f in feature_set.features_by_group(g))

    actual_names = set(result.keys())
    if actual_names != expected_names:
        missing = expected_names - actual_names
        unexpected = actual_names - expected_names
        raise ValueError(
            f"M1 feature drift detected between schema and calculation engine. "
            f"Missing features: {sorted(missing)}, Unexpected features: {sorted(unexpected)}"
        )

    return result
