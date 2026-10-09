"""
File: tests/unit/test_audit_v1.py
Purpose: Proves the small, training-free parts of scripts/audit_v1.py: the feature groups cover the 15 model v1
         features exactly once, the variant helpers drop the right columns, the audit plan has the expected variants,
         and a result row without an MLflow run id is rejected. No data, model or MLflow is needed.
"""

import pandas as pd
import pytest

from scripts.audit_v1 import (
    ALL_FEATURES,
    BALANCE,
    GENERATED,
    TYPE_FLAGS,
    build_variants,
    check_feature_groups,
    drop_group,
    keep_only,
    validate_row,
)

# The 15 features of model v1, in the order prepare_features builds them (scripts/fraud_baseline.py).
MODEL_V1_FEATURES = [
    "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "account_avg_amount",
    "many_small_inbound_count", "large_outbound_amount", "new_device_flag", "geo_velocity_spike",
    "type_CASH_IN", "type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER",
]


def good_row() -> dict:
    """A complete result row."""
    return {"name": "control_all15", "group": "control", "features": ["amount"], "split": "level5_random",
            "pr_auc": 0.99, "mlflow_run_id": "abc123", "device": "cpu", "n_train": 10, "n_test": 5}


def test_groups_cover_the_15_features_exactly_once() -> None:
    """Amount plus the three groups equal the 15 model v1 features, with no overlap."""
    check_feature_groups()
    assert sorted(ALL_FEATURES) == sorted(MODEL_V1_FEATURES)
    assert len(BALANCE) == 4 and len(TYPE_FLAGS) == 5 and len(GENERATED) == 5
    assert not set(BALANCE) & set(GENERATED) and not set(TYPE_FLAGS) & set(GENERATED)


def test_groups_match_the_real_prepare_features() -> None:
    """If the Level 5 libraries are installed, prepare_features really returns these 15 columns."""
    pytest.importorskip("lightgbm", reason="the Level 5 libraries are only installed with requirements-ml.txt")
    pytest.importorskip("catboost", reason="the Level 5 libraries are only installed with requirements-ml.txt")
    from scripts.audit_v1 import level5_features

    raw = pd.DataFrame({"amount": [1.0], "type": ["PAYMENT"], "isFraud": [0],
                        **{name: [0.0] for name in BALANCE + GENERATED}})
    features, _ = level5_features(raw)
    assert sorted(features.columns) == sorted(MODEL_V1_FEATURES)


def test_variant_helpers_drop_and_keep_the_right_columns() -> None:
    """drop_group removes only the named columns; keep_only leaves exactly one."""
    table = pd.DataFrame({name: [1, 2] for name in ALL_FEATURES})
    without_balance = drop_group(table, BALANCE)
    assert set(without_balance.columns) == set(ALL_FEATURES) - set(BALANCE) and "amount" in without_balance.columns
    assert list(keep_only(table, "type_PAYMENT").columns) == ["type_PAYMENT"]


def test_the_plan_has_the_control_ablations_oot_and_15_single_features() -> None:
    """The plan lists 21 variants: control, 3 ablations, 2 out-of-time and one per feature."""
    variants = build_variants()
    names = [variant["name"] for variant in variants]
    assert len(names) == len(set(names)) == 21
    assert {"control_all15", "ablate_no_balance", "ablate_no_type", "ablate_no_generated", "oot_all15",
            "oot_no_balance"} <= set(names)
    assert sorted(name.removeprefix("single_") for name in names if name.startswith("single_")) == sorted(ALL_FEATURES)


def test_a_row_without_a_run_id_is_rejected() -> None:
    """A complete row passes; an empty run id, a missing field or an unknown group is refused."""
    validate_row(good_row())
    with pytest.raises(ValueError, match="no MLflow run id"):
        validate_row({**good_row(), "mlflow_run_id": ""})
    missing = good_row()
    del missing["pr_auc"]
    with pytest.raises(ValueError, match="missing"):
        validate_row(missing)
    with pytest.raises(ValueError, match="unknown group"):
        validate_row({**good_row(), "group": "made_up"})
