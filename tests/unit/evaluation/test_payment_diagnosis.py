import math

import numpy as np
import pytest

from evaluation.payment_diagnosis import dedupe_payees, feature_truth_ap, oracle_recall


def test_diagnosis_helpers_are_the_hand_computed_values():
    labels = np.asarray([1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8)
    assert oracle_recall(labels, 0.2) == 1.0
    assert oracle_recall(labels, 0.1) == 0.5
    assert oracle_recall(np.zeros(4, dtype=np.int8), 0.5) is None
    with pytest.raises(ValueError):
        oracle_recall(labels, 0.0)

    values = np.asarray([0.9, math.nan, 0.8, 0.1], dtype=np.float64)
    ap_value, missing = feature_truth_ap(values, np.asarray([1, 0, 0, 1], dtype=np.int8))
    assert missing == 1
    assert math.isclose(ap_value, (1.0 + 2.0 / 3.0) / 2.0, rel_tol=0.0, abs_tol=1e-12)
    none_ap, _ = feature_truth_ap(np.asarray([1.0, 2.0]), np.asarray([0, 0], dtype=np.int8))
    assert none_ap is None
    with pytest.raises(ValueError):
        feature_truth_ap(np.asarray([1.0]), np.asarray([0, 1], dtype=np.int8))

    held = [("w", "a", 1), ("w", "a", 0), ("w", "b", 0), ("w2", "a", 1)]
    deduped = dedupe_payees(held)
    assert deduped["held_payments"] == 4
    assert deduped["distinct_payees"] == 3
    assert deduped["positive_payees"] == 2
    assert deduped["payee_precision"] == 2.0 / 3.0
    assert deduped["payees_for_half"] == 1
    assert deduped["payees_for_half_share"] == 1.0 / 3.0
    with pytest.raises(ValueError):
        dedupe_payees([])
