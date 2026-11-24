# File: tests/unit/evaluation/test_calibration.py
# Purpose: Pin the A-060/A-060a isotonic calibration rules: the fit, the application, the ranking
# guard, the test bins, the truth mapping, the calibrator file, the task 4 reproduction and the
# account-day loader.

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import xgboost

from core.event_clock import IST
from decisions.mule_account_model import account_model_feature_names
from decisions.pu_learning import PUBaggedAccountModel
from evaluation.calibration import (
    WorldRecord,
    account_split_rows,
    assert_ranking_preserved,
    calibration_metrics,
    isotonic_apply,
    isotonic_fit,
    load_calibrator,
    truth_label,
    verify_task4_reproduction,
    write_calibrator,
)

APPLICATION_RULE = "linear interpolation between the thresholds, clipped to [y_min, y_max]"


def test_isotonic_fit_keeps_the_thresholds_and_apply_interpolates_then_clips():
    calibrator = isotonic_fit([0.1, 0.2, 0.3], [0, 1, 1])

    assert calibrator.x_thresholds == (0.1, 0.2, 0.3)
    assert calibrator.y_thresholds == (0.0, 1.0, 1.0)
    applied = isotonic_apply(calibrator, np.array([0.05, 0.15, 0.25, 0.9]))
    np.testing.assert_allclose(applied, [0.0, 0.5, 1.0, 1.0], rtol=0.0, atol=1e-12)

    with pytest.raises(ValueError):
        isotonic_fit([0.1, 0.2], [0, 0])
    with pytest.raises(ValueError):
        isotonic_fit([0.1, 0.2], [0])
    with pytest.raises(ValueError):
        isotonic_fit([0.1, float("nan")], [0, 1])
    with pytest.raises(ValueError):
        isotonic_fit([0.1, 1.5], [0, 1])
    with pytest.raises(ValueError):
        isotonic_apply(calibrator, [])


def test_isotonic_fit_pools_an_adjacent_violation_into_one_block():
    calibrator = isotonic_fit([0.1, 0.2, 0.3, 0.4], [0, 1, 0, 1])

    assert calibrator.x_thresholds == (0.1, 0.2, 0.3, 0.4)
    assert calibrator.y_thresholds == (0.0, 0.5, 0.5, 1.0)


def test_ranking_guard_accepts_a_monotone_calibration_and_rejects_a_reorder():
    assert assert_ranking_preserved([0.3, 0.1, 0.2], [1.0, 0.0, 0.5], "validation") == {
        "rows": 3,
        "monotone": True,
    }
    with pytest.raises(ValueError):
        assert_ranking_preserved([0.3, 0.1, 0.2], [0.1, 0.5, 0.0], "validation")
    with pytest.raises(ValueError):
        assert_ranking_preserved([0.1, 0.2], [0.5], "validation")


def test_brier_and_ece_match_the_hand_computed_values_and_the_bins_cover_every_row():
    probabilities = [0.1] * 90 + [0.9] * 10
    truth = [0] * 90 + [1] * 10

    metrics = calibration_metrics(probabilities, truth)

    assert metrics["rows"] == 100
    assert metrics["positives"] == 10
    assert metrics["brier"] == pytest.approx(0.01, abs=1e-12)
    assert metrics["ece"] == pytest.approx(0.10, abs=1e-12)
    assert [point["count"] for point in metrics["bins"]] == [10] * 10
    assert sum(point["count"] for point in metrics["bins"]) == 100
    for point in metrics["bins"][:9]:
        assert point["mean_probability"] == pytest.approx(0.1, abs=1e-12)
        assert point["observed_rate"] == 0.0
        assert point["min_probability"] == pytest.approx(0.1, abs=1e-12)
        assert point["max_probability"] == pytest.approx(0.1, abs=1e-12)
    assert metrics["bins"][9]["mean_probability"] == pytest.approx(0.9, abs=1e-12)
    assert metrics["bins"][9]["observed_rate"] == 1.0


def test_truth_label_scores_the_three_mule_roles_high_and_the_two_genuine_roles_low():
    assert truth_label("mule") == 1
    assert truth_label("collector") == 1
    assert truth_label("cash_out") == 1
    assert truth_label("genuine") == 0
    assert truth_label("victim") == 0
    with pytest.raises(ValueError):
        truth_label("retired")


def test_the_calibrator_file_round_trips_and_refuses_a_changed_payload(tmp_path):
    calibrator = isotonic_fit([0.1, 0.2, 0.3, 0.4], [0, 1, 0, 1])
    path = tmp_path / "calibrator.json"
    write_calibrator(path, calibrator)

    assert load_calibrator(path) == calibrator
    np.testing.assert_allclose(
        isotonic_apply(load_calibrator(path), [0.15, 0.25, 0.9]),
        isotonic_apply(calibrator, [0.15, 0.25, 0.9]),
        rtol=0.0,
        atol=0.0,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["application"] == APPLICATION_RULE
    assert payload["y_min"] == 0.0 and payload["y_max"] == 1.0

    extra = dict(payload, extra=1)
    path.write_text(json.dumps(extra), encoding="utf-8")
    with pytest.raises(ValueError):
        load_calibrator(path)
    bad_rule = dict(payload, application="nearest neighbour")
    path.write_text(json.dumps(bad_rule), encoding="utf-8")
    with pytest.raises(ValueError):
        load_calibrator(path)
    decreasing = dict(payload, y_thresholds=[0.0, 0.6, 0.5, 1.0])
    path.write_text(json.dumps(decreasing), encoding="utf-8")
    with pytest.raises(ValueError):
        load_calibrator(path)


def test_task4_reproduction_accepts_the_recorded_shape_and_rejects_any_drift():
    reference = {
        "splits": {
            "test": {
                "worlds": [
                    {"world_id": "w1", "rows": 100, "positives": 10, "truth_ap": 0.5},
                    {"world_id": "w2", "rows": 50, "positives": 5, "truth_ap": 0.25},
                ],
                "aggregate": {"truth_ap": {"median": 0.375}},
            }
        }
    }
    records = [WorldRecord("w1", 100, 10, 0.5), WorldRecord("w2", 50, 5, 0.25)]

    assert verify_task4_reproduction(records, reference, 150, 15) == {
        "worlds": 2,
        "rows": 150,
        "positives": 15,
        "truth_ap_median": 0.375,
    }
    with pytest.raises(ValueError):
        verify_task4_reproduction([WorldRecord("w1", 100, 10, 0.5), WorldRecord("w2", 50, 5, 0.3)], reference, 150, 15)
    with pytest.raises(ValueError):
        verify_task4_reproduction([WorldRecord("w1", 99, 10, 0.5), records[1]], reference, 150, 15)
    with pytest.raises(ValueError):
        verify_task4_reproduction(records, reference, 151, 15)
    with pytest.raises(ValueError):
        verify_task4_reproduction([records[1], records[0]], reference, 150, 15)


def _write_truth_file(run_dir: Path, name: str, rows: list[dict[str, object]]) -> dict[str, object]:
    """Writes one truth JSON Lines file and returns its manifest entry."""
    text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    data = text.encode("utf-8")
    (run_dir / name).write_bytes(data)
    return {"sha256": hashlib.sha256(data).hexdigest(), "row_count": len(rows)}


def test_account_split_rows_keeps_the_latest_day_and_labels_the_truth_role(tmp_path):
    features = account_model_feature_names()
    matrix = np.zeros((4, len(features)))
    members = []
    for seed in (1, 2):
        member = xgboost.XGBClassifier(n_estimators=2, max_depth=1, random_state=seed)
        member.fit(matrix, [0, 1, 0, 1])
        members.append(member)
    model = PUBaggedAccountModel(
        members=tuple(members),
        member_seeds=(1, 2),
        n_bags=2,
        n_estimators=2,
        unlabelled_per_positive=1,
    )

    dataset_dir = tmp_path / "validation-a"
    dataset_dir.mkdir()
    snapshots = pa.table(
        {
            "world_id": ["w1", "w1", "w1"],
            "bank_id": ["bank", "bank", "bank"],
            "account_id": ["acc_a", "acc_a", "acc_b"],
            "snapshot_time": pa.array(
                [
                    datetime(2026, 9, 1, 10, 0, tzinfo=IST),
                    datetime(2026, 9, 1, 21, 0, tzinfo=IST),
                    datetime(2026, 9, 2, 21, 0, tzinfo=IST),
                ],
                type=pa.timestamp("us", tz="Asia/Kolkata"),
            ),
            "snapshot_kind": ["full", "daily", "daily"],
        }
    )
    pq.write_table(snapshots, dataset_dir / "snapshots.parquet")
    (dataset_dir / "manifest.json").write_text(
        json.dumps(
            {
                "split": "validation",
                "content_hash": "validation-hash",
                "worlds": [{"world_id": "w1", "run_id": "run1", "clock_start": "2026-09-01"}],
            }
        ),
        encoding="utf-8",
    )

    truth_root = tmp_path / "truth"
    run_dir = truth_root / "run1"
    run_dir.mkdir(parents=True)
    files = {
        "roles.jsonl": [{"account_id": "acc_a", "valid_from": 1260, "valid_to": None, "role": "mule"}],
        "device_control.jsonl": [],
        "payment_stories.jsonl": [],
        "gangs.jsonl": [],
    }
    manifest = {
        "format": "truth-v2",
        "files": {name: _write_truth_file(run_dir, name, rows) for name, rows in files.items()},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    scores, truth, diagnostics = account_split_rows([dataset_dir], "validation", model, truth_root)

    assert diagnostics == {"rows": 2, "positives": 1, "negatives": 1, "worlds": 1}
    assert truth.tolist() == [1, 0]
    assert scores.shape == (2,)
    assert bool(np.all(np.isfinite(scores)))
    assert bool(np.all((scores > 0.0) & (scores < 1.0)))
