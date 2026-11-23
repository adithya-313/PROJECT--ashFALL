"""Tests for the Level 18 payment model's feature order, rows, device rule and search adapter."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from evaluation.hyperparameter_search import validation_truth_average_precision
from evaluation.leakage_checks_5_6 import (
    bank_visible_fields,
    event_record_fields,
    simulator_only_fields,
    truth_file_columns,
)
from evaluation.payment_model import (
    DEFAULT_FEATURE_ORDER_PATH,
    DEFAULT_SCHEMA_PATH,
    TARGET,
    _subset_ap,
    build_feature_order,
    build_matrix,
    choose_device,
    feature_list_sha256,
    load_feature_order,
    load_schema_features,
    load_split_rows,
    payment_scored,
)

VOCABULARIES = {"account_type": ["current", "savings"], "payee_type": ["merchant", "person"]}


def test_feature_order_places_payee_risk_then_payment_level_then_payer_account() -> None:
    payload = build_feature_order(DEFAULT_SCHEMA_PATH, VOCABULARIES)
    features = payload["features"]
    records = load_schema_features(DEFAULT_SCHEMA_PATH)
    payment_level = [str(record["name"]) for record in records if record["group"] == "payment_level"]
    payer = [str(record["name"]) for record in records if record["group"] != "payment_level"]

    assert features[0] == "payee_risk"
    assert features[1 : 1 + len(payment_level)] == payment_level
    assert features[1 + len(payment_level) :] == payer
    assert len(features) == 1 + len(payment_level) + len(payer) == 54
    assert payload["feature_sha256"] == feature_list_sha256(features)
    assert payload["groups"]["payment_level"] == payment_level
    assert payload["groups"]["payer_account"] == payer


def test_feature_order_rejects_missing_categorical_encoding() -> None:
    with pytest.raises(ValueError, match="account_type"):
        build_feature_order(DEFAULT_SCHEMA_PATH, {"payee_type": ["merchant", "person"]})


def test_committed_feature_order_matches_schema_and_sha() -> None:
    payload = load_feature_order(DEFAULT_FEATURE_ORDER_PATH)
    rebuilt = build_feature_order(DEFAULT_SCHEMA_PATH, payload["categorical_encodings"])
    assert payload["features"] == rebuilt["features"]
    assert payload["feature_sha256"] == feature_list_sha256(payload["features"])
    assert len(payload["features"]) == 54


def test_build_matrix_encodes_categoricals_and_keeps_missing_as_nan() -> None:
    table = pa.table(
        {
            "payee_type": ["merchant", "person", None],
            "amount": pa.array([1.5, None, float("nan")], pa.float64()),
        }
    )
    matrix = build_matrix(table, ["payee_type", "amount"], {"payee_type": ["merchant", "person"]})

    assert matrix.dtype == np.float32
    assert matrix[0, 0] == 0.0
    assert matrix[1, 0] == 1.0
    assert np.isnan(matrix[2, 0])
    assert matrix[0, 1] == pytest.approx(1.5)
    assert np.isnan(matrix[1, 1])
    assert np.isnan(matrix[2, 1])


def test_build_matrix_refuses_a_category_outside_the_frozen_encoding() -> None:
    table = pa.table({"payee_type": ["shop"]})
    with pytest.raises(ValueError, match="'shop'"):
        build_matrix(table, ["payee_type"], {"payee_type": ["merchant", "person"]})


def _write_dataset(folder: Path, world_id: str, payment_ids: list[str], split: str = "training") -> None:
    folder.mkdir(parents=True)
    (folder / "manifest.json").write_text(json.dumps({"split": split}), encoding="utf-8")
    table = pa.table(
        {
            "world_id": [world_id] * len(payment_ids),
            "payment_id": payment_ids,
            "bank_id": ["bank_0"] * len(payment_ids),
            "decision_time": pa.array([1, 2][: len(payment_ids)], pa.timestamp("us", tz="UTC")),
            "receiver_account_id": ["acc"] * len(payment_ids),
            TARGET: [True] * len(payment_ids),
            "payee_same_bank_risk": [0.0] * len(payment_ids),
        }
    )
    pq.write_table(table, folder / "payments.parquet")


def _write_oof(folder: Path, world_id: str, rows: list[tuple[str, float, str]], split: str) -> None:
    per_world = folder / "per_world"
    per_world.mkdir(parents=True)
    table = pa.table(
        {
            "world_id": [world_id] * len(rows),
            "payment_id": [row[0] for row in rows],
            "split": [split] * len(rows),
            "same_bank": [True] * len(rows),
            "payee_risk": pa.array([row[1] for row in rows], pa.float64()),
            "nan_reason": [row[2] for row in rows],
        }
    )
    pq.write_table(table, per_world / f"{world_id}.parquet")


def test_load_split_rows_aligns_payments_and_payee_risk_on_sorted_keys(tmp_path: Path) -> None:
    dataset = tmp_path / "training-ds"
    _write_dataset(dataset, "w1", ["p2", "p1"])
    oof = tmp_path / "oof"
    _write_oof(oof, "w1", [("p1", 0.1, ""), ("p2", float("nan"), "receiver_at_other_bank")], "training")

    rows = load_split_rows([dataset], "training", ["payee_risk", "payee_same_bank_risk"], oof)

    assert rows["payment_id"].to_pylist() == ["p1", "p2"]
    assert rows["payee_risk"].to_pylist()[0] == pytest.approx(0.1)
    assert np.isnan(rows["payee_risk"].to_pylist()[1])
    assert rows["same_bank"].to_pylist() == [True, True]


def test_load_split_rows_refuses_rows_that_do_not_align(tmp_path: Path) -> None:
    dataset = tmp_path / "training-ds"
    _write_dataset(dataset, "w1", ["p1", "p2"])
    oof = tmp_path / "oof"
    _write_oof(oof, "w1", [("p1", 0.1, "")], "training")

    with pytest.raises(ValueError, match="rows"):
        load_split_rows([dataset], "training", ["payee_risk"], oof)


def test_load_split_rows_refuses_the_wrong_split(tmp_path: Path) -> None:
    dataset = tmp_path / "validation-ds"
    _write_dataset(dataset, "w1", ["p1"], split="validation")
    oof = tmp_path / "oof"
    _write_oof(oof, "w1", [("p1", 0.1, "")], "validation")

    with pytest.raises(ValueError, match="passed as 'training'"):
        load_split_rows([dataset], "training", ["payee_risk"], oof)


def test_device_rule_uses_the_005_boundary_strictly_at_the_limit() -> None:
    device, reason = choose_device(cpu_ap=0.1, cpu_seconds=2.0, cuda_ap=0.105, cuda_seconds=1.0, cuda_error=None)
    assert device == "cuda", reason

    device, reason = choose_device(cpu_ap=0.4, cpu_seconds=2.0, cuda_ap=0.4051, cuda_seconds=1.0, cuda_error=None)
    assert device == "cpu", reason


def test_device_rule_keeps_cpu_when_cuda_fails_or_is_slower() -> None:
    device, reason = choose_device(
        cpu_ap=0.4, cpu_seconds=2.0, cuda_ap=None, cuda_seconds=0.5, cuda_error="XGBoostError: no GPU"
    )
    assert device == "cpu"
    assert "no GPU" in reason

    device, reason = choose_device(cpu_ap=0.4, cpu_seconds=2.0, cuda_ap=0.4, cuda_seconds=3.0, cuda_error=None)
    assert device == "cpu"
    assert "not lower" in reason


def test_subset_ap_reports_rows_positives_and_score() -> None:
    result = _subset_ap([0.9, 0.1, 0.8], [1, 0, 1], np.array([True, True, False]), "same-bank")
    assert result["rows"] == 2
    assert result["positives"] == 1
    assert result["truth_ap"] == pytest.approx(1.0)


def test_search_objective_reads_the_adapter_role() -> None:
    score = validation_truth_average_precision([0.9, 0.1], payment_scored(["mule", "genuine"]))
    assert score == pytest.approx(1.0)


def test_payment_model_inputs_are_bank_visible_and_never_truth_or_simulator_fields(tmp_path: Path) -> None:
    truth_path = tmp_path / "roles.jsonl"
    truth_path.write_text(
        json.dumps({"world_id": "w", "account_id": "a", "minute": 0, "role": "mule"}) + "\n",
        encoding="utf-8",
    )
    world_fields = event_record_fields() | truth_file_columns(truth_path)
    simulator_only = simulator_only_fields(world_fields, bank_visible_fields())
    truth_fields = truth_file_columns(truth_path) | {"truth_role_receiver", "truth_owner", "truth_depth"}
    catalog = {str(record["name"]): record for record in load_schema_features(DEFAULT_SCHEMA_PATH)}
    inputs = load_feature_order(DEFAULT_FEATURE_ORDER_PATH)["features"]

    assert "purpose" in simulator_only
    for name in inputs:
        assert name not in simulator_only, f"{name} is a simulator-only field"
        assert name not in truth_fields, f"{name} is a truth field"
        assert name not in (TARGET, "label_payment_named_by_cutoff"), f"{name} is a label"
        if name == "payee_risk":
            continue
        assert name in catalog, f"{name} is not a bank-visible M1 feature"
        source = catalog[name].get("bank_visible_source_records")
        assert isinstance(source, str) and source, f"{name} has no bank-visible source record"


def test_feature_order_file_lives_in_the_committed_config_folder() -> None:
    assert DEFAULT_FEATURE_ORDER_PATH.parent.name == "config"
    assert yaml.safe_load(DEFAULT_FEATURE_ORDER_PATH.read_text(encoding="utf-8"))["target"] == TARGET
