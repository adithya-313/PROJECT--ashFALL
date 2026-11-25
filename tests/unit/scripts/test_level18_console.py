"""
File: tests/unit/scripts/test_level18_console.py
Purpose: Proves the Level 18 console generator copies frozen numbers, records a manifest of input hashes and the
         commit hash, reads its lists from the source keys, computes only the E1 median, and refuses missing,
         malformed or contradictory inputs while leaving every input untouched.

The whole run is written into a temporary repo-shaped tree, and the real generator is called on it.
"""

import hashlib
import importlib
import json
from pathlib import Path
from types import ModuleType

import pytest

CONSOLE_FILES = (
    "calibration.json",
    "comparison.json",
    "held_out_families.json",
    "hidden_depth.json",
    "leakage.json",
    "onnx_parity.json",
    "payment_results.json",
    "reason_codes.json",
    "yield_e1.json",
)
LEAKAGE_ORDER = (
    "truth_leak",
    "split_leak",
    "check_5a",
    "check_5b",
    "check_6a",
    "check_6b",
    "section_23_check_7",
)


def console() -> ModuleType:
    """Return the real console generator module."""
    return importlib.import_module("scripts.level18_console")


def _write(root: Path, rel_path: str, document: dict) -> Path:
    """Write one fixture JSON document under the temporary root."""
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _spread(value: float, worlds: int = 3) -> dict:
    """Return a frozen aggregate block with a median and a spread."""
    return {"median": value, "p10": value / 2.0, "p90": value, "unit": "fraction", "world_count": worlds}


def _scorer_block(measured: float, reweighted: float) -> dict:
    """Return one frozen scorer block carrying every metric the generator copies."""
    return {
        "recall_at_capacity": _spread(0.5),
        "yield_measured": _spread(measured),
        "yield_reweighted": _spread(reweighted),
    }


def _run_a(per_world: dict) -> dict:
    """Return a frozen Run A result shaped like the real one."""
    return {
        "across_worlds": {
            "test": {"rules": _scorer_block(0.4, 0.1), "xgboost": _scorer_block(0.9, 0.2)},
            "stress": {
                "xgboost": {"recall_at_capacity": {"status": "not run: no stress worlds supplied", "world_count": 0}}
            },
        },
        "per_world": per_world,
    }


def _extra() -> dict:
    """Return a frozen extra-reporting document with two depths and one family."""
    return {
        "manifest": {"xgboost_cut": 0.75},
        "test": {
            "worlds_processed": 3,
            "world_cards_checked": 3,
            "aggregate": {
                "hidden_by_depth": {
                    "1": {"accounts_total": 6, "accounts_median": 6.0, "recall": _spread(1.0)},
                    "2": {"accounts_total": 3, "accounts_median": 3.0, "recall": _spread(0.5)},
                },
                "families": {
                    "slow_drip": {
                        "accounts_total": 4,
                        "recall_at_capacity": _spread(0.5),
                        "recall_at_cut": _spread(0.9),
                        "worlds_used": 3,
                        "worlds_absent": 0,
                    }
                },
            },
        },
    }


def _leakage_a() -> dict:
    """Return the first frozen leakage document: three checks, trivial world and the check-4 record."""
    return {
        "checks": {
            "truth_leak": {"status": "pass", "feature_count": 42},
            "split_leak": {"status": "pass", "overlaps": []},
            "section_23_check_7": {"status": "pass", "validation": {}},
            "trivial_world": {"status": "pass", "best_feature": "account_age_days"},
            "section_23_checks_4_to_7": {"status": "not run", "reason": "later round", "checks": [4, 5, 6, 7]},
        }
    }


def _leakage_b() -> dict:
    """Return the second frozen leakage document: checks 5a, 5b, 6a and 6b."""
    return {
        "status": "fail",
        "check_5a": {"status": "pass"},
        "check_5b": {"status": "pass"},
        "check_6a": {"status": "pass"},
        "check_6b": {"status": "fail", "average_precision_over_base_rate": 10.4},
    }


def _payment_aggregate() -> dict:
    """Return one payment split aggregate block."""
    return {
        "truth_ap": _spread(0.6),
        "recall": _spread(0.2),
        "precision": _spread(0.9),
        "hold_rate": _spread(0.005),
        "same_bank_truth_ap": _spread(0.8),
        "other_bank_truth_ap": _spread(0.4),
    }


def _payment_result() -> dict:
    """Return a frozen payment test result shaped like the real one."""
    return {
        "threshold": 0.2,
        "hold_share": 0.005,
        "splits": {
            "test": {
                "aggregate": _payment_aggregate(),
                "nan_payee_risk_share": 0.6,
                "nan_reason_counts": {"scored": 1},
            },
            "stress": {"aggregate": _payment_aggregate(), "nan_payee_risk_share": 0.5},
        },
        "manifest": {
            "git_hash": "d" * 40,
            "model": {
                "sha256": "e" * 64,
                "feature_sha256": "f" * 64,
                "feature_count": 54,
                "rounds": 647,
                "seed": 1717,
            },
        },
    }


def _diagnosis() -> dict:
    """Return a frozen payment diagnosis shaped like the real one."""
    return {
        "splits": {
            "test": {"payee_precision": {"payee_precision": 0.911}, "feature_truth_ap": {"payee_risk": _spread(0.9)}},
            "stress": {"feature_truth_ap": {"payee_risk": _spread(0.5)}},
        },
        "leakage": {
            "status": "fail",
            "item_status": {"id_leak": "fail", "temporal_cutoff": "pass"},
            "items": {"id_leak": {"run_a_check_6b": {"status": "fail", "average_precision_over_base_rate": 10.4}}},
        },
    }


def _calibration_state(offset: float) -> dict:
    """Return one calibration state with two reliability bins."""
    return {
        "rows": 10,
        "positives": 2,
        "brier": 0.1,
        "ece": 0.02,
        "bins": [
            {
                "bin": 0,
                "count": 5,
                "mean_probability": offset,
                "observed_rate": 0.0,
                "min_probability": 0.0,
                "max_probability": 0.5,
            },
            {
                "bin": 1,
                "count": 5,
                "mean_probability": offset + 0.1,
                "observed_rate": 0.4,
                "min_probability": 0.5,
                "max_probability": 1.0,
            },
        ],
    }


def _calibration_model(name: str) -> dict:
    """Return one model's calibration block with both states and a calibrator."""
    return {
        "test": {"before_calibration": _calibration_state(0.0), "after_calibration": _calibration_state(0.01)},
        "calibrator": {
            "path": f"C:\\work\\{name}_isotonic.json",
            "sha256": "a" * 64,
            "application": "linear interpolation between the thresholds, clipped to [y_min, y_max]",
        },
    }


def _reason_codes(name: str) -> dict:
    """Return one reason-code document with one published code."""
    return {
        "model": name,
        "feature_count": 42,
        "feature_sha256": "b" * 64,
        "frequency": {
            "rows": 20,
            "rows_without_reason": 0,
            "codes": [
                {
                    "code": "ROUTINE_LIFE",
                    "group": "routine_life",
                    "plain_reason": "routine life",
                    "rows": 10,
                    "share": 0.5,
                    "first": 1,
                }
            ],
        },
        "manifest": {"sample_rows": 20, "seed": 1717},
    }


def _parity() -> dict:
    """Return one frozen parity block."""
    return {"rows": 10, "nan_rows": 0, "max_abs_diff": 1e-07, "mean_abs_diff": 1e-08, "rows_over_tolerance": 0}


def _onnx() -> dict:
    """Return a frozen ONNX export document for both models."""
    return {
        "account": {
            "onnx": {"path": "C:\\work\\mule_account_xgb.onnx", "sha256": "c" * 64},
            "parity": {"validation": _parity(), "test": _parity()},
            "merge_proof": {"spearman": 1.0, "decision_flips": 0},
        },
        "payment": {
            "onnx": {"path": "C:\\work\\mule_payment_xgb.onnx", "sha256": "d" * 64},
            "parity": {"validation": _parity(), "test": _parity()},
        },
        "tolerances": {"parity": 1e-05},
        "converter": {"onnxmltools": "1.16.0"},
    }


def _default_worlds() -> dict:
    """Return three worlds: two with a positive card factor and one with a zero factor."""
    return {
        "test-0000": ("run01", 2.0, 1.0, 0.5),
        "test-0001": ("run02", 4.0, 0.8, 0.25),
        "test-0002": ("run03", 0.0, 0.9, None),
    }


def _fixture_root(tmp_path: Path, worlds: dict | None = None) -> Path:
    """Write a complete frozen-like input tree under tmp_path and return its root."""
    module = console()
    worlds = _default_worlds() if worlds is None else worlds
    per_world = {}
    for world_key, (run_id, factor, measured, reweighted) in worlds.items():
        per_world[world_key] = {
            "run_id": run_id,
            "scorers": {"xgboost": {"yield_rate": {"measured": measured, "reweighted": reweighted}}},
        }
        card = {"fraud_inflation_factor": factor, "complaints": 6, "complaints_per_100000": 26.0}
        _write(tmp_path, f"{module.WORLD_CARDS_DIR}/{run_id}/base_rate.json", card)
    _write(tmp_path, module.RUN_A_RESULT, _run_a(per_world))
    _write(tmp_path, module.RUN_A_EXTRA, _extra())
    _write(tmp_path, module.RUN_A_LEAKAGE, _leakage_a())
    _write(tmp_path, module.RUN_A_LEAKAGE_5_6, _leakage_b())
    _write(tmp_path, module.PAYMENT_RESULT, _payment_result())
    _write(tmp_path, module.PAYMENT_DIAGNOSIS, _diagnosis())
    calibration = {"models": {"account": _calibration_model("account"), "payment": _calibration_model("payment")}}
    _write(tmp_path, module.CALIBRATION, calibration)
    _write(tmp_path, module.REASON_ACCOUNT, _reason_codes("account"))
    _write(tmp_path, module.REASON_PAYMENT, _reason_codes("payment"))
    _write(tmp_path, module.ONNX_EXPORT, _onnx())
    return tmp_path


def _data(root: Path, name: str) -> dict:
    """Return the data block of one generated console file."""
    document = json.loads((root / "reports" / "level_18" / "console" / name).read_text(encoding="utf-8"))
    return document["data"]


def test_build_writes_nine_files_with_manifests(tmp_path: Path) -> None:
    """Every console file carries simulated true and a manifest hashing every input it read."""
    root = _fixture_root(tmp_path)
    paths = console().build(root)
    assert sorted(path.name for path in paths) == sorted(CONSOLE_FILES)
    for path in paths:
        document = json.loads(path.read_text(encoding="utf-8"))
        assert document["simulated"] is True
        assert document["manifest"]["generator"] == "scripts/level18_console.py"
        assert len(document["manifest"]["git_hash"]) == 40
        assert document["manifest"]["sources"]
        for rel_path, digest in document["manifest"]["sources"].items():
            assert digest == hashlib.sha256((root / rel_path).read_bytes()).hexdigest()
        assert path.read_bytes().endswith(b"\n")


def test_lists_come_from_sources(tmp_path: Path) -> None:
    """Scorers, depths, families, reason models and the seven checks are read from the source keys."""
    root = _fixture_root(tmp_path)
    console().build(root)
    assert sorted(_data(root, "comparison.json")["test"]) == ["rules", "xgboost"]
    assert list(_data(root, "hidden_depth.json")["hidden_by_depth"]) == ["1", "2"]
    assert list(_data(root, "held_out_families.json")["families"]) == ["slow_drip"]
    assert sorted(_data(root, "reason_codes.json")["models"]) == ["account", "payment"]
    checks = _data(root, "leakage.json")["checks"]
    assert [check["check"] for check in checks] == list(LEAKAGE_ORDER)
    assert [check["status"] for check in checks] == ["pass", "pass", "pass", "pass", "pass", "fail", "pass"]
    assert _data(root, "leakage.json")["trivial_world"]["status"] == "pass"


def test_yield_e1_median_and_guard(tmp_path: Path) -> None:
    """The E1 median uses only positive factors and weaves the undefined world through as null."""
    root = _fixture_root(tmp_path)
    console().build(root)
    data = _data(root, "yield_e1.json")
    assert [world["fraud_inflation_factor"] for world in data["worlds"]] == [2.0, 4.0, 0.0]
    assert data["worlds"][2]["reweighted"] is None
    assert data["inflation_factor"] == {"worlds_defined": 2, "worlds_zero_factor": 1, "median_defined": 3.0}
    assert data["scorers"]["xgboost"]["yield_measured"]["median"] == 0.9


def test_yield_e1_disagreement_raises(tmp_path: Path) -> None:
    """A zero card factor with a frozen re-weighted yield stops the build instead of being copied."""
    root = _fixture_root(tmp_path, {"test-0000": ("run01", 0.0, 0.5, 0.5)})
    with pytest.raises(ValueError, match="disagree"):
        console().build(root)


def test_missing_input_raises(tmp_path: Path) -> None:
    """A missing input file raises a ValueError naming the path."""
    root = _fixture_root(tmp_path)
    (root / console().ONNX_EXPORT).unlink()
    with pytest.raises(ValueError, match="console input is missing: runs/full/level18b/onnx/onnx_export.json"):
        console().build(root)


def test_malformed_input_raises(tmp_path: Path) -> None:
    """Invalid JSON in an input raises a ValueError naming the path."""
    root = _fixture_root(tmp_path)
    (root / console().CALIBRATION).write_text("{", encoding="utf-8")
    message = "console input is not valid JSON: runs/full/level18b/calibration/calibration_data.json"
    with pytest.raises(ValueError, match=message):
        console().build(root)


def test_missing_key_raises(tmp_path: Path) -> None:
    """A required key removed from an input raises a ValueError naming the key."""
    root = _fixture_root(tmp_path)
    extra = json.loads((root / console().RUN_A_EXTRA).read_text(encoding="utf-8"))
    del extra["manifest"]["xgboost_cut"]
    _write(root, console().RUN_A_EXTRA, extra)
    with pytest.raises(ValueError, match="xgboost_cut"):
        console().build(root)


def test_inputs_not_modified(tmp_path: Path) -> None:
    """The build leaves every input byte-identical."""
    root = _fixture_root(tmp_path)
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in root.rglob("*.json")}
    console().build(root)
    for path, digest in before.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
