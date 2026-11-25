"""
File: tests/unit/scripts/test_level18_model_cards.py
Purpose: Proves the Level 18 model-card generator copies the committed console numbers into both cards, states
         the five limitations with their console values, renders the same numbers in Markdown, records the A-063
         manifest and refuses missing, malformed or key-short console files while leaving every input untouched.

The whole run is written into a temporary repo-shaped tree, and the real generator is called on it.
"""

import hashlib
import importlib
import json
from pathlib import Path
from types import ModuleType

import pytest

CONSOLE_DIR = "reports/level_18/console"
ACCOUNT_SOURCES = (
    f"{CONSOLE_DIR}/comparison.json",
    f"{CONSOLE_DIR}/hidden_depth.json",
    f"{CONSOLE_DIR}/held_out_families.json",
    f"{CONSOLE_DIR}/calibration.json",
    f"{CONSOLE_DIR}/leakage.json",
    f"{CONSOLE_DIR}/onnx_parity.json",
)
PAYMENT_SOURCES = (
    f"{CONSOLE_DIR}/payment_results.json",
    f"{CONSOLE_DIR}/calibration.json",
    f"{CONSOLE_DIR}/onnx_parity.json",
)


def cards() -> ModuleType:
    """Return the real model-card generator module."""
    return importlib.import_module("scripts.level18_model_cards")


def _spread(median: float, worlds: int = 5) -> dict:
    """Return one frozen median/p10/p90 block with its world count."""
    return {"median": median, "p10": median / 2.0, "p90": median, "world_count": worlds, "unit": "fraction"}


def _comparison() -> dict:
    """Return the console comparison data block with one published xgboost block."""
    return {
        "decision_cut": {"key": "manifest.xgboost_cut", "source": "runs/x/extra_reporting.json", "value": 0.75},
        "test_world_count": 5,
        "test": {
            "xgboost": {
                "recall_at_capacity": _spread(0.64),
                "genuine_restriction_rate": _spread(0.006),
                "median_time_to_detection_hours": _spread(11.5),
                "yield_measured": _spread(0.92),
                "yield_reweighted": {
                    "median": 0.27,
                    "p10": 0.13,
                    "p90": 1.0,
                    "worlds_used": 4,
                    "worlds_defined": 4,
                    "worlds_undefined": 1,
                },
            }
        },
    }


def _hidden_depth() -> dict:
    """Return the console hidden-depth data block with depths 1 and 5."""
    return {
        "worlds_processed": 5,
        "world_cards_checked": 5,
        "hidden_by_depth": {
            "1": {"accounts_total": 12, "accounts_median": 6.0, "recall": _spread(1.0)},
            "5": {
                "accounts_total": 4,
                "accounts_median": 2.0,
                "recall": {"median": 0.5, "p10": 0.25, "p90": 1.0, "world_count": 2, "worlds_used": 2},
            },
        },
    }


def _families() -> dict:
    """Return the console held-out family data block with one family."""
    return {
        "worlds_processed": 5,
        "families": {
            "device_rotation": {
                "accounts_total": 7,
                "recall_at_cut": _spread(0.7, worlds=3),
                "recall_at_capacity": _spread(0.4, worlds=3),
                "worlds_absent": 2,
                "worlds_used": 3,
            }
        },
    }


def _calibration() -> dict:
    """Return the console calibration data block for both models."""

    def model(brier: float, ece: float, name: str) -> dict:
        return {
            "raw": {"brier": brier, "ece": ece, "rows": 10, "positives": 2},
            "calibrated": {"brier": brier / 4.0, "ece": ece / 4.0, "rows": 10, "positives": 2},
            "calibrator": {
                "file": name,
                "sha256": "a" * 64,
                "application": "linear interpolation between the thresholds, clipped to [y_min, y_max]",
            },
        }

    return {
        "points": {},
        "summaries": {
            "account": {
                key: value for key, value in model(0.058, 0.127, "account_isotonic.json").items() if key != "calibrator"
            },
            "payment": {
                key: value for key, value in model(0.017, 0.012, "payment_isotonic.json").items() if key != "calibrator"
            },
        },
        "calibrator": {
            "account": model(0.058, 0.127, "account_isotonic.json")["calibrator"],
            "payment": model(0.017, 0.012, "payment_isotonic.json")["calibrator"],
        },
    }


def _leakage() -> dict:
    """Return the console leakage data block with the seven checks in order and check 6b failed."""
    return {
        "checks": [
            {"check": "truth_leak", "status": "pass"},
            {"check": "split_leak", "status": "pass"},
            {"check": "check_5a", "status": "pass"},
            {"check": "check_5b", "status": "pass"},
            {"check": "check_6a", "status": "pass"},
            {"check": "check_6b", "status": "fail"},
            {"check": "section_23_check_7", "status": "pass"},
        ],
        "trivial_world": {"status": "pass"},
        "not_implemented": [],
        "section_23_checks_4_to_7": {},
    }


def _aggregate(truth_ap: float, other_bank: float) -> dict:
    """Return one payment split aggregate block."""
    return {
        "truth_ap": _spread(truth_ap),
        "precision": _spread(0.99),
        "recall": _spread(0.25),
        "hold_rate": _spread(0.005),
        "same_bank_truth_ap": _spread(0.87),
        "other_bank_truth_ap": _spread(other_bank),
    }


def _payment_results() -> dict:
    """Return the console payment-results data block for the test and stress splits."""
    return {
        "threshold": 0.2,
        "hold_share": 0.005,
        "frozen_model": {
            "sha256": "e" * 64,
            "feature_sha256": "f" * 64,
            "feature_count": 54,
            "rounds": 647,
            "seed": 1717,
            "git_hash": "d" * 40,
        },
        "splits": {
            "test": {
                "aggregate": _aggregate(0.62, 0.47),
                "nan_payee_risk_share": 0.617,
                "nan_reason_counts": {"scored": 60, "receiver_at_other_bank": 1234, "no_prior_account_day": 5},
            },
            "stress": {
                "aggregate": _aggregate(0.07, 0.08),
                "nan_payee_risk_share": 0.61,
                "nan_reason_counts": {"scored": 30, "receiver_at_other_bank": 3000},
            },
        },
        "payee_precision": {
            "payee_precision": 0.911,
            "reported_per_world_precision_median": 0.994,
            "distinct_payees": 10,
            "held_payments": 100,
        },
        "feature_truth_ap": {
            "test": {"median": 0.93, "p10": 0.5, "p90": 1.0, "worlds_used": 4},
            "stress": {"median": 0.55, "p10": 0.2, "p90": 0.9, "worlds_used": 4},
        },
        "leakage": {
            "status": "fail",
            "item_status": {"id_leak": "fail", "temporal_cutoff": "pass"},
            "run_a_check_6b": {
                "status": "fail",
                "average_precision_over_base_rate": 10.4,
                "max_ap_over_base_rate": 2.0,
                "validation_average_precision": 0.6,
                "validation_base_rate": 0.06,
            },
        },
    }


def _parity_block(nan_rows: int = 3) -> dict:
    """Return one split's parity record as the console publishes it."""
    return {
        "max_abs_diff": 1.234e-06,
        "mean_abs_diff": 3.0e-07,
        "rows": nan_rows,
        "nan_rows": nan_rows,
        "rows_over_tolerance": 0,
    }


def _onnx_parity() -> dict:
    """Return the console ONNX-parity data block for both models."""
    return {
        "tolerances": {"max_abs_diff": 1e-05},
        "converter": {"package": "skl2onnx"},
        "models": {
            "account": {
                "file": "mule_account_xgb.onnx",
                "sha256": "b" * 64,
                "parity": {"validation": _parity_block(), "test": _parity_block()},
                "merge_proof": {"spearman": 1.0, "decision_flips": 0, "top_set_identical": True},
            },
            "payment": {
                "file": "mule_payment_xgb.onnx",
                "sha256": "c" * 64,
                "parity": {"validation": _parity_block(), "test": _parity_block()},
            },
        },
    }


def _documents() -> dict:
    """Return every console document the card generator reads, keyed by its relative path."""
    return {
        f"{CONSOLE_DIR}/comparison.json": _comparison(),
        f"{CONSOLE_DIR}/hidden_depth.json": _hidden_depth(),
        f"{CONSOLE_DIR}/held_out_families.json": _families(),
        f"{CONSOLE_DIR}/calibration.json": _calibration(),
        f"{CONSOLE_DIR}/leakage.json": _leakage(),
        f"{CONSOLE_DIR}/payment_results.json": _payment_results(),
        f"{CONSOLE_DIR}/onnx_parity.json": _onnx_parity(),
    }


def _envelope(data: dict) -> dict:
    """Return one console document in the A-063 envelope."""
    return {
        "simulated": True,
        "manifest": {"generator": "scripts/level18_console.py", "git_hash": "0" * 40, "sources": {}},
        "data": data,
    }


def _tree(root: Path) -> dict:
    """Write every fixture console document under the temporary root and return the documents."""
    documents = _documents()
    for rel_path, data in documents.items():
        path = root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_envelope(data)), encoding="utf-8")
    return documents


def _build(root: Path) -> tuple[dict, dict]:
    """Run the real generator into a temporary output folder and return both card documents."""
    cards().build(root=root, output_dir=root / "out")
    account = json.loads((root / "out" / "mule_account_xgb.json").read_text(encoding="utf-8"))
    payment = json.loads((root / "out" / "mule_payment_xgb.json").read_text(encoding="utf-8"))
    return account, payment


def test_build_writes_four_files_with_envelopes(tmp_path):
    """The run writes both cards as JSON and Markdown, each JSON with the A-063 envelope and source digests."""
    documents = _tree(tmp_path)
    account, payment = _build(tmp_path)

    for name in ("mule_account_xgb.json", "mule_account_xgb.md", "mule_payment_xgb.json", "mule_payment_xgb.md"):
        assert (tmp_path / "out" / name).is_file()

    for document, sources in ((account, ACCOUNT_SOURCES), (payment, PAYMENT_SOURCES)):
        assert document["simulated"] is True
        assert document["manifest"]["generator"] == "scripts/level18_model_cards.py"
        assert len(document["manifest"]["git_hash"]) == 40
        assert set(document["manifest"]["sources"]) == set(sources)
        for rel_path in sources:
            digest = hashlib.sha256((tmp_path / rel_path).read_bytes()).hexdigest()
            assert document["manifest"]["sources"][rel_path] == digest
        data = document["data"]
        assert isinstance(data["intended_use"], str)
        assert isinstance(data["not_intended_for"], list)
        assert data["training_data"]["label"] == "SIMULATED"

    assert sorted(account["data"]["metrics"]["recall_by_depth"]) == ["1", "5"]
    assert sorted(payment["data"]["metrics"]["truth_ap"]) == ["stress", "test"]
    assert set(documents) == set(ACCOUNT_SOURCES) | set(PAYMENT_SOURCES)


def test_card_numbers_equal_console_values(tmp_path):
    """Every copied metric carries the exact console value, spread and world count."""
    _tree(tmp_path)
    account, payment = _build(tmp_path)
    metrics = account["data"]["metrics"]

    assert metrics["recall_at_capacity"] == {
        "median": 0.64,
        "p10": 0.32,
        "p90": 0.64,
        "worlds": 5,
    }
    assert metrics["yield_reweighted"] == {
        "median": 0.27,
        "p10": 0.13,
        "p90": 1.0,
        "worlds": 4,
        "worlds_defined": 4,
        "worlds_undefined": 1,
    }
    assert metrics["recall_by_depth"]["5"] == {
        "median": 0.5,
        "p10": 0.25,
        "p90": 1.0,
        "worlds": 2,
        "accounts_median": 2.0,
        "accounts_total": 4,
    }
    family = metrics["held_out_families"]["device_rotation"]
    assert family["accounts_total"] == 7
    assert family["recall_at_cut"]["median"] == 0.7
    assert family["worlds_absent"] == 2
    assert account["data"]["calibration"]["calibrated"]["ece"] == 0.127 / 4.0
    assert account["data"]["onnx"]["parity"]["test"]["max_abs_diff"] == 1.234e-06
    assert account["data"]["onnx"]["merge_proof"]["decision_flips"] == 0
    assert [(check["check"], check["status"]) for check in account["data"]["leakage"]["checks"]][5] == (
        "check_6b",
        "fail",
    )

    pay_metrics = payment["data"]["metrics"]
    assert pay_metrics["truth_ap"]["stress"]["median"] == 0.07
    assert pay_metrics["precision"]["test"]["median"] == 0.99
    assert pay_metrics["other_bank_truth_ap"]["median"] == 0.47
    assert pay_metrics["other_bank_truth_ap"]["stress"]["median"] == 0.08
    assert pay_metrics["other_bank_truth_ap"]["worlds"] == 5
    assert pay_metrics["payee_precision"]["payee_precision"] == 0.911
    assert pay_metrics["feature_truth_ap_payee_risk"]["test"]["median"] == 0.93
    assert pay_metrics["nan_payee_risk_share"] == {"test": 0.617, "stress": 0.61}
    assert pay_metrics["nan_reason_counts"]["test"]["receiver_at_other_bank"] == 1234
    assert payment["data"]["threshold"] == 0.2
    assert payment["data"]["hold_share"] == 0.005
    assert payment["data"]["frozen_model"]["rounds"] == 647
    assert payment["data"]["leakage"]["run_a_check_6b"]["average_precision_over_base_rate"] == 10.4
    assert payment["data"]["calibration"]["calibrator"]["file"] == "payment_isotonic.json"


def test_limitations_embed_console_numbers(tmp_path):
    """The payment card states all five limitations and the first three carry the console numbers."""
    _tree(tmp_path)
    account, payment = _build(tmp_path)
    limitations = payment["data"]["known_limitations"]

    assert len(limitations) == 5
    assert "0.911000" in limitations[0]
    assert "0.994000" in limitations[0]
    assert "10 distinct payees" in limitations[0]
    assert "100 held payments" in limitations[0]
    assert "0.620000" in limitations[1]
    assert "0.070000" in limitations[1]
    assert "0.550000" in limitations[1]
    assert "0.930000" in limitations[1]
    assert "0.617000" in limitations[2]
    assert "1234" in limitations[2]
    assert "world/bank_connectors.py:344-369" in limitations[3]
    assert "evaluation/level17_scoring.py:527" in limitations[4]

    account_limitations = account["data"]["known_limitations"]
    assert len(account_limitations) == 3
    assert "check 6b" in account_limitations[2]


def test_markdown_shows_the_same_numbers(tmp_path):
    """The Markdown twin shows the same numbers, limitation text and manifest as the JSON card."""
    _tree(tmp_path)
    _build(tmp_path)
    account_md = (tmp_path / "out" / "mule_account_xgb.md").read_text(encoding="utf-8")
    payment_md = (tmp_path / "out" / "mule_payment_xgb.md").read_text(encoding="utf-8")

    assert "| recall at capacity | 0.640000 | 0.320000 | 0.640000 | 5 |" in account_md
    assert "| device_rotation | 0.700000 | 0.350000 | 0.700000 | 0.400000 | 0.200000 | 0.400000 | 7 | 3 | 2 |" in (
        account_md
    )
    assert "0.000001 (1.234000e-06) on validation" in account_md
    assert "| check_6b | fail |" in account_md
    assert "Trivial-world check: pass." in account_md
    assert "scripts/level18_model_cards.py" in account_md
    assert f"{CONSOLE_DIR}/calibration.json" in account_md

    assert "| truth AP | 0.620000 | 0.310000 | 0.620000 | 5 |" in payment_md
    assert "| payee_precision | 0.911000 |" in payment_md
    assert "Per-payee precision is 0.911000, not the reported per-world median 0.994000" in payment_md
    assert "Run A check 6b: status fail" in payment_md
    assert "| id_leak | fail |" in payment_md


def test_missing_input_raises(tmp_path):
    """A missing console file stops the run with a ValueError naming the file."""
    _tree(tmp_path)
    (tmp_path / CONSOLE_DIR / "calibration.json").unlink()

    with pytest.raises(ValueError, match="console input is missing: reports/level_18/console/calibration.json"):
        cards().build(root=tmp_path, output_dir=tmp_path / "out")


def test_malformed_input_raises(tmp_path):
    """A console file that is not valid JSON stops the run with a ValueError naming the file."""
    _tree(tmp_path)
    (tmp_path / CONSOLE_DIR / "leakage.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError, match="console input is not valid JSON: reports/level_18/console/leakage.json"):
        cards().build(root=tmp_path, output_dir=tmp_path / "out")


def test_key_short_input_raises(tmp_path):
    """A console document without its data block stops the run with a ValueError naming the key."""
    _tree(tmp_path)
    path = tmp_path / CONSOLE_DIR / "comparison.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    del document["data"]
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="missing key 'data' at reports/level_18/console/comparison.json"):
        cards().build(root=tmp_path, output_dir=tmp_path / "out")


def test_inputs_not_modified(tmp_path):
    """Every console file is byte-identical after the run."""
    _tree(tmp_path)
    before = {rel_path: hashlib.sha256((tmp_path / rel_path).read_bytes()).hexdigest() for rel_path in _documents()}

    _build(tmp_path)

    after = {rel_path: hashlib.sha256((tmp_path / rel_path).read_bytes()).hexdigest() for rel_path in _documents()}
    assert after == before
