"""
WHAT: Reads the Level 18 Run B validation tuning checkpoint and result file, then writes
the tuned baseline rule thresholds configuration into config/rules_tuned.yaml.

WHY: Serving rules require proven, frozen thresholds tuned on validation worlds rather
than arbitrary untuned starting defaults (closes register item 17-5).
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

BASELINE_THRESHOLD_KEYS: frozenset[str] = frozenset({
    "large_credit_amount_paise",
    "outflow_share",
    "time_window_seconds",
    "new_account_age_days",
    "payer_count",
    "near_zero_balance_paise",
})

DERIVED_FROM_STATEMENT: str = (
    "Level 18 Run B checkpoint, rules tuned on validation worlds only (closes 17-5)"
)


def compute_sha256(path: Path) -> str:
    """
    PURPOSE:
        Computes the SHA-256 hexadecimal digest of a file.

    INPUTS:
        path: Filesystem path to the file.

    RETURNS:
        str: 64-character hexadecimal SHA-256 digest.

    ERRORS:
        Raises FileNotFoundError if the file does not exist.
    """
    hasher = hashlib.sha256()
    hasher.update(path.read_bytes())
    return hasher.hexdigest()


def read_tuned_thresholds(thresholds_path: Path) -> dict[str, int | float]:
    """
    PURPOSE:
        Reads the tuned rules thresholds from the checkpoint thresholds.json file.

    INPUTS:
        thresholds_path: Path to checkpoint thresholds.json.

    RETURNS:
        dict[str, int | float]: Mapping of threshold names to tuned numeric values.

    ERRORS:
        Raises ValueError if rules_thresholds key is missing or threshold names do not
        match the 6 canonical baseline threshold keys.
    """
    data = json.loads(thresholds_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "rules_thresholds" not in data:
        raise ValueError(f"Missing 'rules_thresholds' key in {thresholds_path}")

    thresholds: dict[str, Any] = data["rules_thresholds"]
    if not isinstance(thresholds, dict):
        raise ValueError(f"'rules_thresholds' must be a dictionary in {thresholds_path}")

    actual_keys = set(thresholds.keys())
    if actual_keys != BASELINE_THRESHOLD_KEYS:
        diff_missing = sorted(BASELINE_THRESHOLD_KEYS - actual_keys)
        diff_extra = sorted(actual_keys - BASELINE_THRESHOLD_KEYS)
        raise ValueError(
            f"Threshold keys in {thresholds_path} do not match baseline keys. "
            f"Missing: {diff_missing}, Extra: {diff_extra}"
        )

    result: dict[str, int | float] = {}
    for key, value in thresholds.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"Threshold '{key}' has non-numeric value: {value}")
        result[key] = value

    return result


def read_validation_dataset_ids(result_path: Path) -> list[str]:
    """
    PURPOSE:
        Reads the validation dataset identifiers from the checkpoint result.json manifest.

    INPUTS:
        result_path: Path to result.json.

    RETURNS:
        list[str]: Sorted list of validation dataset ID strings.

    ERRORS:
        Raises ValueError if manifest or validation datasets are missing or empty.
    """
    data = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = data.get("manifest") if isinstance(data, dict) else None
    if not isinstance(manifest, dict):
        raise ValueError(f"Missing 'manifest' dictionary in {result_path}")

    # Support either direct validation_dataset_ids list or datasets array with split='validation'
    if "validation_dataset_ids" in manifest and isinstance(manifest["validation_dataset_ids"], list):
        dataset_ids = manifest["validation_dataset_ids"]
    elif "datasets" in manifest and isinstance(manifest["datasets"], list):
        dataset_ids = [
            item["dataset_id"]
            for item in manifest["datasets"]
            if isinstance(item, dict) and item.get("split") == "validation" and "dataset_id" in item
        ]
    else:
        raise ValueError(f"Could not find validation datasets in manifest of {result_path}")

    if not dataset_ids:
        raise ValueError(f"Missing or empty validation datasets in {result_path}")

    for item in dataset_ids:
        if not isinstance(item, str) or not item:
            raise ValueError(f"Invalid dataset ID in {result_path}: {item}")

    return sorted(dataset_ids)


def generate_serving_rules(
    repo_root: Path,
    thresholds_path: Path,
    result_path: Path,
    output_path: Path,
) -> Path:
    """
    PURPOSE:
        Generates config/rules_tuned.yaml from Run B checkpoint and result files.

    INPUTS:
        repo_root: Root directory of the repository for relative path computation.
        thresholds_path: Path to checkpoint thresholds.json.
        result_path: Path to checkpoint result.json.
        output_path: Output path for rules_tuned.yaml.

    RETURNS:
        Path: Resolved output file path.
    """
    thresholds = read_tuned_thresholds(thresholds_path)
    val_dataset_ids = read_validation_dataset_ids(result_path)

    thresholds_rel = thresholds_path.relative_to(repo_root).as_posix()
    result_rel = result_path.relative_to(repo_root).as_posix()

    thresholds_sha = compute_sha256(thresholds_path)
    result_sha = compute_sha256(result_path)

    payload: dict[str, Any] = {
        "derived_from": DERIVED_FROM_STATEMENT,
        "sources": {
            "thresholds": {
                "path": thresholds_rel,
                "sha256": thresholds_sha,
            },
            "result": {
                "path": result_rel,
                "sha256": result_sha,
            },
        },
        "validation_dataset_ids": val_dataset_ids,
        "thresholds": thresholds,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_text = yaml.safe_dump(payload, sort_keys=False)
    output_path.write_text(yaml_text, encoding="utf-8")
    return output_path


def main() -> None:
    """
    PURPOSE:
        Entry point to write config/rules_tuned.yaml using Run B checkpoint files.
    """
    repo_root = Path(__file__).resolve().parent.parent
    thresholds_path = repo_root / "runs/full/run_b320/final/45c496799170/checkpoint/thresholds.json"
    result_path = repo_root / "runs/full/run_b320/final/45c496799170/result.json"
    output_path = repo_root / "config/rules_tuned.yaml"

    generate_serving_rules(
        repo_root=repo_root,
        thresholds_path=thresholds_path,
        result_path=result_path,
        output_path=output_path,
    )
    print(f"Wrote tuned serving rules configuration to {output_path}")


if __name__ == "__main__":
    main()
