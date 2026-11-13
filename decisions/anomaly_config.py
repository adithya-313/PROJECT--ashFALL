# File: decisions/anomaly_config.py
# Purpose: Configuration loading and schema validation for unsupervised anomaly detection and cold-start policy.
# Issue: Level 17b Round 1 anomaly detector and cold-start policy.
"""
WHAT:
    Defines immutable data containers and strict validation routines for the
    isolation forest anomaly detector and the bank cold-start routing policy.

WHY:
    Anomaly detection relies on early unlabelled traffic without labels or label-derived
    features. Strict validation of excluded features, training splits, and metadata
    guarantees that benchmark boundaries are preserved and training data leakage is prevented.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from features.account_snapshots import account_feature_names

__all__ = [
    "ALLOWED_ANOMALY_CONFIG_KEYS",
    "ALLOWED_COLD_START_CONFIG_KEYS",
    "AnomalyDetectorConfig",
    "ColdStartConfig",
    "input_feature_names",
    "load_anomaly_detector_config",
    "load_cold_start_config",
]

ALLOWED_ANOMALY_CONFIG_KEYS: frozenset[str] = frozenset(
    {
        "excluded_features",
        "early_window_days",
        "n_estimators",
        "max_samples",
        "contamination",
        "random_state",
        "allowed_training_splits",
    }
)

ALLOWED_COLD_START_CONFIG_KEYS: frozenset[str] = frozenset(
    {
        "min_positive_labels_per_bank",
        "counted_verdict",
    }
)


@dataclass(frozen=True)
class AnomalyDetectorConfig:
    """
    PURPOSE:
        Encapsulates validated hyperparameters and feature exclusion rules for the Isolation Forest detector.

    ATTRIBUTES:
        excluded_features: Ordered tuple of feature names excluded from the anomaly matrix.
        excluded_features_reasons: Mapping of excluded feature names to their exclusion justification.
        early_window_days: Integer duration in days defining the early unlabelled training window.
        n_estimators: Number of isolation trees in the ensemble.
        max_samples: Number of samples or string ('auto') drawn to train each base estimator.
        contamination: Proportion of outliers in the data or string ('auto').
        random_state: Integer random seed for deterministic tree generation.
        allowed_training_splits: Tuple of dataset splits allowed for fitting (must not contain test or stress).
    """

    excluded_features: tuple[str, ...]
    excluded_features_reasons: dict[str, str]
    early_window_days: int
    n_estimators: int
    max_samples: str | int | float
    contamination: str | float
    random_state: int
    allowed_training_splits: tuple[str, ...]


@dataclass(frozen=True)
class ColdStartConfig:
    """
    PURPOSE:
        Encapsulates policy threshold parameters for routing between heuristic rules and ML models.

    ATTRIBUTES:
        min_positive_labels_per_bank: Minimum verified fraud labels required before model activation.
        counted_verdict: Specific verdict string (e.g. 'fraud') counted toward the threshold.
    """

    min_positive_labels_per_bank: int
    counted_verdict: str


def input_feature_names(config: AnomalyDetectorConfig | None = None) -> tuple[str, ...]:
    """
    PURPOSE:
        Returns canonical account feature names excluding configured anomaly detector exclusions.

    INPUTS:
        config: Optional pre-loaded AnomalyDetectorConfig. If None, default config is loaded.

    KEY LOGIC:
        Filters account_feature_names() preserving catalog order, removing any feature
        declared in config.excluded_features.

    RETURNS:
        tuple[str, ...]: Canonical input feature names for the anomaly detector matrix.
    """
    if config is None:
        config = load_anomaly_detector_config()

    excluded = set(config.excluded_features)
    canonical = account_feature_names()
    return tuple(name for name in canonical if name not in excluded)


def _check_block_metadata(key: str, block: Any, file_label: str) -> None:
    """Validates that a configuration block contains non-empty 'label' and 'derived_from' fields."""
    if not isinstance(block, dict):
        raise ValueError(f"Block '{key}' in {file_label} must be a dictionary; got {type(block).__name__}")

    label = block.get("label")
    if not label or not isinstance(label, str) or not label.strip():
        raise ValueError(f"Block '{key}' in {file_label} missing or empty required field 'label'")

    derived_from = block.get("derived_from")
    if not derived_from or not isinstance(derived_from, str) or not derived_from.strip():
        raise ValueError(f"Block '{key}' in {file_label} missing or empty required field 'derived_from'")


def load_anomaly_detector_config(path: Path | str | None = None) -> AnomalyDetectorConfig:
    """
    PURPOSE:
        Loads and validates configuration for the Isolation Forest anomaly detector from YAML.

    INPUTS:
        path: Optional filesystem path. Defaults to config/anomaly_detector.yaml.

    KEY LOGIC:
        1. Validates root mapping and verifies no unknown keys are present.
        2. Validates that every parameter block contains required governance metadata (label, derived_from).
        3. Validates that all excluded features exist in canonical account_feature_names().
        4. Validates that early_window_days is strictly positive (> 0).
        5. Validates that allowed_training_splits contains neither 'test' nor 'stress'.
        6. Constructs immutable AnomalyDetectorConfig.

    SIDE EFFECTS / ERRORS:
        Raises FileNotFoundError: If path does not exist.
        Raises ValueError: On schema violations, unknown keys, missing metadata, or invalid parameters.
    """
    if path is None:
        root_dir = Path(__file__).resolve().parent.parent
        resolved_path = root_dir / "config" / "anomaly_detector.yaml"
    else:
        resolved_path = Path(path)

    if not resolved_path.exists():
        raise FileNotFoundError(f"Anomaly detector configuration not found at: {resolved_path}")

    with open(resolved_path, "r", encoding="utf-8") as f:
        raw: Any = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Invalid YAML content in {resolved_path}; expected mapping at root.")

    # 1. Check for unknown keys
    unknown_keys = set(raw.keys()) - ALLOWED_ANOMALY_CONFIG_KEYS
    if unknown_keys:
        raise ValueError(f"Unknown keys in {resolved_path}: {sorted(unknown_keys)}")

    # 2. Check metadata on all parameter blocks
    for key in ALLOWED_ANOMALY_CONFIG_KEYS:
        if key not in raw:
            raise ValueError(f"Missing required parameter '{key}' in {resolved_path}")
        _check_block_metadata(key, raw[key], str(resolved_path))

    # 3. Validate excluded_features against account_feature_names()
    excluded_block = raw["excluded_features"]
    raw_excluded_value = excluded_block.get("value")
    excluded_reasons: dict[str, str] = {}
    if isinstance(raw_excluded_value, dict):
        for feat_name, reason in raw_excluded_value.items():
            if isinstance(reason, dict):
                excluded_reasons[str(feat_name)] = str(reason.get("reason", ""))
            else:
                excluded_reasons[str(feat_name)] = str(reason)
    elif isinstance(raw_excluded_value, list):
        for item in raw_excluded_value:
            if isinstance(item, dict) and "name" in item:
                excluded_reasons[str(item["name"])] = str(item.get("reason", ""))
            else:
                excluded_reasons[str(item)] = ""
    else:
        raise ValueError(f"Invalid excluded_features value in {resolved_path}; expected mapping or list.")

    valid_account_features = set(account_feature_names())
    for feat in excluded_reasons:
        if feat not in valid_account_features:
            raise ValueError(
                f"Excluded feature '{feat}' in {resolved_path} is not in canonical account_feature_names()"
            )

    # 4. Validate early_window_days > 0
    early_days_block = raw["early_window_days"]
    early_days_val = early_days_block.get("value")
    if not isinstance(early_days_val, int) or isinstance(early_days_val, bool):
        raise ValueError(f"early_window_days value must be integer in {resolved_path}; got {type(early_days_val)}")
    if early_days_val <= 0:
        raise ValueError(f"early_window_days must be > 0 in {resolved_path}; got {early_days_val}")

    # 5. Validate hyperparameters
    n_est = raw["n_estimators"].get("value")
    if not isinstance(n_est, int) or isinstance(n_est, bool) or n_est <= 0:
        raise ValueError(f"n_estimators must be a positive integer in {resolved_path}; got {n_est}")

    max_samples = raw["max_samples"].get("value")
    contamination = raw["contamination"].get("value")
    random_state = raw["random_state"].get("value")
    if not isinstance(random_state, int) or isinstance(random_state, bool):
        raise ValueError(f"random_state must be an integer in {resolved_path}; got {random_state}")

    # 6. Validate allowed_training_splits does not contain test or stress
    splits_val = raw["allowed_training_splits"].get("value")
    if not isinstance(splits_val, (list, tuple)):
        raise ValueError(f"allowed_training_splits must be a sequence in {resolved_path}; got {type(splits_val)}")

    parsed_splits: tuple[str, ...] = tuple(str(s) for s in splits_val)
    for forbidden in ("test", "stress"):
        if forbidden in parsed_splits:
            raise ValueError(
                f"allowed_training_splits in {resolved_path} cannot contain '{forbidden}'; got {parsed_splits}"
            )

    return AnomalyDetectorConfig(
        excluded_features=tuple(excluded_reasons.keys()),
        excluded_features_reasons=excluded_reasons,
        early_window_days=early_days_val,
        n_estimators=n_est,
        max_samples=max_samples,
        contamination=contamination,
        random_state=random_state,
        allowed_training_splits=parsed_splits,
    )


def load_cold_start_config(path: Path | str | None = None) -> ColdStartConfig:
    """
    PURPOSE:
        Loads and validates configuration for the cold-start review and model routing policy.

    INPUTS:
        path: Optional filesystem path. Defaults to config/cold_start.yaml.

    KEY LOGIC:
        1. Validates root mapping and checks for unknown keys.
        2. Validates metadata (label, derived_from) on all parameter blocks.
        3. Parses min_positive_labels_per_bank (must be positive integer).
        4. Parses counted_verdict (must be non-empty string).
        5. Constructs immutable ColdStartConfig.

    SIDE EFFECTS / ERRORS:
        Raises FileNotFoundError: If path does not exist.
        Raises ValueError: On schema violations, unknown keys, missing metadata, or invalid values.
    """
    if path is None:
        root_dir = Path(__file__).resolve().parent.parent
        resolved_path = root_dir / "config" / "cold_start.yaml"
    else:
        resolved_path = Path(path)

    if not resolved_path.exists():
        raise FileNotFoundError(f"Cold-start configuration not found at: {resolved_path}")

    with open(resolved_path, "r", encoding="utf-8") as f:
        raw: Any = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Invalid YAML content in {resolved_path}; expected mapping at root.")

    # 1. Check for unknown keys
    unknown_keys = set(raw.keys()) - ALLOWED_COLD_START_CONFIG_KEYS
    if unknown_keys:
        raise ValueError(f"Unknown keys in {resolved_path}: {sorted(unknown_keys)}")

    # 2. Check metadata
    for key in ALLOWED_COLD_START_CONFIG_KEYS:
        if key not in raw:
            raise ValueError(f"Missing required parameter '{key}' in {resolved_path}")
        _check_block_metadata(key, raw[key], str(resolved_path))

    # 3. Parse and validate values
    min_labels = raw["min_positive_labels_per_bank"].get("value")
    if not isinstance(min_labels, int) or isinstance(min_labels, bool) or min_labels <= 0:
        raise ValueError(
            f"min_positive_labels_per_bank must be positive integer in {resolved_path}; got {min_labels}"
        )

    counted_verdict = raw["counted_verdict"].get("value")
    if not counted_verdict or not isinstance(counted_verdict, str) or not counted_verdict.strip():
        raise ValueError(f"counted_verdict must be non-empty string in {resolved_path}; got {counted_verdict}")

    return ColdStartConfig(
        min_positive_labels_per_bank=min_labels,
        counted_verdict=counted_verdict.strip(),
    )

