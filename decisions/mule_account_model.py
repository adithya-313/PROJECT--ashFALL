"""
WHAT: Trains and applies the Level 18 pilot XGBoost account-day model from bank-visible labels.

WHY: The pilot compares a supervised account model with the rules and the anomaly detector on the
same account-days. The model may only learn from labels a bank could actually hold (complaints,
police bulk reports and registry reports), never from hidden simulation truth, so this module
lives in decisions/ and must not import evaluation.truth.

TERM: account-day: one account's latest snapshot on one IST calendar day.
TERM: positive account: an account with at least one bank-visible label in its world.
TERM: label day: the IST calendar date of a label's arrival_time.

Inputs are plain snapshot and label rows. Missing or malformed required fields raise ValueError;
there is no silent fallback.
"""

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

import xgboost

from core.event_clock import IST
from decisions.anomaly_config import load_anomaly_detector_config
from features.account_snapshots import account_feature_names

__all__ = [
    "account_model_feature_names",
    "build_training_rows",
    "load_model",
    "predict_account_days",
    "save_model",
    "train_account_model",
]


def account_model_feature_names() -> tuple[str, ...]:
    """
    WHAT: Returns the canonical account features minus the anomaly config's excluded features.

    WHY: The pilot must use the same bank-visible feature contract as the anomaly detector, and the
    exclusion list is policy read from config/anomaly_detector.yaml rather than copied here.

    Returns:
        Canonical feature names in catalog order, with every configured exclusion removed.

    Raises:
        ValueError: If the anomaly config is missing or malformed (raised by its loader).
    """
    excluded = set(load_anomaly_detector_config().excluded_features)
    return tuple(name for name in account_feature_names() if name not in excluded)


def _require_string(row: dict[str, Any], key: str, caller: str) -> str:
    """
    WHAT: Returns a required non-empty string field from a row.

    Raises:
        ValueError: If the field is missing, empty, or not a string.
    """
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{caller} requires a non-empty string '{key}' on every row")
    return value


def _require_datetime(row: dict[str, Any], key: str, caller: str) -> datetime:
    """
    WHAT: Returns a required timezone-aware datetime field from a row.

    Raises:
        ValueError: If the field is missing, not a datetime, or timezone-naive.
    """
    value = row.get(key)
    if not isinstance(value, datetime):
        raise ValueError(f"{caller} requires a datetime '{key}' on every row")
    if value.tzinfo is None:
        raise ValueError(f"{caller} requires a timezone-aware '{key}'")
    return value


def _ist_day(moment: datetime) -> date:
    """
    WHAT: Returns the Asia/Kolkata calendar date of a timezone-aware moment.

    Raises:
        ValueError: If the moment is not timezone-aware.
    """
    if moment.tzinfo is None:
        raise ValueError("IST day requires a timezone-aware datetime")
    return moment.astimezone(IST).date()


def _one_row_per_account_day(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    WHAT: Keeps the latest snapshot per (world, bank, account, IST day), preferring 'daily' on a tie.

    WHY: The evaluation package owns the canonical one_row_per_account_day, but decisions/ is an
    operational package and the truth-isolation guard forbids it from importing evaluation at all.
    This local copy keeps the same selection rule without weakening the boundary, including the
    world in the key so account identifiers repeated across simulated worlds stay apart.

    Raises:
        ValueError: If a row lacks a required string field, a timezone-aware snapshot_time, or a
            valid snapshot_kind.
    """
    selected: dict[tuple[object, str, str, date], dict[str, Any]] = {}
    for row in rows:
        world_id = row.get("world_id")
        bank_id = _require_string(row, "bank_id", "one_row_per_account_day")
        account_id = _require_string(row, "account_id", "one_row_per_account_day")
        day = _ist_day(_require_datetime(row, "snapshot_time", "one_row_per_account_day"))
        key = (world_id, bank_id, account_id, day)
        current = selected.get(key)
        if current is None:
            selected[key] = row
            continue
        moment = _require_datetime(row, "snapshot_time", "one_row_per_account_day")
        current_moment = _require_datetime(current, "snapshot_time", "one_row_per_account_day")
        if moment > current_moment:
            selected[key] = row
        elif moment == current_moment:
            kind = row.get("snapshot_kind")
            current_kind = current.get("snapshot_kind")
            if kind not in {"daily", "full"}:
                raise ValueError(f"snapshot_kind must be 'daily' or 'full'; got {kind!r}")
            if current_kind not in {"daily", "full"}:
                raise ValueError(f"snapshot_kind must be 'daily' or 'full'; got {current_kind!r}")
            if kind == "daily" and current_kind != "daily":
                selected[key] = row
    return list(selected.values())


def _feature_value(value: Any, name: str) -> float:
    """
    WHAT: Converts one feature value to float, turning None into NaN.

    WHY: XGBoost handles missing values natively, so an absent feature must stay missing rather
    than be guessed as zero.

    Raises:
        ValueError: If the value is not numeric (a text feature would silently break the matrix).
    """
    if value is None:
        return float("nan")
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    raise ValueError(f"Feature {name!r} must be numeric or None; got {value!r}")


def build_training_rows(
    snapshot_rows: list[dict[str, Any]],
    label_rows: list[dict[str, Any]],
) -> tuple[list[list[float]], list[int], list[tuple[str, str, str, date]]]:
    """
    WHAT: Builds the pilot model's training matrix and targets from snapshots and bank-visible labels.

    WHY: Only account-days a bank could have seen before the first label are usable; after a label
    the bank already knows the account is suspicious, so those days would leak the answer. Unlabelled
    accounts keep every day as a negative because the bank saw no complaint against them.

    Args:
        snapshot_rows: Raw snapshot rows carrying world_id, bank_id, account_id, snapshot_time and
            snapshot_kind, plus the account features.
        label_rows: Bank-visible label rows carrying world_id, account_id and arrival_time.

    Returns:
        (matrix, targets, row_keys) where matrix rows follow account_model_feature_names() order and
        contain floats with NaN for missing values, targets are 1 for positive accounts and 0 for
        unlabelled accounts, and row_keys are (world_id, bank_id, account_id, IST day) per kept row.

    Key logic:
        One row per account-day is selected inside each world, because account identifiers repeat
        across simulated worlds. A positive account keeps only the days on or before the IST day of
        its first label.

    Raises:
        ValueError: If a required field is missing or malformed.
    """
    sources = sorted({_require_string(row, "source", "build_training_rows") for row in label_rows})
    verdicts = sorted({_require_string(row, "verdict", "build_training_rows") for row in label_rows})
    print(f"Label values: source={sources} verdict={verdicts}")

    first_label_day: dict[tuple[str, str], date] = {}
    for row in label_rows:
        world_id = _require_string(row, "world_id", "build_training_rows")
        account_id = _require_string(row, "account_id", "build_training_rows")
        arrival_time = _require_datetime(row, "arrival_time", "build_training_rows")
        day = _ist_day(arrival_time)
        key = (world_id, account_id)
        current = first_label_day.get(key)
        if current is None or day < current:
            first_label_day[key] = day

    feature_names = account_model_feature_names()
    rows_by_world: dict[str, list[dict[str, Any]]] = {}
    for row in snapshot_rows:
        world_id = _require_string(row, "world_id", "build_training_rows")
        rows_by_world.setdefault(world_id, []).append(row)

    matrix: list[list[float]] = []
    targets: list[int] = []
    row_keys: list[tuple[str, str, str, date]] = []
    for world_id in sorted(rows_by_world):
        day_rows = _one_row_per_account_day(rows_by_world[world_id])
        for row in day_rows:
            bank_id = _require_string(row, "bank_id", "build_training_rows")
            account_id = _require_string(row, "account_id", "build_training_rows")
            moment = _require_datetime(row, "snapshot_time", "build_training_rows")
            day = _ist_day(moment)
            key = (world_id, account_id)
            is_positive = key in first_label_day
            if is_positive and day > first_label_day[key]:
                continue
            matrix.append([_feature_value(row.get(name), name) for name in feature_names])
            targets.append(1 if is_positive else 0)
            row_keys.append((world_id, bank_id, account_id, day))
    return matrix, targets, row_keys


def train_account_model(
    rows: list[list[float]],
    targets: list[int],
    seed: int,
    n_estimators: int = 100,
) -> tuple[xgboost.XGBClassifier, float]:
    """
    WHAT: Fits the pilot account model and returns it with its positive-class weight.

    WHY: Fraud accounts are rare, so scale_pos_weight balances the two classes. Every other
    hyperparameter stays at the XGBoost default; n_estimators is chosen on validation by the runner.

    Args:
        rows: Feature rows in account_model_feature_names() order.
        targets: Binary labels aligned with rows.
        seed: Random seed recorded in the model sidecar.
        n_estimators: Number of boosting rounds; chosen on validation, never on test.

    Returns:
        (model, scale_pos_weight) where scale_pos_weight is negatives / positives.

    Raises:
        ValueError: If rows and targets differ in length, or there are no positives or no negatives.
    """
    if len(rows) != len(targets):
        raise ValueError(f"rows length {len(rows)} does not match targets length {len(targets)}")
    positives = sum(1 for target in targets if target == 1)
    negatives = sum(1 for target in targets if target == 0)
    if positives == 0:
        raise ValueError("train_account_model requires at least one positive account-day")
    if negatives == 0:
        raise ValueError("train_account_model requires at least one negative account-day")
    scale_pos_weight = negatives / positives

    model = xgboost.XGBClassifier(
        objective="binary:logistic",
        tree_method="hist",
        random_state=seed,
        scale_pos_weight=scale_pos_weight,
        n_estimators=n_estimators,
    )
    model.fit(rows, targets)
    return model, scale_pos_weight


def predict_account_days(model: xgboost.XGBClassifier, day_rows: list[dict[str, Any]]) -> list[float]:
    """
    WHAT: Returns the model's fraud probability for every supplied account-day row, in row order.

    WHY: Evaluation keeps the same row order as one_row_per_account_day, so callers can attach a
    probability to each ScoredDay without a separate join key.

    Args:
        model: Fitted XGBoost account model.
        day_rows: Account-day rows carrying the canonical features.

    Returns:
        One probability per input row, in the same order.

    Raises:
        ValueError: If a feature value is not numeric, or the rows cannot be scored.
    """
    feature_names = account_model_feature_names()
    matrix = [[_feature_value(row.get(name), name) for name in feature_names] for row in day_rows]
    if not matrix:
        raise ValueError("predict_account_days requires at least one account-day row")
    probabilities = model.predict_proba(matrix)[:, 1]
    return [float(probability) for probability in probabilities]


def _sidecar_path(path: Path) -> Path:
    """
    WHAT: Returns the JSON sidecar path for a native XGBoost model file.
    """
    return Path(str(path) + ".json")


def save_model(model: xgboost.XGBClassifier, path: Path, metadata: dict[str, Any]) -> None:
    """
    WHAT: Saves the native XGBoost model and a JSON sidecar describing the matrix it expects.

    WHY: A model file alone does not say which feature order produced it, so loading must compare
    the sidecar against the current canonical feature order.

    Args:
        model: Fitted XGBoost account model.
        path: Destination for the native .ubj model file.
        metadata: At least seed, scale_pos_weight and training_dataset_ids.

    Raises:
        ValueError: If a required metadata key is missing.
    """
    required = {"seed", "scale_pos_weight", "training_dataset_ids"}
    missing = sorted(required - set(metadata))
    if missing:
        raise ValueError(f"save_model metadata is missing required keys: {missing}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))
    sidecar = dict(metadata)
    sidecar["feature_names"] = list(account_model_feature_names())
    sidecar["xgboost_version"] = xgboost.__version__
    _sidecar_path(path).write_text(
        json.dumps(sidecar, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_model(path: Path) -> xgboost.XGBClassifier:
    """
    WHAT: Loads a native XGBoost model after proving its sidecar feature order still matches.

    WHY: A feature-order mismatch would silently score the wrong columns, so loading refuses it.

    Args:
        path: Native .ubj model file written by save_model.

    Returns:
        The loaded XGBoost classifier.

    Raises:
        ValueError: If the sidecar is missing, malformed, or disagrees with the canonical order.
    """
    path = Path(path)
    sidecar_path = _sidecar_path(path)
    if not sidecar_path.is_file():
        raise ValueError(f"Model sidecar is missing: {sidecar_path}")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if not isinstance(sidecar, dict):
        raise ValueError(f"Model sidecar {sidecar_path} must hold a mapping")
    recorded = sidecar.get("feature_names")
    expected = list(account_model_feature_names())
    if recorded != expected:
        raise ValueError(
            f"Model sidecar {sidecar_path} feature order does not match account_model_feature_names()"
        )

    model = xgboost.XGBClassifier()
    model.load_model(str(path))
    return model
