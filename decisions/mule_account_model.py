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
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import xgboost
import yaml

from core.event_clock import IST
from decisions.anomaly_config import load_anomaly_detector_config
from features.account_snapshots import account_feature_names

__all__ = [
    "ALLOWED_LABEL_CUTS",
    "MuleModelConfig",
    "account_model_feature_names",
    "active_mule_model_config_path",
    "build_training_rows",
    "load_model",
    "load_mule_model_config",
    "predict_account_days",
    "save_model",
    "set_mule_model_config_path",
    "train_account_model",
]

ALLOWED_LABEL_CUTS: frozenset[str] = frozenset({"arrival_time", "label_day"})
ALLOWED_PU_METHODS: frozenset[str] = frozenset({"none", "bagging", "elkan_noto"})
ALLOWED_MULE_MODEL_KEYS: frozenset[str] = frozenset(
    {
        "feature_fill_flag_share",
        "label_cut",
        "label_maturity_days",
        "use_label_network_features",
        "label_network_features",
        "feedback_network_features_not_used",
        "rules_plus_labels_and_registry",
        "pu_method",
        "pu_bagging_n_bags",
        "pu_unlabelled_per_positive_candidates",
        "pu_probability_clip",
    }
)
OPTIONAL_MULE_MODEL_KEYS: frozenset[str] = frozenset(
    {
        "use_label_network_features",
        "label_network_features",
        "feedback_network_features_not_used",
        "rules_plus_labels_and_registry",
        "pu_method",
        "pu_bagging_n_bags",
        "pu_unlabelled_per_positive_candidates",
        "pu_probability_clip",
    }
)


@dataclass(frozen=True)
class MuleModelConfig:
    """
    WHAT: Holds the validated policy switches that shape the account model's training rows.

    Attributes:
        label_cut: How a positive account's rows are cut at its first label; one of
            ALLOWED_LABEL_CUTS.
        label_maturity_days: Number of trailing IST days dropped at the end of every world, or None
            when the maturity window is off because the label delay has not been measured yet.
        feature_fill_flag_share: Share of rows at or above which a feature's most common value makes
            the feature unable to separate anything, so the sanity report flags it.
        use_label_network_features: Whether the account model may use the configured label-based
            network features, or None when the config file does not carry the switch.
        label_network_features: Bank-visible label-based network features the model may use.
        feedback_network_features_not_used: Label-based network features built from bank_feedback,
            which no built world writes; recorded so the omission is explicit.
        rules_plus_labels_and_registry: Whether the rules table also carries rule_6, the label-based
            OR rule, or None when the config file does not carry the switch.
        pu_method: One of ALLOWED_PU_METHODS, or None when the config file does not carry the switch.
        pu_bagging_n_bags: Number of bagged models the PU bagging method trains, or None when the
            config file does not carry the value.
        pu_unlabelled_per_positive_candidates: Unlabelled rows drawn per positive row for each
            bagging candidate, or None when the config file does not carry the value.
        pu_probability_clip: Cap below 1 on g(x) so the Elkan-Noto weight stays finite, or None when
            the config file does not carry the value.
    """

    label_cut: str
    label_maturity_days: int | None
    feature_fill_flag_share: float
    use_label_network_features: bool | None
    label_network_features: tuple[str, ...]
    feedback_network_features_not_used: tuple[str, ...]
    rules_plus_labels_and_registry: bool | None
    pu_method: str | None
    pu_bagging_n_bags: int | None
    pu_unlabelled_per_positive_candidates: tuple[int, ...] | None
    pu_probability_clip: float | None


def _parse_feature_name_list(block: dict[str, Any], key: str, path: Path) -> tuple[str, ...]:
    """
    WHAT: Returns a non-empty, duplicate-free list of feature names from a config block.

    Raises:
        ValueError: If the value is not a non-empty list of non-empty distinct strings.
    """
    value = block.get("value")
    if not isinstance(value, list) or not value:
        raise ValueError(f"Block '{key}' in {path} must hold a non-empty list of feature names")
    for name in value:
        if not isinstance(name, str) or not name:
            raise ValueError(f"Block '{key}' in {path} holds a feature name that is not a non-empty string")
    if len(set(value)) != len(value):
        raise ValueError(f"Block '{key}' in {path} holds duplicate feature names")
    return tuple(value)


def _validate_network_feature_names(
    label_network_features: tuple[str, ...],
    feedback_network_features_not_used: tuple[str, ...],
    path: Path,
) -> None:
    """
    WHAT: Requires every listed network feature to exist in the canonical catalogue and in the
    anomaly detector's exclusion list.

    WHY: The model may only add features the catalogue defines, and adding one the anomaly
    detector still uses would silently change that detector's no-labels contract.

    Raises:
        ValueError: Naming the first listed feature that fails either check.
    """
    canonical = set(account_feature_names())
    excluded = set(load_anomaly_detector_config().excluded_features)
    for name in (*label_network_features, *feedback_network_features_not_used):
        if name not in canonical:
            raise ValueError(f"Feature {name!r} in {path} is not in the canonical account feature catalogue")
        if name not in excluded:
            raise ValueError(f"Feature {name!r} in {path} is not in the anomaly exclusion list")


_COMMITTED_MULE_MODEL_CONFIG_PATH: Path = Path(__file__).resolve().parent.parent / "config" / "mule_model.yaml"
_MULE_MODEL_CONFIG_OVERRIDE: Path | None = None


def set_mule_model_config_path(path: Path | str | None) -> None:
    """
    WHAT: Points every loader in this process at one model config, or clears the override with None.

    WHY: One override point rather than a config parameter on each function: every caller of
    load_mule_model_config must see the same file during a run, and threading a parameter through
    the feature-name builder, the training-row builder and every evaluation caller risks one call
    site keeping the committed file and silently mixing two configs in one result.

    The override is process-wide state, set once at start-up and cleared when the run ends; it is
    not safe to change while a run is in progress.

    Args:
        path: Config file to use instead of the committed config; None restores the committed path.

    Raises:
        FileNotFoundError: If a given path does not exist, naming that path.
    """
    global _MULE_MODEL_CONFIG_OVERRIDE
    if path is None:
        _MULE_MODEL_CONFIG_OVERRIDE = None
        return
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Mule model configuration not found at: {resolved}")
    _MULE_MODEL_CONFIG_OVERRIDE = resolved


def active_mule_model_config_path() -> Path:
    """
    WHAT: Returns the config file load_mule_model_config reads when no explicit path is given.

    WHY: The run manifest records which config actually produced a result, so the overridden and
    committed cases must both be answerable from one place.
    """
    if _MULE_MODEL_CONFIG_OVERRIDE is not None:
        return _MULE_MODEL_CONFIG_OVERRIDE
    return _COMMITTED_MULE_MODEL_CONFIG_PATH


def load_mule_model_config(path: Path | str | None = None) -> MuleModelConfig:
    """
    WHAT: Loads and validates the account model's policy switches from the active config file.

    WHY: Whether the cut uses the label's exact arrival time is a modelling decision, so it is read
    from a tracked config file instead of being typed into the builder.

    Args:
        path: Optional config path; an explicit path wins over the process-wide override; when
            omitted the active path is config/mule_model.yaml unless set_mule_model_config_path
            pointed elsewhere.

    Returns:
        The validated switches.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the root, parameters, block metadata or a parameter value is missing or invalid.
    """
    if path is None:
        resolved_path = active_mule_model_config_path()
    else:
        resolved_path = Path(path)

    if not resolved_path.exists():
        raise FileNotFoundError(f"Mule model configuration not found at: {resolved_path}")

    with open(resolved_path, "r", encoding="utf-8") as f:
        raw: Any = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Invalid YAML content in {resolved_path}; expected mapping at root.")
    unknown_root = set(raw.keys()) - {"parameters"}
    if unknown_root:
        raise ValueError(f"Unknown keys in {resolved_path}: {sorted(unknown_root)}")

    parameters = raw.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError(f"Missing 'parameters' mapping in {resolved_path}")

    unknown_keys = set(parameters.keys()) - ALLOWED_MULE_MODEL_KEYS
    if unknown_keys:
        raise ValueError(f"Unknown keys in {resolved_path}: {sorted(unknown_keys)}")

    for key in ALLOWED_MULE_MODEL_KEYS:
        if key in OPTIONAL_MULE_MODEL_KEYS and key not in parameters:
            continue
        block = parameters.get(key)
        if not isinstance(block, dict):
            raise ValueError(f"Missing parameter block '{key}' in {resolved_path}")
        label = block.get("label")
        if not isinstance(label, str) or not label:
            raise ValueError(f"Block '{key}' in {resolved_path} is missing a non-empty 'label'")
        derived_from = block.get("derived_from")
        if not isinstance(derived_from, str) or not derived_from:
            raise ValueError(f"Block '{key}' in {resolved_path} is missing a non-empty 'derived_from'")

    label_cut = parameters["label_cut"].get("value")
    if not isinstance(label_cut, str) or label_cut not in ALLOWED_LABEL_CUTS:
        raise ValueError(f"label_cut in {resolved_path} must be one of {sorted(ALLOWED_LABEL_CUTS)}; got {label_cut!r}")

    maturity_block = parameters["label_maturity_days"]
    maturity_value = maturity_block.get("value")
    if maturity_value is None:
        if maturity_block.get("not_found") is not True:
            raise ValueError(f"label_maturity_days in {resolved_path} is null but not_found is not true")
        label_maturity_days: int | None = None
    elif isinstance(maturity_value, int) and not isinstance(maturity_value, bool) and maturity_value >= 0:
        label_maturity_days = maturity_value
    else:
        raise ValueError(
            f"label_maturity_days in {resolved_path} must be null or a non-negative integer; got {maturity_value!r}"
        )

    fill_share_block = parameters["feature_fill_flag_share"]
    fill_share_value = fill_share_block.get("value")
    if (
        not isinstance(fill_share_value, (int, float))
        or isinstance(fill_share_value, bool)
        or not 0.0 < fill_share_value <= 1.0
    ):
        raise ValueError(
            f"feature_fill_flag_share in {resolved_path} must be a number in (0.0, 1.0]; got {fill_share_value!r}"
        )

    use_label_network_features: bool | None = None
    if "use_label_network_features" in parameters:
        switch_value = parameters["use_label_network_features"].get("value")
        if not isinstance(switch_value, bool):
            raise ValueError(
                f"use_label_network_features in {resolved_path} must be a boolean; got {switch_value!r}"
            )
        use_label_network_features = switch_value

    label_network_features: tuple[str, ...] = ()
    if "label_network_features" in parameters:
        label_network_features = _parse_feature_name_list(
            parameters["label_network_features"], "label_network_features", resolved_path
        )
    elif use_label_network_features is True:
        raise ValueError(
            f"label_network_features in {resolved_path} is required when use_label_network_features is true"
        )

    feedback_network_features_not_used: tuple[str, ...] = ()
    if "feedback_network_features_not_used" in parameters:
        feedback_network_features_not_used = _parse_feature_name_list(
            parameters["feedback_network_features_not_used"], "feedback_network_features_not_used", resolved_path
        )

    _validate_network_feature_names(label_network_features, feedback_network_features_not_used, resolved_path)

    rules_plus_labels_and_registry: bool | None = None
    if "rules_plus_labels_and_registry" in parameters:
        rules_switch_value = parameters["rules_plus_labels_and_registry"].get("value")
        if not isinstance(rules_switch_value, bool):
            raise ValueError(
                f"rules_plus_labels_and_registry in {resolved_path} must be a boolean; got {rules_switch_value!r}"
            )
        rules_plus_labels_and_registry = rules_switch_value

    pu_method: str | None = None
    if "pu_method" in parameters:
        pu_method_value = parameters["pu_method"].get("value")
        if not isinstance(pu_method_value, str) or pu_method_value not in ALLOWED_PU_METHODS:
            raise ValueError(
                f"pu_method in {resolved_path} must be one of {sorted(ALLOWED_PU_METHODS)}; got {pu_method_value!r}"
            )
        pu_method = pu_method_value

    pu_bagging_n_bags: int | None = None
    if "pu_bagging_n_bags" in parameters:
        n_bags_value = parameters["pu_bagging_n_bags"].get("value")
        if not isinstance(n_bags_value, int) or isinstance(n_bags_value, bool) or n_bags_value <= 0:
            raise ValueError(f"pu_bagging_n_bags in {resolved_path} must be a positive integer; got {n_bags_value!r}")
        pu_bagging_n_bags = n_bags_value

    pu_unlabelled_per_positive_candidates: tuple[int, ...] | None = None
    if "pu_unlabelled_per_positive_candidates" in parameters:
        candidates_value = parameters["pu_unlabelled_per_positive_candidates"].get("value")
        if not isinstance(candidates_value, list) or not candidates_value:
            raise ValueError(
                f"pu_unlabelled_per_positive_candidates in {resolved_path} "
                "must be a non-empty list of positive integers"
            )
        for candidate in candidates_value:
            if not isinstance(candidate, int) or isinstance(candidate, bool) or candidate <= 0:
                raise ValueError(
                    f"pu_unlabelled_per_positive_candidates in {resolved_path} must hold positive integers only; "
                    f"got {candidate!r}"
                )
        pu_unlabelled_per_positive_candidates = tuple(candidates_value)

    pu_probability_clip: float | None = None
    if "pu_probability_clip" in parameters:
        clip_value = parameters["pu_probability_clip"].get("value")
        if (
            not isinstance(clip_value, (int, float))
            or isinstance(clip_value, bool)
            or not 0.0 < clip_value < 1.0
        ):
            raise ValueError(
                f"pu_probability_clip in {resolved_path} must be a number in (0.0, 1.0); got {clip_value!r}"
            )
        pu_probability_clip = float(clip_value)

    return MuleModelConfig(
        label_cut=label_cut,
        label_maturity_days=label_maturity_days,
        feature_fill_flag_share=float(fill_share_value),
        use_label_network_features=use_label_network_features,
        label_network_features=label_network_features,
        feedback_network_features_not_used=feedback_network_features_not_used,
        rules_plus_labels_and_registry=rules_plus_labels_and_registry,
        pu_method=pu_method,
        pu_bagging_n_bags=pu_bagging_n_bags,
        pu_unlabelled_per_positive_candidates=pu_unlabelled_per_positive_candidates,
        pu_probability_clip=pu_probability_clip,
    )


def account_model_feature_names(use_label_network_features: bool | None = None) -> tuple[str, ...]:
    """
    WHAT: Returns the canonical account features minus the anomaly config's excluded features, plus
    the configured label network features when that switch is on.

    WHY: The pilot must use the same bank-visible feature contract as the anomaly detector, and the
    exclusion list is policy read from config/anomaly_detector.yaml rather than copied here. The
    label network switch is policy too, so it is read from config/mule_model.yaml unless the caller
    states it explicitly.

    Args:
        use_label_network_features: Overrides the config switch when given; when None the committed
            config decides.

    Returns:
        Canonical feature names in catalog order, with every configured exclusion removed except
        the label network features the switch admits.

    Raises:
        ValueError: If the anomaly config is missing or malformed (raised by its loader), the switch
            is absent from the config and not given, or a configured network feature fails
            validation (raised by the config loader).
    """
    config = load_mule_model_config()
    if use_label_network_features is None:
        if config.use_label_network_features is None:
            raise ValueError("use_label_network_features is missing from the mule model configuration")
        use_label_network_features = config.use_label_network_features
    excluded = set(load_anomaly_detector_config().excluded_features)
    if not use_label_network_features:
        return tuple(name for name in account_feature_names() if name not in excluded)
    admitted = set(config.label_network_features)
    return tuple(name for name in account_feature_names() if name not in excluded or name in admitted)


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
    label_cut: str | None = None,
    label_maturity_days: int | None = None,
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
        label_cut: "arrival_time" keeps a positive account's snapshots only strictly before the
            exact moment of its first label; "label_day" is the legacy cut after the IST day of that
            label. Defaults to the value in config/mule_model.yaml.
        label_maturity_days: Drops each world's final N IST days, where labels have not had time to
            arrive and positives are undercounted. Defaults to the value in config/mule_model.yaml,
            where null keeps every day.

    Returns:
        (matrix, targets, row_keys) where matrix rows follow account_model_feature_names() order and
        contain floats with NaN for missing values, targets are 1 for positive accounts and 0 for
        unlabelled accounts, and row_keys are (world_id, bank_id, account_id, IST day) per kept row.

    Key logic:
        One row per account-day is selected inside each world, because account identifiers repeat
        across simulated worlds. A positive account keeps only snapshots up to its first label:
        under "arrival_time" the bank sees nothing at or after that instant, while "label_day"
        keeps the whole IST day and exists only to reproduce the first pilot dataset. The maturity window is
        measured per world against that world's own last snapshot day.

    Raises:
        ValueError: If a required field is missing or malformed, a switch is not allowed, or
            label_maturity_days is negative.
    """
    if label_cut is None or label_maturity_days is None:
        config = load_mule_model_config()
        if label_cut is None:
            label_cut = config.label_cut
        if label_maturity_days is None:
            label_maturity_days = config.label_maturity_days
    if label_cut not in ALLOWED_LABEL_CUTS:
        raise ValueError(f"label_cut must be one of {sorted(ALLOWED_LABEL_CUTS)}; got {label_cut!r}")
    if label_maturity_days is not None:
        if not isinstance(label_maturity_days, int) or isinstance(label_maturity_days, bool) or label_maturity_days < 0:
            raise ValueError(f"label_maturity_days must be a non-negative integer; got {label_maturity_days!r}")

    sources = sorted({_require_string(row, "source", "build_training_rows") for row in label_rows})
    verdicts = sorted({_require_string(row, "verdict", "build_training_rows") for row in label_rows})
    print(f"Label values: source={sources} verdict={verdicts}")

    first_label_day: dict[tuple[str, str], date] = {}
    first_label_arrival: dict[tuple[str, str], datetime] = {}
    for row in label_rows:
        world_id = _require_string(row, "world_id", "build_training_rows")
        account_id = _require_string(row, "account_id", "build_training_rows")
        arrival_time = _require_datetime(row, "arrival_time", "build_training_rows")
        day = _ist_day(arrival_time)
        key = (world_id, account_id)
        current = first_label_day.get(key)
        if current is None or day < current:
            first_label_day[key] = day
        current_arrival = first_label_arrival.get(key)
        if current_arrival is None or arrival_time < current_arrival:
            first_label_arrival[key] = arrival_time

    feature_names = account_model_feature_names()
    rows_by_world: dict[str, list[dict[str, Any]]] = {}
    for row in snapshot_rows:
        world_id = _require_string(row, "world_id", "build_training_rows")
        rows_by_world.setdefault(world_id, []).append(row)

    matrix: list[list[float]] = []
    targets: list[int] = []
    row_keys: list[tuple[str, str, str, date]] = []
    for world_id in sorted(rows_by_world):
        world_rows = rows_by_world[world_id]
        maturity_cutoff: date | None = None
        if label_maturity_days is not None:
            last_day = max(
                _ist_day(_require_datetime(row, "snapshot_time", "build_training_rows")) for row in world_rows
            )
            maturity_cutoff = last_day - timedelta(days=label_maturity_days)
        day_rows = _one_row_per_account_day(world_rows)
        for row in day_rows:
            bank_id = _require_string(row, "bank_id", "build_training_rows")
            account_id = _require_string(row, "account_id", "build_training_rows")
            moment = _require_datetime(row, "snapshot_time", "build_training_rows")
            day = _ist_day(moment)
            if maturity_cutoff is not None and day > maturity_cutoff:
                continue
            key = (world_id, account_id)
            is_positive = key in first_label_day
            if is_positive:
                if label_cut == "arrival_time":
                    if moment >= first_label_arrival[key]:
                        continue
                elif day > first_label_day[key]:
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
    *,
    sample_weight: list[float] | None = None,
    scale_pos_weight_override: float | None = None,
) -> tuple[xgboost.XGBClassifier, float]:
    """
    WHAT: Fits the pilot account model and returns it with its positive-class weight.

    WHY: Fraud accounts are rare, so scale_pos_weight balances the two classes. Every other
    hyperparameter stays at the XGBoost default; n_estimators is chosen on validation by the runner.
    The weighted Elkan-Noto path needs its own per-row weights and its own class weight, so both can
    be supplied without changing the default supervised call.

    Args:
        rows: Feature rows in account_model_feature_names() order.
        targets: Binary labels aligned with rows.
        seed: Random seed recorded in the model sidecar.
        n_estimators: Number of boosting rounds; chosen on validation, never on test.
        sample_weight: Optional non-negative weight per row; None fits without weights exactly as
            before.
        scale_pos_weight_override: Optional positive class weight replacing negatives / positives;
            None keeps negatives / positives.

    Returns:
        (model, scale_pos_weight) where scale_pos_weight is the override when given, otherwise
        negatives / positives.

    Raises:
        ValueError: If rows and targets differ in length, there are no positives or no negatives,
            the sample weights do not align with the rows or include a non-finite or negative
            value, or the override is not a positive finite number.
    """
    if len(rows) != len(targets):
        raise ValueError(f"rows length {len(rows)} does not match targets length {len(targets)}")
    if sample_weight is not None:
        if len(sample_weight) != len(rows):
            raise ValueError(
                f"sample_weight length {len(sample_weight)} does not match rows length {len(rows)}"
            )
        for weight in sample_weight:
            if not isinstance(weight, (int, float)) or isinstance(weight, bool):
                raise ValueError(f"sample_weight values must be numbers; got {weight!r}")
            if not math.isfinite(weight) or weight < 0.0:
                raise ValueError(f"sample_weight values must be non-negative and finite; got {weight!r}")
    if scale_pos_weight_override is not None:
        if (
            not isinstance(scale_pos_weight_override, (int, float))
            or isinstance(scale_pos_weight_override, bool)
            or not math.isfinite(scale_pos_weight_override)
            or scale_pos_weight_override <= 0.0
        ):
            raise ValueError(
                "scale_pos_weight_override must be a positive finite number; "
                f"got {scale_pos_weight_override!r}"
            )
    positives = sum(1 for target in targets if target == 1)
    negatives = sum(1 for target in targets if target == 0)
    if positives == 0:
        raise ValueError("train_account_model requires at least one positive account-day")
    if negatives == 0:
        raise ValueError("train_account_model requires at least one negative account-day")
    scale_pos_weight = (
        float(scale_pos_weight_override)
        if scale_pos_weight_override is not None
        else negatives / positives
    )

    model = xgboost.XGBClassifier(
        objective="binary:logistic",
        tree_method="hist",
        random_state=seed,
        scale_pos_weight=scale_pos_weight,
        n_estimators=n_estimators,
    )
    if sample_weight is None:
        model.fit(rows, targets)
    else:
        model.fit(rows, targets, sample_weight=sample_weight)
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
