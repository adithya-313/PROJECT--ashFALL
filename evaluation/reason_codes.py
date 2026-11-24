# File: evaluation/reason_codes.py
# Purpose: Build the Level 18 reason-code map and explain a fixed validation sample with TreeSHAP
# for the frozen account and payment models.
"""
WHAT:
    Generates the one-code-per-feature-group map from the M1 schema and the committed payment
    feature order, then computes TreeSHAP values for the frozen account and payment models on a
    fixed 20,000-row validation sample (seed 1717), keeps each row's top three positive
    contributions, maps them to reason codes and writes both samples with their frequency tables.

WHY:
    A-059 fixes the SHAP space, the ranking rule, the mapping source and the sampling rule before
    any value is computed, so the reported reasons describe the frozen models instead of a choice
    made after seeing them. No model is fitted, tuned or changed, and no test or stress row is read.

TERM: reason feature: a feature whose SHAP contribution at a row is positive and among the three
      largest positive contributions.
TERM: reason code: the plain code of the feature's schema group, one code per group.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import shap
import xgboost
import yaml

from decisions.mule_account_model import _feature_value
from decisions.pu_learning import PUBaggedAccountModel
from evaluation.level17_scoring import one_row_per_account_day
from evaluation.payment_model import build_matrix, load_feature_order, load_split_rows
from evaluation.payment_oof_risk import (
    _CommitSampler,
    _git_head,
    _read_world_snapshots,
    load_frozen_recipe,
    validate_dataset_split,
)
from evaluation.payment_test_score import verify_booster_rounds, verify_frozen_artifacts

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEMA_PATH = REPO_ROOT / "data" / "schemas" / "feature_set_m1.yaml"
DEFAULT_PAYMENT_FEATURE_ORDER = REPO_ROOT / "config" / "mule_payment_feature_order.yaml"
DEFAULT_CODES_PATH = REPO_ROOT / "config" / "feature_reason_codes.yaml"
DEFAULT_MODEL_DIR = REPO_ROOT / "runs" / "full" / "payment_model_v1"
DEFAULT_OOF_ROOT = REPO_ROOT / "runs" / "full" / "payment_oof"
DEFAULT_OUT_ROOT = REPO_ROOT / "runs" / "full" / "level18b" / "reason_codes"
DEFAULT_EXTRA_REPORTING = REPO_ROOT / "runs" / "full" / "run_a320_stage5c" / "extra_reporting.json"

#: A-059 fixes the sample at 20,000 validation rows drawn with seed 1717.
SEED = 1717
SAMPLE_ROWS = 20000
TOP_REASONS = 3
ACCOUNT_DAY_KEY_FIELDS = ("dataset", "world_id", "bank_id", "account_id", "snapshot_time")
PAYMENT_KEY_FIELDS = ("dataset", "world_id", "payment_id")
_SAMPLING_RULE = (
    "the 20,000 rows with the smallest sha256('1717|' + '|'.join(key fields)) digest over the "
    "validation rows; output ordered by the key fields"
)


@dataclass(frozen=True)
class ReasonCode:
    """One reason code, its schema group and the group's plain name."""

    group: str
    code: str
    plain_reason: str


@dataclass(frozen=True)
class ReasonCodeMap:
    """The generated feature-to-code mapping and the sources it was built from."""

    schema_path: str
    schema_sha256: str
    payment_feature_order_path: str
    payment_feature_order_sha256: str
    codes: tuple[ReasonCode, ...]
    features: dict[str, str]

    def payload(self) -> dict[str, Any]:
        """Returns the committed YAML payload for this map."""
        return {
            "version": 1,
            "schema": self.schema_path,
            "schema_sha256": self.schema_sha256,
            "payment_feature_order": self.payment_feature_order_path,
            "payment_feature_order_sha256": self.payment_feature_order_sha256,
            "reason_codes": [
                {"group": entry.group, "code": entry.code, "plain_reason": entry.plain_reason} for entry in self.codes
            ],
            "features": dict(self.features),
        }


@dataclass(frozen=True)
class ReasonFeature:
    """One ranked reason feature and its SHAP contribution."""

    feature: str
    shap: float


@dataclass(frozen=True)
class RowReason:
    """One ranked reason feature with its reason code."""

    feature: str
    shap: float
    code: str


def _sha256_file(path: Path) -> str:
    """Returns the sha256 hex digest of a file's bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _display_path(path: Path) -> str:
    """Returns the repository-relative POSIX path when possible, so the committed YAML is stable."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _schema_feature_groups(schema_path: Path) -> list[tuple[str, str]]:
    """
    WHAT: Returns (feature name, group) pairs from the M1 schema in schema order.

    Raises:
        ValueError: If the schema holds no feature list, a feature lacks a non-empty name or group,
            or two features share a name.
    """
    document = yaml.safe_load(Path(schema_path).read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not isinstance(document.get("features"), list):
        raise ValueError(f"schema {schema_path} holds no 'features' list")
    pairs: list[tuple[str, str]] = []
    for record in document["features"]:
        if not isinstance(record, dict):
            raise ValueError(f"schema {schema_path} holds a feature that is not a mapping")
        name = record.get("name")
        group = record.get("group")
        if not isinstance(name, str) or not name:
            raise ValueError(f"schema {schema_path} holds a feature without a name")
        if not isinstance(group, str) or not group:
            raise ValueError(f"schema {schema_path} feature {name!r} has no group")
        pairs.append((name, group))
    names = [name for name, _ in pairs]
    if len(names) != len(set(names)):
        raise ValueError(f"schema {schema_path} repeats a feature name")
    return pairs


def _payment_feature_order(order_path: Path) -> tuple[list[str], dict[str, list[str]]]:
    """
    WHAT: Returns the committed payment feature order and its group memberships.

    Raises:
        ValueError: If the file is not a mapping with a unique feature list and string group lists.
    """
    payload = yaml.safe_load(Path(order_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{order_path} is not a mapping")
    features = payload.get("features")
    if not isinstance(features, list) or not features or not all(isinstance(name, str) and name for name in features):
        raise ValueError(f"{order_path} holds no feature list")
    if len(features) != len(set(features)):
        raise ValueError(f"{order_path} repeats a feature name")
    groups = payload.get("groups")
    if not isinstance(groups, dict):
        raise ValueError(f"{order_path} holds no groups mapping")
    typed: dict[str, list[str]] = {}
    for group, members in groups.items():
        if not isinstance(group, str) or not group:
            raise ValueError(f"{order_path} has a group without a name")
        if not isinstance(members, list) or not all(isinstance(name, str) and name for name in members):
            raise ValueError(f"{order_path} group {group!r} holds no feature list")
        typed[group] = list(members)
    return list(features), typed


def build_reason_code_map(
    schema_path: Path = DEFAULT_SCHEMA_PATH,
    payment_feature_order_path: Path = DEFAULT_PAYMENT_FEATURE_ORDER,
) -> ReasonCodeMap:
    """
    WHAT: Builds the one-code-per-group map from the schema and the committed payment feature order.

    WHY: The mapping is generated rather than typed, so a new schema feature or payment feature
    lands in its group automatically or stops the build instead of being silently unmapped.

    Raises:
        ValueError: If a feature has no group, a non-schema payment feature shares its group with
            another feature, or two groups would share a code.
    """
    schema_pairs = _schema_feature_groups(schema_path)
    payment_features, payment_groups = _payment_feature_order(payment_feature_order_path)
    schema_groups = {name: group for name, group in schema_pairs}

    extra_groups: dict[str, str] = {}
    for group, members in payment_groups.items():
        for member in members:
            if member in schema_groups:
                continue
            if len(members) != 1:
                raise ValueError(
                    f"payment feature {member!r} has no schema group and shares feature-order group {group!r}"
                )
            extra_groups[member] = group
    for member in extra_groups:
        if member not in payment_features:
            raise ValueError(f"payment feature-order group names {member!r}, which is not in the feature list")

    group_order: list[str] = []
    for _, group in schema_pairs:
        if group not in group_order:
            group_order.append(group)
    for group in extra_groups.values():
        if group not in group_order:
            group_order.append(group)

    codes: list[ReasonCode] = []
    seen_codes: set[str] = set()
    for group in group_order:
        code = group.upper()
        if code in seen_codes:
            raise ValueError(f"two feature groups share the reason code {code!r}")
        seen_codes.add(code)
        codes.append(ReasonCode(group=group, code=code, plain_reason=group.replace("_", " ")))

    code_by_group = {entry.group: entry.code for entry in codes}
    features: dict[str, str] = {name: code_by_group[group] for name, group in schema_pairs}
    for member, group in extra_groups.items():
        features[member] = code_by_group[group]

    missing = [name for name in payment_features if name not in features]
    if missing:
        raise ValueError(f"payment features have no reason code: {missing}")

    return ReasonCodeMap(
        schema_path=_display_path(schema_path),
        schema_sha256=_sha256_file(schema_path),
        payment_feature_order_path=_display_path(payment_feature_order_path),
        payment_feature_order_sha256=_sha256_file(payment_feature_order_path),
        codes=tuple(codes),
        features=features,
    )


_MAP_HEADER = (
    "# Generated by `py -3.12 -m evaluation.reason_codes build-map` from\n"
    "# data/schemas/feature_set_m1.yaml and config/mule_payment_feature_order.yaml. Do not edit by hand.\n"
)


def write_reason_code_map(
    path: Path = DEFAULT_CODES_PATH,
    schema_path: Path = DEFAULT_SCHEMA_PATH,
    payment_feature_order_path: Path = DEFAULT_PAYMENT_FEATURE_ORDER,
) -> ReasonCodeMap:
    """
    WHAT: Writes the generated reason-code map to `path` as YAML.

    WHY: The committed file is generated, so a stale or hand-edited copy cannot silently disagree
    with the schema and the payment feature order.
    """
    reason_map = build_reason_code_map(schema_path, payment_feature_order_path)
    text = _MAP_HEADER + yaml.safe_dump(reason_map.payload(), sort_keys=False, allow_unicode=True)
    Path(path).write_text(text, encoding="utf-8", newline="\n")
    return reason_map


def load_reason_code_map(
    path: Path = DEFAULT_CODES_PATH,
    schema_path: Path = DEFAULT_SCHEMA_PATH,
    payment_feature_order_path: Path = DEFAULT_PAYMENT_FEATURE_ORDER,
) -> ReasonCodeMap:
    """
    WHAT: Loads the committed map after proving it equals a fresh generation.

    WHY: The run must explain rows with the same mapping a reviewer can regenerate from the two
    committed sources; a stale copy is a stop, not a warning.

    Raises:
        ValueError: If the file is missing, holds no YAML mapping, or disagrees with its sources.
    """
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"reason-code map is missing: {path}")
    committed = yaml.safe_load(path.read_text(encoding="utf-8"))
    fresh = build_reason_code_map(schema_path, payment_feature_order_path)
    if committed != fresh.payload():
        raise ValueError(
            f"{path} is not the generated map for its sources; rerun `py -3.12 -m evaluation.reason_codes build-map`"
        )
    return fresh


def _key_text(row: Mapping[str, Any], field: str) -> str:
    """
    WHAT: Returns one sample key field as a stable string.

    WHY: The sample is ranked by a sha256 of the key fields, so a datetime must render the same way
    on every run instead of formatting differently.

    Raises:
        ValueError: If the field is missing or is not a non-empty string, integer or datetime.
    """
    if field not in row:
        raise ValueError(f"sample row lacks key field {field!r}")
    value = row[field]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, str) and value:
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    raise ValueError(f"sample key field {field!r} is not a non-empty string, integer or datetime: {value!r}")


def _sample_key(row: Mapping[str, Any], key_fields: Sequence[str]) -> tuple[str, ...]:
    """Returns the row's sample key as a tuple of stable strings."""
    return tuple(_key_text(row, field) for field in key_fields)


def _row_digest(row: Mapping[str, Any], key_fields: Sequence[str], seed: int) -> bytes:
    """Returns the sha256 digest that ranks a row in the sample."""
    text = "|".join([str(seed), *_sample_key(row, key_fields)])
    return hashlib.sha256(text.encode("utf-8")).digest()


def select_sample(
    rows: Iterator[Mapping[str, Any]] | Sequence[Mapping[str, Any]],
    key_fields: Sequence[str],
    seed: int = SEED,
    sample_rows: int = SAMPLE_ROWS,
) -> list[dict[str, Any]]:
    """
    WHAT: Returns the `sample_rows` rows with the smallest sha256(seed and key fields) digest.

    WHY: A-059 fixes this rule so the same validation sample is reproducible without depending on
    a random-number library's implementation; the digest also ranks any two rows without a tie.

    Raises:
        ValueError: If no key field is given, the sample size is not positive, or two selected
            rows share a key.
    """
    if not key_fields:
        raise ValueError("select_sample needs at least one key field")
    if sample_rows <= 0:
        raise ValueError(f"sample_rows must be positive; got {sample_rows!r}")
    candidates = ((_row_digest(row, key_fields, seed), index, dict(row)) for index, row in enumerate(rows))
    selected = heapq.nsmallest(sample_rows, candidates, key=lambda item: (item[0], item[1]))
    keys = [_sample_key(row, key_fields) for _, _, row in selected]
    if len(keys) != len(set(keys)):
        raise ValueError("select_sample selected two rows with the same key")
    return [row for _, _, row in selected]


def top_reason_features(
    shap_row: Sequence[float],
    feature_names: Sequence[str],
    top_n: int = TOP_REASONS,
) -> list[ReasonFeature]:
    """
    WHAT: Returns the strongest positive SHAP contributions of one row, strongest first.

    WHY: A reason is a feature that pushed the score up, so zero and negative contributions are
    dropped before the top three are taken; ties keep the frozen feature order.

    Raises:
        ValueError: If the row and names differ in length, a value is not finite, or top_n is not
            positive.
    """
    if top_n <= 0:
        raise ValueError(f"top_n must be positive; got {top_n!r}")
    values = np.asarray(shap_row, dtype=np.float64)
    if values.shape != (len(feature_names),):
        raise ValueError(f"SHAP row holds {values.size} values for {len(feature_names)} feature names")
    if not bool(np.all(np.isfinite(values))):
        raise ValueError("SHAP values must all be finite")
    ranked = sorted(
        ((float(value), index) for index, value in enumerate(values) if float(value) > 0.0),
        key=lambda item: (-item[0], item[1]),
    )
    return [ReasonFeature(feature=feature_names[index], shap=value) for value, index in ranked[:top_n]]


def reason_codes_for_features(features: Sequence[str], feature_codes: Mapping[str, str]) -> list[str]:
    """
    WHAT: Maps reason features to their codes, keeping one occurrence of each code in rank order.

    WHY: Two reasons in one schema group must not repeat the same code, so the row carries one code
    per group it actually touched.

    Raises:
        ValueError: If a feature has no reason code.
    """
    codes: list[str] = []
    for feature in features:
        code = feature_codes.get(feature)
        if code is None:
            raise ValueError(f"feature {feature!r} has no reason code")
        if code not in codes:
            codes.append(code)
    return codes


def row_reasons(
    shap_row: Sequence[float],
    feature_names: Sequence[str],
    feature_codes: Mapping[str, str],
) -> tuple[list[RowReason], list[str]]:
    """
    WHAT: Returns one row's ranked reasons with their codes and its deduplicated code list.

    WHY: Keeping the reasons and the codes in one place stops a row's two outputs from disagreeing.
    """
    reasons = top_reason_features(shap_row, feature_names)
    codes = reason_codes_for_features([reason.feature for reason in reasons], feature_codes)
    coded = [
        RowReason(feature=reason.feature, shap=reason.shap, code=feature_codes[reason.feature]) for reason in reasons
    ]
    return coded, codes


def _validate_shap(values: np.ndarray, rows: int, feature_count: int, model_name: str) -> np.ndarray:
    """Proves a SHAP matrix covers the sampled rows and features and holds only finite values."""
    if values.shape != (rows, feature_count):
        raise ValueError(f"{model_name} SHAP values have shape {values.shape}, expected {(rows, feature_count)}")
    if not bool(np.all(np.isfinite(values))):
        raise ValueError(f"{model_name} SHAP values hold a non-finite value")
    return values


def account_shap_values(
    model: PUBaggedAccountModel,
    matrix: list[list[float]],
    feature_count: int,
) -> np.ndarray:
    """
    WHAT: Returns the bagged margin-space SHAP values for every row of `matrix`.

    WHY: A-059 uses the production `PUBaggedAccountModel.shap_values`, which averages each member's
    explanation with weight 1/50 in margin space; this wrapper only proves the returned shape.

    Raises:
        ValueError: If the values do not cover every row and feature, or hold a non-finite value.
    """
    values = np.asarray(model.shap_values(matrix), dtype=np.float64)
    return _validate_shap(values, len(matrix), feature_count, "account")


def payment_shap_values(
    model: xgboost.XGBClassifier,
    matrix: np.ndarray,
    feature_count: int,
) -> np.ndarray:
    """
    WHAT: Returns the frozen payment booster's margin-space TreeSHAP values for `matrix`.

    WHY: A-059 explains the single supervised booster rather than an ensemble, in the same margin
    space the account ensemble averages in.

    Raises:
        ValueError: If the values do not cover every row and feature, or hold a non-finite value.
    """
    explainer = shap.TreeExplainer(model)
    values = np.asarray(explainer.shap_values(matrix), dtype=np.float64)
    return _validate_shap(values, matrix.shape[0], feature_count, "payment")


def frequency_table(rows: Sequence[Mapping[str, Any]], codes: Sequence[ReasonCode]) -> dict[str, Any]:
    """
    WHAT: Returns per-code sample counts and the count of rows with no reason.

    WHY: A-059 reports how often each code appears, so the table counts a code once per row on its
    deduplicated list and also records how often it is the first, strongest reason.

    Raises:
        ValueError: If a row's codes are not a list of known codes.
    """
    counts: dict[str, dict[str, int]] = {entry.code: {"rows": 0, "first": 0} for entry in codes}
    without_reason = 0
    for row in rows:
        row_codes = row.get("codes")
        if not isinstance(row_codes, list) or not all(isinstance(code, str) for code in row_codes):
            raise ValueError("a sample row holds no code list")
        if not row_codes:
            without_reason += 1
            continue
        for position, code in enumerate(row_codes):
            count = counts.get(code)
            if count is None:
                raise ValueError(f"a sample row carries unknown code {code!r}")
            count["rows"] += 1
            if position == 0:
                count["first"] += 1
    total = len(rows)
    entries: list[dict[str, Any]] = [
        {
            "code": entry.code,
            "group": entry.group,
            "plain_reason": entry.plain_reason,
            "rows": counts[entry.code]["rows"],
            "share": counts[entry.code]["rows"] / total if total else 0.0,
            "first": counts[entry.code]["first"],
        }
        for entry in codes
    ]
    entries.sort(key=lambda item: (-int(item["rows"]), str(item["code"])))
    return {"rows": total, "rows_without_reason": without_reason, "codes": entries}


def verify_account_checkpoint(
    checkpoint_root: Path,
    extra_reporting_path: Path = DEFAULT_EXTRA_REPORTING,
) -> dict[str, str]:
    """
    WHAT: Recomputes every checkpoint sha256 recorded in the extra-reporting manifest.

    WHY: A-059 names those hashes as the account model's identity; a changed ensemble, member or
    threshold file must stop the phase instead of being explained as if it were frozen.

    Raises:
        ValueError: If the manifest or its hash map is missing, a recorded file is absent, a digest
            differs, or model_bags holds a file the manifest does not record.
    """
    checkpoint_root = Path(checkpoint_root)
    reporting = json.loads(Path(extra_reporting_path).read_text(encoding="utf-8"))
    manifest = reporting.get("manifest") if isinstance(reporting, dict) else None
    hashes = manifest.get("checkpoint_sha256") if isinstance(manifest, dict) else None
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError(f"{extra_reporting_path} holds no checkpoint_sha256 map")
    digests: dict[str, str] = {}
    for relative, expected in hashes.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ValueError(f"{extra_reporting_path} checkpoint_sha256 holds a non-string entry")
        path = checkpoint_root / relative
        if not path.is_file():
            raise ValueError(f"recorded checkpoint file is missing: {path}")
        digest = _sha256_file(path)
        if digest != expected:
            raise ValueError(f"checkpoint file {relative} sha256 {digest} != recorded {expected}")
        digests[relative] = digest
    model_bags = checkpoint_root / "model_bags"
    if not model_bags.is_dir():
        raise ValueError(f"checkpoint has no model_bags folder: {model_bags}")
    present = {path.relative_to(checkpoint_root).as_posix() for path in model_bags.rglob("*") if path.is_file()}
    recorded_bags = {relative for relative in hashes if relative.startswith("model_bags/")}
    if present != recorded_bags:
        unrecorded = sorted(present - recorded_bags)
        missing = sorted(recorded_bags - present)
        raise ValueError(
            f"checkpoint model_bags files differ from the manifest: unrecorded={unrecorded} missing={missing}"
        )
    if "thresholds.json" not in hashes:
        raise ValueError("the checkpoint manifest records no thresholds.json sha256")
    return digests


def _validation_datasets(validation_dirs: Sequence[Path]) -> list[Path]:
    """
    WHAT: Returns the validation folders sorted, proving each is validation and their names differ.

    Raises:
        ValueError: If no folder is given, two folders share a name, or a folder is not validation.
    """
    if not validation_dirs:
        raise ValueError("reason codes need at least one validation dataset")
    datasets = sorted(Path(directory) for directory in validation_dirs)
    names = [directory.name for directory in datasets]
    if len(names) != len(set(names)):
        raise ValueError(f"validation dataset folder names repeat: {names}")
    for directory in datasets:
        validate_dataset_split(directory, "validation")
    return datasets


def _content_hashes(datasets: Sequence[Path]) -> dict[str, Any]:
    """Returns each validation dataset's recorded content hash by folder name."""
    return {dataset.name: validate_dataset_split(dataset, "validation").get("content_hash") for dataset in datasets}


def _account_day_candidates(datasets: Sequence[Path]) -> Iterator[dict[str, Any]]:
    """
    WHAT: Yields every validation account-day row, tagged with its dataset folder name.

    WHY: The sample is taken over all validation account-days, and each world is read on its own so
    only one world's snapshots are held while the sample is selected.

    Raises:
        ValueError: If a manifest holds no world list, a world entry has no world_id, or a snapshot
            row already carries a dataset field.
    """
    for dataset_dir in datasets:
        manifest = validate_dataset_split(dataset_dir, "validation")
        worlds = manifest.get("worlds")
        if not isinstance(worlds, list):
            raise ValueError(f"{dataset_dir.name} manifest holds no world list")
        for entry in worlds:
            world_id = entry.get("world_id") if isinstance(entry, dict) else None
            if not isinstance(world_id, str) or not world_id:
                raise ValueError(f"{dataset_dir.name} manifest holds a world entry without world_id")
            for row in one_row_per_account_day(_read_world_snapshots(dataset_dir, world_id)):
                if "dataset" in row:
                    raise ValueError(f"{dataset_dir.name} snapshot rows already carry a dataset field")
                candidate = dict(row)
                candidate["dataset"] = dataset_dir.name
                yield candidate


def _sorted_by_key(rows: Sequence[dict[str, Any]], key_fields: Sequence[str]) -> list[dict[str, Any]]:
    """Returns the rows ordered by their sample key."""
    return sorted(rows, key=lambda row: _sample_key(row, key_fields))


def _run_account(datasets: Sequence[Path], code_map: ReasonCodeMap) -> dict[str, Any]:
    """
    WHAT: Explains the account model's validation sample and returns the sample, frequency table and
    manifest.

    Raises:
        ValueError: If a frozen checkpoint hash or recipe value disagrees, or the sample is empty.
    """
    recipe = load_frozen_recipe()
    checkpoint_hashes = verify_account_checkpoint(recipe.checkpoint_dir.parent)
    model = PUBaggedAccountModel.load(recipe.checkpoint_dir)
    if model.n_bags != recipe.n_bags:
        raise ValueError(f"loaded account ensemble has {model.n_bags} bags, recipe says {recipe.n_bags}")
    if list(model.member_seeds) != list(range(recipe.seed, recipe.seed + recipe.n_bags)):
        raise ValueError("loaded account ensemble member seeds are not the frozen seed block")

    sample = select_sample(_account_day_candidates(datasets), ACCOUNT_DAY_KEY_FIELDS, SEED, SAMPLE_ROWS)
    if not sample:
        raise ValueError("the validation sample holds no account-day rows")
    sample = _sorted_by_key(sample, ACCOUNT_DAY_KEY_FIELDS)

    matrix = [[_feature_value(row.get(name), name) for name in recipe.feature_names] for row in sample]
    shap_values = account_shap_values(model, matrix, len(recipe.feature_names))
    margins = model.predict_margins(matrix)
    probabilities = model.predict_probabilities(matrix)

    rows: list[dict[str, Any]] = []
    for row, shap_row, margin, probability in zip(sample, shap_values, margins, probabilities, strict=True):
        reasons, codes = row_reasons(shap_row.tolist(), recipe.feature_names, code_map.features)
        rows.append(
            {
                "dataset": _key_text(row, "dataset"),
                "world_id": _key_text(row, "world_id"),
                "bank_id": _key_text(row, "bank_id"),
                "account_id": _key_text(row, "account_id"),
                "snapshot_time": _key_text(row, "snapshot_time"),
                "margin": float(margin),
                "probability": float(probability),
                "reasons": [
                    {"feature": reason.feature, "shap": reason.shap, "code": reason.code} for reason in reasons
                ],
                "codes": codes,
            }
        )
    return {
        "model": "account",
        "feature_count": len(recipe.feature_names),
        "feature_sha256": recipe.feature_sha256,
        "frequency": frequency_table(rows, code_map.codes),
        "sample": rows,
        "manifest": {
            "amendment": "A-059",
            "simulated": True,
            "git_hash": _git_head(),
            "seed": SEED,
            "sample_rows": SAMPLE_ROWS,
            "sampled_rows": len(rows),
            "top_reasons": TOP_REASONS,
            "sampling_rule": _SAMPLING_RULE,
            "validation_dataset_ids": [dataset.name for dataset in datasets],
            "validation_content_hashes": _content_hashes(datasets),
            "code_map": {"path": str(DEFAULT_CODES_PATH), "sha256": _sha256_file(DEFAULT_CODES_PATH)},
            "recipe": {
                "config": str(recipe.config_path),
                "config_sha256": recipe.config_sha256,
                "checkpoint": str(recipe.checkpoint_dir),
                "seed": recipe.seed,
                "n_bags": recipe.n_bags,
                "n_estimators": recipe.n_estimators,
                "feature_sha256": recipe.feature_sha256,
                "feature_count": len(recipe.feature_names),
            },
            "checkpoint_sha256": checkpoint_hashes,
        },
    }


def _run_payment(
    datasets: Sequence[Path],
    model_dir: Path,
    oof_root: Path,
    code_map: ReasonCodeMap,
) -> dict[str, Any]:
    """
    WHAT: Explains the payment model's validation sample and returns the sample, frequency table and
    manifest.

    Raises:
        ValueError: If a frozen artifact hash or shape differs, a dataset holds no payment rows, or
            the selected rows do not match the sampled keys.
    """
    model_dir = Path(model_dir)
    model_sha256, feature_file_sha256 = verify_frozen_artifacts(model_dir)
    payload = load_feature_order(model_dir / "feature_order.yaml")
    features = payload["features"]
    encodings = payload["categorical_encodings"]
    if not isinstance(features, list) or not isinstance(encodings, dict):
        raise ValueError(f"{model_dir} holds malformed feature-order data")
    metadata = json.loads((model_dir / "model_metadata.json").read_text(encoding="utf-8"))
    if (
        not isinstance(metadata, dict)
        or payload["feature_sha256"] != metadata.get("feature_sha256")
        or len(features) != metadata.get("feature_count")
    ):
        raise ValueError("frozen feature order does not match model_metadata.json")
    model = xgboost.XGBClassifier()
    model.load_model(str(model_dir / "frozen_model.json"))
    booster = model.get_booster()
    if booster.num_features() != len(features):
        raise ValueError(f"frozen model takes {booster.num_features()} features but the order holds {len(features)}")
    rounds = verify_booster_rounds(booster, metadata)

    table = load_split_rows(list(datasets), "validation", features, oof_root)
    if table.num_rows == 0:
        raise ValueError("the validation datasets hold no payment rows")
    dataset_of: dict[tuple[str, str], str] = {}
    for dataset_dir in datasets:
        keys = pq.read_table(dataset_dir / "payments.parquet", columns=["world_id", "payment_id"])
        if keys.num_rows == 0:
            raise ValueError(f"validation dataset {dataset_dir.name} holds no payment rows")
        for world_id, payment_id in zip(keys["world_id"].to_pylist(), keys["payment_id"].to_pylist(), strict=True):
            key = (world_id, payment_id)
            if key in dataset_of:
                raise ValueError(f"validation payment key {key} appears in two datasets")
            dataset_of[key] = dataset_dir.name

    def candidates() -> Iterator[dict[str, Any]]:
        for world_id, payment_id in zip(table["world_id"].to_pylist(), table["payment_id"].to_pylist(), strict=True):
            if (world_id, payment_id) not in dataset_of:
                raise ValueError(f"validation payment key {(world_id, payment_id)} belongs to no dataset")
            yield {"dataset": dataset_of[(world_id, payment_id)], "world_id": world_id, "payment_id": payment_id}

    sample = select_sample(candidates(), PAYMENT_KEY_FIELDS, SEED, SAMPLE_ROWS)
    if not sample:
        raise ValueError("the validation sample holds no payment rows")

    wanted = {_sample_key(row, PAYMENT_KEY_FIELDS) for row in sample}
    located: list[dict[str, Any]] = []
    positions: list[int] = []
    for index, (world_id, payment_id) in enumerate(
        zip(table["world_id"].to_pylist(), table["payment_id"].to_pylist(), strict=True)
    ):
        row = {"dataset": dataset_of[(world_id, payment_id)], "world_id": world_id, "payment_id": payment_id}
        if _sample_key(row, PAYMENT_KEY_FIELDS) in wanted:
            located.append(row)
            positions.append(index)
    combined = table.take(pa.array(positions, type=pa.int64()))
    rows = [
        {**row, "decision_time": decision_time}
        for row, decision_time in zip(located, combined["decision_time"].to_pylist(), strict=True)
    ]
    order = sorted(range(len(rows)), key=lambda index: _sample_key(rows[index], PAYMENT_KEY_FIELDS))
    rows = [rows[index] for index in order]
    combined = combined.take(pa.array(order, type=pa.int64()))
    if sorted(_sample_key(row, PAYMENT_KEY_FIELDS) for row in sample) != [
        _sample_key(row, PAYMENT_KEY_FIELDS) for row in rows
    ]:
        raise ValueError("selected payment rows do not match the sampled keys")

    matrix = build_matrix(combined, features, encodings)
    shap_values = payment_shap_values(model, matrix, len(features))
    margins = np.asarray(booster.predict(xgboost.DMatrix(matrix), output_margin=True), dtype=np.float64)
    probabilities = np.asarray(model.predict_proba(matrix)[:, 1], dtype=np.float64)

    explained: list[dict[str, Any]] = []
    for row, shap_row, margin, probability in zip(rows, shap_values, margins, probabilities, strict=True):
        reasons, codes = row_reasons(shap_row.tolist(), features, code_map.features)
        explained.append(
            {
                "dataset": _key_text(row, "dataset"),
                "world_id": _key_text(row, "world_id"),
                "payment_id": _key_text(row, "payment_id"),
                "decision_time": _key_text(row, "decision_time"),
                "margin": float(margin),
                "probability": float(probability),
                "reasons": [
                    {"feature": reason.feature, "shap": reason.shap, "code": reason.code} for reason in reasons
                ],
                "codes": codes,
            }
        )
    return {
        "model": "payment",
        "feature_count": len(features),
        "feature_sha256": payload["feature_sha256"],
        "frequency": frequency_table(explained, code_map.codes),
        "sample": explained,
        "manifest": {
            "amendment": "A-059",
            "simulated": True,
            "git_hash": _git_head(),
            "seed": SEED,
            "sample_rows": SAMPLE_ROWS,
            "sampled_rows": len(explained),
            "top_reasons": TOP_REASONS,
            "sampling_rule": _SAMPLING_RULE,
            "validation_dataset_ids": [dataset.name for dataset in datasets],
            "validation_content_hashes": _content_hashes(datasets),
            "code_map": {"path": str(DEFAULT_CODES_PATH), "sha256": _sha256_file(DEFAULT_CODES_PATH)},
            "model": {
                "path": str(model_dir / "frozen_model.json"),
                "sha256": model_sha256,
                "feature_order_path": str(model_dir / "feature_order.yaml"),
                "feature_order_file_sha256": feature_file_sha256,
                "feature_sha256": payload["feature_sha256"],
                "feature_count": len(features),
                "rounds": rounds,
                "seed": metadata.get("seed"),
                "device": metadata.get("device"),
            },
        },
    }


def _summary(result: Mapping[str, Any]) -> dict[str, Any]:
    """Returns the short console summary for one model's result."""
    frequency = result["frequency"]
    return {
        "model": result["model"],
        "sampled_rows": len(result["sample"]),
        "rows_without_reason": frequency["rows_without_reason"],
        "top_codes": frequency["codes"][:5],
    }


def run(validation_dirs: Sequence[Path], out_root: Path = DEFAULT_OUT_ROOT) -> dict[str, Any]:
    """
    WHAT: Explains both frozen models on their validation samples and writes the two result files.

    WHY: One entry point keeps the frozen-artifact checks, the sample rule and the committed code
    map that produced the numbers in one run, and writes a manifest beside each result.

    Raises:
        ValueError: If no validation folder is given or the committed code map is stale.
    """
    datasets = _validation_datasets(validation_dirs)
    code_map = load_reason_code_map()
    started = time.perf_counter()
    sampler = _CommitSampler()
    with sampler:
        account_started = time.perf_counter()
        account = _run_account(datasets, code_map)
        account_seconds = time.perf_counter() - account_started
        payment_started = time.perf_counter()
        payment = _run_payment(datasets, DEFAULT_MODEL_DIR, DEFAULT_OOF_ROOT, code_map)
        payment_seconds = time.perf_counter() - payment_started
    peak = sampler.peak_gb
    total_seconds = time.perf_counter() - started
    for result, seconds in ((account, account_seconds), (payment, payment_seconds)):
        result["manifest"]["wall_seconds"] = seconds
        result["manifest"]["peak_committed_gb"] = peak
        result["manifest"]["total_wall_seconds"] = total_seconds

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "account_model.json").write_text(json.dumps(account, indent=2), encoding="utf-8")
    (out_root / "payment_model.json").write_text(json.dumps(payment, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "account": _summary(account),
                "payment": _summary(payment),
                "peak_committed_gb": peak,
                "wall_seconds": total_seconds,
                "out_root": str(out_root),
            },
            indent=2,
        ),
        flush=True,
    )
    return {"account": account, "payment": payment}


def main(argv: Sequence[str] | None = None) -> int:
    """Runs the reason-code sample or regenerates the committed mapping."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run", help="Explain the fixed validation sample")
    run_parser.add_argument(
        "--validation", action="append", required=True, help="Validation dataset folder (repeatable)"
    )
    run_parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT), help="Output folder for the two result files")
    commands.add_parser("build-map", help="Regenerate config/feature_reason_codes.yaml")
    args = parser.parse_args(argv)
    if args.command == "build-map":
        reason_map = write_reason_code_map()
        print(json.dumps({"codes": len(reason_map.codes), "features": len(reason_map.features)}, indent=2), flush=True)
        return 0
    run([Path(path) for path in args.validation], Path(args.out_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
