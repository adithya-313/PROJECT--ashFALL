"""
WHAT: Leakage checks for the Level 18 pilot datasets, starting with Section 23 check 7.

WHY: The account model may use features built from labels a bank has already received, but a value
of zero hops must mean the label really had arrived by the snapshot time. This module compares the
stored snapshot values with the dataset's label arrival times, world by world, without a tolerance
window. Label rows are the bank's own feed: only a real complaint (source "complaint", no routing
channel) may support hops_to_complained_account, because that is exactly what fills the complained
set in features/account_labels.py.
"""

from datetime import datetime
from typing import Any, Sequence

from decisions.mule_account_model import account_model_feature_names, build_training_rows

__all__ = [
    "check_label_features_by_arrival",
    "combine_label_feature_checks",
    "feedback_distinct_values",
]

COMPLAINED_FEATURE = "hops_to_complained_account"
FEEDBACK_FEATURES: tuple[str, ...] = (
    "hops_to_confirmed_label",
    "shared_counterparties_with_flagged_7d",
)
REGISTRY_FEATURES: tuple[str, ...] = ("registry_match_count", "registry_match_age_days")
REGISTRY_VISIBILITY_TESTS: tuple[str, ...] = (
    "tests/unit/test_registry_as_of.py::test_as_of_visibility_before_and_after",
    "tests/unit/test_registry_as_of.py::test_as_of_revocation_before_and_after",
    "tests/unit/test_registry_as_of.py::test_as_of_dispute_contested_and_resolved",
)
MAX_EXAMPLES = 5


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


def _require_aware_datetime(row: dict[str, Any], key: str, caller: str) -> datetime:
    """
    WHAT: Returns a required timezone-aware datetime field from a row.

    Raises:
        ValueError: If the field is missing, not a datetime, or timezone-naive.
    """
    value = row.get(key)
    if not isinstance(value, datetime):
        raise ValueError(f"{caller} requires a datetime '{key}' on every row")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{caller} requires a timezone-aware '{key}'")
    return value


def _complained_label_arrivals(
    label_rows: Sequence[dict[str, Any]],
) -> dict[tuple[str, str], list[datetime]]:
    """
    WHAT: Indexes real complaint arrivals per (world_id, account_id).

    WHY: The complained set is filled from bank_complaints rows only, so only source "complaint"
    with no routing channel (via is None) can justify a zero hop. A registry, police_bulk or
    hold_request label is a different kind and must not satisfy the check.
    """
    arrivals: dict[tuple[str, str], list[datetime]] = {}
    for row in label_rows:
        if row.get("source") != "complaint" or row.get("via") is not None:
            continue
        world_id = _require_string(row, "world_id", "check_label_features_by_arrival")
        account_id = _require_string(row, "account_id", "check_label_features_by_arrival")
        arrival_time = _require_aware_datetime(row, "arrival_time", "check_label_features_by_arrival")
        arrivals.setdefault((world_id, account_id), []).append(arrival_time)
    return arrivals


def _complained_block(
    snapshot_rows: Sequence[dict[str, Any]],
    label_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """
    WHAT: Proves every zero hops_to_complained_account snapshot has a complaint that arrived by then.

    Returns:
        Snapshots checked, violation count and up to MAX_EXAMPLES example keys.
    """
    arrivals = _complained_label_arrivals(label_rows)
    checked = 0
    violations = 0
    examples: list[dict[str, Any]] = []
    for row in snapshot_rows:
        hops = row.get(COMPLAINED_FEATURE)
        if hops is None or hops != 0:
            continue
        checked += 1
        world_id = _require_string(row, "world_id", "check_label_features_by_arrival")
        account_id = _require_string(row, "account_id", "check_label_features_by_arrival")
        snapshot_time = _require_aware_datetime(row, "snapshot_time", "check_label_features_by_arrival")
        visible = any(arrival <= snapshot_time for arrival in arrivals.get((world_id, account_id), ()))
        if visible:
            continue
        violations += 1
        if len(examples) < MAX_EXAMPLES:
            examples.append(
                {
                    "world_id": world_id,
                    "account_id": account_id,
                    "snapshot_time": snapshot_time.isoformat(),
                    "hops_to_complained_account": hops,
                }
            )
    return {
        "feature": COMPLAINED_FEATURE,
        "snapshots_checked": checked,
        "violations": violations,
        "examples": examples,
    }


def feedback_distinct_values(snapshot_rows: Sequence[dict[str, Any]]) -> dict[str, set[Any]]:
    """
    WHAT: Returns the distinct non-null values of each feedback-derived feature.

    WHY: The check runs one world at a time so whole-split rows are never held, and the split-wide
    distinct counts then have to be rebuilt by unioning these small value sets across worlds
    instead of re-reading the rows.

    Args:
        snapshot_rows: One world's snapshot rows.

    Returns:
        Feature name -> set of the non-null values seen for that feature.
    """
    values: dict[str, set[Any]] = {name: set() for name in FEEDBACK_FEATURES}
    for row in snapshot_rows:
        for name in FEEDBACK_FEATURES:
            value = row.get(name)
            if value is None:
                continue
            values[name].add(value)
    return values


def _feedback_block(snapshot_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """
    WHAT: Counts snapshots carrying a feedback-derived feature value and checks it never varies.

    WHY: No replay component writes bank_feedback, so these features are empty (all null) or carry
    one constant empty-value (zero); a column that takes more than one value has no feedback source
    to explain it and is itself a leak signal worth investigating.
    """
    counts = {name: 0 for name in FEEDBACK_FEATURES}
    rows_with_any: set[int] = set()
    for index, row in enumerate(snapshot_rows):
        for name in FEEDBACK_FEATURES:
            if row.get(name) is None:
                continue
            counts[name] += 1
            rows_with_any.add(index)
    distinct_counts = {
        name: len(values) for name, values in feedback_distinct_values(snapshot_rows).items()
    }
    varied = any(distinct_count > 1 for distinct_count in distinct_counts.values())
    return {
        "features": list(FEEDBACK_FEATURES),
        "snapshots_checked": len(snapshot_rows),
        "non_null_snapshot_counts": counts,
        "distinct_value_counts": distinct_counts,
        "snapshots_with_any_non_null": len(rows_with_any),
        "status": "fail" if varied else "no source data",
    }


def _positive_rows_with_zero_complained_hops(
    snapshot_rows: Sequence[dict[str, Any]],
    label_rows: Sequence[dict[str, Any]],
) -> int:
    """
    WHAT: Counts positive training rows whose hops_to_complained_account is zero.

    WHY: A positive row is cut at the label's arrival, so no positive row should already see a
    complaint that justifies a zero hop; a nonzero count means the cut let a labelled row through.
    """
    names = account_model_feature_names()
    if COMPLAINED_FEATURE not in names:
        raise ValueError(
            f"{COMPLAINED_FEATURE} is not in account_model_feature_names(); check 7 needs the label network switch on"
        )
    index = names.index(COMPLAINED_FEATURE)
    matrix, targets, _ = build_training_rows(list(snapshot_rows), list(label_rows))
    return sum(1 for row, target in zip(matrix, targets, strict=True) if target == 1 and row[index] == 0)


def check_label_features_by_arrival(
    snapshot_rows: Sequence[dict[str, Any]],
    label_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """
    WHAT: Runs Section 23 check 7 over one split's dataset rows.

    WHY: Label-based features are only legitimate if their values can be explained by labels that
    had already arrived. The check compares stored values against stored arrival times exactly.

    Args:
        snapshot_rows: Dataset snapshot rows carrying world_id, account_id, snapshot_time and the
            label-based feature columns.
        label_rows: Dataset label rows carrying world_id, account_id, source, via and arrival_time.

    Returns:
        A block with the complained violations, the feedback source-data status, the positive-row
        count and the registry coverage note. Status is "pass" only when the complained feature has
        no violations and the feedback features carry no source data.

    Raises:
        ValueError: If a required field is missing or malformed, or the committed model config does
            not carry the complained feature.
    """
    complained = _complained_block(snapshot_rows, label_rows)
    feedback = _feedback_block(snapshot_rows)
    positive_zero = _positive_rows_with_zero_complained_hops(snapshot_rows, label_rows)
    passed = complained["violations"] == 0 and feedback["status"] == "no source data"
    return {
        "status": "pass" if passed else "fail",
        "complained": complained,
        "feedback_features": feedback,
        "positive_training_rows_with_zero_complained_hops": positive_zero,
        "registry_features": {
            "features": list(REGISTRY_FEATURES),
            "status": "covered by function tests",
            "tests": list(REGISTRY_VISIBILITY_TESTS),
        },
    }


def combine_label_feature_checks(
    blocks: Sequence[dict[str, Any]],
    distinct_value_sets: Sequence[dict[str, set[Any]]],
    world_count: int,
) -> dict[str, Any]:
    """
    WHAT: Combines per-world Section 23 check 7 blocks into the split-wide block.

    WHY: Running the check one world at a time keeps whole-split rows out of memory, but the
    published block is still defined over the split: counts add up, the first MAX_EXAMPLES
    violations are kept in world order, and the feedback distinct counts are rebuilt by unioning
    the per-world value sets instead of re-reading the rows.

    Args:
        blocks: Per-world blocks in sorted world_id order.
        distinct_value_sets: Per-world feedback_distinct_values outputs, aligned with blocks.
        world_count: Number of worlds combined, recorded in the result.

    Returns:
        A block with the same keys as check_label_features_by_arrival, plus world_count.

    Raises:
        ValueError: If no block was supplied, the inputs disagree in length, or a value set does
            not name exactly the feedback features.
    """
    if not blocks:
        raise ValueError("combine_label_feature_checks needs at least one world block")
    if len(distinct_value_sets) != len(blocks):
        raise ValueError(
            f"combine_label_feature_checks got {len(distinct_value_sets)} value sets for {len(blocks)} blocks"
        )
    if world_count != len(blocks):
        raise ValueError(
            f"combine_label_feature_checks got world_count {world_count} for {len(blocks)} blocks"
        )
    examples: list[dict[str, Any]] = []
    for block in blocks:
        for example in block["complained"]["examples"]:
            if len(examples) < MAX_EXAMPLES:
                examples.append(example)
    merged_values: dict[str, set[Any]] = {name: set() for name in FEEDBACK_FEATURES}
    for values in distinct_value_sets:
        if set(values) != set(FEEDBACK_FEATURES):
            raise ValueError(
                f"feedback value sets must name exactly {list(FEEDBACK_FEATURES)}; got {sorted(values)}"
            )
        for name in FEEDBACK_FEATURES:
            merged_values[name].update(values[name])
    distinct_counts = {name: len(values) for name, values in merged_values.items()}
    feedback_status = "fail" if any(count > 1 for count in distinct_counts.values()) else "no source data"
    violations = sum(block["complained"]["violations"] for block in blocks)
    passed = violations == 0 and feedback_status == "no source data"
    return {
        "status": "pass" if passed else "fail",
        "complained": {
            "feature": COMPLAINED_FEATURE,
            "snapshots_checked": sum(block["complained"]["snapshots_checked"] for block in blocks),
            "violations": violations,
            "examples": examples,
        },
        "feedback_features": {
            "features": list(FEEDBACK_FEATURES),
            "snapshots_checked": sum(
                block["feedback_features"]["snapshots_checked"] for block in blocks
            ),
            "non_null_snapshot_counts": {
                name: sum(
                    block["feedback_features"]["non_null_snapshot_counts"][name] for block in blocks
                )
                for name in FEEDBACK_FEATURES
            },
            "distinct_value_counts": distinct_counts,
            "snapshots_with_any_non_null": sum(
                block["feedback_features"]["snapshots_with_any_non_null"] for block in blocks
            ),
            "status": feedback_status,
        },
        "positive_training_rows_with_zero_complained_hops": sum(
            block["positive_training_rows_with_zero_complained_hops"] for block in blocks
        ),
        "registry_features": blocks[0]["registry_features"],
        "world_count": world_count,
    }
