"""
WHAT: Unit tests for the Level 18a4 sanity report: feature-fill flagging at the 0.999 boundary, an
oracle scored through the pilot's escalation and metric path, and the CLI report.

WHY: A sparse feature or a capacity ceiling must be visible before a model is blamed; the oracle
test proves the report reaches full precision when the ranking is perfect.
"""

import json
from pathlib import Path

import orjson

from decisions.mule_account_model import account_model_feature_names
from evaluation.dataset_truth_join import join_truth
from evaluation.level18_sanity import (
    FEATURE_FILL_FLAG_SHARE,
    build_sanity_report,
    feature_fill_report,
    main,
    oracle_report,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES = REPO_ROOT / "showcase" / "fixtures" / "level_17"
VALIDATION = FIXTURES / "datasets" / "validation-20261031T1830-e5593f4c3f8e"
TRUTH_ROOT = FIXTURES / "truth"


def _filled_row() -> dict[str, object]:
    """Builds one snapshot row carrying a value for every account-model feature."""
    return {name: 1.0 for name in account_model_feature_names()}


def test_feature_fill_flags_a_feature_missing_from_every_row() -> None:
    """A feature that is None on every row is reported with a zero fill share."""
    rows = [_filled_row() for _ in range(10)]
    for row in rows:
        row["distinct_payers_24h"] = None

    report = feature_fill_report(rows)

    assert report["feature_fill_flag_share"] == FEATURE_FILL_FLAG_SHARE
    assert report["row_count"] == 10
    assert report["flagged_features"] == [{"feature": "distinct_payers_24h", "fill_share": 0.0}]


def test_feature_fill_flags_only_features_strictly_below_the_share() -> None:
    """A feature carried by exactly 999 of 1000 rows is not flagged; 998 rows is."""
    rows = [_filled_row() for _ in range(1000)]
    rows[0]["distinct_payers_24h"] = None
    rows[0]["distinct_payers_7d"] = None
    rows[1]["distinct_payers_7d"] = None

    report = feature_fill_report(rows)

    flagged = {entry["feature"]: entry["fill_share"] for entry in report["flagged_features"]}
    assert flagged == {"distinct_payers_7d": 0.998}


def test_oracle_reaches_full_yield_through_the_pilot_path() -> None:
    """With perfect ranking every escalated account-day is a mule day, so measured yield is 1.0."""
    _, snapshots = join_truth(VALIDATION, TRUTH_ROOT)
    report = oracle_report(snapshots.to_pylist())

    assert report["escalations"] > 0
    assert report["mule_accounts"] > 0
    assert report["yield_measured"] == 1.0
    assert report["recall_at_capacity"] > 0.0


def test_build_sanity_report_covers_every_validation_world() -> None:
    """The report names the dataset and carries one oracle block per world in its manifest."""
    report = build_sanity_report((VALIDATION,), TRUTH_ROOT)
    manifest = json.loads((VALIDATION / "manifest.json").read_text(encoding="utf-8"))

    assert report["validation_datasets"] == [VALIDATION.name]
    assert [block["world_id"] for block in report["oracle_by_world"]] == [
        str(world["world_id"]) for world in manifest["worlds"]
    ]
    assert report["feature_fill"]["row_count"] > 0


def test_cli_writes_the_report(tmp_path: Path) -> None:
    """The CLI writes the same report structure as build_sanity_report."""
    out = tmp_path / "sanity.json"

    exit_code = main(
        [
            "--validation",
            str(VALIDATION),
            "--truth-root",
            str(TRUTH_ROOT),
            "--out",
            str(out),
        ]
    )
    payload = orjson.loads(out.read_bytes())

    assert exit_code == 0
    assert set(payload) == {"feature_fill", "oracle_by_world", "validation_datasets"}
    assert payload["oracle_by_world"][0]["oracle"]["yield_measured"] == 1.0
