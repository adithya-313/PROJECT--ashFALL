"""
WHAT: Tests that scoring test worlds one at a time gives exactly the results the old whole-dataset code did.

WHY: _evaluate_test_worlds used to turn each whole dataset's tables into Python rows and filter them per
world, which held many gigabytes at full scale. It now filters the Arrow tables to one world first. That is
only safe if every per-world result is identical, so a test-local copy of the old code is run on the very
arguments a real pilot run passed.
"""

from pathlib import Path
from typing import Any

import pytest

import evaluation.run_level_18_pilot as pilot
from evaluation.run_level_18_pilot import run_pilot
from showcase.level_17_checks import TEST_DATASET, TRAINING_DATASET, VALIDATION_DATASET

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES = REPO_ROOT / "showcase" / "fixtures" / "level_17"
TRUTH_ROOT = FIXTURES / "truth"
WORLD_CARDS_ROOT = FIXTURES / "world_cards"


def _old_evaluate_test_worlds(
    test_pairs: Any,
    truth_root: Path,
    world_cards_root: Path,
    tuned: Any,
    detector: Any,
    xgboost_model: Any,
    xgboost_threshold: float,
    settings: Any,
    collect_day_rows: bool = False,
) -> dict[str, dict[str, Any]]:
    """The previous implementation: whole-dataset to_pylist, then a list comprehension per world."""
    per_world: dict[str, dict[str, Any]] = {}
    repeat_policy = pilot.load_escalation_repeat_policy()
    for dataset_dir, dataset in test_pairs:
        payments, snapshots = pilot.join_truth(dataset_dir, truth_root)
        payment_rows = payments.to_pylist()
        snapshot_rows = snapshots.to_pylist()
        labels_rows = dataset.labels.to_pylist()
        for world in dataset.manifest["worlds"]:
            world_id = str(world["world_id"])
            run_id = str(world["run_id"])
            world_payments = [row for row in payment_rows if row["world_id"] == world_id]
            world_snapshots = [row for row in snapshot_rows if row["world_id"] == world_id]
            day_rows = pilot.one_row_per_account_day(world_snapshots)
            scored = pilot.score_account_days(day_rows, tuned.rules_config, detector)
            probabilities = pilot.predict_account_days(xgboost_model, day_rows)
            xgboost_scored = [
                pilot.replace(scored_day, anomaly_score=probability)
                for scored_day, probability in zip(scored, probabilities, strict=True)
            ]
            snapshot_times = pilot.snapshot_time_by_account_day(day_rows)
            truth_store = pilot.load_truth(truth_root, run_id)
            first_credit = pilot.first_illicit_credit(world_payments, truth_store.story_of)
            reported = pilot.reported_accounts(labels_rows, run_id)
            inflation_factor = pilot._world_inflation_factor(
                pilot.read_world_base_rate(world_cards_root, run_id), run_id
            )
            scorers: dict[str, Any] = {}
            for scorer in pilot._LEVEL17_SCORERS:
                scorers[scorer] = pilot.evaluate_world(
                    scored,
                    scorer,
                    tuned.anomaly_result.threshold,
                    settings.capacity_pct_active_accounts_per_day,
                    inflation_factor,
                    truth_store.persona_of,
                    settings.persona_weights,
                    settings.default_persona_weight,
                    snapshot_times,
                    first_credit,
                    reported,
                    repeat_policy=repeat_policy,
                )
            scorers["xgboost"] = pilot.evaluate_world(
                xgboost_scored,
                "anomaly",
                xgboost_threshold,
                settings.capacity_pct_active_accounts_per_day,
                inflation_factor,
                truth_store.persona_of,
                settings.persona_weights,
                settings.default_persona_weight,
                snapshot_times,
                first_credit,
                reported,
                repeat_policy=repeat_policy,
            )
            per_world[world_id] = {"run_id": run_id, "split": "test", "scorers": scorers}
    return per_world


def _capture_main_test_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[tuple[Any, ...], dict[str, Any], Any]:
    """Runs the fixture pilot and returns the arguments and per-world result of its main test scoring call."""
    captured: list[tuple[tuple[Any, ...], dict[str, Any], Any]] = []
    original = pilot._evaluate_test_worlds

    def spy(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        if kwargs.get("reservoir") is not None:
            captured.append((args, kwargs, result))
        return result

    monkeypatch.setattr(pilot, "_evaluate_test_worlds", spy)
    exit_code = run_pilot(
        [TRAINING_DATASET], [VALIDATION_DATASET], [TEST_DATASET], TRUTH_ROOT, WORLD_CARDS_ROOT, tmp_path
    )
    assert exit_code == 0
    assert len(captured) == 1
    return captured[0]


def test_per_world_results_equal_the_old_whole_dataset_code(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    args, _kwargs, (new_per_world, _rows) = _capture_main_test_call(monkeypatch, tmp_path)

    old_per_world = _old_evaluate_test_worlds(*args)

    assert list(new_per_world) == list(old_per_world)
    assert new_per_world == old_per_world


def test_world_rows_equals_a_list_comprehension_over_the_whole_table() -> None:
    _payments, snapshots = pilot.join_truth(TEST_DATASET, TRUTH_ROOT)
    world_id = snapshots.column("world_id")[0].as_py()

    whole = [row for row in snapshots.to_pylist() if row["world_id"] == world_id]

    assert pilot._world_rows(snapshots, "world_id", world_id) == whole


def _two_world_tables() -> tuple[Any, Any, Any]:
    """Payments, snapshots and labels for two worlds whose rows and registry labels differ."""
    from datetime import datetime, timezone

    import pyarrow

    moment = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)
    payments = pyarrow.table(
        {
            "world_id": ["world_a", "world_b", "world_a"],
            "payment_id": ["pay_a1", "pay_b1", "pay_a2"],
        }
    )
    snapshots = pyarrow.table(
        {
            "world_id": ["world_a", "world_b", "world_a"],
            "bank_id": ["bank_a", "bank_a", "bank_a"],
            "account_id": ["acc_1", "acc_1", "acc_2"],
            "snapshot_time": [moment, moment, moment],
            "snapshot_kind": ["daily", "daily", "daily"],
        }
    )
    labels = pyarrow.table(
        {
            "run_id": ["run_a", "run_b", "run_a"],
            "source": ["registry", "registry", "complaint"],
            "account_id": ["acc_1_reported_in_a", "acc_1_reported_in_b", "acc_2_complaint_in_a"],
        }
    )
    return payments, snapshots, labels


def test_each_world_is_scored_with_only_its_own_payments_snapshots_and_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    payments, snapshots, labels = _two_world_tables()
    recorded: dict[str, dict[str, Any]] = {}

    def record_world(scored: Any, *_: Any, **__: Any) -> dict[str, Any]:
        # evaluate_world's arguments: scored rows, then (first_credit, reported) as the 10th and 11th.
        arguments = _
        recorded.setdefault(scored[0]["world_id"], {})
        recorded[scored[0]["world_id"]]["scored_accounts"] = sorted(row["account_id"] for row in scored)
        recorded[scored[0]["world_id"]]["first_credit"] = sorted(arguments[8])
        recorded[scored[0]["world_id"]]["reported"] = sorted(arguments[9])
        return {}

    monkeypatch.setattr(pilot, "join_truth", lambda _dir, _root: (payments, snapshots))
    monkeypatch.setattr(pilot, "load_truth", lambda _root, _run: SimpleNamespace(story_of=None, persona_of=None))
    monkeypatch.setattr(pilot, "first_illicit_credit", lambda rows, _story: {row["payment_id"]: None for row in rows})
    monkeypatch.setattr(pilot, "score_account_days", lambda rows, _config, _detector: list(rows))
    monkeypatch.setattr(pilot, "predict_account_days", lambda _model, rows: [0.5] * len(rows))
    monkeypatch.setattr(pilot, "replace", lambda day, anomaly_score: day)
    monkeypatch.setattr(pilot, "snapshot_time_by_account_day", lambda _rows: {})
    monkeypatch.setattr(
        pilot, "read_world_base_rate", lambda _root, _run: {"complaints": 1, "fraud_inflation_factor": 2.0}
    )
    monkeypatch.setattr(pilot, "load_escalation_repeat_policy", lambda: "every_day")
    monkeypatch.setattr(pilot, "evaluate_world", record_world)
    dataset = SimpleNamespace(
        labels=labels,
        manifest={
            "worlds": [
                {"world_id": "world_a", "run_id": "run_a"},
                {"world_id": "world_b", "run_id": "run_b"},
            ]
        },
    )

    # Stand-ins: evaluate_world and every loader are replaced above, so these only need the fields read.
    test_pairs: Any = [(Path("dataset"), dataset)]
    tuned: Any = SimpleNamespace(rules_config=None, anomaly_result=SimpleNamespace(threshold=0.5))
    settings: Any = SimpleNamespace(
        capacity_pct_active_accounts_per_day=1.0, persona_weights={}, default_persona_weight=1
    )
    no_detector: Any = None
    no_model: Any = None

    pilot._evaluate_test_worlds(test_pairs, Path("truth"), Path("cards"), tuned, no_detector, no_model, 0.5, settings)

    assert recorded["world_a"]["scored_accounts"] == ["acc_1", "acc_2"]
    assert recorded["world_b"]["scored_accounts"] == ["acc_1"]
    assert recorded["world_a"]["first_credit"] == ["pay_a1", "pay_a2"]
    assert recorded["world_b"]["first_credit"] == ["pay_b1"]
    assert recorded["world_a"]["reported"] == ["acc_1_reported_in_a"]
    assert recorded["world_b"]["reported"] == ["acc_1_reported_in_b"]
