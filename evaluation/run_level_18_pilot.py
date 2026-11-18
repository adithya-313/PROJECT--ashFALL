"""
WHAT: Runs the Level 18 pilot: trains the XGBoost account model on bank-visible labels, tunes its
alarm cut on validation, evaluates it beside the rules and anomaly detector on the same test
account-days, and writes result.json, leakage_checks.json and shap_top_features.json.

WHY: The pilot only means something if all four scorers see identical account-days, identical
truth and the same metric code, and if every possible leak is checked and printed. Test worlds are
used exactly once, after every choice was made on training and validation data.

TERM: pilot: a small simulated run whose numbers are plumbing checks, not production results.
TERM: split leak: a run_id or world_id that appears in more than one of training, validation and test.

Every dataset folder, truth file and world card is required. Missing or malformed inputs raise a
plain ValueError naming the offending path.
"""

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import numpy
import orjson
import shap
import xgboost

from core.event_clock import IST
from core.truth_import_check import find_truth_violations
from decisions.anomaly_config import load_anomaly_detector_config
from decisions.anomaly_detector import TrainedAnomalyDetector, train_anomaly_detector
from decisions.mule_account_model import (
    account_model_feature_names,
    active_mule_model_config_path,
    build_training_rows,
    load_mule_model_config,
    predict_account_days,
    save_model,
    set_mule_model_config_path,
    train_account_model,
)
from decisions.pu_learning import PUBaggedAccountModel
from decisions.rules_config import (
    RulesConfig,
    load_rules_config,
    load_rules_config_with_labels,
    with_thresholds,
)
from evaluation.dataset_truth_join import join_truth
from evaluation.leakage_checks import COMPLAINED_FEATURE, check_label_features_by_arrival
from evaluation.level17_evaluation import (
    evaluate_world,
    first_illicit_credit,
    world_start_by_world,
)
from evaluation.level17_scoring import ScoredDay, one_row_per_account_day, score_account_days
from evaluation.level18_sanity import load_escalation_repeat_policy
from evaluation.operating_points import operating_points, rules_curve, score_curve
from evaluation.rules_tuning import (
    AnomalyTuningResult,
    TuningConfig,
    load_labels_tuning_config,
    load_tuning_config,
    tune_anomaly_threshold,
    tune_rules,
)
from evaluation.run_level_17 import (
    EvaluationSettings,
    TunedThresholds,
    _dataset_entry,
    _sha256_of_file,
    _world_inflation_factor,
    build_result_manifest,
    load_evaluation_settings,
    load_split_datasets,
    read_world_base_rate,
    reported_accounts,
    run_key,
    snapshot_time_by_account_day,
    summarise_worlds,
    write_result,
)
from evaluation.trivial_world import average_precision, best_direction_ap, load_evaluation_config
from evaluation.truth import load_truth
from replay.dataset_builder import Dataset

__all__ = ["PILOT_SCORERS", "main", "run_pilot"]

REPO_ROOT = Path(__file__).resolve().parent.parent

PILOT_SCORERS: tuple[str, ...] = ("rules", "anomaly", "rules_and_anomaly", "xgboost")

N_ESTIMATOR_CANDIDATES: tuple[int, ...] = (100, 200, 400)

_LEVEL17_SCORERS: tuple[str, ...] = ("rules", "anomaly", "rules_and_anomaly")


def _dataset_identities(datasets: Sequence[Dataset]) -> tuple[set[str], set[str]]:
    """
    WHAT: Returns the run ids and world ids recorded by a split's datasets.

    Args:
        datasets: Loaded datasets of one split.

    Returns:
        (run_ids, world_ids) as sets of strings.
    """
    run_ids: set[str] = set()
    world_ids: set[str] = set()
    for dataset in datasets:
        for world in dataset.manifest["worlds"]:
            run_ids.add(str(world["run_id"]))
            world_ids.add(str(world["world_id"]))
    return run_ids, world_ids


def _split_leak(
    training_datasets: Sequence[Dataset],
    validation_datasets: Sequence[Dataset],
    test_datasets: Sequence[Dataset],
) -> dict[str, Any]:
    """
    WHAT: Checks whether any run_id or world_id is shared by two dataset splits.

    WHY: A shared run or world would let validation or test information into training, so the
    pilot must refuse to run on it and must record the exact overlapping ids.

    Args:
        training_datasets: Loaded training datasets.
        validation_datasets: Loaded validation datasets.
        test_datasets: Loaded test datasets.

    Returns:
        A check block with status pass or fail, each split's ids, and every overlap found.
    """
    splits = {
        "training": _dataset_identities(training_datasets),
        "validation": _dataset_identities(validation_datasets),
        "test": _dataset_identities(test_datasets),
    }
    overlaps: list[dict[str, Any]] = []
    for left, right in (
        ("training", "validation"),
        ("training", "test"),
        ("validation", "test"),
    ):
        shared_runs = sorted(splits[left][0] & splits[right][0])
        shared_worlds = sorted(splits[left][1] & splits[right][1])
        if shared_runs or shared_worlds:
            overlaps.append(
                {
                    "splits": [left, right],
                    "run_ids": shared_runs,
                    "world_ids": shared_worlds,
                }
            )
    return {
        "status": "fail" if overlaps else "pass",
        "run_ids": {name: sorted(ids[0]) for name, ids in splits.items()},
        "world_ids": {name: sorted(ids[1]) for name, ids in splits.items()},
        "overlaps": overlaps,
    }


def _write_leakage(path: Path, checks: dict[str, Any]) -> None:
    """
    WHAT: Writes the leakage report next to result.json with the pilot markers.

    Args:
        path: Destination leakage_checks.json.
        checks: Check name -> check block.
    """
    payload = {"simulated": True, "pilot": True, "checks": checks}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(orjson.dumps(payload, option=orjson.OPT_SORT_KEYS | orjson.OPT_INDENT_2) + b"\n")


def _not_run_checks(reason: str) -> dict[str, Any]:
    """
    WHAT: Returns the leakage blocks that were not run, each carrying the given reason.

    Args:
        reason: Why the checks did not run, for example "split leak" or "pilot".
    """
    return {
        "truth_leak": {"status": "not run", "reason": reason},
        "trivial_world": {"status": "not run", "reason": reason},
        "section_23_checks_4_to_7": {"status": "not run", "reason": reason, "checks": [4, 5, 6, 7]},
    }


def _day_rows_by_world(snapshot_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    WHAT: Deduplicates snapshots to one row per account-day inside each world.

    WHY: Account identifiers repeat across simulated worlds, so a single global dedupe would merge
    unrelated accounts. Grouping by world_id first keeps every world's account-days intact.

    Args:
        snapshot_rows: Raw snapshot rows of one or more worlds.

    Returns:
        One row per (world, bank, account, IST day), in world-sorted first-seen order.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in snapshot_rows:
        world_id = row.get("world_id")
        if not isinstance(world_id, str) or not world_id:
            raise ValueError("Snapshot rows require a non-empty string 'world_id'")
        grouped.setdefault(world_id, []).append(row)
    day_rows: list[dict[str, Any]] = []
    for world_id in sorted(grouped):
        day_rows.extend(one_row_per_account_day(grouped[world_id]))
    return day_rows


def _row_key(row: dict[str, Any]) -> tuple[str, str, str, object]:
    """
    WHAT: Returns the (world, bank, account, IST day) key used to align training rows with snapshots.

    Raises:
        ValueError: If a required identifier or snapshot time is missing.
    """
    world_id = row.get("world_id")
    bank_id = row.get("bank_id")
    account_id = row.get("account_id")
    moment = row.get("snapshot_time")
    if not isinstance(world_id, str) or not world_id:
        raise ValueError("Snapshot rows require a non-empty string 'world_id'")
    if not isinstance(bank_id, str) or not bank_id:
        raise ValueError("Snapshot rows require a non-empty string 'bank_id'")
    if not isinstance(account_id, str) or not account_id:
        raise ValueError("Snapshot rows require a non-empty string 'account_id'")
    if not hasattr(moment, "astimezone"):
        raise ValueError("Snapshot rows require a timezone-aware 'snapshot_time'")
    return (world_id, bank_id, account_id, moment.astimezone(IST).date())


def _align_rows(
    day_rows: Sequence[dict[str, Any]],
    keys: Sequence[tuple[str, str, str, object]],
) -> list[dict[str, Any]]:
    """
    WHAT: Returns the day rows that produced a build_training_rows key, in key order.

    WHY: build_training_rows returns matrix rows and keys but not the original snapshot rows, and
    predict_account_days takes rows so its feature order stays identical to training.

    Raises:
        ValueError: If a key has no matching account-day row.
    """
    by_key = {_row_key(row): row for row in day_rows}
    aligned: list[dict[str, Any]] = []
    for key in keys:
        row = by_key.get(key)
        if row is None:
            raise ValueError(f"Training row key {key!r} has no matching account-day snapshot")
        aligned.append(row)
    return aligned


def _feature_matrix(day_rows: Sequence[dict[str, Any]], feature_names: Sequence[str]) -> list[list[float]]:
    """
    WHAT: Builds the numeric matrix the model expects, with None represented as NaN.

    WHY: SHAP needs the exact matrix that produced the probabilities, and XGBoost treats NaN as a
    missing value rather than zero.

    Raises:
        ValueError: If a feature value is not numeric.
    """
    matrix: list[list[float]] = []
    for row in day_rows:
        values: list[float] = []
        for name in feature_names:
            value = row.get(name)
            if value is None:
                values.append(float("nan"))
            elif isinstance(value, bool):
                values.append(float(value))
            elif isinstance(value, (int, float)):
                values.append(float(value))
            else:
                raise ValueError(f"Feature {name!r} must be numeric or None; got {value!r}")
        matrix.append(values)
    return matrix


def _tune_on_validation(
    validation_dirs: Sequence[Path],
    truth_root: Path,
    base_rules_config: RulesConfig,
    tuning: TuningConfig,
    detector: TrainedAnomalyDetector,
    budget_pct: float,
    *,
    collect_grid_points: bool = False,
) -> tuple[TunedThresholds, list[ScoredDay], list[dict[str, Any]]]:
    """
    WHAT: Tunes the rules thresholds and both anomaly-style cuts on the validation account-days.

    WHY: All four scorers must be set to the same genuine-restriction budget on validation, never on
    test. The XGBoost cut uses the same tuner and the same rows as the detector cut, so the only
    difference between them is the score itself.

    Args:
        validation_dirs: Validation dataset folders.
        truth_root: Root folder holding one truth folder per run_id.
        base_rules_config: Shipped rules table supplying the untrained thresholds.
        tuning: Candidate grid and allowed tuning splits.
        detector: Trained anomaly detector.
        budget_pct: Maximum share of genuine account-days either system may restrict.
        collect_grid_points: When true, the rules result keeps every grid point so the pilot can
            report operating points at the extra committed budgets from the same scoring pass.

    Returns:
        (tuned thresholds, validation scored days with detector scores, the account-day rows
        those scores were computed from, in score order).

    Raises:
        ValueError: If the datasets cannot be joined or a tuner refuses its inputs.
    """
    snapshot_rows: list[dict[str, Any]] = []
    for validation_dir in validation_dirs:
        _, snapshots = join_truth(validation_dir, truth_root)
        snapshot_rows.extend(snapshots.to_pylist())
    day_rows = one_row_per_account_day(snapshot_rows)

    rules_result = tune_rules(
        day_rows,
        base_rules_config,
        tuning,
        "validation",
        budget_pct,
        collect_grid_points=collect_grid_points,
    )
    tuned_config = with_thresholds(
        base_rules_config,
        rules_result.thresholds,
        "tuned on pilot validation datasets " + ", ".join(sorted(directory.name for directory in validation_dirs)),
    )
    scored = score_account_days(day_rows, base_rules_config, detector)
    anomaly_result = tune_anomaly_threshold(scored, tuning, "validation", budget_pct)

    tuned = TunedThresholds(
        rules_config=tuned_config,
        rules_result=rules_result,
        anomaly_result=anomaly_result,
    )
    return tuned, scored, day_rows


def _evaluate_test_worlds(
    test_pairs: Sequence[tuple[Path, Dataset]],
    truth_root: Path,
    world_cards_root: Path,
    tuned: TunedThresholds,
    detector: TrainedAnomalyDetector,
    xgboost_model: xgboost.XGBClassifier,
    xgboost_threshold: float,
    settings: EvaluationSettings,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """
    WHAT: Scores every test world with rules, anomaly, their union and the XGBoost account model.

    WHY: All four scorers must see the same rows and the same truth. The XGBoost probability replaces
    ScoredDay.anomaly_score, and evaluate_world is then called with the existing "anomaly" scorer at
    the XGBoost cut; this reuses the exact ranking and metric code without adding a parallel metric
    implementation.

    Args:
        test_pairs: (dataset folder, loaded Dataset) pairs of the test split.
        truth_root: Root folder holding one truth folder per run_id.
        world_cards_root: Root folder holding one world card per run_id.
        tuned: Rules config, rules tuning result and detector cut chosen on validation.
        detector: Trained anomaly detector.
        xgboost_model: Fitted XGBoost account model.
        xgboost_threshold: XGBoost cut chosen on validation.
        settings: Capacity, persona weights and world-count policy.

    Returns:
        (per-world results, every test account-day row scored, for the SHAP summary).

    Raises:
        ValueError: If a world repeats, a world card is missing or invalid, or a metric refuses the data.
    """
    per_world: dict[str, dict[str, Any]] = {}
    test_day_rows: list[dict[str, Any]] = []
    repeat_policy = load_escalation_repeat_policy()
    for dataset_dir, dataset in test_pairs:
        payments, snapshots = join_truth(dataset_dir, truth_root)
        payment_rows = payments.to_pylist()
        snapshot_rows = snapshots.to_pylist()
        labels_rows = dataset.labels.to_pylist()
        for world in dataset.manifest["worlds"]:
            world_id = str(world["world_id"])
            run_id = str(world["run_id"])
            if world_id in per_world:
                raise ValueError(f"World {world_id!r} appears in more than one test dataset")

            world_payments = [row for row in payment_rows if row["world_id"] == world_id]
            world_snapshots = [row for row in snapshot_rows if row["world_id"] == world_id]
            day_rows = one_row_per_account_day(world_snapshots)
            test_day_rows.extend(day_rows)
            scored = score_account_days(day_rows, tuned.rules_config, detector)
            probabilities = predict_account_days(xgboost_model, day_rows)
            xgboost_scored = [
                replace(scored_day, anomaly_score=probability)
                for scored_day, probability in zip(scored, probabilities, strict=True)
            ]
            snapshot_times = snapshot_time_by_account_day(day_rows)

            truth_store = load_truth(truth_root, run_id)
            first_credit = first_illicit_credit(world_payments, truth_store.story_of)
            reported = reported_accounts(labels_rows, run_id)
            inflation_factor = _world_inflation_factor(
                read_world_base_rate(world_cards_root, run_id), run_id
            )

            scorers: dict[str, Any] = {}
            for scorer in _LEVEL17_SCORERS:
                scorers[scorer] = evaluate_world(
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
            scorers["xgboost"] = evaluate_world(
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
    return per_world, test_day_rows


def _summarise_pilot(
    per_world: dict[str, dict[str, Any]],
    min_worlds: int,
) -> dict[str, dict[str, dict[str, Any]]]:
    """
    WHAT: Summarises all four scorers across worlds, including XGBoost.

    WHY: run_level_17.summarise_worlds knows only the three Level 17 scorers. Calling it a second
    time with the XGBoost result placed under its "anomaly" key reuses that exact summariser, then
    the result is renamed to "xgboost".

    Args:
        per_world: world_id -> result block carrying all four scorers.
        min_worlds: Smallest world count for which p10 and p90 are reported.

    Returns:
        window -> scorer -> metric -> summary, with xgboost included.
    """
    across_worlds = summarise_worlds(per_world, min_worlds)
    xgboost_only = {
        world_id: {
            "split": world["split"],
            "scorers": {"anomaly": world["scorers"]["xgboost"]},
        }
        for world_id, world in per_world.items()
    }
    xgboost_summaries = summarise_worlds(xgboost_only, min_worlds)
    for window in ("test", "stress"):
        across_worlds[window]["xgboost"] = xgboost_summaries[window]["anomaly"]
    return across_worlds


def _test_metric_median(test_across: dict[str, Any], scorer: str, metric: str) -> float | None:
    """
    WHAT: Returns one scorer's test median for a headline metric, or None when it is not numeric.

    Args:
        test_across: across_worlds["test"] block.
        scorer: scorer name such as "rules" or "xgboost".
        metric: headline metric name such as "recall_at_capacity".

    Returns:
        The median, or None when the summariser recorded no numeric value.
    """
    block = test_across[scorer][metric]
    if "median" not in block:
        return None
    median = block["median"]
    return float(median) if isinstance(median, (int, float)) else None


def _success_criterion(
    validation_scorers: dict[str, Any],
    test_across: dict[str, Any],
    budget_pct: float,
) -> dict[str, Any]:
    """
    WHAT: Applies the committed success rule for the main budget.

    WHY: The brief allows saying the model beats the rules only when both scorers met the validation
    harm budget and XGBoost beats rules on BOTH test median recall at capacity and test median yield.
    A tie never passes, and a missing median makes the comparison unjudgeable rather than a win.

    Args:
        validation_scorers: the main budget's per-scorer validation block from the operating points.
        test_across: across_worlds["test"] on the main budget.
        budget_pct: the committed main genuine-restriction budget the operating points were built for.

    Returns:
        A block with status "not_judgeable", "pass" or "fail", a reason, the committed budget, the
        validation rates and both scorers' test medians and genuine-restriction rates.

    Key logic:
        A validation budget miss by rules or XGBoost, or any missing rules/XGBoost test median for
        recall at capacity or yield, makes the comparison not judgeable and the reason names each
        cause. Otherwise the status is pass only when XGBoost's median recall and median yield both
        strictly exceed the rules values; a tie fails.

    Raises:
        KeyError: If the blocks do not carry the expected scorer keys.
    """
    rules_validation = validation_scorers["rules"]
    xgboost_validation = validation_scorers["xgboost"]
    rules_recall = _test_metric_median(test_across, "rules", "recall_at_capacity")
    xgboost_recall = _test_metric_median(test_across, "xgboost", "recall_at_capacity")
    rules_yield = _test_metric_median(test_across, "rules", "yield_measured")
    xgboost_yield = _test_metric_median(test_across, "xgboost", "yield_measured")
    rules_harm = _test_metric_median(test_across, "rules", "genuine_restriction_rate")
    xgboost_harm = _test_metric_median(test_across, "xgboost", "genuine_restriction_rate")

    problems: list[str] = []
    if not rules_validation["meets_budget"]:
        problems.append(f"validation budget missed by rules at {rules_validation['validation_rate_pct']}%")
    if not xgboost_validation["meets_budget"]:
        problems.append(
            f"validation budget missed by xgboost at {xgboost_validation['validation_rate_pct']}%"
        )
    missing_medians = [
        name
        for name, value in (
            ("rules recall_at_capacity", rules_recall),
            ("xgboost recall_at_capacity", xgboost_recall),
            ("rules yield_measured", rules_yield),
            ("xgboost yield_measured", xgboost_yield),
        )
        if value is None
    ]
    if missing_medians:
        problems.append("missing test median(s): " + ", ".join(missing_medians))

    if problems:
        status = "not_judgeable"
        reason = "; ".join(problems)
    elif (
        xgboost_recall is not None
        and rules_recall is not None
        and xgboost_yield is not None
        and rules_yield is not None
        and xgboost_recall > rules_recall
        and xgboost_yield > rules_yield
    ):
        status = "pass"
        reason = "xgboost median recall and median yield both exceed rules on test"
    else:
        status = "fail"
        reason = "xgboost does not strictly exceed rules on both median recall and median yield on test"

    return {
        "budget_pct": budget_pct,
        "status": status,
        "reason": reason,
        "rules_validation_rate_pct": rules_validation["validation_rate_pct"],
        "xgboost_validation_rate_pct": xgboost_validation["validation_rate_pct"],
        "rules_test_median_recall_at_capacity": rules_recall,
        "xgboost_test_median_recall_at_capacity": xgboost_recall,
        "rules_test_median_yield_measured": rules_yield,
        "xgboost_test_median_yield_measured": xgboost_yield,
        "rules_test_genuine_restriction_rate": rules_harm,
        "xgboost_test_genuine_restriction_rate": xgboost_harm,
    }


def _by_budget_test_results(
    test_pairs: Sequence[tuple[Path, Dataset]],
    truth_root: Path,
    world_cards_root: Path,
    base_rules_config: RulesConfig,
    tuned: TunedThresholds,
    validation_operating_points: dict[str, Any],
    detector: TrainedAnomalyDetector,
    best_model: xgboost.XGBClassifier,
    settings: EvaluationSettings,
) -> dict[str, Any]:
    """
    WHAT: Evaluates every scorer on the test worlds at each committed budget's validation-tuned point.

    WHY: Harm must be matched at the extra budgets too, or a comparison at 0.5 percent alone could
    hide that the model only wins when one side is allowed more harm.

    Args:
        test_pairs: (dataset folder, loaded Dataset) pairs of the test split.
        truth_root: Root folder holding one truth folder per run_id.
        world_cards_root: Root folder holding one world card per run_id.
        base_rules_config: Baseline rules table supplying the committed thresholds.
        tuned: main-budget tuning results, reused for per-rule standalone data.
        validation_operating_points: operating_points block with one entry per budget.
        detector: Trained anomaly detector.
        best_model: Fitted XGBoost account model.
        settings: Capacity, persona weights and world-count policy.

    Returns:
        {"budgets_pct": [...], "test": {budget label: {"budget_pct", "validation", "across_worlds"}}}
        where across_worlds is the same summary shape as the existing main-budget block.

    Raises:
        ValueError: If a world repeats, a world card is missing or invalid, or a metric refuses the data.
    """
    results: dict[str, Any] = {}
    for point in validation_operating_points["points"]:
        budget_pct = point["budget_pct"]
        scorers = point["scorers"]
        rules_at_budget = with_thresholds(
            base_rules_config,
            scorers["rules"]["thresholds"],
            f"pilot validation operating point at {budget_pct}%",
        )
        anomaly_at_budget = AnomalyTuningResult(
            threshold=scorers["anomaly"]["threshold"],
            validation_rate_pct=scorers["anomaly"]["validation_rate_pct"],
            validation_recall=scorers["anomaly"]["validation_recall"],
            meets_budget=scorers["anomaly"]["meets_budget"],
        )
        tuned_at_budget = TunedThresholds(
            rules_config=rules_at_budget,
            rules_result=tuned.rules_result,
            anomaly_result=anomaly_at_budget,
        )
        per_world_at_budget, _ = _evaluate_test_worlds(
            test_pairs,
            truth_root,
            world_cards_root,
            tuned_at_budget,
            detector,
            best_model,
            scorers["xgboost"]["threshold"],
            settings,
        )
        across_worlds_at_budget = _summarise_pilot(per_world_at_budget, settings.min_test_worlds)
        results[f"{budget_pct:g}"] = {
            "budget_pct": budget_pct,
            "validation": scorers,
            "across_worlds": across_worlds_at_budget["test"],
        }
    return {"budgets_pct": list(validation_operating_points["budgets_pct"]), "test": results}


def _truth_leak_check() -> dict[str, Any]:
    """
    WHAT: Proves the model matrix has no truth columns and the model module imports no evaluation code.

    WHY: The account model is operational code; the truth-isolation guard must stay intact, and a
    column such as truth_role would hand the model the answer.

    Returns:
        A check block with pass or fail plus the exact offending names and violations.
    """
    feature_names = account_model_feature_names()
    truth_prefixed = [name for name in feature_names if name.startswith("truth_")]
    order_is_canonical = tuple(feature_names) == account_model_feature_names()
    violations = [
        violation
        for violation in find_truth_violations(REPO_ROOT)
        if violation.startswith("decisions/mule_account_model.py:")
    ]
    passed = not truth_prefixed and order_is_canonical and not violations
    return {
        "status": "pass" if passed else "fail",
        "feature_count": len(feature_names),
        "truth_prefixed_names": truth_prefixed,
        "order_matches_account_model_feature_names": order_is_canonical,
        "module_violations": violations,
    }


def _trivial_world_check(
    training_matrix: Sequence[Sequence[float]],
    training_targets: Sequence[int],
) -> dict[str, Any]:
    """
    WHAT: Measures the best single feature's best-direction AUPRC on the training account-days.

    WHY: If one feature alone separates fraud almost perfectly, the world is too easy and any model
    would look good. The threshold is policy read from config/evaluation.yaml.

    Args:
        training_matrix: Training feature rows in account_model_feature_names() order.
        training_targets: Binary bank-visible targets aligned with the rows.

    Returns:
        A check block with pass or fail, the top three features and the configured threshold.
    """
    config = load_evaluation_config()
    threshold = float(config["trivial_world_check"]["fail_threshold"])
    feature_names = account_model_feature_names()
    scores: list[tuple[str, float]] = []
    for index, name in enumerate(feature_names):
        values = [float(row[index]) for row in training_matrix]
        scores.append((name, best_direction_ap(values, training_targets)))
    ranked = sorted(scores, key=lambda item: item[1], reverse=True)
    top_three = [
        {"feature": name, "best_direction_ap": value} for name, value in ranked[:3]
    ]
    best_feature, best_ap = ranked[0]
    return {
        "status": "fail" if best_ap >= threshold else "pass",
        "best_feature": best_feature,
        "best_direction_ap": best_ap,
        "fail_threshold": threshold,
        "top_3": top_three,
    }


def _write_shap(
    path: Path,
    model: xgboost.XGBClassifier | PUBaggedAccountModel,
    feature_names: Sequence[str],
    day_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """
    WHAT: Writes the mean |SHAP| top ten features on the test account-days.

    WHY: The pilot needs to show which features drive the account model, without plots. A SHAP
    runtime failure is recorded verbatim and does not stop the evaluation, as the architect ruled.
    A PU bagging ensemble explains itself with the mean of its members' margin-space SHAP values.

    Args:
        path: Destination shap_top_features.json.
        model: Fitted XGBoost account model or PUBaggedAccountModel ensemble.
        feature_names: Canonical model feature order.
        day_rows: Test account-day rows that were scored.

    Returns:
        The written payload.

    Errors:
        A SHAP/runtime exception is caught and written as {"status": "not run", "reason": "..."}.
    """
    payload: dict[str, Any]
    try:
        matrix = _feature_matrix(day_rows, feature_names)
        if isinstance(model, PUBaggedAccountModel):
            values = model.shap_values(matrix)
        else:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(matrix)
            if isinstance(shap_values, list):
                values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            else:
                values = shap_values
        array = numpy.asarray(values)
        if array.ndim == 3:
            array = array[:, :, 1]
        mean_abs = numpy.mean(numpy.abs(array), axis=0)
        ranked = sorted(
            ((name, float(value)) for name, value in zip(feature_names, mean_abs, strict=True)),
            key=lambda item: item[1],
            reverse=True,
        )
        payload = {
            "simulated": True,
            "pilot": True,
            "status": "run",
            "top_features": [
                {"feature": name, "mean_abs_shap": value} for name, value in ranked[:10]
            ],
        }
    except Exception as exc:
        payload = {
            "simulated": True,
            "pilot": True,
            "status": "not run",
            "reason": f"{type(exc).__name__}: {exc}",
        }
    path.write_bytes(orjson.dumps(payload, option=orjson.OPT_SORT_KEYS | orjson.OPT_INDENT_2) + b"\n")
    return payload


def run_pilot(
    training_dirs: Sequence[Path],
    validation_dirs: Sequence[Path],
    test_dirs: Sequence[Path],
    truth_root: Path,
    world_cards_root: Path,
    out_root: Path,
) -> int:
    """
    WHAT: Runs the complete Level 18 pilot and writes its three output files.

    WHY: One entry point fixes the order: load and check splits, train on training only, choose every
    cut on validation, then score test once. Nothing can reach test before the choices are frozen.

    Args:
        training_dirs: Training dataset folders.
        validation_dirs: Validation dataset folders.
        test_dirs: Test dataset folders.
        truth_root: Root folder holding one truth folder per run_id.
        world_cards_root: Root folder holding one world card per run_id.
        out_root: Root folder for the pilot result.

    Returns:
        Process exit code: 0 when every run leakage check passes, 1 otherwise.

    Key logic:
        The mule model config's rules_plus_labels_and_registry switch decides the rules table:
        true loads rules 1 to 6 and the merged tuning grid, false keeps rules 1 to 5 exactly as
        before. The manifest records which variant ran and the sha256 of each rules config file.

    Raises:
        ValueError: If a split is mismatched, a split leaks, a world card or truth manifest is
            missing, or any input is malformed.
    """
    settings = load_evaluation_settings()
    mule_model_config = load_mule_model_config()
    rules_config_paths: tuple[str, ...]
    if mule_model_config.rules_plus_labels_and_registry is True:
        base_rules_config = load_rules_config_with_labels()
        tuning = load_labels_tuning_config()
        rules_variant = "rules_1_to_6"
        rules_config_paths = (
            "config/rules_baseline.yaml",
            "config/rules_labels_extension.yaml",
        )
    else:
        tuning = load_tuning_config()
        base_rules_config = load_rules_config()
        rules_variant = "rules_1_to_5"
        rules_config_paths = ("config/rules_baseline.yaml",)
    anomaly_config = load_anomaly_detector_config()

    training_datasets = load_split_datasets("training", training_dirs)
    validation_datasets = load_split_datasets("validation", validation_dirs)
    test_datasets = load_split_datasets("test", test_dirs)

    dataset_entries = [
        _dataset_entry(dataset_dir, dataset)
        for dataset_dir, dataset in [
            *zip(training_dirs, training_datasets, strict=True),
            *zip(validation_dirs, validation_datasets, strict=True),
            *zip(test_dirs, test_datasets, strict=True),
        ]
    ]
    key = run_key([str(entry["dataset_id"]) for entry in dataset_entries])
    result_dir = out_root / key
    leakage_path = result_dir / "leakage_checks.json"

    split_leak = _split_leak(training_datasets, validation_datasets, test_datasets)
    if split_leak["status"] == "fail":
        checks = {"split_leak": split_leak, **_not_run_checks("split leak")}
        _write_leakage(leakage_path, checks)
        print("LEAKAGE CHECK FAILED")
        return 1

    training_snapshot_rows = [
        row for dataset in training_datasets for row in dataset.snapshots.to_pylist()
    ]
    training_label_rows = [
        row for dataset in training_datasets for row in dataset.labels.to_pylist()
    ]
    training_matrix, training_targets, _ = build_training_rows(
        training_snapshot_rows, training_label_rows
    )
    training_positive_count = sum(training_targets)

    seed = anomaly_config.random_state
    validation_snapshot_rows: list[dict[str, Any]] = []
    validation_label_rows: list[dict[str, Any]] = []
    for validation_dir, dataset in zip(validation_dirs, validation_datasets, strict=True):
        _, snapshots = join_truth(validation_dir, truth_root)
        validation_snapshot_rows.extend(snapshots.to_pylist())
        validation_label_rows.extend(dataset.labels.to_pylist())
    validation_matrix, validation_targets, validation_keys = build_training_rows(
        validation_snapshot_rows, validation_label_rows
    )
    validation_day_rows = _day_rows_by_world(validation_snapshot_rows)
    validation_rows_for_keys = _align_rows(validation_day_rows, validation_keys)

    n_estimators_scores: dict[str, float] = {}
    best_model: xgboost.XGBClassifier | None = None
    best_scale_pos_weight = 0.0
    best_n_estimators = 0
    best_validation_ap = -1.0
    for n_estimators in N_ESTIMATOR_CANDIDATES:
        model, scale_pos_weight = train_account_model(
            training_matrix, training_targets, seed, n_estimators=n_estimators
        )
        probabilities = predict_account_days(model, validation_rows_for_keys)
        score = average_precision(probabilities, validation_targets)
        n_estimators_scores[str(n_estimators)] = score
        if score > best_validation_ap:
            best_validation_ap = score
            best_model = model
            best_scale_pos_weight = scale_pos_weight
            best_n_estimators = n_estimators
    if best_model is None:
        raise ValueError("No n_estimators candidate produced a validation score")

    world_starts = world_start_by_world(training_snapshot_rows)
    detector = train_anomaly_detector(
        training_snapshot_rows, "training", world_starts, anomaly_config
    )
    tuned, validation_scored, validation_scored_rows = _tune_on_validation(
        validation_dirs,
        truth_root,
        base_rules_config,
        tuning,
        detector,
        settings.genuine_restriction_budget_pct,
        collect_grid_points=True,
    )
    validation_probabilities = predict_account_days(best_model, validation_scored_rows)
    validation_xgboost_scored = [
        replace(scored_day, anomaly_score=probability)
        for scored_day, probability in zip(validation_scored, validation_probabilities, strict=True)
    ]
    xgboost_tuning = tune_anomaly_threshold(
        validation_xgboost_scored,
        tuning,
        "validation",
        settings.genuine_restriction_budget_pct,
    )
    validation_operating_points = operating_points(
        rules_grid_points=tuned.rules_result.grid_points,
        anomaly_scored=validation_scored,
        xgboost_scored=validation_xgboost_scored,
        run_manifest_id=key,
    )

    test_pairs = list(zip(test_dirs, test_datasets, strict=True))
    per_world, test_day_rows = _evaluate_test_worlds(
        test_pairs,
        truth_root,
        world_cards_root,
        tuned,
        detector,
        best_model,
        xgboost_tuning.threshold,
        settings,
    )
    across_worlds = _summarise_pilot(per_world, settings.min_test_worlds)
    by_budget = _by_budget_test_results(
        test_pairs,
        truth_root,
        world_cards_root,
        base_rules_config,
        tuned,
        validation_operating_points,
        detector,
        best_model,
        settings,
    )
    main_budget_pct = validation_operating_points["budgets_pct"][0]
    main_budget_label = f"{main_budget_pct:g}"
    success_criterion = _success_criterion(
        validation_operating_points["points"][0]["scorers"],
        by_budget["test"][main_budget_label]["across_worlds"],
        main_budget_pct,
    )

    model_path = result_dir / "account_model.ubj"
    save_model(
        best_model,
        model_path,
        {
            "seed": seed,
            "scale_pos_weight": best_scale_pos_weight,
            "training_dataset_ids": sorted(directory.name for directory in training_dirs),
        },
    )
    sidecar_path = Path(str(model_path) + ".json")

    run_ids = sorted({str(world["run_id"]) for world in per_world.values()})
    truth_manifests = {
        run_id: _sha256_of_file(truth_root / run_id / "manifest.json") for run_id in run_ids
    }
    manifest = build_result_manifest(
        dataset_entries,
        truth_manifests,
        detector,
        anomaly_config.random_state,
        tuned,
        settings,
    )
    manifest["versions"] = {
        "xgboost": xgboost.__version__,
        "shap": shap.__version__,
        "numpy": numpy.__version__,
    }
    repo_root = Path(__file__).resolve().parents[1]
    config_path = active_mule_model_config_path().resolve()
    if config_path.is_relative_to(repo_root):
        recorded_config_path = str(config_path.relative_to(repo_root))
    else:
        recorded_config_path = str(config_path)
    committed_config_path = (repo_root / "config" / "mule_model.yaml").resolve()
    manifest["mule_model_config"] = {
        "path": recorded_config_path,
        "sha256": _sha256_of_file(config_path),
        "overridden": config_path != committed_config_path,
        "use_label_network_features": mule_model_config.use_label_network_features,
        "rules_plus_labels_and_registry": mule_model_config.rules_plus_labels_and_registry,
        "label_cut": mule_model_config.label_cut,
        "label_maturity_days": mule_model_config.label_maturity_days,
    }
    rules_config_files: list[dict[str, str]] = []
    for relative_path in rules_config_paths:
        rules_config_files.append(
            {"path": relative_path, "sha256": _sha256_of_file(REPO_ROOT / relative_path)}
        )
    manifest["rules_variant"] = rules_variant
    manifest["rules_config_files"] = rules_config_files
    manifest["pilot_account_model"] = {
        "n_estimators_scores": n_estimators_scores,
        "chosen_n_estimators": best_n_estimators,
        "chosen_validation_average_precision": best_validation_ap,
        "scale_pos_weight": best_scale_pos_weight,
        "training_row_count": len(training_matrix),
        "training_positive_count": training_positive_count,
        "seed": seed,
        "feature_count": len(account_model_feature_names()),
        "model_path": str(model_path),
        "model_sha256": _sha256_of_file(model_path),
        "model_sidecar_path": str(sidecar_path),
        "model_sidecar_sha256": _sha256_of_file(sidecar_path),
        "xgboost_threshold": xgboost_tuning.threshold,
        "xgboost_validation_rate_pct": xgboost_tuning.validation_rate_pct,
        "xgboost_validation_recall": xgboost_tuning.validation_recall,
        "xgboost_meets_budget": xgboost_tuning.meets_budget,
    }
    manifest["per_rule_standalone"] = [
        {
            "reason_code": rule.reason_code,
            "validation_rate_pct": rule.validation_rate_pct,
            "validation_recall": rule.validation_recall,
        }
        for rule in tuned.rules_result.per_rule_standalone
    ]
    manifest["validation_operating_points"] = validation_operating_points
    manifest["by_budget"] = by_budget
    manifest["success_criterion"] = success_criterion
    validation_curves = {
        "simulated": True,
        "run_manifest_id": key,
        "score_curves": {
            "anomaly": score_curve(validation_scored),
            "xgboost": score_curve(validation_xgboost_scored),
        },
        "rules_curve": rules_curve(tuned.rules_result.grid_points),
    }

    result_path = write_result(out_root, key, manifest, per_world, across_worlds)
    curves_path = result_dir / "validation_curves.json"
    curves_path.write_bytes(
        orjson.dumps(validation_curves, option=orjson.OPT_SORT_KEYS | orjson.OPT_INDENT_2) + b"\n"
    )
    payload = orjson.loads(result_path.read_bytes())
    payload["pilot"] = True
    result_path.write_bytes(
        orjson.dumps(payload, option=orjson.OPT_SORT_KEYS | orjson.OPT_INDENT_2) + b"\n"
    )

    truth_leak = _truth_leak_check()
    trivial_world = _trivial_world_check(training_matrix, training_targets)
    if COMPLAINED_FEATURE in account_model_feature_names():
        check_7_training = check_label_features_by_arrival(training_snapshot_rows, training_label_rows)
        check_7_validation = check_label_features_by_arrival(validation_snapshot_rows, validation_label_rows)
        check_7_status = "fail" if "fail" in {check_7_training["status"], check_7_validation["status"]} else "pass"
        check_7 = {
            "status": check_7_status,
            "training": check_7_training,
            "validation": check_7_validation,
        }
    else:
        check_7 = {"status": "not run", "reason": "label network features are off"}
    checks = {
        "split_leak": split_leak,
        "truth_leak": truth_leak,
        "trivial_world": trivial_world,
        "section_23_checks_4_to_7": {
            "status": "not run",
            "reason": "later round",
            "checks": [4, 5, 6],
        },
        "section_23_check_7": check_7,
    }
    _write_leakage(leakage_path, checks)

    shap_payload = _write_shap(
        result_dir / "shap_top_features.json",
        best_model,
        account_model_feature_names(),
        test_day_rows,
    )
    if shap_payload["status"] != "run":
        print(f"SHAP not run: {shap_payload['reason']}")

    failed = [name for name, check in checks.items() if check["status"] == "fail"]
    if failed:
        print("LEAKAGE CHECK FAILED")
        return 1
    print(str(result_path))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """
    WHAT: Command-line entry point for the Level 18 pilot.

    Args:
        argv: Optional argument list; defaults to sys.argv.

    Returns:
        The runner's process exit code.
    """
    parser = argparse.ArgumentParser(description="Run the Level 18 pilot and write its result files")
    parser.add_argument(
        "--training", type=Path, action="append", required=True, help="training dataset folder; repeatable"
    )
    parser.add_argument(
        "--validation", type=Path, action="append", required=True, help="validation dataset folder; repeatable"
    )
    parser.add_argument(
        "--test", type=Path, action="append", required=True, help="test dataset folder; repeatable"
    )
    parser.add_argument(
        "--truth-root", type=Path, required=True, help="root folder holding one truth folder per run_id"
    )
    parser.add_argument(
        "--world-cards-root",
        type=Path,
        required=True,
        help="root folder holding one world card per run_id",
    )
    parser.add_argument(
        "--out-root", type=Path, required=True, help="root folder for the pilot result"
    )
    parser.add_argument(
        "--mule-model-config",
        type=Path,
        default=None,
        help="model config to use instead of the committed config/mule_model.yaml",
    )
    args = parser.parse_args(argv)
    if args.mule_model_config is not None:
        set_mule_model_config_path(args.mule_model_config)
    try:
        return run_pilot(
            tuple(args.training),
            tuple(args.validation),
            tuple(args.test),
            args.truth_root,
            args.world_cards_root,
            args.out_root,
        )
    finally:
        set_mule_model_config_path(None)


if __name__ == "__main__":
    raise SystemExit(main())
