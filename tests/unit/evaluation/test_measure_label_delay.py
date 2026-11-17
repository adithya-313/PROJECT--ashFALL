"""
WHAT: Unit tests for the Level 18a4 label-delay measurement: first-credit delays, the first label
per account, censored mules, the empty case, the pooled fixture report and the CLI.

WHY: The maturity window is only as trustworthy as the delay it is based on, so the measured days
must follow the evaluation's first-credit clock and never drop an unlabelled mule silently.
"""

from datetime import datetime, timezone
from pathlib import Path

import orjson
import pytest

from evaluation.measure_label_delay import (
    build_label_delay_report,
    main,
    measure_label_delay,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES = REPO_ROOT / "showcase" / "fixtures" / "level_17"
TRAINING = FIXTURES / "datasets" / "training-20261031T1830-624c2ad40f4c"
TRUTH_ROOT = FIXTURES / "truth"


def _moment(day: int, hour: int = 0) -> datetime:
    """Returns a UTC moment in September 2026."""
    return datetime(2026, 9, day, hour, 0, tzinfo=timezone.utc)


def _payment(payment_id: str, receiver: str, moment: datetime, role: str = "mule") -> dict[str, object]:
    """Builds one joined payment row as first_illicit_credit expects it."""
    return {
        "payment_id": payment_id,
        "receiver_account_id": receiver,
        "decision_time": moment,
        "truth_role_receiver": role,
    }


def _story_of(_payment_id: str) -> dict[str, object]:
    """Marks every payment as fraud-origin."""
    return {"origin": "fraud"}


def test_label_delay_counts_censored_mules_and_percentiles() -> None:
    """Two labelled mules give their delay in days; the unlabelled mule is counted as censored."""
    payments = [
        _payment("p1", "acc_a", _moment(1)),
        _payment("p2", "acc_b", _moment(1)),
        _payment("p3", "acc_c", _moment(1)),
        _payment("p4", "acc_d", _moment(1), role="genuine"),
    ]
    labels = [
        {"account_id": "acc_a", "arrival_time": _moment(2)},
        {"account_id": "acc_b", "arrival_time": _moment(4)},
    ]

    report = measure_label_delay(payments, labels, _story_of)

    assert report["accounts_with_first_credit"] == 3
    assert report["labelled_accounts"] == 2
    assert report["delays_days"] == [1.0, 3.0]
    assert report["censored_count"] == 1
    assert report["censored_accounts"] == ["acc_c"]
    assert report["delay_percentiles_days"] == pytest.approx({"p50": 2.0, "p90": 2.8, "p95": 2.9})
    assert report["min_delay_days"] == 1.0
    assert report["max_delay_days"] == 3.0


def test_label_delay_uses_the_earliest_label_and_allows_zero() -> None:
    """An account labelled at the credit moment keeps a zero delay; a later label does not replace it."""
    payments = [_payment("p1", "acc_a", _moment(1))]
    labels = [
        {"account_id": "acc_a", "arrival_time": _moment(6)},
        {"account_id": "acc_a", "arrival_time": _moment(1)},
    ]

    report = measure_label_delay(payments, labels, _story_of)

    assert report["delays_days"] == [0.0]
    assert report["delay_percentiles_days"] == pytest.approx({"p50": 0.0, "p90": 0.0, "p95": 0.0})


def test_label_delay_with_no_mule_credit_is_empty_not_an_error() -> None:
    """A world with no fraud-origin credit reports zeros instead of raising."""
    payments = [_payment("p1", "acc_a", _moment(1), role="genuine")]
    labels = [{"account_id": "acc_a", "arrival_time": _moment(2)}]

    report = measure_label_delay(payments, labels, _story_of)

    assert report["accounts_with_first_credit"] == 0
    assert report["delays_days"] == []
    assert report["delay_percentiles_days"] == {}
    assert report["min_delay_days"] is None
    assert report["max_delay_days"] is None


def test_build_label_delay_report_reads_every_training_world() -> None:
    """The fixture training dataset yields one block per world and pooled percentiles."""
    dataset = orjson.loads((TRAINING / "manifest.json").read_bytes())

    report = build_label_delay_report((TRAINING,), TRUTH_ROOT)

    assert report["training_datasets"] == [TRAINING.name]
    assert [block["world_id"] for block in report["by_world"]] == [
        str(world["world_id"]) for world in dataset["worlds"]
    ]
    assert report["pooled"]["accounts_with_first_credit"] > 0
    assert report["pooled"]["labelled_accounts"] > 0
    assert set(report["pooled"]["delay_percentiles_days"]) == {"p50", "p90", "p95"}


def test_cli_writes_the_report(tmp_path: Path) -> None:
    """The CLI writes the same report structure as build_label_delay_report."""
    out = tmp_path / "label_delay.json"

    exit_code = main(["--training", str(TRAINING), "--truth-root", str(TRUTH_ROOT), "--out", str(out)])
    payload = orjson.loads(out.read_bytes())

    assert exit_code == 0
    assert set(payload) == {"by_world", "pooled", "training_datasets"}
    assert payload["pooled"]["accounts_with_first_credit"] > 0
