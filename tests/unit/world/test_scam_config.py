"""
Tests for the scam schedule resolvers in world/scam_config.py. Each resolver is checked against the committed
files and against a narrowed copy of the config, which proves the value is read from config and not typed in code.
"""

from pathlib import Path
from typing import Any, Callable

import pytest
import yaml

from world.scam_config import (
    SCAM_SCHEDULES_PATH,
    WORLD_RANGES_PATH,
    load_scam_schedules,
    resolve_debit_amount_inr,
    resolve_debit_payment_count,
    resolve_debit_pays_back,
    resolve_debit_spread_minutes,
    resolve_first_payment_amount_inr,
    resolve_investment_or_job_share,
    resolve_investment_share_of_investment_or_job,
    resolve_min_minutes_between_victim_payments,
    resolve_payment_escalation_multiple,
    resolve_payment_spread_days,
    resolve_payments_per_victim,
    resolve_payout_back_share,
    resolve_remainder_scam_type,
)


def _write_schedules(tmp_path: Path, edit: Callable[[dict[str, Any]], None]) -> Path:
    """Writes a copy of the committed schedule file after applying edit to its parsed contents."""
    data = yaml.safe_load(SCAM_SCHEDULES_PATH.read_text(encoding="utf-8"))
    edit(data)
    path = tmp_path / "scam_schedules.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _write_world_ranges(tmp_path: Path, parameter: str, split: str, bounds: list[float]) -> Path:
    """Writes a copy of the committed world ranges with one parameter's split narrowed to bounds."""
    data = yaml.safe_load(WORLD_RANGES_PATH.read_text(encoding="utf-8"))
    data["parameters"][parameter][split] = bounds
    path = tmp_path / "world_ranges.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_resolvers_return_the_committed_training_values() -> None:
    """Checks every resolver against the values the committed files hold for the training split."""
    assert resolve_investment_or_job_share("training") == (0.55, 0.8)
    assert resolve_investment_share_of_investment_or_job("training") == (0.4, 0.6)
    assert resolve_remainder_scam_type() == "unauthorised_debit"
    assert resolve_payments_per_victim("training") == (1, 6)
    assert resolve_payment_spread_days("training") == (0, 10)
    assert resolve_first_payment_amount_inr("training") == (2000.0, 50000.0)
    assert resolve_payment_escalation_multiple("training") == (1.0, 2.0)
    assert resolve_payout_back_share("training") == (0.05, 0.3)
    assert resolve_min_minutes_between_victim_payments() == 2
    assert resolve_debit_payment_count("training") == (1, 3)
    assert resolve_debit_spread_minutes("training") == (0, 60)
    assert resolve_debit_amount_inr("training") == (2000.0, 50000.0)
    assert resolve_debit_pays_back() is False


def test_stress_split_reads_the_stress_values() -> None:
    """Checks that the split argument selects a different range, so no resolver is fixed to training."""
    assert resolve_investment_share_of_investment_or_job("stress") == (0.1, 0.9)
    assert resolve_payment_escalation_multiple("stress") == (1.0, 4.0)
    assert resolve_debit_payment_count("stress") == (1, 10)
    assert resolve_debit_spread_minutes("stress") == (0, 240)


@pytest.mark.parametrize(
    ("section", "setting", "resolver"),
    [
        ("scam_type_shares", "investment_share_of_investment_or_job", resolve_investment_share_of_investment_or_job),
        ("investment_and_job_schedule", "payment_escalation_multiple", resolve_payment_escalation_multiple),
        ("unauthorised_debit_schedule", "debit_payments", resolve_debit_payment_count),
        ("unauthorised_debit_schedule", "debit_spread_minutes", resolve_debit_spread_minutes),
    ],
)
def test_own_ranges_follow_a_narrowed_schedule_file(
    tmp_path: Path, section: str, setting: str, resolver: Callable[..., tuple[float, float]]
) -> None:
    """Narrows each range stored in the schedule file and checks the resolver returns the narrowed values."""
    path = _write_schedules(tmp_path, lambda data: data[section][setting].update({"training": [2, 2]}))

    assert resolver("training", path) == (2, 2)


@pytest.mark.parametrize(
    ("section", "setting", "parameter", "resolver"),
    [
        (
            "scam_type_shares", "investment_or_job_share", "share_of_scams_investment_or_job",
            resolve_investment_or_job_share,
        ),
        ("investment_and_job_schedule", "payments_per_victim", "scam_payments_per_victim", resolve_payments_per_victim),
        ("investment_and_job_schedule", "payment_spread_days", "scam_payment_spread_days", resolve_payment_spread_days),
        (
            "investment_and_job_schedule", "first_payment_amount_inr", "mule_transfer_amount_inr",
            resolve_first_payment_amount_inr,
        ),
        ("investment_and_job_schedule", "payout_back_share", "scam_small_payout_back_share", resolve_payout_back_share),
        ("unauthorised_debit_schedule", "debit_amount_inr", "mule_transfer_amount_inr", resolve_debit_amount_inr),
    ],
)
def test_referenced_ranges_follow_a_narrowed_world_ranges_file(
    tmp_path: Path, section: str, setting: str, parameter: str, resolver: Callable[..., tuple[float, float]]
) -> None:
    """Narrows the world_ranges parameter a setting names and checks the resolver returns the narrowed values."""
    ranges_path = _write_world_ranges(tmp_path, parameter, "training", [3, 4])

    assert resolver("training", None, ranges_path) == (3, 4)


def test_reference_name_is_read_from_the_schedule_file(tmp_path: Path) -> None:
    """Points a setting at a different world_ranges parameter and checks the resolver follows the new name."""
    path = _write_schedules(
        tmp_path,
        lambda data: data["investment_and_job_schedule"]["payments_per_victim"].update(
            {"world_ranges_reference": "scam_payment_spread_days"}
        ),
    )

    assert resolve_payments_per_victim("training", path) == resolve_payment_spread_days("training")


def _narrow_single_values(data: dict[str, Any]) -> None:
    """Changes the three single-value settings of a parsed schedule file to values different from the committed ones."""
    data["scam_type_shares"]["remainder_type"]["value"] = "other_type"
    data["investment_and_job_schedule"]["min_minutes_between_victim_payments"]["value"] = 5
    data["unauthorised_debit_schedule"]["payout_back"]["value"] = True


def test_single_values_follow_a_narrowed_schedule_file(tmp_path: Path) -> None:
    """Narrows the three single-value settings and checks each resolver returns the narrowed value."""
    path = _write_schedules(tmp_path, _narrow_single_values)

    assert resolve_remainder_scam_type(path) == "other_type"
    assert resolve_min_minutes_between_victim_payments(path) == 5
    assert resolve_debit_pays_back(path) is True


def test_missing_schedule_file_is_refused(tmp_path: Path) -> None:
    """Checks that a path with no file raises FileNotFoundError instead of falling back to the committed file."""
    with pytest.raises(FileNotFoundError, match="Scam schedule config not found"):
        load_scam_schedules(tmp_path / "missing.yaml")


def test_missing_split_is_refused(tmp_path: Path) -> None:
    """Checks that a split absent from a setting raises a sentence naming the setting and the split."""
    path = _write_schedules(tmp_path, lambda data: data["unauthorised_debit_schedule"]["debit_payments"].pop("test"))

    with pytest.raises(ValueError, match="'debit_payments' has no range for split 'test'"):
        resolve_debit_payment_count("test", path)


def test_missing_setting_and_section_are_refused(tmp_path: Path) -> None:
    """Checks that a removed setting and a removed section each raise a plain sentence."""
    path = _write_schedules(tmp_path, lambda data: data["unauthorised_debit_schedule"].pop("debit_payments"))
    with pytest.raises(ValueError, match="has no setting 'debit_payments'"):
        resolve_debit_payment_count("training", path)

    path = _write_schedules(tmp_path, lambda data: data.pop("unauthorised_debit_schedule"))
    with pytest.raises(ValueError, match="has no section 'unauthorised_debit_schedule'"):
        resolve_debit_payment_count("training", path)


@pytest.mark.parametrize(
    ("bounds", "message"),
    [
        ([1], "must be a list of two numbers"),
        ([3, 1], "lower 3 above upper 1"),
        (["a", 2], "must hold numbers"),
        ([1.5, 3], "must hold whole numbers"),
    ],
)
def test_malformed_ranges_are_refused(tmp_path: Path, bounds: list[Any], message: str) -> None:
    """Checks that a range with the wrong length, order, type or fractions raises a plain sentence."""
    path = _write_schedules(
        tmp_path, lambda data: data["unauthorised_debit_schedule"]["debit_payments"].update({"training": bounds})
    )

    with pytest.raises(ValueError, match=message):
        resolve_debit_payment_count("training", path)


def test_escalation_below_one_is_refused(tmp_path: Path) -> None:
    """Checks that an escalation range that lets payments shrink is refused."""
    path = _write_schedules(
        tmp_path,
        lambda data: data["investment_and_job_schedule"]["payment_escalation_multiple"].update(
            {"training": [0.5, 2.0]}
        ),
    )

    with pytest.raises(ValueError, match="must not go below 1.0"):
        resolve_payment_escalation_multiple("training", path)


def test_invalid_single_values_are_refused(tmp_path: Path) -> None:
    """Checks that a too-small gap, a non-boolean payout flag and an empty type name are refused."""
    path = _write_schedules(
        tmp_path,
        lambda data: data["investment_and_job_schedule"]["min_minutes_between_victim_payments"].update({"value": 1}),
    )
    with pytest.raises(ValueError, match="at least 2"):
        resolve_min_minutes_between_victim_payments(path)

    path = _write_schedules(
        tmp_path, lambda data: data["unauthorised_debit_schedule"]["payout_back"].update({"value": "no"})
    )
    with pytest.raises(ValueError, match="must be true or false"):
        resolve_debit_pays_back(path)

    path = _write_schedules(tmp_path, lambda data: data["scam_type_shares"]["remainder_type"].update({"value": ""}))
    with pytest.raises(ValueError, match="non-empty string"):
        resolve_remainder_scam_type(path)


def test_setting_without_reference_and_unknown_reference_are_refused(tmp_path: Path) -> None:
    """Checks that a referenced setting with no reference, or a reference to a missing parameter, is refused."""
    path = _write_schedules(
        tmp_path, lambda data: data["investment_and_job_schedule"]["payments_per_victim"].pop("world_ranges_reference")
    )
    with pytest.raises(ValueError, match="must name a world_ranges_reference"):
        resolve_payments_per_victim("training", path)

    path = _write_schedules(
        tmp_path,
        lambda data: data["investment_and_job_schedule"]["payments_per_victim"].update(
            {"world_ranges_reference": "nope"}
        ),
    )
    with pytest.raises(ValueError, match="unknown world_ranges parameter 'nope'"):
        resolve_payments_per_victim("training", path)


def test_missing_world_ranges_file_is_refused(tmp_path: Path) -> None:
    """Checks that a world ranges path with no file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="World ranges config not found"):
        resolve_payments_per_victim("training", None, tmp_path / "missing.yaml")
