"""
tests/unit/world/test_phone_book.py

Unit tests for synthetic phone directory generation, telecom number recycling,
and shared-household telephone assignments (world/phone_book.py).
"""

import hashlib
from typing import Any

import pytest

from showcase.level_12_support_b import get_showcase_b_settings
from showcase.level_12_support_c import load_showcase_c_fixture
from world.gang_config import GangConfig
from world.interactive_records import InteractiveWorldOutput, RunnerSettings
from world.interactive_runner import run_interactive_world
from world.phone_book import MINUTES_PER_DAY, build_phone_book


def _run_test_world_fraud_on(gang: GangConfig) -> InteractiveWorldOutput:
    """Helper to run a small fraud-on simulation world with the given gang."""
    base_settings, code_version = get_showcase_b_settings()
    return run_interactive_world(
        base_settings,
        RunnerSettings(fraud_enabled=True),
        gang_configs=(gang,),
        code_version=code_version,
    )


@pytest.fixture(scope="module")
def fraud_on_world() -> InteractiveWorldOutput:
    """Runs a fraud-on world once using the Level 12 Part C cash-out fixture."""
    return _run_test_world_fraud_on(load_showcase_c_fixture())


@pytest.fixture(scope="module")
def fraud_off_world() -> InteractiveWorldOutput:
    """Runs a standard fraud-off world using Level 11 baseline settings."""
    base_settings, code_version = get_showcase_b_settings()
    return run_interactive_world(
        base_settings,
        RunnerSettings(fraud_enabled=False),
        code_version=code_version,
    )


def test_same_seed_gives_same_phone_book(fraud_on_world: InteractiveWorldOutput) -> None:
    """The same run seed produces an identical phone book across repeated invocations.

    Recreates a situation where an evaluation harness or simulation replay generates
    phone intelligence multiple times. Catches a bug where unseeded random generators or
    non-deterministic dictionary ordering cause phone numbers or recycling assignments
    to drift between identical runs.
    """
    seed = 101
    book_1 = build_phone_book(fraud_on_world, run_seed=seed)
    book_2 = build_phone_book(fraud_on_world, run_seed=seed)

    assert book_1.run_seed == book_2.run_seed == seed
    assert book_1.recycled_count == book_2.recycled_count
    assert book_1.shared_count == book_2.shared_count
    assert set(book_1.records.keys()) == set(book_2.records.keys())

    for acc_id, rec_1 in book_1.records.items():
        rec_2 = book_2.records[acc_id]
        assert rec_1 == rec_2


def test_recycled_number_deactivation_is_at_least_90_days_before_reassignment(
    fraud_on_world: InteractiveWorldOutput,
) -> None:
    """Recycled numbers enforce the statutory minimum 90-day quarantine period.

    Recreates the regulatory requirement from the Supreme Court of India judgment (30 Oct 2023)
    mandating that telecom providers hold deactivated numbers for at least 90 days before
    reallocating them. Catches a bug where recycled numbers are reassigned prematurely or
    where timestamps fail to reflect the required pre-world separation.
    """
    phone_config: dict[str, Any] = {
        "recycle_min_days": 90,
        "recycled_number_share": 0.2,
    }
    book = build_phone_book(fraud_on_world, run_seed=101, phone_config=phone_config)
    assert book.recycled_count > 0

    ninety_days_minutes = 90 * MINUTES_PER_DAY

    for rec in book.records.values():
        if not rec.is_recycled:
            assert rec.previous_holder_deactivated_minute is None
            assert rec.reassigned_minute is None
            continue

        assert rec.previous_holder_deactivated_minute is not None
        assert rec.reassigned_minute is not None

        # Pre-world history must occur before day 0 (minute 0)
        assert rec.previous_holder_deactivated_minute < 0
        assert rec.reassigned_minute <= 0

        # Statutory delay: reassignment must be at least 90 days (129,600 minutes) after deactivation
        quarantine_minutes = rec.reassigned_minute - rec.previous_holder_deactivated_minute
        assert quarantine_minutes >= ninety_days_minutes


def test_fraud_off_output_is_unaffected_by_phone_book(
    fraud_off_world: InteractiveWorldOutput,
) -> None:
    """Building the phone book does not alter finished world output, events, or ledgers.

    Recreates post-simulation intelligence ingestion on a pristine fraud-off baseline run.
    Catches a bug where the phone book builder accidentally mutates the world state,
    advances core RNG children, or touches frozen anchor event logs.
    """
    events_count_before = len(fraud_off_world.result.run_result.events)
    events_repr_before = repr(fraud_off_world.result.run_result.events)
    ledgers_repr_before = repr(fraud_off_world.world.ledgers)

    book = build_phone_book(fraud_off_world, run_seed=101)
    assert len(book.records) > 0

    events_count_after = len(fraud_off_world.result.run_result.events)
    events_repr_after = repr(fraud_off_world.result.run_result.events)
    ledgers_repr_after = repr(fraud_off_world.world.ledgers)

    assert events_count_before == events_count_after
    hash_events_before = hashlib.sha256(events_repr_before.encode()).hexdigest()
    hash_events_after = hashlib.sha256(events_repr_after.encode()).hexdigest()
    assert hash_events_before == hash_events_after

    hash_ledgers_before = hashlib.sha256(ledgers_repr_before.encode()).hexdigest()
    hash_ledgers_after = hashlib.sha256(ledgers_repr_after.encode()).hexdigest()
    assert hash_ledgers_before == hash_ledgers_after


def test_shared_numbers_exist_when_share_positive_and_none_when_zero(
    fraud_on_world: InteractiveWorldOutput,
) -> None:
    """Shared household numbers appear when configured above zero and are absent when set to zero.

    Recreates household telephone sharing where mule accounts share a phone line with
    an innocent family member. Catches a bug where sharing rates fail to control number
    linkage or where zero-share configurations inadvertently link unrelated accounts.
    """
    # 1. Share set to 0.5: shared numbers must exist
    book_shared = build_phone_book(
        fraud_on_world,
        run_seed=101,
        phone_config={"identifier_shared_across_mule_accounts_share": 0.5},
    )
    assert book_shared.shared_count > 0

    shared_records = [r for r in book_shared.records.values() if r.is_shared]
    assert len(shared_records) == book_shared.shared_count
    for r in shared_records:
        assert r.shared_with_account_id is not None
        assert r.shared_with_account_id in book_shared.records
        # The mule and the genuine family member must share the exact same synthetic number
        genuine_phone = book_shared.records[r.shared_with_account_id].phone_number
        assert r.phone_number == genuine_phone

    # 2. Share set to 0.0: no shared numbers must exist
    book_none = build_phone_book(
        fraud_on_world,
        run_seed=101,
        phone_config={"identifier_shared_across_mule_accounts_share": 0.0},
    )
    assert book_none.shared_count == 0
    for r in book_none.records.values():
        assert r.is_shared is False
        assert r.shared_with_account_id is None


def test_missing_recycled_number_share_raises(fraud_on_world: InteractiveWorldOutput) -> None:
    """Omitting recycled_number_share raises ValueError naming the key and config/nationalsim.yaml.

    Recreates a misconfigured deployment where the research assumption for recycled phone
    shares is omitted. Catches a bug where missing configuration values silently default to 0.05
    rather than failing fast.
    """
    with pytest.raises(ValueError, match="recycled_number_share.*config/nationalsim.yaml"):
        build_phone_book(
            fraud_on_world,
            run_seed=101,
            phone_config={"recycled_number_share": None},
        )


def test_missing_split_raises(fraud_on_world: InteractiveWorldOutput) -> None:
    """Missing range_name in the world's effective settings raises ValueError.

    Recreates an execution path where the simulation split was not recorded in effective settings.
    Catches a bug where an unconfigured split silently defaults to 'training' rather than raising.
    """
    from dataclasses import replace

    bad_effective_config = {
        "world_configs": fraud_on_world.effective_config["world_configs"],
        "settings": {
            k: v for k, v in fraud_on_world.effective_config["settings"].items() if k != "range_name"
        },
    }
    bad_world = replace(fraud_on_world, effective_config=bad_effective_config)
    with pytest.raises(ValueError, match="range_name"):
        build_phone_book(bad_world, run_seed=101)


def test_stress_split_world_draws_from_stress_range(
    fraud_on_world: InteractiveWorldOutput,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stress-split world draws the shared phone share strictly from the stress range bounds.

    Recreates a stress-testing simulation run where mule identifier sharing rates must follow
    the wider stress evaluation bounds rather than baseline training bounds. Catches a bug
    where the phone book generator hardcodes the 'training' partition or ignores effective ranges.
    """
    from dataclasses import replace

    import world.phone_book as phone_book_module
    from world.config_reader import get_parameter_entry

    world_ranges = fraud_on_world.effective_config["world_configs"]["world_ranges"]
    entry = get_parameter_entry(world_ranges, "identifier_shared_across_mule_accounts_share")
    training_bounds = entry["training"]
    stress_bounds = entry["stress"]

    # Verify that the stress range actually differs from the training range
    assert stress_bounds != training_bounds

    drawn_values: list[float] = []
    real_draw_in_range = phone_book_module.draw_in_range

    def spy_draw(
        entry: dict[str, Any],
        rng: Any,
        range_name: str = "training",
        param_name: str = "unnamed_parameter",
    ) -> float:
        val = real_draw_in_range(entry, rng, range_name=range_name, param_name=param_name)
        if param_name == "identifier_shared_across_mule_accounts_share":
            assert range_name == "stress"
            drawn_values.append(val)
        return val

    monkeypatch.setattr(phone_book_module, "draw_in_range", spy_draw)

    stress_effective = dict(fraud_on_world.effective_config)
    stress_effective["settings"] = dict(fraud_on_world.effective_config["settings"])
    stress_effective["settings"]["range_name"] = "stress"
    stress_world = replace(fraud_on_world, effective_config=stress_effective)

    build_phone_book(stress_world, run_seed=101)

    assert len(drawn_values) == 1
    drawn_share = drawn_values[0]
    assert float(stress_bounds[0]) <= drawn_share <= float(stress_bounds[1])

