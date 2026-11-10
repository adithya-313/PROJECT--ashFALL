"""
Tests that the committed Level 16 showcase world contains every guarantee (plan only, no replay).

WHY: The 12 checks pass only on real counts above zero, so the world they run on must hold a crackdown, a cross-bank
complaint, a listed recycled number, a G3 account and a false report, and must keep holding them if config changes.

TERM: G3: one receiving account collecting complaints from several victims over weeks.
"""

from functools import lru_cache

from showcase.level_16_world import (
    GuaranteeCounts,
    guarantee_counts,
    level_16_registry_settings,
    level_16_world_configs,
    load_level_16_config,
    plan_level_16_world,
    unmet_guarantees,
)
from world.label_builder import build_label_events

CFG = load_level_16_config()


@lru_cache(maxsize=1)
def planned():  # type: ignore[no-untyped-def]
    """The committed seed's planned world, built once for this file."""
    return plan_level_16_world(CFG)


def test_every_guarantee_meets_its_floor() -> None:
    """Checks the committed seed meets all six floors in config/showcase_level_16.yaml."""
    counts = guarantee_counts(planned(), level_16_registry_settings(CFG))
    assert unmet_guarantees(counts, CFG) == [], counts


def test_the_recorded_scan_counts_match_the_world() -> None:
    """Checks the counts written in the YAML are what the committed seed really gives."""
    counts = guarantee_counts(planned(), level_16_registry_settings(CFG))
    rec = CFG["scan"]["chosen_seed_counts"]
    assert counts.bulk_confirmations_with_accounts == rec["bulk_confirmations_with_accounts"]
    assert counts.cross_bank_complaints == rec["cross_bank_complaints"]
    assert counts.recycled_numbers_listed == rec["recycled_numbers_listed_before_day_0"]
    assert counts.g3_complaints == rec["g3_complaints_on_one_account"]
    assert round(counts.g3_spread_days, 1) == rec["g3_spread_days"]
    assert counts.false_reported_accounts == rec["false_reported_accounts"]
    assert counts.hold_requests == rec["hold_requests"]
    assert counts.innocent_holder_listings == rec["innocent_holder_listings"]


def test_g3_arrivals_spread_over_at_least_14_days_from_several_victims() -> None:
    """Checks one account gets complaints from at least 3 different victims spread over at least 14 days."""
    counts = guarantee_counts(planned(), level_16_registry_settings(CFG))
    assert counts.g3_victims >= 3 and counts.g3_spread_days >= 14


def test_a_cross_bank_complaint_names_a_payee_at_another_bank() -> None:
    """Checks at least one complaint's payee bank differs from the complaint bank."""
    events = [e for e in build_label_events(planned().output) if e.kind == "complaint"]
    assert any(e.reported_bank_index != e.complainant_bank_index for e in events)


def test_pins_are_showcase_only_and_nest_inside_the_sourced_ranges() -> None:
    """Checks the pinned configs build (nesting holds) and the false-complaint pin is the stress maximum."""
    ranges = level_16_world_configs(CFG).world_ranges["parameters"]
    assert ranges["false_complaints_per_100000_genuine_payments"]["training"] == [50, 50]
    assert all(pin["showcase_only"] is True for pin in CFG["pins"].values())
    assert isinstance(guarantee_counts(planned(), level_16_registry_settings(CFG)), GuaranteeCounts)


def test_bank_2_is_the_showcase_non_participant_and_the_others_take_part() -> None:
    """Checks the showcase registry settings switch bank_2 off and leave bank_0 and bank_1 enrolled."""
    assert level_16_registry_settings(CFG).participants == ["bank_0", "bank_1"]
