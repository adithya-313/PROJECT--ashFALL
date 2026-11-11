"""
Tests that registry identifier hashes namespace accounts by bank index but not phones.

WHY: A phone shared across banks must hash the same everywhere so banks can match it; an account id must hash
differently at two banks.

TERM: namespacing: prefixing the raw value with the bank index before hashing.
"""

from nationalsim.keyed_hash import compute_identifier_hash

KEY = "unit-test-key"
RUN = "run_x"


def test_same_phone_at_two_bank_indexes_gives_the_same_hash() -> None:
    """Checks a phone hashes identically whatever bank index is passed."""
    first = compute_identifier_hash(KEY, RUN, "phone", "sim-0000000001", bank_index=0)
    second = compute_identifier_hash(KEY, RUN, "phone", "sim-0000000001", bank_index=1)
    assert first == second == compute_identifier_hash(KEY, RUN, "phone", "sim-0000000001")


def test_same_account_id_at_two_bank_indexes_gives_different_hashes() -> None:
    """Checks the same account id hashes differently at two bank indexes."""
    first = compute_identifier_hash(KEY, RUN, "account", "acc_person_000001", bank_index=0)
    second = compute_identifier_hash(KEY, RUN, "account", "acc_person_000001", bank_index=1)
    assert first != second
