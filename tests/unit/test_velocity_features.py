# File: tests/unit/test_velocity_features.py
# Purpose: Unit tests for velocity features in features/velocity.py using an in-memory fakeredis client.
from datetime import datetime, timedelta

import fakeredis
import pytest

from features.velocity import (
    WINDOW_1H_SECONDS,
    WINDOW_24H_SECONDS,
    compute_all_velocities,
    compute_velocity,
    record_transaction,
    update_account_state,
)

NOW = datetime(2024, 1, 15, 12, 0, 0)
FINGERPRINT = "dev_id=abc123|os=Windows|browser=Chrome"


@pytest.fixture
def client():
    return fakeredis.FakeRedis(decode_responses=True)


def test_three_recent_transactions_counted_in_1h_window(client):
    for i in range(3):
        record_transaction(client, "acct1", f"txn_{i}", NOW - timedelta(minutes=10 * i))
    assert compute_velocity(client, "acct1", WINDOW_1H_SECONDS, NOW) == 3


def test_transactions_outside_window_not_counted(client):
    for i in range(3):
        record_transaction(client, "acct1", f"txn_{i}", NOW - timedelta(hours=2, minutes=i))
    assert compute_velocity(client, "acct1", WINDOW_1H_SECONDS, NOW) == 0


def test_velocity_for_unknown_account_is_zero(client):
    assert compute_velocity(client, "nobody", WINDOW_24H_SECONDS, NOW) == 0


def test_compute_all_velocities_five_recent_transactions(client):
    for i in range(5):
        record_transaction(client, "acct1", f"txn_{i}", NOW - timedelta(minutes=5 * i))
    result = compute_all_velocities(client, "acct1", NOW)
    assert result == {"velocity_1h": 5, "velocity_6h": 5, "velocity_24h": 5}


def test_compute_all_velocities_returns_exactly_three_keys(client):
    assert set(compute_all_velocities(client, "acct1", NOW)) == {"velocity_1h", "velocity_6h", "velocity_24h"}


def test_compute_all_velocities_old_transaction_only_in_larger_windows(client):
    record_transaction(client, "acct1", "txn_old", NOW - timedelta(hours=3))
    result = compute_all_velocities(client, "acct1", NOW)
    assert result == {"velocity_1h": 0, "velocity_6h": 1, "velocity_24h": 1}


def test_update_account_state_preserves_first_seen(client):
    second = NOW + timedelta(hours=1)
    update_account_state(client, "acct1", "t1", "12.9716,77.5946", "m1", FINGERPRINT, NOW)
    update_account_state(client, "acct1", "t2", "28.6139,77.2090", "m2", FINGERPRINT, second)
    assert client.get("acct:first_seen:acct1") == NOW.isoformat()


def test_update_account_state_overwrites_last_location_and_time(client):
    second = NOW + timedelta(hours=1)
    update_account_state(client, "acct1", "t1", "12.9716,77.5946", "m1", FINGERPRINT, NOW)
    update_account_state(client, "acct1", "t2", "28.6139,77.2090", "m2", FINGERPRINT, second)
    assert client.get("acct:last_loc:acct1") == "28.6139,77.2090"
    assert client.get("acct:last_txn_time:acct1") == second.isoformat()


def test_update_account_state_stores_parsed_dev_id_and_merchants_newest_first(client):
    update_account_state(client, "acct1", "t1", "1,1", "m1", FINGERPRINT, NOW)
    update_account_state(client, "acct1", "t2", "1,1", "m2", FINGERPRINT, NOW)
    assert client.lrange("dev:history:acct1", 0, -1) == ["abc123", "abc123"]
    assert client.lrange("merch:history:acct1", 0, -1) == ["m2", "m1"]
