# File: tests/unit/test_network_batching.py
# Purpose: Proves the batched network reads return exactly what the one-account-at-a-time reads returned.
# Issue: Level 15 decision latency (network group round trips).
"""
WHAT: Builds one fixed 18-account payment graph in fakeredis (chain, hub, random extra edges, labelled and
complained accounts, one record that arrives after the cutoff) and checks:
  - same_bank_counterparties_many(a) equals the frozen same-bank neighbour set of every account;
  - the full network-group output for 5 start accounts (one hits the node budget) equals frozen values;
  - read_side_records_many equals read_account_snapshot(...).side_records for every account.

WHY: The batching only changes how many Redis round trips are made. Every feature value must stay identical,
so the expected values below are literals produced ONCE by running the unbatched code of commit be01c79
(same_bank_counterparties and compute_network_features) on this exact graph. They are not recomputed here.
"""

import logging
import random
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import fakeredis
import pytest

from features.account_keys import (
    AccountFeaturesSettings,
    complained_accounts_key,
    labelled_accounts_key,
    load_account_features_settings,
)
from features.account_labels import same_bank_counterparties, same_bank_counterparties_many
from features.account_reader import read_account_snapshot, read_side_records_many
from features.account_records import PaymentSideRecord
from features.account_summaries import update_account_summaries
from features.account_updater import apply_payment_side
from features.m1_network import compute_network_features
from features.velocity import to_unix_seconds

RUN_ID = "66db9aa439b092ee"
BANK_0 = "bank_0"
DECISION_TIME = datetime(2026, 10, 28, 12, 0, 0, tzinfo=timezone.utc)
GRAPH_SEED = 20261031
ACCOUNTS = [f"a{i:02d}" for i in range(18)]
FLAGGED = ("a03", "a10", "a11", "a12", "a13")
COMPLAINED = ("a07", "a14")
NODE_BUDGET = 4


def _edge(
    client: fakeredis.FakeRedis,
    settings: AccountFeaturesSettings,
    number: int,
    sender: str,
    receiver: str,
    event_time: datetime,
    arrival_time: datetime,
    with_summaries: bool,
) -> None:
    """Store both sides of one payment between two accounts of bank_0 (a debit for sender, a credit for receiver)."""
    for side, account_id, counterparty in (("debit", sender, receiver), ("credit", receiver, sender)):
        record = PaymentSideRecord(
            run_id=RUN_ID,
            bank_id=BANK_0,
            payment_id=f"{RUN_ID}-p-{number:03d}",
            account_id=account_id,
            side=side,
            counterparty_vpa=counterparty,
            counterparty_bank_id=BANK_0,
            amount_paise=100000 + number,
            status="success",
            failure_domain=None,
            payee_type="person",
            channel="upi_app",
            sender_device_fingerprint=None,
            sender_os_generation=None,
            event_time=event_time,
            arrival_time=arrival_time,
            balance_after_paise=900000,
        )
        apply_payment_side(client, record, settings)
        if with_summaries:
            update_account_summaries(client, record, settings)


def build_graph(client: fakeredis.FakeRedis, settings: AccountFeaturesSettings) -> None:
    """Write the fixed graph: a chain a00-a01-a02-a03, a hub at a02, seeded extra edges, labels and complaints."""
    rng = random.Random(GRAPH_SEED)
    edges: list[tuple[str, str, bool]] = [
        ("a00", "a01", False),
        ("a01", "a02", False),
        ("a02", "a03", False),
        ("a00", "a14", False),
        ("a10", "a14", False),
        ("a11", "a14", False),
        ("a12", "a15", False),
        ("a13", "a16", False),
    ]
    edges += [("a02", f"a{i:02d}", rng.random() < 0.5) for i in range(4, 10)]
    while len(edges) < 32:
        first, second = rng.sample(ACCOUNTS, 2)
        edges.append((first, second, rng.random() < 0.5))

    for number, (sender, receiver, is_old) in enumerate(edges):
        if is_old:
            # An old edge: more than a week back, so it only exists as closed counterparty days.
            event_time = DECISION_TIME - timedelta(days=rng.randint(10, 14))
        else:
            event_time = DECISION_TIME - timedelta(hours=rng.randint(1, 150))
        _edge(client, settings, number, sender, receiver, event_time, event_time + timedelta(seconds=1), is_old)

    # One recent record that reaches the store AFTER the arrival cutoff, so readers must leave it out.
    late_arrival = DECISION_TIME + timedelta(hours=1)
    _edge(client, settings, len(edges), "a05", "a17", DECISION_TIME - timedelta(hours=2), late_arrival, False)

    visible_from = to_unix_seconds(DECISION_TIME - timedelta(days=1))
    client.zadd(labelled_accounts_key(RUN_ID, BANK_0), {account: visible_from for account in FLAGGED})
    client.zadd(complained_accounts_key(RUN_ID, BANK_0), {account: visible_from for account in COMPLAINED})


def network_outputs(client: fakeredis.FakeRedis, settings: AccountFeaturesSettings) -> dict[str, dict[str, object]]:
    """The full network-group output for 5 start accounts; the last one runs with a small node budget."""
    cases = [
        ("a00", settings),
        ("a02", settings),
        ("a09", settings),
        ("a16", settings),
        ("a00_budget", replace(settings, network_max_nodes_visited=NODE_BUDGET)),
    ]
    results: dict[str, dict[str, object]] = {}
    for label, case_settings in cases:
        account_id = label.split("_")[0]
        snapshot = read_account_snapshot(
            client, RUN_ID, BANK_0, account_id, DECISION_TIME, DECISION_TIME, case_settings
        )
        results[label] = compute_network_features(
            client, RUN_ID, BANK_0, account_id, snapshot, DECISION_TIME, DECISION_TIME, case_settings
        )
    return results


def fresh_graph() -> tuple[fakeredis.FakeRedis, AccountFeaturesSettings]:
    """A new fakeredis holding the fixed graph, with the repository settings."""
    settings = load_account_features_settings()
    client = fakeredis.FakeRedis()
    build_graph(client, settings)
    return client, settings


# Frozen expected values. Generated ONCE by running the unbatched code of commit be01c79
# (same_bank_counterparties and compute_network_features) on build_graph(). Do not regenerate them
# from the code under test.
NEIGHBOURS_EXPECTED: dict[str, frozenset[str]] = {
    "a00": frozenset({"a01", "a07", "a14"}),
    "a01": frozenset({"a00", "a02", "a08"}),
    "a02": frozenset({"a01", "a03", "a04", "a05", "a06", "a07", "a08", "a09"}),
    "a03": frozenset({"a02", "a05", "a07", "a11"}),
    "a04": frozenset({"a02", "a06"}),
    "a05": frozenset({"a02", "a03", "a08", "a13"}),
    "a06": frozenset({"a02", "a04"}),
    "a07": frozenset({"a00", "a02", "a03", "a15", "a16"}),
    "a08": frozenset({"a01", "a02", "a05"}),
    "a09": frozenset({"a02"}),
    "a10": frozenset({"a14", "a17"}),
    "a11": frozenset({"a03", "a12", "a14"}),
    "a12": frozenset({"a11", "a15"}),
    "a13": frozenset({"a05", "a16"}),
    "a14": frozenset({"a00", "a10", "a11"}),
    "a15": frozenset({"a07", "a12"}),
    "a16": frozenset({"a07", "a13"}),
    "a17": frozenset({"a10"}),
}
NETWORK_EXPECTED: dict[str, dict[str, object]] = {
    "a00": {
        "hops_to_confirmed_label": 2,
        "shared_counterparties_with_flagged_7d": 2,
        "local_cluster_density": 0.0,
        "hops_to_complained_account": 1,
    },
    "a02": {
        "hops_to_confirmed_label": 1,
        "shared_counterparties_with_flagged_7d": 3,
        "local_cluster_density": 0.17857142857142858,
        "hops_to_complained_account": 1,
    },
    "a09": {
        "hops_to_confirmed_label": 2,
        "shared_counterparties_with_flagged_7d": 0,
        "local_cluster_density": None,
        "hops_to_complained_account": 2,
    },
    "a16": {
        "hops_to_confirmed_label": 1,
        "shared_counterparties_with_flagged_7d": 1,
        "local_cluster_density": 0.0,
        "hops_to_complained_account": 1,
    },
    "a00_budget": {
        "hops_to_confirmed_label": None,
        "shared_counterparties_with_flagged_7d": 2,
        "local_cluster_density": 0.0,
        "hops_to_complained_account": 1,
    },
}


def test_same_bank_counterparties_many_matches_frozen_neighbours() -> None:
    """Every account's batched neighbour set equals the set the one-at-a-time code returned."""
    client, settings = fresh_graph()
    batched = same_bank_counterparties_many(client, RUN_ID, BANK_0, ACCOUNTS, DECISION_TIME, DECISION_TIME, settings)
    assert batched == NEIGHBOURS_EXPECTED


def test_same_bank_counterparties_single_call_matches_frozen_neighbours() -> None:
    """The one-account function gives the same sets (it is a thin call into the batched one)."""
    client, settings = fresh_graph()
    for account_id in ACCOUNTS:
        single = same_bank_counterparties(client, RUN_ID, BANK_0, account_id, DECISION_TIME, DECISION_TIME, settings)
        assert single == NEIGHBOURS_EXPECTED[account_id], account_id


def test_late_arriving_record_is_not_a_neighbour() -> None:
    """a05 paid a17 but that record arrived after the cutoff, so neither sees the other."""
    client, settings = fresh_graph()
    batched = same_bank_counterparties_many(
        client, RUN_ID, BANK_0, ["a05", "a17"], DECISION_TIME, DECISION_TIME, settings
    )
    assert "a17" not in batched["a05"]
    assert "a05" not in batched["a17"]


def test_network_group_outputs_match_frozen_values(caplog: pytest.LogCaptureFixture) -> None:
    """The full network group for 5 start accounts equals the frozen outputs, including the budget case."""
    client, settings = fresh_graph()
    with caplog.at_level(logging.WARNING, logger="features.m1_network"):
        outputs = network_outputs(client, settings)
    assert outputs == NETWORK_EXPECTED


def test_node_budget_exceeded_returns_none_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    """With a budget of 4 nodes the search from a00 gives up before reaching the flagged account."""
    client, settings = fresh_graph()
    with caplog.at_level(logging.WARNING, logger="features.m1_network"):
        outputs = network_outputs(client, settings)
    assert NETWORK_EXPECTED["a00"]["hops_to_confirmed_label"] == 2
    assert outputs["a00_budget"]["hops_to_confirmed_label"] is None
    assert any("network_max_nodes_visited reached" in message for message in caplog.messages)


def test_read_side_records_many_equals_snapshot_side_records() -> None:
    """For every account the batched read returns exactly the snapshot's side records, in the same order."""
    client, settings = fresh_graph()
    batched = read_side_records_many(client, RUN_ID, BANK_0, ACCOUNTS, DECISION_TIME, DECISION_TIME, settings)
    assert set(batched) == set(ACCOUNTS)
    for account_id in ACCOUNTS:
        snapshot = read_account_snapshot(client, RUN_ID, BANK_0, account_id, DECISION_TIME, DECISION_TIME, settings)
        assert batched[account_id] == snapshot.side_records, account_id
    assert sum(len(records) for records in batched.values()) > 30


def test_read_side_records_many_applies_the_arrival_filter() -> None:
    """The record that arrives after the cutoff is visible once the cutoff moves past it, and not before."""
    client, settings = fresh_graph()
    before = read_side_records_many(client, RUN_ID, BANK_0, ["a17"], DECISION_TIME, DECISION_TIME, settings)
    after = read_side_records_many(
        client, RUN_ID, BANK_0, ["a17"], DECISION_TIME, DECISION_TIME + timedelta(hours=2), settings
    )
    assert len(after["a17"]) == len(before["a17"]) + 1


def test_read_side_records_many_with_no_accounts_makes_no_round_trip() -> None:
    """An empty account list returns {} without touching Redis."""
    client, settings = fresh_graph()
    assert read_side_records_many(client, RUN_ID, BANK_0, [], DECISION_TIME, DECISION_TIME, settings) == {}
