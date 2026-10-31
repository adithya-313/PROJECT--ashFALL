# File: tests/unit/test_decision_reads.py
# Purpose: Proves that sharing one set of reads across a decision's feature functions changes no value.
# Issue: Level 15 decision latency (duplicate reads within one decision).
"""
WHAT: Seeds five payers in fakeredis (same-bank and other-bank peers, failed attempts, labelled and complained
accounts) and checks that compute_feature_values (which reads the payer snapshot and label lists once) returns
exactly the values the separate feature functions return when each reads for itself (reads=None); that reads
made for a different account or arrival cutoff are refused; and that a whole decision reads the payer's
snapshot exactly once.

WHY: DecisionReads only removes duplicate Redis reads. Every logged feature value, and so every replay, must
stay identical, and a wrong set of reads must fail loudly instead of quietly producing wrong features.
"""

import random
from datetime import datetime, timedelta, timezone
from typing import Any

import fakeredis
import pytest

from api.v1.models import DecisionRequest
from features.account_keys import (
    AccountFeaturesSettings,
    complained_accounts_key,
    event_times_key,
    labelled_accounts_key,
    load_account_features_settings,
)
from features.account_records import PaymentSideRecord
from features.account_summaries import update_account_summaries
from features.account_updater import apply_payment_side
from features.decision_features import compute_feature_values
from features.decision_reads import read_decision_reads
from features.feature_log import canonical_feature_names
from features.m1_features import compute_account_features
from features.m1_payment import (
    ProposedPayment,
    compute_payee_vpa_payment_features,
    compute_payer_only_payment_features,
    compute_payment_features,
)
from features.velocity import to_unix_seconds

RUN_ID = "66db9aa439b092ee"
BANK_0 = "bank_0"
BANK_1 = "bank_1"
DECISION_TIME = datetime(2026, 10, 28, 12, 0, 0, tzinfo=timezone.utc)
SEED = 20261101
PAYERS = [f"acc_payer_{i}" for i in range(5)]
PEERS = [f"acc_peer_{i:02d}" for i in range(8)]
# The payee each payer is about to pay: a labelled account, a complained account, a peer, other-bank twice.
PAYEES = [
    ("bank_0", "acc_peer_00"),
    ("bank_0", "acc_peer_01"),
    ("bank_0", "acc_peer_02"),
    ("bank_1", "x@bank1"),
    ("bank_1", "y@bank1"),
]
ACCOUNT_GROUPS = [
    "inflow",
    "outflow",
    "balance_pattern",
    "routine_life",
    "lifecycle",
    "control",
    "region",
    "network",
    "intelligence",
]


def _store(
    client: fakeredis.FakeRedis,
    settings: AccountFeaturesSettings,
    number: int,
    account_id: str,
    side: str,
    counterparty_bank: str,
    counterparty: str,
    amount_paise: int,
    status: str,
    event_time: datetime,
) -> None:
    record = PaymentSideRecord(
        run_id=RUN_ID,
        bank_id=BANK_0,
        payment_id=f"{RUN_ID}-p-{number:04d}",
        account_id=account_id,
        side=side,
        counterparty_vpa=counterparty,
        counterparty_bank_id=counterparty_bank,
        amount_paise=amount_paise,
        status=status,
        failure_domain="customer" if status != "success" else None,
        payee_type="person",
        channel="upi_app",
        sender_device_fingerprint=None,
        sender_os_generation=None,
        event_time=event_time,
        arrival_time=event_time + timedelta(seconds=1),
        balance_after_paise=900000,
    )
    apply_payment_side(client, record, settings)
    update_account_summaries(client, record, settings)


def seeded_client() -> tuple[fakeredis.FakeRedis, AccountFeaturesSettings]:
    """A fakeredis with fixed-seed histories for the five payers and their peers, plus label lists."""
    settings = load_account_features_settings()
    client = fakeredis.FakeRedis()
    rng = random.Random(SEED)
    number = 0
    for payer in PAYERS:
        for _ in range(12):
            number += 1
            peer = rng.choice(PEERS)
            bank = rng.choice([BANK_0, BANK_1])
            age = timedelta(hours=rng.randint(1, 150)) if rng.random() < 0.8 else timedelta(days=rng.randint(10, 14))
            status = "success" if rng.random() < 0.85 else "failed"
            side = rng.choice(["debit", "credit"])
            _store(
                client,
                settings,
                number,
                payer,
                side,
                bank,
                peer,
                rng.randint(1, 90) * 1000,
                status,
                DECISION_TIME - age,
            )
    for peer in PEERS:
        for _ in range(3):
            number += 1
            other = rng.choice(PEERS + PAYERS)
            if other != peer:
                _store(
                    client,
                    settings,
                    number,
                    peer,
                    "debit",
                    BANK_0,
                    other,
                    5000,
                    "success",
                    DECISION_TIME - timedelta(hours=rng.randint(1, 150)),
                )
    visible_from = to_unix_seconds(DECISION_TIME - timedelta(days=1))
    client.zadd(labelled_accounts_key(RUN_ID, BANK_0), {"acc_peer_00": visible_from, "acc_peer_05": visible_from})
    client.zadd(complained_accounts_key(RUN_ID, BANK_0), {"acc_peer_01": visible_from})
    return client, settings


def _request(payer: str, payee: tuple[str, str]) -> DecisionRequest:
    return DecisionRequest(
        payment_id=f"{RUN_ID}-dec-{payer}",
        sender_account_id=payer,
        receiver_bank_id=payee[0],
        receiver_vpa=payee[1],
        amount_paise=45000,
        event_time=DECISION_TIME,
        channel="upi_app",
        payee_type="person",
    )


def _unshared_values(
    client: Any, settings: AccountFeaturesSettings, payer: str, payee: tuple[str, str]
) -> dict[str, Any]:
    """What the decision computed before sharing: every feature function reads for itself (reads=None)."""
    account_feats = compute_account_features(
        client=client,
        run_id=RUN_ID,
        bank_id=BANK_0,
        account_id=payer,
        decision_time=DECISION_TIME,
        arrival_cutoff=DECISION_TIME,
        settings=settings,
        groups=ACCOUNT_GROUPS,
        reads=None,
    )
    proposed = ProposedPayment(
        run_id=RUN_ID,
        bank_id=BANK_0,
        payer_account_id=payer,
        payee_bank_id=payee[0],
        payee_vpa=payee[1],
        payee_type="person",
        amount_paise=45000,
        event_time=DECISION_TIME,
    )
    payment_feats = compute_payment_features(client, proposed, DECISION_TIME, settings, reads=None)
    values: dict[str, Any] = {}
    for name in canonical_feature_names():
        values[name] = account_feats[name] if name in account_feats else payment_feats.get(name)
    return values


@pytest.mark.parametrize("index", range(5))
def test_shared_reads_give_identical_values(index: int) -> None:
    """Same values, and the same types, whether the decision shares its reads or every function reads again."""
    client, settings = seeded_client()
    payer, payee = PAYERS[index], PAYEES[index]
    shared = compute_feature_values(
        client, BANK_0, RUN_ID, _request(payer, payee), DECISION_TIME, DECISION_TIME, settings=settings
    ).feature_values
    unshared = _unshared_values(client, settings, payer, payee)

    assert set(shared) == set(unshared)
    for name, value in unshared.items():
        assert shared[name] == value, name
        assert type(shared[name]) is type(value), name
    assert sum(value is not None for value in shared.values()) > 20


def _reads_for(client: Any, settings: AccountFeaturesSettings, account: str, cutoff: datetime) -> Any:
    return read_decision_reads(client, RUN_ID, BANK_0, account, DECISION_TIME, cutoff, settings)


def _all_feature_calls(client: Any, settings: AccountFeaturesSettings, reads: Any) -> list[Any]:
    """The four feature entry points, each handed `reads` for the payer acc_payer_0 at DECISION_TIME."""
    proposed = ProposedPayment(
        run_id=RUN_ID,
        bank_id=BANK_0,
        payer_account_id=PAYERS[0],
        payee_bank_id=BANK_0,
        payee_vpa="acc_peer_02",
        payee_type="person",
        amount_paise=45000,
        event_time=DECISION_TIME,
    )
    return [
        lambda: compute_account_features(
            client,
            RUN_ID,
            BANK_0,
            PAYERS[0],
            DECISION_TIME,
            DECISION_TIME,
            settings,
            groups=ACCOUNT_GROUPS,
            reads=reads,
        ),
        lambda: compute_payment_features(client, proposed, DECISION_TIME, settings, reads=reads),
        lambda: compute_payer_only_payment_features(
            client, RUN_ID, BANK_0, PAYERS[0], 45000, DECISION_TIME, DECISION_TIME, settings, reads=reads
        ),
        lambda: compute_payee_vpa_payment_features(
            client,
            RUN_ID,
            BANK_0,
            PAYERS[0],
            BANK_0,
            "acc_peer_02",
            DECISION_TIME,
            DECISION_TIME,
            settings,
            reads=reads,
        ),
    ]


def test_reads_for_the_matching_decision_are_accepted() -> None:
    """Control: reads built for the same account and cutoff are used without error by all four entry points."""
    client, settings = seeded_client()
    reads = _reads_for(client, settings, PAYERS[0], DECISION_TIME)
    for call in _all_feature_calls(client, settings, reads):
        call()


def test_reads_for_a_different_account_are_refused() -> None:
    """DecisionReads of another payer raise ValueError instead of silently feeding the wrong history."""
    client, settings = seeded_client()
    reads = _reads_for(client, settings, PAYERS[1], DECISION_TIME)
    for call in _all_feature_calls(client, settings, reads):
        with pytest.raises(ValueError, match="account_id"):
            call()


def test_reads_for_a_different_arrival_cutoff_are_refused() -> None:
    """DecisionReads taken at an earlier arrival cutoff raise ValueError."""
    client, settings = seeded_client()
    reads = _reads_for(client, settings, PAYERS[0], DECISION_TIME - timedelta(hours=1))
    for call in _all_feature_calls(client, settings, reads):
        with pytest.raises(ValueError, match="arrival_cutoff"):
            call()


class _CountingClient:
    """Forwards every call to a real client and records (method name, first argument)."""

    def __init__(self, inner: fakeredis.FakeRedis) -> None:
        self._inner = inner
        self.calls: list[tuple[str, Any]] = []

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(self._inner, name)
        if not callable(attribute):
            return attribute

        def counted(*args: Any, **kwargs: Any) -> Any:
            self.calls.append((name, args[0] if args else None))
            return attribute(*args, **kwargs)

        return counted

    def payer_snapshot_reads(self, payer: str) -> int:
        """Snapshot reads of this payer: lifecycle events are only ever read by read_account_snapshot."""
        key = event_times_key(RUN_ID, BANK_0, payer)
        return sum(1 for name, first in self.calls if name == "zrangebyscore" and first == key)


def test_a_whole_decision_reads_the_payer_snapshot_once() -> None:
    """compute_feature_values reads the payer snapshot one time; the unshared functions read it three times."""
    client, settings = seeded_client()
    payer, payee = PAYERS[0], PAYEES[0]

    shared_client = _CountingClient(client)
    compute_feature_values(
        shared_client, BANK_0, RUN_ID, _request(payer, payee), DECISION_TIME, DECISION_TIME, settings=settings
    )
    assert shared_client.payer_snapshot_reads(payer) == 1

    unshared_client = _CountingClient(client)
    _unshared_values(unshared_client, settings, payer, payee)
    assert unshared_client.payer_snapshot_reads(payer) == 3
