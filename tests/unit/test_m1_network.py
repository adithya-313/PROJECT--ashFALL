# File: tests/unit/test_m1_network.py
# Purpose: Unit tests for Level 15b M1 network features (graph hops, shared counterparties, cluster density).
# Issue: Level 15b M1 network features (plan Task 1, G1, R8).
"""
WHAT: Unit tests validating graph traversal and topological feature extraction:
  1. hops_to_confirmed_label / hops_to_complained_account:
     - Chain A-B-C-D with D labelled returns 3 hops from A.
     - max_hops=2 returns None when D is 3 hops away.
     - A itself labelled returns 0.
  2. Tenant isolation: a labelled account at another bank is never reached (returns None).
  3. Node budget: hitting network_max_nodes_visited returns None and logs a warning.
  4. shared_counterparties_with_flagged_7d: 2 of A's 3 counterparties also transacting with a labelled account.
  5. local_cluster_density: N = {B, C, D} with only B-C connected returns 1/3; returns None when |N| < 2.
"""

from datetime import datetime, timedelta, timezone

import fakeredis
import pytest

from features.account_keys import (
    AccountFeaturesSettings,
    complained_accounts_key,
    labelled_accounts_key,
    load_account_features_settings,
)
from features.account_records import PaymentSideRecord
from features.account_updater import apply_payment_side
from features.m1_features import compute_account_features

RUN_ID = "66db9aa439b092ee"
BANK_0 = "bank_0"
BANK_1 = "bank_1"
NOW_UTC = datetime(2026, 10, 28, 12, 0, 0, tzinfo=timezone.utc)
IST = timezone(timedelta(hours=5, minutes=30))


def _make_payment_side(
    payment_id: str,
    account_id: str,
    side: str,
    counterparty_bank_id: str,
    counterparty_vpa: str,
    event_time: datetime,
    amount_paise: int = 100000,
    status: str = "success",
    bank_id: str = BANK_0,
) -> PaymentSideRecord:
    return PaymentSideRecord(
        run_id=RUN_ID,
        bank_id=bank_id,
        account_id=account_id,
        payment_id=payment_id,
        side=side,
        counterparty_bank_id=counterparty_bank_id,
        counterparty_vpa=counterparty_vpa,
        amount_paise=amount_paise,
        status=status,
        failure_domain=None,
        payee_type="person",
        channel="upi_app",
        sender_device_fingerprint=None,
        sender_os_generation=None,
        event_time=event_time,
        arrival_time=event_time,
        balance_after_paise=1000000,
    )


@pytest.fixture
def redis_client() -> fakeredis.FakeRedis:
    return fakeredis.FakeRedis()


def test_hops_chain_and_max_hops_and_self(redis_client: fakeredis.FakeRedis) -> None:
    """
    CRITICAL: Validates BFS hops calculation:
      - Chain A -> B -> C -> D at bank_0 with D labelled has hops == 3 from A.
      - With max_hops == 2, returns None (D is beyond horizon).
      - When A itself is labelled, returns 0.
    """
    settings = load_account_features_settings()
    t_ev = NOW_UTC - timedelta(hours=2)

    # Chain of successful transactions within bank_0:
    # 1. A pays B
    apply_payment_side(redis_client, _make_payment_side(f"{RUN_ID}-p1-deb", "A", "debit", BANK_0, "B", t_ev), settings)
    apply_payment_side(redis_client, _make_payment_side(f"{RUN_ID}-p1-crd", "B", "credit", BANK_0, "A", t_ev), settings)

    # 2. B pays C
    apply_payment_side(redis_client, _make_payment_side(f"{RUN_ID}-p2-deb", "B", "debit", BANK_0, "C", t_ev), settings)
    apply_payment_side(redis_client, _make_payment_side(f"{RUN_ID}-p2-crd", "C", "credit", BANK_0, "B", t_ev), settings)

    # 3. C pays D
    apply_payment_side(redis_client, _make_payment_side(f"{RUN_ID}-p3-deb", "C", "debit", BANK_0, "D", t_ev), settings)
    apply_payment_side(redis_client, _make_payment_side(f"{RUN_ID}-p3-crd", "D", "credit", BANK_0, "C", t_ev), settings)

    # D is a confirmed fraud labelled account
    score = int(t_ev.timestamp())
    redis_client.zadd(labelled_accounts_key(RUN_ID, BANK_0), {"D": score})

    # Test from A: A -> B (hop 1) -> C (hop 2) -> D (hop 3)
    # Arithmetic: smallest hop count to labelled D is exactly 3
    feats_3 = compute_account_features(
        client=redis_client,
        run_id=RUN_ID,
        bank_id=BANK_0,
        account_id="A",
        decision_time=NOW_UTC,
        arrival_cutoff=NOW_UTC,
        settings=settings,
        groups=["network"],
    )
    assert feats_3["hops_to_confirmed_label"] == 3

    # With max_hops=2, distance 3 exceeds horizon -> returns None
    settings_2hops = AccountFeaturesSettings(
        windows_seconds=settings.windows_seconds,
        retention_seconds=settings.retention_seconds,
        allowed_lateness_seconds=settings.allowed_lateness_seconds,
        key_expiry_margin_seconds=settings.key_expiry_margin_seconds,
        key_prefix=settings.key_prefix,
        run_id_hex_length=settings.run_id_hex_length,
        account_event_batch_size=settings.account_event_batch_size,
        account_event_overlap_rows=settings.account_event_overlap_rows,
        summary_horizon_days=settings.summary_horizon_days,
        routine_life_warm_up_days=settings.routine_life_warm_up_days,
        fingerprint_warning_count=settings.fingerprint_warning_count,
        routine_cycle_days=settings.routine_cycle_days,
        routine_rhythm_tolerance_days=settings.routine_rhythm_tolerance_days,
        routine_min_history_days=settings.routine_min_history_days,
        machine_timing_min_payments=settings.machine_timing_min_payments,
        device_change_window_seconds=settings.device_change_window_seconds,
        network_max_hops=2,
        network_max_nodes_visited=settings.network_max_nodes_visited,
    )
    feats_2 = compute_account_features(
        client=redis_client,
        run_id=RUN_ID,
        bank_id=BANK_0,
        account_id="A",
        decision_time=NOW_UTC,
        arrival_cutoff=NOW_UTC,
        settings=settings_2hops,
        groups=["network"],
    )
    assert feats_2["hops_to_confirmed_label"] is None

    # When A itself is labelled -> distance is 0
    redis_client.zadd(labelled_accounts_key(RUN_ID, BANK_0), {"A": score})
    feats_self = compute_account_features(
        client=redis_client,
        run_id=RUN_ID,
        bank_id=BANK_0,
        account_id="A",
        decision_time=NOW_UTC,
        arrival_cutoff=NOW_UTC,
        settings=settings,
        groups=["network"],
    )
    assert feats_self["hops_to_confirmed_label"] == 0


def test_other_bank_labelled_never_reached(redis_client: fakeredis.FakeRedis) -> None:
    """
    CRITICAL: Verifies tenant isolation:
    A labelled account at ANOTHER bank is never reached by graph traversal (returns None).
    (Sabotage proof: allowing BFS to step to accounts at other banks fails this test).
    """
    settings = load_account_features_settings()
    t_ev = NOW_UTC - timedelta(hours=2)

    # Account A at bank_0 pays external account Ext at bank_1
    p_ext = _make_payment_side(f"{RUN_ID}-p-ext", "A", "debit", BANK_1, "Ext", t_ev, bank_id=BANK_0)
    apply_payment_side(redis_client, p_ext, settings)

    # Ext is labelled in bank_1's namespace, and also present in bank_0's labelled set
    # (so if cross-bank edges leaked into bank_0's graph, Ext would be falsely reached)
    score = int(t_ev.timestamp())
    redis_client.zadd(labelled_accounts_key(RUN_ID, BANK_1), {"Ext": score})
    redis_client.zadd(labelled_accounts_key(RUN_ID, BANK_0), {"Ext": score})

    # At bank_0, A's graph search must never cross to bank_1
    feats = compute_account_features(
        client=redis_client,
        run_id=RUN_ID,
        bank_id=BANK_0,
        account_id="A",
        decision_time=NOW_UTC,
        arrival_cutoff=NOW_UTC,
        settings=settings,
        groups=["network"],
    )
    assert feats["hops_to_confirmed_label"] is None


def test_node_limit_reached_returns_none_and_warns(
    redis_client: fakeredis.FakeRedis,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    CRITICAL: Verifies policy rule:
    When network_max_nodes_visited is exceeded, the result is None and a warning is logged,
    never a partial number.
    (Sabotage proof: returning a partial hop count when the limit is hit fails this test).
    """
    settings = load_account_features_settings()
    t_ev = NOW_UTC - timedelta(hours=2)

    # A pays B and C at bank_0; B pays D (which is labelled)
    apply_payment_side(redis_client, _make_payment_side(f"{RUN_ID}-p-ab", "A", "debit", BANK_0, "B", t_ev), settings)
    apply_payment_side(redis_client, _make_payment_side(f"{RUN_ID}-p-ac", "A", "debit", BANK_0, "C", t_ev), settings)
    apply_payment_side(redis_client, _make_payment_side(f"{RUN_ID}-p-bd", "B", "debit", BANK_0, "D", t_ev), settings)

    score = int(t_ev.timestamp())
    redis_client.zadd(labelled_accounts_key(RUN_ID, BANK_0), {"D": score})

    # Set network_max_nodes_visited=2 (A is node 1, B is node 2, visiting C exceeds 2)
    tight_settings = AccountFeaturesSettings(
        windows_seconds=settings.windows_seconds,
        retention_seconds=settings.retention_seconds,
        allowed_lateness_seconds=settings.allowed_lateness_seconds,
        key_expiry_margin_seconds=settings.key_expiry_margin_seconds,
        key_prefix=settings.key_prefix,
        run_id_hex_length=settings.run_id_hex_length,
        account_event_batch_size=settings.account_event_batch_size,
        account_event_overlap_rows=settings.account_event_overlap_rows,
        summary_horizon_days=settings.summary_horizon_days,
        routine_life_warm_up_days=settings.routine_life_warm_up_days,
        fingerprint_warning_count=settings.fingerprint_warning_count,
        routine_cycle_days=settings.routine_cycle_days,
        routine_rhythm_tolerance_days=settings.routine_rhythm_tolerance_days,
        routine_min_history_days=settings.routine_min_history_days,
        machine_timing_min_payments=settings.machine_timing_min_payments,
        device_change_window_seconds=settings.device_change_window_seconds,
        network_max_hops=5,
        network_max_nodes_visited=2,
    )

    with caplog.at_level("WARNING"):
        feats = compute_account_features(
            client=redis_client,
            run_id=RUN_ID,
            bank_id=BANK_0,
            account_id="A",
            decision_time=NOW_UTC,
            arrival_cutoff=NOW_UTC,
            settings=tight_settings,
            groups=["network"],
        )

    # Must return None and log warning naming the limit
    assert feats["hops_to_confirmed_label"] is None
    assert any("network_max_nodes_visited reached" in record.message for record in caplog.records)


def test_shared_counterparties_with_flagged_7d(redis_client: fakeredis.FakeRedis) -> None:
    """
    CRITICAL: Validates shared_counterparties_with_flagged_7d:
    Count of A's 7-day raw window counterparties that also appear in 7d raw window of a labelled account.
    Hand-worked setup:
      - Account A has 3 counterparties: CP1, CP2, CP3.
      - Labelled mule M has counterparties: CP1, CP2, CP4.
      - Intersection is {CP1, CP2}; expected count is exactly 2.
    """
    settings = load_account_features_settings()
    t_ev = NOW_UTC - timedelta(hours=3)

    # 1. Account A transactions with CP1, CP2, CP3
    apply_payment_side(redis_client, _make_payment_side(f"{RUN_ID}-pa1", "A", "debit", BANK_0, "CP1", t_ev), settings)
    apply_payment_side(redis_client, _make_payment_side(f"{RUN_ID}-pa2", "A", "debit", BANK_0, "CP2", t_ev), settings)
    apply_payment_side(redis_client, _make_payment_side(f"{RUN_ID}-pa3", "A", "debit", BANK_0, "CP3", t_ev), settings)

    # 2. Mule account M transactions with CP1, CP2, CP4
    apply_payment_side(redis_client, _make_payment_side(f"{RUN_ID}-pm1", "M", "debit", BANK_0, "CP1", t_ev), settings)
    apply_payment_side(redis_client, _make_payment_side(f"{RUN_ID}-pm2", "M", "debit", BANK_0, "CP2", t_ev), settings)
    apply_payment_side(redis_client, _make_payment_side(f"{RUN_ID}-pm4", "M", "debit", BANK_0, "CP4", t_ev), settings)

    # Mark M as labelled confirmed fraud
    score = int(t_ev.timestamp())
    redis_client.zadd(labelled_accounts_key(RUN_ID, BANK_0), {"M": score})

    feats = compute_account_features(
        client=redis_client,
        run_id=RUN_ID,
        bank_id=BANK_0,
        account_id="A",
        decision_time=NOW_UTC,
        arrival_cutoff=NOW_UTC,
        settings=settings,
        groups=["network"],
    )

    # Arithmetic comment:
    # A's counterparties = {bank_0:CP1, bank_0:CP2, bank_0:CP3}
    # M's counterparties = {bank_0:CP1, bank_0:CP2, bank_0:CP4}
    # Overlap = {bank_0:CP1, bank_0:CP2}; count == 2
    assert feats["shared_counterparties_with_flagged_7d"] == 2


def test_local_cluster_density(redis_client: fakeredis.FakeRedis) -> None:
    """
    CRITICAL: Validates local_cluster_density:
    For account A with same-bank counterparties N = {B, C, D}:
      - Total unordered pairs among N: 3 * (3 - 1) / 2 = 3 pairs ({B, C}, {B, D}, {C, D}).
      - Only B and C have an edge (B paid C); {B, D} and {C, D} are disconnected.
      - Expected density = 1 / 3 == 0.3333333333333333.
      - An account with |N| < 2 returns None.
    """
    settings = load_account_features_settings()
    t_ev = NOW_UTC - timedelta(hours=4)

    # Account A is connected to B, C, D:
    apply_payment_side(redis_client, _make_payment_side(f"{RUN_ID}-p-ab", "A", "debit", BANK_0, "B", t_ev), settings)
    apply_payment_side(redis_client, _make_payment_side(f"{RUN_ID}-p-ac", "A", "debit", BANK_0, "C", t_ev), settings)
    apply_payment_side(redis_client, _make_payment_side(f"{RUN_ID}-p-ad", "A", "debit", BANK_0, "D", t_ev), settings)

    # Between N={B, C, D}, only B and C are connected:
    p_bc_deb = _make_payment_side(f"{RUN_ID}-p-bc-deb", "B", "debit", BANK_0, "C", t_ev)
    p_bc_crd = _make_payment_side(f"{RUN_ID}-p-bc-crd", "C", "credit", BANK_0, "B", t_ev)
    apply_payment_side(redis_client, p_bc_deb, settings)
    apply_payment_side(redis_client, p_bc_crd, settings)

    feats = compute_account_features(
        client=redis_client,
        run_id=RUN_ID,
        bank_id=BANK_0,
        account_id="A",
        decision_time=NOW_UTC,
        arrival_cutoff=NOW_UTC,
        settings=settings,
        groups=["network"],
    )

    # Arithmetic comment:
    # N = {B, C, D}, |N| = 3.
    # Total unordered pairs = 3 * 2 / 2 = 3 ({B, C}, {B, D}, {C, D}).
    # Connected pairs = 1 ({B, C}).
    # Density = 1 / 3.
    assert feats["local_cluster_density"] == pytest.approx(1.0 / 3.0)

    # Solitary account with only 1 counterparty (|N| = 1 < 2) returns None
    p_solo = _make_payment_side(f"{RUN_ID}-p-solo", "Solo", "debit", BANK_0, "B", t_ev)
    apply_payment_side(redis_client, p_solo, settings)
    feats_solo = compute_account_features(
        client=redis_client,
        run_id=RUN_ID,
        bank_id=BANK_0,
        account_id="Solo",
        decision_time=NOW_UTC,
        arrival_cutoff=NOW_UTC,
        settings=settings,
        groups=["network"],
    )
    assert feats_solo["local_cluster_density"] is None


def test_hops_to_complained_account(redis_client: fakeredis.FakeRedis) -> None:
    """Verifies that hops_to_complained_account correctly searches graph for complained accounts."""
    settings = load_account_features_settings()
    t_ev = NOW_UTC - timedelta(hours=2)

    # A -> B -> C at bank_0
    apply_payment_side(redis_client, _make_payment_side(f"{RUN_ID}-c-ab", "A", "debit", BANK_0, "B", t_ev), settings)
    apply_payment_side(redis_client, _make_payment_side(f"{RUN_ID}-c-bc", "B", "debit", BANK_0, "C", t_ev), settings)

    # Mark C as complained account
    score = int(t_ev.timestamp())
    redis_client.zadd(complained_accounts_key(RUN_ID, BANK_0), {"C": score})

    feats = compute_account_features(
        client=redis_client,
        run_id=RUN_ID,
        bank_id=BANK_0,
        account_id="A",
        decision_time=NOW_UTC,
        arrival_cutoff=NOW_UTC,
        settings=settings,
        groups=["network"],
    )

    # A -> B (hop 1) -> C (hop 2)
    assert feats["hops_to_complained_account"] == 2
