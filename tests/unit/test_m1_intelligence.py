# File: tests/unit/test_m1_intelligence.py
# Purpose: Unit tests proving M1 intelligence features and payee registry screening.
# Issue: Level 15b intelligence features (plan Task 1, G1, R8, architect rules 1, 2, 3, decision 9c).
"""
WHAT: Unit tests proving:
  - registry_match_count and registry_match_age_days calculations with FakeRegistryScreener.
  - Honest degradation: unknown screening outcome yields None, never 0 or False (Rule 3).
  - Unconfigured key yields None.
  - payee_registry_match on compute_payment_features (True, False, None).
  - registry-outage stress test scenario: non-participating bank (HTTP 403) yields unknown -> None.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import fakeredis
import httpx
import pytest

from core.event_clock import IST, ManualClock
from features.account_keys import (
    fingerprints_key,
    load_account_features_settings,
)
from features.m1_features import compute_account_features
from features.m1_intelligence import compute_intelligence_features
from features.m1_payment import ProposedPayment, compute_payment_features
from features.registry_screening import (
    FakeRegistryScreener,
    NationalsimRegistryScreener,
    ScreenOutcome,
)
from nationalsim.app import create_app
from nationalsim.bank_client import NationalsimBankClient
from nationalsim.keyed_hash import compute_identifier_hash
from world.layers import read_layer

RUN_ID = "66db9aa439b092ee"
BANK_0 = "bank_0"
BANK_1 = "bank_1"
DPIP_KEY = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
NOW_UTC = datetime(2026, 10, 28, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def redis_client() -> fakeredis.FakeRedis:
    return fakeredis.FakeRedis()


def test_intelligence_features_match_count_and_age(redis_client: fakeredis.FakeRedis) -> None:
    """
    CRITICAL: Validates intelligence feature calculation:
      - Account hash matches (first visible 10 days ago).
      - First device fingerprint matches (first visible 15 days ago).
      - Second device fingerprint does not match.
      - Expected: registry_match_count == 2, registry_match_age_days == 15.
    """
    settings = load_account_features_settings()
    account_id = "acc_intel_01"
    fp_alpha = "fp_intel_alpha"
    fp_beta = "fp_intel_beta"

    # Populate lifetime device fingerprints in Redis
    t_fp = NOW_UTC - timedelta(days=20)
    score = int(t_fp.timestamp())
    fp_k = fingerprints_key(RUN_ID, BANK_0, account_id)
    redis_client.zadd(fp_k, {fp_alpha: score, fp_beta: score})

    # Compute expected hashes
    acc_hash = compute_identifier_hash(DPIP_KEY, RUN_ID, "account", account_id, bank_index=0)
    fp_alpha_hash = compute_identifier_hash(DPIP_KEY, RUN_ID, "device", fp_alpha)
    fp_beta_hash = compute_identifier_hash(DPIP_KEY, RUN_ID, "device", fp_beta)

    t_vis_acc = NOW_UTC - timedelta(days=10)
    t_vis_fp = NOW_UTC - timedelta(days=15)

    screener = FakeRegistryScreener()
    screener.set_outcome(acc_hash, "account", ScreenOutcome(status="match", first_visible_at=t_vis_acc))
    screener.set_outcome(fp_alpha_hash, "device", ScreenOutcome(status="match", first_visible_at=t_vis_fp))
    screener.set_outcome(fp_beta_hash, "device", ScreenOutcome(status="no_match", first_visible_at=None))

    feats = compute_intelligence_features(
        client=redis_client,
        run_id=RUN_ID,
        bank_id=BANK_0,
        account_id=account_id,
        decision_time=NOW_UTC,
        arrival_cutoff=NOW_UTC,
        settings=settings,
        screener=screener,
        dpip_key=DPIP_KEY,
    )

    # Arithmetic comment:
    # 2 matching identifiers (account and fp_alpha).
    # Earliest visibility is t_vis_fp (15 days before NOW_UTC).
    # registry_match_count == 2, registry_match_age_days == 15.
    assert feats["registry_match_count"] == 2
    assert feats["registry_match_age_days"] == 15

    # Also verify through compute_account_features orchestrator
    full_feats = compute_account_features(
        client=redis_client,
        run_id=RUN_ID,
        bank_id=BANK_0,
        account_id=account_id,
        decision_time=NOW_UTC,
        arrival_cutoff=NOW_UTC,
        settings=settings,
        groups=["intelligence"],
        screener=screener,
        dpip_key=DPIP_KEY,
    )
    assert full_feats["registry_match_count"] == 2
    assert full_feats["registry_match_age_days"] == 15


def test_one_screen_unknown_returns_none_for_both(redis_client: fakeredis.FakeRedis) -> None:
    """
    CRITICAL: Validates Rule 3 (honest degradation):
      If ANY identifier query returns 'unknown', the count must be None, NEVER a partial count.
      (Sabotage proof: returning partial count 1 fails this test).
    """
    settings = load_account_features_settings()
    account_id = "acc_intel_02"
    fp_alpha = "fp_alpha_unknown"

    score = int((NOW_UTC - timedelta(days=5)).timestamp())
    redis_client.zadd(fingerprints_key(RUN_ID, BANK_0, account_id), {fp_alpha: score})

    acc_hash = compute_identifier_hash(DPIP_KEY, RUN_ID, "account", account_id, bank_index=0)
    fp_hash = compute_identifier_hash(DPIP_KEY, RUN_ID, "device", fp_alpha)

    # Account matches, but fingerprint is unknown (e.g. timeout or error)
    screener = FakeRegistryScreener()
    screener.set_outcome(acc_hash, "account", ScreenOutcome(status="match", first_visible_at=NOW_UTC))
    screener.set_outcome(fp_hash, "device", ScreenOutcome(status="unknown", first_visible_at=None))

    feats = compute_intelligence_features(
        client=redis_client,
        run_id=RUN_ID,
        bank_id=BANK_0,
        account_id=account_id,
        decision_time=NOW_UTC,
        arrival_cutoff=NOW_UTC,
        settings=settings,
        screener=screener,
        dpip_key=DPIP_KEY,
    )

    assert feats["registry_match_count"] is None
    assert feats["registry_match_age_days"] is None


def test_no_key_configured_returns_none(redis_client: fakeredis.FakeRedis) -> None:
    """
    CRITICAL: If no DPIP secret key is configured, features evaluate honestly to None (unknown).
    """
    settings = load_account_features_settings()
    screener = FakeRegistryScreener()

    feats = compute_intelligence_features(
        client=redis_client,
        run_id=RUN_ID,
        bank_id=BANK_0,
        account_id="acc_no_key",
        decision_time=NOW_UTC,
        arrival_cutoff=NOW_UTC,
        settings=settings,
        screener=screener,
        dpip_key="",  # Empty key
    )
    assert feats["registry_match_count"] is None
    assert feats["registry_match_age_days"] is None


def test_payee_registry_match_true_false_and_none(redis_client: fakeredis.FakeRedis) -> None:
    """
    CRITICAL: Validates payee_registry_match calculation on compute_payment_features:
      - Match -> True
      - No match -> False
      - Unknown outcome -> None (unknown)
      - No screener / no key -> None (unknown)
    """
    settings = load_account_features_settings()
    payer = "acc_payer_payee_test"
    payee = "acc_payee_target"

    payee_hash = compute_identifier_hash(DPIP_KEY, RUN_ID, "account", payee, bank_index=0)
    screener = FakeRegistryScreener()

    prop = ProposedPayment(
        run_id=RUN_ID,
        bank_id=BANK_0,
        payer_account_id=payer,
        payee_bank_id=BANK_0,
        payee_vpa=payee,
        payee_type="person",
        amount_paise=50000,
        event_time=NOW_UTC,
    )

    # 1. Match -> True
    screener.set_outcome(payee_hash, "account", ScreenOutcome(status="match", first_visible_at=NOW_UTC))
    feats_true = compute_payment_features(
        redis_client, prop, NOW_UTC, settings, screener=screener, dpip_key=DPIP_KEY
    )
    assert feats_true["payee_registry_match"] is True

    # 2. No match -> False
    screener.set_outcome(payee_hash, "account", ScreenOutcome(status="no_match", first_visible_at=None))
    feats_false = compute_payment_features(
        redis_client, prop, NOW_UTC, settings, screener=screener, dpip_key=DPIP_KEY
    )
    assert feats_false["payee_registry_match"] is False

    # 3. Unknown -> None
    screener.set_outcome(payee_hash, "account", ScreenOutcome(status="unknown", first_visible_at=None))
    feats_unknown = compute_payment_features(
        redis_client, prop, NOW_UTC, settings, screener=screener, dpip_key=DPIP_KEY
    )
    assert feats_unknown["payee_registry_match"] is None

    # 4. No screener provided -> None
    feats_no_screener = compute_payment_features(redis_client, prop, NOW_UTC, settings, screener=None)
    assert feats_no_screener["payee_registry_match"] is None


def test_registry_outage_stress_layer_causes_unknown_and_none(redis_client: fakeredis.FakeRedis) -> None:
    """
    CRITICAL: Validates registry outage stress scenario (scenarios/stress/registry-outage.yaml):
      - Loads the layer and confirms bank participation drops to ~0%.
      - Serves an in-process mock registry via ASGITransport with bank_0 as a non-participant.
      - A real NationalsimBankClient receives HTTP 403 Forbidden, mapping to status 'unavailable' (match=None).
      - Features honestly degrade to None, never False or 0.
    """
    # 1. Load the stress test layer through the project's existing layer loader
    stress_file = Path("scenarios/stress/registry-outage.yaml")
    assert stress_file.is_file(), f"Missing stress scenario {stress_file}"
    layer = read_layer(stress_file)
    assert layer.name == "registry-outage"

    # Confirm it modifies probability_a_bank_takes_part_in_dpip down to ~0.0–0.01
    prob_range = layer.changes["world_ranges"]["probability_a_bank_takes_part_in_dpip"]["training"]
    assert prob_range == [0.0, 0.01]

    # 2. Build in-process registry app where bank_0 is a non-participant
    clock = ManualClock(NOW_UTC.astimezone(IST))
    app = create_app(
        clock=clock,
        key_provider=lambda: DPIP_KEY,
        settings={
            "listing_delay_minutes": 15,
            "banks": {
                BANK_0: {"dpip_participation": False},  # Non-participant under stress
                BANK_1: {"dpip_participation": True},
            },
        },
    )

    # 3. Bank client communicates with app via ASGITransport
    transport = httpx.ASGITransport(app=app)
    client = NationalsimBankClient(
        base_url="http://mock-nationalsim",
        bank_id=BANK_0,
        run_id=RUN_ID,
        transport=transport,
    )
    screener = NationalsimRegistryScreener(client)

    # 4. Screening bank_0 identifiers yields 'unknown'
    account_id = "acc_outage_01"
    acc_hash = compute_identifier_hash(DPIP_KEY, RUN_ID, "account", account_id, bank_index=0)
    outcome = screener.screen(acc_hash, "account", as_of=NOW_UTC)
    assert outcome.status == "unknown"
    assert outcome.first_visible_at is None

    # 5. Features evaluate to None (unknown), never False or 0
    settings = load_account_features_settings()
    feats = compute_account_features(
        client=redis_client,
        run_id=RUN_ID,
        bank_id=BANK_0,
        account_id=account_id,
        decision_time=NOW_UTC,
        arrival_cutoff=NOW_UTC,
        settings=settings,
        groups=["intelligence"],
        screener=screener,
        dpip_key=DPIP_KEY,
    )
    assert feats["registry_match_count"] is None
    assert feats["registry_match_age_days"] is None

    # 6. Payment feature also evaluates to None
    prop = ProposedPayment(
        run_id=RUN_ID,
        bank_id=BANK_0,
        payer_account_id=account_id,
        payee_bank_id=BANK_0,
        payee_vpa="acc_payee_outage",
        payee_type="person",
        amount_paise=10000,
        event_time=NOW_UTC,
    )
    pay_feats = compute_payment_features(
        redis_client, prop, NOW_UTC, settings, screener=screener, dpip_key=DPIP_KEY
    )
    assert pay_feats["payee_registry_match"] is None
    assert pay_feats["payee_registry_match"] is not False
