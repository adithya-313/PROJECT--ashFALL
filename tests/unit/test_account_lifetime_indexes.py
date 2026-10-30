# File: tests/unit/test_account_lifetime_indexes.py
# Purpose: Unit tests validating lifetime indexes in Redis (region, account type, device, payee-type).
# Issue: Level 15b Round 2b-2 lifetime state (A2, A3, A4, A5, A6).
"""
WHAT: Tests lifetime Redis index maintenance and querying:
  1. Repeat-safety (idempotency): reapplying records produces bitwise identical Redis state.
  2. Visibility by available_from: records are visible only when available_from <= min(decision, cutoff).
  3. Lifetime persistence: TTL for all lifetime keys is -1 (no expiration).
  4. Tenant bank isolation: bank_0 queries strictly observe only bank_0 namespace.
  5. Payee-type conflict: debit sides with conflicting payee types log a warning and preserve original.
  6. Account opened validation: account_opened missing home_region or account_type raises InvalidAccountEventError.
"""

from datetime import datetime, timedelta, timezone
from typing import Any

import fakeredis
import pytest

from features.account_keys import (
    account_type_key,
    counterparty_payee_type_key,
    device_accounts_key,
    device_changes_key,
    home_region_key,
    load_account_features_settings,
    opened_at_key,
)
from features.account_records import (
    AccountEventRecord,
    InvalidAccountEventError,
    PaymentSideRecord,
)
from features.account_summaries import (
    read_account_type,
    read_accounts_on_device,
    read_device_changes,
    read_home_region,
    read_lifetime,
    read_payee_type,
    update_account_summaries,
)

IST = timezone(timedelta(hours=5, minutes=30))
RUN_ID = "66db9aa439b092ee"
BANK_0 = "bank_0"
BANK_1 = "bank_1"
ACC_1 = "acc_person_000001"
ACC_2 = "acc_person_000002"


@pytest.fixture
def redis_client() -> fakeredis.FakeRedis:
    """Provides an isolated in-memory Redis instance."""
    return fakeredis.FakeRedis()


def _make_account_event(
    event_id: str,
    kind: str,
    event_time: datetime,
    arrival_time: datetime,
    details: dict[str, Any] | None = None,
    bank_id: str = BANK_0,
    account_id: str = ACC_1,
) -> AccountEventRecord:
    """Helper constructing a valid AccountEventRecord."""
    return AccountEventRecord(
        run_id=RUN_ID,
        bank_id=bank_id,
        account_id=account_id,
        event_id=event_id,
        kind=kind,
        event_time=event_time,
        arrival_time=arrival_time,
        details=details if details is not None else {},
    )


def _make_payment_side(
    payment_id: str,
    event_time: datetime,
    arrival_time: datetime,
    side: str = "debit",
    counterparty_bank_id: str = BANK_1,
    counterparty_vpa: str = "vpa_target",
    payee_type: str = "merchant",
    sender_device_fingerprint: str | None = "fp_phone_alpha",
    bank_id: str = BANK_0,
    account_id: str = ACC_1,
) -> PaymentSideRecord:
    """Helper constructing a valid PaymentSideRecord."""
    return PaymentSideRecord(
        run_id=RUN_ID,
        bank_id=bank_id,
        account_id=account_id,
        payment_id=payment_id,
        side=side,
        counterparty_bank_id=counterparty_bank_id,
        counterparty_vpa=counterparty_vpa,
        amount_paise=150000,
        status="success",
        failure_domain=None,
        payee_type=payee_type,
        channel="upi_app",
        sender_device_fingerprint=sender_device_fingerprint,
        sender_os_generation=1,
        event_time=event_time,
        arrival_time=arrival_time,
        balance_after_paise=400000,
    )


def test_lifetime_indexes_repeat_safety(redis_client: fakeredis.FakeRedis) -> None:
    """Verifies that reapplying opening and payment records is idempotent and repeat-safe."""
    settings = load_account_features_settings()
    t_event = datetime(2026, 10, 1, 10, 0, tzinfo=IST)
    t_arrival = datetime(2026, 10, 1, 10, 1, tzinfo=IST)

    open_ev = _make_account_event(
        f"{RUN_ID}-open-01",
        "account_opened",
        t_event,
        t_arrival,
        details={"home_region": 12, "account_type": "savings"},
    )
    pay_side = _make_payment_side(f"{RUN_ID}-p-01", t_event, t_arrival)

    # First write
    update_account_summaries(redis_client, open_ev, settings)
    update_account_summaries(redis_client, pay_side, settings)

    raw_keys: Any = redis_client.keys("*")
    keys = sorted(raw_keys)
    state_first = {k: (redis_client.type(k), redis_client.dump(k)) for k in keys}

    # Second write (repeat)
    update_account_summaries(redis_client, open_ev, settings)
    update_account_summaries(redis_client, pay_side, settings)

    state_second = {k: (redis_client.type(k), redis_client.dump(k)) for k in keys}
    assert state_first == state_second


def test_visibility_by_available_from(redis_client: fakeredis.FakeRedis) -> None:
    """Verifies that lifetime values remain hidden until min(decision_time, arrival_cutoff) >= available_from."""
    settings = load_account_features_settings()
    t_event = datetime(2026, 10, 1, 10, 0, tzinfo=IST)
    # Event arrives 2 hours later
    t_arrival = datetime(2026, 10, 1, 12, 0, tzinfo=IST)

    open_ev = _make_account_event(
        f"{RUN_ID}-open-01",
        "account_opened",
        t_event,
        t_arrival,
        details={"home_region": 25, "account_type": "current"},
    )
    update_account_summaries(redis_client, open_ev, settings)

    # Query before arrival time (11:00 IST) -> not yet visible
    t_before = datetime(2026, 10, 1, 11, 0, tzinfo=IST)
    assert read_home_region(redis_client, RUN_ID, BANK_0, ACC_1, t_before, t_before, settings) is None
    assert read_account_type(redis_client, RUN_ID, BANK_0, ACC_1, t_before, t_before, settings) is None
    lifetime_early = read_lifetime(redis_client, RUN_ID, BANK_0, ACC_1, t_before, t_before, settings)
    assert lifetime_early.opened_at is None

    # Query at/after arrival time (12:30 IST) -> visible
    t_after = datetime(2026, 10, 1, 12, 30, tzinfo=IST)
    assert read_home_region(redis_client, RUN_ID, BANK_0, ACC_1, t_after, t_after, settings) == "25"
    assert read_account_type(redis_client, RUN_ID, BANK_0, ACC_1, t_after, t_after, settings) == "current"
    lifetime_late = read_lifetime(redis_client, RUN_ID, BANK_0, ACC_1, t_after, t_after, settings)
    assert lifetime_late.opened_at == t_event


def test_lifetime_keys_have_no_ttl(redis_client: fakeredis.FakeRedis) -> None:
    """Verifies that lifetime index keys have TTL == -1 (no expiration)."""
    settings = load_account_features_settings()
    t_now = datetime(2026, 10, 1, 10, 0, tzinfo=IST)

    open_ev = _make_account_event(
        f"{RUN_ID}-open-01",
        "account_opened",
        t_now,
        t_now,
        details={"home_region": 7, "account_type": "savings"},
    )
    dev_ev = _make_account_event(
        f"{RUN_ID}-dev-01",
        "device_binding_change",
        t_now,
        t_now,
        details={"fingerprint_before": "fp_old", "fingerprint_after": "fp_new"},
    )
    pay_side = _make_payment_side(
        f"{RUN_ID}-p-01", t_now, t_now, payee_type="merchant", sender_device_fingerprint="fp_new"
    )

    update_account_summaries(redis_client, open_ev, settings)
    update_account_summaries(redis_client, dev_ev, settings)
    update_account_summaries(redis_client, pay_side, settings)

    assert redis_client.ttl(home_region_key(RUN_ID, BANK_0, ACC_1)) == -1
    assert redis_client.ttl(account_type_key(RUN_ID, BANK_0, ACC_1)) == -1
    assert redis_client.ttl(opened_at_key(RUN_ID, BANK_0, ACC_1)) == -1
    assert redis_client.ttl(device_accounts_key(RUN_ID, BANK_0, "fp_new")) == -1
    assert redis_client.ttl(counterparty_payee_type_key(RUN_ID, BANK_0, ACC_1, "debit")) == -1
    assert redis_client.ttl(device_changes_key(RUN_ID, BANK_0, ACC_1)) == -1


def test_bank_isolation_on_lifetime_indexes(redis_client: fakeredis.FakeRedis) -> None:
    """Verifies that bank_0 and bank_1 namespaces never leak across tenant boundaries."""
    settings = load_account_features_settings()
    t_now = datetime(2026, 10, 1, 10, 0, tzinfo=IST)

    # Bank 0 registers ACC_1 with home_region 1
    ev_b0 = _make_account_event(
        f"{RUN_ID}-open-b0",
        "account_opened",
        t_now,
        t_now,
        details={"home_region": 1, "account_type": "savings"},
        bank_id=BANK_0,
        account_id=ACC_1,
    )
    # Bank 1 registers ACC_2 with home_region 2
    ev_b1 = _make_account_event(
        f"{RUN_ID}-open-b1",
        "account_opened",
        t_now,
        t_now,
        details={"home_region": 2, "account_type": "current"},
        bank_id=BANK_1,
        account_id=ACC_2,
    )
    update_account_summaries(redis_client, ev_b0, settings)
    update_account_summaries(redis_client, ev_b1, settings)

    # Query bank 0 for ACC_2 -> must be None (isolation)
    assert read_home_region(redis_client, RUN_ID, BANK_0, ACC_2, t_now, t_now, settings) is None
    assert read_account_type(redis_client, RUN_ID, BANK_0, ACC_2, t_now, t_now, settings) is None

    # Shared device fingerprint between different banks:
    # Bank 0 sees ACC_1 on fp_shared; Bank 1 sees ACC_2 on fp_shared
    p0 = _make_payment_side(
        f"{RUN_ID}-p-b0", t_now, t_now, sender_device_fingerprint="fp_shared", bank_id=BANK_0, account_id=ACC_1
    )
    p1 = _make_payment_side(
        f"{RUN_ID}-p-b1", t_now, t_now, sender_device_fingerprint="fp_shared", bank_id=BANK_1, account_id=ACC_2
    )
    update_account_summaries(redis_client, p0, settings)
    update_account_summaries(redis_client, p1, settings)

    b0_accs = read_accounts_on_device(redis_client, RUN_ID, BANK_0, "fp_shared", t_now, t_now, settings)
    b1_accs = read_accounts_on_device(redis_client, RUN_ID, BANK_1, "fp_shared", t_now, t_now, settings)

    assert b0_accs == (ACC_1,)
    assert b1_accs == (ACC_2,)


def test_payee_type_conflict_logs_warning_and_preserves_first(
    redis_client: fakeredis.FakeRedis,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verifies that conflicting payee_type updates log a warning and preserve the first observed classification."""
    settings = load_account_features_settings()
    t1 = datetime(2026, 10, 1, 10, 0, tzinfo=IST)
    t2 = datetime(2026, 10, 2, 10, 0, tzinfo=IST)

    cp = "bank_1:vpa_merchant_01"
    p1 = _make_payment_side(
        f"{RUN_ID}-p-01", t1, t1, side="debit", counterparty_vpa="vpa_merchant_01", payee_type="merchant"
    )
    p2 = _make_payment_side(
        f"{RUN_ID}-p-02", t2, t2, side="debit", counterparty_vpa="vpa_merchant_01", payee_type="person"
    )

    update_account_summaries(redis_client, p1, settings)
    assert read_payee_type(redis_client, RUN_ID, BANK_0, ACC_1, "debit", cp, settings) == "merchant"

    with caplog.at_level("WARNING"):
        update_account_summaries(redis_client, p2, settings)

    # First value is preserved
    assert read_payee_type(redis_client, RUN_ID, BANK_0, ACC_1, "debit", cp, settings) == "merchant"
    # Warning was logged
    assert "already has payee_type 'merchant'; keeping first value instead of 'person'" in caplog.text


def test_account_opened_validation_missing_fields(redis_client: fakeredis.FakeRedis) -> None:
    """Verifies that account_opened lacking home_region or account_type raises InvalidAccountEventError."""
    settings = load_account_features_settings()
    t_now = datetime(2026, 10, 1, 10, 0, tzinfo=IST)

    ev_missing_region = _make_account_event(
        f"{RUN_ID}-open-bad1",
        "account_opened",
        t_now,
        t_now,
        details={"account_type": "savings"},
    )
    with pytest.raises(InvalidAccountEventError, match="missing required fields"):
        update_account_summaries(redis_client, ev_missing_region, settings)

    ev_missing_type = _make_account_event(
        f"{RUN_ID}-open-bad2",
        "account_opened",
        t_now,
        t_now,
        details={"home_region": 10},
    )
    with pytest.raises(InvalidAccountEventError, match="missing required fields"):
        update_account_summaries(redis_client, ev_missing_type, settings)


def test_first_phone_does_not_count_as_device_change(redis_client: fakeredis.FakeRedis) -> None:
    """Verifies that an account's initial phone does not register in device_changes ZSET."""
    settings = load_account_features_settings()
    t_open = datetime(2026, 9, 1, 10, 0, tzinfo=IST)
    t_pay = datetime(2026, 9, 2, 10, 0, tzinfo=IST)
    t_change = datetime(2026, 9, 15, 10, 0, tzinfo=IST)

    # 1. Opening event with initial phone
    open_ev = _make_account_event(
        f"{RUN_ID}-open-01",
        "account_opened",
        t_open,
        t_open,
        details={"home_region": 5, "account_type": "savings", "fingerprint": "fp_initial"},
    )
    update_account_summaries(redis_client, open_ev, settings)

    # Initial phone must not create an entry in device_changes
    changes = read_device_changes(redis_client, RUN_ID, BANK_0, ACC_1, t_pay, t_pay, settings)
    assert len(changes) == 0

    # 2. Payment from same phone
    p1 = _make_payment_side(f"{RUN_ID}-p-01", t_pay, t_pay, sender_device_fingerprint="fp_initial")
    update_account_summaries(redis_client, p1, settings)
    changes = read_device_changes(redis_client, RUN_ID, BANK_0, ACC_1, t_pay, t_pay, settings)
    assert len(changes) == 0

    # 3. Explicit device_binding_change
    dev_change_ev = _make_account_event(
        f"{RUN_ID}-dev-01",
        "device_binding_change",
        t_change,
        t_change,
        details={"fingerprint_before": "fp_initial", "fingerprint_after": "fp_second"},
    )
    update_account_summaries(redis_client, dev_change_ev, settings)

    t_after = t_change + timedelta(hours=1)
    changes = read_device_changes(redis_client, RUN_ID, BANK_0, ACC_1, t_after, t_after, settings)
    assert len(changes) == 1
    assert changes[0][0] == f"{RUN_ID}-dev-01"
