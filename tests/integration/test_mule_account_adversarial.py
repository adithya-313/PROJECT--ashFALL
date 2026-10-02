# File: tests/integration/test_mule_account_adversarial.py
# Purpose: Verifies velocity windows count a money-mule burst (12 small inbound txns in 24h) correctly.
from datetime import datetime, timedelta

import fakeredis

from features.velocity import compute_all_velocities, record_transaction

# Mule account pattern: a mule receives many small inbound transfers within 24 hours, then sends one large
# outbound payment. The classic fraud signal is the 24h transaction count, so the velocity pipeline must count
# every one of those inbound transactions. If velocity_24h is not 12 here, the window logic is broken.


def test_mule_account_twelve_small_inbound_in_24h():
    client = fakeredis.FakeRedis(decode_responses=True)
    fake_account_id = "mule_acct_test_001"
    base_time = datetime(2024, 1, 15, 10, 0, 0)

    # 12 transactions, 90 minutes apart: offsets 0h, 1.5h, ... 16.5h from base_time.
    for i in range(12):
        record_transaction(client, fake_account_id, f"txn_{i + 1:04d}", base_time + timedelta(minutes=90 * i))

    # 23h after base_time: the 24h window starts at base_time - 1h, so all 12 transactions are inside it.
    now = base_time + timedelta(hours=23, minutes=0)
    velocities = compute_all_velocities(client, fake_account_id, now)

    assert velocities["velocity_24h"] == 12

    # The last transaction was at +16.5h, i.e. 6.5h before now. That is outside the 6h window (starts at +17h)
    # and outside the 1h window, so both counts are exactly 0.
    assert velocities["velocity_6h"] == 0
    assert velocities["velocity_1h"] == 0
