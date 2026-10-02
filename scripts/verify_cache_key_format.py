# File: scripts/verify_cache_key_format.py
# Purpose: Checks that the Redis keys written by update_account_state match data/schemas/feature_cache_schema.yaml.
import sys
from datetime import datetime
from pathlib import Path

import fakeredis

# Allow running as "python scripts/verify_cache_key_format.py" from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features.velocity import ALL_WINDOWS, update_account_state  # noqa: E402

ACCOUNT_ID = "verify_test_001"


def build_expected_keys(account_id: str) -> list[str]:
    """Build the key names the schema says update_account_state must write."""
    keys = [
        f"acct:first_seen:{account_id}",
        f"acct:last_loc:{account_id}",
        f"acct:last_txn_time:{account_id}",
        f"merch:history:{account_id}",
        f"dev:history:{account_id}",
    ]
    keys += [f"vel:{account_id}:{window}" for window in ALL_WINDOWS]
    return sorted(keys)


def main() -> int:
    """Write one transaction, then compare the keys in Redis with the schema-derived keys."""
    client = fakeredis.FakeRedis(decode_responses=True)
    update_account_state(
        client,
        ACCOUNT_ID,
        "txn_verify_001",
        "12.9716,77.5946",
        "merchant_42",
        "dev_id=dev_abc|os=Windows|browser=Chrome",
        datetime(2024, 1, 15, 10, 0, 0),
    )

    written = sorted(client.keys("*"))
    expected = build_expected_keys(ACCOUNT_ID)

    for key in written:
        print(f"WRITTEN KEY: {key}")
    print()
    print("EXPECTED KEYS FROM SCHEMA:")
    for key in expected:
        print(f"  {key}")
    print()

    if written == expected:
        print("RESULT: MATCH (one-to-one)")
        return 0
    print("RESULT: MISMATCH")
    print(f"  missing from written: {sorted(set(expected) - set(written))}")
    print(f"  unexpected written:   {sorted(set(written) - set(expected))}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
