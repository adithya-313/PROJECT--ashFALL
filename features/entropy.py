# File: features/entropy.py
# Purpose: Merchant-entropy and device-ratio session features computed from an account's recent history.
#
# These features measure how VARIED an account's behaviour is. Normal people shop at a mix of places on their own
# devices. Fraud often looks different: a stolen card hammering the same merchant, or one account suddenly
# appearing from many different devices (or many accounts sharing one device, in a mule ring).
import logging
import math

logger = logging.getLogger(__name__)

DEVICE_ID_PREFIX = "dev_id="


def compute_merchant_entropy(merchant_ids: list[str]) -> float:
    """
    Return the Shannon entropy (base 2, in bits) of how an account's transactions are spread across merchants.

    Why this function exists: entropy is a single number for "how varied is this?". A fraudster who has stolen a
    card often hits the same merchant over and over (card testing, or cashing out through one outlet), which gives
    LOW entropy. Normal behaviour is spread across many merchants (groceries, fuel, restaurants), which gives
    HIGHER entropy. A sudden drop in entropy, or an unusual value, helps the model spot a change in behaviour.

    Args:
        merchant_ids: the merchant IDs of the account's recent transactions, one entry per transaction, repeats
            included (for example ["A", "A", "B"]).

    Returns:
        Entropy in bits as a float, 0.0 or more. 0.0 means "all the same merchant". Two equally likely merchants
        give 1.0 bit, four give 2.0 bits, and so on (log2 of the number of equally likely merchants).

    Edge cases:
        - Empty list: returns 0.0 instead of raising an error. A brand-new account simply has no history yet, which
          is normal, not a mistake. The caller treats 0.0 as "no history available".
        - A single repeated merchant also gives 0.0 (no variety), so 0.0 alone cannot tell "no history" from "only
          one merchant". Callers who need to tell them apart should check the list length separately.
    """
    # NOTE: merchant_id is used as a proxy for merchant category because no merchant category field exists in the
    # schema. Entropy over individual merchants is a slightly stricter measure of variety than entropy over
    # categories, but it captures the same idea.
    if not merchant_ids:
        return 0.0

    # Step 1 of 3 - count occurrences: tally how many times each distinct merchant appears. This turns the raw
    # list into "merchant A: 5 times, merchant B: 1 time", which is what a distribution is built from.
    counts: dict[str, int] = {}
    for merchant_id in merchant_ids:
        counts[merchant_id] = counts.get(merchant_id, 0) + 1

    # Step 2 of 3 - compute probability: the share of all transactions that went to each merchant is
    # count / total, so the shares add up to 1. We need the total to divide by.
    total = len(merchant_ids)

    # Step 3 of 3 - apply -p * log2(p) to each merchant and add them up. Intuition: a rare merchant (small p) is a
    # "surprising" visit and contributes a lot of information; a very common one contributes little. Summing the
    # surprise weighted by how often it happens gives the average surprise per transaction, which is the entropy.
    # The minus sign is there because log2(p) is negative for p < 1, so it makes the total positive. Every p here
    # is above 0 because it comes from a merchant that was actually seen, so log2 never receives zero.
    # abs() only avoids returning -0.0 when there is a single merchant (p = 1, log2(1) = 0); it never changes a
    # real value because entropy cannot be negative.
    return abs(-sum((count / total) * math.log2(count / total) for count in counts.values()))


def parse_device_id(device_fingerprint: str) -> str:
    """
    Extract the device identifier from a device fingerprint string.

    Why this function exists: the API sends one combined string, "dev_id=<value>|os=<value>|browser=<value>". The
    OS and browser can change with every update, but the dev_id identifies the physical device. Fraud checks care
    about devices ("is this account suddenly on many new devices?"), so we isolate just that part.

    Args:
        device_fingerprint: the full fingerprint text, with segments separated by "|".

    Returns:
        The text after "dev_id=" up to the next "|" (or the end of the string).

    Edge cases:
        - Raises ValueError if no segment starts with "dev_id=". We fail loudly because a fingerprint with no
          device ID is malformed data, and guessing a value would corrupt the device history.
        - The dev_id segment may appear in any position, not only first.
        - An empty value ("dev_id=") returns an empty string; the segment was found, it just has no value.
    """
    # Split on "|" so each key=value segment can be checked on its own. Checking the start of each segment, rather
    # than searching the whole string, avoids false matches such as "other_dev_id=" inside another value.
    for segment in device_fingerprint.split("|"):
        if segment.startswith(DEVICE_ID_PREFIX):
            # Everything after the "dev_id=" prefix is the identifier.
            return segment[len(DEVICE_ID_PREFIX):]

    # We checked every segment and found no dev_id, so the input is malformed.
    raise ValueError(f"No 'dev_id=' segment found in device_fingerprint: {device_fingerprint!r}")


def compute_device_ratio(device_fingerprints: list[str]) -> float:
    """
    Return unique device IDs divided by transaction count, or 0.0 for an empty list.

    Skipped entries (unparseable fingerprints) are excluded from the unique-device count but still count toward the
    total transaction denominator, because an unparseable fingerprint is still a transaction.

    Why this function exists: the ratio shows how many different devices an account uses relative to how often it
    transacts. A ratio near 0 means "same device every time" (normal). A ratio near 1 means almost every
    transaction came from a different device, which is a classic sign of account takeover, credential stuffing or an
    account being operated by several people.

    Args:
        device_fingerprints: the raw fingerprint strings of the account's recent transactions, one per transaction.

    Returns:
        A float between 0.0 and 1.0: (number of distinct dev_ids) / (number of transactions in the list).

    Edge cases:
        - Empty list: returns 0.0 (no history, so no device variety to report).
        - Unparseable fingerprints are skipped with a logged warning, so one bad record cannot stop the feature
          from being computed. They still count in the denominator. A missing fingerprint is still a transaction
          event, so removing it from the denominator would inflate the ratio (for example 1 unique device over 1
          counted transaction = 1.0 even though there were 2 transactions) and distort the signal.
    """
    # With no transactions there is nothing to divide by, and no device variety to report.
    if not device_fingerprints:
        return 0.0

    # Pull the dev_id out of every fingerprint. A fingerprint we cannot parse is logged and skipped rather than
    # raising, so a single corrupt record does not make the whole feature unavailable.
    device_ids = []
    for fingerprint in device_fingerprints:
        try:
            device_ids.append(parse_device_id(fingerprint))
        except ValueError as exc:
            logger.warning("Skipping unparseable device_fingerprint: %s", exc)

    # A set keeps each device ID once, so its length is the number of distinct devices. The denominator is the
    # original list length, NOT the number of parsed IDs, so skipped entries still count as transactions.
    return len(set(device_ids)) / len(device_fingerprints)
