"""
WHAT: The encumbrance book: a record of every lien and debit hold placed on accounts during a run, which of them
are still active, and what each account may still spend. It answers one question for the runner before every
outgoing payment piece: does an active lien or hold block this debit?

WHY: A lien or hold must stop money leaving an account without moving any money itself. If the block were done by
moving money into a special account, the bank ledgers, their hashes and the money-conservation checks would all
change. Keeping blocks in a separate book leaves the ledgers exactly as Level 11 defines them.

TERM: encumbrance: one lien or debit hold on one account, from the minute it is placed until it is released.
TERM: lien: keeps part of an account's money from being spent; e.g. balance Rs 2,000 with a Rs 500 lien leaves
      Rs 1,500 spendable.
TERM: debit hold: blocks every outgoing payment from the account, whatever the balance; incoming money still arrives.
TERM: active: placed and not yet released.
TERM: available balance: balance minus all active liens on the account; it can go below zero, e.g. balance
      Rs 300 with a Rs 500 lien gives available -Rs 200.
TERM: debit: money leaving an account. A credit is money arriving.
TERM: paise: the smallest unit of rupees; Rs 1 = 100 paise, so Rs 500 = 50,000 paise.
"""

from dataclasses import dataclass, field

from world.outside_events import OutsideEvent

# The two kinds that place an encumbrance; "release" ends one, and "resume_payment" is not an encumbrance.
PLACING_KINDS: tuple[str, ...] = ("lien", "debit_hold")


@dataclass(frozen=True)
class Encumbrance:
    """
    WHAT: One lien or debit hold as placed: its name, the account, its kind, the amount it keeps back (0 for a
    debit hold, which blocks everything) and the minute it was placed.

    WHY: Recording when and how much keeps the block auditable: a later level can show exactly which money was
    kept back, for how long, and why a payment failed.

    Args:
        encumbrance_id: Its name, e.g. "enc-000001".
        account_id: The account it applies to, e.g. "acc_person_000003".
        kind: "lien" or "debit_hold".
        amount_paise: For a lien, the paise kept back, e.g. 50000 (Rs 500); 0 for a debit hold.
        placed_at_minute: When it was placed, e.g. 1440.

    Returns:
        A frozen Encumbrance.

    Example:
        Encumbrance("enc-000001", "acc_person_000003", "lien", 50000, 1440) is a Rs 500 lien from minute 1440.
    """

    encumbrance_id: str
    account_id: str
    kind: str
    amount_paise: int
    placed_at_minute: int


@dataclass
class EncumbranceBook:
    """
    WHAT: Holds every encumbrance placed in a run, the active ones per account, and a log of every placement and
    release as (minute, action, encumbrance id). The methods place, release and answer spending questions.

    WHY: The runner asks "is this debit blocked?" for every outgoing payment piece, so active encumbrances are kept
    grouped by account and the answer for an account with none is a single quick lookup. The log feeds the Level 12
    showcase and the observation feed that a later part builds.

    Args:
        placed: Every encumbrance ever placed, by id.
        active_by_account: For each account, its active encumbrances by id.
        log: (minute, action, encumbrance id) for every placement and release, in the order they happened.

    Returns:
        A new, empty book when built with no arguments.

    Example:
        book = EncumbranceBook(); book.place(OutsideEvent(1440, "lien", "acc_a", 50000), "enc-000001") leaves
        book.log == [(1440, "place_lien", "enc-000001")].
    """

    placed: dict[str, Encumbrance] = field(default_factory=dict)
    active_by_account: dict[str, dict[str, Encumbrance]] = field(default_factory=dict)
    log: list[tuple[int, str, str]] = field(default_factory=list)

    def place(self, event: OutsideEvent, encumbrance_id: str) -> Encumbrance:
        """
        WHAT: Places the lien or debit hold that a lien or debit_hold outside event describes, under the given id,
        makes it active on its account, and writes "place_lien" or "place_debit_hold" to the log.

        WHY: Ids must be unique so a later release ends exactly one block. A debit hold keeps no amount because it
        blocks every debit; storing 0 for it keeps the available-balance sum about liens only.

        Args:
            event: A lien or debit_hold OutsideEvent that has already passed validate_outside_event.
            encumbrance_id: The name to give it, e.g. "enc-000001".

        Returns:
            The new Encumbrance.

        Example:
            place(OutsideEvent(1440, "lien", "acc_a", 50000), "enc-000001") returns
            Encumbrance("enc-000001", "acc_a", "lien", 50000, 1440).
        """
        # Only liens and debit holds place something; anything else is a caller mistake.
        if event.kind not in PLACING_KINDS or event.account_id is None:
            raise ValueError(f"only a lien or debit_hold with an account can be placed, got '{event.kind}'")
        # Two blocks with one name could not be told apart at release time.
        if encumbrance_id in self.placed:
            raise ValueError(f"encumbrance {encumbrance_id} has already been placed")
        # A debit hold blocks everything, so its amount is not used.
        amount = event.amount_paise if event.kind == "lien" else 0
        encumbrance = Encumbrance(encumbrance_id, event.account_id, event.kind, amount, event.minute)
        # Record it as placed, active on its account, and in the log.
        self.placed[encumbrance_id] = encumbrance
        self.active_by_account.setdefault(event.account_id, {})[encumbrance_id] = encumbrance
        self.log.append((event.minute, f"place_{event.kind}", encumbrance_id))
        return encumbrance

    def release(self, encumbrance_id: str, minute: int) -> Encumbrance:
        """
        WHAT: Ends one active encumbrance at the given minute and writes "release" to the log. An id that was never
        placed, or was already released, raises ValueError.

        WHY: Releasing an unknown or already ended block means the outside events disagree with each other. Saying
        so loudly is safer than ignoring it, because a missing release would keep money blocked for the whole run.

        Args:
            encumbrance_id: The block to end, e.g. "enc-000001".
            minute: When it ends, e.g. 2880.

        Returns:
            The Encumbrance that was released.

        Example:
            After place(lien, "enc-000001") at minute 1440, release("enc-000001", 2880) returns it and the log ends
            with (2880, "release", "enc-000001").
        """
        # The id must have been placed at some point.
        if encumbrance_id not in self.placed:
            raise ValueError(f"encumbrance {encumbrance_id} was never placed, so it cannot be released")
        encumbrance = self.placed[encumbrance_id]
        # It must still be active on its account.
        active_here = self.active_by_account.get(encumbrance.account_id, {})
        if encumbrance_id not in active_here:
            raise ValueError(f"encumbrance {encumbrance_id} has already been released")
        # Remove it from the active set and note the release.
        del active_here[encumbrance_id]
        self.log.append((minute, "release", encumbrance_id))
        return encumbrance

    def available_balance(self, account_id: str, balance_paise: int) -> int:
        """
        WHAT: Returns what the account may still spend: its balance minus the amounts of all active liens on it.
        Debit holds are not subtracted here; they block debits outright in blocks_debit. The result can be negative.

        WHY: A lien keeps money in the account but out of reach. Subtracting it from the balance, instead of moving
        money, leaves the ledger untouched while still stopping the money from being spent.

        Args:
            account_id: The account, e.g. "acc_a".
            balance_paise: Its current ledger balance in paise, e.g. 200000 (Rs 2,000).

        Returns:
            The available balance in paise.

        Example:
            With one active lien of 50000 on "acc_a", available_balance("acc_a", 200000) is 200000 - 50000 = 150000.
        """
        # Add up the liens still active on this account; debit holds carry no amount.
        active = self.active_by_account.get(account_id, {})
        kept_back = sum(item.amount_paise for item in active.values() if item.kind == "lien")
        return balance_paise - kept_back

    def blocks_debit(self, account_id: str, balance_paise: int, amount_paise: int) -> bool:
        """
        WHAT: Says whether a debit of amount_paise from the account is blocked: True when an active debit hold is on
        the account, or when the amount is more than the available balance (balance minus active liens).

        WHY: This is the one question the runner asks before every outgoing payment piece. It does not look at
        credits, so money arriving into a held or liened account is never blocked.

        Args:
            account_id: The paying account, e.g. "acc_a".
            balance_paise: Its current ledger balance, e.g. 200000.
            amount_paise: The debit being tried, e.g. 160000.

        Returns:
            True if the debit must fail because of a lien or hold, otherwise False.

        Example:
            With a 50000 lien on "acc_a": blocks_debit("acc_a", 200000, 160000) is True (160000 > 150000),
            blocks_debit("acc_a", 200000, 150000) is False.
        """
        # An account with no active encumbrance can never be blocked; this is the common, quick case.
        active = self.active_by_account.get(account_id)
        if not active:
            return False
        # A debit hold blocks every outgoing payment, whatever the amount.
        if any(item.kind == "debit_hold" for item in active.values()):
            return True
        # Otherwise only liens matter: the debit must fit inside what is still available.
        return amount_paise > self.available_balance(account_id, balance_paise)
