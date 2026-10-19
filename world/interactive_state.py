"""
WHAT: The runner's own lookup tables for a fraud-on run: which bank each account belongs to, which person owns each
account, and which phones each person has used. It also holds register_account, the one supported way to open an
account while a world is running, and counts the money such accounts bring in.

WHY: Level 11's run_world builds these tables once at the start and never changes them, so an account opened later
(for example a mule account a gang opens on day 12) would be unknown to it. The fraud-on runner keeps its own copy,
built with the same Level 11 helpers so it starts out identical, and updates it whenever an account is registered.

TERM: lookup table: a dictionary that answers "which X belongs to this Y" in one step, e.g. account -> bank number.
TERM: bank map: account id -> bank number, e.g. "acc_person_000003" -> 0.
TERM: register an account: open it in a bank ledger and add it to every lookup table, so payments can use it.
TERM: opening balance: money an account starts with when it is opened, in paise; it is new money in the world.
TERM: money created: the total opening balance of accounts registered by the runner; conservation checks add it to
      the world's starting total.
TERM: device state: for each person, the phone fingerprints seen so far, the count of phone changes, and the first
      phone's id; used to mark a payment as made from a new device.
TERM: paise: Rs 1 = 100 paise.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from world.calendar_runner import _build_account_bank_map
from world.config_reader import get_parameter_entry, read_world_ranges
from world.device_runtime import DeviceRuntimeState, start_device_runtime
from world.money import rupees_to_paise
from world.people import create_single_person
from world.regions import assign_region
from world.shops import (
    SHOP_OPENING_BALANCE_RUPEES,
    choose_bank_index,
    determine_shop_kind,
)


@dataclass
class RunnerState:
    """
    WHAT: The fraud-on runner's mutable lookup tables plus the running total of money created by registered accounts.
    It starts as an exact copy of what run_world would build, and only register_account changes it.

    WHY: Owning these tables lets the runner accept accounts opened mid-run without editing Level 11. Starting from
    Level 11's own helpers keeps the empty-plan run identical to run_world.

    Args:
        world: The World whose ledgers and people the tables describe.
        bank_map: Account id -> bank number.
        people_by_id: Person id -> Person.
        people_by_account: Account id -> Person (shops are not people and have no entry).
        device_state: Phones each person has used, as Level 11's DeviceRuntimeState.
        money_created_paise: Opening balances of registered accounts, in paise.
        next_person_index: Next sequential index for person account opening.
        next_shop_index: Next sequential index for shop account opening.
        opened_accounts: Forensic audit log of accounts registered mid-run.

    Returns:
        A RunnerState.

    Example:
        start_runner_state(world) for the seed-101 tiny world maps 72 accounts (60 people + 12 shops) to banks.
    """

    world: Any
    bank_map: dict[str, int]
    people_by_id: dict[str, Any]
    people_by_account: dict[str, Any]
    device_state: DeviceRuntimeState
    money_created_paise: int = 0
    next_person_index: int = 0
    next_shop_index: int = 0
    opened_accounts: list[tuple[int, str, str, str, int, int]] = field(default_factory=list)

    def register_account(
        self, account_id: str, bank_index: int, person: Any = None, opening_balance_paise: int = 0, minute: int = 0
    ) -> None:
        """
        WHAT: Opens a new account in bank number bank_index, adds it to the bank map, and, when a person owns it, adds
        that person to the people tables and records their current phone as already known. The opening balance is
        added to money_created_paise.

        WHY: Every lookup a payment needs must know the account, or the payment fails with an unclear error. Counting
        the opening balance lets money-conservation checks expect exactly the money the run added.

        Args:
            account_id: The new account's id, e.g. "acc_mule_000001".
            bank_index: The bank number, 0 to 2 in a three-bank world.
            person: The Person who owns it, or None for an account with no phone (like a shop).
            opening_balance_paise: Money it starts with, e.g. 500000 (Rs 5,000).
            minute: When it is opened, for the ledger's opening entry.

        Returns:
            None.

        Example:
            register_account("acc_mule_000001", 1, None, 500000, 0) makes bank_map["acc_mule_000001"] == 1 and adds
            500000 to money_created_paise.
        """
        # One account id can exist only once in the world.
        if account_id in self.bank_map:
            raise ValueError(f"account {account_id} is already registered")
        # Open it in the ledger (this writes an opening entry when the balance is above zero).
        self.world.ledgers[bank_index].add_account(account_id, opening_balance_paise, minute)
        self.bank_map[account_id] = bank_index
        self.money_created_paise += opening_balance_paise
        # A person-owned account also needs the person and their phone in the tables.
        if person is not None:
            self.people_by_id[person.person_id] = person
            self.people_by_account[account_id] = person
            self.device_state.known_fingerprints[person.person_id] = {person.device.fingerprint}
            self.device_state.phone_change_counts[person.person_id] = 0
            self.device_state.original_device_ids[person.person_id] = person.device.device_id

    def open_account(
        self, owner_kind: str, opening_stream: str, minute: int, rng: np.random.Generator
    ) -> str:
        """
        Creates and registers one person or shop account mid-run in standard Level 11 ID format.

        Allocates the sequential ID from the single runner counter per owner kind, constructs the
        entity with authentic attributes and opening balance, and registers the account in the ledger.
        """
        if owner_kind not in ("person", "shop"):
            raise ValueError(f"owner_kind must be 'person' or 'shop', got {owner_kind!r}")

        world = self.world
        n_regions = int(world.parameters_drawn["n_regions"])
        banks = world.banks
        range_name = str(world.parameters_drawn.get("range_name", "training"))

        if owner_kind == "person":
            person_idx = self.next_person_index
            self.next_person_index += 1
            employer_ids = [s.shop_id for s in world.shops if s.kind == "employer"]
            ranges = read_world_ranges()
            med_e = get_parameter_entry(ranges, "genuine_payment_median_amount_inr")
            sig_e = get_parameter_entry(ranges, "genuine_payment_amount_spread_sigma")
            person = create_single_person(
                person_idx, n_regions, banks, employer_ids, med_e, sig_e, rng, range_name
            )
            account_id = person.account_id
            bank_index = person.bank_index
            opening_balance_paise = person.opening_balance_paise
            self.register_account(
                account_id, bank_index, person=person, opening_balance_paise=opening_balance_paise, minute=minute
            )
        else:
            shop_idx = self.next_shop_index
            self.next_shop_index += 1
            s_id = f"shop_{shop_idx:04d}"
            _ = determine_shop_kind(shop_idx, rng)
            _ = assign_region(n_regions, rng)
            b_idx = choose_bank_index(banks, rng)
            account_id = f"acc_{s_id}"
            opening_balance_paise = rupees_to_paise(SHOP_OPENING_BALANCE_RUPEES)
            self.register_account(
                account_id, b_idx, person=None, opening_balance_paise=opening_balance_paise, minute=minute
            )
            bank_index = b_idx

        self.opened_accounts.append(
            (minute, account_id, owner_kind, opening_stream, bank_index, opening_balance_paise)
        )
        return account_id


def start_runner_state(world: Any) -> RunnerState:
    """
    WHAT: Builds a RunnerState for a world with the same Level 11 helpers run_world uses at its start: the bank map
    from the ledgers, the two people tables, and the starting device state.

    WHY: Building the tables the same way is what makes an empty fraud-on run identical to run_world: every payment
    finds the same bank and every first payment from a phone is marked new or not exactly as Level 11 marks it.

    Args:
        world: A prepared World, before any payment ran.

    Returns:
        A RunnerState with money_created_paise 0.

    Example:
        start_runner_state(prepare_world(FastRunSettings(101, 60, 12, 61, "training"), None).world).bank_map has
        72 entries.
    """
    # The same three tables and device state that run_world builds at calendar_runner.py lines 517-518.
    return RunnerState(
        world=world,
        bank_map=_build_account_bank_map(world.ledgers),
        people_by_id={person.person_id: person for person in world.people},
        people_by_account={person.account_id: person for person in world.people},
        device_state=start_device_runtime(world.people),
        next_person_index=len(world.people),
        next_shop_index=len(world.shops),
    )
