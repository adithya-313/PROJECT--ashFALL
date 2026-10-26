"""
world/phone_book.py

Post-run synthetic phone directory generator for simulated national fraud systems.
Provides pure mapping of person accounts to synthetic phone numbers, modeling
telecom number recycling and shared-household phone lines.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core.config_validator import load_yaml
from world.config_reader import draw_in_range, get_parameter_entry, read_world_ranges

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
NATIONALSIM_CONFIG_PATH: Path = REPO_ROOT / "config" / "nationalsim.yaml"

# Fixed named integer salt ensuring the phone book random stream remains independent
# from core simulation RNG children 0 to 3 and Level 12 runner streams.
PHONE_BOOK_SALT: int = 13002

# Standard minutes in one full 24-hour calendar day.
MINUTES_PER_DAY: int = 1440


@dataclass(frozen=True)
class PhoneRecord:
    """Record of a synthetic phone number assignment for an account holder.

    What it is for:
        Maintains the phone mapping for an individual person account, including
        telecom recycling history and shared household relationships.

    Inputs:
        account_id: Unique account identifier string (e.g. 'acc_person_000001').
        phone_number: Synthetic phone string with 'sim-' prefix (e.g. 'sim-0000000001').
        is_recycled: True if number was previously held by a pre-world mule account.
        previous_holder_deactivated_minute: Negative minute when previous holder disconnected, or None.
        reassigned_minute: Minute when telecom reassigned number to genuine holder, or None.
        is_shared: True if phone number is shared with a genuine household member.
        shared_with_account_id: Account ID of the genuine household member sharing this line, or None.

    What it returns:
        An immutable PhoneRecord dataclass instance.

    How it works:
        1. Stores phone identifier and associated account holder ID.
        2. Preserves temporal markers for recycled deactivation and reassignment.
        3. Preserves household sharing linkage.

    Errors and side effects:
        None. Dataclass is frozen and immutable.

    Limits:
        Applies only to person account holders; shops and institutions do not hold personal phone lines.
    """

    account_id: str
    phone_number: str
    is_recycled: bool = False
    previous_holder_deactivated_minute: int | None = None
    reassigned_minute: int | None = None
    is_shared: bool = False
    shared_with_account_id: str | None = None


@dataclass(frozen=True)
class PhoneBook:
    """Immutable phone directory generated after a world run.

    What it is for:
        Provides national intelligence feeds (I4C), DPIP screening endpoints, and dispute
        resolution mechanisms with synthetic phone numbers and telecom timeline data.

    Inputs:
        records: Mapping of account_id to its corresponding PhoneRecord.
        by_phone: Mapping of phone_number to list of PhoneRecords sharing that number.
        run_seed: Master seed of the world run.
        recycled_count: Total number of recycled phone assignments.
        shared_count: Total number of shared phone assignments.

    What it returns:
        An immutable PhoneBook dataclass instance.

    How it works:
        1. Indexes records by account identifier for fast adapter lookups.
        2. Indexes records by phone number to query multiple accounts sharing one line.
        3. Exposes reassigned_minute through public query methods for registry dispute handling.

    Errors and side effects:
        None. Dataclass is frozen.

    Limits:
        Generated strictly after world execution completes; cannot alter simulation events.
    """

    records: dict[str, PhoneRecord]
    by_phone: dict[str, list[PhoneRecord]]
    run_seed: int
    recycled_count: int
    shared_count: int

    def get_record(self, account_id: str) -> PhoneRecord | None:
        """Retrieve the phone record for an account ID, or None if unlisted."""
        return self.records.get(account_id)

    def get_records_by_phone(self, phone_number: str) -> list[PhoneRecord]:
        """Retrieve all phone records sharing a synthetic phone number."""
        return self.by_phone.get(phone_number, [])

    def get_reassigned_minute(self, phone_number: str) -> int | None:
        """Return the telecom reassignment minute for public dispute verification.

        What it is for:
            Allows the national registry to inspect telecom reassignment timing without
            accessing hidden simulation truth or private account holder names.

        Inputs:
            phone_number: Synthetic phone string (e.g. 'sim-0000000001').

        What it returns:
            Integer minute of telecom reassignment if recycled, or None otherwise.

        How it works:
            1. Scans records registered to the given phone number.
            2. Returns the reassigned_minute of any recycled record found.
            3. Returns None if the number was never recycled.

        Errors and side effects:
            None.

        Limits:
            Returns public telecom event time only.
        """
        for record in self.by_phone.get(phone_number, []):
            if record.is_recycled and record.reassigned_minute is not None:
                return record.reassigned_minute
        return None


def _load_default_phone_config() -> dict[str, Any]:
    """Load default recycling configuration from config/nationalsim.yaml.

    What it is for:
        Reads the mandatory minimum quarantine days and assumed recycled share from
        the national simulation configuration file.

    Inputs:
        None. Reads config/nationalsim.yaml relative to repository root.

    What it returns:
        Dictionary containing recycle_min_days and recycled_number_share values.

    How it works:
        1. Read config/nationalsim.yaml from disk.
        2. Extract nationalsim section.
        3. Parse recycle_min_days and recycled_number_share value fields.

    Errors and side effects:
        Raises ValueError if configuration file or expected sections are missing.

    Limits:
        File read on invocation; does not monitor file for updates.
    """
    if not NATIONALSIM_CONFIG_PATH.is_file():
        raise ValueError(f"Missing configuration file '{NATIONALSIM_CONFIG_PATH}'.")
    raw = load_yaml(NATIONALSIM_CONFIG_PATH)
    ns = raw.get("nationalsim", {})
    recycle_min_days_block = ns.get("recycle_min_days", {})
    recycled_share_block = ns.get("recycled_number_share", {})
    return {
        "recycle_min_days": int(recycle_min_days_block.get("value", 90)),
        "recycled_number_share": float(recycled_share_block.get("value", 0.05)),
    }


def build_phone_book(
    output: Any,
    run_seed: int,
    phone_config: dict[str, Any] | None = None,
) -> PhoneBook:
    """Build an immutable synthetic phone book from completed world output.

    What it is for:
        Constructs synthetic telephone number assignments for all person account holders
        after simulation completion. Simulates telecom mobile number recycling (numbers
        deactivated by pre-world mules and reassigned after the statutory 90-day quarantine)
        and shared household phones (where an illicit mule and an innocent genuine household
        member share a single line, causing unintended collateral registry matches).

    Inputs:
        output: Completed InteractiveWorldOutput (or World) structure from the simulation run.
        run_seed: Master integer seed of the world run.
        phone_config: Optional configuration overrides for recycling and sharing rates.

    What it returns:
        An immutable PhoneBook holding all account phone mappings and telecom history.

    How it works:
        1. Initializes an independent NumPy random generator using run_seed and PHONE_BOOK_SALT
           via SeedSequence to ensure frozen anchors (children 0-3) remain completely untouched.
        2. Resolves configuration bounds for recycle quarantine days, recycled number share,
           and shared mule account share from config/nationalsim.yaml and config/world_ranges.yaml.
        3. Collects all person account holders from output.world.people in deterministic order.
        4. Identifies mule accounts from hidden truth roles if available, distinguishing genuine
           accounts from illicit mule accounts.
        5. Assigns baseline unique synthetic phone numbers formatted as 'sim-0000000001'.
        6. Allocates shared household phone numbers: a configured share of mule accounts share
           their phone number with a genuine household member, creating realistic collateral matches.
        7. Allocates recycled phone numbers: a configured share of genuine people hold numbers
           previously held by pre-world mules that were deactivated at least recycle_min_days
           before day 0 and reassigned before day 0.
        8. Indexes records by account identifier and phone number, returning a frozen PhoneBook.

    Errors and side effects:
        Raises ValueError if run_seed is invalid or output does not contain person accounts.
        Does not mutate output, RNG streams, or world ledgers.

    Limits:
        Phone numbers are generated post-run; they do not influence simulated transaction paths.
    """
    # 1. Derive an isolated random generator stream without disturbing world anchors
    seed_sequence = np.random.SeedSequence([int(run_seed), PHONE_BOOK_SALT])
    rng = np.random.default_rng(seed_sequence)

    # 2. Resolve configuration parameters
    default_cfg = _load_default_phone_config()
    cfg = dict(default_cfg)
    if phone_config is not None:
        cfg.update(phone_config)

    recycle_min_days: int = int(cfg.get("recycle_min_days", 90))
    recycled_number_share: float = float(cfg.get("recycled_number_share", 0.05))

    # Resolve shared mule account share from config or world_ranges
    if "identifier_shared_across_mule_accounts_share" in cfg:
        shared_mule_share = float(cfg["identifier_shared_across_mule_accounts_share"])
    elif "shared_mule_share" in cfg:
        shared_mule_share = float(cfg["shared_mule_share"])
    else:
        world_ranges = None
        if hasattr(output, "effective_config") and isinstance(output.effective_config, dict):
            world_ranges = output.effective_config.get("world_configs", {}).get("world_ranges")
        if world_ranges is None:
            world_ranges = read_world_ranges()
        range_name = "training"
        if hasattr(output, "effective_config") and isinstance(output.effective_config, dict):
            range_name = output.effective_config.get("settings", {}).get("range_name", "training")
        param_entry = get_parameter_entry(world_ranges, "identifier_shared_across_mule_accounts_share")
        shared_mule_share = draw_in_range(
            param_entry,
            rng,
            range_name=range_name,
            param_name="identifier_shared_across_mule_accounts_share",
        )

    # 3. Extract person account holders
    world = getattr(output, "world", output)
    people = getattr(world, "people", [])
    person_account_ids: list[str] = [p.account_id for p in people]

    # Include mid-run opened person accounts if present
    result = getattr(output, "result", None)
    if result is not None and hasattr(result, "opened_accounts"):
        for opened_item in result.opened_accounts:
            # opened_item structure: (minute, account_id, owner_kind, ...)
            if len(opened_item) >= 3 and opened_item[2] == "person":
                acc_id = opened_item[1]
                if acc_id not in person_account_ids:
                    person_account_ids.append(acc_id)

    person_account_ids.sort()

    # 4. Identify mule accounts versus genuine accounts
    mule_account_ids: set[str] = set()
    if result is not None and hasattr(result, "truth") and hasattr(result.truth, "roles"):
        for role_span in result.truth.roles:
            if role_span.role in ("mule", "collector", "cash_out"):
                mule_account_ids.add(role_span.account_id)

    genuine_accounts: list[str] = [acc for acc in person_account_ids if acc not in mule_account_ids]
    mule_person_accounts: list[str] = [acc for acc in person_account_ids if acc in mule_account_ids]

    # 5. Baseline synthetic phone assignment: each person receives sim-0000000001, etc.
    records: dict[str, PhoneRecord] = {}
    for index, acc_id in enumerate(person_account_ids, start=1):
        phone_str = f"sim-{index:010d}"
        records[acc_id] = PhoneRecord(
            account_id=acc_id,
            phone_number=phone_str,
            is_recycled=False,
            previous_holder_deactivated_minute=None,
            reassigned_minute=None,
            is_shared=False,
            shared_with_account_id=None,
        )

    # 6. Apply shared household phones
    # In fraud investigations, "shared" denotes a phone line shared with an innocent household member.
    # When the mule account is flagged in national databases, the innocent family member's account
    # triggers a collateral match on the shared phone number during registry screening.
    if shared_mule_share > 0.0 and mule_person_accounts and genuine_accounts:
        n_shared = int(np.round(len(mule_person_accounts) * shared_mule_share))
        n_shared = max(1, min(n_shared, len(mule_person_accounts)))
        chosen_mules = [str(x) for x in rng.choice(mule_person_accounts, size=n_shared, replace=False)]
        chosen_genuine = [str(x) for x in rng.choice(genuine_accounts, size=n_shared, replace=True)]

        for mule_acc, genuine_acc in zip(chosen_mules, chosen_genuine, strict=True):
            shared_phone = records[genuine_acc].phone_number
            records[mule_acc] = PhoneRecord(
                account_id=mule_acc,
                phone_number=shared_phone,
                is_recycled=False,
                previous_holder_deactivated_minute=None,
                reassigned_minute=None,
                is_shared=True,
                shared_with_account_id=genuine_acc,
            )

    # 7. Apply recycled phone numbers
    # A recycled number belonged to a previous mule account disconnected prior to day 0.
    # Under Supreme Court of India directives (30 Oct 2023), telecom operators cannot reallocate
    # a disconnected number for at least 90 days. Thus, deactivation and reassignment occur
    # strictly in pre-world negative minutes, separated by at least recycle_min_days * 1440.
    if recycled_number_share > 0.0 and genuine_accounts:
        n_recycled = int(np.round(len(genuine_accounts) * recycled_number_share))
        n_recycled = max(1, min(n_recycled, len(genuine_accounts)))
        chosen_genuine_recycled = [str(x) for x in rng.choice(genuine_accounts, size=n_recycled, replace=False)]

        for genuine_acc in chosen_genuine_recycled:
            # Sample deactivation between recycle_min_days and (recycle_min_days + 90) days before day 0
            deactivated_days = int(rng.integers(recycle_min_days, recycle_min_days + 90))
            deactivated_minute = -deactivated_days * MINUTES_PER_DAY

            # Quarantine delay must satisfy at least recycle_min_days
            delay_days = int(rng.integers(recycle_min_days, deactivated_days + 1))
            reassigned_minute = deactivated_minute + (delay_days * MINUTES_PER_DAY)

            current_rec = records[genuine_acc]
            records[genuine_acc] = PhoneRecord(
                account_id=genuine_acc,
                phone_number=current_rec.phone_number,
                is_recycled=True,
                previous_holder_deactivated_minute=deactivated_minute,
                reassigned_minute=reassigned_minute,
                is_shared=current_rec.is_shared,
                shared_with_account_id=current_rec.shared_with_account_id,
            )

    # 8. Index records by phone number and compute counts
    by_phone: dict[str, list[PhoneRecord]] = {}
    for record in records.values():
        by_phone.setdefault(record.phone_number, []).append(record)

    recycled_count = sum(1 for r in records.values() if r.is_recycled)
    shared_count = sum(1 for r in records.values() if r.is_shared)

    return PhoneBook(
        records=records,
        by_phone=by_phone,
        run_seed=run_seed,
        recycled_count=recycled_count,
        shared_count=shared_count,
    )
