"""
WHAT: Records and configuration helpers for complaints as label events (Level 12e, round E-1, amendment A-020).
Defines the label event a bank would receive, the inputs the complaint planners work from (victim losses and
genuine payments), the per-world settings drawn once per world, and the readers for config/labels.yaml and the
complaint-delay ranges of the run's own world ranges.

WHY: A label event must carry only what a bank could know, so the record has no field that says whether the
complaint is true; evaluation finds that later by matching payment ids against truth. Keeping the settings and
bounds logic here keeps the planners in world/complaints.py pure functions over plain data.

TERM: label event: a message a bank receives saying a payment was fraud, made some minutes or days after the payment.
TERM: claimed type: the scam type the complainant says happened; it is a claim and is never checked on the event.
TERM: layered world ranges: the world_ranges dictionary of a run after stress layers were applied to it.
TERM: stacking: several victims complaining about the same receiving account.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core.config_validator import load_yaml
from world.config_reader import draw_in_range, get_parameter_entry
from world.device_schedule import MINUTES_PER_DAY
from world.scam_config import resolve_remainder_scam_type
from world.scam_planner import SCAM_INVESTMENT, SCAM_JOB
from world.truth_records import load_truth_config

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
LABELS_CONFIG_PATH: Path = REPO_ROOT / "config" / "labels.yaml"

LABEL_KIND_COMPLAINT: str = "complaint"
LABEL_KINDS: tuple[str, ...] = (LABEL_KIND_COMPLAINT,)
DELAY_UNIT_MINUTES: str = "minutes"
DELAY_UNIT_DAYS: str = "days"
DELAY_UNITS: tuple[str, ...] = (DELAY_UNIT_MINUTES, DELAY_UNIT_DAYS)
REGISTRY_REPORT_KIND: str = "registry_report"
RANGE_SPLITS: tuple[str, ...] = ("training", "test", "stress")
# Key inside complaint_delay_by_scam_type that documents the block; it is not a scam type.
DELAY_BLOCK_DOCUMENTATION_KEY: str = "reason"


@dataclass(frozen=True)
class LabelEvent:
    """
    WHAT: One complaint as a bank receives it: when it arrives, who complains, which account is reported, the
    payments it names and the scam type the complainant claims.

    WHY: The bank feed must not reveal truth, so no field says whether the complaint is right. label_id is empty
    until assign_label_ids numbers the events of a run.

    Args:
        label_id: Run-scoped id such as "<run_id>-L0001"; empty before numbering.
        kind: One of LABEL_KINDS.
        arrival_minute: Simulation minute at which the bank receives the complaint (may lie after the world ends).
        complainant_account: Account of the person who complains.
        complainant_bank_index: Bank index of the complainant's account.
        reported_account: Account the complaint names as the receiver of the bad payments.
        reported_bank_index: Bank index of the reported account.
        payment_ids: Ids of the payments the complaint names; never empty.
        claimed_type: The scam type the complainant claims.

    Raises:
        ValueError: If kind is unknown, payment_ids is empty or arrival_minute is negative.

    Example:
        LabelEvent("", "complaint", 5000, "acc_a", 0, "acc_b", 1, ("p1",), "investment")
    """

    label_id: str
    kind: str
    arrival_minute: int
    complainant_account: str
    complainant_bank_index: int
    reported_account: str
    reported_bank_index: int
    payment_ids: tuple[str, ...]
    claimed_type: str

    def __post_init__(self) -> None:
        """Rejects an unknown kind, an empty payment list or a negative arrival minute when the event is built."""
        if self.kind not in LABEL_KINDS:
            raise ValueError(f"LabelEvent kind '{self.kind}' must be one of {LABEL_KINDS}")
        if len(self.payment_ids) == 0:
            raise ValueError("LabelEvent payment_ids must not be empty")
        if self.arrival_minute < 0:
            raise ValueError(f"LabelEvent arrival_minute {self.arrival_minute} must not be negative")


@dataclass(frozen=True)
class VictimLoss:
    """
    WHAT: What one scam victim lost to one receiving account: the successful victim payments, in event order, and
    the minute of the last one.

    WHY: A victim complains at most once per receiving account, measured from the last successful payment, so the
    planner needs exactly these facts and nothing else.

    Args:
        victim_account: The victim's account.
        victim_bank_index: Bank index of the victim's account.
        mule_account: The receiving account the victim paid.
        mule_bank_index: Bank index of the receiving account.
        scam_type: The scam type of the victim's schedule.
        payment_ids: Ids of the successful victim payments to the receiving account, in event order; never empty.
        last_paid_minute: Minute of the last successful victim payment.

    Raises:
        ValueError: If payment_ids is empty or last_paid_minute is negative.

    Example:
        VictimLoss("acc_v", 0, "acc_m", 1, "investment", ("p1", "p2"), 12000)
    """

    victim_account: str
    victim_bank_index: int
    mule_account: str
    mule_bank_index: int
    scam_type: str
    payment_ids: tuple[str, ...]
    last_paid_minute: int

    def __post_init__(self) -> None:
        """Rejects an empty payment list or a negative last payment minute when the loss is built."""
        if len(self.payment_ids) == 0:
            raise ValueError("VictimLoss payment_ids must not be empty (a victim with no successful payment lost)")
        if self.last_paid_minute < 0:
            raise ValueError(f"VictimLoss last_paid_minute {self.last_paid_minute} must not be negative")


@dataclass(frozen=True, slots=True)
class GenuinePayment:
    """
    WHAT: One successful ordinary payment that could draw a false complaint, with the facts the false-complaint
    planner needs: the payer, the payee and whether the payee is a shop.

    WHY: Payments to shops draw false complaints more often than payments to people (issue A3), so the planner needs
    the payee kind. Slots keep a few hundred thousand of these small in memory.

    Args:
        payment_id: Id of the payment.
        minute: Minute of the payment.
        payer_account: Account that paid; it is the complainant of a false complaint.
        payer_bank_index: Bank index of the payer.
        payee_account: Account that was paid; it is the reported account.
        payee_bank_index: Bank index of the payee.
        payee_is_shop: True when the payee is a shop account.

    Example:
        GenuinePayment("p9", 800, "acc_a", 0, "acc_shop_1", 2, True)
    """

    payment_id: str
    minute: int
    payer_account: str
    payer_bank_index: int
    payee_account: str
    payee_bank_index: int
    payee_is_shop: bool


@dataclass(frozen=True)
class RegistryReport:
    """
    WHAT: One report a bank makes to the national registry about an account, as written in the Level 12e contract
    in docs/interfaces/national_systems.md. The field names and their order are the contract.

    WHY: Level 13 reads these events. Like a label event, a report never says whether the account really is a mule,
    so every wrong registry entry traces back to a false complaint and to nothing else.

    Args:
        report_id: Run-scoped id such as "<run_id>-R000001".
        kind: Always REGISTRY_REPORT_KIND.
        reported_minute: When the report reaches the registry (confirmed_minute plus the drawn delay).
        reporting_bank_index: Bank index of the reported account.
        account_id: The reported account.
        device_fingerprint: The holder's device fingerprint for a Level 11 person; None for shops and accounts
            opened mid-run.
        confirmed_minute: Arrival minute of the first complaint naming the account.
        complaint_ids: Ids of every complaint naming the account that arrived at or before confirmed_minute.

    Raises:
        ValueError: If kind is wrong, complaint_ids is empty, or reported_minute is before confirmed_minute.

    Example:
        RegistryReport("r-R000001", "registry_report", 4321, 1, "acc_a", None, 3900, ("r-L000003",))
    """

    report_id: str
    kind: str
    reported_minute: int
    reporting_bank_index: int
    account_id: str
    device_fingerprint: str | None
    confirmed_minute: int
    complaint_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        """Rejects a wrong kind, an empty complaint list, or a report that arrives before its confirmation."""
        if self.kind != REGISTRY_REPORT_KIND:
            raise ValueError(f"RegistryReport kind '{self.kind}' must be '{REGISTRY_REPORT_KIND}'")
        if len(self.complaint_ids) == 0:
            raise ValueError("RegistryReport complaint_ids must not be empty")
        if self.reported_minute < self.confirmed_minute:
            raise ValueError(
                f"RegistryReport reported_minute {self.reported_minute} is before confirmed_minute "
                f"{self.confirmed_minute}"
            )


@dataclass(frozen=True)
class LabelWorldSettings:
    """
    WHAT: The three numbers drawn once per world that steer its complaints: the share of victims who complain, the
    false-complaint rate per 100,000 genuine payments and the merchant weight.

    WHY: Drawing them once per world, inside the split's range, gives each world its own complaint behaviour.

    Args:
        victim_complaint_share: Chance that a victim complains, between 0 and 1.
        false_complaint_rate_per_100000: Expected false complaints per 100,000 successful genuine payments (>= 0).
        false_complaint_merchant_weight: How many times likelier a shop payment is to draw a false complaint than a
            payment to a person (> 0).

    Raises:
        ValueError: If a value is outside the ranges above.

    Example:
        LabelWorldSettings(0.4, 1.0, 5.0)
    """

    victim_complaint_share: float
    false_complaint_rate_per_100000: float
    false_complaint_merchant_weight: float

    def __post_init__(self) -> None:
        """Rejects a share outside [0, 1], a negative rate or a merchant weight that is not above zero."""
        if not 0.0 <= self.victim_complaint_share <= 1.0:
            raise ValueError(f"victim_complaint_share {self.victim_complaint_share} must be between 0 and 1")
        if self.false_complaint_rate_per_100000 < 0.0:
            raise ValueError(f"false_complaint_rate_per_100000 {self.false_complaint_rate_per_100000} must be >= 0")
        if self.false_complaint_merchant_weight <= 0.0:
            raise ValueError(f"false_complaint_merchant_weight {self.false_complaint_merchant_weight} must be > 0")


def load_labels_config(path: Path | None = None) -> dict[str, Any]:
    """
    WHAT: Reads config/labels.yaml (or the given file), validates the `labels` block and returns it.

    WHY: Every label function takes this block as an argument, so a malformed file fails once, here, with a plain
    message instead of deep inside a planner.

    Args:
        path: Optional YAML path; defaults to config/labels.yaml.

    Returns:
        The validated `labels` block as a dictionary.

    Raises:
        ValueError: If the file has no `labels` block or the block fails validate_labels_config.

    Example:
        load_labels_config()["rate_check_standard_errors"] == 4
    """
    config_path = path if path is not None else LABELS_CONFIG_PATH
    data = load_yaml(config_path)
    if not isinstance(data, dict) or "labels" not in data:
        raise ValueError(f"{config_path} must contain a top-level 'labels' block")
    block = data["labels"]
    validate_labels_config(block)
    return block  # type: ignore[no-any-return]


def delay_scam_types(config: dict[str, Any]) -> list[str]:
    """
    WHAT: Lists the scam-type keys of complaint_delay_by_scam_type: every key except the documentation key `reason`.

    WHY: The block mixes one entry per scam type with a `reason` string, so the validator and the bounds reader must
    agree on which keys are scam types.

    Args:
        config: The `labels` block.

    Returns:
        The scam-type keys in file order.

    Raises:
        ValueError: If complaint_delay_by_scam_type is missing or not a mapping.

    Example:
        delay_scam_types(load_labels_config()) == ["investment", "job", "unauthorised_debit"]
    """
    if "complaint_delay_by_scam_type" not in config:
        raise ValueError("labels config is missing 'complaint_delay_by_scam_type'")
    delay_block = config["complaint_delay_by_scam_type"]
    if not isinstance(delay_block, dict):
        raise ValueError("labels config 'complaint_delay_by_scam_type' must be a mapping")
    return [key for key in delay_block if key != DELAY_BLOCK_DOCUMENTATION_KEY]


def _require_reference(block: Any, where: str) -> None:
    """Checks a block names its world_ranges entry with a non-empty string; raises ValueError naming `where`."""
    if not isinstance(block, dict) or "world_ranges_reference" not in block:
        raise ValueError(f"labels config '{where}' must have a 'world_ranges_reference'")
    reference = block["world_ranges_reference"]
    if not isinstance(reference, str) or reference == "":
        raise ValueError(f"labels config '{where}' world_ranges_reference must be a non-empty string")


def validate_labels_config(config: dict[str, Any]) -> None:
    """
    WHAT: Checks the `labels` block: every required key is present, the delay keys (all but `reason`) equal exactly
    the scam types the planner can draw, units are minutes or days, and the check settings are positive numbers.

    WHY: A missing or extra scam type would silently skip delays or leave a type with no range; failing at load time
    keeps labels honest. Required keys are never defaulted.

    Args:
        config: The `labels` block of config/labels.yaml.

    Raises:
        ValueError: On the first problem found, naming the key.

    Example:
        validate_labels_config(load_labels_config())  # returns None when valid
    """
    required_keys = (
        "complaint_delay_by_scam_type", "victim_complaint_share", "false_complaint_rate_per_100000",
        "false_complaint_merchant_weight", "rate_check_min_sample", "rate_check_standard_errors",
    )
    for key in required_keys:
        if key not in config:
            raise ValueError(f"labels config is missing required key '{key}'")

    expected_types = {SCAM_INVESTMENT, SCAM_JOB, resolve_remainder_scam_type()}
    found_types = delay_scam_types(config)
    if set(found_types) != expected_types or len(found_types) != len(expected_types):
        raise ValueError(
            f"complaint_delay_by_scam_type keys {sorted(found_types)} must equal the scam types "
            f"{sorted(expected_types)}"
        )
    for scam_type in found_types:
        delay_block = config["complaint_delay_by_scam_type"][scam_type]
        _require_reference(delay_block, f"complaint_delay_by_scam_type.{scam_type}")
        if "unit" not in delay_block or delay_block["unit"] not in DELAY_UNITS:
            raise ValueError(
                f"complaint_delay_by_scam_type.{scam_type} unit must be one of {DELAY_UNITS}, "
                f"got {delay_block.get('unit')!r}"
            )
    for key in ("victim_complaint_share", "false_complaint_rate_per_100000", "false_complaint_merchant_weight"):
        _require_reference(config[key], key)

    min_sample = config["rate_check_min_sample"]
    if isinstance(min_sample, bool) or not isinstance(min_sample, int) or min_sample <= 0:
        raise ValueError(f"rate_check_min_sample must be a positive integer, got {min_sample!r}")
    standard_errors = config["rate_check_standard_errors"]
    if isinstance(standard_errors, bool) or not isinstance(standard_errors, (int, float)) or standard_errors <= 0:
        raise ValueError(f"rate_check_standard_errors must be a positive number, got {standard_errors!r}")
    _validate_registry_and_bulk_keys(config)


def _validate_registry_and_bulk_keys(config: dict[str, Any]) -> None:
    """
    Checks the round E-2 keys of the labels block: the bank reporting share and bulk references name a world_ranges
    entry, registry_report_delay_minutes has a minutes pair with 0 <= low <= high for every split, and
    mule_side_roles is a non-empty list of roles that exist in config/truth_store.yaml. Raises ValueError otherwise.
    """
    required_keys = (
        "bank_registry_report_share", "registry_report_delay_minutes", "bulk_confirmation_events_per_year",
        "bulk_confirmation_size_accounts", "bulk_confirmation_delay_days", "mule_side_roles",
    )
    for key in required_keys:
        if key not in config:
            raise ValueError(f"labels config is missing required key '{key}'")
    for key in ("bank_registry_report_share", "bulk_confirmation_events_per_year",
                "bulk_confirmation_size_accounts", "bulk_confirmation_delay_days"):
        _require_reference(config[key], key)

    delay_block = config["registry_report_delay_minutes"]
    if not isinstance(delay_block, dict) or delay_block.get("unit") != DELAY_UNIT_MINUTES:
        raise ValueError(f"registry_report_delay_minutes must be a mapping with unit '{DELAY_UNIT_MINUTES}'")
    for split in RANGE_SPLITS:
        if split not in delay_block:
            raise ValueError(f"registry_report_delay_minutes has no '{split}' pair")
        pair = delay_block[split]
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"registry_report_delay_minutes '{split}' must be a pair, got {pair!r}")
        if pair[0] < 0 or pair[0] > pair[1]:
            raise ValueError(f"registry_report_delay_minutes '{split}' must satisfy 0 <= low <= high, got {pair!r}")

    mule_side_roles = config["mule_side_roles"]
    if not isinstance(mule_side_roles, list) or len(mule_side_roles) == 0:
        raise ValueError("mule_side_roles must be a non-empty list of role names")
    truth_roles = load_truth_config()["truth_store"]["roles"]
    for role in mule_side_roles:
        if role not in truth_roles:
            raise ValueError(
                f"mule_side_roles name '{role}' is not a role in config/truth_store.yaml: {list(truth_roles)}"
            )


def resolve_registry_delay_bounds_minutes(split: str, config: dict[str, Any]) -> tuple[int, int]:
    """
    WHAT: Returns the (low, high) registry report delay in whole minutes for a split, read from the labels config
    itself, because this range lives in config/labels.yaml and not in world_ranges.

    WHY: No published figure exists, so the project owns this range (amendment A-021); reading it here keeps the
    split handling in one place.

    Args:
        split: Range split name, one of training, test or stress.
        config: The validated `labels` block.

    Returns:
        (low_minutes, high_minutes), 0 <= low <= high.

    Raises:
        ValueError: If the split has no pair or the pair is malformed.

    Example:
        resolve_registry_delay_bounds_minutes("training", config) == (60, 1440)
    """
    delay_block = config["registry_report_delay_minutes"]
    if split not in delay_block:
        raise ValueError(f"registry_report_delay_minutes has no split '{split}'")
    pair = delay_block[split]
    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
        raise ValueError(f"registry_report_delay_minutes '{split}' must be a pair, got {pair!r}")
    low_minutes = int(round(float(pair[0])))
    high_minutes = int(round(float(pair[1])))
    if low_minutes < 0 or low_minutes > high_minutes:
        raise ValueError(f"registry_report_delay_minutes '{split}' must satisfy 0 <= low <= high, got {pair!r}")
    return low_minutes, high_minutes


def resolve_delay_bounds_minutes(
    world_ranges: dict[str, Any], split: str, scam_type: str, config: dict[str, Any]
) -> tuple[int, int]:
    """
    WHAT: Returns the (low, high) complaint delay in whole minutes for a scam type and split, read from the GIVEN
    world_ranges dictionary (the run's layered copy, never the file on disk). Day ranges are converted with
    MINUTES_PER_DAY, and each bound is rounded to a whole minute.

    WHY: Reading the run's own ranges lets a stress layer such as complaint-backlog change label delays with no extra
    code. The configured unit must match the unit stored on the range entry, so a unit mix-up cannot pass silently.

    Args:
        world_ranges: The run's (possibly layered) world_ranges dictionary.
        split: Range split name, such as "training", "test" or "stress".
        scam_type: One of the scam types in complaint_delay_by_scam_type; `reason` is not a scam type.
        config: The `labels` block.

    Returns:
        (low_minutes, high_minutes), 0 <= low <= high.

    Raises:
        ValueError: If the scam type is unknown, the split or its pair is missing or malformed, the units disagree,
            low is negative or low exceeds high after rounding.

    Example:
        resolve_delay_bounds_minutes(ranges, "training", "unauthorised_debit", config) == (10, 240)
    """
    scam_types = delay_scam_types(config)
    if scam_type not in scam_types:
        raise ValueError(f"Unknown scam type '{scam_type}' for complaint delay; known types are {scam_types}")
    delay_block = config["complaint_delay_by_scam_type"][scam_type]
    reference = delay_block["world_ranges_reference"]
    entry = get_parameter_entry(world_ranges, reference)
    if split not in entry:
        raise ValueError(f"world_ranges entry '{reference}' has no split '{split}'")
    bounds = entry[split]
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        raise ValueError(f"world_ranges entry '{reference}' split '{split}' must be a pair, got {bounds!r}")
    configured_unit = delay_block["unit"]
    if "unit" not in entry or entry["unit"] != configured_unit:
        raise ValueError(
            f"world_ranges entry '{reference}' has unit {entry.get('unit')!r} "
            f"but labels config says '{configured_unit}'"
        )
    factor = MINUTES_PER_DAY if configured_unit == DELAY_UNIT_DAYS else 1
    low_minutes = int(round(float(bounds[0]) * factor))
    high_minutes = int(round(float(bounds[1]) * factor))
    if low_minutes < 0:
        raise ValueError(f"Complaint delay for '{scam_type}' has a negative low bound of {low_minutes} minutes")
    if low_minutes > high_minutes:
        raise ValueError(f"Complaint delay for '{scam_type}' has low {low_minutes} above high {high_minutes} minutes")
    return low_minutes, high_minutes


def draw_label_world_settings(
    world_ranges: dict[str, Any], split: str, rng: np.random.Generator, config: dict[str, Any]
) -> LabelWorldSettings:
    """
    WHAT: Draws the world's three complaint settings, each uniformly inside its range for the split: first the
    victim complaint share, then the false-complaint rate, then the merchant weight. It always makes exactly three
    uniform draws, in that order.

    WHY: A fixed draw count and order keep the stream aligned whatever the configuration says.

    Args:
        world_ranges: The run's (possibly layered) world_ranges dictionary.
        split: Range split name.
        rng: Generator for label_world_settings; advanced by exactly three draws.
        config: The `labels` block.

    Returns:
        The drawn LabelWorldSettings.

    Raises:
        ValueError: If an entry or split is missing or a drawn value breaks LabelWorldSettings limits.

    Example:
        draw_label_world_settings(ranges, "training", np.random.default_rng(1), config)
    """
    share_name = config["victim_complaint_share"]["world_ranges_reference"]
    rate_name = config["false_complaint_rate_per_100000"]["world_ranges_reference"]
    weight_name = config["false_complaint_merchant_weight"]["world_ranges_reference"]
    share = draw_in_range(get_parameter_entry(world_ranges, share_name), rng, split, share_name)
    rate = draw_in_range(get_parameter_entry(world_ranges, rate_name), rng, split, rate_name)
    weight = draw_in_range(get_parameter_entry(world_ranges, weight_name), rng, split, weight_name)
    return LabelWorldSettings(share, rate, weight)
