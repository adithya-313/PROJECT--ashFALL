"""
WHAT: Resolvers for the scam schedule settings in config/scam_schedules.yaml (Level 12c C-2). Each function returns
one setting for one evaluation split. A setting is either stored in the schedule file itself or named there by a
world_ranges_reference and then read from config/world_ranges.yaml.

WHY: The gang planner must not hold any scam number in code. One function per setting keeps every read visible,
validated, and testable with a narrowed copy of the config files.

TERM: split: the evaluation range of the world: training, test or stress.
TERM: world_ranges_reference: a setting that names a parameter of config/world_ranges.yaml instead of holding values.
TERM: scam type: the story that makes a victim pay: investment, job or unauthorised_debit.
TERM: payout back: a small payment from the receiving mule back to the victim to build trust.
"""

from pathlib import Path
from typing import Any

from core.config_validator import load_yaml
from world.run_identity import REPO_ROOT

SCAM_SCHEDULES_PATH: Path = REPO_ROOT / "config" / "scam_schedules.yaml"
WORLD_RANGES_PATH: Path = REPO_ROOT / "config" / "world_ranges.yaml"

# Section names of config/scam_schedules.yaml, named once so a rename is a one-line change.
SHARES_SECTION: str = "scam_type_shares"
INVESTMENT_JOB_SECTION: str = "investment_and_job_schedule"
DEBIT_SECTION: str = "unauthorised_debit_schedule"


def load_scam_schedules(config_path: Path | None = None) -> dict[str, Any]:
    """
    WHAT: Reads and parses config/scam_schedules.yaml, or the file at config_path when one is given.

    WHY: Every resolver starts from this one read, so a narrowed copy of the file changes all of them at once.

    Args:
        config_path: Optional path of a scam schedule file; None means the committed file.

    Returns:
        The parsed YAML mapping.

    Raises:
        FileNotFoundError: If the file does not exist.

    Example:
        load_scam_schedules()["scam_type_shares"]["remainder_type"]["value"] == "unauthorised_debit"
    """
    path = config_path if config_path is not None else SCAM_SCHEDULES_PATH
    if not path.exists():
        raise FileNotFoundError(f"Scam schedule config not found: {path}")
    return load_yaml(path)


def _read_setting(section: str, name: str, config_path: Path | None) -> dict[str, Any]:
    """
    WHAT: Returns the labelled block called name inside the given top-level section of the schedule file.

    WHY: A missing section or setting must stop with a sentence naming it, not a bare KeyError.

    Args:
        section: Top-level section name, e.g. "unauthorised_debit_schedule".
        name: Setting name inside the section, e.g. "debit_payments".
        config_path: Optional path of a scam schedule file.

    Returns:
        The setting's block.

    Raises:
        ValueError: If the section or the setting is missing.

    Example:
        _read_setting(DEBIT_SECTION, "debit_payments", None)["label"] == "assumption"
    """
    schedules = load_scam_schedules(config_path)
    if section not in schedules:
        raise ValueError(f"Scam schedule config has no section '{section}'.")
    if name not in schedules[section]:
        raise ValueError(f"Scam schedule section '{section}' has no setting '{name}'.")
    return schedules[section][name]


def _split_bounds(entry: dict[str, Any], split: str, description: str) -> tuple[float, float]:
    """
    WHAT: Returns the [lower, upper] pair stored under the split name in a config entry, checked to be two numbers
    in order.

    WHY: A malformed or inverted range would otherwise surface later as a confusing random-draw error.

    Args:
        entry: A block holding a list for each split, e.g. {"training": [1, 3]}.
        split: Split name, e.g. "training".
        description: Setting name used in error sentences.

    Returns:
        (lower, upper) as floats.

    Raises:
        ValueError: If the split is missing, the value is not two numbers, or lower is above upper.

    Example:
        _split_bounds({"training": [1, 3]}, "training", "debit_payments") == (1.0, 3.0)
    """
    if split not in entry:
        raise ValueError(f"Scam setting '{description}' has no range for split '{split}'.")
    bounds = entry[split]
    if not isinstance(bounds, list) or len(bounds) != 2:
        raise ValueError(f"Scam setting '{description}' split '{split}' must be a list of two numbers.")
    lower, upper = bounds[0], bounds[1]
    for value in (lower, upper):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Scam setting '{description}' split '{split}' must hold numbers, got {value!r}.")
    if lower > upper:
        raise ValueError(f"Scam setting '{description}' split '{split}' has lower {lower} above upper {upper}.")
    return float(lower), float(upper)


def _own_range(
    section: str, name: str, split: str, config_path: Path | None
) -> tuple[float, float]:
    """
    WHAT: Reads a setting whose split ranges are stored in the schedule file itself.

    WHY: Separates the two storage styles so each public resolver says in one line where its value lives.

    Args:
        section: Top-level section name.
        name: Setting name inside the section.
        split: Split name.
        config_path: Optional path of a scam schedule file.

    Returns:
        (lower, upper) as floats.

    Raises:
        ValueError: If the setting or split is missing or malformed.

    Example:
        _own_range(DEBIT_SECTION, "debit_payments", "training", None) == (1.0, 3.0)
    """
    entry = _read_setting(section, name, config_path)
    return _split_bounds(entry, split, name)


def _referenced_range(
    section: str, name: str, split: str, config_path: Path | None, world_ranges_path: Path | None
) -> tuple[float, float]:
    """
    WHAT: Reads a setting that names a world_ranges parameter, then returns that parameter's range for the split.

    WHY: The values stay in config/world_ranges.yaml, the one place that holds their label and source.

    Args:
        section: Top-level section name.
        name: Setting name inside the section.
        split: Split name.
        config_path: Optional path of a scam schedule file.
        world_ranges_path: Optional path of a world ranges file; None means the committed file.

    Returns:
        (lower, upper) as floats.

    Raises:
        ValueError: If the setting has no reference, the referenced parameter is missing, or its split is malformed.
        FileNotFoundError: If the world ranges file does not exist.

    Example:
        _referenced_range(INVESTMENT_JOB_SECTION, "payments_per_victim", "training", None, None) == (1.0, 6.0)
    """
    entry = _read_setting(section, name, config_path)
    if "world_ranges_reference" not in entry:
        raise ValueError(f"Scam setting '{name}' must name a world_ranges_reference.")
    reference = entry["world_ranges_reference"]
    ranges_path = world_ranges_path if world_ranges_path is not None else WORLD_RANGES_PATH
    if not ranges_path.exists():
        raise FileNotFoundError(f"World ranges config not found: {ranges_path}")
    parameters = load_yaml(ranges_path)["parameters"]
    if reference not in parameters:
        raise ValueError(f"Scam setting '{name}' references unknown world_ranges parameter '{reference}'.")
    return _split_bounds(parameters[reference], split, reference)


def _as_int_range(bounds: tuple[float, float], name: str) -> tuple[int, int]:
    """
    WHAT: Converts a (lower, upper) pair to whole numbers, refusing a pair that holds fractions.

    WHY: Counts and day numbers feed integer random draws; a fractional bound in config is a config mistake.

    Args:
        bounds: (lower, upper) as floats.
        name: Setting name used in the error sentence.

    Returns:
        (lower, upper) as ints.

    Raises:
        ValueError: If either bound is not a whole number.

    Example:
        _as_int_range((1.0, 6.0), "payments_per_victim") == (1, 6)
    """
    lower, upper = bounds
    if lower != int(lower) or upper != int(upper):
        raise ValueError(f"Scam setting '{name}' must hold whole numbers, got {lower} and {upper}.")
    return int(lower), int(upper)


def resolve_investment_or_job_share(
    split: str, config_path: Path | None = None, world_ranges_path: Path | None = None
) -> tuple[float, float]:
    """
    WHAT: Returns the range of the chance that a victim's scam is investment or job, from world_ranges.

    WHY: The planner draws one chance per scam step from this range and gives every victim that chance.

    Args:
        split: Split name.
        config_path: Optional path of a scam schedule file.
        world_ranges_path: Optional path of a world ranges file.

    Returns:
        (lower, upper) fractions.

    Example:
        resolve_investment_or_job_share("training") == (0.55, 0.8)
    """
    return _referenced_range(SHARES_SECTION, "investment_or_job_share", split, config_path, world_ranges_path)


def resolve_investment_share_of_investment_or_job(
    split: str, config_path: Path | None = None
) -> tuple[float, float]:
    """
    WHAT: Returns the range of the chance that an investment-or-job victim has the investment scam, not the job one.

    WHY: Gives the planner the split between the two escalating scam types.

    Args:
        split: Split name.
        config_path: Optional path of a scam schedule file.

    Returns:
        (lower, upper) fractions.

    Example:
        resolve_investment_share_of_investment_or_job("training") == (0.4, 0.6)
    """
    return _own_range(SHARES_SECTION, "investment_share_of_investment_or_job", split, config_path)


def resolve_remainder_scam_type(config_path: Path | None = None) -> str:
    """
    WHAT: Returns the scam type given to every victim who is neither investment nor job.

    WHY: Keeps the name of the catch-all type in config, with its reason, instead of in planner code.

    Args:
        config_path: Optional path of a scam schedule file.

    Returns:
        The scam type name.

    Raises:
        ValueError: If the value is not a non-empty string.

    Example:
        resolve_remainder_scam_type() == "unauthorised_debit"
    """
    value = _read_setting(SHARES_SECTION, "remainder_type", config_path)["value"]
    if not isinstance(value, str) or not value:
        raise ValueError(f"Scam setting 'remainder_type' must be a non-empty string, got {value!r}.")
    return value


def resolve_payments_per_victim(
    split: str, config_path: Path | None = None, world_ranges_path: Path | None = None
) -> tuple[int, int]:
    """
    WHAT: Returns the range of how many times one investment or job victim pays.

    WHY: The planner draws each victim's payment count from this range.

    Args:
        split: Split name.
        config_path: Optional path of a scam schedule file.
        world_ranges_path: Optional path of a world ranges file.

    Returns:
        (lower, upper) counts.

    Example:
        resolve_payments_per_victim("training") == (1, 6)
    """
    bounds = _referenced_range(
        INVESTMENT_JOB_SECTION, "payments_per_victim", split, config_path, world_ranges_path
    )
    return _as_int_range(bounds, "payments_per_victim")


def resolve_payment_spread_days(
    split: str, config_path: Path | None = None, world_ranges_path: Path | None = None
) -> tuple[int, int]:
    """
    WHAT: Returns the range of days over which one investment or job victim's payments are spread.

    WHY: The planner draws each victim's spread from this range and places later payments inside it.

    Args:
        split: Split name.
        config_path: Optional path of a scam schedule file.
        world_ranges_path: Optional path of a world ranges file.

    Returns:
        (lower, upper) days.

    Example:
        resolve_payment_spread_days("training") == (0, 10)
    """
    bounds = _referenced_range(
        INVESTMENT_JOB_SECTION, "payment_spread_days", split, config_path, world_ranges_path
    )
    return _as_int_range(bounds, "payment_spread_days")


def resolve_first_payment_amount_inr(
    split: str, config_path: Path | None = None, world_ranges_path: Path | None = None
) -> tuple[float, float]:
    """
    WHAT: Returns the range of a victim's first payment in rupees.

    WHY: The planner draws the first amount here and builds later payments from it.

    Args:
        split: Split name.
        config_path: Optional path of a scam schedule file.
        world_ranges_path: Optional path of a world ranges file.

    Returns:
        (lower, upper) rupees.

    Example:
        resolve_first_payment_amount_inr("training") == (2000.0, 50000.0)
    """
    return _referenced_range(
        INVESTMENT_JOB_SECTION, "first_payment_amount_inr", split, config_path, world_ranges_path
    )


def resolve_payment_escalation_multiple(
    split: str, config_path: Path | None = None
) -> tuple[float, float]:
    """
    WHAT: Returns the range of how many times larger a victim's next payment is than the one before it.

    WHY: The planner draws one multiple per victim; 1.0 means no escalation.

    Args:
        split: Split name.
        config_path: Optional path of a scam schedule file.

    Returns:
        (lower, upper) multiples.

    Raises:
        ValueError: If the lower bound is below 1.0, because payments must not shrink.

    Example:
        resolve_payment_escalation_multiple("training") == (1.0, 2.0)
    """
    bounds = _own_range(INVESTMENT_JOB_SECTION, "payment_escalation_multiple", split, config_path)
    if bounds[0] < 1.0:
        raise ValueError(f"Scam setting 'payment_escalation_multiple' must not go below 1.0, got {bounds[0]}.")
    return bounds


def resolve_payout_back_share(
    split: str, config_path: Path | None = None, world_ranges_path: Path | None = None
) -> tuple[float, float]:
    """
    WHAT: Returns the range of the share of the victim's payments so far that the mule pays back.

    WHY: The planner draws one share per payout; 0.0 means the payout is skipped.

    Args:
        split: Split name.
        config_path: Optional path of a scam schedule file.
        world_ranges_path: Optional path of a world ranges file.

    Returns:
        (lower, upper) fractions.

    Example:
        resolve_payout_back_share("training") == (0.05, 0.3)
    """
    return _referenced_range(INVESTMENT_JOB_SECTION, "payout_back_share", split, config_path, world_ranges_path)


def resolve_min_minutes_between_victim_payments(config_path: Path | None = None) -> int:
    """
    WHAT: Returns the smallest number of minutes between two payments of one investment or job victim.

    WHY: With at least two minutes between payments, the payout back always has a whole minute strictly between
    the first and the last payment.

    Args:
        config_path: Optional path of a scam schedule file.

    Returns:
        Minutes, at least 2.

    Raises:
        ValueError: If the value is not a whole number of at least 2.

    Example:
        resolve_min_minutes_between_victim_payments() == 2
    """
    value = _read_setting(INVESTMENT_JOB_SECTION, "min_minutes_between_victim_payments", config_path)["value"]
    if isinstance(value, bool) or not isinstance(value, int) or value < 2:
        raise ValueError(
            f"Scam setting 'min_minutes_between_victim_payments' must be a whole number of at least 2, got {value!r}."
        )
    return value


def resolve_debit_payment_count(split: str, config_path: Path | None = None) -> tuple[int, int]:
    """
    WHAT: Returns the range of how many unauthorised debits one victim suffers.

    WHY: The planner draws each unauthorised-debit victim's debit count here.

    Args:
        split: Split name.
        config_path: Optional path of a scam schedule file.

    Returns:
        (lower, upper) counts.

    Example:
        resolve_debit_payment_count("training") == (1, 3)
    """
    bounds = _own_range(DEBIT_SECTION, "debit_payments", split, config_path)
    return _as_int_range(bounds, "debit_payments")


def resolve_debit_spread_minutes(split: str, config_path: Path | None = None) -> tuple[int, int]:
    """
    WHAT: Returns the range of minutes over which one victim's unauthorised debits are spread.

    WHY: The planner places each later debit inside the drawn spread after the first.

    Args:
        split: Split name.
        config_path: Optional path of a scam schedule file.

    Returns:
        (lower, upper) minutes.

    Example:
        resolve_debit_spread_minutes("training") == (0, 60)
    """
    bounds = _own_range(DEBIT_SECTION, "debit_spread_minutes", split, config_path)
    return _as_int_range(bounds, "debit_spread_minutes")


def resolve_debit_amount_inr(
    split: str, config_path: Path | None = None, world_ranges_path: Path | None = None
) -> tuple[float, float]:
    """
    WHAT: Returns the range of the size of one unauthorised debit in rupees.

    WHY: Each debit is drawn independently from this range.

    Args:
        split: Split name.
        config_path: Optional path of a scam schedule file.
        world_ranges_path: Optional path of a world ranges file.

    Returns:
        (lower, upper) rupees.

    Example:
        resolve_debit_amount_inr("training") == (2000.0, 50000.0)
    """
    return _referenced_range(DEBIT_SECTION, "debit_amount_inr", split, config_path, world_ranges_path)


def resolve_debit_pays_back(config_path: Path | None = None) -> bool:
    """
    WHAT: Returns whether a mule pays anything back to an unauthorised-debit victim.

    WHY: Keeps the "no payout after a takeover" rule in config with its reason.

    Args:
        config_path: Optional path of a scam schedule file.

    Returns:
        True when a payout back is planned for such victims.

    Raises:
        ValueError: If the value is not a boolean.

    Example:
        resolve_debit_pays_back() is False
    """
    value = _read_setting(DEBIT_SECTION, "payout_back", config_path)["value"]
    if not isinstance(value, bool):
        raise ValueError(f"Scam setting 'payout_back' must be true or false, got {value!r}.")
    return value
