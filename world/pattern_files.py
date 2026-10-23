"""
WHAT: Loads a pattern file (a YAML description of one gang in the tactic language), refuses it with a plain reason if
its keys, version or gang are wrong, and previews it: the gang is planned once in a small training world and the number
of mule accounts, victims and money per week it would move is reported.

WHY: A written-down pattern is only useful if a mistake in it is found at once and named. The preview plans the gang
without running the world: a full run takes about ten seconds, and the budget is enforced at planning time, so planning
is enough to refuse an over-budget pattern. Pattern text is data (ground rule 19): the name and description are kept as
strings and never executed or formatted into code, and patterns use the tactic language only (ground rule 20).

ISSUE IDs: none (Level 12f task 7).
TERM: pattern file: a YAML file with a top-level `pattern` block holding a version, a name, a description and a gang.
TERM: preview: a plan of the gang in one small world, an estimate of what the pattern would do, not a run.
TERM: mule account: an account the gang controls to receive and pass on money.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from world.fast_mode import FastRunSettings
from world.gang_config import GangConfig, gang_config_from_plain
from world.gang_planner import GangBudgetExceeded, GangPlan, plan_gang
from world.interactive_runner import prepare_world
from world.truth_records import SCAM_STEP_VICTIM_PAYMENT
from world.world_sizing import resolve_settings

DEFAULT_PATTERN_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "pattern_files.yaml"
MINUTES_PER_DAY = 1440
BUDGET_KEYS = ("money_paise", "time_minutes", "max_risk")


class PatternFileError(ValueError):
    """
    WHAT: The error raised for any pattern file that cannot be used. Its message starts with the file name and then
    gives one reason, such as "malformed: missing key 'gang'" or "over budget: ...".

    WHY: One error type lets a caller catch every pattern problem together, and the leading file name tells the
    person which file to fix.
    """


@dataclass(frozen=True)
class PatternFile:
    """
    WHAT: A pattern file that passed every check: its name, description, version, gang and where it came from.

    WHY: Later steps receive an object that is already valid, so they need not check it again.

    Args:
        name: Short name written in the file.
        description: Free text written in the file; kept as plain text and never executed.
        version: The pattern_version number.
        gang: The validated GangConfig.
        path: The file it was read from.
    """

    name: str
    description: str
    version: int
    gang: GangConfig
    path: Path


@dataclass(frozen=True)
class BudgetUse:
    """
    WHAT: How much of each budget limit a planned gang used: money in paise, time in minutes and risk.

    WHY: Shows how close a pattern is to its limits even when it passes.

    Args:
        money_used: Planned spending in paise.
        money_limit: Budget money in paise.
        time_used: Planned campaign time in minutes.
        time_limit: Budget time in minutes.
        risk_used: Planned combined risk.
        risk_limit: Budget maximum risk.
    """

    money_used: int
    money_limit: int
    time_used: int
    time_limit: int
    risk_used: float
    risk_limit: float


@dataclass(frozen=True)
class PatternPreview:
    """
    WHAT: What planning a pattern's gang once in the preview world showed: mule accounts, victims, planned payments,
    money into mules, the span in days, money per week and the budget used.

    WHY: Gives a person a rough idea of a pattern before it is used; every number is an estimate from one plan.

    Args:
        mule_accounts: Distinct accounts with the role "mule".
        victims: Distinct payers of victim payments.
        payments_planned: Number of planned payments.
        money_into_mules_paise: Sum of planned payments whose receiving account is a mule, in paise.
        span_days: Days from the first to the last planned payment, rounded up, at least 1.
        money_per_week_paise: money_into_mules_paise * 7 / span_days, rounded down.
        budget_used: The BudgetUse of the plan.
        preview_preset: Name of the size preset the preview world used.
    """

    mule_accounts: int
    victims: int
    payments_planned: int
    money_into_mules_paise: int
    span_days: int
    money_per_week_paise: int
    budget_used: BudgetUse
    preview_preset: str

    def summary_text(self) -> str:
        """Returns one plain sentence with the mule count and rupees per week, marked as an estimate."""
        rupees = self.money_per_week_paise // 100
        return (
            f"about {self.mule_accounts} mule accounts moving about Rs {rupees:,} per week "
            f"(ESTIMATE: one {self.preview_preset} training world plan, not a run)"
        )


def load_pattern_config(path: Path | str | None = None) -> dict[str, Any]:
    """
    WHAT: Reads config/pattern_files.yaml (or the given path) and returns its pattern_files block.

    WHY: The supported version, preview world and allowed keys are policy, so they live in YAML and not in code.

    Args:
        path: Optional YAML path.

    Returns:
        The pattern_files dictionary.

    Raises:
        ValueError: If the file has no pattern_files block.
    """
    chosen = DEFAULT_PATTERN_CONFIG_PATH if path is None else Path(path)
    with open(chosen, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict) or "pattern_files" not in data:
        raise ValueError(f"{chosen} has no pattern_files block")
    block: dict[str, Any] = data["pattern_files"]
    return block


def _fail(path: Path, reason: str) -> PatternFileError:
    """Builds the error for a file: its name first, then the reason."""
    return PatternFileError(f"{path.name}: {reason}")


def _read_yaml(path: Path) -> Any:
    """Reads a pattern file with yaml.safe_load, so no text in the file can build or run an object."""
    try:
        with open(path, encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    except yaml.YAMLError:
        raise _fail(path, "malformed: not valid YAML") from None


def _check_keys(path: Path, block: dict[str, Any], allowed: list[str], required: list[str]) -> None:
    """Refuses a mapping with a key outside the allowed list, or without one of the required keys."""
    for key in block:
        if key not in allowed:
            raise _fail(path, f"malformed: unknown key '{key}'")
    for key in required:
        if key not in block:
            raise _fail(path, f"malformed: missing key '{key}'")


def _check_type(path: Path, key: str, value: Any, expected: type, label: str) -> None:
    """Refuses a value of the wrong type; a bool is not accepted where a whole number is expected."""
    if not isinstance(value, expected) or (expected is int and isinstance(value, bool)):
        raise _fail(path, f"malformed: '{key}' must be {label}")


def _check_gang_shape(path: Path, gang: Any, cfg: dict[str, Any]) -> None:
    """Checks the gang block's keys and types so gang_config_from_plain never fails with a bare KeyError."""
    _check_type(path, "gang", gang, dict, "a mapping")
    _check_keys(path, gang, cfg["allowed_gang_keys"]["keys"], ["gang_id", "strength", "budget", "steps"])
    _check_type(path, "gang_id", gang["gang_id"], str, "text")
    _check_type(path, "strength", gang["strength"], int, "a whole number")
    _check_type(path, "budget", gang["budget"], dict, "a mapping")
    _check_keys(path, gang["budget"], list(BUDGET_KEYS), list(BUDGET_KEYS))
    _check_type(path, "steps", gang["steps"], list, "a list")
    for step in gang["steps"]:
        _check_type(path, "steps", step, dict, "a list of mappings")
        _check_keys(path, step, ["block", "params"], ["block"])


def load_pattern_file(path: Path | str, pattern_cfg: dict[str, Any] | None = None) -> PatternFile:
    """
    WHAT: Reads a pattern file and checks it in a fixed order: YAML, top-level keys, pattern keys, types, version, then
    the gang. Every problem raises PatternFileError whose message starts with the file name.

    WHY: A fixed order gives the same reason for the same mistake every time. Gang problems keep the wording of
    gang_config_from_plain, prefixed with "invalid gang:", so "unknown block" stays recognisable.

    Args:
        path: The pattern file.
        pattern_cfg: The pattern_files block; defaults to config/pattern_files.yaml.

    Returns:
        The validated PatternFile.

    Raises:
        PatternFileError: For malformed YAML, unknown or missing keys, wrong types, an unsupported version or an
            invalid gang.
    """
    file_path = Path(path)
    cfg = load_pattern_config() if pattern_cfg is None else pattern_cfg
    data = _read_yaml(file_path)
    if not isinstance(data, dict):
        raise _fail(file_path, "malformed: top level must be a mapping with only 'pattern'")
    _check_keys(file_path, data, cfg["allowed_top_level_keys"]["keys"], ["pattern"])
    block = data["pattern"]
    _check_type(file_path, "pattern", block, dict, "a mapping")
    _check_keys(file_path, block, cfg["allowed_pattern_keys"]["keys"], cfg["allowed_pattern_keys"]["keys"])
    _check_type(file_path, "pattern_version", block["pattern_version"], int, "a whole number")
    _check_type(file_path, "name", block["name"], str, "text")
    _check_type(file_path, "description", block["description"], str, "text")
    if block["pattern_version"] != cfg["supported_pattern_version"]["value"]:
        raise _fail(file_path, f"unsupported version {block['pattern_version']}")
    _check_gang_shape(file_path, block["gang"], cfg)
    try:
        gang = gang_config_from_plain(block["gang"])
    except (ValueError, KeyError, TypeError) as problem:
        raise _fail(file_path, f"invalid gang: {problem}") from None
    return PatternFile(block["name"], block["description"], block["pattern_version"], gang, file_path)


def _preview_from_plan(plan: GangPlan, gang: GangConfig, preset: str) -> PatternPreview:
    """Turns a planned gang into the preview numbers described on PatternPreview."""
    mules = {span.account_id for span in plan.truth.roles if span.role == "mule"}
    victims: set[str] = set()
    money_into_mules = 0
    first_minute = 0
    last_minute = 0
    for index in range(len(plan.payments)):
        payment = plan.payments[index]
        story = plan.truth.payment_stories[index]
        if story.scam_step == SCAM_STEP_VICTIM_PAYMENT:
            victims.add(payment.from_account)
        if payment.to_account in mules:
            money_into_mules += payment.amount_paise
        if index == 0 or payment.minute < first_minute:
            first_minute = payment.minute
        if index == 0 or payment.minute > last_minute:
            last_minute = payment.minute
    span_minutes = last_minute - first_minute
    # Rounding up keeps the span at least one day, so money per week never divides by zero.
    span_days = max(1, -(-span_minutes // MINUTES_PER_DAY))
    budget = gang.budget
    used = BudgetUse(
        plan.totals.money_paise, budget.money_paise, plan.totals.time_minutes, budget.time_minutes,
        plan.totals.risk, budget.max_risk,
    )
    return PatternPreview(
        len(mules), len(victims), len(plan.payments), money_into_mules, span_days,
        money_into_mules * 7 // span_days, used, preset,
    )


def preview_pattern(pattern: PatternFile, preview_cfg: dict[str, Any] | None = None) -> PatternPreview:
    """
    WHAT: Plans the pattern's gang once in the preview world (the configured preset and training seed, range
    "training") and returns the preview numbers. It plans only; no payment is run.

    WHY: Planning is enough to refuse an over-budget pattern, because the budget is enforced while planning, and it is
    far quicker than a run of about ten seconds. The random streams are built as the interactive runner builds them
    (children 6 and 7 of the seed), so the preview shows the plan the runner would make for a first gang.

    Args:
        pattern: A PatternFile that passed load_pattern_file.
        preview_cfg: The pattern_files block; defaults to config/pattern_files.yaml.

    Returns:
        The PatternPreview.

    Raises:
        PatternFileError: "over budget: ..." if planning exceeds the budget, or "cannot be planned: ..." if the world
            cannot supply what the gang needs.
    """
    cfg = load_pattern_config() if preview_cfg is None else preview_cfg
    world_cfg = cfg["preview_world"]
    settings: FastRunSettings = resolve_settings(
        seed=world_cfg["seed"], preset=world_cfg["preset"], range_name="training"
    )
    prepared = prepare_world(settings, None)
    children = np.random.SeedSequence(settings.seed).spawn(10)
    dev_rng = np.random.default_rng(children[6])
    plan_rng = np.random.default_rng(children[7])
    reserved: set[str] = set()
    if pattern.gang.source is not None:
        reserved.add(pattern.gang.source.account_id)
    try:
        plan = plan_gang(pattern.gang, prepared.world, plan_rng, reserved_accounts=reserved, gang_devices_rng=dev_rng)
    except GangBudgetExceeded as problem:
        raise _fail(pattern.path, f"over budget: {problem}") from None
    except ValueError as problem:
        raise _fail(pattern.path, f"cannot be planned: {problem}") from None
    return _preview_from_plan(plan, pattern.gang, world_cfg["preset"])


def validate_pattern_file(path: Path | str) -> PatternPreview:
    """
    WHAT: Loads a pattern file and previews it, so one call checks everything: keys, version, gang and budget.

    WHY: A person wants a single command that says "fine, here is roughly what it does" or "no, because ...".

    Args:
        path: The pattern file.

    Returns:
        The PatternPreview of a valid file.

    Raises:
        PatternFileError: With the file name and the reason, as in load_pattern_file and preview_pattern.
    """
    return preview_pattern(load_pattern_file(path))
