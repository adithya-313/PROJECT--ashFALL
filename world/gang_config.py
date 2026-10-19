"""
WHAT: Data structures and structural validation for gang tactical configurations (Level 12b B-2a).
Defines immutable records for gang budgets, fixture money sources, block steps, and gang configurations,
along with conversion to canonical plain-data dictionaries and strict policy validation.

WHY: Simulation realism requires structured tactical plans before execution. Validating gang configurations
against declarative tactic catalogues and dividing lines ensures that invalid orders, unsupported strengths,
parameter boundary breaches, and unauthorized tactics are rejected cleanly before world planning begins.

TERM: gang configuration: specification of a criminal gang including budget, source account, strength, and tactic steps.
TERM: gang budget: three-dimensional spending cap in money (paise), simulated time (minutes), and cumulative risk.
TERM: fixture source: initial collector account pre-funded with opening money from which fraud routing originates.
TERM: block step: single parametrized tactic building block in a gang's sequential action plan.
TERM: strength level: the sophistication and camouflage applied by a gang (strength 1 = fixed, 2 = camouflaged).
TERM: dividing line: boundary separating training tactics from held-out tactics to prevent data leakage.
"""

from dataclasses import dataclass
from typing import Any

from world.tactic_config import get_block_definition, resolve_dividing_line


@dataclass(frozen=True)
class GangBudget:
    """
    WHAT: Represents spending limits for a gang campaign across money, time, and cumulative risk.

    WHY: Constrains gang operations within realistic resource boundaries per Decision D8.

    Args:
        money_paise: Maximum money allocated to transaction fees in paise.
        time_minutes: Maximum duration in simulated minutes allocated to campaign steps.
        max_risk: Maximum acceptable cumulative probability of detection (in [0, 1]).

    Returns:
        An initialized immutable GangBudget instance.

    Example:
        # budget = GangBudget(money_paise=500_000, time_minutes=2880, max_risk=0.05)
    """

    money_paise: int
    time_minutes: int
    max_risk: float


@dataclass(frozen=True)
class FixtureSource:
    """
    WHAT: Pre-registered collector account and initial funding allocation for a gang campaign.

    WHY: Ground Rule 15 and Decision D3 require fraud money to originate from a designated fixture account.

    Args:
        account_id: Account identifier for the fixture collector (e.g., 'acc_collector_01').
        bank_index: Index of the bank holding the collector account.
        opening_balance_paise: Initial balance in paise deposited into the account at start.
        start_minute: Simulation minute at which gang routing begins.

    Returns:
        An initialized immutable FixtureSource instance.

    Example:
        # source = FixtureSource("acc_collector_01", 0, 10_000_000, 0)
    """

    account_id: str
    bank_index: int
    opening_balance_paise: int
    start_minute: int


@dataclass(frozen=True)
class BlockStep:
    """
    WHAT: Single parametrized building block step in a gang campaign plan.

    WHY: Composes sequential tactical maneuvers while remaining hashable and canonical via tuple parameters.

    Args:
        block: String identifier of the tactic building block (acquire, warm_up, route, disguise).
        params: Immutable tuple of (parameter_name, value) pairs.

    Returns:
        An initialized immutable BlockStep instance.

    Example:
        # step = BlockStep("acquire", (("count", 5),))
    """

    block: str
    params: tuple[tuple[str, Any], ...] = ()

    @property
    def params_dict(self) -> dict[str, Any]:
        """
        WHAT: Converts the immutable tuple of parameter (name, value) pairs into a standard Python dictionary.

        WHY: While BlockStep instances must remain frozen and hashable for canonical representation in run_id
        effective configuration records, validation rules and planning algorithms require fast key-based parameter
        lookups to inspect counts, layers, amounts, and dwell times.

        Returns:
            Dictionary mapping string parameter names to their configured values.

        Example:
            # step = BlockStep("acquire", (("count", 5),))
            # step.params_dict["count"] == 5
        """
        # 1. Convert tuple pairs to a dictionary.
        return dict(self.params)


@dataclass(frozen=True)
class GangConfig:
    """
    WHAT: Complete configuration defining an illicit syndicate's tactical campaign.

    WHY: Serves as the top-level specification driving deterministic fraud planning and truth generation.

    Args:
        gang_id: Unique string identifier for the gang (e.g., 'gang_alpha').
        strength: Integer sophistication level: 1 (fixed parameters) or 2 (camouflaged).
        budget: GangBudget specifying monetary, temporal, and risk boundaries.
        source: FixtureSource designating the starting collector account and capital.
        steps: Ordered sequence of BlockStep items specifying the campaign itinerary.

    Returns:
        An initialized immutable GangConfig instance.

    Example:
        # config = GangConfig("gang_alpha", 1, budget, source, (step1, step2))
    """

    gang_id: str
    strength: int
    budget: GangBudget
    source: FixtureSource
    steps: tuple[BlockStep, ...]


def gang_config_to_plain(config: GangConfig) -> dict[str, Any]:
    """
    WHAT: Converts a GangConfig instance into a canonical plain Python dictionary.

    WHY: Used by the interactive runner (Decision D5) to fingerprint effective_config into the run_id.

    Args:
        config: GangConfig instance to convert.

    Returns:
        Plain dictionary holding serialized primitives.

    Example:
        # plain = gang_config_to_plain(config)
        # plain["gang_id"] == "gang_alpha"
    """
    # 1. Build canonical dictionary of budget, source, and step specifications.
    return {
        "gang_id": config.gang_id,
        "strength": config.strength,
        "budget": {
            "money_paise": config.budget.money_paise,
            "time_minutes": config.budget.time_minutes,
            "max_risk": config.budget.max_risk,
        },
        "source": {
            "account_id": config.source.account_id,
            "bank_index": config.source.bank_index,
            "opening_balance_paise": config.source.opening_balance_paise,
            "start_minute": config.source.start_minute,
        },
        "steps": [
            {
                "block": step.block,
                "params": list(step.params),
            }
            for step in config.steps
        ],
    }


def _validate_block_existence_and_params(gang_id: str, steps: tuple[BlockStep, ...]) -> None:
    """
    WHAT: Asserts that all block names exist in tactic_blocks.yaml and parameter names match block schemas.

    WHY: Catches mistyped block identifiers and invalid parameters before planning.

    Args:
        gang_id: Gang identifier for error diagnostics.
        steps: Sequence of BlockStep items to validate.

    Returns:
        None. Raises ValueError if a block or parameter is unrecognized.

    Example:
        # _validate_block_existence_and_params("gang_a", (BlockStep("acquire", (("count", 5),)),))
    """
    # 1. Iterate across steps to check block definitions and parameter names.
    for step in steps:
        try:
            block_def = get_block_definition(step.block)
        except KeyError:
            raise ValueError(f"Gang '{gang_id}' uses unknown block '{step.block}'.")

        known_params = set(block_def.get("parameters", {}).keys())
        for param_name, _ in step.params:
            if param_name not in known_params:
                raise ValueError(f"Gang '{gang_id}' block '{step.block}' has unknown parameter '{param_name}'.")


def _validate_block_strengths(gang_id: str, strength: int, steps: tuple[BlockStep, ...]) -> None:
    """
    WHAT: Asserts that the gang's strength level is permitted for every block in its step itinerary.

    WHY: Decision D7 permits disguise only at strength 2; invalid strength combinations must be refused.

    Args:
        gang_id: Gang identifier for error diagnostics.
        strength: Integer strength level of the gang.
        steps: Sequence of BlockStep items to validate.

    Returns:
        None. Raises ValueError if a block does not permit the gang's strength.

    Example:
        # Disguise at strength 1 raises ValueError.
        # _validate_block_strengths("gang_a", 1, (BlockStep("disguise"),))
    """
    # 1. Assert gang strength is supported by each block in the sequence.
    for step in steps:
        block_def = get_block_definition(step.block)
        allowed_strengths = block_def.get("allowed_strengths", [])
        if strength not in allowed_strengths:
            raise ValueError(
                f"Gang '{gang_id}' strength {strength} is not allowed for block '{step.block}'; "
                f"allowed strengths are {allowed_strengths}."
            )


def _validate_step_ordering(gang_id: str, steps: tuple[BlockStep, ...]) -> None:
    """
    WHAT: Enforces procedural order constraints: acquire before route, route needs acquire, warm_up after acquire.

    WHY: A syndicate cannot route stolen funds or warm up accounts without first acquiring mule accounts.

    Args:
        gang_id: Gang identifier for error diagnostics.
        steps: Sequence of BlockStep items to validate.

    Returns:
        None. Raises ValueError if step ordering rules are violated.

    Example:
        # Route before acquire raises ValueError.
        # _validate_step_ordering("gang_a", (BlockStep("route"), BlockStep("acquire")))
    """
    # 1. Track whether accounts have been acquired before downstream operations.
    has_acquired = False
    has_routed = False

    for step in steps:
        if step.block == "acquire":
            has_acquired = True
        elif step.block == "warm_up":
            if not has_acquired:
                raise ValueError(f"Gang '{gang_id}' cannot warm up before acquiring accounts.")
        elif step.block == "route":
            if not has_acquired:
                raise ValueError(f"Gang '{gang_id}' must acquire accounts before routing money.")
            has_routed = True

    # 2. Assert that routing requires at least one acquisition step.
    if has_routed and not has_acquired:
        raise ValueError(f"Gang '{gang_id}' must acquire accounts before routing money.")


def _validate_route_parameters(gang_id: str, steps: tuple[BlockStep, ...]) -> None:
    """
    WHAT: Asserts route steps have layers >= 1, dwell <= training maximum, and amount >= slow drip minimum.

    WHY: Enforces held-out dividing lines (12-8) to guarantee training routes never leak into held-out tactics.

    Args:
        gang_id: Gang identifier for error diagnostics.
        steps: Sequence of BlockStep items to validate.

    Returns:
        None. Raises ValueError if route parameters breach dividing lines or structural limits.

    Example:
        # Route dwell of 90 minutes exceeds training maximum 60.
        # _validate_route_parameters("gang_a", (BlockStep("route", (("dwell_minutes", 90),)),))
    """
    # 1. Resolve dynamic held-out dividing lines from world ranges.
    max_dwell = resolve_dividing_line("training_max_dwell_minutes")
    min_amount = resolve_dividing_line("slow_drip_max_amount_inr")

    # 2. Verify route parameters against dividing line thresholds.
    for step in steps:
        if step.block != "route":
            continue
        params_dict = step.params_dict
        if "layers" in params_dict and params_dict["layers"] < 1:
            raise ValueError(f"Gang '{gang_id}' route layers must be at least 1, got {params_dict['layers']}.")
        if "dwell_minutes" in params_dict and params_dict["dwell_minutes"] > max_dwell:
            raise ValueError(
                f"Gang '{gang_id}' route dwell {params_dict['dwell_minutes']} "
                f"exceeds training maximum dwell of {max_dwell} minutes."
            )
        if "amount_inr" in params_dict and params_dict["amount_inr"] < min_amount:
            raise ValueError(
                f"Gang '{gang_id}' route amount {params_dict['amount_inr']} INR "
                f"is below training minimum slow drip amount of {min_amount} INR."
            )


def validate_gang_config(config: GangConfig) -> None:
    """
    WHAT: Comprehensive validator enforcing block schemas, strength constraints, step ordering, and dividing lines.
    Raises ValueError with a descriptive message upon any validation failure.

    WHY: Core entry point verifying gang specifications before simulation planning to ensure full policy adherence.

    Args:
        config: GangConfig instance to validate.

    Returns:
        None. Raises ValueError if any rule is violated.

    Example:
        # validate_gang_config(valid_config)  # passes cleanly
    """
    # 1. Validate block identifiers and parameter names.
    _validate_block_existence_and_params(config.gang_id, config.steps)

    # 2. Validate strength capability per block.
    _validate_block_strengths(config.gang_id, config.strength, config.steps)

    # 3. Validate logical step ordering.
    _validate_step_ordering(config.gang_id, config.steps)

    # 4. Validate route parameters against dividing lines.
    _validate_route_parameters(config.gang_id, config.steps)
