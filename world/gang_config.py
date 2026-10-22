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

from world.tactic_config import (
    get_block_definition,
    get_origin_mix_options,
    get_scam_receiving_mules_range,
    get_scam_victims_per_mule_range,
    get_take_control_devices_range,
    resolve_dividing_line,
)


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
        source: FixtureSource designating the starting collector account and capital, or None when the gang's
            money comes from a scam step. A gang has exactly one money source: this fixture source or a scam step.
        steps: Ordered sequence of BlockStep items specifying the campaign itinerary.
        start_minute: Simulation minute at which the campaign begins when there is no fixture source (the fixture
            source carries its own start_minute); None when a fixture source is set.

    Returns:
        An initialized immutable GangConfig instance.

    Example:
        # config = GangConfig("gang_alpha", 1, budget, source, (step1, step2))
        # scam_config = GangConfig("gang_beta", 1, budget, None, (acquire, scam), start_minute=1440)
    """

    gang_id: str
    strength: int
    budget: GangBudget
    source: FixtureSource | None = None
    steps: tuple[BlockStep, ...] = ()
    start_minute: int | None = None


def gang_start_minute(config: GangConfig) -> int:
    """
    WHAT: Returns the simulation minute at which a gang's campaign begins: the fixture source's start_minute when
    the gang has a fixture source, otherwise the gang's own start_minute.

    WHY: The planner needs one start time whichever money source the gang has.

    Args:
        config: A GangConfig that passed validation.

    Returns:
        The start minute.

    Raises:
        ValueError: If the gang has neither a fixture source nor a start_minute.

    Example:
        # gang_start_minute(config_with_source_at_1440) == 1440
    """
    if config.source is not None:
        return config.source.start_minute
    if config.start_minute is None:
        raise ValueError(f"Gang '{config.gang_id}' has neither a fixture source nor a start_minute.")
    return config.start_minute


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
    # 1. Build the canonical dictionary of identity and budget; source, start minute and steps follow.
    plain: dict[str, Any] = {
        "gang_id": config.gang_id,
        "strength": config.strength,
        "budget": {
            "money_paise": config.budget.money_paise,
            "time_minutes": config.budget.time_minutes,
            "max_risk": config.budget.max_risk,
        },
    }
    # 2. Write source and start_minute only when set, so a fixture gang's plain form and run id stay as before.
    if config.source is not None:
        plain["source"] = {
            "account_id": config.source.account_id,
            "bank_index": config.source.bank_index,
            "opening_balance_paise": config.source.opening_balance_paise,
            "start_minute": config.source.start_minute,
        }
    if config.start_minute is not None:
        plain["start_minute"] = config.start_minute
    plain["steps"] = [
        {
            "block": step.block,
            "params": list(step.params),
        }
        for step in config.steps
    ]
    return plain


def gang_config_from_plain(data: dict[str, Any]) -> GangConfig:
    """
    WHAT: Reconstructs an immutable GangConfig instance from a plain Python dictionary.
    It converts primitive budget, source, and step specifications back into strongly typed dataclasses
    (tuples for sequences and parameter pairs), validating all rules before returning.

    WHY: Enables loading gang specifications from YAML fixtures and effective_config representations
    while guaranteeing that all schemas, strengths, step orderings, and dividing line rules are enforced.

    Args:
        data: Plain Python dictionary containing serialized gang specification fields.

    Returns:
        A validated, fully typed GangConfig instance.

    Raises:
        ValueError: If the configuration fails structural or semantic validation.

    Example:
        # config = gang_config_from_plain(plain_data)
        # assert config.gang_id == "gang_alpha"
    """
    # 1. Unpack budget and source specifications from target dictionary.
    target = data.get("gang", data)
    b_data = target["budget"]
    budget = GangBudget(int(b_data["money_paise"]), int(b_data["time_minutes"]), float(b_data["max_risk"]))
    source: FixtureSource | None = None
    if "source" in target:
        s_data = target["source"]
        source = FixtureSource(
            str(s_data["account_id"]), int(s_data["bank_index"]),
            int(s_data["opening_balance_paise"]), int(s_data["start_minute"])
        )
    start_minute: int | None = int(target["start_minute"]) if "start_minute" in target else None
    # 2. Reconstruct tuple of BlockStep items with (name, value) parameter tuples.
    steps: list[BlockStep] = []
    for s in target.get("steps", []):
        raw_p = s.get("params", [])
        p_items = raw_p.items() if isinstance(raw_p, dict) else raw_p
        steps.append(BlockStep(str(s["block"]), tuple((str(k), v) for k, v in p_items)))
    # 3. Instantiate GangConfig and validate constraints before returning.
    config = GangConfig(str(target["gang_id"]), int(target["strength"]), budget, source, tuple(steps), start_minute)
    validate_gang_config(config)
    return config


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
    WHAT: Enforces procedural order constraints: acquire before route, route needs acquire, warm_up after acquire,
    and a scam step after acquire and before route.

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

    for idx, step in enumerate(steps):
        if step.block == "acquire":
            has_acquired = True
        elif step.block == "scam":
            if not has_acquired:
                raise ValueError(f"Gang '{gang_id}' must acquire accounts before a scam step.")
            if has_routed:
                raise ValueError(f"Gang '{gang_id}' must scam before routing money, not after.")
        elif step.block == "take_control":
            if not has_acquired:
                raise ValueError(f"Gang '{gang_id}' cannot take control before acquiring accounts.")
        elif step.block == "warm_up":
            if not has_acquired:
                raise ValueError(f"Gang '{gang_id}' cannot warm up before acquiring accounts.")
        elif step.block == "route":
            if not has_acquired:
                raise ValueError(f"Gang '{gang_id}' must acquire accounts before routing money.")
            has_routed = True
        elif step.block == "cash_out":
            if not has_routed:
                raise ValueError(f"Gang '{gang_id}' must route money before cashing out.")
        elif step.block == "split_below_threshold":
            subsequent = [s.block for s in steps[idx + 1:]]
            if not any(b in ("route", "cash_out") for b in subsequent):
                raise ValueError(
                    f"Gang '{gang_id}' split_below_threshold must be followed by a route or cash_out step."
                )

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


def _validate_take_control_parameters(gang_id: str, steps: tuple[BlockStep, ...]) -> None:
    """
    WHAT: Asserts take_control steps configure devices count within configured bounds and strict boolean compromise flags.

    WHY: Decision E1 models device takeover with devices bounded by config and strict boolean flags per Decision E6.

    Args:
        gang_id: Gang identifier for error diagnostics.
        steps: Sequence of BlockStep items to validate.

    Returns:
        None. Raises ValueError if devices count is out of range or flags are not booleans.

    Example:
        # Non-boolean sim_swap raises ValueError.
        # _validate_take_control_parameters("gang_a", (BlockStep("take_control", (("sim_swap", "yes"),)),))
    """
    # 1. Fetch allowed devices range from tactic blocks configuration.
    min_devices, max_devices = get_take_control_devices_range()
    boolean_flags = ("sim_swap", "linked_number_switch", "emulator", "remote_access")
    for step in steps:
        if step.block != "take_control":
            continue
        params = step.params_dict
        if "devices" in params:
            devices_count = params["devices"]
            if not isinstance(devices_count, int) or isinstance(devices_count, bool) or not (min_devices <= devices_count <= max_devices):
                raise ValueError(
                    f"Gang '{gang_id}' take_control devices must be an integer in [{min_devices}, {max_devices}], got {devices_count}."
                )
        for flag in boolean_flags:
            if flag in params and not isinstance(params[flag], bool):
                raise ValueError(
                    f"Gang '{gang_id}' take_control parameter '{flag}' must be boolean, got {type(params[flag]).__name__}."
                )


def _validate_money_source(config: GangConfig) -> None:
    """
    WHAT: Asserts that the gang has exactly one money source: a fixture source or one scam step, never both and
    never neither, and that start_minute is given exactly when there is no fixture source.

    WHY: Money in the world must come from one declared place. A fixture source already carries its own start minute,
    so a second one on the gang could silently disagree with it.

    Args:
        config: GangConfig to validate.

    Returns:
        None. Raises ValueError with one plain sentence when the rule is broken.

    Example:
        # A gang with no source and no scam step raises ValueError.
        # _validate_money_source(GangConfig("g", 1, budget, None, (BlockStep("acquire"),), 0))
    """
    gang_id = config.gang_id
    scam_steps = [step for step in config.steps if step.block == "scam"]
    if config.source is None and not scam_steps:
        raise ValueError(
            f"Gang '{gang_id}' needs exactly one money source but has neither a fixture source nor a scam step."
        )
    if config.source is not None and scam_steps:
        raise ValueError(
            f"Gang '{gang_id}' needs exactly one money source but has both a fixture source and a scam step."
        )
    if len(scam_steps) > 1:
        raise ValueError(f"Gang '{gang_id}' may have only one scam step, got {len(scam_steps)}.")
    if config.source is not None and config.start_minute is not None:
        raise ValueError(
            f"Gang '{gang_id}' sets start_minute together with a fixture source, which already carries one."
        )
    if config.source is None:
        start_minute = config.start_minute
        if isinstance(start_minute, bool) or not isinstance(start_minute, int) or start_minute < 0:
            raise ValueError(
                f"Gang '{gang_id}' without a fixture source needs a start_minute of 0 or more, got {start_minute!r}."
            )


def _check_int_in_range(gang_id: str, name: str, value: Any, lower: int, upper: int) -> None:
    """
    WHAT: Asserts that a scam parameter is a true integer (not a boolean) between lower and upper inclusive.

    WHY: Both scam counts feed integer draws and list sizes; a float or boolean would fail far from the config.

    Args:
        gang_id: Gang identifier for error diagnostics.
        name: Parameter name for the error sentence.
        value: The configured value.
        lower: Smallest allowed value.
        upper: Largest allowed value.

    Returns:
        None. Raises ValueError if the value is not an integer in range.

    Example:
        # _check_int_in_range("g", "receiving_mules", 3, 1, 10) passes; value 11 raises ValueError.
    """
    if isinstance(value, bool) or not isinstance(value, int) or not (lower <= value <= upper):
        raise ValueError(f"Gang '{gang_id}' scam {name} must be an integer in [{lower}, {upper}], got {value!r}.")


def _validate_scam_parameters(gang_id: str, steps: tuple[BlockStep, ...]) -> None:
    """
    WHAT: Asserts that a scam step names receiving_mules as an integer inside the block's configured range, and that
    an explicit victims_per_receiving_mule is an integer inside the configured victims range.

    WHY: Every victim is attached to a receiving mule, so the mule count must be explicit and bounded; the victim
    count may be left out to be drawn.

    Args:
        gang_id: Gang identifier for error diagnostics.
        steps: Sequence of BlockStep items to validate.

    Returns:
        None. Raises ValueError if a scam parameter is missing, not an integer, or out of range.

    Example:
        # _validate_scam_parameters("g", (BlockStep("scam"),)) raises: receiving_mules is required.
    """
    min_mules, max_mules = get_scam_receiving_mules_range()
    min_victims, max_victims = get_scam_victims_per_mule_range()
    for step in steps:
        if step.block != "scam":
            continue
        params = step.params_dict
        if "receiving_mules" not in params:
            raise ValueError(f"Gang '{gang_id}' scam step needs receiving_mules.")
        _check_int_in_range(gang_id, "receiving_mules", params["receiving_mules"], min_mules, max_mules)
        if "victims_per_receiving_mule" in params:
            _check_int_in_range(
                gang_id, "victims_per_receiving_mule", params["victims_per_receiving_mule"], min_victims, max_victims
            )


def _validate_acquire_parameters(gang_id: str, steps: tuple[BlockStep, ...]) -> None:
    """
    WHAT: Asserts that an acquire step's origin_mix, when given, is one of the values allowed in
    config/tactic_blocks.yaml.

    WHY: A misspelled mix would otherwise be treated as a different, unintended mix by the planner.

    Args:
        gang_id: Gang identifier for error diagnostics.
        steps: Sequence of BlockStep items to validate.

    Returns:
        None. Raises ValueError if origin_mix is not an allowed value.

    Example:
        # _validate_acquire_parameters("g", (BlockStep("acquire", (("origin_mix", "random"),)),)) raises.
    """
    allowed_values, _default = get_origin_mix_options()
    for step in steps:
        if step.block != "acquire":
            continue
        params = step.params_dict
        if "origin_mix" in params and params["origin_mix"] not in allowed_values:
            raise ValueError(
                f"Gang '{gang_id}' acquire origin_mix must be one of {', '.join(allowed_values)}, "
                f"got {params['origin_mix']!r}."
            )


def _validate_scam_route_agreement(gang_id: str, steps: tuple[BlockStep, ...]) -> None:
    """
    WHAT: When a gang has both a scam step and a route step, asserts that route accounts_per_layer equals the scam
    step's receiving_mules and that the route step gives no amount_inr.

    WHY: Layer 1 of the route must be exactly the receiving mules, and the route amount is the victim money that
    arrives, so a configured amount could only disagree with it.

    Args:
        gang_id: Gang identifier for error diagnostics.
        steps: Sequence of BlockStep items to validate.

    Returns:
        None. Raises ValueError if the route step disagrees with the scam step.

    Example:
        # scam receiving_mules=2 with route accounts_per_layer=3 raises ValueError.
    """
    scam_steps = [step for step in steps if step.block == "scam"]
    route_steps = [step for step in steps if step.block == "route"]
    if not scam_steps or not route_steps:
        return
    receiving_mules = scam_steps[0].params_dict["receiving_mules"]
    for route_step in route_steps:
        route_params = route_step.params_dict
        if "amount_inr" in route_params:
            raise ValueError(f"Gang '{gang_id}' route must not set amount_inr when its money comes from a scam step.")
        if "accounts_per_layer" not in route_params or route_params["accounts_per_layer"] != receiving_mules:
            raise ValueError(
                f"Gang '{gang_id}' route accounts_per_layer must equal the scam step's "
                f"receiving_mules of {receiving_mules}."
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

    # 5. Validate take control parameters.
    _validate_take_control_parameters(config.gang_id, config.steps)

    # 6. Validate the money source, scam parameters, and the scam-to-route agreement.
    _validate_money_source(config)
    _validate_scam_parameters(config.gang_id, config.steps)
    _validate_acquire_parameters(config.gang_id, config.steps)
    _validate_scam_route_agreement(config.gang_id, config.steps)
