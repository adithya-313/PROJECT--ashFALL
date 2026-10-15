"""
WHAT: World configuration resolution, mode dispatching, and capacity decision engine.
It resolves simulation parameters from preset templates, knob overrides, or expert configuration files.
It computes resource feasibility against machine limits, generating plain-English refusals with actionable guidance.

WHY: Provides an intuitive parameterization interface allowing operators to scale worlds safely.
Preventing oversized simulation execution protects host machines from memory exhaustion and runaway runtimes.
Clear diagnostic feedback suggests the largest fitting preset or population size to assist operator planning.

TERM: preset: A named pre-configured world configuration specifying standard population and shop counts,
      e.g. the tiny or small preset.
TERM: knob: A parameter override that automatically scales dependent world dimensions by preset ratios,
      e.g. adjusting people with shops following proportionally.
TERM: ratio scaling: Proportionally calculating commercial shop counts from citizen population using baseline ratios,
      e.g. scaling shops as people change.
TERM: expert file: A structured YAML file providing complete explicit simulation parameters for advanced scenarios,
      e.g. an expert.yaml file.
TERM: refusal: A safety rejection preventing execution of oversized worlds with diagnostic feedback and suggestions,
      e.g. declining a run that exceeds available RAM.
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.config_validator import load_yaml
from world.background_config import read_background_realism
from world.estimates import (
    MachineProfile,
    WorldEstimate,
    describe_estimate,
    estimate_world,
)
from world.fast_mode import FastRunSettings
from world.machine_limits import MachineState, find_problems


@dataclass(frozen=True)
class SizingDecision:
    """
    WHAT: Immutable record holding the feasibility evaluation of a proposed world simulation run.
    It encapsulates whether the world fits within host limits, projected resource estimates,
    any identified limit violations, and a plain-English message explaining the decision.

    WHY: Decouples capacity evaluation from execution control flow, allowing CLI tools to inspect outcomes cleanly.
    Freezing the decision ensures that planning metadata cannot be altered during downstream execution.
    It standardizes the structure of acceptance and refusal diagnostics across interfaces.

    Args:
        fits: Boolean flag indicating True if all operational machine limits are satisfied.
        estimate: WorldEstimate record containing projected transaction counts, duration, file size, and heap memory.
        problems: List of plain-English strings describing each broken machine capacity limit.
        message: Multi-line formatted diagnostic message presenting decision details, problems, and suggestions.

    Example:
        >>> from world.estimates import WorldEstimate
        >>> from world.world_sizing import SizingDecision
        >>> est = WorldEstimate(1000, 1.0, 350000, 1000000)
        >>> dec = SizingDecision(True, est, [], "Fits on this machine.")
        >>> (dec.fits, len(dec.problems))
        (True, 0)
    """

    fits: bool
    estimate: WorldEstimate
    problems: list[str]
    message: str


def _load_expert_doc(expert_path: Path | str) -> dict[str, Any]:
    """
    WHAT: Reads and parses an expert YAML configuration document, enforcing dictionary type and schema keys.
    It verifies that the root structure is a mapping and contains exactly the mandatory five simulation parameters.
    Any extraneous unknown fields or missing required fields trigger an immediate, explanatory ValueError.

    WHY: Isolates file loading and schema validation logic into a dedicated, testable helper routine.
    Strict key validation catches typos such as 'n_peopel' early before parameters propagate into execution.
    Rejecting non-dictionary root structures prevents downstream type errors when indexing configuration keys.

    Args:
        expert_path: Filesystem path to the expert configuration YAML document.

    Returns:
        Dictionary mapping validated configuration keys to their corresponding parsed YAML values.

    Raises:
        ValueError: If the file is not a YAML dictionary or contains unknown or missing fields.

    Example:
        >>> from pathlib import Path
        >>> import tempfile
        >>> with tempfile.NamedTemporaryFile("w", delete=False, suffix=".yaml") as f:
        ...     _ = f.write("seed: 1\\nn_people: 10\\nn_shops: 2\\nn_days: 61\\nrange_name: training\\n")
        ...     p = f.name
        >>> d = _load_expert_doc(p)
        >>> Path(p).unlink()
        >>> (d["seed"], d["n_people"])
        (1, 10)
    """
    # 1. Parse YAML file into Python document object.
    doc = load_yaml(Path(expert_path))
    if not isinstance(doc, dict):
        raise ValueError(f"Expert file '{expert_path}' must contain a YAML mapping.")

    # 2. Check for unknown fields not recognized by the simulation schema.
    expected_fields = {"seed", "n_people", "n_shops", "n_days", "range_name"}
    actual_fields = set(doc.keys())
    unknown = actual_fields - expected_fields
    if unknown:
        raise ValueError(f"Expert file '{expert_path}' contains unknown field(s): {', '.join(sorted(unknown))}")

    # 3. Check for missing fields required by the simulation schema.
    missing = expected_fields - actual_fields
    if missing:
        raise ValueError(f"Expert file '{expert_path}' is missing required field(s): {', '.join(sorted(missing))}")

    return doc


def _resolve_expert(
    expert_path: Path | str,
    seed: int | None,
    preset: str | None,
    n_people: int | None,
    n_shops: int | None,
    n_days: int | None,
    range_name: str | None,
) -> FastRunSettings:
    """
    WHAT: Resolves world settings from an expert configuration file while ensuring no CLI options conflict.
    It inspects all optional command-line flags, raising ValueError if any flag is supplied alongside --expert.
    Validated dictionary attributes from the YAML file are packaged into a FastRunSettings record.

    WHY: Enforces mutual exclusivity between file-based configuration and command-line overrides.
    Allowing mixed configuration sources risks ambiguity regarding which source takes precedence.
    Delegating file parsing to a validated helper keeps this resolution function compact and focused.

    Args:
        expert_path: Filesystem path to the expert YAML configuration file.
        seed: Random seed parameter passed on CLI, which must be None in expert mode.
        preset: Preset template name passed on CLI, which must be None in expert mode.
        n_people: Population count passed on CLI, which must be None in expert mode.
        n_shops: Shop count passed on CLI, which must be None in expert mode.
        n_days: Horizon days passed on CLI, which must be None in expert mode.
        range_name: Split name passed on CLI, which must be None in expert mode.

    Returns:
        FastRunSettings populated directly from the parsed expert YAML file.

    Raises:
        ValueError: If conflicting options are passed, or if the file schema is invalid.

    Example:
        >>> from pathlib import Path
        >>> import tempfile
        >>> with tempfile.NamedTemporaryFile("w", delete=False, suffix=".yaml") as f:
        ...     _ = f.write("seed: 10\\nn_people: 100\\nn_shops: 20\\nn_days: 61\\nrange_name: training\\n")
        ...     p = f.name
        >>> s = _resolve_expert(p, None, None, None, None, None, None)
        >>> Path(p).unlink()
        >>> (s.seed, s.n_people)
        (10, 100)
    """
    # 1. Reject invocations that mix expert mode with other configuration parameters.
    extra_options = [
        opt for opt, val in [
            ("--seed", seed), ("--preset", preset), ("--people", n_people),
            ("--shops", n_shops), ("--days", n_days), ("--range-name", range_name),
        ] if val is not None
    ]
    if extra_options:
        raise ValueError(
            f"Expert mode requires only --expert; conflicting options passed: {', '.join(extra_options)}"
        )

    # 2. Parse file and build validated settings container.
    doc = _load_expert_doc(expert_path)
    return FastRunSettings(
        seed=doc["seed"],
        n_people=doc["n_people"],
        n_shops=doc["n_shops"],
        n_days=doc["n_days"],
        range_name=doc["range_name"],
    )


def _compute_preset_shops(
    base_people: int,
    base_shops: int,
    n_people: int | None,
    n_shops: int | None,
) -> int:
    """
    WHAT: Determines commercial shop count from explicit override, preset baseline, or proportional ratio scaling.
    If an explicit shop count is provided, it is returned directly without any alteration.
    Otherwise, if people were overridden, shops scale via max(1, floor(people * base_shops / base_people + 0.5)).

    WHY: Maintains realistic merchant-to-citizen ratios when operators scale population via the people knob.
    Permitting explicit shop overrides allows specialized stress scenarios with unusual shop concentrations.
    Enforcing a floor of at least 1 shop prevents degenerate economic environments without merchants.

    Args:
        base_people: Baseline citizen population defined by the chosen preset template.
        base_shops: Baseline commercial shop count defined by the chosen preset template.
        n_people: Citizen population override passed by operator, or None if using preset baseline.
        n_shops: Commercial shop count override passed by operator, or None for automatic calculation.

    Returns:
        Integer commercial shop count reflecting overrides, baseline, or proportional scaling.

    Example:
        >>> _compute_preset_shops(300, 40, 600, None)
        80
        >>> _compute_preset_shops(300, 40, None, None)
        40
        >>> _compute_preset_shops(300, 40, 600, 55)
        55
    """
    # 1. Return direct user override if explicitly supplied.
    if n_shops is not None:
        return n_shops

    # 2. Return unscaled preset baseline if population was not adjusted.
    if n_people is None:
        return base_shops

    # 3. Apply rounded proportional ratio scaling based on population adjustment.
    scaled = math.floor(n_people * base_shops / base_people + 0.5)
    return max(1, scaled)


def _resolve_preset(
    preset: str,
    seed: int | None,
    n_people: int | None,
    n_shops: int | None,
    n_days: int | None,
    range_name: str | None,
    sizing_config: dict[str, Any],
) -> FastRunSettings:
    """
    WHAT: Resolves world settings from a named preset template with optional parameter knob overrides.
    It requires an explicit random seed and scales commercial shops proportionally if people are overridden.
    Direct overrides for shops, days, or dataset range take precedence over preset baseline values.

    WHY: Simplifies simulation setup by providing curated standard world sizes chosen for testing.
    Proportional shop scaling ensures merchant-to-citizen economic ratios remain realistic when population shifts.
    Requiring explicit seeds prevents operators from accidentally running simulations with unintended seeds.

    Args:
        preset: Named preset template identifier (e.g. 'tiny', 'small', 'medium', 'large').
        seed: Master random seed integer required for reproducibility.
        n_people: Optional citizen population count overriding the preset baseline.
        n_shops: Optional commercial shop count overriding preset or ratio-scaled shops.
        n_days: Optional simulation duration in days overriding the preset baseline.
        range_name: Optional dataset split name overriding the preset baseline.
        sizing_config: Configuration dictionary defining preset templates and baseline dimensions.

    Returns:
        FastRunSettings populated from preset baselines and applied knob overrides.

    Raises:
        ValueError: If the preset name is unknown or if the random seed is omitted.

    Example:
        >>> from world.estimates import read_sizing_config
        >>> s = _resolve_preset("small", 7, 600, None, None, None, read_sizing_config())
        >>> (s.n_people, s.n_shops)
        (600, 80)
    """
    # 1. Verify preset name exists in configuration.
    presets = sizing_config.get("presets", {})
    if preset not in presets or not isinstance(presets[preset], dict):
        available = [p for p in presets if isinstance(presets[p], dict)]
        raise ValueError(f"Unknown preset '{preset}'. Available presets: {', '.join(available)}")

    # 2. Enforce explicit random seed requirement in preset mode.
    if seed is None:
        raise ValueError("Preset mode requires an explicit --seed parameter.")

    # 3. Retrieve baseline template values and resolve overrides.
    base = presets[preset]
    resolved_days = n_days if n_days is not None else base["n_days"]
    resolved_range = range_name if range_name is not None else base["range_name"]
    resolved_people = n_people if n_people is not None else base["n_people"]
    resolved_shops = _compute_preset_shops(base["n_people"], base["n_shops"], n_people, n_shops)

    # 4. Assemble resolved settings structure.
    return FastRunSettings(
        seed=seed,
        n_people=resolved_people,
        n_shops=resolved_shops,
        n_days=resolved_days,
        range_name=resolved_range,
    )


def _resolve_explicit(
    seed: int | None,
    n_people: int | None,
    n_shops: int | None,
    n_days: int | None,
    range_name: str | None,
) -> FastRunSettings:
    """
    WHAT: Resolves simulation settings in explicit parameter mode where every option is supplied individually.
    It verifies that seed, population, shops, duration, and partition range are all non-None.
    If any required argument is missing, it raises a plain-English ValueError listing all missing flags.

    WHY: Preserves exact backwards compatibility with legacy CLI invocations specifying explicit world parameters.
    Explicit mode prevents silent default parameter assumptions, ensuring operator intent is fully specified.
    Listing all missing parameters at once helps operators rectify invocation commands in a single pass.

    Args:
        seed: Master random seed integer.
        n_people: Total citizen population count.
        n_shops: Total commercial shop count.
        n_days: Simulation calendar duration in full days.
        range_name: Dataset partition name ('training', 'test', 'stress').

    Returns:
        FastRunSettings populated with validated explicit command-line values.

    Raises:
        ValueError: If any of the five required simulation parameters is omitted.

    Example:
        >>> s = _resolve_explicit(101, 60, 12, 61, "training")
        >>> (s.seed, s.n_people, s.n_shops)
        (101, 60, 12)
    """
    # 1. Identify any omitted parameters from the explicit configuration tuple.
    missing = [
        param for param, val in [
            ("seed", seed), ("n_people", n_people), ("n_shops", n_shops),
            ("n_days", n_days), ("range_name", range_name),
        ] if val is None
    ]
    if missing:
        raise ValueError(f"Explicit mode requires all parameters; missing: {', '.join(missing)}")

    # 2. Package explicit values into FastRunSettings container.
    return FastRunSettings(
        seed=seed,  # type: ignore[arg-type]
        n_people=n_people,  # type: ignore[arg-type]
        n_shops=n_shops,  # type: ignore[arg-type]
        n_days=n_days,  # type: ignore[arg-type]
        range_name=range_name,  # type: ignore[arg-type]
    )


def _validate_resolved(settings: FastRunSettings) -> None:
    """
    WHAT: Validates that resolved simulation settings satisfy integer positivity and calendar horizon rules.
    It verifies that seed, population, shops, and days are strictly positive integers.
    It reads background realism configuration to enforce that duration exceeds festival season peak bounds.

    WHY: Catches invalid, negative, or degenerate world parameters before initiating expensive simulation runs.
    Ensuring calendar duration extends past festival peak windows protects calendar runners from indexing errors.
    Clear error messages guide operators to the smallest permissible duration for their chosen range.

    Args:
        settings: FastRunSettings container holding candidate simulation parameters.

    Returns:
        None upon successful validation without errors.

    Raises:
        ValueError: If any parameter count is non-positive, or if n_days is too short for the range.

    Example:
        >>> from world.fast_mode import FastRunSettings
        >>> s = FastRunSettings(101, 60, 12, 61, "training")
        >>> _validate_resolved(s)
    """
    # 1. Enforce positive integer requirements across entity counts and simulation duration.
    for name, val in [
        ("seed", settings.seed),
        ("n_people", settings.n_people),
        ("n_shops", settings.n_shops),
        ("n_days", settings.n_days),
    ]:
        if not isinstance(val, int) or val <= 0:
            raise ValueError(f"Parameter {name} ({val}) must be a positive integer.")

    # 2. Read authoritative festival season peak bounds from background realism configuration.
    bg_config = read_background_realism()
    festival_bounds = bg_config["parameters"]["festival_peak_day_in_world"]
    if settings.range_name not in festival_bounds:
        raise ValueError(
            f"Unknown range_name '{settings.range_name}'. Allowed: {', '.join(festival_bounds.keys())}"
        )

    # 3. Assert simulation duration exceeds range festival peak upper bound.
    upper_bound = festival_bounds[settings.range_name][1]
    if settings.n_days <= upper_bound:
        min_allowed = upper_bound + 1
        raise ValueError(
            f"Simulation duration n_days ({settings.n_days}) must be greater than {upper_bound} "
            f"for range '{settings.range_name}'; the smallest allowed n_days is {min_allowed}."
        )


def resolve_settings(
    seed: int | None = None,
    preset: str | None = None,
    n_people: int | None = None,
    n_shops: int | None = None,
    n_days: int | None = None,
    range_name: str | None = None,
    expert_path: Path | str | None = None,
    sizing_config: dict[str, Any] | None = None,
) -> FastRunSettings:
    """
    WHAT: Dispatches simulation parameter resolution across expert, preset, and explicit invocation modes.
    It enforces that exactly one parameterization mode is selected and resolves all entity counts and horizons.
    The resolved configuration is validated for positivity and festival calendar horizon constraints.

    WHY: Serves as the authoritative single entry point for simulation configuration resolution.
    Disallowing ambiguous combinations of modes prevents unexpected parameter overrides.
    Early validation protects subsequent calibration and execution routines from malformed configurations.

    Args:
        seed: Master random seed integer.
        preset: Named world size template string.
        n_people: Total citizen population count.
        n_shops: Total commercial shop count.
        n_days: Simulation calendar duration in full days.
        range_name: Dataset partition name ('training', 'test', 'stress').
        expert_path: Filesystem path to an expert YAML configuration file.
        sizing_config: Sizing policy dictionary loaded from config/world_sizing.yaml.

    Returns:
        Validated FastRunSettings instance ready for simulation execution.

    Raises:
        ValueError: If multiple modes are combined, required fields are missing, or constraints are violated.

    Example:
        >>> from world.estimates import read_sizing_config
        >>> from world.world_sizing import resolve_settings
        >>> cfg = read_sizing_config()
        >>> s = resolve_settings(seed=7, preset="tiny", sizing_config=cfg)
        >>> (s.seed, s.n_people, s.n_shops, s.n_days, s.range_name)
        (7, 60, 12, 61, 'training')
    """
    # 1. Fall back to default sizing configuration if not provided by caller.
    if sizing_config is None:
        from world.estimates import read_sizing_config
        sizing_config = read_sizing_config()

    # 2. Dispatch to the appropriate resolution pathway based on mode flags.
    if expert_path is not None:
        settings = _resolve_expert(expert_path, seed, preset, n_people, n_shops, n_days, range_name)
    elif preset is not None:
        settings = _resolve_preset(preset, seed, n_people, n_shops, n_days, range_name, sizing_config)
    else:
        settings = _resolve_explicit(seed, n_people, n_shops, n_days, range_name)

    # 3. Perform universal validation across resolved parameters.
    _validate_resolved(settings)
    return settings


def largest_fitting_people(
    n_days: int,
    profile: MachineProfile,
    state: MachineState,
    sizing_config: dict[str, Any],
    max_minutes: float,
) -> int:
    """
    WHAT: Determines the maximum citizen population that safely fits on the machine for a given calendar duration.
    It calculates a theoretical upper bound across memory, disk, and time limits, then steps down to find the maximum.
    If even a single citizen exceeds machine capacity limits, the function returns 0.

    WHY: Provides concrete, actionable sizing recommendations when a requested world is refused for being too large.
    Stepping down from an arithmetic upper bound ensures the returned population count fits while count+1 fails.
    Calculating limits dynamically reflects real-time host memory and disk availability.

    Args:
        n_days: Simulation calendar horizon duration in full days.
        profile: MachineProfile holding calibrated transaction throughput and per-payment byte costs.
        state: MachineState holding available free physical memory and free disk storage.
        sizing_config: Configuration dictionary defining machine limits and safety margins.
        max_minutes: Maximum permissible execution duration in minutes.

    Returns:
        Integer population count representing the largest fitting world, or 0 if no world fits.

    Example:
        >>> from world.estimates import MachineProfile, read_sizing_config
        >>> from world.machine_limits import MachineState
        >>> from world.world_sizing import largest_fitting_people
        >>> prof = MachineProfile(5000.0, 0.8, 350.0, 1024.0, 2915)
        >>> state = MachineState(100_000_000, 500_000_000)
        >>> largest_fitting_people(61, prof, state, read_sizing_config(), 30.0)
        870
    """
    # 1. Calculate per-person transaction generation rate scaled by safety margin.
    pppd = profile.payments_per_person_day
    margin = float(sizing_config["payment_count_margin"]["value"])
    person_rate = n_days * pppd * margin

    # 2. Extract policy thresholds for allowable memory and disk consumption.
    limits = sizing_config["machine_limits"]
    mem_share = float(limits.get("memory_share", 0.5))
    disk_share = float(limits.get("disk_share", 0.5))

    # 3. Calculate maximum allowable transaction count across all three operational constraints.
    max_pmts_time = max_minutes * 60.0 * profile.payments_per_second
    max_pmts_mem = (state.free_memory_bytes * mem_share) / profile.memory_bytes_per_payment
    max_pmts_disk = (state.free_disk_bytes * disk_share) / profile.file_bytes_per_payment
    upper_pmts = min(max_pmts_time, max_pmts_mem, max_pmts_disk)

    # 4. Formulate tight arithmetic upper bound and step down until all constraints are satisfied.
    guess = max(0, int(upper_pmts / person_rate) + 2)
    for people in range(guess, 0, -1):
        estimate = estimate_world(people, n_days, profile, sizing_config)
        if not find_problems(estimate, state, limits, max_minutes):
            return people

    return 0


def largest_fitting_preset(
    profile: MachineProfile,
    state: MachineState,
    sizing_config: dict[str, Any],
    max_minutes: float,
) -> str | None:
    """
    WHAT: Identifies the largest standard preset world whose resource demands fit entirely within host limits.
    It evaluates configured presets in descending order of citizen population and checks each against machine limits.
    It returns the name string of the largest feasible preset template, or None if no presets fit.

    WHY: Offers instant, standardized fallback options when custom or oversized simulations are refused.
    Sorting by population guarantees that the operator is directed to the most substantial fitting workload.
    Returning None clearly communicates when host resources are too constrained for any preset.

    Args:
        profile: MachineProfile holding empirical throughput and cost rates.
        state: MachineState holding real-time available physical memory and free disk space.
        sizing_config: Sizing configuration dictionary defining presets and machine limits.
        max_minutes: Maximum permissible execution duration in minutes.

    Returns:
        String name of the largest fitting preset, or None if no preset fits on the machine.

    Example:
        >>> from world.estimates import MachineProfile, read_sizing_config
        >>> from world.machine_limits import MachineState
        >>> from world.world_sizing import largest_fitting_preset
        >>> prof = MachineProfile(5000.0, 0.8, 350.0, 1024.0, 2915)
        >>> state = MachineState(100_000_000, 500_000_000)
        >>> largest_fitting_preset(prof, state, read_sizing_config(), 30.0)
        'small'
    """
    # 1. Filter preset definitions to extract mapping configurations with population counts.
    presets = sizing_config.get("presets", {})
    names = [p for p in presets if isinstance(presets[p], dict) and "n_people" in presets[p]]

    # 2. Sort presets in descending order of population to evaluate the largest options first.
    sorted_names = sorted(names, key=lambda p: presets[p]["n_people"], reverse=True)

    # 3. Test each preset against machine constraints and return first fitting candidate.
    limits = sizing_config["machine_limits"]
    for name in sorted_names:
        p_data = presets[name]
        est = estimate_world(p_data["n_people"], p_data["n_days"], profile, sizing_config)
        if not find_problems(est, state, limits, max_minutes):
            return name

    return None


def _build_refusal_suggestions(
    settings: FastRunSettings,
    estimate: WorldEstimate,
    problems: list[str],
    profile: MachineProfile,
    state: MachineState,
    sizing_config: dict[str, Any],
    max_minutes: float,
) -> list[str]:
    """
    WHAT: Generates actionable plain-English suggestion bullets explaining how to fit a refused simulation.
    It proposes the largest fitting standard preset, the maximum fitting population, and runtime budget increases.
    Suggestions are conditionally included only when relevant to the detected capacity bottlenecks.

    WHY: Transforms raw limit failure diagnostics into helpful guidance that unblocks operators immediately.
    Suggesting specific parameter overrides prevents operators from having to guess feasible configurations.
    Limiting budget increase suggestions to time-limited runs prevents irrelevant advice on memory bottlenecks.

    Args:
        settings: Proposed FastRunSettings specifying the requested duration and population.
        estimate: Computed WorldEstimate detailing projected resource requirements.
        problems: List of plain-English strings describing broken capacity limits.
        profile: Calibrated MachineProfile capturing machine execution characteristics.
        state: MachineState measuring host physical memory and disk availability.
        sizing_config: Sizing configuration dictionary defining presets and machine policies.
        max_minutes: Active runtime duration threshold in minutes.

    Returns:
        List of formatted recommendation strings explaining actionable alternatives.

    Example:
        >>> from world.estimates import MachineProfile, WorldEstimate, read_sizing_config
        >>> from world.fast_mode import FastRunSettings
        >>> from world.machine_limits import MachineState
        >>> prof = MachineProfile(5000.0, 0.8, 350.0, 1024.0, 2915)
        >>> state = MachineState(100_000_000, 500_000_000)
        >>> settings = FastRunSettings(101, 5000, 500, 61, "training")
        >>> est = WorldEstimate(500000, 100.0, 150000000, 500000000)
        >>> cfg = read_sizing_config()
        >>> sugs = _build_refusal_suggestions(settings, est, ["time: over limit"], prof, state, cfg, 30.0)
        >>> len(sugs) >= 1
        True
    """
    suggestions: list[str] = []

    # 1. Suggest largest fitting preset template if one exists, or state none fit.
    best_preset = largest_fitting_preset(profile, state, sizing_config, max_minutes)
    if best_preset:
        p_data = sizing_config["presets"][best_preset]
        p_est = estimate_world(p_data["n_people"], p_data["n_days"], profile, sizing_config)
        suggestions.append(
            f"  - Use the biggest preset that fits: --preset {best_preset} ({describe_estimate(p_est)})"
        )
    else:
        suggestions.append("  - None of the presets fit on this machine right now.")

    # 2. Suggest largest population count that fits with current duration if at least 1 person fits.
    max_p = largest_fitting_people(settings.n_days, profile, state, sizing_config, max_minutes)
    if max_p >= 1:
        suggestions.append(f"  - Or keep your settings with fewer people: --people {max_p}")

    # 3. Suggest overriding runtime budget if execution duration was an identified problem.
    if any(p.startswith("time:") for p in problems):
        ceil_min = math.ceil(estimate.seconds / 60.0)
        suggestions.append(f"  - Or allow more time for this run: --max-minutes {ceil_min}")

    return suggestions


def _format_refusal_message(
    estimate: WorldEstimate,
    problems: list[str],
    settings: FastRunSettings,
    profile: MachineProfile,
    state: MachineState,
    sizing_config: dict[str, Any],
    max_minutes: float,
) -> str:
    """
    WHAT: Formats a complete multi-line plain-English refusal message when a world exceeds machine limits.
    It combines the refusal header, resource estimate, enumerated problem causes, and actionable suggestions.
    The message concludes with a reassurance that estimates lean high on purpose to avoid underestimating.

    WHY: Standardizes refusal output formatting so operators receive clear, consistent diagnostics.
    Displaying the exact resource estimate alongside broken thresholds clarifies why the world was declined.
    Concluding with sizing margin context reminds operators that refusal thresholds are intentionally conservative.

    Args:
        estimate: Computed WorldEstimate detailing projected resource requirements.
        problems: List of plain-English strings describing broken capacity limits.
        settings: Proposed FastRunSettings specifying requested world dimensions.
        profile: Calibrated MachineProfile capturing machine execution characteristics.
        state: MachineState measuring host physical memory and disk availability.
        sizing_config: Sizing configuration dictionary defining presets and machine policies.
        max_minutes: Active runtime duration threshold in minutes.

    Returns:
        Formatted multi-line refusal message string.

    Example:
        >>> from world.estimates import MachineProfile, WorldEstimate, read_sizing_config
        >>> from world.fast_mode import FastRunSettings
        >>> from world.machine_limits import MachineState
        >>> prof = MachineProfile(5000.0, 0.8, 350.0, 1024.0, 2915)
        >>> state = MachineState(100_000_000, 500_000_000)
        >>> settings = FastRunSettings(101, 5000, 500, 61, "training")
        >>> est = WorldEstimate(500000, 100.0, 150000000, 500000000)
        >>> cfg = read_sizing_config()
        >>> msg = _format_refusal_message(est, ["memory: needs 500 MB"], settings, prof, state, cfg, 30.0)
        >>> "Refused:" in msg
        True
    """
    # 1. Assemble refusal diagnostic header and problem enumeration.
    lines = [
        "Refused: this world is too big for this machine. Nothing was run.",
        f"Estimate: {describe_estimate(estimate)}",
        "Why:",
    ]
    for p in problems:
        lines.append(f"  - {p}")
    lines.append("What you can do:")

    # 2. Append actionable suggestions for adapting simulation settings.
    suggestions = _build_refusal_suggestions(
        settings, estimate, problems, profile, state, sizing_config, max_minutes
    )
    lines.extend(suggestions)

    # 3. Append concluding estimation rationale.
    lines.append("Estimates lean high on purpose, so the real run is usually smaller and faster.")
    return "\n".join(lines)


def decide_run(
    settings: FastRunSettings,
    profile: MachineProfile,
    state: MachineState,
    sizing_config: dict[str, Any],
    max_minutes: float,
) -> SizingDecision:
    """
    WHAT: Evaluates resource feasibility for proposed simulation settings and builds a comprehensive sizing decision.
    If constraints are broken, it formats a refusal diagnostic with problem causes and actionable alternatives.
    If all constraints are satisfied, it returns an approval decision linking the projected resource estimate.

    WHY: Enforces simulation safety policies before any disk writes or heavy computation take place.
    Explaining refusal reasons and offering concrete suggestions empowers operators to resolve capacity issues.
    Structuring decision outputs enables both interactive command shells and automated pipelines to handle outcomes.

    Args:
        settings: FastRunSettings specifying candidate simulation population, shops, and duration.
        profile: MachineProfile holding empirical host throughput and per-payment resource rates.
        state: MachineState holding real-time available memory and free disk capacity.
        sizing_config: Sizing configuration dictionary defining presets, margins, and limits.
        max_minutes: Maximum permissible execution duration in minutes.

    Returns:
        SizingDecision container holding approval status, estimate, problems list, and message string.

    Example:
        >>> from world.estimates import MachineProfile, read_sizing_config
        >>> from world.fast_mode import FastRunSettings
        >>> from world.machine_limits import MachineState
        >>> from world.world_sizing import decide_run
        >>> prof = MachineProfile(5000.0, 0.8, 350.0, 1024.0, 2915)
        >>> state = MachineState(100_000_000, 500_000_000)
        >>> settings = FastRunSettings(101, 60, 12, 61, "training")
        >>> dec = decide_run(settings, prof, state, read_sizing_config(), 30.0)
        >>> dec.fits
        True
    """
    # 1. Compute resource estimate and evaluate against machine limits.
    estimate = estimate_world(settings.n_people, settings.n_days, profile, sizing_config)
    problems = find_problems(estimate, state, sizing_config["machine_limits"], max_minutes)

    # 2. Return positive approval decision if no constraints were violated.
    if not problems:
        return SizingDecision(
            fits=True,
            estimate=estimate,
            problems=[],
            message=f"Fits on this machine. {describe_estimate(estimate)}",
        )

    # 3. Format refusal diagnostic message explaining limit violations and suggesting remedies.
    msg = _format_refusal_message(
        estimate, problems, settings, profile, state, sizing_config, max_minutes
    )
    return SizingDecision(fits=False, estimate=estimate, problems=problems, message=msg)
