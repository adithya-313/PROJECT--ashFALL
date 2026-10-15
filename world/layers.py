"""
WHAT: Config layer parsing, validation, stacking, and effective-world description module.
It reads overlay YAML layers that modify specific parameter ranges on top of baseline configs.
It validates nesting invariants across all stacked changes and prints an audit trail summary.

WHY: Enables targeted scenario and sensitivity testing without modifying baseline config files.
Tracking individual layer changes and merge order provides complete auditability of simulation inputs.
Enforcing nesting rules guarantees that layered configurations never violate valid parameter domains.

TERM: layer: A small configuration overlay file specifying range overrides for existing settings,
      e.g. a YAML file shifting the Diwali festival peak window.
TERM: merge order: The sequential evaluation order where subsequent layers override previous layers on clashes,
      e.g. layer B winning over layer A when both modify the same range.
TERM: effective world: The resulting composite configuration state after stacking all specified layers in order,
      e.g. displaying all applied parameter mutations alongside unchanged settings.
TERM: nesting: The invariant requiring parameter ranges to satisfy training inside test inside stress,
      e.g. ensuring [low, high] for training is contained within test bounds.
TERM: override: A targeted replacement of an existing parameter range value by a layer,
      e.g. updating training [30, 50] to [40, 45].
"""

import copy
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from core.config_validator import check_nested_ranges
from world.world_configs import CONFIG_NAMES, WorldConfigs, read_world_configs


@dataclass(frozen=True)
class Layer:
    """
    WHAT: Immutable record representing a validated configuration layer loaded from a YAML file.
    It encapsulates the layer name, rationale, parameter changes mapping, SHA-256 digest, and source path.
    Downstream layering logic consumes this structure to stack parameter overrides in order.

    WHY: Bundles file metadata and configuration overrides into a typed, validated structure.
    Freezing the dataclass prevents accidental alteration of layer specifications after loading.
    Retaining SHA-256 digests and source paths enables accurate run manifest tracking.

    Args:
        name: Unique string identifier for the layer.
        reason: Explanation describing why this layer was created and what scenarios it tests.
        changes: Nested dictionary mapping configuration names to settings and range overrides.
        sha256: 64-character SHA-256 hexadecimal hash computed over the source file's raw bytes.
        path: Filesystem path string indicating where the layer file was loaded from.

    Returns:
        Layer dataclass instance holding parsed layer properties.

    Example:
        >>> h = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        >>> layer = Layer("l1", "test reason", {}, h, "p.yaml")
        >>> (layer.name, layer.reason)
        ('l1', 'test reason')
    """

    name: str
    reason: str
    changes: dict[str, Any]
    sha256: str
    path: str


@dataclass(frozen=True)
class LayerChange:
    """
    WHAT: Immutable record detailing an individual parameter range transition produced by a layer.
    It captures the domain config name, setting name, range partition, prior bounds, new bounds, and author layer.
    Summary routines inspect these records to present before-and-after audit logs to operators.

    WHY: Provides fine-grained observability into each parameter alteration during layer stacking.
    Recording the prior value allows clash reporting where successive layers overwrite the same setting.
    Freezing the record prevents tampering with the audit log.

    Args:
        config_name: Canonical name of the domain configuration being altered.
        setting: Name of the parameter setting whose range was updated.
        range_name: Partition range modified ('training', 'test', or 'stress').
        before: Two-element list representing the range bounds immediately prior to this layer.
        after: Two-element list representing the new range bounds established by this layer.
        layer_name: Identifier of the layer responsible for this specific change.

    Returns:
        LayerChange dataclass instance describing the transition.

    Example:
        >>> ch = LayerChange("background_realism", "festival_peak_day_in_world", "training", [30, 50], [40, 45], "late")
        >>> (ch.setting, ch.before, ch.after)
        ('festival_peak_day_in_world', [30, 50], [40, 45])
    """

    config_name: str
    setting: str
    range_name: str
    before: list[Any]
    after: list[Any]
    layer_name: str


def _validate_layer_changes(changes: dict[str, Any], layer_name: str) -> None:
    """
    WHAT: Validates that layer changes target existing configs, settings, and valid nested range names.
    It checks that each config belongs to CONFIG_NAMES and that settings exist in baseline parameters.
    It verifies that every range name is valid and pairs of numbers satisfy low <= high ordering.

    WHY: Prevents misconfigured or misspelled layer overrides from silently corrupting simulations.
    Validating setting existence upfront ensures typos in layer YAML files fail with clear error messages.
    Enforcing numeric interval validity protects downstream pseudo-random samplers from inversion errors.

    Args:
        changes: Dictionary mapping configuration names to settings and range replacement bounds.
        layer_name: Identifier name of the layer being validated for diagnostic reporting.

    Returns:
        None upon successful structural and domain validation.

    Raises:
        ValueError: If config name, setting name, range name, or interval bounds are invalid.

    Example:
        >>> sample = {"background_realism": {"festival_peak_day_in_world": {"training": [40, 45]}}}
        >>> _validate_layer_changes(sample, "demo-layer")
    """
    # 1. Verify changes mapping is a non-empty dictionary.
    if not isinstance(changes, dict) or not changes:
        raise ValueError(f"Layer '{layer_name}': changes must be a non-empty mapping.")

    # 2. Read baseline configurations to verify existence of target settings.
    base_configs = read_world_configs()

    # 3. Iterate over each target configuration partition in the layer.
    for config_name, settings_map in changes.items():
        if config_name not in CONFIG_NAMES:
            allowed = ", ".join(CONFIG_NAMES)
            raise ValueError(f"Layer '{layer_name}': unknown config name '{config_name}'. Allowed: {allowed}")
        if not isinstance(settings_map, dict) or not settings_map:
            raise ValueError(f"Layer '{layer_name}': settings mapping for '{config_name}' must be non-empty.")
        cfg_dict = getattr(base_configs, config_name)

        # 4. Iterate over each setting name within the configuration.
        for setting_name, ranges_map in settings_map.items():
            if setting_name not in cfg_dict["parameters"]:
                raise ValueError(f"Layer '{layer_name}': unknown setting '{setting_name}' in config '{config_name}'.")
            if not isinstance(ranges_map, dict) or not ranges_map:
                raise ValueError(f"Layer '{layer_name}': ranges mapping for '{setting_name}' must be non-empty.")

            # 5. Validate range names and numeric bounds.
            for range_name, bounds in ranges_map.items():
                if range_name not in ("training", "test", "stress"):
                    raise ValueError(
                        f"Layer '{layer_name}': invalid range name '{range_name}' for setting '{setting_name}'."
                    )
                if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                    raise ValueError(
                        f"Layer '{layer_name}': range for '{setting_name}' ({range_name}) must have 2 numbers."
                    )
                low, high = bounds
                if (
                    isinstance(low, bool)
                    or isinstance(high, bool)
                    or not isinstance(low, (int, float))
                    or not isinstance(high, (int, float))
                ):
                    raise ValueError(
                        f"Layer '{layer_name}': range values for '{setting_name}' ({range_name}) must be numeric."
                    )
                if low > high:
                    raise ValueError(
                        f"Layer '{layer_name}': range [{low}, {high}] for '{setting_name}' ({range_name}) "
                        "has low > high."
                    )


def read_layer(path: Path) -> Layer:
    """
    WHAT: Reads and validates a YAML layer file from disk and calculates its SHA-256 digest.
    It verifies top-level keys ('name', 'reason', 'changes') and ensures name and reason are non-empty.
    It delegates domain validation of parameter changes to _validate_layer_changes before returning a Layer.

    WHY: Guarantees every layer read into memory is structurally sound, traceable, and cryptographically hashed.
    Hashing the file content upfront enables exact reproducibility tracking in simulation run manifests.
    Failing early on malformed YAML prevents inconsistent state from propagating into simulation runs.

    Args:
        path: Filesystem path to the YAML layer file to be loaded and validated.

    Returns:
        Layer dataclass instance holding name, reason, changes dictionary, SHA-256 digest, and path.

    Raises:
        ValueError: If file structure, top-level keys, or layer changes violate validation rules.

    Example:
        >>> import tempfile
        >>> from pathlib import Path
        >>> from world.layers import read_layer
        >>> with tempfile.NamedTemporaryFile("w", delete=False, suffix=".yaml") as f:
        ...     _ = f.write("name: demo\\nreason: testing\\nchanges:\\n")
        ...     _ = f.write("  background_realism:\\n    festival_peak_day_in_world:\\n      training: [40, 45]\\n")
        ...     demo_path = Path(f.name)
        >>> layer = read_layer(demo_path)
        >>> (layer.name, layer.reason, len(layer.sha256))
        ('demo', 'testing', 64)
    """
    # 1. Read raw file bytes and compute cryptographic SHA-256 hash digest.
    raw_bytes = path.read_bytes()
    sha256 = hashlib.sha256(raw_bytes).hexdigest()

    # 2. Parse YAML content and ensure result is a dictionary.
    data = yaml.safe_load(raw_bytes.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Layer '{path}': content must be a YAML mapping.")

    # 3. Enforce that top-level keys are exactly name, reason, and changes.
    actual_keys = set(data.keys())
    if actual_keys != {"name", "reason", "changes"}:
        raise ValueError(f"Layer '{path}': top-level keys must be exactly name, reason, changes.")

    # 4. Enforce that name and reason are non-empty string records.
    name, reason, changes = data["name"], data["reason"], data["changes"]
    if not isinstance(name, str) or not name.strip() or not isinstance(reason, str) or not reason.strip():
        raise ValueError(f"Layer '{path}': name and reason must be non-empty strings.")

    # 5. Validate the structured changes mapping against baseline domain configurations.
    _validate_layer_changes(changes, name)

    # 6. Assemble and return validated Layer record.
    return Layer(name=name, reason=reason, changes=changes, sha256=sha256, path=str(path))


def apply_layers(
    configs: WorldConfigs, layers: list[Layer]
) -> tuple[WorldConfigs, list[LayerChange]]:
    """
    WHAT: Deep-copies baseline configs, sequentially applies overlay layers, and verifies nested bounds.
    It records a LayerChange for every range modified, preserving before-and-after values across overrides.
    After all layers are stacked, it runs check_nested_ranges to enforce training <= test <= stress invariants.

    WHY: Implements deterministic layer composition where subsequent layers win on clashes.
    Tracking intermediate changes enables comprehensive effective-world reporting for transparency.
    Deep-copying guarantees that caller-supplied baseline configs remain completely unmutated.

    Args:
        configs: WorldConfigs container holding baseline domain configuration dictionaries.
        layers: Ordered list of Layer instances to apply sequentially on top of the configs.

    Returns:
        Tuple of (modified WorldConfigs container, list of chronological LayerChange records).

    Raises:
        ValueError: If any modified setting violates the training inside test inside stress nesting rule.

    Example:
        >>> from world.world_configs import read_world_configs
        >>> from world.layers import Layer, apply_layers
        >>> base = read_world_configs()
        >>> chgs = {"background_realism": {"festival_peak_day_in_world": {"training": [40, 45]}}}
        >>> l = Layer("l1", "test", chgs, "abc", "p")
        >>> new_cfgs, changes = apply_layers(base, [l])
        >>> (len(changes), changes[0].before, changes[0].after)
        (1, [30, 50], [40, 45])
    """
    # 1. Deep-copy baseline configurations to guarantee input immutability.
    applied_configs = WorldConfigs(
        world_ranges=copy.deepcopy(configs.world_ranges),
        background_realism=copy.deepcopy(configs.background_realism),
        failure_realism=copy.deepcopy(configs.failure_realism),
        device_realism=copy.deepcopy(configs.device_realism),
    )
    changes: list[LayerChange] = []
    last_layer_for_setting: dict[tuple[str, str], str] = {}

    # 2. Sequentially apply layer overrides in list order.
    for layer in layers:
        for cfg_name, settings_map in layer.changes.items():
            cfg_dict = getattr(applied_configs, cfg_name)
            for setting_name, ranges_map in settings_map.items():
                last_layer_for_setting[(cfg_name, setting_name)] = layer.name
                param_block = cfg_dict["parameters"][setting_name]
                for r_name, r_val in ranges_map.items():
                    before = list(param_block[r_name])
                    after = list(r_val)
                    param_block[r_name] = after
                    changes.append(LayerChange(cfg_name, setting_name, r_name, before, after, layer.name))

    # 3. Validate nesting constraints on every modified parameter setting.
    for (cfg_name, setting_name), layer_name in last_layer_for_setting.items():
        param_block = getattr(applied_configs, cfg_name)["parameters"][setting_name]
        errors = check_nested_ranges(param_block)
        if errors:
            err_text = "; ".join(errors)
            raise ValueError(
                f"Layer '{layer_name}' broke nesting in {cfg_name}.{setting_name}: {err_text}"
            )

    return applied_configs, changes


def describe_effective_world(
    layers: list[Layer],
    changes: list[LayerChange],
    configs: WorldConfigs,
) -> str:
    """
    WHAT: Generates a human-readable textual summary of applied layers, overrides, and untouched settings.
    It lists layer names in application order, line-by-line transitions for every changed range,
    and the computed count of untouched settings compared to total simulation parameters.

    WHY: Provides clear auditability for simulation operators before running full lifecycles.
    Displaying both steps when multiple layers modify the same range makes merge resolution explicit.
    Dynamic computation guarantees the summary always reflects the true count of parameters.

    Args:
        layers: List of Layer instances that were processed for the simulation.
        changes: List of LayerChange records detailing all range transitions performed.
        configs: WorldConfigs container used to compute the total count of domain parameters.

    Returns:
        Formatted multi-line string detailing the effective configuration state.

    Example:
        >>> from world.world_configs import read_world_configs
        >>> from world.layers import describe_effective_world
        >>> cfgs = read_world_configs()
        >>> text = describe_effective_world([], [], cfgs)
        >>> text.splitlines()[0]
        'Layers applied (in order): none'
        >>> text.splitlines()[-1]
        'Everything else: unchanged (112 of 112 settings)'
    """
    # 1. Format the layer name execution sequence header.
    names = ", ".join(layer.name for layer in layers) if layers else "none"
    lines = [f"Layers applied (in order): {names}"]

    # 2. Append each individual range change record.
    for ch in changes:
        lines.append(f"  {ch.setting} ({ch.range_name}): {ch.before} -> {ch.after}   from {ch.layer_name}")

    # 3. Compute total parameter entries across all four configuration domains.
    total = sum(len(getattr(configs, name)["parameters"]) for name in CONFIG_NAMES)
    distinct_settings = len({(ch.config_name, ch.setting) for ch in changes})
    unchanged = total - distinct_settings

    # 4. Append unchanged parameter summary line and join.
    lines.append(f"Everything else: unchanged ({unchanged} of {total} settings)")
    return "\n".join(lines)
