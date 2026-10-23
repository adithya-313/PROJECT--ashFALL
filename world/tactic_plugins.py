"""
WHAT: Local tactic plugins. A plugin is a Python file under patterns/plugins/ that defines one module-level PLUGIN
(a TacticPlugin) describing a new timing-only tactic block, plus a plan function. This module lints the file before
importing it, validates its description like a built-in block, keeps a process-local registry of loaded plugins, and
runs a plugin step.

WHY: A plugin is code, so it is checked before it runs: only the gang's own random stream may be used (determinism),
the hidden truth and other world modules may not be imported (truth isolation, rule 15), and there is no file access.
A plugin receives only PluginContext (params, clock minute, strength, the gang's rng) and returns only
PluginStepResult (minutes advanced, money in paise). It cannot create payments, roles or device events, so it cannot
retire or rotate devices, which structurally keeps the 12-8 dividing line training_devices_retired_on_schedule true.
The planner draws the step's risk from the plugin's own risk_per_use range. Lint is the stated 12f scope; it is not a
sandbox, and a plugin must still be read by a person before it is trusted.

ISSUE IDs: 12-10, 12-8.
TERM: plugin: a local Python file that adds one tactic block to the gang language.
TERM: lint: reading the file's syntax tree for forbidden imports and calls without running it.
TERM: registry: the process-local table of loaded plugins; it is never saved and never shared between processes.
"""

import ast
import hashlib
import importlib.util
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core.config_validator import load_yaml
from world.run_identity import REPO_ROOT

PLUGIN_CONFIG_PATH: Path = REPO_ROOT / "config" / "tactic_plugins.yaml"
TACTIC_BLOCKS_PATH: Path = REPO_ROOT / "config" / "tactic_blocks.yaml"
DEFAULT_PLUGIN_ROOT: Path = REPO_ROOT / "patterns" / "plugins"


@dataclass(frozen=True)
class PluginContext:
    """
    WHAT: Everything a plugin may see when it plans a step: its parameters, the gang's clock minute, the gang's
    strength and the gang's own random generator.

    WHY: Handing over only these four values keeps the world, ledgers, accounts and truth out of reach.

    Args:
        params: The step's parameters, as written in the gang config.
        clock_minute: The gang's clock when the step starts.
        strength: The gang's strength level, 1 or 2.
        rng: The gang's planning stream; the plugin must draw from nothing else.
    """

    params: dict[str, Any]
    clock_minute: int
    strength: int
    rng: np.random.Generator


@dataclass(frozen=True)
class PluginStepResult:
    """
    WHAT: What a plugin step returns: how many minutes it advances the gang's clock and how much money it costs.

    WHY: These are the only two effects a plugin has, so it cannot create payments, roles or device events.

    Args:
        minutes_advanced: Whole minutes added to the gang's clock; at least 0.
        money_paise: Whole paise the step costs; at least 0.
    """

    minutes_advanced: int
    money_paise: int


@dataclass(frozen=True)
class TacticPlugin:
    """
    WHAT: A plugin's description of itself, written like a block of config/tactic_blocks.yaml, plus its plan function.

    WHY: Describing the block the same way as a built-in lets the gang validator and planner treat it alike.

    Args:
        name: The block name used in gang steps; it must not equal a built-in block name.
        meaning: One plain sentence on what the block does.
        allowed_strengths: Strength levels that may use it, a subset of [1, 2].
        parameters: Name to {"range": [low, high]} or {"type": "boolean"}.
        risk_per_use: [low, high] range the planner draws the step's risk from, inside [0, 1].
        label: Must be "assumption".
        reason: Why these numbers; must not be empty.
        plan: Function from PluginContext to PluginStepResult.
    """

    name: str
    meaning: str
    allowed_strengths: tuple[int, ...]
    parameters: dict[str, dict[str, Any]]
    risk_per_use: tuple[float, float]
    label: str
    reason: str
    plan: Callable[[PluginContext], PluginStepResult]


@dataclass(frozen=True)
class LoadedPlugin:
    """
    WHAT: A plugin that passed lint and validation, with the SHA-256 of its file bytes and its path.

    WHY: The hash goes into the gang's plain form, so the run id changes when the plugin code changes.

    Args:
        plugin: The TacticPlugin.
        sha256: Hex SHA-256 of the plugin file's bytes.
        path: The resolved file path.
    """

    plugin: TacticPlugin
    sha256: str
    path: Path


_REGISTRY: dict[str, LoadedPlugin] = {}


def load_plugin_config(path: Path | None = None) -> dict[str, Any]:
    """
    WHAT: Reads config/tactic_plugins.yaml and returns its tactic_plugins block.

    WHY: The forbidden lists are policy, so they live in YAML and not in the linter code.

    Args:
        path: Optional YAML path.

    Returns:
        The tactic_plugins dictionary.
    """
    data = load_yaml(PLUGIN_CONFIG_PATH if path is None else path)
    block: dict[str, Any] = data["tactic_plugins"]
    return block


def builtin_block_names() -> set[str]:
    """Returns the block names of config/tactic_blocks.yaml, read directly so tactic_config need not be imported."""
    blocks = load_yaml(TACTIC_BLOCKS_PATH)["tactic_blocks"]["blocks"]
    return set(blocks)


def _dotted(node: ast.AST, aliases: dict[str, str]) -> str | None:
    """Turns a Name or Attribute chain into its real dotted name, replacing an import alias by what it stands for."""
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(aliases.get(current.id, current.id))
    parts.reverse()
    return ".".join(parts)


def _collect_imports(tree: ast.AST, filename: str, cfg: dict[str, Any], problems: list[str]) -> dict[str, str]:
    """Checks every import against the lists and returns a map from the local name to the real dotted name."""
    aliases: dict[str, str] = {}
    forbidden = set(cfg["forbidden_imports"]["modules"])
    allowed_world = set(cfg["allowed_world_imports"]["modules"])
    allowed_numpy_random = set(cfg["numpy_random_allowed"]["names"])
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                full = alias.name
                aliases[alias.asname or alias.name.split(".")[0]] = full if alias.asname else full.split(".")[0]
                _check_module(full, node.lineno, filename, forbidden, allowed_world, problems)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            _check_module(module, node.lineno, filename, forbidden, allowed_world, problems)
            for alias in node.names:
                full = f"{module}.{alias.name}"
                aliases[alias.asname or alias.name] = full
                if module == "world":
                    _check_module(full, node.lineno, filename, forbidden, allowed_world, problems)
                if module.startswith("numpy.random") and full not in allowed_numpy_random:
                    problems.append(f"{filename}:{node.lineno}: imports {full}, a second random source")
    return aliases


def _check_module(
    module: str, line: int, filename: str, forbidden: set[str], allowed_world: set[str], problems: list[str]
) -> None:
    """Records a problem for a forbidden module (random sources, truth) or a world module that is not allowed."""
    top = module.split(".")[0]
    if top in forbidden:
        problems.append(f"{filename}:{line}: imports '{module}', which a plugin may not use")
    if top == "world" and module != "world" and module not in allowed_world:
        problems.append(
            f"{filename}:{line}: imports '{module}'; only {sorted(allowed_world)} may be imported from world"
        )


def lint_plugin_source(source: str, filename: str) -> list[str]:
    """
    WHAT: Reads plugin source with ast, without running it, and returns every problem found: forbidden imports
    (random, secrets, evaluation, other world modules), numpy.random names other than the Generator type, os.urandom,
    the clock (time.time, time.time_ns, datetime now), and open, exec, eval and __import__. Import aliases are followed.

    WHY: Determinism needs the gang's stream to be the only random source; truth isolation (rule 15) forbids reaching
    evaluation or world modules; no I/O keeps a plugin a pure function of its context. The lists are in
    config/tactic_plugins.yaml.

    Args:
        source: The plugin file's text.
        filename: Name used in the messages.

    Returns:
        A list of problem strings; empty when the file is clean.
    """
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as error:
        return [f"{filename}:{error.lineno}: not valid Python ({error.msg})"]
    cfg = load_plugin_config()
    problems: list[str] = []
    aliases = _collect_imports(tree, filename, cfg, problems)
    forbidden_calls = set(cfg["forbidden_calls"]["names"])
    allowed_numpy_random = set(cfg["numpy_random_allowed"]["names"])
    for node in ast.walk(tree):
        line = getattr(node, "lineno", 0)
        if isinstance(node, ast.Attribute):
            name = _dotted(node, aliases)
            # Any numpy.random.<name> other than the allowed type is a second random source (determinism).
            if name is not None and name.startswith("numpy.random.") and name not in allowed_numpy_random:
                problems.append(f"{filename}:{line}: uses {name}, a second random source")
        if isinstance(node, ast.Call):
            called = _dotted(node.func, aliases)
            if called in forbidden_calls:
                problems.append(f"{filename}:{line}: calls {called}, which a plugin may not use")
    return sorted(set(problems))


def _validate_definition(plugin: Any, filename: str) -> None:
    """Validates a plugin's description like a built-in block and raises ValueError with a plain reason."""
    if not isinstance(plugin, TacticPlugin):
        raise ValueError(f"{filename}: PLUGIN must be built with TacticPlugin")
    if not plugin.name or not isinstance(plugin.name, str):
        raise ValueError(f"{filename}: plugin name must be non-empty text")
    if plugin.name in builtin_block_names():
        raise ValueError(f"{filename}: plugin name '{plugin.name}' clashes with a built-in block name")
    if not plugin.allowed_strengths or not set(plugin.allowed_strengths) <= {1, 2}:
        raise ValueError(f"{filename}: allowed_strengths must be a non-empty subset of [1, 2]")
    for parameter, spec in plugin.parameters.items():
        _validate_parameter(filename, parameter, spec)
    low, high = plugin.risk_per_use
    if not 0.0 <= low <= high <= 1.0:
        raise ValueError(f"{filename}: risk_per_use must satisfy 0 <= low <= high <= 1, got [{low}, {high}]")
    if plugin.label != "assumption":
        raise ValueError(f"{filename}: label must be 'assumption', got '{plugin.label}'")
    if not plugin.reason or not plugin.reason.strip():
        raise ValueError(f"{filename}: reason must not be empty")


def _validate_parameter(filename: str, parameter: str, spec: dict[str, Any]) -> None:
    """Validates one parameter description: an ordered numeric range, or type boolean."""
    if spec.get("type") == "boolean":
        return
    bounds = spec.get("range")
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        raise ValueError(f"{filename}: parameter '{parameter}' needs a range [low, high] or type boolean")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in bounds):
        raise ValueError(f"{filename}: parameter '{parameter}' range must hold numbers")
    if bounds[0] > bounds[1]:
        raise ValueError(f"{filename}: parameter '{parameter}' range [{bounds[0]}, {bounds[1]}] is not ordered")


def load_plugin(path: Path | str, allowed_root: Path | None = None) -> LoadedPlugin:
    """
    WHAT: Loads one plugin: confines the path to the plugin folder, lints the source (listing every problem), imports
    it only if the lint is clean, and validates its PLUGIN description.

    WHY: Plugins are local code, so a path outside the plugin folder is refused, and a file that fails lint is never
    imported, which means none of its top-level code runs.

    Args:
        path: The plugin file.
        allowed_root: Folder the file must be inside; defaults to patterns/plugins. Tests pass their own folder.

    Returns:
        The LoadedPlugin with the file's SHA-256.

    Raises:
        ValueError: If the path is outside the folder, the lint finds a problem, the file has no PLUGIN or its
            description is invalid.
    """
    root = (DEFAULT_PLUGIN_ROOT if allowed_root is None else Path(allowed_root)).resolve()
    resolved = Path(path).resolve()
    if root not in resolved.parents:
        raise ValueError(f"{resolved.name}: plugins load only from {root}; this path is outside it")
    raw = resolved.read_bytes()
    problems = lint_plugin_source(raw.decode("utf-8"), resolved.name)
    if problems:
        raise ValueError(f"plugin {resolved.name} refused: " + "; ".join(problems))
    spec = importlib.util.spec_from_file_location(f"tactic_plugin_{resolved.stem}", resolved)
    if spec is None or spec.loader is None:
        raise ValueError(f"{resolved.name}: cannot be loaded as a Python module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "PLUGIN"):
        raise ValueError(f"{resolved.name}: the file defines no PLUGIN")
    _validate_definition(module.PLUGIN, resolved.name)
    return LoadedPlugin(module.PLUGIN, hashlib.sha256(raw).hexdigest(), resolved)


def register_plugin(loaded: LoadedPlugin) -> None:
    """
    WHAT: Adds a loaded plugin to the process-local registry under its block name.

    WHY: The gang validator and planner look plugins up here. The registry lives in this process only; it is not saved
    and not shared, so use plugins_registered in tests and the showcase to keep one plugin from leaking into another.

    Args:
        loaded: A plugin from load_plugin.

    Raises:
        ValueError: If a different plugin is already registered under that name.
    """
    name = loaded.plugin.name
    if name in _REGISTRY and _REGISTRY[name].sha256 != loaded.sha256:
        raise ValueError(f"a different plugin is already registered as '{name}'")
    _REGISTRY[name] = loaded


def get_registered_plugin(name: str) -> LoadedPlugin | None:
    """Returns the registered plugin with this block name, or None when there is none."""
    return _REGISTRY.get(name)


@contextmanager
def plugins_registered(paths: list[Path | str], allowed_root: Path | None = None) -> Iterator[list[LoadedPlugin]]:
    """
    WHAT: Loads and registers the given plugin files for the duration of a with block, then always puts the registry
    back as it was on entry, even if the block raises.

    WHY: A registry that outlives its test would make a later test pass or fail because of an earlier one.

    Args:
        paths: Plugin files to load.
        allowed_root: Folder the files must be inside; defaults to patterns/plugins.

    Yields:
        The LoadedPlugin records, in order.
    """
    before = dict(_REGISTRY)
    try:
        loaded_list = [load_plugin(path, allowed_root) for path in paths]
        for loaded in loaded_list:
            register_plugin(loaded)
        yield loaded_list
    finally:
        _REGISTRY.clear()
        _REGISTRY.update(before)


def plugin_block_definition(loaded: LoadedPlugin) -> dict[str, Any]:
    """Returns the plugin's description in the shape of a config/tactic_blocks.yaml block."""
    plugin = loaded.plugin
    return {
        "label": plugin.label,
        "reason": plugin.reason,
        "meaning": plugin.meaning,
        "allowed_strengths": list(plugin.allowed_strengths),
        "parameters": plugin.parameters,
        "risk_per_use": list(plugin.risk_per_use),
    }


def _check_param(name: str, value: Any, spec: dict[str, Any]) -> None:
    """Refuses a parameter value that is not a boolean when type boolean, or is outside its range."""
    if spec.get("type") == "boolean":
        if not isinstance(value, bool):
            raise ValueError(f"plugin parameter '{name}' must be true or false, got {value!r}")
        return
    low, high = spec["range"]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high:
        raise ValueError(f"plugin parameter '{name}'={value!r} is outside its range [{low}, {high}]")


def run_plugin_step(
    loaded: LoadedPlugin, params: dict[str, Any], clock_minute: int, strength: int, rng: np.random.Generator
) -> PluginStepResult:
    """
    WHAT: Runs one plugin step: checks the strength and every parameter against the plugin's description, calls its
    plan function with a PluginContext, and checks the result is a PluginStepResult of whole, non-negative numbers.

    WHY: A plugin is trusted only as far as its description; the checks stop a bad parameter or a negative time from
    reaching the planner's clock and budget.

    Args:
        loaded: A registered plugin.
        params: The step's parameters.
        clock_minute: The gang's clock when the step starts.
        strength: The gang's strength level.
        rng: The gang's planning stream.

    Returns:
        The validated PluginStepResult.

    Raises:
        ValueError: For an unsupported strength, an unknown or out-of-range parameter, or an invalid result.
    """
    plugin = loaded.plugin
    if strength not in plugin.allowed_strengths:
        raise ValueError(f"plugin '{plugin.name}' does not allow strength {strength}")
    for name, value in params.items():
        if name not in plugin.parameters:
            raise ValueError(f"plugin '{plugin.name}' has no parameter '{name}'")
        _check_param(name, value, plugin.parameters[name])
    result = plugin.plan(PluginContext(dict(params), clock_minute, strength, rng))
    if not isinstance(result, PluginStepResult):
        raise ValueError(f"plugin '{plugin.name}' must return a PluginStepResult")
    for field_name, number in (("minutes_advanced", result.minutes_advanced), ("money_paise", result.money_paise)):
        if isinstance(number, bool) or not isinstance(number, int) or number < 0:
            raise ValueError(
                f"plugin '{plugin.name}' {field_name} must be a whole number of at least 0, got {number!r}"
            )
    return result
