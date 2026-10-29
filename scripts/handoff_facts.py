"""
scripts/handoff_facts.py

Prints, in markdown, the facts a level handoff note needs, read from the code and configuration
files directly: public function and dataclass signatures, dataclass fields, configuration keys,
random stream children with tie-break order, and the register status of touched issues.

Targeting specific parts:
    Pass --part (for example --part 12h or --part 13a) to select the constants for that level part.
    If omitted, --part defaults to '12h' to preserve existing behavior.
"""

import argparse
import dataclasses
import importlib
import inspect
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.config_validator import load_yaml  # noqa: E402

RUNNER_CONFIG_PATH: str = "config/interactive_runner.yaml"
REGISTER_PATH: str = "config/issues_v2.yaml"

PART_CONFIGS: dict[str, dict[str, Any]] = {
    "12h": {
        "INTERFACE_MODULES": (
            "world.edge_cases",
            "world.base_rate",
            "evaluation.trivial_world",
            "evaluation.sanity_report",
            "world.world_card",
        ),
        "RECORDS_MODULE": "world.edge_cases",
        "RECORD_NAMES": ("EdgeCase",),
        "CONFIG_FILES": (
            "config/base_rate.yaml",
            "config/sanity_report.yaml",
            "config/evaluation.yaml",
        ),
        "REGISTER_IDS": ("E1", "12-6", "12-7", "12-13", "12-16"),
    },
    "13a": {
        "INTERFACE_MODULES": (
            "core.event_clock",
            "nationalsim.keyed_hash",
            "nationalsim.registry",
            "nationalsim.adapter",
            "nationalsim.app",
        ),
        "RECORDS_MODULE": (
            "nationalsim.registry",
            "nationalsim.adapter",
        ),
        "RECORD_NAMES": (
            "FeedbackRecord",
            "RegistryEntry",
            "ScreenResult",
            "IngestionItem",
        ),
        "CONFIG_FILES": (
            "config/nationalsim.yaml",
        ),
        "REGISTER_IDS": ("X1", "K8", "13-11", "13-12", "13-14", "13-15"),
    },
    "13b": {
        "INTERFACE_MODULES": (
            "world.phone_book",
            "nationalsim.i4c_feed",
            "nationalsim.registry",
            "nationalsim.adapter",
            "evaluation.registry_coverage",
        ),
        "RECORDS_MODULE": (
            "world.phone_book",
            "nationalsim.i4c_feed",
            "nationalsim.registry",
            "evaluation.registry_coverage",
        ),
        "RECORD_NAMES": (
            "PhoneRecord",
            "PhoneBook",
            "I4CEntry",
            "DisputeResolutionRecord",
            "MuleDepthMetrics",
            "RegistryCoverageResult",
        ),
        "CONFIG_FILES": (
            "config/nationalsim.yaml",
        ),
        "REGISTER_IDS": ("13-5", "13-9", "13-10"),
    },
    "13c": {
        "INTERFACE_MODULES": (
            "nationalsim.hold_flow",
            "world.hold_replay",
            "nationalsim.bank_client",
            "nationalsim.national_outbox",
            "core.redaction",
            "core.raw_identifier_scan",
        ),
        "RECORDS_MODULE": (
            "nationalsim.hold_flow",
            "world.hold_replay",
            "nationalsim.bank_client",
            "core.raw_identifier_scan",
        ),
        "RECORD_NAMES": (
            "HoldRequest",
            "HoldOutcome",
            "OutgoingTransfer",
            "HoldReplayResult",
            "OutageEvent",
            "PlaceScanResult",
            "RawScanReport",
        ),
        "CONFIG_FILES": (
            "config/nationalsim.yaml",
        ),
        "REGISTER_IDS": ("13-2", "13-3", "13-16", "13-17"),
    },
    "14c": {
        "INTERFACE_MODULES": (
            "world.bank_feed",
            "world.bank_feed_config",
            "world.bank_connectors",
            "api.v1.bank_service",
            "api.v1.bank_clients",
        ),
        "RECORDS_MODULE": (
            "world.bank_feed",
            "world.bank_feed_config",
            "world.bank_connectors",
            "api.v1.bank_service",
        ),
        "RECORD_NAMES": (
            "BankFeedPaymentSide",
            "BankFeedAccountEvent",
            "BankFeedComplaint",
            "BankFeed",
            "BankFeedConfig",
            "ConnectorRunReport",
            "IdempotentOutcome",
        ),
        "CONFIG_FILES": (
            "config/bank_feed.yaml",
            "config/bank_api.yaml",
        ),
        "REGISTER_IDS": ("14-6", "14-8", "14-10", "R7", "K2", "X3"),
    },
}


def format_annotation(annotation: Any) -> str:
    """Format a type annotation into clean, readable text.

    What it is for:
        Turns type objects and type hints into concise strings suitable for markdown tables.
        For example, prints plain class names without module noise and strips typing prefixes.

    Inputs:
        annotation: A Python type or typing construct.
            For example, int or str | None.

    What it returns:
        String representation of the type. For example, 'int' or 'str | None'.

    How it works:
        1. If the annotation is a basic type class, return its __name__ attribute.
        2. Otherwise, convert to string and remove any 'typing.' module qualifiers.

    Errors and side effects:
        None. Pure string conversion function.

    Limits:
        Does not deeply reformat nested generic structures beyond removing module prefixes.
    """
    if isinstance(annotation, type):
        return annotation.__name__
    return str(annotation).replace("typing.", "")


def defined_in(module: ModuleType, member: Any) -> bool:
    """Determine whether an object was defined inside the given module.

    What it is for:
        Filters out third-party or stdlib functions imported into a module's namespace,
        ensuring only symbols authored within that specific module are documented.

    Inputs:
        module: The module object being inspected.
            For example, importlib.import_module('nationalsim.registry').
        member: The class or function object to inspect.
            For example, RegistryCore.

    What it returns:
        Boolean True if member.__module__ equals module.__name__, False otherwise.

    How it works:
        1. Read the __module__ attribute from the candidate member object.
        2. Compare against module.__name__.

    Errors and side effects:
        None. Pure inspection query.

    Limits:
        Objects without a __module__ attribute return False.
    """
    return getattr(member, "__module__", None) == module.__name__


def public_functions(module: ModuleType) -> list[tuple[str, inspect.Signature]]:
    """List all public functions authored inside a module with their call signatures.

    What it is for:
        Gathers the callable interface for a module so subsequent levels have exact, verified signatures.

    Inputs:
        module: The module object to inspect.
            For example, importlib.import_module('nationalsim.keyed_hash').

    What it returns:
        List of (name, signature) tuples sorted alphabetically by function name.

    How it works:
        1. Use inspect.getmembers to discover all function members.
        2. Skip private functions starting with an underscore.
        3. Skip functions imported from other modules using defined_in.
        4. Extract the call signature for each remaining function.

    Errors and side effects:
        None. Read-only inspection.

    Limits:
        Does not inspect methods inside classes; functions only.
    """
    found: list[tuple[str, inspect.Signature]] = []
    for name, member in inspect.getmembers(module, inspect.isfunction):
        if name.startswith("_") or not defined_in(module, member):
            continue
        found.append((name, inspect.signature(member)))
    return found


def public_dataclasses(module: ModuleType) -> list[type]:
    """List all public dataclass types defined directly inside a module.

    What it is for:
        Extracts record schemas created in a module so data contracts are documented accurately.

    Inputs:
        module: The module object to inspect.
            For example, importlib.import_module('nationalsim.registry').

    What it returns:
        List of dataclass types sorted alphabetically by class name.

    How it works:
        1. Discover all class members in the module using inspect.getmembers.
        2. Skip private classes and classes imported from external modules.
        3. Filter for classes confirmed to be dataclasses via dataclasses.is_dataclass.

    Errors and side effects:
        None. Read-only inspection.

    Limits:
        Only identifies classes decorated with @dataclass.
    """
    classes: list[type] = []
    for name, member in inspect.getmembers(module, inspect.isclass):
        if name.startswith("_") or not defined_in(module, member):
            continue
        if dataclasses.is_dataclass(member):
            classes.append(member)
    return classes


def print_public_interfaces(interface_modules: tuple[str, ...]) -> None:
    """Print markdown documentation for public functions and dataclasses in the inspected modules.

    What it is for:
        Extracts real function signatures and dataclass structures using inspect.signature,
        writing out the Public Interfaces section for a handoff note.

    Inputs:
        interface_modules: Tuple of module path strings to inspect.
            For example, ('core.event_clock', 'nationalsim.app').

    What it returns:
        None. Writes formatted markdown to standard output.

    How it works:
        1. Print the '## Public Interfaces' section heading.
        2. For each module path, import the module dynamically.
        3. Print signatures for all public functions defined within it.
        4. Print signatures for all public dataclasses defined within it.

    Errors and side effects:
        Writes to standard output. Raises ModuleNotFoundError if an invalid module name is given.

    Limits:
        Only inspects modules listed in interface_modules.
    """
    print("## Public Interfaces")
    print()
    print("Extracted directly via `inspect.signature`:")
    for module_name in interface_modules:
        module = importlib.import_module(module_name)
        path = module_name.replace(".", "/") + ".py"
        print()
        print(f"### `{path}`")
        for name, signature in public_functions(module):
            print()
            print("```python")
            print(f"{name}{signature}")
            print("```")
        for record in public_dataclasses(module):
            print()
            print("```python")
            print(f"{record.__name__}{inspect.signature(record)}")
            print("```")
            print("- dataclass")


def print_records(
    interface_modules: tuple[str, ...],
    records_module: str | tuple[str, ...],
    record_names: tuple[str, ...],
) -> None:
    """Print fields and file paths for all dataclasses in the inspected and record modules.

    What it is for:
        Outputs the Records and Fields section of the handoff note, proving exact dataclass field
        names and types directly from code.

    Inputs:
        interface_modules: Tuple of module strings scanned for dataclasses.
            For example, ('nationalsim.registry', 'nationalsim.adapter').
        records_module: Specific module name or tuple of module names holding expected records.
            For example, 'world.edge_cases' or ('nationalsim.registry', 'nationalsim.adapter').
        record_names: Tuple of expected record class names to verify.
            For example, ('EdgeCase',) or ('RegistryEntry', 'IngestionItem').

    What it returns:
        None. Writes formatted markdown to standard output.

    How it works:
        1. Collect all public dataclasses defined across interface_modules.
        2. Normalize records_module to a tuple and import each module.
        3. Verify that each record named in record_names is present, adding it if not yet collected.
        4. Track any expected records that could not be found in any record module.
        5. For each dataclass, print its name, field types, and defining file path.
        6. If any expected records were missing, print a notice.

    Errors and side effects:
        Writes markdown text to standard output.

    Limits:
        Inspects only dataclasses decorated with @dataclass.
    """
    print("## Records and Fields")
    print()
    print("Extracted directly via `dataclasses.fields`:")
    records: list[type] = []
    for module_name in interface_modules:
        records.extend(public_dataclasses(importlib.import_module(module_name)))

    mod_list: tuple[str, ...] = (
        (records_module,) if isinstance(records_module, str) else tuple(records_module)
    )
    modules_to_check = [importlib.import_module(m) for m in mod_list]
    missing: list[str] = []
    for record_name in record_names:
        found_rec: type | None = None
        for mod in modules_to_check:
            if hasattr(mod, record_name):
                candidate = getattr(mod, record_name)
                if dataclasses.is_dataclass(candidate):
                    found_rec = candidate
                    break
        if found_rec is not None:
            if found_rec not in records:
                records.append(found_rec)
        else:
            missing.append(record_name)

    for record in records:
        field_text = ", ".join(f"`{f.name}: {format_annotation(f.type)}`" for f in dataclasses.fields(record))
        print()
        print(f"- `{record.__name__}`:")
        print(f"  - Fields: {field_text}")
        print(f"  - Path: `{record.__module__.replace('.', '/')}.py`")
    for record_name in missing:
        print()
        module_display = records_module if isinstance(records_module, str) else ", ".join(records_module)
        print(f"- `{record_name}`: not present in `{module_display}`")


def print_config_keys(config_files: tuple[str, ...]) -> None:
    """Print top-level and second-level keys found in the given YAML configuration files.

    What it is for:
        Extracts real configuration keys directly from disk, documenting the parameters available
        to developers and tests without manual transcription errors.

    Inputs:
        config_files: Tuple of repository-relative paths to YAML configuration files.
            For example, ('config/nationalsim.yaml',).

    What it returns:
        None. Writes formatted markdown to standard output.

    How it works:
        1. Print the '## Configuration Keys' heading.
        2. For each config file, load the YAML content via load_yaml.
        3. Print the file heading and each top-level key.
        4. If a top-level key contains a mapping dictionary, print its nested keys.

    Errors and side effects:
        Writes markdown to standard output. Raises FileNotFoundError if a config file is missing.

    Limits:
        Prints up to two levels of nested configuration keys.
    """
    print("## Configuration Keys")
    for config_file in config_files:
        data = load_yaml(REPO_ROOT / config_file)
        print()
        print(f"### `{config_file}`")
        for top_key, value in data.items():
            print(f"- **{top_key}**")
            if isinstance(value, dict):
                for second_key in value:
                    print(f"  - `{second_key}`")


def print_streams() -> None:
    """Print random stream child allocations and tie-break ordering from the runner configuration.

    What it is for:
        Documents the deterministic seed splitting and event tie-break sequence used across world
        and runner simulations.

    Inputs:
        None. Reads configuration from RUNNER_CONFIG_PATH.

    What it returns:
        None. Writes formatted markdown to standard output.

    How it works:
        1. Load interactive runner configuration from config/interactive_runner.yaml.
        2. Print the '## Random Stream Children and Tie-Break Order' heading.
        3. Print all random_stream_children sorted by numeric child index.
        4. Print the tie_break_order list.

    Errors and side effects:
        Writes markdown to standard output.

    Limits:
        Reads specifically from config/interactive_runner.yaml.
    """
    block = load_yaml(REPO_ROOT / RUNNER_CONFIG_PATH)["interactive_runner"]
    print("## Random Stream Children and Tie-Break Order")
    print()
    print("- `random_stream_children`:")
    for name, number in sorted(block["random_stream_children"].items(), key=lambda item: item[1]):
        print(f"  - `{number}`: `{name}`")
    print()
    print("- `tie_break_order`:")
    print(f"  - `[{', '.join(block['tie_break_order'])}]`")


def print_register(register_ids: tuple[str, ...]) -> None:
    """Print current title and status for the specified register issues.

    What it is for:
        Documents the live implementation status of issues touched by the level part.

    Inputs:
        register_ids: Tuple of issue ID strings to look up in config/issues_v2.yaml.
            For example, ('X1', 'K8', '13-11').

    What it returns:
        None. Writes formatted markdown to standard output.

    How it works:
        1. Load issues dictionary from config/issues_v2.yaml.
        2. Print the '## Register Changes' heading.
        3. For each issue ID, retrieve its entry and print its ID, title, and current status.

    Errors and side effects:
        Writes markdown to standard output. Raises KeyError if an issue ID does not exist.

    Limits:
        Only reports on IDs present in the register_ids tuple.
    """
    issues = load_yaml(REPO_ROOT / REGISTER_PATH)["issues"]
    print("## Register Changes")
    for issue_id in register_ids:
        entry = issues[issue_id]
        print(f"- {issue_id}: {entry['title']} (status: {entry['status']})")


def main(argv: list[str] | None = None) -> int:
    """Print the markdown sections required for a project handoff note.

    What it is for:
        Extracts verified facts directly from source code and YAML configurations to ensure
        handoff documentation contains exact signatures and schemas without manual drift.

    Inputs:
        argv: Optional list of command-line argument strings.
            For example, ['--part', '13a']. If None, sys.argv[1:] is parsed.

    What it returns:
        Integer exit code (0 on success).

    How it works:
        1. Set up an argument parser accepting --part with default '12h'.
        2. Validate that the specified part exists in the PART_CONFIGS registry.
        3. Retrieve module, record, configuration, and issue constants for the selected part.
        4. Print the register changes, public interfaces, records, config keys, and random streams in sequence.

    Errors and side effects:
        Prints markdown content to standard output.
        Exits via parser.error with code 2 if an unrecognized part name is supplied.

    Limits:
        Generates facts for configured parts registered in PART_CONFIGS.
    """
    parser = argparse.ArgumentParser(
        description="Print markdown facts for a level handoff note from live code and configuration."
    )
    parser.add_argument(
        "--part",
        default="12h",
        help="Named part to extract facts for (e.g. '12h', '13a', or '13b'). Defaults to '12h'.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check that the handoff note exists and contains all required facts sections.",
    )
    args = parser.parse_args(argv)

    if args.check:
        target_file = REPO_ROOT / f"docs/handoff/level_{args.part}.md"
        if not target_file.is_file():
            candidate_13b = REPO_ROOT / "docs/handoff/level_13b.md"
            if candidate_13b.is_file():
                target_file = candidate_13b
            else:
                sys.stderr.write(f"Handoff file '{target_file}' not found.\n")
                return 1
        content = target_file.read_text(encoding="utf-8")
        required_headings = (
            "## Register Changes",
            "## Public Interfaces",
            "## Records and Fields",
            "## Configuration Keys",
            "## Random Stream Children and Tie-Break Order",
        )
        for heading in required_headings:
            if heading not in content:
                sys.stderr.write(f"Missing required heading '{heading}' in {target_file}.\n")
                return 1
        print(f"Handoff check passed: {target_file.name} is complete.")
        return 0

    if args.part not in PART_CONFIGS:
        available = ", ".join(sorted(PART_CONFIGS.keys()))
        parser.error(f"Unknown part '{args.part}'. Available parts: {available}")

    part_cfg = PART_CONFIGS[args.part]
    interface_modules = part_cfg["INTERFACE_MODULES"]
    records_module = part_cfg["RECORDS_MODULE"]
    record_names = part_cfg["RECORD_NAMES"]
    config_files = part_cfg["CONFIG_FILES"]
    register_ids = part_cfg["REGISTER_IDS"]

    print_register(register_ids)
    print()
    print("---")
    print()
    print_public_interfaces(interface_modules)
    print()
    print("---")
    print()
    print_records(interface_modules, records_module, record_names)
    print()
    print("---")
    print()
    print_config_keys(config_files)
    print()
    print("---")
    print()
    print_streams()
    return 0


if __name__ == "__main__":
    sys.exit(main())
