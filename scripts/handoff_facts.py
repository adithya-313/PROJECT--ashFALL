"""
WHAT: Prints, in markdown, the facts a Level 12e handoff note needs, read from the code and the config files
themselves: public function and dataclass signatures, dataclass fields, configuration keys, random stream
children with the tie-break order, and the register status of the issues Level 12e touches.

WHY: A handoff note typed by hand drifts from the code. Pasting this script's output into the note keeps every
signature, field and key exactly as the code has it, and a reader can run the same command to check the note.

TERM: public function: a function defined in a module whose name does not start with an underscore.
TERM: dataclass signature: the call signature of a dataclass, which lists its fields in order with their types.
TERM: random stream child: a numbered, independent source of random numbers split off the world's seed.
TERM: tie-break order: which kind of item goes first when several fall on the same simulated minute.
"""

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

INTERFACE_MODULES: tuple[str, ...] = (
    "world.world_sampler",
    "world.sampled_world",
    "world.gang_mix",
    "world.pattern_files",
    "world.tactic_plugins",
    "world.world_invariants",
)
RECORDS_MODULE: str = "world.world_sampler"
RECORD_NAMES: tuple[str, ...] = ("WorldSpec",)
CONFIG_FILES: tuple[str, ...] = (
    "config/world_sampler.yaml",
    "config/gang_mix.yaml",
    "config/tactic_plugins.yaml",
    "config/pattern_files.yaml",
    "config/interactive_runner.yaml",
)
REGISTER_IDS: tuple[str, ...] = ("12-8", "12-10", "12-14", "12-18", "G4")
RUNNER_CONFIG_PATH: str = "config/interactive_runner.yaml"
REGISTER_PATH: str = "config/issues_v2.yaml"


def format_annotation(annotation: Any) -> str:
    """
    WHAT: Turns a type annotation into readable text: a plain class prints as its name, anything else as its
    string form with the "typing." prefix removed.

    WHY: Field types appear in the handoff note; "int" reads better than "<class 'int'>".

    Args:
        annotation: A type, or a typing construct such as int | None.

    Returns:
        The text form of the annotation.

    Example:
        format_annotation(int) == "int"
    """
    if isinstance(annotation, type):
        return annotation.__name__
    return str(annotation).replace("typing.", "")


def defined_in(module: ModuleType, member: Any) -> bool:
    """
    WHAT: Tells whether a function or class is defined in the given module, not merely imported into it.

    WHY: Imported names would otherwise be listed under every module that uses them.

    Args:
        module: The module being described.
        member: A function or class found in the module.

    Returns:
        True when the member's own module is the given module.

    Example:
        defined_in(importlib.import_module("os"), os.path.join) is False
    """
    return getattr(member, "__module__", None) == module.__name__


def public_functions(module: ModuleType) -> list[tuple[str, inspect.Signature]]:
    """
    WHAT: Lists the public functions defined in a module, in name order, each with its signature.

    WHY: These are the module's callable interface for later levels.

    Args:
        module: The module to inspect.

    Returns:
        (name, signature) pairs for functions whose names do not start with an underscore.

    Example:
        public_functions(importlib.import_module("world.gang_feed")) includes ("find_holding_gang", ...)
    """
    found: list[tuple[str, inspect.Signature]] = []
    for name, member in inspect.getmembers(module, inspect.isfunction):
        if name.startswith("_") or not defined_in(module, member):
            continue
        found.append((name, inspect.signature(member)))
    return found


def public_dataclasses(module: ModuleType) -> list[type]:
    """
    WHAT: Lists the dataclasses defined in a module, in name order.

    WHY: Dataclasses are the records later levels read and build.

    Args:
        module: The module to inspect.

    Returns:
        The dataclass types whose names do not start with an underscore.

    Example:
        public_dataclasses(importlib.import_module("world.gang_feed")) includes GangHolding
    """
    classes: list[type] = []
    for name, member in inspect.getmembers(module, inspect.isclass):
        if name.startswith("_") or not defined_in(module, member):
            continue
        if dataclasses.is_dataclass(member):
            classes.append(member)
    return classes


def print_public_interfaces() -> None:
    """
    WHAT: Prints the "Public Interfaces" section: for each inspected module, its public functions and its
    dataclass signatures, each followed by the file path.

    WHY: Gives the handoff note the exact signatures, copied from the code by inspect.signature.

    Args:
        None.

    Returns:
        None. Writes markdown to standard output.

    Example:
        print_public_interfaces() prints "## Public Interfaces" first.
    """
    print("## Public Interfaces")
    print()
    print("Extracted directly via `inspect.signature`:")
    for module_name in INTERFACE_MODULES:
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


def print_records() -> None:
    """
    WHAT: Prints the "Records and Fields" section: the fields of every dataclass in the inspected modules, and of
    the named Level 12e records in world/label_records.py. A named record that is missing is listed as such.

    WHY: Field lists are the contract between the runner, the truth files and the showcase.

    Args:
        None.

    Returns:
        None. Writes markdown to standard output.

    Example:
        print_records() prints "## Records and Fields" first.
    """
    print("## Records and Fields")
    print()
    print("Extracted directly via `dataclasses.fields`:")
    records: list[type] = []
    for module_name in INTERFACE_MODULES:
        records.extend(public_dataclasses(importlib.import_module(module_name)))
    records_module = importlib.import_module(RECORDS_MODULE)
    missing: list[str] = []
    for record_name in RECORD_NAMES:
        if hasattr(records_module, record_name):
            records.append(getattr(records_module, record_name))
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
        print(f"- `{record_name}`: not present in `{RECORDS_MODULE}`")


def print_config_keys() -> None:
    """
    WHAT: Prints the "Configuration Keys" section: for each listed config file, its top-level keys and, under each
    one that holds a mapping, its second-level keys.

    WHY: The note should name the keys a reader can set, as the files have them today.

    Args:
        None.

    Returns:
        None. Writes markdown to standard output.

    Example:
        print_config_keys() lists "gang_reactions" under config/gang_reactions.yaml.
    """
    print("## Configuration Keys")
    for config_file in CONFIG_FILES:
        data = load_yaml(REPO_ROOT / config_file)
        print()
        print(f"### `{config_file}`")
        for top_key, value in data.items():
            print(f"- **{top_key}**")
            if isinstance(value, dict):
                for second_key in value:
                    print(f"  - `{second_key}`")


def print_streams() -> None:
    """
    WHAT: Prints the "Random Stream Children and Tie-Break Order" section from config/interactive_runner.yaml:
    each child number with its name, in number order, and the tie-break order list.

    WHY: These two lists decide which draws and which items come first, so the note must match the file exactly.

    Args:
        None.

    Returns:
        None. Writes markdown to standard output.

    Example:
        print_streams() lists "16: account_market" among the children.
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


def print_register() -> None:
    """
    WHAT: Prints the "Register Changes" section: the id, title and status in config/issues_v2.yaml of each issue
    Level 12e touches.

    WHY: The note should state where those issues stand now, not where they stood when it was written.

    Args:
        None.

    Returns:
        None. Writes markdown to standard output.

    Raises:
        KeyError: If a listed id is not in the register.

    Example:
        print_register() prints a line for 12-8.
    """
    issues = load_yaml(REPO_ROOT / REGISTER_PATH)["issues"]
    print("## Register Changes")
    for issue_id in REGISTER_IDS:
        entry = issues[issue_id]
        print(f"- {issue_id}: {entry['title']} (status: {entry['status']})")


def main() -> int:
    """
    WHAT: Prints all five sections in the order of the handoff note, separated by horizontal rules.

    WHY: One command gives everything the generated parts of the note need.

    Args:
        None.

    Returns:
        0 on success.

    Example:
        python scripts/handoff_facts.py prints the five sections.
    """
    sections = (print_register, print_public_interfaces, print_records, print_config_keys, print_streams)
    for position, section in enumerate(sections):
        if position:
            print()
            print("---")
            print()
        section()
    return 0


if __name__ == "__main__":
    sys.exit(main())
