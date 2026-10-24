"""
File: core/config_validator.py
Purpose: A shared checker for the Level 10 research config files (world ranges, legal actions, evaluation rules).
         Every decision in those files is a "labelled block": a small dictionary with a "label" and a "reason".
         This module finds all such blocks in a YAML file and checks that each one follows the rules.

Why it exists: A number in a config file must say where it came from. A sourced number needs a link, a date and a
short quote. A guess must be registered as an assumption. This checker makes it impossible to forget that.

Use from the command line:
    python -m core.config_validator config/world_ranges.yaml config/legal_actions.yaml
"""

import copy
import hashlib
import os
import sys
from typing import Any

import yaml

ALLOWED_LABELS = ["sourced", "reported", "assumption", "policy"]

# A quote from a source must stay short (copyright and readability).
MAX_QUOTE_WORDS = 25

# Content-addressed cache mapping SHA-256 digests of raw file bytes to parsed YAML objects.
# Capped at 256 entries to prevent unbounded memory usage over long processes.
# DECISION: A clear is simple and correct, and a few hundred distinct config contents is far more than one process uses.
_YAML_CACHE_CAPACITY: int = 256
_PARSED_BY_CONTENT: dict[str, Any] = {}


def load_yaml(path: str | os.PathLike[str]) -> Any:
    """
    WHAT: Reads a YAML file from disk, parses its contents into Python data structures, and returns a deep copy.

    WHY: Repeatedly parsing YAML files across high-volume transaction evaluation is CPU-intensive. Caching parsed
    structures keyed by the SHA-256 digest of raw file bytes (rather than path or modification time) ensures automatic,
    exact cache invalidation across swapped paths or temporary test configs. Returning a deep copy guarantees every
    caller receives an isolated, private mutable structure that cannot leak state mutations to other callers.
    The cache is capped at 256 entries and cleared when full.

    Args:
        path: Path string or path-like object pointing to the YAML file on disk.

    Returns:
        A private deep copy of the parsed YAML structure (dictionary or list).

    Raises:
        FileNotFoundError: If the file does not exist at the given path.
        yaml.YAMLError: If the file content contains invalid YAML syntax.

    Example:
        # data = load_yaml("config/world_ranges.yaml")
    """
    with open(path, "rb") as raw_file:
        raw_bytes = raw_file.read()

    digest = hashlib.sha256(raw_bytes).hexdigest()
    if digest not in _PARSED_BY_CONTENT:
        if len(_PARSED_BY_CONTENT) >= _YAML_CACHE_CAPACITY:
            # DECISION: Clearing when full is simple and correct; a few hundred config variants exceeds normal usage.
            _PARSED_BY_CONTENT.clear()
        parsed = yaml.safe_load(raw_bytes.decode("utf-8"))
        _PARSED_BY_CONTENT[digest] = parsed

    return copy.deepcopy(_PARSED_BY_CONTENT[digest])


def clear_yaml_cache() -> None:
    """
    WHAT: Clears all cached parsed YAML structures from the in-memory content cache.

    WHY: Helper routine designed strictly for test harnesses to reset cache state between test isolation runs.
    """
    _PARSED_BY_CONTENT.clear()


def find_labelled_blocks(data, path=""):
    """Walk through nested dictionaries and lists. Return a list of (path, block) for every dict that has a "label"."""
    found = []

    if isinstance(data, dict):
        if "label" in data:
            found.append((path or "(top)", data))
        for key, value in data.items():
            child_path = f"{path}.{key}" if path else str(key)
            found.extend(find_labelled_blocks(value, child_path))
    elif isinstance(data, list):
        for index, item in enumerate(data):
            found.extend(find_labelled_blocks(item, f"{path}[{index}]"))

    return found


def is_filled(value):
    """True when a value is present and, for text, not just spaces."""
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() != ""
    return True


def is_number(value):
    """True for ints and floats. A bool is not counted as a number."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def check_source(block, need_quote):
    """Check the "source" part of a block. Return a list of error messages."""
    source = block.get("source")
    label = block.get("label")

    if not isinstance(source, dict):
        return [f"{label} block has no source.url"]

    errors = []

    url = source.get("url")
    if not is_filled(url) or not str(url).startswith("http"):
        errors.append(f"{label} block has no source.url that starts with http")

    if not is_filled(source.get("date")):
        errors.append(f"{label} block has no source.date")

    if need_quote:
        quote = source.get("quote")
        if not is_filled(quote):
            errors.append(f"{label} block has no source.quote")
        elif len(str(quote).split()) >= MAX_QUOTE_WORDS:
            errors.append(f"source.quote has {len(str(quote).split())} words; it must be under {MAX_QUOTE_WORDS}")

    return errors


def check_range(range_value, name):
    """A range must be a list of two numbers [min, max] with min < max. Return a list of error messages."""
    if not isinstance(range_value, list) or len(range_value) != 2:
        return [f"{name} must be a list of two numbers [min, max]"]
    if not is_number(range_value[0]) or not is_number(range_value[1]):
        return [f"{name} must hold numbers only"]
    if range_value[0] >= range_value[1]:
        return [f"{name} has min {range_value[0]} that is not below max {range_value[1]}"]
    return []


def contains(outer, inner):
    """True when the outer range fully covers the inner range."""
    return outer[0] <= inner[0] and outer[1] >= inner[1]


def check_nested_ranges(block):
    """Training, test and stress ranges must be valid and nested: training inside test inside stress."""
    errors = []
    for name in ["training", "test", "stress"]:
        errors.extend(check_range(block[name], name))
    if errors:
        # The nesting check only makes sense when all three ranges are well formed.
        return errors

    if not contains(block["test"], block["training"]):
        errors.append("training range is not inside the test range")
    if not contains(block["stress"], block["test"]):
        errors.append("test range is not inside the stress range")
    return errors


def check_block(path, block):
    """Check one labelled block. Return a list of error strings (empty list = fine)."""
    errors = []

    label = block.get("label")
    if label not in ALLOWED_LABELS:
        errors.append(f"label {label!r} is not allowed")

    if not is_filled(block.get("reason")):
        errors.append("reason is missing or empty")

    if label == "sourced":
        errors.extend(check_source(block, need_quote=True))
    elif label == "reported":
        errors.extend(check_source(block, need_quote=False))

    # An empty value is only allowed when the block says plainly that nothing was found.
    if "value" in block and block["value"] is None and block.get("not_found") is not True:
        errors.append("value is empty but not_found is not true")

    if "training" in block and "test" in block and "stress" in block:
        errors.extend(check_nested_ranges(block))

    return [f"{path}: {message}" for message in errors]


def validate_file(path):
    """Check a whole file. Return a list of error strings (empty list = valid)."""
    data = load_yaml(path)
    blocks = find_labelled_blocks(data)

    if not blocks:
        return ["the file has no labelled blocks"]

    errors = []
    for block_path, block in blocks:
        errors.extend(check_block(block_path, block))
    return errors


def count_labels(path):
    """Count the labelled blocks in a file by label. Used later for the status table."""
    counts = {label: 0 for label in ALLOWED_LABELS}
    for _, block in find_labelled_blocks(load_yaml(path)):
        label = block.get("label")
        if label in counts:
            counts[label] += 1
    return counts


def main():
    """Command line entry point. Print every error and exit with 1 if there was any."""
    file_paths = sys.argv[1:]
    if not file_paths:
        print("Usage: python -m core.config_validator <file> [<file> ...]")
        return 2

    error_count = 0
    for file_path in file_paths:
        for error in validate_file(file_path):
            print(f"{file_path}: {error}")
            error_count += 1

    return 1 if error_count else 0


if __name__ == "__main__":
    sys.exit(main())
