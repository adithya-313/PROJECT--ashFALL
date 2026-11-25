# File: scripts/mule_model_quality_gate.py
# Purpose: Level 18 task 10 - the CI "Mule model quality gate". Reads the committed threshold config and the
#          committed model cards, prints one PASS/FAIL line per threshold and exits 1 on any failure.
#
# What it checks:
#   1. config/mule_model_quality_gate.yaml is readable and carries exactly threshold_count thresholds.
#   2. Every threshold names a committed card, a dot path into that card's data, one rule and one value, and a
#      non-empty derived_from.
#   3. Every card value satisfies its rule. A missing card, path or value, an unknown card, a duplicate id, a
#      wrong type or an unknown rule is a failure, never a skip.
#
# The gate has no flag, environment variable or switch that skips a threshold or overrides a value. Model v1's
# scripts/ci_quality_gate.py is a separate gate and is not touched.
#
# Rules: docs/research/amendments.md, A-064 ruling 7.
#
# Usage:
#   python scripts/mule_model_quality_gate.py

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from core.config_validator import load_yaml
from world.run_identity import REPO_ROOT

CONFIG_PATH = "config/mule_model_quality_gate.yaml"
CARD_PATHS: dict[str, str] = {
    "mule_account_xgb": "reports/level_18/model_cards/mule_account_xgb.json",
    "mule_payment_xgb": "reports/level_18/model_cards/mule_payment_xgb.json",
}
RULES: tuple[str, ...] = ("at_least", "at_most", "equals")


def _fmt(value: Any) -> str:
    """Render one value for the gate line: floats to six significant digits, everything else as text."""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _load_card(root: Path, model: str) -> dict[str, Any]:
    """Return one committed card's data block, refusing an unknown model, a missing file or a malformed card."""
    rel_path = CARD_PATHS.get(model)
    if rel_path is None:
        raise ValueError(f"unknown card '{model}'")
    path = root / rel_path
    if not path.is_file():
        raise ValueError(f"card is missing: {rel_path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"card is not valid JSON: {rel_path}: {error}") from error
    if not isinstance(document, dict):
        raise ValueError(f"card must be a JSON object: {rel_path}")
    data = document.get("data")
    if not isinstance(data, dict):
        raise ValueError(f"card has no data object: {rel_path}")
    return data


def _parse_threshold(entry: Any, index: int) -> tuple[str, str, str, str, Any]:
    """Return one threshold's id, card, path, rule and value, refusing an incomplete entry."""
    where = f"thresholds[{index}]"
    if not isinstance(entry, dict):
        raise ValueError(f"expected an object at {where}")
    for key in ("id", "card", "path", "derived_from"):
        value = entry.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{where} has no usable '{key}'")
    present = [rule for rule in RULES if rule in entry]
    if len(present) != 1:
        raise ValueError(f"threshold '{entry['id']}' must carry exactly one of {', '.join(RULES)}")
    return entry["id"], entry["card"], entry["path"], present[0], entry[present[0]]


def _resolve(data: dict[str, Any], path: str, threshold_id: str) -> Any:
    """Follow one dot path inside a card's data block, refusing a missing step."""
    block: Any = data
    for key in path.split("."):
        if not isinstance(block, dict) or key not in block:
            raise ValueError(f"threshold '{threshold_id}': the card has no value at '{path}'")
        block = block[key]
    return block


def _compare(actual: Any, rule: str, expected: Any, threshold_id: str) -> bool:
    """Apply one rule, refusing a value that cannot be compared under it."""
    if rule == "equals":
        if isinstance(actual, bool) != isinstance(expected, bool):
            raise ValueError(
                f"threshold '{threshold_id}': the card value {actual!r} does not match the type of {expected!r}"
            )
        return actual == expected
    for value, label in ((actual, "card value"), (expected, "threshold")):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"threshold '{threshold_id}': {label} {value!r} is not a number")
    if rule == "at_least":
        return actual >= expected
    return actual <= expected


def run(root: Path = REPO_ROOT) -> int:
    """Check every threshold and return 0 only when all of them pass."""
    try:
        config = load_yaml(root / CONFIG_PATH)
    except (OSError, ValueError) as error:
        print(f"FAIL config: cannot read {CONFIG_PATH}: {error}")
        return 1
    if not isinstance(config, dict):
        print(f"FAIL config: {CONFIG_PATH} must hold an object")
        return 1
    entries = config.get("thresholds")
    threshold_count = config.get("threshold_count")
    if not isinstance(entries, list) or not entries:
        print(f"FAIL config: {CONFIG_PATH} has no thresholds list")
        return 1
    if isinstance(threshold_count, bool) or not isinstance(threshold_count, int):
        print(f"FAIL config: {CONFIG_PATH} has no integer threshold_count")
        return 1
    if threshold_count != len(entries):
        print(
            f"FAIL config: {CONFIG_PATH} carries {len(entries)} thresholds but declares {threshold_count}; "
            "a removed threshold is a failure, not a skip"
        )
        return 1
    try:
        parsed = [_parse_threshold(entry, index) for index, entry in enumerate(entries)]
    except ValueError as error:
        print(f"FAIL config: {error}")
        return 1
    ids = [item[0] for item in parsed]
    duplicates = sorted({threshold_id for threshold_id in ids if ids.count(threshold_id) > 1})
    if duplicates:
        print(f"FAIL config: duplicate threshold ids {', '.join(duplicates)}")
        return 1
    cards: dict[str, dict[str, Any]] = {}
    for model in sorted({item[1] for item in parsed}):
        try:
            cards[model] = _load_card(root, model)
        except ValueError as error:
            print(f"FAIL card {model}: {error}")
            return 1
    failures = 0
    for threshold_id, model, path, rule, expected in parsed:
        try:
            actual = _resolve(cards[model], path, threshold_id)
            passed = _compare(actual, rule, expected, threshold_id)
        except ValueError as error:
            print(f"FAIL {threshold_id}: {error}")
            failures += 1
            continue
        print(f"{'PASS' if passed else 'FAIL'} {threshold_id} {_fmt(actual)} {rule.replace('_', ' ')} {_fmt(expected)}")
        if not passed:
            failures += 1
    print(f"{len(parsed) - failures} passed, {failures} failed")
    return 0 if failures == 0 else 1


def main() -> int:
    """Run the gate and return its exit code."""
    return run()


if __name__ == "__main__":
    sys.exit(main())
