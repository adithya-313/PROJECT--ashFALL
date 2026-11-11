"""
File: scripts/list_level_16_switches.py
Purpose: Generates the Markdown switch table documenting all Level 16 runtime and replay switches
         dynamically loaded from configuration files.
"""

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
REPLAY_CONFIG_PATH = REPO_ROOT / "config" / "replay.yaml"

# Metadata mapping each switch name to its live default, what OFF reproduces, and display label
SWITCH_METADATA: dict[str, dict[str, str]] = {
    "decisions_per_event_payment": {
        "default_live": "false",
        "off_reproduces": "Level 16b per-batch payment attempt scoring (decisions evaluated per attempt)",
    },
    "idle_partition_advance": {
        "default_live": "false",
        "off_reproduces": "Level 16c-2 quiet payment partition watermark stalling",
    },
    "kyc_phone_on_account_opened": {
        "default_live": "false",
        "off_reproduces": "Level 16c feeds without KYC phone payload on account_opened events",
    },
    "world_registry_reporting": {
        "default_live": "false",
        "off_reproduces": "Level 16c replays without in-process DPIP registry labels",
    },
    "replay_speed": {
        "default_live": "1.0",
        "off_reproduces": "Real-time paced replay execution (with inter-step sleep delays)",
    },
}


def load_replay_switches(config_path: Path | str | None = None) -> list[dict[str, Any]]:
    """
    WHAT: Reads config/replay.yaml and builds structured metadata for each Level 16 switch.
    WHY: Centralizes switch configuration introspection so docs stay in sync with config.
    """
    target = REPLAY_CONFIG_PATH if config_path is None else Path(config_path)
    with open(target, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    workspace_cfg = data.get("replay_workspace", {})
    switches: list[dict[str, Any]] = []

    for key, meta in SWITCH_METADATA.items():
        if key in workspace_cfg:
            item = workspace_cfg[key]
            val = item.get("value")
            val_str = f'"{val}"' if isinstance(val, str) else str(val).lower()
            switches.append(
                {
                    "name": key,
                    "file": "config/replay.yaml",
                    "default_live": meta["default_live"],
                    "replay_value": val_str,
                    "off_reproduces": meta["off_reproduces"],
                    "reason": item.get("reason", ""),
                }
            )
    return switches


def generate_switch_table_markdown(switches: list[dict[str, Any]] | None = None) -> str:
    """
    WHAT: Formats the list of switches into a GitHub Flavored Markdown table.
    WHY: Embeds directly into docs/handoff/level_16.md without manual transcription.
    """
    items = load_replay_switches() if switches is None else switches
    lines = [
        "| Switch Name | Configuration File | Default Live | Replay Value | What OFF Reproduces |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]
    for s in items:
        lines.append(
            f"| `{s['name']}` | `{s['file']}` | `{s['default_live']}` | `{s['replay_value']}` | {s['off_reproduces']} |"
        )
    return "\n".join(lines)


def main() -> None:
    table = generate_switch_table_markdown()
    print(table)


if __name__ == "__main__":
    main()
