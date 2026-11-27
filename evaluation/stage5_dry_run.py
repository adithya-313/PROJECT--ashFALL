"""
WHAT: Rehearses stage 5 of the full run on training and validation data only and writes dry_run.json.

WHY: Stage 5 is where full-scale memory runs out, and the full run only reaches it hours in. This
runs the same code up to the point where test data would be read, so the memory and timing can be
proven first. It is given no test or stress folder and run_pilot refuses any, so it cannot touch the
final exam.

Run by hand:
    py -3.12 -m evaluation.stage5_dry_run --training <dir> --validation <dir>
        --truth-root <dir> --world-cards-root <dir> --mule-model-config <file.yaml> --out <dir>

The model config is made active exactly as run_full does for stage 5: set_mule_model_config_path
before run_pilot, restored afterwards.
"""

import argparse
from pathlib import Path
from typing import Sequence

from decisions.mule_account_model import active_mule_model_config_path, set_mule_model_config_path
from evaluation.run_level_18_pilot import run_pilot

__all__ = ["main"]


def main(argv: Sequence[str] | None = None) -> int:
    """
    WHAT: Parses the command line and runs the pilot up to its test stop point.

    Returns:
        The pilot's exit code (0 once the stop point is reached).
    """
    parser = argparse.ArgumentParser(description="Rehearse stage 5 without loading test or stress data")
    parser.add_argument(
        "--training", type=Path, action="append", required=True, help="training dataset folder; repeatable"
    )
    parser.add_argument(
        "--validation", type=Path, action="append", required=True, help="validation dataset folder; repeatable"
    )
    parser.add_argument(
        "--truth-root", type=Path, required=True, help="root folder holding one truth folder per run_id"
    )
    parser.add_argument(
        "--world-cards-root", type=Path, required=True, help="root folder holding one world card per run_id"
    )
    parser.add_argument("--mule-model-config", type=Path, required=True, help="model config to rehearse with")
    parser.add_argument("--out", type=Path, required=True, help="folder for the dry-run output")
    args = parser.parse_args(argv)

    previous_config = active_mule_model_config_path()
    set_mule_model_config_path(args.mule_model_config)
    try:
        return run_pilot(
            tuple(args.training),
            tuple(args.validation),
            (),
            args.truth_root,
            args.world_cards_root,
            args.out,
            (),
            stop_before_test=True,
        )
    finally:
        set_mule_model_config_path(previous_config)


if __name__ == "__main__":
    raise SystemExit(main())
