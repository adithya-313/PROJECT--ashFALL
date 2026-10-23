"""
WHAT: Plans a few sampled worlds, runs each with the complaints gang and prints how long each took, with the mean,
the maximum and the machine it ran on. It writes no files and changes no config.

WHY: How many worlds the project can afford depends on how long one takes. The numbers printed here are an ESTIMATE
from one machine and one moment; a later amendment decides the counts, this script decides nothing.

TERM: sampled world: a world planned by world/world_sampler.py and run with its key parameters pinned.
"""

import argparse
import os
import platform
import sys
import time
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.config_validator import load_yaml  # noqa: E402
from world.gang_config import gang_config_from_plain  # noqa: E402
from world.sampled_world import run_sampled_world  # noqa: E402
from world.world_configs import read_world_configs  # noqa: E402
from world.world_sampler import load_sampler_config, sample_worlds  # noqa: E402

GANG_PATH: Path = REPO_ROOT / "showcase" / "fixtures" / "level_12" / "gang_complaints.yaml"
CODE_VERSION: str = "measure-sampled-world"


def read_arguments(argv: list[str]) -> argparse.Namespace:
    """Parses the command line: preset, number of worlds, sampler seed and split."""
    parser = argparse.ArgumentParser(description="Measure how long sampled worlds take to run (ESTIMATE only).")
    parser.add_argument("--preset", default="tiny", help="size preset of config/world_sizing.yaml")
    parser.add_argument("--count", type=int, default=3, help="number of worlds to plan and run")
    parser.add_argument("--sampler-seed", type=int, default=1, help="master seed of the sampler")
    parser.add_argument("--split", default="training", help="training, validation, test or stress")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Plans and runs the worlds, printing one line per world and then the summary. Returns the exit code."""
    args = read_arguments(sys.argv[1:] if argv is None else argv)
    cfg = load_sampler_config()
    ranges = read_world_configs().world_ranges
    gang = gang_config_from_plain(load_yaml(GANG_PATH))
    specs = sample_worlds(args.split, args.count, args.sampler_seed, cfg, ranges)

    print("ESTIMATE: one machine, one moment; this script measures and decides nothing.")
    seconds_list: list[float] = []
    for spec in specs:
        started = time.perf_counter()
        output = run_sampled_world(spec, args.preset, (gang,), cfg, CODE_VERSION)
        seconds = time.perf_counter() - started
        seconds_list.append(seconds)
        executed = len(output.result.run_result.events)
        print(f"{spec.world_id} seed={spec.seed} payments_executed={executed} seconds={seconds:.2f}")

    print(f"mean_seconds={sum(seconds_list) / len(seconds_list):.2f} max_seconds={max(seconds_list):.2f}")
    print(f"python={platform.python_version()} cpu_count={os.cpu_count()} platform={platform.platform()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
