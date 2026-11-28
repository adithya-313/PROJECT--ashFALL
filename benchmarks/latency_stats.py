"""
WHAT: Shared statistical calculation functions and latency configuration loaders used by
decision and Triton latency benchmarks.

WHY: Moving pure statistical routines into a dedicated module allows benchmarks to compute
percentiles, medians, and load configuration parameters without pulling heavy application
dependencies (such as asyncpg, database pools, or Triton gRPC clients).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
CONFIG_PATH: Path = REPO_ROOT / "config" / "evaluation.yaml"


def load_latency_method(config_file: Path | None = None) -> dict[str, Any]:
    """
    PURPOSE:
        Reads the pre-registered latency measurement method parameters from config/evaluation.yaml.
        These parameters specify how benchmarks collect timing data to produce a verdict.

    INPUTS:
        config_file: Optional Path to the evaluation configuration file. If omitted or None,
            the default CONFIG_PATH ("config/evaluation.yaml") is used.

    OUTPUTS:
        dict[str, Any]: A dictionary containing:
            - "warm_up_calls": int, number of unmeasured warm-up iterations.
            - "measured_calls": int, number of timed iterations per run.
            - "runs": int, number of distinct benchmark runs.
            - "judged_on": str, the statistical criterion (must be 'median_of_run_p99').
            - "slowest_calls_reported": int, number of tail outliers recorded.

    KEY STEPS:
        1. Open the specified YAML configuration file with UTF-8 encoding.
        2. Parse the YAML content and locate the 'latency_method' section.
        3. Validate that 'judged_on' matches 'median_of_run_p99' as pre-registered in A-035.
        4. Return a dictionary containing the validated numerical parameters.

    ERRORS RAISED:
        FileNotFoundError: If config_file does not exist on disk.
        ValueError: If 'latency_method' section is missing or 'judged_on' is not 'median_of_run_p99'.

    LIMITS:
        Reads only local configuration files; does not modify any files.
    """
    path = config_file or CONFIG_PATH
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict) or "latency_method" not in data:
        raise ValueError(f"Missing 'latency_method' section in evaluation config: {path}")

    method = data["latency_method"]
    if method.get("judged_on") != "median_of_run_p99":
        raise ValueError(
            f"Unsupported latency_method.judged_on {method.get('judged_on')!r}; expected 'median_of_run_p99'"
        )

    return {
        "warm_up_calls": int(method["warm_up_calls"]),
        "measured_calls": int(method["measured_calls"]),
        "runs": int(method["runs"]),
        "judged_on": str(method["judged_on"]),
        "slowest_calls_reported": int(method["slowest_calls_reported"]),
    }


def median_of_run_values(values: Sequence[float]) -> float:
    """
    PURPOSE:
        Computes the statistical median of a sequence of floating-point values across benchmark runs.
        This represents the canonical verdict metric (e.g. median of per-run p99s).

    INPUTS:
        values: A sequence of floating-point numbers (e.g. [1.2, 1.5, 1.4]).

    OUTPUTS:
        float: The median value of the input sequence.

    KEY STEPS:
        1. Validate that the input sequence is non-empty to prevent invalid mathematical operations.
        2. Convert the sequence to a NumPy array of 64-bit floats.
        3. Compute the median using NumPy's median implementation.
        4. Cast and return the result as a standard Python float.

    ERRORS RAISED:
        ValueError: If values sequence is empty.

    LIMITS:
        Requires at least one numerical element; handles odd and even sequence lengths via interpolation.
    """
    if not values:
        raise ValueError("median_of_run_values needs at least one run")
    return float(np.median(np.asarray(values, dtype=float)))


def compute_percentiles(values: Sequence[float]) -> dict[str, float]:
    """
    PURPOSE:
        Calculates comprehensive distribution summary statistics (count, min, p50, p95, p99, max)
        for a sequence of latency measurements in milliseconds.

    INPUTS:
        values: A sequence of floating-point latency timings (e.g. list of recorded milliseconds).

    OUTPUTS:
        dict[str, float]: Summary dictionary with keys:
            - "count": float, total number of observations.
            - "min": float, minimum observed latency.
            - "p50": float, 50th percentile (median) latency.
            - "p95": float, 95th percentile latency.
            - "p99": float, 99th percentile tail latency.
            - "max": float, maximum observed latency.

    KEY STEPS:
        1. Check that the input sequence is non-empty.
        2. Convert elements to a NumPy float array.
        3. Extract count, min, p50, p95, p99, and max using NumPy quantile/percentile methods.
        4. Return dictionary with all values cast to Python float.

    ERRORS RAISED:
        ValueError: If values sequence is empty.

    LIMITS:
        Assumes valid real numbers; NaNs or infinities in values will propagate according to NumPy rules.
    """
    if not values:
        raise ValueError("Cannot compute percentiles on an empty sequence")

    arr = np.array(values, dtype=float)
    return {
        "count": float(len(arr)),
        "min": float(np.min(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }
