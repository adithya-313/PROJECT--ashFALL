"""
WHAT: Calibration and resource estimation harness projecting world simulation requirements.
It benchmarks execution throughput, per-event file bytes, and peak heap allocation on the host machine.
It projects transaction counts, wall-clock run time, events file size, and memory usage for any world scale.

WHY: Prevents simulation crashes and out-of-memory errors by forecasting resource demands prior to execution.
Machine hardware varies widely across environments, requiring empirical runtime calibration
rather than static constants.
Safety margins ensure estimates lean conservatively high so operators never under-provision host resources.

TERM: calibration: Running a standardized benchmark workload on the host machine to measure real execution rates,
      e.g. timing a 60-person world to evaluate CPU speed.
TERM: estimate: A projected calculation of resource usage based on measured rates and mathematical scaling,
      e.g. forecasting run duration for a 1000-person world.
TERM: safety margin: A deliberate multiplier added to baseline projections to prevent underestimating resource needs,
      e.g. scaling transaction counts by 1.15.
TERM: payments per second: The number of simulated financial transactions generated and settled per wall-clock second,
      e.g. processing 5,000 payments/s.
TERM: peak memory: The maximum resident heap memory allocated by the Python process during simulation execution,
      e.g. tracing 55 MB of memory consumption.
"""

import argparse
import math
import tempfile
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.config_validator import load_yaml
from world.fast_mode import FastRunSettings, run_fast_world, write_events_file


def read_sizing_config(path: Path = Path("config/world_sizing.yaml")) -> dict[str, Any]:
    """
    WHAT: Loads and parses the world sizing policy configuration file from disk.
    It reads calibration benchmark parameters, safety margin multipliers, and validation check ratios.
    The resulting dictionary provides authoritative inputs for machine calibration and scaling calculations.

    WHY: Decouples empirical policy ratios from estimation code to eliminate hardcoded sizing heuristics.
    Centralizing sizing parameters in validated YAML ensures that all estimates follow approved operational policies.
    It guarantees that benchmark seeds and check thresholds remain auditable across environments.

    Args:
        path: Filesystem path pointing to the world sizing YAML configuration document.

    Returns:
        Structured dictionary containing parsed sizing policies, margins, and check thresholds.

    Example:
        >>> from world.estimates import read_sizing_config
        >>> config = read_sizing_config()
        >>> config["calibration_world"]["n_people"]
        60
    """
    # Load and validate the YAML document containing simulation sizing policies.
    return load_yaml(path)


@dataclass(frozen=True)
class MachineProfile:
    """
    WHAT: Immutable hardware execution profile capturing empirical benchmark rates on the host machine.
    It records transaction throughput per second, per-person-day transaction rates, and resource costs per payment.
    Estimation routines apply these rates to project total run time, file storage, and heap memory requirements.

    WHY: Encapsulates empirical machine performance measurements into a typed, validated data structure.
    Separating hardware rate measurement from scaling logic allows testing estimation algorithms
    with synthetic profiles.
    Freezing the profile prevents accidental mutation of measured baseline rates during simulation planning.

    Args:
        payments_per_second: Measured payment transaction generation and ledger settlement rate per second.
        payments_per_person_day: Average transaction generation frequency per citizen per calendar day.
        file_bytes_per_payment: Mean serialized JSON Lines byte footprint per payment event in events.jsonl.
        memory_bytes_per_payment: Peak heap allocation byte cost attributed to each simulated payment.
        calibration_payments: Total count of payment events generated during the calibration benchmark run.

    Example:
        >>> from world.estimates import MachineProfile
        >>> profile = MachineProfile(5000.0, 0.8, 350.0, 1024.0, 2915)
        >>> (profile.calibration_payments, profile.payments_per_second > 0)
        (2915, True)
    """

    payments_per_second: float
    payments_per_person_day: float
    file_bytes_per_payment: float
    memory_bytes_per_payment: float
    calibration_payments: int


def calibrate_machine(sizing_config: dict[str, Any]) -> MachineProfile:
    """
    WHAT: Runs a standardized baseline simulation on the current host to measure empirical performance metrics.
    It executes a timed run to determine throughput and serialized file size, followed by a memory tracing run.
    The combined measurements form a host hardware profile used to extrapolate resource requirements for larger worlds.

    WHY: Hardware speeds and allocation overheads vary substantially across developer laptops and cloud servers.
    Benchmarking on the active runtime environment eliminates brittle static latency assumptions.
    Separating the memory measurement into a dedicated run prevents memory tracer overhead from skewing timing metrics.

    Args:
        sizing_config: Dictionary containing calibration_world settings loaded from world_sizing.yaml.

    Returns:
        MachineProfile holding measured throughput, transaction frequencies, and byte cost rates.

    Example:
        >>> from world.estimates import calibrate_machine, read_sizing_config
        >>> profile = calibrate_machine(read_sizing_config())
        >>> # Calibration rates depend on machine hardware; we assert rates are positive
        >>> (profile.calibration_payments, profile.payments_per_second > 0)
        (2915, True)
    """
    # 1. Build canonical baseline world settings from configuration.
    cal = sizing_config["calibration_world"]
    settings = FastRunSettings(cal["seed"], cal["n_people"], cal["n_shops"], cal["n_days"], cal["range_name"])

    # 2. Execute timed simulation run to measure payment throughput.
    start = time.perf_counter()
    output = run_fast_world(settings)
    duration = time.perf_counter() - start
    payments = len(output.result.events)
    payments_per_second = payments / duration
    payments_per_person_day = payments / float(settings.n_people * settings.n_days)

    # 3. Write simulated events to measure average serialized file size per transaction.
    with tempfile.TemporaryDirectory() as tmp_dir:
        events_path = Path(tmp_dir) / "events.jsonl"
        write_events_file(output, events_path)
        file_bytes_per_payment = events_path.stat().st_size / float(payments)

    # 4. Execute separate run under memory tracing to isolate peak allocation without timing distortion.
    tracemalloc.start()
    _output_mem = run_fast_world(settings)
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    memory_bytes_per_payment = peak / float(payments)

    # 5. Assemble immutable hardware calibration profile.
    return MachineProfile(
        payments_per_second,
        payments_per_person_day,
        file_bytes_per_payment,
        memory_bytes_per_payment,
        payments,
    )


@dataclass(frozen=True)
class WorldEstimate:
    """
    WHAT: Immutable container aggregating resource demand forecasts for a projected world simulation.
    It encapsulates total expected transaction counts, wall-clock duration in seconds, events file size,
    and peak memory.
    Planning tools inspect these projections to ensure target workloads fit within host operational boundaries.

    WHY: Groups all resource dimensions into a single unified record to support clear capacity decisions.
    Providing explicit file and memory projections protects operators against storage exhaustion or kernel termination.
    Freezing the dataclass guarantees that computed estimates remain unmodified during evaluation workflows.

    Args:
        payments: Projected total payment transaction event count, scaled by safety margin.
        seconds: Projected execution duration in wall-clock seconds based on measured host throughput.
        file_bytes: Projected total size in bytes for the serialized events.jsonl output log.
        memory_bytes: Projected peak heap memory footprint in bytes required during simulation execution.

    Example:
        >>> from world.estimates import WorldEstimate
        >>> estimate = WorldEstimate(55000, 5.0, 19000000, 55000000)
        >>> (estimate.payments, estimate.file_bytes)
        (55000, 19000000)
    """

    payments: int
    seconds: float
    file_bytes: int
    memory_bytes: int


def estimate_world(
    n_people: int,
    n_days: int,
    profile: MachineProfile,
    sizing_config: dict[str, Any],
) -> WorldEstimate:
    """
    WHAT: Computes resource demand projections for a specified world size using measured machine rates.
    It applies the policy safety margin to transaction counts and extrapolates run time, file size, and peak memory.
    Inputs are strictly validated to require positive integer dimensions for citizen population and horizon days.

    WHY: Enables simulation operators to preview resource consumption before launching large compute jobs.
    Applying a conservative safety margin ensures estimates lean high, preventing unexpected resource exhaustion.
    Rejecting invalid or non-positive dimensions immediately protects downstream routines from zero-division errors.

    Args:
        n_people: Total citizen population count planned for the simulation world.
        n_days: Calendar horizon duration in full days planned for the simulation world.
        profile: MachineProfile containing empirical hardware throughput and cost-per-payment rates.
        sizing_config: Dictionary containing payment_count_margin policy parameters.

    Returns:
        WorldEstimate holding projected transaction counts, wall-clock seconds, file bytes, and memory bytes.

    Raises:
        ValueError: If n_people or n_days is not a positive integer.

    Example:
        >>> from world.estimates import MachineProfile, estimate_world, read_sizing_config
        >>> profile = MachineProfile(5000.0, 0.8, 350.0, 1000.0, 2915)
        >>> est = estimate_world(100, 10, profile, read_sizing_config())
        >>> (est.payments, est.file_bytes, est.memory_bytes)
        (920, 322000, 920000)
    """
    # Validate that population and duration are positive integer numbers.
    if not isinstance(n_people, int) or not isinstance(n_days, int) or n_people <= 0 or n_days <= 0:
        raise ValueError(
            f"World size parameters n_people ({n_people}) and n_days ({n_days}) must be positive integers."
        )

    # Retrieve policy margin multiplier to ensure transaction estimates lean conservatively high.
    margin = float(sizing_config["payment_count_margin"]["value"])

    # Calculate projected transaction count scaled by empirical frequency and safety margin.
    payments = math.ceil(n_people * n_days * profile.payments_per_person_day * margin)

    # Extrapolate execution duration based on calibrated hardware throughput.
    seconds = payments / profile.payments_per_second

    # Extrapolate events file size and peak memory allocation from per-payment cost rates.
    file_bytes = math.ceil(payments * profile.file_bytes_per_payment)
    memory_bytes = math.ceil(payments * profile.memory_bytes_per_payment)

    # Package calculated resource requirements into immutable estimate record.
    return WorldEstimate(payments, seconds, file_bytes, memory_bytes)


def describe_estimate(estimate: WorldEstimate) -> str:
    """
    WHAT: Formats a WorldEstimate into a concise, human-readable summary line with standard metric units.
    It formats transaction counts with comma separators, files in megabytes, memory in megabytes, and rounded seconds.
    The output string explicitly clarifies that projections lean conservatively high to assist operator planning.

    WHY: Translates raw byte counts and floating-point seconds into intuitive summaries for command-line users.
    Consistent formatting ensures operator logs and reports present uniform sizing information across CLI invocations.
    Clarifying conservative estimation prevents confusion when real simulations complete slightly faster than predicted.

    Args:
        estimate: WorldEstimate record containing projected transactions, duration, file size, and memory.

    Returns:
        Formatted single-line description string summarizing projected resource requirements.

    Example:
        >>> from world.estimates import WorldEstimate, describe_estimate
        >>> est = WorldEstimate(56000, 4.2, 19900000, 55000000)
        >>> print("\\n".join(describe_estimate(est).split(", ")))
        about 56,000 payments
        a 19.9 MB events file
        about 55 MB of memory and about 4 seconds on this machine (estimates lean high)
    """
    # Convert file size and peak memory bytes to decimal megabytes (1 MB = 1,000,000 bytes).
    file_mb = estimate.file_bytes / 1_000_000.0
    memory_mb = round(estimate.memory_bytes / 1_000_000.0)

    # Round projected execution duration to nearest whole second.
    seconds_rounded = round(estimate.seconds)

    # Assemble plain-English descriptive summary line.
    return (
        f"about {estimate.payments:,} payments, a {file_mb:.1f} MB events file, "
        f"about {memory_mb} MB of memory and about {seconds_rounded} seconds on this machine (estimates lean high)"
    )


def main(argv: list[str] | None = None) -> int:
    """
    WHAT: Command-line entry point for running machine calibration and estimating world resource requirements.
    It parses required --people and --days options, performs host calibration, and displays formatted estimates.
    Errors resulting from invalid arguments or negative dimensions print clean messages and return exit code 1.

    WHY: Provides interactive CLI access for operators sizing simulation workloads before running full batches.
    Automates live machine calibration so estimates accurately reflect the performance profile of the active host.
    Catching domain validation errors prevents unhandled exception tracebacks from cluttering terminal output.

    Args:
        argv: Optional list of command-line argument strings, defaulting to sys.argv when None.

    Returns:
        Integer exit status code (0 for success, 1 on ValueError or invalid arguments).

    Example:
        >>> import contextlib, io
        >>> from world.estimates import main
        >>> buf = io.StringIO()
        >>> with contextlib.redirect_stdout(buf):
        ...     code = main(["--people", "0", "--days", "61"])
        >>> code
        1
    """
    # Configure command-line parser requiring explicit people and days parameters.
    parser = argparse.ArgumentParser(description="Estimate world simulation resource requirements.")
    parser.add_argument("--people", type=int, required=True, help="Citizen population count.")
    parser.add_argument("--days", type=int, required=True, help="Simulation duration in days.")

    # Parse arguments and run estimation workflow, catching validation errors cleanly.
    try:
        args = parser.parse_args(argv)
        # Validate that population and simulation horizon parameters are strictly positive integers before calibrating.
        if args.people <= 0 or args.days <= 0:
            raise ValueError(
                f"World size parameters people ({args.people}) and days ({args.days}) must be positive integers."
            )
        sizing_config = read_sizing_config()
        print("Calibrating on this machine...")
        profile = calibrate_machine(sizing_config)
        print(f"payments per second: {profile.payments_per_second:.1f}")
        print(f"payments per person per day: {profile.payments_per_person_day:.4f}")
        print(f"bytes per payment: {profile.file_bytes_per_payment:.1f}")
        print(f"memory per payment: {profile.memory_bytes_per_payment:.1f}")
        estimate = estimate_world(args.people, args.days, profile, sizing_config)
        print(describe_estimate(estimate))
        return 0
    except ValueError as err:
        print(str(err))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
