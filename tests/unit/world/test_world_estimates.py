"""
WHAT: Unit tests verifying machine calibration, resource estimation scaling, and execution throughput stability.
It validates that sizing configuration complies with schema policies, empirical rates are positive,
and projections consistently lean high without exceeding safety ratio bounds.
It also confirms that simulation speed remains steady as worlds scale up from small to large population counts.

WHY: Guarantees that operators receive accurate, conservatively high resource estimates before launching big runs.
Validating that scaling holds across multi-thousand person worlds prevents throughput collapses and memory overruns.
Automated checks ensure that configuration parameters and safety margins remain synchronized with code behavior.

TERM: machine calibration: Measuring execution rates on the active host hardware to ground scaling calculations,
      e.g. running a small 60-person world to measure real CPU and memory throughput.
TERM: resource estimate: A projected calculation of transaction counts, run seconds, file bytes, and heap memory,
      e.g. estimating 55,000 transactions for a 1000-person simulation.
TERM: safety margin: A policy multiplier applied to prevent under-estimating resource demands,
      e.g. increasing transaction projections by 15 percent.
TERM: throughput stability: Maintaining steady transaction processing rates per second as world size increases,
      e.g. ensuring a 1000-person world runs at least half as fast per transaction as a 60-person world.
TERM: peak memory: Maximum recorded heap allocation reached during world simulation execution,
      e.g. tracing 15 megabytes of memory for a 300-person world.
"""

import time
import tracemalloc
from pathlib import Path
from typing import Any

import pytest

from core.config_validator import validate_file
from world.estimates import (
    MachineProfile,
    calibrate_machine,
    estimate_world,
    read_sizing_config,
)
from world.fast_mode import FastRunSettings, run_fast_world, write_events_file


@pytest.fixture(scope="module")
def sizing_context() -> tuple[dict[str, Any], MachineProfile]:
    """
    WHAT: Module-scoped test fixture loading world sizing configuration and executing host calibration once.
    It returns the parsed configuration dictionary alongside the empirical MachineProfile benchmark record.
    Reusing this fixture across test cases avoids repeating calibration overhead for every individual test.

    WHY: Running hardware calibration once per test module keeps the test suite execution fast and responsive.
    Providing a shared, immutable profile ensures all estimate assertions evaluate against identical baseline rates.
    It encapsulates test setup into a standardized pytest fixture structure.

    Args:
        None: This pytest fixture accepts no input parameters.

    Returns:
        Tuple containing the parsed sizing configuration dictionary and the calibrated MachineProfile.

    Example:
        # Fixture invoked automatically by pytest when requested by test arguments
        # cfg, profile = sizing_context
    """
    # Load authoritative sizing policies from the configuration file.
    config = read_sizing_config()
    # Execute empirical benchmark on the active host to derive hardware profile rates.
    profile = calibrate_machine(config)
    # Pass back the configuration and calibration profile pair.
    return config, profile


def test_sizing_config_is_valid() -> None:
    """
    WHAT: Asserts that config/world_sizing.yaml passes all structural validation rules with zero errors.
    It validates that all required policy blocks, labels, values, and justifications are properly formed.
    The test invokes validate_file directly against the YAML document path.

    WHY: Prevents misconfigured sizing policies from causing runtime failures during machine calibration.
    Enforces repository standards requiring all operational numbers to carry documented justifications.
    Validating configuration upfront protects downstream estimation functions from missing keys.

    Args:
        None: This test accepts no input parameters.

    Returns:
        None: Asserts empty error list and returns nothing.

    Example:
        # Validates configuration syntax and schema rules
        # test_sizing_config_is_valid()
    """
    # 1. Run core validator against the world sizing configuration file.
    errors = validate_file(Path("config/world_sizing.yaml"))

    # 2. Assert zero schema validation errors were reported.
    assert errors == []


def test_calibration_values_are_positive(sizing_context: tuple[dict[str, Any], MachineProfile]) -> None:
    """
    WHAT: Asserts that all empirical throughput and cost rates derived during calibration are strictly positive.
    It checks payments per second, payments per person day, bytes per payment, memory per payment, and event count.
    Each rate is confirmed to exceed zero, ensuring the benchmark execution produced valid measurements.

    WHY: Protects against corrupted or aborted calibration runs producing zero or negative hardware rates.
    Zero rates would cause division-by-zero crashes or degenerate zero-cost projections during estimation.
    Confirming positive rates guarantees the host benchmark completed successfully.

    Args:
        sizing_context: Pytest fixture supplying the sizing configuration and calibrated MachineProfile.

    Returns:
        None: Asserts strictly positive rate properties and returns nothing.

    Example:
        # Verifies all calibrated rates are positive floating-point values
        # test_calibration_values_are_positive(sizing_context)
    """
    # 1. Extract calibrated machine hardware profile from test fixture.
    _config, profile = sizing_context

    # 2. Assert all measured throughput, frequency, and resource rates are strictly positive.
    assert profile.payments_per_second > 0.0
    assert profile.payments_per_person_day > 0.0
    assert profile.file_bytes_per_payment > 0.0
    assert profile.memory_bytes_per_payment > 0.0
    assert profile.calibration_payments > 0


def test_payment_estimate_leans_high(sizing_context: tuple[dict[str, Any], MachineProfile]) -> None:
    """
    WHAT: Asserts that estimated payment count for a 300-person world bounds the real count conservatively.
    It runs a 61-day simulation with seed 3 and compares observed transactions against the projected estimate.
    The estimate must be greater than or equal to the real count, and within the configured upper ratio bound.

    WHY: Guarantees that payment projections lean safely high so operators do not under-provision workloads.
    Bounding the estimate from above ensures projections remain tight and realistic rather than wildly exaggerated.
    Printing measured counts provides audit visibility into estimation accuracy.

    Args:
        sizing_context: Pytest fixture supplying the sizing configuration and calibrated MachineProfile.

    Returns:
        None: Asserts payment estimate bounds and returns nothing.

    Example:
        # Asserts real payments <= estimated payments <= real payments * upper ratio
        # test_payment_estimate_leans_high(sizing_context)
    """
    # 1. Unpack configuration and machine profile from fixture.
    config, profile = sizing_context
    checks = config["estimate_checks"]
    payments_upper_ratio = checks["payments_upper_ratio"]

    # 2. Execute actual simulation run for 300 people across 61 days with seed 3.
    settings = FastRunSettings(seed=3, n_people=300, n_shops=40, n_days=61, range_name="training")
    output = run_fast_world(settings)
    real_payments = len(output.result.events)

    # 3. Compute projected estimate using calibrated machine rates.
    estimate = estimate_world(300, 61, profile, config)

    # 4. Print measured comparison for audit logging.
    print(
        f"MEASURED: real_payments={real_payments} est_payments={estimate.payments} "
        f"max_allowed={round(real_payments * payments_upper_ratio, 1)}"
    )

    # 5. Assert estimated payments bound real transactions from above without exceeding safety ceiling.
    assert real_payments <= estimate.payments <= real_payments * payments_upper_ratio


def test_file_size_estimate_leans_high(
    sizing_context: tuple[dict[str, Any], MachineProfile],
    tmp_path: Path,
) -> None:
    """
    WHAT: Asserts that estimated events file size for a 300-person world bounds real serialized bytes.
    It writes events.jsonl to a temporary directory and measures the exact resulting file size in bytes.
    The estimate must be greater than or equal to the real bytes, and within the configured file ratio bound.

    WHY: Ensures disk space projections reliably prevent disk-full conditions on target execution environments.
    Validating that estimates stay within reasonable bounds prevents unnecessary rejection of feasible runs.
    Audit output documents exact file byte measurements alongside projected estimates.

    Args:
        sizing_context: Pytest fixture supplying sizing configuration and calibrated MachineProfile.
        tmp_path: Pytest temporary directory fixture for isolated events file output.

    Returns:
        None: Asserts file size estimate bounds and returns nothing.

    Example:
        # Asserts real bytes <= estimated bytes <= real bytes * file upper ratio
        # test_file_size_estimate_leans_high(sizing_context, tmp_path)
    """
    # 1. Unpack configuration and machine profile from fixture.
    config, profile = sizing_context
    checks = config["estimate_checks"]
    file_bytes_upper_ratio = checks["file_bytes_upper_ratio"]

    # 2. Run simulation and serialize events to temporary output file.
    settings = FastRunSettings(seed=3, n_people=300, n_shops=40, n_days=61, range_name="training")
    output = run_fast_world(settings)
    events_path = tmp_path / "events_300.jsonl"
    write_events_file(output, events_path)
    real_bytes = events_path.stat().st_size

    # 3. Calculate projected file size using empirical byte rate.
    estimate = estimate_world(300, 61, profile, config)

    # 4. Print measured file size comparison for audit logs.
    print(
        f"MEASURED: real_bytes={real_bytes} est_bytes={estimate.file_bytes} "
        f"max_allowed={round(real_bytes * file_bytes_upper_ratio, 1)}"
    )

    # 5. Assert estimate covers real file size and does not exceed upper bound.
    assert real_bytes <= estimate.file_bytes <= real_bytes * file_bytes_upper_ratio


def test_memory_estimate_covers_real_peak(sizing_context: tuple[dict[str, Any], MachineProfile]) -> None:
    """
    WHAT: Asserts that estimated peak memory for a 300-person world covers the real observed peak heap allocation.
    It runs the 300-person simulation under tracemalloc tracing to record the true peak memory consumption.
    The test verifies that the projected memory estimate is greater than or equal to the real measured peak.

    WHY: Protects operators against out-of-memory errors by ensuring memory estimates lean conservatively high.
    Memory estimation is critical for scheduling multi-worker jobs on memory-constrained servers.
    Printing measured peak heap bytes verifies empirical memory scaling behavior.

    Args:
        sizing_context: Pytest fixture supplying sizing configuration and calibrated MachineProfile.

    Returns:
        None: Asserts peak memory coverage and returns nothing.

    Example:
        # Asserts real peak memory <= estimated memory bytes
        # test_memory_estimate_covers_real_peak(sizing_context)
    """
    # 1. Unpack configuration and profile from fixture.
    config, profile = sizing_context

    # 2. Execute simulation under tracemalloc to capture actual peak heap memory.
    settings = FastRunSettings(seed=3, n_people=300, n_shops=40, n_days=61, range_name="training")
    tracemalloc.start()
    _output = run_fast_world(settings)
    _cur, real_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # 3. Compute projected memory requirement.
    estimate = estimate_world(300, 61, profile, config)

    # 4. Output measured peak memory comparison.
    print(f"MEASURED: real_peak={real_peak} est_memory={estimate.memory_bytes}")

    # 5. Assert estimated memory conservatively covers observed peak heap allocation.
    assert real_peak <= estimate.memory_bytes


def test_time_estimate_within_margin(sizing_context: tuple[dict[str, Any], MachineProfile]) -> None:
    """
    WHAT: Asserts that real execution time for a 300-person world stays within the configured time margin.
    It times a 61-day simulation run using time.perf_counter() without memory tracing overhead.
    The observed execution seconds must not exceed the estimated run duration multiplied by the time ratio bound.

    WHY: Ensures runtime estimates provide reliable scheduling guidance for automated pipeline executions.
    Allowing a noise margin accounts for concurrent operating system background tasks.
    Logging measured execution times enables continuous monitoring of simulation throughput.

    Args:
        sizing_context: Pytest fixture supplying sizing configuration and calibrated MachineProfile.

    Returns:
        None: Asserts execution time bound and returns nothing.

    Example:
        # Asserts real seconds <= estimated seconds * time upper ratio
        # test_time_estimate_within_margin(sizing_context)
    """
    # 1. Unpack configuration and machine profile from fixture.
    config, profile = sizing_context
    time_upper_ratio = config["estimate_checks"]["time_upper_ratio"]

    # 2. Measure actual wall-clock execution time without tracing overhead.
    settings = FastRunSettings(seed=3, n_people=300, n_shops=40, n_days=61, range_name="training")
    start = time.perf_counter()
    _output = run_fast_world(settings)
    real_seconds = time.perf_counter() - start

    # 3. Derive projected duration from calibrated throughput.
    estimate = estimate_world(300, 61, profile, config)

    # 4. Output measured timing results for audit reporting.
    print(
        f"MEASURED: real_seconds={round(real_seconds, 3)} est_seconds={round(estimate.seconds, 3)} "
        f"max_allowed={round(estimate.seconds * time_upper_ratio, 3)}"
    )

    # 5. Assert actual execution duration does not exceed the allowed upper time bound.
    assert real_seconds <= estimate.seconds * time_upper_ratio


def test_speed_stays_steady_as_worlds_grow(sizing_context: tuple[dict[str, Any], MachineProfile]) -> None:
    """
    WHAT: Asserts that simulation throughput does not collapse as world population scales from 60 to 1000 citizens.
    It executes a 1000-person world and measures transaction processing speed in payments per second.
    The speed must remain at or above the minimum steady speed ratio relative to the calibration world.

    WHY: Proves that index-based random sampling prevents quadratic scaling slowdowns in larger worlds.
    Guarantees that calibration rates measured on small worlds remain valid predictors for large simulations.
    Failing this test indicates an algorithmic regression in entity lookup or payment scheduling.

    Args:
        sizing_context: Pytest fixture supplying sizing configuration and calibrated MachineProfile.

    Returns:
        None: Asserts throughput stability and returns nothing.

    Example:
        # Asserts 1000-person speed >= calibration speed * steady speed min ratio
        # test_speed_stays_steady_as_worlds_grow(sizing_context)
    """
    # 1. Unpack configuration and profile from fixture.
    config, profile = sizing_context
    steady_speed_min_ratio = config["estimate_checks"]["steady_speed_min_ratio"]

    # 2. Execute 1000-person simulation with seed 4 and measure transaction processing speed.
    settings = FastRunSettings(seed=4, n_people=1000, n_shops=120, n_days=61, range_name="training")
    start = time.perf_counter()
    output = run_fast_world(settings)
    seconds = time.perf_counter() - start
    speed_1000 = len(output.result.events) / seconds

    # 3. Output measured speed comparison for audit records.
    print(
        f"MEASURED: speed_1000={round(speed_1000, 1)} pps_cal={round(profile.payments_per_second, 1)} "
        f"min_allowed={round(profile.payments_per_second * steady_speed_min_ratio, 1)}"
    )

    # 4. Assert 1000-person throughput maintains steady speed relative to calibration baseline.
    assert speed_1000 >= profile.payments_per_second * steady_speed_min_ratio


def test_estimate_scales_and_refuses_bad_input(sizing_context: tuple[dict[str, Any], MachineProfile]) -> None:
    """
    WHAT: Asserts that estimates scale linearly with population and reject invalid non-positive inputs.
    It verifies that doubling citizen population doubles projected payments to within rounding margin (+-1).
    It also verifies that passing zero population or duration raises ValueError with clear messages.

    WHY: Validates mathematical consistency of the estimation model across varying world dimensions.
    Enforces strict input validation to protect downstream components from invalid zero or negative configurations.
    Confirms robust error handling before operators initiate batch simulation runs.

    Args:
        sizing_context: Pytest fixture supplying sizing configuration and calibrated MachineProfile.

    Returns:
        None: Asserts linear scaling and exception raising and returns nothing.

    Example:
        # Asserts doubling population doubles payments and rejects zero people
        # test_estimate_scales_and_refuses_bad_input(sizing_context)
    """
    # 1. Extract configuration and profile from fixture.
    config, profile = sizing_context

    # 2. Compute estimates for 100 people and 200 people across 61 days.
    est_100 = estimate_world(100, 61, profile, config)
    est_200 = estimate_world(200, 61, profile, config)

    # 3. Assert doubling population scales payment estimate linearly to within rounding tolerance.
    assert abs(est_200.payments - est_100.payments * 2) <= 1

    # 4. Assert zero population is rejected with ValueError.
    with pytest.raises(ValueError) as exc_people:
        estimate_world(0, 61, profile, config)
    assert "positive integers" in str(exc_people.value)

    # 5. Assert zero duration is rejected with ValueError.
    with pytest.raises(ValueError) as exc_days:
        estimate_world(100, 0, profile, config)
    assert "positive integers" in str(exc_days.value)
