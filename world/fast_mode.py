"""
WHAT: Fast-mode execution orchestrator running a full world simulation and exporting output artifacts.
It executes the canonical world generation, planning, device schedule, episode generation, and payment pipeline.
It serializes every simulated payment event into JSON Lines format and writes a cryptographic run manifest.

WHY: Provides a deterministic, single-command harness for executing full-lifecycle simulations.
Downstream benchmark suites, evaluation pipelines, and fraud model trainers require reproducible event logs.
Emitting standard manifest records guarantees auditability, cryptographic integrity, and data provenance.

TERM: fast mode: A streamlined simulation execution pipeline running an entire world directly in memory,
      e.g. running 60 days of town transactions in under a second without real-time delays.
TERM: JSON Lines: A text format storing structured records where each line is an independent JSON object,
      e.g. appending one transaction event per line in a log file.
TERM: run manifest: A metadata document summarizing simulation configuration, entity counts, and data hashes,
      e.g. a manifest.json file recording settings and file fingerprints for reproducibility audits.
TERM: SHA-256 fingerprint: A 256-bit cryptographic hash uniquely representing the binary content of a file,
      e.g. a 64-character hex string verifying an events file has not been altered.
TERM: deterministic: A process that produces identical results given the exact same initial inputs and seed,
      e.g. generating bit-for-bit identical transaction logs on every run.
"""

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from world.background_config import read_background_realism
from world.build_world import World, build_world, world_fingerprint
from world.calendar_runner import RunResult, run_world
from world.clock import SimClock
from world.config_reader import read_world_ranges
from world.device_runtime import DeviceStamp
from world.device_schedule import draw_device_event_schedule
from world.episode_payments import plan_episode_payments
from world.episodes import Episode, draw_episode_schedule
from world.failures import PaymentEvent
from world.world_plan import WorldPlan, draw_world_plan, spawn_life_streams, spawn_world_streams

if TYPE_CHECKING:
    from world.world_sizing import SizingDecision


@dataclass(frozen=True)
class FastRunSettings:
    """
    WHAT: Immutable configuration parameters required to run a full world simulation in fast mode.
    It encapsulates the random seed, citizen population, commercial shop count, simulation horizon, and dataset split.
    Downstream simulation workflows pass this structure to guarantee completely reproducible environment construction.

    WHY: Groups simulation setup parameters into a single typed dataclass to eliminate loose argument lists.
    Freezing the dataclass prevents accidental runtime modification of simulation parameters.
    It guarantees that identical settings produce bit-for-bit identical world states and event sequences.

    Args:
        seed: Master integer random seed initializing all pseudo-random number generator streams.
        n_people: Total citizen population count generated within the simulated town.
        n_shops: Total commercial shop establishments participating in the local economy.
        n_days: Simulation calendar horizon duration measured in full calendar days.
        range_name: Configuration partition name determining parameter ranges (e.g. 'training').

    Example:
        >>> from world.fast_mode import FastRunSettings
        >>> settings = FastRunSettings(seed=42, n_people=100, n_shops=10, n_days=61, range_name="training")
        >>> (settings.seed, settings.n_people, settings.range_name)
        (42, 100, 'training')
    """

    seed: int
    n_people: int
    n_shops: int
    n_days: int
    range_name: str


@dataclass(frozen=True)
class FastRunOutput:
    """
    WHAT: Immutable container aggregating all entities, schedules, and results from a fast-mode simulation run.
    It holds the generated town world, frozen world plan, payment execution result, episodes,
    and episode summary report.
    Export routines read from this structure to serialize transactions and build cryptographic manifests.

    WHY: Decouples simulation computation from disk serialization and file formatting logic.
    Retaining rich domain dataclasses in memory allows tests and benchmarks to inspect results directly.
    Freezing the container guarantees that completed run artifacts cannot be mutated after execution.

    Args:
        world: World container housing all town entities, bank accounts, and ledgers.
        plan: WorldPlan structure holding calendar dates, festival schedules, and failure dynamics.
        result: RunResult record containing chronological payment events, telemetry, and execution metrics.
        episodes: List of scheduled Episode instances drawn for ordinary citizens.
        episode_report: Summary dictionary describing episode payment planning counts and constraints.

    Example:
        >>> from world.fast_mode import FastRunOutput, FastRunSettings, run_fast_world
        >>> settings = FastRunSettings(seed=101, n_people=10, n_shops=4, n_days=61, range_name="training")
        >>> output = run_fast_world(settings)
        >>> len(output.result.events)
        499
    """

    world: World
    plan: WorldPlan
    result: RunResult
    episodes: list[Episode]
    episode_report: dict[str, Any]


def run_fast_world(settings: FastRunSettings) -> FastRunOutput:
    """
    WHAT: Executes a complete simulated world lifecycle according to the canonical test recipe.
    It builds entities, derives independent random streams, draws schedules, and runs payments.
    The sequence strictly preserves the deterministic order established in golden regression tests.

    WHY: Centralizes the end-to-end simulation lifecycle into a single authoritative function.
    Preventing divergent execution logic guarantees exact parity with existing golden benchmarks.
    Isolating lifecycle streams ensures future modifications cannot inadvertently disrupt random draws.

    Args:
        settings: FastRunSettings specifying seed, entity counts, horizon duration, and split.

    Returns:
        FastRunOutput holding completed world entities, calendar plan, run results, and episode reports.

    Example:
        >>> from world.fast_mode import FastRunSettings, run_fast_world
        >>> settings = FastRunSettings(seed=101, n_people=10, n_shops=4, n_days=61, range_name="training")
        >>> output = run_fast_world(settings)
        >>> (len(output.result.events), len(output.episodes))
        (499, 0)
    """
    # 1. Build foundational town entities including banks, double-entry ledgers, shops, and citizens.
    world = build_world(seed=settings.seed, n_people=settings.n_people, n_shops=settings.n_shops)

    # 2. Derive independent pseudo-random streams for calendar planning and dynamic transaction execution.
    plan_rng, run_rng = spawn_world_streams(settings.seed)

    # 3. Derive auxiliary pseudo-random streams for device lifecycle transitions and burst episodes.
    device_rng, episode_rng = spawn_life_streams(settings.seed)

    # 4. Construct the temporal world plan containing clock, festival calendar, and failure settings.
    plan = draw_world_plan(plan_rng, n_days=settings.n_days, range_name=settings.range_name)

    # 5. Read empirical world configuration bounds and device realism parameters.
    ranges = read_world_ranges()
    device_config = read_background_realism(Path("config/device_realism.yaml"))

    # 6. Pre-draw device hardware updates, replacements, and OS rollout wave schedules.
    device_events = draw_device_event_schedule(device_config, device_rng, world.people, plan.festivals, plan.n_days)

    # 7. Schedule deceptive high-velocity payment burst episodes for ordinary citizens.
    episodes = draw_episode_schedule(ranges, episode_rng, world.people, plan.n_days, range_name=settings.range_name)

    # 8. Plan concrete inbound and outbound transaction flows for each scheduled episode.
    episode_plan = plan_episode_payments(world, plan, episodes, ranges, episode_rng, range_name=settings.range_name)

    # 9. Execute all scheduled routine, merchant, and episode payments through bank ledgers.
    result = run_world(world, plan, run_rng, device_events=device_events, episode_payments=episode_plan.payments)

    # Pass back the complete assembled simulation state and execution artifacts.
    return FastRunOutput(
        world=world,
        plan=plan,
        result=result,
        episodes=episodes,
        episode_report=episode_plan.report,
    )


def event_to_record(
    event: PaymentEvent,
    purpose: str,
    stamp: DeviceStamp,
    clock: SimClock,
) -> dict[str, Any]:
    """
    WHAT: Converts a simulated payment event, telemetry stamp, and purpose into a JSON-ready dictionary.
    It computes an ISO-8601 calendar timestamp from the minute offset and serializes exactly 14 field keys.
    Values are copied faithfully without type mutations, preserving null markers for missing attributes.

    WHY: Provides a normalized flat schema representation for transaction event logging.
    Eliminating timezone offsets prevents geographical or system clock ambiguity across environments.
    Standardizing field names ensures downstream fraud detection models receive consistent feature names.

    Args:
        event: PaymentEvent record containing transaction minute, accounts, amount, and status.
        purpose: Real-world semantic transaction purpose string (e.g. 'salary', 'everyday').
        stamp: DeviceStamp telemetry containing hardware fingerprint, OS version, and novelty flag.
        clock: SimClock instance translating simulation minute offsets into Gregorian calendar datetimes.

    Returns:
        Dictionary holding exactly the 14 standardized event fields required by the events-v1 format.

    Example:
        >>> from world.clock import SimClock, WORLD_START_DATE
        >>> from world.device_runtime import DeviceStamp
        >>> from world.failures import PaymentEvent
        >>> from world.fast_mode import event_to_record
        >>> clock = SimClock(WORLD_START_DATE)
        >>> ev = PaymentEvent(60, "acc_1", "acc_2", 5000, 0, 1, "success", "ok", None)
        >>> stamp = DeviceStamp("fp_abc", 3, False)
        >>> rec = event_to_record(ev, "everyday", stamp, clock)
        >>> (rec["timestamp"], rec["amount_paise"], rec["purpose"])
        ('2026-09-01T01:00:00', 5000, 'everyday')
    """
    # Map the simulated minute offset to an ISO-8601 calendar timestamp string without timezone.
    iso_timestamp = clock.to_datetime(event.minute).isoformat()

    # Assemble the flat dictionary record with exactly the 14 standardized transaction fields.
    return {
        "timestamp": iso_timestamp,
        "minute": event.minute,
        "from_account": event.from_account,
        "to_account": event.to_account,
        "amount_paise": event.amount_paise,
        "from_bank_index": event.from_bank_index,
        "to_bank_index": event.to_bank_index,
        "status": event.status,
        "reason": event.reason,
        "failure_domain": event.failure_domain,
        "purpose": purpose,
        "sender_fingerprint": stamp.fingerprint,
        "sender_os_generation": stamp.os_generation,
        "sender_is_new_device": stamp.is_new_device,
    }


def write_events_file(output: FastRunOutput, path: Path) -> str:
    """
    WHAT: Writes all chronological payment events from a simulation run to a JSON Lines file.
    It serializes each event dictionary with sorted keys and compact separators into a newline-terminated line.
    After completing writing, it reads back the file bytes and calculates its SHA-256 cryptographic digest.

    WHY: Emits transaction events in a standard, stream-friendly JSON Lines representation.
    Enforcing explicit UTF-8 encoding and LF newlines ensures bit-for-bit identical outputs across OS platforms.
    Returning the SHA-256 digest allows immediate verification of file content and inclusion in the manifest.

    Args:
        output: FastRunOutput container holding the simulated payment events, telemetry, and clock.
        path: Destination filesystem Path where the events.jsonl file will be written.

    Returns:
        Hexadecimal string representing the SHA-256 cryptographic digest of the written events file.

    Example:
        >>> import tempfile
        >>> from pathlib import Path
        >>> from world.fast_mode import FastRunSettings, run_fast_world, write_events_file
        >>> settings = FastRunSettings(seed=101, n_people=10, n_shops=4, n_days=61, range_name="training")
        >>> output = run_fast_world(settings)
        >>> with tempfile.TemporaryDirectory() as tmp_dir:
        ...     digest = write_events_file(output, Path(tmp_dir) / "events.jsonl")
        >>> digest[:8]
        '3b969c22'
    """
    # Open destination file with explicit UTF-8 encoding and standard Unix line endings.
    with open(path, "w", encoding="utf-8", newline="\n") as events_file:
        # Iterate over parallel sequences of payment events, transaction purposes, and device telemetry stamps.
        for event, purpose, stamp in zip(
            output.result.events,
            output.result.event_purposes,
            output.result.event_sender_devices,
        ):
            record = event_to_record(event, purpose, stamp, output.plan.clock)
            # Serialize the record with sorted keys and compact separators to ensure deterministic bytes.
            events_file.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")

    # Read back raw file bytes to calculate the authoritative SHA-256 cryptographic digest.
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_manifest(
    settings: FastRunSettings,
    output: FastRunOutput,
    events_sha256: str,
) -> dict[str, Any]:
    """
    WHAT: Constructs a metadata dictionary summarizing simulation parameters, counts, and cryptographic digests.
    It captures format version, generator identifier, world fingerprint, event statistics, and file checksums.
    The dictionary omits all non-deterministic environment properties such as wall-clock times or local file paths.

    WHY: Documents the exact provenance, configuration settings, and output integrity of a simulation run.
    Excluding variable host properties ensures that identical seeds produce identical manifest contents.
    Downstream verification tools rely on this document to audit data integrity before ingestion.

    Args:
        settings: FastRunSettings used to initialize and configure the simulation world.
        output: FastRunOutput container holding world entities, execution results, and episode reports.
        events_sha256: Hexadecimal SHA-256 digest of the associated events.jsonl file.

    Returns:
        Structured dictionary adhering strictly to the events-v1 manifest specification schema.

    Example:
        >>> from world.fast_mode import FastRunSettings, run_fast_world, build_manifest
        >>> settings = FastRunSettings(seed=101, n_people=10, n_shops=4, n_days=61, range_name="training")
        >>> output = run_fast_world(settings)
        >>> manifest = build_manifest(settings, output, "dummy_hash")
        >>> (manifest["format"], manifest["event_count"], manifest["events_file"])
        ('events-v1', 499, 'events.jsonl')
    """
    # Extract root simulation clock start date formatted as an ISO-8601 string.
    clock_start_iso = output.plan.clock.to_datetime(0).isoformat()

    # Calculate authoritative cryptographic fingerprint of all initialized world entities.
    world_digest = world_fingerprint(output.world)

    # Assemble manifest metadata adhering strictly to the deterministic events-v1 schema.
    return {
        "format": "events-v1",
        "generator": "world.fast_mode",
        "settings": {
            "seed": settings.seed,
            "n_people": settings.n_people,
            "n_shops": settings.n_shops,
            "n_days": settings.n_days,
            "range_name": settings.range_name,
        },
        "clock_start": clock_start_iso,
        "world_fingerprint": world_digest,
        "events_file": "events.jsonl",
        "events_sha256": events_sha256,
        "event_count": len(output.result.events),
        "counts": output.result.counts,
        "device_counts": output.result.device_counts,
        "episode_count": len(output.episodes),
        "episode_report": output.episode_report,
    }


def write_world_files(settings: FastRunSettings, out_dir: Path) -> Path:
    """
    WHAT: Orchestrates a fast-mode simulation run and writes both events.jsonl and manifest.json to disk.
    It verifies that the target directory is empty before executing, creating the folder if needed.
    The simulation completes entirely in memory before any data files are written to the filesystem.

    WHY: Guarantees transactional atomicity of simulation exports by preventing partial file writes.
    Refusing non-empty directories protects existing simulation runs from accidental data overwrites.
    Structuring exports into paired events and manifest files standardizes dataset distribution.

    Args:
        settings: FastRunSettings specifying simulation seed, entity counts, horizon, and split.
        out_dir: Filesystem Path designating the directory where run artifacts must be written.

    Returns:
        Path to the written manifest.json metadata file.

    Raises:
        FileExistsError: If out_dir already exists and contains one or more files or subdirectories.

    Example:
        >>> import tempfile
        >>> from pathlib import Path
        >>> from world.fast_mode import FastRunSettings, write_world_files
        >>> settings = FastRunSettings(seed=101, n_people=10, n_shops=4, n_days=61, range_name="training")
        >>> with tempfile.TemporaryDirectory() as tmp_dir:
        ...     manifest_path = write_world_files(settings, Path(tmp_dir) / "run")
        ...     manifest_name = manifest_path.name
        >>> manifest_name
        'manifest.json'
    """
    # Guard against overwriting existing data by rejecting non-empty output folders.
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(
            f"Output folder '{out_dir}' already exists and is not empty; nothing was written."
        )

    # Execute simulation lifecycle completely in memory before creating any disk files.
    output = run_fast_world(settings)

    # Ensure parent output directory hierarchy exists prior to writing output files.
    out_dir.mkdir(parents=True, exist_ok=True)

    # Serialize chronological payment events and obtain file cryptographic digest.
    events_path = out_dir / "events.jsonl"
    events_sha256 = write_events_file(output, events_path)

    # Build manifest metadata dictionary linking event counts, settings, and file hashes.
    manifest = build_manifest(settings, output, events_sha256)

    # Write formatted manifest JSON file with explicit UTF-8 encoding and final newline.
    manifest_path = out_dir / "manifest.json"
    manifest_text = json.dumps(manifest, sort_keys=True, indent=2) + "\n"
    manifest_path.write_text(manifest_text, encoding="utf-8", newline="\n")

    # Pass back the filesystem path of the completed run manifest.
    return manifest_path


def _build_parser() -> argparse.ArgumentParser:
    """
    WHAT: Constructs and configures the command-line argument parser for fast-mode world simulation runs.
    It registers options for preset templates, expert files, random seed, entity counts, horizon, and limits.
    Flag definitions specify appropriate data types and documentation strings to assist operators in shells.

    WHY: Separates argument parsing configuration from command-line execution and lifecycle control flow.
    Isolating parser construction into a dedicated helper keeps main() compact and focused on orchestration.
    Centralizing CLI parameter specifications enables unit tests to inspect flag definitions directly.

    Args:
        None: This builder routine takes no positional or keyword parameters.

    Returns:
        Configured ArgumentParser instance ready to parse command-line argument lists.

    Example:
        >>> parser = _build_parser()
        >>> args = parser.parse_args(["--out", "run_dir", "--estimate-only"])
        >>> (str(args.out), args.estimate_only)
        ('run_dir', True)
    """
    parser = argparse.ArgumentParser(description="Run simulated world in fast mode and export events.")
    parser.add_argument("--preset", type=str, default=None, help="Preset template name (tiny, small, medium, large).")
    parser.add_argument("--expert", type=Path, default=None, help="Path to expert YAML configuration file.")
    parser.add_argument("--seed", type=int, default=None, help="Master random seed integer.")
    parser.add_argument("--people", type=int, default=None, help="Citizen population count override.")
    parser.add_argument("--shops", type=int, default=None, help="Commercial shop count override.")
    parser.add_argument("--days", type=int, default=None, help="Simulation duration in days override.")
    parser.add_argument("--range-name", type=str, default=None, help="Dataset split name override.")
    parser.add_argument("--out", type=Path, required=True, help="Target output folder path.")
    parser.add_argument("--max-minutes", type=float, default=None, help="Maximum allowed runtime in minutes.")
    parser.add_argument("--estimate-only", action="store_true", help="Estimate feasibility without running.")
    return parser


def _evaluate_capacity(
    settings: FastRunSettings,
    out_dir: Path,
    sizing_config: dict[str, Any],
    max_minutes: float | None,
) -> "SizingDecision":
    """
    WHAT: Benchmarks host hardware and computes resource feasibility against operational machine limits.
    It calibrates throughput and memory cost rates on the local machine before reading available resources.
    The resulting SizingDecision is returned alongside printing diagnostic messages and recommendations.

    WHY: Enforces simulation safety checks before committing to heavy computation or file generation.
    Separating capacity evaluation into a focused helper keeps the CLI orchestrator modular and readable.
    Executing calibration on the target machine guarantees that sizing decisions reflect real hardware speeds.

    Args:
        settings: FastRunSettings specifying citizen population, shop counts, and simulation horizon.
        out_dir: Target filesystem path where simulation output artifacts would be exported.
        sizing_config: Policy configuration dictionary loaded from config/world_sizing.yaml.
        max_minutes: Optional operator runtime override in minutes, defaulting to config limit when None.

    Returns:
        SizingDecision container holding approval status, resource estimate, and diagnostic text.

    Example:
        >>> import contextlib, io, tempfile
        >>> from pathlib import Path
        >>> from world.estimates import read_sizing_config
        >>> from world.fast_mode import FastRunSettings, _evaluate_capacity
        >>> settings = FastRunSettings(101, 10, 4, 61, "training")
        >>> with tempfile.TemporaryDirectory() as tmp_dir:
        ...     buf = io.StringIO()
        ...     with contextlib.redirect_stdout(buf):
        ...         dec = _evaluate_capacity(settings, Path(tmp_dir), read_sizing_config(), 30.0)
        >>> dec.fits
        True
    """
    from world.estimates import calibrate_machine
    from world.machine_limits import read_machine_state
    from world.world_sizing import decide_run

    # 3. Print calibration notice and benchmark hardware execution performance.
    print("Calibrating on this machine...")
    profile = calibrate_machine(sizing_config)

    # 4. Read available host resources and evaluate sizing feasibility.
    default_limit = float(sizing_config["machine_limits"]["max_minutes"])
    effective_max_minutes = max_minutes if max_minutes is not None else default_limit
    state = read_machine_state(out_dir)
    decision = decide_run(settings, profile, state, sizing_config, effective_max_minutes)
    print(decision.message)

    return decision


def _export_and_report(settings: FastRunSettings, out_dir: Path) -> int:
    """
    WHAT: Executes the complete fast-mode simulation and writes output artifacts to the target folder.
    It serializes events to JSON Lines format, constructs the run manifest, and calculates cryptographic digests.
    Summary lines reporting manifest path, event count, and SHA-256 digest are printed to standard output.

    WHY: Isolates export orchestration and summary reporting into a dedicated, clean execution helper.
    Encapsulating disk serialization keeps the top-level main function within strict length boundaries.
    Providing immediate cryptographic confirmation enables shell scripts and operators to audit outputs.

    Args:
        settings: FastRunSettings specifying the validated simulation configuration.
        out_dir: Target filesystem path where events and manifest files must be saved.

    Returns:
        Integer exit code 0 indicating successful simulation execution and export.

    Example:
        >>> import contextlib, io, tempfile
        >>> from pathlib import Path
        >>> from world.fast_mode import FastRunSettings, _export_and_report
        >>> settings = FastRunSettings(101, 10, 4, 61, "training")
        >>> with tempfile.TemporaryDirectory() as tmp_dir:
        ...     buf = io.StringIO()
        ...     with contextlib.redirect_stdout(buf):
        ...         code = _export_and_report(settings, Path(tmp_dir) / "run")
        >>> code
        0
    """
    # 7. Write simulation files to disk and parse manifest metadata.
    manifest_path = write_world_files(settings, out_dir)
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Output manifest metadata and file digests for verification.
    print(f"Manifest: {manifest_path}")
    print(f"Events: {manifest_data['event_count']}")
    print(f"SHA-256: {manifest_data['events_sha256']}")
    return 0


def _resolve_cli_settings(
    args: argparse.Namespace,
    sizing_config: dict[str, Any],
) -> FastRunSettings | None:
    """
    WHAT: Resolves simulation settings from parsed command-line options and outputs diagnostic confirmation.
    It passes options to the resolution engine, intercepting ValueErrors to output plain-English explanations.
    On successful resolution, it prints the resolved population, shops, duration, split, and seed.

    WHY: Decouples command-line parameter extraction and error handling from downstream execution flow.
    Isolating parameter resolution keeps main() compact and focused on lifecycle orchestration.
    Printing resolved dimensions provides immediate feedback verifying knob scaling and preset selection.

    Args:
        args: Parsed command-line Namespace containing candidate parameter flags.
        sizing_config: Policy configuration dictionary loaded from config/world_sizing.yaml.

    Returns:
        Validated FastRunSettings instance, or None if parameter resolution encountered a ValueError.

    Example:
        >>> import argparse, contextlib, io
        >>> from world.estimates import read_sizing_config
        >>> from world.fast_mode import _resolve_cli_settings
        >>> args = argparse.Namespace(
        ...     seed=101, preset=None, people=60, shops=12,
        ...     days=61, range_name="training", expert=None
        ... )
        >>> buf = io.StringIO()
        >>> with contextlib.redirect_stdout(buf):
        ...     s = _resolve_cli_settings(args, read_sizing_config())
        >>> (s.n_people, s.n_shops)
        (60, 12)
    """
    from world.world_sizing import resolve_settings

    # 1. Resolve simulation parameters across expert, preset, and explicit pathways.
    try:
        settings = resolve_settings(
            seed=args.seed,
            preset=args.preset,
            n_people=args.people,
            n_shops=args.shops,
            n_days=args.days,
            range_name=args.range_name,
            expert_path=args.expert,
            sizing_config=sizing_config,
        )
    except ValueError as err:
        print(str(err))
        return None

    # Print confirmed world dimensions for operator visibility.
    people_info = f"{settings.n_people} people, {settings.n_shops} shops"
    time_info = f"{settings.n_days} days, range '{settings.range_name}', seed {settings.seed}"
    print(f"Resolved settings: {people_info}, {time_info}")
    return settings


def main(argv: list[str] | None = None) -> int:
    """
    WHAT: Command-line interface entry point for executing world simulations in fast mode.
    It parses parameters across preset, knob, and expert modes, verifies machine capacity, and runs simulation.
    If resources are insufficient, it displays a plain-English refusal with actionable suggestions and exits 1.

    WHY: Enables operators to run calibrated full-world simulations safely from shells or automation scripts.
    Pre-flight capacity checks protect developer laptops and servers against runaway memory and storage usage.
    Providing clear parameter resolution and cryptographic digests ensures transparency and reproducibility.

    Args:
        argv: Optional list of command-line argument strings, defaulting to sys.argv when None.

    Returns:
        Integer exit status code (0 for success or fitting estimate, 1 on refusal or configuration error).

    Example:
        >>> import contextlib, io, tempfile
        >>> from world.fast_mode import main
        >>> with tempfile.TemporaryDirectory() as tmp_dir:
        ...     buf = io.StringIO()
        ...     with contextlib.redirect_stdout(buf):
        ...         code = main([
        ...             "--seed", "101", "--people", "10", "--shops", "4",
        ...             "--days", "61", "--range-name", "training", "--out", tmp_dir + "/run"
        ...         ])
        >>> code
        0
    """
    from world.estimates import read_sizing_config

    # 1. Parse command-line arguments and resolve simulation parameters.
    parser = _build_parser()
    args = parser.parse_args(argv)
    sizing_config = read_sizing_config()
    settings = _resolve_cli_settings(args, sizing_config)
    if settings is None:
        return 1

    # 2. Reject existing non-empty output directories before performing calibration.
    if args.out.exists() and any(args.out.iterdir()):
        print(f"Output folder '{args.out}' already exists and is not empty; nothing was written.")
        return 1

    # 3 & 4. Calibrate host performance and evaluate machine limits.
    decision = _evaluate_capacity(settings, args.out, sizing_config, args.max_minutes)

    # 5. Handle estimate-only evaluation without writing simulation files.
    if args.estimate_only:
        return 0 if decision.fits else 1

    # 6. Halt execution if the proposed world violates host limits.
    if not decision.fits:
        return 1

    # 7. Execute full simulation and serialize events and manifest to disk.
    return _export_and_report(settings, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
