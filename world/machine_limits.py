"""
WHAT: Hardware capacity monitoring module inspecting available physical memory and disk storage.
It reads real-time host resource availability using psutil and standard filesystem utilities.
It compares projected simulation resource estimates against configurable operational thresholds to detect violations.

WHY: Prevents simulation runs from crashing host environments due to memory exhaustion or out-of-disk errors.
Proactively evaluating system capacity before running heavy workloads maintains workstation usability.
Clear plain-English diagnostics explain exactly which resource boundaries a projected simulation would exceed.

TERM: free memory: Physical RAM currently unoccupied and available to applications without swapping,
      e.g. 1,200 MB of available memory.
TERM: free disk: Unallocated storage capacity on the target filesystem partition available for writing files,
      e.g. 5,000 MB of free storage.
TERM: machine limit: An operational threshold restricting resource utilization to protect host stability,
      e.g. capping execution time at 30 minutes.
TERM: psutil: A cross-platform Python system monitoring library retrieving hardware utilization statistics,
      e.g. reading available physical memory.
"""

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil

from world.estimates import WorldEstimate


@dataclass(frozen=True)
class MachineState:
    """
    WHAT: Immutable snapshot of available physical memory and filesystem storage capacity on the host.
    It encapsulates available free memory in bytes and available free disk storage in bytes for a target path.
    Operational limit checkers evaluate these numbers to determine whether a simulation safely fits on the machine.

    WHY: Captures physical host capacity into a unified, typed data structure for limit evaluation.
    Freezing the snapshot prevents runtime modification during simulation planning and safety audits.
    Separating capacity measurement from policy checks allows testing limit logic using synthetic hardware states.

    Args:
        free_memory_bytes: Real-time available physical memory in bytes retrieved from the host operating system.
        free_disk_bytes: Available filesystem storage capacity in bytes on the target output drive partition.

    Example:
        >>> from world.machine_limits import MachineState
        >>> state = MachineState(free_memory_bytes=1_200_000_000, free_disk_bytes=5_000_000_000)
        >>> (state.free_memory_bytes, state.free_disk_bytes)
        (1200000000, 5000000000)
    """

    free_memory_bytes: int
    free_disk_bytes: int


def read_machine_state(out_dir: Path) -> MachineState:
    """
    WHAT: Inspects the host machine to read available physical RAM and available disk space for the output directory.
    It queries psutil for system-wide available memory and queries the filesystem for free disk space.
    If the target output directory does not exist yet, it traverses upwards to find the nearest existing parent folder.

    WHY: Provides ground-truth hardware metrics reflecting the real-time capacity of the active execution environment.
    Checking existing parent directories prevents filesystem query errors when targeting new output folders.
    Measuring both memory and disk ensures neither resource is inadvertently exhausted during world generation.

    Args:
        out_dir: Destination filesystem path where simulation output artifacts are planned to be written.

    Returns:
        MachineState record containing available free memory bytes and available free disk bytes.

    Example:
        >>> from pathlib import Path
        >>> from world.machine_limits import read_machine_state
        >>> state = read_machine_state(Path("."))
        >>> (state.free_memory_bytes > 0, state.free_disk_bytes > 0)
        (True, True)
    """
    # 1. Query host operating system for available physical memory without swap.
    free_memory_bytes = int(psutil.virtual_memory().available)

    # 2. Locate nearest existing directory in target path hierarchy to check storage capacity.
    target = out_dir.resolve()
    while not target.exists() and target != target.parent:
        target = target.parent

    # 3. Query filesystem partition hosting target path for available unallocated storage bytes.
    free_disk_bytes = int(shutil.disk_usage(target).free)

    # 4. Package real-time measurements into an immutable machine state snapshot.
    return MachineState(free_memory_bytes=free_memory_bytes, free_disk_bytes=free_disk_bytes)


def find_problems(
    estimate: WorldEstimate,
    state: MachineState,
    limits: dict[str, Any],
    max_minutes: float,
) -> list[str]:
    """
    WHAT: Evaluates projected simulation requirements against host capacity limits and returns any broken constraints.
    It checks wall-clock duration against max_minutes, heap memory against memory_share, and disk against disk_share.
    Each violated threshold produces a plain-English explanatory line; an empty list indicates the simulation fits.

    WHY: Identifies specific resource bottlenecks preventing a simulation from safely running on the active machine.
    Enforcing policy shares (such as 50% of free RAM and disk) keeps the machine responsive for concurrent tasks.
    Providing clear diagnostic messages enables human operators to adjust simulation scale or override limits.

    Args:
        estimate: WorldEstimate containing projected execution seconds, events file size, and peak memory.
        state: MachineState holding real-time available physical memory and free disk space.
        limits: Dictionary containing policy thresholds including memory_share and disk_share.
        max_minutes: Maximum permissible execution duration in minutes for this simulation run.

    Returns:
        List of plain-English violation strings describing each exceeded threshold, empty if all fit.

    Example:
        >>> from world.estimates import WorldEstimate
        >>> from world.machine_limits import MachineState, find_problems
        >>> est = WorldEstimate(50000, 45.0 * 60, 3_400_000_000, 900_000_000)
        >>> state = MachineState(free_memory_bytes=1_200_000_000, free_disk_bytes=5_000_000_000)
        >>> limits = {"memory_share": 0.5, "disk_share": 0.5}
        >>> problems = find_problems(est, state, limits, 30.0)
        >>> len(problems)
        3
    """
    # Initialize problem container collecting human-readable constraint violation messages.
    problems: list[str] = []

    # 1. Check projected execution duration against maximum allowed time budget.
    estimate_minutes = estimate.seconds / 60.0
    if estimate_minutes > max_minutes:
        limit_desc = f"{int(max_minutes)}-minute" if max_minutes == int(max_minutes) else f"{max_minutes}-minute"
        problems.append(f"time: about {round(estimate_minutes)} minutes, over the {limit_desc} limit")

    # 2. Check projected heap memory against allowable share of available host physical RAM.
    memory_share = float(limits.get("memory_share", 0.5))
    if estimate.memory_bytes > state.free_memory_bytes * memory_share:
        memory_mb = round(estimate.memory_bytes / 1_000_000.0)
        free_memory_mb = round(state.free_memory_bytes / 1_000_000.0)
        share_desc = "half" if memory_share == 0.5 else f"{int(round(memory_share * 100))}%"
        problems.append(
            f"memory: needs about {memory_mb:,} MB, over {share_desc} of the {free_memory_mb:,} MB free"
        )

    # 3. Check projected events file storage against allowable share of available partition disk space.
    disk_share = float(limits.get("disk_share", 0.5))
    if estimate.file_bytes > state.free_disk_bytes * disk_share:
        disk_mb = round(estimate.file_bytes / 1_000_000.0)
        free_disk_mb = round(state.free_disk_bytes / 1_000_000.0)
        share_desc = "half" if disk_share == 0.5 else f"{int(round(disk_share * 100))}%"
        problems.append(f"disk: needs about {disk_mb:,} MB, over {share_desc} of the {free_disk_mb:,} MB free")

    # Pass back list of identified resource bottlenecks.
    return problems
