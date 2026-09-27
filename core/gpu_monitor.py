"""
File: core/gpu_monitor.py
Purpose: Polls nvidia-smi once per second and updates Prometheus gauges with GPU stats.
Connects: api/main.py starts poll_gpu_metrics() as a background task on app startup.
          Prometheus scrapes the gauges via the /metrics endpoint.
"""

import asyncio
import subprocess
from typing import Optional

from prometheus_client import Gauge

from core.logging import get_logger

logger = get_logger(__name__)

# Prometheus gauges — each one holds the latest reading from nvidia-smi.
# Prometheus scrapes these values whenever it hits /metrics.
gpu_utilization_percent = Gauge(
    "gpu_utilization_percent",
    "GPU compute utilization as a percentage (0-100)",
)
gpu_vram_used_mb = Gauge(
    "gpu_vram_used_mb",
    "GPU VRAM currently in use, in megabytes",
)
gpu_power_draw_watts = Gauge(
    "gpu_power_draw_watts",
    "GPU power draw in watts",
)


def get_gpu_stats() -> Optional[tuple[float, float, float]]:
    """
    Runs nvidia-smi and parses three numbers: utilization, VRAM used, power draw.

    Returns a tuple of (utilization_percent, vram_used_mb, power_watts),
    or None if nvidia-smi is not available or returns unexpected output.
    """
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,power.draw",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=5,  # Don't wait more than 5 seconds for a response
    )

    if result.returncode != 0:
        return None

    # Output is a single comma-separated line like: "45, 2048, 85.5"
    parts = result.stdout.strip().split(",")
    if len(parts) != 3:
        return None

    return (
        float(parts[0].strip()),
        float(parts[1].strip()),
        float(parts[2].strip()),
    )


async def poll_gpu_metrics() -> None:
    """
    Background task that reads GPU stats once per second and updates the Prometheus gauges.

    If nvidia-smi is not available (e.g. on a CI runner or CPU-only machine),
    logs a single warning and exits cleanly — it does not crash the whole application.
    """
    warned_already = False

    while True:
        try:
            stats = get_gpu_stats()
            if stats is not None:
                util, vram, power = stats
                gpu_utilization_percent.set(util)
                gpu_vram_used_mb.set(vram)
                gpu_power_draw_watts.set(power)
        except FileNotFoundError:
            # nvidia-smi binary is not installed on this machine — log once and stop.
            if not warned_already:
                logger.warning("nvidia-smi not found — GPU metrics will not be collected")
                warned_already = True
            return
        except Exception as exc:
            # Any other unexpected error — log once and stop to avoid log spam.
            if not warned_already:
                logger.warning("GPU metrics polling failed", error=str(exc))
                warned_already = True
            return

        # Wait one second before the next poll.
        await asyncio.sleep(1)
