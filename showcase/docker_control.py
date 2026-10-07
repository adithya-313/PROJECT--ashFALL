# File: showcase/docker_control.py
# Purpose: Stop and start the Redpanda broker from a showcase (to prove that a stopped broker loses nothing), and
#          wait until it answers again. Uses the docker command line through subprocess, with timeouts.
import asyncio
import subprocess
import time
from pathlib import Path

from aiokafka.admin import AIOKafkaAdminClient

from core.config import config

REPO_ROOT = Path(__file__).resolve().parent.parent
BROKER_SERVICE = "redpanda"
COMMAND_TIMEOUT_SECONDS = 60
# One try of "is the broker answering yet?" may take at most this long.
PROBE_TIMEOUT_SECONDS = 5.0


def run_docker_command(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    """Run one docker compose command from the repo root (so it finds docker-compose.yml), with a timeout."""
    return subprocess.run(
        ["docker", "compose", *arguments],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=COMMAND_TIMEOUT_SECONDS,
    )


def run_or_raise(arguments: list[str]) -> None:
    """Run a docker compose command and raise RuntimeError with the short error text if it did not succeed."""
    result = run_docker_command(arguments)
    if result.returncode != 0:
        short_error = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "no error text"
        raise RuntimeError(f"docker compose {' '.join(arguments)} failed: {short_error}")


def stop_broker() -> None:
    """Stop the Redpanda broker container."""
    run_or_raise(["stop", BROKER_SERVICE])


def start_broker() -> None:
    """Start the Redpanda broker container again."""
    run_or_raise(["start", BROKER_SERVICE])


def docker_available() -> str | None:
    """Return None if docker works and the broker container exists, otherwise a plain-English problem."""
    try:
        result = run_docker_command(["ps", "-q", BROKER_SERVICE])
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"docker is not usable ({type(error).__name__}) - install or start Docker Desktop"
    if result.returncode != 0:
        return "docker compose could not list services - run this from the project folder with Docker running"
    if not result.stdout.strip():
        return "the redpanda container does not exist - run: docker compose up -d redpanda"
    return None


async def broker_answers() -> bool:
    """One try: can we connect and list topics within a few seconds?"""
    admin = AIOKafkaAdminClient(bootstrap_servers=config.kafka_bootstrap_servers)
    try:
        await asyncio.wait_for(admin.start(), PROBE_TIMEOUT_SECONDS)
        await asyncio.wait_for(admin.list_topics(), PROBE_TIMEOUT_SECONDS)
        return True
    except Exception:  # noqa: BLE001 - any failure just means "not ready yet"
        return False
    finally:
        try:
            await asyncio.wait_for(admin.close(), PROBE_TIMEOUT_SECONDS)
        except Exception:  # noqa: BLE001
            pass


async def wait_for_broker(timeout_seconds: float = 90) -> float:
    """Poll once a second until the broker answers. Returns the seconds waited, or raises TimeoutError."""
    started = time.monotonic()
    while True:
        if await broker_answers():
            return time.monotonic() - started
        if time.monotonic() - started >= timeout_seconds:
            raise TimeoutError(f"the Kafka broker did not answer within {timeout_seconds:.0f} seconds")
        await asyncio.sleep(1)
