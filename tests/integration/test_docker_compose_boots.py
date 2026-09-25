"""
File: tests/integration/test_docker_compose_boots.py
Purpose: Integration test to verify that the full multi-service Docker Compose topology boots cleanly.
Connects: Interacts directly with the local Docker daemon to validate docker-compose.yml.

EXECUTION SAFETY & CI/LOCAL CONTEXT APPLICABILITY:
  - Shell-out mechanism: This test explicitly executes `docker compose up -d` and
    `docker compose ps` via Python's `subprocess.run`.
  - Local workstation: Safe to run on local developer machines where Docker Desktop
    or Docker Engine is installed, running, and accessible.
  - CI environment: Safe only in CI environments with Docker daemon access (e.g.
    GitHub Actions Ubuntu runners or runners with /var/run/docker.sock mounted).
    It will fail in rootless, sandboxed, or DinD-restricted runner environments
    lacking Docker CLI access.
"""

import json
import shutil
import subprocess
import time
import pytest


def test_docker_compose_boots_all_services() -> None:
    """
    Executes 'docker compose up -d', waits 10 seconds for initial container health checks,
    and asserts that all five expected services (api, postgres, redis, prometheus, grafana)
    are present and reporting a running state.
    """
    # Verify Docker CLI is available on PATH before attempting execution
    docker_bin = shutil.which("docker")
    if not docker_bin:
        pytest.skip("Docker CLI not found on host PATH — cannot run integration test")

    # Boot all services in detached mode
    up_cmd = subprocess.run(
        ["docker", "compose", "up", "-d"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert up_cmd.returncode == 0, f"docker compose up -d failed: {up_cmd.stderr or up_cmd.stdout}"

    # Wait 10 seconds as specified by the test contract
    time.sleep(10)

    # Query service status via docker compose ps (JSON format)
    ps_cmd = subprocess.run(
        ["docker", "compose", "ps", "--format", "json"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert ps_cmd.returncode == 0, f"docker compose ps failed: {ps_cmd.stderr or ps_cmd.stdout}"

    stdout = ps_cmd.stdout.strip()
    assert stdout, "docker compose ps returned empty output"

    # Docker Compose JSON output can be either a JSON array or newline-delimited JSON objects
    services_state: dict[str, str] = {}
    try:
        parsed = json.loads(stdout)
        if isinstance(parsed, list):
            for item in parsed:
                service_name = item.get("Service") or item.get("Name", "")
                services_state[service_name] = item.get("State", "").lower()
        elif isinstance(parsed, dict):
            service_name = parsed.get("Service") or parsed.get("Name", "")
            services_state[service_name] = parsed.get("State", "").lower()
    except json.JSONDecodeError:
        # Fallback to newline-delimited JSON lines
        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            service_name = item.get("Service") or item.get("Name", "")
            services_state[service_name] = item.get("State", "").lower()

    # The five services defined in Level 0 docker-compose.yml
    expected_services = {"api", "postgres", "redis", "prometheus", "grafana"}

    for expected in expected_services:
        assert expected in services_state, (
            f"Expected service '{expected}' not found in docker compose ps output. "
            f"Discovered services: {list(services_state.keys())}"
        )
        assert services_state[expected] == "running", (
            f"Service '{expected}' is not running! Current state: '{services_state[expected]}'"
        )
