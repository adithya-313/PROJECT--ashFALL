# File: tests/unit/test_showcase_docker_control.py
# Purpose: Proves the docker helpers run the exact commands we expect, raise on failure, and give clear messages.
#          A fake replaces subprocess.run, so no real Docker is touched.
import subprocess
from typing import Any

import pytest

from showcase import docker_control


class FakeRun:
    """A stand-in for subprocess.run that records its commands and returns a prepared result."""

    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "", error: Exception | None = None):
        self.calls: list[list[str]] = []
        self.returncode, self.stdout, self.stderr, self.error = returncode, stdout, stderr, error

    def __call__(self, command: list[str], **options: Any) -> subprocess.CompletedProcess[str]:
        """Record the command, then return the prepared result (or raise the prepared error)."""
        self.calls.append(command)
        if self.error is not None:
            raise self.error
        return subprocess.CompletedProcess(command, self.returncode, self.stdout, self.stderr)


def test_stop_and_start_use_the_exact_commands(monkeypatch):
    """stop_broker and start_broker run 'docker compose stop/start redpanda'."""
    fake = FakeRun()
    monkeypatch.setattr(docker_control.subprocess, "run", fake)
    docker_control.stop_broker()
    docker_control.start_broker()
    assert fake.calls == [["docker", "compose", "stop", "redpanda"], ["docker", "compose", "start", "redpanda"]]


def test_non_zero_return_code_raises_runtime_error_with_short_text(monkeypatch):
    """A failing docker command raises RuntimeError that includes the last line of the error text."""
    fake = FakeRun(returncode=1, stderr="line one\nno such service")
    monkeypatch.setattr(docker_control.subprocess, "run", fake)
    with pytest.raises(RuntimeError, match="no such service"):
        docker_control.stop_broker()


def test_docker_available_returns_none_when_container_exists(monkeypatch):
    """docker_available is None when 'ps -q redpanda' succeeds and prints a container id."""
    fake = FakeRun(stdout="abc123\n")
    monkeypatch.setattr(docker_control.subprocess, "run", fake)
    assert docker_control.docker_available() is None
    assert fake.calls == [["docker", "compose", "ps", "-q", "redpanda"]]


def test_docker_available_explains_each_problem(monkeypatch):
    """Missing docker, a failing command and a missing container each give a different plain message."""
    monkeypatch.setattr(docker_control.subprocess, "run", FakeRun(error=FileNotFoundError("docker")))
    assert "not usable" in str(docker_control.docker_available())
    monkeypatch.setattr(docker_control.subprocess, "run", FakeRun(returncode=1))
    assert "could not list services" in str(docker_control.docker_available())
    monkeypatch.setattr(docker_control.subprocess, "run", FakeRun(stdout=""))
    assert "does not exist" in str(docker_control.docker_available())
