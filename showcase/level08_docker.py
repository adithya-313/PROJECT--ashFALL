# File: showcase/level08_docker.py
# Purpose: The Docker side of the Level 8 "no outgoing network" check. It starts a throwaway COPY of the stack (project
#          name ashfall-noegress, on an internal network, no published ports), watches every container's network with a
#          tcpdump sidecar, runs two control experiments that prove the watching works, and removes everything again.
#
# Safety: every compose command comes from compose_command() in level08_helpers.py, which refuses any other project
# name. Other containers and networks are only removed one by one, and only if their name starts with
# "ashfall-noegress".
import json
import subprocess
import time
from pathlib import Path

from showcase.level08_helpers import (
    NETWORK_NAME,
    OBJECT_PREFIX,
    PROJECT_NAME,
    SNIFFER_IMAGE,
    SniffResult,
    compose_command,
    count_dns_failures,
    count_outside_packets,
    network_problem,
    owns_name,
    parse_out_no_routes,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = "showcase/fixtures/level_08"
DOCKER_TIMEOUT = 60
UP_TIMEOUT = 330
OPEN_NETWORK = f"{OBJECT_PREFIX}-open"

# What the sniffer sidecar runs. It reads the kernel's counters before and after, and runs two tcpdump captures at once:
# (1) packets that leave the container's own network (outside the internal subnet and not loopback; IPv6 multicast
# housekeeping such as neighbour discovery is ignored), (2) DNS traffic on the loopback interface (Docker's own DNS).
SNIFF_SCRIPT = (
    "cat /proc/net/snmp > /tmp/before; "
    "timeout {seconds} tcpdump -i any -nn -l "
    "'(ip and not net {subnet} and not net 127.0.0.0/8) or (ip6 and not ip6 multicast)' > /tmp/outside.txt 2>&1 & "
    "timeout {seconds} tcpdump -i lo -nn -l 'udp port 53' > /tmp/dns.txt 2>&1 & "
    "wait; cat /proc/net/snmp > /tmp/after; "
    "echo '=== BEFORE'; cat /tmp/before; echo '=== OUTSIDE'; cat /tmp/outside.txt; echo '=== DNS'; cat /tmp/dns.txt; "
    "echo '=== AFTER'; cat /tmp/after; echo '=== END'"
)


Completed = subprocess.CompletedProcess[str]


def run(command: list[str], timeout: int = DOCKER_TIMEOUT, text_input: str | None = None) -> Completed:
    """Run a command from the repo root with a timeout, capturing its output."""
    return subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout, input=text_input)


def run_compose(args: list[str], timeout: int = DOCKER_TIMEOUT, text_input: str | None = None) -> Completed:
    """Run a compose command for the throwaway copy (the guard in compose_command checks the project name)."""
    return run(compose_command(args), timeout, text_input)


def list_services() -> list[str]:
    """Every service name in the compose file (a read-only question)."""
    return run_compose(["config", "--services"]).stdout.split()


def build_sniffer_image() -> str | None:
    """Build the tcpdump image. Returns a problem text, or None when it is ready."""
    dockerfile = f"{FIXTURE_DIR}/sniffer.Dockerfile"
    result = run(["docker", "build", "-q", "-f", dockerfile, "-t", SNIFFER_IMAGE, FIXTURE_DIR], 300)
    return None if result.returncode == 0 else result.stderr.strip()[-300:]


def names_of(command: list[str]) -> list[str]:
    """Names printed by a `docker ... --format {{.Names}}`-style command, only those that belong to the copy."""
    return [name for name in run(command).stdout.split() if owns_name(name)]


def cleanup_copy() -> None:
    """
    Remove everything that belongs to the copy: the compose project, then leftover sidecars, controls and networks.

    Safe to run twice and safe when nothing exists. Objects are removed one by one and only if their name starts with
    the copy's prefix, so nothing from the real stack can be touched.
    """
    run_compose(["down", "-v", "--remove-orphans"], UP_TIMEOUT)
    for container in names_of(["docker", "ps", "-a", "--format", "{{.Names}}"]):
        run(["docker", "rm", "-f", container])
    for network in names_of(["docker", "network", "ls", "--format", "{{.Name}}"]):
        run(["docker", "network", "rm", network])


def start_copy(services: list[str]) -> str | None:
    """Start the chosen services and wait for their health checks. Returns a problem text, or None on success."""
    result = run_compose(["up", "-d", "--wait", "--wait-timeout", "240", *services], UP_TIMEOUT)
    return None if result.returncode == 0 else (result.stderr.strip()[-600:] or "docker compose up failed")


def project_containers() -> list[str]:
    """Names of all containers of the copy (running or not)."""
    label = f"label=com.docker.compose.project={PROJECT_NAME}"
    return run(["docker", "ps", "-a", "--filter", label, "--format", "{{.Names}}"]).stdout.split()


def networks_of(container: str) -> list[str]:
    """The names of the networks a container is attached to."""
    result = run(["docker", "inspect", "-f", "{{json .NetworkSettings.Networks}}", container])
    return sorted(json.loads(result.stdout or "{}"))


def verify_network_is_closed() -> str | None:
    """A problem text when egress could be open (network not internal, or a container on a second network)."""
    inspect = run(["docker", "network", "inspect", "-f", "{{.Internal}}", NETWORK_NAME])
    internal = inspect.stdout.strip() == "true"
    return network_problem(internal, {name: networks_of(name) for name in project_containers()})


def network_subnet(network: str) -> str:
    """The IPv4 subnet of a Docker network."""
    result = run(["docker", "network", "inspect", "-f", "{{range .IPAM.Config}}{{.Subnet}} {{end}}", network])
    return next((part for part in result.stdout.split() if ":" not in part), "")


def read_env_value(name: str) -> str:
    """A value from .env.example (the copy's settings), such as POSTGRES_USER."""
    for line in (REPO_ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{name}="):
            return line.split("=", 1)[1].strip()
    return ""


def apply_migrations() -> str | None:
    """Apply every migration file inside the copy's Postgres (SQL is piped in; no host port is used)."""
    user, database = read_env_value("POSTGRES_USER"), read_env_value("POSTGRES_DB")
    for path in sorted((REPO_ROOT / "migrations").glob("*.sql")):
        args = ["exec", "-T", "postgres", "psql", "-q", "-U", user, "-d", database, "-v", "ON_ERROR_STOP=1", "-f", "-"]
        result = run_compose(args, DOCKER_TIMEOUT, path.read_text(encoding="utf-8"))
        if result.returncode != 0:
            return f"migration {path.name} failed: {result.stderr.strip()[-300:]}"
    return None


def container_state(container: str) -> dict[str, str]:
    """State, restart count and health of one container."""
    fmt = "{{.State.Status}}|{{.RestartCount}}|{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}"
    status, restarts, health = run(["docker", "inspect", "-f", fmt, container]).stdout.strip().split("|")
    return {"container": container, "status": status, "restarts": restarts, "health": health}


def start_sniffer(target: str, seconds: int, subnet: str) -> str:
    """Start a tcpdump sidecar that shares the target's network space. Returns the sidecar's name."""
    sidecar = f"{OBJECT_PREFIX}-sniff-{target}"
    script = SNIFF_SCRIPT.format(seconds=seconds, subnet=subnet)
    network = f"container:{target}"
    run(["docker", "run", "-d", "--name", sidecar, "--network", network, SNIFFER_IMAGE, "sh", "-c", script])
    return sidecar


def section(output: str, name: str) -> str:
    """The text between '=== NAME' and the next '=== ' marker."""
    parts = output.split(f"=== {name}\n", 1)
    return parts[1].split("=== ", 1)[0] if len(parts) == 2 else ""


def collect_sniffer(sidecar: str, target: str, seconds: int) -> SniffResult:
    """Wait for the sidecar to finish, read its numbers, and remove it."""
    run(["docker", "wait", sidecar], seconds + 60)
    output = run(["docker", "logs", sidecar]).stdout
    run(["docker", "rm", "-f", sidecar])
    before = parse_out_no_routes(section(output, "BEFORE"))
    after = parse_out_no_routes(section(output, "AFTER"))
    complete = "=== END" in output and before is not None and after is not None
    delta = (after - before) if complete and before is not None and after is not None else 0
    return SniffResult(target, count_outside_packets(section(output, "OUTSIDE")),
                       count_dns_failures(section(output, "DNS")), delta, complete)


def observe_copy(seconds: int) -> list[SniffResult]:
    """Watch every container of the copy for `seconds` at the same time."""
    subnet = network_subnet(NETWORK_NAME)
    targets = project_containers()
    sidecars = {target: start_sniffer(target, seconds, subnet) for target in targets}
    return [collect_sniffer(sidecar, target, seconds) for target, sidecar in sidecars.items()]


def run_control(network: str, name: str, commands: str, seconds: int = 12) -> SniffResult:
    """Run deliberate outside attempts from a throwaway container on `network` while a sidecar watches it."""
    run(["docker", "run", "-d", "--name", name, "--network", network, "alpine", "sleep", "120"])
    try:
        sidecar = start_sniffer(name, seconds, network_subnet(network))
        time.sleep(3)  # let tcpdump start before the attempts
        run(["docker", "exec", name, "sh", "-c", commands])
        return collect_sniffer(sidecar, name, seconds)
    finally:
        run(["docker", "rm", "-f", name])


def run_control_a() -> SniffResult:
    """Control A: on a NORMAL network the packet filter must see the outside attempt (proves it can count)."""
    run(["docker", "network", "create", OPEN_NETWORK])
    try:
        commands = "wget -T 2 -q -O- http://1.1.1.1 >/dev/null 2>&1; true"
        return run_control(OPEN_NETWORK, f"{OBJECT_PREFIX}-controlA", commands)
    finally:
        run(["docker", "network", "rm", OPEN_NETWORK])


def run_control_b() -> SniffResult:
    """Control B: on the copy's internal network an outside lookup must show as a DNS failure."""
    commands = "nslookup example.com >/dev/null 2>&1; wget -T 2 -q -O- http://1.1.1.1 >/dev/null 2>&1; true"
    return run_control(NETWORK_NAME, f"{OBJECT_PREFIX}-controlB", commands)
