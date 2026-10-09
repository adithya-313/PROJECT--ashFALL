# File: showcase/level08_helpers.py
# Purpose: The small, Docker-free building blocks of the Level 8 "no outgoing network" check: the command guard that
#          makes sure we only ever touch our own throwaway copy of the stack, which services to start, and the functions
#          that read the numbers out of the sniffer's text output. Everything that talks to Docker lives in
#          level08_docker.py.
import re
from dataclasses import dataclass

# The ONLY compose project this code may start, stop or remove. Your normal stack uses a different project name.
PROJECT_NAME = "ashfall-noegress"
OVERRIDE_FILE = "showcase/fixtures/level_08/compose.noegress.yml"
ENV_FILE = ".env.example"  # the copy never uses the real .env
SNIFFER_IMAGE = "ashfall-noegress-sniffer"
NETWORK_NAME = f"{PROJECT_NAME}_nonet"
OBJECT_PREFIX = "ashfall-noegress"  # every container or network we create outside compose starts with this

QUICK_SERVICES = ["postgres", "redis", "redpanda"]
NEVER_STARTED = ["triton"]  # needs a GPU image; it has profiles and is never part of the copy


def compose_command(args: list[str], project: str = PROJECT_NAME) -> list[str]:
    """
    Build a `docker compose` command for the throwaway copy. Refuses any other project name.

    This is the only place that builds compose commands for the copy, so a typo can never point a stop or remove at
    the real stack.
    """
    if project != PROJECT_NAME:
        raise ValueError(f"refusing to run compose for project '{project}': only '{PROJECT_NAME}' is allowed")
    return ["docker", "compose", "-f", "docker-compose.yml", "-f", OVERRIDE_FILE, "--env-file", ENV_FILE,
            "-p", project, *args]


def services_for_mode(quick: bool, all_services: list[str]) -> list[str]:
    """The services to start: three in quick mode, otherwise all of them except the ones that are never started."""
    if quick:
        return list(QUICK_SERVICES)
    return [name for name in all_services if name not in NEVER_STARTED]


def owns_name(name: str) -> bool:
    """True only for names that belong to the copy (so cleanup never removes anything else)."""
    return name.startswith(OBJECT_PREFIX)


def network_problem(network_is_internal: bool, networks_by_container: dict[str, list[str]]) -> str | None:
    """
    Return a plain-English problem if egress could be open, or None when the copy is closed.

    Closed means: the network is internal, and every container is attached to that network and to nothing else.
    """
    if not network_is_internal:
        return f"network {NETWORK_NAME} is not internal, so egress could be open"
    if not networks_by_container:
        return "no containers were found in the copy"
    for container, networks in networks_by_container.items():
        if networks != [NETWORK_NAME]:
            return f"container {container} is on {networks}, not only on {NETWORK_NAME}: egress could be open"
    return None


def count_dns_failures(text: str) -> int:
    """How many DNS replies in tcpdump text are failures (ServFail or NXDomain)."""
    return len(re.findall(r"servfail|nxdomain", text, re.IGNORECASE))


def count_outside_packets(text: str) -> int:
    """How many captured packets are in tcpdump text (lines that start with a time such as 12:34:56.789)."""
    return len(re.findall(r"^\d{2}:\d{2}:\d{2}\.\d+ ", text, re.MULTILINE))


def parse_out_no_routes(snmp_text: str) -> int | None:
    """
    Read the OutNoRoutes number from the text of /proc/net/snmp.

    The file has a line of column names starting with "Ip:" followed by a line of values starting with "Ip:".
    The column is found by name, so a changed column order cannot break it. None means it could not be read.
    """
    ip_lines = [line.split() for line in snmp_text.splitlines() if line.startswith("Ip:")]
    if len(ip_lines) < 2 or "OutNoRoutes" not in ip_lines[0]:
        return None
    position = ip_lines[0].index("OutNoRoutes")
    if position >= len(ip_lines[1]):
        return None
    return int(ip_lines[1][position])


@dataclass
class SniffResult:
    """What one sniffer saw on one container's network."""

    container: str
    outside_packets: int
    dns_failures: int
    out_no_routes_delta: int
    complete: bool  # False when the capture did not run to the end
