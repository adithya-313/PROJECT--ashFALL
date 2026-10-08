"""
File: tests/unit/test_level08_helpers.py
Purpose: Proves the Docker-free helpers of the Level 8 showcase: the compose guard refuses any project but the throwaway
         one, the service choice, the safety check on the network, and the three parsers. No Docker is needed.
"""

import pytest

from showcase.level08_helpers import (
    NETWORK_NAME,
    PROJECT_NAME,
    compose_command,
    count_dns_failures,
    count_outside_packets,
    network_problem,
    owns_name,
    parse_out_no_routes,
    services_for_mode,
)

SNMP_SAMPLE = (
    "Ip: Forwarding DefaultTTL InReceives OutRequests OutNoRoutes ReasmTimeout\n"
    "Ip: 1 64 100 50 7 0\n"
    "Icmp: InMsgs\n"
    "Icmp: 3\n"
)


def test_compose_guard_refuses_any_other_project_name() -> None:
    """Only the throwaway project name is accepted, so a stop or remove can never target the real stack."""
    for wrong in ("project-ashfall", "ashfall", "", "ashfall-noegress-2"):
        with pytest.raises(ValueError, match="refusing"):
            compose_command(["down"], project=wrong)


def test_compose_command_always_names_the_project_and_the_override() -> None:
    """The built command uses the override file, the example env file and the throwaway project name."""
    command = compose_command(["ps"])
    assert command[:2] == ["docker", "compose"]
    assert command[command.index("-p") + 1] == PROJECT_NAME
    assert "showcase/fixtures/level_08/compose.noegress.yml" in command
    assert command[command.index("--env-file") + 1] == ".env.example"
    assert command[-1] == "ps"


def test_services_for_mode() -> None:
    """Quick mode starts three services; full mode starts everything except triton."""
    everything = ["api", "postgres", "triton", "redis", "redpanda", "outbox-relay"]
    assert services_for_mode(True, everything) == ["postgres", "redis", "redpanda"]
    assert services_for_mode(False, everything) == ["api", "postgres", "redis", "redpanda", "outbox-relay"]


def test_owns_name_only_matches_the_copy() -> None:
    """Cleanup may only touch names that start with the copy's prefix."""
    assert owns_name("ashfall-noegress-controlA")
    assert not owns_name("project-ashfall-api-1")


def test_network_check_catches_an_open_network_or_a_second_network() -> None:
    """A non-internal network, or a container on another network too, is reported; a closed copy is not."""
    closed = {"api-1": [NETWORK_NAME], "db-1": [NETWORK_NAME]}
    assert network_problem(True, closed) is None
    assert "not internal" in str(network_problem(False, closed))
    assert "api-1" in str(network_problem(True, {"api-1": [NETWORK_NAME, "bridge"]}))
    assert "no containers" in str(network_problem(True, {}))


def test_dns_failures_are_counted() -> None:
    """ServFail and NXDomain replies are counted, and an empty capture counts zero."""
    text = "12:00:00.1 lo In IP 127.0.0.11.53 > 127.0.0.1.4000: 1 ServFail- 0/0/0 (29)\n12:00:00.2 x NXDomain 0/0/0\n"
    assert count_dns_failures(text) == 2
    assert count_dns_failures("") == 0


def test_outside_packets_are_counted_by_their_timestamp_lines() -> None:
    """Only lines that start with a time count as packets; warnings and blank lines do not."""
    text = (
        "tcpdump: listening on any
12:00:00.123456 eth0 Out IP 1.2.3.4 > 5.6.7.8: Flags [S]

"
        "12:00:01.5 eth0 In IP6 x
"
    )
    assert count_outside_packets(text) == 2
    assert count_outside_packets("") == 0


def test_out_no_routes_is_found_by_column_name() -> None:
    """OutNoRoutes is read from the right column, and None is returned when the text is unusable."""
    assert parse_out_no_routes(SNMP_SAMPLE) == 7
    reordered = "Ip: OutNoRoutes Forwarding\nIp: 3 1\n"
    assert parse_out_no_routes(reordered) == 3
    assert parse_out_no_routes("") is None
    assert parse_out_no_routes("Ip: Forwarding\nIp: 1\n") is None
