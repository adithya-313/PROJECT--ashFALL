"""
WHAT: Tests for world/run_identity.py: how a run finds its code version, and how run ids and payment ids are built.
Every test is exact; nothing here is random or statistical.

WHY: Level 12 keeps many worlds side by side, and each must carry a name that changes whenever its seed, settings
or code change. The code version must also degrade gracefully: a container with no git must still run and say
"unknown" instead of crashing or guessing.

TERM: code version: which commit of the code made a run, e.g. "d41b176", or "unknown" when it cannot be found.
TERM: environment variable: a named setting handed to a program when it starts, e.g. ASHFALL_CODE_VERSION=build-42.
TERM: run id: a 16-character fingerprint of seed, settings and code version, e.g. "66db9aa439b092ee".
TERM: payment id: a run id, a dash and a 6-digit position, e.g. "66db9aa439b092ee-000042".
TERM: injection: handing a function the thing it should use (here, which git program to run) instead of letting it
      find that thing itself, so a test can swap in a missing program without changing the computer's settings.
TERM: timeout: the longest wait for another program, after which we give up, e.g. 5 seconds for git.
"""

import re

from world.run_identity import (
    REPO_ROOT,
    find_code_version,
    load_runner_config,
    make_payment_id,
    make_run_id,
)

# The shape every payment id must have: 16 hexadecimal characters, a dash, then 6 digits.
PAYMENT_ID_PATTERN = re.compile(r"^[0-9a-f]{16}-\d{6}$")


def test_environment_variable_wins_over_git() -> None:
    """
    WHAT: Sets the configured variable (ASHFALL_CODE_VERSION) to "build-42" and checks that this value is returned
    even though this folder is a git repository that could answer too.

    WHY: Containers have no git history, so whoever starts them must be able to state the version. When they do,
    that statement must win, or a container built from a copied folder could report a misleading commit.

    Args:
        None.

    Returns:
        None. The test fails if git's answer is used instead.

    Example:
        find_code_version({"ASHFALL_CODE_VERSION": "build-42"}, REPO_ROOT, 5) returns "build-42".
    """
    # Hand in an environment that names a version explicitly.
    config = load_runner_config()
    environment = {config["code_version_env_var"]: "build-42"}
    assert find_code_version(environment, REPO_ROOT, config["code_version_git_timeout_seconds"]) == "build-42"


def test_git_answer_is_used_without_environment_variable() -> None:
    """
    WHAT: Passes an empty environment and checks that the version is git's short commit hash, optionally followed
    by "-dirty", for example "d41b176" or "d41b176-dirty".

    WHY: On a developer laptop git knows the commit, so a run made there can be traced back to its exact code.

    Args:
        None.

    Returns:
        None. The test fails if the answer does not look like a short git hash.

    Example:
        find_code_version({}, REPO_ROOT, 5) returned "d41b176-dirty" while this file was unsaved work.
    """
    # With no variable set, git is asked.
    config = load_runner_config()
    version = find_code_version({}, REPO_ROOT, config["code_version_git_timeout_seconds"])
    assert re.fullmatch(r"[0-9a-f]{7,40}(-dirty)?", version), f"unexpected version text: {version!r}"


def test_missing_git_gives_unknown() -> None:
    """
    WHAT: Injects a git program path that does not exist and checks the configured word "unknown" comes back,
    with no exception raised.

    WHY: A machine without git must still run worlds. The run records "unknown" openly so readers know the version
    is missing. Injecting the path tests this without changing the computer's PATH for any other test.

    Args:
        None.

    Returns:
        None. The test fails if anything other than "unknown" is returned or an exception escapes.

    Example:
        find_code_version({}, REPO_ROOT, 5, git_executable="no-such-git-program") returns "unknown".
    """
    # Point at a program that cannot exist, so starting it fails.
    config = load_runner_config()
    version = find_code_version(
        {}, REPO_ROOT, config["code_version_git_timeout_seconds"], git_executable="no-such-git-program"
    )
    assert version == config["unknown_code_version"]


def test_git_timeout_gives_unknown() -> None:
    """
    WHAT: Asks git with a timeout of one microsecond, which no program can meet, and checks "unknown" comes back.

    WHY: A hung git (for example on a broken network drive) must not hang the world run. The timeout ends the wait
    and the run falls back to "unknown" instead of failing.

    Args:
        None.

    Returns:
        None. The test fails if the timeout does not lead to "unknown".

    Example:
        find_code_version({}, REPO_ROOT, 0.000001) returns "unknown".
    """
    # One microsecond is far shorter than any program needs to start.
    config = load_runner_config()
    assert find_code_version({}, REPO_ROOT, 0.000001) == config["unknown_code_version"]


def test_run_id_changes_when_any_input_changes() -> None:
    """
    WHAT: Builds a run id for seed 101, a small config and version "d41b176", then changes one input at a time:
    the seed to 102, one config value from 60 to 61 people, and the version to "d41b176-dirty". Each change must
    give a different id, and the same inputs must give the same id again.

    WHY: The id exists to keep different runs apart. If any input could change without changing the id, two
    different worlds could be stored under one name.

    Args:
        None.

    Returns:
        None. The test fails if a change leaves the id unchanged, or equal inputs give different ids.

    Example:
        make_run_id(101, {"n_people": 60}, "d41b176", 16) is "66db9aa439b092ee"; with seed 102 it is
        "909b343fc34b8d17".
    """
    # The id length comes from config.
    length = load_runner_config()["run_id_hex_length"]
    base = make_run_id(101, {"n_people": 60}, "d41b176", length)
    # Same inputs, same id.
    assert make_run_id(101, {"n_people": 60}, "d41b176", length) == base
    # Each single change gives a new id.
    variants = [
        make_run_id(102, {"n_people": 60}, "d41b176", length),
        make_run_id(101, {"n_people": 61}, "d41b176", length),
        make_run_id(101, {"n_people": 60}, "d41b176-dirty", length),
    ]
    assert all(variant != base for variant in variants)
    assert len(base) == length and base == "66db9aa439b092ee"


def test_payment_ids_are_unique_and_well_formed() -> None:
    """
    WHAT: Builds payment ids for positions 0 to 2914 (the 2,915 events of the tiny world) and checks every id has
    the shape 16 hex characters, a dash, 6 digits, and that no two ids are the same.

    WHY: Downstream levels look events up by payment id. A duplicate would merge two payments; a wrong shape would
    break readers that split the id at the dash.

    Args:
        None.

    Returns:
        None. The test fails on a duplicate or a malformed id.

    Example:
        make_payment_id("66db9aa439b092ee", 42, 6) is "66db9aa439b092ee-000042".
    """
    # Use the configured lengths and one real-looking run id.
    config = load_runner_config()
    run_id = make_run_id(101, {"n_people": 60}, "d41b176", config["run_id_hex_length"])
    payment_ids = [make_payment_id(run_id, position, config["position_digits"]) for position in range(2915)]
    # Every id has the expected shape, and all 2,915 are different.
    assert all(PAYMENT_ID_PATTERN.match(payment_id) for payment_id in payment_ids)
    assert len(set(payment_ids)) == 2915
    assert payment_ids[42] == "66db9aa439b092ee-000042"
