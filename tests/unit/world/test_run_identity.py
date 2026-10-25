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
import subprocess
from collections.abc import Sequence
from pathlib import Path

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


class StandinGit:
    """
    WHAT: A test double callable for git execution that returns predetermined command outputs or raises exceptions.
    Records every invoked command's argument list so callers can verify invocation counts and parameters.

    WHY: Decouples code version tests from real OS git processes and network drives, allowing instant, deterministic
    simulation of timeouts, clean working trees, and dirty working trees without touching disk state.

    Args:
        responses: Sequence of string outputs to return in order, or an Exception to raise when invoked.

    Returns:
        A StandinGit instance ready to be passed to find_code_version as run_git.

    Example:
        standin = StandinGit(["abc1234", ""]); find_code_version({}, REPO_ROOT, 5, run_git=standin) returns "abc1234".
    """

    def __init__(self, responses: Sequence[str] | Exception) -> None:
        """
        WHAT: Initializes the stand-in git runner with preset responses or an exception to raise upon invocation.

        WHY: Captures whether the test simulates successful sequential git queries or an abrupt process timeout.

        Args:
            responses: Preset outputs to yield on successive calls, or an exception to raise on call.

        Returns:
            None.

        Example:
            standin = StandinGit(["abc1234", ""])
        """
        # Store preset responses and initialize empty history of command arguments.
        self._responses: list[str] = list(responses) if isinstance(responses, Sequence) else []
        self._exception: Exception | None = responses if isinstance(responses, Exception) else None
        self.calls: list[list[str]] = []

    def __call__(self, executable: str, cwd: Path, args: Sequence[str], timeout: float) -> str:
        """
        WHAT: Records the invoked command arguments and either raises the configured exception
        or returns the next response.

        WHY: Implements the Callable[[str, Path, Sequence[str], float], str] protocol expected by find_code_version.

        Args:
            executable: Path or name of the git executable.
            cwd: Working directory where git command is executed.
            args: Sub-command arguments passed to git.
            timeout: Timeout duration in seconds.

        Returns:
            The next scheduled string output for the command.

        Example:
            standin("git", Path("."), ["rev-parse", "HEAD"], 5.0)
        """
        # Record arguments for test assertions.
        self.calls.append(list(args))
        if self._exception is not None:
            raise self._exception
        return self._responses.pop(0)


def test_git_timeout_gives_unknown() -> None:
    """
    WHAT: Injects a stand-in git callable that raises subprocess.TimeoutExpired and verifies fallback to "unknown".
    Asserts that the stand-in was called exactly once and that no exception escapes.

    WHY: When git hangs (for example on a broken network drive or busy lock), the world run must not hang or fail.
    The timeout ends the wait and find_code_version falls back cleanly to the configured "unknown" version.

    Args:
        None.

    Returns:
        None. The test fails if anything other than "unknown" is returned or the stand-in is called unexpectedly.

    Example:
        Passing a stand-in run_git that times out causes find_code_version({}, REPO_ROOT, 5) to return "unknown".
    """
    # Create stand-in git runner configured to raise TimeoutExpired on the first call.
    timeout_git = StandinGit(subprocess.TimeoutExpired(cmd="git", timeout=5.0))
    config = load_runner_config()
    version = find_code_version({}, REPO_ROOT, 5.0, run_git=timeout_git)
    assert version == config["unknown_code_version"]
    assert len(timeout_git.calls) == 1


def test_injected_git_clean_working_tree_gives_commit() -> None:
    """
    WHAT: Injects a stand-in git callable that returns commit "abc1234" for rev-parse and empty string "" for status,
    verifying that find_code_version returns exactly "abc1234".

    WHY: Confirms that when the working tree is clean (no modified or untracked files), find_code_version returns
    the exact commit hash without appending a "-dirty" suffix.

    Args:
        None.

    Returns:
        None. The test fails if the returned version is not "abc1234".

    Example:
        A stand-in returning "abc1234" and "" causes find_code_version({}, REPO_ROOT, 5) to return "abc1234".
    """
    # Stand-in yields "abc1234" on rev-parse and empty string on porcelain status.
    clean_git = StandinGit(["abc1234", ""])
    version = find_code_version({}, REPO_ROOT, 5.0, run_git=clean_git)
    assert version == "abc1234"
    assert len(clean_git.calls) == 2


def test_injected_git_dirty_working_tree_gives_commit_dirty() -> None:
    """
    WHAT: Injects a stand-in git callable that returns commit "abc1234" for rev-parse and " M world/x.py" for status,
    verifying that find_code_version appends the "-dirty" suffix to produce "abc1234-dirty".

    WHY: Confirms that unsaved changes in the repository are properly flagged with the "-dirty" suffix so that runs
    built with uncommitted modifications are distinguished from pure commits.

    Args:
        None.

    Returns:
        None. The test fails if the returned version is not "abc1234-dirty".

    Example:
        A stand-in returning "abc1234" and " M world/x.py" returns "abc1234-dirty".
    """
    # Stand-in yields "abc1234" on rev-parse and non-empty status on porcelain status.
    dirty_git = StandinGit(["abc1234", " M world/x.py"])
    version = find_code_version({}, REPO_ROOT, 5.0, run_git=dirty_git)
    assert version == "abc1234-dirty"
    assert len(dirty_git.calls) == 2


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
