"""
WHAT: Names a world run and every payment in it. A run id is a short fingerprint of three things: the seed, the
full settings the world was built with, and the version of the code. A payment id is that run id plus the
payment's position in the run, for example "66db9aa439b092ee-000042" for the 43rd event.

WHY: Later levels keep many runs side by side (training, validation and test worlds). Two runs with the same seed
but a different setting, or built by different code, must never be mistaken for each other. Folding all three
into the id makes that impossible, and the payment id lets any one event be traced back to the run it came from.

TERM: run id: a 16-character fingerprint naming one run, e.g. "66db9aa439b092ee" (hexadecimal letters 0-9, a-f).
TERM: payment id: a run id, a dash and a zero-padded position, e.g. "66db9aa439b092ee-000000" for the first event.
TERM: code version: which commit of the code made the run, e.g. "d41b176", with "-dirty" added when files had
      unsaved changes, or "unknown" when no answer can be found.
TERM: environment variable: a named setting handed to a program by whoever starts it,
      e.g. ASHFALL_CODE_VERSION=d41b176 set by a container start script.
TERM: canonical JSON: data written with sorted keys and no spaces, so the same data always gives the same text.
TERM: SHA-256: a function that turns any text into a 64-character fingerprint; one changed character changes it.
TERM: timeout: the longest time we wait for another program before giving up, e.g. 5 seconds for git.
"""

import hashlib
import json
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from core.config_validator import load_yaml

# The repo root, found from this file's location so the config loads from any working directory.
REPO_ROOT: Path = Path(__file__).resolve().parent.parent
RUNNER_CONFIG_PATH: Path = REPO_ROOT / "config" / "interactive_runner.yaml"


def load_runner_config(path: Path = RUNNER_CONFIG_PATH) -> dict[str, Any]:
    """
    WHAT: Reads config/interactive_runner.yaml and returns its one block, interactive_runner, as a dictionary.
    The runner, the run-identity helpers and the tests all take their numbers from this one place.

    WHY: Every number the runner uses (which random streams, how long ids are, how long to wait for git) is a
    labelled policy in config, never a literal hidden in code. Reading it through one function keeps that true.

    Args:
        path: Where the config file is; the default is config/interactive_runner.yaml in this repo.

    Returns:
        The interactive_runner block, for example with "run_id_hex_length" set to 16.

    Example:
        >>> load_runner_config()["run_id_hex_length"]
        16
        >>> load_runner_config()["random_stream_children"]["fraud_payment_failures"]
        4
    """
    # The file holds one labelled block; hand back only that block.
    return load_yaml(path)["interactive_runner"]


def _run_git(git_executable: str, repo_root: Path, arguments: Sequence[str], timeout_seconds: float) -> str:
    """
    WHAT: Runs one read-only git command in repo_root and returns what it printed, without the trailing newline.
    It raises if git cannot be started, takes longer than the timeout, or reports an error.

    WHY: Both git questions the code version needs (which commit, and are there unsaved changes) go through the
    same safety rules. Keeping them here means the timeout and the error handling are written exactly once.

    Args:
        git_executable: The program to run, normally "git"; tests pass a missing path to simulate no git.
        repo_root: The folder of the repository to ask about.
        arguments: The git words after the folder, e.g. ["rev-parse", "--short", "HEAD"].
        timeout_seconds: The longest wait before giving up, e.g. 5.

    Returns:
        The command's printed text with surrounding spaces and newlines removed, e.g. "d41b176".

    Example:
        _run_git("git", REPO_ROOT, ["rev-parse", "--short", "HEAD"], 5) returned "d41b176" at that commit.
    """
    # check=True turns a git error (such as "not a repository") into an exception the caller handles.
    completed = subprocess.run(
        [git_executable, "-C", str(repo_root), *arguments],
        capture_output=True, text=True, timeout=timeout_seconds, check=True,
    )
    return completed.stdout.strip()


def find_code_version(
    env: Mapping[str, str],
    repo_root: Path,
    timeout_seconds: float,
    *,
    env_var: str | None = None,
    unknown: str | None = None,
    git_executable: str = "git",
    run_git: Callable[[str, Path, Sequence[str], float], str] = _run_git,
) -> str:
    """
    WHAT: Works out which version of the code is running. First choice is the environment variable named in
    config (ASHFALL_CODE_VERSION); second is git's short commit hash, with "-dirty" when there are unsaved
    changes; last resort is the configured word "unknown". It never raises.

    WHY: The Docker image has no git history, so a container must be told its version from outside. On a laptop
    git knows. When neither works the run still goes ahead and says "unknown" openly, because a wrong guess
    would be worse than an honest gap.

    Args:
        env: The environment to read, normally os.environ; tests pass a small dictionary.
        repo_root: The repository folder to ask git about.
        timeout_seconds: The longest wait for each git command, from config (5 seconds).
        env_var: The environment variable to read; None means the name in config.
        unknown: The word recorded when nothing works; None means the word in config.
        git_executable: The git program to run; tests pass a missing path to simulate a machine without git.
        run_git: The runner for git commands; defaults to _run_git, tests pass a stand-in callable.

    Returns:
        A short version string, e.g. "d41b176", "d41b176-dirty", "build-42" from the variable, or "unknown".

    Example:
        find_code_version({"ASHFALL_CODE_VERSION": "build-42"}, REPO_ROOT, 5) gives "build-42".
        find_code_version({}, REPO_ROOT, 5, git_executable="no-such-git") gives "unknown".
    """
    # Fill in the variable name and the fallback word from config when the caller did not choose them.
    config = load_runner_config()
    variable_name = env_var if env_var is not None else config["code_version_env_var"]
    fallback = unknown if unknown is not None else config["unknown_code_version"]
    # 1. Whoever started the program may tell us the version directly; that answer wins.
    from_environment = env.get(variable_name, "")
    if from_environment:
        return from_environment
    # 2. Ask git for the commit, then whether any tracked or new files have unsaved changes.
    try:
        commit = run_git(git_executable, repo_root, ["rev-parse", "--short", "HEAD"], timeout_seconds)
        changes = run_git(git_executable, repo_root, ["status", "--porcelain"], timeout_seconds)
    except (OSError, subprocess.SubprocessError):
        # 3. No git, too slow, or not a repository: record the honest fallback instead of failing the run.
        return fallback
    if not commit:
        return fallback
    return f"{commit}-dirty" if changes else commit


def make_run_id(seed: int, effective_config: Mapping[str, Any], code_version: str, length: int) -> str:
    """
    WHAT: Builds the run id: the SHA-256 fingerprint of the seed, the full effective settings and the code
    version, written as canonical JSON, cut to its first `length` characters.

    WHY: Any change to any of the three inputs gives a different id, so two different runs can never share a
    name. Canonical JSON makes the id the same on every machine for the same inputs.

    Args:
        seed: The world's master seed, e.g. 101.
        effective_config: Everything the world was built from (settings and the four config files), as plain data.
        code_version: The value from find_code_version, e.g. "d41b176".
        length: How many characters to keep, from config (16).

    Returns:
        A lowercase hexadecimal string of `length` characters.

    Example:
        >>> make_run_id(101, {"n_people": 60}, "d41b176", 16)
        '66db9aa439b092ee'
        >>> make_run_id(102, {"n_people": 60}, "d41b176", 16)
        '909b343fc34b8d17'
    """
    # Put the three inputs in one dictionary and write it in canonical form.
    identity = {"seed": seed, "config": effective_config, "code_version": code_version}
    canonical_text = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    # Fingerprint the text and keep the configured number of characters.
    return hashlib.sha256(canonical_text.encode("utf-8")).hexdigest()[:length]


def make_payment_id(run_id: str, position: int, digits: int) -> str:
    """
    WHAT: Builds one payment id from the run id and a position, padding the position with zeros to `digits`
    characters, for example position 42 with 6 digits becomes "000042".

    WHY: Zero padding keeps ids the same length, so they sort in run order as plain text. The shape matches
    Level 11's live-mode event id, so anyone reading both sees one familiar pattern.

    Args:
        run_id: The run id, e.g. "66db9aa439b092ee".
        position: The zero-based position, e.g. 42.
        digits: The padded width, from config (6).

    Returns:
        The id, e.g. "66db9aa439b092ee-000042".

    Example:
        >>> make_payment_id("66db9aa439b092ee", 42, 6)
        '66db9aa439b092ee-000042'
    """
    # Pad the position on the left with zeros to the configured width.
    return f"{run_id}-{position:0{digits}d}"
