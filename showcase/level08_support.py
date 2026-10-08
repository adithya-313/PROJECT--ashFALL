# File: showcase/level08_support.py
# Purpose: Helpers for checks 1, 2 and 3 of the Level 8 showcase (no Docker needed): running commands with a timeout,
#          making a temporary copy of the tracked files, changing one response field in that copy, and a small DVC
#          round trip with generated files.
import hashlib
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# Only these tracked folders and files are copied for check 2 (enough to import the API and run the baseline tests).
COPY_FOLDERS = ["api", "core", "features", "scripts", "tests", "reports/model", "reports/calibration"]
COPY_FILES = ["docker-compose.yml", "pyproject.toml", ".env.example"]
RESPONSE_FILE = "api/routers/scoring.py"
DVC_TIMEOUT = 180


def run_command(
    command: list[str], cwd: Path, timeout: int, extra_env: dict[str, str] | None = None,
) -> tuple[int, str]:
    """Run a command with a timeout and return (exit code, combined output). A timeout counts as exit code 124."""
    environment = dict(os.environ)
    environment.update(extra_env or {})
    try:
        result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=timeout, env=environment)
    except subprocess.TimeoutExpired:
        return 124, f"timed out after {timeout} s: {' '.join(command[:4])}"
    return result.returncode, result.stdout + result.stderr


def last_lines(text: str, count: int = 15) -> str:
    """The last few lines of some output."""
    return "\n".join(text.strip().splitlines()[-count:])


def hide(text: str, secret: str) -> str:
    """Replace a private value (such as the local DVC remote folder) with a placeholder before text is saved."""
    return text.replace(secret, "<local remote>") if secret else text


def count_differences(output: str) -> int:
    """The number from 'changed (N difference(s))' in a baseline failure message, or 0."""
    match = re.search(r"\((\d+) difference", output)
    return int(match.group(1)) if match else 0


def parse_check_summary(output: str) -> tuple[int, int]:
    """From the table printed by scripts/check.py: (steps that passed, steps that ran). SKIPPED is not counted."""
    passed = len(re.findall(r"^PASS\b", output, re.MULTILINE))
    failed = len(re.findall(r"^FAIL\b", output, re.MULTILINE))
    return passed, passed + failed


def rename_field_in_text(text: str) -> str | None:
    """
    Rename the field `model_version` inside the class ScoreResponse (only the field line in the class definition).

    Returns the changed text, or None if the class or the field line is not found exactly once.
    """
    start = text.find("class ScoreResponse")
    if start == -1:
        return None
    end = text.find("\nclass ", start + 1)
    block = text[start:] if end == -1 else text[start:end]
    field_line = re.compile(r"^    model_version: str\b", re.MULTILINE)
    if len(field_line.findall(block)) != 1:
        return None
    changed = field_line.sub("    model_version_renamed: str", block, count=1)
    return text[:start] + changed + text[start + len(block):]


def sha256_of(path: Path) -> str:
    """SHA-256 of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tracked_files() -> list[str]:
    """Paths of the files Git tracks, from `git ls-files`."""
    result = subprocess.run(["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=60)
    return result.stdout.splitlines()


def copy_tracked_files(destination: Path) -> int:
    """Copy the chosen tracked files into `destination` (plus .env.example as .env). Returns how many were copied."""
    copied = 0
    for name in tracked_files():
        wanted = name in COPY_FILES or any(name.startswith(folder + "/") for folder in COPY_FOLDERS)
        if wanted and (REPO_ROOT / name).is_file():
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO_ROOT / name, target)
            copied += 1
    shutil.copy2(destination / ".env.example", destination / ".env")  # the real .env is never copied
    return copied


def remove_tree(path: Path) -> None:
    """Delete a folder, including read-only files (DVC cache files are read-only, which Windows refuses to delete)."""
    def make_writable_and_retry(function, failed_path, _error) -> None:  # type: ignore[no-untyped-def]
        Path(failed_path).chmod(stat.S_IWRITE)
        function(failed_path)

    shutil.rmtree(path, onerror=make_writable_and_retry)


def dvc(args: list[str], cwd: Path) -> str | None:
    """Run one dvc command. Returns a problem text, or None when it worked."""
    code, output = run_command(["dvc", *args], cwd, DVC_TIMEOUT)
    return None if code == 0 else f"dvc {' '.join(args)} failed: {last_lines(output, 5)}"


def make_generated_files(folder: Path, count: int = 3) -> dict[str, str]:
    """Create small files with fixed contents and return {file name: sha256}."""
    folder.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for number in range(count):
        path = folder / f"artifact_{number}.bin"
        path.write_bytes(bytes((number * 31 + step) % 256 for step in range(2000 + number)))
        hashes[path.name] = sha256_of(path)
    return hashes


def dvc_round_trip(workdir: Path) -> tuple[int, str | None]:
    """
    Prove DVC can rebuild files from a remote: add and push 3 generated files, delete the files AND the cache, pull.

    Returns (number of files whose SHA-256 matches afterwards, problem text or None).
    """
    project, remote = workdir / "project", workdir / "remote"
    project.mkdir()
    remote.mkdir()
    steps = [["init", "--no-scm"], ["config", "core.analytics", "false"], ["remote", "add", "-d", "store", str(remote)]]
    hashes = make_generated_files(project / "data")
    steps.append(["add", *[f"data/{name}" for name in hashes]])
    steps.append(["push"])
    for step in steps:
        problem = dvc(step, project)
        if problem:
            return 0, problem
    for name in hashes:
        (project / "data" / name).unlink()
    remove_tree(project / ".dvc" / "cache")
    problem = dvc(["pull"], project)
    if problem:
        return 0, problem
    matching = sum(1 for name, digest in hashes.items() if (project / "data" / name).is_file()
                   and sha256_of(project / "data" / name) == digest)
    return matching, None if matching == len(hashes) else "some restored files differ from the originals"


PYTHON = sys.executable
