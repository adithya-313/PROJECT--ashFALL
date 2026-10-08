# File: scripts/verify_dvc_restore.py
# Purpose: Prove that a brand-new clone of the repo plus the DVC remote rebuilds every DVC-tracked file exactly.
#
# Run:  python scripts/verify_dvc_restore.py --dvc-remote-dir <the dvc-remote folder>
#
# Steps: (1) clone the repo into a temporary folder OUTSIDE this repo, (2) point the clone at the remote folder,
# (3) run `dvc pull` there, (4) compare every tracked file in the clone with the same file in THIS repo (size and
# SHA-256). Nothing in this repo is changed. Exit code 0 means everything matched.
import argparse
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.backup_mlflow import sha256_of  # noqa: E402

DEFAULT_REPO_URL = "https://github.com/adithya-313/PROJECT--ashFALL"
TEMP_PREFIX = "ashfall-verify-"
NOT_RESTORED_BY_DVC = [".env", "mlflow.db", "mlruns/", "data/raw/paysim.csv", "data/processed/*"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read the command line."""
    parser = argparse.ArgumentParser(description="Check that a fresh clone plus dvc pull restores every tracked file.")
    parser.add_argument("--dvc-remote-dir", required=True, type=Path)
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    parser.add_argument("--keep-temp", action="store_true", help="do not delete the temporary clone")
    return parser.parse_args(argv)


def run(command: list[str], folder: Path) -> str:
    """Run a command in a folder and return its output. Raises RuntimeError with the output if it fails."""
    result = subprocess.run(command, cwd=folder, capture_output=True, text=True, timeout=1800)
    if result.returncode != 0:
        raise RuntimeError(f"`{' '.join(command)}` failed:\n{result.stdout[-800:]}\n{result.stderr[-800:]}")
    return result.stdout.strip()


def tracked_paths(repo: Path) -> list[Path]:
    """Read the output path out of every *.dvc file (not hard-coded), as paths relative to the repo."""
    paths = []
    for dvc_file in sorted(repo.rglob("*.dvc")):
        if not dvc_file.is_file():
            continue  # the folder named ".dvc" also matches "*.dvc"; only real files count
        if ".dvc" in dvc_file.relative_to(repo).parts[:-1]:
            continue  # skip anything inside the .dvc folder itself
        match = re.search(r"^\s*path:\s*(.+?)\s*$", dvc_file.read_text(encoding="utf-8"), re.MULTILINE)
        if match:
            paths.append(dvc_file.parent.relative_to(repo) / match.group(1))
    return paths


def compare_file(original: Path, restored: Path) -> str:
    """OK when the restored file exists with the same size and SHA-256 as the original, otherwise a reason."""
    if not restored.is_file():
        return "MISSING"
    if original.stat().st_size != restored.stat().st_size:
        return "MISMATCH (size)"
    if sha256_of(original) != sha256_of(restored):
        return "MISMATCH (sha256)"
    return "OK"


def make_clone(repo_url: str, temp_dir: Path) -> Path:
    """Clone the repo's main branch into the temporary folder."""
    clone_dir = temp_dir / "clone"
    run(["git", "clone", "--branch", "main", repo_url, str(clone_dir)], temp_dir)
    return clone_dir


def prepare_clone(clone_dir: Path, remote_dir: Path) -> None:
    """Point the clone at the remote and make sure it uses its own, empty cache."""
    run(["dvc", "remote", "add", "--local", "-d", "backup", str(remote_dir)], clone_dir)
    cache_dir = Path(run(["dvc", "cache", "dir"], clone_dir)).resolve()
    if clone_dir.resolve() not in cache_dir.parents:
        raise RuntimeError(f"The clone's DVC cache ({cache_dir}) is not inside the clone. Stopping.")
    if cache_dir.exists() and any(cache_dir.iterdir()):
        raise RuntimeError(f"The clone's DVC cache ({cache_dir}) is not empty. Stopping.")


def remove_read_only(function, path, _error) -> None:  # type: ignore[no-untyped-def]
    """Helper for rmtree on Windows: git marks some files read-only."""
    Path(path).chmod(stat.S_IWRITE)
    function(path)


def delete_temp_dir(temp_dir: Path) -> None:
    """Delete the temporary folder, but only if it is one this script made under the system temp folder."""
    system_temp = Path(tempfile.gettempdir()).resolve()
    if not (temp_dir.name.startswith(TEMP_PREFIX) and system_temp in temp_dir.resolve().parents):
        return
    # Windows may keep a file open for a moment after a program ends, so try a few times before giving up.
    for attempt in range(5):
        try:
            shutil.rmtree(temp_dir, onerror=remove_read_only)
            return
        except OSError:
            time.sleep(2)
    print(f"Could not delete the temporary folder {temp_dir}. It is safe to delete by hand.")


def verify(clone_dir: Path) -> bool:
    """Compare every tracked file; print a table; return True only if all are OK."""
    all_ok = True
    print(f"{'path':60} result")
    for relative in tracked_paths(clone_dir):
        result = compare_file(REPO_ROOT / relative, clone_dir / relative)
        all_ok = all_ok and result == "OK"
        print(f"{relative.as_posix():60} {result}")
    return all_ok


def main(argv: list[str] | None = None) -> int:
    """Clone, pull, compare, clean up."""
    args = parse_args(argv)
    temp_dir = Path(tempfile.mkdtemp(prefix=TEMP_PREFIX))
    try:
        clone_dir = make_clone(args.repo_url, temp_dir)
        prepare_clone(clone_dir, args.dvc_remote_dir)
        print(run(["dvc", "pull"], clone_dir).splitlines()[-1])
        all_ok = verify(clone_dir)
    except (RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"Verification could not finish: {error}")
        all_ok = False
    finally:
        if args.keep_temp:
            print(f"Temporary clone kept at {temp_dir}")
        else:
            delete_temp_dir(temp_dir)
    print("A clone does NOT get these from `dvc pull`: " + ", ".join(NOT_RESTORED_BY_DVC))
    print("RESULT: every tracked file matches" if all_ok else "RESULT: FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
