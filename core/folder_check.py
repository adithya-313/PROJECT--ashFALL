"""
File: core/folder_check.py
Purpose: Checks that every top-level folder of the repo is listed in config/folder_classification.yaml as either
         "engine" or "not_engine", and that the engine folders are also listed in config/new_engine_paths.yaml.

Why it exists: The pre-registration check only watches the paths in config/new_engine_paths.yaml. If new-engine code
could appear in an unlisted folder, the check would miss it. This check forces every new folder to be classified.

Use from the command line:
    python -m core.folder_check
"""

import os
import subprocess
import sys
from pathlib import Path

from core.config_validator import load_yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CLASSIFICATION_FILE = REPO_ROOT / "config" / "folder_classification.yaml"
NEW_ENGINE_PATHS_FILE = REPO_ROOT / "config" / "new_engine_paths.yaml"


def top_level_folders(root, use_git=True):
    """Names of the top-level folders. With git: folders that hold tracked files. Without: folders on disk."""
    if use_git:
        output = subprocess.run(["git", "-C", str(root), "ls-files"], capture_output=True, text=True, check=True)
        names = {line.split("/")[0] for line in output.stdout.splitlines() if "/" in line}
        return sorted(names)
    return sorted(entry.name for entry in os.scandir(root) if entry.is_dir())


def check_folders(folders, classification, engine_paths):
    """Return a list of error messages (empty = fine).

    folders is a list of top-level folder names, classification is the dict from the config file
    (with "engine" and "not_engine" lists), engine_paths is the list from config/new_engine_paths.yaml.
    """
    engine = set(classification["engine"])
    not_engine = set(classification["not_engine"])
    errors = []

    # A folder can only be one kind.
    for name in sorted(engine & not_engine):
        errors.append(f"folder '{name}' is listed as both engine and not_engine")

    # Every existing folder must be classified.
    for name in folders:
        if name not in engine and name not in not_engine:
            errors.append(f"folder '{name}' is not classified: add it to engine or not_engine "
                          f"in config/folder_classification.yaml")

    # Every engine folder must be watched by the order check.
    for name in sorted(engine):
        if f"{name}/" not in engine_paths:
            errors.append(f"engine folder '{name}' is not in config/new_engine_paths.yaml (as '{name}/')")

    # Every watched path must start in a classified folder.
    for path in engine_paths:
        first_part = path.split("/")[0]
        if first_part not in engine and first_part not in not_engine:
            errors.append(f"new_engine_paths entry '{path}' starts in folder '{first_part}', which is not classified")

    return errors


def main():
    classification = load_yaml(CLASSIFICATION_FILE)["folder_classification"]
    engine_paths = load_yaml(NEW_ENGINE_PATHS_FILE)["new_engine_paths"]["paths"]
    errors = check_folders(top_level_folders(REPO_ROOT), classification, engine_paths)
    for error in errors:
        print(error)
    if not errors:
        print("OK: every top-level folder is classified, and every engine folder is watched.")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
