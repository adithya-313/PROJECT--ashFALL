"""
File: tests/unit/nationalsim/test_image_requirements.py
Purpose: Verifies that every third-party import used by nationalsim is declared in requirements.txt.
"""

import ast
import sys
from pathlib import Path

from core.imports_check import (
    REPO_ROOT,
    declared_packages,
    is_own_code,
    normalise,
    packages_for_module,
)


def collect_third_party_imports(file_paths: list[Path]) -> dict[str, Path]:
    """Collect unique third-party module names imported by the specified Python files.

    What it is for:
        Extracts top-level imported module names from source files using the Python AST,
        filtering out Python standard library modules and this repository's own code.

    Inputs:
        file_paths: List of Path objects to Python source files to inspect.

    What it returns:
        A dictionary mapping each third-party module name to the first file that imports it.

    How it works:
        1. Parse each file into an abstract syntax tree.
        2. Walk the tree to find all Import and ImportFrom statements.
        3. Extract the root module name from each import statement.
        4. Skip standard library modules and internal repository packages.
        5. Record remaining third-party module names with the source file path.

    Errors and side effects:
        None.

    Limits:
        Inspects top-level imports statically; does not execute the files or trace dynamic imports.
    """
    found: dict[str, Path] = {}
    for path in file_paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                continue
            for name in names:
                root_name = name.split(".")[0]
                if root_name in sys.stdlib_module_names or is_own_code(root_name, REPO_ROOT):
                    continue
                found.setdefault(root_name, path)
    return found


def test_nationalsim_image_dependencies_declared_in_requirements_txt() -> None:
    """Verifies that all third-party imports in nationalsim are declared in requirements.txt.

    What it is for:
        The nationalsim Docker container installs requirements.txt alone (without requirements-dev.txt
        or requirements-test.txt). This test prevents container boot failures by ensuring every
        third-party dependency required by nationalsim is declared directly in requirements.txt.
        This recreates and guards against the real Level 13a failure where PyYAML was pinned only in
        requirements-test.txt, allowing local unit tests to pass while the container crashed on boot
        with ModuleNotFoundError: No module named 'yaml'.

    Inputs:
        None. Inspects all .py files in nationalsim/ and core/event_clock.py.

    What it returns:
        None. Asserts that no third-party import is missing from requirements.txt.

    How it works:
        1. Find all Python files in nationalsim/ plus core/event_clock.py.
        2. Extract all third-party imported module names using AST analysis.
        3. Parse declared packages from requirements.txt using declared_packages.
        4. Resolve distribution package names for each module using packages_for_module.
        5. Verify that each package is either declared in requirements.txt or provided as a core
           dependency of a declared package (such as pydantic from fastapi and pydantic-settings).
        6. Fail with an informative message if any required package is missing.

    Errors and side effects:
        Raises AssertionError listing every missing package and the file importing it.

    Limits:
        Checks packages required by the nationalsim image runtime, not development or test tooling.
    """
    nationalsim_files = sorted((REPO_ROOT / "nationalsim").glob("*.py"))
    clock_file = REPO_ROOT / "core" / "event_clock.py"
    files_to_check = nationalsim_files + [clock_file]

    third_party_imports = collect_third_party_imports(files_to_check)
    declared = declared_packages(REPO_ROOT / "requirements.txt")

    missing: list[str] = []
    for module_name, path in sorted(third_party_imports.items()):
        packages = packages_for_module(module_name)
        # Check direct declaration in requirements.txt
        is_declared = any(normalise(pkg) in declared for pkg in packages)
        # pydantic is a mandatory dependency of fastapi and pydantic-settings, both in requirements.txt
        if not is_declared and module_name == "pydantic":
            if any(p.startswith("pydantic-") or p == "fastapi" for p in declared):
                is_declared = True

        if not is_declared:
            relative_path = path.relative_to(REPO_ROOT).as_posix()
            pkg_name = packages[0] if packages else module_name
            missing.append(
                f"{relative_path}: imports '{module_name}' (package '{pkg_name}'), which is not in requirements.txt"
            )

    assert not missing, "Third-party imports missing from requirements.txt:\n" + "\n".join(missing)
