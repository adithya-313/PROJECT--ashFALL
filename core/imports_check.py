"""
File: core/imports_check.py
Purpose: Fails if a third-party package that core/ or showcase/ imports is missing from the dev requirements file.
         It reads the import lines with the ast module, turns each import name into its package name, and compares
         that with the packages listed in requirements-dev.txt (and every file it includes with -r).

Why it exists: A package can work on one laptop only because it happens to be installed there. CI installs ONLY the
requirements file, so a missing line shows up as a failure here instead of as a surprise later.

Use from the command line:
    python -m core.imports_check
"""

import ast
import re
import sys
from functools import lru_cache
from importlib import metadata
from pathlib import Path

from core.config_validator import load_yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
REQUIREMENTS_FILE = REPO_ROOT / "requirements-dev.txt"
EXCEPTIONS_FILE = REPO_ROOT / "config" / "import_exceptions.yaml"
FOLDERS_TO_CHECK = ["core", "showcase"]


def normalise(name):
    """Package names are the same when case, '-', '_' and '.' are ignored."""
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_requirement_line(line):
    """Return ('include', file name) for '-r file', ('package', name) for a package line, or None to skip."""
    line = line.split("#", 1)[0].strip()
    if not line:
        return None
    if line.startswith("-r "):
        return "include", line[3:].strip()
    if line.startswith("-"):
        return None
    # Cut off extras, versions and environment markers: 'anyio[trio]==4.7.0 ; python_version < "3.12"' -> 'anyio'.
    name = re.split(r"[\[<>=!~; ]", line, maxsplit=1)[0]
    return "package", name


def declared_packages(requirements_file=REQUIREMENTS_FILE):
    """All package names in a requirements file and in every file it includes. Names are normalised."""
    packages = set()
    for line in Path(requirements_file).read_text(encoding="utf-8").splitlines():
        parsed = parse_requirement_line(line)
        if parsed is None:
            continue
        kind, value = parsed
        if kind == "include":
            packages |= declared_packages(Path(requirements_file).parent / value)
        else:
            packages.add(normalise(value))
    return packages


def imported_module_names(folder):
    """Top-level module names imported by every .py file in a folder. Returns {module name: first file using it}."""
    found = {}
    for path in sorted(Path(folder).rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                continue
            for name in names:
                found.setdefault(name.split(".")[0], path)
    return found


def is_own_code(module_name, repo_root):
    """True if the import is this repo's own code (a folder or a .py file at the top of the repo)."""
    return (Path(repo_root) / module_name).is_dir() or (Path(repo_root) / f"{module_name}.py").exists()


@lru_cache(maxsize=1)
def installed_modules():
    """Which installed packages provide each import name. Slow to build on a big environment, so it is built once."""
    return metadata.packages_distributions()


def packages_for_module(module_name):
    """Every package (distribution) that provides an import name, e.g. 'yaml' -> ['PyYAML'].

    Some import names are shared by several packages (for example 'opentelemetry'). Falls back to the same name.
    """
    return installed_modules().get(module_name) or [module_name]


def load_exceptions(exceptions_file=EXCEPTIONS_FILE):
    """Modules that are deliberately declared in ANOTHER requirements file. Returns {module: file name}."""
    if not Path(exceptions_file).exists():
        return {}
    return load_yaml(exceptions_file)["declared_in_other_files"]["modules"]


def find_missing(repo_root=REPO_ROOT, requirements_file=REQUIREMENTS_FILE, folders=None, exceptions=None):
    """Return a list of 'file: imports X (package Y) but it is not in the requirements' messages."""
    declared = declared_packages(requirements_file)
    exceptions = load_exceptions() if exceptions is None else exceptions
    problems = []
    for folder in folders or FOLDERS_TO_CHECK:
        for module_name, path in imported_module_names(Path(repo_root) / folder).items():
            if module_name in sys.stdlib_module_names or is_own_code(module_name, repo_root):
                continue
            packages = packages_for_module(module_name)
            if any(normalise(package) in declared for package in packages):
                continue
            if module_name in exceptions:
                # An exception only counts if the module really is declared in the other file.
                other_file = Path(repo_root) / exceptions[module_name]
                if other_file.exists() and any(normalise(package) in declared_packages(other_file)
                                               for package in packages):
                    continue
            relative = path.relative_to(repo_root).as_posix()
            problems.append(f"{relative}: imports '{module_name}' (package '{packages[0]}'), "
                            f"which is not in {Path(requirements_file).name}")
    return problems


def main():
    problems = find_missing()
    for problem in problems:
        print(problem)
    if not problems:
        print("OK: every third-party import in core/ and showcase/ is declared in the requirements.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
