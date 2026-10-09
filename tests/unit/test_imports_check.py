"""
File: tests/unit/test_imports_check.py
Purpose: Tests the check that every package imported in core/ and showcase/ is declared in the requirements file.
         The fake projects are built in a temporary folder with package names that are not installed anywhere.
"""
from core import imports_check


def make_project(tmp_path, source, requirements, extra_files=None):
    """A tiny fake repo: core/example.py with the given source, plus requirements-dev.txt."""
    (tmp_path / "core").mkdir()
    (tmp_path / "core" / "example.py").write_text(source, encoding="utf-8")
    (tmp_path / "requirements-dev.txt").write_text(requirements, encoding="utf-8")
    for name, text in (extra_files or {}).items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    return tmp_path


def run_check(repo, exceptions=None):
    return imports_check.find_missing(repo, repo / "requirements-dev.txt", ["core"], exceptions or {})


def test_real_repo_has_no_missing_packages():
    assert imports_check.find_missing() == []


def test_declared_package_passes(tmp_path):
    repo = make_project(tmp_path, "import os\nimport fakepkg_one\n", "fakepkg-one==1.0\n")
    assert run_check(repo) == []


def test_missing_package_is_reported(tmp_path):
    repo = make_project(tmp_path, "import fakepkg_two\n", "fakepkg-one==1.0\n")
    problems = run_check(repo)
    assert len(problems) == 1
    assert "core/example.py" in problems[0] and "fakepkg_two" in problems[0]


def test_from_import_and_submodule_use_the_top_name(tmp_path):
    repo = make_project(tmp_path, "from fakepkg_three.sub import thing\n", "")
    assert "fakepkg_three" in run_check(repo)[0]


def test_standard_library_and_own_code_are_ignored(tmp_path):
    repo = make_project(tmp_path, "import json\nfrom core import other\nfrom pathlib import Path\n", "")
    assert run_check(repo) == []


def test_included_requirements_file_is_followed(tmp_path):
    more = {"more.txt": "fakepkg-four[extra]>=2 # note\n"}
    repo = make_project(tmp_path, "import fakepkg_four\n", "-r more.txt\n", more)
    assert run_check(repo) == []


def test_exception_counts_only_if_declared_in_the_other_file(tmp_path):
    repo = make_project(tmp_path, "import fakepkg_five\n", "", {"requirements-ml.txt": "fakepkg-five==1\n"})
    assert run_check(repo, {"fakepkg_five": "requirements-ml.txt"}) == []
    assert len(run_check(repo, {"fakepkg_five": "requirements-other.txt"})) == 1


def test_parse_requirement_line():
    assert imports_check.parse_requirement_line("anyio[trio]==4.7.0 ; python_version < '3.12'") == ("package", "anyio")
    assert imports_check.parse_requirement_line("-r requirements.txt") == ("include", "requirements.txt")
    assert imports_check.parse_requirement_line("# only a comment") is None
    assert imports_check.parse_requirement_line("   ") is None
