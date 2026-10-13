from pathlib import Path


def find_stdlib_json_imports(search_dirs: list[Path]) -> list[str]:
    """Search files recursively in the given directories for occurrences of 'import json'."""
    violations: list[str] = []
    for directory in search_dirs:
        if not directory.exists():
            continue
        # Walk every file in the directory tree.
        for path in directory.rglob("*"):
            if path.is_file():
                # Read text using UTF-8 with errors='ignore' so binary or special encodings don't raise.
                content = path.read_text(encoding="utf-8", errors="ignore")
                if "import json" in content:
                    violations.append(str(path))
    return sorted(violations)


def test_no_stdlib_json():
    """
    Test that 'import json' is not used anywhere in api/ or core/.
    This ensures we stick to orjson for performance.
    """
    repo_root = Path(__file__).parent.parent.parent
    target_dirs = [repo_root / "api", repo_root / "core"]
    # A missing folder must fail, not silently pass.
    for folder in target_dirs:
        assert folder.is_dir(), f"Expected folder is missing: {folder}"
    violations = find_stdlib_json_imports(target_dirs)
    assert not violations, "Found 'import json' in the codebase:\n" + "\n".join(violations)


def test_no_stdlib_json_catches_violations(tmp_path: Path):
    """
    Verify that find_stdlib_json_imports reliably detects files containing 'import json'.
    """
    sub_dir = tmp_path / "pkg"
    sub_dir.mkdir()
    bad_file = sub_dir / "service.py"
    bad_file.write_text("import json\n\ndef parse(): pass\n", encoding="utf-8")
    good_file = sub_dir / "clean.py"
    good_file.write_text("import orjson\n\ndef parse(): pass\n", encoding="utf-8")

    matches = find_stdlib_json_imports([sub_dir])
    assert matches == [str(bad_file)]
