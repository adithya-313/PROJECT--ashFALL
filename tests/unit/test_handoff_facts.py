"""
tests/unit/test_handoff_facts.py

Unit tests for scripts/handoff_facts.py command-line argument handling and part selection.
Proves that specifying unknown parts fails loudly with a descriptive error message,
while valid parts ('12h', '13a') execute cleanly.
"""

import pytest

from scripts.handoff_facts import main


def test_handoff_facts_fails_loudly_on_unknown_part(capsys: pytest.CaptureFixture[str]) -> None:
    """Proves that scripts/handoff_facts.py rejects unknown part names with a clear error message.

    What it is for:
        When an engineer or automated pipeline passes an invalid --part argument, the script
        must reject the request with exit code 2 and print available options rather than
        silently falling back to a default part.

    Inputs:
        capsys: Pytest fixture capturing standard output and standard error streams.

    What it returns:
        None. Asserts SystemExit is raised and the error text lists available parts.

    How it works:
        1. Invoke handoff_facts.main with an unrecognized part argument ['--part', 'unknown_part_xyz'].
        2. Catch the resulting SystemExit exception and assert its code is 2.
        3. Read stderr from capsys and verify it names the invalid part and lists valid options ('12h', '13a').

    Errors and side effects:
        Raises AssertionError if the script fails to reject the unknown part or omits available options.

    Limits:
        Tests argument validation; does not evaluate fact extraction logic.
    """
    with pytest.raises(SystemExit) as excinfo:
        main(["--part", "unknown_part_xyz"])

    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert "Unknown part 'unknown_part_xyz'" in captured.err
    assert "Available parts: 12h, 13a, 13b" in captured.err


def test_handoff_facts_succeeds_on_known_parts(capsys: pytest.CaptureFixture[str]) -> None:
    """Proves that scripts/handoff_facts.py accepts registered parts ('12h', '13a', '13b') and generates output.

    What it is for:
        Verifies that valid part names ('12h', '13a', and '13b') are recognized and successfully produce
        the required markdown sections without raising exceptions.

    Inputs:
        capsys: Pytest fixture capturing standard output and standard error streams.

    What it returns:
        None. Asserts main returns 0 and expected headings appear in standard output.

    How it works:
        1. Run main with ['--part', '12h']; assert exit code is 0 and output contains '## Register Changes'.
        2. Run main with ['--part', '13a']; assert exit code is 0 and output contains 'nationalsim'.
        3. Run main with ['--part', '13b']; assert exit code is 0 and output contains 'world.phone_book'.

    Errors and side effects:
        None.

    Limits:
        Tests that top-level execution completes without unhandled errors.
    """
    exit_12h = main(["--part", "12h"])
    assert exit_12h == 0
    out_12h = capsys.readouterr().out
    assert "## Register Changes" in out_12h
    assert "world.edge_cases" in out_12h

    exit_13a = main(["--part", "13a"])
    assert exit_13a == 0
    out_13a = capsys.readouterr().out
    assert "## Register Changes" in out_13a
    assert "nationalsim.registry" in out_13a

    exit_13b = main(["--part", "13b"])
    assert exit_13b == 0
    out_13b = capsys.readouterr().out
    assert "## Register Changes" in out_13b
    assert "world.phone_book" in out_13b
