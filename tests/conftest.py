"""
File: tests/conftest.py
Purpose: Pytest configuration and shared fixtures for the test suite.
"""

import pytest

# Makes the run fail when a test is skipped and FAIL_ON_SKIP=1 (see tests/fail_on_skip.py).
from tests.fail_on_skip import pytest_sessionfinish  # noqa: F401


@pytest.fixture
def anyio_backend():
    """
    Restricts AnyIO async tests to the 'asyncio' event loop backend.
    Ensures compatibility across environments and avoids trio version mismatches.
    """
    return "asyncio"
