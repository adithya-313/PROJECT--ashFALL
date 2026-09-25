"""
File: tests/conftest.py
Purpose: Pytest configuration and shared fixtures for the test suite.
"""

import pytest


@pytest.fixture
def anyio_backend():
    """
    Restricts AnyIO async tests to the 'asyncio' event loop backend.
    Ensures compatibility across environments and avoids trio version mismatches.
    """
    return "asyncio"
