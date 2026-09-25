"""
File: core/exceptions.py
Purpose: Application-wide base and domain-specific exception classes.
Connects: Imported by API route handlers, core services, and pipeline components
          to provide clear, strongly-typed error handling across PROJECT-ashFALL.
"""


class AshfallException(Exception):
    """
    Base exception class for all custom errors in PROJECT-ashFALL.
    
    Subclasses should define specific error conditions so callers can handle
    expected domain failures cleanly without broad except clauses.
    """
    pass


class ConfigurationError(AshfallException):
    """Raised when environment variables or application configuration are invalid."""
    pass
