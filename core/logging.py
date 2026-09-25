"""
File: core/logging.py
Purpose: Configures the application-wide structured JSON logger.
Connects: Any module in the application can import `get_logger` from this file to log events.
"""

import logging
import sys
import structlog


def configure_logging(log_level: str) -> None:
    """
    Sets up the structlog library to output JSON formatted logs.
    
    Args:
        log_level: The verbosity level for standard logging (e.g., 'DEBUG', 'INFO').
                   Events with a severity lower than this will be ignored.
                   
    Returns:
        None
    """
    # Parse the string log level into a standard Python logging integer level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Set the root logger configuration so that all standard library logs go to stdout
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=numeric_level,
    )
    
    # Configure structlog processors to add timestamps, log levels, and format as JSON
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.stdlib.add_log_level,
            structlog.processors.JSONRenderer()
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Retrieves a configured logger instance bound to a specific module name.
    
    Args:
        name: The name of the module or component requesting the logger (typically __name__).
        
    Returns:
        structlog.BoundLogger: A logger instance that will emit JSON logs.
    """
    return structlog.get_logger(name)
