"""
File: core/logging.py
Purpose: Configures the application-wide structured JSON logger with context propagation.
Connects: Any module in the application can import `get_logger` from this file to log events.
          `request_id_ctx_var` is set by api/middleware/request_id.py to attach request tracing.
"""

import logging
import sys
from contextvars import ContextVar
from typing import Any, Optional

import structlog

# Context variable holding the current request ID for the active async call stack.
# Initialized with default=None when outside of an active HTTP request context.
request_id_ctx_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)


def inject_trace_id(logger: Any, method_name: str, event_dict: dict) -> dict:
    """
    Structlog processor that injects the active OpenTelemetry trace_id into every log line.
    When a trace_id is present, log lines and traces in Grafana can be linked by this value.
    """
    # Import here to avoid a circular import at module load time.
    from core.tracing import trace_id_ctx_var
    event_dict["trace_id"] = trace_id_ctx_var.get()
    return event_dict


def inject_request_id(logger: Any, method_name: str, event_dict: dict) -> dict:
    """
    Structlog processor that injects the current request_id from contextvars.
    
    Ensures every structured log line emitted within an HTTP request's call stack
    carries the associated request_id, enabling end-to-end request tracing.
    """
    event_dict["request_id"] = request_id_ctx_var.get()
    return event_dict


def inject_environment(logger: Any, method_name: str, event_dict: dict) -> dict:
    """
    Structlog processor that injects the deployment environment name into every log entry.
    """
    if "environment" not in event_dict:
        from core.config import config
        event_dict["environment"] = config.environment
    return event_dict


def configure_logging(log_level: str) -> None:
    """
    Sets up structlog to output JSON formatted logs with standard contextual fields:
    timestamp, level, event, environment, and request_id.
    
    Args:
        log_level: The verbosity level for standard logging (e.g., 'DEBUG', 'INFO').
                   Events with a severity lower than this will be ignored.
    """
    # Parse the string log level into a standard Python logging integer level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Set the root logger configuration so that all standard library logs go to stdout.
    # force=True ensures the stdout handler is attached even when invoked under test runners.
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=numeric_level,
        force=True,
    )
    
    # Configure structlog processors to ensure timestamp, level, event, environment,
    # and request_id are present on every entry, then format as JSON.
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.processors.TimeStamper(fmt="iso", key="timestamp"),
            structlog.stdlib.add_log_level,
            inject_environment,
            inject_request_id,
            inject_trace_id,
            structlog.processors.JSONRenderer(),
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
        structlog.BoundLogger: A logger instance that emits structured JSON logs.
    """
    return structlog.get_logger(name)
