"""
File: core/logging.py
Purpose: Configures the application-wide structured JSON logger with context propagation.
Connects: Any module in the application can import `get_logger` from this file to log events.
          `request_id_ctx_var` is set by api/middleware/request_id.py to attach request tracing.
"""

import queue
import sys
from contextvars import ContextVar
import logging
import logging.handlers
from typing import Any, Optional
import structlog

# Context variable holding the current request ID for the active async call stack.
# Initialized with default=None when outside of an active HTTP request context.
request_id_ctx_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)

# Global reference to the log queue, ensuring all loggers can push to the same queue.
_log_queue: queue.Queue = queue.Queue(-1)


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
    
    # Set up the non-blocking logging pipeline.
    # structlog will format the JSON string and hand it to the root logger.
    # The root logger puts the string on the queue (taking ~1-2 microseconds).
    # A background QueueListener thread will drain the queue and execute the blocking sys.stdout.write.
    
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers to prevent duplicate logs if called multiple times
    root_logger.handlers.clear()
    
    # Attach the non-blocking QueueHandler to the root logger
    queue_handler = logging.handlers.QueueHandler(_log_queue)
    root_logger.addHandler(queue_handler)
    
    # Configure structlog processors to ensure timestamp, level, event, environment,
    # and request_id are present on every entry, then format as JSON.
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.processors.TimeStamper(fmt="iso", key="timestamp"),
            structlog.stdlib.add_log_level,
            inject_environment,
            inject_request_id,
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


def start_log_listener() -> logging.handlers.QueueListener:
    """
    Creates and starts the background QueueListener thread for non-blocking I/O.
    
    This function should be called within the FastAPI lifespan context manager
    to ensure the background thread is safely initialized AFTER the worker process
    forks. Starting threads at the module level is unsafe if a pre-fork model
    is ever introduced.
    
    Returns:
        logging.handlers.QueueListener: The started listener, to be stored and stopped later.
    """
    stream_handler = logging.StreamHandler(sys.stdout)
    # Since structlog has already formatted the message as a complete JSON string,
    # the stdlib formatter just outputs the raw message.
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    
    listener = logging.handlers.QueueListener(_log_queue, stream_handler)
    listener.start()
    return listener


def stop_log_listener(listener: logging.handlers.QueueListener) -> None:
    """
    Cleanly stops the background QueueListener thread.
    
    This blocks until all pending log messages in the queue have been flushed
    to stdout, preventing lost logs during graceful worker shutdown.
    
    Args:
        listener: The QueueListener instance returned by start_log_listener.
    """
    listener.stop()
