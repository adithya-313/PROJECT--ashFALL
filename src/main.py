"""
File: src/main.py
Purpose: The main entrypoint of the application.
Connects: This script initializes configuration and logging, serving as the foundational harness.
"""

from src.core.config import config
from src.core.logging import configure_logging, get_logger

def main() -> None:
    """
    Initializes the foundation layer and emits a test log to verify functionality.
    
    This function sets up the logging according to the environment variables loaded
    in the configuration step. It then logs a structured JSON message and exits.
    
    Args:
        None
        
    Returns:
        None
    """
    # Initialize the logging framework using the log level from the environment
    configure_logging(config.log_level)
    
    # Retrieve a logger instance bound to this main module
    logger = get_logger(__name__)
    
    # Emit a test structured log line to verify that everything is working.
    # The 'environment' key provides context on where this log came from.
    logger.info("Application starting up", environment=config.environment)

if __name__ == "__main__":
    main()
