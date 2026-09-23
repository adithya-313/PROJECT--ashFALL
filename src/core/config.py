"""
File: src/core/config.py
Purpose: Defines the application configuration schema and loads variables from the environment.
Connects: Used throughout the application whenever a configuration value is needed. It relies on pydantic-settings to validate the environment variables.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict

class AppConfig(BaseSettings):
    """
    Defines the structured configuration for the application.
    
    Attributes:
        environment: The deployment environment (e.g., development, production).
        log_level: The verbosity of the logging output.
    """
    environment: str = "development"
    log_level: str = "INFO"

    # SettingsConfigDict configures how pydantic-settings behaves.
    # We tell it to look for a '.env' file, and if an environment variable exists,
    # it will override the file's value. Extra variables are ignored.
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8", 
        extra="ignore"
    )

def get_config() -> AppConfig:
    """
    Instantiates and returns the application configuration.
    
    Returns:
        AppConfig: A validated configuration object containing all settings.
    
    Why this function exists:
    This factory function allows us to potentially cache the configuration
    or inject dependencies later without changing how consumers access it.
    """
    return AppConfig()

# We create a single instance of the configuration to be imported by other modules.
config = get_config()
