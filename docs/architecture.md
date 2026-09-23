# Architecture Overview (Level 0 - Foundation)

## Introduction
This document describes the foundational architecture of the PROJECT--ashFALL system. This layer establishes the base directory structure, configuration management, logging strategy, and Docker orchestration, serving as the bedrock for all future services and APIs.

## Directory Structure
The repository uses a standard `src/` layout:
- `src/core/`: Contains core utilities used throughout the application, such as configuration loading (`config.py`) and logging setup (`logging.py`).
- `src/main.py`: The main entrypoint for the application. Currently serves as a minimal test harness for the foundation layer.
- `docs/`: Contains architectural documentation and guides.

We chose this layout to prevent import clutter at the repository root and to cleanly separate source code from configuration and orchestration files.

## Configuration System
The application is configured using environment variables.
- We use `pydantic-settings` to load, validate, and type-check these variables.
- A `.env.example` file is provided as a template. Developers should copy this to `.env` and provide real values. The `.env` file is explicitly ignored by version control to prevent leaking sensitive information.
- The `src/core/config.py` module defines the schema and provides a centralized way to access configuration values.

## Structured Logging
Logging is critical for observability in distributed systems.
- We use `structlog` to emit logs as structured JSON lines.
- JSON logs allow log aggregation systems to easily parse and query specific fields (like `environment`, `level`, or future fields like `request_id`).
- The logger is configured in `src/core/logging.py` and outputs directly to `stdout`, which is the standard best practice for containerized applications.

## Docker and GPU Orchestration
The application is containerized using Docker to ensure consistent execution environments.
- **Dockerfile**: A multi-stage definition starting from `python:3.11-slim` that installs dependencies and prepares the Python environment.
- **docker-compose.yml**: Orchestrates the local development environment. Critically, it requests exactly one NVIDIA GPU using device reservations to match our target hardware (1x RTX 4050, 6GB VRAM).

## External Services
If and when the system integrates with NVIDIA's model server, it will be strictly referred to as **Triton Inference Server**. This explicit naming avoids any ambiguity with the Triton language used for OpenAI's GPU kernels.
