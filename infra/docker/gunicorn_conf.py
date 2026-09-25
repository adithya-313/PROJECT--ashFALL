"""
File: infra/docker/gunicorn_conf.py
Purpose: Configuration file for the gunicorn process manager inside the Docker container.
Connects: Gunicorn reads this file when launched via the Dockerfile CMD.
          It pulls the worker count from the environment (GUNICORN_WORKERS).
          All other gunicorn settings are set here.

DECISION:
  gunicorn_conf.py is located under infra/docker/ because it configures the containerized
  ASGI server process topology (workers, connections, bindings, process management)
  specifically for container execution.
"""

import os

# The module path of the FastAPI 'app' object gunicorn will serve.
# Gunicorn uses this to import and bind the ASGI application.
wsgi_app = "api.main:app"

# Worker class: tells gunicorn to spawn uvicorn-based workers.
# Each worker runs its own asyncio event loop with uvloop and httptools.
worker_class = "uvicorn.workers.UvicornWorker"

# Number of worker processes. Each worker is a separate OS process.
# Read from the environment so it can be tuned without rebuilding.
workers = int(os.environ.get("GUNICORN_WORKERS", "2"))

# Number of simultaneous connections per worker.
worker_connections = 1000

# Bind address. 0.0.0.0 makes the service reachable from outside the container.
bind = "0.0.0.0:8000"

# Gunicorn's own access log. Disabled because structured application logging covers requests.
accesslog = None

# Gunicorn's own error log (worker startup/shutdown events). "-" means stderr.
errorlog = "-"

# Log level for gunicorn's own error log.
loglevel = "info"
