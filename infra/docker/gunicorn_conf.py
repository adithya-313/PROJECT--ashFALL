"""
File: infra/docker/gunicorn_conf.py
Purpose: Configuration file for the gunicorn process manager inside the Docker container.
Connects: Gunicorn reads this file when launched via the Dockerfile CMD.
          It pulls the worker count from the environment (WORKER_COUNT).
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
workers = int(os.environ.get("WORKER_COUNT", "2"))

# Number of simultaneous connections per worker.
worker_connections = 1000

# The maximum number of pending connections.
backlog = int(os.environ.get("BACKLOG", "2048"))

# Workers silent for more than this many seconds are killed and restarted.
timeout = int(os.environ.get("TIMEOUT", "30"))

# The number of seconds to wait for requests on a Keep-Alive connection.
keepalive = int(os.environ.get("KEEPALIVE", "2"))

# The maximum number of requests a worker will process before restarting.
max_requests = int(os.environ.get("MAX_REQUESTS", "0"))

# The maximum jitter to add to the max_requests setting.
max_requests_jitter = int(os.environ.get("MAX_REQUESTS_JITTER", "0"))

# Bind address. 0.0.0.0 makes the service reachable from outside the container.
bind = "0.0.0.0:8000"

# Gunicorn's own access log. Disabled because structured application logging covers requests.
accesslog = None

# Gunicorn's own error log (worker startup/shutdown events). "-" means stderr.
errorlog = "-"

# Log level for gunicorn's own error log.
loglevel = "info"
