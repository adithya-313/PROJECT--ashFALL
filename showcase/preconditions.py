# File: showcase/preconditions.py
# Purpose: Quick "is this service even there?" checks a level runs BEFORE its real checks. A missing service is
#          reported in plain English within a couple of seconds, instead of as a confusing error deep in a check.
import socket

import asyncpg

# How long we wait before deciding a service is not there. Short on purpose, so "missing" is reported fast.
CONNECT_TIMEOUT_SECONDS = 2.0


def check_tcp_port(host: str, port: int, what: str, fix_hint: str) -> str | None:
    """Return None if host:port accepts a connection, otherwise a plain-English problem with the fix."""
    # Try to open a plain network connection. We do not talk to the service, we only see if something answers.
    try:
        with socket.create_connection((host, port), timeout=CONNECT_TIMEOUT_SECONDS):
            return None
    except OSError as error:
        return f"{what} not reachable at {host}:{port} ({type(error).__name__}) - run: {fix_hint}"


async def check_database(database_url: str, fix_hint: str) -> str | None:
    """Return None if Postgres accepts a connection, otherwise a plain-English problem (never shows a password)."""
    # Try a real login with a short timeout, then close right away.
    try:
        connection = await asyncpg.connect(dsn=database_url, timeout=CONNECT_TIMEOUT_SECONDS)
    except (OSError, asyncpg.PostgresError, TimeoutError) as error:
        return f"Postgres not reachable ({type(error).__name__}) - run: {fix_hint}"
    await connection.close()
    return None


def check_redis(redis_url: str, fix_hint: str) -> str | None:
    """Return None if Redis accepts a connection, otherwise a plain-English problem."""
    import redis

    try:
        client = redis.Redis.from_url(redis_url, socket_timeout=CONNECT_TIMEOUT_SECONDS)
        client.ping()
        client.close()
        return None
    except Exception as error:
        return f"Redis not reachable ({type(error).__name__}) - run: {fix_hint}"
