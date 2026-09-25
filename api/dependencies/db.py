"""
File: api/dependencies/db.py
Purpose: Provides FastAPI dependencies for database access.
"""

from typing import AsyncGenerator

import asyncpg
import fastapi

async def get_db_pool(request: fastapi.Request) -> AsyncGenerator[asyncpg.Pool, None]:
    """
    FastAPI dependency that yields the asyncpg connection pool from app.state.
    
    This ensures handlers only access the initialized pool and do not create
    new connections directly.
    """
    yield request.app.state.db_pool
