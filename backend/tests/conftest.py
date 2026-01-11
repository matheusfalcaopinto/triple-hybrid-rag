"""Test configuration and fixtures."""

import asyncio
from typing import AsyncGenerator
from uuid import uuid4

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from control_plane.app import app
from control_plane.db.models.base import Base
from control_plane.db.session import get_session


# Test database URL - use SQLite for testing
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture(scope="function")
async def test_engine():
    """Create test database engine."""
    engine = create_async_engine(
        TEST_DATABASE_URL,
        echo=False,
    )
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create a test database session."""
    async_session_factory = async_sessionmaker(
        test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    
    async with async_session_factory() as session:
        yield session
        await session.rollback()


@pytest.fixture
def test_app(db_session: AsyncSession) -> FastAPI:
    """Create test FastAPI application."""
    
    async def override_get_session():
        yield db_session
    
    app.dependency_overrides[get_session] = override_get_session
    
    yield app
    
    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def client(test_app: FastAPI) -> AsyncGenerator[AsyncClient, None]:
    """Create test HTTP client."""
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
def user_id() -> str:
    """Generate a random user ID."""
    return str(uuid4())


@pytest.fixture
def establishment_id() -> str:
    """Generate a random establishment ID."""
    return str(uuid4())


@pytest.fixture
def auth_headers(user_id: str) -> dict[str, str]:
    """Generate auth headers for testing.
    
    Note: In real tests, this would create a proper JWT token.
    """
    from control_plane.auth.jwt import create_access_token
    
    token = create_access_token(
        data={"sub": user_id},
        expires_delta=None,
    )
    return {"Authorization": f"Bearer {token}"}
