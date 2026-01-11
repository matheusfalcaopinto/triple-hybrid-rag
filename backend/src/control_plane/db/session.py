"""Database session management."""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from control_plane.config import settings

# Engine and session factory
_engine: AsyncEngine | None = None
_async_session_factory: async_sessionmaker[AsyncSession] | None = None


async def init_db() -> None:
    """Initialize the database engine and session factory."""
    global _engine, _async_session_factory

    _engine = create_async_engine(
        settings.database_url,
        echo=settings.db_echo,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
    )

    _async_session_factory = async_sessionmaker(
        bind=_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )


async def close_db() -> None:
    """Close the database engine."""
    global _engine, _async_session_factory

    if _engine:
        await _engine.dispose()
        _engine = None
        _async_session_factory = None


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Get an async database session."""
    if _async_session_factory is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")

    async with _async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


def get_engine() -> AsyncEngine:
    """Get the database engine."""
    if _engine is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    return _engine
