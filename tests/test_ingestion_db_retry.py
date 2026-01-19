"""Tests for ingestion DB retry behavior."""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, cast
from uuid import uuid4

import pytest

from triple_hybrid_rag.config import RAGConfig
from triple_hybrid_rag.ingestion.ingest import IngestStats, Ingestor
from triple_hybrid_rag.types import ChildChunk, Modality, ParentChunk

class FakeConn:
    """Fake DB connection that fails once before succeeding."""

    def __init__(self) -> None:
        self.calls = 0

    async def fetch(self, *_args: Any, **_kwargs: Any):
        """
        Simulate batch fetch that fails on first call.
        
        The new batch insert implementation uses conn.fetch() with UNNEST arrays.
        """
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("transient db error")
        # Return a list of row dicts (batch insert returns multiple rows)
        return [{"id": uuid4()}]
    
    async def fetchrow(self, *_args: Any, **_kwargs: Any):
        """Legacy single-row fetch (no longer used in batch mode)."""
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("transient db error")
        return {"id": uuid4()}

class FakePool:
    """Fake asyncpg pool with acquire context manager."""

    def __init__(self, conn: FakeConn) -> None:
        self._conn = conn

    @asynccontextmanager
    async def acquire(self):
        yield self._conn

@pytest.mark.asyncio
async def test_store_chunks_retries_and_tracks_stats() -> None:
    config = RAGConfig(
        rag_ingest_db_retry_attempts=2,
        rag_ingest_retry_backoff_min=0.01,
        rag_ingest_retry_backoff_max=0.01,
    )
    ingestor = Ingestor(config=config)

    conn = FakeConn()
    pool = FakePool(conn)

    async def _fake_get_pool():
        return pool

    ingestor._get_db_pool = cast(Any, _fake_get_pool)

    stats = IngestStats()
    parent = ParentChunk(text="parent", page_start=1)
    child = ChildChunk(
        text="child",
        page=1,
        modality=Modality.TEXT,
        start_char_offset=0,
        end_char_offset=5,
        parent_id=parent.id,
    )

    parent_ids, child_ids = await ingestor._store_chunks(
        parent_chunks=[parent],
        child_chunks=[child],
        title="title",
        category="general",
        knowledge_base_id=None,
        stats=stats,
    )

    assert len(parent_ids) == 1
    assert len(child_ids) == 1
    assert stats.db_retries == 1
