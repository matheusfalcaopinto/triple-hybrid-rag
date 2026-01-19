"""Tests for SQLGraphFallback module."""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, List
from uuid import uuid4

import pytest

from triple_hybrid_rag.graph.sql_fallback import SQLGraphFallback
from triple_hybrid_rag.types import SearchChannel


class FakeRow(dict):
    def get(self, key, default=None):
        return super().get(key, default)


class FakeConn:
    """Fake async connection returning stubbed rows."""

    def __init__(self, entities: List[dict], chunks: List[dict], relations: List[dict] | None = None) -> None:
        self._entities = entities
        self._chunks = chunks
        self._relations = relations or []

    async def fetch(self, query: str, *args: Any) -> List[FakeRow]:
        if "rag_entities" in query:
            return [FakeRow(row) for row in self._entities]
        if "rag_relations" in query:
            return [FakeRow(row) for row in self._relations]
        if "rag_entity_mentions" in query or "rag_child_chunks" in query:
            return [FakeRow(row) for row in self._chunks]
        return []


class FakePool:
    def __init__(self, conn: FakeConn) -> None:
        self._conn = conn

    @asynccontextmanager
    async def acquire(self):
        yield self._conn


@pytest.mark.asyncio
async def test_search_by_keywords_returns_results() -> None:
    entity_id = uuid4()
    child_id = uuid4()
    parent_id = uuid4()
    doc_id = uuid4()

    entities = [{"id": entity_id, "name": "Acme Corp"}]
    chunks = [
        {
            "child_id": child_id,
            "parent_id": parent_id,
            "document_id": doc_id,
            "text": "Acme Corp is a company.",
            "page": 1,
            "modality": "text",
            "match_count": 1,
        }
    ]

    pool = FakePool(FakeConn(entities, chunks))
    fallback = SQLGraphFallback(pool)

    results = await fallback.search_by_keywords(["Acme"], tenant_id="t1", limit=10)

    assert len(results) == 1
    assert results[0].text == "Acme Corp is a company."
    assert results[0].source_channel == SearchChannel.GRAPH
    assert results[0].graph_score > 0


@pytest.mark.asyncio
async def test_search_by_keywords_empty_when_no_keywords() -> None:
    pool = FakePool(FakeConn([], []))
    fallback = SQLGraphFallback(pool)

    results = await fallback.search_by_keywords([], tenant_id="t1", limit=10)

    assert results == []


@pytest.mark.asyncio
async def test_find_related_chunks_traverses_relations() -> None:
    entity_a = uuid4()
    entity_b = uuid4()
    child_id = uuid4()
    parent_id = uuid4()
    doc_id = uuid4()

    entities = [{"id": entity_a, "name": "Alpha"}]
    relations = [{"related_id": entity_b}]
    chunks = [
        {
            "child_id": child_id,
            "parent_id": parent_id,
            "document_id": doc_id,
            "text": "Chunk related to Alpha and Beta",
            "page": 2,
            "modality": "text",
            "match_count": 2,
        }
    ]

    pool = FakePool(FakeConn(entities, chunks, relations))
    fallback = SQLGraphFallback(pool)

    results = await fallback.find_related_chunks(["Alpha"], tenant_id="t1", limit=10)

    assert len(results) == 1
    assert results[0].graph_score > 0
