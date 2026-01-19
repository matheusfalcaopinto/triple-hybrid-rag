"""
SQL fallback for graph search when PuppyGraph is unavailable.
"""

from __future__ import annotations

import logging
from typing import List, Optional
from uuid import UUID

import asyncpg

from triple_hybrid_rag.types import SearchResult, SearchChannel, Modality

logger = logging.getLogger(__name__)


class SQLGraphFallback:
    """Simple SQL-based graph traversal using rag_entities and rag_relations."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self.pool = pool

    async def search_by_keywords(
        self,
        keywords: List[str],
        tenant_id: str,
        limit: int = 50,
    ) -> List[SearchResult]:
        if not keywords:
            return []
        patterns = [f"%{kw}%" for kw in keywords]
        async with self.pool.acquire() as conn:
            entities = await conn.fetch(
                """
                SELECT id, name
                FROM rag_entities
                WHERE tenant_id = $1
                  AND (
                    name ILIKE ANY($2::text[])
                    OR canonical_name ILIKE ANY($2::text[])
                  )
                LIMIT $3
                """,
                tenant_id,
                patterns,
                limit,
            )

            if not entities:
                return []

            entity_ids = [row["id"] for row in entities]

            rows = await conn.fetch(
                """
                SELECT
                    c.id as child_id,
                    c.parent_id,
                    c.document_id,
                    c.text,
                    c.page,
                    c.modality,
                    COUNT(em.entity_id) as match_count
                FROM rag_entity_mentions em
                JOIN rag_child_chunks c ON c.id = em.child_chunk_id
                WHERE em.entity_id = ANY($1::uuid[])
                  AND c.tenant_id = $2
                GROUP BY c.id
                ORDER BY match_count DESC
                LIMIT $3
                """,
                entity_ids,
                tenant_id,
                limit,
            )

        results: List[SearchResult] = []
        for row in rows:
            modality_val = row.get("modality", "text")
            modality = Modality(modality_val) if modality_val in [m.value for m in Modality] else Modality.TEXT
            result = SearchResult(
                chunk_id=row["child_id"],
                parent_id=row["parent_id"],
                document_id=row["document_id"],
                text=row["text"],
                page=row.get("page"),
                modality=modality,
                source_channel=SearchChannel.GRAPH,
            )
            match_count = float(row.get("match_count", 1))
            result.graph_score = match_count / max(len(keywords), 1)
            results.append(result)

        return results

    async def traverse_relations(
        self,
        entity_ids: List[UUID],
        tenant_id: str,
        limit: int = 50,
    ) -> List[UUID]:
        if not entity_ids:
            return []
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT DISTINCT object_entity_id as related_id
                FROM rag_relations
                WHERE tenant_id = $1 AND subject_entity_id = ANY($2::uuid[])
                LIMIT $3
                """,
                tenant_id,
                entity_ids,
                limit,
            )
        return [row["related_id"] for row in rows]

    async def find_related_chunks(
        self,
        keywords: List[str],
        tenant_id: str,
        limit: int = 50,
    ) -> List[SearchResult]:
        if not keywords:
            return []

        patterns = [f"%{kw}%" for kw in keywords]
        async with self.pool.acquire() as conn:
            entities = await conn.fetch(
                """
                SELECT id
                FROM rag_entities
                WHERE tenant_id = $1
                  AND (
                    name ILIKE ANY($2::text[])
                    OR canonical_name ILIKE ANY($2::text[])
                  )
                LIMIT $3
                """,
                tenant_id,
                patterns,
                limit,
            )

        if not entities:
            return []

        entity_ids = [row["id"] for row in entities]

        related_ids = await self.traverse_relations(entity_ids, tenant_id, limit=limit)
        all_ids = list({*entity_ids, *related_ids})
        if not all_ids:
            return await self.search_by_keywords([kw for kw in keywords if kw], tenant_id, limit=limit)

        return await self._chunks_for_entity_ids(all_ids, tenant_id, limit)

    async def _chunks_for_entity_ids(
        self,
        entity_ids: List[UUID],
        tenant_id: str,
        limit: int,
    ) -> List[SearchResult]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    c.id as child_id,
                    c.parent_id,
                    c.document_id,
                    c.text,
                    c.page,
                    c.modality,
                    COUNT(em.entity_id) as match_count
                FROM rag_entity_mentions em
                JOIN rag_child_chunks c ON c.id = em.child_chunk_id
                WHERE em.entity_id = ANY($1::uuid[])
                  AND c.tenant_id = $2
                GROUP BY c.id
                ORDER BY match_count DESC
                LIMIT $3
                """,
                entity_ids,
                tenant_id,
                limit,
            )

        results: List[SearchResult] = []
        for row in rows:
            modality_val = row.get("modality", "text")
            modality = Modality(modality_val) if modality_val in [m.value for m in Modality] else Modality.TEXT
            result = SearchResult(
                chunk_id=row["child_id"],
                parent_id=row["parent_id"],
                document_id=row["document_id"],
                text=row["text"],
                page=row.get("page"),
                modality=modality,
                source_channel=SearchChannel.GRAPH,
            )
            match_count = float(row.get("match_count", 1))
            result.graph_score = match_count / max(len(entity_ids), 1)
            results.append(result)
        return results
