"""
High-level RAG orchestrator for ingestion and retrieval.

This class wires together:
- Hierarchical chunking
- Multimodal embeddings
- Lexical + semantic + graph retrieval
- Optional entity/relation extraction (NER/RE)
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from uuid import UUID, uuid4

import asyncpg

from triple_hybrid_rag.config import RAGConfig, get_settings
from triple_hybrid_rag.core import (
    HierarchicalChunker,
    MultimodalEmbedder,
    QueryPlanner,
    RRFFusion,
    Reranker,
    EntityRelationExtractor,
    GraphEntityStore,
)
from triple_hybrid_rag.graph.puppygraph import PuppyGraphClient
from triple_hybrid_rag.graph.sql_fallback import SQLGraphFallback
from triple_hybrid_rag.types import (
    ChildChunk,
    IngestionResult,
    IngestionStatus,
    Modality,
    ParentChunk,
    QueryPlan,
    RetrievalResult,
    SearchChannel,
    SearchResult,
)

logger = logging.getLogger(__name__)


class RAG:
    """
    Orchestrates ingestion and retrieval across all three channels.

    Notes:
        - File ingestion currently supports plain text files only.
        - Entity extraction is optional and controlled by config.
    """

    def __init__(self, config: Optional[RAGConfig] = None) -> None:
        self.config = config or get_settings()
        self.chunker = HierarchicalChunker(self.config)
        self.embedder = MultimodalEmbedder(self.config)
        self.query_planner = QueryPlanner(self.config)
        self.fusion = RRFFusion(self.config)
        self.reranker = Reranker(self.config)
        self.graph_client = PuppyGraphClient(self.config)
        self.entity_extractor = EntityRelationExtractor(self.config)
        self.entity_store = GraphEntityStore(self.config)
        self._pool: Optional[asyncpg.Pool] = None

    async def _get_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                self.config.database_url,
                min_size=1,
                max_size=self.config.database_pool_size,
            )
        return self._pool

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
        await self.embedder.close()
        await self.query_planner.close()
        await self.reranker.close()
        await self.graph_client.close()
        await self.entity_store.close()

    async def ingest(
        self,
        file_path: str,
        tenant_id: str = "default",
        collection: str = "general",
        title: Optional[str] = None,
    ) -> IngestionResult:
        """
        Ingest a file into Postgres and optionally run entity extraction.

        Returns an IngestionResult describing stored rows.
        """
        path = Path(file_path)
        text = _load_text_file(path)
        file_hash = _sha256(text)
        document_id = uuid4()

        parents, children = self.chunker.split_document(
            text=text,
            document_id=document_id,
            tenant_id=tenant_id,
        )

        children = await self.embedder.embed_chunks(children)

        result = IngestionResult(document_id=document_id, success=False)

        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                await _upsert_document(
                    conn,
                    document_id=document_id,
                    tenant_id=tenant_id,
                    file_name=path.name,
                    file_path=str(path),
                    file_hash=file_hash,
                    collection=collection,
                    title=title or path.stem,
                )
                await _insert_parent_chunks(conn, parents)
                await _insert_child_chunks(conn, children)

        result.parent_chunks_created = len(parents)
        result.child_chunks_created = len(children)

        if self.config.rag_entity_extraction_enabled:
            extraction = await self.entity_extractor.extract(children)
            stats = await self.entity_store.store(
                extraction,
                children,
                tenant_id=tenant_id,
                document_id=document_id,
            )
            result.entities_extracted = stats.get("entities", 0)
            result.relations_extracted = stats.get("relations", 0)

        await _update_document_status(pool, document_id, IngestionStatus.COMPLETED)
        result.success = True
        return result

    async def retrieve(
        self,
        query: str,
        tenant_id: str = "default",
        collection: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> RetrievalResult:
        """
        Run triple-hybrid retrieval (lexical + semantic + graph) and fuse results.
        """
        query_plan = await self.query_planner.plan(query)
        top_k = top_k or self.config.rag_final_top_k

        lexical_results: List[SearchResult] = []
        semantic_results: List[SearchResult] = []
        graph_results: List[SearchResult] = []

        if self.config.rag_lexical_enabled:
            lexical_query = " ".join(query_plan.keywords) or query
            lexical_results = await _lexical_search(
                await self._get_pool(),
                tenant_id,
                lexical_query,
                query_plan.lexical_top_k,
                collection,
            )

        if self.config.rag_semantic_enabled:
            embedding = await self.embedder.embed_text(query_plan.semantic_query_text)
            semantic_results = await _semantic_search(
                await self._get_pool(),
                tenant_id,
                embedding,
                query_plan.semantic_top_k,
                collection,
            )

        if self.config.rag_graph_enabled:
            graph_results = await self._graph_search(query_plan, tenant_id)

        rerank_top_k = self.config.rag_rerank_top_k
        fused = self.fusion.fuse(
            lexical_results=lexical_results,
            semantic_results=semantic_results,
            graph_results=graph_results,
            query_plan=query_plan,
            top_k=rerank_top_k,
            apply_safety=False,
            apply_denoise=False,
        )

        expanded = await _expand_to_parents(await self._get_pool(), fused)

        if self.config.rag_rerank_enabled and expanded:
            expanded = await _rerank_parents(self.reranker, query, expanded)

        final_results, refused, reason, max_score = _apply_post_rerank_safety(
            expanded,
            safety_threshold=self.config.rag_safety_threshold,
            denoise_alpha=self.config.rag_denoise_alpha,
            denoise_enabled=self.config.rag_denoise_enabled,
            top_k=top_k,
        )

        retrieval = RetrievalResult(
            query=query,
            query_plan=query_plan,
            results=final_results,
            lexical_results=lexical_results,
            semantic_results=semantic_results,
            graph_results=graph_results,
        )

        if refused:
            retrieval.metadata.update(
                {
                    "refused": True,
                    "refusal_reason": reason,
                    "max_score": max_score,
                }
            )

        return retrieval

    async def _graph_search(self, plan: QueryPlan, tenant_id: str) -> List[SearchResult]:
        if not self.config.rag_graph_enabled:
            return []

        pool = await self._get_pool()
        sql_fallback = SQLGraphFallback(pool)

        if plan.cypher_query:
            try:
                return await self.graph_client.execute_query_plan_cypher(
                    plan.cypher_query,
                    tenant_id,
                    limit=plan.graph_top_k,
                )
            except Exception:
                return await sql_fallback.find_related_chunks(
                    plan.keywords,
                    tenant_id,
                    limit=plan.graph_top_k,
                )

        if plan.keywords:
            try:
                return await self.graph_client.search_by_keywords_graph(
                    plan.keywords,
                    tenant_id,
                    limit=plan.graph_top_k,
                )
            except Exception:
                return await sql_fallback.find_related_chunks(
                    plan.keywords,
                    tenant_id,
                    limit=plan.graph_top_k,
                )

        return []


def _load_text_file(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() not in {".txt", ".md"}:
        raise ValueError("Only .txt and .md files are supported for ingestion")

    return path.read_text(encoding="utf-8")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _vector_literal(values: List[float]) -> str:
    return "[" + ",".join(f"{v:.6f}" for v in values) + "]"


async def _upsert_document(
    conn: asyncpg.Connection,
    document_id: UUID,
    tenant_id: str,
    file_name: str,
    file_path: str,
    file_hash: str,
    collection: str,
    title: str,
) -> None:
    await conn.execute(
        """
        INSERT INTO rag_documents (
            id, tenant_id, hash_sha256, file_name, file_path, collection, title, ingestion_status, created_at, updated_at
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $9)
        ON CONFLICT (tenant_id, hash_sha256)
        DO UPDATE SET
            file_name = EXCLUDED.file_name,
            file_path = EXCLUDED.file_path,
            collection = EXCLUDED.collection,
            title = EXCLUDED.title,
            updated_at = EXCLUDED.updated_at
        """,
        document_id,
        tenant_id,
        file_hash,
        file_name,
        file_path,
        collection,
        title,
        IngestionStatus.PROCESSING.value,
        datetime.utcnow(),
    )


async def _update_document_status(
    pool: asyncpg.Pool,
    document_id: UUID,
    status: IngestionStatus,
) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE rag_documents
            SET ingestion_status = $2,
                updated_at = $3
            WHERE id = $1
            """,
            document_id,
            status.value,
            datetime.utcnow(),
        )


async def _insert_parent_chunks(conn: asyncpg.Connection, parents: List[ParentChunk]) -> None:
    await conn.executemany(
        """
        INSERT INTO rag_parent_chunks (
            id, document_id, tenant_id, index_in_document, text, token_count,
            page_start, page_end, section_heading, ocr_confidence, metadata
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
        """,
        [
            (
                p.id,
                p.document_id,
                p.tenant_id,
                p.index_in_document,
                p.text,
                p.token_count,
                p.page_start,
                p.page_end,
                p.section_heading,
                p.ocr_confidence,
                p.metadata,
            )
            for p in parents
        ],
    )


async def _insert_child_chunks(conn: asyncpg.Connection, children: List[ChildChunk]) -> None:
    await conn.executemany(
        """
        INSERT INTO rag_child_chunks (
            id, parent_id, document_id, tenant_id, index_in_parent, text, token_count,
            start_char_offset, end_char_offset, page, modality, content_hash,
            embedding_1024, image_embedding_1024, image_data, metadata
        )
        VALUES (
            $1, $2, $3, $4, $5, $6, $7,
            $8, $9, $10, $11, $12,
            $13::vector, $14::vector, $15, $16
        )
        """,
        [
            (
                c.id,
                c.parent_id,
                c.document_id,
                c.tenant_id,
                c.index_in_parent,
                c.text,
                c.token_count,
                c.start_char_offset,
                c.end_char_offset,
                c.page,
                c.modality.value,
                c.content_hash,
                _vector_literal(c.embedding or [0.0] * 1024),
                _vector_literal(c.image_embedding or [0.0] * 1024)
                if c.image_embedding
                else None,
                c.image_data,
                c.metadata,
            )
            for c in children
        ],
    )


async def _lexical_search(
    pool: asyncpg.Pool,
    tenant_id: str,
    query: str,
    limit: int,
    collection: Optional[str],
) -> List[SearchResult]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM rag_lexical_search($1, $2, $3, $4)",
            tenant_id,
            query,
            limit,
            collection,
        )
    return [_row_to_result(row, SearchChannel.LEXICAL, score_field="rank") for row in rows]


async def _semantic_search(
    pool: asyncpg.Pool,
    tenant_id: str,
    embedding: List[float],
    limit: int,
    collection: Optional[str],
) -> List[SearchResult]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM rag_semantic_search($1, $2::vector, $3, $4)",
            tenant_id,
            _vector_literal(embedding),
            limit,
            collection,
        )
    return [_row_to_result(row, SearchChannel.SEMANTIC, score_field="similarity") for row in rows]


def _row_to_result(
    row: asyncpg.Record,
    channel: SearchChannel,
    score_field: str,
) -> SearchResult:
    modality_val = row.get("modality", "text")
    modality = Modality(modality_val) if modality_val in [m.value for m in Modality] else Modality.TEXT
    result = SearchResult(
        chunk_id=row["child_id"],
        parent_id=row["parent_id"],
        document_id=row["document_id"],
        text=row["text"],
        page=row.get("page"),
        modality=modality,
        source_channel=channel,
    )
    if channel == SearchChannel.LEXICAL:
        result.lexical_score = row.get(score_field, 0.0)
    elif channel == SearchChannel.SEMANTIC:
        result.semantic_score = row.get(score_field, 0.0)
    return result


async def _expand_to_parents(
    pool: asyncpg.Pool,
    results: List[SearchResult],
) -> List[SearchResult]:
    if not results:
        return []

    parent_ids = list({r.parent_id for r in results if r.parent_id})
    if not parent_ids:
        return results

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, text, page_start, page_end, section_heading
            FROM rag_parent_chunks
            WHERE id = ANY($1::uuid[])
            """,
            parent_ids,
        )

    parent_lookup = {row["id"]: row for row in rows}
    merged: dict = {}

    for result in results:
        parent_id = result.parent_id
        if not parent_id:
            continue
        parent_row = parent_lookup.get(parent_id)
        if not parent_row:
            continue

        existing = merged.get(parent_id)
        if not existing:
            parent_result = SearchResult(
                chunk_id=parent_id,
                parent_id=parent_id,
                document_id=result.document_id,
                text=parent_row["text"],
                page=parent_row.get("page_start"),
                modality=result.modality,
                source_channel=result.source_channel,
            )
            parent_result.metadata["child_ids"] = [str(result.chunk_id)]
            parent_result.metadata["child_texts"] = [result.text]
            parent_result.metadata["section_heading"] = parent_row.get("section_heading")
            parent_result.rrf_score = result.rrf_score
            parent_result.lexical_score = result.lexical_score
            parent_result.semantic_score = result.semantic_score
            parent_result.graph_score = result.graph_score
            merged[parent_id] = parent_result
        else:
            existing.metadata.setdefault("child_ids", []).append(str(result.chunk_id))
            existing.metadata.setdefault("child_texts", []).append(result.text)
            existing.rrf_score = max(existing.rrf_score, result.rrf_score)
            existing.lexical_score = max(existing.lexical_score, result.lexical_score)
            existing.semantic_score = max(existing.semantic_score, result.semantic_score)
            existing.graph_score = max(existing.graph_score, result.graph_score)

    expanded = list(merged.values())
    expanded.sort(key=lambda r: r.rrf_score, reverse=True)
    return expanded


async def _rerank_parents(
    reranker: Reranker,
    query: str,
    results: List[SearchResult],
) -> List[SearchResult]:
    if not results:
        return []

    documents = [r.text for r in results]
    scores = await reranker.rerank(query, documents)
    for result, score in zip(results, scores):
        result.rerank_score = score
        result.final_score = score

    results.sort(key=lambda r: r.rerank_score or 0.0, reverse=True)
    return results


def _apply_post_rerank_safety(
    results: List[SearchResult],
    safety_threshold: float,
    denoise_alpha: float,
    denoise_enabled: bool,
    top_k: int,
) -> tuple[List[SearchResult], bool, Optional[str], float]:
    if not results:
        return [], True, "No candidates", 0.0

    max_score = max((r.rerank_score or r.final_score or 0.0) for r in results)
    if safety_threshold and max_score < safety_threshold:
        return [], True, "Below safety threshold", max_score

    filtered = results
    if denoise_enabled and max_score > 0:
        min_score = denoise_alpha * max_score
        filtered = [r for r in results if (r.rerank_score or r.final_score or 0.0) >= min_score]

    return filtered[:top_k], False, None, max_score
