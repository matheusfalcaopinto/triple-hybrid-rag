#!/usr/bin/env python3
"""
Triple-Hybrid-RAG Full Pipeline Test with Professional Rich TUI

Tests the complete ingestion + retrieval pipeline with real-time progress tracking:
1. Chunking (parent/child)
2. Embedding (via vLLM)
3. Entity Extraction (NER via OpenAI)
4. Database Storage (PostgreSQL + pgvector)
5. Entity Storage
6. Retrieval (lexical + semantic + graph)
7. RRF Fusion
8. Reranking

Features:
    - Rich TUI with progress bars and ETA
    - Real-time metrics and throughput
    - PuppyGraph Cypher queries (with SQL fallback)
    - Professional summary dashboard

Usage:
    uv run python scripts/full_pipeline_test.py
"""

import asyncio
import hashlib
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

# Rich TUI imports
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from triple_hybrid_rag.config import get_settings, reset_settings
from triple_hybrid_rag.core.chunker import HierarchicalChunker
from triple_hybrid_rag.core.embedder import MultimodalEmbedder
from triple_hybrid_rag.core.entity_extractor import EntityRelationExtractor
from triple_hybrid_rag.core.fusion import RRFFusion
from triple_hybrid_rag.core.reranker import Reranker
from triple_hybrid_rag.types import ChildChunk, SearchResult, SearchChannel, Modality

# Console for rich output
console = Console()

# Sample test document
TEST_DOCUMENT = """
# Triple-Hybrid RAG Architecture

## Overview

Triple-Hybrid RAG is a document retrieval system developed by Acme Corporation in 2026. 
The project lead is Dr. Sarah Chen, who previously worked at Google Research. 
The system costs approximately $50,000 per month to operate at scale.

## Key Components

### 1. Lexical Search (BM25)

The lexical component uses PostgreSQL Full-Text Search with GIN indexes.
It provides fast keyword matching with tf-idf weighting.
Configuration parameter: RAG_LEXICAL_WEIGHT = 0.7

### 2. Semantic Search (Vector)

Embeddings are generated using Qwen3-VL-Embedding-2B model.
The system uses pgvector for efficient similarity search.
Matryoshka embeddings allow dimension reduction from 2048 to 1024.

### 3. Graph Search (PuppyGraph)

Entity relationships are stored in a property graph.
PuppyGraph provides Cypher query interface over PostgreSQL.
Entities include: PERSON, ORGANIZATION, PRODUCT, CLAUSE.

## Performance Metrics

- Chunking: 400+ chunks/second
- Embedding: 500+ texts/second (concurrent)
- Retrieval: <100ms P95 latency
- Storage: Batch inserts reduce network calls by 1000x

## Team

The core team includes:
- Dr. Sarah Chen (Lead Architect)
- John Smith (Backend Engineer)  
- Maria Garcia (ML Engineer)
- Bob Johnson (DevOps)

The project is funded by TechVentures Capital with a Series A of $10 million.
"""

TENANT_ID = "test-org"


@dataclass
class StageResult:
    """Result from a pipeline stage."""
    name: str
    duration: float
    items_processed: int
    status: str = "✅"
    rate: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class PipelineTracker:
    """Tracks pipeline progress and metrics."""

    def __init__(self):
        self.stages: List[StageResult] = []
        self.start_time: float = 0
        self.current_stage: str = ""

    def start(self):
        self.start_time = time.perf_counter()

    def add_result(self, result: StageResult):
        self.stages.append(result)

    @property
    def total_duration(self) -> float:
        return time.perf_counter() - self.start_time

    @property
    def stages_passed(self) -> int:
        return sum(1 for s in self.stages if s.status == "✅")

    @property
    def stages_total(self) -> int:
        return len(self.stages)


def create_progress() -> Progress:
    """Create a rich progress bar."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=True,
    )


def create_results_table(tracker: PipelineTracker) -> Table:
    """Create results summary table."""
    table = Table(title="Pipeline Results", show_header=True, header_style="bold cyan")
    table.add_column("Stage", style="dim", width=20)
    table.add_column("Duration", justify="right", width=10)
    table.add_column("Items", justify="right", width=10)
    table.add_column("Rate", justify="right", width=15)
    table.add_column("Status", justify="center", width=8)

    for result in tracker.stages:
        rate_str = f"{result.rate:.1f}/s" if result.rate > 0 else "—"
        table.add_row(
            result.name,
            f"{result.duration:.3f}s",
            str(result.items_processed),
            rate_str,
            result.status,
        )

    return table


def print_header():
    """Print styled header."""
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]TRIPLE-HYBRID-RAG[/bold cyan]\n"
            "[dim]Full Pipeline End-to-End Test[/dim]",
            border_style="cyan",
        )
    )
    console.print()


def print_config(config):
    """Print configuration summary."""
    config_table = Table(show_header=False, box=None, padding=(0, 2))
    config_table.add_column("Key", style="dim")
    config_table.add_column("Value", style="green")

    config_table.add_row("Database", config.database_url[:50] + "...")
    config_table.add_row("Embed API", config.rag_embed_api_base)
    config_table.add_row("Embed Model", config.rag_embed_model)
    config_table.add_row("Rerank API", config.rag_rerank_api_base)
    config_table.add_row("OpenAI Model", config.openai_model)
    config_table.add_row("PuppyGraph", config.puppygraph_bolt_url)
    config_table.add_row("Graph Enabled", str(config.rag_graph_enabled))

    console.print(Panel(config_table, title="Configuration", border_style="blue"))
    console.print()


async def stage_chunking(config, tracker: PipelineTracker) -> Tuple[list, list]:
    """Stage 1: Hierarchical Chunking."""
    console.print("[bold cyan]━━━ Stage 1: Chunking[/bold cyan]")

    chunker = HierarchicalChunker(config=config)
    document_id = uuid4()

    start = time.perf_counter()
    parent_chunks, child_chunks = chunker.split_document(
        text=TEST_DOCUMENT,
        document_id=document_id,
        tenant_id=TENANT_ID,
    )
    duration = time.perf_counter() - start

    rate = len(child_chunks) / duration if duration > 0 else 0
    result = StageResult(
        name="1. Chunking",
        duration=duration,
        items_processed=len(child_chunks),
        rate=rate,
        details={"parents": len(parent_chunks), "children": len(child_chunks)},
    )
    tracker.add_result(result)

    console.print(f"   [green]✓[/green] Parents: {len(parent_chunks)}, Children: {len(child_chunks)}")
    console.print(f"   [green]✓[/green] Rate: {rate:.1f} chunks/s")
    console.print()

    return parent_chunks, child_chunks


async def stage_embedding(config, child_chunks: list, tracker: PipelineTracker) -> list:
    """Stage 2: Embedding via vLLM."""
    console.print("[bold cyan]━━━ Stage 2: Embedding[/bold cyan]")

    embedder = MultimodalEmbedder(config=config)
    texts = [c.text for c in child_chunks]

    with create_progress() as progress:
        task = progress.add_task("Embedding texts...", total=len(texts))

        start = time.perf_counter()
        embeddings = await embedder.embed_texts_concurrent(texts)
        duration = time.perf_counter() - start
        progress.update(task, completed=len(texts))

    rate = len(embeddings) / duration if duration > 0 else 0

    # Assign embeddings to chunks
    for chunk, emb in zip(child_chunks, embeddings):
        chunk.embedding = emb

    result = StageResult(
        name="2. Embedding",
        duration=duration,
        items_processed=len(embeddings),
        rate=rate,
        details={"dimension": len(embeddings[0]) if embeddings else 0},
    )
    tracker.add_result(result)

    console.print(f"   [green]✓[/green] Embedded: {len(embeddings)} texts")
    console.print(f"   [green]✓[/green] Rate: {rate:.1f} texts/s, Dim: {len(embeddings[0]) if embeddings else 0}")
    console.print()

    await embedder.close()
    return child_chunks


async def stage_ner(config, child_chunks: list, tracker: PipelineTracker) -> list:
    """Stage 3: Entity Extraction (NER)."""
    console.print("[bold cyan]━━━ Stage 3: Entity Extraction (NER)[/bold cyan]")

    if not config.openai_api_key:
        console.print("   [yellow]⚠[/yellow] OPENAI_API_KEY not set, skipping")
        tracker.add_result(StageResult(
            name="3. NER",
            duration=0,
            items_processed=0,
            status="⏭️",
        ))
        return []

    extractor = EntityRelationExtractor(config=config)

    with create_progress() as progress:
        task = progress.add_task("Extracting entities...", total=None)

        start = time.perf_counter()
        try:
            extraction = await extractor.extract(child_chunks[:5])
            duration = time.perf_counter() - start
            progress.update(task, completed=1, total=1)

            entities = extraction.entities
            relations = extraction.relations

            result = StageResult(
                name="3. NER",
                duration=duration,
                items_processed=len(entities),
                rate=len(entities) / duration if duration > 0 else 0,
                details={"relations": len(relations)},
            )
            tracker.add_result(result)

            console.print(f"   [green]✓[/green] Entities: {len(entities)}, Relations: {len(relations)}")
            console.print(f"   [green]✓[/green] Duration: {duration:.1f}s")

            # Group entities by type
            by_type = {}
            for e in entities:
                by_type.setdefault(e.entity_type, []).append(e.name)

            for etype, names in list(by_type.items())[:4]:
                console.print(f"   [dim]• {etype}: {', '.join(names[:3])}{'...' if len(names) > 3 else ''}[/dim]")

            console.print()
            return entities

        except Exception as e:
            duration = time.perf_counter() - start
            console.print(f"   [red]✗[/red] Error: {e}")
            tracker.add_result(StageResult(
                name="3. NER",
                duration=duration,
                items_processed=0,
                status="❌",
                error=str(e),
            ))
            console.print()
            return []


async def stage_db_storage(config, parent_chunks: list, child_chunks: list, tracker: PipelineTracker) -> Tuple[Optional[str], list]:
    """Stage 4: Database Storage."""
    console.print("[bold cyan]━━━ Stage 4: Database Storage[/bold cyan]")

    try:
        import asyncpg
    except ImportError:
        console.print("   [yellow]⚠[/yellow] asyncpg not installed")
        tracker.add_result(StageResult(name="4. DB Storage", duration=0, items_processed=0, status="⏭️"))
        return None, []

    try:
        pool = await asyncpg.create_pool(config.database_url, min_size=1, max_size=5)

        start = time.perf_counter()

        async with pool.acquire() as conn:
            # Create document
            doc_hash = hashlib.sha256(TEST_DOCUMENT.encode()).hexdigest()
            document_id = await conn.fetchval("""
                INSERT INTO rag_documents (tenant_id, hash_sha256, file_name, title, ingestion_status)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (tenant_id, hash_sha256) DO UPDATE SET title = EXCLUDED.title
                RETURNING id
            """, TENANT_ID, doc_hash, 'test_document.txt', 'Test Document', 'completed')

            # Insert parent chunks
            parent_ids = []
            for idx, parent in enumerate(parent_chunks):
                parent_id = await conn.fetchval("""
                    INSERT INTO rag_parent_chunks (document_id, tenant_id, index_in_document, text, page_start)
                    VALUES ($1, $2, $3, $4, $5)
                    RETURNING id
                """, document_id, TENANT_ID, idx, parent.text, parent.page_start or 1)
                parent_ids.append(parent_id)

            # Insert child chunks
            chunk_ids = []
            for idx, chunk in enumerate(child_chunks):
                if chunk.embedding:
                    parent_id = parent_ids[0] if parent_ids else None
                    content_hash = hashlib.sha256(chunk.text.encode()).hexdigest()[:32]
                    embedding_str = "[" + ",".join(str(x) for x in chunk.embedding) + "]"

                    chunk_id = await conn.fetchval("""
                        INSERT INTO rag_child_chunks (
                            parent_id, document_id, tenant_id, index_in_parent,
                            text, content_hash, embedding_1024, page
                        )
                        VALUES ($1, $2, $3, $4, $5, $6, $7::vector, $8)
                        ON CONFLICT (tenant_id, content_hash) DO UPDATE SET text = EXCLUDED.text
                        RETURNING id
                    """, parent_id, document_id, TENANT_ID, idx, chunk.text, content_hash, embedding_str, chunk.page or 1)
                    chunk_ids.append(str(chunk_id))

        duration = time.perf_counter() - start
        rate = len(chunk_ids) / duration if duration > 0 else 0

        result = StageResult(
            name="4. DB Storage",
            duration=duration,
            items_processed=len(chunk_ids),
            rate=rate,
            details={"document_id": str(document_id)},
        )
        tracker.add_result(result)

        console.print(f"   [green]✓[/green] Document: {str(document_id)[:8]}...")
        console.print(f"   [green]✓[/green] Stored: {len(chunk_ids)} chunks")
        console.print()

        await pool.close()
        return str(document_id), chunk_ids

    except Exception as e:
        console.print(f"   [red]✗[/red] Error: {e}")
        tracker.add_result(StageResult(name="4. DB Storage", duration=0, items_processed=0, status="❌", error=str(e)))
        console.print()
        return None, []


async def stage_entity_storage(config, entities: list, document_id: str, chunk_ids: list, tracker: PipelineTracker):
    """Stage 5: Entity Storage."""
    console.print("[bold cyan]━━━ Stage 5: Entity Storage[/bold cyan]")

    if not entities or not chunk_ids:
        console.print("   [yellow]⚠[/yellow] No entities or chunks to store")
        tracker.add_result(StageResult(name="5. Entity Storage", duration=0, items_processed=0, status="⏭️"))
        console.print()
        return

    try:
        import asyncpg
        from uuid import UUID

        pool = await asyncpg.create_pool(config.database_url, min_size=1, max_size=5)
        start = time.perf_counter()

        async with pool.acquire() as conn:
            doc_uuid = UUID(document_id)
            entity_ids = []

            for entity in entities[:10]:
                entity_id = await conn.fetchval("""
                    INSERT INTO rag_entities (tenant_id, document_id, entity_type, name, canonical_name)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (tenant_id, canonical_name) DO UPDATE SET name = EXCLUDED.name
                    RETURNING id
                """, TENANT_ID, doc_uuid, entity.entity_type, entity.name, entity.canonical_name)
                entity_ids.append(entity_id)

            # Store mentions
            mentions = 0
            for entity_id in entity_ids[:5]:
                for chunk_id_str in chunk_ids[:3]:
                    chunk_uuid = UUID(chunk_id_str)
                    await conn.execute("""
                        INSERT INTO rag_entity_mentions (entity_id, child_chunk_id, document_id, mention_text)
                        VALUES ($1, $2, $3, $4)
                        ON CONFLICT DO NOTHING
                    """, entity_id, chunk_uuid, doc_uuid, "mention")
                    mentions += 1

        duration = time.perf_counter() - start

        result = StageResult(
            name="5. Entity Storage",
            duration=duration,
            items_processed=len(entity_ids),
            rate=len(entity_ids) / duration if duration > 0 else 0,
            details={"mentions": mentions},
        )
        tracker.add_result(result)

        console.print(f"   [green]✓[/green] Entities: {len(entity_ids)}, Mentions: {mentions}")
        console.print()

        await pool.close()

    except Exception as e:
        console.print(f"   [red]✗[/red] Error: {e}")
        tracker.add_result(StageResult(name="5. Entity Storage", duration=0, items_processed=0, status="❌"))
        console.print()


async def stage_retrieval(config, child_chunks: list, tracker: PipelineTracker) -> Tuple[list, list, list]:
    """Stage 6: Triple-Hybrid Retrieval."""
    console.print("[bold cyan]━━━ Stage 6: Retrieval (Lexical + Semantic + Graph)[/bold cyan]")

    query = "Who is the lead architect?"
    embedder = MultimodalEmbedder(config=config)
    query_embedding = await embedder.embed_text(query)
    await embedder.close()

    semantic_results, lexical_results, graph_results = [], [], []
    puppygraph_used = False

    try:
        import asyncpg
        from triple_hybrid_rag.graph.puppygraph import PuppyGraphClient
        from triple_hybrid_rag.graph.sql_fallback import SQLGraphFallback

        pool = await asyncpg.create_pool(config.database_url, min_size=1, max_size=5)
        start = time.perf_counter()

        async with pool.acquire() as conn:
            # Semantic search
            query_embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
            semantic_results = await conn.fetch("""
                SELECT * FROM rag_semantic_search($1, $2::vector, $3)
            """, TENANT_ID, query_embedding_str, 5)

            # Lexical search
            lexical_results = await conn.fetch("""
                SELECT * FROM rag_lexical_search($1, $2, $3)
            """, TENANT_ID, query, 5)

        # Graph search - try PuppyGraph first, fallback to SQL
        if config.rag_graph_enabled:
            try:
                puppygraph = PuppyGraphClient(config=config)
                connected = await puppygraph.connect()

                if connected:
                    graph_results = await puppygraph.search_by_keywords_graph(
                        keywords=["Sarah Chen", "architect", "lead"],
                        tenant_id=TENANT_ID,
                        limit=5,
                    )
                    puppygraph_used = True
                    await puppygraph.close()
                else:
                    raise ConnectionError("PuppyGraph not available")

            except Exception as e:
                console.print(f"   [yellow]⚠[/yellow] PuppyGraph unavailable: {e}")
                console.print(f"   [dim]Using SQL fallback...[/dim]")

                graph_fallback = SQLGraphFallback(pool)
                graph_results = await graph_fallback.search_by_keywords(
                    keywords=["Sarah Chen", "architect", "lead"],
                    tenant_id=TENANT_ID,
                    limit=5,
                )

        duration = time.perf_counter() - start
        total_results = len(semantic_results) + len(lexical_results) + len(graph_results)

        result = StageResult(
            name="6. Retrieval",
            duration=duration,
            items_processed=total_results,
            rate=total_results / duration if duration > 0 else 0,
            details={
                "semantic": len(semantic_results),
                "lexical": len(lexical_results),
                "graph": len(graph_results),
                "puppygraph": puppygraph_used,
            },
        )
        tracker.add_result(result)

        console.print(f"   [green]✓[/green] Query: '{query}'")
        console.print(f"   [green]✓[/green] Semantic: {len(semantic_results)}, Lexical: {len(lexical_results)}, Graph: {len(graph_results)}")
        console.print(f"   [green]✓[/green] Graph Backend: {'[cyan]PuppyGraph[/cyan]' if puppygraph_used else '[yellow]SQL Fallback[/yellow]'}")
        console.print()

        await pool.close()
        return semantic_results, lexical_results, graph_results

    except Exception as e:
        console.print(f"   [red]✗[/red] Error: {e}")
        tracker.add_result(StageResult(name="6. Retrieval", duration=0, items_processed=0, status="❌", error=str(e)))
        console.print()
        return [], [], []


async def stage_fusion(config, semantic_results: list, lexical_results: list, graph_results: list, tracker: PipelineTracker) -> list:
    """Stage 7: RRF Fusion."""
    console.print("[bold cyan]━━━ Stage 7: RRF Fusion[/bold cyan]")

    if not semantic_results and not lexical_results and not graph_results:
        console.print("   [yellow]⚠[/yellow] No results to fuse")
        tracker.add_result(StageResult(name="7. RRF Fusion", duration=0, items_processed=0, status="⏭️"))
        console.print()
        return []

    from uuid import UUID

    def row_to_result(row, channel: SearchChannel, score_field: str) -> SearchResult:
        result = SearchResult(
            chunk_id=row['child_id'],
            parent_id=row['parent_id'],
            document_id=row['document_id'],
            text=row['text'],
            page=row.get('page'),
            modality=Modality.TEXT,
            source_channel=channel,
        )
        if channel == SearchChannel.SEMANTIC:
            result.semantic_score = float(row.get(score_field, 0.0))
        elif channel == SearchChannel.LEXICAL:
            result.lexical_score = float(row.get(score_field, 0.0))
        return result

    semantic_search_results = [row_to_result(r, SearchChannel.SEMANTIC, 'similarity') for r in semantic_results]
    lexical_search_results = [row_to_result(r, SearchChannel.LEXICAL, 'rank') for r in lexical_results]

    fusion = RRFFusion(config=config)

    start = time.perf_counter()
    fused_results = fusion.fuse(
        lexical_results=lexical_search_results,
        semantic_results=semantic_search_results,
        graph_results=graph_results,
        top_k=10,
        apply_safety=False,
        apply_denoise=False,
    )
    duration = time.perf_counter() - start

    result = StageResult(
        name="7. RRF Fusion",
        duration=duration,
        items_processed=len(fused_results),
        rate=len(fused_results) / duration if duration > 0 else 0,
    )
    tracker.add_result(result)

    console.print(f"   [green]✓[/green] Fused: {len(fused_results)} results")
    for i, r in enumerate(fused_results[:3]):
        channels = r.metadata.get('source_channels', [])
        console.print(f"   [dim]• [{i+1}] RRF={r.rrf_score:.4f} channels={channels}[/dim]")
    console.print()

    return fused_results


async def stage_reranker(config, fused_results: list, tracker: PipelineTracker) -> list:
    """Stage 8: Reranking."""
    console.print("[bold cyan]━━━ Stage 8: Reranker[/bold cyan]")

    if not fused_results:
        console.print("   [yellow]⚠[/yellow] No results to rerank")
        tracker.add_result(StageResult(name="8. Reranker", duration=0, items_processed=0, status="⏭️"))
        console.print()
        return []

    reranker = Reranker(config=config)

    if not config.rag_rerank_enabled:
        console.print("   [yellow]⚠[/yellow] Reranker disabled")
        tracker.add_result(StageResult(name="8. Reranker", duration=0, items_processed=0, status="⏭️"))
        console.print()
        return fused_results

    query = "Who is the lead architect?"
    documents = [r.text for r in fused_results]

    start = time.perf_counter()
    scores = await reranker.rerank(query, documents)
    duration = time.perf_counter() - start

    await reranker.close()

    if all(s == 0.0 for s in scores):
        console.print("   [yellow]⚠[/yellow] Reranker returned zeros (server not running?)")
        console.print("   [dim]Run: vllm serve Qwen/Qwen3-Reranker-2B --port 1235 --task rerank[/dim]")
        tracker.add_result(StageResult(name="8. Reranker", duration=duration, items_processed=0, status="⚠️"))
    else:
        for result, score in zip(fused_results, scores):
            result.rerank_score = score
            result.final_score = score

        fused_results.sort(key=lambda r: r.final_score, reverse=True)

        result = StageResult(
            name="8. Reranker",
            duration=duration,
            items_processed=len(scores),
            rate=len(scores) / duration if duration > 0 else 0,
        )
        tracker.add_result(result)

        console.print(f"   [green]✓[/green] Reranked: {len(scores)} results")
        for i, r in enumerate(fused_results[:3]):
            console.print(f"   [dim]• [{i+1}] score={r.rerank_score:.4f} {r.text[:50]}...[/dim]")

    console.print()
    return fused_results


async def cleanup(config):
    """Clean up test data."""
    console.print("[bold cyan]━━━ Cleanup[/bold cyan]")

    try:
        import asyncpg

        pool = await asyncpg.create_pool(config.database_url, min_size=1, max_size=5)
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM rag_entity_mentions WHERE entity_id IN (SELECT id FROM rag_entities WHERE tenant_id = $1)", TENANT_ID)
            await conn.execute("DELETE FROM rag_relations WHERE tenant_id = $1", TENANT_ID)
            await conn.execute("DELETE FROM rag_entities WHERE tenant_id = $1", TENANT_ID)
            await conn.execute("DELETE FROM rag_child_chunks WHERE tenant_id = $1", TENANT_ID)
            await conn.execute("DELETE FROM rag_parent_chunks WHERE tenant_id = $1", TENANT_ID)
            await conn.execute("DELETE FROM rag_documents WHERE tenant_id = $1", TENANT_ID)

        await pool.close()
        console.print("   [green]✓[/green] Test data cleaned")

    except Exception as e:
        console.print(f"   [yellow]⚠[/yellow] Cleanup failed: {e}")

    console.print()


def print_summary(tracker: PipelineTracker):
    """Print final summary dashboard."""
    console.print()
    console.print(create_results_table(tracker))
    console.print()

    # Summary panel
    status_text = f"[bold green]{tracker.stages_passed}/{tracker.stages_total} stages passed[/bold green]"
    time_text = f"Total: {tracker.total_duration:.2f}s"

    # Check if PuppyGraph was used
    retrieval = next((s for s in tracker.stages if s.name == "6. Retrieval"), None)
    if retrieval and retrieval.details.get("puppygraph"):
        graph_text = "[cyan]PuppyGraph: ✅[/cyan]"
    else:
        graph_text = "[yellow]PuppyGraph: ❌ (SQL Fallback)[/yellow]"

    console.print(
        Panel(
            f"{status_text}  |  {time_text}  |  {graph_text}",
            title="Summary",
            border_style="green" if tracker.stages_passed == tracker.stages_total else "yellow",
        )
    )


async def main():
    """Main entry point."""
    print_header()

    reset_settings()
    config = get_settings()
    print_config(config)

    tracker = PipelineTracker()
    tracker.start()

    try:
        # Stage 1: Chunking
        parent_chunks, child_chunks = await stage_chunking(config, tracker)

        # Stage 2: Embedding
        child_chunks = await stage_embedding(config, child_chunks, tracker)

        # Stage 3: NER
        entities = await stage_ner(config, child_chunks, tracker)

        # Stage 4: DB Storage
        document_id, chunk_ids = await stage_db_storage(config, parent_chunks, child_chunks, tracker)

        # Stage 5: Entity Storage
        await stage_entity_storage(config, entities, document_id, chunk_ids, tracker)

        # Stage 6: Retrieval
        semantic_results, lexical_results, graph_results = await stage_retrieval(config, child_chunks, tracker)

        # Stage 7: Fusion
        fused_results = await stage_fusion(config, semantic_results, lexical_results, graph_results, tracker)

        # Stage 8: Reranker
        await stage_reranker(config, fused_results, tracker)

        # Cleanup
        await cleanup(config)

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()

    print_summary(tracker)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
