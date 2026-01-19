"""
Knowledge Base Ingestor

Main entry point for the RAG ingestion pipeline:
- Orchestrates loading, OCR, chunking, embedding, and storage
- Handles deduplication via content hashing
- Provides progress tracking and error handling
- Supports tenacity retry for transient failures
"""

import asyncio
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from uuid import uuid4

from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from triple_hybrid_rag.config import RAGConfig
from triple_hybrid_rag.core.chunker import HierarchicalChunker
from triple_hybrid_rag.core.embedder import MultimodalEmbedder
from triple_hybrid_rag.ingestion.loaders import DocumentLoader, FileType, LoadedDocument
from triple_hybrid_rag.ingestion.ocr import OCRProcessor, OCRResult
from triple_hybrid_rag.types import ChildChunk, ParentChunk

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    """Get current UTC time with timezone awareness."""
    return datetime.now(timezone.utc)


@dataclass
class IngestStats:
    """Statistics from ingestion process."""
    documents_processed: int = 0
    pages_processed: int = 0
    chunks_created: int = 0
    parent_chunks: int = 0
    child_chunks: int = 0
    chunks_embedded: int = 0
    chunks_stored: int = 0
    chunks_deduplicated: int = 0
    ocr_pages_processed: int = 0
    ocr_retries: int = 0
    embed_retries: int = 0
    db_retries: int = 0
    errors: List[str] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    @property
    def duration_seconds(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0


@dataclass
class IngestResult:
    """Result of ingesting a document or batch."""
    success: bool
    source_document: str
    stats: IngestStats
    chunk_ids: List[str] = field(default_factory=list)
    parent_ids: List[str] = field(default_factory=list)
    error: Optional[str] = None


class Ingestor:
    """
    Main ingestion pipeline for the RAG system.
    
    Orchestrates:
    1. Document loading
    2. OCR processing (if needed)
    3. Hierarchical chunking (parent/child)
    4. Embedding generation
    5. Deduplication
    6. Database storage
    """
    
    def __init__(
        self,
        org_id: str = "default",
        category: str = "general",
        loader: Optional[DocumentLoader] = None,
        chunker: Optional[HierarchicalChunker] = None,
        embedder: Optional[MultimodalEmbedder] = None,
        ocr_processor: Optional[OCRProcessor] = None,
        dedup_enabled: bool = True,
        config: Optional[RAGConfig] = None,
    ):
        """
        Initialize the ingestor.
        
        Args:
            org_id: Organization ID for data isolation
            category: Default category for ingested content
            loader: Document loader instance
            chunker: Chunker instance
            embedder: Embedder instance
            ocr_processor: OCR processor instance
            dedup_enabled: Whether to enable deduplication
            config: RAG configuration
        """
        self.org_id = org_id
        self.category = category
        self.dedup_enabled = dedup_enabled
        self.config = config or RAGConfig()
        
        # Initialize components with defaults
        self.loader = loader or DocumentLoader()
        self.chunker = chunker or HierarchicalChunker(config=self.config)
        self.embedder = embedder or MultimodalEmbedder(config=self.config)
        self.ocr_processor = ocr_processor or OCRProcessor()
        
        self._db_pool = None
    
    async def ingest_file(
        self,
        file_path: Union[str, Path],
        title: Optional[str] = None,
        category: Optional[str] = None,
        knowledge_base_id: Optional[str] = None,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> IngestResult:
        """
        Ingest a single file into the knowledge base.
        
        Args:
            file_path: Path to the file
            title: Optional title for the document
            category: Category override
            knowledge_base_id: Parent knowledge_base entry ID
            progress_callback: Callback(stage, current, total) for progress
            
        Returns:
            IngestResult with stats and chunk IDs
        """
        stats = IngestStats(start_time=_utcnow())
        path = Path(file_path)
        source_document = path.name
        category = category or self.category
        title = title or path.stem
        document_id = uuid4()
        
        try:
            # Stage 1: Load document
            if progress_callback:
                progress_callback("loading", 0, 1)
            
            document = self.loader.load(path)
            
            if document.error:
                stats.errors.append(f"Loading failed: {document.error}")
                stats.end_time = _utcnow()
                return IngestResult(
                    success=False,
                    source_document=source_document,
                    stats=stats,
                    error=document.error,
                )
            
            stats.documents_processed = 1
            stats.pages_processed = len(document.pages)
            
            # Stage 2: OCR processing for scanned pages
            if progress_callback:
                progress_callback("ocr", 0, len(document.pages))
            
            await self._process_ocr(document, stats, progress_callback)
            
            # Stage 3: Chunking
            if progress_callback:
                progress_callback("chunking", 0, 1)
            
            # Combine all page text
            full_text = "\n\n".join(
                page.text for page in document.pages if page.text.strip()
            )
            
            # Use hierarchical chunking
            parent_chunks, child_chunks = self.chunker.split_document(
                text=full_text,
                document_id=document_id,
                tenant_id=self.org_id,
            )
            
            stats.parent_chunks = len(parent_chunks)
            stats.child_chunks = len(child_chunks)
            stats.chunks_created = len(parent_chunks) + len(child_chunks)
            
            # Stage 4: Deduplication (on child chunks only)
            if self.dedup_enabled:
                child_chunks, dedup_count = await self._deduplicate_chunks(child_chunks)
                stats.chunks_deduplicated = dedup_count
            
            # Stage 5: Embedding (child chunks only - they're the retrieval units)
            if progress_callback:
                progress_callback("embedding", 0, len(child_chunks))
            
            texts = [chunk.text for chunk in child_chunks]
            embeddings = await self._embed_texts_with_retry(texts, stats)
            
            for chunk, embedding in zip(child_chunks, embeddings):
                chunk.embedding = embedding
            
            stats.chunks_embedded = len(embeddings)
            
            # Stage 6: Store in database
            if progress_callback:
                progress_callback("storing", 0, len(child_chunks))
            
            parent_ids, chunk_ids = await self._store_chunks(
                parent_chunks=parent_chunks,
                child_chunks=child_chunks,
                title=title,
                category=category,
                knowledge_base_id=knowledge_base_id,
                stats=stats,
            )
            
            stats.end_time = _utcnow()
            
            return IngestResult(
                success=True,
                source_document=source_document,
                stats=stats,
                chunk_ids=chunk_ids,
                parent_ids=parent_ids,
            )
            
        except Exception as e:
            logger.exception(f"Ingestion failed for {file_path}: {e}")
            stats.errors.append(str(e))
            stats.end_time = _utcnow()
            
            return IngestResult(
                success=False,
                source_document=source_document,
                stats=stats,
                error=str(e),
            )
    
    async def ingest_text(
        self,
        text: str,
        title: str,
        category: Optional[str] = None,
        source_document: str = "inline",
        knowledge_base_id: Optional[str] = None,
    ) -> IngestResult:
        """
        Ingest raw text into the knowledge base.
        
        Args:
            text: Text content to ingest
            title: Title for the content
            category: Category override
            source_document: Source identifier
            knowledge_base_id: Parent knowledge_base entry ID
            
        Returns:
            IngestResult with stats and chunk IDs
        """
        stats = IngestStats(start_time=_utcnow())
        category = category or self.category
        document_id = uuid4()
        
        try:
            # Chunk the text hierarchically
            parent_chunks, child_chunks = self.chunker.split_document(
                text=text,
                document_id=document_id,
                tenant_id=self.org_id,
            )
            
            stats.parent_chunks = len(parent_chunks)
            stats.child_chunks = len(child_chunks)
            stats.chunks_created = len(parent_chunks) + len(child_chunks)
            
            # Deduplication
            if self.dedup_enabled:
                child_chunks, dedup_count = await self._deduplicate_chunks(child_chunks)
                stats.chunks_deduplicated = dedup_count
            
            # Embed child chunks
            texts = [chunk.text for chunk in child_chunks]
            embeddings = await self._embed_texts_with_retry(texts, stats)
            
            for chunk, embedding in zip(child_chunks, embeddings):
                chunk.embedding = embedding
            
            stats.chunks_embedded = len(embeddings)
            
            # Store
            parent_ids, chunk_ids = await self._store_chunks(
                parent_chunks=parent_chunks,
                child_chunks=child_chunks,
                title=title,
                category=category,
                knowledge_base_id=knowledge_base_id,
                stats=stats,
            )
            
            stats.end_time = _utcnow()
            
            return IngestResult(
                success=True,
                source_document=source_document,
                stats=stats,
                chunk_ids=chunk_ids,
                parent_ids=parent_ids,
            )
            
        except Exception as e:
            logger.exception(f"Ingestion failed for text: {e}")
            stats.errors.append(str(e))
            stats.end_time = _utcnow()
            
            return IngestResult(
                success=False,
                source_document=source_document,
                stats=stats,
                error=str(e),
            )
    
    async def ingest_batch(
        self,
        file_paths: List[Union[str, Path]],
        category: Optional[str] = None,
        max_concurrent: int = 4,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> List[IngestResult]:
        """
        Ingest multiple files concurrently.
        
        Args:
            file_paths: List of file paths to ingest
            category: Category override for all files
            max_concurrent: Maximum concurrent ingestions
            progress_callback: Callback(filename, current, total)
            
        Returns:
            List of IngestResult for each file
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        results = []
        
        async def ingest_with_limit(file_path: Union[str, Path], idx: int) -> IngestResult:
            async with semaphore:
                if progress_callback:
                    progress_callback(str(file_path), idx, len(file_paths))
                return await self.ingest_file(file_path, category=category)
        
        tasks = [
            ingest_with_limit(fp, i)
            for i, fp in enumerate(file_paths)
        ]
        results = await asyncio.gather(*tasks)
        
        return list(results)
    
    async def _process_ocr(
        self,
        document: LoadedDocument,
        stats: IngestStats,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> None:
        """Process OCR for pages that need it."""
        for i, page in enumerate(document.pages):
            if page.is_scanned and page.image_data:
                result = await self.ocr_processor.process_image(page.image_data)
                
                if result.text and not result.error:
                    # Append OCR text to page
                    page.text = f"{page.text}\n{result.text}".strip()
                    stats.ocr_pages_processed += 1
                    stats.ocr_retries += result.retry_count + result.network_retry_count
                else:
                    stats.errors.append(
                        f"OCR failed for page {page.page_number}: {result.error}"
                    )
                
                if progress_callback:
                    progress_callback("ocr", i + 1, len(document.pages))
    
    async def _deduplicate_chunks(
        self,
        chunks: List[ChildChunk],
    ) -> Tuple[List[ChildChunk], int]:
        """
        Deduplicate chunks based on content hash.
        
        Returns:
            Tuple of (deduplicated chunks, count of duplicates removed)
        """
        seen_hashes = set()
        unique_chunks = []
        duplicates = 0
        
        for chunk in chunks:
            # Compute hash of chunk text
            text_hash = hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()
            
            if text_hash not in seen_hashes:
                seen_hashes.add(text_hash)
                unique_chunks.append(chunk)
            else:
                duplicates += 1
        
        return unique_chunks, duplicates
    
    async def _embed_texts_with_retry(
        self,
        texts: List[str],
        stats: IngestStats,
    ) -> List[List[float]]:
        """Embed texts with retry on transient errors."""
        if not texts:
            return []

        retry_attempts = self.config.rag_ingest_embed_retry_attempts
        backoff_min = self.config.rag_ingest_retry_backoff_min
        backoff_max = self.config.rag_ingest_retry_backoff_max

        async for attempt in AsyncRetrying(
            retry=retry_if_exception_type(Exception),
            stop=stop_after_attempt(retry_attempts),
            wait=wait_exponential(multiplier=1, min=backoff_min, max=backoff_max),
            reraise=True,
        ):
            with attempt:
                if attempt.retry_state.attempt_number > 1:
                    stats.embed_retries += 1
                    logger.warning(
                        "Embedding retry %s/%s",
                        attempt.retry_state.attempt_number - 1,
                        retry_attempts - 1,
                    )
                return await self.embedder.embed_texts(texts, raise_on_error=True)

        return []

    async def _store_chunks(
        self,
        parent_chunks: List[ParentChunk],
        child_chunks: List[ChildChunk],
        title: str,
        category: str,
        knowledge_base_id: Optional[str],
        stats: IngestStats,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Tuple[List[str], List[str]]:
        """
        Store chunks in the database with retry for transient failures.
        
        OPTIMIZATION: Uses batch inserts to minimize network round-trips.
        With 289k chunks and batch_size=1000, this reduces calls from 289k to 290.
        
        Returns tuple of (parent_ids, child_ids).
        """
        retry_attempts = self.config.rag_ingest_db_retry_attempts
        backoff_min = self.config.rag_ingest_retry_backoff_min
        backoff_max = self.config.rag_ingest_retry_backoff_max
        batch_size = self.config.rag_db_batch_size

        async for attempt in AsyncRetrying(
            retry=retry_if_exception_type(Exception),
            stop=stop_after_attempt(retry_attempts),
            wait=wait_exponential(multiplier=1, min=backoff_min, max=backoff_max),
            reraise=True,
        ):
            with attempt:
                if attempt.retry_state.attempt_number > 1:
                    stats.db_retries += 1
                    logger.warning(
                        "DB store retry %s/%s",
                        attempt.retry_state.attempt_number - 1,
                        retry_attempts - 1,
                    )

                parent_ids = []
                chunk_ids = []

                # Get database connection
                pool = await self._get_db_pool()

                async with pool.acquire() as conn:
                    # ═══════════════════════════════════════════════════════════
                    # STORE PARENT CHUNKS (batched)
                    # ═══════════════════════════════════════════════════════════
                    parent_id_map = {}  # Map internal UUID to DB ID
                    
                    # Prepare parent chunk data
                    parent_records = [
                        (
                            parent.text,
                            getattr(parent, 'source_document', None),
                            parent.page_start,
                            parent.section_heading,
                            category,
                            self.org_id,
                            {"title": title, "knowledge_base_id": knowledge_base_id},
                            str(parent.id),  # Internal ID for mapping
                        )
                        for parent in parent_chunks
                    ]
                    
                    # Batch insert parents
                    for batch_start in range(0, len(parent_records), batch_size):
                        batch = parent_records[batch_start:batch_start + batch_size]
                        
                        # Use executemany with RETURNING via CTE
                        rows = await conn.fetch(
                            """
                            INSERT INTO rag_parent_chunks (
                                content, source_document, page_number,
                                section_heading, category, org_id, metadata
                            )
                            SELECT * FROM UNNEST($1::text[], $2::text[], $3::int[], 
                                                  $4::text[], $5::text[], $6::text[], $7::jsonb[])
                            RETURNING id
                            """,
                            [r[0] for r in batch],  # content
                            [r[1] for r in batch],  # source_document
                            [r[2] for r in batch],  # page_number
                            [r[3] for r in batch],  # section_heading
                            [r[4] for r in batch],  # category
                            [r[5] for r in batch],  # org_id
                            [r[6] for r in batch],  # metadata
                        )
                        
                        # Map internal UUIDs to DB IDs
                        for i, row in enumerate(rows):
                            db_id = str(row["id"])
                            parent_ids.append(db_id)
                            internal_id = batch[i][7]
                            parent_id_map[internal_id] = db_id

                    # ═══════════════════════════════════════════════════════════
                    # STORE CHILD CHUNKS (batched)
                    # ═══════════════════════════════════════════════════════════
                    
                    # Prepare child chunk data
                    child_records = []
                    for chunk in child_chunks:
                        parent_db_id = (
                            parent_id_map.get(str(chunk.parent_id))
                            if chunk.parent_id
                            else None
                        )
                        child_records.append((
                            chunk.text,
                            chunk.embedding,
                            getattr(chunk, 'source_document', None),
                            chunk.modality.value if hasattr(chunk, 'modality') else 'text',
                            parent_db_id,
                            chunk.page,
                            chunk.start_char_offset,
                            chunk.end_char_offset,
                            category,
                            self.org_id,
                            {"title": title, "knowledge_base_id": knowledge_base_id},
                        ))
                    
                    # Batch insert children
                    total_batches = (len(child_records) + batch_size - 1) // batch_size
                    for batch_idx, batch_start in enumerate(range(0, len(child_records), batch_size)):
                        batch = child_records[batch_start:batch_start + batch_size]
                        
                        rows = await conn.fetch(
                            """
                            INSERT INTO rag_chunks (
                                content, embedding, source_document, chunk_type,
                                parent_id, page_number, char_start, char_end,
                                category, org_id, metadata
                            )
                            SELECT * FROM UNNEST(
                                $1::text[], $2::vector[], $3::text[], $4::text[],
                                $5::uuid[], $6::int[], $7::int[], $8::int[],
                                $9::text[], $10::text[], $11::jsonb[]
                            )
                            RETURNING id
                            """,
                            [r[0] for r in batch],   # content
                            [r[1] for r in batch],   # embedding
                            [r[2] for r in batch],   # source_document
                            [r[3] for r in batch],   # chunk_type
                            [r[4] for r in batch],   # parent_id
                            [r[5] for r in batch],   # page_number
                            [r[6] for r in batch],   # char_start
                            [r[7] for r in batch],   # char_end
                            [r[8] for r in batch],   # category
                            [r[9] for r in batch],   # org_id
                            [r[10] for r in batch],  # metadata
                        )
                        
                        for row in rows:
                            chunk_ids.append(str(row["id"]))
                        
                        stats.chunks_stored += len(rows)
                        
                        if progress_callback:
                            progress_callback(batch_idx + 1, total_batches)
                    
                    logger.info(
                        f"Stored {len(parent_ids)} parents and {len(chunk_ids)} chunks "
                        f"in {total_batches} batches"
                    )

                return parent_ids, chunk_ids

        return [], []
    
    async def _get_db_pool(self):
        """Get or create database connection pool."""
        if self._db_pool is None:
            try:
                import asyncpg
                self._db_pool = await asyncpg.create_pool(
                    self.config.database_url,
                    min_size=2,
                    max_size=self.config.database_pool_size,
                )
            except ImportError:
                raise RuntimeError("asyncpg is required for database operations")
        return self._db_pool
    
    async def close(self) -> None:
        """Close database connections."""
        if self._db_pool:
            await self._db_pool.close()
            self._db_pool = None
