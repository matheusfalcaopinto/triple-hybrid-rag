"""
Knowledge Base Ingestor

Main entry point for the RAG ingestion pipeline:
- Orchestrates loading, OCR, chunking, embedding, and storage
- Handles deduplication
- Provides progress tracking and error handling
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from voice_agent.config import SETTINGS
from voice_agent.ingestion.chunker import Chunk, ChunkType, Chunker
from voice_agent.ingestion.embedder import Embedder, EmbeddingResult
from voice_agent.ingestion.loader import DocumentLoader, FileType, LoadedDocument
from voice_agent.ingestion.ocr import OCRProcessor, OCRResult
from voice_agent.observability.rag_metrics import rag_metrics
from voice_agent.utils.db import get_supabase_client

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
    chunks_embedded: int = 0
    chunks_stored: int = 0
    chunks_deduplicated: int = 0
    ocr_pages_processed: int = 0
    ocr_retries: int = 0
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
    error: Optional[str] = None


class KnowledgeBaseIngestor:
    """
    Main ingestion pipeline for the RAG system.
    
    Orchestrates:
    1. Document loading
    2. OCR processing (if needed)
    3. Chunking
    4. Embedding generation
    5. Deduplication
    6. Database storage
    """
    
    def __init__(
        self,
        org_id: str,
        category: str = "general",
        loader: Optional[DocumentLoader] = None,
        chunker: Optional[Chunker] = None,
        embedder: Optional[Embedder] = None,
        ocr_processor: Optional[OCRProcessor] = None,
        dedup_enabled: bool = True,
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
        """
        self.org_id = org_id
        self.category = category
        self.dedup_enabled = dedup_enabled if dedup_enabled is not None else SETTINGS.rag_dedup_enabled
        
        # Initialize components with defaults from config
        self.loader = loader or DocumentLoader()
        self.chunker = chunker or Chunker(
            chunk_size=SETTINGS.rag_chunk_size,
            chunk_overlap=SETTINGS.rag_chunk_overlap,
            preserve_tables=SETTINGS.rag_preserve_tables,
        )
        self.embedder = embedder or Embedder()
        self.ocr_processor = ocr_processor or OCRProcessor()
        
        self._supabase = None
    
    @property
    def supabase(self):
        """Lazy-load Supabase client."""
        if self._supabase is None:
            self._supabase = get_supabase_client()
        return self._supabase
    
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
            
            chunks = self.chunker.chunk_document(document)
            stats.chunks_created = len(chunks)
            
            # Update chunk metadata
            for chunk in chunks:
                chunk.source_document = source_document
            
            # Stage 4: Deduplication
            if self.dedup_enabled:
                chunks, dedup_count = await self._deduplicate_chunks(chunks)
                stats.chunks_deduplicated = dedup_count
            
            # Stage 5: Embedding
            if progress_callback:
                progress_callback("embedding", 0, len(chunks))
            
            def embed_progress(current: int, total: int):
                if progress_callback:
                    progress_callback("embedding", current, total)
            
            embedding_results = await self.embedder.embed_chunks(
                chunks,
                progress_callback=embed_progress,
            )
            
            stats.chunks_embedded = sum(1 for r in embedding_results if r.text_embedding or r.image_embedding)
            
            # Stage 6: Store in database
            if progress_callback:
                progress_callback("storing", 0, len(embedding_results))
            
            chunk_ids = await self._store_chunks(
                embedding_results,
                title=title,
                category=category,
                knowledge_base_id=knowledge_base_id,
                stats=stats,
            )
            
            stats.end_time = _utcnow()
            
            # Record metrics
            rag_metrics.documents_ingested.inc()
            rag_metrics.chunks_created.inc(stats.chunks_created)
            rag_metrics.chunks_stored.inc(stats.chunks_stored)
            rag_metrics.chunks_deduplicated.inc(stats.chunks_deduplicated)
            rag_metrics.chunks_per_document.observe(float(stats.chunks_created))
            rag_metrics.ingestion_duration.observe(stats.duration_seconds)
            
            return IngestResult(
                success=True,
                source_document=source_document,
                stats=stats,
                chunk_ids=chunk_ids,
            )
            
        except Exception as e:
            logger.exception(f"Ingestion failed for {file_path}: {e}")
            stats.errors.append(str(e))
            stats.end_time = _utcnow()
            
            # Record failure metric
            rag_metrics.documents_failed.inc()
            
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
        
        try:
            # Chunk the text
            chunks = self.chunker.chunk_text_simple(
                text=text,
                source_document=source_document,
                page_number=1,
            )
            stats.chunks_created = len(chunks)
            
            # Deduplication
            if self.dedup_enabled:
                chunks, dedup_count = await self._deduplicate_chunks(chunks)
                stats.chunks_deduplicated = dedup_count
            
            # Embed
            embedding_results = await self.embedder.embed_chunks(chunks)
            stats.chunks_embedded = sum(1 for r in embedding_results if r.text_embedding)
            
            # Store
            chunk_ids = await self._store_chunks(
                embedding_results,
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
            )
            
        except Exception as e:
            logger.exception(f"Text ingestion failed: {e}")
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
        max_concurrent: int = 2,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> List[IngestResult]:
        """
        Ingest multiple files.
        
        Args:
            file_paths: List of file paths
            category: Category for all files
            max_concurrent: Maximum concurrent file processing
            progress_callback: Callback for overall progress
            
        Returns:
            List of IngestResult for each file
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        results = []
        total = len(file_paths)
        
        async def process_file(idx: int, path: Union[str, Path]) -> IngestResult:
            async with semaphore:
                if progress_callback:
                    progress_callback("file", idx, total)
                return await self.ingest_file(path, category=category)
        
        tasks = [process_file(i, p) for i, p in enumerate(file_paths)]
        results = await asyncio.gather(*tasks)
        
        return list(results)
    
    async def _process_ocr(
        self,
        document: LoadedDocument,
        stats: IngestStats,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> None:
        """Process OCR for scanned pages."""
        pages_needing_ocr = [
            (i, page) for i, page in enumerate(document.pages)
            if page.is_scanned and page.image_data
        ]
        
        if not pages_needing_ocr:
            return
        
        for idx, (page_idx, page) in enumerate(pages_needing_ocr):
            if progress_callback:
                progress_callback("ocr", idx, len(pages_needing_ocr))
            
            if page.image_data:
                result = await self.ocr_processor.process_image(page.image_data)
                
                # Update page with OCR results
                if result.text:
                    document.pages[page_idx].text = result.text
                    document.pages[page_idx].is_scanned = False  # Now has text
                
                if result.tables:
                    document.pages[page_idx].tables.extend(result.tables)
                
                stats.ocr_pages_processed += 1
                stats.ocr_retries += result.retry_count
                
                if result.error:
                    stats.errors.append(f"OCR page {page_idx + 1}: {result.error}")
    
    async def _deduplicate_chunks(
        self,
        chunks: List[Chunk],
    ) -> tuple[List[Chunk], int]:
        """
        Remove duplicate chunks based on content hash.
        
        Returns:
            Tuple of (unique_chunks, dedup_count)
        """
        # Check existing hashes in database
        hashes = [c.content_hash for c in chunks]
        
        try:
            response = self.supabase.table("knowledge_base_chunks").select(
                "content_hash"
            ).eq(
                "org_id", self.org_id
            ).in_(
                "content_hash", hashes
            ).execute()
            
            existing_hashes = {row["content_hash"] for row in response.data}
        except Exception as e:
            logger.warning(f"Dedup check failed: {e}")
            existing_hashes = set()
        
        # Also dedupe within batch
        seen_hashes = set()
        unique_chunks = []
        
        for chunk in chunks:
            if chunk.content_hash in existing_hashes:
                continue
            if chunk.content_hash in seen_hashes:
                continue
            seen_hashes.add(chunk.content_hash)
            unique_chunks.append(chunk)
        
        dedup_count = len(chunks) - len(unique_chunks)
        return unique_chunks, dedup_count
    
    async def _store_chunks(
        self,
        embedding_results: List[EmbeddingResult],
        title: str,
        category: str,
        knowledge_base_id: Optional[str],
        stats: IngestStats,
    ) -> List[str]:
        """Store embedded chunks in the database."""
        chunk_ids = []
        
        for result in embedding_results:
            if result.error and not result.text_embedding:
                stats.errors.append(f"Embedding failed: {result.error}")
                continue
            
            chunk = result.chunk
            
            try:
                data = {
                    "org_id": self.org_id,
                    "knowledge_base_id": knowledge_base_id,
                    "category": category,
                    "title": title,
                    "source_document": chunk.source_document,
                    "modality": chunk.chunk_type.value,
                    "page": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                    "content": chunk.content,
                    "content_hash": chunk.content_hash,
                    "ocr_confidence": chunk.ocr_confidence,
                    "is_table": chunk.is_table,
                    "table_context": chunk.table_context if chunk.is_table else None,
                    "alt_text": chunk.alt_text if chunk.alt_text else None,
                }
                
                # Add embeddings
                if result.text_embedding:
                    data["vector_embedding"] = result.text_embedding
                
                if result.image_embedding:
                    data["vector_image"] = result.image_embedding
                
                # Remove None values
                data = {k: v for k, v in data.items() if v is not None}
                
                response = self.supabase.table("knowledge_base_chunks").insert(data).execute()
                
                if response.data:
                    chunk_ids.append(response.data[0]["id"])
                    stats.chunks_stored += 1
                    
            except Exception as e:
                logger.error(f"Failed to store chunk {chunk.chunk_index}: {e}")
                stats.errors.append(f"Storage failed: {str(e)}")
        
        return chunk_ids
    
    async def delete_by_source(self, source_document: str) -> int:
        """
        Delete all chunks from a source document.
        
        Args:
            source_document: Source document name
            
        Returns:
            Number of chunks deleted
        """
        try:
            response = self.supabase.table("knowledge_base_chunks").delete().eq(
                "org_id", self.org_id
            ).eq(
                "source_document", source_document
            ).execute()
            
            return len(response.data) if response.data else 0
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            return 0
