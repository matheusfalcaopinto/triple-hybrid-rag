"""
RAG 2.0 Ingestion Pipeline

Orchestrates document ingestion into the RAG 2.0 schema:
1. Document registration (with idempotency via hash)
2. Text extraction (reuse existing loader/OCR)
3. Hierarchical chunking (parent/child)
4. Embedding generation (Matryoshka 4096→1024)
5. Deduplication
6. Database storage
"""

import asyncio
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union
from uuid import uuid4

from voice_agent.ingestion.loader import DocumentLoader, LoadedDocument
from voice_agent.ingestion.ocr import OCRProcessor
from voice_agent.rag2.chunker import (
    ChildChunk,
    HierarchicalChunker,
    ParentChunk,
    get_hierarchical_chunker,
)
from voice_agent.rag2.embedder import RAG2Embedder, get_rag2_embedder
from voice_agent.utils.db import get_supabase_client

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    """Get current UTC time."""
    return datetime.now(timezone.utc)


@dataclass
class IngestStats:
    """Statistics from RAG2 ingestion process."""
    documents_registered: int = 0
    documents_skipped: int = 0  # Already exists (idempotent)
    parent_chunks_created: int = 0
    child_chunks_created: int = 0
    child_chunks_embedded: int = 0
    child_chunks_deduplicated: int = 0
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
    """Result of ingesting a document."""
    success: bool
    document_id: Optional[str]
    stats: IngestStats
    error: Optional[str] = None


class RAG2Ingestor:
    """
    RAG 2.0 Ingestion Pipeline.
    
    Ingests documents into the new RAG2 schema:
    - rag_documents
    - rag_parent_chunks
    - rag_child_chunks
    """
    
    def __init__(
        self,
        org_id: str,
        collection: str = "general",
        loader: Optional[DocumentLoader] = None,
        chunker: Optional[HierarchicalChunker] = None,
        embedder: Optional[RAG2Embedder] = None,
        ocr_processor: Optional[OCRProcessor] = None,
        dedup_enabled: bool = True,
    ):
        """
        Initialize the RAG2 ingestor.
        
        Args:
            org_id: Organization ID for data isolation
            collection: Default collection name
            loader: Document loader instance
            chunker: Hierarchical chunker instance
            embedder: RAG2 embedder instance
            ocr_processor: OCR processor instance
            dedup_enabled: Whether to enable chunk-level deduplication
        """
        self.org_id = org_id
        self.collection = collection
        self.dedup_enabled = dedup_enabled
        
        # Initialize components
        self.loader = loader or DocumentLoader()
        self.chunker = chunker or get_hierarchical_chunker()
        self.embedder = embedder or get_rag2_embedder()
        self.ocr_processor = ocr_processor or OCRProcessor()
        
        self._supabase = None
    
    @property
    def supabase(self) -> Any:
        """Lazy-load Supabase client."""
        if self._supabase is None:
            self._supabase = get_supabase_client()
        return self._supabase
    
    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA-256 hash of file contents."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    async def ingest_file(
        self,
        file_path: Union[str, Path],
        title: Optional[str] = None,
        collection: Optional[str] = None,
        tags: Optional[List[str]] = None,
        force: bool = False,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> IngestResult:
        """
        Ingest a single file into RAG2 schema.
        
        Args:
            file_path: Path to the file
            title: Optional document title
            collection: Collection override
            tags: Optional tags
            force: Force re-ingestion even if document exists
            progress_callback: Callback(stage, current, total)
            
        Returns:
            IngestResult with document_id and stats
        """
        stats = IngestStats(start_time=_utcnow())
        path = Path(file_path)
        collection = collection or self.collection
        title = title or path.stem
        tags = tags or []
        
        try:
            # Step 1: Compute file hash for idempotency
            file_hash = self._compute_file_hash(path)
            
            if progress_callback:
                progress_callback("registering", 1, 6)
            
            # Step 2: Check if document already exists
            existing = self.supabase.table("rag_documents").select("id").eq(
                "org_id", self.org_id
            ).eq("hash_sha256", file_hash).execute()
            
            if existing.data and not force:
                logger.info(f"Document already exists: {path.name}")
                stats.documents_skipped = 1
                stats.end_time = _utcnow()
                return IngestResult(
                    success=True,
                    document_id=existing.data[0]["id"],
                    stats=stats,
                )
            
            # Step 3: Register document
            doc_id = str(uuid4())
            doc_data = {
                "id": doc_id,
                "org_id": self.org_id,
                "hash_sha256": file_hash,
                "file_name": path.name,
                "mime_type": self._get_mime_type(path),
                "collection": collection,
                "title": title,
                "tags": tags,
                "ingestion_status": "processing",
            }
            
            self.supabase.table("rag_documents").upsert(doc_data).execute()
            stats.documents_registered = 1
            
            if progress_callback:
                progress_callback("loading", 2, 6)
            
            # Step 4: Load document content
            loaded_doc = await self._load_document(path)
            full_text = self._get_full_text(loaded_doc)
            
            # Build page mapping for provenance
            source_pages = self._build_page_map(loaded_doc)
            
            if progress_callback:
                progress_callback("chunking", 3, 6)
            
            # Step 5: Create hierarchical chunks
            parents = self.chunker.chunk_document(
                text=full_text,
                doc_hash=file_hash[:16],  # Use first 16 chars for stable IDs
                source_pages=source_pages,
            )
            
            stats.parent_chunks_created = len(parents)
            
            if progress_callback:
                progress_callback("embedding", 4, 6)
            
            # Step 6: Embed and store chunks
            await self._store_chunks(
                doc_id=doc_id,
                parents=parents,
                stats=stats,
            )
            
            if progress_callback:
                progress_callback("finalizing", 5, 6)
            
            # Step 7: Update document status
            self.supabase.table("rag_documents").update({
                "ingestion_status": "completed",
            }).eq("id", doc_id).execute()
            
            if progress_callback:
                progress_callback("complete", 6, 6)
            
            stats.end_time = _utcnow()
            
            logger.info(
                f"Ingested {path.name}: {stats.parent_chunks_created} parents, "
                f"{stats.child_chunks_created} children, "
                f"{stats.child_chunks_deduplicated} deduped"
            )
            
            return IngestResult(
                success=True,
                document_id=doc_id,
                stats=stats,
            )
            
        except Exception as e:
            logger.exception(f"Ingestion failed for {file_path}: {e}")
            stats.errors.append(str(e))
            stats.end_time = _utcnow()
            
            return IngestResult(
                success=False,
                document_id=None,
                stats=stats,
                error=str(e),
            )
    
    def _get_mime_type(self, path: Path) -> str:
        """Get MIME type from file extension."""
        ext_map = {
            ".pdf": "application/pdf",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ".csv": "text/csv",
            ".txt": "text/plain",
            ".md": "text/markdown",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
        }
        return ext_map.get(path.suffix.lower(), "application/octet-stream")
    
    def _get_full_text(self, doc: LoadedDocument) -> str:
        """Extract full text from loaded document."""
        return "\n\n".join(page.text for page in doc.pages if page.text)
    
    async def _load_document(self, path: Path) -> LoadedDocument:
        """Load document using existing loader."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.loader.load, path)
    
    def _build_page_map(self, doc: LoadedDocument) -> List[Tuple[int, int, int]]:
        """Build page mapping: (start_char, end_char, page_num)."""
        page_map = []
        current_pos = 0
        
        for page in doc.pages:
            content_len = len(page.text) if page.text else 0
            page_map.append((current_pos, current_pos + content_len, page.page_number))
            current_pos += content_len + 1  # +1 for newline
        
        return page_map
    
    async def _store_chunks(
        self,
        doc_id: str,
        parents: List[ParentChunk],
        stats: IngestStats,
    ) -> None:
        """Store parent and child chunks in database."""
        
        # Collect all child texts for batch embedding
        all_children: List[Tuple[ParentChunk, ChildChunk]] = []
        for parent in parents:
            for child in parent.children:
                all_children.append((parent, child))
        
        # Batch embed all child chunks
        child_texts = [child.text for _, child in all_children]
        embeddings = await self.embedder.embed_texts_async(child_texts)
        
        # Check for dedup
        existing_hashes = set()
        if self.dedup_enabled and all_children:
            hashes = [child.content_hash for _, child in all_children]
            result = self.supabase.table("rag_child_chunks").select("content_hash").eq(
                "org_id", self.org_id
            ).in_("content_hash", hashes).execute()
            existing_hashes = {r["content_hash"] for r in result.data}
        
        # Store parent chunks
        for parent in parents:
            parent_data = {
                "id": str(uuid4()),
                "document_id": doc_id,
                "org_id": self.org_id,
                "index_in_document": parent.index_in_document,
                "text": parent.text,
                "token_count": parent.token_count,
                "page_start": parent.page_start,
                "page_end": parent.page_end,
                "section_heading": parent.section_heading,
                "ocr_confidence": parent.ocr_confidence,
                "metadata": parent.metadata,
            }
            
            result = self.supabase.table("rag_parent_chunks").insert(parent_data).execute()
            db_parent_id = result.data[0]["id"]
            
            # Store child chunks for this parent
            for i, (p, child) in enumerate(all_children):
                if p.id != parent.id:
                    continue
                
                # Check dedup
                if child.content_hash in existing_hashes:
                    stats.child_chunks_deduplicated += 1
                    continue
                
                embedding = embeddings[i]
                if embedding.error:
                    stats.errors.append(f"Embedding failed: {embedding.error}")
                    continue
                
                child_data = {
                    "id": str(uuid4()),
                    "parent_id": db_parent_id,
                    "document_id": doc_id,
                    "org_id": self.org_id,
                    "index_in_parent": child.index_in_parent,
                    "text": child.text,
                    "token_count": child.token_count,
                    "start_char_offset": child.start_char_offset,
                    "end_char_offset": child.end_char_offset,
                    "page": child.page,
                    "modality": child.modality.value,
                    "content_hash": child.content_hash,
                    "metadata": child.metadata,
                    "embedding_1024": embedding.embedding,
                }
                
                try:
                    self.supabase.table("rag_child_chunks").insert(child_data).execute()
                    stats.child_chunks_created += 1
                    stats.child_chunks_embedded += 1
                except Exception as e:
                    # Handle unique constraint violation (dedup)
                    if "duplicate" in str(e).lower() or "unique" in str(e).lower():
                        stats.child_chunks_deduplicated += 1
                    else:
                        raise


async def ingest_file(
    org_id: str,
    file_path: Union[str, Path],
    **kwargs: Any,
) -> IngestResult:
    """Convenience function to ingest a file."""
    ingestor = RAG2Ingestor(org_id=org_id)
    return await ingestor.ingest_file(file_path, **kwargs)
