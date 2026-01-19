"""
Hierarchical Chunker for Triple-Hybrid-RAG

Two-level recursive character splitter:
1. Split document into PARENT chunks (800-1000 tokens)
2. Split each parent into CHILD chunks (~200 tokens)

Benefits:
- Child chunks are retrieved (smaller = more precise matches)
- Parent chunks provide context (larger = better understanding)
- When child is retrieved, we can expand to parent for LLM context

Optimizations:
- LRU cache for token estimation (avoid repeated tiktoken calls)
- Iterative work queue instead of recursion (stack safety + performance)
- O(n) child chunk offset search (track position, avoid O(n²))
"""

import hashlib
import logging
import re
from functools import lru_cache
from typing import Callable, List, Optional, Tuple
from uuid import UUID, uuid4

import tiktoken

from triple_hybrid_rag.config import RAGConfig, get_settings
from triple_hybrid_rag.types import ChildChunk, Modality, PageContent, ParentChunk

logger = logging.getLogger(__name__)

# Default separators for recursive splitting (ordered by priority)
DEFAULT_SEPARATORS = [
    "\n\n\n",      # Triple newline (major sections)
    "\n\n",        # Double newline (paragraphs)
    "\n",          # Single newline
    ". ",          # Sentence end
    "? ",          # Question end
    "! ",          # Exclamation end
    "; ",          # Semicolon
    ", ",          # Comma
    " ",           # Space
    "",            # Character-level (last resort)
]

# Global LRU cache for token estimation
# Cache size of 65536 entries should cover most chunking scenarios
# Uses character-based estimation for cached entries (len/4 approximation)
_TOKEN_CACHE_MAX_TEXT_LEN = 4000  # Only cache texts up to this length


@lru_cache(maxsize=65536)
def _estimate_tokens_cached(text: str) -> int:
    """
    Cached token estimation using character-based approximation.
    
    This provides a fast approximation for repeated token counts.
    For English text, ~4 characters per token is a reasonable estimate.
    """
    return len(text) // 4


class HierarchicalChunker:
    """
    Two-level hierarchical chunker for RAG.
    
    Creates parent chunks (800-1000 tokens) that contain child chunks (~200 tokens).
    Child chunks are the retrieval units with embeddings.
    Parent chunks provide context for LLM generation.
    """
    
    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        parent_chunk_tokens: Optional[int] = None,
        parent_chunk_max_tokens: Optional[int] = None,
        child_chunk_tokens: Optional[int] = None,
        chunk_overlap_tokens: Optional[int] = None,
        separators: Optional[List[str]] = None,
        encoding_name: str = "cl100k_base",
    ):
        """
        Initialize the hierarchical chunker.
        
        Args:
            config: RAGConfig instance (uses defaults if not provided)
            parent_chunk_tokens: Target parent chunk size in tokens
            parent_chunk_max_tokens: Maximum parent chunk size
            child_chunk_tokens: Target child chunk size in tokens
            chunk_overlap_tokens: Overlap between chunks
            separators: List of separators for recursive splitting
            encoding_name: Tiktoken encoding name
        """
        self.config = config or get_settings()
        
        # Use provided values or fall back to config
        self.parent_chunk_tokens = parent_chunk_tokens or self.config.rag_parent_chunk_tokens
        self.parent_chunk_max_tokens = parent_chunk_max_tokens or self.config.rag_parent_chunk_max_tokens
        self.child_chunk_tokens = child_chunk_tokens or self.config.rag_child_chunk_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens or self.config.rag_chunk_overlap_tokens
        
        self.separators = separators or DEFAULT_SEPARATORS
        
        # Initialize tokenizer
        try:
            self.encoding = tiktoken.get_encoding(encoding_name)
        except Exception:
            logger.warning(f"Failed to load encoding {encoding_name}, using cl100k_base")
            self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str, use_cache: bool = True) -> int:
        """
        Count tokens in text.
        
        Uses LRU cached estimation for small texts to improve performance.
        For larger texts or when precision is needed, uses tiktoken directly.
        
        Args:
            text: Text to count tokens for
            use_cache: Whether to use cached estimation for small texts
            
        Returns:
            Token count (exact or estimated)
        """
        if not text:
            return 0
        
        # Use cached estimation for small texts (fast path)
        if use_cache and len(text) <= _TOKEN_CACHE_MAX_TEXT_LEN:
            return _estimate_tokens_cached(text)
        
        # Use tiktoken for larger texts (accurate path)
        return len(self.encoding.encode(text))
    
    def count_tokens_exact(self, text: str) -> int:
        """Count tokens exactly using tiktoken (no caching)."""
        if not text:
            return 0
        return len(self.encoding.encode(text))
    
    def _compute_hash(self, text: str) -> str:
        """Compute SHA-256 hash of text for deduplication."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]
    
    def _recursive_split(
        self,
        text: str,
        target_tokens: int,
        max_tokens: int,
        separators: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Split text into chunks of target size using iterative work queue.
        
        OPTIMIZATION: Uses iterative approach instead of recursion to:
        - Avoid stack overflow on very large documents
        - Improve performance by reducing function call overhead
        
        Uses a hierarchical approach, trying larger separators first,
        then falling back to smaller ones if chunks are still too large.
        """
        if not text:
            return []
        
        separators = separators or self.separators
        
        # If text is already small enough, return as-is
        if self.count_tokens(text) <= max_tokens:
            return [text] if text.strip() else []
        
        # Work queue: (text_segment, separator_index)
        # Using list as stack (append/pop from end) for efficiency
        work_queue: List[Tuple[str, int]] = [(text, 0)]
        final_chunks: List[str] = []
        
        while work_queue:
            current_text, sep_idx = work_queue.pop()
            
            # Skip empty text
            if not current_text or not current_text.strip():
                continue
            
            # If text is small enough, add to final chunks
            if self.count_tokens(current_text) <= max_tokens:
                if current_text.strip():
                    final_chunks.append(current_text)
                continue
            
            # If we've exhausted all separators, split by tokens
            if sep_idx >= len(separators):
                token_chunks = self._split_by_tokens(current_text, target_tokens, max_tokens)
                final_chunks.extend(token_chunks)
                continue
            
            sep = separators[sep_idx]
            
            # Empty separator means split by tokens
            if sep == "":
                token_chunks = self._split_by_tokens(current_text, target_tokens, max_tokens)
                final_chunks.extend(token_chunks)
                continue
            
            # Try splitting with current separator
            if sep not in current_text:
                # Separator not found, try next separator
                work_queue.append((current_text, sep_idx + 1))
                continue
            
            # Split and merge small pieces
            splits = current_text.split(sep)
            merged_chunks = []
            current_chunk = ""
            
            for split in splits:
                split = split.strip()
                if not split:
                    continue
                
                # Check if adding this split would exceed max
                test_chunk = f"{current_chunk}{sep}{split}" if current_chunk else split
                test_tokens = self.count_tokens(test_chunk)
                
                if test_tokens <= max_tokens:
                    current_chunk = test_chunk
                else:
                    # Save current chunk if it has content
                    if current_chunk:
                        merged_chunks.append(current_chunk)
                    current_chunk = split
            
            if current_chunk:
                merged_chunks.append(current_chunk)
            
            # Process merged chunks - add small ones to final, queue large ones
            for chunk in merged_chunks:
                if self.count_tokens(chunk) <= max_tokens:
                    if chunk.strip():
                        final_chunks.append(chunk)
                else:
                    # Queue for further splitting with finer separator
                    work_queue.append((chunk, sep_idx + 1))
        
        return final_chunks
    
    def _split_by_tokens(
        self,
        text: str,
        target_tokens: int,
        max_tokens: int,
    ) -> List[str]:
        """Split text into chunks by token count (last resort)."""
        tokens = self.encoding.encode(text)
        chunks = []
        
        for i in range(0, len(tokens), target_tokens):
            chunk_tokens = tokens[i:i + max_tokens]
            chunk_text = self.encoding.decode(chunk_tokens)
            if chunk_text.strip():
                chunks.append(chunk_text)
        
        return chunks
    
    def _apply_overlap(
        self,
        chunks: List[str],
        overlap_tokens: int,
    ) -> List[str]:
        """Apply overlap between consecutive chunks."""
        if overlap_tokens <= 0 or len(chunks) <= 1:
            return chunks
        
        overlapped = []
        prev_suffix = ""
        
        for i, chunk in enumerate(chunks):
            # Add suffix from previous chunk as prefix
            if prev_suffix:
                chunk = prev_suffix + " " + chunk
            
            overlapped.append(chunk)
            
            # Extract suffix for next chunk
            tokens = self.encoding.encode(chunk)
            if len(tokens) > overlap_tokens:
                suffix_tokens = tokens[-overlap_tokens:]
                prev_suffix = self.encoding.decode(suffix_tokens)
            else:
                prev_suffix = ""
        
        return overlapped
    
    def _extract_section_heading(self, text: str) -> Optional[str]:
        """Extract section heading from text if present."""
        # Look for common heading patterns
        patterns = [
            r'^#+\s+(.+?)$',           # Markdown headings
            r'^([A-Z][A-Z\s]+)$',      # ALL CAPS headings
            r'^(\d+\.?\s+.+?)$',       # Numbered headings
            r'^(Chapter\s+\d+.*)$',    # Chapter headings
            r'^(Section\s+\d+.*)$',    # Section headings
        ]
        
        first_line = text.split('\n')[0].strip()
        
        for pattern in patterns:
            match = re.match(pattern, first_line, re.MULTILINE | re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def split_into_parents(
        self,
        text: str,
        document_id: UUID,
        tenant_id: str = "",
        pages: Optional[List[PageContent]] = None,
    ) -> List[ParentChunk]:
        """
        Split text into parent chunks.
        
        Args:
            text: Full document text
            document_id: UUID of the source document
            tenant_id: Tenant ID for multi-tenancy
            pages: Optional page content for metadata
            
        Returns:
            List of ParentChunk objects
        """
        # Split into parent-sized chunks
        chunks_text = self._recursive_split(
            text,
            target_tokens=self.parent_chunk_tokens,
            max_tokens=self.parent_chunk_max_tokens,
        )
        
        # Apply overlap
        chunks_text = self._apply_overlap(chunks_text, self.chunk_overlap_tokens)
        
        # Create ParentChunk objects
        parent_chunks = []
        char_offset = 0
        
        for i, chunk_text in enumerate(chunks_text):
            if not chunk_text.strip():
                continue
            
            # Calculate page range if page info available
            page_start = None
            page_end = None
            ocr_confidence = None
            
            if pages:
                # Find pages that overlap with this chunk
                chunk_start = text.find(chunk_text, char_offset)
                chunk_end = chunk_start + len(chunk_text)
                
                current_pos = 0
                for page in pages:
                    page_len = len(page.text)
                    page_end_pos = current_pos + page_len
                    
                    if current_pos <= chunk_start < page_end_pos:
                        page_start = page.page_number
                    if current_pos < chunk_end <= page_end_pos:
                        page_end = page.page_number
                    if page.ocr_confidence is not None:
                        if ocr_confidence is None:
                            ocr_confidence = page.ocr_confidence
                        else:
                            ocr_confidence = min(ocr_confidence, page.ocr_confidence)
                    
                    current_pos = page_end_pos
            
            parent_chunk = ParentChunk(
                id=uuid4(),
                document_id=document_id,
                tenant_id=tenant_id,
                index_in_document=i,
                text=chunk_text,
                token_count=self.count_tokens(chunk_text),
                page_start=page_start,
                page_end=page_end,
                section_heading=self._extract_section_heading(chunk_text),
                ocr_confidence=ocr_confidence,
            )
            
            parent_chunks.append(parent_chunk)
            char_offset = text.find(chunk_text, char_offset) + len(chunk_text)
        
        logger.debug(f"Created {len(parent_chunks)} parent chunks from document {document_id}")
        return parent_chunks
    
    def split_parent_into_children(
        self,
        parent: ParentChunk,
    ) -> List[ChildChunk]:
        """
        Split a parent chunk into child chunks.
        
        OPTIMIZATION: Uses O(n) offset search by:
        - Tracking current search position (avoids searching from start)
        - Using first 50 chars as search key (faster than full text match)
        
        Args:
            parent: ParentChunk to split
            
        Returns:
            List of ChildChunk objects
        """
        # Split into child-sized chunks
        chunks_text = self._recursive_split(
            parent.text,
            target_tokens=self.child_chunk_tokens,
            max_tokens=int(self.child_chunk_tokens * 1.2),  # 20% buffer
        )
        
        # Create ChildChunk objects with O(n) offset tracking
        child_chunks = []
        current_search_pos = 0  # Track position to avoid O(n²) searching
        parent_text = parent.text
        
        for i, chunk_text in enumerate(chunks_text):
            chunk_text_stripped = chunk_text.strip()
            if not chunk_text_stripped:
                continue
            
            # OPTIMIZATION: Use search key (first 50 chars) for faster matching
            # This avoids comparing the full chunk text each time
            search_key = chunk_text_stripped[:50] if len(chunk_text_stripped) > 50 else chunk_text_stripped
            
            # Search from current position (O(n) total instead of O(n²))
            start_offset = parent_text.find(search_key, current_search_pos)
            
            if start_offset == -1:
                # Fallback: search from beginning (shouldn't happen normally)
                start_offset = parent_text.find(search_key)
                if start_offset == -1:
                    # Last resort: use current position
                    start_offset = current_search_pos
            
            end_offset = start_offset + len(chunk_text_stripped)
            
            child_chunk = ChildChunk(
                id=uuid4(),
                parent_id=parent.id,
                document_id=parent.document_id,
                tenant_id=parent.tenant_id,
                index_in_parent=i,
                text=chunk_text_stripped,
                token_count=self.count_tokens(chunk_text_stripped),
                start_char_offset=start_offset,
                end_char_offset=end_offset,
                page=parent.page_start,  # Use parent's page start
                modality=Modality.TEXT,
                content_hash=self._compute_hash(chunk_text_stripped),
            )
            
            child_chunks.append(child_chunk)
            
            # Update search position for next iteration
            current_search_pos = end_offset
        
        return child_chunks
    
    def split_document(
        self,
        text: str,
        document_id: UUID,
        tenant_id: str = "",
        pages: Optional[List[PageContent]] = None,
    ) -> Tuple[List[ParentChunk], List[ChildChunk]]:
        """
        Split a document into hierarchical parent and child chunks.
        
        Args:
            text: Full document text
            document_id: UUID of the source document
            tenant_id: Tenant ID for multi-tenancy
            pages: Optional page content for metadata
            
        Returns:
            Tuple of (parent_chunks, child_chunks)
        """
        # First create parent chunks
        parent_chunks = self.split_into_parents(text, document_id, tenant_id, pages)
        
        # Then split each parent into children
        all_children = []
        for parent in parent_chunks:
            children = self.split_parent_into_children(parent)
            parent.children = children  # Link children to parent
            all_children.extend(children)
        
        logger.info(
            f"Document {document_id}: {len(parent_chunks)} parents, "
            f"{len(all_children)} children"
        )
        
        return parent_chunks, all_children
    
    def create_image_chunk(
        self,
        image_data: bytes,
        parent: ParentChunk,
        index_in_parent: int,
        alt_text: Optional[str] = None,
    ) -> ChildChunk:
        """
        Create a child chunk for an image.
        
        Args:
            image_data: Raw image bytes
            parent: Parent chunk containing the image
            index_in_parent: Index within parent
            alt_text: Alternative text description
            
        Returns:
            ChildChunk with image data
        """
        return ChildChunk(
            id=uuid4(),
            parent_id=parent.id,
            document_id=parent.document_id,
            tenant_id=parent.tenant_id,
            index_in_parent=index_in_parent,
            text=alt_text or "[Image]",
            token_count=self.count_tokens(alt_text or "[Image]"),
            page=parent.page_start,
            modality=Modality.IMAGE,
            content_hash=self._compute_hash(str(image_data[:100])),
            image_data=image_data,
        )
