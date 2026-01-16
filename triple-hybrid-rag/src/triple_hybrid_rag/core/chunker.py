"""
Hierarchical Chunker for Triple-Hybrid-RAG

Two-level recursive character splitter:
1. Split document into PARENT chunks (800-1000 tokens)
2. Split each parent into CHILD chunks (~200 tokens)

Benefits:
- Child chunks are retrieved (smaller = more precise matches)
- Parent chunks provide context (larger = better understanding)
- When child is retrieved, we can expand to parent for LLM context
"""

import hashlib
import logging
import re
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
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
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
        Recursively split text into chunks of target size.
        
        Uses a hierarchical approach, trying larger separators first,
        then falling back to smaller ones if chunks are still too large.
        """
        if not text:
            return []
        
        separators = separators or self.separators
        
        # If text is already small enough, return as-is
        if self.count_tokens(text) <= max_tokens:
            return [text] if text.strip() else []
        
        # Try each separator
        for sep in separators:
            if sep == "":
                # Last resort: split by characters
                return self._split_by_tokens(text, target_tokens, max_tokens)
            
            if sep in text:
                splits = text.split(sep)
                
                # Merge small splits and split large ones
                chunks = []
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
                            chunks.append(current_chunk)
                        
                        # Check if split itself is too large
                        if self.count_tokens(split) > max_tokens:
                            # Recursively split with finer separators
                            remaining_seps = separators[separators.index(sep) + 1:]
                            sub_chunks = self._recursive_split(
                                split, target_tokens, max_tokens, remaining_seps
                            )
                            chunks.extend(sub_chunks)
                            current_chunk = ""
                        else:
                            current_chunk = split
                
                if current_chunk:
                    chunks.append(current_chunk)
                
                if chunks:
                    return chunks
        
        # Fallback: split by tokens
        return self._split_by_tokens(text, target_tokens, max_tokens)
    
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
        
        # Create ChildChunk objects
        child_chunks = []
        char_offset = 0
        
        for i, chunk_text in enumerate(chunks_text):
            if not chunk_text.strip():
                continue
            
            # Calculate character offsets
            start_offset = parent.text.find(chunk_text, char_offset)
            end_offset = start_offset + len(chunk_text)
            
            child_chunk = ChildChunk(
                id=uuid4(),
                parent_id=parent.id,
                document_id=parent.document_id,
                tenant_id=parent.tenant_id,
                index_in_parent=i,
                text=chunk_text,
                token_count=self.count_tokens(chunk_text),
                start_char_offset=start_offset,
                end_char_offset=end_offset,
                page=parent.page_start,  # Use parent's page start
                modality=Modality.TEXT,
                content_hash=self._compute_hash(chunk_text),
            )
            
            child_chunks.append(child_chunk)
            char_offset = end_offset
        
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
