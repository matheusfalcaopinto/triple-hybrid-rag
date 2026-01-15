"""
RAG 2.0 Hierarchical Chunker

Creates parent/child chunk hierarchy:
- Parent chunks: 800-1000 tokens (context for LLM)
- Child chunks: ~200 tokens (retrieval units)
- Maintains parent-child linkage
- Preserves tables and heading context
"""

import hashlib
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional, Tuple

from voice_agent.config import SETTINGS

logger = logging.getLogger(__name__)


class ChunkModality(Enum):
    """Type of content in a chunk."""
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    IMAGE_CAPTION = "image_caption"


@dataclass
class ParentChunk:
    """A parent chunk (800-1000 tokens) for LLM context."""
    id: str  # Stable ID: doc_hash:parent_idx
    index_in_document: int
    text: str
    token_count: int
    page_start: int
    page_end: int
    section_heading: Optional[str] = None
    ocr_confidence: Optional[float] = None
    metadata: dict = field(default_factory=dict)
    
    # Child chunks derived from this parent
    children: List["ChildChunk"] = field(default_factory=list)


@dataclass
class ChildChunk:
    """A child chunk (~200 tokens) for retrieval."""
    id: str  # Stable ID: doc_hash:parent_idx:child_idx
    parent_id: str
    index_in_parent: int
    text: str
    token_count: int
    start_char_offset: int
    end_char_offset: int
    page: int
    modality: ChunkModality = ChunkModality.TEXT
    content_hash: str = ""
    metadata: dict = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Compute content hash after initialization."""
        if not self.content_hash:
            self.content_hash = self._compute_hash()
    
    def _compute_hash(self) -> str:
        """Compute SHA-256 hash of normalized content."""
        normalized = self._normalize_for_hash(self.text)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    
    @staticmethod
    def _normalize_for_hash(text: str) -> str:
        """Normalize text for consistent hashing."""
        text = re.sub(r"\s+", " ", text)
        text = text.strip().lower()
        return text


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text.
    
    Uses simple heuristic: ~4 chars per token for English/Portuguese.
    For production, consider using tiktoken or the model's tokenizer.
    """
    return len(text) // 4


def find_sentence_boundary(text: str, target_pos: int, window: int = 100) -> int:
    """
    Find the nearest sentence boundary near target_pos.
    
    Args:
        text: Full text
        target_pos: Target character position
        window: Window to search for boundary
        
    Returns:
        Position of sentence boundary (or target_pos if none found)
    """
    # Search backward for sentence end
    start = max(0, target_pos - window)
    end = min(len(text), target_pos + window)
    
    search_region = text[start:end]
    
    # Find sentence-ending punctuation
    sentence_ends = []
    for match in re.finditer(r'[.!?]\s+', search_region):
        absolute_pos = start + match.end()
        sentence_ends.append(absolute_pos)
    
    if not sentence_ends:
        return target_pos
    
    # Find closest to target
    closest = min(sentence_ends, key=lambda x: abs(x - target_pos))
    return closest


class HierarchicalChunker:
    """
    Create parent/child chunk hierarchy for RAG 2.0.
    
    Workflow:
    1. Split document into parent chunks (800-1000 tokens)
    2. Split each parent into child chunks (~200 tokens)
    3. Maintain linkage and stable IDs
    """
    
    def __init__(
        self,
        parent_target_tokens: Optional[int] = None,
        parent_max_tokens: Optional[int] = None,
        child_target_tokens: Optional[int] = None,
        preserve_tables: bool = True,
    ):
        """
        Initialize the chunker.
        
        Args:
            parent_target_tokens: Target tokens for parent chunks (default: 800)
            parent_max_tokens: Max tokens for parent chunks (default: 1000)
            child_target_tokens: Target tokens for child chunks (default: 200)
            preserve_tables: Keep tables as single chunks
        """
        self.parent_target_tokens = parent_target_tokens or SETTINGS.rag2_parent_chunk_tokens
        self.parent_max_tokens = parent_max_tokens or SETTINGS.rag2_parent_chunk_max_tokens
        self.child_target_tokens = child_target_tokens or SETTINGS.rag2_child_chunk_tokens
        self.preserve_tables = preserve_tables
        
        # Regex patterns
        self.table_pattern = re.compile(
            r'(\|[^\n]+\|\n)+',  # Markdown tables
            re.MULTILINE
        )
        self.heading_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    
    def chunk_document(
        self,
        text: str,
        doc_hash: str,
        source_pages: Optional[List[Tuple[int, int, int]]] = None,
    ) -> List[ParentChunk]:
        """
        Chunk a document into parent/child hierarchy.
        
        Args:
            text: Full document text
            doc_hash: Document hash for stable ID generation
            source_pages: Optional list of (start_char, end_char, page_num) for provenance
            
        Returns:
            List of ParentChunk objects, each containing ChildChunk objects
        """
        # Extract and preserve tables
        tables = list(self.table_pattern.finditer(text))
        table_ranges = [(m.start(), m.end()) for m in tables]
        
        # Extract headings for context
        headings = []
        for match in self.heading_pattern.finditer(text):
            headings.append((match.start(), match.group(2).strip()))
        
        # Create parent chunks
        parents = self._create_parent_chunks(
            text=text,
            doc_hash=doc_hash,
            table_ranges=table_ranges,
            headings=headings,
            source_pages=source_pages,
        )
        
        # Create child chunks for each parent
        for parent in parents:
            parent.children = self._create_child_chunks(
                parent=parent,
                table_ranges=table_ranges,
            )
        
        return parents
    
    def _get_current_heading(self, position: int, headings: List[Tuple[int, str]]) -> Optional[str]:
        """Get the most recent heading before a position."""
        current_heading = None
        for heading_pos, heading_text in headings:
            if heading_pos <= position:
                current_heading = heading_text
            else:
                break
        return current_heading
    
    def _get_page_for_position(
        self,
        position: int,
        source_pages: Optional[List[Tuple[int, int, int]]]
    ) -> int:
        """Get page number for a character position."""
        if not source_pages:
            return 1
        
        for start, end, page in source_pages:
            if start <= position <= end:
                return page
        
        return 1
    
    def _is_in_table(self, position: int, table_ranges: List[Tuple[int, int]]) -> bool:
        """Check if position is inside a table."""
        for start, end in table_ranges:
            if start <= position <= end:
                return True
        return False
    
    def _create_parent_chunks(
        self,
        text: str,
        doc_hash: str,
        table_ranges: List[Tuple[int, int]],
        headings: List[Tuple[int, str]],
        source_pages: Optional[List[Tuple[int, int, int]]],
    ) -> List[ParentChunk]:
        """Create parent chunks from document text."""
        parents = []
        current_pos = 0
        parent_idx = 0
        
        while current_pos < len(text):
            # Calculate target end position
            target_chars = self.parent_target_tokens * 4  # ~4 chars per token
            max_chars = self.parent_max_tokens * 4
            
            target_end = current_pos + target_chars
            max_end = current_pos + max_chars
            
            # Don't split in the middle of a table
            for table_start, table_end in table_ranges:
                if current_pos < table_start < target_end < table_end:
                    # Table starts before target but doesn't end
                    target_end = table_end
                elif table_start <= current_pos < table_end:
                    # We're inside a table, extend to table end
                    target_end = table_end
            
            # Find sentence boundary near target
            if target_end < len(text):
                target_end = find_sentence_boundary(text, target_end)
            
            # Clamp to max
            target_end = min(target_end, max_end, len(text))
            
            # Extract parent text
            parent_text = text[current_pos:target_end].strip()
            
            if not parent_text:
                current_pos = target_end
                continue
            
            # Get metadata
            section_heading = self._get_current_heading(current_pos, headings)
            page_start = self._get_page_for_position(current_pos, source_pages)
            page_end = self._get_page_for_position(target_end - 1, source_pages)
            
            parent = ParentChunk(
                id=f"{doc_hash}:{parent_idx}",
                index_in_document=parent_idx,
                text=parent_text,
                token_count=estimate_tokens(parent_text),
                page_start=page_start,
                page_end=page_end,
                section_heading=section_heading,
            )
            
            parents.append(parent)
            parent_idx += 1
            current_pos = target_end
        
        logger.debug(f"Created {len(parents)} parent chunks")
        return parents
    
    def _create_child_chunks(
        self,
        parent: ParentChunk,
        table_ranges: List[Tuple[int, int]],
    ) -> List[ChildChunk]:
        """Create child chunks from a parent chunk."""
        children = []
        text = parent.text
        current_pos = 0
        child_idx = 0
        
        # Target child size
        target_chars = self.child_target_tokens * 4
        
        while current_pos < len(text):
            target_end = current_pos + target_chars
            
            # Find sentence boundary
            if target_end < len(text):
                target_end = find_sentence_boundary(text, target_end, window=50)
            
            target_end = min(target_end, len(text))
            
            child_text = text[current_pos:target_end].strip()
            
            if not child_text:
                current_pos = target_end
                continue
            
            # Determine modality
            modality = ChunkModality.TEXT
            if self.table_pattern.search(child_text):
                modality = ChunkModality.TABLE
            
            child = ChildChunk(
                id=f"{parent.id}:{child_idx}",
                parent_id=parent.id,
                index_in_parent=child_idx,
                text=child_text,
                token_count=estimate_tokens(child_text),
                start_char_offset=current_pos,
                end_char_offset=target_end,
                page=parent.page_start,  # Approximate
                modality=modality,
            )
            
            children.append(child)
            child_idx += 1
            current_pos = target_end
        
        logger.debug(f"Created {len(children)} child chunks for parent {parent.id}")
        return children


def get_hierarchical_chunker(**kwargs: Any) -> HierarchicalChunker:
    """Get a configured hierarchical chunker instance."""
    return HierarchicalChunker(**kwargs)
