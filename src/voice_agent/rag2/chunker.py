"""
RAG 2.0 Hierarchical Chunker with Recursive Character Splitting

Creates parent/child chunk hierarchy using recursive splitting:
- Parent chunks: 800-1000 tokens (context for LLM)
- Child chunks: ~200 tokens (retrieval units)
- Recursive Character Splitting: \n\n → \n → . → space hierarchy
- Maintains parent-child linkage
- Preserves tables, headings, and narrative context
"""

import hashlib
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional, Tuple

from voice_agent.config import SETTINGS

logger = logging.getLogger(__name__)


# =============================================================================
# RECURSIVE CHARACTER SPLITTING CONSTANTS
# =============================================================================

# Separator hierarchy for recursive splitting (ordered by preference)
# Each level falls back to the next if chunks are still too large
SEPARATORS_HIERARCHY = [
    "\n\n\n",      # Triple newline (major section breaks)
    "\n\n",        # Double newline (paragraph breaks)
    "\n",          # Single newline (line breaks)
    ". ",          # Sentence end with space
    ".\n",         # Sentence end with newline
    "? ",          # Question end
    "! ",          # Exclamation end
    "; ",          # Semicolon
    ", ",          # Comma (last resort before space)
    " ",           # Space (word-level split, last resort)
]

# Special separators for markdown content
MARKDOWN_SEPARATORS = [
    "\n## ",       # H2 headers
    "\n### ",      # H3 headers  
    "\n#### ",     # H4 headers
    "\n- ",        # List items
    "\n* ",        # List items (alternative)
    "\n1. ",       # Numbered list
]


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


# =============================================================================
# RECURSIVE CHARACTER TEXT SPLITTER
# =============================================================================

class RecursiveCharacterTextSplitter:
    """
    LangChain-style Recursive Character Text Splitter.
    
    Splits text using a hierarchy of separators, falling back to smaller
    separators when chunks are still too large. This preserves semantic
    coherence by preferring to split at natural boundaries (paragraphs,
    sentences) rather than arbitrary character positions.
    """
    
    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        separators: Optional[List[str]] = None,
        keep_separator: bool = True,
        is_separator_regex: bool = False,
    ):
        """
        Initialize the recursive splitter.
        
        Args:
            chunk_size: Target size for chunks (in tokens)
            chunk_overlap: Overlap between chunks (in tokens)
            separators: Ordered list of separators to try
            keep_separator: Whether to keep separator in chunk
            is_separator_regex: Whether separators are regex patterns
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or SEPARATORS_HIERARCHY
        self.keep_separator = keep_separator
        self.is_separator_regex = is_separator_regex
    
    def _split_text_by_separator(
        self,
        text: str,
        separator: str,
    ) -> List[str]:
        """Split text by a single separator."""
        if self.is_separator_regex:
            parts = re.split(separator, text)
        else:
            parts = text.split(separator)
        
        # Re-add separator to end of each part (except last)
        if self.keep_separator and separator and not self.is_separator_regex:
            result = []
            for i, part in enumerate(parts):
                if i < len(parts) - 1:
                    result.append(part + separator)
                else:
                    result.append(part)
            return [p for p in result if p]
        
        return [p for p in parts if p]
    
    def _merge_splits(
        self,
        splits: List[str],
        separator: str = "",
    ) -> List[str]:
        """
        Merge splits into chunks respecting chunk_size.
        
        Combines small splits until they approach chunk_size,
        then starts a new chunk with overlap.
        """
        chunks = []
        current_chunk: List[str] = []
        current_length = 0
        
        for split in splits:
            split_length = estimate_tokens(split)
            
            # If adding this split would exceed chunk_size
            if current_length + split_length > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = separator.join(current_chunk)
                chunks.append(chunk_text)
                
                # Start new chunk with overlap
                overlap_tokens = 0
                overlap_parts: List[str] = []
                
                # Take parts from end of current chunk for overlap
                for part in reversed(current_chunk):
                    part_tokens = estimate_tokens(part)
                    if overlap_tokens + part_tokens <= self.chunk_overlap:
                        overlap_parts.insert(0, part)
                        overlap_tokens += part_tokens
                    else:
                        break
                
                current_chunk = overlap_parts
                current_length = overlap_tokens
            
            current_chunk.append(split)
            current_length += split_length
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_text = separator.join(current_chunk)
            chunks.append(chunk_text)
        
        return chunks
    
    def _split_text_recursive(
        self,
        text: str,
        separators: List[str],
    ) -> List[str]:
        """
        Recursively split text using separator hierarchy.
        
        Tries each separator in order. If resulting chunks are still
        too large, recursively splits them with the next separator.
        """
        if not text:
            return []
        
        # Check if text is already small enough
        if estimate_tokens(text) <= self.chunk_size:
            return [text]
        
        # No more separators, must split by character
        if not separators:
            # Last resort: split by character position
            chunks = []
            target_chars = self.chunk_size * 4
            current = 0
            while current < len(text):
                end = min(current + target_chars, len(text))
                chunks.append(text[current:end])
                current = end - (self.chunk_overlap * 4)  # Overlap
            return chunks
        
        # Try current separator
        separator = separators[0]
        remaining_separators = separators[1:]
        
        # Split by this separator
        splits = self._split_text_by_separator(text, separator)
        
        if len(splits) == 1:
            # Separator not found, try next
            return self._split_text_recursive(text, remaining_separators)
        
        # Recursively split any chunks that are still too large
        final_splits = []
        for split in splits:
            if estimate_tokens(split) > self.chunk_size:
                # This split is too large, recurse with remaining separators
                sub_splits = self._split_text_recursive(split, remaining_separators)
                final_splits.extend(sub_splits)
            else:
                final_splits.append(split)
        
        # Merge small splits back together
        return self._merge_splits(final_splits, "")
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks using recursive character splitting.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        return self._split_text_recursive(text, self.separators)
    
    def create_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
    ) -> List[Tuple[str, dict]]:
        """
        Create document chunks from a list of texts.
        
        Args:
            texts: List of texts to split
            metadatas: Optional list of metadata dicts
            
        Returns:
            List of (text, metadata) tuples
        """
        documents = []
        metadatas = metadatas or [{}] * len(texts)
        
        for text, metadata in zip(texts, metadatas):
            chunks = self.split_text(text)
            for chunk in chunks:
                documents.append((chunk, metadata.copy()))
        
        return documents


class HierarchicalChunker:
    """
    Create parent/child chunk hierarchy for RAG 2.0.
    
    Uses Recursive Character Splitting for semantic coherence:
    1. Split document into parent chunks (800-1000 tokens) using hierarchy
    2. Split each parent into child chunks (~200 tokens) for retrieval
    3. Maintain linkage and stable IDs
    4. Preserve tables and headings as atomic units
    """
    
    def __init__(
        self,
        parent_target_tokens: Optional[int] = None,
        parent_max_tokens: Optional[int] = None,
        child_target_tokens: Optional[int] = None,
        parent_overlap_tokens: Optional[int] = None,
        child_overlap_tokens: Optional[int] = None,
        preserve_tables: bool = True,
        use_markdown_separators: bool = True,
    ):
        """
        Initialize the chunker.
        
        Args:
            parent_target_tokens: Target tokens for parent chunks (default: 800)
            parent_max_tokens: Max tokens for parent chunks (default: 1000)
            child_target_tokens: Target tokens for child chunks (default: 200)
            parent_overlap_tokens: Overlap between parent chunks (default: 100)
            child_overlap_tokens: Overlap between child chunks (default: 50)
            preserve_tables: Keep tables as single chunks
            use_markdown_separators: Include markdown-specific separators
        """
        self.parent_target_tokens = parent_target_tokens or SETTINGS.rag2_parent_chunk_tokens
        self.parent_max_tokens = parent_max_tokens or SETTINGS.rag2_parent_chunk_max_tokens
        self.child_target_tokens = child_target_tokens or SETTINGS.rag2_child_chunk_tokens
        self.parent_overlap_tokens = parent_overlap_tokens or getattr(SETTINGS, 'rag2_parent_overlap_tokens', 100)
        self.child_overlap_tokens = child_overlap_tokens or getattr(SETTINGS, 'rag2_child_overlap_tokens', 50)
        self.preserve_tables = preserve_tables
        self.use_markdown_separators = use_markdown_separators
        
        # Build separator hierarchy
        separators = []
        if use_markdown_separators:
            separators.extend(MARKDOWN_SEPARATORS)
        separators.extend(SEPARATORS_HIERARCHY)
        
        # Initialize recursive splitters
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.parent_target_tokens,
            chunk_overlap=self.parent_overlap_tokens,
            separators=separators,
            keep_separator=True,
        )
        
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.child_target_tokens,
            chunk_overlap=self.child_overlap_tokens,
            separators=SEPARATORS_HIERARCHY,  # Don't use markdown for children
            keep_separator=True,
        )
        
        # Regex patterns
        self.table_pattern = re.compile(
            r'(\|[^\n]+\|\n)+',  # Markdown tables
            re.MULTILINE
        )
        self.heading_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    
    def _extract_tables(self, text: str) -> Tuple[str, List[Tuple[str, int]]]:
        """
        Extract tables from text, replacing with placeholders.
        
        Returns:
            (text_with_placeholders, [(table_content, original_position), ...])
        """
        tables = []
        offset = 0
        result_text = text
        
        for match in self.table_pattern.finditer(text):
            table_text = match.group()
            original_pos = match.start()
            placeholder = f"\n[[TABLE_{len(tables)}]]\n"
            
            # Replace in result text (accounting for offset from previous replacements)
            start = match.start() - offset
            end = match.end() - offset
            result_text = result_text[:start] + placeholder + result_text[end:]
            
            offset += len(table_text) - len(placeholder)
            tables.append((table_text, original_pos))
        
        return result_text, tables
    
    def _restore_tables(self, text: str, tables: List[Tuple[str, int]]) -> str:
        """Restore tables from placeholders."""
        result = text
        for i, (table_text, _) in enumerate(tables):
            placeholder = f"[[TABLE_{i}]]"
            result = result.replace(placeholder, table_text)
        return result
    
    def chunk_document(
        self,
        text: str,
        doc_hash: str,
        source_pages: Optional[List[Tuple[int, int, int]]] = None,
    ) -> List[ParentChunk]:
        """
        Chunk a document into parent/child hierarchy using recursive splitting.
        
        Args:
            text: Full document text
            doc_hash: Document hash for stable ID generation
            source_pages: Optional list of (start_char, end_char, page_num) for provenance
            
        Returns:
            List of ParentChunk objects, each containing ChildChunk objects
        """
        # Extract tables to preserve them
        if self.preserve_tables:
            text_no_tables, tables = self._extract_tables(text)
        else:
            text_no_tables = text
            tables = []
        
        # Extract headings for context tracking
        headings = []
        for match in self.heading_pattern.finditer(text):
            headings.append((match.start(), match.group(2).strip()))
        
        # Use recursive character splitting for parent chunks
        parent_texts = self.parent_splitter.split_text(text_no_tables)
        
        # Create parent chunks
        parents = []
        current_char_offset = 0
        
        for parent_idx, parent_text in enumerate(parent_texts):
            # Restore tables in this parent
            if tables:
                parent_text = self._restore_tables(parent_text, tables)
            
            parent_text = parent_text.strip()
            if not parent_text:
                continue
            
            # Find position in original text for metadata
            try:
                pos_in_original = text.find(parent_text[:100])  # First 100 chars
                if pos_in_original == -1:
                    pos_in_original = current_char_offset
            except Exception:
                pos_in_original = current_char_offset
            
            # Get metadata
            section_heading = self._get_current_heading(pos_in_original, headings)
            page_start = self._get_page_for_position(pos_in_original, source_pages)
            page_end = self._get_page_for_position(
                pos_in_original + len(parent_text) - 1, source_pages
            )
            
            parent = ParentChunk(
                id=f"{doc_hash}:{parent_idx}",
                index_in_document=parent_idx,
                text=parent_text,
                token_count=estimate_tokens(parent_text),
                page_start=page_start,
                page_end=page_end,
                section_heading=section_heading,
            )
            
            # Create child chunks using recursive splitting
            parent.children = self._create_child_chunks_recursive(
                parent=parent,
                doc_hash=doc_hash,
            )
            
            parents.append(parent)
            current_char_offset = pos_in_original + len(parent_text)
        
        logger.debug(f"Created {len(parents)} parent chunks using recursive splitting")
        return parents
    
    def _create_child_chunks_recursive(
        self,
        parent: ParentChunk,
        doc_hash: str,
    ) -> List[ChildChunk]:
        """Create child chunks from a parent using recursive splitting."""
        children = []
        
        # Use recursive splitter for children
        child_texts = self.child_splitter.split_text(parent.text)
        
        current_offset = 0
        for child_idx, child_text in enumerate(child_texts):
            child_text = child_text.strip()
            if not child_text:
                continue
            
            # Find offset in parent text
            try:
                offset_in_parent = parent.text.find(child_text[:50])
                if offset_in_parent == -1:
                    offset_in_parent = current_offset
            except Exception:
                offset_in_parent = current_offset
            
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
                start_char_offset=offset_in_parent,
                end_char_offset=offset_in_parent + len(child_text),
                page=parent.page_start,
                modality=modality,
            )
            
            children.append(child)
            current_offset = offset_in_parent + len(child_text)
        
        logger.debug(f"Created {len(children)} child chunks for parent {parent.id}")
        return children
    
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


def get_hierarchical_chunker(**kwargs: Any) -> HierarchicalChunker:
    """Get a configured hierarchical chunker instance."""
    return HierarchicalChunker(**kwargs)
