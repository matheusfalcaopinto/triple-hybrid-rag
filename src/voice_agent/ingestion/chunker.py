"""
Text Chunking Module

Handles intelligent text chunking for RAG:
- Character-based chunking with overlap
- Table-aware chunking (tables kept intact)
- Heading preservation
- Semantic boundary detection
"""

import hashlib
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from voice_agent.ingestion.loader import LoadedDocument, PageContent

logger = logging.getLogger(__name__)


class ChunkType(Enum):
    """Type of content in a chunk."""
    TEXT = "text"
    TABLE = "table"
    IMAGE_CAPTION = "image_caption"
    IMAGE = "image"


@dataclass
class Chunk:
    """A single chunk of content ready for embedding."""
    content: str
    chunk_type: ChunkType
    page_number: int
    chunk_index: int
    source_document: str
    
    # Metadata
    content_hash: str = ""
    heading_context: str = ""  # Closest heading above this chunk
    is_table: bool = False
    table_context: str = ""  # Description of table contents
    alt_text: str = ""  # For image captions
    ocr_confidence: Optional[float] = None
    image_data: Optional[bytes] = None  # For image embedding
    
    def __post_init__(self):
        """Compute content hash after initialization."""
        if not self.content_hash:
            self.content_hash = self._compute_hash()
    
    def _compute_hash(self) -> str:
        """Compute SHA-256 hash of normalized content."""
        normalized = self._normalize_for_hash(self.content)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    
    @staticmethod
    def _normalize_for_hash(text: str) -> str:
        """Normalize text for consistent hashing."""
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)
        # Remove leading/trailing whitespace
        text = text.strip()
        # Convert to lowercase for case-insensitive dedup
        text = text.lower()
        return text


class Chunker:
    """
    Chunk documents for RAG embedding.
    
    Features:
    - Character-based chunking with configurable size and overlap
    - Table preservation (tables are never split)
    - Heading context tracking
    - Semantic boundary detection at sentence/paragraph breaks
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        preserve_tables: bool = True,
        min_chunk_size: int = 100,
        sentence_boundary: bool = True,
    ):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between consecutive chunks
            preserve_tables: Keep tables intact (don't split)
            min_chunk_size: Minimum chunk size (smaller chunks are merged)
            sentence_boundary: Try to break at sentence boundaries
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.preserve_tables = preserve_tables
        self.min_chunk_size = min_chunk_size
        self.sentence_boundary = sentence_boundary
        
        # Heading pattern (Markdown style)
        self.heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
        
        # Sentence boundary pattern
        self.sentence_end_pattern = re.compile(r"[.!?]\s+")
        
        # Table pattern
        self.table_pattern = re.compile(r"^\|.*\|$", re.MULTILINE)
    
    def chunk_document(self, document: LoadedDocument) -> List[Chunk]:
        """
        Chunk a loaded document into embedding-ready pieces.
        
        Args:
            document: LoadedDocument from the loader
            
        Returns:
            List of Chunk objects
        """
        all_chunks = []
        chunk_index = 0
        
        for page in document.pages:
            page_chunks = self._chunk_page(
                page=page,
                source_document=document.file_path,
                start_chunk_index=chunk_index,
            )
            all_chunks.extend(page_chunks)
            chunk_index += len(page_chunks)
        
        return all_chunks
    
    def _chunk_page(
        self,
        page: PageContent,
        source_document: str,
        start_chunk_index: int = 0,
    ) -> List[Chunk]:
        """Chunk a single page of content."""
        chunks = []
        chunk_index = start_chunk_index
        
        # First, handle tables separately if preserve_tables is True
        if self.preserve_tables and page.tables:
            for table in page.tables:
                table_chunk = Chunk(
                    content=table,
                    chunk_type=ChunkType.TABLE,
                    page_number=page.page_number,
                    chunk_index=chunk_index,
                    source_document=source_document,
                    is_table=True,
                    table_context=self._generate_table_context(table),
                )
                chunks.append(table_chunk)
                chunk_index += 1
        
        # Process main text
        text = page.text
        if not text or len(text.strip()) < self.min_chunk_size:
            # Page has no meaningful text (possibly scanned)
            if page.is_scanned and page.image_data:
                # Create placeholder for OCR processing
                chunks.append(Chunk(
                    content="[Scanned page - requires OCR]",
                    chunk_type=ChunkType.IMAGE,
                    page_number=page.page_number,
                    chunk_index=chunk_index,
                    source_document=source_document,
                    image_data=page.image_data,
                ))
            return chunks
        
        # Remove tables from text if they're handled separately
        if self.preserve_tables and page.tables:
            for table in page.tables:
                text = text.replace(table, "")
        
        # Chunk the remaining text
        text_chunks = self._chunk_text(
            text=text.strip(),
            page_number=page.page_number,
            source_document=source_document,
            start_index=chunk_index,
        )
        chunks.extend(text_chunks)
        
        return chunks
    
    def _chunk_text(
        self,
        text: str,
        page_number: int,
        source_document: str,
        start_index: int = 0,
    ) -> List[Chunk]:
        """
        Chunk text content with heading preservation and semantic boundaries.
        """
        chunks = []
        
        # Extract headings and their positions
        headings = list(self.heading_pattern.finditer(text))
        
        # Split text into segments based on headings
        segments = self._split_by_headings(text, headings)
        
        chunk_index = start_index
        for segment_text, heading_context in segments:
            segment_chunks = self._chunk_segment(
                text=segment_text,
                heading_context=heading_context,
                page_number=page_number,
                source_document=source_document,
                start_index=chunk_index,
            )
            chunks.extend(segment_chunks)
            chunk_index += len(segment_chunks)
        
        return chunks
    
    def _split_by_headings(
        self,
        text: str,
        headings: List[re.Match],
    ) -> List[tuple]:
        """Split text into segments based on headings."""
        if not headings:
            return [(text, "")]
        
        segments = []
        current_heading = ""
        last_end = 0
        
        for match in headings:
            # Text before this heading
            if match.start() > last_end:
                segment_text = text[last_end:match.start()].strip()
                if segment_text:
                    segments.append((segment_text, current_heading))
            
            # Update current heading
            current_heading = match.group(2).strip()
            last_end = match.end()
        
        # Remaining text after last heading
        if last_end < len(text):
            segment_text = text[last_end:].strip()
            if segment_text:
                segments.append((segment_text, current_heading))
        
        return segments if segments else [(text, "")]
    
    def _chunk_segment(
        self,
        text: str,
        heading_context: str,
        page_number: int,
        source_document: str,
        start_index: int = 0,
    ) -> List[Chunk]:
        """Chunk a text segment with overlap."""
        if len(text) <= self.chunk_size:
            # Entire segment fits in one chunk
            return [Chunk(
                content=text,
                chunk_type=ChunkType.TEXT,
                page_number=page_number,
                chunk_index=start_index,
                source_document=source_document,
                heading_context=heading_context,
            )]
        
        chunks = []
        chunk_index = start_index
        position = 0
        
        while position < len(text):
            # Determine chunk end
            end = min(position + self.chunk_size, len(text))
            
            # Try to break at sentence boundary
            if self.sentence_boundary and end < len(text):
                # Look for sentence end within the last 20% of chunk
                search_start = position + int(self.chunk_size * 0.8)
                search_text = text[search_start:end]
                
                matches = list(self.sentence_end_pattern.finditer(search_text))
                if matches:
                    # Use the last sentence boundary found
                    last_match = matches[-1]
                    end = search_start + last_match.end()
            
            # Extract chunk text
            chunk_text = text[position:end].strip()
            
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(Chunk(
                    content=chunk_text,
                    chunk_type=ChunkType.TEXT,
                    page_number=page_number,
                    chunk_index=chunk_index,
                    source_document=source_document,
                    heading_context=heading_context,
                ))
                chunk_index += 1
            
            # Move position with overlap
            position = end - self.chunk_overlap
            if position >= len(text) or end >= len(text):
                break
        
        return chunks
    
    def _generate_table_context(self, table_markdown: str) -> str:
        """Generate a brief description of table contents."""
        lines = table_markdown.strip().split("\n")
        
        if len(lines) < 2:
            return "Table data"
        
        # Get headers from first row
        header_line = lines[0]
        headers = [h.strip() for h in header_line.split("|") if h.strip()]
        
        # Count data rows (exclude header and separator)
        data_rows = len([l for l in lines[2:] if l.strip()])
        
        if headers:
            header_text = ", ".join(headers[:5])  # First 5 headers
            if len(headers) > 5:
                header_text += f", ... ({len(headers)} columns)"
            return f"Table with columns: {header_text}. {data_rows} rows."
        
        return f"Table with {data_rows} rows"
    
    def chunk_text_simple(
        self,
        text: str,
        source_document: str = "inline",
        page_number: int = 1,
    ) -> List[Chunk]:
        """
        Simple text chunking for direct text input (not from files).
        
        Args:
            text: Text to chunk
            source_document: Source identifier
            page_number: Page number for metadata
            
        Returns:
            List of Chunk objects
        """
        return self._chunk_text(
            text=text,
            page_number=page_number,
            source_document=source_document,
            start_index=0,
        )
