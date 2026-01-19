"""
Semantic Chunking with Boundary Detection

Uses embedding similarity to detect natural topic boundaries
and create semantically coherent chunks.

Features:
- Embedding-based boundary detection
- Adaptive chunk sizing based on content
- Structure-aware splitting (preserves headers, lists, code blocks)
- Table and figure preservation
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Callable, Any
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)

class ContentType(Enum):
    """Types of content that require special handling."""
    TEXT = "text"
    HEADING = "heading"
    LIST = "list"
    CODE = "code"
    TABLE = "table"
    QUOTE = "quote"
    FIGURE = "figure"

@dataclass
class ContentBlock:
    """A block of content with metadata."""
    text: str
    content_type: ContentType
    level: int = 0  # Heading level, list nesting, etc.
    start_idx: int = 0
    end_idx: int = 0
    metadata: dict = field(default_factory=dict)
    
    @property
    def word_count(self) -> int:
        return len(self.text.split())
    
    @property
    def char_count(self) -> int:
        return len(self.text)

@dataclass
class SemanticChunk:
    """A semantically coherent chunk of content."""
    text: str
    chunk_index: int
    start_block: int
    end_block: int
    
    # Scores and metadata
    coherence_score: float = 0.0
    boundary_confidence: float = 0.0
    
    # Content analysis
    primary_content_type: ContentType = ContentType.TEXT
    has_code: bool = False
    has_table: bool = False
    has_list: bool = False
    
    # Token/word counts
    word_count: int = 0
    estimated_tokens: int = 0
    
    # Hash for deduplication
    content_hash: str = ""
    
    metadata: dict = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.content_hash and self.text:
            self.content_hash = hashlib.md5(self.text.encode()).hexdigest()[:16]
        if not self.word_count:
            self.word_count = len(self.text.split())
        if not self.estimated_tokens:
            # Rough estimate: 1 token ≈ 0.75 words for English
            self.estimated_tokens = int(self.word_count / 0.75)

@dataclass
class SemanticChunkerConfig:
    """Configuration for semantic chunking."""
    
    # Target chunk sizes (in tokens)
    target_chunk_tokens: int = 512
    min_chunk_tokens: int = 100
    max_chunk_tokens: int = 1024
    
    # Boundary detection
    similarity_threshold: float = 0.5  # Below this = boundary
    window_size: int = 3  # Sentences to compare
    
    # Structure preservation
    preserve_code_blocks: bool = True
    preserve_tables: bool = True
    preserve_lists: bool = True
    split_on_headings: bool = True
    
    # Overlap for context
    overlap_tokens: int = 50
    
    # Embedding settings
    use_embeddings: bool = True
    embedding_batch_size: int = 32

class StructureParser:
    """Parse document structure into content blocks."""
    
    # Patterns for structure detection
    HEADING_PATTERNS = [
        (r'^#{1,6}\s+(.+)$', 'markdown_heading'),
        (r'^(.+)\n[=]{3,}$', 'underline_h1'),
        (r'^(.+)\n[-]{3,}$', 'underline_h2'),
    ]
    
    CODE_BLOCK_PATTERN = r'```[\w]*\n(.*?)```'
    INLINE_CODE_PATTERN = r'`[^`]+`'
    
    TABLE_PATTERNS = [
        r'\|.+\|',  # Markdown tables
        r'^\s*\+[-+]+\+\s*$',  # ASCII tables
    ]
    
    LIST_PATTERNS = [
        r'^\s*[-*+]\s+',  # Unordered
        r'^\s*\d+\.\s+',  # Ordered
    ]
    
    QUOTE_PATTERN = r'^\s*>\s+'
    
    def parse(self, text: str) -> List[ContentBlock]:
        """Parse text into content blocks."""
        blocks = []
        
        # First, extract special blocks that shouldn't be split
        text, code_blocks = self._extract_code_blocks(text)
        text, table_blocks = self._extract_tables(text)
        
        # Split remaining text into paragraphs and analyze
        paragraphs = self._split_paragraphs(text)
        
        current_idx = 0
        for para in paragraphs:
            if not para.strip():
                continue
            
            content_type, level = self._detect_content_type(para)
            
            block = ContentBlock(
                text=para,
                content_type=content_type,
                level=level,
                start_idx=current_idx,
                end_idx=current_idx + len(para),
            )
            blocks.append(block)
            current_idx += len(para) + 1
        
        # Re-insert code and table blocks at appropriate positions
        blocks = self._merge_special_blocks(blocks, code_blocks, table_blocks)
        
        return blocks
    
    def _extract_code_blocks(self, text: str) -> Tuple[str, List[ContentBlock]]:
        """Extract code blocks from text."""
        code_blocks = []
        
        def replace_code(match):
            code_content = match.group(0)
            placeholder = f"__CODE_BLOCK_{len(code_blocks)}__"
            code_blocks.append(ContentBlock(
                text=code_content,
                content_type=ContentType.CODE,
                metadata={"original_match": match.group(0)},
            ))
            return placeholder
        
        modified_text = re.sub(
            self.CODE_BLOCK_PATTERN,
            replace_code,
            text,
            flags=re.MULTILINE | re.DOTALL,
        )
        
        return modified_text, code_blocks
    
    def _extract_tables(self, text: str) -> Tuple[str, List[ContentBlock]]:
        """Extract tables from text."""
        table_blocks = []
        lines = text.split('\n')
        modified_lines = []
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Check if this looks like a table
            if self._is_table_line(line):
                table_lines = [line]
                i += 1
                
                # Collect all table lines
                while i < len(lines) and self._is_table_line(lines[i]):
                    table_lines.append(lines[i])
                    i += 1
                
                if len(table_lines) >= 2:  # At least 2 lines for a table
                    table_text = '\n'.join(table_lines)
                    placeholder = f"__TABLE_BLOCK_{len(table_blocks)}__"
                    table_blocks.append(ContentBlock(
                        text=table_text,
                        content_type=ContentType.TABLE,
                    ))
                    modified_lines.append(placeholder)
                else:
                    modified_lines.extend(table_lines)
            else:
                modified_lines.append(line)
                i += 1
        
        return '\n'.join(modified_lines), table_blocks
    
    def _is_table_line(self, line: str) -> bool:
        """Check if a line is part of a table."""
        return bool(re.match(r'^\s*\|.*\|\s*$', line)) or \
               bool(re.match(r'^\s*\+[-+]+\+\s*$', line)) or \
               bool(re.match(r'^\s*[-|:]+\s*$', line))
    
    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        # Split on double newlines or before headings
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _detect_content_type(self, text: str) -> Tuple[ContentType, int]:
        """Detect the content type and level of a text block."""
        text_stripped = text.strip()
        
        # Check for headings
        for pattern, _ in self.HEADING_PATTERNS:
            match = re.match(pattern, text_stripped, re.MULTILINE)
            if match:
                # Count # for level
                level = 1
                if text_stripped.startswith('#'):
                    hash_match = re.match(r'^#+', text_stripped)
                    if hash_match:
                        level = len(hash_match.group())
                return ContentType.HEADING, level
        
        # Check for lists
        lines = text_stripped.split('\n')
        list_lines = sum(1 for line in lines if re.match(r'^\s*[-*+\d.]+\s+', line))
        if list_lines > len(lines) * 0.5:  # More than half are list items
            return ContentType.LIST, 0
        
        # Check for quotes
        if re.match(self.QUOTE_PATTERN, text_stripped):
            return ContentType.QUOTE, 0
        
        return ContentType.TEXT, 0
    
    def _merge_special_blocks(
        self,
        blocks: List[ContentBlock],
        code_blocks: List[ContentBlock],
        table_blocks: List[ContentBlock],
    ) -> List[ContentBlock]:
        """Merge special blocks back into the main block list."""
        result = []
        
        for block in blocks:
            # Check for code block placeholders
            code_match = re.search(r'__CODE_BLOCK_(\d+)__', block.text)
            if code_match:
                idx = int(code_match.group(1))
                if idx < len(code_blocks):
                    result.append(code_blocks[idx])
                continue
            
            # Check for table block placeholders
            table_match = re.search(r'__TABLE_BLOCK_(\d+)__', block.text)
            if table_match:
                idx = int(table_match.group(1))
                if idx < len(table_blocks):
                    result.append(table_blocks[idx])
                continue
            
            result.append(block)
        
        return result

class SemanticBoundaryDetector:
    """Detect semantic boundaries using embeddings."""
    
    def __init__(
        self,
        config: SemanticChunkerConfig,
        embed_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
    ):
        self.config = config
        self.embed_fn = embed_fn
    
    def detect_boundaries(
        self,
        blocks: List[ContentBlock],
    ) -> List[Tuple[int, float]]:
        """
        Detect semantic boundaries between blocks.
        
        Returns:
            List of (block_index, boundary_confidence) tuples
        """
        if len(blocks) < 2:
            return []
        
        boundaries = []
        
        # Always create boundaries at headings
        for i, block in enumerate(blocks):
            if block.content_type == ContentType.HEADING:
                boundaries.append((i, 1.0))
        
        # Use embedding similarity if available
        if self.config.use_embeddings and self.embed_fn:
            embedding_boundaries = self._detect_embedding_boundaries(blocks)
            boundaries.extend(embedding_boundaries)
        else:
            # Fallback to heuristic-based detection
            heuristic_boundaries = self._detect_heuristic_boundaries(blocks)
            boundaries.extend(heuristic_boundaries)
        
        # Deduplicate and sort
        boundary_dict = {}
        for idx, conf in boundaries:
            if idx not in boundary_dict or conf > boundary_dict[idx]:
                boundary_dict[idx] = conf
        
        return sorted(boundary_dict.items())
    
    def _detect_embedding_boundaries(
        self,
        blocks: List[ContentBlock],
    ) -> List[Tuple[int, float]]:
        """Detect boundaries using embedding similarity."""
        boundaries = []
        
        # Get texts for embedding
        texts = [block.text for block in blocks]
        
        # Embed all blocks
        try:
            embeddings = self.embed_fn(texts)
        except Exception as e:
            logger.warning(f"Embedding failed, using heuristics: {e}")
            return self._detect_heuristic_boundaries(blocks)
        
        # Calculate similarity between adjacent blocks
        for i in range(1, len(embeddings)):
            similarity = self._cosine_similarity(embeddings[i-1], embeddings[i])
            
            # Low similarity = boundary
            if similarity < self.config.similarity_threshold:
                confidence = 1.0 - similarity  # Higher confidence for lower similarity
                boundaries.append((i, confidence))
        
        return boundaries
    
    def _detect_heuristic_boundaries(
        self,
        blocks: List[ContentBlock],
    ) -> List[Tuple[int, float]]:
        """Detect boundaries using heuristics."""
        boundaries = []
        
        for i in range(1, len(blocks)):
            prev_block = blocks[i-1]
            curr_block = blocks[i]
            
            # Content type change is a boundary signal
            if prev_block.content_type != curr_block.content_type:
                boundaries.append((i, 0.6))
            
            # Large size difference suggests different topics
            size_ratio = max(prev_block.char_count, curr_block.char_count) / \
                        max(1, min(prev_block.char_count, curr_block.char_count))
            if size_ratio > 3:
                boundaries.append((i, 0.4))
            
            # Check for topic shift keywords
            if self._has_topic_shift_indicator(curr_block.text):
                boundaries.append((i, 0.5))
        
        return boundaries
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if not vec1 or not vec2:
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _has_topic_shift_indicator(self, text: str) -> bool:
        """Check for indicators of topic shift."""
        shift_patterns = [
            r'^(however|meanwhile|in contrast|on the other hand)',
            r'^(next|furthermore|additionally|moreover)',
            r'^(in conclusion|finally|to summarize)',
            r'^(chapter|section|part)\s+\d+',
        ]
        
        text_lower = text.lower().strip()
        return any(re.match(p, text_lower) for p in shift_patterns)

class SemanticChunker:
    """
    Create semantically coherent chunks from documents.
    
    Uses embedding-based boundary detection combined with
    structure-aware parsing.
    """
    
    def __init__(
        self,
        config: Optional[SemanticChunkerConfig] = None,
        embed_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
    ):
        self.config = config or SemanticChunkerConfig()
        self.embed_fn = embed_fn
        self.parser = StructureParser()
        self.boundary_detector = SemanticBoundaryDetector(self.config, embed_fn)
    
    def chunk(self, text: str) -> List[SemanticChunk]:
        """
        Chunk text into semantically coherent pieces.
        
        Args:
            text: The document text to chunk
            
        Returns:
            List of SemanticChunk objects
        """
        if not text or not text.strip():
            return []
        
        # Parse into content blocks
        blocks = self.parser.parse(text)
        
        if not blocks:
            return []
        
        # Detect semantic boundaries
        boundaries = self.boundary_detector.detect_boundaries(blocks)
        
        # Create chunks based on boundaries and size constraints
        chunks = self._create_chunks(blocks, boundaries)
        
        # Add overlap for context
        if self.config.overlap_tokens > 0:
            chunks = self._add_overlap(chunks)
        
        logger.debug(
            f"Semantic chunking: {len(blocks)} blocks -> {len(chunks)} chunks"
        )
        
        return chunks
    
    def _create_chunks(
        self,
        blocks: List[ContentBlock],
        boundaries: List[Tuple[int, float]],
    ) -> List[SemanticChunk]:
        """Create chunks respecting boundaries and size constraints."""
        chunks = []
        boundary_set = {b[0]: b[1] for b in boundaries}
        
        current_blocks: List[int] = []
        current_tokens = 0
        
        for i, block in enumerate(blocks):
            block_tokens = int(block.word_count / 0.75)
            
            # Check if we should start a new chunk
            should_split = False
            split_confidence = 0.0
            
            # Boundary detected
            if i in boundary_set:
                should_split = True
                split_confidence = boundary_set[i]
            
            # Size limit reached
            if current_tokens + block_tokens > self.config.max_chunk_tokens:
                should_split = True
                split_confidence = max(split_confidence, 0.3)
            
            # Special blocks shouldn't be split
            if block.content_type in [ContentType.CODE, ContentType.TABLE]:
                # If current chunk has content and this block is too big
                if current_blocks and block_tokens > self.config.min_chunk_tokens:
                    should_split = True
                    split_confidence = max(split_confidence, 0.8)
            
            # Create chunk if needed
            if should_split and current_blocks:
                chunk = self._blocks_to_chunk(
                    blocks, current_blocks, len(chunks), split_confidence
                )
                if chunk:
                    chunks.append(chunk)
                current_blocks = []
                current_tokens = 0
            
            # Add block to current chunk
            current_blocks.append(i)
            current_tokens += block_tokens
        
        # Create final chunk
        if current_blocks:
            chunk = self._blocks_to_chunk(
                blocks, current_blocks, len(chunks), 1.0
            )
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    def _blocks_to_chunk(
        self,
        blocks: List[ContentBlock],
        block_indices: List[int],
        chunk_index: int,
        boundary_confidence: float,
    ) -> Optional[SemanticChunk]:
        """Convert a list of blocks to a chunk."""
        if not block_indices:
            return None
        
        selected_blocks = [blocks[i] for i in block_indices]
        text = '\n\n'.join(b.text for b in selected_blocks)
        
        if not text.strip():
            return None
        
        # Analyze content
        content_types = [b.content_type for b in selected_blocks]
        primary_type = max(set(content_types), key=content_types.count)
        
        chunk = SemanticChunk(
            text=text,
            chunk_index=chunk_index,
            start_block=block_indices[0],
            end_block=block_indices[-1],
            boundary_confidence=boundary_confidence,
            primary_content_type=primary_type,
            has_code=ContentType.CODE in content_types,
            has_table=ContentType.TABLE in content_types,
            has_list=ContentType.LIST in content_types,
        )
        
        return chunk
    
    def _add_overlap(self, chunks: List[SemanticChunk]) -> List[SemanticChunk]:
        """Add overlap between chunks for context."""
        if len(chunks) <= 1:
            return chunks
        
        # For now, we just note that overlap should be handled
        # Full implementation would prepend context from previous chunk
        for i, chunk in enumerate(chunks):
            if i > 0:
                chunk.metadata["has_overlap"] = True
                chunk.metadata["overlap_from"] = i - 1
        
        return chunks

def create_semantic_chunker(
    config: Optional[SemanticChunkerConfig] = None,
    embed_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
) -> SemanticChunker:
    """Factory function to create a semantic chunker."""
    return SemanticChunker(config=config, embed_fn=embed_fn)
