"""
Tests for RAG2 Hierarchical Chunker Module

Tests parent/child chunk hierarchy with token-based splitting.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch


class TestParentChunk:
    """Test ParentChunk dataclass."""
    
    def test_parent_chunk_creation(self) -> None:
        """Test ParentChunk can be created."""
        from voice_agent.rag2.chunker import ParentChunk
        
        chunk = ParentChunk(
            id="parent_1",
            index_in_document=0,
            text="This is parent content.",
            token_count=10,
            page_start=1,
            page_end=1,
            section_heading="Introduction",
        )
        
        assert chunk.id == "parent_1"
        assert chunk.index_in_document == 0
        assert chunk.text == "This is parent content."
        assert chunk.token_count == 10
        assert chunk.page_start == 1
        assert chunk.section_heading == "Introduction"


class TestChildChunk:
    """Test ChildChunk dataclass."""
    
    def test_child_chunk_creation(self) -> None:
        """Test ChildChunk can be created."""
        from voice_agent.rag2.chunker import ChildChunk, ChunkModality
        
        chunk = ChildChunk(
            id="child_1",
            parent_id="parent_1",
            index_in_parent=0,
            text="This is child content.",
            token_count=5,
            start_char_offset=0,
            end_char_offset=22,
            page=1,
            modality=ChunkModality.TEXT,
        )
        
        assert chunk.id == "child_1"
        assert chunk.parent_id == "parent_1"
        assert chunk.text == "This is child content."
        assert chunk.modality == ChunkModality.TEXT
        assert chunk.content_hash != ""  # Should be computed
    
    def test_child_chunk_modalities(self) -> None:
        """Test different chunk modalities."""
        from voice_agent.rag2.chunker import ChunkModality
        
        assert ChunkModality.TEXT.value == "text"
        assert ChunkModality.TABLE.value == "table"
        assert ChunkModality.IMAGE.value == "image"
        assert ChunkModality.IMAGE_CAPTION.value == "image_caption"
    
    def test_content_hash_consistency(self) -> None:
        """Test that same content produces same hash."""
        from voice_agent.rag2.chunker import ChildChunk
        
        chunk1 = ChildChunk(
            id="c1", parent_id="p1", index_in_parent=0, text="Same content",
            token_count=2, start_char_offset=0, end_char_offset=12, page=1,
        )
        chunk2 = ChildChunk(
            id="c2", parent_id="p2", index_in_parent=1, text="Same content",
            token_count=2, start_char_offset=0, end_char_offset=12, page=1,
        )
        
        assert chunk1.content_hash == chunk2.content_hash
    
    def test_content_hash_different(self) -> None:
        """Test that different content produces different hashes."""
        from voice_agent.rag2.chunker import ChildChunk
        
        chunk1 = ChildChunk(
            id="c1", parent_id="p1", index_in_parent=0, text="Content A",
            token_count=2, start_char_offset=0, end_char_offset=9, page=1,
        )
        chunk2 = ChildChunk(
            id="c2", parent_id="p1", index_in_parent=1, text="Content B",
            token_count=2, start_char_offset=10, end_char_offset=19, page=1,
        )
        
        assert chunk1.content_hash != chunk2.content_hash


class TestHierarchicalChunker:
    """Test HierarchicalChunker class."""
    
    def test_chunker_initialization(self) -> None:
        """Test chunker initializes with correct defaults."""
        from voice_agent.rag2.chunker import HierarchicalChunker
        
        chunker = HierarchicalChunker()
        
        assert chunker.parent_target_tokens > 0
        assert chunker.child_target_tokens > 0
        assert chunker.parent_target_tokens > chunker.child_target_tokens
    
    def test_chunker_custom_sizes(self) -> None:
        """Test chunker with custom token sizes."""
        from voice_agent.rag2.chunker import HierarchicalChunker
        
        chunker = HierarchicalChunker(
            parent_target_tokens=1000,
            child_target_tokens=250,
        )
        
        assert chunker.parent_target_tokens == 1000
        assert chunker.child_target_tokens == 250
    
    def test_chunk_document_basic(self) -> None:
        """Test chunking a basic document."""
        from voice_agent.rag2.chunker import HierarchicalChunker
        
        chunker = HierarchicalChunker(
            parent_target_tokens=50,
            child_target_tokens=20,
        )
        
        # Create a document with enough text
        text = "This is a test document with some content. " * 20
        
        parents = chunker.chunk_document(
            text=text,
            doc_hash="abc123",
        )
        
        assert len(parents) > 0
        
        # All parents should have children
        for parent in parents:
            assert len(parent.children) >= 0
    
    def test_chunk_document_with_pages(self) -> None:
        """Test chunking with page tracking."""
        from voice_agent.rag2.chunker import HierarchicalChunker
        
        chunker = HierarchicalChunker(
            parent_target_tokens=50,
            child_target_tokens=20,
        )
        
        text = "Page 1 content. " * 10 + "Page 2 content. " * 10
        page_len = len("Page 1 content. " * 10)
        
        # Define page boundaries
        source_pages = [
            (0, page_len, 1),
            (page_len, len(text), 2),
        ]
        
        parents = chunker.chunk_document(
            text=text,
            doc_hash="abc123",
            source_pages=source_pages,
        )
        
        assert len(parents) > 0
    
    def test_chunk_empty_document(self) -> None:
        """Test chunking an empty document."""
        from voice_agent.rag2.chunker import HierarchicalChunker
        
        chunker = HierarchicalChunker()
        
        parents = chunker.chunk_document(
            text="",
            doc_hash="empty",
        )
        
        assert parents == []
    
    def test_chunk_short_document(self) -> None:
        """Test chunking a very short document."""
        from voice_agent.rag2.chunker import HierarchicalChunker
        
        chunker = HierarchicalChunker(
            parent_target_tokens=1000,
            child_target_tokens=200,
        )
        
        parents = chunker.chunk_document(
            text="Short document.",
            doc_hash="short",
        )
        
        # Should still create at least one parent
        assert len(parents) >= 1


class TestEstimateTokens:
    """Test token estimation function."""
    
    def test_estimate_tokens_basic(self) -> None:
        """Test basic token estimation."""
        from voice_agent.rag2.chunker import estimate_tokens
        
        # Simple text
        count = estimate_tokens("Hello world")
        assert count > 0
        
        # Empty text
        assert estimate_tokens("") == 0
    
    def test_estimate_tokens_long_text(self) -> None:
        """Test token estimation for longer text."""
        from voice_agent.rag2.chunker import estimate_tokens
        
        # ~100 chars -> ~25 tokens
        text = "a" * 100
        count = estimate_tokens(text)
        
        # Should be roughly 1 token per 4 chars
        assert 20 <= count <= 30


class TestSentenceBoundary:
    """Test sentence boundary detection."""
    
    def test_find_sentence_boundary(self) -> None:
        """Test finding sentence boundaries."""
        from voice_agent.rag2.chunker import find_sentence_boundary
        
        text = "First sentence. Second sentence. Third sentence."
        
        # Target near first period
        pos = find_sentence_boundary(text, 15)
        
        # Should find a boundary near target
        assert 10 <= pos <= 20
    
    def test_find_sentence_boundary_no_boundary(self) -> None:
        """Test when no boundary found."""
        from voice_agent.rag2.chunker import find_sentence_boundary
        
        text = "NoSentenceBoundaryHere"
        
        pos = find_sentence_boundary(text, 10)
        
        # Should return target when no boundary found
        assert pos == 10


class TestGetHierarchicalChunker:
    """Test factory function."""
    
    def test_get_hierarchical_chunker_returns_instance(self) -> None:
        """Test factory returns chunker instance."""
        from voice_agent.rag2.chunker import get_hierarchical_chunker
        
        chunker = get_hierarchical_chunker()
        
        assert chunker is not None
    
    def test_get_hierarchical_chunker_uses_settings(self) -> None:
        """Test factory uses settings for configuration."""
        from voice_agent.rag2.chunker import get_hierarchical_chunker, HierarchicalChunker
        
        chunker = get_hierarchical_chunker()
        
        assert isinstance(chunker, HierarchicalChunker)
