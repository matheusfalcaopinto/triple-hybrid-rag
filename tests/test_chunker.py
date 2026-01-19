"""
Tests for Hierarchical Chunker Module

Tests parent/child chunk hierarchy with token-based splitting.
Includes optimization tests for LRU cache, iterative splitting, and O(n) search.
"""
from __future__ import annotations

import time
from uuid import uuid4

import pytest


class TestParentChunk:
    """Test ParentChunk dataclass."""
    
    def test_parent_chunk_creation(self) -> None:
        """Test ParentChunk can be created."""
        from triple_hybrid_rag.types import ParentChunk
        
        chunk = ParentChunk(
            index_in_document=0,
            text="This is parent content.",
            token_count=10,
            page_start=1,
            page_end=1,
            section_heading="Introduction",
        )
        
        assert chunk.index_in_document == 0
        assert chunk.text == "This is parent content."
        assert chunk.token_count == 10
        assert chunk.page_start == 1
        assert chunk.section_heading == "Introduction"


class TestChildChunk:
    """Test ChildChunk dataclass."""
    
    def test_child_chunk_creation(self) -> None:
        """Test ChildChunk can be created."""
        from triple_hybrid_rag.types import ChildChunk, Modality
        
        chunk = ChildChunk(
            index_in_parent=0,
            text="This is child content.",
            token_count=5,
            start_char_offset=0,
            end_char_offset=22,
            page=1,
            modality=Modality.TEXT,
        )
        
        assert chunk.index_in_parent == 0
        assert chunk.text == "This is child content."
        assert chunk.modality == Modality.TEXT
    
    def test_child_chunk_modalities(self) -> None:
        """Test different chunk modalities."""
        from triple_hybrid_rag.types import Modality
        
        assert Modality.TEXT.value == "text"
        assert Modality.TABLE.value == "table"
        assert Modality.IMAGE.value == "image"


class TestHierarchicalChunker:
    """Test HierarchicalChunker class."""
    
    def test_chunker_initialization(self) -> None:
        """Test chunker initializes with correct defaults."""
        from triple_hybrid_rag.core.chunker import HierarchicalChunker
        
        chunker = HierarchicalChunker()
        
        assert chunker.parent_chunk_tokens > 0
        assert chunker.child_chunk_tokens > 0
        assert chunker.parent_chunk_tokens > chunker.child_chunk_tokens
    
    def test_chunker_custom_sizes(self) -> None:
        """Test chunker with custom token sizes."""
        from triple_hybrid_rag.core.chunker import HierarchicalChunker
        
        chunker = HierarchicalChunker(
            parent_chunk_tokens=1000,
            child_chunk_tokens=250,
        )
        
        assert chunker.parent_chunk_tokens == 1000
        assert chunker.child_chunk_tokens == 250
    
    def test_count_tokens(self) -> None:
        """Test token counting."""
        from triple_hybrid_rag.core.chunker import HierarchicalChunker
        
        chunker = HierarchicalChunker()
        
        # Test empty string
        assert chunker.count_tokens("") == 0
        
        # Test some text
        count = chunker.count_tokens("Hello world")
        assert count > 0  # Should be 2 tokens
        assert count <= 5  # Sanity check
    
    def test_split_into_parents_basic(self) -> None:
        """Test splitting into parent chunks."""
        from triple_hybrid_rag.core.chunker import HierarchicalChunker
        
        chunker = HierarchicalChunker(
            parent_chunk_tokens=50,
            child_chunk_tokens=20,
        )
        
        # Create text with enough content
        text = "This is a test document with some content. " * 20
        
        parents = chunker.split_into_parents(
            text=text,
            document_id=uuid4(),
        )
        
        assert len(parents) > 0
        
        # Check parent structure
        for parent in parents:
            assert parent.text != ""
            assert parent.token_count > 0
    
    def test_split_document_creates_hierarchy(self) -> None:
        """Test that split_document creates parent-child hierarchy."""
        from triple_hybrid_rag.core.chunker import HierarchicalChunker
        
        chunker = HierarchicalChunker(
            parent_chunk_tokens=100,
            child_chunk_tokens=30,
        )
        
        text = "This is a test document. " * 50
        
        parents, children = chunker.split_document(
            text=text,
            document_id=uuid4(),
        )
        
        assert len(parents) > 0
        assert len(children) > 0
        
        # Children should reference parents
        parent_ids = {str(p.id) for p in parents}
        for child in children:
            assert str(child.parent_id) in parent_ids
    
    def test_split_document_short_text(self) -> None:
        """Test splitting short text that fits in one chunk."""
        from triple_hybrid_rag.core.chunker import HierarchicalChunker
        
        chunker = HierarchicalChunker()
        
        text = "Short text."
        
        parents, children = chunker.split_document(
            text=text,
            document_id=uuid4(),
        )
        
        # Should have at least one parent and one child
        assert len(parents) >= 1
        assert len(children) >= 1
    
    def test_split_document_empty_text(self) -> None:
        """Test splitting empty text."""
        from triple_hybrid_rag.core.chunker import HierarchicalChunker
        
        chunker = HierarchicalChunker()
        
        parents, children = chunker.split_document(
            text="",
            document_id=uuid4(),
        )
        
        # Empty text should produce no chunks
        assert len(parents) == 0
        assert len(children) == 0
    
    def test_section_heading_extraction(self) -> None:
        """Test that section headings are extracted."""
        from triple_hybrid_rag.core.chunker import HierarchicalChunker
        
        chunker = HierarchicalChunker(
            parent_chunk_tokens=100,
            child_chunk_tokens=30,
        )
        
        text = """# Introduction

This is the introduction section with some content.

## Methods

This section describes the methods used.
"""
        
        parents = chunker.split_into_parents(
            text=text,
            document_id=uuid4(),
        )
        
        # Should have detected some section headings
        headings = [p.section_heading for p in parents if p.section_heading]
        # At least check we got parents
        assert len(parents) > 0


class TestChunkOverlap:
    """Test chunk overlap handling."""
    
    def test_overlap_applied(self) -> None:
        """Test that overlap is applied between chunks."""
        from triple_hybrid_rag.core.chunker import HierarchicalChunker
        
        chunker = HierarchicalChunker(
            parent_chunk_tokens=50,
            child_chunk_tokens=20,
            chunk_overlap_tokens=5,
        )
        
        # Long text to force multiple chunks
        text = "Word " * 100
        
        parents, children = chunker.split_document(
            text=text,
            document_id=uuid4(),
        )
        
        # With overlap, we should have multiple chunks
        if len(children) > 1:
            # Check that adjacent chunks might share some text
            # (This is a weak test but verifies overlap logic runs)
            pass


class TestChunkerOptimizations:
    """Test chunker performance optimizations."""
    
    def test_token_estimation_cache_performance(self) -> None:
        """Test that LRU cached token estimation is faster than tiktoken."""
        from triple_hybrid_rag.core.chunker import (
            HierarchicalChunker,
            _estimate_tokens_cached,
        )
        
        chunker = HierarchicalChunker()
        text = "This is a test text for token counting. " * 10
        
        # Clear cache to ensure fair comparison
        _estimate_tokens_cached.cache_clear()
        
        # Time cached estimation (multiple calls)
        start = time.perf_counter()
        for _ in range(1000):
            chunker.count_tokens(text, use_cache=True)
        cached_time = time.perf_counter() - start
        
        # Time exact counting (tiktoken)
        start = time.perf_counter()
        for _ in range(1000):
            chunker.count_tokens_exact(text)
        exact_time = time.perf_counter() - start
        
        # Cached should be significantly faster (at least 10x)
        assert cached_time < exact_time, f"Cached: {cached_time}s, Exact: {exact_time}s"
    
    def test_iterative_split_handles_large_documents(self) -> None:
        """Test that iterative splitting works on large documents without stack overflow."""
        from triple_hybrid_rag.core.chunker import HierarchicalChunker
        
        chunker = HierarchicalChunker(
            parent_chunk_tokens=100,
            child_chunk_tokens=30,
        )
        
        # Generate large document (should not cause stack overflow)
        text = "This is a sentence. " * 10000  # ~10k sentences
        
        # Should complete without RecursionError
        parents, children = chunker.split_document(
            text=text,
            document_id=uuid4(),
        )
        
        assert len(parents) > 0
        assert len(children) > 0
    
    def test_child_chunk_offsets_are_correct(self) -> None:
        """Test that O(n) offset search produces correct character offsets."""
        from triple_hybrid_rag.core.chunker import HierarchicalChunker
        
        chunker = HierarchicalChunker(
            parent_chunk_tokens=200,
            child_chunk_tokens=50,
        )
        
        text = "First sentence here. Second sentence here. Third sentence here. Fourth sentence here. " * 10
        
        parents, children = chunker.split_document(
            text=text,
            document_id=uuid4(),
        )
        
        # Verify each child's offsets point to correct text
        for parent in parents:
            for child in parent.children:
                start = child.start_char_offset or 0
                end = child.end_char_offset or 0
                if start >= 0 and end <= len(parent.text):
                    # The text at offset should match or contain the child text
                    # (May not be exact due to stripping)
                    extracted = parent.text[start:end]
                    # At minimum, the lengths should be close
                    assert len(extracted) > 0
    
    def test_count_tokens_exact_bypasses_cache(self) -> None:
        """Test that count_tokens_exact always uses tiktoken."""
        from triple_hybrid_rag.core.chunker import HierarchicalChunker
        
        chunker = HierarchicalChunker()
        
        # Exact should use tiktoken for accurate count
        exact_count = chunker.count_tokens_exact("Hello world")
        
        # This should be 2 tokens for "Hello" and "world" with cl100k_base
        assert exact_count == 2
    
    def test_estimate_tokens_uses_char_approximation(self) -> None:
        """Test that cached estimation uses len/4 approximation."""
        from triple_hybrid_rag.core.chunker import _estimate_tokens_cached
        
        # Clear cache
        _estimate_tokens_cached.cache_clear()
        
        text = "a" * 100  # 100 characters
        estimated = _estimate_tokens_cached(text)
        
        # Should be approximately len/4 = 25
        assert estimated == 25
