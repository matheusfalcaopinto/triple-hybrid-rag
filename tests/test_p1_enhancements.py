"""
Tests for P1 Enhancement Modules

Tests for:
- Adaptive Query-Aware Fusion (AQAF)
- Semantic Chunking
"""

import pytest
from typing import List

from triple_hybrid_rag.retrieval.adaptive_fusion import (
    QueryFeatureExtractor,
    AdaptiveFusionPredictor,
    AdaptiveRRFFusion,
    AdaptiveFusionConfig,
    QueryIntent,
    QueryFeatures,
    FusionWeights,
)
from triple_hybrid_rag.ingestion.semantic_chunker import (
    SemanticChunker,
    SemanticChunkerConfig,
    StructureParser,
    SemanticBoundaryDetector,
    ContentType,
    ContentBlock,
    SemanticChunk,
)
from triple_hybrid_rag.types import SearchResult


# ============================================================================
# Adaptive Fusion Tests
# ============================================================================

class TestQueryFeatureExtractor:
    """Test query feature extraction."""
    
    @pytest.fixture
    def extractor(self):
        return QueryFeatureExtractor()
    
    def test_basic_extraction(self, extractor):
        """Test basic feature extraction."""
        query = "What is the refund policy?"
        features = extractor.extract(query)
        
        assert features.word_count == 5
        assert features.char_count == len(query)
        assert features.question_type == "what"
    
    def test_factual_intent_detection(self, extractor):
        """Test factual intent detection."""
        query = "When was the company founded?"
        features = extractor.extract(query)
        
        assert features.primary_intent == QueryIntent.FACTUAL
        assert features.question_type == "when"
    
    def test_procedural_intent_detection(self, extractor):
        """Test procedural intent detection."""
        query = "How to configure the database connection?"
        features = extractor.extract(query)
        
        assert features.primary_intent == QueryIntent.PROCEDURAL
        assert features.question_type == "how"
    
    def test_conceptual_intent_detection(self, extractor):
        """Test conceptual intent detection."""
        query = "Explain the concept of microservices architecture"
        features = extractor.extract(query)
        
        assert features.primary_intent == QueryIntent.CONCEPTUAL
    
    def test_comparative_intent_detection(self, extractor):
        """Test comparative intent detection."""
        query = "What is the difference between SQL and NoSQL?"
        features = extractor.extract(query)
        
        assert features.primary_intent == QueryIntent.COMPARATIVE
    
    def test_relational_intent_detection(self, extractor):
        """Test relational intent detection."""
        query = "How are users related to orders?"
        features = extractor.extract(query)
        
        assert features.primary_intent == QueryIntent.RELATIONAL
    
    def test_navigational_intent_detection(self, extractor):
        """Test navigational intent detection."""
        query = "Find the authentication section in the documentation"
        features = extractor.extract(query)
        
        assert features.primary_intent == QueryIntent.NAVIGATIONAL
    
    def test_technical_intent_detection(self, extractor):
        """Test technical intent detection."""
        query = "API endpoint for user authentication"
        features = extractor.extract(query)
        
        assert features.primary_intent == QueryIntent.TECHNICAL
        assert features.has_technical_terms
    
    def test_temporal_detection_past(self, extractor):
        """Test past temporal detection."""
        query = "What was the previous version of the API?"
        features = extractor.extract(query)
        
        assert features.has_temporal_reference
        assert features.temporal_type == "past"
    
    def test_temporal_detection_future(self, extractor):
        """Test future temporal detection."""
        query = "What will be in the next release?"
        features = extractor.extract(query)
        
        assert features.has_temporal_reference
        assert features.temporal_type == "future"
    
    def test_code_reference_detection(self, extractor):
        """Test code reference detection."""
        query = "How to use `async/await` in JavaScript?"
        features = extractor.extract(query)
        
        assert features.has_code_reference
    
    def test_multiple_clauses_detection(self, extractor):
        """Test multiple clauses detection."""
        query = "Find all users who have orders and calculate their total spending"
        features = extractor.extract(query)
        
        assert features.has_multiple_clauses
    
    def test_abstraction_level_high(self, extractor):
        """Test high abstraction level."""
        query = "Explain the difference between monolithic and microservices"
        features = extractor.extract(query)
        
        assert features.abstraction_level == "high"
    
    def test_abstraction_level_low(self, extractor):
        """Test low abstraction level."""
        query = "When was version 2.0 released?"
        features = extractor.extract(query)
        
        assert features.abstraction_level == "low"
    
    def test_keyword_density(self, extractor):
        """Test keyword density calculation."""
        # Query with many stopwords
        query = "the is a an of in to"
        features = extractor.extract(query)
        
        assert features.stopword_ratio > 0.8
        assert features.keyword_density < 0.3


class TestAdaptiveFusionPredictor:
    """Test adaptive fusion weight prediction."""
    
    @pytest.fixture
    def predictor(self):
        return AdaptiveFusionPredictor()
    
    def test_default_weights_when_disabled(self):
        """Test default weights when fusion is disabled."""
        config = AdaptiveFusionConfig(enabled=False)
        predictor = AdaptiveFusionPredictor(config)
        
        weights = predictor.predict("any query")
        
        assert weights.lexical_weight == 0.7
        assert weights.semantic_weight == 0.8
        assert weights.graph_weight == 1.0
        assert weights.confidence == 1.0
    
    def test_factual_query_weights(self, predictor):
        """Test weights for factual queries."""
        query = "When was the company founded?"
        weights = predictor.predict(query)
        
        # Factual queries should boost lexical
        assert weights.lexical_weight > 0.8
    
    def test_conceptual_query_weights(self, predictor):
        """Test weights for conceptual queries."""
        query = "Explain how neural networks learn"
        weights = predictor.predict(query)
        
        # Conceptual queries should boost semantic
        assert weights.semantic_weight > 0.9
    
    def test_relational_query_weights(self, predictor):
        """Test weights for relational queries."""
        query = "How are products related to categories?"
        weights = predictor.predict(query)
        
        # Relational queries should boost graph
        assert weights.graph_weight >= 1.0
    
    def test_technical_query_adjustments(self, predictor):
        """Test adjustments for technical queries."""
        query = "API endpoint for JSON configuration"
        weights = predictor.predict(query)
        
        # Technical queries should boost lexical and SPLADE
        assert weights.lexical_weight > 0.7
    
    def test_temporal_query_adjustments(self, predictor):
        """Test adjustments for temporal queries."""
        query = "What was the previous version?"
        weights = predictor.predict(query)
        
        # Temporal queries should have reasonable lexical weight
        assert weights.lexical_weight > 0.7
    
    def test_weights_have_reasoning(self, predictor):
        """Test that weights include reasoning."""
        query = "How to configure authentication?"
        weights = predictor.predict(query)
        
        assert weights.reasoning
        assert "Intent:" in weights.reasoning
    
    def test_weights_normalize(self):
        """Test weight normalization."""
        weights = FusionWeights(
            lexical_weight=2.0,
            semantic_weight=2.0,
            graph_weight=2.0,
            splade_weight=0.0,
        )
        
        normalized = weights.normalize()
        total = normalized.lexical_weight + normalized.semantic_weight + normalized.graph_weight
        
        assert abs(total - 1.0) < 0.01


class TestAdaptiveRRFFusion:
    """Test adaptive RRF fusion."""
    
    def _create_results(self, ids: List[int]) -> List[SearchResult]:
        """Helper to create mock search results."""
        from uuid import UUID
        return [
            SearchResult(
                chunk_id=UUID(int=i),
                document_id=UUID(int=1),
                text=f"Result {i}",
                semantic_score=1.0 - (idx * 0.1),
            )
            for idx, i in enumerate(ids)
        ]
    
    def test_basic_fusion(self):
        """Test basic fusion of results."""
        fusion = AdaptiveRRFFusion()
        
        lexical = self._create_results([1, 2, 3])
        semantic = self._create_results([2, 3, 4])
        graph = self._create_results([3, 4, 5])
        
        results, weights = fusion.fuse(
            query="What is the refund policy?",
            lexical_results=lexical,
            semantic_results=semantic,
            graph_results=graph,
        )
        
        assert len(results) == 5  # Unique chunk IDs
        assert weights is not None
        assert weights.reasoning
    
    def test_fusion_ranking(self):
        """Test that fusion ranks by RRF score."""
        from uuid import UUID
        fusion = AdaptiveRRFFusion()
        
        # Chunk 2 appears in all three
        lexical = self._create_results([1, 2])
        semantic = self._create_results([2, 3])
        graph = self._create_results([2, 4])
        
        results, _ = fusion.fuse(
            query="test query",
            lexical_results=lexical,
            semantic_results=semantic,
            graph_results=graph,
        )
        
        # Chunk 2 should be ranked highest
        assert results[0].chunk_id == UUID(int=2)
    
    def test_weights_affect_ranking(self):
        """Test that weights affect final ranking."""
        # High graph weight
        config = AdaptiveFusionConfig(
            default_graph_weight=2.0,
            default_lexical_weight=0.5,
            default_semantic_weight=0.5,
        )
        
        fusion = AdaptiveRRFFusion(config)
        
        lexical = self._create_results([1])
        semantic = self._create_results([2])
        graph = self._create_results([3])
        
        # Use a query
        results, weights = fusion.fuse(
            query="Test query",
            lexical_results=lexical,
            semantic_results=semantic,
            graph_results=graph,
        )
        
        # Weights should be computed and within valid range
        assert 0 < weights.graph_weight <= 2.5
        assert 0 < weights.lexical_weight <= 2.5
        assert 0 < weights.semantic_weight <= 2.5
        assert weights.query_features is not None


# ============================================================================
# Semantic Chunking Tests
# ============================================================================

class TestStructureParser:
    """Test document structure parsing."""
    
    @pytest.fixture
    def parser(self):
        return StructureParser()
    
    def test_parse_simple_text(self, parser):
        """Test parsing simple text."""
        text = "This is a simple paragraph.\n\nThis is another paragraph."
        blocks = parser.parse(text)
        
        assert len(blocks) == 2
        assert all(b.content_type == ContentType.TEXT for b in blocks)
    
    def test_parse_headings(self, parser):
        """Test parsing markdown headings."""
        text = "# Main Title\n\nSome content.\n\n## Section\n\nMore content."
        blocks = parser.parse(text)
        
        heading_blocks = [b for b in blocks if b.content_type == ContentType.HEADING]
        assert len(heading_blocks) >= 2
    
    def test_parse_code_blocks(self, parser):
        """Test parsing code blocks."""
        text = "Some text.\n\n```python\nprint('hello')\n```\n\nMore text."
        blocks = parser.parse(text)
        
        code_blocks = [b for b in blocks if b.content_type == ContentType.CODE]
        assert len(code_blocks) == 1
        assert "print" in code_blocks[0].text
    
    def test_parse_tables(self, parser):
        """Test parsing markdown tables."""
        text = "Some text.\n\n| Col1 | Col2 |\n|------|------|\n| A | B |\n\nMore text."
        blocks = parser.parse(text)
        
        table_blocks = [b for b in blocks if b.content_type == ContentType.TABLE]
        assert len(table_blocks) == 1
    
    def test_parse_lists(self, parser):
        """Test parsing lists."""
        text = "Some text.\n\n- Item 1\n- Item 2\n- Item 3\n\nMore text."
        blocks = parser.parse(text)
        
        list_blocks = [b for b in blocks if b.content_type == ContentType.LIST]
        assert len(list_blocks) == 1
    
    def test_code_block_not_split(self, parser):
        """Test that code blocks are preserved as single blocks."""
        text = """```python
def function():
    x = 1
    y = 2
    return x + y
```"""
        blocks = parser.parse(text)
        
        code_blocks = [b for b in blocks if b.content_type == ContentType.CODE]
        assert len(code_blocks) == 1
        assert "def function" in code_blocks[0].text


class TestSemanticBoundaryDetector:
    """Test semantic boundary detection."""
    
    def test_heading_boundaries(self):
        """Test that headings create boundaries."""
        config = SemanticChunkerConfig(use_embeddings=False)
        detector = SemanticBoundaryDetector(config, embed_fn=None)
        
        blocks = [
            ContentBlock(text="Introduction", content_type=ContentType.HEADING),
            ContentBlock(text="Some content here.", content_type=ContentType.TEXT),
            ContentBlock(text="Next Section", content_type=ContentType.HEADING),
            ContentBlock(text="More content.", content_type=ContentType.TEXT),
        ]
        
        boundaries = detector.detect_boundaries(blocks)
        
        # Headings at index 0 and 2 should be boundaries
        boundary_indices = [b[0] for b in boundaries]
        assert 0 in boundary_indices
        assert 2 in boundary_indices
    
    def test_content_type_change_boundary(self):
        """Test that content type changes create boundaries."""
        config = SemanticChunkerConfig(use_embeddings=False)
        detector = SemanticBoundaryDetector(config, embed_fn=None)
        
        blocks = [
            ContentBlock(text="Some text.", content_type=ContentType.TEXT),
            ContentBlock(text="```code```", content_type=ContentType.CODE),
            ContentBlock(text="More text.", content_type=ContentType.TEXT),
        ]
        
        boundaries = detector.detect_boundaries(blocks)
        
        # Should have boundaries at type changes
        assert len(boundaries) >= 2
    
    def test_heuristic_boundaries(self):
        """Test heuristic-based boundary detection."""
        config = SemanticChunkerConfig(use_embeddings=False)
        detector = SemanticBoundaryDetector(config, embed_fn=None)
        
        blocks = [
            ContentBlock(text="Short.", content_type=ContentType.TEXT),
            ContentBlock(
                text="This is a much longer paragraph with lots of content " * 10,
                content_type=ContentType.TEXT,
            ),
        ]
        
        boundaries = detector.detect_boundaries(blocks)
        
        # Size difference should trigger boundary
        assert len(boundaries) >= 1


class TestSemanticChunker:
    """Test the complete semantic chunker."""
    
    @pytest.fixture
    def chunker(self):
        config = SemanticChunkerConfig(
            use_embeddings=False,
            max_chunk_tokens=500,
            min_chunk_tokens=50,
        )
        return SemanticChunker(config=config)
    
    def test_chunk_empty_text(self, chunker):
        """Test chunking empty text."""
        chunks = chunker.chunk("")
        assert chunks == []
        
        chunks = chunker.chunk("   ")
        assert chunks == []
    
    def test_chunk_simple_text(self, chunker):
        """Test chunking simple text."""
        text = "This is a simple document with some content."
        chunks = chunker.chunk(text)
        
        assert len(chunks) >= 1
        assert chunks[0].text.strip()
    
    def test_chunk_with_headings(self, chunker):
        """Test chunking with headings creates boundaries."""
        text = """# Introduction

This is the introduction section with some content.

# Methods

This describes the methods used in the study.

# Results

Here are the results of our analysis."""
        
        chunks = chunker.chunk(text)
        
        # Should create at least 3 chunks (one per section)
        assert len(chunks) >= 3
    
    def test_chunk_preserves_code(self, chunker):
        """Test that code blocks are preserved."""
        text = """Some explanation text.

```python
def complex_function():
    for i in range(100):
        process(i)
    return result
```

More explanation after the code."""
        
        chunks = chunker.chunk(text)
        
        # Find chunk with code
        code_chunk = None
        for chunk in chunks:
            if "def complex_function" in chunk.text:
                code_chunk = chunk
                break
        
        assert code_chunk is not None
        assert code_chunk.has_code
    
    def test_chunk_preserves_tables(self, chunker):
        """Test that tables are preserved."""
        text = """Introduction text.

| Header 1 | Header 2 |
|----------|----------|
| Data 1   | Data 2   |
| Data 3   | Data 4   |

Conclusion text."""
        
        chunks = chunker.chunk(text)
        
        # Find chunk with table
        table_chunk = None
        for chunk in chunks:
            if "Header 1" in chunk.text:
                table_chunk = chunk
                break
        
        assert table_chunk is not None
        assert table_chunk.has_table
    
    def test_chunk_respects_size_limits(self, chunker):
        """Test that chunks respect size limits."""
        # Create a large document
        text = "\n\n".join([f"Paragraph {i}. " * 20 for i in range(20)])
        
        chunks = chunker.chunk(text)
        
        # All chunks should be under max size
        for chunk in chunks:
            assert chunk.estimated_tokens <= chunker.config.max_chunk_tokens + 100  # Allow some slack
    
    def test_chunk_has_metadata(self, chunker):
        """Test that chunks have proper metadata."""
        text = "# Title\n\nContent here.\n\n# Another Section\n\nMore content."
        chunks = chunker.chunk(text)
        
        for chunk in chunks:
            assert chunk.chunk_index >= 0
            assert chunk.word_count > 0
            assert chunk.estimated_tokens > 0
            assert chunk.content_hash
    
    def test_chunk_content_hash_unique(self, chunker):
        """Test that different chunks have different hashes."""
        text = "# Section 1\n\nContent A.\n\n# Section 2\n\nContent B."
        chunks = chunker.chunk(text)
        
        if len(chunks) >= 2:
            hashes = [c.content_hash for c in chunks]
            assert len(hashes) == len(set(hashes))  # All unique


class TestSemanticChunkerWithEmbeddings:
    """Test semantic chunker with mock embeddings."""
    
    def test_chunker_with_embedding_function(self):
        """Test chunker with a mock embedding function."""
        def mock_embed(texts: List[str]) -> List[List[float]]:
            # Return different embeddings for different texts
            return [[hash(t) % 100 / 100.0] * 10 for t in texts]
        
        config = SemanticChunkerConfig(
            use_embeddings=True,
            similarity_threshold=0.5,
        )
        chunker = SemanticChunker(config=config, embed_fn=mock_embed)
        
        text = "First topic about A.\n\nSecond topic about B.\n\nThird topic about C."
        chunks = chunker.chunk(text)
        
        assert len(chunks) >= 1


# ============================================================================
# Integration Tests
# ============================================================================

class TestAdaptiveFusionIntegration:
    """Integration tests for adaptive fusion."""
    
    def test_end_to_end_fusion(self):
        """Test end-to-end adaptive fusion."""
        from uuid import UUID
        fusion = AdaptiveRRFFusion()
        
        # Create realistic results
        lexical = [
            SearchResult(chunk_id=UUID(int=1), document_id=UUID(int=1), text="Refund policy states...", lexical_score=0.9),
            SearchResult(chunk_id=UUID(int=2), document_id=UUID(int=1), text="Returns are processed...", lexical_score=0.8),
        ]
        semantic = [
            SearchResult(chunk_id=UUID(int=2), document_id=UUID(int=1), text="Returns are processed...", semantic_score=0.85),
            SearchResult(chunk_id=UUID(int=3), document_id=UUID(int=1), text="Money back guarantee...", semantic_score=0.75),
        ]
        graph = [
            SearchResult(chunk_id=UUID(int=4), document_id=UUID(int=2), text="Related: Customer service...", graph_score=0.7),
        ]
        
        results, weights = fusion.fuse(
            query="What is the refund policy?",
            lexical_results=lexical,
            semantic_results=semantic,
            graph_results=graph,
        )
        
        # Check results
        assert len(results) == 4
        assert all(r.rrf_score is not None for r in results)
        
        # Check weights
        assert weights.lexical_weight > 0
        assert weights.semantic_weight > 0
        assert weights.graph_weight > 0
        assert weights.query_features is not None


class TestSemanticChunkerIntegration:
    """Integration tests for semantic chunking."""
    
    def test_chunk_real_document(self):
        """Test chunking a realistic document."""
        document = """# User Guide

Welcome to our application. This guide will help you get started.

## Installation

To install the application, follow these steps:

1. Download the installer
2. Run the setup wizard
3. Configure your settings

```bash
./install.sh --prefix=/usr/local
```

## Configuration

The configuration file is located at `/etc/app/config.yaml`.

| Option | Default | Description |
|--------|---------|-------------|
| port   | 8080    | Server port |
| debug  | false   | Debug mode  |

## Usage

After installation, you can start using the application.

### Basic Commands

- `app start` - Start the server
- `app stop` - Stop the server
- `app status` - Check status

## Troubleshooting

If you encounter issues, check the logs at `/var/log/app/`.

Common issues:
- Port already in use
- Permission denied
- Configuration errors
"""
        
        config = SemanticChunkerConfig(
            use_embeddings=False,
            max_chunk_tokens=300,
            min_chunk_tokens=50,
        )
        chunker = SemanticChunker(config=config)
        
        chunks = chunker.chunk(document)
        
        # Should create multiple chunks
        assert len(chunks) >= 3
        
        # Check that code is preserved
        code_found = any(chunk.has_code for chunk in chunks)
        assert code_found
        
        # Check that tables are preserved
        table_found = any(chunk.has_table for chunk in chunks)
        assert table_found
        
        # All chunks should have content
        for chunk in chunks:
            assert chunk.text.strip()
            assert chunk.word_count > 0
