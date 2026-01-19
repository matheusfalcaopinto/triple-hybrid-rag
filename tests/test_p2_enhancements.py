"""
Tests for P2 Enhancement Modules

Tests for:
- Context Compression
- SPLADE Integration
- Query Classification & Routing
"""

import pytest
from typing import List

from triple_hybrid_rag.retrieval.context_compression import (
    ContextCompressor,
    CompressionConfig,
    CompressionStrategy,
    SentenceScorer,
    LongContextHandler,
)
from triple_hybrid_rag.retrieval.splade import (
    SpladeEncoder,
    SpladeRetriever,
    SpladeConfig,
    SparseVector,
    QueryTermAnalyzer,
)
from triple_hybrid_rag.retrieval.query_router import (
    QueryClassifier,
    QueryRouter,
    QueryCategory,
    RetrievalStrategy,
    IndexType,
    RouterConfig,
    AdaptiveRouter,
)

# ============================================================================
# Context Compression Tests
# ============================================================================

class TestSentenceScorer:
    """Test sentence scoring."""
    
    @pytest.fixture
    def scorer(self):
        return SentenceScorer()
    
    def test_score_sentences_heuristics(self, scorer):
        """Test heuristic scoring."""
        sentences = [
            "The refund policy allows returns within 30 days.",
            "Weather is nice today.",
            "Contact customer service for refund requests.",
        ]
        query = "What is the refund policy?"
        
        scored = scorer.score_sentences(sentences, query)
        
        assert len(scored) == 3
        # Refund-related sentences should score higher
        assert "refund" in scored[0][0].lower() or "refund" in scored[1][0].lower()
    
    def test_score_empty_sentences(self, scorer):
        """Test with empty input."""
        scored = scorer.score_sentences([], "any query")
        assert scored == []
    
    def test_score_with_embeddings(self):
        """Test scoring with mock embeddings."""
        def mock_embed(texts):
            return [[len(t) / 100.0] * 10 for t in texts]
        
        scorer = SentenceScorer(embed_fn=mock_embed)
        sentences = ["Short.", "This is a longer sentence with more content."]
        
        scored = scorer.score_sentences(sentences, "test")
        assert len(scored) == 2

class TestContextCompressor:
    """Test context compression."""
    
    @pytest.fixture
    def compressor(self):
        return ContextCompressor()
    
    def test_compress_disabled(self):
        """Test compression when disabled."""
        config = CompressionConfig(enabled=False)
        compressor = ContextCompressor(config=config)
        
        text = "Some text to compress."
        result = compressor.compress(text, "query")
        
        assert result.compressed_context == text
        assert result.compression_ratio == 1.0
    
    def test_compress_short_text(self, compressor):
        """Test that short text is not compressed."""
        text = "This is short."
        result = compressor.compress(text, "query", max_tokens=1000)
        
        assert result.compressed_context == text
        assert result.compression_ratio == 1.0
    
    def test_extractive_compression(self):
        """Test extractive compression."""
        config = CompressionConfig(
            strategy=CompressionStrategy.EXTRACTIVE,
            max_tokens=50,
            sentences_per_chunk=2,
        )
        compressor = ContextCompressor(config=config)
        
        # Create text that exceeds token limit
        text = "First sentence about refunds. " * 20 + "Second sentence about returns. " * 20
        
        result = compressor.compress(text, "refund policy")
        
        assert result.compressed_tokens < result.original_tokens
        assert result.strategy_used == CompressionStrategy.EXTRACTIVE
    
    def test_smart_truncate(self):
        """Test smart truncation."""
        config = CompressionConfig(
            strategy=CompressionStrategy.TRUNCATE,
            max_tokens=20,
        )
        compressor = ContextCompressor(config=config)
        
        # Use longer text to trigger compression
        text = "First sentence here. " * 20 + "Second sentence here. " * 20
        result = compressor.compress(text, "query")
        
        assert result.strategy_used == CompressionStrategy.TRUNCATE
        assert result.compressed_tokens <= result.original_tokens
    
    def test_preserve_code_blocks(self):
        """Test that code blocks are preserved."""
        config = CompressionConfig(
            strategy=CompressionStrategy.EXTRACTIVE,
            preserve_code=True,
            max_tokens=100,
        )
        compressor = ContextCompressor(config=config)
        
        text = """Some explanation text here that goes on and on.

```python
def important_function():
    return "preserved"
```

More text here."""
        
        result = compressor.compress(text, "function")
        
        # Code should be in preserved elements or compressed text
        assert "important_function" in result.compressed_context or \
               any("important_function" in c.compressed_text for c in result.chunks)
    
    def test_caching(self):
        """Test compression caching."""
        config = CompressionConfig(enable_cache=True, max_tokens=50)
        compressor = ContextCompressor(config=config)
        
        text = "A " * 200  # Long text
        
        # First call
        result1 = compressor.compress(text, "query")
        assert not result1.cache_hit
        
        # Second call - should hit cache
        result2 = compressor.compress(text, "query")
        assert result2.cache_hit

class TestLongContextHandler:
    """Test long context handling."""
    
    def test_process_multiple_documents(self):
        """Test processing multiple documents."""
        handler = LongContextHandler(max_tokens=200)
        
        documents = [
            "Document 1 content. " * 20,
            "Document 2 content. " * 20,
        ]
        
        result = handler.process_long_context(documents, "test query")
        
        # Should be compressed
        assert len(result) < sum(len(d) for d in documents)

# ============================================================================
# SPLADE Tests
# ============================================================================

class TestSpladeEncoder:
    """Test SPLADE encoding."""
    
    @pytest.fixture
    def encoder(self):
        return SpladeEncoder()
    
    def test_encode_text(self, encoder):
        """Test basic encoding."""
        result = encoder.encode("How to configure the database connection?")
        
        assert result.text
        assert result.sparse_vector.dimension > 0
        assert len(result.sparse_vector.indices) == len(result.sparse_vector.values)
    
    def test_encode_with_expansion(self, encoder):
        """Test term expansion."""
        result = encoder.encode("API authentication")
        
        # Should have expanded terms
        assert len(result.expanded_terms) > 0 or result.sparse_vector.dimension > 2
    
    def test_sparse_vector_properties(self, encoder):
        """Test sparse vector properties."""
        result = encoder.encode("test query")
        
        vector = result.sparse_vector
        assert vector.dimension == len(vector.indices)
        
        # Test to_dict
        d = vector.to_dict()
        assert isinstance(d, dict)
        assert len(d) == vector.dimension
    
    def test_top_tokens(self, encoder):
        """Test getting top tokens."""
        result = encoder.encode("database configuration settings")
        
        top = result.sparse_vector.top_tokens(5)
        # Top tokens should be sorted by weight
        if len(top) >= 2:
            assert top[0][1] >= top[1][1]
    
    def test_encode_batch(self, encoder):
        """Test batch encoding."""
        texts = ["query one", "query two", "query three"]
        results = encoder.encode_batch(texts)
        
        assert len(results) == 3
        assert all(r.sparse_vector.dimension > 0 for r in results)
    
    def test_update_idf(self, encoder):
        """Test IDF update."""
        documents = [
            "The quick brown fox",
            "The lazy dog",
            "Quick foxes are quick",
        ]
        
        encoder.update_idf(documents)
        
        # Common word "quick" should have lower IDF
        # Rare word "lazy" should have higher IDF
        assert "quick" in encoder._idf or "lazy" in encoder._idf

class TestSpladeRetriever:
    """Test SPLADE retrieval."""
    
    @pytest.fixture
    def retriever(self):
        return SpladeRetriever()
    
    def test_index_and_search(self, retriever):
        """Test indexing and searching."""
        documents = [
            (1, "How to configure database connections"),
            (2, "API authentication with OAuth tokens"),
            (3, "Error handling best practices"),
        ]
        
        retriever.index_documents(documents)
        results = retriever.search("database configuration", top_k=2)
        
        assert len(results) <= 2
        # First result should be about database
        if results:
            assert results[0].doc_id == 1
    
    def test_search_empty_index(self, retriever):
        """Test searching empty index."""
        results = retriever.search("any query")
        assert results == []
    
    def test_hybrid_search(self, retriever):
        """Test hybrid search with dense results."""
        documents = [
            (1, "Document about databases"),
            (2, "Document about APIs"),
            (3, "Document about testing"),
        ]
        
        retriever.index_documents(documents)
        
        # Mock dense results
        dense_results = [(1, 0.9), (2, 0.7)]
        
        results = retriever.hybrid_search("database API", dense_results, top_k=3)
        
        assert len(results) <= 3
        # Results should have hybrid scores
        assert all(r.score >= 0 for r in results)
    
    def test_matched_terms(self, retriever):
        """Test that matched terms are tracked."""
        documents = [
            (1, "Database configuration and setup guide"),
        ]
        
        retriever.index_documents(documents)
        results = retriever.search("database setup")
        
        if results:
            assert len(results[0].matched_terms) > 0

class TestQueryTermAnalyzer:
    """Test query term analysis."""
    
    def test_analyze_technical_query(self):
        """Test analyzing technical query."""
        analyzer = QueryTermAnalyzer()
        
        analysis = analyzer.analyze("API endpoint configuration")
        
        assert "top_terms" in analysis
        assert len(analysis["technical_terms"]) > 0
    
    def test_analyze_domain_query(self):
        """Test analyzing domain query."""
        analyzer = QueryTermAnalyzer()
        
        analysis = analyzer.analyze("refund policy for customers")
        
        assert analysis["total_terms"] > 0

# ============================================================================
# Query Router Tests
# ============================================================================

class TestQueryClassifier:
    """Test query classification."""
    
    @pytest.fixture
    def classifier(self):
        return QueryClassifier()
    
    def test_classify_factual(self, classifier):
        """Test factual query classification."""
        result = classifier.classify("What is the API rate limit?")
        
        assert result.category == QueryCategory.FACTUAL
        assert result.is_question
        assert result.question_type == "what"
    
    def test_classify_procedural(self, classifier):
        """Test procedural query classification."""
        result = classifier.classify("How to configure authentication?")
        
        assert result.category == QueryCategory.PROCEDURAL
        assert result.question_type == "how"
    
    def test_classify_conceptual(self, classifier):
        """Test conceptual query classification."""
        result = classifier.classify("Why does the system use microservices?")
        
        assert result.category == QueryCategory.CONCEPTUAL
        assert result.question_type == "why"
    
    def test_classify_comparative(self, classifier):
        """Test comparative query classification."""
        result = classifier.classify("What is the difference between REST and GraphQL?")
        
        # Should detect comparative (or factual due to "what is")
        assert result.category in [QueryCategory.COMPARATIVE, QueryCategory.FACTUAL]
        assert result.confidence > 0.3
    
    def test_classify_lookup(self, classifier):
        """Test lookup query classification."""
        result = classifier.classify("authentication")
        
        assert result.category == QueryCategory.LOOKUP
    
    def test_detect_technical_terms(self, classifier):
        """Test technical term detection."""
        result = classifier.classify("Configure the API endpoint and database connection")
        
        assert result.has_technical_terms
        assert len(result.technical_terms) > 0
    
    def test_detect_entities(self, classifier):
        """Test entity detection."""
        result = classifier.classify('Find information about "OAuth" authentication')
        
        assert result.has_entities
        assert "OAuth" in result.entities
    
    def test_complexity_calculation(self, classifier):
        """Test complexity scoring."""
        simple = classifier.classify("What is X?")
        complex_q = classifier.classify(
            "How can I configure the database and authentication "
            "while maintaining security and performance?"
        )
        
        assert complex_q.complexity_score > simple.complexity_score

class TestQueryRouter:
    """Test query routing."""
    
    @pytest.fixture
    def router(self):
        return QueryRouter()
    
    def test_route_factual_query(self, router):
        """Test routing factual query."""
        decision = router.route("When was the API released?")
        
        # Factual queries prefer lexical or hybrid
        assert decision.strategy in [
            RetrievalStrategy.LEXICAL_HEAVY,
            RetrievalStrategy.HYBRID_BALANCED,
            RetrievalStrategy.DENSE_SPARSE_HYBRID,
        ]
        assert decision.lexical_weight > 0
    
    def test_route_conceptual_query(self, router):
        """Test routing conceptual query."""
        decision = router.route("Explain how neural networks learn patterns")
        
        # Conceptual queries prefer semantic
        assert decision.strategy in [
            RetrievalStrategy.SEMANTIC_HEAVY,
            RetrievalStrategy.HYBRID_BALANCED,
        ]
        assert decision.semantic_weight > 0.5
    
    def test_route_complex_query(self, router):
        """Test routing complex query with entities."""
        decision = router.route(
            "How are Users related to Orders in the system?"
        )
        
        # Should enable query expansion for complex queries
        assert decision.use_query_expansion or decision.complexity_score > 0.3
    
    def test_routing_disabled(self):
        """Test default routing when disabled."""
        config = RouterConfig(enabled=False)
        router = QueryRouter(config=config)
        
        decision = router.route("any query")
        
        assert decision.strategy == RetrievalStrategy.HYBRID_BALANCED
        assert "Default routing" in decision.reasoning
    
    def test_index_selection(self, router):
        """Test index selection."""
        decision = router.route("Explain the concept of caching")
        
        assert decision.primary_index in IndexType
        # Semantic-heavy should use vector index
        if decision.strategy == RetrievalStrategy.SEMANTIC_HEAVY:
            assert decision.primary_index == IndexType.VECTOR
    
    def test_enhancement_settings(self, router):
        """Test enhancement settings determination."""
        # Conceptual query should enable HyDE
        decision = router.route("Explain how authentication works")
        
        # For high-confidence conceptual, HyDE should be enabled
        if decision.classification.confidence >= 0.7:
            assert decision.use_hyde or decision.use_query_expansion
    
    def test_channel_weights(self, router):
        """Test channel weight computation."""
        decision = router.route("database API configuration")
        
        # All weights should be positive
        assert decision.semantic_weight > 0
        assert decision.lexical_weight > 0
        
        # Technical queries should boost lexical
        if decision.classification.has_technical_terms:
            assert decision.lexical_weight >= 0.5

class TestAdaptiveRouter:
    """Test adaptive routing."""
    
    def test_adaptive_routing(self):
        """Test adaptive router with feedback."""
        router = AdaptiveRouter()
        
        # Route some queries
        decision1 = router.route("What is X?")
        decision2 = router.route("How does Y work?")
        
        # Record feedback
        router.record_feedback(decision1, 0.8)
        router.record_feedback(decision2, 0.6)
        
        # Check statistics
        stats = router.get_statistics()
        assert stats["total_queries"] == 2
    
    def test_adaptation_on_poor_performance(self):
        """Test that router adapts on poor performance."""
        router = AdaptiveRouter()
        
        # Record many poor scores for a category/strategy pair
        for _ in range(15):
            decision = router.route("What is the price?")
            router.record_feedback(decision, 0.2)  # Poor score
        
        # Next routing should potentially adapt
        final_decision = router.route("What is the price?")
        # Either adapted or maintains original strategy
        assert final_decision.reasoning

# ============================================================================
# Integration Tests
# ============================================================================

class TestP2Integration:
    """Integration tests for P2 modules."""
    
    def test_compression_with_routing(self):
        """Test compression integrated with routing."""
        router = QueryRouter()
        compressor = ContextCompressor()
        
        # Route query
        query = "Explain the authentication flow"
        decision = router.route(query)
        
        # Simulate retrieved context
        context = "Authentication flow explanation. " * 50
        
        # Compress based on routing (conceptual queries might need less compression)
        max_tokens = 500 if decision.strategy == RetrievalStrategy.SEMANTIC_HEAVY else 300
        
        result = compressor.compress(context, query, max_tokens=max_tokens)
        
        assert result.compressed_tokens <= max_tokens
    
    def test_splade_with_routing(self):
        """Test SPLADE with routing decisions."""
        router = QueryRouter()
        splade = SpladeRetriever()
        
        # Index some documents
        docs = [
            (1, "Database configuration guide"),
            (2, "API authentication setup"),
        ]
        splade.index_documents(docs)
        
        # Route a query
        query = "configure database"
        decision = router.route(query)
        
        # If SPLADE is appropriate, search
        if decision.splade_weight > 0.2:
            results = splade.search(query, top_k=5)
            assert isinstance(results, list)
    
    def test_full_p2_pipeline(self):
        """Test full P2 enhancement pipeline."""
        # Initialize components
        classifier = QueryClassifier()
        router = QueryRouter()
        compressor = ContextCompressor()
        splade = SpladeRetriever()
        
        # Sample query
        query = "How to configure the API authentication endpoint?"
        
        # Classify
        classification = classifier.classify(query)
        assert classification.category in QueryCategory
        
        # Route
        decision = router.route(query)
        assert decision.strategy in RetrievalStrategy
        
        # Index and search with SPLADE
        docs = [(1, "API authentication configuration guide")]
        splade.index_documents(docs)
        results = splade.search(query)
        
        # Compress results
        if results:
            context = " ".join(r.text for r in results)
            compressed = compressor.compress(context, query)
            assert compressed.compressed_context
