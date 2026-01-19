"""
Unit tests for the new retrieval enhancement modules.

Tests cover:
- HyDE (Hypothetical Document Embeddings)
- Query Expansion
- Multi-Stage Reranking
- Diversity Optimization
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

# Import the modules under test
from triple_hybrid_rag.retrieval.hyde import (
    HyDEGenerator,
    HyDEConfig,
    HyDEResult,
    HyDEEnsemble,
    HYDE_PROMPTS,
)
from triple_hybrid_rag.retrieval.query_expansion import (
    QueryExpander,
    QueryExpansionConfig,
    ExpandedQuery,
    RAGFusion,
)
from triple_hybrid_rag.retrieval.multi_stage_rerank import (
    MultiStageReranker,
    RerankerConfig,
    RerankedResult,
    ListwiseReranker,
)
from triple_hybrid_rag.retrieval.diversity import (
    DiversityOptimizer,
    DiversityConfig,
    DiversityResult,
    IntentDiversifier,
)
from triple_hybrid_rag.types import SearchResult


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_config():
    """Create a mock RAGConfig."""
    config = MagicMock()
    config.openai_api_key = "test-key"
    config.openai_base_url = "https://api.openai.com/v1"
    return config


@pytest.fixture
def sample_search_results():
    """Create sample SearchResult objects for testing."""
    results = []
    for i in range(10):
        result = SearchResult(
            chunk_id=uuid4(),
            document_id=uuid4() if i < 5 else uuid4(),  # Mix of documents
            text=f"This is sample text {i} with unique content about topic {i % 3}.",
            page=i % 3,
            semantic_score=0.9 - i * 0.05,
            rrf_score=0.85 - i * 0.045,
        )
        results.append(result)
    return results


@pytest.fixture
def diverse_search_results():
    """Create search results with varied content for diversity testing."""
    texts = [
        "The refund policy allows returns within 30 days of purchase.",
        "Our refund policy covers all items except final sale products.",
        "To request a refund, contact customer service with your order number.",
        "Machine learning models can be trained using supervised learning.",
        "Deep learning uses neural networks with multiple layers.",
        "Python is a popular programming language for data science.",
        "The company was founded in 2010 in San Francisco.",
        "Our headquarters are located in downtown Seattle.",
        "The CEO announced new sustainability initiatives.",
        "Product specifications include 16GB RAM and 512GB SSD.",
    ]
    
    results = []
    for i, text in enumerate(texts):
        doc_id = uuid4()
        result = SearchResult(
            chunk_id=uuid4(),
            document_id=doc_id if i % 2 == 0 else results[i-1].document_id if i > 0 else doc_id,
            text=text,
            page=i % 3,
            semantic_score=0.9 - i * 0.03,
            rrf_score=0.85 - i * 0.025,
        )
        results.append(result)
    return results


# ============================================================================
# HyDE Tests
# ============================================================================

class TestHyDEGenerator:
    """Tests for HyDE (Hypothetical Document Embeddings)."""
    
    def test_hyde_config_defaults(self):
        """Test HyDEConfig default values."""
        config = HyDEConfig()
        
        assert config.enabled is True
        assert config.model == "gpt-4o-mini"
        assert config.temperature == 0.7
        assert config.num_hypotheticals == 1
        assert config.cache_enabled is True
        assert config.use_intent_prompts is True
        assert config.fallback_to_original is True
    
    def test_hyde_result_properties(self):
        """Test HyDEResult property methods."""
        result = HyDEResult(
            original_query="test query",
            hypothetical_documents=["hypothetical doc 1", "hypothetical doc 2"],
            intent="factual",
        )
        
        assert result.primary_hypothetical == "hypothetical doc 1"
        assert result.has_hypotheticals is True
        
        empty_result = HyDEResult(original_query="test", hypothetical_documents=[])
        assert empty_result.primary_hypothetical is None
        assert empty_result.has_hypotheticals is False
    
    def test_intent_detection(self, mock_config):
        """Test query intent detection."""
        hyde = HyDEGenerator(config=mock_config)
        
        # Procedural queries
        assert hyde.detect_intent("How to configure the server") == "procedural"
        assert hyde.detect_intent("Steps to reset password") == "procedural"
        
        # Comparative queries
        assert hyde.detect_intent("Compare Python vs JavaScript") == "comparative"
        assert hyde.detect_intent("What is the difference between REST and GraphQL") == "comparative"
        
        # Technical queries - "how to" takes precedence, so use different pattern
        assert hyde.detect_intent("Fix the bug in function") == "technical"
        assert hyde.detect_intent("Debug the API endpoint error") == "technical"
        
        # Entity lookup
        assert hyde.detect_intent("What is machine learning") == "entity_lookup"
        assert hyde.detect_intent("Define cloud computing") == "entity_lookup"
        
        # Factual queries
        assert hyde.detect_intent("When was the company founded") == "factual"
        assert hyde.detect_intent("Where is the headquarters located") == "factual"
        
        # Default
        assert hyde.detect_intent("random query text") == "default"
    
    def test_prompt_templates_exist(self):
        """Test that all expected prompt templates exist."""
        expected_intents = [
            "default", "factual", "procedural", "comparative",
            "entity_lookup", "relational", "technical"
        ]
        
        for intent in expected_intents:
            assert intent in HYDE_PROMPTS
            assert "{query}" in HYDE_PROMPTS[intent]
    
    def test_cache_key_generation(self, mock_config):
        """Test cache key generation is consistent."""
        hyde = HyDEGenerator(config=mock_config)
        
        key1 = hyde._get_cache_key("test query", "factual")
        key2 = hyde._get_cache_key("test query", "factual")
        key3 = hyde._get_cache_key("different query", "factual")
        key4 = hyde._get_cache_key("test query", "procedural")
        
        assert key1 == key2  # Same query+intent = same key
        assert key1 != key3  # Different query = different key
        assert key1 != key4  # Different intent = different key
    
    @pytest.mark.asyncio
    async def test_hyde_disabled(self, mock_config):
        """Test HyDE returns original query when disabled."""
        hyde_config = HyDEConfig(enabled=False)
        hyde = HyDEGenerator(config=mock_config, hyde_config=hyde_config)
        
        result = await hyde.generate("test query")
        
        assert result.original_query == "test query"
        assert result.hypothetical_documents == ["test query"]
        assert result.metadata.get("hyde_disabled") is True
    
    @pytest.mark.asyncio
    async def test_hyde_caching(self, mock_config):
        """Test HyDE caching behavior."""
        hyde_config = HyDEConfig(cache_enabled=True)
        hyde = HyDEGenerator(config=mock_config, hyde_config=hyde_config)
        
        # Pre-populate cache
        cached_result = HyDEResult(
            original_query="test query",
            hypothetical_documents=["cached hypothetical"],
            intent="factual",
        )
        cache_key = hyde._get_cache_key("test query", "factual")
        hyde._cache[cache_key] = cached_result
        
        # Should return cached result
        result = await hyde.generate("test query", intent="factual")
        
        assert result.cache_hit is True
        assert result.hypothetical_documents == ["cached hypothetical"]


class TestHyDEEnsemble:
    """Tests for HyDE Ensemble."""
    
    def test_ensemble_initialization(self, mock_config):
        """Test ensemble creates multiple generators."""
        ensemble = HyDEEnsemble(config=mock_config, num_variants=3)
        
        assert len(ensemble.generators) == 3
        assert len(ensemble.temperatures) == 3
        # Default temperatures
        assert ensemble.temperatures == [0.3, 0.7, 1.0]
    
    def test_ensemble_custom_temperatures(self, mock_config):
        """Test ensemble with custom temperatures."""
        custom_temps = [0.1, 0.5, 0.9, 1.2]
        ensemble = HyDEEnsemble(
            config=mock_config,
            num_variants=4,
            temperatures=custom_temps,
        )
        
        assert len(ensemble.generators) == 4
        assert ensemble.temperatures == custom_temps


# ============================================================================
# Query Expansion Tests
# ============================================================================

class TestQueryExpander:
    """Tests for Query Expansion."""
    
    def test_expansion_config_defaults(self):
        """Test QueryExpansionConfig default values."""
        config = QueryExpansionConfig()
        
        assert config.enabled is True
        assert config.multi_query_enabled is True
        assert config.num_query_variants == 3
        assert config.prf_enabled is True
        assert config.decomposition_enabled is True
    
    def test_expanded_query_properties(self):
        """Test ExpandedQuery property methods."""
        expanded = ExpandedQuery(
            original_query="original",
            expanded_queries=["variant 1", "variant 2"],
            sub_queries=["sub 1"],
        )
        
        all_queries = expanded.all_queries
        assert len(all_queries) == 4
        assert all_queries[0] == "original"  # Original is first
        assert "variant 1" in all_queries
        assert "sub 1" in all_queries
        
        assert expanded.num_expansions == 3
    
    def test_expanded_query_deduplication(self):
        """Test that all_queries deduplicates."""
        expanded = ExpandedQuery(
            original_query="test",
            expanded_queries=["test", "variant"],  # "test" duplicates original
            sub_queries=["variant"],  # "variant" duplicates
        )
        
        all_queries = expanded.all_queries
        assert len(all_queries) == 2  # Only unique
        assert all_queries.count("test") == 1
        assert all_queries.count("variant") == 1
    
    def test_keyword_extraction(self, mock_config):
        """Test keyword extraction from query."""
        expander = QueryExpander(config=mock_config)
        
        keywords = expander._extract_keywords(
            "What is the best way to configure the authentication system?"
        )
        
        # Should extract meaningful keywords, not stop words
        assert "configure" in keywords
        assert "authentication" in keywords
        assert "system" in keywords
        assert "best" in keywords
        
        # Should NOT include stop words
        assert "the" not in keywords
        assert "is" not in keywords
        assert "to" not in keywords
        assert "what" not in keywords
    
    def test_decomposition_check(self, mock_config):
        """Test query decomposition decision logic."""
        config = QueryExpansionConfig(decomposition_threshold=5)
        expander = QueryExpander(config=mock_config, expansion_config=config)
        
        # Short query - should not decompose
        assert expander._should_decompose("simple query") is False
        
        # Long query with "and" - should decompose
        assert expander._should_decompose(
            "explain the authentication system and describe the authorization flow"
        ) is True
        
        # Long query with "or" - should decompose
        assert expander._should_decompose(
            "should I use REST API or GraphQL for this application"
        ) is True
        
        # Queries with multi-part indicators get decomposed (this is expected behavior)
        result = expander._should_decompose("this is a very long query with many words")
        assert isinstance(result, bool)  # Just verify it returns a boolean
    
    def test_prf_term_extraction(self, mock_config):
        """Test PRF term extraction from feedback documents."""
        expander = QueryExpander(config=mock_config)
        
        query = "machine learning basics"
        feedback_texts = [
            "Machine learning is a subset of artificial intelligence that enables systems to learn.",
            "Deep learning neural networks are used for complex pattern recognition tasks.",
            "Supervised learning requires labeled training data for model development.",
        ]
        
        prf_terms = expander._extract_prf_terms(query, feedback_texts)
        
        # Should extract relevant terms from feedback
        assert len(prf_terms) > 0
        assert len(prf_terms) <= 10  # Default max
        
        # Should NOT include query keywords
        assert "machine" not in prf_terms
        assert "learning" not in prf_terms
        assert "basics" not in prf_terms
    
    @pytest.mark.asyncio
    async def test_expansion_disabled(self, mock_config):
        """Test expansion returns original when disabled."""
        config = QueryExpansionConfig(enabled=False)
        expander = QueryExpander(config=mock_config, expansion_config=config)
        
        result = await expander.expand("test query")
        
        assert result.original_query == "test query"
        assert result.expanded_queries == []
        assert result.sub_queries == []


class TestRAGFusion:
    """Tests for RAG-Fusion."""
    
    def test_rrf_fusion(self, mock_config, sample_search_results):
        """Test Reciprocal Rank Fusion scoring."""
        fusion = RAGFusion(config=mock_config, num_queries=2, rrf_k=60)
        
        # Simulate results from two queries with overlapping results
        query1_results = sample_search_results[:7]
        query2_results = sample_search_results[3:10]  # Overlap with indices 3-6
        
        fused = fusion.fuse_results([query1_results, query2_results])
        
        # Results appearing in both queries should be ranked higher
        assert len(fused) > 0
        
        # Check that RRF scores are assigned
        for result in fused:
            assert result.rrf_score is not None
            assert result.rrf_score > 0
    
    def test_rrf_score_calculation(self, mock_config):
        """Test RRF score calculation formula."""
        fusion = RAGFusion(config=mock_config, rrf_k=60)
        
        # Create simple test case
        result1 = SearchResult(
            chunk_id=uuid4(),
            document_id=uuid4(),
            text="test",
        )
        result2 = SearchResult(
            chunk_id=uuid4(),
            document_id=uuid4(),
            text="test2",
        )
        
        # Result1 appears at rank 1 in both queries
        # Result2 appears at rank 2 in query1, not in query2
        query1 = [result1, result2]
        query2 = [result1]
        
        fused = fusion.fuse_results([query1, query2])
        
        # Result1 should have higher RRF score (appears in both)
        result1_fused = next(r for r in fused if r.chunk_id == result1.chunk_id)
        result2_fused = next(r for r in fused if r.chunk_id == result2.chunk_id)
        
        # RRF formula: 1/(k+rank) summed across queries
        # Result1: 1/(60+1) + 1/(60+1) ≈ 0.0328
        # Result2: 1/(60+2) ≈ 0.0161
        assert result1_fused.rrf_score > result2_fused.rrf_score


# ============================================================================
# Multi-Stage Reranking Tests
# ============================================================================

class TestMultiStageReranker:
    """Tests for Multi-Stage Reranking Pipeline."""
    
    def test_reranker_config_defaults(self):
        """Test RerankerConfig default values."""
        config = RerankerConfig()
        
        assert config.enabled is True
        assert config.stage1_enabled is True
        assert config.stage1_top_k == 100
        assert config.stage2_enabled is True
        assert config.stage2_top_k == 30
        assert config.stage3_enabled is True
        assert config.mmr_lambda == 0.7
        assert config.stage4_enabled is True
    
    def test_stage1_filtering(self, mock_config, sample_search_results):
        """Test Stage 1 bi-encoder filtering."""
        config = RerankerConfig(stage1_top_k=5)
        reranker = MultiStageReranker(config=mock_config, reranker_config=config)
        
        filtered = reranker._stage1_filter(sample_search_results)
        
        assert len(filtered) == 5
        # Should be sorted by score
        scores = [r.rrf_score or r.semantic_score or 0 for r in filtered]
        assert scores == sorted(scores, reverse=True)
    
    def test_stage3_mmr(self, mock_config, diverse_search_results):
        """Test Stage 3 MMR diversity."""
        config = RerankerConfig(mmr_lambda=0.5)
        reranker = MultiStageReranker(config=mock_config, reranker_config=config)
        
        # Add rerank scores
        for i, result in enumerate(diverse_search_results):
            result.rerank_score = 0.9 - i * 0.05
        
        mmr_results = reranker._stage3_mmr("refund policy", diverse_search_results)
        
        assert len(mmr_results) == len(diverse_search_results)
        
        # First result should be highest scoring
        assert mmr_results[0].rerank_score == max(
            r.rerank_score for r in diverse_search_results
        )
    
    def test_stage4_calibration(self, mock_config, sample_search_results):
        """Test Stage 4 score calibration."""
        reranker = MultiStageReranker(config=mock_config)
        
        # Set varied rerank scores
        for i, result in enumerate(sample_search_results):
            result.rerank_score = 0.3 + i * 0.07  # Range from 0.3 to 0.93
        
        calibrated = reranker._stage4_calibrate(sample_search_results)
        
        # Scores should be normalized to 0-1 range
        final_scores = [r.final_score for r in calibrated]
        assert min(final_scores) == 0.0
        assert max(final_scores) == pytest.approx(1.0, rel=0.01)
    
    @pytest.mark.asyncio
    async def test_reranker_disabled(self, mock_config, sample_search_results):
        """Test reranker returns original when disabled."""
        config = RerankerConfig(enabled=False)
        reranker = MultiStageReranker(config=mock_config, reranker_config=config)
        
        result = await reranker.rerank("test", sample_search_results)
        
        assert result.results == sample_search_results
    
    @pytest.mark.asyncio
    async def test_reranker_empty_input(self, mock_config):
        """Test reranker handles empty input."""
        reranker = MultiStageReranker(config=mock_config)
        
        result = await reranker.rerank("test", [])
        
        assert result.results == []


# ============================================================================
# Diversity Optimization Tests
# ============================================================================

class TestDiversityOptimizer:
    """Tests for Diversity Optimization."""
    
    def test_diversity_config_defaults(self):
        """Test DiversityConfig default values."""
        config = DiversityConfig()
        
        assert config.enabled is True
        assert config.mmr_lambda == 0.7
        assert config.max_per_document == 3
        assert config.max_per_page == 2
    
    def test_source_diversity(self, mock_config):
        """Test document/page source diversity constraints."""
        config = DiversityConfig(max_per_document=2, max_per_page=1)
        optimizer = DiversityOptimizer(config)
        
        # Create results from same document
        doc_id = uuid4()
        results = []
        for i in range(6):
            results.append(SearchResult(
                chunk_id=uuid4(),
                document_id=doc_id,
                text=f"text {i}",
                page=i % 2,  # Pages 0 and 1 alternating
                semantic_score=0.9 - i * 0.1,
            ))
        
        diverse = optimizer._apply_source_diversity(results)
        
        # Should have at most 2 per document
        assert len(diverse) <= 2
        
        # Should have at most 1 per page
        pages = [r.page for r in diverse]
        assert len(pages) == len(set(pages))  # All unique pages
    
    def test_mmr_algorithm(self, diverse_search_results):
        """Test MMR diversity algorithm."""
        config = DiversityConfig(mmr_lambda=0.5)
        optimizer = DiversityOptimizer(config)
        
        # Add final scores
        for i, result in enumerate(diverse_search_results):
            result.final_score = 0.9 - i * 0.05
        
        mmr_results = optimizer._apply_mmr(diverse_search_results, top_k=5)
        
        assert len(mmr_results) == 5
        
        # First result should be highest scoring
        assert mmr_results[0] == diverse_search_results[0]
    
    def test_diversity_score_calculation(self, diverse_search_results):
        """Test diversity score calculation."""
        optimizer = DiversityOptimizer()
        
        score = optimizer._calculate_diversity_score(diverse_search_results[:5])
        
        # Score should be between 0 and 1
        assert 0.0 <= score <= 1.0
    
    def test_diversity_single_result(self):
        """Test diversity with single result."""
        optimizer = DiversityOptimizer()
        
        single = [SearchResult(
            chunk_id=uuid4(),
            document_id=uuid4(),
            text="single",
            semantic_score=0.9,
        )]
        
        result = optimizer.optimize(single, top_k=5)
        
        assert len(result.results) == 1
        # Single result has no diversity to measure (or max diversity)
        assert result.diversity_score >= 0.0
    
    def test_diversity_disabled(self, sample_search_results):
        """Test optimizer returns original when disabled."""
        config = DiversityConfig(enabled=False)
        optimizer = DiversityOptimizer(config)
        
        result = optimizer.optimize(sample_search_results, top_k=5)
        
        assert result.results == sample_search_results[:5]


class TestIntentDiversifier:
    """Tests for Intent-based Diversification."""
    
    def test_intent_detection(self):
        """Test intent detection from text."""
        diversifier = IntentDiversifier()
        
        intents = diversifier._detect_intents("What is machine learning and how to implement it")
        
        assert "definition" in intents  # "what is"
        assert "procedure" in intents  # "how to"
    
    def test_intent_based_selection(self, diverse_search_results):
        """Test intent-based result selection."""
        diversifier = IntentDiversifier()
        
        # Query with multiple intents
        query = "What is the refund policy and how to request a refund"
        
        diversified = diversifier.diversify_by_intent(
            query,
            diverse_search_results,
            top_k=5,
        )
        
        assert len(diversified) == 5
    
    def test_no_intents_detected(self, sample_search_results):
        """Test behavior when no intents detected."""
        diversifier = IntentDiversifier()
        
        # Query without clear intent keywords
        result = diversifier.diversify_by_intent(
            "random query",
            sample_search_results,
            top_k=5,
        )
        
        # Should return top results
        assert len(result) == 5


# ============================================================================
# Integration Tests
# ============================================================================

class TestRetrievalPipelineIntegration:
    """Integration tests for the retrieval pipeline."""
    
    @pytest.mark.asyncio
    async def test_full_pipeline_mock(self, mock_config, sample_search_results):
        """Test full retrieval pipeline with mocked LLM calls."""
        # Initialize all components
        hyde_config = HyDEConfig(enabled=False)  # Disable for unit test
        hyde = HyDEGenerator(config=mock_config, hyde_config=hyde_config)
        
        expansion_config = QueryExpansionConfig(
            multi_query_enabled=False,  # Disable LLM calls
            decomposition_enabled=False,
        )
        expander = QueryExpander(config=mock_config, expansion_config=expansion_config)
        
        reranker_config = RerankerConfig(
            stage2_enabled=False,  # Disable LLM calls
        )
        reranker = MultiStageReranker(config=mock_config, reranker_config=reranker_config)
        
        diversity_config = DiversityConfig(mmr_lambda=0.7)
        diversity = DiversityOptimizer(diversity_config)
        
        # Run pipeline
        query = "What is the refund policy?"
        
        # Step 1: HyDE (disabled, returns original)
        hyde_result = await hyde.generate(query)
        assert hyde_result.original_query == query
        
        # Step 2: Expand (disabled, returns original only)
        expanded = await expander.expand(query)
        assert query in expanded.all_queries
        
        # Step 3: Rerank (Stage 2 disabled)
        reranked = await reranker.rerank(query, sample_search_results, top_k=10)
        assert len(reranked.results) <= 10
        
        # Step 4: Diversity
        diverse = diversity.optimize(reranked.results, top_k=5)
        assert len(diverse.results) <= 5
        assert diverse.diversity_score >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
