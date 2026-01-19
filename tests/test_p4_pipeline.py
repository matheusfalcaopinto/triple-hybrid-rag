"""
Tests for P4 Enhanced Pipeline

Tests for:
- EnhancedRAGPipeline
- PipelineBuilder
- Pipeline integration
"""

import pytest
import time
from typing import List, Dict, Any
from dataclasses import dataclass

from triple_hybrid_rag.pipeline import (
    EnhancedRAGPipeline,
    PipelineConfig,
    PipelineResult,
    RetrievalContext,
    PipelineBuilder,
)
from triple_hybrid_rag.pipeline.builder import create_pipeline

# ============================================================================
# Mock Components
# ============================================================================

@dataclass
class MockResult:
    """Mock search result."""
    doc_id: str
    text: str
    score: float
    source: str = "semantic"

def create_mock_results(prefix: str, count: int, source: str = "semantic") -> List[MockResult]:
    """Create mock search results."""
    return [
        MockResult(
            doc_id=f"{prefix}_{i}",
            text=f"Content {i} from {source}",
            score=1.0 - (i * 0.1),
            source=source,
        )
        for i in range(count)
    ]

def mock_semantic_search(query: str, top_k: int = 10) -> List[MockResult]:
    """Mock semantic search."""
    return create_mock_results("sem", min(top_k, 5), "semantic")

def mock_lexical_search(query: str, top_k: int = 10) -> List[MockResult]:
    """Mock lexical search."""
    return create_mock_results("lex", min(top_k, 5), "lexical")

def mock_graph_search(query: str, top_k: int = 10) -> List[MockResult]:
    """Mock graph search."""
    return create_mock_results("graph", min(top_k, 3), "graph")

# ============================================================================
# Pipeline Config Tests
# ============================================================================

class TestPipelineConfig:
    """Test pipeline configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = PipelineConfig()
        
        assert config.enable_hyde is True
        assert config.enable_reranking is True
        assert config.semantic_top_k == 50
        assert config.final_top_k == 10
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = PipelineConfig(
            enable_hyde=False,
            final_top_k=20,
            semantic_weight=0.8,
        )
        
        assert config.enable_hyde is False
        assert config.final_top_k == 20
        assert config.semantic_weight == 0.8

# ============================================================================
# Enhanced RAG Pipeline Tests
# ============================================================================

class TestEnhancedRAGPipeline:
    """Test enhanced RAG pipeline."""
    
    @pytest.fixture
    def minimal_pipeline(self):
        """Create minimal pipeline."""
        config = PipelineConfig(
            enable_hyde=False,
            enable_query_expansion=False,
            enable_splade=False,
            enable_reranking=False,
            enable_diversity=False,
            enable_compression=False,
            enable_routing=False,
            enable_caching=False,
            enable_observability=False,
        )
        return EnhancedRAGPipeline(
            config=config,
            semantic_search_fn=mock_semantic_search,
            lexical_search_fn=mock_lexical_search,
        )
    
    def test_basic_retrieval(self, minimal_pipeline):
        """Test basic retrieval without enhancements."""
        result = minimal_pipeline.retrieve("What is machine learning?")
        
        assert result.success
        assert len(result.results) > 0
        assert result.query == "What is machine learning?"
    
    def test_result_properties(self, minimal_pipeline):
        """Test result has expected properties."""
        result = minimal_pipeline.retrieve("test query")
        
        assert isinstance(result, PipelineResult)
        assert hasattr(result, 'query')
        assert hasattr(result, 'results')
        assert hasattr(result, 'context')
        assert hasattr(result, 'total_time_ms')
        assert hasattr(result, 'stages_completed')
    
    def test_context_populated(self, minimal_pipeline):
        """Test that context is populated."""
        result = minimal_pipeline.retrieve("test query")
        
        ctx = result.context
        assert ctx.query == "test query"
        assert len(ctx.semantic_results) > 0
        assert len(ctx.lexical_results) > 0
    
    def test_timing_recorded(self, minimal_pipeline):
        """Test that timing is recorded."""
        result = minimal_pipeline.retrieve("test query")
        
        assert result.total_time_ms > 0
        assert 'semantic' in result.context.timings
        assert 'lexical' in result.context.timings
    
    def test_stages_completed(self, minimal_pipeline):
        """Test stages are tracked."""
        result = minimal_pipeline.retrieve("test query")
        
        assert 'retrieval' in result.stages_completed
        assert 'fusion' in result.stages_completed
    
    def test_multi_channel_fusion(self, minimal_pipeline):
        """Test that results from multiple channels are fused."""
        result = minimal_pipeline.retrieve("test query")
        
        # Should have candidates from both channels
        assert result.total_candidates > 0
        
        # Fused results should exist
        assert len(result.results) > 0
    
    def test_top_k_limit(self, minimal_pipeline):
        """Test top_k limits results."""
        result = minimal_pipeline.retrieve("test query", top_k=3)
        
        assert len(result.results) <= 3
    
    def test_empty_query_handling(self, minimal_pipeline):
        """Test handling of empty query."""
        result = minimal_pipeline.retrieve("")
        
        # Should still return a result object
        assert isinstance(result, PipelineResult)

class TestPipelineWithEnhancements:
    """Test pipeline with enhancement components."""
    
    def test_with_routing(self):
        """Test pipeline with routing enabled."""
        pipeline = (
            PipelineBuilder()
            .with_semantic_search(mock_semantic_search)
            .with_lexical_search(mock_lexical_search)
            .enable_routing()
            .disable_hyde()
            .disable_query_expansion()
            .disable_reranking()
            .disable_diversity()
            .disable_compression()
            .disable_caching()
            .build()
        )
        
        result = pipeline.retrieve("What is Python?")
        
        assert result.success
        assert 'routing' in result.stages_completed
    
    def test_with_reranking(self):
        """Test pipeline with reranking enabled."""
        pipeline = (
            PipelineBuilder()
            .with_semantic_search(mock_semantic_search)
            .enable_reranking()
            .disable_hyde()
            .disable_query_expansion()
            .disable_routing()
            .disable_diversity()
            .disable_compression()
            .disable_caching()
            .build()
        )
        
        result = pipeline.retrieve("test query")
        
        assert result.success
        assert 'reranking' in result.stages_completed
    
    def test_with_diversity(self):
        """Test pipeline with diversity enabled."""
        pipeline = (
            PipelineBuilder()
            .with_semantic_search(mock_semantic_search)
            .enable_diversity()
            .disable_hyde()
            .disable_query_expansion()
            .disable_routing()
            .disable_reranking()
            .disable_compression()
            .disable_caching()
            .build()
        )
        
        result = pipeline.retrieve("test query")
        
        assert result.success
        assert 'diversity' in result.stages_completed
    
    def test_with_caching(self):
        """Test pipeline with caching enabled."""
        pipeline = (
            PipelineBuilder()
            .with_semantic_search(mock_semantic_search)
            .enable_caching()
            .disable_hyde()
            .disable_query_expansion()
            .disable_routing()
            .disable_reranking()
            .disable_diversity()
            .disable_compression()
            .build()
        )
        
        # First query - cache miss
        result1 = pipeline.retrieve("test query")
        assert not result1.cache_hit
        
        # Second query - cache hit
        result2 = pipeline.retrieve("test query")
        assert result2.cache_hit
    
    def test_with_observability(self):
        """Test pipeline with observability enabled."""
        pipeline = (
            PipelineBuilder()
            .with_semantic_search(mock_semantic_search)
            .enable_observability()
            .disable_hyde()
            .disable_query_expansion()
            .disable_routing()
            .disable_reranking()
            .disable_diversity()
            .disable_compression()
            .disable_caching()
            .build()
        )
        
        result = pipeline.retrieve("test query")
        
        assert result.success
        assert result.context.trace_id is not None

# ============================================================================
# Pipeline Builder Tests
# ============================================================================

class TestPipelineBuilder:
    """Test pipeline builder."""
    
    def test_basic_builder(self):
        """Test basic builder usage."""
        pipeline = (
            PipelineBuilder()
            .with_semantic_search(mock_semantic_search)
            .minimal()
            .build()
        )
        
        assert isinstance(pipeline, EnhancedRAGPipeline)
    
    def test_fluent_api(self):
        """Test fluent API chaining."""
        builder = PipelineBuilder()
        
        result = (
            builder
            .with_semantic_search(mock_semantic_search)
            .with_lexical_search(mock_lexical_search)
            .with_config(final_top_k=5)
        )
        
        assert result is builder  # Returns self for chaining
    
    def test_config_override(self):
        """Test config override via builder."""
        pipeline = (
            PipelineBuilder()
            .with_semantic_search(mock_semantic_search)
            .with_config(final_top_k=5, semantic_weight=0.9)
            .minimal()
            .build()
        )
        
        assert pipeline.config.final_top_k == 5
        assert pipeline.config.semantic_weight == 0.9
    
    def test_minimal_preset(self):
        """Test minimal preset."""
        pipeline = (
            PipelineBuilder()
            .with_semantic_search(mock_semantic_search)
            .minimal()
            .build()
        )
        
        assert not pipeline.config.enable_hyde
        assert not pipeline.config.enable_reranking
        assert not pipeline.config.enable_diversity
    
    def test_enable_disable_features(self):
        """Test enabling and disabling features."""
        pipeline = (
            PipelineBuilder()
            .with_semantic_search(mock_semantic_search)
            .enable_reranking()
            .enable_caching()
            .disable_hyde()
            .disable_diversity()
            .build()
        )
        
        assert pipeline.config.enable_reranking
        assert pipeline.config.enable_caching
        assert not pipeline.config.enable_hyde
        assert not pipeline.config.enable_diversity

class TestCreatePipeline:
    """Test create_pipeline factory function."""
    
    def test_minimal_preset(self):
        """Test minimal preset factory."""
        pipeline = create_pipeline(
            semantic_search=mock_semantic_search,
            preset="minimal",
        )
        
        assert isinstance(pipeline, EnhancedRAGPipeline)
        assert not pipeline.config.enable_hyde
    
    def test_fast_preset(self):
        """Test fast preset factory."""
        pipeline = create_pipeline(
            semantic_search=mock_semantic_search,
            preset="fast",
        )
        
        assert pipeline.config.enable_routing
        assert pipeline.config.enable_caching
    
    def test_balanced_preset(self):
        """Test balanced preset factory."""
        pipeline = create_pipeline(
            semantic_search=mock_semantic_search,
            preset="balanced",
        )
        
        assert pipeline.config.enable_routing
        assert pipeline.config.enable_reranking
        assert pipeline.config.enable_diversity
        assert pipeline.config.enable_caching
    
    def test_with_multiple_search_functions(self):
        """Test factory with multiple search functions."""
        pipeline = create_pipeline(
            semantic_search=mock_semantic_search,
            lexical_search=mock_lexical_search,
            graph_search=mock_graph_search,
            preset="minimal",
        )
        
        result = pipeline.retrieve("test query")
        
        assert result.success
        assert len(result.context.semantic_results) > 0
        assert len(result.context.lexical_results) > 0
        assert len(result.context.graph_results) > 0
    
    def test_config_kwargs(self):
        """Test config override via kwargs."""
        pipeline = create_pipeline(
            semantic_search=mock_semantic_search,
            preset="minimal",
            final_top_k=3,
        )
        
        assert pipeline.config.final_top_k == 3

# ============================================================================
# Integration Tests
# ============================================================================

class TestP4Integration:
    """Integration tests for P4 pipeline."""
    
    def test_full_pipeline_flow(self):
        """Test complete pipeline flow with multiple components."""
        pipeline = (
            PipelineBuilder()
            .with_semantic_search(mock_semantic_search)
            .with_lexical_search(mock_lexical_search)
            .enable_routing()
            .enable_reranking()
            .enable_diversity()
            .enable_caching()
            .enable_observability()
            .with_config(final_top_k=5)
            .build()
        )
        
        # First query
        result1 = pipeline.retrieve("What is machine learning?")
        
        assert result1.success
        assert len(result1.results) <= 5
        assert not result1.cache_hit
        
        # Expected stages
        expected_stages = ['routing', 'retrieval', 'fusion', 'reranking', 'diversity']
        for stage in expected_stages:
            assert stage in result1.stages_completed, f"Missing stage: {stage}"
        
        # Second query - should hit cache
        result2 = pipeline.retrieve("What is machine learning?")
        assert result2.cache_hit
    
    def test_performance_timing(self):
        """Test that performance timing is tracked."""
        pipeline = (
            PipelineBuilder()
            .with_semantic_search(mock_semantic_search)
            .with_lexical_search(mock_lexical_search)
            .enable_routing()
            .enable_diversity()
            .minimal()  # Override to minimal but keep routing/diversity
            .enable_routing()
            .enable_diversity()
            .build()
        )
        
        result = pipeline.retrieve("test query")
        
        # Should have timing for various stages
        timings = result.context.timings
        assert 'semantic' in timings
        assert 'lexical' in timings
        assert 'routing' in timings
        assert 'diversity' in timings
        
        # Total time should be reasonable
        assert result.total_time_ms > 0
        assert result.total_time_ms < 5000  # Should be fast for mocks
    
    def test_error_resilience(self):
        """Test that pipeline handles component failures gracefully."""
        def failing_search(query: str, top_k: int = 10):
            raise ValueError("Search failed")
        
        # Pipeline should not crash if one component fails
        pipeline = (
            PipelineBuilder()
            .with_semantic_search(mock_semantic_search)
            .with_graph_search(failing_search)  # This will fail
            .minimal()
            .build()
        )
        
        result = pipeline.retrieve("test query")
        
        # Should still return results from working components
        assert result.success
        assert len(result.results) > 0
    
    def test_concurrent_queries(self):
        """Test pipeline handles concurrent queries."""
        import threading
        
        pipeline = (
            PipelineBuilder()
            .with_semantic_search(mock_semantic_search)
            .enable_caching()
            .minimal()
            .enable_caching()
            .build()
        )
        
        results = []
        errors = []
        
        def query(q: str):
            try:
                r = pipeline.retrieve(q)
                results.append(r)
            except Exception as e:
                errors.append(e)
        
        # Run concurrent queries
        threads = [
            threading.Thread(target=query, args=(f"query {i}",))
            for i in range(5)
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        assert len(results) == 5
        assert all(r.success for r in results)
    
    def test_result_quality_metrics(self):
        """Test that quality metrics are calculated."""
        pipeline = (
            PipelineBuilder()
            .with_semantic_search(mock_semantic_search)
            .with_lexical_search(mock_lexical_search)
            .minimal()
            .build()
        )
        
        result = pipeline.retrieve("test query")
        
        assert result.total_candidates > 0
        assert result.final_count > 0
        assert result.avg_score >= 0.0
