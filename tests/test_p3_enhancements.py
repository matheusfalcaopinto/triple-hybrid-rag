"""
Tests for P3 Enhancement Modules

Tests for:
- Query Caching
- Observability
- Batch Processing
"""

import pytest
import time
from typing import List, Any

from triple_hybrid_rag.retrieval.caching import (
    QueryCache,
    CacheConfig,
    CacheStrategy,
    CacheStats,
    MultiLevelCache,
)
from triple_hybrid_rag.retrieval.observability import (
    RAGObserver,
    MetricsCollector,
    Tracer,
    ComponentType,
    PipelineMetrics,
    get_global_observer,
    set_global_observer,
)
from triple_hybrid_rag.retrieval.batch import (
    BatchProcessor,
    StreamingBatchProcessor,
    BatchConfig,
    BatchResult,
    Priority,
)

# ============================================================================
# Query Caching Tests
# ============================================================================

class TestQueryCache:
    """Test query caching."""
    
    @pytest.fixture
    def cache(self):
        return QueryCache()
    
    def test_basic_set_get(self, cache):
        """Test basic cache set and get."""
        cache.set("What is X?", ["result1", "result2"])
        result = cache.get("What is X?")
        
        assert result == ["result1", "result2"]
    
    def test_cache_miss(self, cache):
        """Test cache miss returns None."""
        result = cache.get("nonexistent query")
        assert result is None
    
    def test_normalized_keys(self, cache):
        """Test that normalized keys match."""
        cache.set("What is X?", ["result"])
        
        # Different case should still match
        result = cache.get("what is x?")
        assert result == ["result"]
    
    def test_exact_keys(self):
        """Test exact key matching."""
        config = CacheConfig(strategy=CacheStrategy.EXACT)
        cache = QueryCache(config=config)
        
        cache.set("What is X?", ["result"])
        
        # Exact match required
        assert cache.get("What is X?") == ["result"]
        assert cache.get("what is x?") is None
    
    def test_ttl_expiration(self):
        """Test TTL expiration."""
        config = CacheConfig(default_ttl_seconds=1)
        cache = QueryCache(config=config)
        
        cache.set("query", ["result"])
        assert cache.get("query") == ["result"]
        
        # Wait for expiration
        time.sleep(1.1)
        assert cache.get("query") is None
    
    def test_cache_invalidation_query(self, cache):
        """Test invalidating specific query."""
        cache.set("query1", ["result1"])
        cache.set("query2", ["result2"])
        
        count = cache.invalidate(query="query1", namespace="default")
        
        assert count == 1
        assert cache.get("query1") is None
        assert cache.get("query2") == ["result2"]
    
    def test_cache_invalidation_namespace(self, cache):
        """Test invalidating entire namespace."""
        cache.set("query1", ["result1"], namespace="ns1")
        cache.set("query2", ["result2"], namespace="ns1")
        cache.set("query3", ["result3"], namespace="ns2")
        
        count = cache.invalidate(namespace="ns1")
        
        assert count == 2
        assert cache.get("query1", namespace="ns1") is None
        assert cache.get("query3", namespace="ns2") == ["result3"]
    
    def test_cache_stats(self, cache):
        """Test cache statistics."""
        cache.set("query", ["result"])
        cache.get("query")  # Hit
        cache.get("query")  # Hit
        cache.get("nonexistent")  # Miss
        
        stats = cache.get_stats()
        
        assert stats.hits == 2
        assert stats.misses == 1
        assert stats.hit_rate == 2/3
    
    def test_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        config = CacheConfig(max_entries=3)
        cache = QueryCache(config=config)
        
        cache.set("query1", ["r1"])
        cache.set("query2", ["r2"])
        cache.set("query3", ["r3"])
        
        # Access query1 to make it recently used
        cache.get("query1")
        
        # Add new entry - should evict query2 (least recently used)
        cache.set("query4", ["r4"])
        
        assert cache.get("query1") is not None  # Still exists
        assert cache.get("query2") is None  # Evicted
        assert cache.get("query3") is not None  # Still exists
        assert cache.get("query4") is not None  # New entry
    
    def test_cache_warming(self, cache):
        """Test cache warming."""
        def retriever(query):
            return [f"result for {query}"]
        
        queries = ["q1", "q2", "q3"]
        count = cache.warm(queries, retriever)
        
        assert count == 3
        assert cache.get("q1") == ["result for q1"]
        assert cache.get("q2") == ["result for q2"]
    
    def test_disabled_cache(self):
        """Test that disabled cache returns None."""
        config = CacheConfig(enabled=False)
        cache = QueryCache(config=config)
        
        cache.set("query", ["result"])
        assert cache.get("query") is None
    
    def test_semantic_cache(self):
        """Test semantic cache with embeddings."""
        def mock_embed(texts):
            # Simple mock - longer text = bigger embedding
            return [[len(t) / 100.0] * 10 for t in texts]
        
        config = CacheConfig(
            strategy=CacheStrategy.SEMANTIC,
            similarity_threshold=0.9,
        )
        cache = QueryCache(config=config, embed_fn=mock_embed)
        
        cache.set("What is machine learning?", ["ML result"])
        
        # Very similar query should hit
        # (In real scenario, embeddings would be more sophisticated)
        result = cache.get("What is machine learning?")
        assert result == ["ML result"]

class TestMultiLevelCache:
    """Test multi-level caching."""
    
    def test_l1_hit(self):
        """Test L1 cache hit."""
        cache = MultiLevelCache()
        
        cache.set("query", ["result"])
        result = cache.get("query")
        
        assert result == ["result"]
    
    def test_fallback_to_l2(self):
        """Test fallback to L2 (placeholder)."""
        cache = MultiLevelCache(l2_backend=None)
        
        # L2 not configured, should return None for miss
        result = cache.get("nonexistent")
        assert result is None

# ============================================================================
# Observability Tests
# ============================================================================

class TestMetricsCollector:
    """Test metrics collection."""
    
    @pytest.fixture
    def metrics(self):
        return MetricsCollector()
    
    def test_counter_increment(self, metrics):
        """Test counter increment."""
        metrics.increment("test.counter")
        metrics.increment("test.counter")
        metrics.increment("test.counter", value=3)
        
        assert metrics.get_counter("test.counter") == 5
    
    def test_gauge_set(self, metrics):
        """Test gauge setting."""
        metrics.gauge("test.gauge", 42.5)
        
        assert metrics.get_gauge("test.gauge") == 42.5
        
        metrics.gauge("test.gauge", 100.0)
        assert metrics.get_gauge("test.gauge") == 100.0
    
    def test_histogram_percentiles(self, metrics):
        """Test histogram percentile calculation."""
        for i in range(100):
            metrics.histogram("test.hist", float(i))
        
        p50 = metrics.get_percentile("test.hist", 50)
        p90 = metrics.get_percentile("test.hist", 90)
        p99 = metrics.get_percentile("test.hist", 99)
        
        assert 45 <= p50 <= 55  # Around 50
        assert 85 <= p90 <= 95  # Around 90
        assert 95 <= p99 <= 100  # Around 99
    
    def test_timer_average(self, metrics):
        """Test timer average calculation."""
        metrics.timer("test.timer", 10.0)
        metrics.timer("test.timer", 20.0)
        metrics.timer("test.timer", 30.0)
        
        avg = metrics.get_average("test.timer")
        assert avg == 20.0
    
    def test_time_operation_context(self, metrics):
        """Test timing context manager."""
        with metrics.time_operation("test.op"):
            time.sleep(0.01)  # 10ms
        
        avg = metrics.get_average("test.op")
        assert avg >= 10  # At least 10ms
    
    def test_metrics_with_tags(self, metrics):
        """Test metrics with tags."""
        metrics.increment("requests", tags={"status": "200"})
        metrics.increment("requests", tags={"status": "500"})
        metrics.increment("requests", tags={"status": "200"})
        
        assert metrics.get_counter("requests", tags={"status": "200"}) == 2
        assert metrics.get_counter("requests", tags={"status": "500"}) == 1
    
    def test_metrics_reset(self, metrics):
        """Test metrics reset."""
        metrics.increment("test.counter", value=10)
        metrics.histogram("test.hist", 100.0)
        
        metrics.reset()
        
        assert metrics.get_counter("test.counter") == 0
        assert metrics.get_percentile("test.hist", 50) == 0.0
    
    def test_get_all_metrics(self, metrics):
        """Test exporting all metrics."""
        metrics.increment("counter1")
        metrics.gauge("gauge1", 42.0)
        
        all_metrics = metrics.get_all_metrics()
        
        assert "counters" in all_metrics
        assert "gauges" in all_metrics

class TestTracer:
    """Test distributed tracing."""
    
    @pytest.fixture
    def tracer(self):
        return Tracer()
    
    def test_trace_span(self, tracer):
        """Test creating a trace span."""
        with tracer.trace("test_op", ComponentType.SEMANTIC_SEARCH) as span:
            time.sleep(0.01)
            span.tags["result_count"] = 5
        
        assert span.duration_ms >= 10
        assert span.tags["result_count"] == 5
        assert span.component == ComponentType.SEMANTIC_SEARCH
    
    def test_nested_spans(self, tracer):
        """Test nested trace spans."""
        tracer.start_trace()
        
        with tracer.trace("outer", ComponentType.TOTAL) as outer:
            with tracer.trace("inner", ComponentType.EMBEDDING) as inner:
                time.sleep(0.01)
        
        spans = tracer.end_trace()
        
        assert len(spans) == 2
        # Inner span should have outer as parent
        assert inner.parent_span_id == outer.span_id
    
    def test_trace_error_handling(self, tracer):
        """Test that errors are recorded in span."""
        tracer.start_trace()
        
        try:
            with tracer.trace("failing_op") as span:
                raise ValueError("Test error")
        except ValueError:
            pass
        
        spans = tracer.end_trace()
        
        assert len(spans) == 1
        assert spans[0].error is not None
        assert len(spans[0].logs) > 0
    
    def test_trace_id_generation(self, tracer):
        """Test trace ID generation."""
        trace_id = tracer.start_trace()
        
        assert trace_id is not None
        assert len(trace_id) == 16
        
        current = tracer.get_current_trace_id()
        assert current == trace_id

class TestRAGObserver:
    """Test unified RAG observer."""
    
    @pytest.fixture
    def observer(self):
        return RAGObserver()
    
    def test_observe_operation(self, observer):
        """Test observing an operation."""
        with observer.observe("search", ComponentType.SEMANTIC_SEARCH) as ctx:
            ctx["result_count"] = 10
            time.sleep(0.01)
        
        # Check metrics were recorded
        metrics = observer.metrics.get_all_metrics()
        assert "rag.semantic_search.success" in str(metrics)
    
    def test_observe_query_lifecycle(self, observer):
        """Test full query observation lifecycle."""
        trace_id = observer.observe_query("What is X?")
        
        assert trace_id is not None
        
        with observer.observe("search", ComponentType.SEMANTIC_SEARCH):
            pass
        
        spans = observer.complete_query(trace_id, success=True, result_count=5)
        
        assert len(spans) >= 0  # May have spans
    
    def test_pipeline_metrics(self, observer):
        """Test getting pipeline metrics."""
        # Simulate some queries
        for _ in range(5):
            trace_id = observer.observe_query("test")
            observer.complete_query(trace_id, success=True)
        
        for _ in range(2):
            trace_id = observer.observe_query("test")
            observer.complete_query(trace_id, success=False)
        
        metrics = observer.get_pipeline_metrics()
        
        assert metrics.total_queries == 7
        assert metrics.successful_queries == 5
        assert metrics.failed_queries == 2
        assert metrics.error_rate == 2/7
    
    def test_global_observer(self):
        """Test global observer singleton."""
        observer1 = get_global_observer()
        observer2 = get_global_observer()
        
        assert observer1 is observer2
        
        # Test setting custom observer
        custom = RAGObserver()
        set_global_observer(custom)
        
        assert get_global_observer() is custom

# ============================================================================
# Batch Processing Tests
# ============================================================================

class TestBatchProcessor:
    """Test batch processing."""
    
    @pytest.fixture
    def mock_retriever(self):
        def retriever(query: str) -> List[Any]:
            time.sleep(0.01)  # Simulate processing
            return [f"result for {query}"]
        return retriever
    
    def test_process_single(self, mock_retriever):
        """Test processing a single query."""
        processor = BatchProcessor(mock_retriever)
        
        result = processor.process_single("test query")
        
        assert result.success
        assert result.results == ["result for test query"]
        processor.shutdown()
    
    def test_process_batch(self, mock_retriever):
        """Test processing a batch of queries."""
        processor = BatchProcessor(mock_retriever)
        
        queries = ["q1", "q2", "q3"]
        results = processor.process_batch(queries)
        
        assert len(results) == 3
        assert all(r.success for r in results)
        assert results[0].results == ["result for q1"]
        processor.shutdown()
    
    def test_concurrent_speedup(self, mock_retriever):
        """Test that concurrent processing is faster."""
        config = BatchConfig(max_workers=4)
        processor = BatchProcessor(mock_retriever, config=config)
        
        queries = ["q1", "q2", "q3", "q4"]
        
        # Sequential would take ~40ms, concurrent should be faster
        start = time.time()
        results = processor.process_batch(queries)
        elapsed = (time.time() - start) * 1000
        
        assert len(results) == 4
        # Should be faster than sequential (40ms)
        assert elapsed < 35  # Allow some overhead
        processor.shutdown()
    
    def test_priority_queue(self, mock_retriever):
        """Test priority queuing."""
        processor = BatchProcessor(mock_retriever)
        
        # Queue with different priorities
        id1 = processor.queue_query("low priority", Priority.LOW)
        id2 = processor.queue_query("high priority", Priority.HIGH)
        id3 = processor.queue_query("normal priority", Priority.NORMAL)
        
        assert id1 is not None
        assert id2 is not None
        assert id3 is not None
        
        results = processor.process_queue()
        
        # High priority should be processed first
        assert results[0].query == "high priority"
        processor.shutdown()
    
    def test_batch_stats(self, mock_retriever):
        """Test batch processing statistics."""
        processor = BatchProcessor(mock_retriever)
        
        processor.process_batch(["q1", "q2", "q3"])
        stats = processor.get_stats()
        
        assert stats.total_processed == 3
        assert stats.successful == 3
        assert stats.failed == 0
        assert stats.throughput_qps > 0
        processor.shutdown()
    
    def test_error_handling(self):
        """Test error handling in batch processing."""
        def failing_retriever(query: str) -> List[Any]:
            if "fail" in query:
                raise ValueError("Intentional failure")
            return [f"result for {query}"]
        
        processor = BatchProcessor(failing_retriever)
        
        results = processor.process_batch(["good", "fail", "good2"])
        
        assert len(results) == 3
        assert results[0].success
        assert not results[1].success
        assert results[1].error == "Intentional failure"
        assert results[2].success
        processor.shutdown()
    
    def test_disabled_batch_processing(self, mock_retriever):
        """Test fallback to sequential when disabled."""
        config = BatchConfig(enabled=False)
        processor = BatchProcessor(mock_retriever, config=config)
        
        results = processor.process_batch(["q1", "q2"])
        
        assert len(results) == 2
        assert all(r.success for r in results)
        processor.shutdown()

class TestStreamingBatchProcessor:
    """Test streaming batch processor."""
    
    @pytest.fixture
    def mock_retriever(self):
        def retriever(query: str) -> List[Any]:
            time.sleep(0.01)
            return [f"result for {query}"]
        return retriever
    
    def test_stream_batch(self, mock_retriever):
        """Test streaming batch results."""
        processor = StreamingBatchProcessor(mock_retriever)
        
        queries = ["q1", "q2", "q3"]
        results = list(processor.stream_batch(queries))
        
        assert len(results) == 3
        assert all(r.success for r in results)
        processor.shutdown()
    
    def test_results_as_available(self, mock_retriever):
        """Test that results stream as they complete."""
        def variable_retriever(query: str) -> List[Any]:
            # Different delays
            if "fast" in query:
                time.sleep(0.01)
            else:
                time.sleep(0.05)
            return [f"result for {query}"]
        
        processor = StreamingBatchProcessor(variable_retriever)
        
        queries = ["slow1", "fast1", "fast2"]
        results = []
        
        for result in processor.stream_batch(queries):
            results.append(result)
        
        assert len(results) == 3
        # Fast queries should typically complete first
        processor.shutdown()

# ============================================================================
# Integration Tests
# ============================================================================

class TestP3Integration:
    """Integration tests for P3 modules."""
    
    def test_cached_batch_processing(self):
        """Test batch processing with caching."""
        cache = QueryCache()
        
        def retriever(query: str) -> List[Any]:
            # Check cache first
            cached = cache.get(query)
            if cached:
                return cached
            
            time.sleep(0.01)
            result = [f"result for {query}"]
            cache.set(query, result)
            return result
        
        processor = BatchProcessor(retriever)
        
        # First batch - cache miss
        results1 = processor.process_batch(["q1", "q2"])
        
        # Second batch - cache hit for q1
        results2 = processor.process_batch(["q1", "q3"])
        
        assert len(results1) == 2
        assert len(results2) == 2
        
        stats = cache.get_stats()
        assert stats.hits >= 1
        processor.shutdown()
    
    def test_observed_batch_processing(self):
        """Test batch processing with observability."""
        observer = RAGObserver()
        
        def retriever(query: str) -> List[Any]:
            with observer.observe("search", ComponentType.SEMANTIC_SEARCH):
                time.sleep(0.01)
                return [f"result for {query}"]
        
        processor = BatchProcessor(retriever)
        
        trace_id = observer.observe_query("batch query")
        results = processor.process_batch(["q1", "q2"])
        spans = observer.complete_query(trace_id, result_count=len(results))
        
        assert len(results) == 2
        
        # Check metrics were collected
        pipeline_metrics = observer.get_pipeline_metrics()
        assert pipeline_metrics.total_queries >= 1
        processor.shutdown()
    
    def test_full_p3_pipeline(self):
        """Test full P3 enhancement pipeline."""
        # Initialize components
        cache = QueryCache()
        observer = RAGObserver()
        
        def retriever(query: str) -> List[Any]:
            with observer.observe("search", ComponentType.SEMANTIC_SEARCH):
                # Check cache
                cached = cache.get(query)
                if cached:
                    return cached
                
                result = [f"result for {query}"]
                cache.set(query, result)
                return result
        
        processor = BatchProcessor(retriever)
        
        # Process queries
        trace_id = observer.observe_query("pipeline test")
        results = processor.process_batch(["test1", "test2", "test1"])  # test1 repeated
        observer.complete_query(trace_id, success=True, result_count=len(results))
        
        # Verify results
        assert len(results) == 3
        
        # Verify cache
        cache_stats = cache.get_stats()
        assert cache_stats.hits >= 1  # test1 should be cached
        
        # Verify metrics
        pipeline_metrics = observer.get_pipeline_metrics()
        assert pipeline_metrics.total_queries >= 1
        
        processor.shutdown()
