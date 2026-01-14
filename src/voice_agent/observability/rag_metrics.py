"""
RAG Metrics Module

Prometheus-compatible metrics for the RAG (Retrieval-Augmented Generation) pipeline.
Tracks ingestion, search, embedding, and reranking operations.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Lock
from typing import Dict, Generator, List, Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Metric Storage (Thread-safe)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Counter:
    """Thread-safe counter metric."""
    name: str
    help_text: str
    value: float = 0.0
    _lock: Lock = field(default_factory=Lock, repr=False)
    
    def inc(self, amount: float = 1.0) -> None:
        with self._lock:
            self.value += amount
    
    def get(self) -> float:
        with self._lock:
            return self.value


@dataclass
class Gauge:
    """Thread-safe gauge metric."""
    name: str
    help_text: str
    value: float = 0.0
    _lock: Lock = field(default_factory=Lock, repr=False)
    
    def set(self, value: float) -> None:
        with self._lock:
            self.value = value
    
    def inc(self, amount: float = 1.0) -> None:
        with self._lock:
            self.value += amount
    
    def dec(self, amount: float = 1.0) -> None:
        with self._lock:
            self.value -= amount
    
    def get(self) -> float:
        with self._lock:
            return self.value


@dataclass
class Histogram:
    """Thread-safe histogram metric with configurable buckets."""
    name: str
    help_text: str
    buckets: List[float] = field(default_factory=lambda: [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0])
    _observations: List[float] = field(default_factory=list, repr=False)
    _sum: float = 0.0
    _count: int = 0
    _lock: Lock = field(default_factory=Lock, repr=False)
    
    def observe(self, value: float) -> None:
        with self._lock:
            self._observations.append(value)
            self._sum += value
            self._count += 1
    
    def get_buckets(self) -> Dict[str, int]:
        with self._lock:
            result = {}
            for bucket in self.buckets:
                count = sum(1 for v in self._observations if v <= bucket)
                result[str(bucket)] = count
            result["+Inf"] = len(self._observations)
            return result
    
    def get_sum(self) -> float:
        with self._lock:
            return self._sum
    
    def get_count(self) -> int:
        with self._lock:
            return self._count


# ──────────────────────────────────────────────────────────────────────────────
# RAG Metrics Definitions
# ──────────────────────────────────────────────────────────────────────────────

class RAGMetrics:
    """
    Centralized RAG metrics collection.
    
    Usage:
        from voice_agent.observability.rag_metrics import rag_metrics
        
        # Counter
        rag_metrics.documents_ingested.inc()
        
        # Histogram with timing
        with rag_metrics.time_search():
            results = await searcher.search(query)
        
        # Gauge
        rag_metrics.chunks_in_database.set(1000)
    """
    
    def __init__(self):
        # ──────────────────────────────────────────────────────────────────────
        # Ingestion Counters
        # ──────────────────────────────────────────────────────────────────────
        self.documents_ingested = Counter(
            "rag_documents_ingested_total",
            "Total number of documents ingested into the knowledge base"
        )
        self.documents_failed = Counter(
            "rag_documents_failed_total",
            "Total number of document ingestion failures"
        )
        self.chunks_created = Counter(
            "rag_chunks_created_total",
            "Total number of chunks created from documents"
        )
        self.chunks_stored = Counter(
            "rag_chunks_stored_total",
            "Total number of chunks stored in database"
        )
        self.chunks_deduplicated = Counter(
            "rag_chunks_deduplicated_total",
            "Total number of duplicate chunks skipped"
        )
        
        # ──────────────────────────────────────────────────────────────────────
        # Embedding Counters
        # ──────────────────────────────────────────────────────────────────────
        self.text_embeddings_generated = Counter(
            "rag_text_embeddings_generated_total",
            "Total number of text embeddings generated via OpenAI"
        )
        self.image_embeddings_generated = Counter(
            "rag_image_embeddings_generated_total",
            "Total number of image embeddings generated via SigLIP"
        )
        self.embedding_errors = Counter(
            "rag_embedding_errors_total",
            "Total number of embedding generation errors"
        )
        
        # ──────────────────────────────────────────────────────────────────────
        # OCR Counters
        # ──────────────────────────────────────────────────────────────────────
        self.ocr_pages_processed = Counter(
            "rag_ocr_pages_processed_total",
            "Total number of pages processed by OCR"
        )
        self.ocr_retries = Counter(
            "rag_ocr_retries_total",
            "Total number of OCR retry attempts"
        )
        self.ocr_failures = Counter(
            "rag_ocr_failures_total",
            "Total number of OCR failures"
        )
        
        # ──────────────────────────────────────────────────────────────────────
        # Search Counters
        # ──────────────────────────────────────────────────────────────────────
        self.searches_total = Counter(
            "rag_searches_total",
            "Total number of knowledge base searches"
        )
        self.search_errors = Counter(
            "rag_search_errors_total",
            "Total number of search errors"
        )
        self.hybrid_searches = Counter(
            "rag_hybrid_searches_total",
            "Total number of hybrid (BM25 + vector) searches"
        )
        self.vector_only_searches = Counter(
            "rag_vector_only_searches_total",
            "Total number of vector-only searches"
        )
        self.bm25_only_searches = Counter(
            "rag_bm25_only_searches_total",
            "Total number of BM25-only searches"
        )
        
        # ──────────────────────────────────────────────────────────────────────
        # Reranking Counters
        # ──────────────────────────────────────────────────────────────────────
        self.reranks_total = Counter(
            "rag_reranks_total",
            "Total number of reranking operations"
        )
        self.rerank_errors = Counter(
            "rag_rerank_errors_total",
            "Total number of reranking errors"
        )
        self.rerank_fallbacks = Counter(
            "rag_rerank_fallbacks_total",
            "Total number of times lightweight reranker was used as fallback"
        )
        
        # ──────────────────────────────────────────────────────────────────────
        # Latency Histograms (in seconds)
        # ──────────────────────────────────────────────────────────────────────
        latency_buckets = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0]
        
        self.ingestion_duration = Histogram(
            "rag_ingestion_duration_seconds",
            "Time spent ingesting a document",
            buckets=latency_buckets
        )
        self.chunking_duration = Histogram(
            "rag_chunking_duration_seconds",
            "Time spent chunking a document",
            buckets=latency_buckets
        )
        self.embedding_duration = Histogram(
            "rag_embedding_duration_seconds",
            "Time spent generating embeddings for a batch",
            buckets=latency_buckets
        )
        self.ocr_duration = Histogram(
            "rag_ocr_duration_seconds",
            "Time spent on OCR for a page",
            buckets=latency_buckets
        )
        self.search_duration = Histogram(
            "rag_search_duration_seconds",
            "Time spent on a search query",
            buckets=latency_buckets
        )
        self.rerank_duration = Histogram(
            "rag_rerank_duration_seconds",
            "Time spent reranking search results",
            buckets=latency_buckets
        )
        self.vector_search_duration = Histogram(
            "rag_vector_search_duration_seconds",
            "Time spent on vector similarity search",
            buckets=latency_buckets
        )
        self.bm25_search_duration = Histogram(
            "rag_bm25_search_duration_seconds",
            "Time spent on BM25 full-text search",
            buckets=latency_buckets
        )
        
        # ──────────────────────────────────────────────────────────────────────
        # Result Histograms
        # ──────────────────────────────────────────────────────────────────────
        result_buckets = [0.0, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0]
        
        self.search_results_count = Histogram(
            "rag_search_results_count",
            "Number of results returned per search",
            buckets=result_buckets
        )
        self.chunks_per_document = Histogram(
            "rag_chunks_per_document",
            "Number of chunks created per document",
            buckets=result_buckets
        )
        self.rerank_results_count = Histogram(
            "rag_rerank_results_count",
            "Number of results returned after reranking",
            buckets=result_buckets
        )
        
        # ──────────────────────────────────────────────────────────────────────
        # State Gauges
        # ──────────────────────────────────────────────────────────────────────
        self.chunks_in_database = Gauge(
            "rag_chunks_in_database",
            "Current number of chunks in the knowledge base"
        )
        self.siglip_model_loaded = Gauge(
            "rag_siglip_model_loaded",
            "Whether the SigLIP model is loaded (1=yes, 0=no)"
        )
        self.crossencoder_model_loaded = Gauge(
            "rag_crossencoder_model_loaded",
            "Whether the CrossEncoder model is loaded (1=yes, 0=no)"
        )
        self.active_ingestions = Gauge(
            "rag_active_ingestions",
            "Number of currently active document ingestions"
        )
        self.active_searches = Gauge(
            "rag_active_searches",
            "Number of currently active search queries"
        )
    
    # ──────────────────────────────────────────────────────────────────────────
    # Timing Context Managers
    # ──────────────────────────────────────────────────────────────────────────
    
    @contextmanager
    def time_ingestion(self) -> Generator[None, None, None]:
        """Time a document ingestion operation."""
        self.active_ingestions.inc()
        start = time.perf_counter()
        try:
            yield
        finally:
            self.active_ingestions.dec()
            duration = time.perf_counter() - start
            self.ingestion_duration.observe(duration)
    
    @contextmanager
    def time_chunking(self) -> Generator[None, None, None]:
        """Time a chunking operation."""
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self.chunking_duration.observe(duration)
    
    @contextmanager
    def time_embedding(self) -> Generator[None, None, None]:
        """Time an embedding generation operation."""
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self.embedding_duration.observe(duration)
    
    @contextmanager
    def time_ocr(self) -> Generator[None, None, None]:
        """Time an OCR operation."""
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self.ocr_duration.observe(duration)
    
    @contextmanager
    def time_search(self) -> Generator[None, None, None]:
        """Time a search operation."""
        self.active_searches.inc()
        start = time.perf_counter()
        try:
            yield
        finally:
            self.active_searches.dec()
            duration = time.perf_counter() - start
            self.search_duration.observe(duration)
    
    @contextmanager
    def time_rerank(self) -> Generator[None, None, None]:
        """Time a reranking operation."""
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self.rerank_duration.observe(duration)
    
    @contextmanager
    def time_vector_search(self) -> Generator[None, None, None]:
        """Time a vector search operation."""
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self.vector_search_duration.observe(duration)
    
    @contextmanager
    def time_bm25_search(self) -> Generator[None, None, None]:
        """Time a BM25 search operation."""
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self.bm25_search_duration.observe(duration)
    
    # ──────────────────────────────────────────────────────────────────────────
    # Prometheus Export
    # ──────────────────────────────────────────────────────────────────────────
    
    def export_prometheus(self) -> str:
        """Export all RAG metrics in Prometheus text format."""
        lines: List[str] = []
        
        # Export counters
        counters = [
            self.documents_ingested, self.documents_failed,
            self.chunks_created, self.chunks_stored, self.chunks_deduplicated,
            self.text_embeddings_generated, self.image_embeddings_generated, self.embedding_errors,
            self.ocr_pages_processed, self.ocr_retries, self.ocr_failures,
            self.searches_total, self.search_errors,
            self.hybrid_searches, self.vector_only_searches, self.bm25_only_searches,
            self.reranks_total, self.rerank_errors, self.rerank_fallbacks,
        ]
        
        for counter in counters:
            lines.append(f"# HELP {counter.name} {counter.help_text}")
            lines.append(f"# TYPE {counter.name} counter")
            lines.append(f"{counter.name} {counter.get()}")
            lines.append("")
        
        # Export gauges
        gauges = [
            self.chunks_in_database,
            self.siglip_model_loaded,
            self.crossencoder_model_loaded,
            self.active_ingestions,
            self.active_searches,
        ]
        
        for gauge in gauges:
            lines.append(f"# HELP {gauge.name} {gauge.help_text}")
            lines.append(f"# TYPE {gauge.name} gauge")
            lines.append(f"{gauge.name} {gauge.get()}")
            lines.append("")
        
        # Export histograms
        histograms = [
            self.ingestion_duration, self.chunking_duration,
            self.embedding_duration, self.ocr_duration,
            self.search_duration, self.rerank_duration,
            self.vector_search_duration, self.bm25_search_duration,
            self.search_results_count, self.chunks_per_document,
        ]
        
        for histogram in histograms:
            lines.append(f"# HELP {histogram.name} {histogram.help_text}")
            lines.append(f"# TYPE {histogram.name} histogram")
            
            buckets = histogram.get_buckets()
            for bucket_le, count in buckets.items():
                lines.append(f'{histogram.name}_bucket{{le="{bucket_le}"}} {count}')
            
            lines.append(f"{histogram.name}_sum {histogram.get_sum()}")
            lines.append(f"{histogram.name}_count {histogram.get_count()}")
            lines.append("")
        
        return "\n".join(lines)
    
    def reset(self) -> None:
        """Reset all metrics (for testing)."""
        # Reset counters
        for attr in dir(self):
            obj = getattr(self, attr)
            if isinstance(obj, Counter):
                with obj._lock:
                    obj.value = 0.0
            elif isinstance(obj, Gauge):
                with obj._lock:
                    obj.value = 0.0
            elif isinstance(obj, Histogram):
                with obj._lock:
                    obj._observations = []
                    obj._sum = 0.0
                    obj._count = 0


# Global metrics instance
rag_metrics = RAGMetrics()


# ──────────────────────────────────────────────────────────────────────────────
# Convenience Functions
# ──────────────────────────────────────────────────────────────────────────────

def get_rag_metrics() -> RAGMetrics:
    """Get the global RAG metrics instance."""
    return rag_metrics


def export_rag_metrics_prometheus() -> str:
    """Export RAG metrics in Prometheus format."""
    return rag_metrics.export_prometheus()
