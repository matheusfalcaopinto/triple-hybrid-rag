"""
Observability Module

Comprehensive logging, metrics, and tracing for RAG pipeline.

Features:
- Structured logging with context
- Performance metrics collection
- Distributed tracing support
- Error tracking and alerting
"""

import time
import logging
import threading
import functools
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable
from contextlib import contextmanager
from enum import Enum
from collections import defaultdict
import uuid

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"

class ComponentType(Enum):
    """RAG pipeline components."""
    QUERY_ANALYSIS = "query_analysis"
    HYDE = "hyde"
    QUERY_EXPANSION = "query_expansion"
    EMBEDDING = "embedding"
    LEXICAL_SEARCH = "lexical_search"
    SEMANTIC_SEARCH = "semantic_search"
    GRAPH_SEARCH = "graph_search"
    SPLADE_SEARCH = "splade_search"
    FUSION = "fusion"
    RERANKING = "reranking"
    DIVERSITY = "diversity"
    COMPRESSION = "compression"
    TOTAL = "total"

@dataclass
class SpanContext:
    """Context for distributed tracing."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    start_time: float = 0.0
    end_time: float = 0.0
    component: Optional[ComponentType] = None
    operation: str = ""
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[Exception] = None
    
    @property
    def duration_ms(self) -> float:
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time) * 1000
        return 0.0

@dataclass
class MetricValue:
    """A metric data point."""
    name: str
    value: float
    metric_type: MetricType
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)
    
@dataclass
class PipelineMetrics:
    """Aggregated metrics for pipeline performance."""
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    
    # Latency percentiles
    p50_latency_ms: float = 0.0
    p90_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    avg_latency_ms: float = 0.0
    
    # Component breakdown
    component_latencies: Dict[str, float] = field(default_factory=dict)
    
    # Quality metrics
    avg_results_count: float = 0.0
    cache_hit_rate: float = 0.0
    
    # Error rates
    error_rate: float = 0.0
    errors_by_type: Dict[str, int] = field(default_factory=dict)

class MetricsCollector:
    """
    Collect and aggregate metrics from RAG pipeline.
    
    Thread-safe metrics collection with aggregation support.
    """
    
    def __init__(self, max_samples: int = 10000):
        self.max_samples = max_samples
        self._lock = threading.RLock()
        
        # Metric storage
        self._counters: Dict[str, int] = defaultdict(int)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._timers: Dict[str, List[float]] = defaultdict(list)
    
    def increment(
        self,
        name: str,
        value: int = 1,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Increment a counter."""
        key = self._make_key(name, tags)
        with self._lock:
            self._counters[key] += value
    
    def gauge(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Set a gauge value."""
        key = self._make_key(name, tags)
        with self._lock:
            self._gauges[key] = value
    
    def histogram(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record a histogram value."""
        key = self._make_key(name, tags)
        with self._lock:
            samples = self._histograms[key]
            samples.append(value)
            if len(samples) > self.max_samples:
                self._histograms[key] = samples[-self.max_samples:]
    
    def timer(
        self,
        name: str,
        duration_ms: float,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record a timer value."""
        key = self._make_key(name, tags)
        with self._lock:
            samples = self._timers[key]
            samples.append(duration_ms)
            if len(samples) > self.max_samples:
                self._timers[key] = samples[-self.max_samples:]
    
    @contextmanager
    def time_operation(
        self,
        name: str,
        tags: Optional[Dict[str, str]] = None,
    ):
        """Context manager to time an operation."""
        start = time.time()
        try:
            yield
        finally:
            duration_ms = (time.time() - start) * 1000
            self.timer(name, duration_ms, tags)
    
    def get_counter(self, name: str, tags: Optional[Dict[str, str]] = None) -> int:
        """Get counter value."""
        key = self._make_key(name, tags)
        with self._lock:
            return self._counters.get(key, 0)
    
    def get_gauge(self, name: str, tags: Optional[Dict[str, str]] = None) -> float:
        """Get gauge value."""
        key = self._make_key(name, tags)
        with self._lock:
            return self._gauges.get(key, 0.0)
    
    def get_percentile(
        self,
        name: str,
        percentile: float,
        tags: Optional[Dict[str, str]] = None,
    ) -> float:
        """Get histogram/timer percentile."""
        key = self._make_key(name, tags)
        with self._lock:
            samples = self._histograms.get(key) or self._timers.get(key) or []
            if not samples:
                return 0.0
            sorted_samples = sorted(samples)
            idx = int(len(sorted_samples) * percentile / 100)
            return sorted_samples[min(idx, len(sorted_samples) - 1)]
    
    def get_average(
        self,
        name: str,
        tags: Optional[Dict[str, str]] = None,
    ) -> float:
        """Get histogram/timer average."""
        key = self._make_key(name, tags)
        with self._lock:
            samples = self._histograms.get(key) or self._timers.get(key) or []
            return sum(samples) / len(samples) if samples else 0.0
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Export all metrics."""
        with self._lock:
            return {
                'counters': dict(self._counters),
                'gauges': dict(self._gauges),
                'histogram_counts': {k: len(v) for k, v in self._histograms.items()},
                'timer_counts': {k: len(v) for k, v in self._timers.items()},
            }
    
    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._timers.clear()
    
    def _make_key(self, name: str, tags: Optional[Dict[str, str]]) -> str:
        """Create metric key with tags."""
        if not tags:
            return name
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}[{tag_str}]"

class Tracer:
    """
    Distributed tracing for RAG pipeline.
    
    Creates spans for each component to track request flow.
    """
    
    _local = threading.local()
    
    def __init__(self, service_name: str = "triple-hybrid-rag"):
        self.service_name = service_name
        self._spans: Dict[str, List[SpanContext]] = {}
        self._lock = threading.RLock()
    
    @contextmanager
    def trace(
        self,
        operation: str,
        component: Optional[ComponentType] = None,
        tags: Optional[Dict[str, Any]] = None,
    ):
        """
        Context manager to create a trace span.
        
        Example:
            with tracer.trace("search", ComponentType.SEMANTIC_SEARCH) as span:
                results = search(query)
                span.tags["result_count"] = len(results)
        """
        span = self._start_span(operation, component, tags)
        try:
            yield span
        except Exception as e:
            span.error = e
            span.logs.append({
                'event': 'error',
                'message': str(e),
                'timestamp': time.time(),
            })
            raise
        finally:
            self._end_span(span)
    
    def start_trace(self) -> str:
        """Start a new trace and return trace_id."""
        trace_id = str(uuid.uuid4())[:16]
        self._local.trace_id = trace_id
        self._local.span_stack = []
        
        with self._lock:
            self._spans[trace_id] = []
        
        return trace_id
    
    def end_trace(self, trace_id: Optional[str] = None) -> List[SpanContext]:
        """End trace and return all spans."""
        trace_id = trace_id or getattr(self._local, 'trace_id', None)
        if not trace_id:
            return []
        
        with self._lock:
            spans = self._spans.pop(trace_id, [])
        
        # Clear local context
        self._local.trace_id = None
        self._local.span_stack = []
        
        return spans
    
    def get_current_trace_id(self) -> Optional[str]:
        """Get current trace ID."""
        return getattr(self._local, 'trace_id', None)
    
    def _start_span(
        self,
        operation: str,
        component: Optional[ComponentType],
        tags: Optional[Dict[str, Any]],
    ) -> SpanContext:
        """Start a new span."""
        trace_id = getattr(self._local, 'trace_id', None) or self.start_trace()
        span_stack = getattr(self._local, 'span_stack', [])
        
        parent_span_id = span_stack[-1] if span_stack else None
        span_id = str(uuid.uuid4())[:8]
        
        span = SpanContext(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            start_time=time.time(),
            component=component,
            operation=operation,
            tags=tags or {},
        )
        
        # Push to stack
        self._local.span_stack = span_stack + [span_id]
        
        return span
    
    def _end_span(self, span: SpanContext) -> None:
        """End a span."""
        span.end_time = time.time()
        
        # Pop from stack
        span_stack = getattr(self._local, 'span_stack', [])
        if span_stack and span_stack[-1] == span.span_id:
            self._local.span_stack = span_stack[:-1]
        
        # Store span
        with self._lock:
            if span.trace_id in self._spans:
                self._spans[span.trace_id].append(span)
        
        # Log span
        logger.debug(
            f"Span completed: {span.operation} "
            f"[{span.component.value if span.component else 'unknown'}] "
            f"duration={span.duration_ms:.2f}ms"
        )

class RAGObserver:
    """
    Unified observability for RAG pipeline.
    
    Combines metrics, tracing, and logging into a single interface.
    """
    
    def __init__(
        self,
        metrics: Optional[MetricsCollector] = None,
        tracer: Optional[Tracer] = None,
    ):
        self.metrics = metrics or MetricsCollector()
        self.tracer = tracer or Tracer()
    
    @contextmanager
    def observe(
        self,
        operation: str,
        component: ComponentType,
        tags: Optional[Dict[str, Any]] = None,
    ):
        """
        Observe an operation with metrics and tracing.
        
        Example:
            with observer.observe("search", ComponentType.SEMANTIC_SEARCH) as ctx:
                results = search(query)
                ctx['result_count'] = len(results)
        """
        context: Dict[str, Any] = tags.copy() if tags else {}
        
        with self.tracer.trace(operation, component, context) as span:
            start = time.time()
            try:
                yield context
                
                # Record success
                self.metrics.increment(
                    f"rag.{component.value}.success",
                    tags={'operation': operation},
                )
                
            except Exception as e:
                # Record error
                self.metrics.increment(
                    f"rag.{component.value}.error",
                    tags={'operation': operation, 'error_type': type(e).__name__},
                )
                raise
            
            finally:
                # Record latency
                duration_ms = (time.time() - start) * 1000
                self.metrics.timer(
                    f"rag.{component.value}.latency_ms",
                    duration_ms,
                    tags={'operation': operation},
                )
                
                # Update span with context
                span.tags.update(context)
    
    def observe_query(self, query: str) -> str:
        """Start observing a query, returns trace_id."""
        trace_id = self.tracer.start_trace()
        self.metrics.increment("rag.queries.total")
        
        logger.info(f"Query started [trace_id={trace_id}]: {query[:100]}...")
        return trace_id
    
    def complete_query(
        self,
        trace_id: str,
        success: bool = True,
        result_count: int = 0,
    ) -> List[SpanContext]:
        """Complete query observation."""
        spans = self.tracer.end_trace(trace_id)
        
        if success:
            self.metrics.increment("rag.queries.success")
        else:
            self.metrics.increment("rag.queries.failed")
        
        self.metrics.histogram("rag.queries.result_count", result_count)
        
        # Log summary
        total_duration = sum(s.duration_ms for s in spans)
        logger.info(
            f"Query completed [trace_id={trace_id}]: "
            f"success={success}, results={result_count}, "
            f"duration={total_duration:.2f}ms"
        )
        
        return spans
    
    def get_pipeline_metrics(self) -> PipelineMetrics:
        """Get aggregated pipeline metrics."""
        total = self.metrics.get_counter("rag.queries.total")
        success = self.metrics.get_counter("rag.queries.success")
        failed = self.metrics.get_counter("rag.queries.failed")
        
        return PipelineMetrics(
            total_queries=total,
            successful_queries=success,
            failed_queries=failed,
            p50_latency_ms=self.metrics.get_percentile("rag.total.latency_ms", 50),
            p90_latency_ms=self.metrics.get_percentile("rag.total.latency_ms", 90),
            p99_latency_ms=self.metrics.get_percentile("rag.total.latency_ms", 99),
            avg_latency_ms=self.metrics.get_average("rag.total.latency_ms"),
            avg_results_count=self.metrics.get_average("rag.queries.result_count"),
            error_rate=failed / total if total > 0 else 0.0,
        )

def traced(component: ComponentType, operation: Optional[str] = None):
    """
    Decorator to trace a function.
    
    Example:
        @traced(ComponentType.SEMANTIC_SEARCH)
        def search(query: str) -> List[Result]:
            ...
    """
    def decorator(func: Callable) -> Callable:
        op_name = operation or func.__name__
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            observer = get_global_observer()
            with observer.observe(op_name, component):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator

# Global observer instance
_global_observer: Optional[RAGObserver] = None

def get_global_observer() -> RAGObserver:
    """Get or create global observer."""
    global _global_observer
    if _global_observer is None:
        _global_observer = RAGObserver()
    return _global_observer

def set_global_observer(observer: RAGObserver) -> None:
    """Set global observer."""
    global _global_observer
    _global_observer = observer
