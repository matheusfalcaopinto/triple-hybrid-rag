"""
Batch Processing Module

Efficient batch query handling for high-throughput scenarios.

Features:
- Concurrent query processing
- Batched embeddings
- Priority queuing
- Result aggregation
"""

import asyncio
import logging
import time
import threading
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable, Awaitable
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import PriorityQueue
from enum import Enum

logger = logging.getLogger(__name__)

class Priority(Enum):
    """Query priority levels."""
    HIGH = 1
    NORMAL = 2
    LOW = 3

@dataclass
class BatchQuery:
    """A query in the batch."""
    query_id: str
    query: str
    priority: Priority = Priority.NORMAL
    metadata: Dict[str, Any] = field(default_factory=dict)
    submitted_at: float = 0.0
    
    def __lt__(self, other: "BatchQuery") -> bool:
        return (self.priority.value, self.submitted_at) < (other.priority.value, other.submitted_at)

@dataclass
class BatchResult:
    """Result for a batched query."""
    query_id: str
    query: str
    results: List[Any]
    processing_time_ms: float
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BatchStats:
    """Statistics for batch processing."""
    total_processed: int = 0
    successful: int = 0
    failed: int = 0
    avg_processing_time_ms: float = 0.0
    total_time_ms: float = 0.0
    throughput_qps: float = 0.0

@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    enabled: bool = True
    
    # Concurrency settings
    max_workers: int = 4
    max_concurrent_queries: int = 10
    
    # Batching settings
    batch_size: int = 10
    batch_timeout_ms: int = 100  # Wait time to collect batch
    
    # Queue settings
    max_queue_size: int = 1000
    enable_priority: bool = True
    
    # Timeout settings
    query_timeout_seconds: float = 30.0

class BatchProcessor:
    """
    Process multiple queries efficiently.
    
    Uses thread pool for concurrent processing with batched embeddings.
    """
    
    def __init__(
        self,
        retriever_fn: Callable[[str], List[Any]],
        config: Optional[BatchConfig] = None,
        embed_batch_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
    ):
        self.config = config or BatchConfig()
        self.retriever_fn = retriever_fn
        self.embed_batch_fn = embed_batch_fn
        
        self._executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        self._queue: PriorityQueue = PriorityQueue(maxsize=self.config.max_queue_size)
        self._processing = False
        self._stats = BatchStats()
    
    def process_batch(
        self,
        queries: List[str],
        priority: Priority = Priority.NORMAL,
    ) -> List[BatchResult]:
        """
        Process a batch of queries.
        
        Args:
            queries: List of query strings
            priority: Priority level for all queries
            
        Returns:
            List of BatchResult for each query
        """
        if not self.config.enabled:
            return self._process_sequential(queries)
        
        start_time = time.time()
        
        # Create batch queries
        batch_queries = [
            BatchQuery(
                query_id=f"q_{i}_{int(time.time()*1000)}",
                query=q,
                priority=priority,
                submitted_at=time.time(),
            )
            for i, q in enumerate(queries)
        ]
        
        # Process with thread pool
        results = self._process_concurrent(batch_queries)
        
        # Update stats
        total_time = (time.time() - start_time) * 1000
        self._update_stats(results, total_time)
        
        return results
    
    def process_single(
        self,
        query: str,
        priority: Priority = Priority.NORMAL,
    ) -> BatchResult:
        """Process a single query."""
        results = self.process_batch([query], priority)
        return results[0]
    
    async def process_batch_async(
        self,
        queries: List[str],
        priority: Priority = Priority.NORMAL,
    ) -> List[BatchResult]:
        """
        Async version of batch processing.
        
        Args:
            queries: List of query strings
            priority: Priority level
            
        Returns:
            List of BatchResult
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self.process_batch,
            queries,
            priority,
        )
    
    def queue_query(
        self,
        query: str,
        priority: Priority = Priority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Queue a query for processing.
        
        Args:
            query: Query string
            priority: Priority level
            metadata: Additional metadata
            
        Returns:
            Query ID for tracking
        """
        batch_query = BatchQuery(
            query_id=f"q_{int(time.time()*1000000)}",
            query=query,
            priority=priority,
            metadata=metadata or {},
            submitted_at=time.time(),
        )
        
        self._queue.put(batch_query)
        return batch_query.query_id
    
    def process_queue(self, max_queries: Optional[int] = None) -> List[BatchResult]:
        """
        Process queries from the queue.
        
        Args:
            max_queries: Maximum queries to process
            
        Returns:
            List of results
        """
        queries = []
        max_q = max_queries or self.config.batch_size
        
        while not self._queue.empty() and len(queries) < max_q:
            try:
                queries.append(self._queue.get_nowait())
            except Exception:
                break
        
        if not queries:
            return []
        
        return self._process_concurrent(queries)
    
    def get_stats(self) -> BatchStats:
        """Get processing statistics."""
        return BatchStats(
            total_processed=self._stats.total_processed,
            successful=self._stats.successful,
            failed=self._stats.failed,
            avg_processing_time_ms=self._stats.avg_processing_time_ms,
            total_time_ms=self._stats.total_time_ms,
            throughput_qps=self._stats.throughput_qps,
        )
    
    def shutdown(self) -> None:
        """Shutdown the processor."""
        self._executor.shutdown(wait=True)
    
    def _process_sequential(self, queries: List[str]) -> List[BatchResult]:
        """Process queries sequentially."""
        results = []
        for i, query in enumerate(queries):
            start = time.time()
            try:
                result = self.retriever_fn(query)
                results.append(BatchResult(
                    query_id=f"seq_{i}",
                    query=query,
                    results=result,
                    processing_time_ms=(time.time() - start) * 1000,
                ))
            except Exception as e:
                results.append(BatchResult(
                    query_id=f"seq_{i}",
                    query=query,
                    results=[],
                    processing_time_ms=(time.time() - start) * 1000,
                    success=False,
                    error=str(e),
                ))
        return results
    
    def _process_concurrent(
        self,
        batch_queries: List[BatchQuery],
    ) -> List[BatchResult]:
        """Process queries concurrently."""
        results: Dict[str, BatchResult] = {}
        
        # Submit all queries
        futures = {}
        for bq in batch_queries:
            future = self._executor.submit(self._process_single_query, bq)
            futures[future] = bq.query_id
        
        # Collect results
        for future in as_completed(futures, timeout=self.config.query_timeout_seconds):
            query_id = futures[future]
            try:
                result = future.result()
                results[query_id] = result
            except Exception as e:
                # Find original query
                original = next(
                    (bq for bq in batch_queries if bq.query_id == query_id),
                    None
                )
                results[query_id] = BatchResult(
                    query_id=query_id,
                    query=original.query if original else "",
                    results=[],
                    processing_time_ms=0,
                    success=False,
                    error=str(e),
                )
        
        # Return in original order
        return [results.get(bq.query_id) for bq in batch_queries if bq.query_id in results]
    
    def _process_single_query(self, batch_query: BatchQuery) -> BatchResult:
        """Process a single query from the batch."""
        start = time.time()
        try:
            results = self.retriever_fn(batch_query.query)
            return BatchResult(
                query_id=batch_query.query_id,
                query=batch_query.query,
                results=results,
                processing_time_ms=(time.time() - start) * 1000,
                metadata=batch_query.metadata,
            )
        except Exception as e:
            logger.error(f"Query failed: {batch_query.query_id}: {e}")
            return BatchResult(
                query_id=batch_query.query_id,
                query=batch_query.query,
                results=[],
                processing_time_ms=(time.time() - start) * 1000,
                success=False,
                error=str(e),
                metadata=batch_query.metadata,
            )
    
    def _update_stats(
        self,
        results: List[BatchResult],
        total_time_ms: float,
    ) -> None:
        """Update processing statistics."""
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        
        processing_times = [r.processing_time_ms for r in results if r.success]
        avg_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        self._stats.total_processed += len(results)
        self._stats.successful += successful
        self._stats.failed += failed
        self._stats.total_time_ms += total_time_ms
        
        # Running average
        if self._stats.total_processed > 0:
            self._stats.avg_processing_time_ms = (
                (self._stats.avg_processing_time_ms * (self._stats.total_processed - len(results)) +
                 avg_time * len(results)) / self._stats.total_processed
            )
            self._stats.throughput_qps = (
                self._stats.total_processed / (self._stats.total_time_ms / 1000)
            )

class StreamingBatchProcessor:
    """
    Streaming processor for progressive result delivery.
    
    Yields results as they become available.
    """
    
    def __init__(
        self,
        retriever_fn: Callable[[str], List[Any]],
        config: Optional[BatchConfig] = None,
    ):
        self.config = config or BatchConfig()
        self.retriever_fn = retriever_fn
        self._executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
    
    def stream_batch(
        self,
        queries: List[str],
    ):
        """
        Stream results as they complete.
        
        Yields:
            BatchResult as each query completes
        """
        batch_queries = [
            BatchQuery(
                query_id=f"stream_{i}",
                query=q,
                submitted_at=time.time(),
            )
            for i, q in enumerate(queries)
        ]
        
        futures = {
            self._executor.submit(self._process_query, bq): bq
            for bq in batch_queries
        }
        
        for future in as_completed(futures, timeout=self.config.query_timeout_seconds):
            try:
                result = future.result()
                yield result
            except Exception as e:
                bq = futures[future]
                yield BatchResult(
                    query_id=bq.query_id,
                    query=bq.query,
                    results=[],
                    processing_time_ms=0,
                    success=False,
                    error=str(e),
                )
    
    async def stream_batch_async(
        self,
        queries: List[str],
    ):
        """
        Async streaming of batch results.
        
        Yields:
            BatchResult as each query completes
        """
        loop = asyncio.get_event_loop()
        
        batch_queries = [
            BatchQuery(
                query_id=f"async_{i}",
                query=q,
                submitted_at=time.time(),
            )
            for i, q in enumerate(queries)
        ]
        
        # Create async tasks
        tasks = [
            loop.run_in_executor(self._executor, self._process_query, bq)
            for bq in batch_queries
        ]
        
        for coro in asyncio.as_completed(tasks):
            try:
                result = await coro
                yield result
            except Exception as e:
                yield BatchResult(
                    query_id="error",
                    query="",
                    results=[],
                    processing_time_ms=0,
                    success=False,
                    error=str(e),
                )
    
    def _process_query(self, batch_query: BatchQuery) -> BatchResult:
        """Process a single query."""
        start = time.time()
        try:
            results = self.retriever_fn(batch_query.query)
            return BatchResult(
                query_id=batch_query.query_id,
                query=batch_query.query,
                results=results,
                processing_time_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return BatchResult(
                query_id=batch_query.query_id,
                query=batch_query.query,
                results=[],
                processing_time_ms=(time.time() - start) * 1000,
                success=False,
                error=str(e),
            )
    
    def shutdown(self) -> None:
        """Shutdown the processor."""
        self._executor.shutdown(wait=True)

def create_batch_processor(
    retriever_fn: Callable[[str], List[Any]],
    config: Optional[BatchConfig] = None,
    streaming: bool = False,
):
    """Factory function to create a batch processor."""
    if streaming:
        return StreamingBatchProcessor(retriever_fn, config)
    return BatchProcessor(retriever_fn, config)
