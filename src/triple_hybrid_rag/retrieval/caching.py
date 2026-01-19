"""
Query Caching Module

Intelligent caching for RAG retrieval results with:
- Semantic cache keys (similar queries hit same cache)
- TTL-based expiration
- LRU eviction policy
- Cache warming and invalidation
"""

import hashlib
import logging
import time
import threading
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable, Tuple
from collections import OrderedDict
from enum import Enum

logger = logging.getLogger(__name__)

class CacheStrategy(Enum):
    """Cache key generation strategies."""
    EXACT = "exact"  # Exact string match
    NORMALIZED = "normalized"  # Lowercase, stripped
    SEMANTIC = "semantic"  # Embedding-based similarity

@dataclass
class CacheConfig:
    """Configuration for query caching."""
    enabled: bool = True
    strategy: CacheStrategy = CacheStrategy.NORMALIZED
    
    # Cache limits
    max_entries: int = 1000
    max_memory_mb: int = 100
    
    # TTL settings
    default_ttl_seconds: int = 3600  # 1 hour
    hot_query_ttl_seconds: int = 7200  # 2 hours for frequently accessed
    
    # Semantic cache settings
    similarity_threshold: float = 0.95  # For semantic cache hits
    
    # Features
    enable_stats: bool = True
    enable_warming: bool = True

@dataclass
class CacheEntry:
    """A cached query result."""
    key: str
    query: str
    results: Any
    created_at: float
    expires_at: float
    hit_count: int = 0
    last_accessed: float = 0
    size_bytes: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_expired(self) -> bool:
        return time.time() > self.expires_at
    
    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at

@dataclass
class CacheStats:
    """Cache statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0
    total_entries: int = 0
    memory_used_bytes: int = 0
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

class QueryCache:
    """
    LRU cache for query results with TTL support.
    
    Supports exact, normalized, and semantic cache key strategies.
    """
    
    def __init__(
        self,
        config: Optional[CacheConfig] = None,
        embed_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
    ):
        self.config = config or CacheConfig()
        self.embed_fn = embed_fn
        
        # LRU cache using OrderedDict
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        
        # Semantic cache index
        self._query_embeddings: Dict[str, List[float]] = {}
        
        # Statistics
        self._stats = CacheStats()
    
    def get(
        self,
        query: str,
        namespace: str = "default",
    ) -> Optional[Any]:
        """
        Get cached results for a query.
        
        Args:
            query: The query string
            namespace: Cache namespace for isolation
            
        Returns:
            Cached results or None if not found
        """
        if not self.config.enabled:
            return None
        
        key = self._generate_key(query, namespace)
        
        with self._lock:
            # Check exact/normalized key
            entry = self._cache.get(key)
            
            if entry is None and self.config.strategy == CacheStrategy.SEMANTIC:
                # Try semantic match
                entry = self._semantic_lookup(query, namespace)
            
            if entry is None:
                self._stats.misses += 1
                return None
            
            # Check expiration
            if entry.is_expired:
                self._remove_entry(key)
                self._stats.expirations += 1
                self._stats.misses += 1
                return None
            
            # Update access info
            entry.hit_count += 1
            entry.last_accessed = time.time()
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            
            self._stats.hits += 1
            
            logger.debug(f"Cache hit for query: {query[:50]}...")
            return entry.results
    
    def set(
        self,
        query: str,
        results: Any,
        namespace: str = "default",
        ttl_seconds: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Cache results for a query.
        
        Args:
            query: The query string
            results: Results to cache
            namespace: Cache namespace
            ttl_seconds: Override default TTL
            metadata: Additional metadata
        """
        if not self.config.enabled:
            return
        
        key = self._generate_key(query, namespace)
        ttl = ttl_seconds or self.config.default_ttl_seconds
        
        now = time.time()
        
        # Estimate size
        size = self._estimate_size(results)
        
        entry = CacheEntry(
            key=key,
            query=query,
            results=results,
            created_at=now,
            expires_at=now + ttl,
            last_accessed=now,
            size_bytes=size,
            metadata=metadata or {},
        )
        
        with self._lock:
            # Evict if needed
            self._evict_if_needed(size)
            
            # Store entry
            self._cache[key] = entry
            self._cache.move_to_end(key)
            
            # Store embedding for semantic lookup
            if self.config.strategy == CacheStrategy.SEMANTIC and self.embed_fn:
                try:
                    emb = self.embed_fn([query])[0]
                    self._query_embeddings[key] = emb
                except Exception as e:
                    logger.warning(f"Failed to compute query embedding: {e}")
            
            self._update_stats()
        
        logger.debug(f"Cached results for query: {query[:50]}...")
    
    def invalidate(
        self,
        query: Optional[str] = None,
        namespace: Optional[str] = None,
        pattern: Optional[str] = None,
    ) -> int:
        """
        Invalidate cache entries.
        
        Args:
            query: Specific query to invalidate
            namespace: Invalidate all in namespace
            pattern: Invalidate matching pattern
            
        Returns:
            Number of entries invalidated
        """
        count = 0
        
        with self._lock:
            if query and namespace:
                key = self._generate_key(query, namespace)
                if key in self._cache:
                    self._remove_entry(key)
                    count = 1
            
            elif namespace:
                keys_to_remove = [
                    k for k in self._cache.keys()
                    if k.startswith(f"{namespace}:")
                ]
                for key in keys_to_remove:
                    self._remove_entry(key)
                    count += 1
            
            elif pattern:
                import re
                regex = re.compile(pattern)
                keys_to_remove = [
                    k for k, v in self._cache.items()
                    if regex.search(v.query)
                ]
                for key in keys_to_remove:
                    self._remove_entry(key)
                    count += 1
            
            else:
                # Clear all
                count = len(self._cache)
                self._cache.clear()
                self._query_embeddings.clear()
        
        logger.info(f"Invalidated {count} cache entries")
        return count
    
    def warm(
        self,
        queries: List[str],
        retriever_fn: Callable[[str], Any],
        namespace: str = "default",
    ) -> int:
        """
        Warm cache with pre-computed results.
        
        Args:
            queries: Queries to pre-cache
            retriever_fn: Function to retrieve results
            namespace: Cache namespace
            
        Returns:
            Number of entries warmed
        """
        if not self.config.enable_warming:
            return 0
        
        count = 0
        for query in queries:
            if self.get(query, namespace) is None:
                try:
                    results = retriever_fn(query)
                    self.set(query, results, namespace)
                    count += 1
                except Exception as e:
                    logger.warning(f"Failed to warm cache for query: {e}")
        
        logger.info(f"Warmed cache with {count} entries")
        return count
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            self._update_stats()
            return CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                evictions=self._stats.evictions,
                expirations=self._stats.expirations,
                total_entries=self._stats.total_entries,
                memory_used_bytes=self._stats.memory_used_bytes,
            )
    
    def _generate_key(self, query: str, namespace: str) -> str:
        """Generate cache key based on strategy."""
        if self.config.strategy == CacheStrategy.EXACT:
            normalized = query
        else:  # NORMALIZED or SEMANTIC
            normalized = query.lower().strip()
        
        hash_val = hashlib.md5(normalized.encode()).hexdigest()[:16]
        return f"{namespace}:{hash_val}"
    
    def _semantic_lookup(
        self,
        query: str,
        namespace: str,
    ) -> Optional[CacheEntry]:
        """Find semantically similar cached query."""
        if not self.embed_fn or not self._query_embeddings:
            return None
        
        try:
            query_emb = self.embed_fn([query])[0]
            
            best_match = None
            best_sim = 0.0
            
            for key, emb in self._query_embeddings.items():
                if not key.startswith(f"{namespace}:"):
                    continue
                
                sim = self._cosine_similarity(query_emb, emb)
                if sim > best_sim and sim >= self.config.similarity_threshold:
                    best_sim = sim
                    best_match = key
            
            if best_match:
                logger.debug(f"Semantic cache hit (sim={best_sim:.3f})")
                return self._cache.get(best_match)
            
        except Exception as e:
            logger.warning(f"Semantic lookup failed: {e}")
        
        return None
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(y * y for y in b) ** 0.5
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0
    
    def _evict_if_needed(self, new_size: int) -> None:
        """Evict entries if cache is full."""
        # Check entry count
        while len(self._cache) >= self.config.max_entries:
            self._evict_oldest()
        
        # Check memory (rough estimate)
        max_bytes = self.config.max_memory_mb * 1024 * 1024
        current_bytes = sum(e.size_bytes for e in self._cache.values())
        
        while current_bytes + new_size > max_bytes and self._cache:
            evicted = self._evict_oldest()
            if evicted:
                current_bytes -= evicted.size_bytes
    
    def _evict_oldest(self) -> Optional[CacheEntry]:
        """Evict least recently used entry."""
        if not self._cache:
            return None
        
        # Get oldest (first) entry
        key = next(iter(self._cache))
        entry = self._cache.pop(key)
        self._query_embeddings.pop(key, None)
        self._stats.evictions += 1
        
        return entry
    
    def _remove_entry(self, key: str) -> None:
        """Remove a specific entry."""
        self._cache.pop(key, None)
        self._query_embeddings.pop(key, None)
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate object size in bytes."""
        try:
            import sys
            return sys.getsizeof(obj)
        except Exception:
            return 1024  # Default estimate
    
    def _update_stats(self) -> None:
        """Update statistics."""
        self._stats.total_entries = len(self._cache)
        self._stats.memory_used_bytes = sum(
            e.size_bytes for e in self._cache.values()
        )

class MultiLevelCache:
    """
    Multi-level caching with L1 (memory) and L2 (persistent).
    
    L1: Fast in-memory cache for recent queries
    L2: Persistent cache (Redis, disk) for longer-term storage
    """
    
    def __init__(
        self,
        l1_cache: Optional[QueryCache] = None,
        l2_backend: Optional[Any] = None,  # Redis client or similar
    ):
        self.l1 = l1_cache or QueryCache()
        self.l2 = l2_backend
    
    def get(self, query: str, namespace: str = "default") -> Optional[Any]:
        """Get from L1, fallback to L2."""
        # Try L1
        result = self.l1.get(query, namespace)
        if result is not None:
            return result
        
        # Try L2
        if self.l2:
            result = self._l2_get(query, namespace)
            if result is not None:
                # Promote to L1
                self.l1.set(query, result, namespace)
                return result
        
        return None
    
    def set(
        self,
        query: str,
        results: Any,
        namespace: str = "default",
        **kwargs,
    ) -> None:
        """Set in both L1 and L2."""
        self.l1.set(query, results, namespace, **kwargs)
        
        if self.l2:
            self._l2_set(query, results, namespace)
    
    def _l2_get(self, query: str, namespace: str) -> Optional[Any]:
        """Get from L2 backend."""
        # Placeholder for Redis/persistent storage
        return None
    
    def _l2_set(self, query: str, results: Any, namespace: str) -> None:
        """Set in L2 backend."""
        # Placeholder for Redis/persistent storage
        pass

def create_query_cache(
    config: Optional[CacheConfig] = None,
    embed_fn: Optional[Callable] = None,
) -> QueryCache:
    """Factory function to create a query cache."""
    return QueryCache(config=config, embed_fn=embed_fn)
