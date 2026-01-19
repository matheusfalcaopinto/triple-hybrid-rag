"""
Embedding Cache for Triple-Hybrid-RAG

Provides caching of embeddings to avoid re-embedding identical content.
Supports multiple backends:
- In-memory LRU cache (default, no dependencies)
- Redis (optional, for distributed caching)

Cache keys are SHA256 hashes of the text content (first 16 chars).
"""

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CacheStats:
    """Statistics for embedding cache operations."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def reset(self) -> None:
        """Reset all statistics."""
        self.hits = 0
        self.misses = 0
        self.evictions = 0


class EmbeddingCacheBackend(ABC):
    """Abstract base class for embedding cache backends."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[List[float]]:
        """Get embedding from cache by key."""
        pass
    
    @abstractmethod
    async def set(self, key: str, embedding: List[float], ttl_seconds: Optional[int] = None) -> None:
        """Store embedding in cache."""
        pass
    
    @abstractmethod
    async def get_many(self, keys: List[str]) -> Dict[str, Optional[List[float]]]:
        """Get multiple embeddings from cache."""
        pass
    
    @abstractmethod
    async def set_many(self, items: Dict[str, List[float]], ttl_seconds: Optional[int] = None) -> None:
        """Store multiple embeddings in cache."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete embedding from cache."""
        pass
    
    @abstractmethod
    async def clear(self) -> None:
        """Clear all cached embeddings."""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close cache connection."""
        pass


class InMemoryEmbeddingCache(EmbeddingCacheBackend):
    """
    In-memory LRU cache for embeddings.
    
    Best for single-process applications or development.
    Uses a simple dict with LRU eviction.
    """
    
    def __init__(self, max_size: int = 100_000, default_ttl: Optional[int] = None):
        """
        Initialize in-memory cache.
        
        Args:
            max_size: Maximum number of embeddings to cache
            default_ttl: Default TTL in seconds (None = no expiry)
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, List[float]] = {}
        self._access_order: List[str] = []  # For LRU tracking
        self.stats = CacheStats()
    
    async def get(self, key: str) -> Optional[List[float]]:
        """Get embedding from cache."""
        if key in self._cache:
            # Move to end for LRU
            self._access_order.remove(key)
            self._access_order.append(key)
            self.stats.hits += 1
            return self._cache[key]
        
        self.stats.misses += 1
        return None
    
    async def set(self, key: str, embedding: List[float], ttl_seconds: Optional[int] = None) -> None:
        """Store embedding in cache."""
        # Evict if at capacity
        while len(self._cache) >= self.max_size:
            oldest_key = self._access_order.pop(0)
            del self._cache[oldest_key]
            self.stats.evictions += 1
        
        self._cache[key] = embedding
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
    
    async def get_many(self, keys: List[str]) -> Dict[str, Optional[List[float]]]:
        """Get multiple embeddings from cache."""
        result = {}
        for key in keys:
            result[key] = await self.get(key)
        return result
    
    async def set_many(self, items: Dict[str, List[float]], ttl_seconds: Optional[int] = None) -> None:
        """Store multiple embeddings in cache."""
        for key, embedding in items.items():
            await self.set(key, embedding, ttl_seconds)
    
    async def delete(self, key: str) -> bool:
        """Delete embedding from cache."""
        if key in self._cache:
            del self._cache[key]
            self._access_order.remove(key)
            return True
        return False
    
    async def clear(self) -> None:
        """Clear all cached embeddings."""
        self._cache.clear()
        self._access_order.clear()
        self.stats.reset()
    
    async def close(self) -> None:
        """Close cache (no-op for in-memory)."""
        pass
    
    def __len__(self) -> int:
        """Return number of cached embeddings."""
        return len(self._cache)


class RedisEmbeddingCache(EmbeddingCacheBackend):
    """
    Redis-backed embedding cache.
    
    Best for distributed applications or when embeddings should
    persist across restarts.
    """
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        key_prefix: str = "emb:",
        default_ttl: int = 86400 * 7,  # 7 days
    ):
        """
        Initialize Redis cache.
        
        Args:
            redis_url: Redis connection URL
            key_prefix: Prefix for all cache keys
            default_ttl: Default TTL in seconds
        """
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self.default_ttl = default_ttl
        self._redis = None
        self.stats = CacheStats()
    
    async def _get_redis(self):
        """Lazy initialize Redis connection."""
        if self._redis is None:
            try:
                import redis.asyncio as redis
                self._redis = await redis.from_url(self.redis_url)
            except ImportError:
                raise RuntimeError("redis package required for Redis cache: pip install redis")
        return self._redis
    
    def _make_key(self, key: str) -> str:
        """Create full Redis key with prefix."""
        return f"{self.key_prefix}{key}"
    
    async def get(self, key: str) -> Optional[List[float]]:
        """Get embedding from Redis."""
        redis = await self._get_redis()
        full_key = self._make_key(key)
        
        data = await redis.get(full_key)
        if data:
            self.stats.hits += 1
            return json.loads(data)
        
        self.stats.misses += 1
        return None
    
    async def set(self, key: str, embedding: List[float], ttl_seconds: Optional[int] = None) -> None:
        """Store embedding in Redis."""
        redis = await self._get_redis()
        full_key = self._make_key(key)
        ttl = ttl_seconds or self.default_ttl
        
        await redis.setex(full_key, ttl, json.dumps(embedding))
    
    async def get_many(self, keys: List[str]) -> Dict[str, Optional[List[float]]]:
        """Get multiple embeddings from Redis using pipeline."""
        redis = await self._get_redis()
        full_keys = [self._make_key(k) for k in keys]
        
        # Use MGET for efficiency
        values = await redis.mget(full_keys)
        
        result = {}
        for key, value in zip(keys, values):
            if value:
                result[key] = json.loads(value)
                self.stats.hits += 1
            else:
                result[key] = None
                self.stats.misses += 1
        
        return result
    
    async def set_many(self, items: Dict[str, List[float]], ttl_seconds: Optional[int] = None) -> None:
        """Store multiple embeddings in Redis using pipeline."""
        redis = await self._get_redis()
        ttl = ttl_seconds or self.default_ttl
        
        pipe = redis.pipeline()
        for key, embedding in items.items():
            full_key = self._make_key(key)
            pipe.setex(full_key, ttl, json.dumps(embedding))
        
        await pipe.execute()
    
    async def delete(self, key: str) -> bool:
        """Delete embedding from Redis."""
        redis = await self._get_redis()
        full_key = self._make_key(key)
        result = await redis.delete(full_key)
        return result > 0
    
    async def clear(self) -> None:
        """Clear all cached embeddings with prefix."""
        redis = await self._get_redis()
        
        # Find all keys with prefix
        cursor = 0
        while True:
            cursor, keys = await redis.scan(cursor, match=f"{self.key_prefix}*", count=1000)
            if keys:
                await redis.delete(*keys)
            if cursor == 0:
                break
        
        self.stats.reset()
    
    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._redis = None


def compute_text_hash(text: str) -> str:
    """
    Compute cache key from text content.
    
    Uses SHA256 hash truncated to 16 characters for compactness
    while maintaining very low collision probability.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class EmbeddingCache:
    """
    High-level embedding cache with automatic key generation.
    
    Usage:
        cache = EmbeddingCache()
        
        # Check for cached embeddings
        texts = ["hello world", "foo bar"]
        cached, missing_indices = await cache.get_cached_embeddings(texts)
        
        # Embed only missing texts
        missing_texts = [texts[i] for i in missing_indices]
        new_embeddings = await embedder.embed_texts(missing_texts)
        
        # Store new embeddings
        await cache.store_embeddings(missing_texts, new_embeddings)
        
        # Merge results
        all_embeddings = cache.merge_embeddings(cached, missing_indices, new_embeddings)
    """
    
    def __init__(
        self,
        backend: Optional[EmbeddingCacheBackend] = None,
        enabled: bool = True,
    ):
        """
        Initialize embedding cache.
        
        Args:
            backend: Cache backend (defaults to in-memory)
            enabled: Whether caching is enabled
        """
        self.backend = backend or InMemoryEmbeddingCache()
        self.enabled = enabled
    
    async def get_cached_embeddings(
        self,
        texts: List[str],
    ) -> Tuple[Dict[int, List[float]], List[int]]:
        """
        Get cached embeddings for texts.
        
        Args:
            texts: List of texts to look up
            
        Returns:
            Tuple of:
            - Dict mapping index -> embedding for cached texts
            - List of indices for texts that need embedding
        """
        if not self.enabled or not texts:
            return {}, list(range(len(texts)))
        
        # Compute keys
        keys = [compute_text_hash(text) for text in texts]
        
        # Batch lookup
        cached_values = await self.backend.get_many(keys)
        
        # Separate hits and misses
        cached: Dict[int, List[float]] = {}
        missing_indices: List[int] = []
        
        for i, (key, text) in enumerate(zip(keys, texts)):
            embedding = cached_values.get(key)
            if embedding is not None:
                cached[i] = embedding
            else:
                missing_indices.append(i)
        
        return cached, missing_indices
    
    async def store_embeddings(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """
        Store embeddings in cache.
        
        Args:
            texts: List of texts
            embeddings: Corresponding embeddings
            ttl_seconds: Optional TTL override
        """
        if not self.enabled or not texts:
            return
        
        items = {}
        for text, embedding in zip(texts, embeddings):
            key = compute_text_hash(text)
            items[key] = embedding
        
        await self.backend.set_many(items, ttl_seconds)
    
    @staticmethod
    def merge_embeddings(
        cached: Dict[int, List[float]],
        missing_indices: List[int],
        new_embeddings: List[List[float]],
        total_count: int,
    ) -> List[List[float]]:
        """
        Merge cached and newly computed embeddings in original order.
        
        Args:
            cached: Dict of index -> embedding for cached texts
            missing_indices: Indices of texts that were embedded
            new_embeddings: Newly computed embeddings
            total_count: Total number of texts
            
        Returns:
            List of all embeddings in original order
        """
        result: List[List[float]] = [[] for _ in range(total_count)]
        
        # Fill in cached embeddings
        for idx, embedding in cached.items():
            result[idx] = embedding
        
        # Fill in new embeddings
        for idx, embedding in zip(missing_indices, new_embeddings):
            result[idx] = embedding
        
        return result
    
    @property
    def stats(self) -> CacheStats:
        """Get cache statistics."""
        return self.backend.stats
    
    async def clear(self) -> None:
        """Clear all cached embeddings."""
        await self.backend.clear()
    
    async def close(self) -> None:
        """Close cache connections."""
        await self.backend.close()
