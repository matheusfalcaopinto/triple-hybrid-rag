"""
Tests for Embedding Cache Module

Tests cache hit/miss behavior, LRU eviction, and merge operations.
"""
import pytest

from triple_hybrid_rag.core.embedding_cache import (
    CacheStats,
    EmbeddingCache,
    InMemoryEmbeddingCache,
    compute_text_hash,
)


class TestComputeTextHash:
    """Test text hash computation."""
    
    def test_hash_is_consistent(self):
        """Same text should produce same hash."""
        text = "Hello, world!"
        hash1 = compute_text_hash(text)
        hash2 = compute_text_hash(text)
        assert hash1 == hash2
    
    def test_hash_is_16_chars(self):
        """Hash should be 16 characters (truncated SHA256)."""
        hash_val = compute_text_hash("test")
        assert len(hash_val) == 16
    
    def test_different_text_different_hash(self):
        """Different texts should produce different hashes."""
        hash1 = compute_text_hash("text one")
        hash2 = compute_text_hash("text two")
        assert hash1 != hash2


class TestCacheStats:
    """Test cache statistics."""
    
    def test_hit_rate_calculation(self):
        """Test hit rate is calculated correctly."""
        stats = CacheStats(hits=80, misses=20)
        assert stats.hit_rate == 0.8
    
    def test_hit_rate_zero_when_empty(self):
        """Hit rate should be 0 when no operations."""
        stats = CacheStats()
        assert stats.hit_rate == 0.0
    
    def test_reset_clears_stats(self):
        """Reset should clear all counters."""
        stats = CacheStats(hits=10, misses=5, evictions=2)
        stats.reset()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.evictions == 0


class TestInMemoryEmbeddingCache:
    """Test in-memory LRU cache backend."""
    
    @pytest.fixture
    def cache(self):
        return InMemoryEmbeddingCache(max_size=3)
    
    async def test_get_returns_none_for_missing(self, cache):
        """Get should return None for missing keys."""
        result = await cache.get("nonexistent")
        assert result is None
        assert cache.stats.misses == 1
    
    async def test_set_and_get(self, cache):
        """Set should store value retrievable by get."""
        embedding = [1.0, 2.0, 3.0]
        await cache.set("key1", embedding)
        result = await cache.get("key1")
        assert result == embedding
        assert cache.stats.hits == 1
    
    async def test_lru_eviction(self, cache):
        """Oldest entry should be evicted when at capacity."""
        # Fill cache
        await cache.set("key1", [1.0])
        await cache.set("key2", [2.0])
        await cache.set("key3", [3.0])
        
        # Add one more (should evict key1)
        await cache.set("key4", [4.0])
        
        # key1 should be evicted
        assert await cache.get("key1") is None
        assert await cache.get("key4") == [4.0]
        assert cache.stats.evictions >= 1
    
    async def test_lru_access_updates_order(self, cache):
        """Accessing an item should move it to end of LRU."""
        await cache.set("key1", [1.0])
        await cache.set("key2", [2.0])
        await cache.set("key3", [3.0])
        
        # Access key1 (moves to end)
        await cache.get("key1")
        
        # Add key4 (should evict key2, not key1)
        await cache.set("key4", [4.0])
        
        assert await cache.get("key1") is not None
        assert await cache.get("key2") is None
    
    async def test_get_many(self, cache):
        """Get many should return dict of results."""
        await cache.set("a", [1.0])
        await cache.set("b", [2.0])
        
        result = await cache.get_many(["a", "b", "c"])
        
        assert result["a"] == [1.0]
        assert result["b"] == [2.0]
        assert result["c"] is None
    
    async def test_set_many(self, cache):
        """Set many should store multiple values."""
        await cache.set_many({
            "x": [10.0],
            "y": [20.0],
        })
        
        assert await cache.get("x") == [10.0]
        assert await cache.get("y") == [20.0]
    
    async def test_delete(self, cache):
        """Delete should remove entry."""
        await cache.set("key", [1.0])
        assert await cache.delete("key") is True
        assert await cache.get("key") is None
    
    async def test_clear(self, cache):
        """Clear should remove all entries."""
        await cache.set("a", [1.0])
        await cache.set("b", [2.0])
        await cache.clear()
        
        assert len(cache) == 0
        assert await cache.get("a") is None


class TestEmbeddingCache:
    """Test high-level embedding cache."""
    
    @pytest.fixture
    def cache(self):
        return EmbeddingCache()
    
    async def test_get_cached_embeddings_all_miss(self, cache):
        """All texts should be missing on empty cache."""
        texts = ["hello", "world"]
        cached, missing = await cache.get_cached_embeddings(texts)
        
        assert len(cached) == 0
        assert missing == [0, 1]
    
    async def test_get_cached_embeddings_partial(self, cache):
        """Should return partial hits correctly."""
        # Pre-populate cache
        await cache.store_embeddings(["hello"], [[1.0, 2.0]])
        
        texts = ["hello", "world"]
        cached, missing = await cache.get_cached_embeddings(texts)
        
        assert 0 in cached
        assert cached[0] == [1.0, 2.0]
        assert missing == [1]
    
    async def test_store_and_retrieve(self, cache):
        """Store should enable retrieval."""
        texts = ["foo", "bar"]
        embeddings = [[1.0], [2.0]]
        
        await cache.store_embeddings(texts, embeddings)
        
        cached, missing = await cache.get_cached_embeddings(texts)
        
        assert len(cached) == 2
        assert len(missing) == 0
    
    def test_merge_embeddings(self):
        """Merge should combine cached and new embeddings in order."""
        cached = {0: [1.0], 2: [3.0]}
        missing_indices = [1, 3]
        new_embeddings = [[2.0], [4.0]]
        
        result = EmbeddingCache.merge_embeddings(
            cached, missing_indices, new_embeddings, total_count=4
        )
        
        assert result == [[1.0], [2.0], [3.0], [4.0]]
    
    async def test_disabled_cache_returns_all_missing(self):
        """Disabled cache should report all texts as missing."""
        cache = EmbeddingCache(enabled=False)
        
        # Even with pre-stored data
        await cache.backend.set("test", [1.0])
        
        cached, missing = await cache.get_cached_embeddings(["test"])
        
        assert len(cached) == 0
        assert missing == [0]
    
    async def test_stats_tracking(self, cache):
        """Stats should track hits and misses."""
        texts = ["a", "b"]
        
        # First lookup - all misses
        await cache.get_cached_embeddings(texts)
        assert cache.stats.misses == 2
        
        # Store and lookup again - all hits
        await cache.store_embeddings(texts, [[1.0], [2.0]])
        await cache.get_cached_embeddings(texts)
        assert cache.stats.hits == 2
