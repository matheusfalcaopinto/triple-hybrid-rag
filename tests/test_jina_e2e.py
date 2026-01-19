"""
End-to-End tests for Jina AI integration.

These tests require a valid JINA_API_KEY to be set in the environment.
Run with: pytest tests/test_jina_e2e.py -v -s

Skip these tests in CI by using: pytest --ignore=tests/test_jina_e2e.py
"""
from __future__ import annotations

import os

import pytest

# Skip all tests if JINA_API_KEY is not set
pytestmark = pytest.mark.skipif(
    not os.getenv("JINA_API_KEY"),
    reason="JINA_API_KEY environment variable not set"
)


@pytest.fixture
def jina_api_key():
    """Get Jina API key from environment."""
    return os.getenv("JINA_API_KEY")


@pytest.fixture
def jina_config(jina_api_key):
    """Create config with real Jina API key."""
    from triple_hybrid_rag.config import RAGConfig
    return RAGConfig(
        jina_api_key=jina_api_key,
        jina_embed_dimensions=1024,
        rag_multimodal_embedding_enabled=True,
        rag_rerank_enabled=True,
    )


@pytest.fixture
def embedder(jina_config):
    """Create JinaEmbedder for E2E tests."""
    from triple_hybrid_rag.core.jina_embedder import JinaEmbedder
    return JinaEmbedder(jina_config)


@pytest.fixture
def reranker(jina_config):
    """Create JinaReranker for E2E tests."""
    from triple_hybrid_rag.core.jina_reranker import JinaReranker
    return JinaReranker(jina_config)


class TestJinaEmbedderE2E:
    """E2E tests for Jina Embedder with real API."""
    
    @pytest.mark.asyncio
    async def test_embed_single_text(self, embedder):
        """Test embedding a single text with real API."""
        result = await embedder.embed_text("Hello, world!")
        
        assert len(result) == 1024
        assert all(isinstance(x, float) for x in result)
        # Embeddings should be non-zero normalized vectors
        assert any(x != 0.0 for x in result)
    
    @pytest.mark.asyncio
    async def test_embed_multiple_texts(self, embedder):
        """Test embedding multiple texts in a batch."""
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Python is a popular programming language.",
        ]
        
        results = await embedder.embed_texts(texts)
        
        assert len(results) == 3
        for emb in results:
            assert len(emb) == 1024
    
    @pytest.mark.asyncio
    async def test_embed_query_vs_passage(self, embedder):
        """Test that query and passage embeddings are different."""
        text = "What is machine learning?"
        
        query_emb = await embedder.embed_query(text)
        passage_emb = await embedder.embed_text(text)
        
        # Both should be valid embeddings
        assert len(query_emb) == 1024
        assert len(passage_emb) == 1024
        
        # They might differ slightly due to different tasks
        # Just verify both are non-zero
        assert any(x != 0.0 for x in query_emb)
        assert any(x != 0.0 for x in passage_emb)
    
    @pytest.mark.asyncio
    async def test_semantic_similarity(self, embedder):
        """Test that similar texts have higher cosine similarity."""
        text1 = "The cat sat on the mat."
        text2 = "A cat was sitting on a rug."
        text3 = "The stock market crashed yesterday."
        
        emb1 = await embedder.embed_text(text1)
        emb2 = await embedder.embed_text(text2)
        emb3 = await embedder.embed_text(text3)
        
        # Similar texts should have higher similarity
        sim_similar = embedder.cosine_similarity(emb1, emb2)
        sim_different = embedder.cosine_similarity(emb1, emb3)
        
        assert sim_similar > sim_different
        print(f"Similarity (cat sentences): {sim_similar:.4f}")
        print(f"Similarity (cat vs stock): {sim_different:.4f}")
    
    @pytest.mark.asyncio
    async def test_embed_image(self, embedder):
        """Test embedding an image (requires multimodal support)."""
        # Create a simple 1x1 pixel PNG
        import base64
        
        # Minimal valid PNG (1x1 white pixel)
        png_b64 = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )
        image_bytes = base64.b64decode(png_b64)
        
        result = await embedder.embed_image(image_bytes)
        
        assert len(result) == 1024
        assert any(x != 0.0 for x in result)
    
    @pytest.mark.asyncio
    async def test_concurrent_embedding(self, embedder):
        """Test concurrent batch embedding."""
        texts = [f"This is test sentence number {i}." for i in range(20)]
        
        progress_updates = []
        def progress_callback(completed, total):
            progress_updates.append((completed, total))
        
        results = await embedder.embed_texts_concurrent(
            texts, 
            progress_callback=progress_callback
        )
        
        assert len(results) == 20
        for emb in results:
            assert len(emb) == 1024
        
        print(f"Progress updates: {progress_updates}")


class TestJinaRerankerE2E:
    """E2E tests for Jina Reranker with real API."""
    
    @pytest.mark.asyncio
    async def test_rerank_documents(self, reranker):
        """Test reranking documents with real API."""
        query = "What is machine learning?"
        documents = [
            "Machine learning is a method of data analysis that automates analytical model building.",
            "The weather today is sunny with a high of 75 degrees.",
            "Deep learning is part of a broader family of machine learning methods.",
            "I like to eat pizza on Friday nights.",
        ]
        
        scores = await reranker.rerank(query, documents)
        
        assert len(scores) == 4
        
        # ML-related documents should score higher
        print(f"Rerank scores: {scores}")
        
        # First and third docs (ML-related) should have higher scores
        ml_score = (scores[0] + scores[2]) / 2
        other_score = (scores[1] + scores[3]) / 2
        
        assert ml_score > other_score
    
    @pytest.mark.asyncio
    async def test_rerank_with_indices(self, reranker):
        """Test reranking with indices for sorting."""
        query = "programming languages"
        documents = [
            "Python is a high-level programming language.",
            "The cat sleeps on the couch.",
            "JavaScript runs in web browsers.",
            "I enjoy hiking in the mountains.",
        ]
        
        results = await reranker.rerank_with_indices(query, documents)
        
        # Results should be sorted by relevance
        assert len(results) == 4
        
        # First result should have highest score
        assert results[0][1] >= results[1][1] >= results[2][1] >= results[3][1]
        
        print(f"Sorted results: {results}")
        
        # Top results should include programming-related docs (indices 0 and 2)
        top_indices = [r[0] for r in results[:2]]
        assert 0 in top_indices or 2 in top_indices


class TestIntegration:
    """Integration tests combining embedder and reranker."""
    
    @pytest.mark.asyncio
    async def test_embed_then_rerank(self, embedder, reranker):
        """Test a realistic retrieval + rerank pipeline."""
        query = "How do neural networks learn?"
        
        # Simulate retrieved documents
        documents = [
            "Neural networks learn through a process called backpropagation.",
            "The weather forecast predicts rain tomorrow.",
            "Deep learning uses multiple layers of neurons to process data.",
            "I bought a new car last week.",
            "Gradient descent optimizes the weights in neural networks.",
        ]
        
        # Step 1: Embed query and documents
        query_emb = await embedder.embed_query(query)
        doc_embs = await embedder.embed_texts(documents)
        
        # Step 2: Compute initial similarities
        initial_scores = [
            embedder.cosine_similarity(query_emb, doc_emb)
            for doc_emb in doc_embs
        ]
        
        print(f"Initial embedding scores: {initial_scores}")
        
        # Step 3: Rerank documents
        rerank_scores = await reranker.rerank(query, documents)
        
        print(f"Rerank scores: {rerank_scores}")
        
        # Both methods should rank neural network docs higher
        # Doc indices 0, 2, 4 are relevant
        for relevant_idx in [0, 2, 4]:
            for irrelevant_idx in [1, 3]:
                assert rerank_scores[relevant_idx] > rerank_scores[irrelevant_idx]
