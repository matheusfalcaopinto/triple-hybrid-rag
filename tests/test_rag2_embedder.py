"""
Tests for RAG2 Embedder Module

Tests Matryoshka embeddings with truncation from 4096→1024 dimensions.
"""
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestMatryoshkaTruncation:
    """Test Matryoshka embedding truncation logic."""
    
    def test_truncate_matryoshka_basic(self) -> None:
        """Test basic truncation preserves first N dimensions."""
        from voice_agent.rag2.embedder import truncate_matryoshka
        
        # Create 4096-dim embedding
        full_embedding = [float(i) for i in range(4096)]
        
        # Truncate to 1024
        truncated = truncate_matryoshka(full_embedding, target_dim=1024, normalize=False)
        
        assert len(truncated) == 1024
        assert truncated == [float(i) for i in range(1024)]
    
    def test_truncate_matryoshka_preserves_short(self) -> None:
        """Test truncation doesn't alter embeddings shorter than target."""
        from voice_agent.rag2.embedder import truncate_matryoshka
        
        short_embedding = [1.0, 2.0, 3.0]
        truncated = truncate_matryoshka(short_embedding, target_dim=1024, normalize=False)
        
        # Should return as-is (normalized by default, so turn off)
        assert truncated == short_embedding
    
    def test_truncate_matryoshka_empty(self) -> None:
        """Test truncation handles empty embeddings."""
        from voice_agent.rag2.embedder import truncate_matryoshka
        
        truncated = truncate_matryoshka([], target_dim=1024, normalize=False)
        assert truncated == []
    
    def test_truncate_matryoshka_numpy(self) -> None:
        """Test truncation works with numpy arrays converted to list."""
        from voice_agent.rag2.embedder import truncate_matryoshka
        
        embedding = np.arange(4096, dtype=np.float64)
        truncated = truncate_matryoshka(embedding.tolist(), target_dim=1024, normalize=False)
        
        assert len(truncated) == 1024
        assert truncated[0] == 0.0
        assert truncated[-1] == 1023.0
    
    def test_truncate_with_normalize(self) -> None:
        """Test truncation with normalization."""
        from voice_agent.rag2.embedder import truncate_matryoshka
        
        # Simple vector
        embedding = [3.0, 4.0] + [0.0] * 4094  # 4096 dims
        truncated = truncate_matryoshka(embedding, target_dim=2, normalize=True)
        
        # Should be normalized to unit vector
        assert len(truncated) == 2
        norm = (truncated[0]**2 + truncated[1]**2) ** 0.5
        assert abs(norm - 1.0) < 1e-6


class TestNormalization:
    """Test L2 normalization."""
    
    def test_normalize_l2_unit_vector(self) -> None:
        """Test L2 normalization produces unit vector."""
        from voice_agent.rag2.embedder import normalize_l2
        
        embedding = [3.0, 4.0]  # Known 3-4-5 triangle
        normalized = normalize_l2(embedding)
        
        assert len(normalized) == 2
        assert abs(normalized[0] - 0.6) < 1e-6
        assert abs(normalized[1] - 0.8) < 1e-6
        
        # Verify unit norm
        norm = sum(x**2 for x in normalized) ** 0.5
        assert abs(norm - 1.0) < 1e-6
    
    def test_normalize_l2_zero_vector(self) -> None:
        """Test normalization handles zero vectors."""
        from voice_agent.rag2.embedder import normalize_l2
        
        normalized = normalize_l2([0.0, 0.0, 0.0])
        
        # Should return zeros (not NaN or error)
        assert all(x == 0.0 for x in normalized)
    
    def test_normalize_l2_already_normalized(self) -> None:
        """Test normalization of already-normalized vector."""
        from voice_agent.rag2.embedder import normalize_l2
        
        unit = [1.0, 0.0, 0.0]
        normalized = normalize_l2(unit)
        
        for a, b in zip(normalized, unit):
            assert abs(a - b) < 1e-6


class TestRAG2Embedder:
    """Test the RAG2Embedder class."""
    
    @pytest.fixture
    def mock_openai_response(self) -> MagicMock:
        """Create mock OpenAI embedding response."""
        mock_embedding = MagicMock()
        mock_embedding.embedding = [float(i) for i in range(4096)]
        
        mock_response = MagicMock()
        mock_response.data = [mock_embedding]
        
        return mock_response
    
    def test_embedder_initialization(self) -> None:
        """Test embedder initializes with correct defaults."""
        from voice_agent.rag2.embedder import RAG2Embedder
        
        embedder = RAG2Embedder()
        
        assert embedder.model is not None
        assert embedder.store_dim > 0
        assert embedder.model_dim > 0
    
    def test_embed_text_basic(self, mock_openai_response: MagicMock) -> None:
        """Test basic text embedding."""
        from voice_agent.rag2.embedder import RAG2Embedder
        
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = mock_openai_response
        
        embedder = RAG2Embedder()
        embedder._client = mock_client  # Inject mock client directly
        
        result = embedder.embed_text("Hello world")
        
        assert result.embedding is not None
        assert result.text == "Hello world"
        assert result.error is None
    
    def test_embed_query_truncates(self, mock_openai_response: MagicMock) -> None:
        """Test query embedding truncates to target dimension."""
        from voice_agent.rag2.embedder import RAG2Embedder
        
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = mock_openai_response
        
        embedder = RAG2Embedder(store_dim=1024)
        embedder._client = mock_client  # Inject mock client directly
        embedder._client = mock_client
        
        embedding = embedder.embed_query("Test query")
        
        assert len(embedding) == 1024


class TestEmbeddingResult:
    """Test EmbeddingResult dataclass."""
    
    def test_embedding_result_creation(self) -> None:
        """Test EmbeddingResult can be created."""
        from voice_agent.rag2.embedder import EmbeddingResult
        
        result = EmbeddingResult(
            text="test",
            embedding=[1.0, 2.0, 3.0],
        )
        
        assert result.embedding == [1.0, 2.0, 3.0]
        assert result.text == "test"
        assert result.error is None
        assert result.full_embedding is None
    
    def test_embedding_result_with_full(self) -> None:
        """Test EmbeddingResult with full embedding."""
        from voice_agent.rag2.embedder import EmbeddingResult
        
        result = EmbeddingResult(
            text="test",
            embedding=[1.0, 2.0],
            full_embedding=[float(i) for i in range(4096)],
        )
        
        assert len(result.embedding) == 2
        assert result.full_embedding is not None
        assert len(result.full_embedding) == 4096


class TestGetRAG2Embedder:
    """Test factory function."""
    
    def test_get_rag2_embedder_returns_instance(self) -> None:
        """Test factory returns embedder instance."""
        from voice_agent.rag2.embedder import get_rag2_embedder
        
        embedder = get_rag2_embedder()
        
        assert embedder is not None
    
    def test_get_rag2_embedder_uses_settings(self) -> None:
        """Test factory uses settings for configuration."""
        from voice_agent.rag2.embedder import get_rag2_embedder, RAG2Embedder
        
        embedder = get_rag2_embedder()
        
        assert isinstance(embedder, RAG2Embedder)
