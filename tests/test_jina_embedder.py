"""Tests for JinaEmbedder module."""
from __future__ import annotations

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from triple_hybrid_rag.config import RAGConfig
from triple_hybrid_rag.core.jina_embedder import JinaEmbedder


@pytest.fixture
def jina_config():
    """Create a test config with Jina settings."""
    return RAGConfig(
        jina_api_key="test-api-key",
        jina_api_base="https://api.jina.ai/v1",
        jina_embed_model="jina-embeddings-v4",
        jina_embed_dimensions=1024,
        jina_embed_batch_size=100,
        rag_multimodal_embedding_enabled=True,
    )


@pytest.fixture
def embedder(jina_config):
    """Create a JinaEmbedder with test config."""
    return JinaEmbedder(jina_config)


def test_embedder_initialization(jina_config):
    """Test that embedder initializes correctly with config."""
    embedder = JinaEmbedder(jina_config)
    
    assert embedder.api_key == "test-api-key"
    assert embedder.model == "jina-embeddings-v4"
    assert embedder.dimensions == 1024


def test_normalize_embedding(embedder):
    """Test L2 normalization of embeddings."""
    embedding = [3.0, 4.0, 0.0, 0.0]  # |v| = 5
    normalized = embedder._normalize(embedding)
    
    assert len(normalized) == 4
    assert normalized[0] == pytest.approx(0.6, rel=1e-3)
    assert normalized[1] == pytest.approx(0.8, rel=1e-3)


def test_cosine_similarity(embedder):
    """Test cosine similarity computation."""
    e1 = [1.0, 0.0, 0.0]
    e2 = [0.0, 1.0, 0.0]
    e3 = [1.0, 0.0, 0.0]
    
    # Orthogonal vectors have 0 similarity
    assert embedder.cosine_similarity(e1, e2) == pytest.approx(0.0, abs=1e-6)
    
    # Identical vectors have 1.0 similarity
    assert embedder.cosine_similarity(e1, e3) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.asyncio
async def test_embed_texts_no_api_key():
    """Test that missing API key returns zero embeddings."""
    config = RAGConfig(
        jina_api_key="",
        jina_embed_dimensions=4,
    )
    embedder = JinaEmbedder(config)
    
    result = await embedder.embed_texts(["hello", "world"])
    
    assert len(result) == 2
    assert result[0] == [0.0] * 4
    assert result[1] == [0.0] * 4


@pytest.mark.asyncio
async def test_embed_texts_empty_input(embedder):
    """Test empty input returns empty list."""
    result = await embedder.embed_texts([])
    assert result == []


@pytest.mark.asyncio
async def test_embed_text_success(embedder):
    """Test successful single text embedding with mocked API."""
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "data": [{"embedding": [0.1, 0.2, 0.3, 0.4] + [0.0] * 1020}]
    }
    
    with patch.object(embedder, '_get_client') as mock_get_client:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client
        
        result = await embedder.embed_text("hello world")
        
        assert len(result) == 1024
        assert result[0] != 0.0  # Should be normalized


@pytest.mark.asyncio
async def test_embed_query_uses_query_task(embedder):
    """Test that embed_query uses retrieval.query task."""
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "data": [{"embedding": [0.1] * 1024}]
    }
    
    with patch.object(embedder, '_get_client') as mock_get_client:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client
        
        await embedder.embed_query("search query")
        
        # Check that the post was called with retrieval.query task
        call_args = mock_client.post.call_args
        json_data = call_args.kwargs.get('json') or call_args[1].get('json')
        assert json_data["task"] == "retrieval.query"


@pytest.mark.asyncio
async def test_embed_images_disabled(jina_config):
    """Test image embedding when multimodal is disabled."""
    jina_config.rag_multimodal_embedding_enabled = False
    embedder = JinaEmbedder(jina_config)
    
    # Fake image data
    image_bytes = b'\x89PNG\r\n\x1a\n' + b'\x00' * 100
    
    result = await embedder.embed_images([image_bytes])
    
    assert len(result) == 1
    assert result[0] == [0.0] * 1024


@pytest.mark.asyncio
async def test_embed_image_success(embedder):
    """Test successful image embedding with mocked API."""
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "data": [{"embedding": [0.5] * 1024}]
    }
    
    with patch.object(embedder, '_get_client') as mock_get_client:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client
        
        # Fake PNG image
        image_bytes = b'\x89PNG\r\n\x1a\n' + b'\x00' * 100
        result = await embedder.embed_image(image_bytes)
        
        assert len(result) == 1024
        
        # Verify image was sent as base64
        call_args = mock_client.post.call_args
        json_data = call_args.kwargs.get('json') or call_args[1].get('json')
        assert "image" in json_data["input"][0]


@pytest.mark.asyncio
async def test_embed_mixed_averages_embeddings(embedder):
    """Test that mixed embedding averages text and image embeddings."""
    text_emb = [1.0] * 1024
    image_emb = [0.0] * 1024
    
    with patch.object(embedder, 'embed_text', return_value=text_emb):
        with patch.object(embedder, 'embed_image', return_value=image_emb):
            result = await embedder.embed_mixed("text", b"image")
            
            # Average of 1.0 and 0.0 = 0.5, then normalized
            assert len(result) == 1024


@pytest.mark.asyncio 
async def test_batching(embedder):
    """Test that large inputs are batched correctly."""
    embedder.batch_size = 2  # Small batch for testing
    
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "data": [
            {"embedding": [0.1] * 1024},
            {"embedding": [0.2] * 1024},
        ]
    }
    
    with patch.object(embedder, '_get_client') as mock_get_client:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client
        
        # 5 texts should be split into 3 batches (2+2+1)
        texts = ["text1", "text2", "text3", "text4", "text5"]
        
        # We need different responses for different batch sizes
        # For simplicity, just check that multiple calls were made
        mock_client.post.side_effect = [
            mock_response,  # batch 1: 2 items
            mock_response,  # batch 2: 2 items  
            MagicMock(
                raise_for_status=MagicMock(),
                json=MagicMock(return_value={
                    "data": [{"embedding": [0.3] * 1024}]
                })
            ),  # batch 3: 1 item
        ]
        
        result = await embedder.embed_texts(texts)
        
        # Should have embeddings for all 5 texts
        assert len(result) == 5
        # Should have made 3 API calls
        assert mock_client.post.call_count == 3


def test_factory_function_returns_jina():
    """Test that get_embedder returns JinaEmbedder when configured."""
    from triple_hybrid_rag.core.embedder import get_embedder
    
    config = RAGConfig(
        rag_embed_provider="jina",
        jina_api_key="test-key",
    )
    
    embedder = get_embedder(config)
    assert isinstance(embedder, JinaEmbedder)


def test_factory_function_returns_local():
    """Test that get_embedder returns MultimodalEmbedder when configured."""
    from triple_hybrid_rag.core.embedder import get_embedder, MultimodalEmbedder
    
    config = RAGConfig(
        rag_embed_provider="local",
    )
    
    embedder = get_embedder(config)
    assert isinstance(embedder, MultimodalEmbedder)
