"""Tests for JinaReranker module."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from triple_hybrid_rag.config import RAGConfig
from triple_hybrid_rag.core.jina_reranker import JinaReranker, _parse_rerank_scores


@pytest.fixture
def jina_config():
    """Create a test config with Jina settings."""
    return RAGConfig(
        jina_api_key="test-api-key",
        jina_api_base="https://api.jina.ai/v1",
        jina_rerank_model="jina-reranker-v3",
        jina_rerank_top_n=20,
        rag_rerank_enabled=True,
    )


@pytest.fixture
def reranker(jina_config):
    """Create a JinaReranker with test config."""
    return JinaReranker(jina_config)


def test_reranker_initialization(jina_config):
    """Test that reranker initializes correctly with config."""
    reranker = JinaReranker(jina_config)
    
    assert reranker.api_key == "test-api-key"
    assert reranker.model == "jina-reranker-v3"
    assert reranker.top_n == 20


def test_parse_rerank_scores_results_format():
    """Test parsing scores from Jina results format."""
    payload = {
        "results": [
            {"index": 0, "relevance_score": 0.9},
            {"index": 1, "relevance_score": 0.7},
            {"index": 2, "relevance_score": 0.5},
        ]
    }
    
    scores = _parse_rerank_scores(payload, expected_len=3)
    
    assert scores == [0.9, 0.7, 0.5]


def test_parse_rerank_scores_out_of_order():
    """Test parsing scores when results are not in order."""
    payload = {
        "results": [
            {"index": 2, "relevance_score": 0.3},
            {"index": 0, "relevance_score": 0.9},
            {"index": 1, "relevance_score": 0.6},
        ]
    }
    
    scores = _parse_rerank_scores(payload, expected_len=3)
    
    # Scores should be aligned to indices
    assert scores == [0.9, 0.6, 0.3]


def test_parse_rerank_scores_partial_results():
    """Test parsing when not all documents have scores."""
    payload = {
        "results": [
            {"index": 0, "relevance_score": 0.9},
            {"index": 2, "relevance_score": 0.5},
        ]
    }
    
    scores = _parse_rerank_scores(payload, expected_len=3)
    
    # Missing index 1 should default to 0.0
    assert scores == [0.9, 0.0, 0.5]


def test_parse_rerank_scores_empty_payload():
    """Test parsing empty payload."""
    scores = _parse_rerank_scores({}, expected_len=3)
    assert scores == []
    
    # Empty results list should return zeros
    scores = _parse_rerank_scores({"results": []}, expected_len=3)
    assert scores == []  # No results parsed from empty list


@pytest.mark.asyncio
async def test_rerank_no_api_key():
    """Test that missing API key returns zero scores."""
    config = RAGConfig(
        jina_api_key="",
        rag_rerank_enabled=True,
    )
    reranker = JinaReranker(config)
    
    result = await reranker.rerank("query", ["doc1", "doc2"])
    
    assert result == [0.0, 0.0]


@pytest.mark.asyncio
async def test_rerank_disabled():
    """Test that disabled reranker returns zero scores."""
    config = RAGConfig(
        jina_api_key="test-key",
        rag_rerank_enabled=False,
    )
    reranker = JinaReranker(config)
    
    result = await reranker.rerank("query", ["doc1", "doc2"])
    
    assert result == [0.0, 0.0]


@pytest.mark.asyncio
async def test_rerank_empty_documents(reranker):
    """Test reranking with empty document list."""
    result = await reranker.rerank("query", [])
    assert result == []


@pytest.mark.asyncio
async def test_rerank_success(reranker):
    """Test successful reranking with mocked API."""
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "results": [
            {"index": 0, "relevance_score": 0.95},
            {"index": 1, "relevance_score": 0.75},
            {"index": 2, "relevance_score": 0.55},
        ]
    }
    
    with patch.object(reranker, '_get_client') as mock_get_client:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client
        
        result = await reranker.rerank(
            query="search query",
            documents=["doc1", "doc2", "doc3"],
        )
        
        assert result == [0.95, 0.75, 0.55]
        
        # Verify API was called correctly
        call_args = mock_client.post.call_args
        json_data = call_args.kwargs.get('json') or call_args[1].get('json')
        assert json_data["model"] == "jina-reranker-v3"
        assert json_data["query"] == "search query"
        assert json_data["documents"] == ["doc1", "doc2", "doc3"]


@pytest.mark.asyncio
async def test_rerank_with_top_n(reranker):
    """Test reranking with custom top_n."""
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "results": [
            {"index": 0, "relevance_score": 0.9},
        ]
    }
    
    with patch.object(reranker, '_get_client') as mock_get_client:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client
        
        await reranker.rerank(
            query="query",
            documents=["doc1", "doc2", "doc3"],
            top_n=1,
        )
        
        # Verify top_n was passed
        call_args = mock_client.post.call_args
        json_data = call_args.kwargs.get('json') or call_args[1].get('json')
        assert json_data["top_n"] == 1


@pytest.mark.asyncio
async def test_rerank_with_indices(reranker):
    """Test reranking with indices for sorting."""
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "results": [
            {"index": 2, "relevance_score": 0.95},
            {"index": 0, "relevance_score": 0.85},
            {"index": 1, "relevance_score": 0.75},
        ]
    }
    
    with patch.object(reranker, '_get_client') as mock_get_client:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client
        
        result = await reranker.rerank_with_indices(
            query="query",
            documents=["doc1", "doc2", "doc3"],
        )
        
        # Results should be sorted by score descending
        assert result[0] == (2, 0.95)
        assert result[1] == (0, 0.85)
        assert result[2] == (1, 0.75)


def test_factory_function_returns_jina():
    """Test that get_reranker returns JinaReranker when configured."""
    from triple_hybrid_rag.core.reranker import get_reranker
    
    config = RAGConfig(
        rag_rerank_provider="jina",
        jina_api_key="test-key",
    )
    
    reranker = get_reranker(config)
    assert isinstance(reranker, JinaReranker)


def test_factory_function_returns_local():
    """Test that get_reranker returns Reranker when configured."""
    from triple_hybrid_rag.core.reranker import get_reranker, Reranker
    
    config = RAGConfig(
        rag_rerank_provider="local",
    )
    
    reranker = get_reranker(config)
    assert isinstance(reranker, Reranker)
