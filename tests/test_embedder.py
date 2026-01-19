"""Tests for MultimodalEmbedder module."""
from __future__ import annotations

import pytest

from triple_hybrid_rag.config import RAGConfig
from triple_hybrid_rag.core.embedder import MultimodalEmbedder


def test_truncate_embedding_applies_matryoshka() -> None:
    config = RAGConfig(
        rag_matryoshka_embeddings=True,
        rag_embed_dim_store=4,
    )
    embedder = MultimodalEmbedder(config)

    embedding = [1.0, 0.0, 0.0, 0.0, 999.0, 999.0]
    truncated = embedder._truncate_embedding(embedding)

    assert len(truncated) == 4
    # Should be normalized, first value is 1 after normalize
    assert truncated[0] == pytest.approx(1.0, rel=1e-3)


def test_truncate_embedding_skips_when_disabled() -> None:
    config = RAGConfig(
        rag_matryoshka_embeddings=False,
        rag_embed_dim_store=4,
    )
    embedder = MultimodalEmbedder(config)

    embedding = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    result = embedder._truncate_embedding(embedding)

    assert result == embedding  # Unchanged


@pytest.mark.asyncio
async def test_embed_texts_returns_zero_embeddings_on_api_failure() -> None:
    config = RAGConfig(
        rag_embed_api_base="http://bad-url:9999/v1",
        rag_embed_dim_store=4,
    )
    embedder = MultimodalEmbedder(config)

    result = await embedder.embed_texts(["hello", "world"])

    assert len(result) == 2
    assert result[0] == [0.0] * 4
    assert result[1] == [0.0] * 4


@pytest.mark.asyncio
async def test_embed_texts_empty_input_returns_empty() -> None:
    config = RAGConfig()
    embedder = MultimodalEmbedder(config)

    result = await embedder.embed_texts([])

    assert result == []
