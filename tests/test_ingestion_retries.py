"""Tests for ingestion retry helpers."""
from __future__ import annotations

from typing import Any, cast

import pytest

from triple_hybrid_rag.config import RAGConfig
from triple_hybrid_rag.ingestion.ingest import IngestStats, Ingestor


class FlakyEmbedder:
    """Embedder that fails once before succeeding."""

    def __init__(self) -> None:
        self.calls = 0

    async def embed_texts(self, texts: list[str], raise_on_error: bool = False):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("transient error")
        return [[0.1, 0.2, 0.3] for _ in texts]


@pytest.mark.asyncio
async def test_embed_retry_tracks_stats() -> None:
    config = RAGConfig(
        rag_ingest_embed_retry_attempts=2,
        rag_ingest_retry_backoff_min=0.01,
        rag_ingest_retry_backoff_max=0.01,
    )
    ingestor = Ingestor(embedder=cast(Any, FlakyEmbedder()), config=config)
    stats = IngestStats()

    embeddings = await ingestor._embed_texts_with_retry(["hello"], stats)

    assert embeddings == [[0.1, 0.2, 0.3]]
    assert stats.embed_retries == 1
