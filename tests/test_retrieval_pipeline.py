"""Integration-style tests for retrieval fusion and reranking."""
from __future__ import annotations

from typing import List
from uuid import uuid4

import pytest
from unittest.mock import AsyncMock

import triple_hybrid_rag.rag as rag_module
from triple_hybrid_rag.config import RAGConfig
from triple_hybrid_rag.core.fusion import RRFFusion
from triple_hybrid_rag.rag import RAG
from triple_hybrid_rag.types import Modality, QueryPlan, SearchChannel, SearchResult


def _make_result(text: str, channel: SearchChannel) -> SearchResult:
    return SearchResult(
        chunk_id=uuid4(),
        parent_id=uuid4(),
        document_id=uuid4(),
        text=text,
        page=1,
        modality=Modality.TEXT,
        source_channel=channel,
    )


def test_rrf_fusion_combines_channels() -> None:
    config = RAGConfig(rag_denoise_enabled=False, rag_safety_threshold=0.0)
    fusion = RRFFusion(config)

    lexical = [_make_result("lexical", SearchChannel.LEXICAL)]
    semantic = [_make_result("semantic", SearchChannel.SEMANTIC)]

    plan = QueryPlan(
        original_query="test",
        keywords=["test"],
        semantic_query_text="test",
        weights={"lexical": 1.0, "semantic": 1.0, "graph": 0.0},
    )

    fused = fusion.fuse(
        lexical_results=lexical,
        semantic_results=semantic,
        graph_results=[],
        query_plan=plan,
        top_k=5,
        apply_safety=False,
        apply_denoise=False,
    )

    assert len(fused) == 2
    assert all(result.rrf_score > 0 for result in fused)
    source_channels = [
        set(result.metadata.get("source_channels", [])) for result in fused
    ]
    assert {"lexical"} in source_channels
    assert {"semantic"} in source_channels


@pytest.mark.asyncio
async def test_rag_retrieve_reranks_results(monkeypatch: pytest.MonkeyPatch) -> None:
    config = RAGConfig(
        rag_lexical_enabled=True,
        rag_semantic_enabled=True,
        rag_graph_enabled=False,
        rag_rerank_enabled=True,
        rag_denoise_enabled=False,
        rag_safety_threshold=0.0,
    )
    rag = RAG(config)

    plan = QueryPlan(
        original_query="hello",
        keywords=["hello"],
        semantic_query_text="hello",
        lexical_top_k=2,
        semantic_top_k=2,
        graph_top_k=0,
        weights={"lexical": 1.0, "semantic": 1.0, "graph": 0.0},
    )

    rag.query_planner.plan = AsyncMock(return_value=plan)
    rag.embedder.embed_text = AsyncMock(return_value=[0.1, 0.2])

    lexical = [_make_result("lexical", SearchChannel.LEXICAL)]
    semantic = [_make_result("semantic", SearchChannel.SEMANTIC)]

    monkeypatch.setattr(rag_module, "_lexical_search", AsyncMock(return_value=lexical))
    monkeypatch.setattr(rag_module, "_semantic_search", AsyncMock(return_value=semantic))
    monkeypatch.setattr(rag, "_get_pool", AsyncMock(return_value=None))
    monkeypatch.setattr(rag_module, "_expand_to_parents", AsyncMock(side_effect=lambda _pool, results: results))
    rag.reranker.rerank = AsyncMock(return_value=[0.1, 0.9])

    result = await rag.retrieve("hello")

    assert len(result.results) == 2
    assert result.results[0].rerank_score == 0.9
    assert result.results[0].text == "semantic"
    assert result.results[1].rerank_score == 0.1
