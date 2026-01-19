"""Tests for RRFFusion module."""
from __future__ import annotations

from uuid import uuid4

import pytest

from triple_hybrid_rag.config import RAGConfig
from triple_hybrid_rag.core.fusion import RRFFusion
from triple_hybrid_rag.types import Modality, QueryPlan, SearchChannel, SearchResult


def _make_result(text: str, channel: SearchChannel, score: float = 0.0) -> SearchResult:
    result = SearchResult(
        chunk_id=uuid4(),
        parent_id=uuid4(),
        document_id=uuid4(),
        text=text,
        page=1,
        modality=Modality.TEXT,
        source_channel=channel,
    )
    if channel == SearchChannel.LEXICAL:
        result.lexical_score = score
    elif channel == SearchChannel.SEMANTIC:
        result.semantic_score = score
    elif channel == SearchChannel.GRAPH:
        result.graph_score = score
    return result


def test_fuse_assigns_rrf_scores() -> None:
    config = RAGConfig(rag_denoise_enabled=False, rag_safety_threshold=0.0)
    fusion = RRFFusion(config)

    lexical = [_make_result("a", SearchChannel.LEXICAL, 0.9)]
    semantic = [_make_result("b", SearchChannel.SEMANTIC, 0.8)]

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
    assert all(r.rrf_score > 0 for r in fused)


def test_fuse_applies_safety_threshold() -> None:
    config = RAGConfig(rag_denoise_enabled=False, rag_safety_threshold=0.5)
    fusion = RRFFusion(config)

    lexical = [_make_result("low", SearchChannel.LEXICAL, 0.1)]
    semantic = [_make_result("high", SearchChannel.SEMANTIC, 0.9)]

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
        apply_safety=True,
        apply_denoise=False,
    )

    # Only the high score result should remain
    assert len(fused) == 1
    assert fused[0].text == "high"


def test_fuse_empty_returns_empty() -> None:
    config = RAGConfig()
    fusion = RRFFusion(config)

    fused = fusion.fuse(
        lexical_results=[],
        semantic_results=[],
        graph_results=[],
        query_plan=None,
        top_k=5,
        apply_safety=False,
        apply_denoise=False,
    )

    assert fused == []
