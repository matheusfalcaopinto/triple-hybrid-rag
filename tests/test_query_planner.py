"""Tests for QueryPlanner module."""
from __future__ import annotations

import pytest

from triple_hybrid_rag.config import RAGConfig
from triple_hybrid_rag.core.query_planner import QueryPlanner


def test_simple_plan_creates_valid_query_plan() -> None:
    config = RAGConfig(rag_query_planner_enabled=False)
    planner = QueryPlanner(config)

    plan = planner._simple_plan("What is the refund policy?")

    assert plan.original_query == "What is the refund policy?"
    assert "refund" in plan.keywords or "policy" in plan.keywords
    assert plan.semantic_query_text == "What is the refund policy?"
    assert plan.requires_graph is False


@pytest.mark.asyncio
async def test_plan_falls_back_to_simple_when_disabled() -> None:
    config = RAGConfig(rag_query_planner_enabled=False)
    planner = QueryPlanner(config)

    plan = await planner.plan("How do I return an item?")

    assert plan.original_query == "How do I return an item?"
    assert plan.semantic_query_text == "How do I return an item?"
    assert isinstance(plan.keywords, list)


@pytest.mark.asyncio
async def test_plan_falls_back_on_api_error(monkeypatch: pytest.MonkeyPatch) -> None:
    config = RAGConfig(
        rag_query_planner_enabled=True,
        openai_api_key="test",
        openai_base_url="http://bad-url:9999/v1",
    )
    planner = QueryPlanner(config)

    plan = await planner.plan("test query")

    # Should have fallen back to simple plan
    assert plan.original_query == "test query"
    assert plan.semantic_query_text == "test query"
