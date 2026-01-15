"""
Tests for RAG2 Retrieval Pipeline

Tests triple-hybrid retrieval with weighted RRF fusion.
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestRetrievalCandidate:
    """Test RetrievalCandidate dataclass."""
    
    def test_candidate_creation(self) -> None:
        """Test RetrievalCandidate can be created."""
        from voice_agent.rag2.retrieval import RetrievalCandidate
        
        candidate = RetrievalCandidate(
            child_id="child_1",
            parent_id="parent_1",
            document_id="doc_123",
            text="This is some content.",
            page=1,
            modality="text",
        )
        
        assert candidate.child_id == "child_1"
        assert candidate.parent_id == "parent_1"
        assert candidate.text == "This is some content."
        assert candidate.rrf_score == 0.0
    
    def test_candidate_with_scores(self) -> None:
        """Test RetrievalCandidate with scores."""
        from voice_agent.rag2.retrieval import RetrievalCandidate
        
        candidate = RetrievalCandidate(
            child_id="child_1",
            parent_id="parent_1",
            document_id="doc_123",
            text="Content",
            page=1,
            modality="text",
            lexical_rank=1,
            semantic_rank=2,
            rrf_score=0.5,
            rerank_score=0.8,
        )
        
        assert candidate.lexical_rank == 1
        assert candidate.semantic_rank == 2
        assert candidate.rrf_score == 0.5
        assert candidate.rerank_score == 0.8


class TestRetrievalResult:
    """Test RetrievalResult dataclass."""
    
    def test_result_creation_success(self) -> None:
        """Test successful RetrievalResult."""
        from voice_agent.rag2.retrieval import RetrievalResult, RetrievalCandidate
        
        result = RetrievalResult(
            success=True,
            contexts=[
                RetrievalCandidate(
                    child_id="c1", parent_id="p1", document_id="d1",
                    text="Content", page=1, modality="text",
                )
            ],
            max_rerank_score=0.9,
        )
        
        assert result.success
        assert len(result.contexts) == 1
        assert result.max_rerank_score == 0.9
        assert not result.refused
    
    def test_result_creation_refused(self) -> None:
        """Test refused RetrievalResult."""
        from voice_agent.rag2.retrieval import RetrievalResult
        
        result = RetrievalResult(
            success=True,
            contexts=[],
            refused=True,
            refusal_reason="Below safety threshold",
            max_rerank_score=0.3,
        )
        
        assert result.success
        assert result.refused
        assert result.refusal_reason == "Below safety threshold"


class TestRAG2Retriever:
    """Test RAG2Retriever class."""
    
    def test_retriever_initialization(self) -> None:
        """Test retriever initializes correctly."""
        from voice_agent.rag2.retrieval import RAG2Retriever
        
        retriever = RAG2Retriever(org_id="org_123")
        
        assert retriever.org_id == "org_123"
        assert retriever.embedder is not None
        assert retriever.query_planner is not None
    
    def test_retriever_with_graph_disabled(self) -> None:
        """Test retriever with graph disabled."""
        from voice_agent.rag2.retrieval import RAG2Retriever
        
        retriever = RAG2Retriever(
            org_id="org_123",
            graph_enabled=False,
        )
        
        assert not retriever.graph_enabled


class TestRRFFusion:
    """Test Reciprocal Rank Fusion algorithm."""
    
    def test_fuse_single_channel(self) -> None:
        """Test RRF with single channel."""
        from voice_agent.rag2.retrieval import RAG2Retriever, RetrievalCandidate
        
        retriever = RAG2Retriever(org_id="test")
        
        candidates = [
            RetrievalCandidate(
                child_id="c1", parent_id="p1", document_id="d1",
                text="A", page=1, modality="text", lexical_rank=1,
            ),
            RetrievalCandidate(
                child_id="c2", parent_id="p1", document_id="d1",
                text="B", page=1, modality="text", lexical_rank=2,
            ),
        ]
        
        weights = {"lexical": 1.0, "semantic": 0.0, "graph": 0.0}
        
        fused = retriever._fuse_rrf(candidates, weights)
        
        # First should score higher
        assert fused[0].child_id == "c1"
        assert fused[0].rrf_score > fused[1].rrf_score
    
    def test_fuse_multiple_channels(self) -> None:
        """Test RRF with multiple channels."""
        from voice_agent.rag2.retrieval import RAG2Retriever, RetrievalCandidate
        
        retriever = RAG2Retriever(org_id="test")
        
        candidates = [
            RetrievalCandidate(
                child_id="c1", parent_id="p1", document_id="d1",
                text="A", page=1, modality="text",
                lexical_rank=1, semantic_rank=3,
            ),
            RetrievalCandidate(
                child_id="c2", parent_id="p1", document_id="d1",
                text="B", page=1, modality="text",
                lexical_rank=2, semantic_rank=1,
            ),
        ]
        
        weights = {"lexical": 0.5, "semantic": 0.5, "graph": 0.0}
        
        fused = retriever._fuse_rrf(candidates, weights)
        
        # Both should have scores
        assert all(c.rrf_score > 0 for c in fused)
    
    def test_fuse_empty_candidates(self) -> None:
        """Test RRF with no candidates."""
        from voice_agent.rag2.retrieval import RAG2Retriever
        
        retriever = RAG2Retriever(org_id="test")
        weights = {"lexical": 1.0, "semantic": 1.0, "graph": 1.0}
        
        fused = retriever._fuse_rrf([], weights)
        
        assert fused == []


class TestSafetyThreshold:
    """Test safety threshold and denoising."""
    
    def test_apply_safety_passes_threshold(self) -> None:
        """Test safety threshold passes good candidates."""
        from voice_agent.rag2.retrieval import RAG2Retriever, RetrievalCandidate
        
        retriever = RAG2Retriever(org_id="test")
        
        candidates = [
            RetrievalCandidate(
                child_id="c1", parent_id="p1", document_id="d1",
                text="Good content", page=1, modality="text",
                rerank_score=0.8,
            ),
        ]
        
        final, refused, reason, max_score = retriever._apply_safety(candidates, top_k=5)
        
        assert not refused
        assert len(final) > 0
        assert max_score == 0.8
    
    def test_apply_safety_refuses_low_score(self) -> None:
        """Test safety threshold refuses low scores."""
        from voice_agent.rag2.retrieval import RAG2Retriever, RetrievalCandidate
        
        retriever = RAG2Retriever(org_id="test")
        
        candidates = [
            RetrievalCandidate(
                child_id="c1", parent_id="p1", document_id="d1",
                text="Bad match", page=1, modality="text",
                rerank_score=0.2,  # Below threshold
            ),
        ]
        
        final, refused, reason, max_score = retriever._apply_safety(candidates, top_k=5)
        
        # Should refuse due to low score
        assert refused or len(final) == 0
    
    def test_apply_safety_empty_candidates(self) -> None:
        """Test safety with no candidates."""
        from voice_agent.rag2.retrieval import RAG2Retriever
        
        retriever = RAG2Retriever(org_id="test")
        
        final, refused, reason, max_score = retriever._apply_safety([], top_k=5)
        
        assert refused
        assert reason is not None


class TestQueryPlan:
    """Test QueryPlan dataclass."""
    
    def test_query_plan_creation(self) -> None:
        """Test QueryPlan can be created."""
        from voice_agent.rag2.query_planner import QueryPlan
        
        plan = QueryPlan(
            original_query="What is the refund policy?",
            keywords=["refund", "policy"],
            semantic_query_text="refund policy procedure",
            requires_graph=False,
            intent="factual",
        )
        
        assert plan.original_query == "What is the refund policy?"
        assert "refund" in plan.keywords
        assert plan.requires_graph is False
    
    def test_query_plan_with_cypher(self) -> None:
        """Test QueryPlan with Cypher query."""
        from voice_agent.rag2.query_planner import QueryPlan
        
        plan = QueryPlan(
            original_query="Who manages John?",
            keywords=["John", "manager"],
            semantic_query_text="John's manager",
            cypher_query="MATCH (p:Person {name: 'John'})-[:REPORTS_TO]->(m:Person) RETURN m",
            requires_graph=True,
            intent="relational",
        )
        
        assert plan.requires_graph is True
        assert plan.cypher_query is not None
        assert "MATCH" in plan.cypher_query


class TestGraphSearch:
    """Test graph search components."""
    
    def test_graph_node_creation(self) -> None:
        """Test GraphNode can be created."""
        from voice_agent.rag2.graph_search import GraphNode
        
        node = GraphNode(
            id="node_1",
            label="Person",
            properties={"name": "John"},
        )
        
        assert node.id == "node_1"
        assert node.label == "Person"
        assert node.properties["name"] == "John"
    
    def test_graph_edge_creation(self) -> None:
        """Test GraphEdge can be created."""
        from voice_agent.rag2.graph_search import GraphEdge
        
        edge = GraphEdge(
            source_id="node_1",
            target_id="node_2",
            relationship="WORKS_FOR",
            confidence=0.9,
        )
        
        assert edge.source_id == "node_1"
        assert edge.target_id == "node_2"
        assert edge.relationship == "WORKS_FOR"
        assert edge.confidence == 0.9
    
    def test_graph_search_result_creation(self) -> None:
        """Test GraphSearchResult can be created."""
        from voice_agent.rag2.graph_search import GraphSearchResult
        
        result = GraphSearchResult(
            nodes=[],
            edges=[],
            paths=[],
            chunk_ids=["c1", "c2"],
            source="sql_fallback",
        )
        
        assert len(result.chunk_ids) == 2
        assert result.source == "sql_fallback"


class TestRetrieveFunction:
    """Test module-level retrieve function."""
    
    def test_retrieve_function_exists(self) -> None:
        """Test retrieve function can be imported."""
        from voice_agent.rag2.retrieval import retrieve
        
        assert callable(retrieve)


# ──────────────────────────────────────────────────────────────────────────────
# Phase 3: Edge Case Tests - Skip Planning, Skip Rerank, Empty Query
# ──────────────────────────────────────────────────────────────────────────────

class TestRetrievalEdgeCases:
    """Edge case tests for retrieval pipeline."""
    
    @pytest.mark.asyncio
    async def test_retrieve_with_skip_planning(self) -> None:
        """Test retrieval with query planning skipped."""
        from voice_agent.rag2.retrieval import RAG2Retriever, RetrievalResult
        from voice_agent.rag2.query_planner import QueryPlan
        
        retriever = RAG2Retriever(org_id="test-org")
        
        # Mock the search methods to avoid DB calls
        with patch.object(retriever, '_retrieve_candidates', return_value=[]):
            result = await retriever.retrieve(
                query="test query words",
                skip_planning=True,
            )
        
        # When skipping planning, keywords should be simple word split
        assert result.query_plan is not None
        assert result.query_plan.keywords == ["test", "query", "words"]
        assert result.query_plan.semantic_query_text == "test query words"
        assert result.query_plan.original_query == "test query words"
        
        # Cypher should not be set for skip planning
        assert result.query_plan.cypher_query is None or result.query_plan.cypher_query == ""
        
        # Should still have timing for "planning" (even if minimal)
        assert "planning" in result.timings
        assert result.timings["planning"] < 0.1  # Should be very fast
    
    @pytest.mark.asyncio
    async def test_retrieve_skip_planning_preserves_query(self) -> None:
        """Test that skip_planning preserves the original query."""
        from voice_agent.rag2.retrieval import RAG2Retriever
        
        retriever = RAG2Retriever(org_id="test-org")
        
        original_query = "What is the refund policy for damaged items?"
        
        with patch.object(retriever, '_retrieve_candidates', return_value=[]):
            result = await retriever.retrieve(
                query=original_query,
                skip_planning=True,
            )
        
        assert result.query_plan.original_query == original_query
        assert result.query_plan.semantic_query_text == original_query
    
    @pytest.mark.asyncio
    async def test_retrieve_with_skip_rerank(self) -> None:
        """Test retrieval with reranking skipped."""
        from voice_agent.rag2.retrieval import RAG2Retriever, RetrievalCandidate
        
        retriever = RAG2Retriever(org_id="test-org")
        
        # Create mock candidates
        mock_candidates = [
            RetrievalCandidate(
                child_id="c1", parent_id="p1", document_id="d1",
                text="Result 1", page=1, modality="text",
                rrf_score=0.5,
            ),
        ]
        
        with patch.object(retriever, '_retrieve_candidates', return_value=mock_candidates):
            with patch.object(retriever, '_expand_to_parents', return_value=mock_candidates):
                with patch.object(retriever, '_apply_safety', return_value=(mock_candidates, False, None, 0.5)):
                    result = await retriever.retrieve(
                        query="test",
                        skip_rerank=True,
                    )
        
        # Rerank timing should be minimal or not present when skipped
        if "rerank" in result.timings:
            assert result.timings["rerank"] < 0.01
    
    @pytest.mark.asyncio
    async def test_retrieve_empty_query(self) -> None:
        """Test retrieval with empty query."""
        from voice_agent.rag2.retrieval import RAG2Retriever
        
        retriever = RAG2Retriever(org_id="test-org")
        
        # Empty query should still return a valid result
        with patch.object(retriever, '_retrieve_candidates', return_value=[]):
            result = await retriever.retrieve(query="")
        
        # Should handle gracefully (refuse or return empty)
        assert result.success
        assert result.refused or len(result.contexts) == 0
    
    @pytest.mark.asyncio
    async def test_retrieve_whitespace_query(self) -> None:
        """Test retrieval with whitespace-only query."""
        from voice_agent.rag2.retrieval import RAG2Retriever
        
        retriever = RAG2Retriever(org_id="test-org")
        
        with patch.object(retriever, '_retrieve_candidates', return_value=[]):
            result = await retriever.retrieve(query="   ")
        
        assert result.success
        assert result.refused or len(result.contexts) == 0
    
    @pytest.mark.asyncio
    async def test_retrieve_single_word_query(self) -> None:
        """Test retrieval with single word query."""
        from voice_agent.rag2.retrieval import RAG2Retriever
        
        retriever = RAG2Retriever(org_id="test-org")
        
        with patch.object(retriever, '_retrieve_candidates', return_value=[]):
            result = await retriever.retrieve(
                query="refund",
                skip_planning=True,
            )
        
        assert result.query_plan.keywords == ["refund"]
        assert result.query_plan.semantic_query_text == "refund"
    
    @pytest.mark.asyncio
    async def test_retrieve_returns_timing_info(self) -> None:
        """Test that retrieval returns timing information."""
        from voice_agent.rag2.retrieval import RAG2Retriever
        
        retriever = RAG2Retriever(org_id="test-org")
        
        with patch.object(retriever, '_retrieve_candidates', return_value=[]):
            result = await retriever.retrieve(query="test", skip_planning=True)
        
        # Should have timing info for key steps
        assert "planning" in result.timings
        assert "retrieval" in result.timings
        assert all(t >= 0 for t in result.timings.values())


class TestQueryPlanSkipPlanning:
    """Test QueryPlan creation when planning is skipped."""
    
    def test_query_plan_from_skip_planning(self) -> None:
        """Test QueryPlan created via skip_planning path."""
        from voice_agent.rag2.query_planner import QueryPlan
        
        query = "multi word query test"
        
        # This is what RAG2Retriever does with skip_planning=True
        plan = QueryPlan(
            original_query=query,
            keywords=query.split(),
            semantic_query_text=query,
        )
        
        assert plan.original_query == query
        assert plan.keywords == ["multi", "word", "query", "test"]
        assert plan.semantic_query_text == query
        assert plan.cypher_query is None or plan.cypher_query == ""
    
    def test_query_plan_special_characters(self) -> None:
        """Test QueryPlan with special characters in query."""
        from voice_agent.rag2.query_planner import QueryPlan
        
        query = "what's the $100 refund policy?"
        
        plan = QueryPlan(
            original_query=query,
            keywords=query.split(),
            semantic_query_text=query,
        )
        
        # Split includes special chars
        assert "what's" in plan.keywords
        assert "$100" in plan.keywords


class TestNoResultsHandling:
    """Test handling when no results are found."""
    
    @pytest.mark.asyncio
    async def test_no_candidates_returns_refused(self) -> None:
        """Test that no candidates leads to refused result."""
        from voice_agent.rag2.retrieval import RAG2Retriever
        
        retriever = RAG2Retriever(org_id="test-org")
        
        with patch.object(retriever, '_retrieve_candidates', return_value=[]):
            result = await retriever.retrieve(query="test query")
        
        assert result.success
        assert result.refused
        assert result.refusal_reason is not None
        assert "No candidates" in result.refusal_reason or len(result.contexts) == 0

