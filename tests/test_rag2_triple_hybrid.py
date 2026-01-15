"""
Tests for RAG2 Triple-Hybrid Integration

Tests all 3 channels (Lexical + Semantic + Graph) together:
- Full pipeline integration
- RRF fusion weights
- Cross-document graph queries  
- Channel isolation
- Entity graph traversal
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# =============================================================================
# Test: Triple-Hybrid Full Pipeline
# =============================================================================
class TestTripleHybridIntegration:
    """Test all 3 channels working together."""
    
    @pytest.mark.asyncio
    async def test_all_three_channels_return_results(self) -> None:
        """Test that all three channels can return results and fuse."""
        from voice_agent.rag2.retrieval import RAG2Retriever, RetrievalCandidate
        
        retriever = RAG2Retriever(org_id="test_org", graph_enabled=True)
        
        # Mock all three channels
        mock_lexical = [
            {"child_id": "c1", "parent_id": "p1", "document_id": "d1", "text": "Lexical result", "page": 1, "modality": "text"},
            {"child_id": "c2", "parent_id": "p1", "document_id": "d1", "text": "Lexical result 2", "page": 1, "modality": "text"},
        ]
        mock_semantic = [
            {"child_id": "c2", "parent_id": "p1", "document_id": "d1", "text": "Semantic result", "page": 1, "modality": "text"},
            {"child_id": "c3", "parent_id": "p2", "document_id": "d2", "text": "Semantic result 2", "page": 1, "modality": "text"},
        ]
        mock_graph = [
            {"child_id": "c3", "parent_id": "p2", "document_id": "d2", "text": "Graph result", "page": 1, "modality": "text"},
            {"child_id": "c4", "parent_id": "p2", "document_id": "d2", "text": "Graph result 2", "page": 2, "modality": "text"},
        ]
        
        with patch.object(retriever, '_lexical_search', new_callable=AsyncMock) as mock_lex, \
             patch.object(retriever, '_semantic_search', new_callable=AsyncMock) as mock_sem, \
             patch.object(retriever, '_graph_search', new_callable=AsyncMock) as mock_grph:
            
            mock_lex.return_value = mock_lexical
            mock_sem.return_value = mock_semantic
            mock_grph.return_value = mock_graph
            
            # Mock query planner
            with patch.object(retriever, 'query_planner') as mock_planner:
                from voice_agent.rag2.query_planner import QueryPlan
                mock_planner.plan_async = AsyncMock(return_value=QueryPlan(
                    original_query="test query",
                    keywords=["test"],
                    semantic_query_text="test query",
                    requires_graph=True,
                    cypher_query="MATCH (e:Entity) RETURN e",
                ))
                
                # Mock expansion
                with patch.object(retriever, '_expand_to_parents', new_callable=AsyncMock) as mock_expand:
                    mock_expand.side_effect = lambda x: x  # Pass through
                    
                    # Mock safety to not filter results
                    with patch.object(retriever, '_apply_safety') as mock_safety:
                        mock_safety.side_effect = lambda cands, top_k: (cands[:top_k], False, None, 0.9)
                        
                        result = await retriever.retrieve(
                            query="test query",
                            skip_rerank=True,
                        )
            
            # All three should have been called
            mock_lex.assert_called_once()
            mock_sem.assert_called_once()
            mock_grph.assert_called_once()
            
            # Should have candidates from multiple channels
            assert len(result.contexts) > 0
    
    @pytest.mark.asyncio
    async def test_channels_merged_by_child_id(self) -> None:
        """Test that same child from different channels is merged."""
        from voice_agent.rag2.retrieval import RAG2Retriever
        
        retriever = RAG2Retriever(org_id="test_org", graph_enabled=True)
        
        # Same child_id from all channels
        shared_child = {
            "child_id": "shared_c1",
            "parent_id": "p1",
            "document_id": "d1",
            "text": "Shared content",
            "page": 1,
            "modality": "text",
        }
        
        with patch.object(retriever, '_lexical_search', new_callable=AsyncMock) as mock_lex, \
             patch.object(retriever, '_semantic_search', new_callable=AsyncMock) as mock_sem, \
             patch.object(retriever, '_graph_search', new_callable=AsyncMock) as mock_grph:
            
            mock_lex.return_value = [shared_child]
            mock_sem.return_value = [shared_child]
            mock_grph.return_value = [shared_child]
            
            with patch.object(retriever, 'query_planner') as mock_planner:
                from voice_agent.rag2.query_planner import QueryPlan
                mock_planner.plan_async = AsyncMock(return_value=QueryPlan(
                    original_query="test",
                    keywords=["test"],
                    semantic_query_text="test",
                    requires_graph=True,
                    cypher_query="MATCH (e) RETURN e",
                ))
                
                with patch.object(retriever, '_expand_to_parents', new_callable=AsyncMock) as mock_exp:
                    mock_exp.side_effect = lambda x: x
                    
                    # Mock safety to not filter
                    with patch.object(retriever, '_apply_safety') as mock_safety:
                        mock_safety.side_effect = lambda cands, top_k: (cands[:top_k], False, None, 0.9)
                        
                        result = await retriever.retrieve("test", skip_rerank=True)
            
            # Should only have one candidate (merged)
            # It should have ranks from all three channels
            candidates = result.contexts
            assert len(candidates) == 1
            assert candidates[0].child_id == "shared_c1"
            assert candidates[0].lexical_rank is not None
            assert candidates[0].semantic_rank is not None
            assert candidates[0].graph_rank is not None
    
    @pytest.mark.asyncio
    async def test_graph_channel_disabled(self) -> None:
        """Test behavior when graph channel is disabled."""
        from voice_agent.rag2.retrieval import RAG2Retriever
        
        retriever = RAG2Retriever(org_id="test_org", graph_enabled=False)
        
        mock_lexical = [
            {"child_id": "c1", "parent_id": "p1", "document_id": "d1", "text": "Lex", "page": 1, "modality": "text"},
        ]
        mock_semantic = [
            {"child_id": "c2", "parent_id": "p1", "document_id": "d1", "text": "Sem", "page": 1, "modality": "text"},
        ]
        
        with patch.object(retriever, '_lexical_search', new_callable=AsyncMock) as mock_lex, \
             patch.object(retriever, '_semantic_search', new_callable=AsyncMock) as mock_sem, \
             patch.object(retriever, '_graph_search', new_callable=AsyncMock) as mock_grph:
            
            mock_lex.return_value = mock_lexical
            mock_sem.return_value = mock_semantic
            mock_grph.return_value = []  # Should not be called
            
            with patch.object(retriever, 'query_planner') as mock_planner:
                from voice_agent.rag2.query_planner import QueryPlan
                mock_planner.plan_async = AsyncMock(return_value=QueryPlan(
                    original_query="test",
                    keywords=["test"],
                    semantic_query_text="test",
                    requires_graph=True,
                    cypher_query="MATCH (e) RETURN e",
                ))
                
                with patch.object(retriever, '_expand_to_parents', new_callable=AsyncMock) as mock_exp:
                    mock_exp.side_effect = lambda x: x
                    
                    # Mock safety to not filter
                    with patch.object(retriever, '_apply_safety') as mock_safety:
                        mock_safety.side_effect = lambda cands, top_k: (cands[:top_k], False, None, 0.9)
                        
                        result = await retriever.retrieve("test", skip_rerank=True)
            
            # Graph should NOT be called when disabled
            mock_grph.assert_not_called()
            
            # Only lexical and semantic results
            assert len(result.contexts) == 2


# =============================================================================
# Test: Channel Isolation
# =============================================================================
class TestChannelIsolation:
    """Test that each channel works independently."""
    
    @pytest.mark.asyncio
    async def test_lexical_only(self) -> None:
        """Test lexical channel only returns results."""
        from voice_agent.rag2.retrieval import RAG2Retriever
        
        retriever = RAG2Retriever(org_id="test_org", graph_enabled=False)
        
        mock_lexical = [
            {"child_id": "lex1", "parent_id": "p1", "document_id": "d1", "text": "Lexical only", "page": 1, "modality": "text"},
        ]
        
        with patch.object(retriever, '_lexical_search', new_callable=AsyncMock) as mock_lex, \
             patch.object(retriever, '_semantic_search', new_callable=AsyncMock) as mock_sem:
            
            mock_lex.return_value = mock_lexical
            mock_sem.return_value = []  # No semantic results
            
            with patch.object(retriever, 'query_planner') as mock_planner:
                from voice_agent.rag2.query_planner import QueryPlan
                mock_planner.plan_async = AsyncMock(return_value=QueryPlan(
                    original_query="test",
                    keywords=["test"],
                    semantic_query_text="test",
                ))
                
                with patch.object(retriever, '_expand_to_parents', new_callable=AsyncMock) as mock_exp:
                    mock_exp.side_effect = lambda x: x
                    
                    # Mock safety to not filter
                    with patch.object(retriever, '_apply_safety') as mock_safety:
                        mock_safety.side_effect = lambda cands, top_k: (cands[:top_k], False, None, 0.9)
                        
                        result = await retriever.retrieve("test", skip_rerank=True)
            
            assert len(result.contexts) == 1
            assert result.contexts[0].lexical_rank == 1
            assert result.contexts[0].semantic_rank is None
    
    @pytest.mark.asyncio
    async def test_semantic_only(self) -> None:
        """Test semantic channel only returns results."""
        from voice_agent.rag2.retrieval import RAG2Retriever
        
        retriever = RAG2Retriever(org_id="test_org", graph_enabled=False)
        
        mock_semantic = [
            {"child_id": "sem1", "parent_id": "p1", "document_id": "d1", "text": "Semantic only", "page": 1, "modality": "text"},
        ]
        
        with patch.object(retriever, '_lexical_search', new_callable=AsyncMock) as mock_lex, \
             patch.object(retriever, '_semantic_search', new_callable=AsyncMock) as mock_sem:
            
            mock_lex.return_value = []  # No lexical results (no keywords)
            mock_sem.return_value = mock_semantic
            
            with patch.object(retriever, 'query_planner') as mock_planner:
                from voice_agent.rag2.query_planner import QueryPlan
                mock_planner.plan_async = AsyncMock(return_value=QueryPlan(
                    original_query="test",
                    keywords=[],  # No keywords
                    semantic_query_text="test",
                ))
                
                with patch.object(retriever, '_expand_to_parents', new_callable=AsyncMock) as mock_exp:
                    mock_exp.side_effect = lambda x: x
                    
                    # Mock safety to not filter
                    with patch.object(retriever, '_apply_safety') as mock_safety:
                        mock_safety.side_effect = lambda cands, top_k: (cands[:top_k], False, None, 0.9)
                        
                        result = await retriever.retrieve("test", skip_rerank=True)
            
            assert len(result.contexts) == 1
            assert result.contexts[0].semantic_rank == 1
            assert result.contexts[0].lexical_rank is None
    
    @pytest.mark.asyncio
    async def test_graph_only(self) -> None:
        """Test graph channel only returns results."""
        from voice_agent.rag2.retrieval import RAG2Retriever
        
        retriever = RAG2Retriever(org_id="test_org", graph_enabled=True)
        
        mock_graph = [
            {"child_id": "grp1", "parent_id": "p1", "document_id": "d1", "text": "Graph only", "page": 1, "modality": "text"},
        ]
        
        with patch.object(retriever, '_lexical_search', new_callable=AsyncMock) as mock_lex, \
             patch.object(retriever, '_semantic_search', new_callable=AsyncMock) as mock_sem, \
             patch.object(retriever, '_graph_search', new_callable=AsyncMock) as mock_grph:
            
            mock_lex.return_value = []
            mock_sem.return_value = []
            mock_grph.return_value = mock_graph
            
            with patch.object(retriever, 'query_planner') as mock_planner:
                from voice_agent.rag2.query_planner import QueryPlan
                mock_planner.plan_async = AsyncMock(return_value=QueryPlan(
                    original_query="test",
                    keywords=[],
                    semantic_query_text="test",
                    requires_graph=True,
                    cypher_query="MATCH (e) RETURN e",
                ))
                
                with patch.object(retriever, '_expand_to_parents', new_callable=AsyncMock) as mock_exp:
                    mock_exp.side_effect = lambda x: x
                    
                    # Mock safety to not filter
                    with patch.object(retriever, '_apply_safety') as mock_safety:
                        mock_safety.side_effect = lambda cands, top_k: (cands[:top_k], False, None, 0.9)
                        
                        result = await retriever.retrieve("test", skip_rerank=True)
            
            assert len(result.contexts) == 1
            assert result.contexts[0].graph_rank == 1
            assert result.contexts[0].lexical_rank is None
            assert result.contexts[0].semantic_rank is None
    
    @pytest.mark.asyncio
    async def test_all_channels_empty(self) -> None:
        """Test behavior when all channels return empty."""
        from voice_agent.rag2.retrieval import RAG2Retriever
        
        retriever = RAG2Retriever(org_id="test_org", graph_enabled=True)
        
        with patch.object(retriever, '_lexical_search', new_callable=AsyncMock) as mock_lex, \
             patch.object(retriever, '_semantic_search', new_callable=AsyncMock) as mock_sem, \
             patch.object(retriever, '_graph_search', new_callable=AsyncMock) as mock_grph:
            
            mock_lex.return_value = []
            mock_sem.return_value = []
            mock_grph.return_value = []
            
            with patch.object(retriever, 'query_planner') as mock_planner:
                from voice_agent.rag2.query_planner import QueryPlan
                mock_planner.plan_async = AsyncMock(return_value=QueryPlan(
                    original_query="test",
                    keywords=["test"],
                    semantic_query_text="test",
                    requires_graph=True,
                    cypher_query="MATCH (e) RETURN e",
                ))
                
                result = await retriever.retrieve("test", skip_rerank=True)
            
            # Should be refused with reason
            assert result.refused
            assert "No candidates" in (result.refusal_reason or "")


# =============================================================================
# Test: Weighted RRF Fusion
# =============================================================================
class TestWeightedRRF:
    """Test Weighted RRF fusion algorithm."""
    
    def test_rrf_default_weights(self) -> None:
        """Test RRF uses default weights correctly."""
        from voice_agent.rag2.retrieval import RAG2Retriever, RetrievalCandidate
        
        retriever = RAG2Retriever(org_id="test")
        
        candidates = [
            RetrievalCandidate(
                child_id="c1", parent_id="p1", document_id="d1",
                text="A", page=1, modality="text",
                lexical_rank=1, semantic_rank=2, graph_rank=3,
            ),
        ]
        
        # Default weights: lexical=0.7, semantic=0.8, graph=1.0
        weights = {"lexical": 0.7, "semantic": 0.8, "graph": 1.0}
        
        fused = retriever._fuse_rrf(candidates, weights)
        
        # RRF score should combine all three
        assert fused[0].rrf_score > 0
        
        # Manual calculation: 0.7/(60+1) + 0.8/(60+2) + 1.0/(60+3)
        k = 60
        expected = 0.7 / (k + 1) + 0.8 / (k + 2) + 1.0 / (k + 3)
        assert abs(fused[0].rrf_score - expected) < 0.001
    
    def test_rrf_graph_boost(self) -> None:
        """Test that graph channel has higher weight (1.0 vs 0.7/0.8)."""
        from voice_agent.rag2.retrieval import RAG2Retriever, RetrievalCandidate
        
        retriever = RAG2Retriever(org_id="test")
        
        # Three candidates, each only in one channel
        candidates = [
            RetrievalCandidate(
                child_id="lex", parent_id="p1", document_id="d1",
                text="Lexical", page=1, modality="text",
                lexical_rank=1,  # Only lexical
            ),
            RetrievalCandidate(
                child_id="sem", parent_id="p1", document_id="d1",
                text="Semantic", page=1, modality="text",
                semantic_rank=1,  # Only semantic
            ),
            RetrievalCandidate(
                child_id="grp", parent_id="p1", document_id="d1",
                text="Graph", page=1, modality="text",
                graph_rank=1,  # Only graph
            ),
        ]
        
        weights = {"lexical": 0.7, "semantic": 0.8, "graph": 1.0}
        
        fused = retriever._fuse_rrf(candidates, weights)
        
        # Graph-only should rank highest
        assert fused[0].child_id == "grp"
        # Semantic-only should rank second
        assert fused[1].child_id == "sem"
        # Lexical-only should rank third
        assert fused[2].child_id == "lex"
    
    def test_rrf_multi_channel_beats_single(self) -> None:
        """Test that appearing in multiple channels beats single channel."""
        from voice_agent.rag2.retrieval import RAG2Retriever, RetrievalCandidate
        
        retriever = RAG2Retriever(org_id="test")
        
        candidates = [
            RetrievalCandidate(
                child_id="single", parent_id="p1", document_id="d1",
                text="Single channel", page=1, modality="text",
                graph_rank=1,  # Only graph, rank 1
            ),
            RetrievalCandidate(
                child_id="multi", parent_id="p1", document_id="d1",
                text="Multi channel", page=1, modality="text",
                lexical_rank=2, semantic_rank=2, graph_rank=2,  # All three, rank 2
            ),
        ]
        
        weights = {"lexical": 0.7, "semantic": 0.8, "graph": 1.0}
        
        fused = retriever._fuse_rrf(candidates, weights)
        
        # Multi-channel should beat single channel even at lower ranks
        assert fused[0].child_id == "multi"
        assert fused[0].rrf_score > fused[1].rrf_score
    
    def test_rrf_custom_weights(self) -> None:
        """Test RRF with custom weights."""
        from voice_agent.rag2.retrieval import RAG2Retriever, RetrievalCandidate
        
        retriever = RAG2Retriever(org_id="test")
        
        candidates = [
            RetrievalCandidate(
                child_id="c1", parent_id="p1", document_id="d1",
                text="A", page=1, modality="text",
                lexical_rank=1, semantic_rank=1,
            ),
        ]
        
        # Custom weights: heavy lexical
        weights = {"lexical": 2.0, "semantic": 0.5, "graph": 0.0}
        
        fused = retriever._fuse_rrf(candidates, weights)
        
        k = 60
        expected = 2.0 / (k + 1) + 0.5 / (k + 1) + 0.0
        assert abs(fused[0].rrf_score - expected) < 0.001
    
    def test_rrf_preserves_other_fields(self) -> None:
        """Test that RRF preserves all candidate fields."""
        from voice_agent.rag2.retrieval import RAG2Retriever, RetrievalCandidate
        
        retriever = RAG2Retriever(org_id="test")
        
        candidates = [
            RetrievalCandidate(
                child_id="c1", parent_id="p1", document_id="d1",
                text="Some text", page=5, modality="image",
                lexical_rank=1, semantic_rank=2,
                parent_text="Parent context",
                section_heading="Section A",
            ),
        ]
        
        weights = {"lexical": 1.0, "semantic": 1.0, "graph": 1.0}
        
        fused = retriever._fuse_rrf(candidates, weights)
        
        # All fields preserved
        assert fused[0].child_id == "c1"
        assert fused[0].parent_id == "p1"
        assert fused[0].document_id == "d1"
        assert fused[0].text == "Some text"
        assert fused[0].page == 5
        assert fused[0].modality == "image"
        assert fused[0].parent_text == "Parent context"
        assert fused[0].section_heading == "Section A"


# =============================================================================
# Test: Entity Graph Traversal
# =============================================================================
class TestEntityGraphTraversal:
    """Test entity-based graph queries."""
    
    @pytest.mark.asyncio
    async def test_entity_relation_query(self) -> None:
        """Test finding related chunks via entity relations."""
        from voice_agent.rag2.retrieval import RAG2Retriever
        
        retriever = RAG2Retriever(org_id="test_org", graph_enabled=True)
        
        # Graph search returns chunks related via entities
        mock_graph = [
            {"child_id": "related1", "parent_id": "p1", "document_id": "d1", "text": "Related via entity", "page": 1, "modality": "text"},
            {"child_id": "related2", "parent_id": "p2", "document_id": "d2", "text": "Also related", "page": 1, "modality": "text"},
        ]
        
        with patch.object(retriever, '_lexical_search', new_callable=AsyncMock) as mock_lex, \
             patch.object(retriever, '_semantic_search', new_callable=AsyncMock) as mock_sem, \
             patch.object(retriever, '_graph_search', new_callable=AsyncMock) as mock_grph:
            
            mock_lex.return_value = []
            mock_sem.return_value = []
            mock_grph.return_value = mock_graph
            
            with patch.object(retriever, 'query_planner') as mock_planner:
                from voice_agent.rag2.query_planner import QueryPlan
                mock_planner.plan_async = AsyncMock(return_value=QueryPlan(
                    original_query="What companies work with Acme Corp?",
                    keywords=["Acme", "Corp"],
                    semantic_query_text="companies related to Acme Corp",
                    requires_graph=True,
                    cypher_query="MATCH (e1:Entity)-[:RELATED_TO]->(e2:Entity) WHERE e1.name = 'Acme Corp' RETURN e2",
                ))
                
                with patch.object(retriever, '_expand_to_parents', new_callable=AsyncMock) as mock_exp:
                    mock_exp.side_effect = lambda x: x
                    
                    # Mock safety to not filter
                    with patch.object(retriever, '_apply_safety') as mock_safety:
                        mock_safety.side_effect = lambda cands, top_k: (cands[:top_k], False, None, 0.9)
                        
                        result = await retriever.retrieve(
                            "What companies work with Acme Corp?",
                            skip_rerank=True,
                        )
            
            # Should have graph-sourced results
            assert len(result.contexts) == 2
            assert all(c.graph_rank is not None for c in result.contexts)
    
    @pytest.mark.asyncio
    async def test_cypher_query_passed_to_graph_search(self) -> None:
        """Test that cypher query from planner reaches graph search."""
        from voice_agent.rag2.retrieval import RAG2Retriever
        
        retriever = RAG2Retriever(org_id="test_org", graph_enabled=True)
        
        with patch.object(retriever, '_lexical_search', new_callable=AsyncMock) as mock_lex, \
             patch.object(retriever, '_semantic_search', new_callable=AsyncMock) as mock_sem, \
             patch.object(retriever, '_graph_search', new_callable=AsyncMock) as mock_grph:
            
            mock_lex.return_value = []
            mock_sem.return_value = []
            mock_grph.return_value = []
            
            expected_cypher = "MATCH (e:Entity {name: 'Test'}) RETURN e"
            
            with patch.object(retriever, 'query_planner') as mock_planner:
                from voice_agent.rag2.query_planner import QueryPlan
                mock_planner.plan_async = AsyncMock(return_value=QueryPlan(
                    original_query="test",
                    keywords=["Test"],
                    semantic_query_text="test",
                    requires_graph=True,
                    cypher_query=expected_cypher,
                ))
                
                await retriever.retrieve("test", skip_rerank=True)
            
            # Verify cypher was passed
            mock_grph.assert_called_once()
            call_kwargs = mock_grph.call_args
            assert call_kwargs[1]["cypher"] == expected_cypher


# =============================================================================
# Test: Cross-Document Graph Queries
# =============================================================================
class TestCrossDocumentGraph:
    """Test graph queries spanning multiple documents."""
    
    @pytest.mark.asyncio
    async def test_graph_returns_multi_document_results(self) -> None:
        """Test graph channel can return results from multiple documents."""
        from voice_agent.rag2.retrieval import RAG2Retriever
        
        retriever = RAG2Retriever(org_id="test_org", graph_enabled=True)
        
        # Results from different documents
        mock_graph = [
            {"child_id": "c1", "parent_id": "p1", "document_id": "doc_A", "text": "From doc A", "page": 1, "modality": "text"},
            {"child_id": "c2", "parent_id": "p2", "document_id": "doc_B", "text": "From doc B", "page": 1, "modality": "text"},
            {"child_id": "c3", "parent_id": "p3", "document_id": "doc_C", "text": "From doc C", "page": 1, "modality": "text"},
        ]
        
        with patch.object(retriever, '_lexical_search', new_callable=AsyncMock) as mock_lex, \
             patch.object(retriever, '_semantic_search', new_callable=AsyncMock) as mock_sem, \
             patch.object(retriever, '_graph_search', new_callable=AsyncMock) as mock_grph:
            
            mock_lex.return_value = []
            mock_sem.return_value = []
            mock_grph.return_value = mock_graph
            
            with patch.object(retriever, 'query_planner') as mock_planner:
                from voice_agent.rag2.query_planner import QueryPlan
                mock_planner.plan_async = AsyncMock(return_value=QueryPlan(
                    original_query="cross doc query",
                    keywords=[],
                    semantic_query_text="cross doc query",
                    requires_graph=True,
                    cypher_query="MATCH (e)-[:MENTIONS]->(c:Chunk) RETURN c",
                ))
                
                with patch.object(retriever, '_expand_to_parents', new_callable=AsyncMock) as mock_exp:
                    mock_exp.side_effect = lambda x: x
                    
                    # Mock safety to not filter
                    with patch.object(retriever, '_apply_safety') as mock_safety:
                        mock_safety.side_effect = lambda cands, top_k: (cands[:top_k], False, None, 0.9)
                        
                        result = await retriever.retrieve("cross doc query", skip_rerank=True)
            
            # Should have results from 3 different documents
            doc_ids = {c.document_id for c in result.contexts}
            assert len(doc_ids) == 3
            assert "doc_A" in doc_ids
            assert "doc_B" in doc_ids
            assert "doc_C" in doc_ids
    
    @pytest.mark.asyncio
    async def test_semantic_and_graph_different_documents(self) -> None:
        """Test semantic and graph channels return different documents."""
        from voice_agent.rag2.retrieval import RAG2Retriever
        
        retriever = RAG2Retriever(org_id="test_org", graph_enabled=True)
        
        mock_semantic = [
            {"child_id": "sem1", "parent_id": "p1", "document_id": "semantic_doc", "text": "Semantic match", "page": 1, "modality": "text"},
        ]
        mock_graph = [
            {"child_id": "grp1", "parent_id": "p2", "document_id": "graph_doc", "text": "Graph match", "page": 1, "modality": "text"},
        ]
        
        with patch.object(retriever, '_lexical_search', new_callable=AsyncMock) as mock_lex, \
             patch.object(retriever, '_semantic_search', new_callable=AsyncMock) as mock_sem, \
             patch.object(retriever, '_graph_search', new_callable=AsyncMock) as mock_grph:
            
            mock_lex.return_value = []
            mock_sem.return_value = mock_semantic
            mock_grph.return_value = mock_graph
            
            with patch.object(retriever, 'query_planner') as mock_planner:
                from voice_agent.rag2.query_planner import QueryPlan
                mock_planner.plan_async = AsyncMock(return_value=QueryPlan(
                    original_query="find related",
                    keywords=[],
                    semantic_query_text="find related",
                    requires_graph=True,
                    cypher_query="MATCH (e) RETURN e",
                ))
                
                with patch.object(retriever, '_expand_to_parents', new_callable=AsyncMock) as mock_exp:
                    mock_exp.side_effect = lambda x: x
                    
                    # Mock safety to not filter
                    with patch.object(retriever, '_apply_safety') as mock_safety:
                        mock_safety.side_effect = lambda cands, top_k: (cands[:top_k], False, None, 0.9)
                        
                        result = await retriever.retrieve("find related", skip_rerank=True)
            
            # Should have both documents
            doc_ids = {c.document_id for c in result.contexts}
            assert "semantic_doc" in doc_ids
            assert "graph_doc" in doc_ids


# =============================================================================
# Test: Full Pipeline Integration
# =============================================================================
class TestFullPipeline:
    """Test complete ingestion to retrieval flow."""
    
    @pytest.mark.asyncio
    async def test_retrieval_with_expansion(self) -> None:
        """Test retrieval includes parent expansion."""
        from voice_agent.rag2.retrieval import RAG2Retriever
        
        retriever = RAG2Retriever(org_id="test_org")
        
        mock_results = [
            {"child_id": "c1", "parent_id": "p1", "document_id": "d1", "text": "Child text", "page": 1, "modality": "text"},
        ]
        
        # Mock parent expansion to return parent text
        async def mock_expansion(candidates):
            for c in candidates:
                c.parent_text = "Parent text with full context"
                c.section_heading = "Introduction"
            return candidates
        
        with patch.object(retriever, '_lexical_search', new_callable=AsyncMock) as mock_lex, \
             patch.object(retriever, '_semantic_search', new_callable=AsyncMock) as mock_sem:
            
            mock_lex.return_value = mock_results
            mock_sem.return_value = []
            
            with patch.object(retriever, 'query_planner') as mock_planner:
                from voice_agent.rag2.query_planner import QueryPlan
                mock_planner.plan_async = AsyncMock(return_value=QueryPlan(
                    original_query="test",
                    keywords=["test"],
                    semantic_query_text="test",
                ))
                
                with patch.object(retriever, '_expand_to_parents', new_callable=AsyncMock) as mock_exp:
                    mock_exp.side_effect = mock_expansion
                    
                    # Mock safety to not filter
                    with patch.object(retriever, '_apply_safety') as mock_safety:
                        mock_safety.side_effect = lambda cands, top_k: (cands[:top_k], False, None, 0.9)
                        
                        result = await retriever.retrieve("test", skip_rerank=True)
            
            # Parent text should be attached
            assert len(result.contexts) == 1
            assert result.contexts[0].parent_text == "Parent text with full context"
            assert result.contexts[0].section_heading == "Introduction"
    
    @pytest.mark.asyncio
    async def test_timings_tracked(self) -> None:
        """Test that pipeline timings are tracked."""
        from voice_agent.rag2.retrieval import RAG2Retriever
        
        retriever = RAG2Retriever(org_id="test_org")
        
        with patch.object(retriever, '_lexical_search', new_callable=AsyncMock) as mock_lex, \
             patch.object(retriever, '_semantic_search', new_callable=AsyncMock) as mock_sem:
            
            mock_lex.return_value = []
            mock_sem.return_value = []
            
            with patch.object(retriever, 'query_planner') as mock_planner:
                from voice_agent.rag2.query_planner import QueryPlan
                mock_planner.plan_async = AsyncMock(return_value=QueryPlan(
                    original_query="test",
                    keywords=["test"],
                    semantic_query_text="test",
                ))
                
                result = await retriever.retrieve("test", skip_rerank=True)
            
            # Timings should be present
            assert "planning" in result.timings
            assert "retrieval" in result.timings
            assert result.timings["planning"] >= 0
            assert result.timings["retrieval"] >= 0
    
    @pytest.mark.asyncio
    async def test_query_plan_attached_to_result(self) -> None:
        """Test that query plan is attached to result."""
        from voice_agent.rag2.retrieval import RAG2Retriever
        
        retriever = RAG2Retriever(org_id="test_org")
        
        with patch.object(retriever, '_lexical_search', new_callable=AsyncMock) as mock_lex, \
             patch.object(retriever, '_semantic_search', new_callable=AsyncMock) as mock_sem:
            
            mock_lex.return_value = []
            mock_sem.return_value = []
            
            expected_plan_query = "test query for plan"
            
            with patch.object(retriever, 'query_planner') as mock_planner:
                from voice_agent.rag2.query_planner import QueryPlan
                mock_planner.plan_async = AsyncMock(return_value=QueryPlan(
                    original_query=expected_plan_query,
                    keywords=["test"],
                    semantic_query_text="test",
                ))
                
                result = await retriever.retrieve("test query for plan", skip_rerank=True)
            
            # Query plan should be attached
            assert result.query_plan is not None
            assert result.query_plan.original_query == expected_plan_query


# =============================================================================
# Test: Safety Threshold with Triple-Hybrid
# =============================================================================
class TestSafetyWithTripleHybrid:
    """Test safety threshold works with triple-hybrid results."""
    
    def test_safety_applies_to_fused_results(self) -> None:
        """Test safety threshold applies after RRF fusion."""
        from voice_agent.rag2.retrieval import RAG2Retriever, RetrievalCandidate
        
        retriever = RAG2Retriever(org_id="test")
        
        # Candidates with various rerank scores
        candidates = [
            RetrievalCandidate(
                child_id="high", parent_id="p1", document_id="d1",
                text="High quality", page=1, modality="text",
                rerank_score=0.9,
            ),
            RetrievalCandidate(
                child_id="med", parent_id="p1", document_id="d1",
                text="Medium quality", page=1, modality="text",
                rerank_score=0.7,
            ),
            RetrievalCandidate(
                child_id="low", parent_id="p1", document_id="d1",
                text="Low quality", page=1, modality="text",
                rerank_score=0.3,
            ),
        ]
        
        final, refused, reason, max_score = retriever._apply_safety(candidates, top_k=5)
        
        assert not refused
        assert max_score == 0.9
        # Low score might be filtered by denoising
    
    def test_safety_refuses_all_low_scores(self) -> None:
        """Test safety refuses when all scores below threshold."""
        from voice_agent.rag2.retrieval import RAG2Retriever, RetrievalCandidate
        from voice_agent.config import SETTINGS
        
        retriever = RAG2Retriever(org_id="test")
        
        # All below typical threshold (0.6)
        candidates = [
            RetrievalCandidate(
                child_id="c1", parent_id="p1", document_id="d1",
                text="Bad match", page=1, modality="text",
                rerank_score=0.2,
            ),
            RetrievalCandidate(
                child_id="c2", parent_id="p1", document_id="d1",
                text="Another bad", page=1, modality="text",
                rerank_score=0.3,
            ),
        ]
        
        # Temporarily set threshold
        original_threshold = SETTINGS.rag2_safety_threshold
        SETTINGS.rag2_safety_threshold = 0.6
        
        try:
            final, refused, reason, max_score = retriever._apply_safety(candidates, top_k=5)
            
            assert refused
            assert "below threshold" in (reason or "").lower()
        finally:
            SETTINGS.rag2_safety_threshold = original_threshold
    
    def test_denoising_removes_low_relative_scores(self) -> None:
        """Test denoising removes scores below alpha*max."""
        from voice_agent.rag2.retrieval import RAG2Retriever, RetrievalCandidate
        from voice_agent.config import SETTINGS
        
        retriever = RAG2Retriever(org_id="test")
        
        # High max, one candidate far below
        candidates = [
            RetrievalCandidate(
                child_id="high", parent_id="p1", document_id="d1",
                text="Excellent", page=1, modality="text",
                rerank_score=0.95,
            ),
            RetrievalCandidate(
                child_id="low", parent_id="p1", document_id="d1",
                text="Poor", page=1, modality="text",
                rerank_score=0.2,  # Far below 0.95 * alpha
            ),
        ]
        
        # Set denoising alpha (e.g., 0.5)
        original_alpha = SETTINGS.rag2_denoise_alpha
        SETTINGS.rag2_denoise_alpha = 0.5
        
        try:
            final, refused, reason, max_score = retriever._apply_safety(candidates, top_k=5)
            
            # Low score should be filtered
            # Min allowed = 0.95 * 0.5 = 0.475
            # 0.2 < 0.475, so filtered
            assert len(final) == 1
            assert final[0].child_id == "high"
        finally:
            SETTINGS.rag2_denoise_alpha = original_alpha


# =============================================================================
# Test: Convenience Function
# =============================================================================
class TestConvenienceFunction:
    """Test the convenience retrieve() function."""
    
    @pytest.mark.asyncio
    async def test_retrieve_function_creates_retriever(self) -> None:
        """Test that retrieve() function creates retriever internally."""
        from voice_agent.rag2.retrieval import retrieve
        
        with patch('voice_agent.rag2.retrieval.RAG2Retriever') as MockRetriever:
            mock_instance = MagicMock()
            mock_instance.retrieve = AsyncMock(return_value=MagicMock(success=True))
            MockRetriever.return_value = mock_instance
            
            result = await retrieve(org_id="test_org", query="test query")
            
            MockRetriever.assert_called_once_with(org_id="test_org")
            mock_instance.retrieve.assert_called_once_with("test query")
