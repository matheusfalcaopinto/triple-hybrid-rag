"""
Tests for RAG2 Tool Connection

Validates that the agent tool `search_knowledge_base` correctly uses
RAG2Retriever when RAG2_ENABLED=true.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import sys


class TestRAG2ToolConnection:
    """Test that search_knowledge_base uses RAG2 when enabled."""
    
    def test_rag2_enabled_uses_rag2_retriever(self):
        """When RAG2_ENABLED=true, should use RAG2Retriever."""
        with patch("voice_agent.tools.crm_knowledge.SETTINGS") as mock_settings:
            mock_settings.rag2_enabled = True
            mock_settings.rag2_graph_enabled = False
            
            with patch("voice_agent.tools.crm_knowledge._search_knowledge_base_rag2") as mock_rag2:
                mock_rag2.return_value = {
                    "success": True,
                    "query": "test",
                    "results": [],
                    "search_type": "rag2_triple_hybrid",
                }
                
                from voice_agent.tools.crm_knowledge import search_knowledge_base
                result = search_knowledge_base("test query")
                
                mock_rag2.assert_called_once()
                assert result["search_type"] == "rag2_triple_hybrid"
    
    def test_rag2_disabled_uses_hybrid(self):
        """When RAG2_ENABLED=false, should use legacy hybrid search."""
        with patch("voice_agent.tools.crm_knowledge.SETTINGS") as mock_settings:
            mock_settings.rag2_enabled = False
            mock_settings.rag_use_hybrid_bm25 = True
            
            with patch("voice_agent.tools.crm_knowledge._search_knowledge_base_hybrid") as mock_hybrid:
                mock_hybrid.return_value = {
                    "success": True,
                    "query": "test",
                    "results": [],
                    "search_type": "hybrid",
                }
                
                from voice_agent.tools.crm_knowledge import search_knowledge_base
                result = search_knowledge_base("test query")
                
                mock_hybrid.assert_called_once()
                assert result["search_type"] == "hybrid"
    
    def test_rag2_disabled_hybrid_disabled_uses_legacy(self):
        """When both RAG2 and hybrid disabled, should use legacy search."""
        with patch("voice_agent.tools.crm_knowledge.SETTINGS") as mock_settings:
            mock_settings.rag2_enabled = False
            mock_settings.rag_use_hybrid_bm25 = False
            
            with patch("voice_agent.tools.crm_knowledge._search_knowledge_base_legacy") as mock_legacy:
                mock_legacy.return_value = {
                    "success": True,
                    "query": "test",
                    "results": [],
                }
                
                from voice_agent.tools.crm_knowledge import search_knowledge_base
                result = search_knowledge_base("test query", use_hybrid=False)
                
                mock_legacy.assert_called_once()


class TestRAG2SearchFunction:
    """Test the _search_knowledge_base_rag2 function directly."""
    
    def test_rag2_search_returns_correct_format(self):
        """RAG2 search should return results in expected tool format."""
        from voice_agent.rag2.retrieval import RetrievalCandidate, RetrievalResult
        
        # Mock RetrievalResult
        mock_result = RetrievalResult(
            success=True,
            contexts=[
                RetrievalCandidate(
                    child_id="child-123",
                    parent_id="parent-456",
                    document_id="doc-789",
                    text="Child chunk text",
                    page=1,
                    modality="text",
                    lexical_rank=1,
                    semantic_rank=2,
                    graph_rank=None,
                    rrf_score=0.85,
                    parent_text="Expanded parent text with more context",
                    section_heading="Section Title",
                    rerank_score=0.92,
                ),
            ],
            max_rerank_score=0.92,
            refused=False,
            timings={"planning": 0.1, "retrieval": 0.2, "fusion": 0.05},
        )
        
        # Create mock retriever
        mock_retriever = MagicMock()
        mock_retriever.retrieve = AsyncMock(return_value=mock_result)
        
        # Mock the RAG2Retriever class
        mock_retriever_class = MagicMock(return_value=mock_retriever)
        
        with patch("voice_agent.tools.crm_knowledge.get_supabase_client") as mock_supabase:
            mock_client = MagicMock()
            mock_client.table.return_value.select.return_value.limit.return_value.execute.return_value = MagicMock(
                data=[{"org_id": "org-123"}]
            )
            mock_supabase.return_value = mock_client
            
            with patch.dict(sys.modules, {"voice_agent.rag2.retrieval": MagicMock(RAG2Retriever=mock_retriever_class)}):
                with patch("voice_agent.tools.crm_knowledge.SETTINGS") as mock_settings:
                    mock_settings.rag2_graph_enabled = False
                    
                    # Need to reload the module to pick up the patched import
                    import importlib
                    import voice_agent.tools.crm_knowledge as crm_module
                    importlib.reload(crm_module)
                    
                    result = crm_module._search_knowledge_base_rag2("What is the policy?", limit=5)
        
        assert result["success"] is True
        assert result["search_type"] == "rag2_triple_hybrid"
        assert result["result_count"] == 1
        assert "results" in result
        
        # Check result format
        first_result = result["results"][0]
        assert first_result["chunk_id"] == "child-123"
        assert first_result["parent_id"] == "parent-456"
        assert first_result["content"] == "Expanded parent text with more context"
        assert first_result["relevance_rank"] == 1
    
    def test_rag2_search_handles_refusal(self):
        """RAG2 search should handle refusal (safety threshold not met)."""
        from voice_agent.rag2.retrieval import RetrievalResult
        
        mock_result = RetrievalResult(
            success=True,
            contexts=[],
            max_rerank_score=0.3,  # Below threshold
            refused=True,
            refusal_reason="Max rerank score 0.30 below safety threshold 0.60",
            timings={},
        )
        
        mock_retriever = MagicMock()
        mock_retriever.retrieve = AsyncMock(return_value=mock_result)
        mock_retriever_class = MagicMock(return_value=mock_retriever)
        
        with patch("voice_agent.tools.crm_knowledge.get_supabase_client") as mock_supabase:
            mock_client = MagicMock()
            mock_client.table.return_value.select.return_value.limit.return_value.execute.return_value = MagicMock(
                data=[{"org_id": "org-123"}]
            )
            mock_supabase.return_value = mock_client
            
            with patch.dict(sys.modules, {"voice_agent.rag2.retrieval": MagicMock(RAG2Retriever=mock_retriever_class)}):
                with patch("voice_agent.tools.crm_knowledge.SETTINGS") as mock_settings:
                    mock_settings.rag2_graph_enabled = False
                    
                    import importlib
                    import voice_agent.tools.crm_knowledge as crm_module
                    importlib.reload(crm_module)
                    
                    result = crm_module._search_knowledge_base_rag2("Unknown query", limit=5)
        
        assert result["success"] is True
        assert result["refused"] is True
        assert "safety threshold" in result["refusal_reason"]
        assert result["results"] == []


class TestRAG2ToolFallback:
    """Test fallback behavior when RAG2 has issues."""
    
    def test_rag2_error_falls_back_to_legacy(self):
        """When RAG2 raises an error, should fall back to legacy search."""
        with patch("voice_agent.tools.crm_knowledge.SETTINGS") as mock_settings:
            mock_settings.rag2_enabled = True
            
            with patch("voice_agent.tools.crm_knowledge._search_knowledge_base_rag2") as mock_rag2:
                mock_rag2.side_effect = Exception("RAG2 connection failed")
                
                with patch("voice_agent.tools.crm_knowledge._search_knowledge_base_legacy") as mock_legacy:
                    mock_legacy.return_value = {
                        "success": True,
                        "query": "test",
                        "results": [{"content": "fallback result"}],
                    }
                    
                    from voice_agent.tools.crm_knowledge import search_knowledge_base
                    result = search_knowledge_base("test query")
                    
                    # Should have fallen back to legacy
                    mock_legacy.assert_called_once()
                    assert result["success"] is True
    
    def test_rag2_no_org_falls_back_to_hybrid(self):
        """When no RAG2 documents exist, should fall back to hybrid search."""
        with patch("voice_agent.tools.crm_knowledge.get_supabase_client") as mock_supabase:
            mock_client = MagicMock()
            # No rag_documents, then no orgs
            mock_client.table.return_value.select.return_value.limit.return_value.execute.side_effect = [
                MagicMock(data=[]),  # First call: rag_documents
                MagicMock(data=[]),  # Second call: organizations
            ]
            mock_supabase.return_value = mock_client
            
            with patch("voice_agent.tools.crm_knowledge._search_knowledge_base_hybrid") as mock_hybrid:
                mock_hybrid.return_value = {
                    "success": True,
                    "search_type": "hybrid",
                    "results": [],
                }
                
                with patch("voice_agent.tools.crm_knowledge.SETTINGS") as mock_settings:
                    mock_settings.rag2_graph_enabled = False
                    
                    from voice_agent.tools.crm_knowledge import _search_knowledge_base_rag2
                    result = _search_knowledge_base_rag2("test", limit=5)
        
        # Should have fallen back to hybrid
        mock_hybrid.assert_called_once()
        assert result["search_type"] == "hybrid"


class TestRAG2ResponseMapping:
    """Test that RAG2 response is correctly mapped to tool format."""
    
    def test_uses_parent_text_when_available(self):
        """Should prefer parent_text over child text."""
        from voice_agent.rag2.retrieval import RetrievalCandidate, RetrievalResult
        
        mock_result = RetrievalResult(
            success=True,
            contexts=[
                RetrievalCandidate(
                    child_id="c1",
                    parent_id="p1",
                    document_id="d1",
                    text="Short child text",
                    page=1,
                    modality="text",
                    parent_text="Much longer parent text with full context",
                    rerank_score=0.9,
                ),
            ],
            max_rerank_score=0.9,
            refused=False,
            timings={},
        )
        
        mock_retriever = MagicMock()
        mock_retriever.retrieve = AsyncMock(return_value=mock_result)
        mock_retriever_class = MagicMock(return_value=mock_retriever)
        
        with patch("voice_agent.tools.crm_knowledge.get_supabase_client") as mock_supabase:
            mock_client = MagicMock()
            mock_client.table.return_value.select.return_value.limit.return_value.execute.return_value = MagicMock(
                data=[{"org_id": "org-123"}]
            )
            mock_supabase.return_value = mock_client
            
            with patch.dict(sys.modules, {"voice_agent.rag2.retrieval": MagicMock(RAG2Retriever=mock_retriever_class)}):
                with patch("voice_agent.tools.crm_knowledge.SETTINGS") as mock_settings:
                    mock_settings.rag2_graph_enabled = False
                    
                    import importlib
                    import voice_agent.tools.crm_knowledge as crm_module
                    importlib.reload(crm_module)
                    
                    result = crm_module._search_knowledge_base_rag2("test", limit=5)
        
        # Should use parent_text
        assert result["results"][0]["content"] == "Much longer parent text with full context"
    
    def test_uses_child_text_when_no_parent(self):
        """Should fall back to child text if parent_text is None."""
        from voice_agent.rag2.retrieval import RetrievalCandidate, RetrievalResult
        
        mock_result = RetrievalResult(
            success=True,
            contexts=[
                RetrievalCandidate(
                    child_id="c1",
                    parent_id="p1",
                    document_id="d1",
                    text="Child text only",
                    page=1,
                    modality="text",
                    parent_text=None,  # No parent text
                    rerank_score=0.9,
                ),
            ],
            max_rerank_score=0.9,
            refused=False,
            timings={},
        )
        
        mock_retriever = MagicMock()
        mock_retriever.retrieve = AsyncMock(return_value=mock_result)
        mock_retriever_class = MagicMock(return_value=mock_retriever)
        
        with patch("voice_agent.tools.crm_knowledge.get_supabase_client") as mock_supabase:
            mock_client = MagicMock()
            mock_client.table.return_value.select.return_value.limit.return_value.execute.return_value = MagicMock(
                data=[{"org_id": "org-123"}]
            )
            mock_supabase.return_value = mock_client
            
            with patch.dict(sys.modules, {"voice_agent.rag2.retrieval": MagicMock(RAG2Retriever=mock_retriever_class)}):
                with patch("voice_agent.tools.crm_knowledge.SETTINGS") as mock_settings:
                    mock_settings.rag2_graph_enabled = False
                    
                    import importlib
                    import voice_agent.tools.crm_knowledge as crm_module
                    importlib.reload(crm_module)
                    
                    result = crm_module._search_knowledge_base_rag2("test", limit=5)
        
        # Should use child text
        assert result["results"][0]["content"] == "Child text only"
    
    def test_table_modality_sets_is_table_true(self):
        """When modality is 'table', is_table should be True."""
        from voice_agent.rag2.retrieval import RetrievalCandidate, RetrievalResult
        
        mock_result = RetrievalResult(
            success=True,
            contexts=[
                RetrievalCandidate(
                    child_id="c1",
                    parent_id="p1",
                    document_id="d1",
                    text="| Col1 | Col2 |",
                    page=1,
                    modality="table",
                    rerank_score=0.9,
                ),
            ],
            max_rerank_score=0.9,
            refused=False,
            timings={},
        )
        
        mock_retriever = MagicMock()
        mock_retriever.retrieve = AsyncMock(return_value=mock_result)
        mock_retriever_class = MagicMock(return_value=mock_retriever)
        
        with patch("voice_agent.tools.crm_knowledge.get_supabase_client") as mock_supabase:
            mock_client = MagicMock()
            mock_client.table.return_value.select.return_value.limit.return_value.execute.return_value = MagicMock(
                data=[{"org_id": "org-123"}]
            )
            mock_supabase.return_value = mock_client
            
            with patch.dict(sys.modules, {"voice_agent.rag2.retrieval": MagicMock(RAG2Retriever=mock_retriever_class)}):
                with patch("voice_agent.tools.crm_knowledge.SETTINGS") as mock_settings:
                    mock_settings.rag2_graph_enabled = False
                    
                    import importlib
                    import voice_agent.tools.crm_knowledge as crm_module
                    importlib.reload(crm_module)
                    
                    result = crm_module._search_knowledge_base_rag2("test", limit=5)
        
        assert result["results"][0]["is_table"] is True
        assert result["results"][0]["modality"] == "table"
