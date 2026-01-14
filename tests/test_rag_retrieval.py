"""
RAG Retrieval Tests

Unit tests for the RAG retrieval pipeline:
- Hybrid search (BM25 + Vector)
- RRF fusion
- Reranking
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from voice_agent.retrieval.hybrid_search import (
    HybridSearcher,
    SearchResult,
    SearchConfig,
)
from voice_agent.retrieval.reranker import (
    Reranker,
    RerankResult,
    LightweightReranker,
)


# =============================================================================
# Search Config Tests
# =============================================================================

class TestSearchConfig:
    """Tests for search configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = SearchConfig()
        
        assert config.use_hybrid is True
        assert config.use_vector is True
        assert config.use_bm25 is True
        assert config.top_k_retrieve == 50
        assert config.rrf_k == 60
    
    def test_config_from_settings(self):
        """Test configuration from settings."""
        with patch('voice_agent.retrieval.hybrid_search.SETTINGS') as mock_settings:
            mock_settings.rag_use_hybrid_bm25 = True
            mock_settings.rag_top_k_retrieve = 100
            mock_settings.rag_top_k_image = 5
            mock_settings.rag_fts_language = "portuguese"
            mock_settings.rag_enable_image_embeddings = False
            
            config = SearchConfig.from_settings()
            
            assert config.use_hybrid is True
            assert config.top_k_retrieve == 100


# =============================================================================
# Search Result Tests
# =============================================================================

class TestSearchResult:
    """Tests for search result dataclass."""
    
    def test_search_result_creation(self):
        """Test creating a search result."""
        result = SearchResult(
            chunk_id="test-id",
            content="Test content",
            modality="text",
            source_document="test.pdf",
            page=1,
            chunk_index=0,
            similarity_score=0.95,
        )
        
        assert result.chunk_id == "test-id"
        assert result.similarity_score == 0.95
        assert result.is_table is False
    
    def test_search_result_with_table(self):
        """Test search result for table chunk."""
        result = SearchResult(
            chunk_id="test-id",
            content="| Header | Data |",
            modality="table",
            source_document="report.pdf",
            page=5,
            chunk_index=2,
            is_table=True,
            table_context="Sales data by quarter",
        )
        
        assert result.is_table is True
        assert result.table_context == "Sales data by quarter"


# =============================================================================
# RRF Fusion Tests
# =============================================================================

class TestRRFFusion:
    """Tests for Reciprocal Rank Fusion."""
    
    def test_rrf_single_list(self):
        """Test RRF with single result list."""
        searcher = HybridSearcher(org_id="test-org")
        
        results = [
            SearchResult(
                chunk_id="1", content="A", modality="text",
                source_document="test.txt", page=1, chunk_index=0,
                similarity_score=0.9,
            ),
            SearchResult(
                chunk_id="2", content="B", modality="text",
                source_document="test.txt", page=1, chunk_index=1,
                similarity_score=0.8,
            ),
        ]
        
        fused = searcher._rrf_fusion([results])
        
        assert len(fused) == 2
        # First result should have higher RRF score
        assert fused[0].chunk_id == "1"
        assert fused[0].rrf_score > fused[1].rrf_score
    
    def test_rrf_multiple_lists(self):
        """Test RRF with multiple result lists."""
        searcher = HybridSearcher(org_id="test-org")
        
        # Vector results
        vector_results = [
            SearchResult(
                chunk_id="1", content="A", modality="text",
                source_document="test.txt", page=1, chunk_index=0,
                similarity_score=0.9,
            ),
            SearchResult(
                chunk_id="2", content="B", modality="text",
                source_document="test.txt", page=1, chunk_index=1,
                similarity_score=0.7,
            ),
        ]
        
        # BM25 results (different order)
        bm25_results = [
            SearchResult(
                chunk_id="2", content="B", modality="text",
                source_document="test.txt", page=1, chunk_index=1,
                bm25_score=5.0,
            ),
            SearchResult(
                chunk_id="1", content="A", modality="text",
                source_document="test.txt", page=1, chunk_index=0,
                bm25_score=3.0,
            ),
        ]
        
        fused = searcher._rrf_fusion([vector_results, bm25_results])
        
        assert len(fused) == 2
        # Both should appear with RRF scores
        assert all(r.rrf_score > 0 for r in fused)
        assert all(r.retrieval_method == "hybrid" for r in fused)
    
    def test_rrf_document_appears_once(self):
        """Test that same document appears once in fused results."""
        searcher = HybridSearcher(org_id="test-org")
        
        result1 = SearchResult(
            chunk_id="same-id", content="Same", modality="text",
            source_document="test.txt", page=1, chunk_index=0,
            similarity_score=0.9,
        )
        result2 = SearchResult(
            chunk_id="same-id", content="Same", modality="text",
            source_document="test.txt", page=1, chunk_index=0,
            bm25_score=5.0,
        )
        
        fused = searcher._rrf_fusion([[result1], [result2]])
        
        assert len(fused) == 1
        assert fused[0].chunk_id == "same-id"
        # RRF score should be sum from both lists
        assert fused[0].rrf_score > 0


# =============================================================================
# Hybrid Searcher Tests
# =============================================================================

class TestHybridSearcher:
    """Tests for hybrid search."""
    
    def test_searcher_initialization(self):
        """Test searcher initialization."""
        config = SearchConfig(top_k_retrieve=100, rrf_k=30)
        searcher = HybridSearcher(org_id="test-org", config=config)
        
        assert searcher.org_id == "test-org"
        assert searcher.config.top_k_retrieve == 100
        assert searcher.config.rrf_k == 30
    
    def test_apply_filters_category(self):
        """Test category filter application."""
        searcher = HybridSearcher(org_id="test-org")
        
        results = [
            SearchResult(
                chunk_id="1", content="A", modality="text",
                source_document="test.txt", page=1, chunk_index=0,
                category="pricing",
            ),
            SearchResult(
                chunk_id="2", content="B", modality="text",
                source_document="test.txt", page=1, chunk_index=1,
                category="faq",
            ),
        ]
        
        filtered = searcher._apply_filters(results, category="pricing")
        
        assert len(filtered) == 1
        assert filtered[0].category == "pricing"
    
    def test_apply_filters_source(self):
        """Test source document filter."""
        searcher = HybridSearcher(org_id="test-org")
        
        results = [
            SearchResult(
                chunk_id="1", content="A", modality="text",
                source_document="doc1.pdf", page=1, chunk_index=0,
            ),
            SearchResult(
                chunk_id="2", content="B", modality="text",
                source_document="doc2.pdf", page=1, chunk_index=1,
            ),
        ]
        
        filtered = searcher._apply_filters(results, source_document="doc1.pdf")
        
        assert len(filtered) == 1
        assert filtered[0].source_document == "doc1.pdf"
    
    @pytest.mark.asyncio
    async def test_search_with_mocks(self):
        """Test search with mocked dependencies."""
        searcher = HybridSearcher(org_id="test-org")
        
        # Mock embedder
        mock_embedding = [0.1] * 1536
        with patch.object(searcher.embedder, 'embed_query') as mock_embed:
            mock_embed.return_value = (mock_embedding, None)
            
            # Mock vector search
            with patch.object(searcher, '_vector_search') as mock_vector:
                mock_vector.return_value = [
                    SearchResult(
                        chunk_id="v1", content="Vector result", modality="text",
                        source_document="test.txt", page=1, chunk_index=0,
                        similarity_score=0.9,
                    )
                ]
                
                # Mock BM25 search
                with patch.object(searcher, '_bm25_search') as mock_bm25:
                    mock_bm25.return_value = [
                        SearchResult(
                            chunk_id="b1", content="BM25 result", modality="text",
                            source_document="test.txt", page=1, chunk_index=1,
                            bm25_score=5.0,
                        )
                    ]
                    
                    results = await searcher.search("test query", top_k=5)
                    
                    assert len(results) <= 5
                    mock_embed.assert_called_once()
                    mock_vector.assert_called_once()
                    mock_bm25.assert_called_once()


# =============================================================================
# Reranker Tests
# =============================================================================

class TestReranker:
    """Tests for cross-encoder reranking."""
    
    def test_reranker_initialization(self):
        """Test reranker initialization."""
        reranker = Reranker(
            model_name="test-model",
            top_k=10,
            enabled=True,
        )
        
        assert reranker.model_name == "test-model"
        assert reranker.top_k == 10
        assert reranker.enabled is True
    
    def test_prepare_document_simple(self):
        """Test document preparation for reranking."""
        reranker = Reranker()
        
        result = SearchResult(
            chunk_id="1", content="Main content here", modality="text",
            source_document="test.txt", page=1, chunk_index=0,
        )
        
        prepared = reranker._prepare_document(result)
        
        # _prepare_document now returns (doc_text, image_base64) tuple
        doc_text, image_b64 = prepared
        assert "Main content here" in doc_text
    
    def test_prepare_document_with_context(self):
        """Test document preparation with title and table context."""
        reranker = Reranker()
        
        result = SearchResult(
            chunk_id="1", content="Table data", modality="table",
            source_document="report.pdf", page=1, chunk_index=0,
            title="Sales Report",
            is_table=True,
            table_context="Quarterly sales by region",
        )
        
        prepared = reranker._prepare_document(result)
        
        # _prepare_document now returns (doc_text, image_base64) tuple
        doc_text, image_b64 = prepared
        assert "Sales Report" in doc_text
        assert "Quarterly sales" in doc_text
        assert "Table data" in doc_text
    
    @pytest.mark.asyncio
    async def test_rerank_disabled(self):
        """Test reranking when disabled."""
        reranker = Reranker(enabled=False, top_k=3)
        
        results = [
            SearchResult(
                chunk_id=str(i), content=f"Content {i}", modality="text",
                source_document="test.txt", page=1, chunk_index=i,
            )
            for i in range(10)
        ]
        
        reranked = await reranker.rerank("test query", results)
        
        # Should just return top-k without reranking
        assert len(reranked) == 3
        assert reranked[0].chunk_id == "0"  # Order preserved
    
    @pytest.mark.asyncio
    async def test_rerank_empty_results(self):
        """Test reranking empty results."""
        reranker = Reranker(enabled=True)
        
        reranked = await reranker.rerank("test query", [])
        
        assert reranked == []


# =============================================================================
# Lightweight Reranker Tests
# =============================================================================

class TestLightweightReranker:
    """Tests for lightweight reranker fallback."""
    
    @pytest.mark.asyncio
    async def test_term_overlap_scoring(self):
        """Test scoring based on term overlap."""
        reranker = LightweightReranker(top_k=3)
        
        results = [
            SearchResult(
                chunk_id="1", content="Python programming language",
                modality="text", source_document="test.txt", page=1, chunk_index=0,
                rrf_score=0.5,
            ),
            SearchResult(
                chunk_id="2", content="Python is great for data science",
                modality="text", source_document="test.txt", page=1, chunk_index=1,
                rrf_score=0.5,
            ),
            SearchResult(
                chunk_id="3", content="Java programming guide",
                modality="text", source_document="test.txt", page=1, chunk_index=2,
                rrf_score=0.5,
            ),
        ]
        
        reranked = await reranker.rerank("Python programming", results)
        
        # Results with "Python" and "programming" should score higher
        assert len(reranked) == 3
        assert all(r.rerank_score is not None for r in reranked)
    
    @pytest.mark.asyncio
    async def test_table_boost(self):
        """Test boost for table results with data-related queries."""
        reranker = LightweightReranker(top_k=5)
        
        results = [
            SearchResult(
                chunk_id="1", content="Some regular text about data",
                modality="text", source_document="test.txt", page=1, chunk_index=0,
                rrf_score=0.5, similarity_score=0.5,
            ),
            SearchResult(
                chunk_id="2", content="| Value | Count |",
                modality="table", source_document="test.txt", page=1, chunk_index=1,
                rrf_score=0.5, similarity_score=0.5,
                is_table=True,
            ),
        ]
        
        reranked = await reranker.rerank("show me the data table", results)
        
        # Table should be boosted for data-related query
        table_result = next(r for r in reranked if r.is_table)
        text_result = next(r for r in reranked if not r.is_table)
        
        assert table_result.rerank_score is not None
        assert text_result.rerank_score is not None
        assert table_result.rerank_score > text_result.rerank_score


# =============================================================================
# Integration Tests
# =============================================================================

class TestRetrievalIntegration:
    """Integration tests for retrieval pipeline."""
    
    @pytest.mark.asyncio
    async def test_search_and_rerank_integration(self):
        """Test search followed by reranking."""
        searcher = HybridSearcher(org_id="test-org")
        reranker = LightweightReranker(top_k=3)
        
        # Create mock search results
        search_results = [
            SearchResult(
                chunk_id=str(i), content=f"Result content {i}",
                modality="text", source_document="test.txt",
                page=1, chunk_index=i,
                rrf_score=0.5 - (i * 0.1),
                similarity_score=0.5 - (i * 0.1),
            )
            for i in range(5)
        ]
        
        # Rerank
        reranked = await reranker.rerank("content query", search_results)
        
        assert len(reranked) == 3
        assert all(r.rerank_score is not None for r in reranked)
    
    def test_provenance_preservation(self):
        """Test that provenance information is preserved through pipeline."""
        result = SearchResult(
            chunk_id="test-id",
            content="Test content",
            modality="text",
            source_document="important_doc.pdf",
            page=42,
            chunk_index=7,
            ocr_confidence=0.95,
            is_table=False,
            table_context=None,
            alt_text="Image description",
            category="technical",
            title="Chapter 3",
        )
        
        # Check all provenance fields are accessible
        assert result.source_document == "important_doc.pdf"
        assert result.page == 42
        assert result.chunk_index == 7
        assert result.ocr_confidence == 0.95
        assert result.category == "technical"
        assert result.title == "Chapter 3"
