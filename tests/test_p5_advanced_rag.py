"""
Tests for P5 Advanced RAG Components

Tests for:
- Self-RAG (self-reflective retrieval)
- Hierarchical Retrieval (Parent Doc, Sentence Window, Auto-Merge)
- Corrective RAG
"""

import pytest
from dataclasses import dataclass
from typing import List, Any

from triple_hybrid_rag.retrieval import (
    # Self-RAG
    SelfRAG,
    SelfRAGConfig,
    AdaptiveRAG,
    RetrievalDecision,
    SupportLevel,
    RelevanceLevel,
    PassageEvaluation,
    GenerationResult,
    # Hierarchical
    ParentDocumentRetriever,
    ParentDocumentConfig,
    SentenceWindowRetriever,
    SentenceWindowConfig,
    AutoMergingRetriever,
    DocumentNode,
    HierarchicalChunk,
    # Corrective RAG
    CorrectiveRAG,
    CRAGConfig,
    CRAGResult,
    RetrievalQuality,
    CorrectionAction,
    RetrievalAssessment,
    KnowledgeRefiner,
)

# ============================================================================
# Mock Components
# ============================================================================

@dataclass
class MockDocument:
    """Mock document for testing."""
    id: str
    text: str
    score: float = 0.8

def create_mock_documents(count: int) -> List[MockDocument]:
    """Create mock documents."""
    return [
        MockDocument(
            id=f"doc_{i}",
            text=f"This is document {i} with relevant information about the topic.",
            score=0.9 - (i * 0.1),
        )
        for i in range(count)
    ]

async def mock_retrieve(query: str, top_k: int = 10) -> List[MockDocument]:
    """Mock retrieval function."""
    return create_mock_documents(min(top_k, 5))

async def mock_generate(prompt: str) -> str:
    """Mock LLM generation function."""
    if "YES" in prompt.upper() or "RETRIEVE" in prompt.upper():
        return "YES"
    if "relevance" in prompt.lower() and "0-10" in prompt:
        return "8"
    if "FULLY" in prompt.upper():
        return "FULLY_SUPPORTED"
    if "refine" in prompt.lower():
        return "refined query about the topic"
    return "This is a generated response based on the context provided."

# ============================================================================
# Self-RAG Tests
# ============================================================================

class TestSelfRAGConfig:
    """Test Self-RAG configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = SelfRAGConfig()
        
        assert config.retrieval_threshold == 0.6
        assert config.max_retrieval_rounds == 3
        assert config.min_relevance_score == 0.5
        assert config.enable_critique is True
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = SelfRAGConfig(
            max_retrieval_rounds=5,
            enable_critique=False,
        )
        
        assert config.max_retrieval_rounds == 5
        assert config.enable_critique is False

class TestSelfRAG:
    """Test Self-RAG component."""
    
    @pytest.fixture
    def self_rag(self):
        """Create Self-RAG instance."""
        return SelfRAG(
            retrieve_fn=mock_retrieve,
            generate_fn=mock_generate,
        )
    
    @pytest.mark.asyncio
    async def test_generate_with_reflection(self, self_rag):
        """Test self-reflective generation."""
        result = await self_rag.generate_with_reflection("What is Python?")
        
        assert isinstance(result, GenerationResult)
        assert result.query == "What is Python?"
        assert result.response != ""
    
    @pytest.mark.asyncio
    async def test_result_structure(self, self_rag):
        """Test result has expected fields."""
        result = await self_rag.generate_with_reflection("test query")
        
        assert hasattr(result, 'query')
        assert hasattr(result, 'response')
        assert hasattr(result, 'retrieved_passages')
        assert hasattr(result, 'retrieval_rounds')
        assert hasattr(result, 'critique_score')
    
    def test_retrieval_decision_enum(self):
        """Test RetrievalDecision enum."""
        assert RetrievalDecision.YES.value == "yes"
        assert RetrievalDecision.NO.value == "no"
        assert RetrievalDecision.CONTINUE.value == "continue"
    
    def test_support_level_enum(self):
        """Test SupportLevel enum."""
        assert SupportLevel.FULLY_SUPPORTED.value == "fully_supported"
        assert SupportLevel.PARTIALLY_SUPPORTED.value == "partially_supported"
        assert SupportLevel.NO_SUPPORT.value == "no_support"

class TestAdaptiveRAG:
    """Test Adaptive RAG component."""
    
    @pytest.fixture
    def adaptive_rag(self):
        """Create Adaptive RAG instance."""
        return AdaptiveRAG(
            retrieve_fn=mock_retrieve,
            generate_fn=mock_generate,
        )
    
    def test_classify_factual_query(self, adaptive_rag):
        """Test factual query classification."""
        category = adaptive_rag.classify_query("What is Python?")
        assert category == "factual"
    
    def test_classify_analytical_query(self, adaptive_rag):
        """Test analytical query classification."""
        category = adaptive_rag.classify_query("Why does Python use indentation?")
        assert category == "analytical"
    
    def test_classify_creative_query(self, adaptive_rag):
        """Test creative query classification."""
        category = adaptive_rag.classify_query("Write a poem about coding")
        assert category == "creative"
    
    def test_should_retrieve(self, adaptive_rag):
        """Test retrieval decision."""
        should, benefit = adaptive_rag.should_retrieve("What is Python?")
        assert should is True  # Factual queries benefit from retrieval
        assert benefit >= 0.5
    
    def test_update_performance(self, adaptive_rag):
        """Test performance tracking update."""
        initial = adaptive_rag.category_performance['factual']['retrieval_benefit']
        
        # Good quality with retrieval should increase benefit
        adaptive_rag.update_performance("What is X?", used_retrieval=True, quality_score=0.9)
        
        updated = adaptive_rag.category_performance['factual']['retrieval_benefit']
        assert updated >= initial

# ============================================================================
# Hierarchical Retrieval Tests
# ============================================================================

class TestParentDocumentRetriever:
    """Test Parent Document Retriever."""
    
    @pytest.fixture
    def parent_retriever(self):
        """Create parent document retriever."""
        return ParentDocumentRetriever(
            config=ParentDocumentConfig(
                child_chunk_size=100,
                parent_chunk_size=500,
            )
        )
    
    def test_config_defaults(self):
        """Test default configuration."""
        config = ParentDocumentConfig()
        
        assert config.child_chunk_size == 200
        assert config.parent_chunk_size == 1000
        assert config.child_top_k == 20
        assert config.final_top_k == 5
    
    def test_add_documents(self, parent_retriever):
        """Test adding documents."""
        documents = [
            "This is document one with multiple sentences. It has quite a lot of content. " * 10,
            "This is document two with different content. Also quite lengthy. " * 10,
        ]
        
        parent_retriever.add_documents(documents)
        
        assert len(parent_retriever.parent_chunks) > 0
        assert len(parent_retriever.child_chunks) > 0
        assert len(parent_retriever.child_to_parent) > 0
    
    def test_chunk_hierarchy(self, parent_retriever):
        """Test parent-child relationships."""
        documents = ["Content " * 100]
        parent_retriever.add_documents(documents)
        
        # Each child should have a parent
        for child_id, parent_id in parent_retriever.child_to_parent.items():
            assert parent_id in parent_retriever.parent_chunks
    
    def test_retrieve_without_search_fn(self, parent_retriever):
        """Test retrieval without search function returns empty."""
        parent_retriever.add_documents(["Test content " * 50])
        results = parent_retriever.retrieve("test")
        assert results == []

class TestSentenceWindowRetriever:
    """Test Sentence Window Retriever."""
    
    @pytest.fixture
    def sentence_retriever(self):
        """Create sentence window retriever."""
        return SentenceWindowRetriever(
            config=SentenceWindowConfig(window_size=2)
        )
    
    def test_config_defaults(self):
        """Test default configuration."""
        config = SentenceWindowConfig()
        
        assert config.window_size == 3
        assert config.min_sentence_length == 10
        assert config.merge_adjacent is True
    
    def test_add_documents(self, sentence_retriever):
        """Test adding documents."""
        documents = [
            "First sentence here. Second sentence follows. Third comes after. Fourth is last.",
            "Another document. With multiple sentences. Each one indexed.",
        ]
        
        sentence_retriever.add_documents(documents)
        
        assert len(sentence_retriever.sentences) > 0
        assert len(sentence_retriever.doc_sentences) == 2
    
    def test_sentence_metadata(self, sentence_retriever):
        """Test sentence metadata."""
        documents = ["One. Two. Three. Four. Five."]
        sentence_retriever.add_documents(documents)
        
        # Check metadata
        for sent_id, sent in sentence_retriever.sentences.items():
            assert 'doc_id' in sent.metadata
            assert 'sentence_idx' in sent.metadata
            assert 'total_sentences' in sent.metadata

class TestAutoMergingRetriever:
    """Test Auto-Merging Retriever."""
    
    @pytest.fixture
    def auto_retriever(self):
        """Create auto-merging retriever."""
        return AutoMergingRetriever(merge_threshold=0.5)
    
    def test_add_documents_hierarchy(self, auto_retriever):
        """Test hierarchical document structure."""
        documents = [
            """# Section 1
            
            First paragraph with content.
            Second paragraph here.
            
            # Section 2
            
            Another section with text.
            More content follows.
            """
        ]
        
        auto_retriever.add_documents(documents)
        
        # Should have nodes at different levels
        levels = {node.level for node in auto_retriever.nodes.values()}
        assert 0 in levels  # Document level
        assert 1 in levels  # Section level

class TestHierarchicalChunk:
    """Test HierarchicalChunk dataclass."""
    
    def test_chunk_creation(self):
        """Test creating hierarchical chunk."""
        chunk = HierarchicalChunk(
            chunk_id="test_1",
            text="Test content",
            parent_text="Parent content",
            level=2,
            score=0.9,
        )
        
        assert chunk.chunk_id == "test_1"
        assert chunk.text == "Test content"
        assert chunk.parent_text == "Parent content"
        assert chunk.level == 2
        assert chunk.score == 0.9
    
    def test_chunk_with_window(self):
        """Test chunk with window context."""
        chunk = HierarchicalChunk(
            chunk_id="sent_1",
            text="Target sentence.",
            window_text="Previous. Target sentence. Next.",
            level=3,
            score=0.85,
        )
        
        assert "Previous" in chunk.window_text
        assert "Next" in chunk.window_text

# ============================================================================
# Corrective RAG Tests
# ============================================================================

class TestCRAGConfig:
    """Test Corrective RAG configuration."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = CRAGConfig()
        
        assert config.correct_threshold == 0.7
        assert config.ambiguous_threshold == 0.4
        assert config.enable_query_refinement is True
        assert config.max_refinement_attempts == 2
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = CRAGConfig(
            correct_threshold=0.8,
            enable_web_search=True,
        )
        
        assert config.correct_threshold == 0.8
        assert config.enable_web_search is True

class TestRetrievalQuality:
    """Test retrieval quality enum."""
    
    def test_quality_values(self):
        """Test quality enum values."""
        assert RetrievalQuality.CORRECT.value == "correct"
        assert RetrievalQuality.INCORRECT.value == "incorrect"
        assert RetrievalQuality.AMBIGUOUS.value == "ambiguous"

class TestCorrectionAction:
    """Test correction action enum."""
    
    def test_action_values(self):
        """Test action enum values."""
        assert CorrectionAction.USE_RETRIEVED.value == "use_retrieved"
        assert CorrectionAction.WEB_SEARCH.value == "web_search"
        assert CorrectionAction.REFINE_QUERY.value == "refine_query"
        assert CorrectionAction.KNOWLEDGE_ONLY.value == "knowledge_only"
        assert CorrectionAction.HYBRID.value == "hybrid"

class TestCorrectiveRAG:
    """Test Corrective RAG component."""
    
    @pytest.fixture
    def crag(self):
        """Create Corrective RAG instance."""
        return CorrectiveRAG(
            retrieve_fn=mock_retrieve,
            generate_fn=mock_generate,
        )
    
    @pytest.mark.asyncio
    async def test_generate_with_correction(self, crag):
        """Test corrective generation."""
        result = await crag.generate_with_correction("What is Python?")
        
        assert isinstance(result, CRAGResult)
        assert result.query == "What is Python?"
        assert result.response != ""
    
    @pytest.mark.asyncio
    async def test_result_has_assessment(self, crag):
        """Test result includes assessment."""
        result = await crag.generate_with_correction("test query")
        
        assert hasattr(result, 'assessment')
        assert isinstance(result.assessment, RetrievalAssessment)
        assert result.assessment.quality in RetrievalQuality
    
    @pytest.mark.asyncio
    async def test_result_structure(self, crag):
        """Test result has expected fields."""
        result = await crag.generate_with_correction("test query")
        
        assert hasattr(result, 'query')
        assert hasattr(result, 'response')
        assert hasattr(result, 'assessment')
        assert hasattr(result, 'used_documents')
        assert hasattr(result, 'correction_applied')
    
    def test_get_doc_text(self, crag):
        """Test document text extraction."""
        doc = MockDocument(id="1", text="Test content")
        text = crag._get_doc_text(doc)
        assert text == "Test content"
        
        # Test dict
        text = crag._get_doc_text({"text": "Dict content"})
        assert text == "Dict content"
        
        # Test string
        text = crag._get_doc_text("Raw string")
        assert text == "Raw string"

class TestKnowledgeRefiner:
    """Test Knowledge Refiner."""
    
    @pytest.fixture
    def refiner(self):
        """Create knowledge refiner."""
        return KnowledgeRefiner(generate_fn=mock_generate)
    
    @pytest.mark.asyncio
    async def test_refine_documents(self, refiner):
        """Test document refinement."""
        documents = [
            MockDocument(id="1", text="Python is a programming language. It was created by Guido."),
            MockDocument(id="2", text="Python supports multiple paradigms."),
        ]
        
        facts = await refiner.refine("What is Python?", documents)
        
        assert isinstance(facts, list)
    
    @pytest.mark.asyncio
    async def test_decompose_text(self, refiner):
        """Test text decomposition."""
        text = "Python is a language. It was made by Guido. It uses indentation."
        
        facts = await refiner.decompose(text)
        
        assert isinstance(facts, list)

# ============================================================================
# Integration Tests
# ============================================================================

class TestP5Integration:
    """Integration tests for P5 components."""
    
    def test_all_imports(self):
        """Test all P5 components can be imported."""
        # Self-RAG imports
        assert SelfRAG is not None
        assert SelfRAGConfig is not None
        assert AdaptiveRAG is not None
        
        # Hierarchical imports
        assert ParentDocumentRetriever is not None
        assert SentenceWindowRetriever is not None
        assert AutoMergingRetriever is not None
        
        # Corrective RAG imports
        assert CorrectiveRAG is not None
        assert CRAGConfig is not None
        assert KnowledgeRefiner is not None
    
    def test_passage_evaluation_creation(self):
        """Test PassageEvaluation dataclass."""
        eval_obj = PassageEvaluation(
            passage_id="p1",
            text="Test passage",
            relevance=RelevanceLevel.RELEVANT,
            relevance_score=0.85,
        )
        
        assert eval_obj.passage_id == "p1"
        assert eval_obj.relevance == RelevanceLevel.RELEVANT
        assert eval_obj.relevance_score == 0.85
    
    def test_retrieval_assessment_creation(self):
        """Test RetrievalAssessment dataclass."""
        assessment = RetrievalAssessment(
            quality=RetrievalQuality.CORRECT,
            confidence=0.9,
            relevant_docs=[MockDocument("1", "test")],
            action=CorrectionAction.USE_RETRIEVED,
        )
        
        assert assessment.quality == RetrievalQuality.CORRECT
        assert assessment.confidence == 0.9
        assert len(assessment.relevant_docs) == 1
    
    def test_document_node_creation(self):
        """Test DocumentNode dataclass."""
        node = DocumentNode(
            id="node_1",
            text="Node content",
            level=1,
            parent_id="doc_0",
        )
        
        assert node.id == "node_1"
        assert node.level == 1
        assert node.parent_id == "doc_0"
