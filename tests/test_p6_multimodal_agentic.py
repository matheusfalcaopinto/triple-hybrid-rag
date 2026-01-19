"""
Tests for P6 Multimodal and Agentic RAG Components

Tests for:
- Multimodal Retrieval (text, images, tables, code)
- ColBERT-style Late Interaction
- Multi-Vector Retrieval
- Agentic RAG with tools
- Streaming RAG
- RAG Orchestrator
"""

import pytest
from dataclasses import dataclass
from typing import List, Any, Dict

from triple_hybrid_rag.retrieval import (
    # Multimodal
    MultimodalRetriever,
    MultimodalConfig,
    MultimodalContent,
    MultimodalResult,
    ModalityType,
    ColBERTRetriever,
    ColBERTConfig,
    MultiVectorRetriever,
    # Agentic RAG
    AgenticRAG,
    AgenticRAGConfig,
    AgenticRAGResult,
    Tool,
    ToolResult,
    ToolType,
    SearchTool,
    CalculateTool,
    SummarizeTool,
    AgentAction,
    AgentStep,
    StreamingRAG,
    RAGOrchestrator,
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
            text=f"Document {i} contains information about Python programming.",
            score=0.9 - (i * 0.1),
        )
        for i in range(count)
    ]

async def mock_retrieve(query: str, top_k: int = 10) -> List[MockDocument]:
    """Mock retrieval function."""
    return create_mock_documents(min(top_k, 5))

async def mock_generate(prompt: str) -> str:
    """Mock LLM generation."""
    if "Action:" in prompt:
        return """Thought: I need to search for information about Python.
Action: search
Action Input: Python programming"""
    return "This is a generated response based on the context."

def mock_embed(text: str) -> List[float]:
    """Mock embedding function."""
    return [0.1] * 384

def mock_search(query: str, top_k: int = 10) -> List[Dict]:
    """Mock search function."""
    return [{'id': f'doc_{i}', 'score': 0.9 - i * 0.1} for i in range(top_k)]

# ============================================================================
# Multimodal Tests
# ============================================================================

class TestModalityType:
    """Test ModalityType enum."""
    
    def test_modality_values(self):
        """Test modality enum values."""
        assert ModalityType.TEXT.value == "text"
        assert ModalityType.IMAGE.value == "image"
        assert ModalityType.TABLE.value == "table"
        assert ModalityType.CODE.value == "code"
        assert ModalityType.AUDIO.value == "audio"
        assert ModalityType.VIDEO.value == "video"

class TestMultimodalConfig:
    """Test Multimodal configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = MultimodalConfig()
        
        assert config.text_weight == 1.0
        assert config.image_weight == 0.8
        assert config.table_weight == 0.9
        assert config.cross_modal_fusion is True
        assert config.final_top_k == 10
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = MultimodalConfig(
            text_weight=0.9,
            image_weight=1.0,
            late_interaction=True,
        )
        
        assert config.text_weight == 0.9
        assert config.image_weight == 1.0
        assert config.late_interaction is True

class TestMultimodalContent:
    """Test MultimodalContent dataclass."""
    
    def test_text_content(self):
        """Test text content creation."""
        content = MultimodalContent(
            content_id="text_1",
            modality=ModalityType.TEXT,
            content="Sample text content",
            text_representation="Sample text content",
        )
        
        assert content.content_id == "text_1"
        assert content.modality == ModalityType.TEXT
        assert content.text_representation is not None
    
    def test_table_content(self):
        """Test table content creation."""
        table_data = {
            'headers': ['Name', 'Age', 'City'],
            'rows': [['Alice', 30, 'NYC'], ['Bob', 25, 'LA']],
        }
        
        content = MultimodalContent(
            content_id="table_1",
            modality=ModalityType.TABLE,
            content=table_data,
        )
        
        assert content.modality == ModalityType.TABLE
        assert 'headers' in content.content

class TestMultimodalRetriever:
    """Test Multimodal Retriever."""
    
    @pytest.fixture
    def retriever(self):
        """Create multimodal retriever."""
        return MultimodalRetriever(
            text_embed_fn=mock_embed,
            config=MultimodalConfig(cross_modal_fusion=True),
        )
    
    def test_add_text_content(self, retriever):
        """Test adding text content."""
        retriever.add_content(
            content="Python is a programming language.",
            modality=ModalityType.TEXT,
            content_id="doc_1",
        )
        
        assert "doc_1" in retriever.content_store
        assert len(retriever.modality_index[ModalityType.TEXT]) == 1
    
    def test_add_code_content(self, retriever):
        """Test adding code content."""
        code = "def hello(): print('Hello')"
        retriever.add_content(
            content=code,
            modality=ModalityType.CODE,
            content_id="code_1",
        )
        
        assert "code_1" in retriever.content_store
        assert retriever.content_store["code_1"].modality == ModalityType.CODE
    
    def test_add_table_content(self, retriever):
        """Test adding table content."""
        table = {'headers': ['A', 'B'], 'rows': [[1, 2], [3, 4]]}
        retriever.add_content(
            content=table,
            modality=ModalityType.TABLE,
            content_id="table_1",
        )
        
        content = retriever.content_store["table_1"]
        assert content.modality == ModalityType.TABLE
        assert content.text_representation is not None  # Table converted to text
    
    def test_retrieve_empty(self, retriever):
        """Test retrieval with no content."""
        results = retriever.retrieve("test query")
        assert results == []
    
    def test_retrieve_with_content(self, retriever):
        """Test retrieval with content."""
        retriever.add_content("Python is great", ModalityType.TEXT, "doc_1")
        retriever.add_content("Java is also good", ModalityType.TEXT, "doc_2")
        
        results = retriever.retrieve("Python", top_k=5)
        
        assert isinstance(results, list)
    
    def test_modality_weights(self, retriever):
        """Test modality weight retrieval."""
        assert retriever._get_modality_weight(ModalityType.TEXT) == 1.0
        assert retriever._get_modality_weight(ModalityType.IMAGE) == 0.8
        assert retriever._get_modality_weight(ModalityType.TABLE) == 0.9

# ============================================================================
# ColBERT Tests
# ============================================================================

class TestColBERTConfig:
    """Test ColBERT configuration."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = ColBERTConfig()
        
        assert config.max_query_tokens == 32
        assert config.max_doc_tokens == 180
        assert config.interaction_type == "maxsim"
        assert config.top_k == 10

class TestColBERTRetriever:
    """Test ColBERT-style retriever."""
    
    @pytest.fixture
    def retriever(self):
        """Create ColBERT retriever."""
        return ColBERTRetriever()
    
    def test_add_document(self, retriever):
        """Test adding documents."""
        retriever.add_document("doc_1", "Python is a programming language.")
        retriever.add_document("doc_2", "Java is another language.")
        
        assert len(retriever.documents) == 2
        assert "doc_1" in retriever.documents
    
    def test_document_tokenization(self, retriever):
        """Test document gets tokenized."""
        retriever.add_document("doc_1", "Hello world test")
        
        doc = retriever.documents["doc_1"]
        assert 'tokens' in doc
        assert len(doc['tokens']) > 0
    
    def test_retrieve(self, retriever):
        """Test retrieval."""
        retriever.add_document("doc_1", "Python programming language")
        retriever.add_document("doc_2", "Java programming language")
        retriever.add_document("doc_3", "Cooking recipes and food")
        
        results = retriever.retrieve("Python", top_k=2)
        
        assert len(results) == 2
        assert results[0]['doc_id'] == "doc_1"  # Python doc should rank first
    
    def test_token_overlap_scoring(self, retriever):
        """Test token overlap score."""
        score = retriever._token_overlap_score(
            ["python", "programming"],
            ["python", "is", "a", "programming", "language"],
        )
        
        assert score > 0
        assert score == 1.0  # Both query words in doc

# ============================================================================
# Multi-Vector Tests
# ============================================================================

class TestMultiVectorRetriever:
    """Test Multi-Vector retriever."""
    
    @pytest.fixture
    def retriever(self):
        """Create multi-vector retriever."""
        return MultiVectorRetriever(embed_fn=mock_embed)
    
    def test_add_document(self, retriever):
        """Test adding document."""
        retriever.add_document(
            doc_id="doc_1",
            text="Python is a programming language.",
            key_passages=["Python is versatile", "Used in AI"],
            entities=["Python", "programming"],
            summary="Overview of Python.",
        )
        
        assert "doc_1" in retriever.documents
        doc = retriever.documents["doc_1"]
        assert len(doc['vectors']) > 0  # Should have multiple vectors
    
    def test_multiple_vectors_created(self, retriever):
        """Test multiple vectors are created."""
        retriever.add_document(
            doc_id="doc_1",
            text="Main content",
            key_passages=["passage 1", "passage 2"],
            summary="Summary text",
        )
        
        doc = retriever.documents["doc_1"]
        vector_types = [v[0] for v in doc['vectors']]
        
        assert 'full' in vector_types
        assert 'summary' in vector_types
    
    def test_retrieve(self, retriever):
        """Test retrieval with multi-vector."""
        retriever.add_document("doc_1", "Python programming")
        retriever.add_document("doc_2", "Java development")
        
        results = retriever.retrieve("Python", top_k=2)
        
        assert len(results) == 2
        assert all('score' in r for r in results)

# ============================================================================
# Agentic RAG Tests
# ============================================================================

class TestAgenticRAGConfig:
    """Test Agentic RAG configuration."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = AgenticRAGConfig()
        
        assert config.max_iterations == 5
        assert config.max_tool_calls == 10
        assert config.enable_chain_of_thought is True
        assert config.tool_timeout_seconds == 30.0

class TestToolType:
    """Test ToolType enum."""
    
    def test_tool_types(self):
        """Test tool type values."""
        assert ToolType.SEARCH.value == "search"
        assert ToolType.CALCULATE.value == "calculate"
        assert ToolType.SUMMARIZE.value == "summarize"

class TestCalculateTool:
    """Test Calculate tool."""
    
    @pytest.fixture
    def calc_tool(self):
        """Create calculate tool."""
        return CalculateTool()
    
    @pytest.mark.asyncio
    async def test_simple_calculation(self, calc_tool):
        """Test simple math calculation."""
        result = await calc_tool.execute("2 + 2")
        
        assert result.success is True
        assert result.result == 4
        assert result.tool_name == "calculate"
    
    @pytest.mark.asyncio
    async def test_complex_calculation(self, calc_tool):
        """Test complex calculation."""
        result = await calc_tool.execute("(10 + 5) * 2")
        
        assert result.success is True
        assert result.result == 30
    
    @pytest.mark.asyncio
    async def test_invalid_expression(self, calc_tool):
        """Test invalid expression handling."""
        result = await calc_tool.execute("import os")
        
        assert result.success is False
        assert result.error is not None

class TestSearchTool:
    """Test Search tool."""
    
    @pytest.fixture
    def search_tool(self):
        """Create search tool."""
        async def search_fn(query: str):
            return [MockDocument(id="1", text=f"Results for {query}")]
        return SearchTool(search_fn=search_fn)
    
    @pytest.mark.asyncio
    async def test_search_execution(self, search_tool):
        """Test search execution."""
        result = await search_tool.execute("Python")
        
        assert result.success is True
        assert result.tool_name == "search"
        assert len(result.result) > 0

class TestAgentAction:
    """Test AgentAction dataclass."""
    
    def test_action_creation(self):
        """Test creating agent action."""
        action = AgentAction(
            thought="I need to search for information",
            action="search",
            action_input="Python programming",
        )
        
        assert action.thought != ""
        assert action.action == "search"
        assert action.action_input == "Python programming"

class TestAgentStep:
    """Test AgentStep dataclass."""
    
    def test_step_creation(self):
        """Test creating agent step."""
        action = AgentAction("thinking", "search", "query")
        step = AgentStep(
            action=action,
            observation="Found 5 results",
        )
        
        assert step.action == action
        assert step.observation == "Found 5 results"

class TestAgenticRAG:
    """Test Agentic RAG."""
    
    @pytest.fixture
    def agent(self):
        """Create agentic RAG."""
        async def llm_fn(prompt: str):
            # Return finish action to avoid infinite loop
            return """Thought: I have the answer.
Action: finish
Action Input: Python is a programming language."""
        
        return AgenticRAG(
            llm_fn=llm_fn,
            tools=[CalculateTool()],
        )
    
    def test_add_tool(self, agent):
        """Test adding tools."""
        async def dummy_search(q): return []
        agent.add_tool(SearchTool(dummy_search))
        
        assert "search" in agent.tools
    
    @pytest.mark.asyncio
    async def test_run_simple_query(self, agent):
        """Test running simple query."""
        result = await agent.run("What is Python?")
        
        assert isinstance(result, AgenticRAGResult)
        assert result.query == "What is Python?"
        assert result.answer != ""
    
    def test_parse_action(self, agent):
        """Test action parsing."""
        response = """Thought: Need to calculate
Action: calculate
Action Input: 2 + 2"""
        
        action = agent._parse_action(response)
        
        assert action.action == "calculate"
        assert action.action_input == "2 + 2"

# ============================================================================
# Streaming RAG Tests
# ============================================================================

class TestStreamingRAG:
    """Test Streaming RAG."""
    
    @pytest.fixture
    def streaming_rag(self):
        """Create streaming RAG."""
        return StreamingRAG(
            retrieve_fn=mock_retrieve,
            generate_fn=mock_generate,
        )
    
    @pytest.mark.asyncio
    async def test_retrieve_and_generate(self, streaming_rag):
        """Test non-streaming retrieve and generate."""
        result = await streaming_rag.retrieve_and_generate("What is Python?")
        
        assert 'query' in result
        assert 'response' in result
        assert 'documents' in result
    
    def test_build_context(self, streaming_rag):
        """Test context building."""
        docs = create_mock_documents(3)
        context = streaming_rag._build_context(docs)
        
        assert "[1]" in context
        assert "[2]" in context
        assert "[3]" in context
    
    def test_build_context_empty(self, streaming_rag):
        """Test context building with no documents."""
        context = streaming_rag._build_context([])
        
        assert "No relevant documents" in context

# ============================================================================
# RAG Orchestrator Tests
# ============================================================================

class TestRAGOrchestrator:
    """Test RAG Orchestrator."""
    
    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator."""
        async def simple_rag(query):
            return f"Simple response to: {query}"
        
        return RAGOrchestrator(simple_rag=simple_rag)
    
    @pytest.mark.asyncio
    async def test_run_simple_strategy(self, orchestrator):
        """Test running with simple strategy."""
        result = await orchestrator.run("What is Python?", strategy="simple")
        
        assert "Simple response" in result
    
    @pytest.mark.asyncio
    async def test_auto_select_strategy(self, orchestrator):
        """Test automatic strategy selection."""
        strategy = await orchestrator._select_strategy("What is Python?")
        
        assert strategy == "simple"  # Factual query
    
    @pytest.mark.asyncio
    async def test_select_agentic_for_complex(self, orchestrator):
        """Test agentic selection for complex queries."""
        strategy = await orchestrator._select_strategy("Compare Python and Java")
        
        assert strategy == "agentic"

# ============================================================================
# Integration Tests
# ============================================================================

class TestP6Integration:
    """Integration tests for P6 components."""
    
    def test_all_imports(self):
        """Test all P6 components can be imported."""
        # Multimodal imports
        assert MultimodalRetriever is not None
        assert MultimodalConfig is not None
        assert ModalityType is not None
        
        # ColBERT imports
        assert ColBERTRetriever is not None
        assert ColBERTConfig is not None
        
        # Multi-Vector imports
        assert MultiVectorRetriever is not None
        
        # Agentic RAG imports
        assert AgenticRAG is not None
        assert Tool is not None
        assert SearchTool is not None
        assert CalculateTool is not None
        
        # Streaming RAG imports
        assert StreamingRAG is not None
        assert RAGOrchestrator is not None
    
    def test_tool_result_creation(self):
        """Test ToolResult dataclass."""
        result = ToolResult(
            tool_name="search",
            success=True,
            result=["doc1", "doc2"],
            execution_time_ms=50.0,
        )
        
        assert result.success is True
        assert result.tool_name == "search"
        assert result.execution_time_ms == 50.0
    
    def test_multimodal_result_creation(self):
        """Test MultimodalResult dataclass."""
        result = MultimodalResult(
            content_id="img_1",
            modality=ModalityType.IMAGE,
            text="An image of a cat",
            score=0.85,
        )
        
        assert result.modality == ModalityType.IMAGE
        assert result.score == 0.85
    
    def test_agentic_result_creation(self):
        """Test AgenticRAGResult dataclass."""
        result = AgenticRAGResult(
            query="test",
            answer="response",
            tools_used=["search", "calculate"],
            total_iterations=3,
        )
        
        assert len(result.tools_used) == 2
        assert result.total_iterations == 3
