"""
RAG2 Full Integration Tests (Mocked Models)

Tests the complete RAG2 pipeline with mocked model calls to verify
all components are properly connected:
- Ingestion: Load → Chunk → Embed → Store
- Retrieval: Plan → Search (Lexical+Semantic+Graph) → Fuse → Expand → Rerank → Safety

All model calls (embeddings, query planner, reranker) are mocked.
Database operations are REAL to test actual integration.
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dotenv import load_dotenv
load_dotenv()

# Skip if no database
pytestmark = pytest.mark.skipif(
    not os.getenv("SUPABASE_URL") or not os.getenv("SERVICE_ROLE_KEY"),
    reason="Requires database connection"
)

# Use existing E2E test organization
TEST_ORG_ID = "00000000-0000-0000-0000-000000000001"
TEST_COLLECTION = "integration_test"


# ============================================================================
# Mock Helpers
# ============================================================================

def deterministic_embedding(text: str, dim: int = 1024) -> List[float]:
    """Generate deterministic embedding from text hash."""
    hash_bytes = hashlib.sha256(text.encode()).digest()
    import struct
    values = []
    for i in range(dim):
        byte_idx = i % len(hash_bytes)
        values.append((hash_bytes[byte_idx] - 128) / 128.0)
    # Normalize
    norm = sum(v * v for v in values) ** 0.5
    return [v / norm for v in values]


class MockOpenAIClient:
    """Mock OpenAI client for embeddings and chat."""
    
    def __init__(self):
        self.embeddings = MockEmbeddings()
        self.chat = MockChat()


class MockEmbeddings:
    """Mock embeddings endpoint."""
    
    def create(self, model: str, input: Any, **kwargs) -> MagicMock:
        """Create embeddings response."""
        if isinstance(input, str):
            texts = [input]
        else:
            texts = input
        
        response = MagicMock()
        response.data = []
        response.usage = MagicMock(total_tokens=len(texts) * 100)
        
        for i, text in enumerate(texts):
            embedding_obj = MagicMock()
            # Generate 4096d embedding (full Qwen dimension)
            embedding_obj.embedding = deterministic_embedding(text, 4096)
            embedding_obj.index = i
            response.data.append(embedding_obj)
        
        return response


class MockChat:
    """Mock chat completions endpoint."""
    
    @property
    def completions(self):
        return self
    
    def create(self, model: str, messages: List, **kwargs) -> MagicMock:
        """Create chat completion response."""
        # Extract query from messages
        user_msg = next((m for m in messages if m.get("role") == "user"), {})
        query = user_msg.get("content", "")
        
        # Generate mock query plan
        import json
        plan = {
            "keywords": query.lower().split()[:5],
            "semantic_query_text": query,
            "cypher_query": None,
            "requires_graph": False,
            "intent": "factual",
            "weights": {"lexical": 0.7, "semantic": 0.8, "graph": 0.0}
        }
        
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message = MagicMock()
        response.choices[0].message.content = json.dumps(plan)
        
        return response


class MockReranker:
    """Mock reranker."""
    
    async def rerank_async(self, query: str, texts: List[str]) -> List[float]:
        """Score texts based on keyword overlap."""
        query_words = set(query.lower().split())
        scores = []
        for text in texts:
            text_words = set(text.lower().split())
            overlap = len(query_words & text_words)
            scores.append(0.5 + 0.1 * min(overlap, 5))
        return scores
    
    def rerank(self, query: str, texts: List[str]) -> List[float]:
        """Sync version."""
        return asyncio.get_event_loop().run_until_complete(
            self.rerank_async(query, texts)
        )


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_openai():
    """Patch OpenAI client."""
    return MockOpenAIClient()


@pytest.fixture
def mock_reranker():
    """Create mock reranker."""
    return MockReranker()


@pytest.fixture
def supabase_client():
    """Get real Supabase client."""
    from voice_agent.utils.db import get_supabase_client
    return get_supabase_client()


@pytest.fixture
def sample_documents():
    """Create sample test documents."""
    return {
        "hr_policy": """
# HR Policies and Procedures

## Leave Policy

### Vacation Days
All employees receive 20 vacation days per year.
Unused days can be carried over up to 5 days.

### Sick Leave
Employees are entitled to 10 sick days annually.
A doctor's note is required for absences over 3 days.

## Remote Work Policy

Employees may work remotely up to 3 days per week.
Manager approval is required for regular remote work.
Core hours are 10am to 3pm in the employee's timezone.
""",
        "product_docs": """
# Product Documentation

## Voice Agent Features

### Speech Recognition
The system uses advanced ASR to convert speech to text.
Supports multiple languages including Portuguese and English.
Real-time transcription with low latency.

### Natural Language Understanding
Extracts intents and entities from user speech.
Handles context switches and multi-turn conversations.
Custom entity extraction for domain-specific terms.

### Response Generation
Uses RAG to retrieve relevant knowledge base content.
Generates natural conversational responses.
Supports voice synthesis with multiple personas.
""",
        "technical_specs": """
# Technical Specifications

## System Architecture

### Components
- Audio Pipeline: Handles streaming audio I/O
- RAG Module: Retrieves context from knowledge base
- LLM Integration: Generates responses using AI models
- Tool Execution: Performs actions like booking

### Performance Requirements
- Response latency: < 500ms p95
- Transcription accuracy: > 95%
- Concurrent calls: 100+

### Database
PostgreSQL with pgvector for embeddings.
HNSW index for fast similarity search.
Full-text search for keyword matching.
"""
    }


@pytest.fixture
def temp_docs(sample_documents):
    """Create temporary document files."""
    temp_files = {}
    for name, content in sample_documents.items():
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False,
            prefix=f"{name}_"
        )
        f.write(content)
        f.flush()
        f.close()
        temp_files[name] = Path(f.name)
    
    yield temp_files
    
    # Cleanup
    for path in temp_files.values():
        try:
            os.unlink(path)
        except:
            pass


@pytest.fixture(scope="module", autouse=True)
def cleanup_test_data(request):
    """Clean up test data after all tests."""
    yield
    
    # Cleanup after all tests in module
    from voice_agent.utils.db import get_supabase_client
    client = get_supabase_client()
    
    try:
        # Delete test documents by collection
        client.table("rag_child_chunks").delete().eq(
            "org_id", TEST_ORG_ID
        ).execute()
        client.table("rag_parent_chunks").delete().eq(
            "org_id", TEST_ORG_ID
        ).execute()
        client.table("rag_documents").delete().eq(
            "org_id", TEST_ORG_ID
        ).eq("collection", TEST_COLLECTION).execute()
    except Exception as e:
        print(f"Cleanup warning: {e}")


# ============================================================================
# Integration Tests
# ============================================================================

class TestFullIngestionPipeline:
    """Test complete ingestion pipeline with mocked embeddings."""
    
    @pytest.mark.asyncio
    async def test_ingest_single_document(
        self,
        supabase_client,
        temp_docs,
        mock_openai,
    ):
        """Test ingesting a single document through the full pipeline."""
        from voice_agent.rag2.ingest import RAG2Ingestor
        from voice_agent.rag2.embedder import RAG2Embedder
        from voice_agent.ingestion.loader import DocumentLoader
        from voice_agent.rag2.chunker import get_hierarchical_chunker
        
        # Create embedder with mocked client
        embedder = RAG2Embedder()
        embedder._client = mock_openai
        
        # Create ingestor
        ingestor = RAG2Ingestor(
            org_id=TEST_ORG_ID,
            collection=TEST_COLLECTION,
            loader=DocumentLoader(),
            chunker=get_hierarchical_chunker(),
            embedder=embedder,
            dedup_enabled=True,
        )
        
        # Ingest HR policy document
        result = await ingestor.ingest_file(
            file_path=temp_docs["hr_policy"],
            title="HR Policy Document",
            tags=["hr", "policy", "integration_test"],
        )
        
        # Verify success
        assert result.success, f"Ingestion failed: {result.error}"
        assert result.document_id is not None
        
        # Verify database records
        doc = supabase_client.table("rag_documents").select("*").eq(
            "id", result.document_id
        ).single().execute()
        
        assert doc.data["title"] == "HR Policy Document"
        assert doc.data["collection"] == TEST_COLLECTION
        assert doc.data["ingestion_status"] == "completed"
        
        # Verify chunks
        parents = supabase_client.table("rag_parent_chunks").select("*").eq(
            "document_id", result.document_id
        ).execute()
        assert len(parents.data) > 0
        
        children = supabase_client.table("rag_child_chunks").select("*").eq(
            "document_id", result.document_id
        ).execute()
        assert len(children.data) > 0
        
        # Verify embeddings are stored
        for child in children.data:
            embedding = child["embedding_1024"]
            assert embedding is not None
            # Embedding might be returned as string or list depending on DB driver
            if isinstance(embedding, str):
                # Parse string representation: "[0.1,0.2,...]"
                import json
                embedding = json.loads(embedding)
            assert len(embedding) == 1024, f"Expected 1024d, got {len(embedding)}d"
        
        print(f"✅ Ingested: {len(parents.data)} parents, {len(children.data)} children")
    
    @pytest.mark.asyncio
    async def test_ingest_multiple_documents(
        self,
        supabase_client,
        temp_docs,
        mock_openai,
    ):
        """Test ingesting multiple documents."""
        from voice_agent.rag2.ingest import RAG2Ingestor
        from voice_agent.rag2.embedder import RAG2Embedder
        from voice_agent.ingestion.loader import DocumentLoader
        from voice_agent.rag2.chunker import get_hierarchical_chunker
        
        embedder = RAG2Embedder()
        embedder._client = mock_openai
        
        ingestor = RAG2Ingestor(
            org_id=TEST_ORG_ID,
            collection=TEST_COLLECTION,
            loader=DocumentLoader(),
            chunker=get_hierarchical_chunker(),
            embedder=embedder,
        )
        
        doc_ids = []
        for name, path in temp_docs.items():
            result = await ingestor.ingest_file(
                file_path=path,
                title=f"Test Doc: {name}",
                tags=["integration_test", name],
            )
            if result.success:
                doc_ids.append(result.document_id)
        
        # Verify all documents ingested
        assert len(doc_ids) == len(temp_docs)
        
        # Verify total chunks
        total_children = supabase_client.table("rag_child_chunks").select(
            "id", count="exact"
        ).in_("document_id", doc_ids).execute()
        
        assert total_children.count > 0
        print(f"✅ Ingested {len(doc_ids)} documents with {total_children.count} child chunks")


class TestFullRetrievalPipeline:
    """Test complete retrieval pipeline with mocked models."""
    
    @pytest.fixture(autouse=True)
    async def setup_test_data(
        self,
        supabase_client,
        temp_docs,
        mock_openai,
    ):
        """Ensure test documents are ingested."""
        from voice_agent.rag2.ingest import RAG2Ingestor
        from voice_agent.rag2.embedder import RAG2Embedder
        from voice_agent.ingestion.loader import DocumentLoader
        from voice_agent.rag2.chunker import get_hierarchical_chunker
        
        embedder = RAG2Embedder()
        embedder._client = mock_openai
        
        ingestor = RAG2Ingestor(
            org_id=TEST_ORG_ID,
            collection=TEST_COLLECTION,
            loader=DocumentLoader(),
            chunker=get_hierarchical_chunker(),
            embedder=embedder,
        )
        
        # Ingest all docs (will skip if already exist)
        for name, path in temp_docs.items():
            await ingestor.ingest_file(
                file_path=path,
                title=f"Retrieval Test: {name}",
            )
        
        self.mock_openai = mock_openai
        yield
    
    @pytest.mark.asyncio
    async def test_lexical_search_integration(self, supabase_client):
        """Test lexical search finds relevant documents."""
        # Direct RPC call
        result = supabase_client.rpc(
            "rag2_lexical_search",
            {
                "p_org_id": TEST_ORG_ID,
                "p_query": "vacation days policy",
                "p_limit": 10,
                "p_collection": TEST_COLLECTION,
            }
        ).execute()
        
        assert len(result.data) > 0
        
        # Should find HR policy content
        texts = [r["text"].lower() for r in result.data]
        assert any("vacation" in t or "days" in t for t in texts)
        print(f"✅ Lexical search found {len(result.data)} results")
    
    @pytest.mark.asyncio
    async def test_semantic_search_integration(self, supabase_client):
        """Test semantic search finds relevant documents."""
        # Generate query embedding
        query = "employee time off and leave"
        query_embedding = deterministic_embedding(query, 1024)
        
        result = supabase_client.rpc(
            "rag2_semantic_search",
            {
                "p_org_id": TEST_ORG_ID,
                "p_embedding": query_embedding,
                "p_limit": 10,
                "p_collection": TEST_COLLECTION,
            }
        ).execute()
        
        assert len(result.data) > 0
        
        # Results should have similarity scores
        for r in result.data:
            assert "similarity" in r
            assert r["similarity"] is not None
        
        print(f"✅ Semantic search found {len(result.data)} results")
    
    @pytest.mark.asyncio
    async def test_hybrid_search_integration(self, supabase_client):
        """Test hybrid RRF search combines lexical and semantic."""
        query = "speech recognition accuracy"
        query_embedding = deterministic_embedding(query, 1024)
        
        result = supabase_client.rpc(
            "rag2_hybrid_rrf_search",
            {
                "p_org_id": TEST_ORG_ID,
                "p_query": query,
                "p_embedding": query_embedding,
                "p_limit": 10,
                "p_collection": TEST_COLLECTION,
            }
        ).execute()
        
        assert len(result.data) > 0
        
        # Results should have RRF scores
        for r in result.data:
            assert "rrf_score" in r
        
        print(f"✅ Hybrid search found {len(result.data)} results")
    
    @pytest.mark.asyncio
    async def test_full_retrieval_pipeline(
        self,
        supabase_client,
        mock_reranker,
    ):
        """Test complete retrieval pipeline."""
        from voice_agent.rag2.retrieval import RAG2Retriever
        from voice_agent.rag2.embedder import RAG2Embedder
        from voice_agent.rag2.query_planner import QueryPlanner
        
        # Create components with mocks
        embedder = RAG2Embedder()
        embedder._client = self.mock_openai
        
        query_planner = QueryPlanner()
        query_planner._client = self.mock_openai
        
        retriever = RAG2Retriever(
            org_id=TEST_ORG_ID,
            embedder=embedder,
            query_planner=query_planner,
            graph_enabled=False,
        )
        retriever._reranker = mock_reranker
        
        # Execute retrieval
        result = await retriever.retrieve(
            query="What is the vacation policy?",
            collection=TEST_COLLECTION,
            top_k=5,
        )
        
        assert result.success
        assert result.query_plan is not None
        
        # Check timings were recorded
        assert "planning" in result.timings
        assert "retrieval" in result.timings
        assert "fusion" in result.timings
        
        print(f"✅ Full retrieval: {len(result.contexts)} contexts, "
              f"max_score={result.max_rerank_score:.2f}")
    
    @pytest.mark.asyncio
    async def test_retrieval_with_different_queries(
        self,
        supabase_client,
        mock_reranker,
    ):
        """Test retrieval with various query types."""
        from voice_agent.rag2.retrieval import RAG2Retriever
        from voice_agent.rag2.embedder import RAG2Embedder
        from voice_agent.rag2.query_planner import QueryPlanner
        
        embedder = RAG2Embedder()
        embedder._client = self.mock_openai
        
        query_planner = QueryPlanner()
        query_planner._client = self.mock_openai
        
        retriever = RAG2Retriever(
            org_id=TEST_ORG_ID,
            embedder=embedder,
            query_planner=query_planner,
        )
        retriever._reranker = mock_reranker
        
        queries = [
            "How many vacation days do employees get?",
            "What are the remote work requirements?",
            "Speech recognition latency requirements",
            "Database architecture and vector search",
        ]
        
        for query in queries:
            result = await retriever.retrieve(
                query=query,
                collection=TEST_COLLECTION,
                top_k=3,
            )
            
            assert result.success, f"Failed for query: {query}"
            print(f"  ✓ '{query[:40]}...' → {len(result.contexts)} results")
        
        print(f"✅ All {len(queries)} queries executed successfully")


class TestPipelineDataFlow:
    """Test data flows correctly between pipeline stages."""
    
    @pytest.mark.asyncio
    async def test_chunk_content_hash_uniqueness(
        self,
        supabase_client,
        temp_docs,
        mock_openai,
    ):
        """Test that content hashes are unique per chunk."""
        from voice_agent.rag2.ingest import RAG2Ingestor
        from voice_agent.rag2.embedder import RAG2Embedder
        from voice_agent.ingestion.loader import DocumentLoader
        from voice_agent.rag2.chunker import get_hierarchical_chunker
        
        embedder = RAG2Embedder()
        embedder._client = mock_openai
        
        ingestor = RAG2Ingestor(
            org_id=TEST_ORG_ID,
            collection=TEST_COLLECTION,
            loader=DocumentLoader(),
            chunker=get_hierarchical_chunker(),
            embedder=embedder,
        )
        
        # Ingest a document
        result = await ingestor.ingest_file(
            file_path=temp_docs["product_docs"],
            title="Hash Test Doc",
        )
        
        # Get all chunks for this document
        chunks = supabase_client.table("rag_child_chunks").select(
            "id, content_hash, text"
        ).eq("document_id", result.document_id).execute()
        
        # All content hashes should be unique
        hashes = [c["content_hash"] for c in chunks.data]
        assert len(hashes) == len(set(hashes)), "Duplicate content hashes found"
        
        print(f"✅ {len(hashes)} unique content hashes verified")
    
    @pytest.mark.asyncio
    async def test_parent_child_relationship(
        self,
        supabase_client,
        temp_docs,
        mock_openai,
    ):
        """Test parent-child chunk relationships are correct."""
        from voice_agent.rag2.ingest import RAG2Ingestor
        from voice_agent.rag2.embedder import RAG2Embedder
        from voice_agent.ingestion.loader import DocumentLoader
        from voice_agent.rag2.chunker import get_hierarchical_chunker
        
        embedder = RAG2Embedder()
        embedder._client = mock_openai
        
        ingestor = RAG2Ingestor(
            org_id=TEST_ORG_ID,
            collection=TEST_COLLECTION,
            loader=DocumentLoader(),
            chunker=get_hierarchical_chunker(),
            embedder=embedder,
        )
        
        result = await ingestor.ingest_file(
            file_path=temp_docs["technical_specs"],
            title="Relationship Test Doc",
        )
        
        # Get parents
        parents = supabase_client.table("rag_parent_chunks").select("*").eq(
            "document_id", result.document_id
        ).execute()
        
        parent_ids = {p["id"] for p in parents.data}
        
        # Get children
        children = supabase_client.table("rag_child_chunks").select("*").eq(
            "document_id", result.document_id
        ).execute()
        
        # All children should reference valid parents
        for child in children.data:
            assert child["parent_id"] in parent_ids, \
                f"Child {child['id']} references invalid parent {child['parent_id']}"
        
        print(f"✅ {len(children.data)} children correctly linked to {len(parents.data)} parents")
    
    @pytest.mark.asyncio
    async def test_embedding_dimension_correct(
        self,
        supabase_client,
        temp_docs,
        mock_openai,
    ):
        """Test embeddings are stored with correct dimension (1024)."""
        from voice_agent.rag2.ingest import RAG2Ingestor
        from voice_agent.rag2.embedder import RAG2Embedder
        from voice_agent.ingestion.loader import DocumentLoader
        from voice_agent.rag2.chunker import get_hierarchical_chunker
        
        embedder = RAG2Embedder()
        embedder._client = mock_openai
        
        ingestor = RAG2Ingestor(
            org_id=TEST_ORG_ID,
            collection=TEST_COLLECTION,
            loader=DocumentLoader(),
            chunker=get_hierarchical_chunker(),
            embedder=embedder,
        )
        
        result = await ingestor.ingest_file(
            file_path=temp_docs["hr_policy"],
            title="Embedding Dim Test",
        )
        
        # Get a child chunk with embedding
        child = supabase_client.table("rag_child_chunks").select(
            "id, embedding_1024"
        ).eq("document_id", result.document_id).limit(1).single().execute()
        
        embedding = child.data["embedding_1024"]
        # Embedding might be returned as string or list depending on DB driver
        if isinstance(embedding, str):
            import json
            embedding = json.loads(embedding)
        assert len(embedding) == 1024, f"Expected 1024d, got {len(embedding)}d"
        
        print(f"✅ Embedding dimension correct: {len(embedding)}d")


class TestErrorHandling:
    """Test error handling throughout the pipeline."""
    
    @pytest.mark.asyncio
    async def test_ingest_nonexistent_file(self, mock_openai):
        """Test ingestion fails gracefully for missing file."""
        from voice_agent.rag2.ingest import RAG2Ingestor
        from voice_agent.rag2.embedder import RAG2Embedder
        from voice_agent.ingestion.loader import DocumentLoader
        from voice_agent.rag2.chunker import get_hierarchical_chunker
        
        embedder = RAG2Embedder()
        embedder._client = mock_openai
        
        ingestor = RAG2Ingestor(
            org_id=TEST_ORG_ID,
            collection=TEST_COLLECTION,
            loader=DocumentLoader(),
            chunker=get_hierarchical_chunker(),
            embedder=embedder,
        )
        
        result = await ingestor.ingest_file(
            file_path="/nonexistent/path/to/file.pdf",
            title="Should Fail",
        )
        
        assert not result.success
        assert result.error is not None
        print(f"✅ Graceful failure for missing file: {result.error[:50]}...")
    
    @pytest.mark.asyncio
    async def test_retrieval_empty_collection(self, mock_openai, mock_reranker):
        """Test retrieval handles empty collection gracefully."""
        from voice_agent.rag2.retrieval import RAG2Retriever
        from voice_agent.rag2.embedder import RAG2Embedder
        from voice_agent.rag2.query_planner import QueryPlanner
        
        embedder = RAG2Embedder()
        embedder._client = mock_openai
        
        query_planner = QueryPlanner()
        query_planner._client = mock_openai
        
        retriever = RAG2Retriever(
            org_id=TEST_ORG_ID,
            embedder=embedder,
            query_planner=query_planner,
        )
        retriever._reranker = mock_reranker
        
        # Query non-existent collection
        result = await retriever.retrieve(
            query="Some query",
            collection="nonexistent_collection_xyz",
            top_k=5,
        )
        
        # Should succeed but with empty results
        assert result.success
        assert len(result.contexts) == 0 or result.refused
        print("✅ Empty collection handled gracefully")


class TestQueryPlannerIntegration:
    """Test query planner integration."""
    
    def test_query_planner_generates_valid_plan(self, mock_openai):
        """Test query planner generates valid plan."""
        from voice_agent.rag2.query_planner import QueryPlanner, QueryPlan
        
        planner = QueryPlanner()
        planner._client = mock_openai
        
        plan = planner.plan("What are the vacation policies?")
        
        assert isinstance(plan, QueryPlan)
        assert plan.original_query == "What are the vacation policies?"
        assert len(plan.keywords) > 0
        assert plan.semantic_query_text is not None
        assert "lexical" in plan.weights
        assert "semantic" in plan.weights
        
        print(f"✅ Query plan: keywords={plan.keywords}, intent={plan.intent}")
    
    @pytest.mark.asyncio
    async def test_query_planner_async(self, mock_openai):
        """Test async query planner."""
        from voice_agent.rag2.query_planner import QueryPlanner
        
        planner = QueryPlanner()
        planner._client = mock_openai
        
        plan = await planner.plan_async("How do I request time off?")
        
        assert plan is not None
        assert len(plan.keywords) > 0
        print("✅ Async query planner works")


class TestRRFFusion:
    """Test RRF fusion logic."""
    
    def test_rrf_fusion_with_overlapping_results(self):
        """Test RRF fusion handles overlapping results."""
        from voice_agent.rag2.retrieval import RAG2Retriever, RetrievalCandidate
        
        # Create candidates with different channel rankings
        candidates = [
            RetrievalCandidate(
                child_id="1", parent_id="p1", document_id="d1",
                text="Text 1", page=1, modality="text",
                lexical_rank=1, semantic_rank=3,
            ),
            RetrievalCandidate(
                child_id="2", parent_id="p2", document_id="d1",
                text="Text 2", page=1, modality="text",
                lexical_rank=2, semantic_rank=1,
            ),
            RetrievalCandidate(
                child_id="3", parent_id="p3", document_id="d1",
                text="Text 3", page=1, modality="text",
                lexical_rank=None, semantic_rank=2,  # Only in semantic
            ),
        ]
        
        retriever = RAG2Retriever(org_id=TEST_ORG_ID)
        
        weights = {"lexical": 0.7, "semantic": 0.8, "graph": 0.0}
        fused = retriever._fuse_rrf(candidates, weights)
        
        # Should have 3 results
        assert len(fused) == 3
        
        # All should have RRF scores
        for c in fused:
            assert c.rrf_score > 0
        
        # Should be sorted by RRF score descending
        scores = [c.rrf_score for c in fused]
        assert scores == sorted(scores, reverse=True)
        
        print(f"✅ RRF fusion: scores={[f'{s:.3f}' for s in scores]}")


# Run with: pytest tests/test_rag2_integration.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
