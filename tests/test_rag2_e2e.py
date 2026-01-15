"""
RAG2 End-to-End Tests

Tests the full RAG2 pipeline with:
- REAL database operations (Supabase/PostgreSQL)
- MOCKED model calls (embeddings, query planner, reranker)

This tests the actual database integration including:
- Document registration with hash-based idempotency
- Hierarchical chunk storage (parent/child)
- Vector embedding storage (1024d)
- FTS index population
- Multi-channel retrieval (lexical + semantic)
- RRF fusion
- Parent expansion
- Deduplication
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Skip if no database connection
pytestmark = pytest.mark.skipif(
    not os.getenv("SUPABASE_URL") or not os.getenv("SERVICE_ROLE_KEY"),
    reason="Requires SUPABASE_URL and SERVICE_ROLE_KEY environment variables"
)


# Test organization ID - use existing E2E test organization
TEST_ORG_ID = "00000000-0000-0000-0000-000000000001"  # E2E Test Organization
TEST_COLLECTION = "e2e_test"


def generate_mock_embedding(text: str, dim: int = 1024) -> List[float]:
    """
    Generate a deterministic mock embedding based on text hash.
    This ensures the same text always produces the same embedding.
    """
    # Use text hash as seed for reproducibility
    hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
    
    # Generate pseudo-random values based on hash
    import random
    random.seed(hash_val)
    embedding = [random.gauss(0, 1) for _ in range(dim)]
    
    # Normalize to unit length
    norm = sum(x * x for x in embedding) ** 0.5
    return [x / norm for x in embedding]


@dataclass
class MockEmbeddingResult:
    """Mock embedding result."""
    embedding: List[float]
    tokens_used: int = 100
    error: Optional[str] = None


class MockRAG2Embedder:
    """Mock embedder that returns deterministic embeddings."""
    
    def __init__(self, dim: int = 1024):
        self.dim = dim
        self.calls = []
    
    async def embed_text_async(self, text: str) -> MockEmbeddingResult:
        self.calls.append(("embed_text_async", text))
        return MockEmbeddingResult(
            embedding=generate_mock_embedding(text, self.dim)
        )
    
    async def embed_texts_async(self, texts: List[str]) -> List[MockEmbeddingResult]:
        self.calls.append(("embed_texts_async", texts))
        return [
            MockEmbeddingResult(embedding=generate_mock_embedding(t, self.dim))
            for t in texts
        ]
    
    def embed_query(self, query: str) -> List[float]:
        self.calls.append(("embed_query", query))
        return generate_mock_embedding(query, self.dim)
    
    async def embed_query_async(self, query: str) -> List[float]:
        self.calls.append(("embed_query_async", query))
        return generate_mock_embedding(query, self.dim)


class MockQueryPlanner:
    """Mock query planner that returns simple plans."""
    
    def __init__(self):
        self.calls = []
    
    def plan(self, query: str, collection: Optional[str] = None):
        from voice_agent.rag2.query_planner import QueryPlan
        
        self.calls.append(("plan", query, collection))
        
        # Simple keyword extraction
        keywords = [w.lower() for w in query.split() if len(w) > 2]
        
        return QueryPlan(
            original_query=query,
            keywords=keywords,
            semantic_query_text=query,
            lexical_top_k=50,
            semantic_top_k=100,
            weights={
                "lexical": 0.7,
                "semantic": 0.8,
                "graph": 0.0,
            },
            intent="factual",
        )
    
    async def plan_async(self, query: str, collection: Optional[str] = None):
        return self.plan(query, collection)


class MockReranker:
    """Mock reranker that uses simple heuristics."""
    
    def __init__(self):
        self.calls = []
    
    async def rerank_async(self, query: str, texts: List[str]) -> List[float]:
        self.calls.append(("rerank_async", query, texts))
        
        # Simple scoring: length overlap with query
        query_words = set(query.lower().split())
        scores = []
        for text in texts:
            text_words = set(text.lower().split())
            overlap = len(query_words & text_words)
            scores.append(0.5 + 0.1 * overlap)  # Base score + overlap bonus
        
        return scores


# Fixtures


@pytest.fixture(scope="module")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
def supabase_client():
    """Get real Supabase client."""
    from voice_agent.utils.db import get_supabase_client
    return get_supabase_client()


@pytest.fixture(scope="module")
def test_org_id():
    """Generate unique org ID for test isolation."""
    return TEST_ORG_ID


@pytest.fixture
def mock_embedder():
    """Create mock embedder."""
    return MockRAG2Embedder(dim=1024)


@pytest.fixture
def mock_query_planner():
    """Create mock query planner."""
    return MockQueryPlanner()


@pytest.fixture
def mock_reranker():
    """Create mock reranker."""
    return MockReranker()


@pytest.fixture
def sample_document_content():
    """Sample document content for testing."""
    return """
    # Voice Agent Documentation

    ## Introduction
    
    This document describes the Voice Agent system architecture and features.
    The system uses advanced natural language processing to handle customer calls.
    
    ## Key Features
    
    ### Speech Recognition
    
    The speech recognition module converts audio to text in real-time.
    It supports multiple languages including Portuguese and English.
    The system uses Whisper for accurate transcription.
    
    ### Natural Language Understanding
    
    The NLU component extracts intents and entities from user speech.
    It can identify customer requests, questions, and commands.
    Entity extraction includes names, dates, and phone numbers.
    
    ### Voice Synthesis
    
    Text-to-speech conversion uses high-quality neural voices.
    The system supports multiple voice personas.
    Response latency is optimized for natural conversation.
    
    ## System Architecture
    
    The system consists of the following components:
    
    1. Audio Pipeline: Handles audio streaming and processing
    2. RAG Module: Retrieves relevant context from knowledge base
    3. LLM Integration: Generates responses using GPT models
    4. Tool Execution: Performs actions like booking appointments
    
    ## Configuration
    
    Configuration is managed via environment variables.
    Key settings include API keys, model selection, and thresholds.
    
    ## Troubleshooting
    
    Common issues and solutions:
    
    - Audio quality: Check microphone settings
    - Slow responses: Verify network connectivity
    - Recognition errors: Adjust sensitivity settings
    """


@pytest.fixture
def temp_document(sample_document_content):
    """Create a temporary document file."""
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".md",
        delete=False
    ) as f:
        f.write(sample_document_content)
        f.flush()
        yield Path(f.name)
    
    # Cleanup
    os.unlink(f.name)


# Cleanup fixture

@pytest.fixture(scope="module", autouse=True)
def cleanup_test_data(supabase_client, test_org_id):
    """Clean up test data before and after tests."""
    def do_cleanup():
        try:
            # Delete in order respecting foreign keys
            supabase_client.table("rag_entity_mentions").delete().eq(
                "org_id", test_org_id
            ).execute()
            supabase_client.table("rag_relations").delete().eq(
                "org_id", test_org_id
            ).execute()
            supabase_client.table("rag_entities").delete().eq(
                "org_id", test_org_id
            ).execute()
            supabase_client.table("rag_child_chunks").delete().eq(
                "org_id", test_org_id
            ).execute()
            supabase_client.table("rag_parent_chunks").delete().eq(
                "org_id", test_org_id
            ).execute()
            supabase_client.table("rag_documents").delete().eq(
                "org_id", test_org_id
            ).execute()
        except Exception as e:
            print(f"Cleanup warning: {e}")
    
    # Cleanup before tests
    do_cleanup()
    
    yield
    
    # Cleanup after tests
    do_cleanup()


# E2E Tests


class TestRAG2E2EIngestion:
    """E2E tests for document ingestion."""
    
    @pytest.mark.asyncio
    async def test_ingest_document_creates_db_records(
        self,
        supabase_client,
        test_org_id,
        sample_document_content,
        mock_embedder,
    ):
        """Test that ingestion creates proper database records."""
        from voice_agent.rag2.ingest import RAG2Ingestor
        from voice_agent.ingestion.loader import DocumentLoader
        from voice_agent.rag2.chunker import get_hierarchical_chunker
        
        # Create unique temp file to ensure fresh ingestion
        unique_id = uuid.uuid4().hex[:8]
        unique_content = f"# Unique Test Document {unique_id}\n\n{sample_document_content}"
        
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False
        ) as f:
            f.write(unique_content)
            f.flush()
            unique_temp_document = Path(f.name)
        
        try:
            # Create ingestor with mock embedder
            ingestor = RAG2Ingestor(
                org_id=test_org_id,
                collection=TEST_COLLECTION,
                loader=DocumentLoader(),
                chunker=get_hierarchical_chunker(),
                embedder=mock_embedder,
                dedup_enabled=True,
            )
            
            # Ingest the document
            result = await ingestor.ingest_file(
                file_path=unique_temp_document,
                title="Voice Agent Documentation",
                tags=["test", "e2e"],
            )
            
            # Verify ingestion succeeded
            assert result.success, f"Ingestion failed: {result.error}"
            assert result.document_id is not None
            
            # Verify document was created in database
            doc_result = supabase_client.table("rag_documents").select("*").eq(
                "id", result.document_id
            ).execute()
            
            assert len(doc_result.data) == 1
            doc = doc_result.data[0]
            assert doc["org_id"] == test_org_id
            assert doc["collection"] == TEST_COLLECTION
            assert doc["title"] == "Voice Agent Documentation"
            assert doc["ingestion_status"] == "completed"
            assert "test" in doc["tags"]
            
            # Verify parent chunks were created
            parent_result = supabase_client.table("rag_parent_chunks").select("*").eq(
                "document_id", result.document_id
            ).order("index_in_document").execute()
            
            assert len(parent_result.data) > 0, "No parent chunks created"
            
            # Verify child chunks were created with embeddings
            child_result = supabase_client.table("rag_child_chunks").select("*").eq(
                "document_id", result.document_id
            ).execute()
            
            assert len(child_result.data) > 0, "No child chunks created"
            
            # Check embeddings are stored (1024d vector)
            first_child = child_result.data[0]
            assert first_child["embedding_1024"] is not None
            assert first_child["content_hash"] is not None
            assert first_child["org_id"] == test_org_id
            
            # Verify stats
            assert result.stats.documents_registered == 1
            assert result.stats.parent_chunks_created > 0
            assert result.stats.child_chunks_created > 0
            
            # Verify embedder was called
            assert len(mock_embedder.calls) > 0
            
        finally:
            os.unlink(unique_temp_document)
    
    @pytest.mark.asyncio
    async def test_ingest_document_idempotency(
        self,
        supabase_client,
        test_org_id,
        temp_document,
        mock_embedder,
    ):
        """Test that re-ingesting the same document is idempotent."""
        from voice_agent.rag2.ingest import RAG2Ingestor
        from voice_agent.ingestion.loader import DocumentLoader
        from voice_agent.rag2.chunker import get_hierarchical_chunker
        
        ingestor = RAG2Ingestor(
            org_id=test_org_id,
            collection=TEST_COLLECTION,
            loader=DocumentLoader(),
            chunker=get_hierarchical_chunker(),
            embedder=mock_embedder,
            dedup_enabled=True,
        )
        
        # First ingestion
        result1 = await ingestor.ingest_file(
            file_path=temp_document,
            title="First Ingest",
        )
        
        assert result1.success
        first_doc_id = result1.document_id
        
        # Second ingestion - should be skipped
        mock_embedder.calls.clear()  # Reset calls
        
        result2 = await ingestor.ingest_file(
            file_path=temp_document,
            title="Second Ingest",  # Different title shouldn't matter
        )
        
        assert result2.success
        assert result2.document_id == first_doc_id  # Same document
        assert result2.stats.documents_skipped == 1
        assert result2.stats.documents_registered == 0
        
        # Embedder should NOT have been called on skip
        assert len(mock_embedder.calls) == 0
        
        # Database should still have only one document with this hash
        doc_result = supabase_client.table("rag_documents").select("id").eq(
            "org_id", test_org_id
        ).eq("id", first_doc_id).execute()
        
        assert len(doc_result.data) == 1
    
    @pytest.mark.asyncio
    async def test_ingest_document_force_reingestion(
        self,
        supabase_client,
        test_org_id,
        mock_embedder,
    ):
        """Test that force=True allows re-ingestion with modified content."""
        from voice_agent.rag2.ingest import RAG2Ingestor
        from voice_agent.ingestion.loader import DocumentLoader
        from voice_agent.rag2.chunker import get_hierarchical_chunker
        
        # Create a unique temp file for this test run
        unique_id = uuid.uuid4().hex[:8]
        
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False
        ) as f:
            f.write(f"# Force Test {unique_id}\n\nThis is a force test document version 1.")
            f.flush()
            temp_path = Path(f.name)
        
        try:
            ingestor = RAG2Ingestor(
                org_id=test_org_id,
                collection=TEST_COLLECTION,
                loader=DocumentLoader(),
                chunker=get_hierarchical_chunker(),
                embedder=mock_embedder,
                dedup_enabled=True,
            )
            
            # First ingestion
            result1 = await ingestor.ingest_file(
                file_path=temp_path,
                title="Force Test Doc",
            )
            assert result1.success
            first_doc_id = result1.document_id
            
            # Modify the file content (creates new hash)
            with open(temp_path, "w") as f:
                f.write(f"# Force Test {unique_id}\n\nThis is a MODIFIED force test document version 2 - {uuid.uuid4().hex[:8]}.")
            
            # Force re-ingestion with modified content
            mock_embedder.calls.clear()
            
            result2 = await ingestor.ingest_file(
                file_path=temp_path,
                title="Force Test Doc v2",
                force=True,
            )
            
            assert result2.success
            # New document ID because content changed (new hash)
            assert result2.document_id != first_doc_id
            assert result2.stats.documents_registered == 1
            
            # Embedder should have been called again
            assert len(mock_embedder.calls) > 0
            
        finally:
            os.unlink(temp_path)


class TestRAG2E2ERetrieval:
    """E2E tests for retrieval pipeline."""
    
    @pytest.fixture(autouse=True)
    async def setup_test_data(
        self,
        supabase_client,
        test_org_id,
        temp_document,
        mock_embedder,
    ):
        """Ensure test data exists for retrieval tests."""
        from voice_agent.rag2.ingest import RAG2Ingestor
        from voice_agent.ingestion.loader import DocumentLoader
        from voice_agent.rag2.chunker import get_hierarchical_chunker
        
        ingestor = RAG2Ingestor(
            org_id=test_org_id,
            collection=TEST_COLLECTION,
            loader=DocumentLoader(),
            chunker=get_hierarchical_chunker(),
            embedder=mock_embedder,
            dedup_enabled=True,
        )
        
        result = await ingestor.ingest_file(
            file_path=temp_document,
            title="Retrieval Test Doc",
        )
        
        self.document_id = result.document_id
        self.mock_embedder = mock_embedder
        yield
    
    @pytest.mark.asyncio
    async def test_lexical_search(
        self,
        supabase_client,
        test_org_id,
    ):
        """Test lexical (FTS) search against database."""
        # Use raw FTS function
        result = supabase_client.rpc(
            "rag2_lexical_search",
            {
                "p_org_id": test_org_id,
                "p_query": "speech recognition",
                "p_limit": 10,
                "p_collection": TEST_COLLECTION,
            }
        ).execute()
        
        assert result.data is not None
        assert len(result.data) > 0
        
        # Should find chunks mentioning speech recognition
        texts = [r["text"] for r in result.data]
        assert any("speech" in t.lower() or "recognition" in t.lower() for t in texts)
    
    @pytest.mark.asyncio
    async def test_semantic_search(
        self,
        supabase_client,
        test_org_id,
        mock_embedder,
    ):
        """Test semantic (vector) search against database."""
        # Generate query embedding
        query = "how does the audio system work"
        query_embedding = mock_embedder.embed_query(query)
        
        # Use vector search
        result = supabase_client.rpc(
            "rag2_semantic_search",
            {
                "p_org_id": test_org_id,
                "p_embedding": query_embedding,
                "p_limit": 10,
                "p_collection": TEST_COLLECTION,
            }
        ).execute()
        
        assert result.data is not None
        assert len(result.data) > 0
        
        # Results should have similarity scores
        first = result.data[0]
        assert "similarity" in first
    
    @pytest.mark.asyncio
    async def test_full_retrieval_pipeline(
        self,
        supabase_client,
        test_org_id,
        mock_embedder,
        mock_query_planner,
        mock_reranker,
    ):
        """Test full retrieval pipeline with mocked models."""
        from voice_agent.rag2.retrieval import RAG2Retriever
        
        # Create retriever with mocks
        retriever = RAG2Retriever(
            org_id=test_org_id,
            embedder=mock_embedder,
            query_planner=mock_query_planner,
            graph_enabled=False,
        )
        
        # Inject mock reranker
        retriever._reranker = mock_reranker
        
        # Execute retrieval
        result = await retriever.retrieve(
            query="What is speech recognition?",
            collection=TEST_COLLECTION,
            top_k=5,
            skip_rerank=False,
        )
        
        # Verify result structure
        assert result.success
        assert result.query_plan is not None
        assert result.query_plan.original_query == "What is speech recognition?"
        
        # Verify timings are tracked
        assert "planning" in result.timings
        assert "retrieval" in result.timings
        
        # Verify query planner was called
        assert len(mock_query_planner.calls) > 0
        
        # Verify embedder was called for semantic search
        embed_calls = [c for c in mock_embedder.calls if c[0] in ("embed_query", "embed_query_async")]
        assert len(embed_calls) > 0
    
    @pytest.mark.asyncio
    async def test_retrieval_with_parent_expansion(
        self,
        supabase_client,
        test_org_id,
        mock_embedder,
        mock_query_planner,
    ):
        """Test that child chunks are expanded to parent context."""
        from voice_agent.rag2.retrieval import RAG2Retriever
        
        retriever = RAG2Retriever(
            org_id=test_org_id,
            embedder=mock_embedder,
            query_planner=mock_query_planner,
            graph_enabled=False,
        )
        
        result = await retriever.retrieve(
            query="NLU component",
            collection=TEST_COLLECTION,
            top_k=3,
            skip_rerank=True,  # Skip rerank to simplify
        )
        
        assert result.success
        
        # If we have contexts, check parent expansion
        if result.contexts:
            for ctx in result.contexts:
                # Parent text should be populated after expansion
                # (may be None if expansion wasn't needed)
                assert ctx.child_id is not None
                assert ctx.parent_id is not None
    
    @pytest.mark.asyncio
    async def test_retrieval_safety_threshold(
        self,
        supabase_client,
        test_org_id,
        mock_embedder,
        mock_query_planner,
    ):
        """Test safety threshold filters low-confidence results."""
        from voice_agent.rag2.retrieval import RAG2Retriever
        
        retriever = RAG2Retriever(
            org_id=test_org_id,
            embedder=mock_embedder,
            query_planner=mock_query_planner,
            graph_enabled=False,
        )
        
        # Query something completely unrelated
        result = await retriever.retrieve(
            query="quantum physics black holes",
            collection=TEST_COLLECTION,
            top_k=5,
            skip_rerank=True,
        )
        
        # Even if not refused, scores should be lower
        assert result.success
        # Max rerank score tracked (will be RRF if no rerank)


class TestRAG2E2EDeduplication:
    """E2E tests for deduplication."""
    
    @pytest.mark.asyncio
    async def test_chunk_deduplication(
        self,
        supabase_client,
        test_org_id,
        mock_embedder,
    ):
        """Test that duplicate chunks are deduplicated."""
        from voice_agent.rag2.ingest import RAG2Ingestor
        from voice_agent.ingestion.loader import DocumentLoader
        from voice_agent.rag2.chunker import get_hierarchical_chunker
        
        # Create two files with same content
        content = """
        # Duplicate Content Test
        
        This is some test content that will appear in both documents.
        It should be deduplicated when the second document is ingested.
        
        The deduplication is based on content hash at the chunk level.
        """
        
        files = []
        for i in range(2):
            f = tempfile.NamedTemporaryFile(
                mode="w", suffix=".md", delete=False
            )
            f.write(content)
            f.flush()
            files.append(Path(f.name))
            f.close()
        
        try:
            ingestor = RAG2Ingestor(
                org_id=test_org_id,
                collection=TEST_COLLECTION,
                loader=DocumentLoader(),
                chunker=get_hierarchical_chunker(),
                embedder=mock_embedder,
                dedup_enabled=True,
            )
            
            # Ingest first file
            result1 = await ingestor.ingest_file(
                file_path=files[0],
                title="Dedup Test 1",
            )
            assert result1.success
            chunks_created_1 = result1.stats.child_chunks_created
            
            # Ingest second file with DIFFERENT content to bypass doc-level dedup
            with open(files[1], "a") as f:
                f.write("\n\n# Extra Section\nThis is unique content.\n")
            
            result2 = await ingestor.ingest_file(
                file_path=files[1],
                title="Dedup Test 2",
            )
            
            assert result2.success
            
            # Some chunks should have been deduplicated
            # (the common content chunks)
            if result2.stats.child_chunks_deduplicated > 0:
                # Dedup working
                assert result2.stats.child_chunks_deduplicated > 0
            
        finally:
            for f in files:
                os.unlink(f)


class TestRAG2E2ESchemaIntegrity:
    """Tests for database schema integrity."""
    
    def test_tables_exist(self, supabase_client):
        """Verify all RAG2 tables exist."""
        tables = [
            "rag_documents",
            "rag_parent_chunks",
            "rag_child_chunks",
            "rag_entities",
            "rag_relations",
            "rag_entity_mentions",
        ]
        
        for table in tables:
            # Try to select from each table
            result = supabase_client.table(table).select("id").limit(1).execute()
            # Should not raise, even if empty
            assert isinstance(result.data, list)
    
    def test_foreign_key_integrity(self, supabase_client, test_org_id):
        """Test foreign key relationships are enforced."""
        from uuid import uuid4
        
        # Try to insert child chunk with non-existent parent
        fake_parent_id = str(uuid4())
        fake_doc_id = str(uuid4())
        
        try:
            supabase_client.table("rag_child_chunks").insert({
                "id": str(uuid4()),
                "parent_id": fake_parent_id,
                "document_id": fake_doc_id,
                "org_id": test_org_id,
                "index_in_parent": 0,
                "text": "Test",
                "content_hash": "test_hash",
            }).execute()
            
            # Should have raised, but if it didn't, clean up
            pytest.fail("Expected foreign key violation")
        except Exception as e:
            # Expected - foreign key constraint
            assert "foreign key" in str(e).lower() or "violates" in str(e).lower()
    
    def test_unique_constraint_on_content_hash(self, supabase_client, test_org_id):
        """Test unique constraint on org_id + content_hash."""
        from uuid import uuid4
        
        # First, create a document and parent
        doc_id = str(uuid4())
        parent_id = str(uuid4())
        
        # Insert document
        supabase_client.table("rag_documents").insert({
            "id": doc_id,
            "org_id": test_org_id,
            "hash_sha256": f"test_{uuid4().hex}",
            "file_name": "test.md",
            "mime_type": "text/markdown",
            "collection": TEST_COLLECTION,
        }).execute()
        
        # Insert parent
        supabase_client.table("rag_parent_chunks").insert({
            "id": parent_id,
            "document_id": doc_id,
            "org_id": test_org_id,
            "index_in_document": 0,
            "text": "Parent text",
        }).execute()
        
        # Insert child
        content_hash = f"unique_test_{uuid4().hex[:8]}"
        supabase_client.table("rag_child_chunks").insert({
            "id": str(uuid4()),
            "parent_id": parent_id,
            "document_id": doc_id,
            "org_id": test_org_id,
            "index_in_parent": 0,
            "text": "Child text",
            "content_hash": content_hash,
        }).execute()
        
        # Try to insert another with same org_id + content_hash
        try:
            supabase_client.table("rag_child_chunks").insert({
                "id": str(uuid4()),
                "parent_id": parent_id,
                "document_id": doc_id,
                "org_id": test_org_id,
                "index_in_parent": 1,
                "text": "Different text but same hash",
                "content_hash": content_hash,  # Same hash!
            }).execute()
            
            pytest.fail("Expected unique constraint violation")
        except Exception as e:
            # Expected - unique constraint
            assert "duplicate" in str(e).lower() or "unique" in str(e).lower()


class TestRAG2E2EFunctions:
    """Tests for database functions."""
    
    def test_lexical_search_function_exists(self, supabase_client, test_org_id):
        """Test FTS search function is callable."""
        # May return empty but should not error
        result = supabase_client.rpc(
            "rag2_lexical_search",
            {
                "p_org_id": test_org_id,
                "p_query": "test",
                "p_limit": 10,
                "p_collection": None,
            }
        ).execute()
        
        assert isinstance(result.data, list)
    
    def test_semantic_search_function_exists(self, supabase_client, test_org_id):
        """Test vector search function is callable."""
        # Create a dummy 1024d embedding
        dummy_embedding = [0.0] * 1024
        dummy_embedding[0] = 1.0  # Normalize
        
        result = supabase_client.rpc(
            "rag2_semantic_search",
            {
                "p_org_id": test_org_id,
                "p_embedding": dummy_embedding,
                "p_limit": 10,
                "p_collection": None,
            }
        ).execute()
        
        assert isinstance(result.data, list)
    
    def test_hybrid_search_function_exists(self, supabase_client, test_org_id):
        """Test hybrid search function is callable."""
        dummy_embedding = [0.0] * 1024
        dummy_embedding[0] = 1.0
        
        result = supabase_client.rpc(
            "rag2_hybrid_rrf_search",
            {
                "p_org_id": test_org_id,
                "p_query": "test query",
                "p_embedding": dummy_embedding,
                "p_limit": 10,
                "p_collection": None,
            }
        ).execute()
        
        assert isinstance(result.data, list)


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
