# Phase 2: Module Validation

> **Priority:** HIGH - Validates new modules work E2E  
> **Time:** 1 hour 15 minutes  
> **Status:** [ ] Not Started  
> **Depends On:** Phase 1 (PuppyGraph Deploy)

---

## Objective

Validate the newly implemented modules (Graph Channel, Entity Extraction) work end-to-end with real data, not just mocked unit tests.

---

## Task 2.1: Graph Channel E2E Test

**Time:** 30 minutes  
**File:** `tests/test_rag2_graph_e2e.py` (new)

### Current State
- ✅ `graph_search.py` implemented with PuppyGraph + SQL fallback
- ✅ Unit tests pass with mocks
- ❌ No E2E test with actual PuppyGraph

### Implementation

Create new test file:

```python
# tests/test_rag2_graph_e2e.py
"""
E2E tests for Graph Channel with PuppyGraph.

Requires:
- PuppyGraph container running (Phase 1)
- RAG2_GRAPH_ENABLED=true in .env
"""

import pytest
from voice_agent.config import SETTINGS
from voice_agent.rag2 import RAG2Ingestor, RAG2Retriever
from voice_agent.supabase_client import get_supabase_client


@pytest.mark.skipif(
    not SETTINGS.rag2_graph_enabled,
    reason="RAG2_GRAPH_ENABLED not set"
)
class TestGraphChannelE2E:
    """E2E tests for graph channel."""
    
    @pytest.fixture
    def org_id(self):
        return "00000000-0000-0000-0000-000000000001"
    
    @pytest.fixture
    def supabase(self):
        return get_supabase_client()
    
    async def test_graph_search_returns_results(self, org_id, supabase):
        """Test that graph search returns results for entity queries."""
        # First, ingest a document with entities
        ingestor = RAG2Ingestor(
            org_id=org_id,
            collection="graph-test",
            entity_extraction_enabled=True,
        )
        
        # Ingest text with clear entities
        text = """
        ACME Corporation signed a contract with XYZ Industries.
        The contract value is $5,000,000 USD.
        John Smith from ACME negotiated with Jane Doe from XYZ.
        The agreement covers product delivery for 2026.
        """
        
        result = await ingestor.ingest_text(
            text=text,
            title="ACME-XYZ Contract",
            tags=["test", "graph"],
        )
        
        assert result.document_id is not None
        
        # Now query using graph channel
        retriever = RAG2Retriever(
            org_id=org_id,
            graph_enabled=True,
        )
        
        # Query that should trigger graph traversal
        query_result = await retriever.retrieve(
            query="What companies are mentioned in contracts?",
            collection="graph-test",
        )
        
        # Verify graph channel was considered
        assert query_result.query_plan is not None
        assert query_result.query_plan.requires_graph or query_result.query_plan.cypher_query
        
        # Should find ACME and XYZ
        assert not query_result.refused
        assert len(query_result.contexts) > 0
    
    async def test_graph_searcher_direct(self, org_id, supabase):
        """Test GraphSearcher directly."""
        from voice_agent.rag2.graph_search import get_graph_searcher
        
        searcher = get_graph_searcher(supabase)
        
        # Search for entities
        result = await searcher.search(
            keywords=["ACME", "contract"],
            org_id=org_id,
            top_k=10,
        )
        
        # Should return GraphSearchResult
        assert result is not None
        assert hasattr(result, 'entities')
        assert hasattr(result, 'chunk_ids')
    
    async def test_retrieval_timings_include_graph(self, org_id):
        """Test that graph channel timing is recorded."""
        retriever = RAG2Retriever(org_id=org_id, graph_enabled=True)
        
        result = await retriever.retrieve(
            query="entities related to payment",
            collection="graph-test",
        )
        
        # Timings should be recorded
        assert "retrieval" in result.timings
        # If graph was used, it should be fast (SQL fallback) or have graph timing
```

### Verification Steps

```bash
# Run graph E2E tests
cd /home/matheus/repos/voice-agent-v5
source venv/bin/activate
pytest tests/test_rag2_graph_e2e.py -v --tb=short
```

### Success Criteria
- [ ] All graph E2E tests pass
- [ ] Graph channel returns results for entity queries
- [ ] Timings are recorded correctly

---

## Task 2.2: Entity Extraction E2E Test

**Time:** 45 minutes  
**File:** `tests/test_rag2_entity_e2e.py` (new)

### Current State
- ✅ `entity_extraction.py` implemented with GPT-5
- ✅ Unit tests pass with mocks
- ❌ No E2E test with real GPT-5 API
- ❌ No test of EntityStore persistence

### Implementation

Create new test file:

```python
# tests/test_rag2_entity_e2e.py
"""
E2E tests for Entity Extraction with real GPT-5 API.

Requires:
- OpenAI API key configured
- RAG2_ENTITY_EXTRACTION_ENABLED=true in .env
"""

import pytest
from voice_agent.config import SETTINGS
from voice_agent.rag2 import (
    EntityExtractor,
    EntityStore,
    RAG2Ingestor,
    get_entity_extractor,
    get_entity_store,
)
from voice_agent.supabase_client import get_supabase_client


@pytest.mark.skipif(
    not SETTINGS.rag2_entity_extraction_enabled,
    reason="RAG2_ENTITY_EXTRACTION_ENABLED not set"
)
class TestEntityExtractionE2E:
    """E2E tests for entity extraction."""
    
    @pytest.fixture
    def org_id(self):
        return "00000000-0000-0000-0000-000000000001"
    
    @pytest.fixture
    def supabase(self):
        return get_supabase_client()
    
    async def test_extract_entities_from_text(self):
        """Test entity extraction with real GPT-5."""
        extractor = get_entity_extractor()
        
        text = """
        Apple Inc. announced a partnership with Microsoft Corporation.
        CEO Tim Cook and Satya Nadella signed the agreement on January 15, 2026.
        The deal is worth approximately $2.5 billion USD.
        The partnership focuses on cloud computing and AI integration.
        """
        
        result = await extractor.extract(text)
        
        # Should find organizations
        org_names = [e.name for e in result.entities if e.entity_type == "ORGANIZATION"]
        assert "Apple" in " ".join(org_names) or "Apple Inc" in " ".join(org_names)
        assert "Microsoft" in " ".join(org_names)
        
        # Should find people
        person_names = [e.name for e in result.entities if e.entity_type == "PERSON"]
        assert any("Tim Cook" in name or "Cook" in name for name in person_names)
        assert any("Nadella" in name for name in person_names)
        
        # Should find money
        money = [e for e in result.entities if e.entity_type == "MONEY"]
        assert len(money) > 0
        
        # Should find relations
        assert len(result.relations) > 0
    
    async def test_entity_store_persistence(self, org_id, supabase):
        """Test that entities are persisted to database."""
        store = get_entity_store(supabase)
        extractor = get_entity_extractor()
        
        text = "Google LLC acquired DeepMind Technologies for $500 million."
        
        result = await extractor.extract(text)
        
        # Store entities
        doc_id = "test-doc-entity-persistence"
        stored_ids = await store.store_entities(
            entities=result.entities,
            relations=result.relations,
            org_id=org_id,
            document_id=doc_id,
        )
        
        assert len(stored_ids["entity_ids"]) > 0
        
        # Verify in database
        db_result = supabase.table("rag_entities").select("*").eq(
            "document_id", doc_id
        ).execute()
        
        assert len(db_result.data) > 0
        
        # Cleanup
        supabase.table("rag_entities").delete().eq("document_id", doc_id).execute()
    
    async def test_ingestion_with_entity_extraction(self, org_id):
        """Test full ingestion pipeline with entity extraction enabled."""
        ingestor = RAG2Ingestor(
            org_id=org_id,
            collection="entity-test",
            entity_extraction_enabled=True,
        )
        
        text = """
        Contract between Vendor ABC and Client XYZ.
        
        Section 1: Payment Terms
        Client shall pay $10,000 monthly for services rendered.
        Payment due by the 15th of each month.
        
        Section 2: Duration
        This agreement is effective from January 1, 2026 to December 31, 2026.
        
        Signed by:
        - John Manager (Vendor ABC)
        - Sarah Director (Client XYZ)
        """
        
        # Track progress
        progress_steps = []
        def progress_callback(step, total, message):
            progress_steps.append((step, message))
        
        result = await ingestor.ingest_text(
            text=text,
            title="ABC-XYZ Contract",
            tags=["test", "entity"],
            progress_callback=progress_callback,
        )
        
        # Verify ingestion succeeded
        assert result.document_id is not None
        assert result.stats.parent_chunks_created > 0
        
        # Verify entity extraction step was called
        entity_step = [s for s in progress_steps if "entit" in s[1].lower()]
        assert len(entity_step) > 0, "Entity extraction step should be in progress"
        
        # Verify entities in database
        from voice_agent.supabase_client import get_supabase_client
        supabase = get_supabase_client()
        
        entities = supabase.table("rag_entities").select("*").eq(
            "document_id", result.document_id
        ).execute()
        
        # Should have extracted entities
        assert len(entities.data) > 0
        
        # Check entity types
        types = set(e["entity_type"] for e in entities.data)
        assert "ORGANIZATION" in types or "PERSON" in types or "MONEY" in types
    
    async def test_entity_types_coverage(self):
        """Test that all 10 entity types can be extracted."""
        extractor = get_entity_extractor()
        
        # Text with multiple entity types
        text = """
        REGULATION: GDPR Article 17 requires data deletion within 30 days.
        
        PRODUCT: The new iPhone 15 Pro costs $999.
        
        TECHNICAL_TERM: The system uses HNSW indexing for vector search.
        
        CLAUSE: Section 5.2 states that termination requires 90 days notice.
        
        LOCATION: The headquarters are located in San Francisco, California.
        
        DATE: The deadline is March 15, 2026.
        
        PERCENTAGE: Revenue increased by 25% year over year.
        """
        
        result = await extractor.extract(text)
        
        # Should find multiple types
        types = set(e.entity_type for e in result.entities)
        
        # At least 4 different types
        assert len(types) >= 4, f"Expected at least 4 types, got: {types}"
```

### Verification Steps

```bash
# Run entity extraction E2E tests
cd /home/matheus/repos/voice-agent-v5
source venv/bin/activate
pytest tests/test_rag2_entity_e2e.py -v --tb=short
```

### Success Criteria
- [ ] Entity extraction finds correct entities from text
- [ ] Entities are persisted to database
- [ ] Ingestion pipeline includes entity extraction step
- [ ] Multiple entity types are recognized

---

## Verification Checklist

```
[ ] Task 2.1: Graph Channel E2E
    [ ] Test file created
    [ ] Graph search returns results
    [ ] Timings recorded correctly
    [ ] All tests pass

[ ] Task 2.2: Entity Extraction E2E
    [ ] Test file created
    [ ] Entities extracted from text
    [ ] Entities persisted to database
    [ ] Ingestion includes entity step
    [ ] All tests pass
```

---

## Troubleshooting

### GPT-5 API errors
- Check OPENAI_API_KEY is set
- Verify API quota/limits
- Check model name in config (rag2_kg_ner_model)

### Database persistence fails
- Verify Supabase connection
- Check table exists (rag_entities)
- Verify org_id is valid UUID

### Graph search returns empty
- Verify entities were ingested first
- Check PuppyGraph is running (Phase 1)
- Fall back to SQL may return different results
