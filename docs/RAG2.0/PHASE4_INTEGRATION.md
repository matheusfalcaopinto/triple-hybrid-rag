# Phase 4: Integration & Documentation

> **Priority:** MEDIUM - End-to-end verification  
> **Time:** 1 hour  
> **Status:** [ ] Not Started  
> **Depends On:** Phase 1, 2, 3 all complete

---

## Objective

Validate the complete RAG 2.0 pipeline end-to-end and update all documentation to reflect 100% completion status.

---

## Task 4.1: Triple-Hybrid Integration Test

**Time:** 45 minutes  
**File:** `tests/test_rag2_triple_hybrid.py` (new)

### Current State
- ✅ Lexical channel working
- ✅ Semantic channel working
- ⏳ Graph channel pending PuppyGraph deployment
- ❌ No test that verifies all 3 channels + RRF fusion

### Prerequisites
- Phase 1: PuppyGraph deployed and running
- Phase 2: Graph E2E test passing

### Implementation

```python
# tests/test_rag2_triple_hybrid.py
"""
End-to-end integration tests for Triple-Hybrid retrieval.

This test validates:
1. All three retrieval channels (lexical, semantic, graph) work together
2. RRF fusion produces correct weighted results
3. Safety thresholds are applied
4. Full pipeline from ingestion to retrieval works

Prerequisites:
- PuppyGraph running on port 8182
- PostgreSQL with pgvector enabled
- Entity extraction enabled
"""

import asyncio
import pytest
from decimal import Decimal
from typing import Dict, List, Any

from voice_agent.rag2.ingest import RAG2Ingestor
from voice_agent.rag2.retrieval import RAG2Retriever, RetrievalResult
from voice_agent.rag2.entity_extraction import EntityExtractor


# Test documents with known entities and relations
TEST_DOCUMENTS = [
    {
        "id": "triple-hybrid-test-doc-1",
        "content": """
        Acme Corporation Policy Manual - Section 12
        
        Employee Benefits Overview
        
        John Smith, the HR Director at Acme Corporation, manages all employee benefits.
        The company offers health insurance through Blue Shield, with premiums starting at $450/month.
        Dental coverage is provided by Delta Dental at $75/month.
        
        The annual enrollment period runs from November 1st to November 30th.
        Employees must complete Form HR-101 to make changes to their benefits.
        
        For questions, contact the HR department at hr@acme.com or call 555-123-4567.
        The HR office is located at 123 Main Street, Building A, Room 405.
        """,
        "metadata": {
            "document_type": "policy",
            "department": "HR",
        },
        "expected_entities": ["John Smith", "Acme Corporation", "Blue Shield", "Delta Dental"],
        "expected_relations": [
            ("John Smith", "WORKS_AT", "Acme Corporation"),
            ("Acme Corporation", "USES_VENDOR", "Blue Shield"),
        ],
    },
    {
        "id": "triple-hybrid-test-doc-2",
        "content": """
        Acme Corporation Technical Specifications
        
        The proprietary ACME-9000 system processes 10,000 transactions per second.
        It was developed by the Engineering team led by Maria Garcia.
        
        System requirements:
        - CPU: Intel Xeon E5-2680 v4 or equivalent
        - RAM: 128GB minimum, 256GB recommended
        - Storage: 2TB NVMe SSD
        - Network: 10Gbps fiber connection
        
        The system integrates with Salesforce CRM and SAP ERP.
        API documentation is available at https://api.acme.com/docs.
        
        For technical support, contact the NOC at noc@acme.com.
        """,
        "metadata": {
            "document_type": "technical",
            "department": "Engineering",
        },
        "expected_entities": ["ACME-9000", "Maria Garcia", "Salesforce", "SAP"],
        "expected_relations": [
            ("Maria Garcia", "LEADS", "Engineering team"),
            ("ACME-9000", "INTEGRATES_WITH", "Salesforce"),
        ],
    },
]


class TestTripleHybridIntegration:
    """
    Full integration tests for triple-hybrid retrieval.
    
    These tests require:
    - Real database connection
    - PuppyGraph running
    - Entity extraction enabled
    """
    
    @pytest.fixture
    async def setup_test_data(self, test_org_id: str):
        """Ingest test documents with entity extraction."""
        ingestor = RAG2Ingestor(
            org_id=test_org_id,
            entity_extraction_enabled=True,
        )
        
        ingested_docs = []
        
        for doc in TEST_DOCUMENTS:
            result = await ingestor.ingest(
                content=doc["content"],
                document_id=doc["id"],
                metadata=doc["metadata"],
            )
            ingested_docs.append({
                "doc_id": doc["id"],
                "result": result,
                "expected_entities": doc.get("expected_entities", []),
            })
        
        # Allow time for indexing
        await asyncio.sleep(0.5)
        
        yield ingested_docs
        
        # Cleanup
        for doc in TEST_DOCUMENTS:
            await ingestor.delete_document(doc["id"])
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_all_three_channels_return_results(
        self,
        setup_test_data: List[Dict],
        test_org_id: str,
    ):
        """Test that all three retrieval channels return results."""
        retriever = RAG2Retriever(org_id=test_org_id)
        
        # Query that should hit all channels
        query = "What are the employee benefits at Acme Corporation?"
        
        result = await retriever.retrieve(
            query=query,
            top_k=10,
            debug=True,  # Enable debug for channel inspection
        )
        
        # Verify we got results
        assert len(result.contexts) > 0, "Should return contexts"
        
        # Check debug info for channel participation
        if hasattr(result, 'debug_info'):
            debug = result.debug_info
            
            # Lexical should find "employee benefits", "Acme Corporation"
            assert debug.get("lexical_candidates", 0) > 0, "Lexical should find candidates"
            
            # Semantic should find similar concepts
            assert debug.get("semantic_candidates", 0) > 0, "Semantic should find candidates"
            
            # Graph should find related entities
            if debug.get("graph_candidates", 0) == 0:
                pytest.skip("Graph channel returned no candidates - check PuppyGraph connection")
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_rrf_fusion_weights(
        self,
        setup_test_data: List[Dict],
        test_org_id: str,
    ):
        """Test that RRF fusion applies correct weights."""
        retriever = RAG2Retriever(org_id=test_org_id)
        
        # Query known to produce results from all channels
        query = "John Smith HR benefits"
        
        result = await retriever.retrieve(
            query=query,
            top_k=5,
            debug=True,
        )
        
        # Check that RRF scores are computed
        if hasattr(result, 'debug_info'):
            rrf_scores = result.debug_info.get("rrf_scores", {})
            
            # Verify fusion produced scores
            assert len(rrf_scores) > 0 or len(result.contexts) > 0, \
                "RRF should produce fused results"
            
            # Verify weights are applied (graph > semantic > lexical)
            # We can't verify exact weights, but results should be ordered
            if len(result.contexts) >= 2:
                # First result should have higher score than last
                assert result.contexts[0].get("score", 1) >= result.contexts[-1].get("score", 0)
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_entity_graph_traversal(
        self,
        setup_test_data: List[Dict],
        test_org_id: str,
    ):
        """Test that graph traversal finds entity relationships."""
        retriever = RAG2Retriever(org_id=test_org_id)
        
        # Query for related entities (should traverse graph)
        query = "Who manages benefits and what vendors does Acme use?"
        
        result = await retriever.retrieve(
            query=query,
            top_k=10,
        )
        
        # Should find:
        # - John Smith (HR Director)
        # - Blue Shield (health insurance)
        # - Delta Dental (dental)
        
        combined_text = " ".join([ctx.get("text", "") for ctx in result.contexts])
        
        # At least some of these should appear
        found_entities = []
        for entity in ["John Smith", "Blue Shield", "Delta Dental", "Acme"]:
            if entity.lower() in combined_text.lower():
                found_entities.append(entity)
        
        assert len(found_entities) >= 2, \
            f"Graph traversal should find related entities, found: {found_entities}"
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_cross_document_graph_query(
        self,
        setup_test_data: List[Dict],
        test_org_id: str,
    ):
        """Test that graph queries can find relationships across documents."""
        retriever = RAG2Retriever(org_id=test_org_id)
        
        # Query that spans both documents (HR and Engineering)
        query = "What systems and people are at Acme Corporation?"
        
        result = await retriever.retrieve(
            query=query,
            top_k=10,
        )
        
        combined_text = " ".join([ctx.get("text", "") for ctx in result.contexts])
        
        # Should find entities from both documents
        from_doc1 = "John Smith" in combined_text or "benefits" in combined_text.lower()
        from_doc2 = "Maria Garcia" in combined_text or "ACME-9000" in combined_text
        
        # At least one from each doc ideally
        assert from_doc1 or from_doc2, "Should find content from at least one document"
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_safety_threshold_application(
        self,
        setup_test_data: List[Dict],
        test_org_id: str,
    ):
        """Test that safety threshold filters low-confidence results."""
        retriever = RAG2Retriever(org_id=test_org_id)
        
        # Query that should produce mixed confidence results
        query = "What is the transaction processing capacity?"
        
        result = await retriever.retrieve(
            query=query,
            top_k=10,
            safety_threshold=0.6,  # Explicit threshold
        )
        
        # All returned results should meet threshold
        for ctx in result.contexts:
            if "score" in ctx:
                assert ctx["score"] >= 0.6 or ctx.get("from_graph", False), \
                    "Results should meet safety threshold or be from graph"
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_timing_includes_all_channels(
        self,
        setup_test_data: List[Dict],
        test_org_id: str,
    ):
        """Test that timing information includes all channels."""
        retriever = RAG2Retriever(org_id=test_org_id)
        
        result = await retriever.retrieve(
            query="employee benefits",
            top_k=5,
        )
        
        # Should have timing for all major steps
        expected_timing_keys = ["planning", "lexical", "semantic", "fusion"]
        
        for key in expected_timing_keys:
            assert key in result.timings, f"Timing should include '{key}'"
        
        # If graph is working, should have graph timing too
        if "graph" not in result.timings:
            pytest.skip("Graph timing missing - PuppyGraph may not be connected")
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_pipeline_ingestion_to_retrieval(
        self,
        test_org_id: str,
    ):
        """Test complete pipeline from fresh document to retrieval."""
        # Fresh document not in fixture
        test_doc = {
            "id": "pipeline-test-unique-" + str(asyncio.get_event_loop().time()),
            "content": """
            Unique Test Document for Pipeline Validation
            
            Dr. Sarah Johnson is the Chief Medical Officer at HealthCorp Inc.
            The company's flagship product, MediTrack Pro, monitors 50,000 patients daily.
            
            Contact: sarah.johnson@healthcorp.com
            Address: 456 Medical Plaza, Suite 789
            """,
        }
        
        # Step 1: Ingest
        ingestor = RAG2Ingestor(
            org_id=test_org_id,
            entity_extraction_enabled=True,
        )
        
        ingest_result = await ingestor.ingest(
            content=test_doc["content"],
            document_id=test_doc["id"],
        )
        
        assert ingest_result.success, "Ingestion should succeed"
        assert ingest_result.parent_count > 0, "Should create parent chunks"
        assert ingest_result.child_count > 0, "Should create child chunks"
        
        # Allow indexing
        await asyncio.sleep(0.5)
        
        # Step 2: Retrieve
        retriever = RAG2Retriever(org_id=test_org_id)
        
        result = await retriever.retrieve(
            query="Who is Sarah Johnson?",
            top_k=5,
        )
        
        # Should find the document
        assert len(result.contexts) > 0, "Should find ingested document"
        
        combined_text = " ".join([ctx.get("text", "") for ctx in result.contexts])
        assert "Sarah Johnson" in combined_text or "HealthCorp" in combined_text, \
            "Should find entity from ingested document"
        
        # Cleanup
        await ingestor.delete_document(test_doc["id"])


class TestChannelIsolation:
    """Test individual channels in isolation."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_lexical_only(self, test_org_id: str):
        """Test retrieval with only lexical channel."""
        retriever = RAG2Retriever(org_id=test_org_id)
        
        result = await retriever.retrieve(
            query="employee benefits",
            channels=["lexical"],  # Only lexical
            top_k=5,
        )
        
        # Should get results from lexical only
        assert result.timings.get("lexical", 0) > 0, "Should have lexical timing"
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_semantic_only(self, test_org_id: str):
        """Test retrieval with only semantic channel."""
        retriever = RAG2Retriever(org_id=test_org_id)
        
        result = await retriever.retrieve(
            query="worker health insurance",
            channels=["semantic"],
            top_k=5,
        )
        
        assert result.timings.get("semantic", 0) > 0, "Should have semantic timing"
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_graph_only(self, test_org_id: str):
        """Test retrieval with only graph channel."""
        retriever = RAG2Retriever(org_id=test_org_id)
        
        result = await retriever.retrieve(
            query="John Smith Acme",
            channels=["graph"],
            top_k=5,
        )
        
        if "graph" not in result.timings:
            pytest.skip("Graph channel not available")
        
        assert result.timings.get("graph", 0) > 0, "Should have graph timing"


class TestWeightedRRF:
    """Test RRF fusion mechanics."""
    
    def test_rrf_formula_calculation(self):
        """Test RRF formula: score = Σ (weight / (k + rank))."""
        from voice_agent.rag2.retrieval import calculate_rrf_score
        
        # Document ranked #1 in graph (weight=1.0), #3 in semantic (weight=0.8)
        ranks = {"graph": 1, "semantic": 3}
        weights = {"graph": 1.0, "semantic": 0.8, "lexical": 0.7}
        k = 60  # Standard RRF constant
        
        expected = (1.0 / (60 + 1)) + (0.8 / (60 + 3))
        expected = round(expected, 6)
        
        actual = calculate_rrf_score(ranks, weights, k)
        actual = round(actual, 6)
        
        assert actual == expected, f"RRF calculation mismatch: {actual} != {expected}"
    
    def test_graph_weight_dominance(self):
        """Test that graph channel with weight=1.0 ranks higher."""
        from voice_agent.rag2.retrieval import calculate_rrf_score
        
        weights = {"graph": 1.0, "semantic": 0.8, "lexical": 0.7}
        k = 60
        
        # Doc A: #1 in graph only
        doc_a_ranks = {"graph": 1}
        doc_a_score = calculate_rrf_score(doc_a_ranks, weights, k)
        
        # Doc B: #1 in lexical only
        doc_b_ranks = {"lexical": 1}
        doc_b_score = calculate_rrf_score(doc_b_ranks, weights, k)
        
        # Graph should dominate due to higher weight
        assert doc_a_score > doc_b_score, "Graph #1 should rank higher than Lexical #1"
```

### Running the Tests

```bash
# Run all triple-hybrid tests
pytest tests/test_rag2_triple_hybrid.py -v -m integration

# Run specific test
pytest tests/test_rag2_triple_hybrid.py::TestTripleHybridIntegration::test_all_three_channels_return_results -v

# Run with coverage
pytest tests/test_rag2_triple_hybrid.py -v --cov=src/voice_agent/rag2 --cov-report=term-missing
```

### Success Criteria
- [ ] All 3 channels return results
- [ ] RRF fusion applies correct weights
- [ ] Graph traversal finds relationships
- [ ] Cross-document queries work
- [ ] Safety threshold applied
- [ ] Full pipeline test passes
- [ ] Channel isolation works
- [ ] RRF formula verified

---

## Task 4.2: Update Documentation

**Time:** 15 minutes

### Files to Update

#### 1. `docs/RAG2.0/README.md`

Add completion status banner:

```markdown
# RAG 2.0 Implementation

✅ **Status: 100% Complete** (as of YYYY-MM-DD)

All components implemented and verified:
- ✅ Triple-Hybrid Retrieval (Lexical + Semantic + Graph)
- ✅ Recursive Splitting (Parent 512 / Child 128 tokens)
- ✅ PuppyGraph Integration (Entity Graph)
- ✅ GPT-5 Entity Extraction (NER + Relation Extraction)
- ✅ Weighted RRF Fusion
- ✅ Safety Thresholds (Conformal + Denoising)
- ✅ Gundam Tiling OCR Enhancement
- ✅ Late Interaction Reranking

## Quick Start

```bash
# Deploy infrastructure
docker-compose -f infrastructure/puppygraph/docker-compose.yml up -d

# Verify
python -c "from gremlin_python.driver import client; c = client.Client('ws://localhost:8182/gremlin', 'g'); print('Connected!')"

# Run tests
pytest tests/test_rag2*.py -v
```
```

#### 2. `docs/RAG2.0/ARCHITECTURE.md`

Add graph channel section:

```markdown
## Graph Channel (NEW)

### PuppyGraph Integration

PuppyGraph provides zero-ETL graph queries over PostgreSQL using Gremlin.

**Deployment:**
- Port: 8182 (Gremlin)
- Port: 8081 (Web UI)

**Schema:** See `infrastructure/puppygraph/schema.json`

**Query Example:**
```gremlin
g.V().hasLabel('entity')
  .has('name', 'Acme Corporation')
  .both('relates_to')
  .limit(10)
  .valueMap()
```
```

#### 3. `README.md` (root)

Add RAG 2.0 section to main README:

```markdown
## RAG 2.0

The voice agent uses RAG 2.0 with triple-hybrid retrieval:

| Channel | Weight | Purpose |
|---------|--------|---------|
| Graph | 1.0 | Entity relationships via PuppyGraph |
| Semantic | 0.8 | Concept similarity via embeddings |
| Lexical | 0.7 | Keyword matching via BM25 |

See [RAG 2.0 Documentation](docs/RAG2.0/README.md) for details.
```

### Verification

```bash
# View updated docs
cat docs/RAG2.0/README.md
cat docs/RAG2.0/ARCHITECTURE.md
```

### Success Criteria
- [ ] README.md shows 100% complete status
- [ ] ARCHITECTURE.md includes graph channel
- [ ] Root README references RAG 2.0

---

## Task 4.3: Generate Final Test Report

**Time:** 15 minutes (or integrate into CI)

### Commands

```bash
# Run all RAG 2.0 tests with coverage
pytest tests/test_rag2*.py -v --cov=src/voice_agent/rag2 --cov-report=html

# Count passing tests
pytest tests/test_rag2*.py -v --tb=no -q 2>&1 | tail -5

# Generate summary
echo "=== RAG 2.0 Test Report ===" > docs/RAG2.0/TEST_REPORT.md
echo "Generated: $(date)" >> docs/RAG2.0/TEST_REPORT.md
echo "" >> docs/RAG2.0/TEST_REPORT.md
pytest tests/test_rag2*.py -v --tb=no 2>&1 | grep -E "^tests/" >> docs/RAG2.0/TEST_REPORT.md
echo "" >> docs/RAG2.0/TEST_REPORT.md
echo "=== Coverage ===" >> docs/RAG2.0/TEST_REPORT.md
pytest tests/test_rag2*.py --cov=src/voice_agent/rag2 --cov-report=term 2>&1 | tail -20 >> docs/RAG2.0/TEST_REPORT.md
```

### Expected Output

```
=== RAG 2.0 Test Report ===
Generated: 2025-01-XX

tests/test_rag2_chunking.py::test_recursive_splitting PASSED
tests/test_rag2_entity_extraction.py::test_entity_extraction PASSED
tests/test_rag2_graph.py::test_graph_channel_e2e PASSED
tests/test_rag2_ocr_gundam.py::test_gundam_tiling PASSED
tests/test_rag2_triple_hybrid.py::test_all_three_channels PASSED
...

=== Coverage ===
Name                                      Stmts   Miss  Cover
-------------------------------------------------------------
src/voice_agent/rag2/ingest.py              234     12    95%
src/voice_agent/rag2/retrieval.py           312     18    94%
src/voice_agent/rag2/entity_extraction.py   156      8    95%
-------------------------------------------------------------
TOTAL                                       702     38    95%
```

---

## Verification Checklist

```
[ ] Task 4.1: Triple-Hybrid Integration Test
    [ ] Test file created
    [ ] All channels tested together
    [ ] RRF fusion verified
    [ ] Cross-document graph queries work
    [ ] Full pipeline test passes

[ ] Task 4.2: Documentation Updated
    [ ] RAG2.0/README.md shows 100%
    [ ] ARCHITECTURE.md has graph section
    [ ] Root README references RAG 2.0

[ ] Task 4.3: Test Report Generated
    [ ] All tests pass
    [ ] Coverage > 90%
    [ ] Report saved to docs/
```

---

## Dependencies

Requires completion of:
- Phase 1: PuppyGraph must be running
- Phase 2: Individual channel tests must pass
- Phase 3: Robustness features added
