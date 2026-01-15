"""
RAG2 Graph Channel E2E Tests

Tests the Graph Search module with REAL database operations:
- GraphSearcher with SQL fallback
- Entity search across rag_entities table
- Relation traversal across rag_relations table
- Entity-chunk lookup in rag_entity_mentions

Requires:
- SUPABASE_URL and SERVICE_ROLE_KEY environment variables
- RAG2_GRAPH_ENABLED=true (optional, tests SQL fallback regardless)
"""
from __future__ import annotations

import asyncio
import os
import uuid
from typing import List

import pytest
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Skip if no database connection
pytestmark = pytest.mark.skipif(
    not os.getenv("SUPABASE_URL") or not os.getenv("SERVICE_ROLE_KEY"),
    reason="Requires SUPABASE_URL and SERVICE_ROLE_KEY environment variables"
)


# Test organization ID
TEST_ORG_ID = "00000000-0000-0000-0000-000000000001"
TEST_DOC_PREFIX = "graph-e2e-test-"


@pytest.fixture
def org_id() -> str:
    """Test organization ID."""
    return TEST_ORG_ID


@pytest.fixture
def supabase():
    """Get Supabase client."""
    from voice_agent.utils.db import get_supabase_client
    return get_supabase_client()


@pytest.fixture
def test_doc_id() -> str:
    """Generate unique document ID for test isolation.
    
    Note: We use a valid UUID since document_id is a foreign key.
    Tests that need a document should create one first.
    """
    return str(uuid.uuid4())


class TestGraphSearcherSQL:
    """Test GraphSearcher with SQL fallback (no PuppyGraph required)."""
    
    @pytest.fixture(autouse=True)
    async def cleanup(self, supabase, org_id, test_doc_id):
        """Cleanup test data after each test."""
        yield
        # Delete test data by name patterns used in tests
        test_entity_patterns = [
            "ACME%", "John Smith", "TechCorp%", "InnovateTech%",
            "Project Alpha", "Project Beta", "Contract 2026",
            "SecretCorp%"
        ]
        
        try:
            # First delete relations (foreign key constraints)
            for pattern in test_entity_patterns:
                # Get entity IDs matching the pattern
                result = supabase.table("rag_entities").select("id").ilike("name", pattern).execute()
                if result.data:
                    entity_ids = [e["id"] for e in result.data]
                    # Delete relations referencing these entities
                    for eid in entity_ids:
                        supabase.table("rag_relations").delete().eq("subject_entity_id", eid).execute()
                        supabase.table("rag_relations").delete().eq("object_entity_id", eid).execute()
        except Exception:
            pass
        
        try:
            # Delete test entities by name pattern
            for pattern in test_entity_patterns:
                supabase.table("rag_entities").delete().ilike("name", pattern).execute()
        except Exception:
            pass
    
    def test_graph_searcher_initialization(self, supabase):
        """Test that GraphSearcher initializes correctly."""
        from voice_agent.rag2 import get_graph_searcher
        
        searcher = get_graph_searcher(supabase)
        
        assert searcher is not None
        assert searcher.sql_fallback is not None
        assert searcher.puppygraph is not None
    
    @pytest.mark.asyncio
    async def test_search_returns_empty_for_nonexistent_entities(self, supabase, org_id):
        """Test that search returns empty result for non-matching keywords."""
        from voice_agent.rag2 import get_graph_searcher
        
        searcher = get_graph_searcher(supabase)
        
        result = await searcher.search(
            keywords=["nonexistent_entity_xyz_12345"],
            cypher_query=None,
            org_id=org_id,
            top_k=10,
        )
        
        assert result is not None
        assert result.source == "sql_fallback"
        assert len(result.nodes) == 0
        assert len(result.edges) == 0
        assert len(result.chunk_ids) == 0
    
    @pytest.mark.asyncio
    async def test_find_entities_by_keyword(self, supabase, org_id, test_doc_id):
        """Test entity search by keyword matching."""
        from voice_agent.rag2 import get_graph_searcher
        
        # Insert test entities (no confidence column in rag_entities, no document_id for simplicity)
        entities_data = [
            {
                "id": str(uuid.uuid4()),
                "org_id": org_id,
                "name": "ACME Corporation",
                "entity_type": "ORGANIZATION",
                "canonical_name": "ACME Corp",
            },
            {
                "id": str(uuid.uuid4()),
                "org_id": org_id,
                "name": "John Smith",
                "entity_type": "PERSON",
            },
        ]
        
        supabase.table("rag_entities").insert(entities_data).execute()
        
        # Search for entities
        searcher = get_graph_searcher(supabase)
        result = await searcher.search(
            keywords=["ACME"],
            cypher_query=None,
            org_id=org_id,
            top_k=10,
        )
        
        assert result is not None
        assert result.source == "sql_fallback"
        assert len(result.nodes) >= 1
        
        # Verify ACME was found
        node_names = [n.properties.get("name", "") for n in result.nodes]
        assert any("ACME" in name for name in node_names)
    
    @pytest.mark.asyncio
    async def test_find_relations_between_entities(self, supabase, org_id, test_doc_id):
        """Test relation search between entities."""
        from voice_agent.rag2 import get_graph_searcher
        
        # Insert test entities (no confidence column in rag_entities)
        entity1_id = str(uuid.uuid4())
        entity2_id = str(uuid.uuid4())
        
        entities_data = [
            {
                "id": entity1_id,
                "org_id": org_id,
                "name": "TechCorp Inc",
                "entity_type": "ORGANIZATION",
            },
            {
                "id": entity2_id,
                "org_id": org_id,
                "name": "InnovateTech LLC",
                "entity_type": "ORGANIZATION",
            },
        ]
        
        supabase.table("rag_entities").insert(entities_data).execute()
        
        # Insert relation
        relation_data = {
            "id": str(uuid.uuid4()),
            "org_id": org_id,
            "subject_entity_id": entity1_id,
            "object_entity_id": entity2_id,
            "relation_type": "PARTNERS_WITH",
            "confidence": 0.85,
        }
        
        supabase.table("rag_relations").insert(relation_data).execute()
        
        # Search for entities and relations
        searcher = get_graph_searcher(supabase)
        result = await searcher.search(
            keywords=["TechCorp"],
            cypher_query=None,
            org_id=org_id,
            top_k=10,
        )
        
        assert result is not None
        assert len(result.nodes) >= 1
        
        # Should find the relation
        assert len(result.edges) >= 1
        
        # Verify relation details
        edge = result.edges[0]
        assert edge.source_id == entity1_id or edge.target_id == entity1_id
        assert edge.relationship == "PARTNERS_WITH"
    
    @pytest.mark.asyncio
    async def test_search_with_multiple_keywords(self, supabase, org_id, test_doc_id):
        """Test search with multiple keywords."""
        from voice_agent.rag2 import get_graph_searcher
        
        # Insert multiple entities (no confidence column in rag_entities)
        entities_data = [
            {
                "id": str(uuid.uuid4()),
                "org_id": org_id,
                "name": "Project Alpha",
                "entity_type": "PROJECT",
            },
            {
                "id": str(uuid.uuid4()),
                "org_id": org_id,
                "name": "Project Beta",
                "entity_type": "PROJECT",
            },
            {
                "id": str(uuid.uuid4()),
                "org_id": org_id,
                "name": "Contract 2026",
                "entity_type": "CONTRACT",
            },
        ]
        
        supabase.table("rag_entities").insert(entities_data).execute()
        
        # Search with multiple keywords
        searcher = get_graph_searcher(supabase)
        result = await searcher.search(
            keywords=["Alpha", "Beta", "Contract"],
            cypher_query=None,
            org_id=org_id,
            top_k=10,
        )
        
        assert result is not None
        assert len(result.nodes) >= 2  # Should find at least 2 entities
    
    @pytest.mark.asyncio
    async def test_graph_search_result_structure(self, supabase, org_id):
        """Test that GraphSearchResult has correct structure."""
        from voice_agent.rag2 import get_graph_searcher, GraphSearchResult
        
        searcher = get_graph_searcher(supabase)
        result = await searcher.search(
            keywords=["test"],
            cypher_query=None,
            org_id=org_id,
            top_k=10,
        )
        
        # Verify result structure
        assert isinstance(result, GraphSearchResult)
        assert hasattr(result, "nodes")
        assert hasattr(result, "edges")
        assert hasattr(result, "paths")
        assert hasattr(result, "chunk_ids")
        assert hasattr(result, "source")
        assert isinstance(result.nodes, list)
        assert isinstance(result.edges, list)
        assert isinstance(result.paths, list)
        assert isinstance(result.chunk_ids, list)
        assert result.source in ["sql_fallback", "puppygraph"]
    
    @pytest.mark.asyncio
    async def test_org_isolation(self, supabase, test_doc_id):
        """Test that entities are isolated by org_id."""
        from voice_agent.rag2 import get_graph_searcher
        
        org1 = "00000000-0000-0000-0000-000000000001"
        org2 = "00000000-0000-0000-0000-000000000002"
        
        # Insert entity for org1 (no confidence column in rag_entities)
        entity_data = {
            "id": str(uuid.uuid4()),
            "org_id": org1,
            "name": "SecretCorp Exclusive",
            "entity_type": "ORGANIZATION",
        }
        
        supabase.table("rag_entities").insert(entity_data).execute()
        
        # Search from org2 should not find org1's entity
        searcher = get_graph_searcher(supabase)
        result = await searcher.search(
            keywords=["SecretCorp"],
            cypher_query=None,
            org_id=org2,
            top_k=10,
        )
        
        assert len(result.nodes) == 0  # Should not find entity from other org


class TestSQLGraphFallback:
    """Direct tests for SQLGraphFallback class."""
    
    @pytest.fixture(autouse=True)
    async def cleanup(self, supabase, test_doc_id):
        """Cleanup test data after each test."""
        yield
        # Clean up entities by name pattern
        try:
            supabase.table("rag_relations").delete().ilike(
                "relation_type", "%"
            ).execute()
        except Exception:
            pass
        
        try:
            # Delete test entities by name pattern
            supabase.table("rag_entities").delete().ilike(
                "name", "%Test%"
            ).execute()
        except Exception:
            pass
    
    @pytest.mark.asyncio
    async def test_find_entities_direct(self, supabase, org_id, test_doc_id):
        """Test SQLGraphFallback.find_entities directly."""
        from voice_agent.rag2.graph_search import SQLGraphFallback
        
        # Insert test entity (no confidence column in rag_entities)
        entity_id = str(uuid.uuid4())
        entity_data = {
            "id": entity_id,
            "org_id": org_id,
            "name": "DirectTest Company",
            "entity_type": "ORGANIZATION",
        }
        
        supabase.table("rag_entities").insert(entity_data).execute()
        
        # Test find_entities
        fallback = SQLGraphFallback(supabase)
        nodes = await fallback.find_entities(
            keywords=["DirectTest"],
            org_id=org_id,
            limit=10,
        )
        
        assert len(nodes) >= 1
        assert any(n.id == entity_id for n in nodes)
    
    @pytest.mark.asyncio
    async def test_find_relations_direct(self, supabase, org_id, test_doc_id):
        """Test SQLGraphFallback.find_relations directly."""
        from voice_agent.rag2.graph_search import SQLGraphFallback
        
        # Insert test entities and relation (no document_id required)
        entity1_id = str(uuid.uuid4())
        entity2_id = str(uuid.uuid4())
        
        entities_data = [
            {
                "id": entity1_id,
                "org_id": org_id,
                "name": "RelTest Subject",
                "entity_type": "ORGANIZATION",
            },
            {
                "id": entity2_id,
                "org_id": org_id,
                "name": "RelTest Object",
                "entity_type": "ORGANIZATION",
            },
        ]
        
        supabase.table("rag_entities").insert(entities_data).execute()
        
        relation_data = {
            "id": str(uuid.uuid4()),
            "org_id": org_id,
            "subject_entity_id": entity1_id,
            "object_entity_id": entity2_id,
            "relation_type": "OWNS",
            "confidence": 0.9,
        }
        
        supabase.table("rag_relations").insert(relation_data).execute()
        
        # Test find_relations
        fallback = SQLGraphFallback(supabase)
        edges = await fallback.find_relations(
            entity_ids=[entity1_id],
            org_id=org_id,
            limit=10,
        )
        
        assert len(edges) >= 1
        assert edges[0].source_id == entity1_id
        assert edges[0].target_id == entity2_id
        assert edges[0].relationship == "OWNS"


class TestPuppyGraphClient:
    """Tests for PuppyGraph client (connectivity checks)."""
    
    def test_puppygraph_client_initialization(self):
        """Test PuppyGraph client initializes correctly."""
        from voice_agent.rag2.graph_search import PuppyGraphClient
        from voice_agent.config import SETTINGS
        
        client = PuppyGraphClient()
        
        # Should use configured URL
        expected_url = SETTINGS.rag2_puppygraph_url
        assert client.endpoint == expected_url
    
    def test_puppygraph_enabled_check(self):
        """Test enabled property reflects configuration."""
        from voice_agent.rag2.graph_search import PuppyGraphClient
        from voice_agent.config import SETTINGS
        
        client = PuppyGraphClient()
        
        # Enabled if URL is configured
        if SETTINGS.rag2_puppygraph_url:
            assert client.enabled is True
        else:
            assert client.enabled is False
    
    @pytest.mark.asyncio
    async def test_health_check_returns_boolean(self):
        """Test health check returns boolean (regardless of status)."""
        from voice_agent.rag2.graph_search import PuppyGraphClient
        
        client = PuppyGraphClient()
        
        try:
            result = await client.health_check()
            assert isinstance(result, bool)
        finally:
            await client.close()


class TestGraphDataClasses:
    """Tests for graph data structures."""
    
    def test_graph_node_creation(self):
        """Test GraphNode dataclass."""
        from voice_agent.rag2 import GraphNode
        
        node = GraphNode(
            id="node-123",
            label="ORGANIZATION",
            properties={"name": "Test Corp"},
        )
        
        assert node.id == "node-123"
        assert node.label == "ORGANIZATION"
        assert node.properties["name"] == "Test Corp"
    
    def test_graph_node_hashable(self):
        """Test that GraphNode is hashable (for deduplication)."""
        from voice_agent.rag2 import GraphNode
        
        node1 = GraphNode(id="same-id", label="TYPE1")
        node2 = GraphNode(id="same-id", label="TYPE2")
        node3 = GraphNode(id="different-id", label="TYPE1")
        
        # Same ID should have same hash
        assert hash(node1) == hash(node2)
        # Different ID should have different hash
        assert hash(node1) != hash(node3)
    
    def test_graph_edge_creation(self):
        """Test GraphEdge dataclass."""
        from voice_agent.rag2 import GraphEdge
        
        edge = GraphEdge(
            source_id="entity-1",
            target_id="entity-2",
            relationship="EMPLOYS",
            confidence=0.95,
        )
        
        assert edge.source_id == "entity-1"
        assert edge.target_id == "entity-2"
        assert edge.relationship == "EMPLOYS"
        assert edge.confidence == 0.95
    
    def test_graph_search_result_creation(self):
        """Test GraphSearchResult dataclass."""
        from voice_agent.rag2 import GraphSearchResult, GraphNode, GraphEdge
        
        result = GraphSearchResult(
            nodes=[GraphNode(id="n1", label="TYPE")],
            edges=[GraphEdge(source_id="n1", target_id="n2", relationship="REL")],
            paths=[["n1", "n2"]],
            chunk_ids=["chunk-1", "chunk-2"],
            source="sql_fallback",
        )
        
        assert len(result.nodes) == 1
        assert len(result.edges) == 1
        assert len(result.paths) == 1
        assert len(result.chunk_ids) == 2
        assert result.source == "sql_fallback"
