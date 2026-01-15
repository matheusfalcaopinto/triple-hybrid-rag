"""
RAG2 Knowledge Graph Search Module

Integrates with PuppyGraph for Cypher-based graph traversal.
Falls back to SQL-based relation queries if PuppyGraph unavailable.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

from voice_agent.config import SETTINGS

logger = logging.getLogger(__name__)


@dataclass
class GraphNode:
    """Represents a node in the knowledge graph."""
    
    id: str
    label: str  # Entity type: person, organization, product, etc.
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self) -> int:
        return hash(self.id)


@dataclass
class GraphEdge:
    """Represents an edge/relationship in the knowledge graph."""
    
    source_id: str
    target_id: str
    relationship: str
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0


@dataclass
class GraphSearchResult:
    """Result from graph traversal."""
    
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    paths: List[List[str]]  # List of paths (each path is list of node IDs)
    chunk_ids: List[str]  # Related chunk IDs for context expansion
    source: str  # "puppygraph" or "sql_fallback"


class PuppyGraphClient:
    """
    Client for PuppyGraph Cypher queries.
    
    PuppyGraph provides graph database capabilities over PostgreSQL.
    https://www.puppygraph.com/
    """
    
    def __init__(
        self,
        endpoint: Optional[str] = None,
        timeout: float = 30.0,
    ):
        self.endpoint = endpoint or SETTINGS.rag2_puppygraph_url
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
    
    @property
    def enabled(self) -> bool:
        """Check if PuppyGraph is configured."""
        return bool(self.endpoint)
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
            )
        return self._client
    
    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    async def execute_cypher(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute a Cypher query against PuppyGraph.
        
        Args:
            query: Cypher query string
            parameters: Query parameters
            
        Returns:
            Query results as dict
        """
        if not self.enabled:
            raise RuntimeError("PuppyGraph not configured")
        
        client = await self._get_client()
        
        payload = {
            "query": query,
            "parameters": parameters or {},
        }
        
        try:
            response = await client.post(
                f"{self.endpoint}/cypher",
                json=payload,
            )
            response.raise_for_status()
            result: Dict[str, Any] = response.json()
            return result
        except httpx.HTTPError as e:
            logger.error(f"PuppyGraph query failed: {e}")
            raise
    
    async def health_check(self) -> bool:
        """Check if PuppyGraph is available."""
        if not self.enabled:
            return False
        
        try:
            client = await self._get_client()
            response = await client.get(f"{self.endpoint}/health")
            return response.status_code == 200
        except Exception:
            return False


class SQLGraphFallback:
    """
    SQL-based graph traversal fallback.
    
    Uses the entities/relations tables directly when PuppyGraph unavailable.
    """
    
    def __init__(self, supabase_client: Any):
        self.db = supabase_client
    
    async def find_entities(
        self,
        keywords: List[str],
        org_id: str,
        limit: int = 20,
    ) -> List[GraphNode]:
        """Find entities matching keywords."""
        # Build OR query for keywords
        nodes: List[GraphNode] = []
        
        for keyword in keywords[:5]:  # Limit keywords
            try:
                kw = keyword  # Bind loop variable for lambda
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda kw=kw: self.db.table("rag_entities")
                    .select("id, entity_type, name, metadata")
                    .eq("org_id", org_id)
                    .ilike("name", f"%{kw}%")
                    .limit(limit // len(keywords))
                    .execute()
                )
                
                for row in response.data:
                    nodes.append(GraphNode(
                        id=row["id"],
                        label=row["entity_type"],
                        properties={
                            "name": row["name"],
                            **(row.get("metadata") or {}),
                        },
                    ))
            except Exception as e:
                logger.warning(f"Entity search failed for '{keyword}': {e}")
        
        return nodes
    
    async def find_relations(
        self,
        entity_ids: List[str],
        org_id: str,
        limit: int = 50,
    ) -> List[GraphEdge]:
        """Find relations involving given entities."""
        if not entity_ids:
            return []
        
        edges: List[GraphEdge] = []
        select_cols = (
            "id, subject_entity_id, object_entity_id, "
            "relation_type, metadata, confidence"
        )
        
        try:
            # Find outgoing relations
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.db.table("rag_relations")
                .select(select_cols)
                .eq("org_id", org_id)
                .in_("subject_entity_id", entity_ids)
                .limit(limit)
                .execute()
            )
            
            for row in response.data:
                edges.append(GraphEdge(
                    source_id=row["subject_entity_id"],
                    target_id=row["object_entity_id"],
                    relationship=row["relation_type"],
                    properties=row.get("metadata") or {},
                    confidence=row.get("confidence", 1.0),
                ))
            
            # Find incoming relations
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.db.table("rag_relations")
                .select(select_cols)
                .eq("org_id", org_id)
                .in_("object_entity_id", entity_ids)
                .limit(limit)
                .execute()
            )
            
            for row in response.data:
                edges.append(GraphEdge(
                    source_id=row["subject_entity_id"],
                    target_id=row["object_entity_id"],
                    relationship=row["relation_type"],
                    properties=row.get("metadata") or {},
                    confidence=row.get("confidence", 1.0),
                ))
        except Exception as e:
            logger.error(f"Relation search failed: {e}")
        
        return edges
    
    async def find_chunks_for_entities(
        self,
        entity_ids: List[str],
        limit: int = 20,
    ) -> List[str]:
        """Find chunk IDs related to entities."""
        if not entity_ids:
            return []
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.db.table("rag_entity_mentions")
                .select("child_chunk_id")
                .in_("entity_id", entity_ids)
                .limit(limit)
                .execute()
            )
            
            return list({row["child_chunk_id"] for row in response.data})
        except Exception as e:
            logger.error(f"Entity-chunk lookup failed: {e}")
            return []


class GraphSearcher:
    """
    Knowledge graph searcher for RAG2.
    
    Supports both PuppyGraph (Cypher) and SQL fallback modes.
    """
    
    def __init__(
        self,
        supabase_client: Any,
        puppygraph_url: Optional[str] = None,
    ):
        self.db = supabase_client
        self.puppygraph = PuppyGraphClient(puppygraph_url)
        self.sql_fallback = SQLGraphFallback(supabase_client)
    
    async def search(
        self,
        keywords: List[str],
        cypher_query: Optional[str],
        org_id: str,
        top_k: int = 20,
    ) -> GraphSearchResult:
        """
        Execute graph search.
        
        Args:
            keywords: Keywords for entity matching
            cypher_query: Optional Cypher query from query planner
            org_id: Organization ID
            top_k: Maximum results
            
        Returns:
            GraphSearchResult with nodes, edges, and related chunk IDs
        """
        # Try PuppyGraph first if available and we have a Cypher query
        if cypher_query and self.puppygraph.enabled:
            try:
                if await self.puppygraph.health_check():
                    return await self._search_puppygraph(cypher_query, org_id, top_k)
            except Exception as e:
                logger.warning(f"PuppyGraph search failed, using SQL fallback: {e}")
        
        # Fall back to SQL-based search
        return await self._search_sql(keywords, org_id, top_k)
    
    async def _search_puppygraph(
        self,
        cypher_query: str,
        org_id: str,
        top_k: int,
    ) -> GraphSearchResult:
        """Execute search via PuppyGraph Cypher."""
        # Add org_id filter to query if not present
        if "$org_id" in cypher_query:
            params = {"org_id": org_id}
        else:
            params = {}
        
        result = await self.puppygraph.execute_cypher(cypher_query, params)
        
        nodes: List[GraphNode] = []
        edges: List[GraphEdge] = []
        chunk_ids: List[str] = []
        paths: List[List[str]] = []
        
        # Parse PuppyGraph response
        for row in result.get("results", []):
            # Handle nodes
            for node_data in row.get("nodes", []):
                nodes.append(GraphNode(
                    id=node_data["id"],
                    label=node_data.get("labels", ["unknown"])[0],
                    properties=node_data.get("properties", {}),
                ))
            
            # Handle relationships
            for rel_data in row.get("relationships", []):
                edges.append(GraphEdge(
                    source_id=rel_data["startNode"],
                    target_id=rel_data["endNode"],
                    relationship=rel_data.get("type", "RELATED_TO"),
                    properties=rel_data.get("properties", {}),
                ))
            
            # Handle paths
            if "path" in row:
                path_ids = [n["id"] for n in row["path"].get("nodes", [])]
                paths.append(path_ids)
            
            # Handle chunk references
            if "chunk_ids" in row:
                chunk_ids.extend(row["chunk_ids"])
        
        # Deduplicate
        nodes = list({n.id: n for n in nodes}.values())[:top_k]
        chunk_ids = list(set(chunk_ids))[:top_k]
        
        return GraphSearchResult(
            nodes=nodes,
            edges=edges,
            paths=paths,
            chunk_ids=chunk_ids,
            source="puppygraph",
        )
    
    async def _search_sql(
        self,
        keywords: List[str],
        org_id: str,
        top_k: int,
    ) -> GraphSearchResult:
        """Execute search via SQL fallback."""
        # Step 1: Find matching entities
        nodes = await self.sql_fallback.find_entities(keywords, org_id, top_k)
        
        if not nodes:
            return GraphSearchResult(
                nodes=[],
                edges=[],
                paths=[],
                chunk_ids=[],
                source="sql_fallback",
            )
        
        entity_ids = [n.id for n in nodes]
        
        # Step 2: Find relationships
        edges = await self.sql_fallback.find_relations(entity_ids, org_id, top_k * 2)
        
        # Step 3: Find related chunks
        chunk_ids = await self.sql_fallback.find_chunks_for_entities(entity_ids, top_k)
        
        # Build simple paths from relationships
        paths: List[List[str]] = []
        for edge in edges[:top_k]:
            paths.append([edge.source_id, edge.target_id])
        
        return GraphSearchResult(
            nodes=nodes,
            edges=edges,
            paths=paths,
            chunk_ids=chunk_ids,
            source="sql_fallback",
        )
    
    async def close(self) -> None:
        """Close clients."""
        await self.puppygraph.close()


# Factory function
def get_graph_searcher(supabase_client: Any) -> GraphSearcher:
    """Get a GraphSearcher instance."""
    return GraphSearcher(
        supabase_client=supabase_client,
        puppygraph_url=SETTINGS.rag2_puppygraph_url,
    )
