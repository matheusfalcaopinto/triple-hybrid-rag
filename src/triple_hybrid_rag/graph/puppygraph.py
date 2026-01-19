"""
PuppyGraph Client for Triple-Hybrid-RAG

Native Cypher queries via Bolt protocol (Neo4j compatible).
This replaces the SQL fallback from the original voice-agent implementation.

PuppyGraph provides a graph layer over PostgreSQL, exposing:
- Bolt protocol on port 7697 (configurable)
- Full Cypher query support
- Real-time sync with PostgreSQL tables
"""

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession
from neo4j.exceptions import ServiceUnavailable, AuthError

from triple_hybrid_rag.config import RAGConfig, get_settings
from triple_hybrid_rag.types import Entity, Relation, SearchResult, Modality

logger = logging.getLogger(__name__)

class PuppyGraphClient:
    """
    Native PuppyGraph client using Bolt/Cypher protocol.
    
    PuppyGraph is a graph database that reads from PostgreSQL tables
    and exposes them as a property graph via Bolt protocol.
    
    This client provides:
    - Entity neighborhood traversal
    - Relation path finding
    - Chunk retrieval via graph connections
    - Full Cypher query support
    """
    
    def __init__(self, config: Optional[RAGConfig] = None):
        """Initialize the PuppyGraph client."""
        self.config = config or get_settings()
        self.bolt_url = self.config.puppygraph_bolt_url
        self.username = self.config.puppygraph_username
        self.password = self.config.puppygraph_password
        self.timeout = self.config.puppygraph_timeout
        self.enabled = self.config.rag_graph_enabled
        
        self._driver: Optional[AsyncDriver] = None
        self._connected = False
    
    async def connect(self) -> bool:
        """
        Connect to PuppyGraph via Bolt protocol.
        
        Returns:
            True if connection successful, False otherwise
        """
        if not self.enabled:
            logger.info("Graph search disabled, skipping PuppyGraph connection")
            return False
        
        try:
            self._driver = AsyncGraphDatabase.driver(
                self.bolt_url,
                auth=(self.username, self.password),
                connection_timeout=self.timeout,
            )
            
            # Verify connection
            async with self._driver.session() as session:
                result = await session.run("RETURN 1 AS test")
                await result.single()
            
            self._connected = True
            logger.info(f"Connected to PuppyGraph at {self.bolt_url}")
            return True
            
        except (ServiceUnavailable, AuthError) as e:
            logger.error(f"Failed to connect to PuppyGraph: {e}")
            self._connected = False
            return False
        except Exception as e:
            logger.error(f"PuppyGraph connection error: {e}")
            self._connected = False
            return False
    
    async def close(self):
        """Close the PuppyGraph connection."""
        if self._driver:
            await self._driver.close()
            self._driver = None
            self._connected = False
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to PuppyGraph."""
        return self._connected and self._driver is not None
    
    async def _get_session(self) -> AsyncSession:
        """Get a session, connecting if necessary."""
        if not self.is_connected:
            connected = await self.connect()
            if not connected:
                raise ConnectionError("Cannot connect to PuppyGraph")
        
        return self._driver.session()
    
    async def query_cypher(
        self,
        cypher: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query and return results.
        
        Args:
            cypher: Cypher query string
            params: Query parameters
            
        Returns:
            List of result records as dictionaries
        """
        params = params or {}
        
        async with await self._get_session() as session:
            result = await session.run(cypher, params)
            records = await result.data()
            return records
    
    async def entity_neighborhood(
        self,
        entity_id: str,
        tenant_id: str,
        hops: int = 2,
        limit: int = 50,
    ) -> List[SearchResult]:
        """
        Get chunks connected to an entity via graph traversal.
        
        Traverses the graph from an entity node, following relationships
        up to N hops, and returns connected chunks.
        
        Args:
            entity_id: UUID of the entity
            tenant_id: Tenant ID for filtering
            hops: Maximum traversal depth (default 2)
            limit: Maximum results
            
        Returns:
            List of SearchResult objects with graph scores
        """
        cypher = """
        MATCH (e:Entity {id: $entity_id, tenant_id: $tenant_id})
        MATCH path = (e)-[*1..{hops}]-(related:Entity)
        MATCH (related)-[:MENTIONED_IN]->(c:Chunk)
        WITH DISTINCT c, length(path) as distance
        RETURN 
            c.id as chunk_id,
            c.parent_id as parent_id,
            c.document_id as document_id,
            c.text as text,
            c.page as page,
            c.modality as modality,
            1.0 / (1.0 + distance) as graph_score
        ORDER BY graph_score DESC
        LIMIT $limit
        """.replace("{hops}", str(hops))
        
        try:
            records = await self.query_cypher(cypher, {
                "entity_id": entity_id,
                "tenant_id": tenant_id,
                "limit": limit,
            })
            
            return [self._record_to_search_result(r) for r in records]
            
        except Exception as e:
            logger.error(f"Entity neighborhood query failed: {e}")
            return []
    
    async def entity_lookup(
        self,
        entity_name: str,
        tenant_id: str,
        entity_type: Optional[str] = None,
        limit: int = 50,
    ) -> List[SearchResult]:
        """
        Find chunks that mention a specific entity by name.
        
        Args:
            entity_name: Name to search for (case-insensitive)
            tenant_id: Tenant ID for filtering
            entity_type: Optional entity type filter
            limit: Maximum results
            
        Returns:
            List of SearchResult objects
        """
        type_filter = "AND e.entity_type = $entity_type" if entity_type else ""
        
        cypher = f"""
        MATCH (e:Entity)
        WHERE e.tenant_id = $tenant_id
          AND (
            toLower(e.name) CONTAINS toLower($entity_name)
            OR toLower(e.canonical_name) CONTAINS toLower($entity_name)
          )
          {type_filter}
        MATCH (e)-[:MENTIONED_IN]->(c:Chunk)
        RETURN DISTINCT
            c.id as chunk_id,
            c.parent_id as parent_id,
            c.document_id as document_id,
            c.text as text,
            c.page as page,
            c.modality as modality,
            1.0 as graph_score
        LIMIT $limit
        """
        
        params = {
            "entity_name": entity_name,
            "tenant_id": tenant_id,
            "limit": limit,
        }
        if entity_type:
            params["entity_type"] = entity_type
        
        try:
            records = await self.query_cypher(cypher, params)
            return [self._record_to_search_result(r) for r in records]
        except Exception as e:
            logger.error(f"Entity lookup query failed: {e}")
            return []
    
    async def relation_path(
        self,
        subject_name: str,
        object_name: str,
        tenant_id: str,
        max_hops: int = 4,
    ) -> List[SearchResult]:
        """
        Find chunks along the path between two entities.
        
        Uses shortest path algorithm to find connections.
        
        Args:
            subject_name: Name of the subject entity
            object_name: Name of the object entity
            tenant_id: Tenant ID for filtering
            max_hops: Maximum path length
            
        Returns:
            List of SearchResult objects from chunks along the path
        """
        cypher = """
        MATCH (s:Entity), (o:Entity)
        WHERE s.tenant_id = $tenant_id
          AND o.tenant_id = $tenant_id
          AND (toLower(s.name) CONTAINS toLower($subject) OR toLower(s.canonical_name) CONTAINS toLower($subject))
          AND (toLower(o.name) CONTAINS toLower($object) OR toLower(o.canonical_name) CONTAINS toLower($object))
        MATCH path = shortestPath((s)-[*1..{max_hops}]-(o))
        UNWIND nodes(path) as n
        MATCH (n)-[:MENTIONED_IN]->(c:Chunk)
        RETURN DISTINCT
            c.id as chunk_id,
            c.parent_id as parent_id,
            c.document_id as document_id,
            c.text as text,
            c.page as page,
            c.modality as modality,
            1.0 as graph_score
        """.replace("{max_hops}", str(max_hops))
        
        try:
            records = await self.query_cypher(cypher, {
                "subject": subject_name,
                "object": object_name,
                "tenant_id": tenant_id,
            })
            return [self._record_to_search_result(r) for r in records]
        except Exception as e:
            logger.error(f"Relation path query failed: {e}")
            return []
    
    async def related_entities(
        self,
        entity_name: str,
        tenant_id: str,
        relation_type: Optional[str] = None,
        limit: int = 20,
    ) -> List[Entity]:
        """
        Find entities related to a given entity.
        
        Args:
            entity_name: Name of the source entity
            tenant_id: Tenant ID for filtering
            relation_type: Optional relation type filter
            limit: Maximum results
            
        Returns:
            List of related Entity objects
        """
        type_filter = "AND type(r) = $relation_type" if relation_type else ""
        
        cypher = f"""
        MATCH (e:Entity)-[r]-(related:Entity)
        WHERE e.tenant_id = $tenant_id
          AND (toLower(e.name) CONTAINS toLower($entity_name) OR toLower(e.canonical_name) CONTAINS toLower($entity_name))
          {type_filter}
        RETURN DISTINCT
            related.id as id,
            related.name as name,
            related.canonical_name as canonical_name,
            related.entity_type as entity_type,
            related.description as description,
            type(r) as relation
        LIMIT $limit
        """
        
        params = {
            "entity_name": entity_name,
            "tenant_id": tenant_id,
            "limit": limit,
        }
        if relation_type:
            params["relation_type"] = relation_type
        
        try:
            records = await self.query_cypher(cypher, params)
            return [self._record_to_entity(r) for r in records]
        except Exception as e:
            logger.error(f"Related entities query failed: {e}")
            return []
    
    async def search_by_keywords_graph(
        self,
        keywords: List[str],
        tenant_id: str,
        limit: int = 50,
    ) -> List[SearchResult]:
        """
        Search for chunks via entity name matching with keywords.
        
        This uses graph structure to find relevant chunks through
        entity connections rather than direct text search.
        
        Args:
            keywords: List of keywords to match against entity names
            tenant_id: Tenant ID for filtering
            limit: Maximum results
            
        Returns:
            List of SearchResult objects
        """
        # Create keyword pattern for matching
        keyword_conditions = " OR ".join([
            f"toLower(e.name) CONTAINS toLower('{kw}')" 
            for kw in keywords
        ])
        
        # PuppyGraph doesn't support DISTINCT with aggregates
        # Use simple query without aggregation
        cypher = f"""
        MATCH (e:Entity)
        WHERE e.tenant_id = $tenant_id
          AND ({keyword_conditions})
        MATCH (e)-[:MENTIONED_IN]->(c:Chunk)
        RETURN
            c.id as chunk_id,
            c.parent_id as parent_id,
            c.document_id as document_id,
            c.text as text,
            c.page as page,
            c.modality as modality,
            e.name as entity_name
        LIMIT $limit
        """
        
        try:
            records = await self.query_cypher(cypher, {
                "tenant_id": tenant_id,
                "limit": limit,
            })
            
            # Deduplicate by chunk_id and count matches
            chunk_map: Dict[str, Dict] = {}
            for r in records:
                cid = r.get("chunk_id")
                if cid not in chunk_map:
                    chunk_map[cid] = {"record": r, "count": 0}
                chunk_map[cid]["count"] += 1
            
            results = []
            for data in sorted(chunk_map.values(), key=lambda x: x["count"], reverse=True):
                result = self._record_to_search_result(data["record"])
                result.graph_score = data["count"] / len(keywords)
                results.append(result)
            
            return results[:limit]
        except Exception as e:
            logger.error(f"Keyword graph search failed: {e}")
            return []
    
    async def execute_query_plan_cypher(
        self,
        cypher_query: str,
        tenant_id: str,
        limit: int = 50,
    ) -> List[SearchResult]:
        """
        Execute a Cypher query from the query planner.
        
        The query planner may generate custom Cypher queries for
        complex relational queries.
        
        Args:
            cypher_query: Cypher query from query planner
            tenant_id: Tenant ID for filtering
            limit: Maximum results
            
        Returns:
            List of SearchResult objects
        """
        # Inject tenant filter and limit if not present
        if "$tenant_id" in cypher_query and "LIMIT" not in cypher_query.upper():
            cypher_query += f" LIMIT {limit}"
        
        try:
            records = await self.query_cypher(cypher_query, {
                "tenant_id": tenant_id,
                "limit": limit,
            })
            
            # Try to extract chunk info from results
            results = []
            for r in records:
                if "chunk_id" in r:
                    results.append(self._record_to_search_result(r))
            
            return results
        except Exception as e:
            logger.error(f"Query planner Cypher failed: {e}")
            return []
    
    def _record_to_search_result(self, record: Dict[str, Any]) -> SearchResult:
        """Convert a Cypher result record to SearchResult."""
        modality = record.get("modality", "text")
        if isinstance(modality, str):
            modality = Modality(modality) if modality in [m.value for m in Modality] else Modality.TEXT
        
        return SearchResult(
            chunk_id=UUID(record["chunk_id"]) if record.get("chunk_id") else None,
            parent_id=UUID(record["parent_id"]) if record.get("parent_id") else None,
            document_id=UUID(record["document_id"]) if record.get("document_id") else None,
            text=record.get("text", ""),
            page=record.get("page"),
            modality=modality,
            graph_score=record.get("graph_score", 1.0),
            source_channel="graph",
        )
    
    def _record_to_entity(self, record: Dict[str, Any]) -> Entity:
        """Convert a Cypher result record to Entity."""
        from triple_hybrid_rag.types import EntityType
        
        entity_type_str = record.get("entity_type", "CONCEPT")
        try:
            entity_type = EntityType(entity_type_str)
        except ValueError:
            entity_type = EntityType.CONCEPT
        
        return Entity(
            id=UUID(record["id"]) if record.get("id") else None,
            name=record.get("name", ""),
            canonical_name=record.get("canonical_name", ""),
            entity_type=entity_type,
            description=record.get("description"),
        )
