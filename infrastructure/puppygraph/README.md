# PuppyGraph for RAG 2.0 Knowledge Graph

PuppyGraph provides a Cypher query interface over the PostgreSQL-based knowledge graph tables, enabling the third channel (Graph) in the triple-hybrid RAG retrieval system.

## Architecture

```
                    ┌─────────────────────┐
                    │   RAG 2.0 Retrieval │
                    │      Pipeline       │
                    └──────────┬──────────┘
                               │
           ┌───────────────────┼───────────────────┐
           │                   │                   │
           ▼                   ▼                   ▼
    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
    │   Lexical    │    │   Semantic   │    │    Graph     │
    │    (BM25)    │    │   (HNSW)     │    │   (Cypher)   │
    └──────────────┘    └──────────────┘    └──────┬───────┘
           │                   │                   │
           ▼                   ▼                   ▼
    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
    │  PostgreSQL  │    │  PostgreSQL  │    │  PuppyGraph  │
    │  FTS (GIN)   │    │ pgvector     │    │   Cypher     │
    └──────────────┘    └──────────────┘    └──────┬───────┘
                                                   │
                                           ┌───────▼───────┐
                                           │  PostgreSQL   │
                                           │ rag_entities  │
                                           │ rag_relations │
                                           └───────────────┘
```

## Quick Start

### 1. Configure Environment

```bash
# Add to .env file
export PUPPYGRAPH_USERNAME=admin
export PUPPYGRAPH_PASSWORD=puppygraph123
export SUPABASE_DB_HOST=host.docker.internal  # or your Supabase host
export SUPABASE_DB_PORT=54322
export SUPABASE_DB_NAME=postgres
export SUPABASE_DB_USER=postgres
export SUPABASE_DB_PASSWORD=your_password

# Enable graph channel
export RAG2_GRAPH_ENABLED=true
export RAG2_PUPPYGRAPH_URL=http://localhost:8182
```

### 2. Start PuppyGraph

```bash
cd infrastructure/puppygraph
docker-compose up -d
```

### 3. Verify Setup

```bash
# Check health
curl http://localhost:8081/health

# Access Web UI
open http://localhost:8081
```

## Schema Overview

### Vertices (Nodes)

| Label | Source Table | Description |
|-------|-------------|-------------|
| `Entity` | `rag_entities` | Named entities (PERSON, ORG, PRODUCT, etc) |
| `Document` | `rag_documents` | Ingested documents |
| `ParentChunk` | `rag_parent_chunks` | Parent context chunks (800-1000 tokens) |
| `ChildChunk` | `rag_child_chunks` | Retrieval units (~200 tokens) |

### Edges (Relationships)

| Label | Source → Target | Description |
|-------|-----------------|-------------|
| `RELATES_TO` | Entity → Entity | Knowledge graph relations |
| `MENTIONED_IN` | Entity → ChildChunk | Entity mentions in text |
| `BELONGS_TO` | Entity → Document | Entity origin document |
| `HAS_PARENT` | ChildChunk → ParentChunk | Chunk hierarchy |
| `FROM_DOCUMENT` | ParentChunk → Document | Document provenance |

## Example Cypher Queries

### Find entities related to a topic

```cypher
MATCH (e:Entity)-[r:RELATES_TO]->(other:Entity)
WHERE e.name CONTAINS 'payment'
RETURN e, r, other
LIMIT 10
```

### Find chunks mentioning specific entities

```cypher
MATCH (e:Entity)-[:MENTIONED_IN]->(c:ChildChunk)-[:HAS_PARENT]->(p:ParentChunk)
WHERE e.entity_type = 'PERSON' AND e.name = 'John Smith'
RETURN p.id, p.section_heading, c.id
```

### Multi-hop traversal for related context

```cypher
MATCH path = (e1:Entity)-[:RELATES_TO*1..3]-(e2:Entity)
WHERE e1.name = 'Contract X'
UNWIND nodes(path) AS entity
MATCH (entity)-[:MENTIONED_IN]->(c:ChildChunk)
RETURN DISTINCT c.id AS chunk_id
LIMIT 50
```

### Find all entities in a document

```cypher
MATCH (d:Document)<-[:BELONGS_TO]-(e:Entity)
WHERE d.title = 'API Reference'
RETURN e.entity_type, e.name, e.description
ORDER BY e.entity_type, e.name
```

## Integration with RAG 2.0

The graph channel is automatically used when:

1. `RAG2_GRAPH_ENABLED=true` in environment
2. `RAG2_PUPPYGRAPH_URL` is set
3. Query planner determines `requires_graph=true`

### Query Planner Integration

The GPT-5 query planner analyzes each query and generates:
- `keywords`: For lexical search
- `semantic_query_text`: For vector search
- `cypher_query`: For graph traversal (when beneficial)

Example query plan for "What are the payment terms in contract X?":

```json
{
  "keywords": ["payment", "terms", "contract X"],
  "semantic_query_text": "Payment terms and conditions in contract X",
  "cypher_query": "MATCH (c:Entity {entity_type:'CONTRACT'})-[:RELATES_TO]->(clause:Entity) WHERE c.name CONTAINS 'Contract X' AND clause.entity_type='CLAUSE' RETURN clause",
  "requires_graph": true,
  "weights": {"lexical": 0.7, "semantic": 0.8, "graph": 1.0}
}
```

## Weighted RRF Fusion

Results from all three channels are combined using Weighted Reciprocal Rank Fusion:

$$\text{score} = \sum_{c \in \text{channels}} \frac{w_c}{k + \text{rank}_c}$$

Default weights (prioritize Graph for relational queries):
- **Graph**: 1.0
- **Semantic**: 0.8
- **Lexical**: 0.7

## Troubleshooting

### PuppyGraph won't start

```bash
# Check logs
docker-compose logs puppygraph

# Verify PostgreSQL connection
docker exec puppygraph curl http://localhost:8081/health
```

### No results from graph queries

1. Ensure entities/relations are populated (see NER pipeline)
2. Check schema.json matches your table structure
3. Verify `org_id` filtering in queries

### Slow queries

```bash
# Increase memory
JAVA_OPTS="-Xms1g -Xmx4g" docker-compose up -d

# Enable query caching in schema.json
```

## Ports

| Port | Protocol | Purpose |
|------|----------|---------|
| 8182 | HTTP | Gremlin/Cypher API |
| 8081 | HTTP | Web UI |
| 7687 | Bolt | Neo4j-compatible protocol |

## Resources

- [PuppyGraph Documentation](https://docs.puppygraph.com/)
- [Cypher Query Language](https://neo4j.com/developer/cypher/)
- [RAG 2.0 Walkthrough](../../docs/RAG2.0/WALKTHROUGH.md)
