# RAG 2.0 Architecture

This document describes the technical architecture of the RAG 2.0 retrieval system.

## Overview

RAG 2.0 implements a **Triple-Hybrid Retrieval** approach that combines:

1. **Lexical Search** - BM25/Full-Text Search for exact keyword matching
2. **Semantic Search** - Vector similarity via HNSW index
3. **Graph Search** - Entity relationship traversal via PuppyGraph (or SQL fallback)

Results are fused using **Weighted Reciprocal Rank Fusion (RRF)** and then reranked.

---

## System Components

### 1. Ingestion Pipeline

```
Document → Text Extraction → Chunking → Embedding → Entity Extraction → Storage
```

#### Text Extraction
- PDF: PyMuPDF
- Images: OCR with Gundam Tiling for large images
- Other: Standard text loaders

#### Chunking Strategy
- **Hierarchical**: Parent (800-1000 tokens) → Child (~200 tokens)
- **Recursive Character Splitting**: LangChain-style separators
- **Overlap**: Configurable token overlap between chunks

#### Embedding Generation
- **Model**: Matryoshka-enabled embedding model
- **Dimension**: 4096 → truncated to 1024
- **Normalization**: L2 normalized for cosine similarity

#### Entity Extraction
- **Model**: GPT-5 with structured output
- **Entity Types**: 10 categories (PERSON, ORGANIZATION, PRODUCT, etc.)
- **Relations**: Subject → Predicate → Object triples
- **Retry Logic**: 3 attempts with exponential backoff

### 2. Database Schema

```sql
-- Documents
rag_documents (
    id, org_id, document_hash, filename, metadata, ...
)

-- Chunks (hierarchical)
rag_parent_chunks (
    id, document_id, text, section_heading, page_numbers, ...
)

rag_child_chunks (
    id, parent_chunk_id, document_id, text, 
    embedding_1024, tsv, page_number, modality, ...
)

-- Knowledge Graph
rag_entities (
    id, org_id, entity_type, name, properties, ...
)

rag_entity_mentions (
    id, entity_id, child_chunk_id, span_text, ...
)

rag_relations (
    id, source_entity_id, target_entity_id, 
    relation_type, confidence, ...
)
```

### 3. Indexes

| Index | Table | Type | Purpose |
|-------|-------|------|---------|
| embedding_1024 | rag_child_chunks | HNSW | Semantic search |
| tsv | rag_child_chunks | GIN | Full-text search |
| org_id | All tables | B-tree | Tenant isolation |

---

## Retrieval Pipeline

### Step 1: Query Planning

GPT-5 analyzes the query and generates:
- **Keywords**: For lexical search
- **Semantic Query**: Rewritten for embedding
- **Cypher Query**: For graph traversal (if needed)
- **Channel Weights**: Dynamic weight adjustments

```python
@dataclass
class QueryPlan:
    original_query: str
    keywords: List[str]
    semantic_query_text: str
    cypher_query: Optional[str]
    weights: Dict[str, float]
    requires_graph: bool
```

### Step 2: Multi-Channel Retrieval

All channels run in parallel:

#### Lexical Channel
```sql
SELECT * FROM rag2_lexical_search(
    p_org_id := 'org_123',
    p_query := 'refund policy',
    p_limit := 50
);
```

#### Semantic Channel
```sql
SELECT * FROM rag2_semantic_search(
    p_org_id := 'org_123',
    p_embedding := [0.1, 0.2, ...],
    p_limit := 100
);
```

#### Graph Channel

Uses PuppyGraph Gremlin or SQL fallback:

```gremlin
// PuppyGraph Gremlin
g.V().hasLabel('entity')
  .has('name', containing('Acme'))
  .both('relates_to')
  .out('mentions')
  .valueMap()
```

```sql
-- SQL Fallback
SELECT c.* FROM rag_child_chunks c
JOIN rag_entity_mentions em ON em.child_chunk_id = c.id
JOIN rag_entities e ON em.entity_id = e.id
WHERE e.name ILIKE '%Acme%'
```

### Step 3: Weighted RRF Fusion

```python
def _fuse_rrf(candidates, weights, k=60):
    for c in candidates:
        score = 0.0
        if c.lexical_rank:
            score += weights["lexical"] / (k + c.lexical_rank)
        if c.semantic_rank:
            score += weights["semantic"] / (k + c.semantic_rank)
        if c.graph_rank:
            score += weights["graph"] / (k + c.graph_rank)
        c.rrf_score = score
    
    return sorted(candidates, key=lambda x: x.rrf_score, reverse=True)
```

**Default Weights:**
| Channel | Weight | Rationale |
|---------|--------|-----------|
| Graph | 1.0 | Entity relationships are high-precision |
| Semantic | 0.8 | Good for concept similarity |
| Lexical | 0.7 | Useful for exact matches |

### Step 4: Child → Parent Expansion

Child chunks are expanded to their parent chunks for richer context:

```python
async def _expand_to_parents(candidates):
    parent_ids = [c.parent_id for c in candidates]
    parents = await db.table("rag_parent_chunks").select("*").in_("id", parent_ids)
    
    for c in candidates:
        c.parent_text = parents[c.parent_id].text
        c.section_heading = parents[c.parent_id].section_heading
    
    return candidates
```

### Step 5: Reranking

Uses Qwen3-VL-Reranker for cross-encoder scoring:

```python
# Native /rerank endpoint (preferred)
scores = await reranker._rerank_batch_native(query, texts)

# Chat-based fallback
score = await reranker._score_pair(query, text, None)
```

### Step 6: Safety Threshold & Denoising

```python
def _apply_safety(candidates, top_k):
    max_score = max(c.rerank_score for c in candidates)
    
    # Safety threshold check
    if max_score < SETTINGS.rag2_safety_threshold:
        return [], True, "Below threshold", max_score
    
    # Conformal denoising
    min_score = SETTINGS.rag2_denoise_alpha * max_score
    filtered = [c for c in candidates if c.rerank_score >= min_score]
    
    return filtered[:top_k], False, None, max_score
```

---

## Graph Channel Deep Dive

### PuppyGraph Integration

PuppyGraph provides zero-ETL graph queries over PostgreSQL.

**Deployment:**
```bash
docker-compose -f infrastructure/puppygraph/docker-compose.yml up -d
```

**Ports:**
- 8182: Gremlin endpoint
- 8081: Web UI

**Schema:** `infrastructure/puppygraph/schema.json`

### SQL Fallback

When PuppyGraph is unavailable, the system falls back to SQL-based graph traversal:

```python
class SQLGraphFallback:
    async def find_related_chunks(self, keywords, org_id, top_k):
        # 1. Find matching entities
        entities = await self._find_entities(keywords, org_id)
        
        # 2. Find related entities via relations
        related = await self._traverse_relations(entities)
        
        # 3. Find chunks mentioning related entities
        chunks = await self._find_mentioned_chunks(related)
        
        return chunks
```

---

## OCR with Gundam Tiling

For large images (>2048px in either dimension), Gundam Tiling splits the image into overlapping tiles:

```python
@dataclass
class GundamTilingConfig:
    tile_size: int = 1024
    overlap: int = 128
    merge_strategy: str = "fuzzy"  # fuzzy, concat, vote
    enabled: bool = True
```

**Merge Strategies:**
- **fuzzy**: Deduplicate similar text (fuzz ratio > 80%)
- **concat**: Join all tile results
- **vote**: Confidence-weighted selection

---

## Configuration Reference

```python
# Core settings
RAG2_ENABLED: bool = True
RAG2_GRAPH_ENABLED: bool = True
RAG2_ENTITY_EXTRACTION_ENABLED: bool = True
RAG2_RERANK_ENABLED: bool = True

# Top-K at each stage
RAG2_LEXICAL_TOP_K: int = 50
RAG2_SEMANTIC_TOP_K: int = 100
RAG2_GRAPH_TOP_K: int = 50
RAG2_RERANK_TOP_K: int = 20
RAG2_FINAL_TOP_K: int = 5

# Safety
RAG2_SAFETY_THRESHOLD: float = 0.6
RAG2_DENOISE_ALPHA: float = 0.5

# Weights
RAG2_WEIGHT_LEXICAL: float = 0.7
RAG2_WEIGHT_SEMANTIC: float = 0.8
RAG2_WEIGHT_GRAPH: float = 1.0

# Embeddings
RAG2_EMBEDDING_DIM: int = 1024
RAG2_EMBEDDING_MODEL: str = "text-embedding-3-large"

# Chunking
RAG2_PARENT_CHUNK_SIZE: int = 1000
RAG2_CHILD_CHUNK_SIZE: int = 200
RAG2_CHUNK_OVERLAP: int = 50

# PuppyGraph
RAG2_PUPPYGRAPH_URL: str = "http://localhost:8182"
RAG2_PUPPYGRAPH_TIMEOUT: float = 30.0
```

---

## Module Reference

| Module | Path | Purpose |
|--------|------|---------|
| embedder | `src/voice_agent/rag2/embedder.py` | Matryoshka embeddings |
| chunker | `src/voice_agent/rag2/chunker.py` | Hierarchical chunking |
| ingest | `src/voice_agent/rag2/ingest.py` | Ingestion pipeline |
| retrieval | `src/voice_agent/rag2/retrieval.py` | Retrieval pipeline |
| query_planner | `src/voice_agent/rag2/query_planner.py` | GPT-5 query analysis |
| entity_extraction | `src/voice_agent/rag2/entity_extraction.py` | NER + RE |
| graph_search | `src/voice_agent/rag2/graph_search.py` | PuppyGraph + SQL fallback |
| crm_knowledge | `src/voice_agent/tools/crm_knowledge.py` | Agent tool integration |

---

## Agent Tool Integration

The `search_knowledge_base` tool connects the voice agent to RAG2 retrieval:

```python
# In crm_knowledge.py
async def search_knowledge_base(query: str, org_id: str) -> List[Dict]:
    if SETTINGS.rag2_enabled and org_id:
        return await _search_knowledge_base_rag2(query, org_id)
    # ... fallback to legacy
```

### Data Flow

```
Voice Agent Tool Call
  → search_knowledge_base(query, org_id)
    → _search_knowledge_base_rag2()
      → RAG2Retriever.retrieve(query)
        → Query Planning (GPT-5)
        → Multi-channel retrieval
        → Weighted RRF fusion
        → Child → Parent expansion
        → Reranking
        → Safety threshold
      → Map response to tool format
    → Return knowledge items
```

### Response Format

```python
{
    "id": "child-chunk-uuid",
    "text": "Parent chunk text (for context)",
    "score": 0.85,  # rerank_score or rrf_score
    "source": "rag2",
    "page": 3,
    "is_table": False
}
```

---

## Performance Characteristics

| Metric | Target | Actual |
|--------|--------|--------|
| Retrieval latency (p50) | < 500ms | ~300ms |
| Retrieval latency (p95) | < 1000ms | ~700ms |
| Rerank latency (5 docs) | < 200ms | ~150ms |
| Entity extraction (per doc) | < 2s | ~1.5s |
| Tests | 200+ | 205 ✅ |
