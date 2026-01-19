# RAG 2.0 Implementation Walkthrough

> **Status:** ✅ PRODUCTION READY  
> **Last Updated:** 2026-01-14  
> **Test Coverage:** 80/80 tests passing

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Model Stack](#model-stack)
4. [Module Reference](#module-reference)
5. [Database Schema](#database-schema)
6. [Configuration](#configuration)
7. [Usage Guide](#usage-guide)
8. [Test Results](#test-results)
9. [E2E Validation Results](#e2e-validation-results)

---

## Overview

RAG 2.0 is a complete rewrite of the retrieval-augmented generation system, implementing a **triple-hybrid architecture** with:

- **Matryoshka Embeddings**: 2048d → 1024d truncation for storage efficiency
- **Hierarchical Chunking**: Parent (800-1000 tokens) / Child (~200 tokens) structure
- **Triple-Hybrid Retrieval**: Lexical + Semantic + Graph channels
- **Weighted RRF Fusion**: Reciprocal Rank Fusion with per-channel weights
- **Safety Threshold**: Conformal denoising with 0.6 minimum score
- **Knowledge Graph Integration**: PuppyGraph/Cypher with SQL fallback

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Matryoshka 2048→1024 | MRL preserves 95%+ recall at 50% storage |
| Parent/Child hierarchy | Small retrieval units, large LLM context |
| SHA-256 content hash | Chunk-level deduplication across documents |
| Native `/rerank` endpoint | 50% score improvement vs chat-based reranking |
| FTS + HNSW indexes | Combined lexical precision + semantic recall |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INGESTION PIPELINE                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────┐     ┌────────────┐     ┌───────────┐     ┌─────────┐ │
│   │ Loader  │────▶│ Hierarchical│────▶│ Matryoshka│────▶│ Supabase│ │
│   │ + OCR   │     │  Chunker   │     │  Embedder │     │  + pgvec│ │
│   └─────────┘     └────────────┘     └───────────┘     └─────────┘ │
│       ▲               │                    │                        │
│       │               ▼                    ▼                        │
│   ┌───────┐      ┌────────┐          ┌─────────┐                   │
│   │ DOCX  │      │ Parent │          │  1024d  │                   │
│   │ PDF   │      │ Chunks │          │ vectors │                   │
│   │ TXT   │      │ (800t) │          └─────────┘                   │
│   │ MD    │      │   ▼    │                                        │
│   └───────┘      │ Child  │                                        │
│                  │ Chunks │                                        │
│                  │ (200t) │                                        │
│                  └────────┘                                        │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                        RETRIEVAL PIPELINE                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│ Query ─▶ Query    ─▶ ┌─────────────────┐ ─▶ RRF    ─▶ Parent       │
│          Planner     │ Lexical (BM25)  │    Fusion    Expansion    │
│          (GPT-5)     │ Semantic (HNSW) │       │          │        │
│              │       │ Graph (Cypher)  │       ▼          ▼        │
│              ▼       └─────────────────┘   Reranker ─▶ Safety      │
│         Keywords                           (Qwen3)    Threshold    │
│         Semantic Q                              │          │       │
│         Cypher Q                                ▼          ▼       │
│                                            Final Contexts ─▶ LLM   │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Document Loading**: Extract text from DOCX/PDF/TXT/MD via `DocumentLoader`
2. **OCR (optional)**: DeepSeek-OCR for scanned documents → markdown text
3. **Hierarchical Chunking**: Split into ~200 token children under ~800 token parents
4. **Embedding**: Qwen3-VL-Embedding-2B → 2048d → truncate to 1024d (Matryoshka)
5. **Storage**: PostgreSQL + pgvector (HNSW index for ANN search)
6. **Query Planning**: GPT-5 generates keywords + semantic query + optional Cypher
7. **Triple Retrieval**: Parallel lexical, semantic, and graph searches
8. **RRF Fusion**: Weighted combination of all channels
9. **Parent Expansion**: Retrieve parent context for child hits
10. **Reranking**: Qwen3-VL-Reranker-2B with native `/rerank` endpoint
11. **Safety Filter**: Discard results below 0.6 threshold

---

## Model Stack

### Production Configuration (Qwen3-VL-2B)

| Component | Model | Port | Dimensions | Status |
|-----------|-------|------|------------|--------|
| **Embeddings** | `Qwen/Qwen3-VL-Embedding-2B` | 1234 | 2048 → 1024 | ✅ Working |
| **Reranker** | `Qwen/Qwen3-VL-Reranker-2B` | 1235 | - | ✅ Working |
| **OCR** | `deepseek-ai/DeepSeek-OCR` | 1236 | - | ✅ Validated |
| **Query Planner** | GPT-5 (OpenAI) | - | - | ✅ Working |

### vLLM Server Commands

```bash
# Embeddings (port 1234)
vllm serve Qwen/Qwen3-VL-Embedding-2B \
  --port 1234 \
  --task embed

# Reranker (port 1235) - IMPORTANT: use --runner pooling for native /rerank
vllm serve Qwen/Qwen3-VL-Reranker-2B \
  --port 1235 \
  --runner pooling

# OCR (port 1236)
vllm serve deepseek-ai/DeepSeek-OCR \
  --port 1236 \
  --max-model-len 4096
```

### Environment Variables

```bash
# Embedding Configuration
RAG_EMBED_MODEL=Qwen/Qwen3-VL-Embedding-2B
RAG_EMBED_API_BASE=http://127.0.0.1:1234/v1
RAG2_EMBED_DIM_MODEL=2048    # Model outputs 2048d (not 4096 like 8B)
RAG2_EMBED_DIM_STORE=1024    # Matryoshka truncation

# Reranker Configuration
RAG_RERANK_MODEL=Qwen/Qwen3-VL-Reranker-2B
RAG_RERANK_API_BASE=http://127.0.0.1:1235/v1

# OCR Configuration
RAG_OCR_ENDPOINT=http://127.0.0.1:1236/v1
RAG_OCR_MODEL=deepseek-ai/DeepSeek-OCR
```

---

## Module Reference

### `src/voice_agent/rag2/`

| File | Purpose | Lines | Key Classes/Functions |
|------|---------|-------|----------------------|
| `__init__.py` | Module exports | ~80 | All public exports |
| `embedder.py` | Matryoshka MRL embedder | ~250 | `RAG2Embedder`, `truncate_matryoshka()` |
| `chunker.py` | Hierarchical parent/child chunker | ~360 | `HierarchicalChunker`, `ParentChunk`, `ChildChunk` |
| `ingest.py` | Document ingestion pipeline | ~400 | `RAG2Ingestor`, `IngestResult`, `IngestStats` |
| `query_planner.py` | GPT-5 query decomposition | ~200 | `QueryPlanner`, `QueryPlan` |
| `retrieval.py` | Triple-hybrid retrieval + RRF fusion | ~500 | `RAG2Retriever`, `RetrievalResult` |
| `graph_search.py` | PuppyGraph + SQL fallback | ~430 | `PuppyGraphClient`, `SQLGraphFallback` |

### embedder.py

```python
from voice_agent.rag2 import get_rag2_embedder

embedder = get_rag2_embedder()

# Single text embedding
result = await embedder.embed_text_async("Your text here")
print(len(result.embedding))  # 1024

# Batch embedding
results = await embedder.embed_texts_async(["text1", "text2", "text3"])

# Query embedding (for retrieval)
query_vec = embedder.embed_query("How to configure Twilio?")
```

### chunker.py

```python
from voice_agent.rag2 import get_hierarchical_chunker

chunker = get_hierarchical_chunker()

# Chunk document text
parents = chunker.chunk_document(
    full_text="Long document text...",
    doc_hash="abc123",  # For stable IDs
)

for parent in parents:
    print(f"Parent {parent.index_in_document}: {parent.token_count} tokens")
    for child in parent.children:
        print(f"  Child {child.index_in_parent}: {child.token_count} tokens")
```

### ingest.py

```python
from voice_agent.rag2 import RAG2Ingestor

ingestor = RAG2Ingestor(
    org_id="00000000-0000-0000-0000-000000000001",
    collection="my-docs",
)

# Ingest single file
result = await ingestor.ingest_file(
    file_path="/path/to/document.docx",
    title="API Reference",
    tags=["api", "documentation"],
)

print(f"Document ID: {result.document_id}")
print(f"Parents: {result.stats.parent_chunks_created}")
print(f"Children: {result.stats.child_chunks_created}")
```

### retrieval.py

```python
from voice_agent.rag2 import RAG2Retriever

retriever = RAG2Retriever(
    org_id="00000000-0000-0000-0000-000000000001",
)

# Full retrieval pipeline
result = await retriever.retrieve(
    query="How to configure Twilio webhooks?",
    collection="my-docs",
    top_k=5,
)

if result.refused:
    print(f"No results: {result.refusal_reason}")
else:
    for ctx in result.contexts:
        print(f"Score: {ctx.rerank_score:.3f}")
        print(f"Text: {ctx.parent_text[:200]}...")
```

---

## Database Schema

### Tables

```
rag_documents          ─────┬───── rag_parent_chunks ─────── rag_child_chunks
(document registry)         │      (800-1000 tokens)          (~200 tokens)
                            │                                  + embedding_1024
                            │                                  + tsv (FTS)
                            │
                            └───── rag_entities ─────────────── rag_relations
                                   (NER nodes)                  (graph edges)
                                         │
                                         └──── rag_entity_mentions
                                              (entity-chunk links)
```

### Key Columns

#### `rag_documents`
- `hash_sha256`: File-level deduplication
- `collection`: Logical grouping
- `ingestion_status`: pending/processing/completed/failed

#### `rag_child_chunks`
- `embedding_1024 vector(1024)`: Matryoshka-truncated embeddings
- `tsv tsvector`: Full-text search (Portuguese)
- `content_hash`: Chunk-level deduplication

### Indexes

| Index | Type | Purpose |
|-------|------|---------|
| `idx_rag_children_embedding_hnsw` | HNSW | ANN vector search |
| `idx_rag_children_fts` | GIN | Full-text search |
| `idx_rag_children_dedup` | BTREE | Content deduplication |
| `idx_rag_docs_org_hash` | BTREE | Document idempotency |

### Migration

```bash
# Apply schema
docker exec -i supabase-db psql -U postgres -d postgres \
  < database/migrations/20260114_rag2_schema.sql
```

---

## Configuration

### All RAG2 Settings (from `config.py`)

| Setting | Default | Description |
|---------|---------|-------------|
| `RAG2_EMBED_DIM_MODEL` | 2048 | Model output dimensions |
| `RAG2_EMBED_DIM_STORE` | 1024 | Storage dimensions (MRL) |
| `RAG2_PARENT_CHUNK_TOKENS` | 800 | Target parent chunk size |
| `RAG2_PARENT_CHUNK_MAX_TOKENS` | 1000 | Maximum parent chunk size |
| `RAG2_CHILD_CHUNK_TOKENS` | 200 | Target child chunk size |
| `RAG2_LEXICAL_TOP_K` | 50 | Lexical channel candidates |
| `RAG2_SEMANTIC_TOP_K` | 100 | Semantic channel candidates |
| `RAG2_GRAPH_TOP_K` | 50 | Graph channel candidates |
| `RAG2_RERANK_TOP_K` | 20 | Candidates sent to reranker |
| `RAG2_FINAL_TOP_K` | 5 | Final results returned |
| `RAG2_LEXICAL_WEIGHT` | 0.7 | RRF weight for lexical |
| `RAG2_SEMANTIC_WEIGHT` | 0.8 | RRF weight for semantic |
| `RAG2_GRAPH_WEIGHT` | 1.0 | RRF weight for graph |
| `RAG2_SAFETY_THRESHOLD` | 0.6 | Minimum rerank score |
| `RAG2_RERANK_ENABLED` | true | Enable reranking step |
| `RAG2_GRAPH_ENABLED` | false | Enable graph channel |
| `RAG2_QUERY_PLANNER_MODEL` | gpt-5-nano | Query planning model |
| `RAG2_PUPPYGRAPH_URL` | - | PuppyGraph endpoint |

---

## Usage Guide

### CLI Scripts

#### Ingest Documents

```bash
# Ingest single file
python scripts/ingest_rag2.py \
  --file /path/to/document.pdf \
  --collection my-docs \
  --title "My Document" \
  --org-id 00000000-0000-0000-0000-000000000001

# Ingest directory
python scripts/ingest_rag2.py \
  --dir /path/to/docs/ \
  --collection my-docs \
  --org-id 00000000-0000-0000-0000-000000000001
```

#### Test Retrieval

```bash
# Test single query
python scripts/test_rag2.py \
  --query "How to configure Twilio webhooks?" \
  --collection my-docs \
  --org-id 00000000-0000-0000-0000-000000000001 \
  --top-k 5
```

#### Backfill from RAG1

```bash
# Migrate existing kb_chunks to RAG2 schema
python scripts/backfill_rag2.py \
  --org-id 00000000-0000-0000-0000-000000000001 \
  --batch-size 100
```

### Programmatic Usage

```python
import asyncio
from voice_agent.rag2 import RAG2Ingestor, RAG2Retriever

async def main():
    org_id = "00000000-0000-0000-0000-000000000001"
    
    # Ingest
    ingestor = RAG2Ingestor(org_id=org_id, collection="my-docs")
    result = await ingestor.ingest_file("/path/to/doc.pdf")
    print(f"Ingested: {result.document_id}")
    
    # Retrieve
    retriever = RAG2Retriever(org_id=org_id)
    result = await retriever.retrieve("What are the payment terms?")
    
    for ctx in result.contexts:
        print(f"[{ctx.rerank_score:.2f}] {ctx.parent_text[:100]}...")

asyncio.run(main())
```

---

## Test Results

### Test Summary: 80/80 PASSING ✅

| Test Suite | Tests | Status | Description |
|------------|-------|--------|-------------|
| `test_rag2_embedder.py` | 15 | ✅ | Matryoshka truncation, L2 normalization |
| `test_rag2_chunker.py` | 20 | ✅ | Hierarchical chunking, content hashing |
| `test_rag2_retrieval.py` | 15 | ✅ | RRF fusion, safety threshold, query planning |
| `test_rag2_e2e.py` | 15 | ✅ | Real DB operations, mocked models |
| `test_rag2_integration.py` | 15 | ✅ | Full pipeline integration |

### Running Tests

```bash
# Run all RAG2 tests
pytest tests/test_rag2_*.py -v

# Run with coverage
pytest tests/test_rag2_*.py --cov=src/voice_agent/rag2 --cov-report=html

# Run specific test file
pytest tests/test_rag2_embedder.py -v
```

### Key Test Cases

#### Embedder Tests
- ✅ `test_matryoshka_truncation` - 2048→1024 dimension reduction
- ✅ `test_l2_normalization` - Unit-length vectors
- ✅ `test_batch_embedding` - Multiple texts in single call
- ✅ `test_empty_text_handling` - Edge case handling

#### Chunker Tests
- ✅ `test_parent_child_hierarchy` - Correct parent/child structure
- ✅ `test_stable_chunk_ids` - Deterministic IDs from doc hash
- ✅ `test_content_hash_dedup` - Same content = same hash
- ✅ `test_table_preservation` - Tables kept as single chunks
- ✅ `test_sentence_boundary_splitting` - Clean text splits

#### Retrieval Tests
- ✅ `test_rrf_fusion_weights` - Weighted score combination
- ✅ `test_safety_threshold_filtering` - Below 0.6 rejected
- ✅ `test_parent_expansion` - Child→Parent context retrieval
- ✅ `test_lexical_semantic_channels` - Both channels contribute
- ✅ `test_query_planning` - Keywords + semantic query generated

#### E2E Tests (Real Database)
- ✅ `test_document_idempotency` - Same file → same doc ID
- ✅ `test_ingestion_pipeline` - Load → Chunk → Embed → Store
- ✅ `test_semantic_search` - Vector similarity working
- ✅ `test_fts_search` - Portuguese full-text search
- ✅ `test_full_retrieval_pipeline` - Query → Results

---

## E2E Validation Results

### Real Model Validation (2026-01-14)

#### Documents Ingested

| Document | Type | Parents | Children | Status |
|----------|------|---------|----------|--------|
| Voice AI Agent Provider API Reference | DOCX | 11 | 43 | ✅ |
| Micro-Structure Sniper PDF | PDF | 14 | 61 | ✅ |

#### Retrieval Performance

| Query | Max Rerank Score | Source Document |
|-------|------------------|-----------------|
| "What is LOB dynamics?" | **0.829** | Micro-Structure Sniper PDF |
| "How to configure Twilio webhooks?" | **0.915** | Voice AI Agent API DOCX |

#### OCR Validation (DeepSeek-OCR)

| Page | Confidence | Characters | Content Types |
|------|------------|------------|---------------|
| Page 1 | 0.94 | 2,667 | LaTeX equations, headings |
| Page 5 | 0.97 | 4,422 | Technical content, tables |
| Page 10 | 0.97 | 4,220 | Formulas, bold text |

**OCR Fixes Applied:**
- `max_tokens`: 4096 → 2048 (model context limit)
- `OCR_PROMPT`: Simplified (long prompts caused empty responses)

#### Reranker Configuration

**Critical Fix:** Use `--runner pooling` for vLLM reranker server:

```bash
# CORRECT - Native /rerank endpoint enabled
vllm serve Qwen/Qwen3-VL-Reranker-2B --port 1235 --runner pooling

# WRONG - Only chat mode, lower scores
vllm serve Qwen/Qwen3-VL-Reranker-2B --port 1235
```

**Score Comparison:**
- With `--runner pooling`: 0.74-0.91 scores
- Without (chat mode fallback): 0.47 scores

---

## Appendix

### File Structure

```
src/voice_agent/rag2/
├── __init__.py         # Module exports
├── embedder.py         # Matryoshka embedder
├── chunker.py          # Hierarchical chunker
├── ingest.py           # Ingestion pipeline
├── query_planner.py    # GPT-5 query decomposition
├── retrieval.py        # Triple-hybrid retrieval
└── graph_search.py     # Knowledge graph search

tests/
├── test_rag2_embedder.py
├── test_rag2_chunker.py
├── test_rag2_retrieval.py
├── test_rag2_e2e.py
└── test_rag2_integration.py

scripts/
├── ingest_rag2.py      # CLI ingestion
├── test_rag2.py        # CLI retrieval testing
└── backfill_rag2.py    # RAG1 → RAG2 migration

database/migrations/
└── 20260114_rag2_schema.sql
```

### Troubleshooting

#### "Empty embedding returned"
- Check model is running: `curl http://127.0.0.1:1234/v1/models`
- Verify `RAG2_EMBED_DIM_MODEL=2048` for 2B model

#### "Rerank scores below 0.5"
- Ensure vLLM uses `--runner pooling`
- Check native endpoint: `curl http://127.0.0.1:1235/rerank -d '{"query":"test","documents":["doc"]}'`

#### "OCR returns empty text"
- Use simplified `OCR_PROMPT`
- Set `max_tokens=2048` (not 4096)
- Check model context limit

#### "Document already exists"
- Normal behavior (idempotency)
- Use `force=True` to re-ingest

### Performance Benchmarks

| Operation | Time | Notes |
|-----------|------|-------|
| Document ingestion (10 pages) | ~15s | With embeddings |
| Single query retrieval | ~800ms | Full pipeline |
| Embedding (1 text) | ~50ms | 1024d output |
| Reranking (20 candidates) | ~200ms | Native endpoint |

---

*Generated: 2026-01-14 | RAG 2.0 v1.0*
