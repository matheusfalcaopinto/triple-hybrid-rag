# Phase 0: Compatibility Note — Current vs RAG2.0 Mapping

**Date**: 2026-01-14  
**Status**: Complete

---

## 1. Current Code Structure

### Ingestion Pipeline
| Module | Location | Lines | Description |
|--------|----------|-------|-------------|
| `loader.py` | `src/voice_agent/ingestion/` | ~500 | File type detection, PDF/DOCX/XLSX extraction |
| `ocr.py` | `src/voice_agent/ingestion/` | ~250 | Qwen3-VL OCR via LM Studio |
| `chunker.py` | `src/voice_agent/ingestion/` | ~366 | Character-based chunking, table preservation |
| `embedder.py` | `src/voice_agent/ingestion/` | ~500 | Qwen3-VL embeddings (4096→4000 truncate) |
| `kb_ingest.py` | `src/voice_agent/ingestion/` | ~513 | Main orchestrator |

### Retrieval Pipeline
| Module | Location | Lines | Description |
|--------|----------|-------|-------------|
| `hybrid_search.py` | `src/voice_agent/retrieval/` | ~526 | BM25 + vector + RRF fusion |
| `reranker.py` | `src/voice_agent/retrieval/` | ~180 | Qwen3-VL-Reranker with `/no_think` |

### Configuration
- `src/voice_agent/config.py`: ~45 RAG settings (embed model, chunk size, rerank, OCR, etc.)

### Database
- **Current table**: `knowledge_base_chunks` with:
  - `vector_embedding vector(1536)` — legacy (was OpenAI)
  - Actual usage: `halfvec(4000)` (per `20260113_halfvec_4000.sql`)
- **Parent table**: `knowledge_base` (document-level metadata)

---

## 2. Key Decisions for RAG2.0

### 2.1 Tenant Key
- **Current**: Uses `org_id` (UUID FK to `organizations`)
- **RAG2.0 spec**: Uses `tenant_id`
- **Decision**: Keep `org_id` (no rename needed). The new tables will use `org_id` for consistency with existing auth/RLS patterns.

### 2.2 Vector Storage
- **Current**: `halfvec(4000)` HNSW (4096 truncated to 4000)
- **RAG2.0 spec**: `vector(1024)` (4096 truncated to 1024 via Matryoshka)
- **Decision**: 
  - New tables use `vector(1024)` for child chunks
  - Optionally keep `vector(4096)` for offline analysis
  - The 1024d is well within pgvector limits and saves ~75% storage

### 2.3 FTS Location
- **Current**: FTS index on `knowledge_base_chunks.content`
- **RAG2.0 spec**: Child chunks have `tsv` generated column
- **Decision**: Keep FTS on child chunks (they are the retrieval unit)

---

## 3. Table Mapping: Old → New

### knowledge_base_chunks → child_chunks

| Old Column | New Column | Notes |
|------------|------------|-------|
| `id` | `id` | Same (UUID pk) |
| `org_id` | `org_id` | Same |
| `knowledge_base_id` | `document_id` | FK to new `documents` table |
| `category` | (moved) | Move to `documents` table |
| `title` | (moved) | Move to `documents` table |
| `source_document` | (moved) | Move to `documents.file_name` |
| `modality` | `modality` | Keep (text/table/image/image_caption) |
| `page` | `page` | Same |
| `chunk_index` | `index_in_parent` | Now relative to parent |
| `content` | `text` | Renamed for spec consistency |
| `content_hash` | `content_hash` | Same |
| `ocr_confidence` | (moved) | Move to `parent_chunks` |
| `is_table` | (metadata) | Track via `modality='table'` |
| `table_context` | (metadata) | Store in JSONB |
| `alt_text` | (metadata) | Store in JSONB |
| `image_path` | (metadata) | Store in JSONB |
| `vector_embedding` | `embedding_1024` | New type: `vector(1024)` |
| `vector_image` | (removed) | Not needed with unified embeddings |

### knowledge_base → documents

| Old Column | New Column | Notes |
|------------|------------|-------|
| `id` | `id` | Same |
| `org_id` | `org_id` | Same |
| `title` | `title` | Metadata |
| `content` | (N/A) | Content now lives in chunks |
| `category` | `collection` | Spec uses `collection` |
| `doc_checksum` | `hash_sha256` | For idempotency |
| `is_chunked` | (N/A) | Implicit: has children |

### New Tables

- **`parent_chunks`**: Intermediate level (800–1000 tokens) for LLM context
- **`entities`**: NER output for knowledge graph
- **`entity_mentions`**: Links entities to chunks
- **`relations`**: Triplets (subject, predicate, object)

---

## 4. Backward Compatibility Strategy

During migration:

1. **Keep both systems running** — old `knowledge_base_chunks` + new `documents/parent_chunks/child_chunks`
2. **Dual-write** — Ingest to both during transition
3. **Shadow-read** — Compare results before cutover
4. **Feature flags** — `RAG2_ENABLED`, `RAG2_GRAPH_ENABLED`, etc.

Post-cutover:

1. Mark old `knowledge_base_chunks` as deprecated
2. Create migration script to backfill existing data if needed
3. Eventually drop old columns/tables

---

## 5. Config Additions Needed

```python
# RAG2.0 Feature Flags
rag2_enabled: bool = Field(False, alias="RAG2_ENABLED")
rag2_graph_enabled: bool = Field(False, alias="RAG2_GRAPH_ENABLED")
rag2_rerank_enabled: bool = Field(True, alias="RAG2_RERANK_ENABLED")
rag2_denoise_enabled: bool = Field(True, alias="RAG2_DENOISE_ENABLED")

# RAG2.0 Embedding (Matryoshka)
rag2_embed_dim_store: int = Field(1024, alias="RAG2_EMBED_DIM_STORE")
rag2_embed_dim_model: int = Field(4096, alias="RAG2_EMBED_DIM_MODEL")

# RAG2.0 Parent/Child Chunking
rag2_parent_chunk_tokens: int = Field(800, alias="RAG2_PARENT_CHUNK_TOKENS")
rag2_child_chunk_tokens: int = Field(200, alias="RAG2_CHILD_CHUNK_TOKENS")

# RAG2.0 Retrieval
rag2_safety_threshold: float = Field(0.6, alias="RAG2_SAFETY_THRESHOLD")
rag2_denoise_alpha: float = Field(0.6, alias="RAG2_DENOISE_ALPHA")
rag2_lexical_weight: float = Field(0.7, alias="RAG2_LEXICAL_WEIGHT")
rag2_semantic_weight: float = Field(0.8, alias="RAG2_SEMANTIC_WEIGHT")
rag2_graph_weight: float = Field(1.0, alias="RAG2_GRAPH_WEIGHT")

# GPT-5 for Query Planning
rag2_query_planner_model: str = Field("gpt-5-nano", alias="RAG2_QUERY_PLANNER_MODEL")
```

---

## 6. Next Steps

- ✅ Phase 0: This compatibility note
- 🔄 Phase 1: Create new schema migration
- Phase 2: Matryoshka embedder refactor
- Phase 3: Ingestion v2
- Phase 4: Retrieval v2
- Phase 5: Graph channel
- Phase 6: Evaluation
- Phase 7: Cutover
