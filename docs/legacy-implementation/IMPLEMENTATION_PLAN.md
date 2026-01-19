# RAG 2.0 Refactor — Implementation Plan (Triple-Hybrid + Matryoshka 4096→1024)

**Repo**: `voice-agent-v5`  
**Branch context**: `vector-rag`  
**Date**: 2026-01-14  
**Goal**: Refactor the current working Vector RAG into the new **Multimodal Triple‑Hybrid RAG** architecture (Lexical + Semantic + Graph + rerank + safety funnel), while **preserving proven production features** currently present (notably **deduplication**, provenance metadata, table preservation, RRF fusion, and the existing Qwen3-based embedding/rerank stack).

This document is written as an **engineering execution plan**: broken into phases and small steps, with concrete deliverables, contracts, and verification gates. It is intentionally detailed so we can implement it iteratively and keep the system usable throughout.

---

## 0) What stays vs what changes

### 0.1 What we have today (confirmed in `docs/RAG_SYSTEM_GUIDE.md`)

The current system is already “Vector RAG 2.0 (halfvec)” with:

- **Unified embeddings** using **Qwen3‑VL‑Embedding** (4096d) truncated to **4000d** and stored as **`halfvec(4000)`** for HNSW.
- **OCR** via Qwen3‑VL (LM Studio endpoint).
- **Hybrid search**: BM25 (Postgres FTS) + vector + **RRF fusion**.
- **Reranking**: Qwen3‑VL‑Reranker using a `/no_think` constraint.
- **Deduplication**: content-hash based.
- **Table preservation** and provenance fields.

### 0.2 What the new RAG 2.0 spec requires

From `docs/artifacts/RAG2.0/*`:

- Triple channel retrieval funnel:
  - **Lexical**: Postgres FTS/BM25
  - **Semantic**: pgvector HNSW
  - **Graph**: PuppyGraph over Postgres tables (Zero‑ETL)
- **QueryPlannerAgent** (GPT‑5) to produce:
  - keywords
  - semantic_query_text
  - cypher_query
  - per-channel top_k and weights
- **Weighted RRF** fusion with default weights:
  - graph 1.0, semantic 0.8, lexical 0.7
- **Parent/Child hierarchy**:
  - child retrieval → expand to parent context → rerank parents
- **Rerank** with Qwen3‑VL‑Reranker
- **Safety**:
  - if max rerank score < 0.6 → refuse (“insufficient evidence”)
- **Context denoising**:
  - conformal prediction style trimming (start heuristic)
- **Matryoshka Representation Learning (MRL)**:
  - model outputs **4096d**, but primary stored/indexed vector is **1024d** (truncate) to fit pgvector dimension constraints.

### 0.3 Must-carry features missing/underspecified in new spec

We explicitly keep these because they’re already valuable/working:

1. **Deduplication**
   - Two layers:
     - Document-level idempotency (hash of raw file bytes)
     - Chunk-level dedup (hash of normalized chunk text)
2. **OCR quality validation + fallback** (from critique docs)
   - Confidence scoring, table alignment checks, retry/fallback strategy.
3. **Table-aware ingestion**
   - Preserve tables and add “table_context” summary as prefix.
4. **Operational guardrails from the current system**
   - Batch embeddings, feature flags, provenance, structured observability.

---

## 1) Target architecture contract (what “done” means)

### 1.1 Ingestion contract (Parse → Transform → Index)

**Input**: document (upload or URL) + metadata filters (tenant/org, collection, tags).  
**Output**: persisted document, chunks, embeddings, and (optionally) KG entities/relations.

**Pipeline invariants**:

- Idempotent: re-ingesting the same file for same tenant does not duplicate.
- Child chunks are the retrieval unit; parent chunks are the “LLM context unit”.
- Vectors are **generated once** and stored with stable IDs.

### 1.2 Retrieval contract (Recall → Precision funnel)

**Input**: query + filters.  
**Output**: either:

- `answer + contexts + trace`, OR
- `NO_SUITABLE_CONTEXT` with `max_rerank_score`.

**Stages**:

- GPT‑5 query planning
- 3 channel retrieval
- Weighted RRF fusion
- child→parent expansion
- rerank parents (Qwen3‑VL‑Reranker)
- safety threshold + trimming
- answer generation (GPT‑5)

---

## 2) Data model & migrations (planned changes)

### 2.1 Key design choice: reconcile “current knowledge_base_chunks” with “spec documents/parent_chunks/child_chunks”

We will **not** break the whole system at once.

- **Short-term**: introduce new tables (`documents`, `parent_chunks`, `child_chunks`, `entities`, `relations`, …) alongside existing `knowledge_base_chunks`.
- **Mid-term**: migrate existing KB into the new tables, and then move retrieval to the new pipeline.

This gives us a safe incremental cutover.

### 2.2 Embedding storage: Matryoshka 4096 → 1024

The new plan is:

- Keep generating the **full 4096d** vectors from Qwen3‑VL‑Embedding.
- Store/index the **first 1024 dims** (`embedding_1024`) as the primary ANN feature.
- Optionally store:
  - raw 4096 (`embedding_4096`) for offline analysis
  - current production 4000-halfvec vectors only during migration window

**Open question (resolved by implementation)**: 
- The current DB uses `halfvec(4000)` because `vector` had a dimension limit. The new spec assumes a `vector(1024)` is acceptable. We will implement **a config-driven vector storage strategy**:
  - Default: `vector(1024)` (simple)
  - Optional: `halfvec(1024)` (smaller, if halfvec extension still preferred)

### 2.3 Proposed schema (minimal)

Create new migrations in `database/migrations/`.

#### documents
- `id uuid pk`
- `tenant_id` or map to current `org_id` (decide based on existing auth model)
- `hash_sha256` unique per tenant/org (idempotency)
- metadata: `collection`, `tags[]`, `file_name`, `mime_type`, `language`, etc.

#### parent_chunks
- `id uuid pk`
- `document_id`
- `tenant_id`
- `index_in_document`
- `text`
- `token_count`
- provenance: `page_start`, `page_end`, `section_heading`

#### child_chunks
- `id uuid pk`
- `parent_id`, `document_id`, tenant
- `index_in_parent`
- `text`, offsets, `page`
- FTS `tsv` column (generated)
- `embedding_1024 vector(1024)` + HNSW index
- `embedding_4096 vector(4096)` optional
- `content_hash` unique (tenant, content_hash) for dedup

#### entities / relations (+ mentions)
As in the spec: `entities`, `entity_mentions`, `relations`.

---

## 3) Phased execution plan

We’ll execute as a sequence of small, safe phases. Each phase ends with a **verifiable gate** (tests + smoke checks).

### Phase 0 — Alignment + inventory (no behavior change)

**Goal**: map current RAG modules and DB schema to the new target so we can refactor without breaking prod behavior.

Steps:

1. Inventory current code entrypoints:
   - ingestion orchestrator
   - retrieval + reranker
   - DB search functions
   - feature flags
2. Inventory current DB schema:
   - confirm current vector column types (halfvec vs vector)
   - confirm current dimension limits used in prod
3. Decide “tenant_id vs org_id” mapping for the new spec tables.
4. Write a short compatibility note:
   - how old `knowledge_base_chunks` maps to new `documents/parent_chunks/child_chunks`.

**Gate**:
- Documentation updated + no code changes.

---

### Phase 1 — Introduce new schema (dual tables, no cutover)

**Goal**: land the RAG2.0 tables without affecting existing queries.

Steps:

1. Add migration(s) creating:
   - `documents`, `parent_chunks`, `child_chunks`
   - `entities`, `entity_mentions`, `relations`
2. Add indexes:
   - HNSW on `child_chunks.embedding_1024`
   - GIN on `child_chunks.tsv`
   - unique dedup indexes for `(tenant_id, hash_sha256)` and `(tenant_id, content_hash)`
3. Add DB helper functions (optional) for:
   - lexical search
   - vector search
   - hybrid RRF

**Gate**:
- Migration applies cleanly on dev DB.
- DB constraints work (quick insert tests).

---

### Phase 2 — Embedder refactor: Matryoshka 4096→1024 (shared library)

**Goal**: implement a single embedding client that can:

- call Qwen3 embedding endpoint (vLLM/LM Studio)
- return 4096d vectors
- truncate to 1024d (MRL)
- normalize vectors

Steps:

1. Create a small embedding module with a clear interface:
   - `embed_texts(texts: list[str]) -> list[EmbeddingResult]`
2. Add config:
   - `RAG_EMBED_DIM_STORE=1024`
   - `RAG_EMBED_DIM_MODEL=4096`
   - `RAG_EMBED_TRUNCATE_STRATEGY=matryoshka_prefix` (first N dims)
3. Add unit tests:
   - truncation produces exactly 1024 dims
   - normalization is stable

**Gate**:
- Unit tests pass.
- We can generate and store 1024d vectors (even if retrieval still uses old table).

---

### Phase 3 — Ingestion pipeline v2 (write new tables, keep old pipeline alive)

**Goal**: implement ingestion into `documents/parent_chunks/child_chunks`.

Steps:

1. Document registration:
   - compute sha256
   - upsert document row
   - return existing document_id if already ingested (unless forced)
2. Text acquisition:
   - reuse existing loader/OCR logic
   - add OCR validation hooks (score, fallback)
3. Hierarchical chunking:
   - parent (800–1000 tokens)
   - child (~200 tokens)
   - stable IDs + offsets
4. Dedup:
   - chunk-level hash after normalization
   - skip duplicates via DB uniqueness (and log counts)
5. Embedding:
   - embed child chunks
   - store `embedding_1024`

**Gate**:
- Ingestion of a small doc produces:
  - 1 document row
  - N parent rows
  - M child rows with embeddings
- Re-ingestion is idempotent.

---

### Phase 4 — Retrieval pipeline v2 (lexical + semantic only)

**Goal**: implement QueryPlanner + (lexical + semantic) + weighted RRF + expansion + rerank.

Steps:

1. Implement QueryPlannerAgent (GPT‑5) returning structured JSON:
   - keywords
   - semantic_query_text
   - top_k_lexical, top_k_semantic
   - weights
2. Implement LexicalRetrievalAgent (FTS on `child_chunks.tsv`).
3. Implement SemanticRetrievalAgent (HNSW on `child_chunks.embedding_1024`).
4. Implement FusionAgent (Weighted RRF).
5. Implement ContextExpansionAgent (child→parent; aggregate scores).
6. Implement RerankerAgent (Qwen reranker) on parent chunks.
7. Implement SafetyAndDenoisingAgent:
   - hard threshold: max_score < 0.6 → refuse
   - trimming heuristic: drop chunks < α * max_score (start with α=0.6)

**Gate**:
- Deterministic unit tests for:
  - RRF scoring
  - expansion aggregation
  - refusal behavior

---

### Phase 5 — Add Graph channel (PuppyGraph) + KG ingestion

**Goal**: implement the third channel and connect it to the same fusion funnel.

Steps:

1. Implement KnowledgeGraphAgent:
   - GPT‑5 NER + relation extraction outputs entities/relations
   - dedup entities by (tenant, type, canonical_name)
2. Add PuppyGraph schema mapping for the new tables.
3. Implement GraphRetrievalAgent:
   - execute Cypher against PuppyGraph
   - map results back to document/parent/child IDs
4. Add Graph channel into Weighted RRF.

**Gate**:
- Integration test: a query that requires a relation hop returns relevant parents.

---

### Phase 6 — Dual-read shadowing + evaluation

**Goal**: compare old vs new retrieval quality before full cutover.

Steps:

1. Build an eval set (start small, 20–50 queries) from:
   - common voice queries
   - “hard” docs (tables, scanned PDFs)
2. Implement a shadow mode:
   - run both pipelines
   - record metrics and overlaps
3. Track metrics:
   - Recall@k, NDCG@10, MRR
   - refusal rate
   - latency per stage

**Gate**:
- New pipeline matches or beats baseline on the eval set.

---

### Phase 7 — Cutover + cleanup

**Goal**: switch production to the new pipeline and then simplify.

Steps:

1. Enable new retrieval behind a feature flag.
2. Roll out gradually.
3. After stability window, deprecate:
   - old vector columns/dimensions (4000 halfvec)
   - old RAG code paths
4. Document rollback plan.

**Gate**:
- p95 latency and refusal rate within budget.
- No regression in core business flows.

---

## 4) Cross-cutting concerns (do throughout)

### 4.1 Feature flags

Keep these toggles to de-risk rollout:

- `RAG2_ENABLED`
- `RAG2_GRAPH_ENABLED`
- `RAG2_RERANK_ENABLED`
- `RAG2_DENoise_ENABLED`
- `RAG2_MATRYOSHKA_DIM=1024`

### 4.2 Observability

Add stage timings and critical counters:

- ingestion: docs ingested, OCR fallbacks, dedup skips
- retrieval: channel timings, fusion timing, rerank timing
- safety: refusals and max score distribution

### 4.3 Performance guardrails

- Limit rerank candidates to 20–50 parents.
- Limit final context to 5–10 parents.
- Cache query embeddings (short TTL) if needed.

---

## 5) Implementation order (smallest useful increments)

If we want the fastest “walking skeleton”:

1. Phase 1 (schema)
2. Phase 2 (Matryoshka embedder)
3. Phase 3 (ingestion to new tables)
4. Phase 4 (retrieval with lexical+semantic only)

Then add graph (Phase 5) once we have end-to-end correctness.

---

## 6) Notes on the Matryoshka truncation choice

MRL truncation is assumed to be “prefix truncation”: keep the first $d$ dimensions.

If the Qwen embedding endpoint returns a single 4096d vector $v$, we store:

$$v_{1024} = v[0:1024]$$

We should still L2-normalize the stored vector:

$$\hat{v}_{1024} = \frac{v_{1024}}{\|v_{1024}\|_2}$$

This is consistent with typical cosine ANN search.

---

## 7) Open decisions (to close early)

These are the only decisions we need before heavy implementation:

1. **Tenant key**: use `org_id` (existing) vs introduce `tenant_id` (spec).
2. **Vector type** for 1024 dims:
   - `vector(1024)` vs `halfvec(1024)`
3. **Where to store FTS**:
   - child only (recommended) vs parent only

All three can be implemented as configs, but we should choose defaults.
