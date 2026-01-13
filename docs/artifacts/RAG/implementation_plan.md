# RAG One-Shot Implementation Plan (Text: OpenAI text-embedding-3-small @1536d, Image: SigLIP Base @768d)

Status: implementation blueprint. Scope covers schema, ingestion, embeddings, retrieval, evaluation/observability, and rollout. Choices finalized: text embeddings via `text-embedding-3-small` (1536d, SaaS), image embeddings via `google/siglip-base-patch16-384` (768d, multilingual-friendly, good for PT sources). Schema split into document-level (`knowledge_base`) and chunk-level (`knowledge_base_chunks`).

---

## Phase 0 — Pre-flight
- Backup current DB; snapshot existing `knowledge_base` embeddings (1536d) for rollback.
- Confirm env vars available: OpenAI key, DeepSeek OCR endpoint/model, storage bucket for uploads if used.
- Verify current search API/tool entrypoint (likely `src/voice_agent/tools/crm_knowledge.py` or similar) and retrieval service wiring.
- Decide migration strategy:
  - **Dual-write/dual-read** (safe): add `knowledge_base_chunks`, backfill, shadow-read, then cut over.
  - **In-place** (fast): migrate and re-embed before serving.

## Phase 1 — Schema & Indexes
1) **Create chunk table** `knowledge_base_chunks` (new):
   - `id uuid pk default uuid_generate_v4()`
   - `org_id uuid not null references organizations(id)`
   - `knowledge_base_id uuid references knowledge_base(id)` (document-level parent)
   - `category text`, `title text`, `source_document text`
   - `modality text check (modality in ('text','image','image_caption','table'))`
   - `page int`, `chunk_index int`
   - `content text not null`
   - `content_hash text not null`
   - `ocr_confidence real`
   - `is_table boolean`, `table_context text`, `alt_text text`
   - `vector_embedding vector(1536)`  -- text embeddings (3-small)
   - `vector_image vector(768)`        -- SigLIP image embeddings
   - `created_at timestamptz default now()`, `updated_at timestamptz default now()`
2) **Indexes/constraints**:
   - Unique `(org_id, content_hash)` for dedup.
   - B-tree `(org_id, source_document, modality)`.
   - GIN FTS on `to_tsvector('portuguese', content)` (or configurable language).
   - IVFFlat/HNSW on `vector_embedding` with cosine/IP; same for `vector_image`.
3) **Optionally add columns to parent `knowledge_base`** for provenance (if needed): `source_document`, `doc_checksum`, `ingestion_metadata jsonb`.
4) **RLS**: mirror org isolation on `knowledge_base_chunks`.
5) **Migration scripts**: place SQL in `database/migrations/<timestamp>_kb_chunks.sql`; include index creation and RLS.

## Phase 2 — Configuration
- `.env` / config entries:
  - `EMBED_MODEL_TEXT=text-embedding-3-small`
  - `VECTOR_DIM_TEXT=1536`
  - `EMBED_MODEL_IMAGE=google/siglip-base-patch16-384`
  - `VECTOR_DIM_IMAGE=768`
  - `ENABLE_IMAGE_EMBEDDINGS=true`
  - `USE_HYBRID_BM25=true`, `TOP_K_RETRIEVE=50`, `TOP_K_RERANK=5-10`, `TOP_K_IMAGE=3-5`
  - `OCR_MODE=base`, `OCR_CONFIDENCE_THRESHOLD=0.6`, `RETRY_LIMIT_OCR=2`
  - `CHUNK_SIZE=1000_chars`, `CHUNK_OVERLAP=200_chars`, `DEDUP_ENABLED=true`
  - `BATCH_SIZE_EMBED`, `MAX_WORKERS` (set per infra)
- Add startup validation: fail fast if configured dims mismatch DB schema.

## Phase 3 — Ingestion Pipeline (files → chunks → embeddings → DB)
1) **Detect type**: pdf / image / txt / docx / csv / xlsx.
2) **Normalize/load**:
   - PDF: extract native text; if density/confidence low → render 300 DPI images.
   - Images: direct OCR; upscale/tiling for low DPI or large tables.
   - DOCX: structured text via python-docx; OCR embedded images if present.
   - CSV/XLSX: convert to Markdown tables; optional summarization for wide tables; no OCR unless images.
3) **OCR (DeepSeek Base)**:
   - Run base by default; if `ocr_confidence < threshold` or tables misaligned → retry with Large or tiling.
   - Capture `ocr_confidence` per page.
4) **Chunking**:
   - Text: ~1k chars with ~200 overlap; keep headings; assign `page`, `chunk_index`.
   - Tables: do not split; store full Markdown; set `is_table=true`, `table_context` description.
   - Images/visuals: generate caption (`alt_text`) and set `modality=image_caption`; also store image crop path if needed.
5) **Dedup**:
   - Normalize whitespace/headers/footers → hash per org (`content_hash`). Skip insert on collision.
6) **Embeddings**:
   - Text chunks → `text-embedding-3-small` (1536d) → `vector_embedding`. L2-normalize before insert.
   - Image regions (selected pages with charts/diagrams/screenshots) → SigLIP image encoder → `vector_image`. Also store caption as text chunk (embedded via text model) for cross-modal recall.
7) **Insert** into `knowledge_base_chunks` with provenance (`org_id`, `knowledge_base_id`, `source_document`, `page`, `chunk_index`, `modality`, `ocr_confidence`, `is_table`, `table_context`, hashes, vectors).
8) **Logging/metrics**: per doc/page ingest duration, OCR retries, low-confidence pages, dedup skips, embedding batches.

## Phase 4 — Retrieval Pipeline (backend/tooling)
1) **Query embed (text)**: `text-embedding-3-small` → normalized vector.
2) **Hybrid retrieve**:
   - BM25/FTS over `content` (top-50).
   - ANN over `vector_embedding` (top-50).
   - Optional image branch: text→image (use SigLIP text encoder if available, or fuse via captions already text-embedded) to query `vector_image` (top `TOP_K_IMAGE`).
   - RRF fuse results; keep filters `(org_id, category, source_document)`.
3) **Rerank**: Cross-encoder MiniLM on fused top-50 → final top-5/10 with scores.
4) **Return**: `content`, `modality`, `source_document`, `page`, `chunk_index`, `ocr_confidence`, `is_table`, `table_context`, `alt_text`.
5) **API/tool update**: Update `search_knowledge_base` (and any voice-agent tools) to read from `knowledge_base_chunks`, accept hybrid/rerank toggles, and include provenance in responses.

## Phase 5 — Backfill & Cutover
1) Export existing `knowledge_base` rows; for each row, create one or more chunks (simple 1k/200 splitting) and embed with 3-small; write to `knowledge_base_chunks`.
2) Build vector indexes for text and image columns after bulk load.
3) Shadow-read: route a small percent of queries through new retrieval; compare overlap/quality.
4) Cutover: switch default retrieval to chunks table; keep legacy vector in `knowledge_base` only for rollback window; plan removal later.

## Phase 6 — Quality & Evaluation
- Create a PT/EN eval set (50–200 queries) with relevance labels, including tables and image-heavy pages.
- Metrics: NDCG@10, Recall@10/20, MRR, context precision/recall (RAGAS), OCR confidence distribution, latency p50/p95.
- Track cost: embeddings per doc, OCR retries, rerank latency.
- Add CI smoke: fixed queries asserting non-empty results and provenance fields.

## Phase 7 — Observability & Ops
- Metrics: ingest throughput, OCR retry rate, dedup hit rate, embedding QPS, retrieval latency (per stage), rerank latency, cache hit rate if applicable.
- Logs: low OCR confidence warnings, index errors, similarity cutoff misses.
- Alerts: spike in OCR failures, drop in NDCG/Recall (from periodic eval), latency p95 breaches.

## Phase 8 — Performance/Cost Guardrails
- Ingestion: batch embeddings; cap `MAX_WORKERS`; skip pages after `RETRY_LIMIT_OCR` with clear log.
- Retrieval: limit `TOP_K_RETRIEVE` to 50; `TOP_K_RERANK` to 5–10; enforce request timeouts.
- Storage: periodic vacuum/analyze; monitor index size; consider MRL/quantization for image vectors if memory pressure.

## Phase 9 — Deliverables & Code Touchpoints
- **DB**: new migration in `database/migrations/*_kb_chunks.sql` (table, indexes, RLS). Update `database/schema.sql` snapshot if maintained manually.
- **Config**: defaults in `src/voice_agent/config.py` (or equivalent); env template update.
- **Ingestion**: new module `src/voice_agent/ingestion/kb_ingest.py` (loader → OCR → chunker → embedder → writer); CLI entry under `scripts/ingest_kb.py`.
- **Retrieval**: update retrieval service/tool (e.g., `src/voice_agent/tools/crm_knowledge.py` or backend API) to use hybrid + rerank + image fusion; add unit tests.
- **Tests**: ingestion unit tests (chunking, dedup, table handling); retrieval tests (hybrid on/off, rerank on/off, image branch); migration smoke tests.
- **Docs**: this plan; update `docs/artifacts/RAG/voice_agent_specific_impacts.md` if interface changes.

## Phase 10 — Rollout Plan
1) Deploy schema migration; keep legacy path active.
2) Run backfill; build indexes; verify counts.
3) Shadow traffic (10–20%) to new retrieval; monitor metrics.
4) Enable full traffic; keep rollback for 24–48h.
5) Remove legacy vector dependency in `knowledge_base` after stability window.

---

### Default knobs (ready to drop into config)
- `EMBED_MODEL_TEXT=text-embedding-3-small`
- `VECTOR_DIM_TEXT=1536`
- `EMBED_MODEL_IMAGE=google/siglip-base-patch16-384`
- `VECTOR_DIM_IMAGE=768`
- `ENABLE_IMAGE_EMBEDDINGS=true`
- `CHUNK_SIZE=1000` chars, `CHUNK_OVERLAP=200`
- `TOP_K_RETRIEVE=50`, `TOP_K_IMAGE=3`, `TOP_K_RERANK=5`
- `USE_HYBRID_BM25=true`, `RERANKING_ENABLED=true`
- `OCR_MODE=base`, `OCR_CONFIDENCE_THRESHOLD=0.6`, `RETRY_LIMIT_OCR=2`
- `DEDUP_ENABLED=true`, `BATCH_SIZE_EMBED=<tune>`, `MAX_WORKERS=<tune>`

### Notes on text/image fusion
- Primary relevance comes from text embeddings; image vectors are additive for charts/diagrams/screenshots. Store captions as text chunks to ensure text-only queries still match image-heavy pages.
- If using SigLIP text tower for text→image search, add a small wrapper to produce query vectors for `vector_image`; otherwise rely on captions in text space.
