# DeepSeek OCR + RAG Architecture for `voice-agent-v5`

Audience: engineers implementing KB ingestion/retrieval with OCR and optional image semantics.
Status: draft blueprint (implementation-ready checklist).

## 1) Goals & Scope
- Ingest **PDF / images / TXT / DOCX / CSV / XLSX** into the knowledge base with semantic + hybrid search.
- Preserve layout context via DeepSeek OCR; capture visual semantics via **SigLIP base** image embeddings for chart/diagram/screenshot-heavy pages.
- Normalize heterogeneous files into structured text (and optional image crops/captions) with provenance for the LLM tools (`search_knowledge_base`, etc.).
- Bake in production needs: OCR validation/fallback, table-aware handling, deduplication, hybrid + rerank retrieval, and observability.

## 2) Modes (pick based on corpus)
- **Text-first (baseline)**: DeepSeek OCR → structured text → text embeddings → vector search. Use when docs are mostly textual; cheapest and fastest.
- **Text + selective image embeddings**: Add image encoder only for pages with charts/diagrams/screenshots where text is insufficient.
- **Hybrid retrieval**: Combine BM25/FTS + vectors for better recall (optional).

## 3) Data model (final deltas)
Use a companion **`knowledge_base_chunks`** table (parent `knowledge_base` keeps doc-level metadata):
- `modality` (text | image | image_caption | table)
- `source_document` (string)
- `page` (int), `chunk_index` (int)
- `content_hash` (string, unique per org) — dedup
- `ocr_confidence` (float) — OCR quality flag
- `alt_text` (string; OCR summary or caption for images)
- `vector_embedding` (vector(1536)) — text via **OpenAI text-embedding-3-small**
- `vector_image` (vector(768)) — image embeddings via **SigLIP base (patch16-384)**
- `is_table` (bool), `table_context` (string), optional `table_type`
- `created_at`, `updated_at`

## 4) Ingestion pipeline
1) **Load & normalize by type**
   - PDF → per-page images (`pdf2image` 300 DPI) + extract native text layer if present; OCR fallback for images/scans.
   - Images (png/jpg/webp) → direct OCR; crop visuals if needed for image embeddings.
   - TXT → direct text load (no OCR).
   - DOCX → extract text (e.g., `python-docx`) preserving headings/lists; OCR only for embedded images if present.
   - CSV/XLSX → extract tables; render as Markdown tables; optional light summarization of wide tables; OCR not needed unless cells contain images.
2) **OCR (DeepSeek) where needed**: Run Base/Large for pages or embedded images; obtain Markdown/JSON with layout preserved.
3) **OCR validation & fallback**: capture `ocr_confidence`, token density, table alignment; if low confidence or complex tables, retry with higher mode/tiling or table-focused parser.
4) **Segment**:
   - Text: split into 200–400 token chunks, keep headings; fixed-size default, measure on your KB.
   - Tables: detect, keep whole; add `table_context` description; embed context + table markdown together.
   - Images/visuals: for charts/diagrams/screenshots, keep original or crop; generate `alt_text` (caption-first approach preferred).
5) **Deduplicate**: hash normalized content (`content_hash`) per org; skip inserts on collisions.
6) **Embed**:
   - Text chunks → `text-embedding-3-small` (1536d) → `vector_embedding` (normalize before insert).
   - Image regions/pages with charts/diagrams/screenshots → **SigLIP base** (768d) → `vector_image`; also store caption/`alt_text` and a text chunk (embedded with 3-small) for cross-modal recall.
7) **Store**:
   - Insert rows with `org_id`, `category`, `title`, `content` (chunk text/table markdown/caption), `modality`, `page`, `chunk_index`, `source_document`, hashes, confidence, embeddings.
8) **Provenance**: Always keep `source_document` + `page` + `chunk_index` for citations and deduping.

## 5) Retrieval flow
- Query → text embed → BM25/FTS top-50 + Vector ANN top-50 → RRF fusion (default on)
- Optional image branch: text→image or caption-based retrieval on `vector_image` (SigLIP) for visual matches; merge if enabled
- Cross-encoder rerank fused top-50/100 (default on; MiniLM-size) → top-5/10
- Return top-k with snippets, `source_document`, `page`, `modality`, `ocr_confidence`; LLM cites sources.

## 6) Module breakdown
- **Loader/Parser**:
   - PDF: image render + text layer extraction
   - DOCX: structural text extractor (headings/lists)
   - TXT: raw text reader
   - CSV/XLSX: table-to-Markdown converter (optionally summarize wide tables)
   - Images: passed to OCR directly
- **OCR module**: DeepSeek OCR runner (vLLM or Transformers) converting images/scans/embedded images to structured text.
- **OCR validator**: Computes `ocr_confidence`, detects tables/low-DPI; triggers fallback.
- **Chunker**: Semantic/paragraph splitter with table-preserving rules; assigns `page` and `chunk_index`.
- **Embedder (text)**: `text-embedding-3-small` client; batches chunks; writes to `vector_embedding` (1536d).
- **Embedder (image)**: SigLIP base (patch16-384, 768d); processes selected images; writes to `vector_image`; stores `alt_text` and caption text chunk.
- **Ingest writer**: Supabase client to upsert rows into `knowledge_base` (or `_chunks`).
- **Retrieval API/Tool**: Update `search_knowledge_base` to use hybrid + rerank (and optional image branch) with filters.
- **Evaluator/Observability**: RAG metrics (context precision/recall, MRR/NDCG, faithfulness/relevancy), latency p50/p95/p99, OCR error/confidence tracking.

## 7) Configuration knobs
- `OCR_MODE` (tiny/small/base/large/gundam) — trade accuracy vs cost.
- `CHUNK_SIZE`, `CHUNK_OVERLAP` — default 1000 chars, 200 overlap for markdown; adjust for tables.
- `EMBED_MODEL_TEXT` — default `text-embedding-3-small` (1536d fits schema).
- `EMBED_MODEL_IMAGE` — default **`google/siglip-base-patch16-384`** (768d) when image branch enabled.
- `TOP_K_TEXT`, `TOP_K_IMAGE` — e.g., 8 and 3; merge by score.
- `USE_HYBRID_BM25` — default on (BM25 + vector with RRF).
- `RERANKING_ENABLED` — default on; top_k_retrieve=50-100, top_k_rerank=5-10.
- `OCR_CONFIDENCE_THRESHOLD` — triggers fallback OCR mode.
- `DEDUP_ENABLED` — default on.
- `BATCH_SIZE_EMBED`, `MAX_WORKERS` — control async/batched embedding.

## 8) Minimal implementation path (finalized)
1) Schema: add `knowledge_base_chunks` with `modality`, `page`, `chunk_index`, `source_document`, `content_hash`, `ocr_confidence`, `is_table`, `table_context`, `vector_embedding` (1536d), `vector_image` (768d), `alt_text`; keep `knowledge_base` as doc parent.
2) Ingestion: OCR with validation → table-aware chunking → dedup → batched embeddings (3-small for text, SigLIP for visuals) → insert with provenance.
3) Retrieval: hybrid (BM25 + vector + RRF) and reranking ON by default; image branch enabled for visual pages; return provenance + confidence.
4) Evaluation: add RAG metrics (context precision/recall, MRR/NDCG, faithfulness) and latency tracking; set a baseline.
5) Image/diagram captions: store as `modality=image_caption` and also embed images with SigLIP for visual recall.

## 9) Optional enhancements
- Add `vector_image` + `alt_text`; enable text→image search for charts/diagrams.
- Rerank top-20 with a cross-encoder for higher precision.
- Table-aware post-processing: repair markdown tables; keep tables unsplit.
- Deduplication: hash normalized text per `source_document` to avoid near-duplicates (recommended core).

## 10) ASCII dataflow diagram (text-first path)

User Query → Text Embed → Vector Search (text) ┐
                                     Filters ───┤ (org_id/category/source)
Results (chunks + provenance) ────────┘
↓
LLM Prompt w/ cited chunks → Answer

Ingestion: PDF/Image → DeepSeek OCR → Chunks → Text Embeds → DB (knowledge_base)

## 11) ASCII dataflow diagram (with optional image embeddings)

PDF/Image → DeepSeek OCR → Text Chunks → Text Embeds → DB (text vectors)
        └→ Visual Crops (charts/figures) → Image Embeds → DB (image vectors)

Query → Text Embed →
   ├─ Vector Search (text)
   └─ Text→Image Embed → Vector Search (image)
Merge/Rerank → LLM with text snippets + image alt_text + provenance → Answer

## 12) Acceptance checklist
- [ ] Text chunks stored in `knowledge_base_chunks` with `vector_embedding` (1536d), `org_id`, `source_document`, `page`, `chunk_index`, `modality=text`.
- [ ] Image/visual chunks stored with `vector_image` (768d, SigLIP) + `alt_text` and `modality=image`/`image_caption`.
- [ ] `search_knowledge_base` uses hybrid (BM25 + vector + RRF) and reranking; returns provenance and `ocr_confidence`.
- [ ] Image branch enabled for visual pages; results include `modality=image`/`image_caption` + `alt_text`.
- [ ] Ingestion retries failed pages and logs per `source_document`/`page`.
- [ ] Table integrity: tables not split across chunks; markdown validated.
- [ ] Dedup enabled via `content_hash` per org.
- [ ] Evaluation metrics collected (context precision/recall, faithfulness, latency p50/p95).

## 13) Suggested file updates (when implementing)
- `database/schema.sql`: add `modality`, `page`, `chunk_index`, `alt_text`, `vector_image` (if chosen), indexes on `(org_id, source_document, modality)`.
- `src/voice_agent/tools/crm_knowledge.py`: add vector search path; keep filters.
- `scripts/` (new): `ingest_kb.py` to run OCR → chunk → embed → insert.
- `.env`: add model IDs and toggles for image embeddings/hybrid search.

## 14) Defaults to start with
- OCR: DeepSeek Base mode (1024×1024, ~256 tokens) for balanced accuracy/cost.
- Embeddings: `text-embedding-3-small` (fits 1536d column); top_k=8.
- Chunking: 1000 chars, 200 overlap; keep tables intact.
- Hybrid search + rerank: on by default (BM25 + vector + RRF; rerank to top-5/10).
- Dedup: on.
- Image embeddings/captions: off by default; enable for chart-heavy sources only.

## 15) Risks & mitigations
- **Long tables/complex pages**: use higher OCR mode or tile; keep tables unsplit.
- **OCR failures on low-DPI scans**: upscale to 300 DPI; retry with higher mode.
- **Cost/latency**: batch embeds; keep image embeddings selective; track ingestion/query costs.
- **Relevance**: hybrid + reranker enabled by default; monitor precision/recall metrics.
- **Observability**: add alerts on drops in context precision/faithfulness or latency spikes.

## 16) Pre-flight checklist (detailed)

**Data & schema**
- Confirm language mix (PT/EN); if heavy multilingual, validate `text-embedding-3-small` or pick a multilingual model (e.g., bge-m3). Keep 1536d schema alignment or update dim.
- Finalize schema fields: `modality`, `source_document`, `page`, `chunk_index`, `content_hash`, `ocr_confidence`, `is_table`, `table_context`, optional `table_type`, `vector_image` if using image vectors.
- Indexes: `(org_id, source_document)`, `(org_id, content_hash)` unique; FTS GIN on `content`; IVFFlat/HNSW on vectors. Verify RLS covers new fields.

**Ingestion quality controls**
- OCR validation: record `ocr_confidence`, token density, table alignment; thresholds that trigger fallback (higher mode/tiling or alternate parser).
- Tables: detect reliably; keep whole; generate `table_context` (what/when/units); ensure row/col counts make sense post-extraction.
- Content normalization: strip headers/footers/boilerplate before hashing; normalize whitespace/numbers for stable hashes; skip duplicate hashes per org.
- Incremental updates: plan reindex strategy (per `source_document` versioning or soft-delete + reinsert) and a retry queue for failed pages.

**Performance & scaling**
- Batch sizing: profile OCR and embeddings for your hardware; set `BATCH_SIZE_EMBED` and concurrency to avoid OOM/CPU thrash.
- Parallelism: cap concurrent workers to respect DB/API limits; add backpressure; pre-warm OCR/embedding clients on startup.
- Index tuning: choose IVFFlat/HNSW parameters (lists/ef) for target latency and corpus size; schedule `ANALYZE`/reindex maintenance.
- Caching: enable short-term query cache and embedding cache for repeated chunks to shave latency/cost.

**Retrieval & ranking**
- Default pipeline: BM25 + vector + RRF → cross-encoder rerank to top-5/10. Set explicit latency budgets (e.g., p95 retrieve+rerank < 400 ms) and adjust `top_k_retrieve` accordingly.
- Diversity/MMR: if you see near-duplicate hits, add MMR before rerank to increase coverage.
- Filters: enforce `org_id`/`category` filters; ensure provenance (source, page, modality, confidence) returned to LLM.

**Evaluation & observability**
- Ground-truth set: assemble 50–200 domain queries covering tables, references (“Section 3.2”), diagrams, multilingual. Use RAGAS/CoFE-RAG for context precision/recall, faithfulness, relevancy; track MRR/NDCG.
- Canary/CI: daily smoke tests on fixed queries; alert on drops in context precision or faithfulness; monitor OCR confidence drift.
- Operational metrics: retrieval latency p50/p95/p99; rerank latency; cache hit rate; ingestion throughput; OCR error/low-confidence rate by source.

**Security & compliance**
- PII handling: define masking/stripping rules pre-store; ensure logs do not capture raw content.
- Access: verify RLS and API filters; avoid leaking cross-org data in hybrid/rerank paths.

**Optional accelerators**
- Captioning for diagrams/screenshots if visual content matters; store as `modality=image_caption` with embeddings.
- Lightweight reranker fallback (e.g., mono-mpnet) if GPU isn’t available; keep MiniLM as default when possible.
- Query normalization (light): unit/number normalization, abbreviation expansion; avoid heavy LLM rewrite to keep latency low.
