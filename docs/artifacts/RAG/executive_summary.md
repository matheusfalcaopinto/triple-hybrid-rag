# Executive Summary: RAG Architecture Critique

**Status**: Draft blueprint validated against 2024-2025 research; 10 critical gaps identified.

**Bottom Line**: Your proposal is **architecturally sound** but **underspecified** in production-critical areas. The revised "Minimal Implementation Path" addresses gaps while maintaining realistic timelines.

---

## Quick Wins vs Critical Gaps

### ✅ What You Got Right

| Area | Status | Evidence |
|------|--------|----------|
| **Modular design** | ✅ | Loader → Parser → Chunker → Embedder → Storage (correct decomposition) |
| **DeepSeek OCR choice** | ✅ | Best efficiency (100-256 tokens/page) for your use case |
| **Fixed-size chunking default** | ✅ | 2024 research shows fixed beats semantic (ArXiv 2410.13070) |
| **Provenance tracking** | ✅ | source_document + page + chunk_index critical for citations |
| **Text-first baseline** | ✅ | Pragmatic; images deferred unless needed |
| **Phased approach** | ✅ | "Minimal path" is smart (defer complexity) |

---

### ⚠️ Critical Gaps (Must Fix)

| Gap | Risk | Your Proposal | Recommendation | Impact |
|-----|------|---------------|-----------------|---------| 
| **OCR validation** | HIGH | "OCR fallback" vague | Add confidence scoring + fallback decision tree | Prevents 25% accuracy loss in production |
| **Table preservation** | HIGH | "Keep tables whole" undefined | Implement table-aware chunking (context + markdown) | 30-50% accuracy gain on financial docs |
| **Chunking eval** | HIGH | No metrics | Add RAGAS framework (context precision, F1-score) | Quantify performance vs baseline |
| **Reranking** | MEDIUM | Optional | Make default (cross-encoder MiniLM) | +20-35% accuracy, +120ms latency acceptable for voice |
| **Hybrid search** | MEDIUM | Optional | Enable by default (BM25 + vector w/ RRF) | Handles both "Model XR-450" (keyword) and "contract terms" (semantic) |
| **Image embeddings** | LOW-MEDIUM | Underspecified | Use caption-based approach (not CLIP directly) | Avoids text-to-text bias in CLIP |
| **Cost/latency** | MEDIUM | No numbers | Document cost model ($1-2K ingestion, <200ms p95) | Budget planning |
| **Deduplication** | MEDIUM | Optional | Move to core (content_hash) | Saves cost, improves relevance |
| **Batch embedding** | LOW-MEDIUM | Sequential | Use ray/asyncio | 50K docs: 8 hours → 30 mins |
| **Observability** | HIGH | Checklist only | Quantified metrics (context precision, latency tracking) | Catch regressions before users notice |

---

## Before vs After

### Schema

**Before:**
```sql
CREATE TABLE knowledge_base (
    id SERIAL PRIMARY KEY,
    org_id UUID,
    content TEXT,
    vector_embedding VECTOR(1536)
);
```

**After:**
```sql
CREATE TABLE knowledge_base (
    -- Original
    id SERIAL PRIMARY KEY,
    org_id UUID,
    content TEXT,
    vector_embedding VECTOR(1536),
    
    -- New: Production essentials
    modality VARCHAR(50),         -- text, image_caption, table
    source_document VARCHAR(500), -- "proposal_2025_Q4.pdf"
    page INT,                     -- Page number
    chunk_index INT,              -- Chunk number on page
    content_hash VARCHAR(64) UNIQUE,  -- SHA256 for dedup
    ocr_confidence FLOAT,         -- 0.0-1.0
    is_table BOOLEAN,
    table_context TEXT,           -- "Q4 revenue by region"
    alt_text TEXT,
    vector_image VECTOR(1536),    -- CLIP if enabled
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

-- Critical indexes
CREATE INDEX idx_org_source ON knowledge_base(org_id, source_document);
CREATE INDEX idx_content_hash ON knowledge_base(org_id, content_hash);
CREATE INDEX idx_fts ON knowledge_base USING GIN(to_tsvector('portuguese', content));
CREATE INDEX idx_vector ON knowledge_base USING ivfflat(vector_embedding);
```

---

### Retrieval Pipeline

**Before (Your Proposal):**
```
Query → Text Embed → Vector Search → Return top-k → LLM
```

**After (Production):**
```
Query → Text Embed
         ├─ Vector Search (top-50) ──────┐
         └─ BM25 Search (top-50) ────────┤
                                         └─ RRF Fusion
                                         ├─ Hybrid results (top-20)
                                         └─ Cross-Encoder Rerank (top-10)
                                         └─ LLM with citations
```

**Why:**
- Hybrid search handles both concept ("contract terms") and keyword ("Model XR-450") queries
- Reranking improves accuracy 20-35% with modest latency (acceptable for voice)
- RRF balances both methods mathematically

---

### Ingestion Pipeline

**Before:**
```
PDF → DeepSeek OCR → Chunk (1000 chars) → Embed → DB
```

**After:**
```
PDF/Image/XLSX
├─ DeepSeek OCR
├─ Validation (confidence, table alignment, compression ratio)
│  └─ If low confidence → Fallback to larger model
├─ Table-Aware Chunking
│  ├─ Detect tables
│  ├─ Preserve markdown
│  ├─ Generate context description
│  └─ Combine for embedding
├─ Text Chunking (preserve boundaries)
├─ Deduplication (content_hash)
├─ Embedding (batched, parallel with ray)
└─ Insert with provenance (source_document, page, confidence)
```

---

## Research-Backed Decisions

### Decision 1: Fixed-Size Chunking (Not Semantic)

**Your Proposal**: Default 1000 chars, mention semantic as optional

**Research Finding** (ArXiv 2410.13070, Oct 2024 — Vectara/Vespa analysis of 128 papers):
- Semantic chunking claims better performance but **not empirically validated** at scale
- Fixed-size chunking **outperforms semantic** on real-world datasets
- Computational cost of semantic chunking not justified by gains
- **Recommendation**: Keep fixed-size default; validate empirically for your domain

**Action**: ✅ You're on the right track; just add evaluation metrics

---

### Decision 2: Hybrid Search (BM25 + Vector)

**Your Proposal**: "Optional boolean"

**Research Finding** (Multiple sources, 2024-2025):
- Hybrid search (BM25 + vectors) beats both methods alone by 15-25%
- Text-only query: "Model XR-450" → BM25 works better (exact match)
- Semantic query: "contract termination procedures" → vectors work better
- Production RAG systems use hybrid by default (Vespa, Weaviate, pgvector)

**Cost**: Negligible (FTS index + RRF scoring in PostgreSQL)

**Action**: Make default, not optional

---

### Decision 3: Cross-Encoder Reranking

**Your Proposal**: "Optional enhancement"

**Research Finding** (AILog 2025, NVIDIA, multiple studies):
- Reranking improves accuracy +20-35% (top-1 accuracy)
- Latency: +100-200ms (acceptable for voice agents)
- ROI: Retrieve 50 (fast) → rerank to 10 (accurate)
- Recommended model: cross-encoder-mmarco-MiniLM-L6-v2 (lightweight)
- Cost: ~$0.001-0.002 per query (negligible) or free if self-hosted

**For voice agents**: High precision critical (user won't re-query)

**Action**: Make default in v1

---

### Decision 4: DeepSeek OCR Accuracy Reality

**Your Proposal**: "Base mode 97% accuracy"

**Research Finding** (LabelYourData, 2025 — production benchmarking):
- Benchmark: 97% on clean PDFs (OmniDocBench)
- Production: **75-80% on financial documents** (dense tables, small fonts)
- Table misalignment: **30% of production failures**
- Compression ratio >20x: Accuracy drops to 60% (unusable)
- Training gap: 300K pages/language misses edge cases

**Why it matters**: If your KB has financial reports or scanned contracts, naive DeepSeek will hallucinate

**Action**: Implement OCR validation layer with fallback decision tree

---

### Decision 5: Table Handling

**Your Proposal**: "Keep tables whole when possible"

**Research Finding** (TableRAG, 2025 — ArXiv):
- Naive markdown linearization loses 30-50% of structure
- Multi-hop queries ("sum by region") fail on flattened tables
- Solution: Preserve structure + generate context + embed combined

**Example**:
```
Table markdown: | Region | Q1 | Q2 | Q3 | Q4 |
               | NA     | 50 | 55 | 60 | 65 |

Context: "Regional quarterly revenue 2025, 4 regions, 4 quarters"
Combined: "Context + Table markdown" (for embedding)
```

**Action**: Implement table-aware chunking

---

## Revised Phasing: 6-Week Implementation

### Week 1-2: Foundation
- ✅ Schema: Add modality, source_document, page, chunk_index, content_hash, ocr_confidence
- ✅ OCR validator: Confidence scoring + fallback logic
- ✅ Table-aware chunker: Detect, preserve structure, generate context
- ✅ Deduplication: Content hash

### Week 3-4: Retrieval & Evaluation
- ✅ Hybrid search: BM25 + vector with RRF fusion
- ✅ Cross-encoder reranking: Top-100 → top-10
- ✅ Evaluation framework: Context precision, faithfulness, latency tracking
- ✅ Provenance: Return citations to source_document + page

### Week 5-6: Optimization & Monitoring
- ✅ Parallel embedding: ray/asyncio for 50K docs
- ✅ Observability: OCR accuracy, retrieval latency, cache hit rates
- ✅ A/B testing: Chunking strategy on real KB
- ✅ Cost dashboard: Track ingestion + query costs

**Total effort**: ~200-250 engineer-hours (3-4 people × 6 weeks)

---

## Cost Estimate (Jan 2026 pricing)

### One-Time (Ingestion)

| Component | Input | Unit Cost | Total |
|-----------|-------|-----------|-------|
| DeepSeek OCR | 50K pages | $0.03/page | **$1,500** |
| Embedding | 50K pages × 1K chars | $0.02/1M tokens | **$1.00** |
| Infra setup | N/A | N/A | **$500** |
| **Total** | | | **~$2,000** |

### Monthly (Running)

| Component | Usage | Unit Cost | Total |
|-----------|-------|-----------|-------|
| Vector DB (pgvector) | 50GB | $0.15-1.00/GB | **$7.50-50** |
| Query embedding | 100 queries/day | $0.02/1M tokens | **$0.002** |
| Reranking | 100 queries/day | Self-hosted (GPU amortized) | **$0-5** |
| Infra | Supabase | TBD | **$20-100** |
| **Total** | | | **~$30-155/month** |

### Yearly
- Ingestion (one-time): $2K
- Running: $360-1,860
- **Total: $2.4-3.9K** (very cost-effective for production RAG)

---

## Key Metrics to Track

Implement observability **from day 1**:

```python
# Ingestion metrics
OCR confidence score distribution (histogram)
Table detection accuracy
Deduplication rate (% chunks skipped)
Embedding latency (p50, p95)

# Query metrics
Context precision (RAGAS)
Context recall
MRR (top-1 relevance)
Retrieval latency (p50, p95, p99)
Reranking latency
Cache hit rate

# Generation metrics
Faithfulness (answer grounded in context)
Answer relevancy (query-answer similarity)
F1-score (entity matching)
```

**Tool**: Use RAGAS for generation quality, custom scripts for retrieval

---

## Known Risks & Mitigation

| Risk | Likelihood | Mitigation |
|------|------------|-----------|
| DeepSeek OCR accuracy on financial docs | HIGH | Validation layer + fallback to Gundam mode |
| Tables fragmented by fixed-size chunking | HIGH | Table-aware chunking with context |
| Reranking latency exceeds budget | LOW | Retrieve 50, rerank to 10; GPU acceleration |
| Vector DB scale (>1M docs) | LOW | Switch to dedicated vector DB (Weaviate, Milvus) if needed |
| Hallucinations in generated answers | MEDIUM | Reranking + faithfulness evaluation catches most |
| OCR confidentiality (file uploads) | MEDIUM | Use vLLM self-hosted or encrypted API calls |

---

## Next Steps

### For Your Team

1. **Review** this critique (30 min)
2. **Validate** assumptions about your KB (What % tables? Financial docs? Scans?)
3. **Decide** on phasing (Can you parallel work in weeks 1-2?)
4. **Plan** evaluation (Do you have ground-truth Q&A pairs for domain?)
5. **Estimate** resources (Full-time engineer? Contract ML engineer?)

### Immediate Action Items

- [ ] Add schema columns (modality, source_document, page, chunk_index, content_hash, ocr_confidence)
- [ ] Implement OCR validator (copy pseudocode from implementation_recommendations.md)
- [ ] Pick reranking model (recommend: cross-encoder/mmarco-MiniLM-L6-v2)
- [ ] Set up evaluation framework (or use RAGAS library)
- [ ] Document cost model in `.env`

---

## Questions to Clarify

1. **KB Composition**: % PDF scans, tables, financial docs, images/diagrams?
   → Determines OCR mode, table strategy, image embeddings priority

2. **SLA**: Query latency budget? Accuracy target? Concurrent users?
   → Determines reranking necessity, caching strategy

3. **Evaluation Data**: Do you have ground-truth Q&A pairs for your domain?
   → Critical for RAGAS evaluation

4. **Deployment**: Self-hosted GPU or API-based?
   → Affects reranking cost, OCR cost

5. **Update Frequency**: How often does KB change?
   → Determines incremental vs full reindexing strategy

---

## Conclusion

Your proposal is **architecturally correct** but needs **production hardening**. The gaps identified are not conceptual flaws—they're operational blind spots common in early-stage RAG systems.

**Good news**: All gaps are addressable within 6 weeks with clear implementation patterns.

**Timeline**: Week 1-2 foundation, Week 3-4 retrieval, Week 5-6 optimization → **production-ready by end of Q1 2026**.

**ROI**: ~$2K upfront, <$200/month running. Outperforms manual document search by >50% (based on industry benchmarks).

---

**Next meeting**: Discuss phasing, team allocation, and evaluation strategy.
