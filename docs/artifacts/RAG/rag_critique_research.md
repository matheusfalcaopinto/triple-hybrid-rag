# RAG Architecture Critique: DeepSeek OCR + Multimodal Retrieval for voice-agent-v5

## Research Summary
Conducted web research (Jan 2026) on:
- Vector RAG architecture best practices (2024-2025)
- DeepSeek OCR performance & production realities
- Multimodal RAG implementations (CLIP, ViT, image embeddings)
- Chunking strategies (semantic vs fixed-size, empirical results)
- Reranking & cross-encoders (cost/latency trade-offs)
- Table preservation & heterogeneous data handling
- RAG evaluation metrics & production deployment patterns

Key sources: ArXiv, IEEE, ACM, NVIDIA, OpenAI Cookbook, academic papers (2024-2025)

---

## STRENGTHS OF CURRENT PROPOSAL

### 1. **Correct High-Level Architecture**
✅ Modular design (loader, parser, chunker, embedder, retrieval API)  
✅ Separation of concerns (OCR → chunking → embedding → storage)  
✅ Provenance tracking (source_document, page, chunk_index) — critical for citations  
✅ Hybrid storage model (text-first baseline + optional image embeddings)  
✅ Sensible default: text-embedding-3-small (1536d) matching existing schema  

### 2. **Realistic OCR Strategy Selection**
✅ DeepSeek OCR is well-chosen for your use case:
   - **Efficiency**: 100-256 tokens/page (vs GOT-OCR2.0 at 256, MinerU2.0 at 6000+)
   - **Layout preservation**: Markdown output maintains structure
   - **Multimodal heritage**: Retains captions/object grounding even after compression
   - **Language support**: 100+ languages (important for Brazilian Portuguese voice agent)

✅ Correct understanding of OCR modes (tiny/small/base/large/gundam trade accuracy vs cost)

### 3. **Pragmatic Phased Approach**
✅ "Minimal implementation path" (section 8) is sound:
   - Start with text chunks + vector embeddings
   - Defer image embeddings unless needed
   - Avoid premature complexity

---

## CRITICAL ISSUES & GAPS

### **ISSUE 1: DeepSeek OCR Production Reality Gap (HIGH RISK)**

**Current proposal assumption:**
> "OCR: DeepSeek Base mode (1024×1024, ~256 tokens) for balanced accuracy/cost."

**Research finding (LabelYourData, 2025):**
- **Benchmark vs Production**: DeepSeek hits 97% on OmniDocBench but **75-80% on financial documents** (production corpus)
- **Token compression danger**: At 20x compression, accuracy drops to **60%** (unusable for production)
- **Training gap**: 30M pages across 100 languages = **300K per language** — misses edge cases
  - Dense financial reports (tables + footnotes + small fonts)
  - Multi-column newspapers (4,000-5,000 tokens needed)
  - Watermarked documents
  - Domain-specific formats

**Specific for voice-agent-v5:**
If your KB includes financial reports, legal documents, or scanned contracts:
- **80% OCR accuracy** → introduces hallucinations in retrieved context → LLM generates wrong answers
- **Table misalignment**: 30% of production failures in financial documents
- **Small fonts**: Requires 400 DPI scanning (preprocessing cost)

**Recommendation:**
- ⚠️ Add **OCR validation layer**: Post-OCR quality checks for tables, confidence scoring
- ⚠️ Implement **fallback strategy**: For high-DPI PDFs with complex tables, use **MinerU2.0** (better structure) or hybrid (DeepSeek for text, table-aware parser for structures)
- ⚠️ Set **accuracy thresholds**: Flag chunks with low OCR confidence; include confidence scores in metadata
- ⚠️ Preprocess: **Flatten PDFs** (remove annotations), upscale to 300+ DPI, validate token density stays <1200

---

### **ISSUE 2: Chunking Strategy — Contradictory Evidence (MEDIUM RISK)**

**Current proposal:**
> "Default 1000 chars, 200 overlap for markdown; adjust for tables"

**Research reveals conflicting empirical data:**

**Study 1 (ArXiv 2024, Vectara/Vespa):**
- **Semantic chunking FAILS in production**: Minimal gains, high computational cost
- **Winner: Fixed-size chunking** (what you're defaulting to — good!)
- Reason: Semantic chunking assumes clean topic boundaries; real docs have mixed content

**Study 2 (ArXiv 2025, Breaking It Down):**
- **Semantic chunking WINS**: 24x MRR improvement on PubMed corpus
- Domain-specific semantic chunking (PSC/MFC) generalizes well

**Study 3 (ArXiv 2025, Domain-Aware Segmentation):**
- **Domain matters**: Financial docs need larger chunks (aggregation queries); technical specs need smaller
- Sweet spot: 5-20 tokens for precision (IoU=0.071), 200-500 tokens for recall

**The reality:** 
✅ Your **fixed-size approach is defensible** (aligns with Vectara research)  
❌ **BUT you're not measuring**: No evaluation metrics (context precision, answer relevancy, F1-score)  
❌ **1000 chars is arbitrary**: For voice-agent KB, validate empirically

**Recommendation:**
- Add **evaluation framework**: Use CoFE-RAG or RAGAS metrics (context precision, answer relevancy, faithfulness)
- **Experiment**: Run A/B on your actual KB: fixed-size (1000 chars) vs 500-token semantic vs 200-token high-precision
- **For tables**: Don't use fixed-size — implement table-aware chunking (next section)
- Document your choice with evidence

---

### **ISSUE 3: Table Handling — Current Proposal Inadequate (HIGH RISK)**

**Current proposal:**
> "Keep tables whole when possible (especially CSV/XLSX → Markdown tables)"

**Reality (TableRAG, 2025):**
- **Naive table linearization loses 30-50% of queryable structure**
- Markdown tables in RAG:
  - ❌ Lose column relationships for aggregation queries
  - ❌ Fail on multi-hop reasoning ("sum revenue by region")
  - ❌ Row-column semantics compressed to flat text
  
**Example failure:**
```
Table: Regional Revenue Q4 2025
| Region | Q1 | Q2 | Q3 | Q4 | Total |
| NA     | 50 | 55 | 60 | 65 | 230   |
| EMEA   | 40 | 42 | 45 | 48 | 175   |
| APAC   | 30 | 35 | 40 | 45 | 150   |

Query: "What's the highest growing region?"
Current approach: Embeds as flat markdown, loses column structure
Result: LLM must parse text → hallucination risk

Better approach: Preserve table + generate context description
```

**Recommendation:**
Implement **Table-Aware Ingestion** (from KX, 2025):
```python
# For each table:
1. Extract cleanly (pandas, unstructured)
2. Generate context description:
   "Sales table tracking regional revenue Q4 2025 across NA/EMEA/APAC"
3. Normalize to markdown:
   "| Region | Q1 | Q2 | Q3 | Q4 | Total |"
4. Create chunk:
   context_description + normalized_markdown
5. Embed the combined chunk
```

**Schema addition:**
```sql
ALTER TABLE knowledge_base ADD COLUMN (
  is_table BOOLEAN DEFAULT FALSE,
  table_context TEXT,  -- "Regional revenue tracker, Q4 2025, 4 regions"
  table_type VARCHAR(50)  -- 'sales', 'financial', 'technical'
);
```

---

### **ISSUE 4: Missing Reranking (MEDIUM COST)**

**Current proposal:**
> "Reranking: optional boolean" (deferred)

**Research (2024-2025, multiple sources):**
- **Cross-encoder reranking improves accuracy 20-35%** with modest latency cost
- **ROI**: Top-100 retrieve (fast) → rerank to top-10 (accurate) → +120ms latency, +33% accuracy
- **Production sweet spot**: Retrieve 50-100, rerank to 5-10
- **Cost**: Negligible if self-hosted; Google Colab: ~40-50ms for 100 pairs on T4 GPU

**For voice-agent-v5:**
Voice users need **high precision** (user won't re-query) → reranking is **high-ROI**

**Recommendation:**
- **Not optional** — add to "minimal path" (section 8)
- Use **cross-encoder-mmarco-MiniLM-L6-v2** (lightweight, fast)
- Config:
  ```python
  TOP_K_RETRIEVE = 50  # Fast retrieval
  TOP_K_RERANK = 10    # Accurate final set
  RERANKING_ENABLED = True  # Default for v1
  RERANKING_LATENCY_BUDGET = 200  # ms
  ```

---

### **ISSUE 5: No Hybrid Search Strategy (MEDIUM RISK)**

**Current proposal:**
> "Hybrid retrieval: BM25/FTS — optional boolean"

**Evidence (2024-2025 consensus):**
- **Hybrid search (BM25 + vector) consistently beats both alone** by 15-25%
- Text queries: "Model XR-450" (keyword match) works better with BM25
- Concept queries: "contract termination procedures" works better with vectors
- **Production RAG uses hybrid by default** (Vespa, Weaviate, pgvector 0.7+)

**Specific to knowledge base:**
- Product documentation: Needs exact model/feature matching (BM25)
- Process/policy docs: Needs semantic understanding (vectors)
- Voice queries: Often hybrid ("Show me the X procedure document")

**Recommendation:**
- **Enable hybrid by default**, not optional
- Use **Reciprocal Rank Fusion (RRF)** to combine scores:
  ```
  score = (1/(k + rank_bm25)) + (1/(k + rank_vector))
  where k=60 (typical)
  ```
- Add to Supabase config:
  ```sql
  CREATE INDEX idx_kb_fts ON knowledge_base USING GIN(to_tsvector('english', content));
  CREATE INDEX idx_kb_vector ON knowledge_base USING ivfflat(vector_embedding);
  ```

---

### **ISSUE 6: Evaluation & Observability Missing (HIGH RISK)**

**Current proposal:**
- Lists checklist (section 12) but no **actual metrics to measure**
- No comparison baseline

**What production RAG needs (RAGAS, PaSSER, CoFE-RAG frameworks):**
```python
Retrieval metrics:
- Context Precision (% of retrieved chunks relevant to query)
- Context Recall (% of ground-truth evidence retrieved)
- MRR (Mean Reciprocal Rank — top-1 relevance)
- NDCG@10 (ranking quality)

Generation metrics:
- Faithfulness (answer contradicts retrieved context? 0-1)
- Answer Relevancy (query-answer semantic similarity)
- F1-score (exact match on key entities)

Operational:
- Retrieval latency (p50, p95, p99)
- Cache hit rate
- OCR error rate per source_document
```

**Recommendation:**
- Add **evaluation section** to ingest_kb.py:
  ```python
  def evaluate_rag(test_queries, test_answers):
      for query in test_queries:
          retrieved = search_knowledge_base(query, k=10)
          context_precision = evaluate_precision(query, retrieved)
          # ... capture metrics
      return metrics_report
  ```
- Set **baseline**: "Baseline RAG achieves 72% context precision on 100-query test set"
- Track: Changes in accuracy as you evolve chunking, embedding models, reranking

---

### **ISSUE 7: Image Embeddings Strategy Under-Specified (LOW-MEDIUM RISK)**

**Current proposal:**
> "Image embeddings: off by default; enable for chart-heavy sources only"

**Research findings:**
- **Common problem**: Text embeddings are biased toward text-to-text similarity
  - CLIP is trained on millions of image-caption pairs, but bias exists
  - Text-only caption extraction loses 20-40% of visual semantics (charts, diagrams)
  
- **Three approaches to multimodal RAG:**
  1. **Embed all modalities in same space** (CLIP) — simple, but biased
  2. **Ground to text** (caption images, embed captions) — recommended, best accuracy
  3. **Separate stores** (text vectors + image vectors, merge at query) — complex

**For voice agent (charts/diagrams in KB):**
Approach #2 is best: **Caption images with LLM, embed captions alongside text**

**Recommendation:**
```python
if has_images:
    # For each image/diagram/chart:
    image_caption = llm.caption(image)  # "Quarterly revenue breakdown: NA 50%, EMEA 30%, APAC 20%"
    alt_text_embedding = embed(image_caption)
    
    # Store with image modality
    insert_into_kb(
        content=image_caption,
        modality='image_caption',
        vector_embedding=alt_text_embedding,
        alt_text=image_caption,
        image_ref=f"source/{page}/image_{idx}.png"
    )
```

Defer direct image embedding (CLIP) unless user queries with images.

---

### **ISSUE 8: Cost & Latency Not Quantified (MEDIUM RISK)**

**Current proposal:**
- Lists "cost/latency" in risks but no numbers

**Actual production costs (Jan 2026):**
- DeepSeek OCR API: ~$0.01-0.05 per page (Base/Large)
- text-embedding-3-small: $0.02 per 1M tokens (1000 chars ≈ 250 tokens)
- Reranking (cross-encoder, self-hosted): ~$0.001-0.002 per query + GPU (amortized)
- Vector DB (Supabase pgvector): $0.15-1.00 per 1GB/month

**Example KB (10,000 documents, avg 5 pages each):**
- OCR: 50,000 pages × $0.03 = **$1,500** (one-time)
- Embedding: 50,000 pages × 1000 chars × $0.02/1M = **$1.00** (one-time)
- Vector storage: ~50GB = **$7.50-50/month**
- Query cost (assuming 100 daily queries, 50 chunks retrieved):
  - Embedding query: 100 × $0.00002 = **$0.002/day**
  - Reranking: 100 × 50 pairs × cross-encoder cost ≈ **$0.05-0.10/day** (or free if GPU)

**Recommendation:**
- Document cost model in `.env` config
- Add **cost tracking** to ingest script:
  ```python
  OCR_COST_PER_PAGE = 0.03
  EMBED_COST_PER_1M_TOKENS = 0.02
  total_cost = num_pages * OCR_COST_PER_PAGE + tokens * EMBED_COST_PER_1M_TOKENS
  print(f"Estimated ingestion cost: ${total_cost:.2f}")
  ```

---

### **ISSUE 9: Deduplication Not Addressed (MEDIUM RISK)**

**Current proposal:**
> "Deduplication: hash normalized text per source_document to avoid near-duplicates" (optional enhancement, section 9)

**Why critical:**
- PDFs often have boilerplate (headers, footers, legal disclaimers)
- Chunking creates near-duplicates ("Section 3 header" repeated across pages)
- Embedding identical/near-identical chunks wastes storage (and embedding cost)
- **Vector search returns duplicates** → low diversity in retrieved context

**Recommendation:**
- Move to **core implementation** (not optional)
- Add to schema:
  ```sql
  ALTER TABLE knowledge_base ADD COLUMN content_hash VARCHAR(64);
  CREATE UNIQUE INDEX idx_content_hash ON knowledge_base(org_id, content_hash);
  ```
- In ingestion:
  ```python
  from hashlib import sha256
  content_hash = sha256(normalized_content.encode()).hexdigest()
  insert_or_skip_if_hash_exists(content_hash)
  ```

---

### **ISSUE 10: Batch Processing & Async Ingestion (LOW-MEDIUM RISK)**

**Current proposal:**
- No mention of batch embedding or async processing

**Production reality:**
- Embedding 50,000 chunks sequentially = hours
- Voice agent needs **responsive ingestion** (user uploads KB, expects fast indexing)

**Recommendation:**
```python
# In ingest_kb.py:
async def batch_embed_chunks(chunks, batch_size=256):
    embeddings = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        batch_embeddings = await async_embed_client.embed(batch)
        embeddings.extend(batch_embeddings)
        print(f"Embedded {i+batch_size}/{len(chunks)}")
    return embeddings

# Use ray for parallelism:
@ray.remote
def embed_chunk_batch(chunks):
    return embed_client.embed(chunks)

# Ingestion script:
futures = [embed_chunk_batch.remote(chunk_batch) for chunk_batch in chunked_chunks]
embeddings = ray.get(futures)
```

---

## SUMMARY TABLE: Proposal Gaps

| Issue | Severity | Current Status | Recommendation |
|-------|----------|-----------------|-----------------|
| DeepSeek OCR accuracy gap | HIGH | Base mode assumed 97% | Add validation layer, fallback strategy, confidence scoring |
| Chunking evaluation | MEDIUM | No metrics | Implement RAGAS/CoFE-RAG evaluation framework |
| Table handling | HIGH | Naive markdown | Implement table-aware chunking + context descriptions |
| Reranking | MEDIUM | Optional | Make default in v1, use cross-encoder-mmarco-MiniLM-L6 |
| Hybrid search | MEDIUM | Optional | Enable by default, use RRF fusion |
| Evaluation & observability | HIGH | Checklist only | Add quantified metrics (precision, recall, F1, latency) |
| Image embeddings | LOW-MEDIUM | Underspecified | Recommend caption-based approach (multimodal #2) |
| Cost/latency | MEDIUM | No numbers | Document cost model, add tracking to scripts |
| Deduplication | MEDIUM | Optional enhancement | Move to core (prevent hash collisions, save cost) |
| Batch/async ingestion | LOW-MEDIUM | Sequential | Implement parallel embedding with ray or asyncio |

---

## REVISED MINIMAL IMPLEMENTATION PATH

**Phase 1 (Weeks 1-2): Foundation**
1. ✅ Add schema columns: `modality`, `page`, `chunk_index`, `source_document`, `content_hash`, `vector_image` (optional)
2. ✅ Implement fixed-size chunking (1000 chars, 200 overlap)
3. ✅ **NEW**: Add table-aware chunking (detect tables, preserve structure, generate context)
4. ✅ Implement text embedding (text-embedding-3-small)
5. ✅ **NEW**: Add OCR validation layer (flag low-confidence chunks)
6. ✅ **NEW**: Implement deduplication (content_hash)

**Phase 2 (Weeks 3-4): Retrieval & Evaluation**
1. ✅ Update `search_knowledge_base` with vector search + filters
2. ✅ **NEW**: Add BM25/FTS hybrid search (Reciprocal Rank Fusion)
3. ✅ **NEW**: Implement cross-encoder reranking (top-100 → top-10)
4. ✅ **NEW**: Add evaluation framework (context precision, faithfulness, latency tracking)
5. ✅ Return provenance (source_document, page, modality)

**Phase 3 (Weeks 5-6): Optimization & Monitoring**
1. ✅ **NEW**: Batch embedding with ray/asyncio
2. ✅ **NEW**: Add observability (OCR accuracy, retrieval latency, cache hit rates)
3. ✅ A/B test chunking strategies on actual KB
4. ✅ Implement image captioning (if diagrams in KB)
5. ✅ Cost tracking dashboard

---

## QUESTIONS FOR YOUR TEAM

1. **What's your KB composition?** (% text docs, PDF scans, financial reports, technical specs, diagrams)
   → Determines OCR mode, table strategy, image embeddings priority

2. **SLA requirements?** (Query latency budget, accuracy target)
   → Determines reranking necessity, batch size

3. **Volume?** (Documents, daily queries, concurrent users)
   → Determines infrastructure (self-hosted vs API), caching strategy

4. **Data freshness?** (KB updates frequency)
   → Determines incremental vs full reindexing

5. **Availability of evaluation data?** (Ground truth Q&A pairs for your domain)
   → Critical for RAGAS/CoFE-RAG evaluation framework

---

## KEY CITATIONS

1. **DeepSeek OCR Production Accuracy**: LabelYourData (2025) — Production accuracy 75-80% on financial docs vs 97% benchmark
2. **Semantic vs Fixed-Size Chunking**: ArXiv 2410.13070 (Vectara/Vespa, 2024) — Fixed-size often outperforms semantic
3. **Reranking ROI**: ArXiv 2405.12363, AILog (2025) — +20-35% accuracy, +120-200ms latency
4. **Table RAG**: ArXiv 2506.10380 (2025) — TableRAG framework for preserving structure
5. **Hybrid Search**: Multiple sources (NVIDIA, Morphik, 2025) — BM25 + vector combo is production standard
6. **Evaluation Frameworks**: CoFE-RAG, RAGBench, PaSSER (2024-2025)
7. **Multimodal RAG**: NVIDIA, OpenAI Cookbook (2024-2025) — Caption-based approach recommended
8. **Chunking Best Practices**: Unstructured.io, KX.com (2025)