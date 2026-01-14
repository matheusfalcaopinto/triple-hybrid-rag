# RAG System Complete Developer Guide

> **Version**: 2.0 (halfvec)  
> **Last Updated**: January 13, 2026  
> **Status**: Production Ready ✅
> **Storage**: halfvec(4000) with HNSW indexes

This guide provides comprehensive documentation for the Vector RAG (Retrieval-Augmented Generation) system implemented in the voice-agent-v5 project. It covers architecture, configuration, usage, database schema, and all operational aspects.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Database Schema](#3-database-schema)
4. [Configuration Reference](#4-configuration-reference)
5. [Setup & Installation](#5-setup--installation)
6. [Document Ingestion](#6-document-ingestion)
7. [Retrieval & Search](#7-retrieval--search)
8. [Agent Tool Integration](#8-agent-tool-integration)
9. [Backend API Service](#9-backend-api-service)
10. [Testing](#10-testing)
11. [Feature Toggle Reference](#11-feature-toggle-reference)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Overview

### What is this RAG System?

The RAG system enables the voice agent to retrieve relevant information from a knowledge base during conversations. It processes documents (PDFs, DOCX, XLSX, etc.), creates searchable chunks with embeddings, and provides hybrid search capabilities.

### Key Features

| Feature | Description | Status |
|---------|-------------|--------|
| **Multi-format Ingestion** | PDF, DOCX, TXT, CSV, XLSX, Images | ✅ Working |
| **Unified Embeddings** | Qwen3-VL-Embedding (4096d → 4000d truncated, halfvec) | ✅ Working |
| **OCR** | Qwen3-VL via LM Studio | ✅ Working |
| **Hybrid Search** | BM25 + Vector + RRF Fusion | ✅ Working |
| **Multimodal Reranking** | Qwen3-VL-Reranker with `/no_think` flag | ✅ Working |
| **HNSW Indexing** | Fast approximate nearest neighbor via halfvec | ✅ Working |
| **Deduplication** | Content hash-based | ✅ Working |
| **Table Preservation** | Markdown table extraction | ✅ Working |
| **Provenance Tracking** | Source doc, page, chunk index | ✅ Working |

> **Note**: The unified `text-embedding-qwen3-vl-embedding-8b` model outputs 4096 dimensions, which are truncated to 4000 and stored as `halfvec` (16-bit floats) for pgvector HNSW index compatibility. This provides 50% storage savings with negligible quality loss (~2.3% dimensions, <1% retrieval impact).

### Pipeline Overview

```
┌─────────────┐     ┌──────────┐     ┌──────────┐     ┌───────────┐     ┌──────────┐
│  Documents  │────▶│  Loader  │────▶│  Chunker │────▶│  Embedder │────▶│ Supabase │
│ PDF/DOCX/...│     │ + OCR    │     │ + Tables │     │ Qwen3-VL  │     │ pgvector │
└─────────────┘     └──────────┘     └──────────┘     │ 4096→4000 │     │ halfvec  │
                                                      └───────────┘     └──────────┘
                    
┌─────────────┐     ┌──────────┐     ┌──────────┐     ┌───────────┐     ┌──────────┐
│   Query     │────▶│  Embed   │────▶│  Hybrid  │────▶│  Reranker │────▶│ Results  │
│  (Agent)    │     │ 4096→4000│     │  Search  │     │ Qwen3-VL  │     │ + Scores │
└─────────────┘     └──────────┘     │ HNSW+BM25│     │ /no_think │     └──────────┘
                                     └──────────┘     └───────────┘
```

---

## 2. Architecture

### Module Structure

```
src/voice_agent/
├── ingestion/                 # Document ingestion pipeline
│   ├── __init__.py           # Module exports
│   ├── loader.py             # File loading (PDF, DOCX, XLSX, etc.)
│   ├── chunker.py            # Text chunking with table preservation
│   ├── embedder.py           # Text (OpenAI) + Image (SigLIP) embeddings
│   ├── ocr.py                # OCR via Qwen3-VL
│   └── kb_ingest.py          # Main orchestrator
│
├── retrieval/                 # Search and retrieval
│   ├── __init__.py           # Module exports
│   ├── hybrid_search.py      # BM25 + Vector + RRF fusion
│   └── reranker.py           # Qwen3-VL multimodal reranking
│
└── tools/
    └── crm_knowledge.py      # Agent tool: search_knowledge_base
```

### Component Details

#### 2.1 Document Loader (`loader.py`)

Handles file type detection and content extraction:

| File Type | Library | Features |
|-----------|---------|----------|
| PDF | PyMuPDF (fitz) | Text + images + tables, page-by-page |
| DOCX | python-docx | Paragraphs + tables |
| XLSX | openpyxl | Sheet-by-sheet as Markdown tables |
| CSV | stdlib | Full content as Markdown table |
| TXT/MD | stdlib | Raw text content |
| Images | PIL + OCR | OCR via Qwen3-VL |

**Key Classes:**
- `FileType` - Enum for supported file types
- `PageContent` - Content from a single page/section
- `LoadedDocument` - Complete parsed document
- `DocumentLoader` - Main loader class

#### 2.2 Chunker (`chunker.py`)

Splits documents into overlapping chunks:

- **Default chunk size**: 1000 characters
- **Default overlap**: 200 characters
- **Table preservation**: Tables are kept as complete chunks
- **Heading context**: Preserves section headings for context

**Key Classes:**
- `ChunkType` - Enum: TEXT, TABLE, IMAGE, IMAGE_CAPTION
- `Chunk` - Single chunk with metadata and content hash
- `Chunker` - Main chunking logic

#### 2.3 Embedder (`embedder.py`)

Generates embeddings for chunks:

| Mode | Model | Output Dims | Storage Dims | Storage Type |
|------|-------|-------------|--------------|--------------|
| Local (default) | `text-embedding-qwen3-vl-embedding-8b` | 4096 | 4000 | halfvec |
| Fallback | `text-embedding-3-small` | 1536 | 1536 | vector |

**Features:**
- Automatic truncation (4096 → 4000) for pgvector HNSW compatibility
- Lazy-loading of models
- Batch processing for efficiency
- L2 normalization of embeddings

#### 2.4 OCR Processor (`ocr.py`)

Extracts text from images and scanned PDFs:

- **Model**: Qwen3-VL (8B) via LM Studio
- **Endpoint**: `http://127.0.0.1:1234/v1/chat/completions`
- **Confidence tracking**: Estimates OCR quality

#### 2.5 Hybrid Search (`hybrid_search.py`)

Combines multiple search strategies:

1. **BM25 (Full-Text Search)**: PostgreSQL `tsvector` with Portuguese/English
2. **Vector Search**: pgvector cosine similarity
3. **RRF Fusion**: Reciprocal Rank Fusion to combine results

**Algorithm:**
```
RRF_score = Σ (1 / (k + rank_i))

where k = 60 (constant), rank_i = position in each result list
```

#### 2.6 Reranker (`reranker.py`)

Precision reranking using Qwen3-VL-Reranker (multimodal):

- **Primary Model**: `qwen3-vl-reranker-8b` (via LM Studio local API)
- **Critical Flag**: `/no_think` in system prompt (disables thinking mode for consistent yes/no)
- **Multimodal Support**: Text + Images + Mixed content
- **Method**: Yes/No classification with simplified prompt format
- **Fallback Chain**: CrossEncoder → LightweightReranker
- **Score range**: 0.0 to 1.0 (probability of relevance)

**Important**: The `/no_think` flag is essential for Qwen3-VL-Reranker to provide consistent responses.

---

## 3. Database Schema

### 3.1 Entity Relationship Diagram

```
┌────────────────────┐
│   organizations    │
│────────────────────│
│ id (PK)            │◄──────────┐
│ name               │           │
│ ...                │           │
└────────────────────┘           │
                                 │
┌────────────────────┐           │
│   knowledge_base   │           │
│────────────────────│           │
│ id (PK)            │◄────┐     │
│ org_id (FK)        │─────┼─────┘
│ category           │     │
│ title              │     │
│ content            │     │
│ doc_checksum       │     │
│ is_chunked         │     │
│ ...                │     │
└────────────────────┘     │
                           │
┌──────────────────────────┴─────────────────────────────────────────┐
│                    knowledge_base_chunks                            │
│─────────────────────────────────────────────────────────────────────│
│ id                 UUID PRIMARY KEY                                 │
│ org_id             UUID NOT NULL FK → organizations(id)             │
│ knowledge_base_id  UUID FK → knowledge_base(id) (optional)          │
│─────────────────────────────────────────────────────────────────────│
│ category           TEXT                                             │
│ title              TEXT                                             │
│ source_document    TEXT                                             │
│ modality           TEXT CHECK IN ('text','image','image_caption',   │
│                          'table')                                   │
│ page               INTEGER                                          │
│ chunk_index        INTEGER                                          │
│─────────────────────────────────────────────────────────────────────│
│ content            TEXT NOT NULL                                    │
│ content_hash       TEXT NOT NULL                                    │
│─────────────────────────────────────────────────────────────────────│
│ ocr_confidence     REAL                                             │
│ is_table           BOOLEAN DEFAULT FALSE                            │
│ table_context      TEXT                                             │
│ alt_text           TEXT                                             │
│ image_path         TEXT                                             │
│─────────────────────────────────────────────────────────────────────│
│ vector_embedding   halfvec(4000)   -- Qwen3-VL text embeddings (truncated) │
│ vector_image       halfvec(4000)   -- Qwen3-VL image embeddings (truncated)│
│─────────────────────────────────────────────────────────────────────│
│ created_at         TIMESTAMPTZ                                      │
│ updated_at         TIMESTAMPTZ                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Indexes

| Index Name | Type | Columns | Purpose |
|------------|------|---------|---------|
| `idx_kb_chunks_dedup` | UNIQUE | `(org_id, content_hash)` | Deduplication |
| `idx_kb_chunks_org_source` | B-tree | `(org_id, source_document, modality)` | Filtering |
| `idx_kb_chunks_org_category` | B-tree | `(org_id, category)` | Category filter |
| `idx_kb_chunks_fts_pt` | GIN | `to_tsvector('portuguese', content)` | Portuguese FTS |
| `idx_kb_chunks_fts_en` | GIN | `to_tsvector('english', content)` | English FTS |
| `idx_kb_chunks_vector_embedding` | HNSW | `vector_embedding halfvec_cosine_ops` | Text ANN search |
| `idx_kb_chunks_vector_image` | HNSW | `vector_image halfvec_cosine_ops` | Image ANN search |

> **Note**: HNSW indexes use `halfvec_cosine_ops` for 16-bit float vectors. This supports up to 4000 dimensions (vs 2000 for regular `vector` type) while providing 50% storage savings.

### 3.3 SQL Functions

| Function | Parameters | Returns | Purpose |
|----------|------------|---------|---------|
| `kb_chunks_fts_pt` | `org_id, query, limit` | Table | Portuguese full-text search |
| `kb_chunks_vector_search` | `org_id, embedding halfvec(4000), limit, category, source` | Table | Vector similarity search |
| `kb_chunks_image_search` | `org_id, image_embedding halfvec(4000), limit` | Table | Image similarity search |
| `kb_chunks_hybrid_rrf_search` | `org_id, embedding halfvec(4000), query, limit, category, rrf_k, fts_config` | Table | Hybrid RRF search |

> **Note**: All vector functions accept `halfvec(4000)` parameters to match the column types.

### 3.4 Row-Level Security

```sql
-- Only org members can access their own chunks
CREATE POLICY "KB Chunks Org Isolation" ON knowledge_base_chunks
    FOR ALL USING (org_id = get_my_org_id() OR is_super_admin());
```

### 3.5 Migration Files

| File | Purpose |
|------|---------|
| `database/migrations/20260113_add_kb_chunks.sql` | Original schema creation |
| `database/migrations/20260113_halfvec_4000.sql` | Convert to halfvec(4000) + HNSW indexes |

**Apply migrations:**
```bash
# Using Supabase CLI
supabase db push

# Or directly via psql
docker exec -i supabase-db psql -U postgres -d postgres < database/migrations/20260113_add_kb_chunks.sql
docker exec -i supabase-db psql -U postgres -d postgres < database/migrations/20260113_halfvec_4000.sql
```

---

## 4. Configuration Reference

All settings are in `src/voice_agent/config.py` and can be overridden via environment variables.

### 4.1 Unified Embeddings

The RAG system uses a **single unified model** (`text-embedding-qwen3-vl-embedding-8b`) for both text and image embeddings. This provides:
- 4096-dimensional output, truncated to 4000 for pgvector HNSW compatibility
- Stored as `halfvec` (16-bit floats) for 50% storage savings
- Multilingual text support (30+ languages)
- Native multimodal understanding
- Hardware efficiency (one model instead of two)

| Setting | Default | Env Variable | Description |
|---------|---------|--------------|-------------|
| `rag_embed_model` | `text-embedding-qwen3-vl-embedding-8b` | `RAG_EMBED_MODEL` | Unified embedding model |
| `rag_vector_dim` | `4000` | `RAG_VECTOR_DIM` | Storage dimensions (truncated) |
| `rag_model_output_dim` | `4096` | `RAG_MODEL_OUTPUT_DIM` | Model output dimensions |
| `rag_embed_batch_size` | `20` | `RAG_EMBED_BATCH_SIZE` | Batch size for embedding API calls |
| `rag_embed_api_base` | `http://127.0.0.1:1234/v1` | `RAG_EMBED_API_BASE` | Local embedding API base URL |
| `rag_embed_use_local` | `True` | `RAG_EMBED_USE_LOCAL` | Use local API (False = OpenAI) |
| `rag_enable_image_embeddings` | `False` | `RAG_ENABLE_IMAGE_EMBEDDINGS` | Enable image embedding generation |

### 4.2 Chunking

| Setting | Default | Env Variable | Description |
|---------|---------|--------------|-------------|
| `rag_chunk_size` | `1000` | `RAG_CHUNK_SIZE` | Target chunk size in characters |
| `rag_chunk_overlap` | `200` | `RAG_CHUNK_OVERLAP` | Overlap between consecutive chunks |
| `rag_preserve_tables` | `True` | `RAG_PRESERVE_TABLES` | Keep tables as complete chunks |

### 4.3 Search & Retrieval

| Setting | Default | Env Variable | Description |
|---------|---------|--------------|-------------|
| `rag_use_hybrid_bm25` | `True` | `RAG_USE_HYBRID_BM25` | Enable hybrid BM25 + vector search |
| `rag_top_k_retrieve` | `50` | `RAG_TOP_K_RETRIEVE` | Initial retrieval pool size |
| `rag_top_k_image` | `3` | `RAG_TOP_K_IMAGE` | Image results to include |
| `rag_top_k_rerank` | `5` | `RAG_TOP_K_RERANK` | Final results after reranking |
| `rag_reranking_enabled` | `True` | `RAG_RERANKING_ENABLED` | Enable Qwen3-VL reranking |
| `rag_rerank_model` | `qwen3-vl-reranker-8b` | `RAG_RERANK_MODEL` | Reranking model |
| `rag_rerank_api_base` | `http://127.0.0.1:1234/v1` | `RAG_RERANK_API_BASE` | Local API endpoint |
| `rag_rerank_use_local` | `True` | `RAG_RERANK_USE_LOCAL` | Use local Qwen3-VL reranker |
| `rag_fts_language` | `portuguese` | `RAG_FTS_LANGUAGE` | Full-text search language |

### 4.4 OCR

| Setting | Default | Env Variable | Description |
|---------|---------|--------------|-------------|
| `rag_ocr_mode` | `base` | `RAG_OCR_MODE` | OCR mode (base, enhanced) |
| `rag_ocr_confidence_threshold` | `0.6` | `RAG_OCR_CONFIDENCE_THRESHOLD` | Min confidence to accept |
| `rag_ocr_retry_limit` | `2` | `RAG_OCR_RETRY_LIMIT` | Max OCR retries |
| `rag_ocr_endpoint` | `http://127.0.0.1:1234/v1` | `RAG_OCR_ENDPOINT` | Qwen3-VL endpoint |
| `rag_ocr_model` | `qwen/qwen3-vl-8b` | `RAG_OCR_MODEL` | OCR model name |

### 4.5 Processing

| Setting | Default | Env Variable | Description |
|---------|---------|--------------|-------------|
| `rag_dedup_enabled` | `True` | `RAG_DEDUP_ENABLED` | Enable content deduplication |
| `rag_max_workers` | `4` | `RAG_MAX_WORKERS` | Parallel processing workers |

---

## 5. Setup & Installation

### 5.1 Prerequisites

1. **Python 3.12+**
2. **Supabase** (local or cloud) with pgvector extension
3. **OpenAI API key** for text embeddings
4. **LM Studio** (optional) with Qwen3-VL for OCR

### 5.2 Install Dependencies

```bash
# Install base dependencies
pip install -e .

# Install RAG optional dependencies
pip install -e ".[rag]"

# Or install individually
pip install PyMuPDF pypdf python-docx openpyxl
pip install sentence-transformers sentencepiece
pip install torch transformers Pillow numpy
```

### 5.3 Environment Variables

Create or update `.env`:

```bash
# Required
OPENAI_API_KEY=sk-your-key-here
SUPABASE_URL=http://localhost:8000
SERVICE_ROLE_KEY=your-supabase-service-role-key

# Local Qwen3-VL stack (default - recommended)
RAG_EMBED_USE_LOCAL=true
RAG_EMBED_API_BASE=http://127.0.0.1:1234/v1
RAG_EMBED_MODEL=text-embedding-qwen3-vl-embedding-8b
RAG_VECTOR_DIM=4000
RAG_MODEL_OUTPUT_DIM=4096
RAG_RERANK_USE_LOCAL=true
RAG_RERANK_MODEL=qwen3-vl-reranker-8b

# Optional settings
RAG_CHUNK_SIZE=1000
RAG_CHUNK_OVERLAP=200
RAG_USE_HYBRID_BM25=true
RAG_RERANKING_ENABLED=true
RAG_ENABLE_IMAGE_EMBEDDINGS=false
RAG_OCR_ENDPOINT=http://127.0.0.1:1234/v1
RAG_OCR_MODEL=qwen/qwen3-vl-8b

# OpenAI + SigLIP fallback (if local models unavailable)
# RAG_EMBED_USE_LOCAL=false
# RAG_EMBED_MODEL=text-embedding-3-small
# RAG_VECTOR_DIM=1536
# RAG_RERANK_USE_LOCAL=false
```

### 5.4 Database Setup

```bash
# Start local Supabase
cd database && docker-compose up -d

# Apply base migration
docker exec -i supabase-db psql -U postgres -d postgres < database/migrations/20260113_add_kb_chunks.sql

# Apply halfvec migration (for Qwen3-VL 4000d embeddings)
docker exec -i supabase-db psql -U postgres -d postgres < database/migrations/20260113_halfvec_4000.sql

# Verify columns
docker exec supabase-db psql -U postgres -d postgres -c "
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'knowledge_base_chunks' AND column_name LIKE 'vector%';"

# Expected output:
#    column_name    | data_type
# ------------------+-----------
#  vector_embedding | halfvec(4000)
#  vector_image     | halfvec(4000)
```

### 5.5 Start OCR Service (Optional)

1. Download and install [LM Studio](https://lmstudio.ai/)
2. Download model: `qwen/qwen3-vl-8b`
3. Start server on `http://127.0.0.1:1234`

---

## 6. Document Ingestion

### 6.1 CLI Tool

The primary way to ingest documents:

```bash
# Ingest a single file
python scripts/ingest_kb.py docs/product_manual.pdf --category product

# Ingest a directory
python scripts/ingest_kb.py docs/knowledge/ --category general --recursive

# With custom settings
python scripts/ingest_kb.py docs/manual.pdf \
    --org-id "00000000-0000-0000-0000-000000000001" \
    --category "technical" \
    --chunk-size 800 \
    --chunk-overlap 150 \
    --enable-images \
    --verbose
```

**CLI Options:**

| Option | Description |
|--------|-------------|
| `--org-id` | Organization ID (auto-detected if not provided) |
| `--category` | Category for all ingested documents |
| `--chunk-size` | Override chunk size |
| `--chunk-overlap` | Override chunk overlap |
| `--enable-images` | Enable SigLIP image embeddings |
| `--no-dedup` | Disable deduplication |
| `--recursive` | Recursively process directories |
| `--verbose` | Enable debug logging |
| `--dry-run` | Show what would be ingested without saving |

### 6.2 Programmatic Ingestion

```python
import asyncio
from pathlib import Path
from voice_agent.ingestion.kb_ingest import KnowledgeBaseIngestor

async def ingest_documents():
    ingestor = KnowledgeBaseIngestor(
        org_id="00000000-0000-0000-0000-000000000001",
        category="product",
        enable_image_embeddings=False,  # Set True for images
    )
    
    # Ingest single file
    result = await ingestor.ingest_file(Path("docs/manual.pdf"))
    print(f"Chunks created: {result.stats.chunks_created}")
    print(f"Chunks stored: {result.stats.chunks_stored}")
    
    # Ingest directory
    results = await ingestor.ingest_directory(Path("docs/knowledge/"))
    for result in results:
        print(f"{result.file_path}: {result.stats.chunks_stored} chunks")

asyncio.run(ingest_documents())
```

### 6.3 Ingestion Pipeline Flow

```
1. LOAD DOCUMENT
   ├── Detect file type (PDF, DOCX, etc.)
   ├── Extract text content page-by-page
   ├── Extract tables as Markdown
   ├── Identify images for OCR
   └── Calculate file hash for tracking

2. OCR (if needed)
   ├── Send images to Qwen3-VL
   ├── Receive extracted text
   └── Track confidence scores

3. CHUNK CONTENT
   ├── Split text into overlapping chunks
   ├── Preserve tables as complete chunks
   ├── Maintain heading context
   └── Calculate content hashes for dedup

4. GENERATE EMBEDDINGS
   ├── Batch text chunks → OpenAI API
   ├── (Optional) Image chunks → SigLIP
   └── Normalize vectors

5. STORE IN DATABASE
   ├── Upsert chunks (dedup by hash)
   ├── Store vectors in pgvector columns
   └── Update metadata and timestamps
```

### 6.4 Backfill Existing Data

Migrate from old `knowledge_base` table:

```bash
python scripts/backfill_kb_chunks.py \
    --org-id "00000000-0000-0000-0000-000000000001" \
    --batch-size 50 \
    --verbose
```

---

## 7. Retrieval & Search

### 7.1 Hybrid Search Overview

The search combines three strategies:

```
┌──────────────────────────────────────────────────────────────────┐
│                         HYBRID SEARCH                            │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Query: "How to configure voice settings?"                      │
│                                                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   BM25 (FTS)    │  │  Vector Search  │  │  Image Search   │  │
│  │ Portuguese/EN   │  │  Cosine Sim     │  │  (if enabled)   │  │
│  │                 │  │                 │  │                 │  │
│  │ Rank: A=1, B=3  │  │ Rank: B=1, A=2  │  │ Rank: C=1       │  │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘  │
│           │                    │                    │           │
│           └────────────┬───────┴────────────────────┘           │
│                        ▼                                        │
│               ┌─────────────────┐                               │
│               │   RRF Fusion    │                               │
│               │ Score = Σ 1/(k+r)                               │
│               └────────┬────────┘                               │
│                        ▼                                        │
│               ┌─────────────────┐                               │
│               │    Reranker     │                               │
│               │ Cross-Encoder   │                               │
│               └────────┬────────┘                               │
│                        ▼                                        │
│               Final Results (top_k)                             │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 7.2 Programmatic Search

```python
import asyncio
from voice_agent.retrieval.hybrid_search import HybridSearcher, SearchConfig
from voice_agent.retrieval.reranker import Reranker

async def search_knowledge():
    # Configure search
    config = SearchConfig(
        use_hybrid=True,
        use_vector=True,
        use_bm25=True,
        top_k_retrieve=50,
        top_k_final=10,
        fts_language="portuguese",
    )
    
    # Create searcher
    searcher = HybridSearcher(
        org_id="00000000-0000-0000-0000-000000000001",
        config=config,
    )
    
    # Search
    results = await searcher.search(
        query="configurar integração com calendário",
        top_k=10,
        category="technical",  # Optional filter
    )
    
    # Rerank (optional)
    reranker = Reranker()
    reranked = await reranker.rerank(
        query="configurar integração com calendário",
        results=results,
        top_k=5,
    )
    
    for r in reranked:
        print(f"[{r.rerank_score:.2f}] {r.content[:100]}...")

asyncio.run(search_knowledge())
```

### 7.3 Search Result Structure

```python
@dataclass
class SearchResult:
    chunk_id: str              # UUID of the chunk
    content: str               # Chunk text content
    modality: str              # text, image, table, image_caption
    source_document: str       # Original file name
    page: Optional[int]        # Page number (1-indexed)
    chunk_index: int           # Chunk position in document
    category: Optional[str]    # Category tag
    title: Optional[str]       # Document title
    similarity_score: float    # Vector similarity (0-1)
    bm25_score: Optional[float]# BM25 relevance score
    rrf_score: Optional[float] # RRF combined score
    rerank_score: Optional[float] # Cross-encoder score
    ocr_confidence: Optional[float] # OCR quality (0-1)
    is_table: bool             # True if table content
    table_context: Optional[str]   # Table description
    alt_text: Optional[str]    # Image alt text
```

### 7.4 RRF Fusion Algorithm

```python
def rrf_fusion(rankings: List[List[str]], k: int = 60) -> Dict[str, float]:
    """
    Reciprocal Rank Fusion combines multiple rankings.
    
    Formula: score(d) = Σ 1 / (k + rank(d, list_i))
    
    Example:
    - Doc A: BM25 rank=1, Vector rank=3 → score = 1/61 + 1/63 = 0.0323
    - Doc B: BM25 rank=2, Vector rank=1 → score = 1/62 + 1/61 = 0.0325
    
    Higher score = more relevant (appears high in multiple lists)
    """
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            if doc_id not in scores:
                scores[doc_id] = 0.0
            scores[doc_id] += 1.0 / (k + rank)
    return scores
```

---

## 8. Agent Tool Integration

### 8.1 The `search_knowledge_base` Tool

The voice agent uses this tool during conversations:

**Tool Definition:**
```python
def search_knowledge_base(
    query: str,
    category: Optional[str] = None,
    limit: int = 5,
    use_hybrid: bool = True,
) -> Dict[str, Any]:
    """
    Search the knowledge base using hybrid search (BM25 + vector).
    
    Args:
        query: Search query (what the customer is asking about)
        category: Filter by category (pricing, technical, faq, product, etc.)
        limit: Maximum results (default 5)
        use_hybrid: Use hybrid search (default True)
    
    Returns:
        List of relevant knowledge base entries with ranking and provenance
    """
```

### 8.2 Tool Response Format

```json
{
  "success": true,
  "query": "como agendar uma reunião",
  "category": null,
  "result_count": 3,
  "search_type": "hybrid",
  "results": [
    {
      "chunk_id": "a1b2c3d4-...",
      "category": "calendar",
      "title": "Guia de Agendamento",
      "content": "Para agendar uma reunião, diga 'agendar reunião' seguido da data e hora desejada...",
      "source_document": "calendar_guide.pdf",
      "page": 3,
      "chunk_index": 12,
      "modality": "text",
      "relevance_rank": 1,
      "similarity_score": 0.8234,
      "rerank_score": 7.4521,
      "ocr_confidence": null,
      "is_table": false,
      "table_context": null,
      "alt_text": null
    }
  ]
}
```

### 8.3 How the Agent Uses It

1. **User asks a question** about products, pricing, procedures, etc.
2. **Agent detects need** for knowledge base lookup
3. **Agent calls tool**: `search_knowledge_base(query="user question")`
4. **Results returned** with relevant chunks and provenance
5. **Agent synthesizes response** using the retrieved information

### 8.4 Testing the Tool Manually

```python
from voice_agent.tools.crm_knowledge import search_knowledge_base

# Simple search
result = search_knowledge_base(
    query="preços do plano premium",
    category="pricing",
    limit=3,
)

print(f"Found {result['result_count']} results")
for r in result["results"]:
    print(f"[{r['relevance_rank']}] {r['source_document']} p.{r['page']}")
    print(f"    Score: {r['similarity_score']:.4f} | Rerank: {r['rerank_score']:.4f}")
    print(f"    {r['content'][:100]}...")
```

---

## 9. Backend API Service

### 9.1 Endpoints

The backend control-plane provides a reranking API:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/rag/health` | GET | Check reranker status |
| `/api/v1/rag/warmup` | POST | Pre-load CrossEncoder model |
| `/api/v1/rag/rerank` | POST | Rerank documents |
| `/api/v1/rag/score` | POST | Score single query-document pair |

### 9.2 API Usage

**Health Check:**
```bash
curl http://localhost:8080/api/v1/rag/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2"
}
```

**Rerank Documents:**
```bash
curl -X POST http://localhost:8080/api/v1/rag/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning algorithms",
    "documents": [
      {"id": "1", "content": "Machine learning is a subset of AI..."},
      {"id": "2", "content": "Pizza is a popular Italian dish..."}
    ],
    "top_k": 5
  }'
```

Response:
```json
{
  "query": "machine learning algorithms",
  "documents": [
    {"id": "1", "content": "Machine learning...", "score": 9.5952, "rank": 1},
    {"id": "2", "content": "Pizza is...", "score": -11.0775, "rank": 2}
  ],
  "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
  "processing_time_ms": 25.4
}
```

### 9.3 Starting the Backend

```bash
cd backend
uvicorn src.control_plane.app:app --host 0.0.0.0 --port 8080
```

---

## 10. Testing

### 10.1 Unit Tests

```bash
# Run all RAG tests
pytest tests/test_rag_*.py -v

# Run specific test files
pytest tests/test_rag_ingestion.py -v   # 39 tests
pytest tests/test_rag_retrieval.py -v   # 20 tests
pytest tests/test_rag_integration.py -v # 16 tests

# Run with coverage
pytest tests/test_rag_*.py --cov=src/voice_agent --cov-report=html
```

### 10.2 E2E Tests with Real Services

```bash
# Requires: Supabase running, OpenAI key, LM Studio (optional)
python scripts/test_rag_real_docs.py
```

**E2E Test Coverage:**

| Test | What It Verifies |
|------|------------------|
| `supabase_connection` | Database connectivity, table exists |
| `ocr_qwen` | OCR via Qwen3-VL extracts text |
| `openai_embeddings` | 1536d embeddings generated |
| `document_loading` | PDF/DOCX/XLSX all load correctly |
| `full_pipeline` | Load → Chunk → Embed → Store → Search |
| `cross_encoder_reranking` | Reranker scores documents correctly |
| `siglip_embeddings` | 768d image embeddings generated |

### 10.3 Manual Testing

**Test Ingestion:**
```bash
# Create test PDF
echo "Test content for RAG" | enscript -o - | ps2pdf - test.pdf

# Ingest
python scripts/ingest_kb.py test.pdf --category test --verbose

# Verify in database
docker exec supabase-db psql -U postgres -d postgres -c \
  "SELECT source_document, chunk_index, substring(content, 1, 50) FROM knowledge_base_chunks LIMIT 5;"
```

**Test Search:**
```python
from voice_agent.tools.crm_knowledge import search_knowledge_base

result = search_knowledge_base("Test content", limit=3)
print(result)
```

---

## 11. Feature Toggle Reference

### 11.1 Quick Reference Table

| Feature | Toggle | Default | Effect When Disabled |
|---------|--------|---------|---------------------|
| Hybrid Search | `RAG_USE_HYBRID_BM25` | `True` | Vector-only search |
| Cross-Encoder Reranking | `RAG_RERANKING_ENABLED` | `True` | Lightweight reranker fallback |
| Image Embeddings | `RAG_ENABLE_IMAGE_EMBEDDINGS` | `False` | Skip image processing |
| Table Preservation | `RAG_PRESERVE_TABLES` | `True` | Tables split into chunks |
| Deduplication | `RAG_DEDUP_ENABLED` | `True` | Duplicate chunks stored |

### 11.2 Performance vs Quality Trade-offs

| Scenario | Recommended Settings |
|----------|---------------------|
| **Low latency** | `RAG_RERANKING_ENABLED=false`, `RAG_TOP_K_RETRIEVE=20` |
| **High accuracy** | `RAG_RERANKING_ENABLED=true`, `RAG_TOP_K_RETRIEVE=100` |
| **Image-heavy docs** | `RAG_ENABLE_IMAGE_EMBEDDINGS=true`, `RAG_TOP_K_IMAGE=5` |
| **Text-only docs** | `RAG_ENABLE_IMAGE_EMBEDDINGS=false` |
| **Portuguese content** | `RAG_FTS_LANGUAGE=portuguese` |
| **English content** | `RAG_FTS_LANGUAGE=english` |

---

## 12. Troubleshooting

### 12.1 Common Issues

**Issue: "No organization found"**
```
Solution: Create an organization in the database first
docker exec supabase-db psql -U postgres -d postgres -c \
  "INSERT INTO organizations (id, name) VALUES ('00000000-0000-0000-0000-000000000001', 'Test Org');"
```

**Issue: "Vector search returns empty"**
```
Solution: Ensure chunks have embeddings
docker exec supabase-db psql -U postgres -d postgres -c \
  "SELECT COUNT(*) FROM knowledge_base_chunks WHERE vector_embedding IS NOT NULL;"
```

**Issue: "OCR endpoint not responding"**
```
Solution: Start LM Studio with Qwen3-VL model on port 1234
Or disable OCR: RAG_OCR_MODE=disabled
```

**Issue: "SigLIP model not loading"**
```
Solution: Install sentencepiece
pip install sentencepiece
```

**Issue: "Cross-encoder reranking fails"**
```
Solution: Install sentence-transformers
pip install sentence-transformers
```

**Issue: "Dimension mismatch" or "expected 4000 dimensions"**
```
Solution: Ensure embedder is truncating to 4000 dimensions
Check: RAG_VECTOR_DIM=4000 in environment
Verify: python -c "from voice_agent.config import SETTINGS; print(SETTINGS.rag_vector_dim)"
```

**Issue: "Cannot create HNSW index on vector(4096)"**
```
Solution: Use halfvec(4000) instead of vector(4096)
pgvector 0.8.0 limits: vector=2000, halfvec=4000 for HNSW
Apply migration: database/migrations/20260113_halfvec_4000.sql
```

### 12.2 Debug Logging

```bash
# Enable verbose logging
export LOG_LEVEL=DEBUG

# Or in Python
import logging
logging.getLogger("voice_agent").setLevel(logging.DEBUG)
```

### 12.3 Database Diagnostics

```sql
-- Check chunk counts by org
SELECT org_id, COUNT(*) FROM knowledge_base_chunks GROUP BY org_id;

-- Check modality distribution
SELECT modality, COUNT(*) FROM knowledge_base_chunks GROUP BY modality;

-- Check embedding coverage
SELECT 
  COUNT(*) as total,
  COUNT(vector_embedding) as with_text_embedding,
  COUNT(vector_image) as with_image_embedding
FROM knowledge_base_chunks;

-- Verify halfvec column types
SELECT 
    a.attname AS column_name,
    pg_catalog.format_type(a.atttypid, a.atttypmod) AS data_type
FROM pg_catalog.pg_attribute a
WHERE a.attrelid = 'knowledge_base_chunks'::regclass 
AND a.attname LIKE 'vector%';
-- Expected: halfvec(4000) for both columns

-- Check HNSW indexes exist
SELECT indexname, indexdef 
FROM pg_indexes 
WHERE tablename = 'knowledge_base_chunks' AND indexname LIKE '%vector%';
-- Should show hnsw indexes with halfvec_cosine_ops

-- Test vector search function
SELECT * FROM kb_chunks_vector_search(
  '00000000-0000-0000-0000-000000000001'::uuid,
  (SELECT vector_embedding FROM knowledge_base_chunks LIMIT 1),
  5
);
```

---

## Appendix A: File Reference

| File | Purpose |
|------|---------|
| `src/voice_agent/config.py` | All RAG configuration settings |
| `src/voice_agent/ingestion/loader.py` | Document loading and parsing |
| `src/voice_agent/ingestion/chunker.py` | Text chunking with table handling |
| `src/voice_agent/ingestion/embedder.py` | Embedding generation + truncation (4096→4000) |
| `src/voice_agent/ingestion/ocr.py` | OCR via Qwen3-VL |
| `src/voice_agent/ingestion/kb_ingest.py` | Ingestion orchestrator |
| `src/voice_agent/retrieval/hybrid_search.py` | Hybrid search implementation |
| `src/voice_agent/retrieval/reranker.py` | Qwen3-VL + CrossEncoder reranking |
| `src/voice_agent/tools/crm_knowledge.py` | Agent tool interface |
| `scripts/ingest_kb.py` | CLI ingestion tool |
| `scripts/backfill_kb_chunks.py` | Migration from old table |
| `scripts/test_rag_real_docs.py` | E2E testing script |
| `database/migrations/20260113_add_kb_chunks.sql` | Base database schema |
| `database/migrations/20260113_halfvec_4000.sql` | halfvec conversion + HNSW indexes |
| `backend/src/control_plane/api/routers/rag.py` | Backend API endpoints |

---

## Appendix B: Model Specifications

| Model | Type | Output Dims | Storage Dims | Storage Type | Location |
|-------|------|-------------|--------------|--------------|----------|
| `text-embedding-qwen3-vl-embedding-8b` | Text/Image Embedding | 4096 | 4000 | halfvec | Local (LM Studio) |
| `qwen3-vl-reranker-8b` | Multimodal Reranker | N/A | N/A | N/A | Local (LM Studio) |
| `qwen/qwen3-vl-8b` | OCR (Vision LLM) | N/A | N/A | N/A | Local (LM Studio) |
| `text-embedding-3-small` | Text Embedding (fallback) | 1536 | 1536 | vector | OpenAI API |
| `google/siglip-base-patch16-384` | Image Embedding (fallback) | 768 | 768 | vector | Local (~814MB) |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker (fallback) | N/A | N/A | N/A | Local (~80MB) |

### Storage Format Details

| Type | Precision | Bytes/Dim | Max HNSW Dims | Use Case |
|------|-----------|-----------|---------------|----------|
| `vector` | 32-bit float | 4 | 2000 | OpenAI embeddings (1536d) |
| `halfvec` | 16-bit float | 2 | 4000 | Qwen3-VL embeddings (4000d) |

> **Why halfvec?** Qwen3-VL outputs 4096 dimensions, but pgvector 0.8.0 limits HNSW indexes to 2000 dims for `vector` type. Using `halfvec` allows up to 4000 dims while also providing 50% storage savings.

---

## Appendix C: Glossary

| Term | Definition |
|------|------------|
| **BM25** | Best Match 25, probabilistic ranking function for full-text search |
| **Chunk** | A segment of document content, typically 1000 chars |
| **Cross-Encoder** | Neural model that scores query-document pairs for relevance |
| **halfvec** | pgvector 16-bit float vector type, supports up to 4000 dims for HNSW |
| **HNSW** | Hierarchical Navigable Small World, fast approximate nearest neighbor index |
| **OCR** | Optical Character Recognition, extracting text from images |
| **pgvector** | PostgreSQL extension for vector similarity search |
| **RAG** | Retrieval-Augmented Generation, enhancing LLMs with retrieved context |
| **RRF** | Reciprocal Rank Fusion, algorithm to combine multiple rankings |
| **SigLIP** | Sigmoid Loss for Language-Image Pre-training, image embedding model |
| **tsvector** | PostgreSQL full-text search vector type |
| **truncation** | Reducing embedding dimensions (4096→4000) for index compatibility |

---

*This document is maintained alongside the codebase. For updates, check the commit history of `docs/RAG_SYSTEM_GUIDE.md`.*
