# RAG E2E Testing Requirements

**Status**: ✅ FULLY IMPLEMENTED & TESTED  
**Created**: 2026-01-13  
**Last Updated**: 2026-01-13  
**Purpose**: Document all external dependencies for RAG tests and provide setup instructions for full E2E testing with real services.

---

## Executive Summary

The RAG system is now **fully operational** with all external services integrated and tested. The test suite contains:

- **238 unit tests** - All passing
- **75 RAG-specific tests** - All passing (mocked)
- **6 E2E real-service tests** - All passing (real services)

### ✅ All Features Implemented & Tested

| Feature | Status | Service |
|---------|--------|---------|
| Text Embeddings | ✅ Working | OpenAI `text-embedding-3-small` |
| OCR Processing | ✅ Working | Qwen3-VL at `http://127.0.0.1:1234/v1` |
| Vector Search | ✅ Working | Supabase + pgvector |
| BM25 Search | ✅ Working | PostgreSQL FTS |
| Hybrid Search | ✅ Working | RRF Fusion |
| Cross-Encoder Reranking | ✅ Working | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Document Loading | ✅ Working | PDF, DOCX, XLSX, CSV, TXT |
| Backend API | ✅ Working | `/api/v1/rag/rerank` endpoint |

---

## Table of Contents

1. [Current Testing Coverage](#current-testing-coverage)
2. [E2E Test Results](#e2e-test-results)
3. [External Dependencies](#external-dependencies)
4. [Environment Setup](#environment-setup)
5. [Running E2E Tests](#running-e2e-tests)
6. [API Endpoints](#api-endpoints)
7. [Sample Test Fixtures](#sample-test-fixtures)
8. [CI/CD Integration](#cicd-integration)

---

## Current Testing Coverage

### ✅ Unit Tests (238 Total - All Passing)

| Test File | Tests | Description |
|-----------|-------|-------------|
| `test_rag_ingestion.py` | 39 | Document loading, chunking, embedding |
| `test_rag_retrieval.py` | 20 | Hybrid search, RRF fusion, reranking |
| `test_rag_integration.py` | 16 | End-to-end pipeline integration |
| Other project tests | 163 | Voice agent, tools, services |

### ✅ E2E Real-Service Tests (6 Tests - All Passing)

| Test | Service Used | Status |
|------|--------------|--------|
| Supabase Connection | Local Supabase + pgvector | ✅ PASSED |
| OCR with Qwen3-VL | `http://127.0.0.1:1234/v1` | ✅ PASSED |
| OpenAI Embeddings | `api.openai.com` | ✅ PASSED |
| Document Loading | pymupdf, python-docx, openpyxl | ✅ PASSED |
| Full Ingestion Pipeline | All services | ✅ PASSED |
| CrossEncoder Reranking | sentence-transformers | ✅ PASSED |

---

## E2E Test Results

### Last Test Run (2026-01-13)

```
============================================================
RAG Real Documents Integration Tests
============================================================
Project root: /home/matheus/repos/voice-agent-v5
Documents dir: /home/matheus/repos/voice-agent-v5/docs/pdfs
OpenAI configured: Yes
Supabase configured: Yes
OCR endpoint: http://127.0.0.1:1234/v1

TEST SUMMARY
============================================================
  supabase_connection: ✅ PASSED
  ocr_qwen: ✅ PASSED
  openai_embeddings: ✅ PASSED
  document_loading: ✅ PASSED
  full_pipeline: ✅ PASSED
  cross_encoder_reranking: ✅ PASSED

Total: 6 passed, 0 failed
```

### Full Pipeline Test Details

| Metric | Value |
|--------|-------|
| Document | Micro-Structure Sniper PDF (96.8 KB) |
| Pages | 13 |
| Characters | 43,383 |
| Chunks Created | 72 |
| Chunks Embedded | 10 (limited for test) |
| Chunks Stored | 10 in Supabase |
| Search | Successfully retrieved |
| Cleanup | Test data removed |

### CrossEncoder Reranking Test

| Metric | Value |
|--------|-------|
| Model | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Input Documents | 4 |
| Output (top_k) | 3 |
| ML Content Score | +9.5952 (highly relevant) |
| Deep Learning Score | -7.3104 (somewhat relevant) |
| Pizza Content Score | -11.0775 (not relevant) |
| Processing Time | ~25ms |

---

## External Dependencies

### 1. OpenAI API (Text Embeddings)

**Purpose**: Generate 1536-dimensional embeddings using `text-embedding-3-small`

**Requirements**:
- Valid OpenAI API key starting with `sk-`
- Network access to `api.openai.com`
- Estimated cost: ~$0.0001 per 1K tokens

**Environment Variables**:
```bash
OPENAI_API_KEY=sk-your-real-api-key
# Do NOT set OPENAI_BASE_URL (or ensure it points to api.openai.com)
```

**Verification Command**:
```bash
curl https://api.openai.com/v1/embeddings \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": "test", "model": "text-embedding-3-small"}'
```

---

### 2. Supabase Database (Storage & Search)

**Purpose**: Store chunks in `knowledge_base_chunks` table, perform vector/FTS search

**Requirements**:
- Supabase instance (local or cloud)
- pgvector extension enabled
- Migration `20260113_add_kb_chunks.sql` applied

**Environment Variables**:
```bash
# Local Supabase
SUPABASE_URL=http://localhost:8000
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key

# Cloud Supabase
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
```

**Setup Commands** (Local):
```bash
# Start Supabase
cd database && docker compose up -d

# Apply migration
docker exec -i supabase-db psql -U postgres -d postgres \
  < database/migrations/20260113_add_kb_chunks.sql
```

**Verification**:
```sql
SELECT COUNT(*) FROM knowledge_base_chunks;
SELECT * FROM pg_extension WHERE extname = 'vector';
```

---

### 3. SigLIP Image Embeddings (Optional)

**Purpose**: Generate 768-dimensional embeddings for images/diagrams

**Requirements**:
- `transformers` library installed
- `torch` installed
- Model download: `google/siglip-base-patch16-384` (~400MB)
- GPU recommended (CPU works but slow)

**Installation**:
```bash
pip install transformers torch
# First run will download the model
```

**Environment Variables**:
```bash
RAG_ENABLE_IMAGE_EMBEDDINGS=true
RAG_EMBED_MODEL_IMAGE=google/siglip-base-patch16-384
```

**Verification**:
```python
from transformers import AutoModel, AutoProcessor
processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-384")
model = AutoModel.from_pretrained("google/siglip-base-patch16-384")
print("SigLIP loaded successfully")
```

---

### 4. Cross-Encoder Reranking (✅ Implemented)

**Purpose**: Neural reranking with `cross-encoder/ms-marco-MiniLM-L-6-v2`

**Status**: ✅ Fully implemented and tested

**Requirements**:
- `sentence-transformers` library installed (v5.2.0+)
- Model download (~80MB on first use)

**Installation**:
```bash
pip install sentence-transformers
```

**Environment Variables**:
```bash
RAG_RERANKING_ENABLED=true
RAG_RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

**Performance Metrics**:
| Metric | Value |
|--------|-------|
| Model Load Time | ~6 seconds (first load) |
| Batch of 4 docs | ~25ms |
| Score Range | -11 to +10 (higher = more relevant) |
| Device | CPU (GPU optional) |

**Verification**:
```python
from sentence_transformers import CrossEncoder
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
scores = model.predict([["query", "document"]])
print(f"Score: {scores[0]}")
```

**Backend API Endpoints** (also available):
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/rag/health` | GET | Check reranker status |
| `/api/v1/rag/warmup` | POST | Pre-load model |
| `/api/v1/rag/rerank` | POST | Rerank documents |
| `/api/v1/rag/score` | POST | Score single pair |

---

### 5. OCR with Qwen3-VL (✅ Implemented)

**Purpose**: Extract text from scanned documents and images using vision-language model

**Current Configuration**:
- **Endpoint**: `http://127.0.0.1:1234/v1` (LM Studio local server)
- **Model**: `qwen/qwen3-vl-8b`
- **API Type**: OpenAI-compatible vision chat API

**Environment Variables**:
```bash
RAG_OCR_ENDPOINT=http://127.0.0.1:1234/v1
RAG_OCR_MODEL=qwen/qwen3-vl-8b
RAG_OCR_MODE=qwen3-vl
```

**How It Works**:
- Images are base64-encoded and sent to the vision model
- Uses `/chat/completions` endpoint with image_url content type
- Returns extracted text with confidence estimation

**Verification**:
```python
from voice_agent.ingestion.ocr import OCRProcessor

processor = OCRProcessor()
result = await processor.process_image(image_bytes)
print(f"Text: {result.text}")
print(f"Confidence: {result.confidence}")
```

**Alternative OCR Options**:
- Tesseract (open source): `pip install pytesseract`
- AWS Textract
- Google Cloud Vision
- DeepSeek OCR (original implementation)

---

### 6. PDF Parsing (pymupdf)

**Purpose**: Extract text and images from PDF documents

**Requirements**:
- `pymupdf` library installed

**Installation**:
```bash
pip install pymupdf
```

**Verification**:
```python
import fitz  # pymupdf
doc = fitz.open("sample.pdf")
print(f"Pages: {len(doc)}")
```

---

### 7. DOCX Parsing (python-docx)

**Purpose**: Extract text from Word documents

**Requirements**:
- `python-docx` library installed

**Installation**:
```bash
pip install python-docx
```

---

## Running E2E Tests

### Real-Service E2E Test Script

The comprehensive E2E test script is located at: `scripts/test_rag_real_docs.py`

**Run the tests**:
```bash
cd /home/matheus/repos/voice-agent-v5
python scripts/test_rag_real_docs.py
```

### What Gets Tested

| Test | Service | What It Validates |
|------|---------|-------------------|
| `test_supabase_connection` | Supabase | Database connectivity, table access |
| `test_ocr_with_qwen` | Qwen3-VL | Image → Text extraction |
| `test_openai_embeddings` | OpenAI | 1536d embedding generation |
| `test_real_document_loading` | pymupdf, python-docx | PDF, DOCX, XLSX parsing |
| `test_full_ingestion_pipeline` | All | Load → Chunk → Embed → Store → Search |
| `test_cross_encoder_reranking` | sentence-transformers | CrossEncoder scoring |

### Unit Tests (Mocked)

Unit tests use mocks for external services to run quickly in CI:

```bash
# Run all 238 unit tests
python -m pytest tests/ -v

# Run only RAG tests
python -m pytest tests/test_rag_*.py -v
```

| Test File | Tests | Focus |
|-----------|-------|-------|
| `test_rag_ingestion.py` | 39 | Loaders, chunkers, embedders (mocked) |
| `test_rag_retrieval.py` | 20 | Search, RRF, reranking (mocked) |
| `test_rag_integration.py` | 16 | Pipeline integration (mocked) |

---

## API Endpoints

### Reranking API (Backend Control Plane)

The backend service provides reranking as an API at `/api/v1/rag/`:

| Endpoint | Method | Request | Response |
|----------|--------|---------|----------|
| `/health` | GET | None | `{status, model_loaded, model_name}` |
| `/warmup` | POST | None | Pre-loads model, returns status |
| `/rerank` | POST | `{query, documents[], top_k}` | `{documents[], processing_time_ms}` |
| `/score` | POST | `{query, document}` | `{score, processing_time_ms}` |

**Example Rerank Request**:
```json
POST /api/v1/rag/rerank
{
  "query": "What is machine learning?",
  "documents": [
    {"id": "doc1", "content": "ML is a branch of AI...", "title": "ML Guide"},
    {"id": "doc2", "content": "Pizza is delicious...", "title": "Recipes"}
  ],
  "top_k": 5
}
```

**Response**:
```json
{
  "query": "What is machine learning?",
  "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
  "documents": [
    {
      "id": "doc1",
      "content": "ML is a branch of AI...",
      "rerank_score": 9.5,
      "original_rank": 0
    }
  ],
  "processing_time_ms": 25.3
}
```

---

## Environment Setup

### Current Working Configuration

The following `.env` configuration is verified working:

```bash
# ═══════════════════════════════════════════════════════════════════════════════
# OpenAI Configuration (REQUIRED for embeddings)
# ═══════════════════════════════════════════════════════════════════════════════
OPENAI_API_KEY=sk-your-real-openai-api-key
# Leave OPENAI_BASE_URL unset to use real OpenAI API

# ═══════════════════════════════════════════════════════════════════════════════
# Supabase Configuration (REQUIRED for storage/search)
# ═══════════════════════════════════════════════════════════════════════════════
SUPABASE_URL=http://localhost:8000
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key

# ═══════════════════════════════════════════════════════════════════════════════
# RAG Configuration
# ═══════════════════════════════════════════════════════════════════════════════
RAG_EMBED_MODEL_TEXT=text-embedding-3-small
RAG_VECTOR_DIM_TEXT=1536
RAG_EMBED_BATCH_SIZE=20

# Image Embeddings (optional - requires transformers + torch)
RAG_ENABLE_IMAGE_EMBEDDINGS=true
RAG_EMBED_MODEL_IMAGE=google/siglip-base-patch16-384
RAG_VECTOR_DIM_IMAGE=768

# Chunking
RAG_CHUNK_SIZE=1000
RAG_CHUNK_OVERLAP=200
RAG_PRESERVE_TABLES=true

# Hybrid Search
RAG_USE_HYBRID_BM25=true
RAG_TOP_K_RETRIEVE=50
RAG_TOP_K_IMAGE=3
RAG_FTS_LANGUAGE=portuguese

# Reranking (requires sentence-transformers)
RAG_RERANKING_ENABLED=true
RAG_RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RAG_TOP_K_RERANK=5

# OCR with Qwen3-VL (local LM Studio)
RAG_OCR_MODE=qwen3-vl
RAG_OCR_ENDPOINT=http://127.0.0.1:1234/v1
RAG_OCR_MODEL=qwen/qwen3-vl-8b
RAG_OCR_CONFIDENCE_THRESHOLD=0.6
RAG_OCR_RETRY_LIMIT=2

# Deduplication
RAG_DEDUP_ENABLED=true
RAG_MAX_WORKERS=4
```

### Install All Dependencies

```bash
# Core RAG dependencies
pip install pymupdf python-docx openpyxl openai numpy

# Image embeddings (SigLIP)
pip install transformers torch

# Cross-encoder reranking (REQUIRED for full functionality)
pip install sentence-transformers

# Optional: OCR (if not using Qwen3-VL)
pip install pytesseract pillow
```

---

## Implemented E2E Test Script

### scripts/test_rag_real_docs.py

This is the primary E2E test script that validates all RAG components with real services.

**Location**: `scripts/test_rag_real_docs.py`

**Run Command**:
```bash
python scripts/test_rag_real_docs.py
```

### Test Classes and Methods

```python
class TestRunner:
    """Run integration tests with real documents."""
    
    async def test_supabase_connection(self) -> bool:
        """Test Supabase database connection."""
        # Verifies:
        # - Database connectivity
        # - knowledge_base_chunks table access
        # - Test organization creation
    
    async def test_ocr_with_qwen(self) -> bool:
        """Test OCR with Qwen3-VL vision model."""
        # Verifies:
        # - OCR endpoint connectivity
        # - Image → Text extraction
        # - Confidence scoring
    
    async def test_openai_embeddings(self) -> bool:
        """Test OpenAI embedding generation."""
        # Verifies:
        # - API connectivity
        # - 1536-dimension embedding generation
        # - Proper normalization
    
    async def test_real_document_loading(self) -> bool:
        """Test loading real PDF, DOCX, XLSX files."""
        # Verifies:
        # - PDF loading via pymupdf
        # - DOCX loading via python-docx
        # - XLSX loading via openpyxl
        # - Text extraction quality
    
    async def test_full_ingestion_pipeline(self) -> bool:
        """Test full pipeline: Load → Chunk → Embed → Store → Search."""
        # Verifies:
        # - Document loading
        # - Chunking with overlap
        # - Embedding generation
        # - Database storage
        # - Search retrieval
        # - Cleanup
    
    async def test_cross_encoder_reranking(self) -> bool:
        """Test CrossEncoder reranking with real model."""
        # Verifies:
        # - Model loading
        # - Score computation
        # - Correct ranking (ML content > Pizza content)
```

### Sample Output

```
============================================================
RAG Real Documents Integration Tests
============================================================
Project root: /home/matheus/repos/voice-agent-v5
Documents dir: /home/matheus/repos/voice-agent-v5/docs/pdfs
OpenAI configured: Yes
Supabase configured: Yes
OCR endpoint: http://127.0.0.1:1234/v1

TEST: Supabase Connection
============================================================
  Connected to Supabase
  knowledge_base_chunks table accessible
  Test organization exists: 00000000-0000-0000-0000-000000000001
✅ supabase_connection: PASSED

TEST: OCR with Qwen3-VL
============================================================
  OCR Endpoint: http://127.0.0.1:1234/v1
  OCR Model: qwen/qwen3-vl-8b
  Mode used: qwen3-vl
  Confidence: 0.85
✅ ocr_qwen: PASSED

TEST: OpenAI Embeddings
============================================================
  Model: text-embedding-3-small
  Dimension: 1536
  Generated embedding: dim=1536
✅ openai_embeddings: PASSED

TEST: Real Document Loading
============================================================
  Found 2 PDFs, 2 DOCX, 1 XLSX
  Loading: Deep Volatility Arbitrageur.pdf
    ✓ Pages: 11
    ✓ Content: 44231 chars
  Loaded: 5/5
✅ document_loading: PASSED

TEST: Full Ingestion Pipeline
============================================================
  Using: Micro-Structure Sniper PDF (96.8 KB)
  Step 1: Loading document... 43383 chars across 13 pages
  Step 2: Chunking document... Created 72 chunks
  Step 3: Generating embeddings... Embedded 10/10 chunks
  Step 4: Storing in database... Stored 10 chunks
  Step 5: Testing search... Found 3 chunks
  Step 6: Cleanup... Removed test data
✅ full_pipeline: PASSED

TEST: CrossEncoder Reranking
============================================================
  Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  Query: What is artificial intelligence and machine learning?
  Input: 4 documents
  Output: 3 reranked documents
    [1] Score: 9.5952 - Machine learning is a subset of artificial intelli...
    [2] Score: -7.3104 - Deep learning uses neural networks with many layer...
    [3] Score: -11.0775 - Pizza is a popular Italian dish with cheese and to...
✅ cross_encoder_reranking: PASSED

TEST SUMMARY
============================================================
  supabase_connection: ✅ PASSED
  ocr_qwen: ✅ PASSED
  openai_embeddings: ✅ PASSED
  document_loading: ✅ PASSED
  full_pipeline: ✅ PASSED
  cross_encoder_reranking: ✅ PASSED

Total: 6 passed, 0 failed
```

---

## Test Documents

### Real Documents Used (docs/pdfs/)

The E2E tests use real documents from the `docs/pdfs/` directory:

| Document | Type | Size | Pages | Content |
|----------|------|------|-------|---------|
| Deep Volatility Arbitrageur.pdf | PDF | ~50KB | 11 | Academic paper on options pricing |
| Technical Report.pdf | PDF | ~30KB | 6 | Technical documentation |
| Micro-Structure Sniper.pdf | PDF | 96.8KB | 13 | Trading strategy document |
| Open-Source VAD Solutions.docx | DOCX | ~40KB | 1 | Voice activity detection research |
| Algorithmic Trading Report.docx | DOCX | ~100KB | 1 | Trading system feasibility |
| output (2).xlsx | XLSX | ~10KB | 4 | Spreadsheet data |

---

## Summary

### ✅ All Features Implemented & Tested

| Feature | Status | Service | Cost/Resource |
|---------|--------|---------|---------------|
| Text Embeddings | ✅ Working | OpenAI API | ~$0.0001/1K tokens |
| Vector Search | ✅ Working | Supabase + pgvector | Free (local) |
| BM25/FTS Search | ✅ Working | PostgreSQL | Free (local) |
| Hybrid Search | ✅ Working | RRF Fusion | Free |
| Image Embeddings | ✅ Working | SigLIP (transformers) | ~814MB model |
| Cross-Encoder | ✅ Working | sentence-transformers | ~80MB model |
| OCR | ✅ Working | Qwen3-VL (LM Studio) | Free (local) |
| PDF Parsing | ✅ Working | pymupdf | Included |
| DOCX Parsing | ✅ Working | python-docx | Included |
| XLSX Parsing | ✅ Working | openpyxl | Included |
| Backend API | ✅ Working | FastAPI `/api/v1/rag/` | Free |

### Quick Start Checklist

1. [x] Get real OpenAI API key (`sk-...`)
2. [x] Start local Supabase
3. [x] Apply database migrations
4. [x] Install sentence-transformers
5. [x] Configure Qwen3-VL for OCR
6. [x] Run E2E tests - **All 6 passing!**

### Test Commands

```bash
# Run all 238 unit tests
python -m pytest tests/ -v

# Run RAG-specific tests
python -m pytest tests/test_rag_*.py -v

# Run E2E real-service tests
python scripts/test_rag_real_docs.py
```

### Files Created/Modified

| File | Purpose |
|------|---------|
| `scripts/test_rag_real_docs.py` | Main E2E test script (6 tests) |
| `backend/src/control_plane/api/routers/rag.py` | Reranking API endpoints |
| `backend/src/control_plane/schemas/rag.py` | API schemas |
| `src/voice_agent/ingestion/ocr.py` | Qwen3-VL OCR integration |
| `src/voice_agent/ingestion/embedder.py` | SigLIP embedder |
| `src/voice_agent/retrieval/reranker.py` | CrossEncoder reranker |
| `database/migrations/20260113_fix_vector_search_types.sql` | Type cast fix |

---

## CI/CD Integration

### GitHub Actions Workflow

Create `.github/workflows/rag-e2e-tests.yml`:

```yaml
name: RAG E2E Tests

on:
  workflow_dispatch:  # Manual trigger
  schedule:
    - cron: '0 6 * * 1'  # Weekly on Mondays

jobs:
  e2e-tests:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: pgvector/pgvector:pg16
        env:
          POSTGRES_PASSWORD: postgres
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      
      - name: Install dependencies
        run: |
          pip install -e ".[rag,dev]"
          pip install sentence-transformers transformers torch
      
      - name: Apply migrations
        run: |
          psql $DATABASE_URL < database/migrations/20260113_add_kb_chunks.sql
        env:
          DATABASE_URL: postgresql://postgres:postgres@localhost:5432/postgres
      
      - name: Run E2E tests
        run: python scripts/test_rag_real_docs.py
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          SUPABASE_URL: http://localhost:8000
          SUPABASE_SERVICE_ROLE_KEY: ${{ secrets.SUPABASE_KEY }}
          RAG_OCR_ENDPOINT: http://127.0.0.1:1234/v1
          RAG_OCR_MODEL: qwen/qwen3-vl-8b
          RAG_RERANKING_ENABLED: true
```

