# RAG One-Shot Implementation: Complete Context Document

**Status**: Implementation Blueprint  
**Date**: January 2026  
**Scope**: Vector-based Retrieval-Augmented Generation with Hybrid Search, Multi-Modal Embeddings, and Portuguese Language Support  
**Target**: Voice Agent / CRM Knowledge Base System

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Architecture Overview](#architecture-overview)
3. [Models & Embedding Strategy](#models--embedding-strategy)
4. [Database Schema & Migrations](#database-schema--migrations)
5. [Configuration & Environment](#configuration--environment)
6. [Ingestion Pipeline](#ingestion-pipeline)
7. [Retrieval Pipeline](#retrieval-pipeline)
8. [Quality & Evaluation](#quality--evaluation)
9. [Observability & Monitoring](#observability--monitoring)
10. [Performance & Cost Guardrails](#performance--cost-guardrails)
11. [Implementation Phases](#implementation-phases)
12. [Code Touchpoints & Deliverables](#code-touchpoints--deliverables)
13. [Appendix: Dependencies & References](#appendix-dependencies--references)

---

## Executive Summary

### Objective
Implement a production-grade Retrieval-Augmented Generation (RAG) system with:
- **Hybrid search**: BM25 full-text search + dense vector similarity (ANN)
- **Multi-modal embeddings**: Text (OpenAI `text-embedding-3-small`, 1536d) + Image (Google `SigLIP Base`, 768d)
- **Portuguese language optimization**: Native FTS support, multilingual model selection
- **Quality assurance**: Reranking, OCR confidence tracking, deduplication
- **Scalability**: Indexed retrieval, batch embeddings, cost-optimized inference

### Key Decisions Finalized
- **Text Embeddings**: OpenAI `text-embedding-3-small` (1536 dimensions, SaaS-managed)
- **Image Embeddings**: Google `google/siglip-base-patch16-384` (768 dimensions, multilingual-aware, self-hosted)
- **OCR**: DeepSeek Base (fallback to Large on confidence < threshold)
- **Search Strategy**: Dual-stage hybrid (BM25 + ANN) + optional reranking
- **Database Split**: Document-level `knowledge_base` + chunk-level `knowledge_base_chunks`

### Deliverables Timeline
- **Phase 0–2**: Schema design & configuration (~2–3 days)
- **Phase 3–4**: Pipeline implementation (~5–7 days)
- **Phase 5–6**: Backfill, evaluation, integration (~3–4 days)
- **Phase 7–10**: Monitoring, observability, rollout (~2–3 days)

---

## Architecture Overview

### System Diagram (Conceptual)

```
┌─────────────────────────────────────────────────────────────┐
│                    INGESTION PIPELINE                       │
├─────────────────────────────────────────────────────────────┤
│  File Input (PDF, DOCX, Images, CSV/XLSX, TXT)             │
│           ↓                                                  │
│  Type Detection & Normalization                            │
│  (native text, OCR, structured table parsing)              │
│           ↓                                                  │
│  OCR (DeepSeek Base, w/ retry & confidence tracking)       │
│           ↓                                                  │
│  Chunking (1000 char, 200 overlap; preserve tables/images) │
│           ↓                                                  │
│  Deduplication (content_hash per org)                      │
│           ↓                                                  │
│  Embedding Generation                                      │
│  ├─ Text: text-embedding-3-small (1536d)                  │
│  └─ Images: SigLIP Base (768d) + captions                 │
│           ↓                                                  │
│  knowledge_base_chunks (+ provenance, OCR confidence)      │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    RETRIEVAL PIPELINE                       │
├─────────────────────────────────────────────────────────────┤
│  Query Input (text, optional: language hint)               │
│           ↓                                                  │
│  Query Embedding                                           │
│  ├─ Text: text-embedding-3-small (1536d, normalized)      │
│  └─ Optional: Image branch via SigLIP text tower          │
│           ↓                                                  │
│  Hybrid Retrieval                                          │
│  ├─ BM25 FTS on content (Portuguese-aware)                │
│  ├─ ANN on vector_embedding (IVFFlat/HNSW)               │
│  └─ Optional: ANN on vector_image + caption fusion        │
│           ↓                                                  │
│  RRF Fusion (Reciprocal Rank Fusion)                       │
│  ├─ Combine BM25 + ANN scores                             │
│  └─ Apply filters (org_id, category, source_document)    │
│           ↓                                                  │
│  Reranking (optional, CrossEncoder MiniLM)                │
│           ↓                                                  │
│  Return Top-K (with provenance, modality, OCR confidence)  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                 QUALITY & OBSERVABILITY                      │
├─────────────────────────────────────────────────────────────┤
│  Metrics: NDCG@10, Recall, MRR, latency, OCR confidence   │
│  Logs: Low confidence warnings, dedup hits, index errors  │
│  Alerts: OCR failures, NDCG/Recall regression, p95 spike  │
│  Smoke Tests: CI integration, fixed query assertions      │
└─────────────────────────────────────────────────────────────┘
```

---

## Models & Embedding Strategy

### Text Embeddings: OpenAI `text-embedding-3-small`

**Why This Choice:**
- **Dimensionality**: 1536 (vs. 1024 for older models) → higher expressiveness
- **Cost**: ~$0.02/1M tokens (extremely economical)
- **Performance**: Better semantic understanding than Ada; MTEB benchmark top-tier
- **Language Support**: Excellent English; good cross-lingual transfer for Portuguese
- **SaaS**: No self-hosting overhead; automatic model updates

**Integration:**
```python
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=text_chunks,  # up to 100k tokens per request
    encoding_format="float"
)
embeddings = [item.embedding for item in response.data]
# Each embedding: list of 1536 floats
```

**Normalization:**
Always L2-normalize before storage and retrieval:
```python
import numpy as np

embeddings_normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
```

**Batch Processing:**
- Max ~100 items per request (API limit); batch in groups of 20–50 for stability
- Cost: ~$0.02 per 1M tokens → ~$2–3 per 1M chunks of avg. 500 chars
- Latency: ~200–300ms per batch of 20; cache aggressively

### Image Embeddings: Google `google/siglip-base-patch16-384`

**Why This Choice:**
- **Multimodal**: Unified text + image embedding space → text-to-image queries work naturally
- **Multilingual**: Trained on cross-lingual data; handles Portuguese captions well
- **Dimensions**: 768 (vs. 1024 for larger models) → storage/performance balance
- **Self-hosted**: No API dependency; can run on modest GPU (6–8 GB VRAM)
- **Open-source**: Hugging Face; active community, no licensing friction

**Integration:**
```python
from transformers import AutoProcessor, AutoModel
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-384")
model = AutoModel.from_pretrained("google/siglip-base-patch16-384").to(device)

# Image embedding
with torch.no_grad():
    image_inputs = processor(images=[pil_image], return_tensors="pt").to(device)
    image_embeddings = model.get_image_features(**image_inputs)  # [1, 768]

# Text embedding (for text-to-image search)
with torch.no_grad():
    text_inputs = processor(text=["chart showing sales trends"], return_tensors="pt").to(device)
    text_embeddings = model.get_text_features(**text_inputs)  # [1, 768]
```

**L2 Normalization (Critical):**
```python
image_embeddings_norm = torch.nn.functional.normalize(image_embeddings, p=2, dim=-1)
```

**Batch Processing:**
- Process images in batches of 8–16 per GPU iteration (adjust for 6GB VRAM)
- Latency: ~50–100ms per batch of 16 on GPU
- Cost: Free (self-hosted)

### Multi-Modal Fusion Strategy

**Text-Dominant Retrieval (Most Queries):**
1. Query text → `text-embedding-3-small` → query vector (1536d)
2. Retrieve via BM25 + ANN on `vector_embedding` (text)
3. Return text chunks + associated images/captions

**Image-Heavy Retrieval (Optional Branch):**
1. Query text → `SigLIP text tower` → query vector (768d)
2. Retrieve via ANN on `vector_image` (image)
3. Also retrieve via image captions (already embedded as text chunks)
4. Fuse text and image results via RRF

**Key Design Principle:**
- Store image captions as **text chunks** (embedded via `text-embedding-3-small`)
- This ensures text-only queries still surface image-heavy pages
- Direct image vector search is supplementary; doesn't replace caption search

---

## Database Schema & Migrations

### Current State
**Table**: `knowledge_base` (document-level)
- `id` uuid pk
- `org_id` uuid (org isolation)
- `title` text
- `content` text
- `vector_embedding` vector(1536) (legacy from single-model approach)
- Other fields: `created_at`, `updated_at`, metadata

### New Design: Chunk-Level Schema

#### Table: `knowledge_base_chunks`

```sql
CREATE TABLE knowledge_base_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    knowledge_base_id UUID REFERENCES knowledge_base(id) ON DELETE CASCADE,
    
    -- Document provenance
    source_document TEXT NOT NULL,
    category TEXT,
    title TEXT,
    
    -- Modality classification
    modality TEXT NOT NULL CHECK (modality IN ('text', 'image', 'image_caption', 'table')),
    
    -- Chunk positioning
    page INT,
    chunk_index INT,
    
    -- Content & hashing
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    
    -- OCR metadata (populated for images, scanned documents)
    ocr_confidence REAL,
    ocr_model_version TEXT,
    
    -- Table-specific fields
    is_table BOOLEAN DEFAULT FALSE,
    table_context TEXT,
    
    -- Image-specific fields
    alt_text TEXT,
    image_path TEXT,
    
    -- Embeddings (normalized L2)
    vector_embedding VECTOR(1536),      -- text-embedding-3-small
    vector_image VECTOR(768),            -- SigLIP Base
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(org_id, content_hash)
);

-- Indexes for ingestion & retrieval
CREATE INDEX idx_kb_chunks_org_doc ON knowledge_base_chunks(org_id, source_document, modality);
CREATE INDEX idx_kb_chunks_modality ON knowledge_base_chunks(modality);
CREATE INDEX idx_kb_chunks_page ON knowledge_base_chunks(source_document, page);

-- Vector indexes for ANN search
CREATE INDEX idx_kb_chunks_vector_text ON knowledge_base_chunks 
  USING ivfflat (vector_embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX idx_kb_chunks_vector_image ON knowledge_base_chunks 
  USING ivfflat (vector_image vector_cosine_ops) WITH (lists = 50);

-- Full-text search index (Portuguese)
CREATE INDEX idx_kb_chunks_fts_content ON knowledge_base_chunks 
  USING GIN (to_tsvector('portuguese', content));

-- RLS (Row-Level Security)
ALTER TABLE knowledge_base_chunks ENABLE ROW LEVEL SECURITY;

CREATE POLICY org_isolation_kb_chunks ON knowledge_base_chunks
  USING (org_id = current_org_id());
```

#### Optional: Update `knowledge_base` for Cross-Reference

```sql
ALTER TABLE knowledge_base
  ADD COLUMN source_document TEXT,
  ADD COLUMN doc_checksum TEXT,
  ADD COLUMN ingestion_metadata JSONB,
  ADD COLUMN chunk_count INT DEFAULT 0;
```

### Migration File Template

**Location**: `database/migrations/<timestamp>_create_knowledge_base_chunks.sql`

```sql
-- Migration: Create knowledge_base_chunks table with embeddings & FTS
-- Timestamp: 2026-01-13
-- Rollback: DROP TABLE IF EXISTS knowledge_base_chunks CASCADE;

BEGIN;

-- Table creation
CREATE TABLE IF NOT EXISTS knowledge_base_chunks (
    -- [full schema above]
);

-- Indexes (in order for optimal build time)
CREATE INDEX CONCURRENTLY idx_kb_chunks_org_doc 
  ON knowledge_base_chunks(org_id, source_document, modality);

CREATE INDEX CONCURRENTLY idx_kb_chunks_modality 
  ON knowledge_base_chunks(modality);

CREATE INDEX CONCURRENTLY idx_kb_chunks_vector_text 
  ON knowledge_base_chunks USING ivfflat (vector_embedding vector_cosine_ops);

CREATE INDEX CONCURRENTLY idx_kb_chunks_fts_content 
  ON knowledge_base_chunks USING GIN (to_tsvector('portuguese', content));

-- RLS
ALTER TABLE knowledge_base_chunks ENABLE ROW LEVEL SECURITY;
CREATE POLICY org_isolation_kb_chunks ON knowledge_base_chunks
  USING (org_id = current_org_id());

-- Grant access
GRANT SELECT, INSERT, UPDATE, DELETE ON knowledge_base_chunks TO app_role;

COMMIT;
```

### Pgvector Configuration

**Installation** (if not already present):
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

**Index Strategy** (for 1M+ chunks):
- **IVFFlat**: Good for <10M vectors; faster builds; ~4% error rate
- **HNSW**: Recommended for 10M+; slower builds; <1% error rate

**For initial rollout** (estimated <100k chunks), use IVFFlat. Migrate to HNSW if scale exceeds 1M.

---

## Configuration & Environment

### Environment Variables

**Location**: `.env` (or config management system)

```bash
# ============================================================================
# EMBEDDINGS & MODELS
# ============================================================================
EMBED_MODEL_TEXT=text-embedding-3-small
VECTOR_DIM_TEXT=1536
OPENAI_API_KEY=sk-...

EMBED_MODEL_IMAGE=google/siglip-base-patch16-384
VECTOR_DIM_IMAGE=768
ENABLE_IMAGE_EMBEDDINGS=true

# ============================================================================
# OCR
# ============================================================================
OCR_ENDPOINT=https://api.deepseek.com/ocr
OCR_API_KEY=<deepseek-key>
OCR_MODE=base                    # 'base' | 'base_retry' | 'large'
OCR_CONFIDENCE_THRESHOLD=0.6     # Skip chunks below this
RETRY_LIMIT_OCR=2                # Max retries per page
OCR_MODEL_VERSION=2024-01        # Track version for metadata

# ============================================================================
# CHUNKING & DEDUPLICATION
# ============================================================================
CHUNK_SIZE=1000                  # Characters
CHUNK_OVERLAP=200                # Characters
CHUNK_PRESERVE_TABLES=true       # Don't split markdown tables
DEDUP_ENABLED=true               # content_hash dedup per org
DEDUP_LOOKBACK_DAYS=30           # Only check recent chunks

# ============================================================================
# RETRIEVAL
# ============================================================================
USE_HYBRID_BM25=true
RERANKING_ENABLED=true
RERANKING_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

TOP_K_RETRIEVE=50                # Initial BM25 + ANN pool
TOP_K_IMAGE=3                    # Image-branch ANN limit
TOP_K_RERANK=5                   # Final results after reranking
SIMILARITY_CUTOFF=0.3            # Min cosine similarity to include

# ============================================================================
# PERFORMANCE & BATCHING
# ============================================================================
BATCH_SIZE_EMBED=32              # Texts per embedding request
BATCH_SIZE_OCR=4                 # Pages per OCR request
BATCH_SIZE_IMAGE=16              # Images per SigLIP batch
MAX_WORKERS_INGEST=4             # Parallel ingest workers
MAX_WORKERS_EMBED=2              # Parallel embed workers
REQUEST_TIMEOUT_EMBED=30         # Seconds
REQUEST_TIMEOUT_RETRIEVAL=10     # Seconds

# ============================================================================
# STORAGE & CACHE
# ============================================================================
POSTGRES_URL=postgresql://user:pass@host/db
REDIS_URL=redis://localhost:6379
ENABLE_RETRIEVAL_CACHE=true
CACHE_TTL_RETRIEVAL=3600         # Seconds

# ============================================================================
# FEATURE FLAGS & DEBUGGING
# ============================================================================
ENABLE_OCR_RETRY_ADAPTIVE=true   # Retry Large mode if base < threshold
ENABLE_IMAGE_CAPTIONS=true       # Generate captions for image chunks
LOG_LEVEL=INFO
LOG_OCR_CONFIDENCE=true          # Log per-page confidence
DEBUG_MODE=false
```

### Configuration Class (Python)

**Location**: `src/voice_agent/config.py`

```python
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class EmbeddingConfig:
    """Text & image embedding configuration."""
    model_text: str = os.getenv("EMBED_MODEL_TEXT", "text-embedding-3-small")
    dim_text: int = int(os.getenv("VECTOR_DIM_TEXT", "1536"))
    model_image: str = os.getenv("EMBED_MODEL_IMAGE", "google/siglip-base-patch16-384")
    dim_image: int = int(os.getenv("VECTOR_DIM_IMAGE", "768"))
    enable_image: bool = os.getenv("ENABLE_IMAGE_EMBEDDINGS", "true").lower() == "true"
    openai_key: str = os.getenv("OPENAI_API_KEY")
    
    def validate(self) -> None:
        """Fail-fast validation at startup."""
        assert self.openai_key, "OPENAI_API_KEY not set"
        assert self.dim_text == 1536, f"Expected dim_text=1536, got {self.dim_text}"
        assert self.dim_image == 768, f"Expected dim_image=768, got {self.dim_image}"

@dataclass
class OCRConfig:
    """OCR configuration."""
    endpoint: str = os.getenv("OCR_ENDPOINT", "https://api.deepseek.com/ocr")
    api_key: str = os.getenv("OCR_API_KEY")
    mode: str = os.getenv("OCR_MODE", "base")  # 'base' | 'large'
    confidence_threshold: float = float(os.getenv("OCR_CONFIDENCE_THRESHOLD", "0.6"))
    retry_limit: int = int(os.getenv("RETRY_LIMIT_OCR", "2"))
    model_version: str = os.getenv("OCR_MODEL_VERSION", "2024-01")

@dataclass
class ChunkingConfig:
    """Chunking & deduplication configuration."""
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    preserve_tables: bool = os.getenv("CHUNK_PRESERVE_TABLES", "true").lower() == "true"
    dedup_enabled: bool = os.getenv("DEDUP_ENABLED", "true").lower() == "true"
    dedup_lookback_days: int = int(os.getenv("DEDUP_LOOKBACK_DAYS", "30"))

@dataclass
class RetrievalConfig:
    """Retrieval pipeline configuration."""
    use_hybrid_bm25: bool = os.getenv("USE_HYBRID_BM25", "true").lower() == "true"
    reranking_enabled: bool = os.getenv("RERANKING_ENABLED", "true").lower() == "true"
    reranking_model: str = os.getenv("RERANKING_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    top_k_retrieve: int = int(os.getenv("TOP_K_RETRIEVE", "50"))
    top_k_image: int = int(os.getenv("TOP_K_IMAGE", "3"))
    top_k_rerank: int = int(os.getenv("TOP_K_RERANK", "5"))
    similarity_cutoff: float = float(os.getenv("SIMILARITY_CUTOFF", "0.3"))

@dataclass
class PerformanceConfig:
    """Performance & batching tuning."""
    batch_size_embed: int = int(os.getenv("BATCH_SIZE_EMBED", "32"))
    batch_size_ocr: int = int(os.getenv("BATCH_SIZE_OCR", "4"))
    batch_size_image: int = int(os.getenv("BATCH_SIZE_IMAGE", "16"))
    max_workers_ingest: int = int(os.getenv("MAX_WORKERS_INGEST", "4"))
    max_workers_embed: int = int(os.getenv("MAX_WORKERS_EMBED", "2"))
    request_timeout_embed: int = int(os.getenv("REQUEST_TIMEOUT_EMBED", "30"))
    request_timeout_retrieval: int = int(os.getenv("REQUEST_TIMEOUT_RETRIEVAL", "10"))

@dataclass
class RAGConfig:
    """Unified RAG configuration."""
    embedding: EmbeddingConfig
    ocr: OCRConfig
    chunking: ChunkingConfig
    retrieval: RetrievalConfig
    performance: PerformanceConfig
    
    @classmethod
    def from_env(cls) -> "RAGConfig":
        config = cls(
            embedding=EmbeddingConfig(),
            ocr=OCRConfig(),
            chunking=ChunkingConfig(),
            retrieval=RetrievalConfig(),
            performance=PerformanceConfig(),
        )
        config.embedding.validate()
        return config

# Global config instance (loaded at app startup)
rag_config = RAGConfig.from_env()
```

### Startup Validation Hook

**Location**: `src/voice_agent/app.py` (or main entry point)

```python
def startup_validation():
    """Validate config & schema alignment at startup."""
    from src.voice_agent.config import rag_config
    
    rag_config.embedding.validate()
    
    # Verify DB schema has correct vector dimensions
    result = db.execute("""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = 'knowledge_base_chunks'
          AND column_name IN ('vector_embedding', 'vector_image')
    """)
    schema_dims = {row['column_name']: row['data_type'] for row in result}
    
    assert 'vector_embedding' in schema_dims, "Column vector_embedding not found"
    assert 'vector_image' in schema_dims, "Column vector_image not found"
    
    logger.info("✓ RAG configuration & schema validation passed")

# Call at app startup
startup_validation()
```

---

## Ingestion Pipeline

### Overview

**Input**: File (PDF, DOCX, images, CSV/XLSX, TXT)  
**Output**: Chunks in `knowledge_base_chunks` table with embeddings and provenance

**Process**: Detect → Load → OCR → Chunk → Dedup → Embed → Insert

### 1. Type Detection & Normalization

**Location**: `src/voice_agent/ingestion/loader.py`

```python
import mimetypes
from pathlib import Path
from typing import Union, Tuple
from PIL import Image
import PyPDF2
from docx import Document
import openpyxl

class DocumentLoader:
    """Detect and load documents by type."""
    
    SUPPORTED_TYPES = {
        'application/pdf': 'pdf',
        'text/plain': 'txt',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'xlsx',
        'text/csv': 'csv',
        'image/jpeg': 'image',
        'image/png': 'image',
        'image/tiff': 'image',
    }
    
    @staticmethod
    def detect(file_path: str) -> str:
        """Return document type: 'pdf', 'image', 'docx', 'csv', 'xlsx', 'txt'."""
        mime_type, _ = mimetypes.guess_type(file_path)
        return DocumentLoader.SUPPORTED_TYPES.get(mime_type, 'unknown')
    
    @staticmethod
    def load_pdf(file_path: str) -> Tuple[list, list]:
        """
        Load PDF; return (pages: list[str], images_per_page: list[list[Image]]).
        If PDF has native text, use it; otherwise render to images.
        """
        pages = []
        images_per_page = []
        
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page_num, page in enumerate(reader.pages):
                # Try native text extraction
                text = page.extract_text()
                if text and len(text.strip()) > 50:
                    pages.append(text)
                    images_per_page.append([])
                else:
                    # Fallback: render to image
                    import pdf2image
                    img = pdf2image.convert_from_path(file_path, first_page=page_num+1, 
                                                       last_page=page_num+1, dpi=300)[0]
                    pages.append(None)  # Placeholder; will OCR image
                    images_per_page.append([img])
        
        return pages, images_per_page
    
    @staticmethod
    def load_docx(file_path: str) -> Tuple[str, list]:
        """Extract text from DOCX; extract embedded images."""
        doc = Document(file_path)
        text = '\n'.join([p.text for p in doc.paragraphs])
        images = []
        
        for rel in doc.part.rels.values():
            if 'image' in rel.target_ref:
                images.append(Image.open(rel.target_part.blob))
        
        return text, images
    
    @staticmethod
    def load_csv(file_path: str) -> str:
        """Load CSV; convert to Markdown table."""
        import pandas as pd
        df = pd.read_csv(file_path)
        return df.to_markdown(index=False)
    
    @staticmethod
    def load_xlsx(file_path: str) -> str:
        """Load XLSX; convert sheets to Markdown tables with sheet names."""
        import pandas as pd
        xls = pd.ExcelFile(file_path)
        output = []
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            output.append(f"## {sheet_name}\n")
            output.append(df.to_markdown(index=False))
        return '\n\n'.join(output)
    
    @staticmethod
    def load_txt(file_path: str) -> str:
        """Load plain text."""
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()
    
    @staticmethod
    def load_image(file_path: str) -> Image.Image:
        """Load image."""
        return Image.open(file_path)
```

### 2. OCR Pipeline

**Location**: `src/voice_agent/ingestion/ocr.py`

```python
import requests
import base64
from PIL import Image
from io import BytesIO
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class DeepSeekOCRClient:
    """Interface to DeepSeek OCR API."""
    
    def __init__(self, endpoint: str, api_key: str, version: str = "base"):
        self.endpoint = endpoint
        self.api_key = api_key
        self.version = version  # 'base' | 'large'
    
    def ocr_image(self, image: Image.Image, mode: str = "base", 
                  retry_on_low_confidence: bool = True) -> Tuple[str, float]:
        """
        OCR image; return (text, confidence_score).
        If mode='base' and confidence < threshold, optionally retry with 'large'.
        """
        text, confidence = self._call_api(image, mode)
        
        if retry_on_low_confidence and confidence < 0.6 and mode == "base":
            logger.warning(f"OCR confidence {confidence:.2f} < 0.6; retrying with 'large'")
            text, confidence = self._call_api(image, "large")
        
        return text, confidence
    
    def _call_api(self, image: Image.Image, mode: str) -> Tuple[str, float]:
        """Call DeepSeek OCR API."""
        # Encode image to base64
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        image_b64 = base64.b64encode(buffer.getvalue()).decode()
        
        payload = {
            "image": f"data:image/png;base64,{image_b64}",
            "mode": mode,  # 'base' or 'large'
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(f"{self.endpoint}/v1/ocr", json=payload, 
                                    headers=headers, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            text = result.get("text", "")
            confidence = result.get("confidence", 0.5)
            
            return text, confidence
        except Exception as e:
            logger.error(f"OCR API error: {e}")
            return "", 0.0
```

### 3. Chunking & Table Preservation

**Location**: `src/voice_agent/ingestion/chunking.py`

```python
import re
import hashlib
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class Chunk:
    content: str
    page: Optional[int]
    chunk_index: int
    modality: str  # 'text', 'image', 'image_caption', 'table'
    is_table: bool
    table_context: Optional[str]
    alt_text: Optional[str]
    ocr_confidence: Optional[float]

class DocumentChunker:
    """Split documents into chunks; preserve tables & images."""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200, 
                 preserve_tables: bool = True):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.preserve_tables = preserve_tables
    
    def chunk(self, text: str, page: int = None, modality: str = "text", 
              ocr_confidence: float = None) -> List[Chunk]:
        """
        Split text into chunks. If preserve_tables=True, don't split markdown tables.
        """
        chunks = []
        
        if self.preserve_tables and modality == "text":
            chunks = self._chunk_preserve_tables(text, page, ocr_confidence)
        else:
            chunks = self._chunk_simple(text, page, modality, ocr_confidence)
        
        return chunks
    
    def _chunk_preserve_tables(self, text: str, page: int, 
                               ocr_confidence: float) -> List[Chunk]:
        """Split text; preserve markdown tables as single chunks."""
        chunks = []
        chunk_index = 0
        
        # Split by markdown table delimiter (lines starting with |)
        parts = re.split(r'(\n\|.*?\|\n)', text, flags=re.MULTILINE)
        
        for part in parts:
            is_table = part.strip().startswith('|')
            
            if is_table:
                # Keep entire table as one chunk
                chunk = Chunk(
                    content=part.strip(),
                    page=page,
                    chunk_index=chunk_index,
                    modality="table",
                    is_table=True,
                    table_context=f"Table on page {page}" if page else None,
                    alt_text=None,
                    ocr_confidence=ocr_confidence
                )
                chunks.append(chunk)
                chunk_index += 1
            else:
                # Regular text: split by size
                sub_chunks = self._chunk_simple(part, page, "text", ocr_confidence)
                for i, sub_chunk in enumerate(sub_chunks):
                    sub_chunk.chunk_index = chunk_index + i
                chunks.extend(sub_chunks)
                chunk_index += len(sub_chunks)
        
        return chunks
    
    def _chunk_simple(self, text: str, page: int, modality: str,
                      ocr_confidence: float) -> List[Chunk]:
        """Simple overlapping chunking by character count."""
        chunks = []
        chunk_index = 0
        
        for i in range(0, len(text), self.chunk_size - self.overlap):
            chunk_text = text[i:i + self.chunk_size]
            if len(chunk_text.strip()) > 0:
                chunk = Chunk(
                    content=chunk_text,
                    page=page,
                    chunk_index=chunk_index,
                    modality=modality,
                    is_table=False,
                    table_context=None,
                    alt_text=None,
                    ocr_confidence=ocr_confidence
                )
                chunks.append(chunk)
                chunk_index += 1
        
        return chunks
    
    @staticmethod
    def compute_hash(content: str) -> str:
        """Compute content hash for deduplication."""
        # Normalize: strip whitespace, lowercase
        normalized = re.sub(r'\s+', ' ', content.strip().lower())
        return hashlib.sha256(normalized.encode()).hexdigest()
```

### 4. Deduplication

**Location**: `src/voice_agent/ingestion/dedup.py`

```python
import hashlib
from typing import List
import logging

logger = logging.getLogger(__name__)

class DeduplicationService:
    """Check chunks against existing DB entries."""
    
    def __init__(self, db_pool):
        self.db_pool = db_pool
    
    def deduplicate(self, org_id: str, chunks: List[Chunk], 
                    lookback_days: int = 30) -> List[Chunk]:
        """
        Filter out chunks already in DB for this org.
        Return only new chunks.
        """
        if not chunks:
            return []
        
        # Compute hashes
        hashes = [Chunk.compute_hash(c.content) for c in chunks]
        
        # Query DB for existing hashes
        with self.db_pool.connection() as conn:
            cursor = conn.cursor()
            placeholders = ','.join(['%s'] * len(hashes))
            query = f"""
                SELECT content_hash FROM knowledge_base_chunks
                WHERE org_id = %s
                  AND content_hash IN ({placeholders})
                  AND created_at > NOW() - INTERVAL '{lookback_days} days'
            """
            cursor.execute(query, [org_id] + hashes)
            existing_hashes = set(row[0] for row in cursor.fetchall())
        
        # Filter
        new_chunks = [c for c, h in zip(chunks, hashes) if h not in existing_hashes]
        skipped_count = len(chunks) - len(new_chunks)
        
        if skipped_count > 0:
            logger.info(f"Dedup: skipped {skipped_count}/{len(chunks)} chunks")
        
        return new_chunks, existing_hashes
```

### 5. Embedding Generation

**Location**: `src/voice_agent/ingestion/embedder.py`

```python
import numpy as np
from openai import OpenAI
from transformers import AutoProcessor, AutoModel
import torch
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

class TextEmbedder:
    """Generate text embeddings via OpenAI."""
    
    def __init__(self, api_key: str, model: str = "text-embedding-3-small", 
                 batch_size: int = 32):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.batch_size = batch_size
    
    def embed_batch(self, texts: List[str]) -> Tuple[np.ndarray, List[int]]:
        """
        Embed list of texts; return (embeddings [N, 1536], token_counts [N]).
        """
        embeddings = []
        token_counts = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch,
                    encoding_format="float"
                )
                
                for item in response.data:
                    embeddings.append(np.array(item.embedding, dtype=np.float32))
                    token_counts.append(item.index)
                
                logger.info(f"Embedded batch {i//self.batch_size + 1} ({len(batch)} texts)")
            except Exception as e:
                logger.error(f"Embedding error for batch {i}: {e}")
                # Fill with zeros on error (retried later)
                embeddings.extend([np.zeros(1536, dtype=np.float32) for _ in batch])
        
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # L2 normalize
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        embeddings_normalized = embeddings_array / (norms + 1e-8)
        
        return embeddings_normalized, token_counts

class ImageEmbedder:
    """Generate image embeddings via SigLIP."""
    
    def __init__(self, model_name: str = "google/siglip-base-patch16-384", 
                 batch_size: int = 16, device: str = "cuda"):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
    
    def embed_images(self, images: List[Image]) -> np.ndarray:
        """Embed list of PIL images; return [N, 768] normalized."""
        embeddings = []
        
        for i in range(0, len(images), self.batch_size):
            batch = images[i:i + self.batch_size]
            
            with torch.no_grad():
                inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
                outputs = self.model.get_image_features(**inputs)
                
                # L2 normalize
                outputs_norm = torch.nn.functional.normalize(outputs, p=2, dim=-1)
                embeddings.append(outputs_norm.cpu().numpy())
            
            logger.info(f"Embedded image batch {i//self.batch_size + 1} ({len(batch)} images)")
        
        if embeddings:
            return np.vstack(embeddings).astype(np.float32)
        return np.array([], dtype=np.float32).reshape(0, 768)
    
    def embed_text_for_image_search(self, texts: List[str]) -> np.ndarray:
        """Embed texts for image search (via SigLIP text tower)."""
        embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            with torch.no_grad():
                inputs = self.processor(text=batch, return_tensors="pt").to(self.device)
                outputs = self.model.get_text_features(**inputs)
                
                # L2 normalize
                outputs_norm = torch.nn.functional.normalize(outputs, p=2, dim=-1)
                embeddings.append(outputs_norm.cpu().numpy())
        
        if embeddings:
            return np.vstack(embeddings).astype(np.float32)
        return np.array([], dtype=np.float32).reshape(0, 768)
```

### 6. Ingestion Writer

**Location**: `src/voice_agent/ingestion/writer.py`

```python
from typing import List
import logging
import psycopg2.extras

logger = logging.getLogger(__name__)

class KnowledgeBaseWriter:
    """Write chunks to DB with embeddings & provenance."""
    
    def __init__(self, db_pool):
        self.db_pool = db_pool
    
    def write_chunks(self, org_id: str, chunks: List[Dict], 
                     source_document: str, knowledge_base_id: str = None):
        """
        Insert chunks into knowledge_base_chunks.
        chunks: list of dicts with keys: content, modality, page, chunk_index, 
                vector_embedding, vector_image, is_table, table_context, 
                alt_text, ocr_confidence
        """
        with self.db_pool.connection() as conn:
            cursor = conn.cursor()
            
            for chunk in chunks:
                query = """
                    INSERT INTO knowledge_base_chunks 
                    (org_id, knowledge_base_id, source_document, content, 
                     content_hash, modality, page, chunk_index, 
                     vector_embedding, vector_image,
                     is_table, table_context, alt_text, ocr_confidence)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (org_id, content_hash) DO NOTHING
                """
                
                cursor.execute(query, (
                    org_id,
                    knowledge_base_id,
                    source_document,
                    chunk['content'],
                    chunk['content_hash'],
                    chunk['modality'],
                    chunk['page'],
                    chunk['chunk_index'],
                    chunk['vector_embedding'].tobytes() if chunk['vector_embedding'] is not None else None,
                    chunk['vector_image'].tobytes() if chunk['vector_image'] is not None else None,
                    chunk['is_table'],
                    chunk['table_context'],
                    chunk['alt_text'],
                    chunk['ocr_confidence']
                ))
            
            conn.commit()
            logger.info(f"Wrote {len(chunks)} chunks to DB for org {org_id}")

```

### 7. Orchestration: Main Ingestion Script

**Location**: `scripts/ingest_kb.py`

```python
#!/usr/bin/env python3
"""
Ingest documents into knowledge_base_chunks.

Usage:
  python scripts/ingest_kb.py --org-id <uuid> --file <path> --knowledge-base-id <uuid>
"""

import argparse
import logging
from pathlib import Path
import sys

from src.voice_agent.config import rag_config
from src.voice_agent.ingestion.loader import DocumentLoader
from src.voice_agent.ingestion.ocr import DeepSeekOCRClient
from src.voice_agent.ingestion.chunking import DocumentChunker
from src.voice_agent.ingestion.dedup import DeduplicationService
from src.voice_agent.ingestion.embedder import TextEmbedder, ImageEmbedder
from src.voice_agent.ingestion.writer import KnowledgeBaseWriter
from src.db import get_db_pool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ingest_document(file_path: str, org_id: str, knowledge_base_id: str = None):
    """End-to-end ingestion pipeline."""
    
    file_path = Path(file_path)
    source_document = file_path.name
    
    logger.info(f"Starting ingestion: {source_document}")
    
    # Initialize services
    db_pool = get_db_pool()
    ocr_client = DeepSeekOCRClient(rag_config.ocr.endpoint, rag_config.ocr.api_key)
    chunker = DocumentChunker(
        chunk_size=rag_config.chunking.chunk_size,
        overlap=rag_config.chunking.chunk_overlap,
        preserve_tables=rag_config.chunking.preserve_tables
    )
    dedup_service = DeduplicationService(db_pool)
    text_embedder = TextEmbedder(rag_config.embedding.openai_key, 
                                 batch_size=rag_config.performance.batch_size_embed)
    image_embedder = ImageEmbedder(batch_size=rag_config.performance.batch_size_image)
    writer = KnowledgeBaseWriter(db_pool)
    
    # 1. Load document
    doc_type = DocumentLoader.detect(str(file_path))
    logger.info(f"Detected type: {doc_type}")
    
    if doc_type == "pdf":
        pages, images_per_page = DocumentLoader.load_pdf(str(file_path))
    elif doc_type == "docx":
        text, images = DocumentLoader.load_docx(str(file_path))
        pages = [text]
        images_per_page = [images]
    elif doc_type == "csv":
        pages = [DocumentLoader.load_csv(str(file_path))]
        images_per_page = [[]]
    elif doc_type == "xlsx":
        pages = [DocumentLoader.load_xlsx(str(file_path))]
        images_per_page = [[]]
    elif doc_type == "txt":
        pages = [DocumentLoader.load_txt(str(file_path))]
        images_per_page = [[]]
    elif doc_type == "image":
        pages = [None]
        images_per_page = [[DocumentLoader.load_image(str(file_path))]]
    else:
        raise ValueError(f"Unsupported file type: {doc_type}")
    
    # 2. OCR, chunk, embed
    all_chunks = []
    all_embeddings = []
    all_image_embeddings = []
    
    for page_num, (page_text, page_images) in enumerate(zip(pages, images_per_page)):
        # OCR images if present
        image_texts = []
        for img in page_images:
            if img is not None:
                ocr_text, ocr_conf = ocr_client.ocr_image(
                    img, 
                    mode=rag_config.ocr.mode,
                    retry_on_low_confidence=True
                )
                if ocr_conf >= rag_config.ocr.confidence_threshold:
                    image_texts.append((ocr_text, ocr_conf, img))
                else:
                    logger.warning(f"Low OCR confidence {ocr_conf:.2f} on page {page_num}")
        
        # Chunk text
        if page_text:
            text_chunks = chunker.chunk(page_text, page=page_num, modality="text")
            all_chunks.extend(text_chunks)
        
        # Chunk OCR'd images as captions
        for ocr_text, ocr_conf, img in image_texts:
            caption_chunks = chunker.chunk(
                ocr_text, page=page_num, modality="image_caption", 
                ocr_confidence=ocr_conf
            )
            all_chunks.extend(caption_chunks)
    
    # 3. Dedup
    new_chunks, _ = dedup_service.deduplicate(org_id, all_chunks, 
                                              rag_config.chunking.dedup_lookback_days)
    
    if not new_chunks:
        logger.info("All chunks already in DB; exiting")
        return
    
    # 4. Embed
    texts_to_embed = [c.content for c in new_chunks]
    embeddings, _ = text_embedder.embed_batch(texts_to_embed)
    
    # 5. Write to DB
    chunks_to_write = []
    for chunk, embedding in zip(new_chunks, embeddings):
        chunk_dict = {
            'content': chunk.content,
            'content_hash': DocumentChunker.compute_hash(chunk.content),
            'modality': chunk.modality,
            'page': chunk.page,
            'chunk_index': chunk.chunk_index,
            'vector_embedding': embedding,
            'vector_image': None,  # TODO: extract image regions, embed separately
            'is_table': chunk.is_table,
            'table_context': chunk.table_context,
            'alt_text': chunk.alt_text,
            'ocr_confidence': chunk.ocr_confidence
        }
        chunks_to_write.append(chunk_dict)
    
    writer.write_chunks(org_id, chunks_to_write, source_document, knowledge_base_id)
    
    logger.info(f"✓ Ingestion complete: {len(chunks_to_write)} chunks written")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest document into KB")
    parser.add_argument("--org-id", required=True, help="Organization UUID")
    parser.add_argument("--file", required=True, help="File path")
    parser.add_argument("--knowledge-base-id", help="Knowledge base UUID")
    
    args = parser.parse_args()
    
    ingest_document(args.file, args.org_id, args.knowledge_base_id)
```

---

## Retrieval Pipeline

### Overview

**Input**: Query text (+ optional language hint, filters)  
**Output**: Ranked list of top-K chunks with provenance & similarity scores

**Process**: Embed query → BM25 search → ANN search → Fuse (RRF) → Rerank → Return

### 1. Query Embedding

**Location**: `src/voice_agent/retrieval/embedder.py`

```python
import numpy as np
from openai import OpenAI
from transformers import AutoProcessor, AutoModel
import torch

class QueryEmbedder:
    """Embed queries for retrieval."""
    
    def __init__(self, openai_key: str, siglip_model: str = None):
        self.text_embedder = OpenAI(api_key=openai_key)
        if siglip_model:
            self.image_processor = AutoProcessor.from_pretrained(siglip_model)
            self.image_model = AutoModel.from_pretrained(siglip_model).eval()
    
    def embed_text_query(self, query: str) -> np.ndarray:
        """Embed query for text search."""
        response = self.text_embedder.embeddings.create(
            model="text-embedding-3-small",
            input=query,
            encoding_format="float"
        )
        
        embedding = np.array(response.data[0].embedding, dtype=np.float32)
        # L2 normalize
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        return embedding
    
    def embed_text_for_image_search(self, query: str) -> np.ndarray:
        """Embed query text for image-space search (via SigLIP text tower)."""
        with torch.no_grad():
            inputs = self.image_processor(text=query, return_tensors="pt")
            outputs = self.image_model.get_text_features(**inputs)
            embedding = torch.nn.functional.normalize(outputs, p=2, dim=-1)
            return embedding.cpu().numpy()[0].astype(np.float32)
```

### 2. Hybrid Retrieval (BM25 + ANN)

**Location**: `src/voice_agent/retrieval/hybrid_search.py`

```python
import numpy as np
from typing import List, Dict, Tuple
import psycopg2
import logging

logger = logging.getLogger(__name__)

class HybridRetriever:
    """Hybrid search combining BM25 (FTS) + ANN (vector similarity)."""
    
    def __init__(self, db_pool):
        self.db_pool = db_pool
    
    def retrieve_hybrid(self, 
                        org_id: str,
                        query_text: str,
                        query_embedding: np.ndarray,
                        top_k: int = 50,
                        similarity_cutoff: float = 0.3,
                        filters: Dict = None) -> List[Dict]:
        """
        Retrieve chunks via BM25 + ANN; combine results.
        
        filters: dict with keys 'category', 'source_document', 'modality'
        """
        
        # 1. BM25 search (full-text)
        bm25_results = self._bm25_search(
            org_id, query_text, top_k, filters
        )
        logger.info(f"BM25: {len(bm25_results)} results")
        
        # 2. ANN search (vector)
        ann_results = self._ann_search(
            org_id, query_embedding, top_k, similarity_cutoff, filters
        )
        logger.info(f"ANN: {len(ann_results)} results")
        
        # 3. Fuse results (RRF)
        fused = self._rrf_fusion(bm25_results, ann_results)
        
        return fused[:top_k]
    
    def _bm25_search(self, org_id: str, query_text: str, top_k: int,
                     filters: Dict) -> List[Dict]:
        """Full-text search via PostgreSQL FTS."""
        
        with self.db_pool.connection() as conn:
            cursor = conn.cursor()
            
            query = """
                SELECT 
                    id, content, source_document, page, chunk_index, 
                    modality, is_table, table_context, alt_text, 
                    ocr_confidence,
                    ts_rank(to_tsvector('portuguese', content), 
                            plainto_tsquery('portuguese', %s)) as rank
                FROM knowledge_base_chunks
                WHERE org_id = %s
                  AND to_tsvector('portuguese', content) @@ plainto_tsquery('portuguese', %s)
            """
            params = [query_text, org_id, query_text]
            
            # Add optional filters
            if filters:
                if 'category' in filters:
                    query += " AND category = %s"
                    params.append(filters['category'])
                if 'source_document' in filters:
                    query += " AND source_document = %s"
                    params.append(filters['source_document'])
                if 'modality' in filters:
                    query += " AND modality = %s"
                    params.append(filters['modality'])
            
            query += " ORDER BY rank DESC LIMIT %s"
            params.append(top_k)
            
            cursor.execute(query, params)
            results = []
            for row in cursor.fetchall():
                results.append({
                    'id': row[0],
                    'content': row[1],
                    'source_document': row[2],
                    'page': row[3],
                    'chunk_index': row[4],
                    'modality': row[5],
                    'is_table': row[6],
                    'table_context': row[7],
                    'alt_text': row[8],
                    'ocr_confidence': row[9],
                    'score': float(row[10]),
                    'method': 'bm25'
                })
            
            return results
    
    def _ann_search(self, org_id: str, query_embedding: np.ndarray, top_k: int,
                    similarity_cutoff: float, filters: Dict) -> List[Dict]:
        """Vector similarity search via pgvector ANN."""
        
        with self.db_pool.connection() as conn:
            cursor = conn.cursor()
            
            query = """
                SELECT 
                    id, content, source_document, page, chunk_index,
                    modality, is_table, table_context, alt_text,
                    ocr_confidence,
                    1 - (vector_embedding <=> %s) as cosine_similarity
                FROM knowledge_base_chunks
                WHERE org_id = %s
                  AND vector_embedding IS NOT NULL
                  AND 1 - (vector_embedding <=> %s) > %s
            """
            params = [query_embedding.tobytes(), org_id, query_embedding.tobytes(), similarity_cutoff]
            
            # Add filters
            if filters:
                if 'category' in filters:
                    query += " AND category = %s"
                    params.append(filters['category'])
                if 'source_document' in filters:
                    query += " AND source_document = %s"
                    params.append(filters['source_document'])
                if 'modality' in filters:
                    query += " AND modality = %s"
                    params.append(filters['modality'])
            
            query += " ORDER BY cosine_similarity DESC LIMIT %s"
            params.append(top_k)
            
            cursor.execute(query, params)
            results = []
            for row in cursor.fetchall():
                results.append({
                    'id': row[0],
                    'content': row[1],
                    'source_document': row[2],
                    'page': row[3],
                    'chunk_index': row[4],
                    'modality': row[5],
                    'is_table': row[6],
                    'table_context': row[7],
                    'alt_text': row[8],
                    'ocr_confidence': row[9],
                    'score': float(row[10]),
                    'method': 'ann'
                })
            
            return results
    
    def _rrf_fusion(self, bm25_results: List[Dict], 
                    ann_results: List[Dict]) -> List[Dict]:
        """Combine results via Reciprocal Rank Fusion."""
        
        # Assign ranks
        rrf_scores = {}
        
        for rank, result in enumerate(bm25_results):
            chunk_id = result['id']
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, {})
            rrf_scores[chunk_id]['rrf_score'] = 1 / (60 + rank)  # Standard RRF formula
            rrf_scores[chunk_id]['result'] = result
        
        for rank, result in enumerate(ann_results):
            chunk_id = result['id']
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, {})
            rrf_scores[chunk_id]['rrf_score'] = rrf_scores[chunk_id].get('rrf_score', 0) + 1 / (60 + rank)
            rrf_scores[chunk_id]['result'] = result
        
        # Sort by RRF score
        fused = sorted(rrf_scores.values(), key=lambda x: x['rrf_score'], reverse=True)
        return [item['result'] for item in fused]
```

### 3. Reranking (Optional Cross-Encoder)

**Location**: `src/voice_agent/retrieval/reranker.py`

```python
from sentence_transformers import CrossEncoder
import numpy as np
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class CrossEncoderReranker:
    """Rerank results using cross-encoder model."""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 batch_size: int = 32):
        self.model = CrossEncoder(model_name)
        self.batch_size = batch_size
    
    def rerank(self, query: str, chunks: List[Dict], top_k: int = 5) -> List[Dict]:
        """Rerank chunks; return top-K."""
        
        if not chunks:
            return []
        
        # Prepare query-chunk pairs
        pairs = [[query, chunk['content']] for chunk in chunks]
        
        # Score all pairs
        scores = self.model.predict(pairs, batch_size=self.batch_size, 
                                    show_progress_bar=False)
        
        # Add scores to chunks
        for chunk, score in zip(chunks, scores):
            chunk['rerank_score'] = float(score)
        
        # Sort by rerank score and return top-K
        reranked = sorted(chunks, key=lambda x: x['rerank_score'], reverse=True)
        logger.info(f"Reranked {len(chunks)} chunks; top={reranked[0]['rerank_score']:.3f}")
        
        return reranked[:top_k]
```

### 4. Retrieval API

**Location**: `src/voice_agent/tools/crm_knowledge.py` (updated)

```python
from src.voice_agent.config import rag_config
from src.voice_agent.retrieval.embedder import QueryEmbedder
from src.voice_agent.retrieval.hybrid_search import HybridRetriever
from src.voice_agent.retrieval.reranker import CrossEncoderReranker
from src.db import get_db_pool

class KnowledgeSearchTool:
    """Search CRM knowledge base."""
    
    def __init__(self):
        self.db_pool = get_db_pool()
        self.query_embedder = QueryEmbedder(rag_config.embedding.openai_key)
        self.retriever = HybridRetriever(self.db_pool)
        self.reranker = CrossEncoderReranker(
            rag_config.retrieval.reranking_model
        ) if rag_config.retrieval.reranking_enabled else None
    
    def search(self, 
               org_id: str,
               query: str,
               top_k: int = None,
               category: str = None,
               source_document: str = None,
               modality: str = None) -> List[Dict]:
        """
        Search knowledge base.
        
        Returns:
            List of chunks with keys: content, source_document, page, chunk_index,
            modality, ocr_confidence, is_table, table_context, similarity_score,
            rerank_score (if enabled)
        """
        
        top_k = top_k or rag_config.retrieval.top_k_rerank
        
        # 1. Embed query
        query_embedding = self.query_embedder.embed_text_query(query)
        
        # 2. Hybrid retrieve
        filters = {
            k: v for k, v in {'category': category, 'source_document': source_document,
                             'modality': modality}.items() if v is not None
        }
        
        results = self.retriever.retrieve_hybrid(
            org_id,
            query,
            query_embedding,
            top_k=rag_config.retrieval.top_k_retrieve,
            similarity_cutoff=rag_config.retrieval.similarity_cutoff,
            filters=filters
        )
        
        # 3. Rerank (optional)
        if self.reranker:
            results = self.reranker.rerank(query, results, top_k)
        else:
            results = results[:top_k]
        
        # 4. Format response
        return [
            {
                'content': r['content'],
                'source_document': r['source_document'],
                'page': r['page'],
                'chunk_index': r['chunk_index'],
                'modality': r['modality'],
                'is_table': r['is_table'],
                'table_context': r['table_context'],
                'alt_text': r['alt_text'],
                'ocr_confidence': r['ocr_confidence'],
                'similarity_score': r.get('score'),
                'rerank_score': r.get('rerank_score'),
            }
            for r in results
        ]
```

---

## Quality & Evaluation

### Evaluation Dataset

**Location**: `data/eval/pt_kb_eval.json`

**Format**: 200 queries (mix of English & Portuguese) with relevance labels

```json
[
  {
    "query": "Como calcular as margens de lucro em uma venda?",
    "language": "pt",
    "category": "sales",
    "relevant_chunks": [
      {
        "source_document": "sales_guide.pdf",
        "page": 5,
        "content_snippet": "Margem de lucro = (Preço de venda - Custo) / Preço de venda",
        "relevance": 2  // 0: not relevant, 1: relevant, 2: highly relevant
      }
    ]
  },
  {
    "query": "What is the process for customer onboarding?",
    "language": "en",
    "category": "operations",
    "relevant_chunks": [...]
  }
]
```

### Metrics

**Location**: `src/voice_agent/eval/metrics.py`

```python
import numpy as np
from typing import List, Dict

def ndcg_at_k(relevance_scores: List[float], k: int = 10) -> float:
    """Normalized Discounted Cumulative Gain."""
    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance_scores[:k]))
    ideal_rel = sorted(relevance_scores, reverse=True)[:k]
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_rel))
    return dcg / idcg if idcg > 0 else 0.0

def recall_at_k(retrieved: set, relevant: set, k: int = 10) -> float:
    """Recall: fraction of relevant docs retrieved in top-K."""
    if not relevant:
        return 1.0
    return len(retrieved & relevant) / len(relevant)

def mrr(ranks: List[int]) -> float:
    """Mean Reciprocal Rank."""
    return np.mean([1 / r if r > 0 else 0 for r in ranks])

def precision_recall_f1(retrieved: set, relevant: set) -> Dict[str, float]:
    """Precision, recall, F1."""
    tp = len(retrieved & relevant)
    precision = tp / len(retrieved) if retrieved else 0
    recall = tp / len(relevant) if relevant else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return {'precision': precision, 'recall': recall, 'f1': f1}
```

### Evaluation Script

**Location**: `scripts/eval_rag.py`

```python
#!/usr/bin/env python3
"""Evaluate RAG quality on PT/EN eval set."""

import json
import logging
from src.voice_agent.tools.crm_knowledge import KnowledgeSearchTool
from src.voice_agent.eval.metrics import ndcg_at_k, recall_at_k, mrr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate():
    # Load eval set
    with open('data/eval/pt_kb_eval.json') as f:
        eval_data = json.load(f)
    
    search_tool = KnowledgeSearchTool()
    
    ndcg_scores = []
    recall_scores = []
    mrr_scores = []
    
    for item in eval_data:
        query = item['query']
        relevant_ids = set((r['source_document'], r['page']) for r in item['relevant_chunks'])
        
        # Retrieve
        results = search_tool.search(org_id="default", query=query)
        retrieved_ids = set((r['source_document'], r['page']) for r in results[:10])
        
        # Score
        relevance = [1 if (r['source_document'], r['page']) in relevant_ids else 0 
                     for r in results[:10]]
        
        ndcg = ndcg_at_k(relevance, k=10)
        recall = recall_at_k(retrieved_ids, relevant_ids, k=10)
        rank = next((i+1 for i, rel in enumerate(relevance) if rel), 0)
        mrr_score = 1 / rank if rank > 0 else 0
        
        ndcg_scores.append(ndcg)
        recall_scores.append(recall)
        mrr_scores.append(mrr_score)
    
    logger.info(f"NDCG@10:  {np.mean(ndcg_scores):.3f}")
    logger.info(f"Recall@10: {np.mean(recall_scores):.3f}")
    logger.info(f"MRR:       {np.mean(mrr_scores):.3f}")
```

---

## Observability & Monitoring

### Metrics Collection

**Location**: `src/voice_agent/monitoring/metrics.py`

```python
from prometheus_client import Counter, Histogram, Gauge
import time

# Counters
ingestion_docs_total = Counter('rag_ingestion_docs_total', 'Total docs ingested')
ingestion_chunks_total = Counter('rag_ingestion_chunks_total', 'Total chunks ingested')
ingestion_chunks_dedup = Counter('rag_ingestion_chunks_dedup', 'Chunks deduplicated')
ocr_failures = Counter('rag_ocr_failures', 'OCR failures')
ocr_retries = Counter('rag_ocr_retries', 'OCR retries (low confidence)')

# Histograms
ingest_duration_seconds = Histogram('rag_ingest_duration_seconds', 'Ingest duration', buckets=[10, 30, 60, 120, 300])
ocr_confidence = Histogram('rag_ocr_confidence', 'OCR confidence score', buckets=[0.3, 0.5, 0.7, 0.85, 0.95])
retrieval_latency = Histogram('rag_retrieval_latency_ms', 'Retrieval latency', buckets=[50, 100, 200, 500])
embedding_latency = Histogram('rag_embedding_latency_ms', 'Embedding latency')

# Gauges
index_size_bytes = Gauge('rag_index_size_bytes', 'Total index size')
```

### Logging Strategy

```python
import logging

logger = logging.getLogger(__name__)

# Structured logging (JSON)
logger.info("ingestion_completed", extra={
    'org_id': org_id,
    'source_document': source_document,
    'chunks_written': len(chunks),
    'duration_seconds': elapsed,
    'dedup_count': dedup_count
})

logger.warning("ocr_low_confidence", extra={
    'page': page_num,
    'confidence': ocr_conf,
    'threshold': threshold,
    'retry_mode': 'large'
})
```

### Alerting

**Prometheus alerting rules** (`monitoring/alerts.yml`):

```yaml
groups:
  - name: rag
    rules:
      - alert: OCRFailureSpike
        expr: rate(rag_ocr_failures[5m]) > 0.1
        for: 5m
        annotations:
          summary: "OCR failure rate spike"
      
      - alert: NDCGRegression
        expr: rag_eval_ndcg < 0.65
        for: 1h
        annotations:
          summary: "NDCG score dropped below threshold"
      
      - alert: RetrievalLatencyHigh
        expr: histogram_quantile(0.95, rag_retrieval_latency_ms) > 500
        for: 5m
        annotations:
          summary: "Retrieval p95 latency >500ms"
```

---

## Performance & Cost Guardrails

### Ingestion Tuning

**Memory**: Max `MAX_WORKERS * BATCH_SIZE_IMAGE` images in GPU memory

```bash
# For 6GB GPU:
BATCH_SIZE_IMAGE=16
MAX_WORKERS_INGEST=4
# = 64 images max in flight
```

**Cost**: text embeddings ~$2–3 per 1M chunks

```python
# Estimate:
chunks_per_1M_chars = 1_000_000 / 500  # avg chunk size
cost_per_1M_chars = (chunks_per_1M_chars / 1_000_000) * 0.02  # OpenAI pricing
# ~$0.00004 per 1M chars
```

### Retrieval Optimization

**Limits**:
- `TOP_K_RETRIEVE=50`: Initial pool from BM25 + ANN
- `TOP_K_RERANK=5–10`: Final results
- `REQUEST_TIMEOUT_RETRIEVAL=10s`: Hard stop

**Query caching**:

```python
@functools.lru_cache(maxsize=1000)
def cached_retrieve(query_hash, org_id):
    # Implement with Redis for multi-process
    pass
```

**Index maintenance**:

```sql
-- Weekly VACUUM/ANALYZE
VACUUM ANALYZE knowledge_base_chunks;

-- Monitor index size
SELECT pg_size_pretty(pg_relation_size('idx_kb_chunks_vector_text'));
```

---

## Implementation Phases

### Phase 0: Pre-Flight (Days 1–2)

- [ ] Backup existing DB, snapshot current `knowledge_base` embeddings
- [ ] Confirm env vars: OPENAI_API_KEY, DeepSeek endpoint, storage
- [ ] Identify current search API entrypoint
- [ ] Decide migration strategy (dual-write vs. in-place)

### Phase 1: Schema & Configuration (Days 2–3)

- [ ] Create migration `database/migrations/<timestamp>_kb_chunks.sql`
- [ ] Deploy migration to staging
- [ ] Update `.env` template with all RAG config vars
- [ ] Implement `RAGConfig` class with startup validation
- [ ] Test schema, indexes, RLS

### Phase 2: Ingestion Pipeline (Days 3–5)

- [ ] Implement document loader (PDF, DOCX, images, CSV/XLSX)
- [ ] Implement OCR client (DeepSeek)
- [ ] Implement chunking with table preservation
- [ ] Implement deduplication
- [ ] Implement text embedder (OpenAI batch)
- [ ] Implement image embedder (SigLIP)
- [ ] Implement writer to DB
- [ ] Create CLI ingestion script
- [ ] Test with sample documents

### Phase 3: Retrieval Pipeline (Days 5–7)

- [ ] Implement query embedder
- [ ] Implement hybrid retriever (BM25 + ANN)
- [ ] Implement RRF fusion
- [ ] Implement cross-encoder reranker
- [ ] Update search tool API
- [ ] Add unit tests (hybrid on/off, rerank on/off)

### Phase 4: Backfill & Integration (Days 7–8)

- [ ] Export existing `knowledge_base` rows
- [ ] Generate chunks and embeddings for all
- [ ] Load into `knowledge_base_chunks`
- [ ] Build vector indexes
- [ ] Shadow-read: compare old vs. new retrieval (10% traffic)
- [ ] Smoke tests: fixed queries assert non-empty results

### Phase 5: Quality & Evaluation (Days 8–9)

- [ ] Create PT/EN eval set (50–200 queries)
- [ ] Implement NDCG, Recall, MRR metrics
- [ ] Run baseline eval on new system
- [ ] Tune top-K, reranking params
- [ ] CI smoke tests integrated

### Phase 6–10: Observability, Rollout, Monitoring (Days 9–14)

- [ ] Implement Prometheus metrics & logging
- [ ] Deploy alerting rules
- [ ] Gradual traffic shift (10% → 50% → 100%)
- [ ] Keep rollback for 48h
- [ ] Monitor NDCG, latency, OCR confidence
- [ ] Document architecture & runbook

---

## Code Touchpoints & Deliverables

### Database

**File**: `database/migrations/2026_01_13_create_knowledge_base_chunks.sql`

- [ ] Table creation with all columns
- [ ] Indexes (B-tree, GIN, IVFFlat)
- [ ] RLS policies
- [ ] Grants

### Configuration

**File**: `src/voice_agent/config.py`

- [ ] `EmbeddingConfig`, `OCRConfig`, `ChunkingConfig`, `RetrievalConfig`, `PerformanceConfig`
- [ ] `RAGConfig` factory with startup validation

**File**: `.env.example`

- [ ] All RAG env vars documented

### Ingestion

**Files**:
- `src/voice_agent/ingestion/loader.py` — Document loading
- `src/voice_agent/ingestion/ocr.py` — DeepSeek OCR client
- `src/voice_agent/ingestion/chunking.py` — Chunking logic
- `src/voice_agent/ingestion/dedup.py` — Deduplication
- `src/voice_agent/ingestion/embedder.py` — Text & image embedding
- `src/voice_agent/ingestion/writer.py` — DB writer
- `scripts/ingest_kb.py` — CLI entry point

### Retrieval

**Files**:
- `src/voice_agent/retrieval/embedder.py` — Query embedding
- `src/voice_agent/retrieval/hybrid_search.py` — BM25 + ANN fusion
- `src/voice_agent/retrieval/reranker.py` — Cross-encoder reranking
- `src/voice_agent/tools/crm_knowledge.py` — Updated search tool (read from chunks)

### Testing

**Files**:
- `tests/ingestion/test_loader.py` — Document loading
- `tests/ingestion/test_chunking.py` — Chunking, table preservation
- `tests/ingestion/test_dedup.py` — Deduplication
- `tests/retrieval/test_hybrid.py` — Hybrid retrieval on/off
- `tests/retrieval/test_rerank.py` — Reranking on/off
- `tests/migration/test_schema.py` — Migration smoke test

### Evaluation

**Files**:
- `src/voice_agent/eval/metrics.py` — NDCG, Recall, MRR
- `scripts/eval_rag.py` — Eval runner
- `data/eval/pt_kb_eval.json` — 50–200 PT/EN queries with labels

### Observability

**Files**:
- `src/voice_agent/monitoring/metrics.py` — Prometheus metrics
- `monitoring/alerts.yml` — Alert rules
- `docs/runbook_rag.md` — Operations runbook

### Documentation

**File**: `docs/artifacts/RAG/voice_agent_rag_implementation.md`

- [ ] Architecture diagram
- [ ] Configuration reference
- [ ] Ingestion troubleshooting
- [ ] Retrieval tuning guide
- [ ] Cost/performance tradeoffs

---

## Appendix: Dependencies & References

### Python Packages

```
# Core
openai>=1.0.0
psycopg2-binary>=2.9
pgvector>=0.2.0
numpy>=1.24
pandas>=2.0

# Models & embeddings
transformers>=4.40.0
torch>=2.0.0
sentence-transformers>=2.2.0
Pillow>=10.0

# Document processing
PyPDF2>=3.0
python-docx>=0.8.11
pdf2image>=1.16
openpyxl>=3.1

# Database & ORM
sqlalchemy>=2.0
alembic>=1.13

# Monitoring
prometheus-client>=0.19

# Testing
pytest>=7.0
pytest-cov>=4.0
```

### External Services

- **OpenAI API** — text-embedding-3-small
  - Docs: https://platform.openai.com/docs/guides/embeddings
  - Cost: $0.02/1M tokens

- **DeepSeek OCR API** — OCR service
  - Docs: https://api.deepseek.com/docs
  - Fallback to Large mode on low confidence

- **Hugging Face** — SigLIP model
  - Model: google/siglip-base-patch16-384
  - Docs: https://huggingface.co/google/siglip-base-patch16-384

### PostgreSQL Extensions

- `pgvector` — Vector similarity search
  - Docs: https://github.com/pgvector/pgvector

### References

**Vector Search:**
- Approximate Nearest Neighbor (ANN) indexing: https://arxiv.org/abs/2104.14550
- Pgvector IVFFLAT vs HNSW: https://github.com/pgvector/pgvector#ivfflat

**RAG:**
- "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks": https://arxiv.org/abs/2005.11401
- Hybrid search (BM25 + dense): https://www.elastic.co/guide/en/elasticsearch/reference/current/hybrid-search.html
- RRF (Reciprocal Rank Fusion): https://en.wikipedia.org/wiki/Reciprocal_rank_fusion

**Portuguese NLP:**
- spaCy Portuguese models: https://spacy.io
- PostgreSQL `portuguese` text search config: https://www.postgresql.org/docs/current/textsearch-dictionaries.html

**Evaluation:**
- RAGAS framework: https://github.com/explodinggradients/ragas
- Ranking metrics (NDCG, MRR): https://en.wikipedia.org/wiki/Discounted_cumulative_gain

---

## Rollback & Disaster Recovery

### Rollback Plan

**If issues detected post-deployment:**

1. **Stop shadow traffic** to new retrieval; switch to legacy 100%
2. **Disable `knowledge_base_chunks`-based queries**; re-enable old vector search on `knowledge_base`
3. **Check logs** for OCR failures, embedding errors, index build issues
4. **Keep `knowledge_base_chunks` table** (don't drop) for forensic analysis
5. **Run legacy validation** on restored queries

**Estimated rollback time**: <5 minutes

### Keeping Legacy Path Warm

During cutover window (24–48h):
- Query both old (`knowledge_base`) and new (`knowledge_base_chunks`) in parallel
- Compare result quality
- Log discrepancies for post-mortem

---

## Final Checklist Before Deployment

- [ ] Schema migration tested on staging DB
- [ ] All config vars set in `.env.staging` and `.env.prod`
- [ ] Ingestion script tested on sample documents (PDF, DOCX, images)
- [ ] Retrieval e2e tested: query → embed → search → rerank
- [ ] Eval script runs; NDCG/Recall scores logged
- [ ] Monitoring & alerting deployed
- [ ] RLS policies verified for org isolation
- [ ] Backup taken; snapshot documented
- [ ] Team briefed on runbook & rollback procedure
- [ ] CI tests green across all phases

---

**Document Version**: 1.0  
**Last Updated**: January 13, 2026  
**Owner**: AI/ML Engineering  
**Status**: Ready for Implementation
