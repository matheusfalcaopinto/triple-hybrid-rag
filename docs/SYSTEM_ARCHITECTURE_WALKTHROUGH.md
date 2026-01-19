# Triple-Hybrid-RAG: Complete System Architecture Walkthrough

> **A comprehensive technical deep-dive into every component, algorithm, and data flow**

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Overview](#system-overview)
3. [Ingestion Pipeline](#ingestion-pipeline)
   - [Document Loading](#1-document-loading)
   - [Hierarchical Chunking](#2-hierarchical-chunking)
   - [Semantic Chunking](#3-semantic-chunking-phase-1-enhancement)
   - [Embedding Generation](#4-embedding-generation)
   - [Entity Extraction](#5-entity-extraction-nerre)
   - [Database Storage](#6-database-storage)
4. [Retrieval Pipeline](#retrieval-pipeline)
   - [Query Processing](#1-query-processing)
   - [HyDE Generation](#2-hyde-generation-phase-2)
   - [Query Expansion](#3-query-expansion-phase-2)
   - [Triple-Hybrid Search](#4-triple-hybrid-search)
   - [RRF Fusion](#5-reciprocal-rank-fusion-rrf)
   - [Multi-Stage Reranking](#6-multi-stage-reranking-phase-2)
   - [Diversity Optimization](#7-diversity-optimization-phase-2)
   - [Context Compression](#8-context-compression-phase-3)
5. [Advanced RAG Strategies](#advanced-rag-strategies)
   - [Self-RAG](#self-rag-phase-5)
   - [Corrective RAG](#corrective-rag-phase-5)
   - [Agentic RAG](#agentic-rag-phase-6)
6. [Infrastructure Components](#infrastructure-components)
   - [PostgreSQL + pgvector](#postgresql--pgvector)
   - [PuppyGraph](#puppygraph)
   - [Embedding Service (vLLM)](#embedding-service-vllm)
   - [Reranking Service](#reranking-service)
7. [Data Models](#data-models)
8. [Configuration System](#configuration-system)
9. [Performance Characteristics](#performance-characteristics)
10. [Appendix: Algorithm Details](#appendix-algorithm-details)

---

## Executive Summary

Triple-Hybrid-RAG is a state-of-the-art Retrieval-Augmented Generation system that combines three complementary search paradigms:

| Search Type | Technology | Strength |
|-------------|------------|----------|
| **Lexical** | PostgreSQL FTS (BM25) | Exact keyword matching, rare terms |
| **Semantic** | pgvector (HNSW) | Conceptual similarity, paraphrases |
| **Graph** | PuppyGraph (Cypher) | Entity relationships, multi-hop reasoning |

The system implements cutting-edge RAG techniques from recent research:

- **HyDE** (Hypothetical Document Embeddings) - Gao et al., 2022
- **Self-RAG** (Self-Reflective RAG) - Asai et al., 2023
- **RAG-Fusion** (Multi-Query) - Raudaschl, 2023
- **CRAG** (Corrective RAG) - Yan et al., 2024

**Key Metrics:**
- Ingestion: 2,700+ chunks/second
- Embedding: 55+ texts/second
- Retrieval: <100ms P95 latency
- Test Coverage: 377 tests passing

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              TRIPLE-HYBRID-RAG SYSTEM                                │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │                           INGESTION PIPELINE                                  │   │
│  │                                                                               │   │
│  │   Documents → Loaders → Chunking → Embedding → NER → Database Storage        │   │
│  │      │          │          │          │         │           │                │   │
│  │   [PDF,TXT]   [OCR]    [Parent/    [1024d    [GPT-5]   [PostgreSQL]          │   │
│  │   [DOCX]     [Parse]   Child]     Vector]   [Entities]  [pgvector]           │   │
│  │                        [Semantic]           [Relations] [PuppyGraph]          │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │                           RETRIEVAL PIPELINE                                  │   │
│  │                                                                               │   │
│  │   Query → HyDE → Expansion → Search → Fusion → Rerank → Diversity → Results  │   │
│  │     │       │         │         │        │        │          │          │     │   │
│  │   [User]  [LLM]    [Multi-   [Triple-  [RRF]   [Cross-   [MMR]      [Top-K]   │   │
│  │   [Query] [Hypo]   Query]    Hybrid]          Encoder]  [Source]             │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │                         ADVANCED RAG STRATEGIES                               │   │
│  │                                                                               │   │
│  │   Self-RAG (Reflective) │ Corrective RAG (Refinement) │ Agentic RAG (Tools)  │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Ingestion Pipeline

The ingestion pipeline transforms raw documents into searchable chunks with embeddings and knowledge graph relationships.

### 1. Document Loading

**File:** `src/triple_hybrid_rag/ingestion/loaders.py`

**Purpose:** Load various document formats into a unified text representation.

**Supported Formats:**
- PDF (with OCR fallback for scanned documents)
- DOCX (Microsoft Word)
- TXT, MD (Plain text, Markdown)
- HTML (Web pages)
- Images (via OCR)

**Process Flow:**

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ Raw Document │ ──▶ │    Loader    │ ──▶ │  Structured  │
│  (PDF/DOCX)  │     │  (Parser)    │     │    Output    │
└──────────────┘     └──────────────┘     └──────────────┘
                            │
                     ┌──────┴──────┐
                     ▼             ▼
              ┌──────────┐  ┌──────────┐
              │ Text     │  │ Images   │
              │ Content  │  │ (OCR)    │
              └──────────┘  └──────────┘
```

**Data Output:**
```python
@dataclass
class LoadedDocument:
    content: str              # Extracted text
    metadata: Dict[str, Any]  # File metadata
    pages: List[str]          # Per-page content
    images: List[bytes]       # Extracted images
    file_path: str            # Source path
    file_type: str            # MIME type
```

### 2. Hierarchical Chunking

**File:** `src/triple_hybrid_rag/core/chunker.py`

**Purpose:** Split documents into a two-level hierarchy optimized for both retrieval (small chunks) and context (large chunks).

**Why Hierarchical?**
- **Problem:** Small chunks are great for precise retrieval but lose context
- **Solution:** Index small chunks, return parent chunks for LLM context

**Chunk Hierarchy:**

```
┌─────────────────────────────────────────────────────────────────┐
│                     PARENT CHUNK (800-1000 tokens)              │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ Full context for LLM generation                          │   │
│   │ Stored in: rag_parent_chunks                             │   │
│   │ NOT indexed for retrieval                                │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│   │ CHILD CHUNK  │  │ CHILD CHUNK  │  │ CHILD CHUNK  │         │
│   │ (~200 tokens)│  │ (~200 tokens)│  │ (~200 tokens)│         │
│   │ Embedded ✓   │  │ Embedded ✓   │  │ Embedded ✓   │         │
│   │ Indexed ✓    │  │ Indexed ✓    │  │ Indexed ✓    │         │
│   └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

**Configuration:**
```python
@dataclass
class ChunkerConfig:
    parent_chunk_size: int = 800    # Tokens per parent
    parent_chunk_overlap: int = 100  # Token overlap between parents
    child_chunk_size: int = 200      # Tokens per child
    child_chunk_overlap: int = 50    # Token overlap between children
    
    # Semantic boundaries (NLP-aware)
    respect_sentence_boundaries: bool = True
    respect_paragraph_boundaries: bool = True
```

**Algorithm:**

```python
def split_document(text: str, document_id: UUID) -> Tuple[List[ParentChunk], List[ChildChunk]]:
    # Step 1: Split into parent chunks with overlap
    parent_texts = split_with_overlap(
        text, 
        size=parent_chunk_size, 
        overlap=parent_chunk_overlap,
        respect_boundaries=True
    )
    
    parent_chunks = []
    child_chunks = []
    
    for idx, parent_text in enumerate(parent_texts):
        # Create parent chunk
        parent = ParentChunk(
            id=uuid4(),
            document_id=document_id,
            index_in_document=idx,
            text=parent_text,
            page_start=calculate_page(idx),
        )
        parent_chunks.append(parent)
        
        # Step 2: Split parent into children
        child_texts = split_with_overlap(
            parent_text,
            size=child_chunk_size,
            overlap=child_chunk_overlap,
            respect_boundaries=True
        )
        
        for child_idx, child_text in enumerate(child_texts):
            child = ChildChunk(
                id=uuid4(),
                parent_id=parent.id,
                document_id=document_id,
                index_in_parent=child_idx,
                text=child_text,
                content_hash=sha256(child_text),
            )
            child_chunks.append(child)
    
    return parent_chunks, child_chunks
```

**Token Counting:**
- Uses `tiktoken` with `cl100k_base` encoding (GPT-4 tokenizer)
- Handles multi-byte Unicode correctly
- Counts are consistent with OpenAI API limits

### 3. Semantic Chunking (Phase 1 Enhancement)

**File:** `src/triple_hybrid_rag/ingestion/semantic_chunker.py`

**Purpose:** Split documents at natural semantic boundaries rather than fixed token counts.

**Why Semantic Chunking?**
- Fixed-size chunks can split mid-sentence or mid-concept
- Semantic chunks preserve complete ideas
- Better retrieval quality for complex topics

**Algorithm: Sliding Window Similarity**

```
Document: "Machine learning is a subset of AI. It enables computers to learn.
           Neural networks are a type of ML. They mimic brain structure."

Step 1: Compute sentence embeddings
   S1: "Machine learning is a subset of AI" → [0.2, 0.8, ...]
   S2: "It enables computers to learn"      → [0.3, 0.7, ...]
   S3: "Neural networks are a type of ML"   → [0.5, 0.6, ...]
   S4: "They mimic brain structure"         → [0.4, 0.5, ...]

Step 2: Compute cosine similarity between adjacent sentences
   sim(S1, S2) = 0.92  (high - same topic)
   sim(S2, S3) = 0.71  (medium - topic shift)
   sim(S3, S4) = 0.89  (high - same topic)

Step 3: Find breakpoints where similarity drops below threshold
   Threshold = 0.75
   Breakpoint after S2 (similarity = 0.71 < 0.75)

Step 4: Create chunks at breakpoints
   Chunk 1: S1 + S2 → "Machine learning is... computers to learn."
   Chunk 2: S3 + S4 → "Neural networks are... brain structure."
```

**Configuration:**
```python
@dataclass
class SemanticChunkerConfig:
    similarity_threshold: float = 0.75  # Breakpoint threshold
    min_chunk_size: int = 100           # Minimum tokens per chunk
    max_chunk_size: int = 500           # Maximum tokens per chunk
    buffer_size: int = 3                # Sentences to look ahead/behind
```

### 4. Embedding Generation

**File:** `src/triple_hybrid_rag/core/embedder.py`

**Purpose:** Convert text chunks into dense vector representations for semantic search.

**Model:** Qwen3-VL-Embedding-2B
- Native dimension: 2048
- Matryoshka dimension: 1024 (used for storage efficiency)
- Supports text, images, and mixed content

**Architecture:**

```
┌─────────────────────────────────────────────────────────────────┐
│                    MULTIMODAL EMBEDDER                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Input Types:                                                   │
│   ┌──────────┐  ┌──────────┐  ┌──────────────┐                 │
│   │   Text   │  │  Image   │  │ Text + Image │                 │
│   └────┬─────┘  └────┬─────┘  └──────┬───────┘                 │
│        │             │               │                          │
│        ▼             ▼               ▼                          │
│   ┌─────────────────────────────────────────────┐              │
│   │        Qwen3-VL-Embedding-2B (vLLM)         │              │
│   │                                              │              │
│   │   Text → Token Embeddings → Mean Pooling    │              │
│   │   Image → Patch Embeddings → Projection     │              │
│   │   Mixed → Late Fusion                       │              │
│   └─────────────────────────────────────────────┘              │
│                          │                                      │
│                          ▼                                      │
│   ┌─────────────────────────────────────────────┐              │
│   │       Matryoshka Dimension Reduction        │              │
│   │              2048 → 1024                     │              │
│   └─────────────────────────────────────────────┘              │
│                          │                                      │
│                          ▼                                      │
│                   [1024-dim vector]                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Matryoshka Embeddings:**

Matryoshka (nesting doll) embeddings allow using a prefix of the full embedding while preserving semantic quality:

```
Full embedding:      [d1, d2, d3, ..., d1024, ..., d2048]
                      ↓
1024-dim prefix:     [d1, d2, d3, ..., d1024]
                      ↓
512-dim prefix:      [d1, d2, d3, ..., d512]
```

- 1024d retains ~98% of full quality
- 50% storage reduction
- Faster similarity computation

**Concurrent Batching:**

```python
async def embed_texts_concurrent(
    texts: List[str],
    batch_size: int = 32,
    max_concurrent: int = 4,
) -> List[List[float]]:
    """
    Process embeddings concurrently for high throughput.
    
    Example: 1000 texts, batch_size=32, max_concurrent=4
    - Creates 32 batches (1000 / 32 ≈ 32)
    - Processes 4 batches simultaneously
    - Total API calls: 32 (not 1000!)
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def embed_batch(batch: List[str]) -> List[List[float]]:
        async with semaphore:
            response = await client.post(
                f"{api_base}/embeddings",
                json={"input": batch, "model": model_name}
            )
            return [e["embedding"][:1024] for e in response["data"]]
    
    # Create batches
    batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]
    
    # Process concurrently
    results = await asyncio.gather(*[embed_batch(b) for b in batches])
    
    return [emb for batch_result in results for emb in batch_result]
```

**Performance:**
- Single text: ~20ms
- Batch of 32: ~150ms (4.7ms/text)
- Concurrent (4 batches): ~200ms for 128 texts (1.6ms/text)

### 5. Entity Extraction (NER/RE)

**File:** `src/triple_hybrid_rag/core/entity_extractor.py`

**Purpose:** Extract named entities and relationships for the knowledge graph.

**Extraction Types:**

| Entity Type | Examples |
|-------------|----------|
| PERSON | "Dr. Sarah Chen", "John Smith" |
| ORGANIZATION | "Acme Corp", "Google Research" |
| PRODUCT | "Triple-Hybrid RAG", "PostgreSQL" |
| MONETARY | "$50,000", "10 million" |
| DATE | "2026", "Q1 2025" |
| CLAUSE | Legal/contractual terms |
| TECHNICAL | APIs, algorithms, technologies |

**Relationship Types:**

| Relation | Example |
|----------|---------|
| WORKS_FOR | (Person) → WORKS_FOR → (Organization) |
| CREATED | (Person) → CREATED → (Product) |
| COSTS | (Product) → COSTS → (Monetary) |
| LOCATED_IN | (Organization) → LOCATED_IN → (Location) |
| PART_OF | (Component) → PART_OF → (System) |

**Extraction Process:**

```
Input Text: "Dr. Sarah Chen leads the RAG project at Acme Corp. 
             The system costs $50,000/month."

                    ▼
         ┌─────────────────────┐
         │    GPT-5 NER/RE     │
         │    Structured JSON  │
         └─────────────────────┘
                    ▼
Output:
{
  "entities": [
    {"name": "Dr. Sarah Chen", "type": "PERSON", "canonical": "sarah_chen"},
    {"name": "RAG project", "type": "PRODUCT", "canonical": "rag_project"},
    {"name": "Acme Corp", "type": "ORGANIZATION", "canonical": "acme_corp"},
    {"name": "$50,000/month", "type": "MONETARY", "canonical": "50000_monthly"}
  ],
  "relations": [
    {"source": "sarah_chen", "target": "rag_project", "type": "LEADS"},
    {"source": "sarah_chen", "target": "acme_corp", "type": "WORKS_FOR"},
    {"source": "rag_project", "target": "50000_monthly", "type": "COSTS"}
  ]
}
```

**Prompt Engineering:**

```python
EXTRACTION_PROMPT = """
Extract entities and relationships from the following text.

ENTITY TYPES: PERSON, ORGANIZATION, PRODUCT, MONETARY, DATE, CLAUSE, TECHNICAL

RELATION TYPES: WORKS_FOR, CREATED, OWNS, COSTS, LOCATED_IN, PART_OF, 
                USES, DEPENDS_ON, LEADS, FUNDED_BY

TEXT:
{text}

Respond in JSON format:
{
  "entities": [{"name": "...", "type": "...", "canonical": "..."}],
  "relations": [{"source": "...", "target": "...", "type": "..."}]
}
"""
```

**Retry Logic:**
- Uses `tenacity` for exponential backoff
- Retries on: rate limits, timeouts, JSON parse errors
- Max 3 retries with 1s → 2s → 4s delays

### 6. Database Storage

**File:** `src/triple_hybrid_rag/ingestion/ingest.py`

**Purpose:** Persist documents, chunks, embeddings, and entities to PostgreSQL.

**Schema Overview:**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DATABASE SCHEMA (PostgreSQL)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────┐          ┌─────────────────┐                           │
│  │  rag_documents  │◄────────┐│ rag_parent_chunks│                          │
│  │                 │         ││                  │                           │
│  │  id (UUID PK)   │         ││  id (UUID PK)    │                           │
│  │  tenant_id      │         ││  document_id FK  │───────────────┐          │
│  │  hash_sha256    │         ││  tenant_id       │               │          │
│  │  file_name      │         ││  text            │               │          │
│  │  title          │         ││  page_start      │               ▼          │
│  │  ingestion_status│        │└──────────────────┘      ┌─────────────────┐ │
│  └─────────────────┘         │                          │rag_child_chunks │ │
│           │                   │                          │                 │ │
│           │                   │                          │  id (UUID PK)   │ │
│           │                   └─────────────────────────▶│  parent_id FK   │ │
│           │                                              │  document_id FK │ │
│           │                                              │  text           │ │
│           │                                              │  content_hash   │ │
│           │                                              │  embedding_1024 │ │
│           │                                              │  page           │ │
│           │                                              └─────────────────┘ │
│           │                                                      │           │
│           │              ┌─────────────────┐                     │           │
│           └─────────────▶│   rag_entities  │◄────────────────────┤          │
│                          │                 │                     │           │
│                          │  id (UUID PK)   │                     │           │
│                          │  document_id FK │      ┌──────────────┴─────────┐│
│                          │  tenant_id      │      │rag_entity_mentions     ││
│                          │  entity_type    │      │                        ││
│                          │  name           │      │  entity_id FK          ││
│                          │  canonical_name │◄─────│  child_chunk_id FK     ││
│                          └─────────────────┘      │  mention_text          ││
│                                   │               └────────────────────────┘│
│                                   │                                          │
│                                   ▼                                          │
│                          ┌─────────────────┐                                 │
│                          │  rag_relations  │                                 │
│                          │                 │                                 │
│                          │  id (UUID PK)   │                                 │
│                          │  source_id FK   │                                 │
│                          │  target_id FK   │                                 │
│                          │  relation_type  │                                 │
│                          │  tenant_id      │                                 │
│                          └─────────────────┘                                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Indexes:**

```sql
-- Vector similarity search (HNSW)
CREATE INDEX idx_chunks_embedding_hnsw 
ON rag_child_chunks 
USING hnsw (embedding_1024 vector_cosine_ops)
WITH (m = 16, ef_construction = 200);

-- Full-text search (GIN)
CREATE INDEX idx_chunks_fts 
ON rag_child_chunks 
USING gin (to_tsvector('english', text));

-- Tenant isolation
CREATE INDEX idx_chunks_tenant 
ON rag_child_chunks (tenant_id);

-- Content deduplication
CREATE UNIQUE INDEX idx_chunks_hash 
ON rag_child_chunks (tenant_id, content_hash);
```

**Batch Insert Pattern:**

```python
async def store_chunks_batch(
    conn: asyncpg.Connection,
    chunks: List[ChildChunk],
) -> List[UUID]:
    """
    Batch insert using COPY for maximum throughput.
    
    Single INSERT: ~1ms per row
    Batch INSERT (1000 rows): ~100ms total (0.1ms per row)
    COPY (1000 rows): ~20ms total (0.02ms per row)
    """
    # Prepare data as tuples
    records = [
        (
            chunk.id,
            chunk.parent_id,
            chunk.document_id,
            tenant_id,
            chunk.index_in_parent,
            chunk.text,
            chunk.content_hash,
            f"[{','.join(str(x) for x in chunk.embedding)}]",
            chunk.page,
        )
        for chunk in chunks
    ]
    
    # Use COPY for bulk insert
    await conn.copy_records_to_table(
        'rag_child_chunks',
        records=records,
        columns=[
            'id', 'parent_id', 'document_id', 'tenant_id',
            'index_in_parent', 'text', 'content_hash', 
            'embedding_1024', 'page'
        ],
    )
    
    return [chunk.id for chunk in chunks]
```

---

## Retrieval Pipeline

The retrieval pipeline transforms user queries into relevant, diverse, high-quality results.

### 1. Query Processing

**File:** `src/triple_hybrid_rag/core/query_planner.py`

**Purpose:** Analyze query intent and route to appropriate retrieval strategies.

**Query Classification:**

```python
class QueryIntent(Enum):
    FACTUAL = "factual"           # "What is X?"
    PROCEDURAL = "procedural"     # "How do I X?"
    COMPARATIVE = "comparative"   # "What's the difference between X and Y?"
    ENTITY_LOOKUP = "entity"      # "Who is X?"
    RELATIONAL = "relational"     # "How is X related to Y?"
    TECHNICAL = "technical"       # "How do I implement X?"
```

**Intent Detection:**

```python
def detect_intent(query: str) -> QueryIntent:
    query_lower = query.lower()
    
    # Procedural patterns
    if any(p in query_lower for p in ["how to", "how do", "steps to"]):
        return QueryIntent.PROCEDURAL
    
    # Comparative patterns
    if any(p in query_lower for p in ["difference between", "compare", "vs"]):
        return QueryIntent.COMPARATIVE
    
    # Entity lookup patterns
    if any(p in query_lower for p in ["who is", "what is"]):
        return QueryIntent.ENTITY_LOOKUP
    
    # Relational patterns
    if any(p in query_lower for p in ["related to", "works with"]):
        return QueryIntent.RELATIONAL
    
    # Default to factual
    return QueryIntent.FACTUAL
```

**Channel Selection:**

Based on intent, different search channels are weighted:

| Intent | Lexical | Semantic | Graph |
|--------|---------|----------|-------|
| FACTUAL | 0.7 | 0.8 | 0.3 |
| PROCEDURAL | 0.6 | 0.9 | 0.2 |
| ENTITY_LOOKUP | 0.5 | 0.6 | **1.0** |
| RELATIONAL | 0.3 | 0.5 | **1.0** |
| COMPARATIVE | 0.8 | 0.9 | 0.5 |

### 2. HyDE Generation (Phase 2)

**File:** `src/triple_hybrid_rag/retrieval/hyde.py`

**Purpose:** Generate hypothetical documents to bridge the query-document semantic gap.

**The Problem:**
- User queries are short and question-like
- Documents are long and statement-like
- Embedding similarity may not capture this asymmetry

**The Solution:**
1. Generate a hypothetical document that would answer the query
2. Embed the hypothetical (not the query)
3. Search for similar real documents

**Visual Explanation:**

```
Traditional Approach:
┌─────────────────┐                    ┌─────────────────┐
│  User Query     │  ──── embed ────▶  │  Query Vector   │
│  "What is ML?"  │                    │  [0.2, 0.8, ...] │
└─────────────────┘                    └────────┬────────┘
                                                │
                                                │ cosine similarity
                                                ▼
                                       ┌─────────────────┐
                                       │ Document Vectors│
                                       │ (Statement-like)│
                                       └─────────────────┘
                                       
Semantic gap: Questions ≠ Statements

HyDE Approach:
┌─────────────────┐   ┌─────────────────────────────────────────┐
│  User Query     │   │            LLM Generation               │
│  "What is ML?"  │──▶│  "Machine learning is a branch of AI   │
└─────────────────┘   │   that enables computers to learn from │
                      │   data without explicit programming..." │
                      └───────────────────┬─────────────────────┘
                                          │ embed
                                          ▼
                                 ┌─────────────────┐
                                 │ Hypo Vector     │
                                 │ [0.6, 0.3, ...] │
                                 └────────┬────────┘
                                          │ cosine similarity
                                          ▼
                                 ┌─────────────────┐
                                 │ Document Vectors│
                                 │ (Statement-like)│
                                 └─────────────────┘
                                 
No gap: Hypothetical statements match real statements
```

**Implementation:**

```python
class HyDEGenerator:
    async def generate(self, query: str, intent: Optional[str] = None) -> HyDEResult:
        # 1. Detect intent if not provided
        if intent is None:
            intent = self.detect_intent(query)
        
        # 2. Select appropriate prompt template
        prompt = HYDE_PROMPTS[intent].format(query=query)
        
        # 3. Generate hypothetical document
        response = await self._llm_generate(prompt)
        
        return HyDEResult(
            original_query=query,
            hypothetical_documents=[response],
            intent=intent,
        )
```

**Intent-Specific Prompts:**

```python
HYDE_PROMPTS = {
    "factual": """
        Write a detailed factual paragraph that would answer:
        {query}
        Include specific details as if from an authoritative source.
    """,
    
    "procedural": """
        Write a step-by-step procedure that explains:
        {query}
        Include specific steps as if from a technical manual.
    """,
    
    "entity_lookup": """
        Write an encyclopedic entry about:
        {query}
        Include definitions, properties, and context.
    """,
}
```

### 3. Query Expansion (Phase 2)

**File:** `src/triple_hybrid_rag/retrieval/query_expansion.py`

**Purpose:** Generate multiple query variants to improve recall.

**Techniques:**

1. **Multi-Query Generation:** LLM generates alternative phrasings
2. **Query Decomposition:** Break complex queries into simpler sub-queries
3. **Pseudo-Relevance Feedback (PRF):** Use top-k results to expand query

**Multi-Query Example:**

```
Original: "What is the refund policy for damaged items?"

Generated Variants:
1. "How can I get a refund for a damaged product?"
2. "What are the return rules for defective merchandise?"
3. "Refund process for items received in poor condition"
4. "Policy on returning broken or damaged goods"
```

**Query Decomposition Example:**

```
Original: "Compare the pricing and features of PostgreSQL vs MySQL for RAG systems"

Sub-queries:
1. "What is the pricing of PostgreSQL?"
2. "What is the pricing of MySQL?"
3. "What features does PostgreSQL offer for RAG?"
4. "What features does MySQL offer for RAG?"
```

**RAG-Fusion Process:**

```
Original Query
      │
      ▼
┌──────────────────┐
│ Query Expansion  │
└─────────┬────────┘
          │
    ┌─────┴─────┐
    ▼     ▼     ▼
  [Q1]   [Q2]   [Q3]   (Query variants)
    │     │     │
    ▼     ▼     ▼
┌─────┐┌─────┐┌─────┐
│Retr.││Retr.││Retr.│  (Independent retrieval)
└──┬──┘└──┬──┘└──┬──┘
   │      │      │
   └──────┼──────┘
          ▼
   ┌──────────────┐
   │  RRF Fusion  │      (Combine results)
   └──────────────┘
```

### 4. Triple-Hybrid Search

**Files:** 
- `src/triple_hybrid_rag/rag.py`
- `src/triple_hybrid_rag/graph/puppygraph.py`
- `src/triple_hybrid_rag/graph/sql_fallback.py`

**Purpose:** Execute parallel searches across three channels.

#### Lexical Search (BM25/FTS)

**SQL Query:**
```sql
SELECT 
    cc.id as child_id,
    cc.parent_id,
    cc.document_id,
    cc.text,
    cc.page,
    ts_rank_cd(
        to_tsvector('english', cc.text),
        plainto_tsquery('english', $2)
    ) as rank
FROM rag_child_chunks cc
WHERE 
    cc.tenant_id = $1
    AND to_tsvector('english', cc.text) @@ plainto_tsquery('english', $2)
ORDER BY rank DESC
LIMIT $3;
```

**BM25 Scoring:**
```
score(D, Q) = Σ IDF(qi) × (f(qi, D) × (k1 + 1)) / (f(qi, D) + k1 × (1 - b + b × |D|/avgdl))

Where:
- IDF(qi) = log((N - n(qi) + 0.5) / (n(qi) + 0.5))
- f(qi, D) = frequency of term qi in document D
- |D| = document length
- avgdl = average document length
- k1 = 1.2 (term frequency saturation)
- b = 0.75 (length normalization)
```

**Strengths:**
- Exact keyword matching
- Rare term boosting
- No embedding required
- Handles typos with fuzzy matching

#### Semantic Search (HNSW)

**SQL Query:**
```sql
SELECT 
    cc.id as child_id,
    cc.parent_id,
    cc.document_id,
    cc.text,
    cc.page,
    1 - (cc.embedding_1024 <=> $2::vector) as similarity
FROM rag_child_chunks cc
WHERE cc.tenant_id = $1
ORDER BY cc.embedding_1024 <=> $2::vector
LIMIT $3;
```

**HNSW Index:**
```
Hierarchical Navigable Small World Graph

Layer 2:  [A]----[B]----[C]  (sparse, long-range connections)
           │      │      │
Layer 1:  [A]-[D]-[B]-[E]-[C]  (medium density)
           │  │  │  │  │  │
Layer 0:  [A][D][F][B][G][E][H][C]  (dense, all nodes)

Search: Start at top layer, greedily descend
- Logarithmic complexity: O(log N)
- ~10ms for millions of vectors
```

**pgvector Configuration:**
```sql
CREATE INDEX idx_embedding_hnsw
ON rag_child_chunks
USING hnsw (embedding_1024 vector_cosine_ops)
WITH (
    m = 16,              -- Connections per node
    ef_construction = 200 -- Build-time search breadth
);

SET hnsw.ef_search = 100;  -- Query-time search breadth
```

**Strengths:**
- Semantic understanding
- Handles paraphrases
- Works with any language
- Finds conceptually similar content

#### Graph Search (Cypher)

**PuppyGraph Query:**
```cypher
MATCH (e:Entity {tenant_id: $tenant_id})
WHERE 
    e.name CONTAINS $keyword1 
    OR e.name CONTAINS $keyword2
MATCH (e)-[r:MENTIONED_IN]->(c:Chunk)
OPTIONAL MATCH (e)-[rel]-(related:Entity)
RETURN 
    c.id as child_id,
    c.text as text,
    c.parent_id as parent_id,
    collect(DISTINCT e.name) as entities,
    collect(DISTINCT type(rel)) as relations
LIMIT $limit;
```

**SQL Fallback:**
```sql
-- When PuppyGraph is unavailable
SELECT DISTINCT
    cc.id as child_id,
    cc.parent_id,
    cc.document_id,
    cc.text,
    cc.page,
    array_agg(DISTINCT e.name) as entities
FROM rag_entities e
JOIN rag_entity_mentions em ON e.id = em.entity_id
JOIN rag_child_chunks cc ON em.child_chunk_id = cc.id
WHERE 
    e.tenant_id = $1
    AND (
        e.name ILIKE '%' || $2 || '%'
        OR e.canonical_name ILIKE '%' || $2 || '%'
    )
GROUP BY cc.id, cc.parent_id, cc.document_id, cc.text, cc.page
LIMIT $3;
```

**Graph Traversal Example:**

```
Query: "Who created the RAG project?"

Graph:
    [Sarah Chen] ──LEADS──▶ [RAG Project]
         │                       │
    WORKS_FOR              COSTS
         │                       │
         ▼                       ▼
    [Acme Corp]              [$50,000]
         │
    FUNDED_BY
         │
         ▼
    [TechVentures]

Traversal: Start from "RAG project" → follow LEADS edge → return "Sarah Chen"
           Also return 2-hop: Sarah Chen → WORKS_FOR → Acme Corp
```

**Strengths:**
- Multi-hop reasoning
- Entity-centric queries
- Relationship discovery
- Structured knowledge

### 5. Reciprocal Rank Fusion (RRF)

**File:** `src/triple_hybrid_rag/core/fusion.py`

**Purpose:** Combine results from multiple search channels into a unified ranking.

**Algorithm:**

```python
def rrf_score(rank: int, k: int = 60) -> float:
    """
    RRF score for a result at given rank.
    
    Formula: 1 / (k + rank)
    
    Where:
    - rank: Position in result list (1-indexed)
    - k: Constant to prevent high scores for top ranks (default 60)
    """
    return 1.0 / (k + rank)

def fuse_results(
    lexical_results: List[SearchResult],
    semantic_results: List[SearchResult],
    graph_results: List[SearchResult],
    weights: Dict[str, float] = None,
) -> List[SearchResult]:
    """
    Combine results using weighted RRF.
    """
    weights = weights or {
        "lexical": 0.7,
        "semantic": 0.8,
        "graph": 1.0,
    }
    
    # Compute RRF scores per result
    rrf_scores: Dict[UUID, float] = {}
    result_map: Dict[UUID, SearchResult] = {}
    
    for rank, result in enumerate(lexical_results, start=1):
        rrf_scores[result.chunk_id] = rrf_scores.get(result.chunk_id, 0)
        rrf_scores[result.chunk_id] += weights["lexical"] * rrf_score(rank)
        result_map[result.chunk_id] = result
    
    for rank, result in enumerate(semantic_results, start=1):
        rrf_scores[result.chunk_id] = rrf_scores.get(result.chunk_id, 0)
        rrf_scores[result.chunk_id] += weights["semantic"] * rrf_score(rank)
        result_map[result.chunk_id] = result
    
    for rank, result in enumerate(graph_results, start=1):
        rrf_scores[result.chunk_id] = rrf_scores.get(result.chunk_id, 0)
        rrf_scores[result.chunk_id] += weights["graph"] * rrf_score(rank)
        result_map[result.chunk_id] = result
    
    # Sort by combined RRF score
    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    
    fused = []
    for chunk_id in sorted_ids:
        result = result_map[chunk_id]
        result.rrf_score = rrf_scores[chunk_id]
        fused.append(result)
    
    return fused
```

**Example Calculation:**

```
Lexical Results (weight=0.7):
  Rank 1: Doc A → RRF = 0.7 × 1/(60+1) = 0.0115
  Rank 2: Doc B → RRF = 0.7 × 1/(60+2) = 0.0113
  
Semantic Results (weight=0.8):
  Rank 1: Doc B → RRF = 0.8 × 1/(60+1) = 0.0131
  Rank 2: Doc C → RRF = 0.8 × 1/(60+2) = 0.0129
  
Graph Results (weight=1.0):
  Rank 1: Doc A → RRF = 1.0 × 1/(60+1) = 0.0164

Combined Scores:
  Doc A: 0.0115 (lexical) + 0.0164 (graph) = 0.0279 ★ Top
  Doc B: 0.0113 (lexical) + 0.0131 (semantic) = 0.0244
  Doc C: 0.0129 (semantic) = 0.0129

Final Ranking: A, B, C
```

**Adaptive Fusion (Phase 2):**

Dynamically adjust weights based on query characteristics:

```python
class AdaptiveFusion:
    def compute_weights(self, query: str, intent: QueryIntent) -> Dict[str, float]:
        """
        Adjust weights based on query analysis.
        """
        # Extract query features
        has_entities = self._has_named_entities(query)
        has_keywords = self._has_specific_keywords(query)
        is_conceptual = self._is_conceptual_query(query)
        
        # Base weights
        weights = {"lexical": 0.7, "semantic": 0.8, "graph": 0.5}
        
        # Boost graph for entity queries
        if has_entities or intent == QueryIntent.ENTITY_LOOKUP:
            weights["graph"] = 1.0
            weights["lexical"] *= 0.8
        
        # Boost lexical for keyword-heavy queries
        if has_keywords and not is_conceptual:
            weights["lexical"] = 1.0
            weights["semantic"] *= 0.8
        
        # Boost semantic for conceptual queries
        if is_conceptual:
            weights["semantic"] = 1.0
            weights["lexical"] *= 0.7
        
        return weights
```

### 6. Multi-Stage Reranking (Phase 2)

**File:** `src/triple_hybrid_rag/retrieval/reranking.py`

**Purpose:** Progressive refinement of results through multiple reranking stages.

**Pipeline:**

```
     Initial Candidates (100 results)
               │
               ▼
    ┌─────────────────────┐
    │  Stage 1: Fast      │  Lightweight bi-encoder scoring
    │  Bi-encoder Score   │  Filter: top 50
    └──────────┬──────────┘
               │
               ▼
    ┌─────────────────────┐
    │  Stage 2: Cross-    │  Deep semantic scoring
    │  Encoder Rerank     │  Filter: top 20
    └──────────┬──────────┘
               │
               ▼
    ┌─────────────────────┐
    │  Stage 3: MMR       │  Diversity injection
    │  Diversity          │  λ = 0.7
    └──────────┬──────────┘
               │
               ▼
    ┌─────────────────────┐
    │  Stage 4: Business  │  Custom scoring rules
    │  Rules              │  Boost: recent, verified
    └──────────┬──────────┘
               │
               ▼
         Final Results (top 10)
```

**Stage 1: Bi-Encoder Scoring**

Fast scoring using pre-computed embeddings:

```python
def bi_encoder_score(query_emb: List[float], doc_emb: List[float]) -> float:
    """Cosine similarity between query and document embeddings."""
    return cosine_similarity(query_emb, doc_emb)
```

**Stage 2: Cross-Encoder Reranking**

Deep semantic scoring with full attention:

```python
async def cross_encoder_rerank(
    query: str,
    documents: List[str],
) -> List[float]:
    """
    Cross-encoder processes query-document pairs jointly.
    
    Model: Qwen3-Reranker-2B via vLLM
    Input: [CLS] query [SEP] document [SEP]
    Output: Relevance score (0-1)
    """
    response = await client.post(
        f"{rerank_api_base}/rerank",
        json={
            "model": "Qwen/Qwen3-Reranker-2B",
            "query": query,
            "documents": documents,
        }
    )
    return [r["relevance_score"] for r in response.json()["results"]]
```

**Stage 3: MMR Diversity**

Maximal Marginal Relevance balances relevance and diversity:

```python
def mmr_select(
    query_emb: List[float],
    doc_embs: List[List[float]],
    relevance_scores: List[float],
    lambda_param: float = 0.7,
    top_k: int = 10,
) -> List[int]:
    """
    MMR = λ × Relevance - (1-λ) × max(Similarity to selected)
    
    λ = 1.0: Pure relevance (no diversity)
    λ = 0.5: Balanced
    λ = 0.0: Pure diversity (ignore relevance)
    """
    selected = []
    remaining = list(range(len(relevance_scores)))
    
    while len(selected) < top_k and remaining:
        best_score = float("-inf")
        best_idx = -1
        
        for idx in remaining:
            relevance = relevance_scores[idx]
            
            if selected:
                # Max similarity to any selected document
                max_sim = max(
                    cosine_similarity(doc_embs[idx], doc_embs[s])
                    for s in selected
                )
            else:
                max_sim = 0
            
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim
            
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx
        
        selected.append(best_idx)
        remaining.remove(best_idx)
    
    return selected
```

**Stage 4: Business Rules**

Apply domain-specific scoring adjustments:

```python
def apply_business_rules(results: List[SearchResult]) -> List[SearchResult]:
    """Apply custom scoring rules."""
    for result in results:
        score_modifier = 0.0
        
        # Boost recent documents
        if result.metadata.get("created_date"):
            days_old = (datetime.now() - result.metadata["created_date"]).days
            if days_old < 30:
                score_modifier += 0.1
            elif days_old < 90:
                score_modifier += 0.05
        
        # Boost verified/authoritative sources
        if result.metadata.get("verified"):
            score_modifier += 0.15
        
        # Penalize low-quality indicators
        if len(result.text) < 50:
            score_modifier -= 0.1
        
        result.final_score = result.rerank_score + score_modifier
    
    return sorted(results, key=lambda r: r.final_score, reverse=True)
```

### 7. Diversity Optimization (Phase 2)

**File:** `src/triple_hybrid_rag/retrieval/diversity.py`

**Purpose:** Ensure results cover diverse sources and perspectives.

**Source Diversity:**

```python
class DiversityOptimizer:
    def optimize(
        self,
        results: List[SearchResult],
        max_per_document: int = 3,
        max_per_page: int = 2,
    ) -> DiversityResult:
        """
        Apply source diversity constraints.
        
        Prevents one document from dominating results.
        """
        doc_counts = {}
        page_counts = {}
        diverse_results = []
        
        for result in results:
            doc_id = str(result.document_id)
            page_key = f"{doc_id}:{result.page}"
            
            # Check limits
            if doc_counts.get(doc_id, 0) >= max_per_document:
                continue
            if page_counts.get(page_key, 0) >= max_per_page:
                continue
            
            # Accept result
            diverse_results.append(result)
            doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1
            page_counts[page_key] = page_counts.get(page_key, 0) + 1
        
        return DiversityResult(
            results=diverse_results,
            diversity_score=self._compute_diversity_score(diverse_results),
        )
```

**Diversity Score:**

```python
def _compute_diversity_score(self, results: List[SearchResult]) -> float:
    """
    Compute average pairwise dissimilarity.
    
    Score ranges from 0 (all identical) to 1 (all different).
    """
    if len(results) <= 1:
        return 1.0
    
    total_dissim = 0
    pairs = 0
    
    for i, r1 in enumerate(results):
        for r2 in results[i+1:]:
            sim = self._text_similarity(r1.text, r2.text)
            total_dissim += (1 - sim)
            pairs += 1
    
    return total_dissim / pairs
```

### 8. Context Compression (Phase 3)

**File:** `src/triple_hybrid_rag/retrieval/compression.py`

**Purpose:** Extract only the relevant portions of retrieved chunks.

**Why Compress?**
- Retrieved chunks may contain irrelevant information
- LLM context windows are limited
- More focused context = better generation

**Algorithm:**

```python
class ContextCompressor:
    async def compress(
        self,
        query: str,
        documents: List[str],
        max_tokens: int = 2000,
    ) -> List[str]:
        """
        Extract query-relevant passages from documents.
        """
        compressed = []
        
        for doc in documents:
            # Split into sentences
            sentences = self._split_sentences(doc)
            
            # Score each sentence for relevance
            scores = await self._score_sentences(query, sentences)
            
            # Select top sentences up to token budget
            selected = self._select_sentences(
                sentences, scores, max_tokens // len(documents)
            )
            
            compressed.append(" ".join(selected))
        
        return compressed
    
    async def _score_sentences(
        self,
        query: str,
        sentences: List[str],
    ) -> List[float]:
        """Score sentences using LLM."""
        prompt = f"""
        Rate each sentence's relevance to the query (0-10):
        
        Query: {query}
        
        Sentences:
        {chr(10).join(f'{i+1}. {s}' for i, s in enumerate(sentences))}
        
        Return JSON: {{"scores": [...]}}
        """
        
        response = await self._llm_generate(prompt)
        return json.loads(response)["scores"]
```

---

## Advanced RAG Strategies

### Self-RAG (Phase 5)

**File:** `src/triple_hybrid_rag/retrieval/self_rag.py`

**Purpose:** Self-reflective retrieval that assesses and corrects its own results.

**Process:**

```
Query: "What is the capital of France?"
               │
               ▼
    ┌─────────────────────┐
    │  1. Retrieve        │
    │     (Standard RAG)  │
    └──────────┬──────────┘
               │
               ▼
    ┌─────────────────────┐
    │  2. Assess          │
    │  "Is retrieval      │──▶ Score: 0.9 (High)
    │   needed?"          │
    └──────────┬──────────┘
               │
               ▼
    ┌─────────────────────┐
    │  3. Score Relevance │
    │  Per Retrieved Doc  │──▶ Doc 1: 0.95, Doc 2: 0.3
    └──────────┬──────────┘
               │
               ▼
    ┌─────────────────────┐
    │  4. Generate        │
    │  With Best Docs     │
    └──────────┬──────────┘
               │
               ▼
    ┌─────────────────────┐
    │  5. Critique        │
    │  "Is answer         │──▶ Supported: Yes
    │   supported?"       │    Useful: Yes
    └──────────┬──────────┘
               │
               ▼
         Final Answer
```

**Implementation:**

```python
class SelfRAG:
    async def retrieve_and_generate(self, query: str) -> SelfRAGResult:
        # Step 1: Decide if retrieval is needed
        retrieval_needed = await self._assess_retrieval_need(query)
        
        if not retrieval_needed:
            # Answer from parametric knowledge
            answer = await self._generate_without_retrieval(query)
            return SelfRAGResult(answer=answer, used_retrieval=False)
        
        # Step 2: Retrieve documents
        documents = await self._retrieve(query)
        
        # Step 3: Score relevance of each document
        relevance_scores = await self._score_relevance(query, documents)
        
        # Filter low-relevance documents
        relevant_docs = [
            doc for doc, score in zip(documents, relevance_scores)
            if score >= self.config.relevance_threshold
        ]
        
        # Step 4: Generate answer
        answer = await self._generate_with_docs(query, relevant_docs)
        
        # Step 5: Critique the answer
        is_supported = await self._check_support(answer, relevant_docs)
        is_useful = await self._check_usefulness(answer, query)
        
        # Step 6: Potentially regenerate if not supported
        if not is_supported and self.config.allow_regeneration:
            answer = await self._regenerate(query, relevant_docs)
        
        return SelfRAGResult(
            answer=answer,
            documents=relevant_docs,
            relevance_scores=relevance_scores,
            is_supported=is_supported,
            is_useful=is_useful,
        )
```

### Corrective RAG (Phase 5)

**File:** `src/triple_hybrid_rag/retrieval/corrective_rag.py`

**Purpose:** Dynamically correct and refine retrieval based on knowledge assessment.

**Process:**

```
Query: "What is CRAG?"
               │
               ▼
    ┌─────────────────────┐
    │  1. Initial         │
    │     Retrieval       │
    └──────────┬──────────┘
               │
               ▼
    ┌─────────────────────┐
    │  2. Knowledge       │
    │  Assessment         │
    │  ┌───────────────┐  │
    │  │ Correct ✓     │  │──▶ Use as-is
    │  │ Ambiguous ?   │  │──▶ Augment with web search
    │  │ Incorrect ✗   │  │──▶ Replace with web search
    │  └───────────────┘  │
    └──────────┬──────────┘
               │
               ▼
    ┌─────────────────────┐
    │  3. Knowledge       │
    │  Refinement         │
    │  - Decompose docs   │
    │  - Filter irrelevant│
    │  - Reorder by       │
    │    relevance        │
    └──────────┬──────────┘
               │
               ▼
    ┌─────────────────────┐
    │  4. Generate        │
    │  With Refined       │
    │  Knowledge          │
    └──────────┬──────────┘
               │
               ▼
         Final Answer
```

**Implementation:**

```python
class CorrectiveRAG:
    async def retrieve_and_generate(self, query: str) -> CRAGResult:
        # Step 1: Initial retrieval
        documents = await self._retrieve(query)
        
        # Step 2: Assess each document
        assessments = []
        for doc in documents:
            assessment = await self._assess_knowledge(query, doc)
            assessments.append(assessment)
        
        # Step 3: Categorize documents
        correct_docs = [d for d, a in zip(documents, assessments) if a == "correct"]
        ambiguous_docs = [d for d, a in zip(documents, assessments) if a == "ambiguous"]
        incorrect_docs = [d for d, a in zip(documents, assessments) if a == "incorrect"]
        
        # Step 4: Handle based on assessment
        if len(correct_docs) >= self.config.min_correct_docs:
            # Use correct documents
            final_docs = correct_docs
        elif len(ambiguous_docs) > 0:
            # Augment with web search
            web_results = await self._web_search(query)
            final_docs = correct_docs + ambiguous_docs + web_results
        else:
            # Replace entirely with web search
            final_docs = await self._web_search(query)
        
        # Step 5: Refine knowledge
        refined_docs = await self._refine_knowledge(query, final_docs)
        
        # Step 6: Generate answer
        answer = await self._generate(query, refined_docs)
        
        return CRAGResult(
            answer=answer,
            documents=refined_docs,
            assessments=assessments,
        )
```

### Agentic RAG (Phase 6)

**File:** `src/triple_hybrid_rag/retrieval/agentic_rag.py`

**Purpose:** Tool-using agents that can perform complex multi-step reasoning.

**Architecture:**

```
Query: "Calculate total revenue from Q1 2025 sales data and compare to Q4 2024"
                              │
                              ▼
                   ┌─────────────────────┐
                   │    AGENTIC RAG      │
                   │    (ReAct Loop)     │
                   └──────────┬──────────┘
                              │
      ┌───────────────────────┼───────────────────────┐
      │                       │                       │
      ▼                       ▼                       ▼
┌──────────┐           ┌──────────┐           ┌──────────┐
│ Thought  │           │  Action  │           │Observation│
│          │           │          │           │          │
│ "I need  │──────────▶│ search(  │──────────▶│ Found 3  │
│  Q1 data"│           │ "Q1 2025"│           │ documents│
│          │           │  revenue)│           │          │
└──────────┘           └──────────┘           └──────────┘
      │                                             │
      │                                             │
      ▼                                             │
┌──────────┐           ┌──────────┐           ┌────┴─────┐
│ Thought  │           │  Action  │           │Observation│
│          │           │          │           │          │
│ "Now Q4  │──────────▶│ search(  │──────────▶│ Found 2  │
│  2024"   │           │ "Q4 2024"│           │ documents│
│          │           │  revenue)│           │          │
└──────────┘           └──────────┘           └──────────┘
      │                                             │
      │                                             │
      ▼                                             │
┌──────────┐           ┌──────────┐           ┌────┴─────┐
│ Thought  │           │  Action  │           │Observation│
│          │           │          │           │          │
│"Calculate│──────────▶│calculate(│──────────▶│ Q1: $5M  │
│ totals"  │           │ revenues)│           │ Q4: $4M  │
└──────────┘           └──────────┘           └──────────┘
      │
      │
      ▼
┌──────────────────────────────────────────────────────────┐
│  FINAL ANSWER                                            │
│                                                          │
│  Q1 2025 revenue ($5M) is 25% higher than Q4 2024 ($4M) │
└──────────────────────────────────────────────────────────┘
```

**Tools:**

```python
class SearchTool(Tool):
    """Search the document corpus."""
    name = "search"
    description = "Search for documents matching a query"
    
    async def execute(self, query: str, top_k: int = 5) -> List[Document]:
        return await self.search_fn(query, top_k)

class CalculateTool(Tool):
    """Perform calculations."""
    name = "calculate"
    description = "Perform mathematical calculations"
    
    async def execute(self, expression: str) -> float:
        # Safe eval with restricted builtins
        return eval(expression, {"__builtins__": {}}, {"sum": sum, "len": len})

class SQLTool(Tool):
    """Execute SQL queries."""
    name = "sql"
    description = "Execute SQL queries against the database"
    
    async def execute(self, query: str) -> List[Dict]:
        return await self.db.fetch(query)
```

**ReAct Loop:**

```python
class AgenticRAG:
    async def run(self, query: str, max_iterations: int = 10) -> AgentResult:
        thoughts = []
        actions = []
        observations = []
        
        for i in range(max_iterations):
            # Generate thought
            thought = await self._think(query, thoughts, actions, observations)
            thoughts.append(thought)
            
            # Check for final answer
            if thought.is_final_answer:
                return AgentResult(
                    answer=thought.content,
                    thoughts=thoughts,
                    actions=actions,
                    observations=observations,
                )
            
            # Generate action
            action = await self._plan_action(thought)
            actions.append(action)
            
            # Execute action
            observation = await self._execute_action(action)
            observations.append(observation)
        
        # Max iterations reached
        return AgentResult(
            answer=await self._summarize(thoughts, observations),
            thoughts=thoughts,
            actions=actions,
            observations=observations,
            max_iterations_reached=True,
        )
```

---

## Infrastructure Components

### PostgreSQL + pgvector

**Role:** Primary data store for documents, chunks, embeddings, and entities.

**Configuration:**
```yaml
# docker-compose.yml
services:
  postgres:
    image: pgvector/pgvector:pg16
    ports:
      - "54332:5432"
    environment:
      POSTGRES_DB: rag_db
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
    volumes:
      - ./database/schema.sql:/docker-entrypoint-initdb.d/01-schema.sql
```

**Key Extensions:**
- `pgvector`: Vector similarity search
- `pg_trgm`: Trigram similarity for fuzzy matching

### PuppyGraph

**Role:** Graph query layer over PostgreSQL tables.

**Configuration:**
```yaml
services:
  puppygraph:
    image: puppygraph/puppygraph:latest
    ports:
      - "8091:8081"  # Web UI
      - "7697:7687"  # Bolt protocol
    environment:
      PUPPYGRAPH_PASSWORD: puppygraph123
```

**Schema Mapping:**
```json
{
  "vertices": [
    {
      "label": "Entity",
      "tableName": "public.rag_entities",
      "mappings": {
        "id": "id",
        "name": "name",
        "type": "entity_type"
      }
    },
    {
      "label": "Chunk",
      "tableName": "public.rag_child_chunks",
      "mappings": {
        "id": "id",
        "text": "text"
      }
    }
  ],
  "edges": [
    {
      "label": "MENTIONED_IN",
      "from": "Entity",
      "to": "Chunk",
      "tableName": "public.rag_entity_mentions"
    }
  ]
}
```

### Embedding Service (vLLM)

**Role:** High-throughput embedding generation.

**Start Command:**
```bash
vllm serve Qwen/Qwen3-VL-Embedding-2B \
    --port 1234 \
    --task embed \
    --max-model-len 8192
```

**API:**
```bash
curl http://localhost:1234/v1/embeddings \
    -H "Content-Type: application/json" \
    -d '{
        "input": ["Hello world"],
        "model": "Qwen/Qwen3-VL-Embedding-2B"
    }'
```

### Reranking Service

**Role:** Cross-encoder reranking.

**Start Command:**
```bash
vllm serve Qwen/Qwen3-Reranker-2B \
    --port 1235 \
    --task rerank
```

**API:**
```bash
curl http://localhost:1235/v1/rerank \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen3-Reranker-2B",
        "query": "What is machine learning?",
        "documents": ["ML is a subset of AI...", "Python is a language..."]
    }'
```

---

## Data Models

### Core Types

```python
@dataclass
class ParentChunk:
    id: UUID
    document_id: UUID
    tenant_id: str
    index_in_document: int
    text: str
    page_start: int
    page_end: Optional[int] = None
    token_count: int = 0

@dataclass
class ChildChunk:
    id: UUID
    parent_id: UUID
    document_id: UUID
    tenant_id: str
    index_in_parent: int
    text: str
    content_hash: str
    embedding: Optional[List[float]] = None
    page: int = 1
    modality: Modality = Modality.TEXT

@dataclass
class Entity:
    id: UUID
    tenant_id: str
    document_id: UUID
    entity_type: str  # PERSON, ORGANIZATION, etc.
    name: str
    canonical_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Relation:
    id: UUID
    tenant_id: str
    source_id: UUID
    target_id: UUID
    relation_type: str  # WORKS_FOR, CREATED, etc.
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SearchResult:
    chunk_id: UUID
    parent_id: UUID
    document_id: UUID
    text: str
    page: Optional[int] = None
    modality: Modality = Modality.TEXT
    source_channel: SearchChannel = SearchChannel.SEMANTIC
    
    # Scores from different stages
    lexical_score: float = 0.0
    semantic_score: float = 0.0
    graph_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: float = 0.0
    final_score: float = 0.0
    
    # Metadata
    entities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
```

---

## Configuration System

**File:** `src/triple_hybrid_rag/config.py`

**Pattern:** Pydantic Settings with environment variable support.

```python
class RAGConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    # Database
    database_url: str = "postgresql://postgres:postgres@localhost:54332/rag_db"
    
    # Feature Flags
    rag_enabled: bool = True
    rag_lexical_enabled: bool = True
    rag_semantic_enabled: bool = True
    rag_graph_enabled: bool = True
    rag_rerank_enabled: bool = True
    
    # Phase 2 Features
    rag_hyde_enabled: bool = True
    rag_query_expansion_enabled: bool = True
    rag_multistage_rerank_enabled: bool = True
    rag_diversity_enabled: bool = True
    
    # Weights
    rag_lexical_weight: float = 0.7
    rag_semantic_weight: float = 0.8
    rag_graph_weight: float = 1.0
    
    # Models
    rag_embed_model: str = "Qwen/Qwen3-VL-Embedding-2B"
    rag_embed_dimension: int = 1024
    rag_rerank_model: str = "Qwen/Qwen3-Reranker-2B"
    
    # Chunking
    rag_parent_chunk_size: int = 800
    rag_child_chunk_size: int = 200
```

---

## Performance Characteristics

### Throughput Benchmarks

| Operation | Throughput | Notes |
|-----------|------------|-------|
| Chunking | 2,700+ chunks/s | CPU-bound, single-threaded |
| Embedding | 55+ texts/s | GPU-accelerated via vLLM |
| DB Insert (batch) | 10,000+ rows/s | Using COPY command |
| Lexical Search | <10ms | GIN index |
| Semantic Search | <20ms | HNSW index |
| Graph Search | <50ms | PuppyGraph/SQL |
| Reranking | 48ms/batch | vLLM cross-encoder |

### Latency Breakdown (P95)

```
User Query Received
     │
     ├─── Query Processing: 5ms
     ├─── HyDE Generation: 200ms (when enabled)
     ├─── Query Expansion: 150ms (when enabled)
     │
     ├─── Lexical Search: 10ms
     ├─── Semantic Search: 20ms
     ├─── Graph Search: 50ms
     │    (parallel execution)
     │
     ├─── RRF Fusion: 2ms
     ├─── Reranking: 50ms
     ├─── Diversity: 5ms
     │
     └─── Total: ~100ms (without LLM)
              ~450ms (with HyDE + expansion)
```

---

## Appendix: Algorithm Details

### HNSW Index Parameters

| Parameter | Value | Effect |
|-----------|-------|--------|
| `m` | 16 | Connections per node (higher = better recall, more memory) |
| `ef_construction` | 200 | Build-time search breadth (higher = better index quality) |
| `ef_search` | 100 | Query-time search breadth (higher = better recall, slower) |

### RRF Constants

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `k` | 60 | Standard value from original paper; prevents top ranks from dominating |

### Token Limits

| Component | Limit | Reason |
|-----------|-------|--------|
| Parent chunk | 800-1000 | Fits in LLM context with room for prompt |
| Child chunk | 200 | Optimal for embedding similarity |
| Query | 512 | Model input limit |
| Context window | 8192 | Qwen3-VL limit |

---

*Document Version: 1.0 | Last Updated: January 2026*
