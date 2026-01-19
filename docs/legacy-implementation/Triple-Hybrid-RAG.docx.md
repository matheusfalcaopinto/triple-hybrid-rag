**Multimodal Triple-Hybrid RAG System – Full Implementation Context**

This document defines the **complete implementation context** for an agentic coding system that will one‑shot implement a **Multimodal Triple‑Hybrid RAG architecture** as described in the original technical specification. It expands the scope into detailed requirements on:

* System goals and constraints

* End‑to‑end data and inference pipelines

* Agents and responsibilities

* API contracts and message schemas

* Database schemas (relational, vector, graph)

* Model selection and serving topology

* Orchestration patterns, error handling, observability, and ops

All sections are written to be **machine‑consumable** by an autonomous coding agent (e.g. GPT‑5 / high‑reasoning model) and **implementation oriented**.

---

**1\. Business & System Goals**

**1.1 Primary Objectives**

1. **High‑precision multimodal retrieval** over:

   * Scanned / camera‑captured documents (PDF, images)

   * Textual documents (PDF, DOCX, Markdown, HTML, TXT)

   * Structured knowledge (tables, databases, graph relations)

2. **Triple‑hybrid retrieval funnel**:

   * **Lexical channel** via BM25 (Postgres FTS)

   * **Semantic channel** via vector similarity (pgvector \+ Qwen3‑VL‑Embedding)

   * **Graph channel** via PuppyGraph (Zero‑ETL over Supabase)

3. **Robust hallucination control** via:

   * Reranker score thresholds

   * Conformal prediction based context denoising

   * Transparent “information not found” behavior

4. **Agent‑friendly modular architecture**:

   * Each step encapsulated in clear services / agents with well‑defined I/O

   * Minimal hidden coupling; easy to unit‑test in isolation

**1.2 Non‑functional Requirements**

* **Latency**:

  * Retrieval+rerank target: p50 \< 1.5s, p95 \< 3s for moderate corpus (≤1M chunks) on provisioned hardware

* **Scalability**:

  * Horizontal scale on:

    * vLLM embedding/rerank workers

    * API workers for ingestion and query

* **Reliability & robustness**:

  * Idempotent ingestion

  * Exactly‑once semantics for chunk / embedding creation where feasible

  * Clear retry and dead‑letter strategy

* **Security & compliance**:

  * All data stored in Supabase (PostgreSQL \+ pgvector)

  * Access control hooks for future per‑tenant / per‑user ACLs

---

**2\. High‑Level Architecture**

**2.1 Macro Components**

1. **API Gateway / HTTP layer**

   * REST (and optionally gRPC) interface for:

     * /ingest – document ingestion

     * /query – retrieval‑augmented answering

     * /health – health check

2. **Ingestion Pipeline**

   * Orchestrator that executes:

     * File upload & metadata registration

     * DeepSeek‑OCR processing (for images/PDF)

     * Markdown normalization & cleaning

     * Hierarchical chunking (parent/child)

     * GPT‑5‑based NER & relation extraction

     * Relational/graph population in Supabase

     * Embedding generation & vector indexing

3. **Retrieval Pipeline**

   * Orchestrator that executes:

     * Query parsing & intent analysis (GPT‑5)

     * Channel‑specific queries:

       * FTS (BM25)

       * Vector (HNSW)

       * Graph (Cypher via PuppyGraph)

     * Rank fusion (Weighted RRF)

     * Context re‑expansion (child→parent)

     * Reranking (Qwen3‑VL‑Reranker)

     * Safety checks & context denoising

     * Answer generation with GPT‑5

4. **Model Serving Layer**

   * vLLM for:

     * Qwen3‑VL‑Embedding (4B)

     * Qwen3‑VL‑Reranker (4B)

   * External OpenAI GPT‑5 API for:

     * NER & relation extraction

     * Query decomposition & Cypher generation

     * Final answer generation

5. **Storage Layer**

   * **Supabase / Postgres \+ pgvector**

   * **PuppyGraph** overlay (Zero‑ETL) over Supabase

---

**3\. Agents and Responsibilities**

The implementation will be structured as a set of logical **agents/services**. They can be realized as Python modules/classes or microservices, but boundaries should be preserved.

**3.1 Ingestion Agents**

1. **IngestionCoordinatorAgent**

   * Input: Ingestion request (file upload or URL) \+ metadata

   * Responsibilities:

     * Validate and register document

     * Dispatch to **OcrAgent** or **TextExtractAgent**

     * Call **ChunkingAgent**, **KnowledgeGraphAgent**, **EmbeddingAgent**

     * Ensure idempotency (document hash)

   * Output: Ingestion job/result object with IDs for documents, chunks, entities

2. **OcrAgent (DeepSeek‑OCR)**

   * Input: PDF/image paths, pages, metadata

   * Responsibilities:

     * Apply Gundam tiling strategy

     * Execute DeepSeek‑OCR via vLLM or specific inference endpoint

     * Return structured Markdown, plus optional LaTeX & HTML for tables

   * Output:

     * ocr\_markdown (string)

     * page\_regions metadata for potential future use

3. **TextExtractAgent**

   * Input: Text‑based documents (PDF, DOCX, HTML, MD, TXT)

   * Responsibilities:

     * Extract raw text

     * Normalize to Markdown (headings, lists, tables)

   * Output: normalized\_markdown (string)

4. **ChunkingAgent**

   * Input: normalized\_markdown

   * Responsibilities:

     * Parent chunking: 800–1000 tokens

     * Child chunking: \~200 tokens

     * Maintain parent/child linkage

     * Compute stable IDs (e.g., doc\_id:parent\_idx:child\_idx)

   * Output:

     * List of parent\_chunks

     * List of child\_chunks with parent\_id references

5. **KnowledgeGraphAgent (NER & Relations – GPT‑5)**

   * Input: Parent chunks (text \+ optional local structure metadata)

   * Responsibilities:

     * Run GPT‑5 NER & relation extraction prompts

     * Normalize entities and relations into canonical types

     * Deduplicate entities (by name \+ type \+ doc namespace)

   * Output:

     * Entity records

     * Relation records

6. **EmbeddingAgent**

   * Input: Child chunks (text \+ optional local image references)

   * Responsibilities:

     * Call Qwen3‑VL‑Embedding via vLLM

     * Use Matryoshka representation (MRL):

       * Base: 4096‑dim

       * Stored: truncated 1024‑dim if chosen in config

     * Write embeddings to pgvector columns on child\_chunks table

   * Output:

     * Updated child\_chunks rows with embedding vectors

**3.2 Retrieval Agents**

1. **QueryPlannerAgent (GPT‑5)**

   * Input: User natural language query (+ optional image context)

   * Responsibilities:

     * Intent detection

     * Compute:

       * Lexical keywords

       * Embedding query text (possibly reformulated)

       * Cypher query skeleton (graph query)

     * Determine channel weights and k values

   * Output:

     * QueryPlan object with:

       * keywords

       * semantic\_query\_text

       * cypher\_query

       * channel\_weights & top\_k per channel

2. **LexicalRetrievalAgent**

   * Input: keywords, top\_k\_lexical

   * Responsibilities:

     * Use Postgres FTS (TSVECTOR/TSQUERY or BM25 wrapper)

     * Retrieve candidate child\_chunk IDs (or parent IDs)

   * Output: ranked list of candidates with scores

3. **SemanticRetrievalAgent**

   * Input: semantic\_query\_text, top\_k\_semantic

   * Responsibilities:

     * Embed query via Qwen3‑VL‑Embedding

     * Perform HNSW similarity search (pgvector)

   * Output: ranked list of candidates with scores

4. **GraphRetrievalAgent**

   * Input: cypher\_query, top\_k\_graph

   * Responsibilities:

     * Execute Cypher (PuppyGraph) over Supabase

     * Map graph nodes/edges back to document/parent/child IDs

   * Output: ranked list of candidates with graph‑based scores

5. **FusionAgent**

   * Input: candidate lists from three channels

   * Responsibilities:

     * Apply Weighted Reciprocal Rank Fusion (RRF)

     * Default weights:

       * Graph: 1.0

       * Semantic: 0.8

       * Lexical: 0.7

   * Output:

     * Single unified ranked list of child\_chunk\_ids \+ fused scores

6. **ContextExpansionAgent**

   * Input: fused list of child candidates (with scores)

   * Responsibilities:

     * Map child → parent chunks

     * Resolve duplicates

     * Select top N parents by aggregated score (e.g. top 20–50 for reranker)

   * Output:

     * Candidate parent\_chunks with aggregated scores

7. **RerankerAgent**

   * Input: Query \+ candidate parent chunks

   * Responsibilities:

     * Use Qwen3‑VL‑Reranker cross‑encoder

     * Score each pair (query, parent\_chunk)

     * Return re‑ordered top K (e.g. 5–10)

   * Output:

     * Final context parent\_chunks with rerank scores

8. **SafetyAndDenoisingAgent**

   * Input: Reranked context \+ rerank scores

   * Responsibilities:

     * Apply minimum score threshold (e.g. max score \< 0.6 → refuse)

     * Apply conformal prediction‑based trimming:

       * Remove low‑conformity snippets

       * Keep highest quality subset, reducing noise by \~2–3x

   * Output:

     * Filtered context list OR explicit refusal outcome

9. **AnswerGenerationAgent (GPT‑5)**

   * Input: User query \+ final context

   * Responsibilities:

     * Generate grounded answer

     * Cite source documents/chunks in answer

     * Optionally generate reasoning traces (for debugging)

   * Output:

     * Answer payload with references

---

**4\. APIs, Contracts and Data Schemas**

**4.1 External API Endpoints**

**4.1.1 Document Ingestion – POST /ingest**

**Request (JSON, multipart for files)**

{  
"source\_type": "upload | url",  
"file\_id": "optional-string-if-already-uploaded",  
"file\_name": "contract\_2025.pdf",  
"url": "[https://example.com/doc.pdf](https://example.com/doc.pdf)",  
"mime\_type": "application/pdf",  
"language": "pt-BR",  
"metadata": {  
"tenant\_id": "tenant\_123",  
"collection": "contracts",  
"tags": \["finance", "2025", "client\_x"\],  
"external\_id": "optional-external-system-id"  
}  
}

**Response**

{  
"status": "accepted",  
"document\_id": "doc\_uuid",  
"ingestion\_job\_id": "job\_uuid"  
}

Ingestion processing may be async; an optional GET /ingest/status/{job\_id} can be provided.

**4.1.2 Query – POST /query**

**Request**

{  
"query": "Quais são as cláusulas de reajuste de preço no contrato X?",  
"language": "pt-BR",  
"top\_k": 5,  
"filters": {  
"tenant\_id": "tenant\_123",  
"collections": \["contracts"\],  
"tags": \["client\_x"\]  
},  
"return\_context": true,  
"debug": false  
}

**Response (successful)**

{  
"answer": "Resumo das cláusulas de reajuste...",  
"contexts": \[  
{  
"document\_id": "doc\_uuid",  
"parent\_chunk\_id": "p\_1",  
"score": 0.92,  
"snippet": "Texto relevante...",  
"metadata": {  
"page": 5,  
"section\_heading": "Reajuste de Preço"  
}  
}  
\],  
"trace": {  
"max\_rerank\_score": 0.92,  
"channels\_used": \["lexical", "semantic", "graph"\],  
"timings\_ms": {  
"planning": 80,  
"retrieval": 120,  
"fusion": 10,  
"rerank": 90,  
"answer": 250  
}  
}  
}

**Response (refusal / information not found)**

{  
"answer": null,  
"error": {  
"code": "NO\_SUITABLE\_CONTEXT",  
"message": "Não foi encontrada evidência suficiente para responder com segurança.",  
"max\_rerank\_score": 0.45  
}  
}

---

**4.2 Database Schema (Supabase / PostgreSQL)**

Below are the essential tables. Names are suggestions; the agent can adjust while preserving semantics.

**4.2.1 documents**

CREATE TABLE documents (  
id UUID PRIMARY KEY DEFAULT gen\_random\_uuid(),  
external\_id TEXT,  
tenant\_id TEXT NOT NULL,  
collection TEXT,  
file\_name TEXT NOT NULL,  
mime\_type TEXT,  
language TEXT,  
source\_type TEXT CHECK (source\_type IN ('upload', 'url')),  
source\_url TEXT,  
storage\_path TEXT, \-- path in Supabase storage / S3  
hash\_sha256 TEXT NOT NULL, \-- dedup / idempotency  
tags TEXT\[\] DEFAULT '{}',  
metadata JSONB DEFAULT '{}'::jsonb,  
created\_at TIMESTAMPTZ DEFAULT now(),  
updated\_at TIMESTAMPTZ DEFAULT now()  
);  
CREATE UNIQUE INDEX documents\_tenant\_hash\_idx  
ON documents(tenant\_id, hash\_sha256);

**4.2.2 parent\_chunks**

CREATE TABLE parent\_chunks (  
id UUID PRIMARY KEY DEFAULT gen\_random\_uuid(),  
document\_id UUID REFERENCES documents(id) ON DELETE CASCADE,  
tenant\_id TEXT NOT NULL,  
index\_in\_document INT NOT NULL,  
text TEXT NOT NULL,  
token\_count INT,  
page\_start INT,  
page\_end INT,  
section\_heading TEXT,  
tags TEXT\[\] DEFAULT '{}',  
metadata JSONB DEFAULT '{}'::jsonb,  
created\_at TIMESTAMPTZ DEFAULT now()  
);  
CREATE INDEX parent\_chunks\_doc\_idx  
ON parent\_chunks(document\_id, index\_in\_document);  
CREATE INDEX parent\_chunks\_tenant\_idx  
ON parent\_chunks(tenant\_id);

**4.2.3 child\_chunks**

CREATE TABLE child\_chunks (  
id UUID PRIMARY KEY DEFAULT gen\_random\_uuid(),  
parent\_id UUID REFERENCES parent\_chunks(id) ON DELETE CASCADE,  
document\_id UUID REFERENCES documents(id) ON DELETE CASCADE,  
tenant\_id TEXT NOT NULL,  
index\_in\_parent INT NOT NULL,  
text TEXT NOT NULL,  
token\_count INT,  
start\_char\_offset INT,  
end\_char\_offset INT,  
page INT,  
tags TEXT\[\] DEFAULT '{}',  
metadata JSONB DEFAULT '{}'::jsonb,  
\-- vector: use pgvector  
embedding\_4096 VECTOR(4096), \-- optional, raw full vector  
embedding\_1024 VECTOR(1024), \-- truncated MRL representation  
created\_at TIMESTAMPTZ DEFAULT now()  
);

CREATE INDEX child\_chunks\_parent\_idx  
ON child\_chunks(parent\_id, index\_in\_parent);

CREATE INDEX child\_chunks\_tenant\_idx  
ON child\_chunks(tenant\_id);

CREATE INDEX child\_chunks\_embedding\_1024\_hnsw\_idx  
ON child\_chunks  
USING hnsw (embedding\_1024 vector\_cosine\_ops);

**4.2.4 FTS support (lexical channel)**

* Add a generated TSVECTOR column to child\_chunks (or parent\_chunks depending on chosen granularity).

ALTER TABLE child\_chunks  
ADD COLUMN tsv tsvector  
GENERATED ALWAYS AS (  
to\_tsvector('portuguese', coalesce(text, ''))  
) STORED;

CREATE INDEX child\_chunks\_tsv\_idx  
ON child\_chunks  
USING GIN (tsv);

Alternatively, store FTS only on parents; the agent should pick one and be consistent.

**4.2.5 Entities and Relations (for KG \+ PuppyGraph)**

CREATE TABLE entities (  
id UUID PRIMARY KEY DEFAULT gen\_random\_uuid(),  
tenant\_id TEXT NOT NULL,  
document\_id UUID REFERENCES documents(id) ON DELETE CASCADE,  
type TEXT NOT NULL, \-- e.g. PERSON, ORG, CLAUSE, PRODUCT  
name TEXT NOT NULL,  
canonical\_name TEXT,  
description TEXT,  
metadata JSONB DEFAULT '{}'::jsonb,  
created\_at TIMESTAMPTZ DEFAULT now()  
);

CREATE INDEX entities\_tenant\_type\_name\_idx  
ON entities(tenant\_id, type, name);

CREATE TABLE entity\_mentions (  
id UUID PRIMARY KEY DEFAULT gen\_random\_uuid(),  
entity\_id UUID REFERENCES entities(id) ON DELETE CASCADE,  
parent\_chunk\_id UUID REFERENCES parent\_chunks(id) ON DELETE CASCADE,  
child\_chunk\_id UUID REFERENCES child\_chunks(id) ON DELETE CASCADE,  
document\_id UUID REFERENCES documents(id) ON DELETE CASCADE,  
char\_start INT,  
char\_end INT,  
mention\_text TEXT,  
metadata JSONB DEFAULT '{}'::jsonb,  
created\_at TIMESTAMPTZ DEFAULT now()  
);

CREATE TABLE relations (  
id UUID PRIMARY KEY DEFAULT gen\_random\_uuid(),  
tenant\_id TEXT NOT NULL,  
document\_id UUID REFERENCES documents(id) ON DELETE CASCADE,  
type TEXT NOT NULL, \-- e.g. OWNS, EMPLOYS, REFERENCES, DEPENDS\_ON  
subject\_entity\_id UUID REFERENCES entities(id) ON DELETE CASCADE,  
object\_entity\_id UUID REFERENCES entities(id) ON DELETE CASCADE,  
confidence FLOAT,  
metadata JSONB DEFAULT '{}'::jsonb,  
created\_at TIMESTAMPTZ DEFAULT now()  
);  
CREATE INDEX relations\_tenant\_type\_idx  
ON relations(tenant\_id, type);

**PuppyGraph** will expose entities as nodes and relations as edges, plus optional document/chunk nodes.

---

**5\. Model Layer Details**

**5.1 DeepSeek‑OCR**

* **Purpose**: High‑fidelity OCR \+ structure extraction for PDFs/images.

* **Input**:

  * Images or PDF pages after tiling with Gundam strategy

* **Output**:

  * Markdown with:

    * Headings

    * Lists

    * Tables (HTML or Markdown)

    * Inline LaTeX for formulas

* **Serving**:

  * Use vLLM with appropriate DeepSeek‑OCR model

  * Ensure support for long context (complex pages)

**5.2 Qwen3‑VL‑Embedding (4B)**

* **Serving**: vLLM runner="pooling"

* **Options**:

  * FP8 quantization

  * Multi‑GPU with \--mm-encoder-tp-mode data

* **Use**:

  * Input: text-only, image-only, or mixed content

  * Output: 4096‑dim embedding

  * Store:

    * Raw 4096 optional

    * Truncated 1024‑dim vectors for primary HNSW index

**5.3 Qwen3‑VL‑Reranker (4B)**

* **Architecture**: Cross‑encoder

* **Input**:

  * Pair (query, candidate\_text\_or\_multimodal\_snippet)

* **Output**:

  * Relevance score in \[0, 1\] (or convert logit → probability)

* **Usage**:

  * Top 20–50 input candidates

  * Keep top 5–10 for GPT‑5 context

**5.4 GPT‑5 (OpenAI API)**

* **Tasks**:

  1. NER & relation extraction on parent chunks

  2. Query understanding & decomposition

  3. Cypher query generation for PuppyGraph

  4. Final answer generation

* **Key design patterns**:

  1. Use structured outputs via JSON mode wherever possible

  2. Use system prompts that explicitly:

     * Avoid hallucinations

     * Restrict answers to provided context

     * Return explicit “insufficient information” flags

---

**6\. Detailed Pipelines**

**6.1 Ingestion Pipeline (Parse–Transform–Index)**

**6.1.1 Steps**

1. **Register document**

   * Compute SHA‑256 hash for raw content

   * Upsert into documents

   * If existing document with same hash and tenant → reuse / skip ingestion (configurable)

2. **Text acquisition**

   * If mime\_type is image/PDF with scanned content → OcrAgent

   * Else → TextExtractAgent

   * Normalize to Markdown

3. **Parent chunking**

   * Algorithm: recursive character/token splitting respecting:

     * Headings

     * Paragraph boundaries

     * Table boundaries

   * Target size: 800–1000 tokens

4. **Child chunking**

   * For each parent:

     * Sliding window or fixed segmentation (\~200 tokens)

     * Record offsets and page info

   * Insert parent\_chunks and child\_chunks rows

5. **NER & relation extraction**

   * For each parent chunk (or batch):

     * Prompt GPT‑5 with structured output instruction:

       * Entities with types, canonical names

       * Relations as triplets

   * Insert entities, entity\_mentions, relations

   * Basic deduping:

     * Same (tenant, type, lower(name)) may be merged

6. **Embeddings**

   * For each child chunk:

     * Call Qwen3‑VL‑Embedding

     * Store embedding\_4096 / embedding\_1024 as per config

7. **Completion**

   * Mark ingestion job as completed in job tracking table (optional)

   * Emit observability event

**6.1.2 Idempotency & Re‑ingestion**

* Design decision: agent should implement:

  * Dedup by (tenant\_id, hash\_sha256)

  * Option to force re‑ingest for updated documents:

    * Soft delete old parent\_chunks, child\_chunks, entities, relations for that document

    * Re‑run pipeline

---

**6.2 Retrieval Pipeline (Recall‑to‑Precision Funnel)**

**6.2.1 Query planning (GPT‑5)**

* Input: user query \+ metadata filters

* Output structure (example):

{  
"keywords": \["cláusula", "reajuste", "preço", "contrato X"\],  
"semantic\_query\_text": "Resumo das cláusulas de reajuste de preço no contrato X.",  
"cypher\_query": "MATCH (c:Clause)-\[:BELONGS\_TO\]-\>(d:Document) WHERE d.collection \= 'contracts' AND [c.name](http://c.name) CONTAINS 'reajuste' RETURN c, d LIMIT 50;",  
"lexical\_top\_k": 50,  
"semantic\_top\_k": 100,  
"graph\_top\_k": 50,  
"weights": {  
"lexical": 0.7,  
"semantic": 0.8,  
"graph": 1.0  
}  
}

**6.2.2 Channel queries**

1. **LexicalRetrievalAgent**

   * Use FTS query:

SELECT  
id,  
document\_id,  
ts\_rank\_cd(tsv, plainto\_tsquery('portuguese', :keywords)) AS score  
FROM child\_chunks  
WHERE tenant\_id \= :tenant\_id  
AND (:collections IS NULL OR collection \= ANY(:collections))  
ORDER BY score DESC  
LIMIT :lexical\_top\_k;

2. **SemanticRetrievalAgent**

   * Embed query; call HNSW index:

SELECT  
id,  
document\_id,  
1 \- (embedding\_1024 \<=\> :query\_embedding) AS score  
FROM child\_chunks  
WHERE tenant\_id \= :tenant\_id  
ORDER BY embedding\_1024 \<=\> :query\_embedding  
LIMIT :semantic\_top\_k;

3. **GraphRetrievalAgent**

   * Execute Cypher via PuppyGraph, then map node IDs to parent\_chunks or child\_chunks and produce candidate list with some score (e.g. uniform or based on relation depth).

**6.2.3 Weighted RRF fusion**

* For a candidate d:

RRF(d) \= Σ\_channel w\_c \* (1 / (k \+ rank\_c(d)))

* Implementation notes:

  * k constant \~60

  * rank\_c(d) \= 1‑based rank from each channel; if not present, ignore term

  * Normalize final scores to \[0, 1\] for interpretability

**6.2.4 Context expansion**

* Map top N child\_chunk\_ids → parent\_chunks

* Aggregate scores:

  * E.g. max or sum over children belonging to the same parent

* Sort parents by aggregated score

* Take top 20–50 as input for reranker

**6.2.5 Reranking & safety**

1. **Reranking**

   * For each candidate parent:

     * Input (query, parent\_chunk.text)

     * Get score in \[0,1\]

   * Keep top 5–10

2. **Safety threshold**

   * Let s\_max be maximum rerank score

   * If s\_max \< 0.6:

     * Return NO\_SUITABLE\_CONTEXT error response

     * Do NOT call GPT‑5 for answer

**6.2.6 Conformal prediction context trimming**

* Model detail can be approximated as:

  * Compute non‑conformity scores based on e.g. rerank score, chunk length, distance from query

  * Pre‑calibrate threshold τ on held‑out data

  * During runtime:

    * Remove chunks whose non‑conformity \> τ

* Implementation guidance:

  * Start with simple heuristic:

    * Drop chunks with rerank score \< (s\_max \* α), e.g. α=0.6

  * Later can refine to full conformal pipeline

---

**7\. Configuration, Observability, and Ops**

**7.1 Configuration**

* Use a central config.yaml or environment‑driven configuration for:

  * Model endpoints (vLLM base URL, GPT‑5 API settings)

  * Embedding dimensions to store (1024 only vs 4096+1024)

  * Channel weights and top‑k defaults

  * Safety thresholds (rerank min score, conformal trimming factor)

  * Tenant / collection filtering rules

**7.2 Logging and Metrics**

**Logging**

* For each ingestion job:

  * Document ID, tenant, size, time per step

* For each query:

  * Latency per step:

    * planning, lexical, semantic, graph, fusion, rerank, answer

  * Channel usage (which channels were activated)

  * Rerank scores & final decision

**Metrics**

* QPS per endpoint

* Ingestion throughput (docs / chunks / sec)

* Query latency percentiles

* Hit rate of NO\_SUITABLE\_CONTEXT

* Distribution of rerank scores

**7.3 Error Handling**

* All external calls (OCR, vLLM, GPT‑5, PuppyGraph) must have:

  * Timeouts

  * Retries with backoff

  * Circuit‑breaker style limits

* Partial failures:

  * If graph channel fails, system should still operate with lexical+semantic

  * Log channel failure and degrade gracefully

---

**8\. Implementation Stack (Python‑centric)**

**8.1 Core Libraries & Services**

* **Web framework**: FastAPI or equivalent

* **Database**: asyncpg / SQLAlchemy for Postgres/Supabase

* **Vector**: pgvector extension on Supabase

* **HTTP clients**:

  * httpx or aiohttp for async calls to vLLM, GPT‑5, PuppyGraph

* **Task orchestration**:

  * Either sync pipeline or background worker (Celery / RQ / custom queue) for ingestion

**8.2 Directory Layout (suggested)**

app/  
api/  
[ingestion.py](http://ingestion.py)  
[query.py](http://query.py)  
core/  
[config.py](http://config.py)  
[logging.py](http://logging.py)  
ingestion/  
[coordinator.py](http://coordinator.py)  
ocr\_agent.py  
text\_agent.py  
chunking\_agent.py  
kg\_agent.py  
embedding\_agent.py  
retrieval/  
planner\_agent.py  
lexical\_agent.py  
semantic\_agent.py  
graph\_agent.py  
fusion\_agent.py  
context\_agent.py  
reranker\_agent.py  
safety\_agent.py  
answer\_agent.py  
models/  
db\_schemas.sql  
pydantic\_schemas.py  
services/  
vllm\_client.py  
gpt5\_client.py  
puppygraph\_client.py  
storage\_client.py

---

**9\. Agent Usage Guidelines (for One‑Shot Implementation)**

This section tells the coding agent how to approach implementation using this document.

1. **Respect module boundaries**:

   * Implement agents as independent classes/functions with clear I/O types.

   * Avoid leaking internal representations across boundaries; use DTOs / Pydantic models.

2. **Prioritize end‑to‑end vertical slice**:

   * First implement:

     * /ingest → OCR → Chunk → Embedding

     * /query → semantic channel \+ GPT‑5 answer (no KG, no RRF)

   * Then progressively add:

     * Lexical channel

     * Graph channel

     * Weighted RRF

     * Reranker

     * Safety & denoising

3. **Use configuration for experimental knobs**:

   * Thresholds, weights, k‑values should be easily tunable without code changes.

4. **Testability**:

   * Each agent must be unit‑testable with:

     * Mocked external clients (vLLM, GPT‑5, PuppyGraph)

     * Fixtures for DB data

5. **Extensibility**:

   * Design schemas and interfaces so additional modalities (e.g., audio) can be plugged into the same pipelines later.

---

**10\. Summary**

This context document expands the initial “Multimodal Triple‑Hybrid RAG Architecture (2026)” specification into a **complete implementation blueprint**:

* Clear agent decomposition and responsibilities

* Explicit API contracts and message formats

* Concrete Supabase/Postgres schemas for documents, chunks, entities, and relations

* Detailed ingestion and retrieval pipelines, including recall‑to‑precision funnel

* Model serving details for DeepSeek‑OCR, Qwen3‑VL, and GPT‑5

* Safety, denoising, observability, and operational guidance

An autonomous coding agent should now be able to **implement a production‑grade RAG system** that faithfully reflects the original architecture, with minimal additional clarification.