# Triple-Hybrid-RAG

> **State-of-the-Art RAG System: Lexical + Semantic + Graph with Advanced Retrieval Enhancements**

A production-ready Retrieval-Augmented Generation (RAG) library featuring triple-hybrid search, multimodal support, and cutting-edge retrieval techniques including HyDE, Self-RAG, Agentic RAG, and more.

[![Tests](https://img.shields.io/badge/tests-377%20passed-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.12+-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

## ✨ What's New (Phase 1-6 Enhancements)

### 🧠 Advanced RAG Techniques
- **HyDE (Hypothetical Document Embeddings)** - Bridge semantic gap with generated hypotheticals
- **Self-RAG** - Self-reflective retrieval with relevance assessment
- **Corrective RAG** - Dynamic knowledge refinement and correction
- **Agentic RAG** - Tool-using agents with ReAct reasoning
- **Multi-Query Expansion** - RAG-Fusion with query variants

### ⚡ Performance & Infrastructure
- **Semantic Chunking** - Boundary-aware document splitting
- **Multi-Stage Reranking** - Progressive refinement pipeline
- **Query Caching** - Semantic similarity-based cache
- **Batch Processing** - Concurrent ingestion with rate limiting
- **Observability** - Prometheus metrics and tracing

### 🎯 Retrieval Enhancements
- **SPLADE Sparse Retrieval** - Learned sparse representations
- **ColBERT Late Interaction** - Token-level MaxSim scoring
- **MMR Diversity** - Maximal Marginal Relevance optimization
- **Context Compression** - LLM-based context extraction
- **Adaptive Fusion** - Query-aware weight tuning

### 📊 Evaluation Framework
- **LLM-as-Judge** - Automated quality evaluation
- **RAGAS Metrics** - Faithfulness, relevance, coverage
- **Retrieval Metrics** - Precision, recall, MRR, NDCG

## 🎯 Core Features

### Triple-Hybrid Search
- **Lexical Search (BM25/FTS)**: PostgreSQL full-text search with ts_rank_cd
- **Semantic Search (HNSW)**: Vector similarity using pgvector with 1024d embeddings
- **Graph Search (Cypher)**: Knowledge graph traversal via PuppyGraph

### Multimodal Support
- **Text Embeddings**: Qwen3-VL-Embedding-2B (Matryoshka: 2048d → 1024d)
- **Image Embeddings**: Direct vision encoding for image retrieval
- **Mixed Embeddings**: Joint text+image representations
- **Table/Code Support**: Multi-vector retrieval for structured content

### Hierarchical Chunking
- **Parent Chunks**: 800-1000 tokens for context (stored for LLM)
- **Child Chunks**: ~200 tokens for retrieval (indexed with embeddings)
- **Semantic Boundaries**: NLP-aware sentence and paragraph detection

### Knowledge Graph
- **Entity Extraction**: GPT-5 NER/RE during ingestion
- **Relation Mapping**: Typed edges between entities
- **Cypher Queries**: Full graph traversal via PuppyGraph Bolt protocol

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ENHANCED QUERY PIPELINE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  User Query                                                                  │
│      │                                                                       │
│      ▼                                                                       │
│  ┌──────────────────┐                                                        │
│  │ Query Router     │ (Route to appropriate RAG strategy)                   │
│  └────────┬─────────┘                                                        │
│           │                                                                  │
│  ┌────────▼─────────┐                                                        │
│  │ HyDE Generation  │ (Generate hypothetical documents)                     │
│  └────────┬─────────┘                                                        │
│           │                                                                  │
│  ┌────────▼─────────┐                                                        │
│  │ Query Expansion  │ (Multi-query variants + keywords)                     │
│  └────────┬─────────┘                                                        │
│           │                                                                  │
│    ┌──────┴──────┬────────────────┐                                         │
│    ▼             ▼                ▼                                         │
│ ┌──────┐   ┌──────────┐    ┌─────────┐                                      │
│ │Lexical│  │Semantic  │    │  Graph  │                                      │
│ │(FTS)  │  │(HNSW)    │    │(Cypher) │                                      │
│ └───┬───┘  └────┬─────┘    └────┬────┘                                      │
│     │           │               │                                           │
│     └───────────┼───────────────┘                                           │
│                 ▼                                                            │
│     ┌─────────────────────┐                                                  │
│     │ Adaptive RRF Fusion │ (Query-aware weight tuning)                     │
│     └──────────┬──────────┘                                                  │
│                ▼                                                             │
│     ┌─────────────────────┐                                                  │
│     │Multi-Stage Reranker │ (Bi-encoder → Cross-encoder → MMR)              │
│     └──────────┬──────────┘                                                  │
│                ▼                                                             │
│     ┌─────────────────────┐                                                  │
│     │Context Compression  │ (Extract relevant passages)                     │
│     └──────────┬──────────┘                                                  │
│                ▼                                                             │
│     ┌─────────────────────┐                                                  │
│     │ Diversity Optimizer │ (MMR + source diversity)                        │
│     └──────────┬──────────┘                                                  │
│                ▼                                                             │
│         Top-K Results                                                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### 1. Start Infrastructure

```bash
docker compose up -d
```

This starts:
- PostgreSQL + pgvector on port **54332**
- PuppyGraph Bolt on port **7697**
- PuppyGraph Web UI on port **8091**

### 2. Install Package

```bash
# Using uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 4. Run Pipeline Test

```bash
uv run python scripts/full_pipeline_test.py
```

### 5. Use the Library

```python
from triple_hybrid_rag import get_settings
from triple_hybrid_rag.pipeline import PipelineBuilder

# Build an enhanced RAG pipeline
pipeline = (
    PipelineBuilder()
    .with_hyde()                    # Hypothetical document embeddings
    .with_query_expansion()         # Multi-query generation
    .with_multi_stage_reranking()   # Progressive refinement
    .with_diversity_optimization()  # MMR diversity
    .with_caching()                 # Query result caching
    .with_observability()           # Metrics and tracing
    .build()
)

# Execute query
results = await pipeline.retrieve("What is the refund policy?")
```

## 📦 Project Structure

```
triple-hybrid-rag/
├── src/triple_hybrid_rag/
│   ├── __init__.py           # Package exports
│   ├── config.py             # Pydantic settings (all toggles)
│   ├── types.py              # Data classes
│   ├── core/                 # Core components
│   │   ├── chunker.py        # Hierarchical chunking
│   │   ├── embedder.py       # Multimodal embeddings
│   │   ├── fusion.py         # RRF fusion
│   │   └── reranker.py       # Reranking
│   ├── retrieval/            # Advanced retrieval (NEW)
│   │   ├── hyde.py           # HyDE generation
│   │   ├── query_expansion.py # Multi-query expansion
│   │   ├── self_rag.py       # Self-reflective RAG
│   │   ├── corrective_rag.py # Corrective RAG
│   │   ├── agentic_rag.py    # Tool-using agents
│   │   ├── hierarchical.py   # Parent-doc retrieval
│   │   ├── multimodal.py     # Multimodal retrieval
│   │   ├── splade.py         # Sparse retrieval
│   │   ├── diversity.py      # MMR diversity
│   │   ├── caching.py        # Query cache
│   │   ├── batch.py          # Batch processing
│   │   └── observability.py  # Metrics/tracing
│   ├── pipeline/             # Pipeline orchestration (NEW)
│   │   ├── builder.py        # Fluent pipeline builder
│   │   └── enhanced_rag.py   # Enhanced RAG pipeline
│   ├── evaluation/           # Evaluation framework (NEW)
│   │   ├── metrics.py        # RAGAS metrics
│   │   └── judge.py          # LLM-as-Judge
│   ├── ingestion/            # Document processing
│   │   ├── semantic_chunker.py # Semantic chunking (NEW)
│   │   ├── loaders.py        # File loaders
│   │   └── ocr.py            # OCR processing
│   └── graph/                # Knowledge graph
│       ├── puppygraph.py     # PuppyGraph client
│       └── sql_fallback.py   # SQL graph fallback
├── tests/                    # Test suite (377 tests)
├── scripts/
│   └── full_pipeline_test.py # E2E pipeline test
├── docs/
│   └── ENHANCEMENT_GUIDE.md  # Detailed feature guide
└── docker-compose.yml        # Infrastructure
```

## ⚙️ Configuration Reference

### Feature Flags

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG_ENABLED` | `true` | Master enable/disable |
| `RAG_LEXICAL_ENABLED` | `true` | BM25/FTS search |
| `RAG_SEMANTIC_ENABLED` | `true` | Vector (HNSW) search |
| `RAG_GRAPH_ENABLED` | `true` | PuppyGraph Cypher search |
| `RAG_RERANK_ENABLED` | `true` | Reranking |

### Phase 2 Enhancements

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG_HYDE_ENABLED` | `true` | HyDE generation |
| `RAG_QUERY_EXPANSION_ENABLED` | `true` | Multi-query expansion |
| `RAG_MULTISTAGE_RERANK_ENABLED` | `true` | Multi-stage reranking |
| `RAG_DIVERSITY_ENABLED` | `true` | MMR diversity optimization |

### Retrieval Weights

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG_LEXICAL_WEIGHT` | `0.7` | Lexical channel RRF weight |
| `RAG_SEMANTIC_WEIGHT` | `0.8` | Semantic channel RRF weight |
| `RAG_GRAPH_WEIGHT` | `1.0` | Graph channel RRF weight |

## 🔧 API Reference

### Enhanced RAG Pipeline (NEW)

```python
from triple_hybrid_rag.pipeline import EnhancedRAGPipeline, PipelineBuilder

# Using builder pattern
pipeline = (
    PipelineBuilder()
    .with_hyde(model="gpt-4o-mini", num_hypotheticals=1)
    .with_query_expansion(num_variants=3)
    .with_multi_stage_reranking(stages=4)
    .with_diversity_optimization(mmr_lambda=0.7)
    .build()
)

# Execute
results = await pipeline.retrieve(
    query="What is the return policy?",
    top_k=10,
)
```

### HyDE Generator (NEW)

```python
from triple_hybrid_rag.retrieval import HyDEGenerator, HyDEConfig

hyde = HyDEGenerator(
    hyde_config=HyDEConfig(
        enabled=True,
        model="gpt-4o-mini",
        temperature=0.7,
    )
)

result = await hyde.generate("What is machine learning?")
# result.primary_hypothetical contains the generated document
```

### Self-RAG (NEW)

```python
from triple_hybrid_rag.retrieval import SelfRAG, SelfRAGConfig

self_rag = SelfRAG(
    retrieve_fn=my_retrieve,
    generate_fn=my_generate,
    config=SelfRAGConfig(
        max_iterations=3,
        relevance_threshold=0.7,
    )
)

result = await self_rag.retrieve_and_generate("Complex question?")
# Includes relevance scores and reflection steps
```

### Agentic RAG (NEW)

```python
from triple_hybrid_rag.retrieval import AgenticRAG, SearchTool, CalculateTool

agent = AgenticRAG(
    llm_fn=generate,
    tools=[SearchTool(search_fn), CalculateTool()],
)

result = await agent.run("Calculate the total revenue from Q1 data")
# Agent uses tools to answer complex questions
```

### Query Expansion (NEW)

```python
from triple_hybrid_rag.retrieval import QueryExpander, QueryExpansionConfig

expander = QueryExpander(
    expansion_config=QueryExpansionConfig(
        num_query_variants=3,
        prf_enabled=True,
    )
)

expanded = await expander.expand("What is RAG?")
# expanded.all_queries contains original + variants
```

### Diversity Optimizer (NEW)

```python
from triple_hybrid_rag.retrieval import DiversityOptimizer, DiversityConfig

optimizer = DiversityOptimizer(
    config=DiversityConfig(
        mmr_lambda=0.7,
        max_per_document=3,
    )
)

result = optimizer.optimize(results, top_k=10)
# result.results contains diversified results
```

### Multimodal Retriever (NEW)

```python
from triple_hybrid_rag.retrieval import MultimodalRetriever, ModalityType

retriever = MultimodalRetriever(text_embed_fn=embed)

# Add content of different types
retriever.add_content("Python is great", ModalityType.TEXT, "doc_1")
retriever.add_content(table_data, ModalityType.TABLE, "table_1")
retriever.add_content(code_snippet, ModalityType.CODE, "code_1")

# Retrieve across modalities
results = retriever.retrieve("Python features", top_k=10)
```

## 📊 Performance

### Benchmark Results

| Metric | Value |
|--------|-------|
| Chunking Throughput | 2,700+ chunks/s |
| Embedding Throughput | 55+ texts/s |
| Reranking Latency | 48ms |
| Diversity Score | 0.94 |
| Test Coverage | 377 tests passing |

### Pipeline Stages

```
Stage 1: Chunking           ✅ 2678.9/s
Stage 2: Embedding          ✅ 55.7/s
Stage 3: NER                ✅ (with valid API key)
Stage 4: DB Storage         ✅
Stage 5: Entity Storage     ✅
Stage 6: HyDE & Expansion   ✅
Stage 7: Retrieval          ✅ PuppyGraph
Stage 8: RRF Fusion         ✅
Stage 9: Multi-Stage Rerank ✅
Stage 10: Diversity Opt     ✅
Stage 11: Pipeline Demo     ✅
```

## 🖥️ Web Dashboard

A full-featured web dashboard for managing the RAG pipeline visually.

### Features
- **📊 Metrics Dashboard**: Real-time stats, feature toggles, job monitoring
- **⚙️ Configuration Panel**: Edit 54+ parameters by category, save to `.env`
- **📁 File Ingestion**: Drag-drop upload with stage progress tracking
- **🔍 Query Interface**: Triple-hybrid search with score breakdown
- **🗃️ Database Browser**: Document management with delete/download
- **🔗 Graph Viewer**: Embedded PuppyGraph Web UI

### Quick Start

```bash
# 1. Start backend (port 8009)
uv run python -m dashboard.backend.main

# 2. Start frontend (port 5173)
cd dashboard/frontend
npm install && npm run dev

# 3. Open http://localhost:5173
```

### Supported File Formats
- **Documents**: PDF, DOCX, DOC, XLSX, XLS, CSV, TXT, MD
- **Images**: PNG, JPG, JPEG, WEBP, TIFF, TIF, BMP

### API Endpoints
| Endpoint | Description |
|----------|-------------|
| `GET /api/health` | Health check |
| `GET/POST /api/config` | Configuration management |
| `POST /api/ingest/upload` | File upload |
| `GET /api/ingest/status/{id}` | Ingestion progress |
| `POST /api/retrieve` | Triple-hybrid search |
| `GET /api/database/stats` | Database statistics |
| `DELETE /api/database/documents/{id}` | Delete with cascade |
| `GET /api/documents/{id}/download` | Download original file |

📖 See [Dashboard README](dashboard/README.md) for full documentation.

## 📚 Documentation

- [Dashboard Guide](dashboard/README.md) - Web dashboard setup and API reference
- [Enhancement Guide](docs/ENHANCEMENT_GUIDE.md) - Detailed feature documentation
- [Architecture](docs/ARCHITECTURE.md) - System design
- [Benchmark Results](docs/BENCHMARK_RESULTS.md) - Performance data
- [Optimization Report](docs/OPTIMIZATION_REPORT.md) - Optimization details

## 🧪 Testing

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific phase tests
uv run pytest tests/test_p1_enhancements.py -v  # Phase 1
uv run pytest tests/test_p2_enhancements.py -v  # Phase 2
uv run pytest tests/test_p3_enhancements.py -v  # Phase 3
uv run pytest tests/test_p4_pipeline.py -v      # Phase 4
uv run pytest tests/test_p5_advanced_rag.py -v  # Phase 5
uv run pytest tests/test_p6_multimodal_agentic.py -v  # Phase 6

# Run E2E pipeline test
uv run python scripts/full_pipeline_test.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `uv run pytest`
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

Built on these excellent technologies:

- [pgvector](https://github.com/pgvector/pgvector) - Vector similarity for PostgreSQL
- [PuppyGraph](https://www.puppygraph.com/) - Graph layer over relational databases
- [Qwen3-VL](https://github.com/QwenLM/Qwen-VL) - Multimodal embeddings and vision
- [tiktoken](https://github.com/openai/tiktoken) - Token counting

Inspired by research:
- [HyDE](https://arxiv.org/abs/2212.10496) - Hypothetical Document Embeddings
- [Self-RAG](https://arxiv.org/abs/2310.11511) - Self-Reflective RAG
- [RAG-Fusion](https://arxiv.org/abs/2402.03367) - Multi-query fusion
- [RAGAS](https://arxiv.org/abs/2309.15217) - RAG Assessment
