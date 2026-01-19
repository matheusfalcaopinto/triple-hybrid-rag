# Triple-Hybrid-RAG

> **The Ultimate RAG System: Lexical + Semantic + Graph with Multimodal Support**

A standalone, production-ready Retrieval-Augmented Generation (RAG) library extracted and universalized from a voice-agent implementation. This system combines three search paradigms into a unified, highly configurable retrieval pipeline.

## 🎯 Features

### Triple-Hybrid Search
- **Lexical Search (BM25/FTS)**: PostgreSQL full-text search with ts_rank_cd
- **Semantic Search (HNSW)**: Vector similarity using pgvector with 1024d embeddings
- **Graph Search (Cypher)**: Knowledge graph traversal via PuppyGraph

### Multimodal Support
- **Text Embeddings**: Qwen3-VL-Embedding-2B (Matryoshka: 2048d → 1024d)
- **Image Embeddings**: Direct vision encoding for image retrieval
- **Mixed Embeddings**: Joint text+image representations
- **Gundam Tiling OCR**: High-accuracy OCR for large scanned documents

### Hierarchical Chunking
- **Parent Chunks**: 800-1000 tokens for context (stored for LLM)
- **Child Chunks**: ~200 tokens for retrieval (indexed with embeddings)
- **Context Expansion**: Retrieve children, expand to parent for generation

### Knowledge Graph
- **Entity Extraction**: GPT-5 NER/RE during ingestion
- **Relation Mapping**: Typed edges between entities
- **Cypher Queries**: Full graph traversal via PuppyGraph Bolt protocol

### Fully Toggleable
Every feature can be enabled/disabled via environment variables:
```bash
RAG_LEXICAL_ENABLED=true
RAG_SEMANTIC_ENABLED=true
RAG_GRAPH_ENABLED=true
RAG_RERANK_ENABLED=true
RAG_ENTITY_EXTRACTION_ENABLED=true
RAG_OCR_ENABLED=true
RAG_MULTIMODAL_EMBEDDING_ENABLED=true
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        QUERY PIPELINE                                │
├─────────────────────────────────────────────────────────────────────┤
│  User Query                                                          │
│      │                                                               │
│      ▼                                                               │
│  ┌──────────────┐                                                    │
│  │Query Planner │ (GPT-5: decompose query for each channel)         │
│  └──────┬───────┘                                                    │
│         │                                                            │
│    ┌────┴────┬────────────┐                                         │
│    ▼         ▼            ▼                                         │
│ ┌──────┐ ┌──────────┐ ┌─────────┐                                   │
│ │Lexical│ │Semantic  │ │  Graph  │                                   │
│ │(FTS)  │ │(HNSW)    │ │(Cypher) │                                   │
│ └───┬───┘ └────┬─────┘ └────┬────┘                                   │
│     │          │            │                                        │
│     └──────────┼────────────┘                                        │
│                ▼                                                     │
│     ┌─────────────────────┐                                         │
│     │  Weighted RRF Fusion │ (configurable weights per channel)     │
│     └──────────┬──────────┘                                         │
│                ▼                                                     │
│     ┌─────────────────────┐                                         │
│     │  Conformal Denoising │ (filter uncertain results)             │
│     └──────────┬──────────┘                                         │
│                ▼                                                     │
│     ┌─────────────────────┐                                         │
│     │    Reranker         │ (Qwen3-VL-Reranker-2B)                  │
│     └──────────┬──────────┘                                         │
│                ▼                                                     │
│         Top-K Results                                                │
└─────────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### 1. Start Infrastructure

```bash
cd triple-hybrid-rag
docker compose up -d
```

This starts:
- PostgreSQL + pgvector on port **54332**
- PuppyGraph Bolt on port **7697**
- PuppyGraph Web UI on port **8091**

### 2. Install Package

```bash
pip install -e .
```

Or with uv:
```bash
uv pip install -e .
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 4. Use the Library

```python
from triple_hybrid_rag import RAG, RAGConfig, get_settings
from triple_hybrid_rag.core import HierarchicalChunker, MultimodalEmbedder

# Initialize components
config = get_settings()
chunker = HierarchicalChunker(config)
embedder = MultimodalEmbedder(config)

# Or use the high-level orchestrator
rag = RAG(config)

# Chunk a document
text = open("document.txt").read()
parents, children = chunker.split_document(text, document_id=uuid4())

# Embed chunks
children = await embedder.embed_chunks(children)

# ... store in database, then retrieve
```

> **Note:** `RAG.ingest(...)` currently supports `.txt` and `.md` files.

## 📦 Project Structure

```
triple-hybrid-rag/
├── src/triple_hybrid_rag/
│   ├── __init__.py           # Package exports
│   ├── config.py             # Pydantic settings (all toggles)
│   ├── types.py              # Data classes (Document, Chunk, Entity, etc.)
│   ├── core/
│   │   ├── chunker.py        # Hierarchical parent/child chunking
│   │   ├── embedder.py       # Multimodal embeddings (text + image)
│   │   ├── query_planner.py  # GPT-5 query decomposition
│   │   └── fusion.py         # Weighted RRF + conformal denoising
│   └── graph/
│       ├── puppygraph.py     # Native Cypher client (Bolt:7697)
│       └── schema.py         # Entity/relation mapping
├── database/
│   └── schema.sql            # PostgreSQL + pgvector schema
├── infrastructure/
│   └── puppygraph/
│       └── schema.json       # PuppyGraph graph mapping
├── docker-compose.yml        # PostgreSQL + PuppyGraph
├── pyproject.toml            # Python package config
├── .env.example              # All configuration options
└── README.md                 # This file
```

## ⚙️ Configuration Reference

### Feature Flags

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG_ENABLED` | `true` | Master enable/disable |
| `RAG_LEXICAL_ENABLED` | `true` | BM25/FTS search |
| `RAG_SEMANTIC_ENABLED` | `true` | Vector (HNSW) search |
| `RAG_GRAPH_ENABLED` | `true` | PuppyGraph Cypher search |
| `RAG_RERANK_ENABLED` | `true` | Reranking with Qwen3-VL-Reranker-2B |
| `RAG_DENOISE_ENABLED` | `true` | Conformal denoising |
| `RAG_QUERY_PLANNER_ENABLED` | `true` | GPT-5 query decomposition |
| `RAG_ENTITY_EXTRACTION_ENABLED` | `true` | NER/RE during ingestion |
| `RAG_OCR_ENABLED` | `true` | OCR for images |
| `RAG_MULTIMODAL_EMBEDDING_ENABLED` | `true` | Direct image embeddings |

### Retrieval Weights

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG_LEXICAL_WEIGHT` | `0.7` | Lexical channel RRF weight |
| `RAG_SEMANTIC_WEIGHT` | `0.8` | Semantic channel RRF weight |
| `RAG_GRAPH_WEIGHT` | `1.0` | Graph channel RRF weight |

### Chunking Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG_PARENT_CHUNK_TOKENS` | `800` | Target parent chunk size |
| `RAG_PARENT_CHUNK_MAX_TOKENS` | `1000` | Max parent chunk size |
| `RAG_CHILD_CHUNK_TOKENS` | `200` | Target child chunk size |
| `RAG_CHUNK_OVERLAP_TOKENS` | `50` | Overlap between chunks |

### Infrastructure Ports

| Service | Port | Description |
|---------|------|-------------|
| PostgreSQL | `54332` | Database (avoids conflicts) |
| PuppyGraph Bolt | `7697` | Cypher queries |
| PuppyGraph HTTP | `8192` | REST API |
| PuppyGraph Web UI | `8091` | Graph visualization |

## 🔧 API Reference

### HierarchicalChunker

```python
from triple_hybrid_rag.core import HierarchicalChunker

chunker = HierarchicalChunker(
    parent_chunk_tokens=800,
    child_chunk_tokens=200,
    chunk_overlap_tokens=50,
)

# Split document into hierarchical chunks
parents, children = chunker.split_document(
    text="...",
    document_id=uuid4(),
    tenant_id="my-tenant",
)
```

### MultimodalEmbedder

```python
from triple_hybrid_rag.core import MultimodalEmbedder

embedder = MultimodalEmbedder()

# Text embeddings
embeddings = await embedder.embed_texts(["Hello world", "Another text"])

# Image embeddings
image_embedding = await embedder.embed_image(image_bytes)

# Mixed text+image embedding
mixed = await embedder.embed_mixed("Caption text", image_bytes)
```

### PuppyGraphClient

```python
from triple_hybrid_rag.graph import PuppyGraphClient

client = PuppyGraphClient()
await client.connect()

# Entity lookup
results = await client.entity_lookup(
    entity_name="John Doe",
    tenant_id="my-tenant",
)

# Graph traversal
results = await client.entity_neighborhood(
    entity_id="...",
    tenant_id="my-tenant",
    hops=2,
)

# Custom Cypher
records = await client.query_cypher(
    "MATCH (e:Entity)-[:MENTIONED_IN]->(c:Chunk) RETURN c",
    params={"tenant_id": "my-tenant"},
)
```

### EntityRelationExtractor (NER + RE)

```python
from triple_hybrid_rag.core import EntityRelationExtractor, GraphEntityStore

extractor = EntityRelationExtractor()
store = GraphEntityStore()

# Extract entities + relations from child chunks
result = await extractor.extract(child_chunks)

# Persist to PostgreSQL (rag_entities, rag_relations, rag_entity_mentions)
stats = await store.store(
    result,
    chunks=child_chunks,
    tenant_id="my-tenant",
    document_id=document_id,
)
print(stats)
```

### RRFFusion

```python
from triple_hybrid_rag.core import RRFFusion

fusion = RRFFusion()

# Fuse results from all channels
fused = fusion.fuse(
    lexical_results=lexical,
    semantic_results=semantic,
    graph_results=graph,
    top_k=20,
)
```

## 📊 Database Schema

### Tables

- `rag_documents`: Source documents with metadata
- `rag_parent_chunks`: Context units (800-1000 tokens)
- `rag_child_chunks`: Retrieval units (~200 tokens) with embeddings
- `rag_entities`: Knowledge graph nodes
- `rag_relations`: Knowledge graph edges
- `rag_entity_mentions`: Links entities to chunks

### Indexes

- **HNSW**: Vector similarity search (cosine distance)
- **GIN**: Full-text search with tsvector
- **GIN (trigram)**: Fuzzy entity name matching

### Functions

- `rag_lexical_search()`: BM25-style full-text search
- `rag_semantic_search()`: Vector similarity search
- `rag_image_semantic_search()`: Image embedding search
- `rag_get_parent_with_children()`: Context expansion

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest`
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

This project was extracted and universalized from a voice-agent implementation, building on the following technologies:

- [pgvector](https://github.com/pgvector/pgvector) - Vector similarity for PostgreSQL
- [PuppyGraph](https://www.puppygraph.com/) - Graph layer over relational databases
- [Qwen3-VL](https://github.com/QwenLM/Qwen-VL) - Multimodal embeddings and vision
- [tiktoken](https://github.com/openai/tiktoken) - Token counting
