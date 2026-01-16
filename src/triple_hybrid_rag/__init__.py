"""
Triple-Hybrid-RAG: The Ultimate RAG System

A standalone, production-ready RAG library featuring:
- Triple-Hybrid Search: Lexical (BM25) + Semantic (HNSW) + Graph (PuppyGraph Cypher)
- Multimodal Embeddings: Text + Image support with Qwen3-VL
- Hierarchical Chunking: Parent/Child architecture for context preservation
- Weighted RRF Fusion: Configurable channel weights with conformal denoising
- Gundam Tiling OCR: High-accuracy OCR for large scanned documents

Usage:
    from triple_hybrid_rag import RAG, RAGConfig

    config = RAGConfig()
    rag = RAG(config)

    # Ingest documents
    await rag.ingest("document.pdf", collection="policies")

    # Retrieve with triple-hybrid search
    results = await rag.retrieve("What is the refund policy?", top_k=5)
"""

__version__ = "1.0.0"
__author__ = "Triple-Hybrid-RAG Team"

# Core exports
from triple_hybrid_rag.config import RAGConfig, get_settings

# Type exports
from triple_hybrid_rag.types import (
    Document,
    ParentChunk,
    ChildChunk,
    Entity,
    Relation,
    SearchResult,
    RetrievalResult,
    QueryPlan,
)

__all__ = [
    # Version
    "__version__",
    # Config
    "RAGConfig",
    "get_settings",
    # Types
    "Document",
    "ParentChunk",
    "ChildChunk",
    "Entity",
    "Relation",
    "SearchResult",
    "RetrievalResult",
    "QueryPlan",
]
