"""
RAG 2.0 Module

Triple-Hybrid RAG architecture with:
- Lexical search (BM25/FTS)
- Semantic search (vector/HNSW)
- Graph search (PuppyGraph/Cypher)
- Matryoshka embeddings (4096→1024)
- Parent/child chunk hierarchy
- Safety threshold + denoising
"""

from voice_agent.rag2.chunker import (
    ChildChunk,
    ChunkModality,
    HierarchicalChunker,
    ParentChunk,
    RecursiveCharacterTextSplitter,
    get_hierarchical_chunker,
)
from voice_agent.rag2.embedder import (
    EmbeddingResult,
    RAG2Embedder,
    get_rag2_embedder,
    normalize_l2,
    truncate_matryoshka,
)
from voice_agent.rag2.entity_extraction import (
    EntityExtractor,
    EntityStore,
    ExtractedEntity,
    ExtractedRelation,
    ExtractionResult,
    get_entity_extractor,
    get_entity_store,
)
from voice_agent.rag2.graph_search import (
    GraphEdge,
    GraphNode,
    GraphSearcher,
    GraphSearchResult,
    get_graph_searcher,
)
from voice_agent.rag2.ingest import (
    IngestResult,
    IngestStats,
    RAG2Ingestor,
    ingest_file,
)
from voice_agent.rag2.query_planner import (
    QueryPlan,
    QueryPlanner,
    get_query_planner,
)
from voice_agent.rag2.retrieval import (
    RAG2Retriever,
    RetrievalCandidate,
    RetrievalResult,
    retrieve,
)

__all__ = [
    # Embedder
    "RAG2Embedder",
    "EmbeddingResult",
    "get_rag2_embedder",
    "normalize_l2",
    "truncate_matryoshka",
    # Chunker
    "HierarchicalChunker",
    "RecursiveCharacterTextSplitter",
    "ParentChunk",
    "ChildChunk",
    "ChunkModality",
    "get_hierarchical_chunker",
    # Graph Search
    "GraphSearcher",
    "GraphSearchResult",
    "GraphNode",
    "GraphEdge",
    "get_graph_searcher",
    # Entity Extraction
    "EntityExtractor",
    "EntityStore",
    "ExtractedEntity",
    "ExtractedRelation",
    "ExtractionResult",
    "get_entity_extractor",
    "get_entity_store",
    # Ingestion
    "RAG2Ingestor",
    "IngestResult",
    "IngestStats",
    "ingest_file",
    # Query Planning
    "QueryPlanner",
    "QueryPlan",
    "get_query_planner",
    # Retrieval
    "RAG2Retriever",
    "RetrievalResult",
    "RetrievalCandidate",
    "retrieve",
]
