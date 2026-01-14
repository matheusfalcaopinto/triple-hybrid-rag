"""
RAG Ingestion Module

Provides document ingestion pipeline for the RAG system:
- File type detection and loading
- OCR processing with DeepSeek
- Text and table-aware chunking
- Multi-modal embedding generation
- Deduplication and database storage
"""

from voice_agent.ingestion.loader import (
    DocumentLoader,
    LoadedDocument,
    detect_file_type,
    FileType,
)
from voice_agent.ingestion.chunker import (
    Chunker,
    Chunk,
    ChunkType,
)
from voice_agent.ingestion.embedder import (
    Embedder,
    EmbeddingResult,
)
from voice_agent.ingestion.ocr import (
    OCRProcessor,
    OCRResult,
)
from voice_agent.ingestion.kb_ingest import (
    KnowledgeBaseIngestor,
    IngestResult,
)

__all__ = [
    # Loader
    "DocumentLoader",
    "LoadedDocument",
    "detect_file_type",
    "FileType",
    # Chunker
    "Chunker",
    "Chunk",
    "ChunkType",
    # Embedder
    "Embedder",
    "EmbeddingResult",
    # OCR
    "OCRProcessor",
    "OCRResult",
    # Ingestor
    "KnowledgeBaseIngestor",
    "IngestResult",
]
