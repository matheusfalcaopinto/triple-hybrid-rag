"""
Triple-Hybrid-RAG Ingestion Module

Provides document ingestion pipeline for the RAG system:
- File type detection and loading
- OCR processing with Qwen3-VL (Gundam Tiling for large images)
- Hierarchical chunking with parent/child relationships
- Multi-modal embedding generation
- Deduplication and database storage
"""

from triple_hybrid_rag.ingestion.loaders import (
    DocumentLoader,
    LoadedDocument,
    PageContent,
    detect_file_type,
    FileType,
)
from triple_hybrid_rag.ingestion.ocr import (
    OCRProcessor,
    OCRResult,
    OCRMode,
    GundamTilingConfig,
)
from triple_hybrid_rag.ingestion.ingest import (
    Ingestor,
    IngestResult,
    IngestStats,
)

__all__ = [
    # Loader
    "DocumentLoader",
    "LoadedDocument",
    "PageContent",
    "detect_file_type",
    "FileType",
    # OCR
    "OCRProcessor",
    "OCRResult",
    "OCRMode",
    "GundamTilingConfig",
    # Ingestor
    "Ingestor",
    "IngestResult",
    "IngestStats",
]
