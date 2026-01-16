"""
Triple-Hybrid-RAG Data Types

Shared data classes and types used throughout the library.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════════

class FileType(Enum):
    """Supported file types for ingestion."""
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    CSV = "csv"
    XLSX = "xlsx"
    JSON = "json"
    JSONL = "jsonl"
    LOG = "log"
    IMAGE = "image"
    UNKNOWN = "unknown"


class Modality(Enum):
    """Content modality types."""
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    MIXED = "mixed"


class EntityType(Enum):
    """Entity types for knowledge graph."""
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    PRODUCT = "PRODUCT"
    CLAUSE = "CLAUSE"
    DATE = "DATE"
    MONEY = "MONEY"
    PERCENTAGE = "PERCENTAGE"
    LOCATION = "LOCATION"
    TECHNICAL_TERM = "TECHNICAL_TERM"
    REGULATION = "REGULATION"
    EVENT = "EVENT"
    CONCEPT = "CONCEPT"


class RelationType(Enum):
    """Relation types for knowledge graph edges."""
    WORKS_FOR = "WORKS_FOR"
    LOCATED_IN = "LOCATED_IN"
    PART_OF = "PART_OF"
    RELATED_TO = "RELATED_TO"
    MENTIONS = "MENTIONS"
    DEFINES = "DEFINES"
    REFERENCES = "REFERENCES"
    DEPENDS_ON = "DEPENDS_ON"
    CREATED_BY = "CREATED_BY"
    OWNED_BY = "OWNED_BY"


class IngestionStatus(Enum):
    """Document ingestion status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class SearchChannel(Enum):
    """Retrieval search channels."""
    LEXICAL = "lexical"
    SEMANTIC = "semantic"
    GRAPH = "graph"


# ═══════════════════════════════════════════════════════════════════════════════
# DOCUMENT TYPES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Document:
    """
    A document in the RAG system.
    
    Represents a source document that has been or will be ingested.
    """
    id: UUID = field(default_factory=uuid4)
    tenant_id: str = ""  # Multi-tenancy support
    file_path: str = ""
    file_name: str = ""
    file_hash: str = ""  # SHA-256 for deduplication
    file_type: FileType = FileType.UNKNOWN
    mime_type: str = ""
    collection: str = "general"
    title: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    ingestion_status: IngestionStatus = IngestionStatus.PENDING
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    error_message: Optional[str] = None


@dataclass
class PageContent:
    """Content extracted from a single page or section of a document."""
    page_number: int
    text: str
    has_images: bool = False
    image_data: Optional[bytes] = None  # Raw image bytes for OCR/embedding
    tables: List[str] = field(default_factory=list)  # Markdown tables
    is_scanned: bool = False  # True if page appears to be scanned/image-only
    ocr_confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoadedDocument:
    """Result of loading a document from file."""
    file_path: str
    file_type: FileType
    file_hash: str
    pages: List[PageContent]
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════════
# CHUNK TYPES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ParentChunk:
    """
    A parent chunk in the hierarchical chunking system.
    
    Parent chunks provide context (800-1000 tokens) and contain
    multiple child chunks. They store metadata like page numbers,
    section headings, and OCR confidence.
    """
    id: UUID = field(default_factory=uuid4)
    document_id: UUID = field(default_factory=uuid4)
    tenant_id: str = ""
    index_in_document: int = 0
    text: str = ""
    token_count: int = 0
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    section_heading: Optional[str] = None
    ocr_confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    # Child chunks (populated during retrieval)
    children: List["ChildChunk"] = field(default_factory=list)


@dataclass
class ChildChunk:
    """
    A child chunk in the hierarchical chunking system.
    
    Child chunks are the retrieval units (~200 tokens). They have
    embeddings for vector search and are linked to parent chunks
    for context expansion.
    """
    id: UUID = field(default_factory=uuid4)
    parent_id: UUID = field(default_factory=uuid4)
    document_id: UUID = field(default_factory=uuid4)
    tenant_id: str = ""
    index_in_parent: int = 0
    text: str = ""
    token_count: int = 0
    start_char_offset: Optional[int] = None
    end_char_offset: Optional[int] = None
    page: Optional[int] = None
    modality: Modality = Modality.TEXT
    content_hash: str = ""  # For deduplication
    
    # Embeddings
    embedding: Optional[List[float]] = None  # Text embedding (1024d)
    image_embedding: Optional[List[float]] = None  # Image embedding (1024d)
    image_data: Optional[bytes] = None  # Raw image for multimodal
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    # Parent reference (populated during retrieval)
    parent: Optional[ParentChunk] = None


# ═══════════════════════════════════════════════════════════════════════════════
# ENTITY TYPES (Knowledge Graph)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Entity:
    """
    An entity in the knowledge graph.
    
    Entities are nodes extracted from documents using NER.
    """
    id: UUID = field(default_factory=uuid4)
    tenant_id: str = ""
    document_id: Optional[UUID] = None
    entity_type: EntityType = EntityType.CONCEPT
    name: str = ""
    canonical_name: str = ""  # Normalized name for deduplication
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    # Mentions (populated during retrieval)
    mentions: List["EntityMention"] = field(default_factory=list)


@dataclass
class EntityMention:
    """
    A mention of an entity in a chunk.
    
    Links entities to the chunks where they appear.
    """
    id: UUID = field(default_factory=uuid4)
    entity_id: UUID = field(default_factory=uuid4)
    parent_chunk_id: Optional[UUID] = None
    child_chunk_id: Optional[UUID] = None
    document_id: Optional[UUID] = None
    mention_text: str = ""
    confidence: float = 1.0
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Relation:
    """
    A relation (edge) in the knowledge graph.
    
    Relations connect two entities with a typed relationship.
    """
    id: UUID = field(default_factory=uuid4)
    tenant_id: str = ""
    document_id: Optional[UUID] = None
    relation_type: RelationType = RelationType.RELATED_TO
    subject_entity_id: UUID = field(default_factory=uuid4)
    object_entity_id: UUID = field(default_factory=uuid4)
    confidence: float = 1.0
    source_parent_id: Optional[UUID] = None  # Parent chunk where relation was found
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    # Populated during retrieval
    subject_entity: Optional[Entity] = None
    object_entity: Optional[Entity] = None


# ═══════════════════════════════════════════════════════════════════════════════
# SEARCH & RETRIEVAL TYPES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SearchResult:
    """
    A single search result from any channel.
    
    Contains the chunk and scoring information.
    """
    chunk_id: UUID = field(default_factory=uuid4)
    parent_id: UUID = field(default_factory=uuid4)
    document_id: UUID = field(default_factory=uuid4)
    text: str = ""
    page: Optional[int] = None
    modality: Modality = Modality.TEXT
    
    # Scores from different channels
    lexical_score: float = 0.0
    semantic_score: float = 0.0
    graph_score: float = 0.0
    
    # Combined scores
    rrf_score: float = 0.0
    rerank_score: Optional[float] = None
    final_score: float = 0.0
    
    # Source channel
    source_channel: SearchChannel = SearchChannel.SEMANTIC
    
    # Additional context
    title: Optional[str] = None
    collection: Optional[str] = None
    is_table: bool = False
    table_context: Optional[str] = None
    alt_text: Optional[str] = None
    image_data: Optional[bytes] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Parent chunk for context expansion
    parent_chunk: Optional[ParentChunk] = None


@dataclass
class QueryPlan:
    """
    Query plan from the GPT query planner.
    
    Contains decomposed query components for each search channel.
    """
    original_query: str = ""
    
    # Lexical channel
    keywords: List[str] = field(default_factory=list)
    lexical_top_k: int = 50
    
    # Semantic channel
    semantic_query_text: str = ""
    semantic_top_k: int = 100
    
    # Graph channel
    cypher_query: Optional[str] = None
    graph_top_k: int = 50
    requires_graph: bool = False
    
    # Channel weights (can override defaults)
    weights: Dict[str, float] = field(default_factory=lambda: {
        "lexical": 0.7,
        "semantic": 0.8,
        "graph": 1.0,
    })
    
    # Detected query intent
    intent: str = "general"  # factual, procedural, comparative, entity_lookup, relational


@dataclass
class RetrievalResult:
    """
    Final result of the retrieval pipeline.
    
    Contains ranked results after fusion and reranking.
    """
    query: str = ""
    query_plan: Optional[QueryPlan] = None
    results: List[SearchResult] = field(default_factory=list)
    
    # Channel-specific results (before fusion)
    lexical_results: List[SearchResult] = field(default_factory=list)
    semantic_results: List[SearchResult] = field(default_factory=list)
    graph_results: List[SearchResult] = field(default_factory=list)
    
    # Timing information
    total_duration_ms: float = 0.0
    lexical_duration_ms: float = 0.0
    semantic_duration_ms: float = 0.0
    graph_duration_ms: float = 0.0
    fusion_duration_ms: float = 0.0
    rerank_duration_ms: float = 0.0
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def top_result(self) -> Optional[SearchResult]:
        """Get the top result."""
        return self.results[0] if self.results else None
    
    @property
    def context_text(self) -> str:
        """Get concatenated text from all results for LLM context."""
        return "\n\n---\n\n".join(r.text for r in self.results)


# ═══════════════════════════════════════════════════════════════════════════════
# OCR TYPES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class OCRResult:
    """Result of OCR processing."""
    text: str = ""
    confidence: float = 0.0
    has_tables: bool = False
    tables: List[str] = field(default_factory=list)
    mode_used: str = ""
    error: Optional[str] = None
    retry_count: int = 0
    
    # Gundam Tiling metadata
    tiles_processed: int = 0
    tile_confidences: List[float] = field(default_factory=list)
    
    metadata: Dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
# INGESTION TYPES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class IngestionResult:
    """Result of document ingestion."""
    document_id: UUID = field(default_factory=uuid4)
    success: bool = False
    parent_chunks_created: int = 0
    child_chunks_created: int = 0
    entities_extracted: int = 0
    relations_extracted: int = 0
    duration_ms: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractionResult:
    """Result of entity extraction."""
    entities: List[Entity] = field(default_factory=list)
    relations: List[Relation] = field(default_factory=list)
    mentions: List[EntityMention] = field(default_factory=list)
    duration_ms: float = 0.0
    error: Optional[str] = None
