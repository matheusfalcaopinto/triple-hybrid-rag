"""
Dashboard Backend - Triple-Hybrid-RAG

FastAPI backend for the RAG dashboard supporting:
- Multimodal file ingestion (PDF, DOCX, XLSX, CSV, images)
- OCR processing via Local Visual RAG API
- Triple-hybrid retrieval (lexical + semantic + graph)
- Configuration management
- Database browsing
- Metrics monitoring
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
from pathlib import Path
import asyncio
import os
import sys
import uuid
import shutil
import tempfile
import json
import hashlib
from datetime import datetime

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Load .env from project root BEFORE importing RAGConfig
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env", override=True)

from triple_hybrid_rag.config import RAGConfig, get_settings, reset_settings
from triple_hybrid_rag.ingestion.loaders import DocumentLoader, FileType
from triple_hybrid_rag.ingestion.ocr import OCRIngestionMode, resolve_ocr_mode
from triple_hybrid_rag.core.local_vrag_ocr import LocalVisualRAGOCR
from triple_hybrid_rag.core.chunker import HierarchicalChunker
from triple_hybrid_rag.core.embedder import get_embedder, reset_embedder
from triple_hybrid_rag.core.entity_extractor import EntityRelationExtractor, GraphEntityStore
from triple_hybrid_rag.core.fusion import RRFFusion
from triple_hybrid_rag.core.reranker import get_reranker
from triple_hybrid_rag.types import SearchChannel, Modality, SearchResult

app = FastAPI(
    title="Triple-Hybrid-RAG Dashboard API",
    description="API for managing the Triple-Hybrid-RAG pipeline with multimodal support",
    version="2.0.0",
)

# CORS for frontend - Allow all origins for local network HTTPS access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for local network access
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Reset singletons on startup to ensure fresh config is used."""
    reset_embedder()
    reset_settings()

# ═══════════════════════════════════════════════════════════════════════════════
# MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class ConfigUpdate(BaseModel):
    """Configuration update request."""
    updates: Dict[str, Any]

class QueryRequest(BaseModel):
    """Retrieval query request."""
    query: str
    tenant_id: str = "default"
    collection: Optional[str] = None
    top_k: Optional[int] = None

class IngestionStage(BaseModel):
    """Ingestion stage status."""
    name: str
    status: str  # pending, running, completed, failed
    items_processed: int = 0
    error: Optional[str] = None
    messages: List[str] = []  # Detailed log messages for UI
    details: Dict[str, Any] = {}  # Key-value details for UI

class IngestionJob(BaseModel):
    """Ingestion job status."""
    job_id: str
    status: str  # pending, processing, completed, failed
    file_name: str
    file_type: str
    progress: float
    stages: List[IngestionStage]
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: str
    updated_at: str

# In-memory job storage (for demo - use Redis in production)
ingestion_jobs: Dict[str, IngestionJob] = {}

# ═══════════════════════════════════════════════════════════════════════════════
# HEALTH & INFO
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.get("/api/info")
async def get_info():
    """Get API and system information."""
    config = get_settings()
    return {
        "api_version": "2.0.0",
        "rag_enabled": config.rag_enabled,
        "database_url": config.database_url.split("@")[-1] if "@" in config.database_url else "configured",
        "image_processing_mode": config.rag_image_processing_mode,
        "supported_formats": ["pdf", "docx", "doc", "xlsx", "xls", "csv", "txt", "md", "png", "jpg", "jpeg", "webp"],
    }

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

def get_config_metadata() -> Dict[str, Dict]:
    """Get configuration parameter metadata."""
    return {
        # Feature Flags
        "rag_enabled": {"category": "Feature Flags", "type": "boolean", "description": "Master enable/disable"},
        "rag_lexical_enabled": {"category": "Feature Flags", "type": "boolean", "description": "Enable lexical (BM25) search"},
        "rag_semantic_enabled": {"category": "Feature Flags", "type": "boolean", "description": "Enable semantic (vector) search"},
        "rag_graph_enabled": {"category": "Feature Flags", "type": "boolean", "description": "Enable graph (SQL-based) search"},
        "rag_rerank_enabled": {"category": "Feature Flags", "type": "boolean", "description": "Enable reranking"},
        "rag_denoise_enabled": {"category": "Feature Flags", "type": "boolean", "description": "Enable conformal denoising"},
        "rag_query_planner_enabled": {"category": "Feature Flags", "type": "boolean", "description": "Enable query planner"},
        "rag_entity_extraction_enabled": {"category": "Feature Flags", "type": "boolean", "description": "Enable entity extraction"},
        "rag_ocr_enabled": {"category": "Feature Flags", "type": "boolean", "description": "Enable OCR for images"},
        "rag_multimodal_embedding_enabled": {"category": "Feature Flags", "type": "boolean", "description": "Enable multimodal embeddings"},
        
        # OCR Settings
        "rag_image_processing_mode": {"category": "OCR", "type": "string", "description": "Image processing mode: text, image, both, auto"},
        "rag_gundam_tiling_enabled": {"category": "OCR", "type": "boolean", "description": "Enable Gundam Tiling for large images"},
        "local_vrag_api_base": {"category": "OCR", "type": "string", "description": "Local Visual RAG API base URL"},
        "local_vrag_timeout": {"category": "OCR", "type": "integer", "description": "Local Visual RAG API timeout (seconds)", "min": 10, "max": 300},
        
        # HyDE
        "rag_hyde_enabled": {"category": "HyDE", "type": "boolean", "description": "Enable HyDE for query transformation"},
        "rag_hyde_model": {"category": "HyDE", "type": "string", "description": "LLM model for HyDE generation"},
        "rag_hyde_temperature": {"category": "HyDE", "type": "float", "description": "Temperature for HyDE generation", "min": 0, "max": 2},
        "rag_hyde_num_hypotheticals": {"category": "HyDE", "type": "integer", "description": "Number of hypothetical documents", "min": 1, "max": 5},
        
        # Query Expansion
        "rag_query_expansion_enabled": {"category": "Query Expansion", "type": "boolean", "description": "Enable query expansion"},
        "rag_query_expansion_num_variants": {"category": "Query Expansion", "type": "integer", "description": "Number of query variants", "min": 1, "max": 10},
        "rag_query_prf_enabled": {"category": "Query Expansion", "type": "boolean", "description": "Enable Pseudo-Relevance Feedback"},
        "rag_query_decomposition_enabled": {"category": "Query Expansion", "type": "boolean", "description": "Enable query decomposition"},
        
        # Retrieval Weights
        "rag_lexical_weight": {"category": "Retrieval Weights", "type": "float", "description": "Lexical channel RRF weight", "min": 0, "max": 2},
        "rag_semantic_weight": {"category": "Retrieval Weights", "type": "float", "description": "Semantic channel RRF weight", "min": 0, "max": 2},
        "rag_graph_weight": {"category": "Retrieval Weights", "type": "float", "description": "Graph channel RRF weight", "min": 0, "max": 2},
        
        # Top-K Settings
        "rag_lexical_top_k": {"category": "Top-K Settings", "type": "integer", "description": "Max results from FTS", "min": 1, "max": 500},
        "rag_semantic_top_k": {"category": "Top-K Settings", "type": "integer", "description": "Max results from vector search", "min": 1, "max": 500},
        "rag_graph_top_k": {"category": "Top-K Settings", "type": "integer", "description": "Max results from graph search", "min": 1, "max": 500},
        "rag_rerank_top_k": {"category": "Top-K Settings", "type": "integer", "description": "Results to rerank", "min": 1, "max": 100},
        "rag_final_top_k": {"category": "Top-K Settings", "type": "integer", "description": "Final results after rerank", "min": 1, "max": 50},
        
        # Multi-Stage Reranking
        "rag_multistage_rerank_enabled": {"category": "Multi-Stage Reranking", "type": "boolean", "description": "Enable multi-stage reranking"},
        "rag_rerank_stage1_enabled": {"category": "Multi-Stage Reranking", "type": "boolean", "description": "Enable Stage 1 (bi-encoder filtering)"},
        "rag_rerank_stage2_enabled": {"category": "Multi-Stage Reranking", "type": "boolean", "description": "Enable Stage 2 (cross-encoder scoring)"},
        "rag_rerank_stage3_enabled": {"category": "Multi-Stage Reranking", "type": "boolean", "description": "Enable Stage 3 (MMR diversity)"},
        "rag_rerank_stage4_enabled": {"category": "Multi-Stage Reranking", "type": "boolean", "description": "Enable Stage 4 (score calibration)"},
        "rag_rerank_mmr_lambda": {"category": "Multi-Stage Reranking", "type": "float", "description": "MMR lambda (0=diversity, 1=relevance)", "min": 0, "max": 1},
        
        # Diversity
        "rag_diversity_enabled": {"category": "Diversity", "type": "boolean", "description": "Enable diversity optimization"},
        "rag_diversity_mmr_lambda": {"category": "Diversity", "type": "float", "description": "MMR lambda for diversity", "min": 0, "max": 1},
        "rag_diversity_max_per_document": {"category": "Diversity", "type": "integer", "description": "Max results from single document", "min": 1, "max": 10},
        
        # Chunking
        "rag_parent_child_chunking": {"category": "Chunking", "type": "boolean", "description": "Enable hierarchical chunking"},
        "rag_parent_chunk_tokens": {"category": "Chunking", "type": "integer", "description": "Target parent chunk size", "min": 100, "max": 2000},
        "rag_child_chunk_tokens": {"category": "Chunking", "type": "integer", "description": "Target child chunk size", "min": 50, "max": 500},
        "rag_chunk_overlap_tokens": {"category": "Chunking", "type": "integer", "description": "Overlap between chunks", "min": 0, "max": 200},
        
        # Embedding
        "rag_embed_api_base": {"category": "Embedding", "type": "string", "description": "Embedding API base URL (local)"},
        "rag_embed_model": {"category": "Embedding", "type": "string", "description": "Embedding model name (local)"},
        "rag_embed_dim_store": {"category": "Embedding", "type": "integer", "description": "Storage embedding dimension", "min": 256, "max": 4096},
        "rag_embed_batch_size": {"category": "Embedding", "type": "integer", "description": "Batch size for embeddings", "min": 1, "max": 1000},
        
        # Jina AI Provider
        "rag_embed_provider": {"category": "Jina AI", "type": "string", "description": "Embedding provider: jina or local"},
        "rag_rerank_provider": {"category": "Jina AI", "type": "string", "description": "Reranker provider: jina or local"},
        "jina_api_key": {"category": "Jina AI", "type": "string", "description": "Jina AI API key (from jina.ai)"},
        "jina_embed_model": {"category": "Jina AI", "type": "string", "description": "Jina embedding model"},
        "jina_embed_dimensions": {"category": "Jina AI", "type": "integer", "description": "Jina embedding dimensions", "min": 256, "max": 2048},
        "jina_rerank_model": {"category": "Jina AI", "type": "string", "description": "Jina reranker model"},
        
        # Database
        "database_url": {"category": "Database", "type": "string", "description": "PostgreSQL connection URL"},
        "database_pool_size": {"category": "Database", "type": "integer", "description": "Connection pool size", "min": 1, "max": 100},
        
        # Entity Extraction
        "rag_ner_model": {"category": "Entity Extraction", "type": "string", "description": "NER model"},
        "rag_entity_types": {"category": "Entity Extraction", "type": "string", "description": "Comma-separated entity types"},
        
        # Observability
        "rag_metrics_enabled": {"category": "Observability", "type": "boolean", "description": "Enable Prometheus metrics"},
        "log_level": {"category": "Observability", "type": "string", "description": "Log level"},
    }

@app.get("/api/config")
async def get_config():
    """Get current configuration with metadata."""
    config = get_settings()
    metadata = get_config_metadata()
    
    # Build response with values and metadata
    result = {}
    for key, meta in metadata.items():
        value = getattr(config, key, None)
        result[key] = {
            "value": value,
            **meta,
        }
    
    # Get unique categories
    categories = sorted(set(m["category"] for m in metadata.values()))
    
    return {"config": result, "categories": categories}

@app.post("/api/config")
async def update_config(update: ConfigUpdate):
    """Update configuration values (saves to .env)."""
    env_path = PROJECT_ROOT / ".env"
    
    # Read current .env
    env_lines = []
    if env_path.exists():
        env_lines = env_path.read_text().splitlines()
    
    # Update or add values
    updated_keys = set()
    for i, line in enumerate(env_lines):
        if "=" in line and not line.strip().startswith("#"):
            key = line.split("=")[0].strip()
            env_key = key.upper()
            for update_key, update_value in update.updates.items():
                if update_key.upper() == env_key:
                    env_lines[i] = f"{key}={update_value}"
                    updated_keys.add(update_key)
    
    # Add new keys
    for key, value in update.updates.items():
        if key not in updated_keys:
            env_lines.append(f"{key.upper()}={value}")
    
    # Write back
    env_path.write_text("\n".join(env_lines))
    
    # CRITICAL: Force reload environment variables into os.environ
    # This ensures the running process sees the new values immediately
    load_dotenv(env_path, override=True)
    
    # Reset the singleton config to pick up new values
    reset_settings()
    reset_embedder()
    
    return {"status": "updated", "updated_keys": list(update.updates.keys())}

@app.post("/api/config/reload")
async def reload_config():
    """Reload configuration from .env file."""
    reset_settings()
    return {"status": "reloaded"}

# ═══════════════════════════════════════════════════════════════════════════════
# INGESTION - Multimodal Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

# Persistent document storage
UPLOAD_DIR = Path(tempfile.gettempdir()) / "rag-uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Persistent storage for original files (not temp)
DOCUMENT_STORAGE_DIR = PROJECT_ROOT / "storage" / "documents"
DOCUMENT_STORAGE_DIR.mkdir(parents=True, exist_ok=True)

import math

def _safe_float(value: float, default: float = 0.0) -> float:
    """Convert NaN/Inf to a safe JSON-serializable value."""
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return default
    return value

def _vector_literal(values: List[float]) -> str:
    """Convert embedding to PostgreSQL vector literal."""
    return "[" + ",".join(f"{v:.6f}" for v in values) + "]"

async def run_ingestion(job_id: str, file_path: Path) -> None:
    """
    Background task to run multimodal ingestion pipeline.
    
    Pipeline stages:
    1. Document Loading (type-specific loaders)
    2. OCR Processing (for images/scanned pages)
    3. Text Aggregation (combine native + OCR text)
    4. Chunking (hierarchical parent/child)
    5. Embedding (via vLLM)
    6. Database Storage
    7. Entity Extraction (optional)
    """
    job = ingestion_jobs[job_id]
    config = get_settings()
    
    # Initialize stages
    stages = [
        IngestionStage(name="Loading", status="pending"),
        IngestionStage(name="OCR", status="pending"),
        IngestionStage(name="Chunking", status="pending"),
        IngestionStage(name="Embedding", status="pending"),
        IngestionStage(name="Storage", status="pending"),
    ]
    if config.rag_entity_extraction_enabled:
        stages.append(IngestionStage(name="Entity Extraction", status="pending"))
    
    job.stages = stages
    job.status = "processing"
    job.updated_at = datetime.utcnow().isoformat()
    
    try:
        import asyncpg
        from uuid import uuid4
        
        # ═══════════════════════════════════════════════════════════════════
        # Stage 1: Document Loading
        # ═══════════════════════════════════════════════════════════════════
        stages[0].status = "running"
        stages[0].messages.append(f"Loading file: {file_path.name}")
        job.progress = 0.05
        job.updated_at = datetime.utcnow().isoformat()
        
        loader = DocumentLoader(pdf_dpi=300, extract_tables=True)
        document = loader.load(file_path)
        
        if document.error:
            raise Exception(f"Loading failed: {document.error}")
        
        job.file_type = document.file_type.value
        stages[0].status = "completed"
        stages[0].items_processed = len(document.pages)
        stages[0].messages.append(f"Loaded {len(document.pages)} pages")
        stages[0].details = {
            "file_type": document.file_type.value,
            "pages": len(document.pages),
            "has_text": any(p.text.strip() for p in document.pages),
            "has_images": any(p.has_images or p.is_scanned for p in document.pages),
        }
        job.progress = 0.15
        job.updated_at = datetime.utcnow().isoformat()
        
        # ═══════════════════════════════════════════════════════════════════
        # Stage 2: OCR / Image Processing
        # ═══════════════════════════════════════════════════════════════════
        stages[1].status = "running"
        stages[1].name = "OCR/Image"
        job.updated_at = datetime.utcnow().isoformat()
        
        # Determine image ingestion mode based on new config
        image_processing_mode = getattr(config, 'rag_image_processing_mode', 'auto').lower()
        embed_provider = getattr(config, 'rag_embed_provider', 'local').lower()
        multimodal_enabled = getattr(config, 'rag_multimodal_embedding_enabled', False)
        ocr_enabled = getattr(config, 'rag_ocr_enabled', False)
        
        stages[1].messages.append(f"Config: image_processing_mode={image_processing_mode}, embed_provider={embed_provider}")
        stages[1].messages.append(f"Multimodal embedding: {'enabled' if multimodal_enabled else 'disabled'}")
        stages[1].messages.append(f"OCR: {'enabled' if ocr_enabled else 'disabled'}")
        
        # Find pages with images
        pages_with_images = [
            (i, page) for i, page in enumerate(document.pages)
            if page.is_scanned or (page.has_images and page.image_data)
        ]
        
        stages[1].messages.append(f"Found {len(pages_with_images)} pages with images/scans")
        
        ocr_texts: Dict[int, str] = {}
        image_data_for_embedding: Dict[int, bytes] = {}
        
        # Determine processing mode based on rag_image_processing_mode
        # Values: text (OCR only), image (embedding only), both (OCR + embedding), auto
        use_ocr = image_processing_mode in ('text', 'both', 'auto') and ocr_enabled
        use_image_embedding = image_processing_mode in ('image', 'both') and multimodal_enabled
        
        # Auto mode: use OCR if no multimodal, otherwise use both if available
        if image_processing_mode == 'auto':
            use_image_embedding = multimodal_enabled
            use_ocr = ocr_enabled
        
        if use_ocr and use_image_embedding:
            stages[1].details["decision"] = "both_ocr_and_image"
        elif use_image_embedding:
            stages[1].details["decision"] = "image_embedding"
        elif use_ocr:
            stages[1].details["decision"] = "ocr_text"
        else:
            stages[1].details["decision"] = "none"
        
        if pages_with_images:
            # BOTH MODE: Extract OCR text AND collect images for embedding
            if use_ocr and use_image_embedding:
                stages[1].messages.append("→ Using BOTH mode (OCR text + Image embedding)")
                
                ocr_processor = LocalVisualRAGOCR(config=config)
                
                for page_idx, page in pages_with_images:
                    if page.image_data:
                        # Do OCR
                        result = await ocr_processor.process_image(page.image_data)
                        if result.text and not result.error:
                            ocr_texts[page_idx] = result.text
                            stages[1].messages.append(f"  • OCR page {page_idx + 1}: {len(result.text)} chars")
                        
                        # Also collect image for embedding
                        image_data_for_embedding[page_idx] = page.image_data
                
                stages[1].messages.append(f"  ✓ OCR: {len(ocr_texts)} pages, Images: {len(image_data_for_embedding)} pages")
                stages[1].items_processed = len(ocr_texts) + len(image_data_for_embedding)
                stages[1].details["ocr_pages"] = len(ocr_texts)
                stages[1].details["images_collected"] = len(image_data_for_embedding)
            
            # IMAGE EMBEDDING ONLY (no OCR)
            elif use_image_embedding:
                stages[1].messages.append("→ Using image embedding only (multimodal)")
                for page_idx, page in pages_with_images:
                    if page.image_data:
                        image_data_for_embedding[page_idx] = page.image_data
                        stages[1].messages.append(f"  • Collected image from page {page_idx + 1}")
                stages[1].items_processed = len(image_data_for_embedding)
                stages[1].details["images_collected"] = len(image_data_for_embedding)
            
            # OCR ONLY (no image embedding)
            elif use_ocr:
                stages[1].messages.append("→ Using OCR text extraction only")
                ocr_processor = LocalVisualRAGOCR(config=config)
                
                for page_idx, page in pages_with_images:
                    if page.image_data:
                        result = await ocr_processor.process_image(page.image_data)
                        if result.text and not result.error:
                            ocr_texts[page_idx] = result.text
                            stages[1].messages.append(f"  • OCR page {page_idx + 1}: {len(result.text)} chars")
                
                stages[1].items_processed = len(ocr_texts)
                stages[1].details["ocr_pages"] = len(ocr_texts)
            else:
                stages[1].messages.append("⚠ No processing: OCR disabled and image embedding not available")
                stages[1].details["skipped"] = True
        else:
            stages[1].messages.append("No images found - skipping image processing")
        
        stages[1].status = "completed"
        job.progress = 0.30
        job.updated_at = datetime.utcnow().isoformat()
        
        # ═══════════════════════════════════════════════════════════════════
        # Stage 3: Text Aggregation & Chunking
        # ═══════════════════════════════════════════════════════════════════
        stages[2].status = "running"
        stages[2].messages.append("Aggregating text from all sources...")
        job.updated_at = datetime.utcnow().isoformat()
        
        # Aggregate text from all sources
        text_parts = []
        native_text_pages = 0
        for i, page in enumerate(document.pages):
            if page.text.strip():
                text_parts.append(f"[Page {page.page_number}]\n{page.text}")
                native_text_pages += 1
            
            # Add OCR text (but NOT for pages that will get combined text+image embedding)
            # Those will be handled separately as multimodal chunks
            if i in ocr_texts and i not in image_data_for_embedding:
                text_parts.append(f"[Page {page.page_number} - OCR]\n{ocr_texts[i]}")
            
            # Add tables
            for table in page.tables:
                text_parts.append(f"[Table - Page {page.page_number}]\n{table}")
        
        full_text = "\n\n".join(text_parts)
        document_id = uuid4()
        
        # Check if we have text content OR images for direct embedding
        has_text = bool(full_text.strip())
        has_images_for_embedding = bool(image_data_for_embedding)
        
        # Pages with BOTH OCR text AND image (for combined embedding)
        combined_pages = set(ocr_texts.keys()) & set(image_data_for_embedding.keys())
        # Pages with image only (no OCR text)
        image_only_pages = set(image_data_for_embedding.keys()) - set(ocr_texts.keys())
        
        stages[2].messages.append(f"Native text from {native_text_pages} pages")
        stages[2].messages.append(f"OCR text from {len(ocr_texts)} pages")
        stages[2].messages.append(f"Combined text+image pages: {len(combined_pages)}")
        stages[2].messages.append(f"Image-only pages: {len(image_only_pages)}")
        stages[2].details["has_text"] = has_text
        stages[2].details["has_images"] = has_images_for_embedding
        stages[2].details["combined_pages"] = len(combined_pages)
        
        if not has_text and not has_images_for_embedding:
            raise Exception("No text content extracted from document and no images available for direct embedding. Enable OCR or set RAG_IMAGE_PROCESSING_MODE=both with multimodal embedding.")
        
        parent_chunks = []
        child_chunks = []
        
        if has_text:
            stages[2].messages.append("→ Creating text chunks (hierarchical)...")
            chunker = HierarchicalChunker(config=config)
            parent_chunks, child_chunks = chunker.split_document(
                text=full_text,
                document_id=document_id,
                tenant_id="default",
            )
            stages[2].messages.append(f"  • {len(parent_chunks)} parent chunks")
            stages[2].messages.append(f"  • {len(child_chunks)} child chunks")
        
        # Create multimodal chunks for pages with BOTH OCR text AND image
        # These will use combined text+image embedding
        multimodal_chunks = []
        if combined_pages:
            stages[2].messages.append("→ Creating multimodal chunks (OCR text + image)...")
            from triple_hybrid_rag.types import ChildChunk, ParentChunk
            
            for page_idx in sorted(combined_pages):
                page_num = page_idx + 1
                ocr_text = ocr_texts[page_idx]
                img_data = image_data_for_embedding[page_idx]
                
                chunk_id = uuid4()
                parent_id = uuid4()
                
                # Create parent chunk with OCR text
                mm_parent = ParentChunk(
                    id=parent_id,
                    document_id=document_id,
                    tenant_id="default",
                    index_in_document=len(parent_chunks) + page_idx,
                    text=ocr_text[:500] + "..." if len(ocr_text) > 500 else ocr_text,
                    token_count=len(ocr_text.split()),
                    page_start=page_num,
                )
                parent_chunks.append(mm_parent)
                
                # Create child chunk with BOTH text and image
                mm_chunk = ChildChunk(
                    id=chunk_id,
                    parent_id=parent_id,
                    document_id=document_id,
                    tenant_id="default",
                    index_in_parent=0,
                    text=ocr_text,  # Full OCR text
                    token_count=len(ocr_text.split()),
                    page=page_num,
                    modality=Modality.IMAGE,  # Mark as multimodal
                )
                mm_chunk.image_data = img_data
                mm_chunk.ocr_text = ocr_text  # Store OCR text for combined embedding
                multimodal_chunks.append(mm_chunk)
            
            stages[2].messages.append(f"  • {len(multimodal_chunks)} multimodal chunks created")
        
        # Create image-only chunks (no OCR text available)
        image_only_chunks = []
        if image_only_pages:
            stages[2].messages.append("→ Creating image-only chunks...")
            from triple_hybrid_rag.types import ChildChunk, ParentChunk
            
            for page_idx in sorted(image_only_pages):
                page_num = page_idx + 1
                img_data = image_data_for_embedding[page_idx]
                
                chunk_id = uuid4()
                parent_id = uuid4()
                
                img_parent = ParentChunk(
                    id=parent_id,
                    document_id=document_id,
                    tenant_id="default",
                    index_in_document=len(parent_chunks) + page_idx,
                    text=f"[Image Page {page_num}]",
                    token_count=10,
                    page_start=page_num,
                )
                parent_chunks.append(img_parent)
                
                img_chunk = ChildChunk(
                    id=chunk_id,
                    parent_id=parent_id,
                    document_id=document_id,
                    tenant_id="default",
                    index_in_parent=0,
                    text=f"[Image from page {page_num}]",
                    token_count=10,
                    page=page_num,
                    modality=Modality.IMAGE,
                )
                img_chunk.image_data = img_data
                image_only_chunks.append(img_chunk)
            
            stages[2].messages.append(f"  • {len(image_only_chunks)} image-only chunks created")
        
        all_chunks = child_chunks + multimodal_chunks + image_only_chunks
        
        stages[2].status = "completed"
        stages[2].items_processed = len(all_chunks)
        stages[2].details["text_chunks"] = len(child_chunks)
        stages[2].details["multimodal_chunks"] = len(multimodal_chunks)
        stages[2].details["image_only_chunks"] = len(image_only_chunks)
        stages[2].details["total_chunks"] = len(all_chunks)
        job.progress = 0.45
        job.updated_at = datetime.utcnow().isoformat()
        
        # ═══════════════════════════════════════════════════════════════════
        # Stage 4: Embedding (Text, Multimodal, Images)
        # ═══════════════════════════════════════════════════════════════════
        stages[3].status = "running"
        stages[3].messages.append(f"Embedding provider: {embed_provider}")
        job.updated_at = datetime.utcnow().isoformat()
        
        embedder = get_embedder(config)
        stages[3].details["provider"] = embed_provider
        
        try:
            # 1. Embed text-only chunks
            if child_chunks:
                stages[3].messages.append(f"→ Embedding {len(child_chunks)} text chunks...")
                texts = [chunk.text for chunk in child_chunks]
                text_embeddings = await embedder.embed_texts(texts)
                for chunk, embedding in zip(child_chunks, text_embeddings):
                    chunk.embedding = embedding
                stages[3].messages.append(f"  ✓ Text embeddings complete (dim={len(text_embeddings[0]) if text_embeddings else 0})")
                stages[3].details["text_embeddings"] = len(text_embeddings)
            
            # 2. Embed multimodal chunks (combined text+image embedding)
            if multimodal_chunks and hasattr(embedder, 'embed_image'):
                stages[3].messages.append(f"→ Embedding {len(multimodal_chunks)} multimodal chunks (text+image)...")
                embedded_count = 0
                failed_count = 0
                for mm_chunk in multimodal_chunks:
                    if hasattr(mm_chunk, 'image_data') and mm_chunk.image_data:
                        try:
                            # Use combined text+image embedding
                            ocr_text = getattr(mm_chunk, 'ocr_text', mm_chunk.text)
                            combined_embedding = await embedder.embed_image(
                                mm_chunk.image_data, 
                                text=ocr_text  # Pass OCR text for combined embedding
                            )
                            # Store as both text and image embedding for maximum retrieval coverage
                            mm_chunk.embedding = combined_embedding
                            mm_chunk.image_embedding = combined_embedding
                            embedded_count += 1
                            stages[3].messages.append(f"  ✓ Page {mm_chunk.page} embedded (text+image combined)")
                        except Exception as e:
                            failed_count += 1
                            stages[3].messages.append(f"  ✗ Page {mm_chunk.page} failed: {str(e)[:50]}")
                
                stages[3].details["multimodal_embeddings"] = embedded_count
                stages[3].details["multimodal_failures"] = failed_count
            
            # 3. Embed image-only chunks
            if image_only_chunks and hasattr(embedder, 'embed_image'):
                stages[3].messages.append(f"→ Embedding {len(image_only_chunks)} image-only chunks...")
                embedded_count = 0
                failed_count = 0
                for img_chunk in image_only_chunks:
                    if hasattr(img_chunk, 'image_data') and img_chunk.image_data:
                        try:
                            img_embedding = await embedder.embed_image(img_chunk.image_data)
                            img_chunk.image_embedding = img_embedding
                            embedded_count += 1
                            stages[3].messages.append(f"  ✓ Page {img_chunk.page} embedded (image only)")
                        except Exception as e:
                            failed_count += 1
                            stages[3].messages.append(f"  ✗ Page {img_chunk.page} failed: {str(e)[:50]}")
                
                stages[3].details["image_embeddings"] = embedded_count
                stages[3].details["image_failures"] = failed_count
            elif image_only_chunks:
                stages[3].messages.append("⚠ Image chunks exist but embedder doesn't support images")
            
            stages[3].status = "completed"
            stages[3].items_processed = len(child_chunks) + len(multimodal_chunks) + len(image_only_chunks)
        # NOTE: Do NOT close embedder here as it is a singleton shared across requests
        finally:
            # await embedder.close()
            pass
        
        job.progress = 0.65
        job.updated_at = datetime.utcnow().isoformat()
        
        # NOTE: PuppyGraph refresh is triggered AFTER database storage and entity extraction
        # (see "Final Step" section) to ensure data is available when graph syncs.
        
        # ═══════════════════════════════════════════════════════════════════
        # Stage 5: Database Storage
        # ═══════════════════════════════════════════════════════════════════
        stages[4].status = "running"
        stages[4].messages.append("Connecting to PostgreSQL database...")
        job.updated_at = datetime.utcnow().isoformat()
        
        pool = await asyncpg.create_pool(config.database_url, min_size=1, max_size=5)
        stages[4].messages.append("✓ Database connection established")
        
        try:
            async with pool.acquire() as conn:
                # Insert document
                doc_id = await conn.fetchval("""
                    INSERT INTO rag_documents (
                        id, tenant_id, hash_sha256, file_name, file_path,
                        collection, title, ingestion_status, created_at, updated_at
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $9)
                    ON CONFLICT (tenant_id, hash_sha256) DO UPDATE SET
                        title = EXCLUDED.title,
                        ingestion_status = EXCLUDED.ingestion_status,
                        updated_at = EXCLUDED.updated_at
                    RETURNING id
                """, document_id, "default", document.file_hash, file_path.name,
                    str(file_path), "general", file_path.stem, "completed", datetime.utcnow())
                
                # Insert parent chunks
                parent_id_map = {}
                for idx, parent in enumerate(parent_chunks):
                    parent_db_id = await conn.fetchval("""
                        INSERT INTO rag_parent_chunks (
                            id, document_id, tenant_id, index_in_document, text,
                            token_count, page_start, section_heading, created_at
                        )
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                        RETURNING id
                    """, parent.id, doc_id, "default", idx, parent.text,
                        parent.token_count, parent.page_start, parent.section_heading,
                        datetime.utcnow())
                    parent_id_map[str(parent.id)] = parent_db_id
                
                # Insert child chunks (text and image) with embeddings
                chunks_stored = 0
                for idx, chunk in enumerate(all_chunks):
                    parent_db_id = parent_id_map.get(str(chunk.parent_id))
                    content_hash = hashlib.sha256(chunk.text.encode()).hexdigest()
                    
                    # Handle text embedding
                    embedding_str = _vector_literal(chunk.embedding) if getattr(chunk, 'embedding', None) else None
                    
                    # Handle image embedding
                    image_embedding_str = _vector_literal(chunk.image_embedding) if getattr(chunk, 'image_embedding', None) else None
                    
                    # Handle image data (store raw bytes for display)
                    image_bytes = getattr(chunk, 'image_data', None)
                    
                    persisted_id = await conn.fetchval("""
                        INSERT INTO rag_child_chunks (
                            id, parent_id, document_id, tenant_id, index_in_parent,
                            text, token_count, content_hash, embedding_1024, image_embedding_1024,
                            image_data, page, modality, created_at
                        )
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::vector, $10::vector, $11, $12, $13, $14)
                        ON CONFLICT (tenant_id, content_hash) DO UPDATE SET
                            embedding_1024 = COALESCE(EXCLUDED.embedding_1024, rag_child_chunks.embedding_1024),
                            image_embedding_1024 = COALESCE(EXCLUDED.image_embedding_1024, rag_child_chunks.image_embedding_1024),
                            image_data = COALESCE(EXCLUDED.image_data, rag_child_chunks.image_data)
                        RETURNING id
                    """, chunk.id, parent_db_id, doc_id, "default", idx,
                        chunk.text, chunk.token_count, content_hash, embedding_str, image_embedding_str,
                        image_bytes, chunk.page, chunk.modality.value, datetime.utcnow())
                    
                    # CRITICAL: Update in-memory chunk ID to match persisted ID (in case of deduplication)
                    # This ensures Entity Extraction uses the correct FK
                    if persisted_id:
                        chunk.id = persisted_id
                        
                    chunks_stored += 1
            
            stages[4].status = "completed"
            stages[4].items_processed = chunks_stored
            stages[4].messages.append(f"✓ Stored {chunks_stored} chunks in database")
            stages[4].details["chunks_stored"] = chunks_stored
            stages[4].details["document_id"] = str(doc_id)
        finally:
            await pool.close()
        
        job.progress = 0.85
        job.updated_at = datetime.utcnow().isoformat()
        
        # ═══════════════════════════════════════════════════════════════════
        # Stage 6: Entity Extraction (Optional)
        # ═══════════════════════════════════════════════════════════════════
        entities_count = 0
        relations_count = 0
        
        if config.rag_entity_extraction_enabled and len(stages) > 5:
            stages[5].status = "running"
            stages[5].messages.append("Extracting entities and relations...")
            job.updated_at = datetime.utcnow().isoformat()
            
            try:
                # Extract entities and relations from text chunks
                extractor = EntityRelationExtractor(config=config)
                chunks_to_process = [c for c in child_chunks if c.modality.value == "text"][:50]  # Process more chunks
                extraction = await extractor.extract(chunks_to_process)
                
                stages[5].messages.append(f"Extracted {len(extraction.entities)} entities, {len(extraction.relations)} relations")
                
                # CRITICAL: Store entities in database for PuppyGraph
                if not extraction.is_empty:
                    stages[5].messages.append("Storing graph data in PostgreSQL...")
                    entity_store = GraphEntityStore(config=config)
                    try:
                        store_result = await entity_store.store(
                            result=extraction,
                            chunks=chunks_to_process,
                            tenant_id="default",
                            document_id=doc_id,
                        )
                        entities_count = store_result["entities"]
                        relations_count = store_result["relations"]
                        mentions_count = store_result["mentions"]
                        stages[5].messages.append(f"✓ Stored: {entities_count} entities, {mentions_count} mentions, {relations_count} relations")
                        stages[5].details["entities"] = entities_count
                        stages[5].details["mentions"] = mentions_count
                        stages[5].details["relations"] = relations_count
                    finally:
                        await entity_store.close()
                else:
                    entities_count = 0
                    relations_count = 0
                    stages[5].messages.append("No entities extracted from this document")
                
                stages[5].status = "completed"
                stages[5].items_processed = entities_count
            except Exception as e:
                import traceback
                traceback.print_exc()
                stages[5].status = "failed"
                stages[5].error = str(e)
                stages[5].messages.append(f"Error: {str(e)}")
        
        # ═══════════════════════════════════════════════════════════════════
        # Complete
        # ═══════════════════════════════════════════════════════════════════
        job.status = "completed"
        job.progress = 1.0
        job.result = {
            "document_id": str(document_id),
            "file_type": document.file_type.value,
            "pages": len(document.pages),
            "pages_ocr": len(ocr_texts),
            "pages_images": len(image_data_for_embedding),
            "parent_chunks": len(parent_chunks),
            "child_chunks": len(all_chunks),  # text + multimodal + image-only chunks
            "text_chunks": len(child_chunks),
            "multimodal_chunks": len(multimodal_chunks),
            "image_only_chunks": len(image_only_chunks),
            "entities": entities_count,
            "relations": relations_count,
        }
        job.updated_at = datetime.utcnow().isoformat()
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        
        job.status = "failed"
        job.error = str(e)
        
        # Mark current stage as failed
        for stage in job.stages:
            if stage.status == "running":
                stage.status = "failed"
                stage.error = str(e)
                break
        
        job.updated_at = datetime.utcnow().isoformat()

@app.post("/api/ingest/upload")
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload a file for multimodal ingestion."""
    # Validate file type
    suffix = Path(file.filename).suffix.lower()
    supported = {".pdf", ".docx", ".doc", ".xlsx", ".xls", ".csv", ".txt", ".md", 
                 ".png", ".jpg", ".jpeg", ".webp", ".tiff", ".tif", ".bmp"}
    
    if suffix not in supported:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Supported: {', '.join(supported)}"
        )
    
    # Generate unique filename (preserve original name with UUID prefix)
    unique_filename = f"{uuid.uuid4()}_{file.filename}"
    
    # Save to temp upload dir for processing
    file_path = UPLOAD_DIR / unique_filename
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    # Also copy to persistent storage for later download
    persistent_path = DOCUMENT_STORAGE_DIR / unique_filename
    shutil.copy2(file_path, persistent_path)
    
    # Detect file type for job
    from triple_hybrid_rag.ingestion.loaders import detect_file_type
    file_type = detect_file_type(file_path)
    
    # Create job
    job_id = str(uuid.uuid4())
    job = IngestionJob(
        job_id=job_id,
        status="pending",
        file_name=unique_filename,  # Store unique filename for later retrieval
        file_type=file_type.value,
        progress=0.0,
        stages=[],
        created_at=datetime.utcnow().isoformat(),
        updated_at=datetime.utcnow().isoformat(),
    )
    ingestion_jobs[job_id] = job
    
    # Start background ingestion
    background_tasks.add_task(run_ingestion, job_id, file_path)
    
    return {"job_id": job_id, "status": "pending", "file_type": file_type.value, "file_name": file.filename}

@app.get("/api/ingest/status/{job_id}")
async def get_ingestion_status(job_id: str):
    """Get ingestion job status."""
    if job_id not in ingestion_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = ingestion_jobs[job_id]
    return {
        "job_id": job.job_id,
        "status": job.status,
        "file_name": job.file_name,
        "file_type": job.file_type,
        "progress": job.progress,
        "stages": [{
            "name": s.name, 
            "status": s.status, 
            "items_processed": s.items_processed, 
            "error": s.error,
            "messages": s.messages,
            "details": s.details,
        } for s in job.stages],
        "result": job.result,
        "error": job.error,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
    }

@app.get("/api/ingest/jobs")
async def list_ingestion_jobs():
    """List all ingestion jobs."""
    return {
        "jobs": [
            {
                "job_id": job.job_id,
                "status": job.status,
                "file_name": job.file_name,
                "file_type": job.file_type,
                "progress": job.progress,
                "stages": [{
                    "name": s.name, 
                    "status": s.status, 
                    "items_processed": s.items_processed, 
                    "error": s.error,
                    "messages": s.messages,
                    "details": s.details,
                } for s in job.stages],
                "result": job.result,
                "error": job.error,
                "created_at": job.created_at,
                "updated_at": job.updated_at,
            }
            for job in ingestion_jobs.values()
        ]
    }

# ═══════════════════════════════════════════════════════════════════════════════
# RETRIEVAL
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/api/retrieve")
async def retrieve(request: QueryRequest):
    """Execute triple-hybrid retrieval query."""
    config = get_settings()
    
    try:
        import asyncpg
        
        # Embed query
        embedder = get_embedder(config)
        # Use query-specific embedding if Jina provider
        if hasattr(embedder, 'embed_query'):
            query_embedding = await embedder.embed_query(request.query)
        else:
            query_embedding = await embedder.embed_text(request.query)
        await embedder.close()
        
        pool = await asyncpg.create_pool(config.database_url, min_size=1, max_size=5)
        
        semantic_results = []
        lexical_results = []
        graph_results = []
        
        try:
            async with pool.acquire() as conn:
                # Semantic search
                if config.rag_semantic_enabled:
                    embedding_str = _vector_literal(query_embedding)
                    semantic_results = await conn.fetch("""
                        SELECT * FROM rag_semantic_search($1, $2::vector, $3, $4)
                    """, request.tenant_id, embedding_str, 
                        request.top_k or config.rag_semantic_top_k, request.collection)
                
                # Lexical search
                if config.rag_lexical_enabled:
                    lexical_results = await conn.fetch("""
                        SELECT * FROM rag_lexical_search($1, $2, $3, $4)
                    """, request.tenant_id, request.query,
                        request.top_k or config.rag_lexical_top_k, request.collection)
            
            # Graph search (using PostgreSQL)
            if config.rag_graph_enabled:
                try:
                    from triple_hybrid_rag.graph.sql_fallback import SQLGraphFallback
                    graph_search = SQLGraphFallback(pool)
                    keywords = request.query.split()[:5]  # Simple keyword extraction
                    graph_results = await graph_search.search_by_keywords(
                        keywords=keywords,
                        tenant_id=request.tenant_id,
                        limit=config.rag_graph_top_k,
                    )
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).warning(f"Graph search error: {e}")
                    graph_results = []
            
            # Convert to SearchResult objects
            def row_to_search_result(row, channel: SearchChannel, score_field: str) -> SearchResult:
                result = SearchResult(
                    chunk_id=row['child_id'],
                    parent_id=row['parent_id'],
                    document_id=row['document_id'],
                    text=row['text'],
                    page=row.get('page'),
                    modality=Modality.TEXT,
                    source_channel=channel,
                )
                if channel == SearchChannel.SEMANTIC:
                    result.semantic_score = float(row.get(score_field, 0.0))
                elif channel == SearchChannel.LEXICAL:
                    result.lexical_score = float(row.get(score_field, 0.0))
                return result
            
            semantic_search_results = [
                row_to_search_result(r, SearchChannel.SEMANTIC, 'similarity')
                for r in semantic_results
            ]
            lexical_search_results = [
                row_to_search_result(r, SearchChannel.LEXICAL, 'rank')
                for r in lexical_results
            ]
            
            # Fusion
            fusion = RRFFusion(config=config)
            fused = fusion.fuse(
                lexical_results=lexical_search_results,
                semantic_results=semantic_search_results,
                graph_results=graph_results if isinstance(graph_results, list) else [],
                top_k=request.top_k or config.rag_final_top_k,
                apply_safety=False,
                apply_denoise=config.rag_denoise_enabled,
            )
            
            # Reranking
            if config.rag_rerank_enabled and fused:
                reranker = get_reranker(config)
                documents = [r.text for r in fused]
                scores = await reranker.rerank(request.query, documents)
                await reranker.close()
                
                for result, score in zip(fused, scores):
                    result.rerank_score = score
                    result.final_score = score
                
                fused.sort(key=lambda r: r.final_score, reverse=True)
            
            # Get parent context
            parent_texts = {}
            if fused:
                parent_ids = list({str(r.parent_id) for r in fused if r.parent_id})
                if parent_ids:
                    async with pool.acquire() as conn:
                        rows = await conn.fetch("""
                            SELECT id, text FROM rag_parent_chunks WHERE id = ANY($1::uuid[])
                        """, [uuid.UUID(pid) for pid in parent_ids])
                        parent_texts = {str(row['id']): row['text'] for row in rows}
            
            return {
                "query": request.query,
                "results": [
                    {
                        "chunk_id": str(r.chunk_id),
                        "document_id": str(r.document_id),
                        "text": r.text,
                        "parent_text": parent_texts.get(str(r.parent_id)),
                        "lexical_score": _safe_float(r.lexical_score),
                        "semantic_score": _safe_float(r.semantic_score),
                        "graph_score": _safe_float(r.graph_score),
                        "rrf_score": _safe_float(r.rrf_score),
                        "rerank_score": _safe_float(r.rerank_score),
                        "final_score": _safe_float(r.final_score),
                        "metadata": r.metadata,
                    }
                    for r in fused
                ],
                "total_results": len(fused),
                "channels": {
                    "semantic": len(semantic_results),
                    "lexical": len(lexical_results),
                    "graph": len(graph_results) if graph_results else 0,
                },
            }
            
        finally:
            await pool.close()
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE BROWSING
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/database/stats")
async def get_database_stats():
    """Get database statistics."""
    import asyncpg
    config = get_settings()
    
    try:
        conn = await asyncpg.connect(config.database_url)
        
        # Get counts from correct table names
        docs = await conn.fetchval("SELECT COUNT(*) FROM rag_documents")
        parents = await conn.fetchval("SELECT COUNT(*) FROM rag_parent_chunks")
        children = await conn.fetchval("SELECT COUNT(*) FROM rag_child_chunks")
        entities = await conn.fetchval("SELECT COUNT(*) FROM rag_entities")
        relations = await conn.fetchval("SELECT COUNT(*) FROM rag_relations")
        
        await conn.close()
        
        return {
            "documents": docs or 0,
            "parent_chunks": parents or 0,
            "child_chunks": children or 0,
            "entities": entities or 0,
            "relations": relations or 0,
        }
    except Exception as e:
        return {
            "documents": 0,
            "parent_chunks": 0,
            "child_chunks": 0,
            "entities": 0,
            "relations": 0,
            "error": str(e),
        }

@app.get("/api/database/documents")
async def list_documents(limit: int = 50, offset: int = 0):
    """List documents."""
    import asyncpg
    config = get_settings()
    
    try:
        conn = await asyncpg.connect(config.database_url)
        
        rows = await conn.fetch(
            """
            SELECT d.id, d.tenant_id, d.file_name, d.collection, d.title, 
                   d.ingestion_status, d.created_at, d.updated_at,
                   (SELECT COUNT(*) FROM rag_child_chunks c WHERE c.document_id = d.id) as chunk_count
            FROM rag_documents d
            ORDER BY d.created_at DESC
            LIMIT $1 OFFSET $2
            """,
            limit, offset
        )
        
        await conn.close()
        
        def check_file_available(file_name: str) -> bool:
            """Check if file is available for download."""
            if not file_name:
                return False
            stored_file = DOCUMENT_STORAGE_DIR / file_name
            return stored_file.exists()
        
        return {
            "documents": [
                {
                    "id": str(row["id"]),
                    "tenant_id": row["tenant_id"],
                    "file_name": row["file_name"],
                    "collection": row["collection"],
                    "title": row["title"],
                    "status": row["ingestion_status"],
                    "chunk_count": row["chunk_count"] or 0,
                    "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                    "has_file": check_file_available(row["file_name"]),
                    "download_url": f"/api/documents/{row['id']}/download" if row["file_name"] else None,
                }
                for row in rows
            ]
        }
    except Exception as e:
        return {"documents": [], "error": str(e)}

@app.delete("/api/database/documents/{document_id}")
async def delete_document(document_id: str):
    """
    Delete a document and all related data (cascade delete).
    
    Deletes:
    - Document record
    - All parent chunks
    - All child chunks (with embeddings)
    - All entity mentions linked to those chunks
    - All relations linked to those entities (if no other mentions)
    - The stored original file (if exists)
    """
    import asyncpg
    config = get_settings()
    
    try:
        doc_uuid = uuid.UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID format")
    
    try:
        conn = await asyncpg.connect(config.database_url)
        
        # Get document info first (for file path)
        doc = await conn.fetchrow(
            "SELECT file_name, file_path FROM rag_documents WHERE id = $1",
            doc_uuid
        )
        
        if not doc:
            await conn.close()
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Get all child chunk IDs for this document
        chunk_ids = await conn.fetch(
            "SELECT id FROM rag_child_chunks WHERE document_id = $1",
            doc_uuid
        )
        chunk_id_list = [row['id'] for row in chunk_ids]
        
        # Delete entity mentions for these chunks
        if chunk_id_list:
            await conn.execute(
                "DELETE FROM rag_entity_mentions WHERE child_chunk_id = ANY($1::uuid[])",
                chunk_id_list
            )
        
        # Delete child chunks
        result = await conn.execute(
            "DELETE FROM rag_child_chunks WHERE document_id = $1",
            doc_uuid
        )
        deleted_children = int(result.split()[-1]) if result else 0
        
        # Delete parent chunks
        result = await conn.execute(
            "DELETE FROM rag_parent_chunks WHERE document_id = $1",
            doc_uuid
        )
        deleted_parents = int(result.split()[-1]) if result else 0
        
        # Delete document
        await conn.execute(
            "DELETE FROM rag_documents WHERE id = $1",
            doc_uuid
        )
        
        # Clean up orphaned entities (entities with no mentions)
        result = await conn.execute("""
            DELETE FROM rag_entities e
            WHERE NOT EXISTS (
                SELECT 1 FROM rag_entity_mentions m WHERE m.entity_id = e.id
            )
        """)
        deleted_entities = int(result.split()[-1]) if result else 0
        
        await conn.close()
        
        # Delete stored file if exists
        file_deleted = False
        if doc['file_name']:
            # Check in persistent storage
            stored_file = DOCUMENT_STORAGE_DIR / doc['file_name']
            if stored_file.exists():
                stored_file.unlink()
                file_deleted = True
            
            # Also check temp upload dir
            if doc['file_path']:
                temp_file = Path(doc['file_path'])
                if temp_file.exists():
                    temp_file.unlink()
                    file_deleted = True

        return {
            "status": "deleted",
            "document_id": document_id,
            "deleted": {
                "parent_chunks": deleted_parents or 0,
                "child_chunks": deleted_children or 0,
                "orphaned_entities": deleted_entities or 0,
                "file_deleted": file_deleted,
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/database/documents/batch")
async def delete_documents_batch(document_ids: List[str]):
    """Delete multiple documents by ID."""
    results = []
    for doc_id in document_ids:
        try:
            result = await delete_document(doc_id)
            results.append({"id": doc_id, "status": "deleted"})
        except HTTPException as e:
            results.append({"id": doc_id, "status": "error", "error": e.detail})
        except Exception as e:
            results.append({"id": doc_id, "status": "error", "error": str(e)})
    
    return {"results": results}


@app.get("/api/documents/{document_id}/download")
async def download_document(document_id: str):
    """Download the original uploaded document."""
    import asyncpg
    config = get_settings()
    
    try:
        doc_uuid = uuid.UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID format")
    
    try:
        conn = await asyncpg.connect(config.database_url)
        
        doc = await conn.fetchrow(
            "SELECT file_name, file_path FROM rag_documents WHERE id = $1",
            doc_uuid
        )
        await conn.close()
        
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Check persistent storage first
        if doc['file_name']:
            stored_file = DOCUMENT_STORAGE_DIR / doc['file_name']
            if stored_file.exists():
                return FileResponse(
                    path=str(stored_file),
                    filename=doc['file_name'],
                    media_type="application/octet-stream"
                )
        
        # Check temp upload path
        if doc['file_path']:
            temp_file = Path(doc['file_path'])
            if temp_file.exists():
                return FileResponse(
                    path=str(temp_file),
                    filename=doc['file_name'] or temp_file.name,
                    media_type="application/octet-stream"
                )
        
        raise HTTPException(status_code=404, detail="Document file not found on disk")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/database/documents/{document_id}/details")
async def get_document_details(document_id: str):
    """
    Get complete document details including all chunks, entities, and relations.
    Used for visual validation of ingestion.
    """
    import asyncpg
    config = get_settings()
    
    try:
        doc_uuid = uuid.UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID format")
    
    try:
        conn = await asyncpg.connect(config.database_url)
        
        # Get document info
        doc = await conn.fetchrow("""
            SELECT id, tenant_id, file_name, collection, title, 
                   ingestion_status, created_at, updated_at
            FROM rag_documents WHERE id = $1
        """, doc_uuid)
        
        if not doc:
            await conn.close()
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Get parent chunks
        parent_chunks = await conn.fetch("""
            SELECT id, index_in_document, text, token_count, page_start, section_heading
            FROM rag_parent_chunks 
            WHERE document_id = $1 
            ORDER BY index_in_document
        """, doc_uuid)
        
        # Get child chunks (with has_image flag for frontend)
        child_chunks = await conn.fetch("""
            SELECT c.id, c.parent_id, c.index_in_parent, c.text, c.token_count, 
                   c.page, c.modality, c.content_hash,
                   (c.image_data IS NOT NULL) AS has_image
            FROM rag_child_chunks c
            WHERE c.document_id = $1 
            ORDER BY c.page, c.index_in_parent
        """, doc_uuid)
        
        # Get entity mentions for this document's chunks
        chunk_ids = [row['id'] for row in child_chunks]
        entities_data = []
        relations_data = []
        
        if chunk_ids:
            # Get entities with their mentions in this document
            entities_rows = await conn.fetch("""
                SELECT DISTINCT e.id, e.name, e.entity_type, 
                       m.child_chunk_id, m.char_start, m.char_end, m.confidence
                FROM rag_entities e
                JOIN rag_entity_mentions m ON m.entity_id = e.id
                WHERE m.child_chunk_id = ANY($1::uuid[])
                ORDER BY e.entity_type, e.name
            """, chunk_ids)
            
            # Group entities
            entity_map = {}
            for row in entities_rows:
                eid = str(row['id'])
                if eid not in entity_map:
                    entity_map[eid] = {
                        'id': eid,
                        'name': row['name'],
                        'entity_type': row['entity_type'],
                        'mentions': []
                    }
                entity_map[eid]['mentions'].append({
                    'chunk_id': str(row['child_chunk_id']),
                    'start_char': row['char_start'],
                    'end_char': row['char_end'],
                    'confidence': float(row['confidence']) if row['confidence'] else None
                })
            entities_data = list(entity_map.values())
            
            # Get relations between entities in this document
            entity_ids = [uuid.UUID(eid) for eid in entity_map.keys()]
            if entity_ids:
                relations_rows = await conn.fetch("""
                    SELECT r.id, r.subject_entity_id, r.object_entity_id, 
                           r.relation_type, r.confidence,
                           se.name as source_name, se.entity_type as source_type,
                           te.name as target_name, te.entity_type as target_type
                    FROM rag_relations r
                    JOIN rag_entities se ON se.id = r.subject_entity_id
                    JOIN rag_entities te ON te.id = r.object_entity_id
                    WHERE r.subject_entity_id = ANY($1::uuid[]) 
                       OR r.object_entity_id = ANY($1::uuid[])
                    ORDER BY r.relation_type
                """, entity_ids)
                
                relations_data = [
                    {
                        'id': str(row['id']),
                        'source': {
                            'id': str(row['subject_entity_id']),
                            'name': row['source_name'],
                            'type': row['source_type']
                        },
                        'target': {
                            'id': str(row['object_entity_id']),
                            'name': row['target_name'],
                            'type': row['target_type']
                        },
                        'relation_type': row['relation_type'],
                        'confidence': float(row['confidence']) if row['confidence'] else None
                    }
                    for row in relations_rows
                ]
        
        await conn.close()
        
        return {
            'document': {
                'id': str(doc['id']),
                'tenant_id': doc['tenant_id'],
                'file_name': doc['file_name'],
                'collection': doc['collection'],
                'title': doc['title'],
                'status': doc['ingestion_status'],
                'created_at': doc['created_at'].isoformat() if doc['created_at'] else None,
            },
            'parent_chunks': [
                {
                    'id': str(row['id']),
                    'index': row['index_in_document'],
                    'text': row['text'],
                    'token_count': row['token_count'],
                    'page_start': row['page_start'],
                    'section_heading': row['section_heading'],
                }
                for row in parent_chunks
            ],
            'child_chunks': [
                {
                    'id': str(row['id']),
                    'parent_id': str(row['parent_id']) if row['parent_id'] else None,
                    'index': row['index_in_parent'],
                    'text': row['text'],
                    'token_count': row['token_count'],
                    'page': row['page'],
                    'modality': row['modality'],
                    'has_image': row['has_image'],
                }
                for row in child_chunks
            ],
            'entities': entities_data,
            'relations': relations_data,
            'stats': {
                'parent_chunks': len(parent_chunks),
                'child_chunks': len(child_chunks),
                'entities': len(entities_data),
                'relations': len(relations_data),
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/chunks/{chunk_id}/image")
async def get_chunk_image(chunk_id: str):
    """
    Serve the image data for a chunk (multimodal image chunks).
    Returns the raw image bytes as an image response.
    """
    import asyncpg
    from fastapi.responses import Response
    
    config = get_settings()
    
    try:
        chunk_uuid = uuid.UUID(chunk_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid chunk ID format")
    
    try:
        conn = await asyncpg.connect(config.database_url)
        
        row = await conn.fetchrow("""
            SELECT image_data, modality FROM rag_child_chunks WHERE id = $1
        """, chunk_uuid)
        
        await conn.close()
        
        if not row:
            raise HTTPException(status_code=404, detail="Chunk not found")
        
        if not row['image_data']:
            raise HTTPException(status_code=404, detail="No image data for this chunk")
        
        # Return image as PNG (most PDF page images are PNG)
        return Response(
            content=bytes(row['image_data']),
            media_type="image/png",
            headers={"Cache-Control": "max-age=3600"}  # Cache for 1 hour
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/database/entities")
async def list_entities(limit: int = 50, offset: int = 0, entity_type: Optional[str] = None):
    """List entities."""
    import asyncpg
    config = get_settings()
    
    try:
        conn = await asyncpg.connect(config.database_url)
        
        if entity_type:
            rows = await conn.fetch(
                """
                SELECT e.id, e.tenant_id, e.name, e.entity_type, e.created_at,
                       (SELECT COUNT(*) FROM rag_entity_mentions m WHERE m.entity_id = e.id) as mention_count
                FROM rag_entities e
                WHERE e.entity_type = $3
                ORDER BY mention_count DESC
                LIMIT $1 OFFSET $2
                """,
                limit, offset, entity_type
            )
        else:
            rows = await conn.fetch(
                """
                SELECT e.id, e.tenant_id, e.name, e.entity_type, e.created_at,
                       (SELECT COUNT(*) FROM rag_entity_mentions m WHERE m.entity_id = e.id) as mention_count
                FROM rag_entities e
                ORDER BY mention_count DESC
                LIMIT $1 OFFSET $2
                """,
                limit, offset
            )
        
        await conn.close()
        
        return {
            "entities": [
                {
                    "id": str(row["id"]),
                    "name": row["name"],
                    "entity_type": row["entity_type"],
                    "mention_count": row["mention_count"] or 0,
                }
                for row in rows
            ]
        }
    except Exception as e:
        return {"entities": [], "error": str(e)}

# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/graph/data")
async def get_graph_data(limit: int = 500):
    """
    Get all entities and relations for graph visualization.
    Returns nodes (entities) and edges (relations) in a format suitable for force-directed graphs.
    """
    import asyncpg
    config = get_settings()
    
    try:
        conn = await asyncpg.connect(config.database_url)
        
        # Get entities (nodes) with mention counts
        entities = await conn.fetch("""
            SELECT e.id, e.name, e.entity_type,
                   (SELECT COUNT(*) FROM rag_entity_mentions m WHERE m.entity_id = e.id) as mention_count
            FROM rag_entities e
            ORDER BY mention_count DESC
            LIMIT $1
        """, limit)
        
        # Get entity IDs for filtering relations
        entity_ids = [row['id'] for row in entities]
        
        # Get relations (edges) between these entities
        relations = []
        if entity_ids:
            relations = await conn.fetch("""
                SELECT r.id, r.subject_entity_id, r.object_entity_id, r.relation_type, r.confidence
                FROM rag_relations r
                WHERE r.subject_entity_id = ANY($1::uuid[])
                  AND r.object_entity_id = ANY($1::uuid[])
            """, entity_ids)
        
        await conn.close()
        
        # Build nodes list
        nodes = [
            {
                "id": str(row["id"]),
                "name": row["name"],
                "type": row["entity_type"],
                "mentions": row["mention_count"] or 0,
            }
            for row in entities
        ]
        
        # Build edges list
        edges = [
            {
                "id": str(row["id"]),
                "source": str(row["subject_entity_id"]),
                "target": str(row["object_entity_id"]),
                "type": row["relation_type"],
                "confidence": float(row["confidence"]) if row["confidence"] else None,
            }
            for row in relations
        ]
        
        # Get unique entity types for filtering
        entity_types = sorted(set(row["entity_type"] for row in entities))
        
        return {
            "nodes": nodes,
            "edges": edges,
            "stats": {
                "total_nodes": len(nodes),
                "total_edges": len(edges),
                "entity_types": entity_types,
            }
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"nodes": [], "edges": [], "stats": {"total_nodes": 0, "total_edges": 0, "entity_types": []}, "error": str(e)}


@app.get("/api/graph/entity/{entity_id}")
async def get_entity_details(entity_id: str):
    """
    Get details of a specific entity including all its relations.
    Used when clicking on a node in the graph visualization.
    """
    import asyncpg
    config = get_settings()
    
    try:
        entity_uuid = uuid.UUID(entity_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid entity ID format")
    
    try:
        conn = await asyncpg.connect(config.database_url)
        
        # Get entity info
        entity = await conn.fetchrow("""
            SELECT e.id, e.name, e.entity_type, e.tenant_id
            FROM rag_entities e
            WHERE e.id = $1
        """, entity_uuid)
        
        if not entity:
            await conn.close()
            raise HTTPException(status_code=404, detail="Entity not found")
        
        # Get mention count and sample mentions
        mentions = await conn.fetch("""
            SELECT m.child_chunk_id, m.char_start, m.char_end, m.confidence,
                   c.text as chunk_text, d.file_name
            FROM rag_entity_mentions m
            JOIN rag_child_chunks c ON c.id = m.child_chunk_id
            JOIN rag_documents d ON d.id = c.document_id
            WHERE m.entity_id = $1
            LIMIT 10
        """, entity_uuid)
        
        # Get outgoing relations (this entity is subject)
        outgoing = await conn.fetch("""
            SELECT r.id, r.relation_type, r.confidence,
                   te.id as target_id, te.name as target_name, te.entity_type as target_type
            FROM rag_relations r
            JOIN rag_entities te ON te.id = r.object_entity_id
            WHERE r.subject_entity_id = $1
        """, entity_uuid)
        
        # Get incoming relations (this entity is object)
        incoming = await conn.fetch("""
            SELECT r.id, r.relation_type, r.confidence,
                   se.id as source_id, se.name as source_name, se.entity_type as source_type
            FROM rag_relations r
            JOIN rag_entities se ON se.id = r.subject_entity_id
            WHERE r.object_entity_id = $1
        """, entity_uuid)
        
        await conn.close()
        
        return {
            "entity": {
                "id": str(entity["id"]),
                "name": entity["name"],
                "type": entity["entity_type"],
                "tenant_id": entity["tenant_id"],
            },
            "mentions": [
                {
                    "chunk_id": str(row["child_chunk_id"]),
                    "char_start": row["char_start"],
                    "char_end": row["char_end"],
                    "confidence": float(row["confidence"]) if row["confidence"] else None,
                    "text_snippet": row["chunk_text"][:200] + "..." if row["chunk_text"] and len(row["chunk_text"]) > 200 else row["chunk_text"],
                    "document": row["file_name"],
                }
                for row in mentions
            ],
            "relations": {
                "outgoing": [
                    {
                        "id": str(row["id"]),
                        "type": row["relation_type"],
                        "confidence": float(row["confidence"]) if row["confidence"] else None,
                        "target": {
                            "id": str(row["target_id"]),
                            "name": row["target_name"],
                            "type": row["target_type"],
                        }
                    }
                    for row in outgoing
                ],
                "incoming": [
                    {
                        "id": str(row["id"]),
                        "type": row["relation_type"],
                        "confidence": float(row["confidence"]) if row["confidence"] else None,
                        "source": {
                            "id": str(row["source_id"]),
                            "name": row["source_name"],
                            "type": row["source_type"],
                        }
                    }
                    for row in incoming
                ],
            },
            "stats": {
                "mention_count": len(mentions),
                "outgoing_relations": len(outgoing),
                "incoming_relations": len(incoming),
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ═══════════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/metrics")
async def get_metrics():
    """Get pipeline metrics."""
    config = get_settings()
    
    # Get database stats
    stats = await get_database_stats()
    
    return {
        "database": stats,
        "config": {
            "rag_enabled": config.rag_enabled,
            "lexical_enabled": config.rag_lexical_enabled,
            "semantic_enabled": config.rag_semantic_enabled,
            "graph_enabled": config.rag_graph_enabled,
            "rerank_enabled": config.rag_rerank_enabled,
            "hyde_enabled": config.rag_hyde_enabled,
            "query_expansion_enabled": config.rag_query_expansion_enabled,
            "diversity_enabled": config.rag_diversity_enabled,
            "ocr_enabled": config.rag_ocr_enabled,
        },
        "ingestion_jobs": {
            "total": len(ingestion_jobs),
            "pending": sum(1 for j in ingestion_jobs.values() if j.status == "pending"),
            "processing": sum(1 for j in ingestion_jobs.values() if j.status == "processing"),
            "completed": sum(1 for j in ingestion_jobs.values() if j.status == "completed"),
            "failed": sum(1 for j in ingestion_jobs.values() if j.status == "failed"),
        },
    }

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    from pathlib import Path
    
    # SSL certificate paths
    certs_dir = Path(__file__).parent.parent / "certs"
    ssl_keyfile = certs_dir / "key.pem"
    ssl_certfile = certs_dir / "cert.pem"
    
    # Check if SSL certificates exist
    if ssl_keyfile.exists() and ssl_certfile.exists():
        print(f"🔒 Starting HTTPS server with certificates from {certs_dir}")
        uvicorn.run(
            "dashboard.backend.main:app",
            host="0.0.0.0",
            port=8009,
            ssl_keyfile=str(ssl_keyfile),
            ssl_certfile=str(ssl_certfile),
            reload=True,
        )
    else:
        print("⚠️  SSL certificates not found, starting HTTP server")
        print(f"   Generate certificates in {certs_dir} for HTTPS support")
        uvicorn.run("dashboard.backend.main:app", host="0.0.0.0", port=8009, reload=True)
    # Trigger reload for timeout boost
