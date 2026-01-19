"""
Dashboard Backend - Triple-Hybrid-RAG

FastAPI backend for the RAG dashboard supporting:
- Multimodal file ingestion (PDF, DOCX, XLSX, CSV, images)
- OCR processing (Qwen3-VL or DeepSeek)
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
from triple_hybrid_rag.ingestion.ocr import OCRProcessor, OCRIngestionMode, resolve_ocr_mode
from triple_hybrid_rag.core.chunker import HierarchicalChunker
from triple_hybrid_rag.core.embedder import get_embedder
from triple_hybrid_rag.core.entity_extractor import EntityRelationExtractor
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
        "puppygraph_url": config.puppygraph_web_ui_url,
        "ocr_mode": config.rag_ocr_mode,
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
        "rag_graph_enabled": {"category": "Feature Flags", "type": "boolean", "description": "Enable graph (PuppyGraph) search"},
        "rag_rerank_enabled": {"category": "Feature Flags", "type": "boolean", "description": "Enable reranking"},
        "rag_denoise_enabled": {"category": "Feature Flags", "type": "boolean", "description": "Enable conformal denoising"},
        "rag_query_planner_enabled": {"category": "Feature Flags", "type": "boolean", "description": "Enable query planner"},
        "rag_entity_extraction_enabled": {"category": "Feature Flags", "type": "boolean", "description": "Enable entity extraction"},
        "rag_ocr_enabled": {"category": "Feature Flags", "type": "boolean", "description": "Enable OCR for images"},
        "rag_multimodal_embedding_enabled": {"category": "Feature Flags", "type": "boolean", "description": "Enable multimodal embeddings"},
        
        # OCR Settings
        "rag_ocr_mode": {"category": "OCR", "type": "string", "description": "OCR mode: qwen, deepseek, off, auto"},
        "rag_deepseek_ocr_enabled": {"category": "OCR", "type": "boolean", "description": "Use DeepSeek OCR"},
        "rag_gundam_tiling_enabled": {"category": "OCR", "type": "boolean", "description": "Enable Gundam Tiling for large images"},
        
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
        "rag_image_ingestion_mode": {"category": "Jina AI", "type": "string", "description": "Image mode: ocr, direct, or auto"},
        
        # Database
        "database_url": {"category": "Database", "type": "string", "description": "PostgreSQL connection URL"},
        "database_pool_size": {"category": "Database", "type": "integer", "description": "Connection pool size", "min": 1, "max": 100},
        
        # PuppyGraph
        "puppygraph_bolt_url": {"category": "PuppyGraph", "type": "string", "description": "PuppyGraph Bolt protocol URL"},
        "puppygraph_web_ui_url": {"category": "PuppyGraph", "type": "string", "description": "PuppyGraph Web UI URL"},
        
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
    
    # Reload settings
    reset_settings()
    
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
        job.progress = 0.05
        job.updated_at = datetime.utcnow().isoformat()
        
        loader = DocumentLoader(pdf_dpi=300, extract_tables=True)
        document = loader.load(file_path)
        
        if document.error:
            raise Exception(f"Loading failed: {document.error}")
        
        job.file_type = document.file_type.value
        stages[0].status = "completed"
        stages[0].items_processed = len(document.pages)
        job.progress = 0.15
        job.updated_at = datetime.utcnow().isoformat()
        
        # ═══════════════════════════════════════════════════════════════════
        # Stage 2: OCR Processing
        # ═══════════════════════════════════════════════════════════════════
        stages[1].status = "running"
        job.updated_at = datetime.utcnow().isoformat()
        
        # Find pages needing OCR
        pages_needing_ocr = [
            (i, page) for i, page in enumerate(document.pages)
            if page.is_scanned or (page.has_images and page.image_data)
        ]
        
        ocr_texts: Dict[int, str] = {}
        
        if pages_needing_ocr and config.rag_ocr_enabled:
            ocr_mode, _ = resolve_ocr_mode(
                mode=config.rag_ocr_mode,
                document=document,
                file_path=str(file_path),
                config=config,
            )
            
            if ocr_mode != OCRIngestionMode.OFF:
                ocr_processor = OCRProcessor.create_for_mode(
                    ingestion_mode=ocr_mode,
                    settings=config,
                )
                
                for page_idx, page in pages_needing_ocr:
                    if page.image_data:
                        result = await ocr_processor.process_image(page.image_data)
                        if result.text and not result.error:
                            ocr_texts[page_idx] = result.text
                
                stages[1].items_processed = len(ocr_texts)
        
        stages[1].status = "completed"
        job.progress = 0.30
        job.updated_at = datetime.utcnow().isoformat()
        
        # ═══════════════════════════════════════════════════════════════════
        # Stage 3: Text Aggregation & Chunking
        # ═══════════════════════════════════════════════════════════════════
        stages[2].status = "running"
        job.updated_at = datetime.utcnow().isoformat()
        
        # Aggregate text from all sources
        text_parts = []
        for i, page in enumerate(document.pages):
            if page.text.strip():
                text_parts.append(f"[Page {page.page_number}]\n{page.text}")
            
            # Add OCR text
            if i in ocr_texts:
                text_parts.append(f"[Page {page.page_number} - OCR]\n{ocr_texts[i]}")
            
            # Add tables
            for table in page.tables:
                text_parts.append(f"[Table - Page {page.page_number}]\n{table}")
        
        full_text = "\n\n".join(text_parts)
        
        if not full_text.strip():
            raise Exception("No text content extracted from document")
        
        # Chunking
        document_id = uuid4()
        chunker = HierarchicalChunker(config=config)
        parent_chunks, child_chunks = chunker.split_document(
            text=full_text,
            document_id=document_id,
            tenant_id="default",
        )
        
        stages[2].status = "completed"
        stages[2].items_processed = len(child_chunks)
        job.progress = 0.45
        job.updated_at = datetime.utcnow().isoformat()
        
        # ═══════════════════════════════════════════════════════════════════
        # Stage 4: Embedding
        # ═══════════════════════════════════════════════════════════════════
        stages[3].status = "running"
        job.updated_at = datetime.utcnow().isoformat()
        
        embedder = get_embedder(config)
        texts = [chunk.text for chunk in child_chunks]
        
        try:
            embeddings = await embedder.embed_texts(texts)
            
            for chunk, embedding in zip(child_chunks, embeddings):
                chunk.embedding = embedding
            
            stages[3].status = "completed"
            stages[3].items_processed = len(embeddings)
        finally:
            await embedder.close()
        
        job.progress = 0.65
        job.updated_at = datetime.utcnow().isoformat()
        
        # ═══════════════════════════════════════════════════════════════════
        # Stage 5: Database Storage
        # ═══════════════════════════════════════════════════════════════════
        stages[4].status = "running"
        job.updated_at = datetime.utcnow().isoformat()
        
        pool = await asyncpg.create_pool(config.database_url, min_size=1, max_size=5)
        
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
                
                # Insert child chunks with embeddings
                chunks_stored = 0
                for idx, chunk in enumerate(child_chunks):
                    parent_db_id = parent_id_map.get(str(chunk.parent_id))
                    content_hash = hashlib.sha256(chunk.text.encode()).hexdigest()
                    
                    embedding_str = _vector_literal(chunk.embedding) if chunk.embedding else None
                    
                    await conn.execute("""
                        INSERT INTO rag_child_chunks (
                            id, parent_id, document_id, tenant_id, index_in_parent,
                            text, token_count, content_hash, embedding_1024,
                            page, modality, created_at
                        )
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::vector, $10, $11, $12)
                        ON CONFLICT (tenant_id, content_hash) DO UPDATE SET
                            embedding_1024 = EXCLUDED.embedding_1024
                    """, chunk.id, parent_db_id, doc_id, "default", idx,
                        chunk.text, chunk.token_count, content_hash, embedding_str,
                        chunk.page, chunk.modality.value, datetime.utcnow())
                    chunks_stored += 1
            
            stages[4].status = "completed"
            stages[4].items_processed = chunks_stored
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
            job.updated_at = datetime.utcnow().isoformat()
            
            try:
                extractor = EntityRelationExtractor(config=config)
                extraction = await extractor.extract(child_chunks[:10])  # Limit for speed
                
                entities_count = len(extraction.entities)
                relations_count = len(extraction.relations)
                
                stages[5].status = "completed"
                stages[5].items_processed = entities_count
            except Exception as e:
                stages[5].status = "failed"
                stages[5].error = str(e)
        
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
            "parent_chunks": len(parent_chunks),
            "child_chunks": len(child_chunks),
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
        "stages": [{"name": s.name, "status": s.status, "items_processed": s.items_processed, "error": s.error} for s in job.stages],
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
                "stages": [{"name": s.name, "status": s.status, "items_processed": s.items_processed, "error": s.error} for s in job.stages],
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
            
            # Graph search (with fallback)
            if config.rag_graph_enabled:
                try:
                    from triple_hybrid_rag.graph.puppygraph import PuppyGraphClient
                    
                    puppygraph = PuppyGraphClient(config=config)
                    connected = await puppygraph.connect()
                    
                    if connected:
                        keywords = request.query.split()[:5]  # Simple keyword extraction
                        graph_results = await puppygraph.search_by_keywords_graph(
                            keywords=keywords,
                            tenant_id=request.tenant_id,
                            limit=config.rag_graph_top_k,
                        )
                        await puppygraph.close()
                except Exception:
                    # Fallback to SQL
                    from triple_hybrid_rag.graph.sql_fallback import SQLGraphFallback
                    fallback = SQLGraphFallback(pool)
                    keywords = request.query.split()[:5]
                    graph_results = await fallback.search_by_keywords(
                        keywords=keywords,
                        tenant_id=request.tenant_id,
                        limit=config.rag_graph_top_k,
                    )
            
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
                        "lexical_score": r.lexical_score,
                        "semantic_score": r.semantic_score,
                        "graph_score": r.graph_score,
                        "rrf_score": r.rrf_score,
                        "rerank_score": r.rerank_score,
                        "final_score": r.final_score,
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
        
        # Get child chunks
        child_chunks = await conn.fetch("""
            SELECT c.id, c.parent_id, c.index_in_parent, c.text, c.token_count, 
                   c.page, c.modality, c.content_hash
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
# METRICS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/metrics")
async def get_metrics():
    """Get pipeline metrics."""
    config = get_settings()
    
    # Get database stats
    stats = await get_database_stats()
    
    # OCR is enabled if rag_ocr_enabled=True AND mode is not "off"
    ocr_effectively_enabled = config.rag_ocr_enabled and config.rag_ocr_mode.lower() != "off"
    
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
            "ocr_enabled": ocr_effectively_enabled,
            "ocr_mode": config.rag_ocr_mode,
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
            app,
            host="0.0.0.0",
            port=8009,
            ssl_keyfile=str(ssl_keyfile),
            ssl_certfile=str(ssl_certfile),
        )
    else:
        print("⚠️  SSL certificates not found, starting HTTP server")
        print(f"   Generate certificates in {certs_dir} for HTTPS support")
        uvicorn.run(app, host="0.0.0.0", port=8009)
