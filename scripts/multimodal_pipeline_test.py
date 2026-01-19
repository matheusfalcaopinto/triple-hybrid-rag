#!/usr/bin/env python3
"""
Triple-Hybrid-RAG Multimodal Pipeline Test with Professional Rich TUI

Tests the complete ingestion pipeline for non-text documents:
- PDF (with OCR for scanned pages)
- DOCX (structured text extraction)
- XLSX (table extraction)
- Images (PNG, JPG, JPEG - full OCR)

Uses DeepSeek OCR (configured in .env) for vision processing.

Pipeline Stages:
1. File Discovery (scan data/ directory)
2. Document Loading (type-specific loaders)
3. OCR Processing (DeepSeek for images/scanned content)
4. Text Aggregation (combine native + OCR text)
5. Chunking (parent/child + semantic)
6. Embedding (via vLLM)
7. Entity Extraction (NER via OpenAI)
8. Database Storage (PostgreSQL + pgvector)
9. Entity Storage
10. Retrieval (lexical + semantic + graph)
11. RRF Fusion (adaptive)
12. Reranking (multi-stage)

Usage:
    uv run python scripts/multimodal_pipeline_test.py
"""

import asyncio
import hashlib
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

# Rich TUI imports
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from triple_hybrid_rag.config import get_settings, reset_settings
from triple_hybrid_rag.core.chunker import HierarchicalChunker
from triple_hybrid_rag.core.embedder import MultimodalEmbedder
from triple_hybrid_rag.core.entity_extractor import EntityRelationExtractor
from triple_hybrid_rag.core.fusion import RRFFusion
from triple_hybrid_rag.core.reranker import Reranker
from triple_hybrid_rag.ingestion.loaders import DocumentLoader, LoadedDocument, FileType
from triple_hybrid_rag.ingestion.ocr import (
    OCRProcessor,
    OCRIngestionMode,
    OCRResult,
    resolve_ocr_mode,
)
from triple_hybrid_rag.types import ChildChunk, SearchResult, SearchChannel, Modality

# Console for rich output
console = Console()

# Data directory
DATA_DIR = Path(__file__).parent.parent / "data"
TENANT_ID = "multimodal-test"

# Supported file extensions
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".xlsx", ".xls", ".csv", ".png", ".jpg", ".jpeg", ".webp", ".tiff", ".tif", ".bmp", ".txt", ".md"}


@dataclass
class StageResult:
    """Result from a pipeline stage."""
    name: str
    duration: float
    items_processed: int
    status: str = "✅"
    rate: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class DocumentResult:
    """Result of processing a single document."""
    file_path: str
    file_type: FileType
    pages_loaded: int
    pages_with_ocr: int
    total_text_chars: int
    ocr_text_chars: int
    chunks_created: int
    entities_extracted: int
    ocr_confidence: float = 0.0
    error: Optional[str] = None


class PipelineTracker:
    """Tracks pipeline progress and metrics."""

    def __init__(self):
        self.stages: List[StageResult] = []
        self.documents: List[DocumentResult] = []
        self.start_time: float = 0
        self.current_stage: str = ""

    def start(self):
        self.start_time = time.perf_counter()

    def add_result(self, result: StageResult):
        self.stages.append(result)

    def add_document(self, doc: DocumentResult):
        self.documents.append(doc)

    @property
    def total_duration(self) -> float:
        return time.perf_counter() - self.start_time

    @property
    def stages_passed(self) -> int:
        return sum(1 for s in self.stages if s.status == "✅")

    @property
    def stages_total(self) -> int:
        return len(self.stages)


def create_progress() -> Progress:
    """Create a rich progress bar."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=True,
    )


def create_results_table(tracker: PipelineTracker) -> Table:
    """Create results summary table."""
    table = Table(title="Pipeline Results", show_header=True, header_style="bold cyan")
    table.add_column("Stage", style="dim", width=30)
    table.add_column("Duration", justify="right", width=10)
    table.add_column("Items", justify="right", width=10)
    table.add_column("Rate", justify="right", width=15)
    table.add_column("Status", justify="center", width=8)

    for result in tracker.stages:
        rate_str = f"{result.rate:.1f}/s" if result.rate > 0 else "—"
        table.add_row(
            result.name,
            f"{result.duration:.3f}s",
            str(result.items_processed),
            rate_str,
            result.status,
        )

    return table


def create_documents_table(tracker: PipelineTracker) -> Table:
    """Create document processing summary table."""
    table = Table(title="Documents Processed", show_header=True, header_style="bold magenta")
    table.add_column("File", style="dim", width=45)
    table.add_column("Type", width=8)
    table.add_column("Pages", justify="right", width=6)
    table.add_column("OCR", justify="right", width=6)
    table.add_column("Text", justify="right", width=10)
    table.add_column("Chunks", justify="right", width=8)
    table.add_column("Status", justify="center", width=8)

    for doc in tracker.documents:
        status = "✅" if not doc.error else "❌"
        file_name = Path(doc.file_path).name
        if len(file_name) > 42:
            file_name = file_name[:39] + "..."
        
        table.add_row(
            file_name,
            doc.file_type.value,
            str(doc.pages_loaded),
            str(doc.pages_with_ocr),
            f"{doc.total_text_chars:,}",
            str(doc.chunks_created),
            status,
        )

    return table


def print_header():
    """Print styled header."""
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]TRIPLE-HYBRID-RAG[/bold cyan]\n"
            "[dim]Multimodal Document Pipeline Test[/dim]\n"
            "[dim italic]PDF • DOCX • XLSX • Images with DeepSeek OCR[/dim italic]",
            border_style="cyan",
        )
    )
    console.print()


def print_config(config):
    """Print configuration summary."""
    config_table = Table(show_header=False, box=None, padding=(0, 2))
    config_table.add_column("Key", style="dim")
    config_table.add_column("Value", style="green")

    config_table.add_row("Database", config.database_url[:50] + "...")
    config_table.add_row("Embed API", config.rag_embed_api_base)
    config_table.add_row("Embed Model", config.rag_embed_model)
    
    # OCR Configuration
    config_table.add_row("─" * 15, "─" * 20)
    config_table.add_row("OCR Mode", config.rag_ocr_mode)
    
    if config.rag_deepseek_ocr_enabled:
        config_table.add_row("OCR Provider", "[cyan]DeepSeek[/cyan]")
        config_table.add_row("OCR API", config.rag_deepseek_ocr_api_base)
        config_table.add_row("OCR Model", config.rag_deepseek_ocr_model)
    else:
        config_table.add_row("OCR Provider", "[yellow]Qwen3-VL[/yellow]")
        config_table.add_row("OCR API", config.rag_ocr_api_base)
        config_table.add_row("OCR Model", config.rag_ocr_model)
    
    config_table.add_row("Gundam Tiling", "[green]✓[/green]" if config.rag_gundam_tiling_enabled else "[dim]✗[/dim]")
    
    config_table.add_row("─" * 15, "─" * 20)
    config_table.add_row("PuppyGraph", config.puppygraph_bolt_url)
    config_table.add_row("Graph Enabled", str(config.rag_graph_enabled))

    console.print(Panel(config_table, title="Configuration", border_style="blue"))
    console.print()


async def stage_file_discovery(tracker: PipelineTracker) -> List[Path]:
    """Stage 1: Discover files in data/ directory."""
    console.print("[bold cyan]━━━ Stage 1: File Discovery[/bold cyan]")
    
    start = time.perf_counter()
    
    if not DATA_DIR.exists():
        console.print(f"   [red]✗[/red] Data directory not found: {DATA_DIR}")
        tracker.add_result(StageResult(
            name="1. File Discovery",
            duration=0,
            items_processed=0,
            status="❌",
            error="Data directory not found",
        ))
        return []
    
    files = []
    for f in DATA_DIR.iterdir():
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(f)
    
    duration = time.perf_counter() - start
    
    result = StageResult(
        name="1. File Discovery",
        duration=duration,
        items_processed=len(files),
        details={"directory": str(DATA_DIR)},
    )
    tracker.add_result(result)
    
    console.print(f"   [green]✓[/green] Found {len(files)} files in {DATA_DIR}")
    for f in files:
        console.print(f"   [dim]• {f.name} ({f.suffix})[/dim]")
    console.print()
    
    return files


async def stage_document_loading(
    config,
    files: List[Path],
    tracker: PipelineTracker,
) -> List[LoadedDocument]:
    """Stage 2: Load documents using type-specific loaders."""
    console.print("[bold cyan]━━━ Stage 2: Document Loading[/bold cyan]")
    
    if not files:
        console.print("   [yellow]⚠[/yellow] No files to load")
        tracker.add_result(StageResult(name="2. Document Loading", duration=0, items_processed=0, status="⏭️"))
        return []
    
    loader = DocumentLoader(
        pdf_dpi=300,
        extract_tables=True,
    )
    
    documents = []
    start = time.perf_counter()
    
    with create_progress() as progress:
        task = progress.add_task("Loading documents...", total=len(files))
        
        for file_path in files:
            doc = loader.load(file_path)
            documents.append(doc)
            
            if doc.error:
                console.print(f"   [red]✗[/red] {file_path.name}: {doc.error}")
            else:
                pages_with_images = sum(1 for p in doc.pages if p.has_images)
                scanned_pages = sum(1 for p in doc.pages if p.is_scanned)
                console.print(
                    f"   [green]✓[/green] {file_path.name}: "
                    f"{len(doc.pages)} pages, {pages_with_images} with images, "
                    f"{scanned_pages} scanned"
                )
            
            progress.update(task, advance=1)
    
    duration = time.perf_counter() - start
    total_pages = sum(len(d.pages) for d in documents if not d.error)
    
    result = StageResult(
        name="2. Document Loading",
        duration=duration,
        items_processed=len(documents),
        rate=len(documents) / duration if duration > 0 else 0,
        details={"total_pages": total_pages},
    )
    tracker.add_result(result)
    
    console.print(f"   [green]✓[/green] Loaded {len(documents)} documents, {total_pages} total pages")
    console.print()
    
    return documents


async def stage_ocr_processing(
    config,
    documents: List[LoadedDocument],
    tracker: PipelineTracker,
) -> Dict[str, Dict[int, OCRResult]]:
    """Stage 3: OCR processing for images and scanned pages."""
    console.print("[bold cyan]━━━ Stage 3: OCR Processing (DeepSeek)[/bold cyan]")
    
    # Resolve OCR mode
    ocr_mode, analysis = resolve_ocr_mode(
        mode=config.rag_ocr_mode,
        config=config,
    )
    
    if ocr_mode == OCRIngestionMode.OFF:
        console.print("   [yellow]⚠[/yellow] OCR disabled by configuration")
        tracker.add_result(StageResult(name="3. OCR Processing", duration=0, items_processed=0, status="⏭️"))
        return {}
    
    console.print(f"   [dim]OCR Mode: {ocr_mode.value}[/dim]")
    
    # Create OCR processor
    ocr_processor = OCRProcessor.create_for_mode(
        ingestion_mode=ocr_mode,
        settings=config,
    )
    
    # Find pages that need OCR
    pages_to_ocr = []
    for doc in documents:
        if doc.error:
            continue
        for page in doc.pages:
            if page.is_scanned or page.has_images:
                if page.image_data:
                    pages_to_ocr.append((doc.file_path, page.page_number, page.image_data))
    
    if not pages_to_ocr:
        console.print("   [dim]No pages require OCR[/dim]")
        tracker.add_result(StageResult(name="3. OCR Processing", duration=0, items_processed=0, status="⏭️"))
        return {}
    
    console.print(f"   [dim]Processing {len(pages_to_ocr)} pages with OCR...[/dim]")
    
    ocr_results: Dict[str, Dict[int, OCRResult]] = {}
    start = time.perf_counter()
    total_confidence = 0.0
    successful_ocr = 0
    
    with create_progress() as progress:
        task = progress.add_task("Running OCR...", total=len(pages_to_ocr))
        
        for file_path, page_num, image_data in pages_to_ocr:
            try:
                result = await ocr_processor.process_image(image_data)
                
                if file_path not in ocr_results:
                    ocr_results[file_path] = {}
                ocr_results[file_path][page_num] = result
                
                if result.text and not result.error:
                    successful_ocr += 1
                    total_confidence += result.confidence
                    console.print(
                        f"   [green]✓[/green] {Path(file_path).name} p{page_num}: "
                        f"{len(result.text)} chars, conf={result.confidence:.2f}"
                    )
                else:
                    console.print(
                        f"   [yellow]⚠[/yellow] {Path(file_path).name} p{page_num}: "
                        f"{result.error or 'No text extracted'}"
                    )
            except Exception as e:
                console.print(f"   [red]✗[/red] {Path(file_path).name} p{page_num}: {e}")
            
            progress.update(task, advance=1)
    
    duration = time.perf_counter() - start
    avg_confidence = total_confidence / successful_ocr if successful_ocr > 0 else 0.0
    
    # Get OCR metrics
    metrics = ocr_processor.get_metrics()
    
    result = StageResult(
        name="3. OCR Processing",
        duration=duration,
        items_processed=successful_ocr,
        rate=successful_ocr / duration if duration > 0 else 0,
        details={
            "total_pages": len(pages_to_ocr),
            "successful": successful_ocr,
            "avg_confidence": avg_confidence,
            "network_retries": metrics.get("total_network_retries", 0),
        },
    )
    tracker.add_result(result)
    
    console.print(f"   [green]✓[/green] OCR completed: {successful_ocr}/{len(pages_to_ocr)} pages")
    console.print(f"   [dim]Avg confidence: {avg_confidence:.2f}, Duration: {duration:.1f}s[/dim]")
    console.print()
    
    return ocr_results


async def stage_text_aggregation(
    documents: List[LoadedDocument],
    ocr_results: Dict[str, Dict[int, OCRResult]],
    tracker: PipelineTracker,
) -> Dict[str, str]:
    """Stage 4: Aggregate native text + OCR text for each document."""
    console.print("[bold cyan]━━━ Stage 4: Text Aggregation[/bold cyan]")
    
    start = time.perf_counter()
    aggregated_texts: Dict[str, str] = {}
    
    for doc in documents:
        if doc.error:
            continue
        
        text_parts = []
        ocr_chars = 0
        native_chars = 0
        
        for page in doc.pages:
            # Add native text
            if page.text.strip():
                text_parts.append(f"[Page {page.page_number}]\n{page.text}")
                native_chars += len(page.text)
            
            # Add OCR text if available
            doc_ocr = ocr_results.get(doc.file_path, {})
            page_ocr = doc_ocr.get(page.page_number)
            if page_ocr and page_ocr.text.strip():
                text_parts.append(f"[Page {page.page_number} - OCR]\n{page_ocr.text}")
                ocr_chars += len(page_ocr.text)
            
            # Add tables
            for table in page.tables:
                text_parts.append(f"[Table - Page {page.page_number}]\n{table}")
        
        full_text = "\n\n".join(text_parts)
        aggregated_texts[doc.file_path] = full_text
        
        console.print(
            f"   [green]✓[/green] {Path(doc.file_path).name}: "
            f"{native_chars:,} native + {ocr_chars:,} OCR = {len(full_text):,} total chars"
        )
        
        # Track document result
        doc_ocr = ocr_results.get(doc.file_path, {})
        tracker.add_document(DocumentResult(
            file_path=doc.file_path,
            file_type=doc.file_type,
            pages_loaded=len(doc.pages),
            pages_with_ocr=len(doc_ocr),
            total_text_chars=len(full_text),
            ocr_text_chars=ocr_chars,
            chunks_created=0,  # Updated later
            entities_extracted=0,  # Updated later
            ocr_confidence=sum(r.confidence for r in doc_ocr.values()) / len(doc_ocr) if doc_ocr else 0.0,
        ))
    
    duration = time.perf_counter() - start
    total_chars = sum(len(t) for t in aggregated_texts.values())
    
    result = StageResult(
        name="4. Text Aggregation",
        duration=duration,
        items_processed=len(aggregated_texts),
        details={"total_chars": total_chars},
    )
    tracker.add_result(result)
    
    console.print(f"   [green]✓[/green] Aggregated {len(aggregated_texts)} documents, {total_chars:,} total chars")
    console.print()
    
    return aggregated_texts


async def stage_chunking(
    config,
    aggregated_texts: Dict[str, str],
    tracker: PipelineTracker,
) -> Tuple[Dict[str, list], Dict[str, list]]:
    """Stage 5: Hierarchical Chunking."""
    console.print("[bold cyan]━━━ Stage 5: Chunking[/bold cyan]")
    
    if not aggregated_texts:
        console.print("   [yellow]⚠[/yellow] No texts to chunk")
        tracker.add_result(StageResult(name="5. Chunking", duration=0, items_processed=0, status="⏭️"))
        return {}, {}
    
    chunker = HierarchicalChunker(config=config)
    
    all_parents: Dict[str, list] = {}
    all_children: Dict[str, list] = {}
    total_chunks = 0
    
    start = time.perf_counter()
    
    with create_progress() as progress:
        task = progress.add_task("Chunking documents...", total=len(aggregated_texts))
        
        for file_path, text in aggregated_texts.items():
            document_id = uuid4()
            
            parent_chunks, child_chunks = chunker.split_document(
                text=text,
                document_id=document_id,
                tenant_id=TENANT_ID,
            )
            
            all_parents[file_path] = parent_chunks
            all_children[file_path] = child_chunks
            total_chunks += len(child_chunks)
            
            # Update document result
            for doc_result in tracker.documents:
                if doc_result.file_path == file_path:
                    doc_result.chunks_created = len(child_chunks)
                    break
            
            console.print(
                f"   [green]✓[/green] {Path(file_path).name}: "
                f"{len(parent_chunks)} parents, {len(child_chunks)} children"
            )
            
            progress.update(task, advance=1)
    
    duration = time.perf_counter() - start
    rate = total_chunks / duration if duration > 0 else 0
    
    result = StageResult(
        name="5. Chunking",
        duration=duration,
        items_processed=total_chunks,
        rate=rate,
        details={"documents": len(aggregated_texts)},
    )
    tracker.add_result(result)
    
    console.print(f"   [green]✓[/green] Created {total_chunks} chunks at {rate:.1f} chunks/s")
    console.print()
    
    return all_parents, all_children


async def stage_embedding(
    config,
    all_children: Dict[str, list],
    tracker: PipelineTracker,
) -> Dict[str, list]:
    """Stage 6: Embedding via vLLM."""
    console.print("[bold cyan]━━━ Stage 6: Embedding[/bold cyan]")
    
    if not all_children:
        console.print("   [yellow]⚠[/yellow] No chunks to embed")
        tracker.add_result(StageResult(name="6. Embedding", duration=0, items_processed=0, status="⏭️"))
        return {}
    
    embedder = MultimodalEmbedder(config=config)
    
    # Flatten all chunks
    all_texts = []
    chunk_map = []  # (file_path, chunk_index)
    
    for file_path, chunks in all_children.items():
        for i, chunk in enumerate(chunks):
            all_texts.append(chunk.text)
            chunk_map.append((file_path, i))
    
    console.print(f"   [dim]Embedding {len(all_texts)} chunks...[/dim]")
    
    start = time.perf_counter()
    
    with create_progress() as progress:
        task = progress.add_task("Embedding texts...", total=len(all_texts))
        
        embeddings = await embedder.embed_texts_concurrent(all_texts)
        progress.update(task, completed=len(all_texts))
    
    duration = time.perf_counter() - start
    rate = len(embeddings) / duration if duration > 0 else 0
    
    # Assign embeddings back to chunks
    for (file_path, chunk_idx), embedding in zip(chunk_map, embeddings):
        all_children[file_path][chunk_idx].embedding = embedding
    
    result = StageResult(
        name="6. Embedding",
        duration=duration,
        items_processed=len(embeddings),
        rate=rate,
        details={"dimension": len(embeddings[0]) if embeddings else 0},
    )
    tracker.add_result(result)
    
    console.print(f"   [green]✓[/green] Embedded {len(embeddings)} chunks at {rate:.1f} texts/s")
    console.print(f"   [dim]Dimension: {len(embeddings[0]) if embeddings else 0}[/dim]")
    console.print()
    
    await embedder.close()
    return all_children


async def stage_ner(
    config,
    all_children: Dict[str, list],
    tracker: PipelineTracker,
) -> Tuple[list, list]:
    """Stage 7: Entity Extraction (NER)."""
    console.print("[bold cyan]━━━ Stage 7: Entity Extraction (NER)[/bold cyan]")
    
    if not config.openai_api_key:
        console.print("   [yellow]⚠[/yellow] OPENAI_API_KEY not set, skipping NER")
        tracker.add_result(StageResult(name="7. NER", duration=0, items_processed=0, status="⏭️"))
        return [], []
    
    # Collect sample chunks from each document for NER
    sample_chunks = []
    for file_path, chunks in all_children.items():
        sample_chunks.extend(chunks[:3])  # First 3 chunks per document
    
    if not sample_chunks:
        console.print("   [yellow]⚠[/yellow] No chunks for NER")
        tracker.add_result(StageResult(name="7. NER", duration=0, items_processed=0, status="⏭️"))
        return [], []
    
    extractor = EntityRelationExtractor(config=config)
    
    start = time.perf_counter()
    
    try:
        with create_progress() as progress:
            task = progress.add_task("Extracting entities...", total=None)
            
            extraction = await extractor.extract(sample_chunks[:10])  # Limit to 10 chunks
            progress.update(task, completed=1, total=1)
        
        duration = time.perf_counter() - start
        entities = extraction.entities
        relations = extraction.relations
        
        result = StageResult(
            name="7. NER",
            duration=duration,
            items_processed=len(entities),
            rate=len(entities) / duration if duration > 0 else 0,
            details={"relations": len(relations)},
        )
        tracker.add_result(result)
        
        console.print(f"   [green]✓[/green] Entities: {len(entities)}, Relations: {len(relations)}")
        
        # Group entities by type
        by_type = {}
        for e in entities:
            by_type.setdefault(e.entity_type, []).append(e.name)
        
        for etype, names in list(by_type.items())[:5]:
            console.print(f"   [dim]• {etype}: {', '.join(names[:3])}{'...' if len(names) > 3 else ''}[/dim]")
        
        console.print()
        return entities, relations
        
    except Exception as e:
        duration = time.perf_counter() - start
        console.print(f"   [red]✗[/red] NER error: {e}")
        tracker.add_result(StageResult(
            name="7. NER",
            duration=duration,
            items_processed=0,
            status="❌",
            error=str(e),
        ))
        console.print()
        return [], []


async def stage_db_storage(
    config,
    all_parents: Dict[str, list],
    all_children: Dict[str, list],
    documents: List[LoadedDocument],
    tracker: PipelineTracker,
) -> Tuple[Dict[str, str], Dict[str, list]]:
    """Stage 8: Database Storage."""
    console.print("[bold cyan]━━━ Stage 8: Database Storage[/bold cyan]")
    
    try:
        import asyncpg
    except ImportError:
        console.print("   [yellow]⚠[/yellow] asyncpg not installed")
        tracker.add_result(StageResult(name="8. DB Storage", duration=0, items_processed=0, status="⏭️"))
        return {}, {}
    
    if not all_children:
        console.print("   [yellow]⚠[/yellow] No chunks to store")
        tracker.add_result(StageResult(name="8. DB Storage", duration=0, items_processed=0, status="⏭️"))
        return {}, {}
    
    try:
        pool = await asyncpg.create_pool(config.database_url, min_size=1, max_size=5)
        
        start = time.perf_counter()
        document_ids: Dict[str, str] = {}
        chunk_ids: Dict[str, list] = {}
        total_chunks_stored = 0
        
        async with pool.acquire() as conn:
            for doc in documents:
                if doc.error or doc.file_path not in all_children:
                    continue
                
                # Create document record
                doc_hash = doc.file_hash
                file_name = Path(doc.file_path).name
                
                document_id = await conn.fetchval("""
                    INSERT INTO rag_documents (tenant_id, hash_sha256, file_name, title, ingestion_status)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (tenant_id, hash_sha256) DO UPDATE SET title = EXCLUDED.title
                    RETURNING id
                """, TENANT_ID, doc_hash, file_name, file_name, 'completed')
                
                document_ids[doc.file_path] = str(document_id)
                
                # Insert parent chunks
                parent_ids = []
                parents = all_parents.get(doc.file_path, [])
                for idx, parent in enumerate(parents):
                    parent_id = await conn.fetchval("""
                        INSERT INTO rag_parent_chunks (document_id, tenant_id, index_in_document, text, page_start)
                        VALUES ($1, $2, $3, $4, $5)
                        RETURNING id
                    """, document_id, TENANT_ID, idx, parent.text, parent.page_start or 1)
                    parent_ids.append(parent_id)
                
                # Insert child chunks
                children = all_children.get(doc.file_path, [])
                file_chunk_ids = []
                
                for idx, chunk in enumerate(children):
                    if chunk.embedding:
                        parent_id = parent_ids[0] if parent_ids else None
                        content_hash = hashlib.sha256(chunk.text.encode()).hexdigest()[:32]
                        embedding_str = "[" + ",".join(str(x) for x in chunk.embedding) + "]"
                        
                        chunk_id = await conn.fetchval("""
                            INSERT INTO rag_child_chunks (
                                parent_id, document_id, tenant_id, index_in_parent,
                                text, content_hash, embedding_1024, page
                            )
                            VALUES ($1, $2, $3, $4, $5, $6, $7::vector, $8)
                            ON CONFLICT (tenant_id, content_hash) DO UPDATE SET text = EXCLUDED.text
                            RETURNING id
                        """, parent_id, document_id, TENANT_ID, idx, chunk.text, content_hash, embedding_str, chunk.page or 1)
                        
                        file_chunk_ids.append(str(chunk_id))
                        total_chunks_stored += 1
                
                chunk_ids[doc.file_path] = file_chunk_ids
                console.print(f"   [green]✓[/green] {file_name}: {len(file_chunk_ids)} chunks stored")
        
        duration = time.perf_counter() - start
        rate = total_chunks_stored / duration if duration > 0 else 0
        
        result = StageResult(
            name="8. DB Storage",
            duration=duration,
            items_processed=total_chunks_stored,
            rate=rate,
            details={"documents": len(document_ids)},
        )
        tracker.add_result(result)
        
        console.print(f"   [green]✓[/green] Stored {total_chunks_stored} chunks from {len(document_ids)} documents")
        console.print()
        
        await pool.close()
        return document_ids, chunk_ids
        
    except Exception as e:
        console.print(f"   [red]✗[/red] Database error: {e}")
        import traceback
        traceback.print_exc()
        tracker.add_result(StageResult(name="8. DB Storage", duration=0, items_processed=0, status="❌", error=str(e)))
        console.print()
        return {}, {}


async def stage_entity_storage(
    config,
    entities: list,
    relations: list,
    document_ids: Dict[str, str],
    chunk_ids: Dict[str, list],
    tracker: PipelineTracker,
):
    """Stage 9: Entity Storage."""
    console.print("[bold cyan]━━━ Stage 9: Entity Storage[/bold cyan]")
    
    if not entities or not document_ids:
        console.print("   [yellow]⚠[/yellow] No entities or documents to store")
        tracker.add_result(StageResult(name="9. Entity Storage", duration=0, items_processed=0, status="⏭️"))
        console.print()
        return
    
    try:
        import asyncpg
        from uuid import UUID
        import json
        
        pool = await asyncpg.create_pool(config.database_url, min_size=1, max_size=5)
        start = time.perf_counter()
        
        # Use first document for entity storage
        first_doc_path = next(iter(document_ids.keys()))
        doc_uuid = UUID(document_ids[first_doc_path])
        first_chunk_ids = chunk_ids.get(first_doc_path, [])
        
        async with pool.acquire() as conn:
            entity_id_map: Dict[str, UUID] = {}
            entity_ids = []
            
            for entity in entities[:20]:
                entity_id = await conn.fetchval("""
                    INSERT INTO rag_entities (tenant_id, document_id, entity_type, name, canonical_name, description)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (tenant_id, canonical_name) DO UPDATE SET 
                        name = EXCLUDED.name,
                        description = COALESCE(EXCLUDED.description, rag_entities.description)
                    RETURNING id
                """, TENANT_ID, doc_uuid, entity.entity_type, entity.name, entity.canonical_name,
                    getattr(entity, 'description', None))
                entity_ids.append(entity_id)
                entity_id_map[entity.canonical_name] = entity_id
            
            # Store mentions
            mentions = 0
            chunk_uuid_list = [UUID(cid) for cid in first_chunk_ids[:5]]
            
            for entity in entities[:20]:
                entity_id = entity_id_map.get(entity.canonical_name)
                if not entity_id:
                    continue
                
                mention_text = getattr(entity, 'mention_text', None) or entity.name
                
                for chunk_uuid in chunk_uuid_list[:2]:
                    await conn.execute("""
                        INSERT INTO rag_entity_mentions (entity_id, child_chunk_id, document_id, mention_text, confidence)
                        VALUES ($1, $2, $3, $4, $5)
                        ON CONFLICT DO NOTHING
                    """, entity_id, chunk_uuid, doc_uuid, mention_text,
                        getattr(entity, 'confidence', 0.8))
                    mentions += 1
            
            # Store relations
            relation_count = 0
            for relation in relations[:30]:
                subject_id = entity_id_map.get(relation.subject)
                object_id = entity_id_map.get(relation.object)
                
                if not subject_id or not object_id:
                    continue
                
                evidence = getattr(relation, 'evidence', None)
                metadata_dict = {"evidence": evidence} if evidence else {}
                metadata_json = json.dumps(metadata_dict)
                
                await conn.execute("""
                    INSERT INTO rag_relations (tenant_id, document_id, relation_type, subject_entity_id, object_entity_id, confidence, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
                    ON CONFLICT DO NOTHING
                """, TENANT_ID, doc_uuid, relation.relation_type, subject_id, object_id,
                    getattr(relation, 'confidence', 0.7), metadata_json)
                relation_count += 1
        
        duration = time.perf_counter() - start
        
        result = StageResult(
            name="9. Entity Storage",
            duration=duration,
            items_processed=len(entity_ids),
            rate=len(entity_ids) / duration if duration > 0 else 0,
            details={"mentions": mentions, "relations": relation_count},
        )
        tracker.add_result(result)
        
        console.print(f"   [green]✓[/green] Entities: {len(entity_ids)}, Mentions: {mentions}, Relations: {relation_count}")
        console.print()
        
        await pool.close()
        
    except Exception as e:
        console.print(f"   [red]✗[/red] Entity storage error: {e}")
        import traceback
        traceback.print_exc()
        tracker.add_result(StageResult(name="9. Entity Storage", duration=0, items_processed=0, status="❌"))
        console.print()


async def stage_retrieval(
    config,
    all_children: Dict[str, list],
    tracker: PipelineTracker,
) -> Tuple[list, list, list, str]:
    """Stage 10: Triple-Hybrid Retrieval."""
    console.print("[bold cyan]━━━ Stage 10: Retrieval (Lexical + Semantic + Graph)[/bold cyan]")
    
    # Generate query based on document content
    query = "What are the backtested trading strategies for Brazilian mining stocks and their technical levels?"
    
    console.print(f"   [dim]Query: {query}[/dim]")
    
    embedder = MultimodalEmbedder(config=config)
    query_embedding = await embedder.embed_text(query)
    await embedder.close()
    
    semantic_results, lexical_results, graph_results = [], [], []
    puppygraph_used = False
    
    try:
        import asyncpg
        from triple_hybrid_rag.graph.puppygraph import PuppyGraphClient
        from triple_hybrid_rag.graph.sql_fallback import SQLGraphFallback
        
        pool = await asyncpg.create_pool(config.database_url, min_size=1, max_size=5)
        start = time.perf_counter()
        
        async with pool.acquire() as conn:
            # Semantic search
            query_embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
            semantic_results = await conn.fetch("""
                SELECT * FROM rag_semantic_search($1, $2::vector, $3)
            """, TENANT_ID, query_embedding_str, 5)
            
            # Lexical search
            lexical_results = await conn.fetch("""
                SELECT * FROM rag_lexical_search($1, $2, $3)
            """, TENANT_ID, query, 5)
        
        # Graph search
        if config.rag_graph_enabled:
            try:
                puppygraph = PuppyGraphClient(config=config)
                connected = await puppygraph.connect()
                
                if connected:
                    graph_results = await puppygraph.search_by_keywords_graph(
                        keywords=["mining", "stocks", "trading", "strategy"],
                        tenant_id=TENANT_ID,
                        limit=5,
                    )
                    puppygraph_used = True
                    await puppygraph.close()
                else:
                    raise ConnectionError("PuppyGraph not available")
                    
            except Exception as e:
                console.print(f"   [yellow]⚠[/yellow] PuppyGraph unavailable: {e}")
                console.print(f"   [dim]Using SQL fallback...[/dim]")
                
                graph_fallback = SQLGraphFallback(pool)
                graph_results = await graph_fallback.search_by_keywords(
                    keywords=["mining", "stocks", "trading", "strategy"],
                    tenant_id=TENANT_ID,
                    limit=5,
                )
        
        duration = time.perf_counter() - start
        total_results = len(semantic_results) + len(lexical_results) + len(graph_results)
        
        result = StageResult(
            name="10. Retrieval",
            duration=duration,
            items_processed=total_results,
            rate=total_results / duration if duration > 0 else 0,
            details={
                "semantic": len(semantic_results),
                "lexical": len(lexical_results),
                "graph": len(graph_results),
                "puppygraph": puppygraph_used,
            },
        )
        tracker.add_result(result)
        
        console.print(f"   [green]✓[/green] Semantic: {len(semantic_results)}, Lexical: {len(lexical_results)}, Graph: {len(graph_results)}")
        console.print(f"   [green]✓[/green] Graph Backend: {'[cyan]PuppyGraph[/cyan]' if puppygraph_used else '[yellow]SQL Fallback[/yellow]'}")
        console.print()
        
        await pool.close()
        return semantic_results, lexical_results, graph_results, query
        
    except Exception as e:
        console.print(f"   [red]✗[/red] Retrieval error: {e}")
        tracker.add_result(StageResult(name="10. Retrieval", duration=0, items_processed=0, status="❌", error=str(e)))
        console.print()
        return [], [], [], query


async def stage_fusion(
    config,
    semantic_results: list,
    lexical_results: list,
    graph_results: list,
    tracker: PipelineTracker,
) -> list:
    """Stage 11: RRF Fusion."""
    console.print("[bold cyan]━━━ Stage 11: RRF Fusion[/bold cyan]")
    
    if not semantic_results and not lexical_results and not graph_results:
        console.print("   [yellow]⚠[/yellow] No results to fuse")
        tracker.add_result(StageResult(name="11. RRF Fusion", duration=0, items_processed=0, status="⏭️"))
        console.print()
        return []
    
    def row_to_result(row, channel: SearchChannel, score_field: str) -> SearchResult:
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
    
    semantic_search_results = [row_to_result(r, SearchChannel.SEMANTIC, 'similarity') for r in semantic_results]
    lexical_search_results = [row_to_result(r, SearchChannel.LEXICAL, 'rank') for r in lexical_results]
    
    fusion = RRFFusion(config=config)
    
    start = time.perf_counter()
    fused_results = fusion.fuse(
        lexical_results=lexical_search_results,
        semantic_results=semantic_search_results,
        graph_results=graph_results,
        top_k=10,
        apply_safety=False,
        apply_denoise=False,
    )
    duration = time.perf_counter() - start
    
    result = StageResult(
        name="11. RRF Fusion",
        duration=duration,
        items_processed=len(fused_results),
        rate=len(fused_results) / duration if duration > 0 else 0,
    )
    tracker.add_result(result)
    
    console.print(f"   [green]✓[/green] Fused: {len(fused_results)} results")
    for i, r in enumerate(fused_results[:3]):
        channels = r.metadata.get('source_channels', [])
        console.print(f"   [dim]• [{i+1}] RRF={r.rrf_score:.4f} channels={channels}[/dim]")
    console.print()
    
    return fused_results


async def stage_reranking(
    config,
    fused_results: list,
    query: str,
    tracker: PipelineTracker,
) -> list:
    """Stage 12: Reranking."""
    console.print("[bold cyan]━━━ Stage 12: Reranking[/bold cyan]")
    
    if not fused_results:
        console.print("   [yellow]⚠[/yellow] No results to rerank")
        tracker.add_result(StageResult(name="12. Reranking", duration=0, items_processed=0, status="⏭️"))
        console.print()
        return []
    
    if not config.rag_rerank_enabled:
        console.print("   [yellow]⚠[/yellow] Reranker disabled")
        tracker.add_result(StageResult(name="12. Reranking", duration=0, items_processed=0, status="⏭️"))
        console.print()
        return fused_results
    
    reranker = Reranker(config=config)
    documents = [r.text for r in fused_results]
    
    start = time.perf_counter()
    scores = await reranker.rerank(query, documents)
    duration = time.perf_counter() - start
    
    await reranker.close()
    
    if all(s == 0.0 for s in scores):
        console.print("   [yellow]⚠[/yellow] Reranker returned zeros (server not running?)")
        tracker.add_result(StageResult(name="12. Reranking", duration=duration, items_processed=0, status="⚠️"))
    else:
        for result, score in zip(fused_results, scores):
            result.rerank_score = score
            result.final_score = score
        
        fused_results.sort(key=lambda r: r.final_score, reverse=True)
        
        result = StageResult(
            name="12. Reranking",
            duration=duration,
            items_processed=len(scores),
            rate=len(scores) / duration if duration > 0 else 0,
        )
        tracker.add_result(result)
        
        console.print(f"   [green]✓[/green] Reranked: {len(scores)} results")
        for i, r in enumerate(fused_results[:3]):
            text_preview = r.text[:60].replace('\n', ' ') + "..."
            console.print(f"   [dim]• [{i+1}] score={r.rerank_score:.4f} {text_preview}[/dim]")
    
    console.print()
    return fused_results


async def cleanup(config):
    """Clean up test data."""
    console.print("[bold cyan]━━━ Cleanup[/bold cyan]")
    
    try:
        import asyncpg
        
        pool = await asyncpg.create_pool(config.database_url, min_size=1, max_size=5)
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM rag_entity_mentions WHERE entity_id IN (SELECT id FROM rag_entities WHERE tenant_id = $1)", TENANT_ID)
            await conn.execute("DELETE FROM rag_relations WHERE tenant_id = $1", TENANT_ID)
            await conn.execute("DELETE FROM rag_entities WHERE tenant_id = $1", TENANT_ID)
            await conn.execute("DELETE FROM rag_child_chunks WHERE tenant_id = $1", TENANT_ID)
            await conn.execute("DELETE FROM rag_parent_chunks WHERE tenant_id = $1", TENANT_ID)
            await conn.execute("DELETE FROM rag_documents WHERE tenant_id = $1", TENANT_ID)
        
        await pool.close()
        console.print("   [green]✓[/green] Test data cleaned")
        
    except Exception as e:
        console.print(f"   [yellow]⚠[/yellow] Cleanup failed: {e}")
    
    console.print()


def print_summary(tracker: PipelineTracker):
    """Print final summary dashboard."""
    console.print()
    console.print(create_documents_table(tracker))
    console.print()
    console.print(create_results_table(tracker))
    console.print()
    
    # Summary panel
    status_text = f"[bold green]{tracker.stages_passed}/{tracker.stages_total} stages passed[/bold green]"
    time_text = f"Total: {tracker.total_duration:.2f}s"
    docs_text = f"Documents: {len(tracker.documents)}"
    
    # Check OCR stats
    ocr_stage = next((s for s in tracker.stages if "OCR" in s.name), None)
    if ocr_stage and ocr_stage.items_processed > 0:
        ocr_text = f"[cyan]OCR: {ocr_stage.items_processed} pages[/cyan]"
    else:
        ocr_text = "[dim]OCR: No pages processed[/dim]"
    
    # Check retrieval
    retrieval = next((s for s in tracker.stages if "Retrieval" in s.name), None)
    if retrieval and retrieval.details.get("puppygraph"):
        graph_text = "[cyan]PuppyGraph: ✅[/cyan]"
    else:
        graph_text = "[yellow]PuppyGraph: ❌ (SQL Fallback)[/yellow]"
    
    console.print(
        Panel(
            f"{status_text}  |  {time_text}\n{docs_text}  |  {ocr_text}\n{graph_text}",
            title="Summary",
            border_style="green" if tracker.stages_passed == tracker.stages_total else "yellow",
        )
    )


async def main():
    """Main entry point."""
    print_header()
    
    reset_settings()
    config = get_settings()
    print_config(config)
    
    tracker = PipelineTracker()
    tracker.start()
    
    try:
        # Stage 1: File Discovery
        files = await stage_file_discovery(tracker)
        
        # Stage 2: Document Loading
        documents = await stage_document_loading(config, files, tracker)
        
        # Stage 3: OCR Processing
        ocr_results = await stage_ocr_processing(config, documents, tracker)
        
        # Stage 4: Text Aggregation
        aggregated_texts = await stage_text_aggregation(documents, ocr_results, tracker)
        
        # Stage 5: Chunking
        all_parents, all_children = await stage_chunking(config, aggregated_texts, tracker)
        
        # Stage 6: Embedding
        all_children = await stage_embedding(config, all_children, tracker)
        
        # Stage 7: NER
        entities, relations = await stage_ner(config, all_children, tracker)
        
        # Stage 8: DB Storage
        document_ids, chunk_ids = await stage_db_storage(config, all_parents, all_children, documents, tracker)
        
        # Stage 9: Entity Storage
        await stage_entity_storage(config, entities, relations, document_ids, chunk_ids, tracker)
        
        # Stage 10: Retrieval
        semantic_results, lexical_results, graph_results, query = await stage_retrieval(config, all_children, tracker)
        
        # Stage 11: Fusion
        fused_results = await stage_fusion(config, semantic_results, lexical_results, graph_results, tracker)
        
        # Stage 12: Reranking
        fused_results = await stage_reranking(config, fused_results, query, tracker)
        
        # Cleanup (optional - use --keep-data flag to preserve data)
        if "--keep-data" not in sys.argv:
            await cleanup(config)
        else:
            console.print("[bold green]━━━ Data Retained (--keep-data flag)[/bold green]")
            console.print(f"   [dim]Data kept in tenant: {TENANT_ID}[/dim]")
            console.print()
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()
    
    print_summary(tracker)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
