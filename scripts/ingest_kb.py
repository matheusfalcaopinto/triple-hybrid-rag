#!/usr/bin/env python3
"""
Knowledge Base Ingestion CLI

Command-line tool for ingesting documents into the RAG knowledge base.

Usage:
    python scripts/ingest_kb.py <file_or_directory> [options]
    
Examples:
    # Ingest a single file
    python scripts/ingest_kb.py docs/product_manual.pdf --category product
    
    # Ingest all files in a directory
    python scripts/ingest_kb.py docs/knowledge/ --category general
    
    # Ingest with custom settings
    python scripts/ingest_kb.py docs/manual.pdf --org-id abc123 --chunk-size 800
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import List, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from voice_agent.config import SETTINGS
from voice_agent.ingestion.kb_ingest import KnowledgeBaseIngestor, IngestResult
from voice_agent.ingestion.loader import FileType, detect_file_type
from voice_agent.utils.db import get_supabase_client


console = Console()
logger = logging.getLogger(__name__)

# Supported file extensions
SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".doc", ".txt", ".md",
    ".csv", ".xlsx", ".xls",
    ".png", ".jpg", ".jpeg", ".webp", ".tiff", ".tif", ".bmp",
}


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_org_id(org_id: Optional[str] = None) -> str:
    """Get organization ID from argument or database."""
    if org_id:
        return org_id
    
    # Fetch default org from database
    supabase = get_supabase_client()
    response = supabase.table("organizations").select("id").limit(1).execute()
    
    if not response.data:
        console.print("[red]Error: No organization found in database[/red]")
        sys.exit(1)
    
    return response.data[0]["id"]


def collect_files(path: Path, recursive: bool = True) -> List[Path]:
    """Collect all supported files from path."""
    files = []
    
    if path.is_file():
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)
        else:
            console.print(f"[yellow]Warning: Unsupported file type: {path.suffix}[/yellow]")
    elif path.is_dir():
        pattern = "**/*" if recursive else "*"
        for file_path in path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(file_path)
    else:
        console.print(f"[red]Error: Path not found: {path}[/red]")
        sys.exit(1)
    
    return sorted(files)


async def ingest_files(
    files: List[Path],
    org_id: str,
    category: str,
    chunk_size: int,
    chunk_overlap: int,
    dry_run: bool = False,
) -> List[IngestResult]:
    """Ingest a list of files."""
    from voice_agent.ingestion.chunker import Chunker
    
    # Create ingestor with custom settings
    chunker = Chunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        preserve_tables=SETTINGS.rag_preserve_tables,
    )
    
    ingestor = KnowledgeBaseIngestor(
        org_id=org_id,
        category=category,
        chunker=chunker,
    )
    
    results = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        
        overall_task = progress.add_task(
            f"[cyan]Ingesting {len(files)} file(s)...",
            total=len(files),
        )
        
        for file_path in files:
            file_type = detect_file_type(file_path)
            progress.update(
                overall_task,
                description=f"[cyan]Processing: {file_path.name} ({file_type.value})",
            )
            
            if dry_run:
                console.print(f"  [dim]Would ingest: {file_path}[/dim]")
                results.append(IngestResult(
                    success=True,
                    source_document=file_path.name,
                    stats=None,  # type: ignore
                ))
            else:
                result = await ingestor.ingest_file(file_path, category=category)
                results.append(result)
                
                if result.success:
                    console.print(
                        f"  [green]✓[/green] {file_path.name}: "
                        f"{result.stats.chunks_stored} chunks stored"
                    )
                else:
                    console.print(
                        f"  [red]✗[/red] {file_path.name}: {result.error}"
                    )
            
            progress.advance(overall_task)
    
    return results


def print_summary(results: List[IngestResult]) -> None:
    """Print ingestion summary."""
    console.print()
    
    # Summary table
    table = Table(title="Ingestion Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    total_files = len(results)
    successful = sum(1 for r in results if r.success)
    failed = total_files - successful
    
    total_chunks = sum(r.stats.chunks_stored for r in results if r.stats)
    total_embedded = sum(r.stats.chunks_embedded for r in results if r.stats)
    total_deduped = sum(r.stats.chunks_deduplicated for r in results if r.stats)
    total_ocr = sum(r.stats.ocr_pages_processed for r in results if r.stats)
    
    table.add_row("Files Processed", str(total_files))
    table.add_row("Successful", str(successful))
    table.add_row("Failed", str(failed))
    table.add_row("Chunks Created", str(sum(r.stats.chunks_created for r in results if r.stats)))
    table.add_row("Chunks Stored", str(total_chunks))
    table.add_row("Chunks Embedded", str(total_embedded))
    table.add_row("Chunks Deduplicated", str(total_deduped))
    table.add_row("OCR Pages Processed", str(total_ocr))
    
    console.print(table)
    
    # Print errors if any
    all_errors = []
    for r in results:
        if r.stats and r.stats.errors:
            all_errors.extend(r.stats.errors)
        if r.error:
            all_errors.append(f"{r.source_document}: {r.error}")
    
    if all_errors:
        console.print()
        console.print("[yellow]Errors/Warnings:[/yellow]")
        for error in all_errors[:10]:  # Limit to first 10
            console.print(f"  • {error}")
        if len(all_errors) > 10:
            console.print(f"  ... and {len(all_errors) - 10} more")


def main():
    parser = argparse.ArgumentParser(
        description="Ingest documents into the RAG knowledge base",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "path",
        type=Path,
        help="File or directory to ingest",
    )
    
    parser.add_argument(
        "--org-id",
        type=str,
        help="Organization ID (default: first org in database)",
    )
    
    parser.add_argument(
        "--category",
        type=str,
        default="general",
        help="Category for ingested content (default: general)",
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=SETTINGS.rag_chunk_size,
        help=f"Chunk size in characters (default: {SETTINGS.rag_chunk_size})",
    )
    
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=SETTINGS.rag_chunk_overlap,
        help=f"Chunk overlap in characters (default: {SETTINGS.rag_chunk_overlap})",
    )
    
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Don't recursively process directories",
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be ingested without actually processing",
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    
    # Setup
    setup_logging(args.verbose)
    
    # Collect files
    console.print(f"[bold]RAG Knowledge Base Ingestion[/bold]")
    console.print()
    
    files = collect_files(args.path, recursive=not args.no_recursive)
    
    if not files:
        console.print("[yellow]No supported files found to ingest[/yellow]")
        sys.exit(0)
    
    console.print(f"Found [bold]{len(files)}[/bold] file(s) to process")
    
    # Get org ID
    org_id = get_org_id(args.org_id)
    console.print(f"Organization ID: [dim]{org_id}[/dim]")
    console.print(f"Category: [dim]{args.category}[/dim]")
    console.print()
    
    # Run ingestion
    results = asyncio.run(ingest_files(
        files=files,
        org_id=org_id,
        category=args.category,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        dry_run=args.dry_run,
    ))
    
    # Print summary
    if not args.dry_run:
        print_summary(results)


if __name__ == "__main__":
    main()
