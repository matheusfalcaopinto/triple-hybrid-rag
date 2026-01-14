#!/usr/bin/env python3
"""
Knowledge Base Backfill Script

Migrates existing knowledge_base entries to the new knowledge_base_chunks table.
This preserves existing content while enabling hybrid search.

Usage:
    python scripts/backfill_kb_chunks.py [options]
    
Examples:
    # Backfill all existing entries
    python scripts/backfill_kb_chunks.py
    
    # Dry run to see what would be processed
    python scripts/backfill_kb_chunks.py --dry-run
    
    # Backfill specific org only
    python scripts/backfill_kb_chunks.py --org-id abc123
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
from voice_agent.ingestion.chunker import Chunker, ChunkType
from voice_agent.ingestion.embedder import Embedder
from voice_agent.utils.db import get_supabase_client


console = Console()
logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


async def backfill_kb_entries(
    org_id: Optional[str] = None,
    batch_size: int = 10,
    dry_run: bool = False,
    skip_existing: bool = True,
) -> dict:
    """
    Backfill existing knowledge_base entries to knowledge_base_chunks.
    
    Args:
        org_id: Optional org ID to filter
        batch_size: Number of entries to process at once
        dry_run: If True, don't actually write to database
        skip_existing: Skip entries that already have chunks
        
    Returns:
        Statistics dict
    """
    supabase = get_supabase_client()
    chunker = Chunker(
        chunk_size=SETTINGS.rag_chunk_size,
        chunk_overlap=SETTINGS.rag_chunk_overlap,
    )
    embedder = Embedder()
    
    stats = {
        "entries_processed": 0,
        "entries_skipped": 0,
        "chunks_created": 0,
        "chunks_embedded": 0,
        "errors": [],
    }
    
    # Build query for entries to backfill
    query = supabase.table("knowledge_base").select(
        "id, org_id, category, title, content, keywords, source_document, is_chunked"
    )
    
    if org_id:
        query = query.eq("org_id", org_id)
    
    # Only get entries not yet chunked
    if skip_existing:
        query = query.or_("is_chunked.is.null,is_chunked.eq.false")
    
    response = query.execute()
    entries = response.data
    
    if not entries:
        console.print("[yellow]No entries found to backfill[/yellow]")
        return stats
    
    console.print(f"Found [bold]{len(entries)}[/bold] entries to backfill")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        
        task = progress.add_task(
            "[cyan]Backfilling entries...",
            total=len(entries),
        )
        
        for entry in entries:
            try:
                entry_id = entry["id"]
                entry_org_id = entry["org_id"]
                
                progress.update(
                    task,
                    description=f"[cyan]Processing: {entry.get('title', entry_id)[:40]}...",
                )
                
                # Chunk the content
                chunks = chunker.chunk_text_simple(
                    text=entry["content"],
                    source_document=entry.get("source_document") or "knowledge_base",
                    page_number=1,
                )
                
                if not chunks:
                    stats["entries_skipped"] += 1
                    progress.advance(task)
                    continue
                
                if dry_run:
                    console.print(
                        f"  [dim]Would create {len(chunks)} chunks for: "
                        f"{entry.get('title', entry_id)[:40]}[/dim]"
                    )
                    stats["chunks_created"] += len(chunks)
                    stats["entries_processed"] += 1
                    progress.advance(task)
                    continue
                
                # Embed chunks
                embedding_results = await embedder.embed_chunks(chunks)
                
                # Store chunks
                for result in embedding_results:
                    if result.error and not result.text_embedding:
                        stats["errors"].append(f"Embedding failed: {result.error}")
                        continue
                    
                    chunk = result.chunk
                    
                    data = {
                        "org_id": entry_org_id,
                        "knowledge_base_id": entry_id,
                        "category": entry.get("category"),
                        "title": entry.get("title"),
                        "source_document": entry.get("source_document") or "knowledge_base",
                        "modality": chunk.chunk_type.value,
                        "page": chunk.page_number,
                        "chunk_index": chunk.chunk_index,
                        "content": chunk.content,
                        "content_hash": chunk.content_hash,
                        "is_table": chunk.is_table,
                    }
                    
                    if result.text_embedding:
                        data["vector_embedding"] = result.text_embedding
                        stats["chunks_embedded"] += 1
                    
                    # Remove None values
                    data = {k: v for k, v in data.items() if v is not None}
                    
                    try:
                        supabase.table("knowledge_base_chunks").insert(data).execute()
                        stats["chunks_created"] += 1
                    except Exception as e:
                        if "duplicate" in str(e).lower():
                            # Skip duplicates
                            pass
                        else:
                            stats["errors"].append(f"Insert failed: {str(e)}")
                
                # Mark entry as chunked
                supabase.table("knowledge_base").update(
                    {"is_chunked": True}
                ).eq("id", entry_id).execute()
                
                stats["entries_processed"] += 1
                
            except Exception as e:
                logger.error(f"Error processing entry {entry.get('id')}: {e}")
                stats["errors"].append(str(e))
                stats["entries_skipped"] += 1
            
            progress.advance(task)
    
    return stats


def print_summary(stats: dict) -> None:
    """Print backfill summary."""
    console.print()
    
    table = Table(title="Backfill Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Entries Processed", str(stats["entries_processed"]))
    table.add_row("Entries Skipped", str(stats["entries_skipped"]))
    table.add_row("Chunks Created", str(stats["chunks_created"]))
    table.add_row("Chunks Embedded", str(stats["chunks_embedded"]))
    table.add_row("Errors", str(len(stats["errors"])))
    
    console.print(table)
    
    if stats["errors"]:
        console.print()
        console.print("[yellow]Errors:[/yellow]")
        for error in stats["errors"][:10]:
            console.print(f"  • {error}")
        if len(stats["errors"]) > 10:
            console.print(f"  ... and {len(stats['errors']) - 10} more")


def main():
    parser = argparse.ArgumentParser(
        description="Backfill existing knowledge_base entries to knowledge_base_chunks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--org-id",
        type=str,
        help="Organization ID to backfill (default: all orgs)",
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of entries to process at once (default: 10)",
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without writing to database",
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-process entries that have already been chunked",
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    
    # Setup
    setup_logging(args.verbose)
    
    console.print("[bold]Knowledge Base Backfill[/bold]")
    console.print()
    
    if args.dry_run:
        console.print("[yellow]Dry run mode - no changes will be made[/yellow]")
        console.print()
    
    # Run backfill
    stats = asyncio.run(backfill_kb_entries(
        org_id=args.org_id,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        skip_existing=not args.force,
    ))
    
    # Print summary
    print_summary(stats)


if __name__ == "__main__":
    main()
