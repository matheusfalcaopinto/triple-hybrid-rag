#!/usr/bin/env python3
"""
RAG 2.0 Backfill Script

Migrates data from RAG 1.0 schema to RAG 2.0 schema.

This script:
1. Reads existing chunks from knowledge_base_chunks table
2. Creates document records in rag_documents
3. Creates parent chunks from existing data
4. Creates child chunks with new embeddings (1024d Matryoshka)
5. Preserves content_hash for deduplication

Usage:
    python scripts/backfill_rag2.py --org-id <org-id>
    python scripts/backfill_rag2.py --org-id <org-id> --dry-run
    python scripts/backfill_rag2.py --org-id <org-id> --batch-size 50
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from voice_agent.config import SETTINGS
from voice_agent.utils.db import get_supabase_client


@dataclass
class BackfillStats:
    """Statistics from backfill process."""
    documents_created: int = 0
    documents_skipped: int = 0
    parent_chunks_created: int = 0
    child_chunks_created: int = 0
    child_chunks_embedded: int = 0
    errors: Optional[List[str]] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
    
    @property
    def duration_seconds(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def generate_doc_hash(filename: str, org_id: str) -> str:
    """Generate document hash from filename and org."""
    content = f"{org_id}:{filename}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


async def get_existing_chunks(
    supabase: Any,
    org_id: str,
    batch_size: int = 100,
) -> List[Dict[str, Any]]:
    """Fetch existing chunks from RAG 1.0 schema."""
    all_chunks = []
    offset = 0
    
    while True:
        response = supabase.table("knowledge_base_chunks").select(
            "id, chunk_text, chunk_index, source_document, source_page, "
            "source_category, content_hash, ocr_confidence, "
            "created_at"
        ).eq("org_id", org_id).order("source_document").order("chunk_index").range(
            offset, offset + batch_size - 1
        ).execute()
        
        if not response.data:
            break
        
        all_chunks.extend(response.data)
        offset += batch_size
        
        if len(response.data) < batch_size:
            break
    
    return all_chunks


def group_chunks_by_document(chunks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group chunks by source document."""
    grouped = defaultdict(list)
    for chunk in chunks:
        doc_name = chunk.get("source_document", "unknown")
        grouped[doc_name].append(chunk)
    
    # Sort each group by chunk_index
    for doc_name in grouped:
        grouped[doc_name].sort(key=lambda c: c.get("chunk_index", 0))
    
    return dict(grouped)


async def create_document_record(
    supabase: Any,
    org_id: str,
    filename: str,
    collection: Optional[str],
    chunk_count: int,
    dry_run: bool = False,
) -> Optional[str]:
    """Create a document record in rag_documents table."""
    doc_hash = generate_doc_hash(filename, org_id)
    doc_id = f"doc_{doc_hash}"
    
    if dry_run:
        return doc_id
    
    try:
        response = supabase.table("rag_documents").upsert({
            "id": doc_id,
            "org_id": org_id,
            "filename": filename,
            "content_hash": doc_hash,
            "collection": collection,
            "page_count": 1,  # Unknown from legacy data
            "char_count": 0,  # Unknown
            "status": "migrated",
            "metadata": {"source": "rag1_backfill", "original_chunks": chunk_count},
        }, on_conflict="id").execute()
        
        return doc_id
    except Exception as e:
        logging.error(f"Failed to create document {filename}: {e}")
        return None


async def create_parent_chunk(
    supabase: Any,
    org_id: str,
    doc_id: str,
    parent_idx: int,
    text: str,
    page: int,
    section: Optional[str],
    dry_run: bool = False,
) -> Optional[str]:
    """Create a parent chunk."""
    parent_id = f"{doc_id}:p{parent_idx}"
    
    if dry_run:
        return parent_id
    
    try:
        response = supabase.table("rag_parent_chunks").upsert({
            "id": parent_id,
            "document_id": doc_id,
            "org_id": org_id,
            "index_in_document": parent_idx,
            "text": text,
            "token_count": len(text) // 4,
            "page_start": page,
            "page_end": page,
            "section_heading": section,
        }, on_conflict="id").execute()
        
        return parent_id
    except Exception as e:
        logging.error(f"Failed to create parent chunk {parent_id}: {e}")
        return None


async def create_child_chunk(
    supabase: Any,
    org_id: str,
    parent_id: str,
    child_idx: int,
    text: str,
    page: int,
    content_hash: str,
    embedding: Optional[List[float]],
    dry_run: bool = False,
) -> Optional[str]:
    """Create a child chunk."""
    child_id = f"{parent_id}:c{child_idx}"
    
    if dry_run:
        return child_id
    
    try:
        data = {
            "id": child_id,
            "parent_chunk_id": parent_id,
            "org_id": org_id,
            "index_in_parent": child_idx,
            "text": text,
            "token_count": len(text) // 4,
            "start_char_offset": 0,
            "end_char_offset": len(text),
            "page_number": page,
            "modality": "text",
            "content_hash": content_hash,
        }
        
        if embedding:
            data["embedding_1024"] = embedding
        
        response = supabase.table("rag_child_chunks").upsert(
            data, on_conflict="id"
        ).execute()
        
        return child_id
    except Exception as e:
        logging.error(f"Failed to create child chunk {child_id}: {e}")
        return None


async def embed_text_matryoshka(text: str) -> Optional[List[float]]:
    """Embed text using Matryoshka truncation."""
    try:
        from voice_agent.rag2.embedder import get_rag2_embedder
        
        embedder = get_rag2_embedder()
        result = embedder.embed_text(text)
        
        if result.error:
            logging.warning(f"Embedding error: {result.error}")
            return None
        
        return result.embedding
    except Exception as e:
        logging.error(f"Failed to embed text: {e}")
        return None


async def backfill_document(
    supabase: Any,
    org_id: str,
    doc_name: str,
    chunks: List[Dict[str, Any]],
    stats: BackfillStats,
    embed: bool = True,
    dry_run: bool = False,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Backfill a single document."""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Ensure errors list exists
    if stats.errors is None:
        stats.errors = []
    
    collection = chunks[0].get("source_category") if chunks else None
    
    # Create document record
    doc_id = await create_document_record(
        supabase, org_id, doc_name, collection, len(chunks), dry_run
    )
    
    if not doc_id:
        stats.errors.append(f"Failed to create document: {doc_name}")
        return
    
    stats.documents_created += 1
    
    # Group chunks into parent-sized groups
    # Assuming ~5 old chunks per parent (1000 chars each = 5000 chars parent)
    CHUNKS_PER_PARENT = 5
    parent_groups = [
        chunks[i:i + CHUNKS_PER_PARENT]
        for i in range(0, len(chunks), CHUNKS_PER_PARENT)
    ]
    
    for parent_idx, group in enumerate(parent_groups):
        # Combine chunks into parent text
        parent_text = "\n\n".join(c["chunk_text"] for c in group)
        page = group[0].get("source_page", 1)
        
        # Create parent chunk
        parent_id = await create_parent_chunk(
            supabase, org_id, doc_id, parent_idx,
            parent_text, page, None, dry_run
        )
        
        if not parent_id:
            stats.errors.append(f"Failed to create parent {parent_idx} for {doc_name}")
            continue
        
        stats.parent_chunks_created += 1
        
        # Create child chunks (each old chunk becomes a child)
        for child_idx, chunk in enumerate(group):
            content_hash = chunk.get("content_hash", "")
            if not content_hash:
                content_hash = hashlib.sha256(
                    chunk["chunk_text"].strip().lower().encode()
                ).hexdigest()
            
            # Optionally embed
            embedding = None
            if embed and not dry_run:
                embedding = await embed_text_matryoshka(chunk["chunk_text"])
                if embedding:
                    stats.child_chunks_embedded += 1
            
            child_id = await create_child_chunk(
                supabase, org_id, parent_id, child_idx,
                chunk["chunk_text"], chunk.get("source_page", 1),
                content_hash, embedding, dry_run
            )
            
            if child_id:
                stats.child_chunks_created += 1


async def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="RAG 2.0 Backfill from RAG 1.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run to see what would be migrated
  python backfill_rag2.py --org-id org_123 --dry-run
  
  # Full migration with embeddings
  python backfill_rag2.py --org-id org_123
  
  # Migration without re-embedding (faster)
  python backfill_rag2.py --org-id org_123 --skip-embed
""",
    )
    
    parser.add_argument("--org-id", "-o", required=True, help="Organization ID")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    parser.add_argument("--skip-embed", action="store_true", 
                        help="Skip embedding (use existing or null)")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Fetch batch size (default: 100)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting RAG 2.0 backfill for org: {args.org_id}")
    if args.dry_run:
        logger.info("DRY RUN - no changes will be made")
    
    # Initialize
    supabase = get_supabase_client()
    stats = BackfillStats(start_time=datetime.now())
    
    # Fetch existing chunks
    logger.info("Fetching existing chunks from RAG 1.0...")
    chunks = await get_existing_chunks(supabase, args.org_id, args.batch_size)
    logger.info(f"Found {len(chunks)} chunks to migrate")
    
    if not chunks:
        logger.warning("No chunks found to migrate")
        return 0
    
    # Group by document
    grouped = group_chunks_by_document(chunks)
    logger.info(f"Found {len(grouped)} documents")
    
    # Process each document
    for i, (doc_name, doc_chunks) in enumerate(grouped.items(), 1):
        logger.info(f"[{i}/{len(grouped)}] Processing: {doc_name} ({len(doc_chunks)} chunks)")
        
        await backfill_document(
            supabase=supabase,
            org_id=args.org_id,
            doc_name=doc_name,
            chunks=doc_chunks,
            stats=stats,
            embed=not args.skip_embed,
            dry_run=args.dry_run,
            logger=logger,
        )
    
    stats.end_time = datetime.now()
    
    # Summary
    print("\n" + "=" * 50)
    print("BACKFILL SUMMARY")
    print("=" * 50)
    print(f"Duration:           {stats.duration_seconds:.1f}s")
    print(f"Documents created:  {stats.documents_created}")
    print(f"Parent chunks:      {stats.parent_chunks_created}")
    print(f"Child chunks:       {stats.child_chunks_created}")
    print(f"Chunks embedded:    {stats.child_chunks_embedded}")
    
    if stats.errors:
        print(f"\n❌ Errors ({len(stats.errors)}):")
        for error in stats.errors[:10]:
            print(f"  - {error}")
        if len(stats.errors) > 10:
            print(f"  ... and {len(stats.errors) - 10} more")
    
    return 0 if not stats.errors else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
