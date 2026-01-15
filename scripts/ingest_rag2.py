#!/usr/bin/env python3
"""
RAG 2.0 Document Ingestion CLI

Ingest documents into the RAG 2.0 system with:
- Hierarchical parent/child chunking
- Matryoshka embeddings (4096→1024)
- NER entity extraction
- Deduplication via content hash

Usage:
    python scripts/ingest_rag2.py --file document.pdf --org-id <org-id>
    python scripts/ingest_rag2.py --dir /path/to/docs --org-id <org-id>
    python scripts/ingest_rag2.py --file document.pdf --org-id <org-id> --collection policies
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import List

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from voice_agent.rag2.ingest import RAG2Ingestor, IngestResult


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def discover_files(path: Path, extensions: List[str]) -> List[Path]:
    """Discover files to ingest."""
    if path.is_file():
        return [path]
    
    files = []
    for ext in extensions:
        files.extend(path.rglob(f"*{ext}"))
    return sorted(files)


async def ingest_single(
    ingestor: RAG2Ingestor,
    file_path: Path,
    collection: str | None,
) -> IngestResult:
    """Ingest a single file."""
    return await ingestor.ingest_file(str(file_path), collection=collection)


async def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="RAG 2.0 Document Ingestion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest a single PDF
  python ingest_rag2.py --file docs/policy.pdf --org-id org_123
  
  # Ingest all documents in a directory
  python ingest_rag2.py --dir docs/policies/ --org-id org_123 --collection policies
  
  # Verbose output with custom embedding dimension
  python ingest_rag2.py --file doc.pdf --org-id org_123 -v --embed-dim 1024
""",
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--file", "-f", type=Path, help="Single file to ingest")
    input_group.add_argument("--dir", "-d", type=Path, help="Directory to ingest recursively")
    
    # Required
    parser.add_argument("--org-id", "-o", required=True, help="Organization ID")
    
    # Optional
    parser.add_argument("--collection", "-c", help="Collection/category name")
    parser.add_argument("--embed-dim", type=int, default=1024, help="Embedding dimension (default: 1024)")
    parser.add_argument("--extensions", nargs="+", default=[".pdf", ".docx", ".txt", ".md"],
                        help="File extensions to process (default: .pdf .docx .txt .md)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip files that already exist")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without doing it")
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Discover files
    if args.file:
        if not args.file.exists():
            logger.error(f"File not found: {args.file}")
            return 1
        files = [args.file]
    else:
        if not args.dir.exists():
            logger.error(f"Directory not found: {args.dir}")
            return 1
        files = discover_files(args.dir, args.extensions)
    
    if not files:
        logger.warning("No files found to ingest")
        return 0
    
    logger.info(f"Found {len(files)} file(s) to ingest")
    
    if args.dry_run:
        for f in files:
            print(f"  Would ingest: {f}")
        return 0
    
    # Create ingestor
    ingestor = RAG2Ingestor(
        org_id=args.org_id,
    )
    
    # Process files
    results: List[IngestResult] = []
    success_count = 0
    skip_count = 0
    error_count = 0
    
    for i, file_path in enumerate(files, 1):
        logger.info(f"[{i}/{len(files)}] Processing: {file_path}")
        
        try:
            result = await ingest_single(ingestor, file_path, args.collection)
            results.append(result)
            
            if result.success:
                if result.stats.documents_skipped > 0:
                    skip_count += 1
                    logger.info(f"  ⏭️  Skipped (already exists): {result.document_id}")
                else:
                    success_count += 1
                    stats = result.stats
                    logger.info(
                        f"  ✅ Ingested: {stats.parent_chunks_created} parents, "
                        f"{stats.child_chunks_created} children"
                    )
            else:
                error_count += 1
                logger.error(f"  ❌ Failed: {result.error}")
                
        except Exception as e:
            error_count += 1
            logger.exception(f"  ❌ Exception: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print("INGESTION SUMMARY")
    print("=" * 50)
    print(f"Total files:  {len(files)}")
    print(f"✅ Success:   {success_count}")
    print(f"⏭️  Skipped:   {skip_count}")
    print(f"❌ Errors:    {error_count}")
    
    if results:
        total_parents = sum(r.stats.parent_chunks_created for r in results)
        total_children = sum(r.stats.child_chunks_created for r in results)
        total_embedded = sum(r.stats.child_chunks_embedded for r in results)
        print(f"\nChunks created:")
        print(f"  Parent chunks:  {total_parents}")
        print(f"  Child chunks:   {total_children}")
        print(f"  Embedded:       {total_embedded}")
    
    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
