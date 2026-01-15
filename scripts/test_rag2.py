#!/usr/bin/env python3
"""
RAG 2.0 Test CLI

Test the RAG 2.0 retrieval pipeline with sample queries.

Usage:
    python scripts/test_rag2.py --query "What is the policy on vacation?" --org-id <org-id>
    python scripts/test_rag2.py --interactive --org-id <org-id>
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from voice_agent.rag2.retrieval import RAG2Retriever, RetrievalResult


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def print_result(result: RetrievalResult, show_full: bool = False) -> None:
    """Pretty-print retrieval result."""
    print("\n" + "=" * 60)
    print("RETRIEVAL RESULT")
    print("=" * 60)
    
    if not result.success:
        print(f"❌ FAILED: {result.refusal_reason}")
        return
    
    if result.refused:
        print(f"⚠️  REFUSED: {result.refusal_reason}")
        print(f"   Max score: {result.max_rerank_score:.3f}")
        return
    
    print(f"✅ SUCCESS - {len(result.contexts)} contexts retrieved")
    print(f"   Max rerank score: {result.max_rerank_score:.3f}")
    
    if result.query_plan:
        plan = result.query_plan
        print(f"\n📋 Query Plan:")
        print(f"   Intent: {plan.intent}")
        print(f"   Keywords: {plan.keywords}")
        print(f"   Requires graph: {plan.requires_graph}")
        if plan.cypher_query:
            print(f"   Cypher: {plan.cypher_query[:80]}...")
    
    if result.timings:
        print(f"\n⏱️  Timings:")
        for stage, time_ms in result.timings.items():
            print(f"   {stage}: {time_ms:.1f}ms")
    
    print(f"\n📚 Contexts (top {min(5, len(result.contexts))}):")
    for i, ctx in enumerate(result.contexts[:5], 1):
        print(f"\n--- Context {i} ---")
        print(f"   RRF Score: {ctx.rrf_score:.4f}")
        if ctx.rerank_score is not None:
            print(f"   Rerank Score: {ctx.rerank_score:.4f}")
        print(f"   Document: {ctx.document_id[:20]}...")
        print(f"   Page: {ctx.page}")
        if ctx.section_heading:
            print(f"   Section: {ctx.section_heading}")
        
        # Text preview
        text = ctx.text[:300] if not show_full else ctx.text
        if len(ctx.text) > 300 and not show_full:
            text += "..."
        print(f"   Text: {text}")
        
        # Parent context (if available)
        if ctx.parent_text and show_full:
            print(f"   Parent: {ctx.parent_text[:500]}...")


async def run_query(
    retriever: RAG2Retriever,
    query: str,
    collection: str | None,
    top_k: int,
    show_full: bool,
) -> bool:
    """Run a single query and display results."""
    print(f"\n🔍 Query: {query}")
    
    result = await retriever.retrieve(
        query=query,
        collection=collection,
        top_k=top_k,
    )
    
    print_result(result, show_full)
    return result.success and not result.refused


async def interactive_mode(
    retriever: RAG2Retriever,
    collection: str | None,
    top_k: int,
    show_full: bool,
) -> None:
    """Interactive query mode."""
    print("\n🎮 Interactive RAG 2.0 Test Mode")
    print("   Type your queries, or 'quit' to exit")
    print("   Commands: /full (toggle full text), /top N (set top_k)")
    print("-" * 50)
    
    while True:
        try:
            query = input("\n🔍 Query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break
        
        if not query:
            continue
        
        if query.lower() in ("quit", "exit", "q"):
            break
        
        if query.startswith("/full"):
            show_full = not show_full
            print(f"   Full text mode: {'ON' if show_full else 'OFF'}")
            continue
        
        if query.startswith("/top "):
            try:
                top_k = int(query.split()[1])
                print(f"   Top K set to: {top_k}")
            except (IndexError, ValueError):
                print("   Usage: /top N")
            continue
        
        await run_query(retriever, query, collection, top_k, show_full)


async def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="RAG 2.0 Retrieval Test CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single query
  python test_rag2.py --query "What is the refund policy?" --org-id org_123
  
  # Interactive mode
  python test_rag2.py --interactive --org-id org_123
  
  # With collection filter
  python test_rag2.py --query "vacation policy" --org-id org_123 --collection hr
  
  # Show full context
  python test_rag2.py --query "..." --org-id org_123 --full
""",
    )
    
    # Query options
    query_group = parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument("--query", "-q", help="Single query to execute")
    query_group.add_argument("--interactive", "-i", action="store_true", 
                             help="Interactive mode")
    
    # Required
    parser.add_argument("--org-id", "-o", required=True, help="Organization ID")
    
    # Optional
    parser.add_argument("--collection", "-c", help="Collection/category filter")
    parser.add_argument("--top-k", "-k", type=int, default=5, help="Number of results (default: 5)")
    parser.add_argument("--full", "-f", action="store_true", help="Show full text")
    parser.add_argument("--graph", "-g", action="store_true", help="Enable graph search")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    # Create retriever
    retriever = RAG2Retriever(
        org_id=args.org_id,
        graph_enabled=args.graph,
    )
    
    if args.interactive:
        await interactive_mode(
            retriever=retriever,
            collection=args.collection,
            top_k=args.top_k,
            show_full=args.full,
        )
        return 0
    
    # Single query
    result = await retriever.retrieve(
        query=args.query,
        collection=args.collection,
        top_k=args.top_k,
    )
    
    if args.json:
        # JSON output
        output = {
            "success": result.success,
            "refused": result.refused,
            "refusal_reason": result.refusal_reason,
            "max_score": result.max_rerank_score,
            "contexts": [
                {
                    "child_id": ctx.child_id,
                    "document_id": ctx.document_id,
                    "page": ctx.page,
                    "rrf_score": ctx.rrf_score,
                    "rerank_score": ctx.rerank_score,
                    "text": ctx.text[:500],
                    "section": ctx.section_heading,
                }
                for ctx in result.contexts
            ],
            "timings": result.timings,
        }
        print(json.dumps(output, indent=2))
    else:
        print_result(result, args.full)
    
    return 0 if result.success and not result.refused else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
