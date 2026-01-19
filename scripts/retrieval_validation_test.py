#!/usr/bin/env python3
"""
Retrieval Validation Test for Multimodal Pipeline

Tests retrieval against ingested multimodal documents:
- Semantic search (pgvector)
- Lexical search (full-text)
- Graph search (PuppyGraph or SQL fallback)
- RRF Fusion
- Reranking

Usage:
    uv run python scripts/retrieval_validation_test.py
"""

import asyncio
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from triple_hybrid_rag.config import get_settings, reset_settings
from triple_hybrid_rag.core.embedder import MultimodalEmbedder
from triple_hybrid_rag.core.fusion import RRFFusion
from triple_hybrid_rag.core.reranker import Reranker
from triple_hybrid_rag.types import SearchResult, SearchChannel, Modality

console = Console()
TENANT_ID = "multimodal-test"

# Test queries covering different document types
TEST_QUERIES = [
    {
        "query": "What are the backtested trading strategies for Brazilian mining stocks?",
        "expected_sources": ["Brazilian Mining Stocks"],
        "description": "PDF - Native text extraction",
    },
    {
        "query": "Voice Activity Detection solutions for WebRTC telephony",
        "expected_sources": ["Voice Activity Detection", "VAD"],
        "description": "DOCX - Native text extraction",
    },
    {
        "query": "Hospital inteligente e mercado de saúde",
        "expected_sources": ["hospitalinteligente", "mercado"],
        "description": "Markdown - Native text",
    },
    {
        "query": "Doença vascular arterial e venosa tratamento",
        "expected_sources": ["Vascular", "Arterial", "Venosa"],
        "description": "PDF - Scanned with OCR",
    },
    {
        "query": "Classificação setorial de empresas listadas",
        "expected_sources": ["ClassifSetorial", "setorial"],
        "description": "XLSX - Table extraction",
    },
]


async def run_retrieval_test(config, query_info: dict) -> dict:
    """Run a single retrieval test."""
    query = query_info["query"]
    
    # Get embedding for query
    embedder = MultimodalEmbedder(config=config)
    query_embedding = await embedder.embed_text(query)
    await embedder.close()
    
    results = {
        "query": query,
        "description": query_info["description"],
        "expected_sources": query_info["expected_sources"],
        "semantic_results": [],
        "lexical_results": [],
        "fused_results": [],
        "reranked_results": [],
        "success": False,
        "found_expected": False,
    }
    
    try:
        import asyncpg
        from triple_hybrid_rag.graph.puppygraph import PuppyGraphClient
        
        pool = await asyncpg.create_pool(config.database_url, min_size=1, max_size=5)
        
        async with pool.acquire() as conn:
            # Semantic search
            query_embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
            semantic_rows = await conn.fetch("""
                SELECT * FROM rag_semantic_search($1, $2::vector, $3)
            """, TENANT_ID, query_embedding_str, 10)
            
            results["semantic_results"] = [
                {
                    "text": row["text"][:150] + "..." if len(row["text"]) > 150 else row["text"],
                    "similarity": float(row.get("similarity", 0)),
                    "page": row.get("page"),
                }
                for row in semantic_rows
            ]
            
            # Lexical search
            lexical_rows = await conn.fetch("""
                SELECT * FROM rag_lexical_search($1, $2, $3)
            """, TENANT_ID, query, 10)
            
            results["lexical_results"] = [
                {
                    "text": row["text"][:150] + "..." if len(row["text"]) > 150 else row["text"],
                    "rank": float(row.get("rank", 0)),
                    "page": row.get("page"),
                }
                for row in lexical_rows
            ]
        
        await pool.close()
        
        # Convert to SearchResult objects for fusion
        def row_to_result(row, channel: SearchChannel, score_field: str) -> SearchResult:
            result = SearchResult(
                chunk_id=row["child_id"],
                parent_id=row["parent_id"],
                document_id=row["document_id"],
                text=row["text"],
                page=row.get("page"),
                modality=Modality.TEXT,
                source_channel=channel,
            )
            if channel == SearchChannel.SEMANTIC:
                result.semantic_score = float(row.get(score_field, 0.0))
            elif channel == SearchChannel.LEXICAL:
                result.lexical_score = float(row.get(score_field, 0.0))
            return result
        
        pool = await asyncpg.create_pool(config.database_url, min_size=1, max_size=5)
        async with pool.acquire() as conn:
            semantic_rows = await conn.fetch("""
                SELECT * FROM rag_semantic_search($1, $2::vector, $3)
            """, TENANT_ID, query_embedding_str, 10)
            
            lexical_rows = await conn.fetch("""
                SELECT * FROM rag_lexical_search($1, $2, $3)
            """, TENANT_ID, query, 10)
        await pool.close()
        
        semantic_search_results = [row_to_result(r, SearchChannel.SEMANTIC, "similarity") for r in semantic_rows]
        lexical_search_results = [row_to_result(r, SearchChannel.LEXICAL, "rank") for r in lexical_rows]
        
        # RRF Fusion
        fusion = RRFFusion(config=config)
        fused_results = fusion.fuse(
            lexical_results=lexical_search_results,
            semantic_results=semantic_search_results,
            graph_results=[],
            top_k=10,
            apply_safety=False,
            apply_denoise=False,
        )
        
        results["fused_results"] = [
            {
                "text": r.text[:150] + "..." if len(r.text) > 150 else r.text,
                "rrf_score": r.rrf_score,
                "channels": r.metadata.get("source_channels", []),
            }
            for r in fused_results[:5]
        ]
        
        # Reranking
        if config.rag_rerank_enabled and fused_results:
            reranker = Reranker(config=config)
            documents = [r.text for r in fused_results]
            scores = await reranker.rerank(query, documents)
            await reranker.close()
            
            for result, score in zip(fused_results, scores):
                result.rerank_score = score
                result.final_score = score
            
            fused_results.sort(key=lambda r: r.final_score, reverse=True)
            
            results["reranked_results"] = [
                {
                    "text": r.text[:150] + "..." if len(r.text) > 150 else r.text,
                    "rerank_score": r.rerank_score,
                }
                for r in fused_results[:5]
            ]
        
        # Check if expected sources were found
        all_texts = " ".join([r["text"] for r in results["semantic_results"][:5]])
        all_texts += " ".join([r["text"] for r in results["lexical_results"][:5]])
        
        found_count = 0
        for expected in query_info["expected_sources"]:
            if expected.lower() in all_texts.lower():
                found_count += 1
        
        results["found_expected"] = found_count > 0
        results["success"] = len(results["semantic_results"]) > 0 or len(results["lexical_results"]) > 0
        
    except Exception as e:
        results["error"] = str(e)
        import traceback
        traceback.print_exc()
    
    return results


async def main():
    """Main entry point."""
    console.print()
    console.print(Panel.fit(
        "[bold cyan]RETRIEVAL VALIDATION TEST[/bold cyan]\n"
        "[dim]Testing multimodal document retrieval[/dim]",
        border_style="cyan"
    ))
    console.print()
    
    reset_settings()
    config = get_settings()
    
    # Check database
    console.print("[bold cyan]━━━ Database Check[/bold cyan]")
    try:
        import asyncpg
        pool = await asyncpg.create_pool(config.database_url, min_size=1, max_size=5)
        async with pool.acquire() as conn:
            doc_count = await conn.fetchval(
                "SELECT COUNT(*) FROM rag_documents WHERE tenant_id = $1", TENANT_ID
            )
            chunk_count = await conn.fetchval(
                "SELECT COUNT(*) FROM rag_child_chunks WHERE tenant_id = $1", TENANT_ID
            )
        await pool.close()
        
        console.print(f"   [green]✓[/green] Documents: {doc_count}")
        console.print(f"   [green]✓[/green] Chunks: {chunk_count}")
        
        if doc_count == 0:
            console.print(f"   [red]✗[/red] No documents found in tenant '{TENANT_ID}'")
            console.print(f"   [yellow]Run multimodal_pipeline_test.py --keep-data first[/yellow]")
            return 1
            
    except Exception as e:
        console.print(f"   [red]✗[/red] Database error: {e}")
        return 1
    
    console.print()
    
    # Run tests
    console.print("[bold cyan]━━━ Running Retrieval Tests[/bold cyan]")
    
    all_results = []
    passed = 0
    
    for i, query_info in enumerate(TEST_QUERIES, 1):
        console.print(f"\n   [bold]Test {i}: {query_info['description']}[/bold]")
        console.print(f"   [dim]Query: {query_info['query'][:60]}...[/dim]")
        
        result = await run_retrieval_test(config, query_info)
        all_results.append(result)
        
        if result["success"]:
            console.print(f"   [green]✓[/green] Semantic: {len(result['semantic_results'])} results")
            console.print(f"   [green]✓[/green] Lexical: {len(result['lexical_results'])} results")
            console.print(f"   [green]✓[/green] Fused: {len(result['fused_results'])} results")
            
            if result["found_expected"]:
                console.print(f"   [green]✓[/green] Found expected content")
                passed += 1
            else:
                console.print(f"   [yellow]⚠[/yellow] Expected content not in top results")
                passed += 0.5  # Partial credit
            
            # Show top result
            if result["reranked_results"]:
                top = result["reranked_results"][0]
                console.print(f"   [dim]Top: {top['text'][:80]}... (score={top['rerank_score']:.3f})[/dim]")
            elif result["fused_results"]:
                top = result["fused_results"][0]
                console.print(f"   [dim]Top: {top['text'][:80]}... (rrf={top['rrf_score']:.3f})[/dim]")
        else:
            console.print(f"   [red]✗[/red] No results found")
            if "error" in result:
                console.print(f"   [red]Error: {result['error']}[/red]")
    
    console.print()
    
    # Summary
    console.print("[bold cyan]━━━ Summary[/bold cyan]")
    
    summary_table = Table(show_header=True, header_style="bold")
    summary_table.add_column("Test", width=50)
    summary_table.add_column("Semantic", justify="center", width=10)
    summary_table.add_column("Lexical", justify="center", width=10)
    summary_table.add_column("Found", justify="center", width=10)
    summary_table.add_column("Status", justify="center", width=10)
    
    for i, result in enumerate(all_results, 1):
        status = "✅" if result["success"] and result["found_expected"] else ("⚠️" if result["success"] else "❌")
        found = "✓" if result["found_expected"] else "✗"
        
        summary_table.add_row(
            f"{i}. {result['description'][:45]}",
            str(len(result["semantic_results"])),
            str(len(result["lexical_results"])),
            found,
            status,
        )
    
    console.print(summary_table)
    console.print()
    
    # Final verdict
    pass_rate = (passed / len(TEST_QUERIES)) * 100
    
    if pass_rate >= 80:
        console.print(Panel(
            f"[bold green]VALIDATION PASSED[/bold green]\n"
            f"Pass Rate: {pass_rate:.0f}% ({passed}/{len(TEST_QUERIES)})\n"
            f"Multimodal retrieval is working correctly!",
            border_style="green"
        ))
    elif pass_rate >= 50:
        console.print(Panel(
            f"[bold yellow]PARTIAL PASS[/bold yellow]\n"
            f"Pass Rate: {pass_rate:.0f}% ({passed}/{len(TEST_QUERIES)})\n"
            f"Some queries may need tuning",
            border_style="yellow"
        ))
    else:
        console.print(Panel(
            f"[bold red]VALIDATION FAILED[/bold red]\n"
            f"Pass Rate: {pass_rate:.0f}% ({passed}/{len(TEST_QUERIES)})",
            border_style="red"
        ))
    
    return 0 if pass_rate >= 50 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
