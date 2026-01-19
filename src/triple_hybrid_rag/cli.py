"""Command-line interface for Triple-Hybrid-RAG."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from triple_hybrid_rag import RAG, get_settings


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Triple-Hybrid-RAG CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Ingest a document")
    ingest.add_argument("file", type=str, help="Path to a .txt or .md file")
    ingest.add_argument("--tenant", default="default", help="Tenant identifier")
    ingest.add_argument("--collection", default="general", help="Collection name")
    ingest.add_argument("--title", default=None, help="Document title")
    ingest.add_argument(
        "--ocr-mode",
        choices=["qwen", "deepseek", "off", "auto"],
        default=None,
        help="OCR mode: qwen (Qwen3-VL), deepseek (DeepSeek OCR), off (no OCR), auto (system decides)",
    )

    retrieve = subparsers.add_parser("retrieve", help="Run a retrieval query")
    retrieve.add_argument("query", type=str, help="User query")
    retrieve.add_argument("--tenant", default="default", help="Tenant identifier")
    retrieve.add_argument("--collection", default=None, help="Collection name")
    retrieve.add_argument("--top-k", type=int, default=None, help="Number of results")

    return parser


async def _run_ingest(args: argparse.Namespace) -> None:
    settings = get_settings()
    
    # Override OCR mode if specified via CLI
    if args.ocr_mode:
        settings.rag_ocr_mode = args.ocr_mode
    
    rag = RAG(settings)
    try:
        result = await rag.ingest(
            file_path=args.file,
            tenant_id=args.tenant,
            collection=args.collection,
            title=args.title,
        )
        print(json.dumps(result.to_dict(), default=str, indent=2))
    finally:
        await rag.close()


async def _run_retrieve(args: argparse.Namespace) -> None:
    rag = RAG(get_settings())
    try:
        result = await rag.retrieve(
            query=args.query,
            tenant_id=args.tenant,
            collection=args.collection,
            top_k=args.top_k,
        )
        payload = {
            "query": result.query,
            "results": [
                {
                    "id": str(r.chunk_id),
                    "parent_id": str(r.parent_id),
                    "score": r.rerank_score or r.final_score or r.rrf_score,
                    "text": r.text,
                }
                for r in result.results
            ],
            "metadata": result.metadata,
        }
        print(json.dumps(payload, default=str, indent=2))
    finally:
        await rag.close()


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "ingest":
        asyncio.run(_run_ingest(args))
    elif args.command == "retrieve":
        asyncio.run(_run_retrieve(args))


if __name__ == "__main__":
    main()
