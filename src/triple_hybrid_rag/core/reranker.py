"""
Reranker for Triple-Hybrid-RAG

Uses a Qwen3-VL-Reranker endpoint to score (query, document) pairs.
Prefers the native /rerank endpoint and falls back gracefully if unavailable.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import httpx

from triple_hybrid_rag.config import RAGConfig, get_settings

logger = logging.getLogger(__name__)


class Reranker:
    """
    Reranker client using a Qwen3-VL-Reranker OpenAI-compatible API.

    Expected /rerank payload:
    {
        "model": "qwen3-vl-reranker-2b",
        "query": "...",
        "documents": ["doc1", "doc2", ...]
    }
    """

    def __init__(self, config: Optional[RAGConfig] = None) -> None:
        self.config = config or get_settings()
        self.api_base = self.config.rag_rerank_api_base.rstrip("/")
        self.model = self.config.rag_rerank_model
        self.timeout = self.config.rag_rerank_timeout
        self.enabled = self.config.rag_rerank_enabled
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def rerank(self, query: str, documents: List[str]) -> List[float]:
        """Return relevance scores aligned to the documents list."""
        if not self.enabled or not documents:
            return [0.0 for _ in documents]

        try:
            client = await self._get_client()
            response = await client.post(
                f"{self.api_base}/rerank",
                json={
                    "model": self.model,
                    "query": query,
                    "documents": documents,
                },
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            payload = response.json()
            scores = _parse_rerank_scores(payload, len(documents))
            if scores:
                return scores
            logger.warning("Rerank response missing scores; falling back to zeros")
        except httpx.HTTPError as exc:
            logger.error(f"Rerank API error: {exc}")
        except Exception as exc:  # noqa: BLE001 - defensive guard
            logger.error(f"Unexpected rerank error: {exc}")

        return [0.0 for _ in documents]


def _parse_rerank_scores(payload: dict, expected_len: int) -> List[float]:
    """Parse scores from common rerank response formats."""
    if not isinstance(payload, dict):
        return []

    if "results" in payload:
        results = payload.get("results") or []
        scores = [0.0] * expected_len
        for item in results:
            try:
                idx = int(item.get("index"))
                score = float(item.get("relevance_score", item.get("score", 0.0)))
            except (TypeError, ValueError):
                continue
            if 0 <= idx < expected_len:
                scores[idx] = score
        return scores

    if "data" in payload:
        results = payload.get("data") or []
        scores = [0.0] * expected_len
        for i, item in enumerate(results):
            try:
                score = float(item.get("score", item.get("relevance_score", 0.0)))
            except (TypeError, ValueError):
                score = 0.0
            if i < expected_len:
                scores[i] = score
        return scores

    return []
