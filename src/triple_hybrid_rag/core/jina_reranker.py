"""
Jina AI Reranker for Triple-Hybrid-RAG

Uses Jina AI jina-reranker-v3 for high-quality reranking:
- Listwise reranking for SOTA multilingual retrieval
- Support for query-document pair scoring
"""

import logging
from typing import List, Optional

import httpx

from triple_hybrid_rag.config import RAGConfig, get_settings

logger = logging.getLogger(__name__)


class JinaReranker:
    """
    Jina AI Reranker using jina-reranker-v3.
    
    Provides high-quality reranking for search results:
    - Listwise reranking (considers all documents together)
    - Multilingual support
    - Returns relevance scores for (query, document) pairs
    """
    
    JINA_API_BASE = "https://api.jina.ai/v1"
    
    def __init__(self, config: Optional[RAGConfig] = None) -> None:
        """Initialize the Jina reranker with configuration."""
        self.config = config or get_settings()
        
        self.api_key = self.config.jina_api_key
        self.api_base = self.config.jina_api_base.rstrip("/")
        self.model = self.config.jina_rerank_model
        self.top_n = self.config.jina_rerank_top_n
        self.timeout = self.config.jina_rerank_timeout
        self.enabled = self.config.rag_rerank_enabled
        
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
            )
        return self._client
    
    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
    
    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
    ) -> List[float]:
        """
        Rerank documents using Jina reranker.
        
        Args:
            query: The search query
            documents: List of document texts to rerank
            top_n: Optional limit on number of results
            
        Returns:
            List of relevance scores aligned to the documents list
        """
        if not self.enabled or not documents:
            return [0.0 for _ in documents]
        
        if not self.api_key:
            logger.error("JINA_API_KEY not configured")
            return [0.0 for _ in documents]
        
        top_n = top_n or self.top_n
        
        try:
            client = await self._get_client()
            
            response = await client.post(
                f"{self.api_base}/rerank",
                json={
                    "model": self.model,
                    "query": query,
                    "documents": documents,
                    "top_n": min(top_n, len(documents)),
                },
            )
            response.raise_for_status()
            
            payload = response.json()
            scores = self._parse_scores(payload, len(documents))
            
            if scores:
                return scores
            
            logger.warning("Jina rerank response missing scores; falling back to zeros")
            
        except httpx.HTTPError as exc:
            logger.error(f"Jina Rerank API error: {exc}")
        except Exception as exc:
            logger.error(f"Unexpected Jina rerank error: {exc}")
        
        return [0.0 for _ in documents]
    
    async def rerank_with_indices(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
    ) -> List[tuple[int, float]]:
        """
        Rerank documents and return (index, score) pairs sorted by relevance.
        
        Args:
            query: The search query
            documents: List of document texts to rerank
            top_n: Optional limit on number of results
            
        Returns:
            List of (document_index, score) tuples sorted by score descending
        """
        if not self.enabled or not documents:
            return [(i, 0.0) for i in range(len(documents))]
        
        if not self.api_key:
            logger.error("JINA_API_KEY not configured")
            return [(i, 0.0) for i in range(len(documents))]
        
        top_n = top_n or self.top_n
        
        try:
            client = await self._get_client()
            
            response = await client.post(
                f"{self.api_base}/rerank",
                json={
                    "model": self.model,
                    "query": query,
                    "documents": documents,
                    "top_n": min(top_n, len(documents)),
                },
            )
            response.raise_for_status()
            
            payload = response.json()
            
            # Parse results with indices
            results = payload.get("results", [])
            index_scores = []
            
            for item in results:
                try:
                    idx = int(item.get("index", 0))
                    score = float(item.get("relevance_score", 0.0))
                    index_scores.append((idx, score))
                except (TypeError, ValueError):
                    continue
            
            # Sort by score descending
            index_scores.sort(key=lambda x: x[1], reverse=True)
            
            return index_scores
            
        except httpx.HTTPError as exc:
            logger.error(f"Jina Rerank API error: {exc}")
        except Exception as exc:
            logger.error(f"Unexpected Jina rerank error: {exc}")
        
        return [(i, 0.0) for i in range(len(documents))]
    
    def _parse_scores(self, payload: dict, expected_len: int) -> List[float]:
        """Parse scores from Jina rerank response."""
        if not isinstance(payload, dict):
            return []
        
        results = payload.get("results", [])
        if not results:
            return []
        
        # Initialize scores array
        scores = [0.0] * expected_len
        
        for item in results:
            try:
                idx = int(item.get("index", 0))
                score = float(item.get("relevance_score", 0.0))
                if 0 <= idx < expected_len:
                    scores[idx] = score
            except (TypeError, ValueError):
                continue
        
        return scores


def _parse_rerank_scores(payload: dict, expected_len: int) -> List[float]:
    """
    Parse scores from Jina rerank response formats.
    
    Provided for backwards compatibility with existing code.
    """
    if not isinstance(payload, dict):
        return []
    
    results = payload.get("results", [])
    if not results:
        return []
    
    scores = [0.0] * expected_len
    
    for item in results:
        try:
            idx = int(item.get("index", 0))
            score = float(item.get("relevance_score", 0.0))
            if 0 <= idx < expected_len:
                scores[idx] = score
        except (TypeError, ValueError):
            continue
    
    return scores
