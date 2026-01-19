"""
Multi-Stage Reranking Pipeline for Triple-Hybrid-RAG

Implements a progressive reranking pipeline:
Stage 1: Fast bi-encoder filtering (top-100 → top-30)
Stage 2: Cross-encoder scoring with fine-grained relevance
Stage 3: Diversity optimization (MMR)
Stage 4: Relevance calibration

Reference:
- Nogueira et al. "Document Ranking with a Pretrained Sequence-to-Sequence Model"
- Pradeep et al. "The Expando-Mono-Duo Design Pattern for Text Ranking"
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import httpx
import numpy as np

from triple_hybrid_rag.config import RAGConfig, get_settings
from triple_hybrid_rag.types import SearchResult

logger = logging.getLogger(__name__)

@dataclass
class RerankerConfig:
    """Configuration for multi-stage reranking."""
    
    enabled: bool = True
    
    # Stage 1: Bi-encoder filtering
    stage1_enabled: bool = True
    stage1_top_k: int = 100
    
    # Stage 2: Cross-encoder scoring
    stage2_enabled: bool = True
    stage2_top_k: int = 30
    stage2_model: str = "gpt-4o-mini"
    stage2_batch_size: int = 10
    
    # Stage 3: Diversity (MMR)
    stage3_enabled: bool = True
    mmr_lambda: float = 0.7  # Balance relevance vs diversity
    
    # Stage 4: Calibration
    stage4_enabled: bool = True
    
    # General settings
    timeout: float = 60.0
    max_concurrent: int = 4

@dataclass
class RerankedResult:
    """Result of multi-stage reranking."""
    
    results: List[SearchResult] = field(default_factory=list)
    stage1_count: int = 0
    stage2_count: int = 0
    stage3_count: int = 0
    rerank_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

class MultiStageReranker:
    """
    Multi-stage reranking pipeline for improved retrieval quality.
    
    Pipeline:
    1. Bi-encoder filtering: Fast initial filtering
    2. Cross-encoder scoring: LLM-based relevance scoring
    3. MMR diversity: Maximal Marginal Relevance
    4. Calibration: Score normalization
    
    Usage:
        reranker = MultiStageReranker()
        reranked = await reranker.rerank(query, results)
    """
    
    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        reranker_config: Optional[RerankerConfig] = None,
    ):
        """Initialize the multi-stage reranker."""
        self.config = config or get_settings()
        self.reranker_config = reranker_config or RerankerConfig()
        
        self.api_key = self.config.openai_api_key
        self.api_base = self.config.openai_base_url.rstrip("/")
        
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.reranker_config.timeout)
        return self._client
    
    async def close(self):
        """Close HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
    
    async def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: Optional[int] = None,
    ) -> RerankedResult:
        """
        Rerank results through multi-stage pipeline.
        
        Args:
            query: User query
            results: Initial retrieval results
            top_k: Final number of results to return
            
        Returns:
            RerankedResult with reranked results
        """
        import time
        start_time = time.perf_counter()
        
        if not self.reranker_config.enabled or not results:
            return RerankedResult(results=results)
        
        current_results = results.copy()
        output = RerankedResult()
        
        # Stage 1: Bi-encoder filtering
        if self.reranker_config.stage1_enabled:
            current_results = self._stage1_filter(current_results)
            output.stage1_count = len(current_results)
        
        # Stage 2: Cross-encoder scoring
        if self.reranker_config.stage2_enabled and len(current_results) > 0:
            current_results = await self._stage2_crossencoder(query, current_results)
            output.stage2_count = len(current_results)
        
        # Stage 3: MMR diversity
        if self.reranker_config.stage3_enabled and len(current_results) > 1:
            current_results = self._stage3_mmr(query, current_results)
            output.stage3_count = len(current_results)
        
        # Stage 4: Calibration
        if self.reranker_config.stage4_enabled:
            current_results = self._stage4_calibrate(current_results)
        
        # Apply final top_k
        if top_k is not None:
            current_results = current_results[:top_k]
        
        output.results = current_results
        output.rerank_time_ms = (time.perf_counter() - start_time) * 1000
        
        logger.debug(
            f"Multi-stage rerank: {len(results)} → {len(current_results)}, "
            f"time={output.rerank_time_ms:.1f}ms"
        )
        
        return output
    
    def _stage1_filter(self, results: List[SearchResult]) -> List[SearchResult]:
        """Stage 1: Filter by existing scores (bi-encoder proxy)."""
        top_k = self.reranker_config.stage1_top_k
        
        # Sort by combined score (RRF or semantic)
        sorted_results = sorted(
            results,
            key=lambda r: r.rrf_score or r.semantic_score or 0.0,
            reverse=True,
        )
        
        return sorted_results[:top_k]
    
    async def _stage2_crossencoder(
        self,
        query: str,
        results: List[SearchResult],
    ) -> List[SearchResult]:
        """Stage 2: Cross-encoder scoring using LLM."""
        top_k = self.reranker_config.stage2_top_k
        batch_size = self.reranker_config.stage2_batch_size
        
        # Score in batches
        semaphore = asyncio.Semaphore(self.reranker_config.max_concurrent)
        
        async def score_batch(batch: List[SearchResult]) -> List[float]:
            async with semaphore:
                return await self._score_batch_llm(query, batch)
        
        # Create batches
        batches = [
            results[i:i + batch_size]
            for i in range(0, len(results), batch_size)
        ]
        
        # Score concurrently
        batch_scores = await asyncio.gather(
            *[score_batch(batch) for batch in batches],
            return_exceptions=True,
        )
        
        # Flatten scores and assign to results
        all_scores = []
        for i, scores in enumerate(batch_scores):
            if isinstance(scores, Exception):
                logger.warning(f"Batch {i} scoring failed: {scores}")
                all_scores.extend([0.0] * len(batches[i]))
            else:
                all_scores.extend(scores)
        
        # Update results with cross-encoder scores
        for result, score in zip(results, all_scores):
            result.rerank_score = score
        
        # Sort by cross-encoder score
        sorted_results = sorted(results, key=lambda r: r.rerank_score or 0.0, reverse=True)
        
        return sorted_results[:top_k]
    
    async def _score_batch_llm(
        self,
        query: str,
        results: List[SearchResult],
    ) -> List[float]:
        """Score a batch of results using LLM."""
        client = await self._get_client()
        
        # Format passages for scoring
        passages = []
        for i, result in enumerate(results):
            text = result.text[:500] if result.text else ""
            passages.append(f"[{i+1}] {text}")
        
        prompt = f"""Rate the relevance of each passage to the query on a scale of 0-10.
Query: {query}

Passages:
{chr(10).join(passages)}

Return ONLY a comma-separated list of scores (e.g., "8,5,9,3"):"""

        try:
            response = await client.post(
                f"{self.api_base}/chat/completions",
                json={
                    "model": self.reranker_config.stage2_model,
                    "messages": [
                        {"role": "system", "content": "You are a relevance scoring system."},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.0,
                    "max_tokens": 100,
                },
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
            response.raise_for_status()
            
            data = response.json()
            content = data["choices"][0]["message"]["content"].strip()
            
            # Parse scores
            scores = []
            for s in content.replace(" ", "").split(","):
                try:
                    scores.append(float(s) / 10.0)  # Normalize to 0-1
                except ValueError:
                    scores.append(0.5)  # Default score
            
            # Pad if needed
            while len(scores) < len(results):
                scores.append(0.5)
            
            return scores[:len(results)]
            
        except Exception as e:
            logger.error(f"LLM scoring failed: {e}")
            return [0.5] * len(results)
    
    def _stage3_mmr(
        self,
        query: str,
        results: List[SearchResult],
    ) -> List[SearchResult]:
        """Stage 3: Maximal Marginal Relevance for diversity."""
        lambda_param = self.reranker_config.mmr_lambda
        
        if len(results) <= 1:
            return results
        
        # Build similarity matrix (using text similarity as proxy)
        def text_similarity(a: str, b: str) -> float:
            """Simple Jaccard-like similarity."""
            if not a or not b:
                return 0.0
            words_a = set(a.lower().split())
            words_b = set(b.lower().split())
            if not words_a or not words_b:
                return 0.0
            intersection = len(words_a & words_b)
            union = len(words_a | words_b)
            return intersection / union if union > 0 else 0.0
        
        selected: List[SearchResult] = []
        remaining = results.copy()
        
        # Select first by relevance
        remaining.sort(key=lambda r: r.rerank_score or r.final_score or 0.0, reverse=True)
        selected.append(remaining.pop(0))
        
        # Iteratively select by MMR
        while remaining:
            best_mmr = float("-inf")
            best_idx = 0
            
            for i, candidate in enumerate(remaining):
                # Relevance score
                relevance = candidate.rerank_score or candidate.final_score or 0.0
                
                # Max similarity to selected
                max_sim = 0.0
                for sel in selected:
                    sim = text_similarity(candidate.text or "", sel.text or "")
                    max_sim = max(max_sim, sim)
                
                # MMR score
                mmr = lambda_param * relevance - (1 - lambda_param) * max_sim
                
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = i
            
            selected.append(remaining.pop(best_idx))
        
        return selected
    
    def _stage4_calibrate(self, results: List[SearchResult]) -> List[SearchResult]:
        """Stage 4: Score calibration and normalization."""
        if not results:
            return results
        
        # Get all scores
        scores = [r.rerank_score or r.final_score or 0.0 for r in results]
        
        # Min-max normalization
        min_score = min(scores) if scores else 0.0
        max_score = max(scores) if scores else 1.0
        score_range = max_score - min_score
        
        for result in results:
            raw_score = result.rerank_score or result.final_score or 0.0
            if score_range > 0:
                calibrated = (raw_score - min_score) / score_range
            else:
                calibrated = raw_score
            result.final_score = calibrated
        
        return results

class ListwiseReranker:
    """
    Listwise LLM reranker using RankGPT-style prompting.
    
    Instead of scoring documents independently, considers
    the entire list and asks the LLM to rank them.
    """
    
    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        model: str = "gpt-4o-mini",
    ):
        """Initialize listwise reranker."""
        self.config = config or get_settings()
        self.model = model
        self.api_key = self.config.openai_api_key
        self.api_base = self.config.openai_base_url.rstrip("/")
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=60.0)
        return self._client
    
    async def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int = 10,
    ) -> List[SearchResult]:
        """Rerank using listwise LLM prompting."""
        if len(results) <= 1:
            return results
        
        # Limit to reasonable size for LLM
        candidates = results[:20]
        
        # Build passages
        passages = []
        for i, r in enumerate(candidates):
            text = (r.text or "")[:300]
            passages.append(f"[{i+1}] {text}")
        
        prompt = f"""Rank the following passages by relevance to the query.
Return ONLY a comma-separated list of passage numbers in order of relevance (most relevant first).

Query: {query}

Passages:
{chr(10).join(passages)}

Ranking (e.g., "3,1,5,2,4"):"""

        try:
            client = await self._get_client()
            response = await client.post(
                f"{self.api_base}/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "You are a document ranking system."},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.0,
                    "max_tokens": 100,
                },
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
            response.raise_for_status()
            
            data = response.json()
            content = data["choices"][0]["message"]["content"].strip()
            
            # Parse ranking
            ranking = []
            for s in content.replace(" ", "").split(","):
                try:
                    idx = int(s) - 1  # Convert to 0-indexed
                    if 0 <= idx < len(candidates):
                        ranking.append(idx)
                except ValueError:
                    continue
            
            # Build reranked list
            reranked = []
            seen = set()
            for idx in ranking:
                if idx not in seen:
                    seen.add(idx)
                    result = candidates[idx]
                    result.final_score = 1.0 - len(reranked) * 0.05  # Descending scores
                    reranked.append(result)
            
            # Add any missing
            for i, r in enumerate(candidates):
                if i not in seen:
                    r.final_score = 0.1
                    reranked.append(r)
            
            return reranked[:top_k]
            
        except Exception as e:
            logger.error(f"Listwise reranking failed: {e}")
            return results[:top_k]
    
    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()
