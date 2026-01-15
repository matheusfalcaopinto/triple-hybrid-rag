"""
RAG 2.0 Retrieval Pipeline

Triple-hybrid retrieval with:
- Lexical search (FTS/BM25)
- Semantic search (vector/HNSW)
- Graph search (PuppyGraph/Cypher) - optional
- Weighted RRF fusion
- Child → Parent expansion
- Reranking
- Safety threshold + denoising
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from voice_agent.config import SETTINGS
from voice_agent.rag2.embedder import RAG2Embedder, get_rag2_embedder
from voice_agent.rag2.query_planner import QueryPlan, QueryPlanner, get_query_planner
from voice_agent.utils.db import get_supabase_client

logger = logging.getLogger(__name__)


@dataclass
class RetrievalCandidate:
    """A candidate from retrieval."""
    child_id: str
    parent_id: str
    document_id: str
    text: str
    page: int
    modality: str
    
    # Scores from different channels
    lexical_rank: Optional[int] = None
    semantic_rank: Optional[int] = None
    graph_rank: Optional[int] = None
    rrf_score: float = 0.0
    
    # After expansion
    parent_text: Optional[str] = None
    section_heading: Optional[str] = None
    
    # After reranking
    rerank_score: Optional[float] = None


@dataclass 
class RetrievalResult:
    """Result of the retrieval pipeline."""
    success: bool
    contexts: List[RetrievalCandidate]
    
    # Safety
    max_rerank_score: float = 0.0
    refused: bool = False
    refusal_reason: Optional[str] = None
    
    # Debug/trace
    query_plan: Optional[QueryPlan] = None
    timings: Dict[str, float] = field(default_factory=dict)


class RAG2Retriever:
    """
    RAG 2.0 Retrieval Pipeline.
    
    Implements the recall-to-precision funnel:
    1. Query planning (GPT-5)
    2. Multi-channel retrieval (lexical + semantic + graph)
    3. Weighted RRF fusion
    4. Child → Parent expansion
    5. Reranking
    6. Safety threshold + denoising
    """
    
    def __init__(
        self,
        org_id: str,
        embedder: Optional[RAG2Embedder] = None,
        query_planner: Optional[QueryPlanner] = None,
        graph_enabled: bool = False,
    ):
        """
        Initialize the retriever.
        
        Args:
            org_id: Organization ID for data isolation
            embedder: Embedder for query vectorization
            query_planner: Query planner agent
            graph_enabled: Whether to use graph channel
        """
        self.org_id = org_id
        self.embedder = embedder or get_rag2_embedder()
        self.query_planner = query_planner or get_query_planner()
        self.graph_enabled = graph_enabled and SETTINGS.rag2_graph_enabled
        
        self._supabase = None
        self._reranker = None
    
    @property
    def supabase(self) -> Any:
        """Lazy-load Supabase client."""
        if self._supabase is None:
            self._supabase = get_supabase_client()
        return self._supabase
    
    @property
    def reranker(self) -> Any:
        """Lazy-load reranker."""
        if self._reranker is None:
            from voice_agent.retrieval.reranker import Reranker
            self._reranker = Reranker()
        return self._reranker
    
    async def retrieve(
        self,
        query: str,
        collection: Optional[str] = None,
        top_k: Optional[int] = None,
        skip_planning: bool = False,
        skip_rerank: bool = False,
    ) -> RetrievalResult:
        """
        Execute the full retrieval pipeline.
        
        Args:
            query: User query
            collection: Optional collection filter
            top_k: Final number of results (default from config)
            skip_planning: Skip query planning (use query as-is)
            skip_rerank: Skip reranking step
            
        Returns:
            RetrievalResult with contexts or refusal
        """
        import time
        timings = {}
        top_k = top_k or SETTINGS.rag2_final_top_k
        
        # Step 1: Query Planning
        t0 = time.time()
        if skip_planning:
            plan = QueryPlan(
                original_query=query,
                keywords=query.split(),
                semantic_query_text=query,
            )
        else:
            plan = await self.query_planner.plan_async(query, collection)
        timings["planning"] = time.time() - t0
        
        # Step 2: Multi-channel retrieval
        t0 = time.time()
        candidates = await self._retrieve_candidates(plan, collection)
        timings["retrieval"] = time.time() - t0
        
        if not candidates:
            return RetrievalResult(
                success=True,
                contexts=[],
                refused=True,
                refusal_reason="No candidates found",
                query_plan=plan,
                timings=timings,
            )
        
        # Step 3: Weighted RRF Fusion
        t0 = time.time()
        fused = self._fuse_rrf(candidates, plan.weights)
        timings["fusion"] = time.time() - t0
        
        # Step 4: Child → Parent expansion
        t0 = time.time()
        expanded = await self._expand_to_parents(fused[:SETTINGS.rag2_rerank_top_k])
        timings["expansion"] = time.time() - t0
        
        # Step 5: Reranking
        if not skip_rerank and SETTINGS.rag2_rerank_enabled:
            t0 = time.time()
            reranked = await self._rerank(query, expanded)
            timings["rerank"] = time.time() - t0
        else:
            reranked = expanded
        
        # Step 6: Safety threshold + denoising
        t0 = time.time()
        final, refused, reason, max_score = self._apply_safety(reranked, top_k)
        timings["safety"] = time.time() - t0
        
        return RetrievalResult(
            success=True,
            contexts=final,
            max_rerank_score=max_score,
            refused=refused,
            refusal_reason=reason,
            query_plan=plan,
            timings=timings,
        )
    
    async def _retrieve_candidates(
        self,
        plan: QueryPlan,
        collection: Optional[str],
    ) -> List[RetrievalCandidate]:
        """Execute multi-channel retrieval."""
        candidates_map: Dict[str, RetrievalCandidate] = {}
        
        # Lexical channel
        if plan.keywords:
            lexical_results = await self._lexical_search(
                keywords=plan.keywords,
                collection=collection,
                limit=plan.lexical_top_k,
            )
            for rank, r in enumerate(lexical_results, 1):
                cid = r["child_id"]
                if cid not in candidates_map:
                    candidates_map[cid] = RetrievalCandidate(
                        child_id=cid,
                        parent_id=r["parent_id"],
                        document_id=r["document_id"],
                        text=r["text"],
                        page=r.get("page", 1),
                        modality=r.get("modality", "text"),
                    )
                candidates_map[cid].lexical_rank = rank
        
        # Semantic channel
        semantic_results = await self._semantic_search(
            query_text=plan.semantic_query_text,
            collection=collection,
            limit=plan.semantic_top_k,
        )
        for rank, r in enumerate(semantic_results, 1):
            cid = r["child_id"]
            if cid not in candidates_map:
                candidates_map[cid] = RetrievalCandidate(
                    child_id=cid,
                    parent_id=r["parent_id"],
                    document_id=r["document_id"],
                    text=r["text"],
                    page=r.get("page", 1),
                    modality=r.get("modality", "text"),
                )
            candidates_map[cid].semantic_rank = rank
        
        # Graph channel (if enabled and query needs it)
        if self.graph_enabled and plan.requires_graph and plan.cypher_query:
            graph_results = await self._graph_search(
                cypher=plan.cypher_query,
                keywords=plan.keywords,
                collection=collection,
                limit=plan.graph_top_k,
            )
            for rank, r in enumerate(graph_results, 1):
                cid = r["child_id"]
                if cid not in candidates_map:
                    candidates_map[cid] = RetrievalCandidate(
                        child_id=cid,
                        parent_id=r["parent_id"],
                        document_id=r["document_id"],
                        text=r["text"],
                        page=r.get("page", 1),
                        modality=r.get("modality", "text"),
                    )
                candidates_map[cid].graph_rank = rank
        
        return list(candidates_map.values())
    
    async def _lexical_search(
        self,
        keywords: List[str],
        collection: Optional[str],
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Execute lexical (FTS) search."""
        query_str = " ".join(keywords)
        
        result = self.supabase.rpc(
            "rag2_lexical_search",
            {
                "p_org_id": self.org_id,
                "p_query": query_str,
                "p_limit": limit,
                "p_collection": collection,
            }
        ).execute()
        
        return result.data or []
    
    async def _semantic_search(
        self,
        query_text: str,
        collection: Optional[str],
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Execute semantic (vector) search."""
        # Embed query
        query_embedding = self.embedder.embed_query(query_text)
        
        result = self.supabase.rpc(
            "rag2_semantic_search",
            {
                "p_org_id": self.org_id,
                "p_embedding": query_embedding,
                "p_limit": limit,
                "p_collection": collection,
            }
        ).execute()
        
        return result.data or []
    
    async def _graph_search(
        self,
        cypher: str,
        keywords: List[str],
        collection: Optional[str],
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Execute graph search via PuppyGraph or SQL fallback."""
        from voice_agent.rag2.graph_search import get_graph_searcher
        
        try:
            graph_searcher = get_graph_searcher(self.supabase)
            result = await graph_searcher.search(
                keywords=keywords,
                cypher_query=cypher,
                org_id=self.org_id,
                top_k=limit,
            )
            
            if not result.chunk_ids:
                return []
            
            # Fetch chunk details for the returned chunk IDs
            chunk_result = self.supabase.table("rag_child_chunks").select(
                "id, parent_id, document_id, text, page, modality"
            ).in_("id", result.chunk_ids[:limit]).execute()
            
            return [
                {
                    "child_id": r["id"],
                    "parent_id": r["parent_id"],
                    "document_id": r["document_id"],
                    "text": r["text"],
                    "page": r.get("page", 1),
                    "modality": r.get("modality", "text"),
                }
                for r in chunk_result.data
            ]
        except Exception as e:
            logger.warning(f"Graph search failed: {e}")
            return []
    
    def _fuse_rrf(
        self,
        candidates: List[RetrievalCandidate],
        weights: Dict[str, float],
        k: int = 60,
    ) -> List[RetrievalCandidate]:
        """Apply Weighted Reciprocal Rank Fusion."""
        for c in candidates:
            score = 0.0
            if c.lexical_rank:
                score += weights.get("lexical", 0.7) / (k + c.lexical_rank)
            if c.semantic_rank:
                score += weights.get("semantic", 0.8) / (k + c.semantic_rank)
            if c.graph_rank:
                score += weights.get("graph", 1.0) / (k + c.graph_rank)
            c.rrf_score = score
        
        # Sort by RRF score descending
        return sorted(candidates, key=lambda x: x.rrf_score, reverse=True)
    
    async def _expand_to_parents(
        self,
        candidates: List[RetrievalCandidate],
    ) -> List[RetrievalCandidate]:
        """Expand child chunks to parent chunks for context."""
        if not candidates:
            return []
        
        # Get unique parent IDs
        parent_ids = list(set(c.parent_id for c in candidates))
        
        # Fetch parent texts
        result = self.supabase.table("rag_parent_chunks").select(
            "id, text, section_heading"
        ).in_("id", parent_ids).execute()
        
        parent_map = {r["id"]: r for r in result.data}
        
        # Attach parent info to candidates
        for c in candidates:
            if c.parent_id in parent_map:
                p = parent_map[c.parent_id]
                c.parent_text = p["text"]
                c.section_heading = p.get("section_heading")
        
        return candidates
    
    async def _rerank(
        self,
        query: str,
        candidates: List[RetrievalCandidate],
    ) -> List[RetrievalCandidate]:
        """Rerank candidates using cross-encoder.
        
        Tries native /rerank endpoint first (vLLM with --runner pooling),
        falls back to chat-based scoring if not available.
        """
        if not candidates:
            return []
        
        # Use parent text for reranking (more context)
        texts = [c.parent_text or c.text for c in candidates]
        
        try:
            from voice_agent.retrieval.reranker import Qwen3VLReranker
            reranker = Qwen3VLReranker()
            
            # Try native /rerank endpoint first (batch scoring)
            try:
                scores = await reranker._rerank_batch_native(query, texts)
                for idx, score in enumerate(scores):
                    candidates[idx].rerank_score = score
                logger.debug(f"Used native /rerank endpoint, scores: {scores[:3]}...")
            except Exception as native_err:
                # Fall back to chat-based scoring
                logger.debug(f"Native rerank unavailable ({native_err}), using chat fallback")
                
                import asyncio
                semaphore = asyncio.Semaphore(5)
                
                async def score_candidate(idx: int, text: str) -> tuple:
                    async with semaphore:
                        score = await reranker._score_pair(query, text, None)
                        return idx, score
                
                tasks = [score_candidate(i, text) for i, text in enumerate(texts)]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in results:
                    if isinstance(result, BaseException):
                        logger.warning(f"Rerank error: {result}")
                        continue
                    if isinstance(result, tuple) and len(result) == 2:
                        idx, score = result
                        candidates[idx].rerank_score = score
            
            # Sort by rerank score
            return sorted(candidates, key=lambda x: x.rerank_score or 0, reverse=True)
            
        except Exception as e:
            logger.warning(f"Reranking failed: {e}")
            return candidates
    
    def _apply_safety(
        self,
        candidates: List[RetrievalCandidate],
        top_k: int,
    ) -> Tuple[List[RetrievalCandidate], bool, Optional[str], float]:
        """
        Apply safety threshold and denoising.
        
        Returns:
            (final_candidates, refused, reason, max_score)
        """
        if not candidates:
            return [], True, "No candidates after reranking", 0.0
        
        # Get max score
        max_score = max(c.rerank_score or c.rrf_score for c in candidates)
        
        # Check safety threshold
        threshold = SETTINGS.rag2_safety_threshold
        if max_score < threshold:
            return [], True, f"Max score {max_score:.2f} below threshold {threshold}", max_score
        
        # Denoise: keep only chunks above alpha * max_score
        alpha = SETTINGS.rag2_denoise_alpha
        min_score = alpha * max_score
        
        filtered = [
            c for c in candidates
            if (c.rerank_score or c.rrf_score) >= min_score
        ]
        
        # Take top_k
        final = filtered[:top_k]
        
        return final, False, None, max_score


async def retrieve(
    org_id: str,
    query: str,
    **kwargs: Any,
) -> RetrievalResult:
    """Convenience function for retrieval."""
    retriever = RAG2Retriever(org_id=org_id)
    return await retriever.retrieve(query, **kwargs)
