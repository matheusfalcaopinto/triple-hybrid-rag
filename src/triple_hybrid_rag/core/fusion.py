"""
RRF Fusion for Triple-Hybrid-RAG

Implements:
- Weighted Reciprocal Rank Fusion (RRF) for combining search channels
- Conformal denoising for filtering low-quality results
- Safety threshold filtering
"""

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from triple_hybrid_rag.config import RAGConfig, get_settings
from triple_hybrid_rag.types import QueryPlan, SearchChannel, SearchResult

logger = logging.getLogger(__name__)

# Default RRF constant (k=60 is standard)
RRF_K = 60

class RRFFusion:
    """
    Weighted Reciprocal Rank Fusion for combining multiple search channels.
    
    Features:
    - Configurable weights per channel (lexical, semantic, graph)
    - Conformal denoising to filter uncertain results
    - Safety threshold filtering
    """
    
    def __init__(self, config: Optional[RAGConfig] = None):
        """Initialize the RRF fusion module."""
        self.config = config or get_settings()
        
        # Default weights
        self.default_weights = {
            "lexical": self.config.rag_lexical_weight,
            "semantic": self.config.rag_semantic_weight,
            "graph": self.config.rag_graph_weight,
        }
        
        # Safety threshold
        self.safety_threshold = self.config.rag_safety_threshold
        
        # Conformal denoising
        self.denoise_enabled = self.config.rag_denoise_enabled
        self.denoise_alpha = self.config.rag_denoise_alpha
    
    def fuse(
        self,
        lexical_results: List[SearchResult],
        semantic_results: List[SearchResult],
        graph_results: List[SearchResult],
        query_plan: Optional[QueryPlan] = None,
        top_k: Optional[int] = None,
    ) -> List[SearchResult]:
        """
        Fuse results from all three channels using weighted RRF.
        
        Args:
            lexical_results: Results from lexical (BM25/FTS) search
            semantic_results: Results from semantic (vector) search
            graph_results: Results from graph (Cypher) search
            query_plan: Optional query plan with custom weights
            top_k: Maximum results to return
            
        Returns:
            Fused and ranked list of SearchResult objects
        """
        # Get weights (use query plan weights if available)
        weights = query_plan.weights if query_plan else self.default_weights
        
        # Calculate RRF scores for each channel
        lexical_scores = self._compute_rrf_scores(
            lexical_results, 
            weights.get("lexical", self.default_weights["lexical"])
        )
        semantic_scores = self._compute_rrf_scores(
            semantic_results,
            weights.get("semantic", self.default_weights["semantic"])
        )
        graph_scores = self._compute_rrf_scores(
            graph_results,
            weights.get("graph", self.default_weights["graph"])
        )
        
        # Merge scores by chunk_id
        merged_scores: Dict[str, Dict] = defaultdict(lambda: {
            "rrf_score": 0.0,
            "lexical_score": 0.0,
            "semantic_score": 0.0,
            "graph_score": 0.0,
            "result": None,
            "sources": set(),
        })
        
        # Add lexical results
        for result in lexical_results:
            key = str(result.chunk_id)
            merged_scores[key]["rrf_score"] += lexical_scores.get(key, 0.0)
            merged_scores[key]["lexical_score"] = result.lexical_score
            merged_scores[key]["sources"].add(SearchChannel.LEXICAL)
            if merged_scores[key]["result"] is None:
                merged_scores[key]["result"] = result
        
        # Add semantic results
        for result in semantic_results:
            key = str(result.chunk_id)
            merged_scores[key]["rrf_score"] += semantic_scores.get(key, 0.0)
            merged_scores[key]["semantic_score"] = result.semantic_score
            merged_scores[key]["sources"].add(SearchChannel.SEMANTIC)
            if merged_scores[key]["result"] is None:
                merged_scores[key]["result"] = result
        
        # Add graph results
        for result in graph_results:
            key = str(result.chunk_id)
            merged_scores[key]["rrf_score"] += graph_scores.get(key, 0.0)
            merged_scores[key]["graph_score"] = result.graph_score
            merged_scores[key]["sources"].add(SearchChannel.GRAPH)
            if merged_scores[key]["result"] is None:
                merged_scores[key]["result"] = result
        
        # Create fused results
        fused_results = []
        for key, data in merged_scores.items():
            result = data["result"]
            if result is None:
                continue
            
            # Update result with fused scores
            result.rrf_score = data["rrf_score"]
            result.lexical_score = data["lexical_score"]
            result.semantic_score = data["semantic_score"]
            result.graph_score = data["graph_score"]
            result.final_score = data["rrf_score"]  # Will be updated by reranker
            
            # Store source channels in metadata
            result.metadata["source_channels"] = [s.value for s in data["sources"]]
            
            fused_results.append(result)
        
        # Sort by RRF score
        fused_results.sort(key=lambda r: r.rrf_score, reverse=True)
        
        # Apply safety threshold filtering
        fused_results = self._apply_safety_threshold(fused_results)
        
        # Apply conformal denoising
        if self.denoise_enabled:
            fused_results = self._apply_conformal_denoising(fused_results)
        
        # Return top_k
        if top_k:
            fused_results = fused_results[:top_k]
        
        logger.debug(
            f"Fused {len(lexical_results)} lexical + {len(semantic_results)} semantic + "
            f"{len(graph_results)} graph → {len(fused_results)} results"
        )
        
        return fused_results
    
    def _compute_rrf_scores(
        self,
        results: List[SearchResult],
        weight: float,
    ) -> Dict[str, float]:
        """
        Compute weighted RRF scores for a list of results.
        
        RRF formula: score = weight * (1 / (k + rank))
        where k is a constant (default 60) and rank is 1-indexed.
        """
        scores = {}
        
        for rank, result in enumerate(results, start=1):
            key = str(result.chunk_id)
            rrf_score = weight * (1.0 / (RRF_K + rank))
            scores[key] = rrf_score
        
        return scores
    
    def _apply_safety_threshold(
        self,
        results: List[SearchResult],
    ) -> List[SearchResult]:
        """
        Filter results below the safety threshold.
        
        Uses the maximum of semantic and lexical scores for filtering.
        """
        if self.safety_threshold <= 0:
            return results
        
        filtered = []
        for result in results:
            # Use max of available scores
            max_score = max(
                result.semantic_score or 0.0,
                result.lexical_score or 0.0,
                result.graph_score or 0.0,
            )
            
            if max_score >= self.safety_threshold:
                filtered.append(result)
            else:
                logger.debug(
                    f"Filtered result {result.chunk_id} below safety threshold "
                    f"({max_score:.3f} < {self.safety_threshold})"
                )
        
        return filtered
    
    def _apply_conformal_denoising(
        self,
        results: List[SearchResult],
    ) -> List[SearchResult]:
        """
        Apply conformal denoising to filter uncertain results.
        
        Uses the score distribution to identify outliers/noise.
        Based on conformal prediction for set-valued prediction.
        """
        if len(results) < 3:
            return results
        
        # Get RRF scores
        scores = np.array([r.rrf_score for r in results])
        
        # Compute threshold using conformal prediction
        # alpha percentile of scores defines the acceptance threshold
        threshold = np.percentile(scores, (1 - self.denoise_alpha) * 100)
        
        # Filter results below threshold
        denoised = [r for r in results if r.rrf_score >= threshold]
        
        if len(denoised) < len(results):
            logger.debug(
                f"Conformal denoising removed {len(results) - len(denoised)} results "
                f"(threshold={threshold:.4f})"
            )
        
        return denoised
    
    def fuse_two_channels(
        self,
        results_a: List[SearchResult],
        results_b: List[SearchResult],
        weight_a: float = 1.0,
        weight_b: float = 1.0,
        top_k: Optional[int] = None,
    ) -> List[SearchResult]:
        """
        Fuse results from two channels (simpler variant).
        
        Useful when graph search is disabled.
        """
        # Compute RRF scores
        scores_a = self._compute_rrf_scores(results_a, weight_a)
        scores_b = self._compute_rrf_scores(results_b, weight_b)
        
        # Merge
        merged: Dict[str, Tuple[float, SearchResult]] = {}
        
        for result in results_a:
            key = str(result.chunk_id)
            merged[key] = (scores_a.get(key, 0.0), result)
        
        for result in results_b:
            key = str(result.chunk_id)
            if key in merged:
                existing_score, existing_result = merged[key]
                merged[key] = (existing_score + scores_b.get(key, 0.0), existing_result)
            else:
                merged[key] = (scores_b.get(key, 0.0), result)
        
        # Sort and return
        fused = sorted(merged.values(), key=lambda x: x[0], reverse=True)
        results = [r for _, r in fused]
        
        for score, result in fused:
            result.rrf_score = score
            result.final_score = score
        
        if top_k:
            results = results[:top_k]
        
        return results
    
    def normalize_scores(
        self,
        results: List[SearchResult],
        score_field: str = "final_score",
    ) -> List[SearchResult]:
        """
        Normalize scores to [0, 1] range using min-max normalization.
        """
        if not results:
            return results
        
        scores = [getattr(r, score_field, 0.0) for r in results]
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            for r in results:
                setattr(r, score_field, 1.0)
        else:
            for r in results:
                score = getattr(r, score_field, 0.0)
                normalized = (score - min_score) / (max_score - min_score)
                setattr(r, score_field, normalized)
        
        return results
