"""
Diversity Optimization for Triple-Hybrid-RAG

Implements algorithms to improve result diversity:
1. Maximal Marginal Relevance (MMR)
2. Determinantal Point Process (DPP) sampling
3. Intent-based diversification

Reference:
- Carbonell & Goldstein "The use of MMR, diversity-based reranking"
- Kulesza & Taskar "Determinantal Point Processes for Machine Learning"
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from triple_hybrid_rag.types import SearchResult

logger = logging.getLogger(__name__)

@dataclass
class DiversityConfig:
    """Configuration for diversity optimization."""
    
    enabled: bool = True
    
    # MMR settings
    mmr_lambda: float = 0.7  # 0=max diversity, 1=max relevance
    
    # DPP settings (if embeddings available)
    dpp_enabled: bool = False
    dpp_temperature: float = 1.0
    
    # Intent diversity
    intent_diversity_enabled: bool = True
    max_per_document: int = 3  # Max results per source document
    max_per_page: int = 2  # Max results per page

@dataclass
class DiversityResult:
    """Result of diversity optimization."""
    
    results: List[SearchResult] = field(default_factory=list)
    removed_count: int = 0
    diversity_score: float = 0.0
    optimization_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

class DiversityOptimizer:
    """
    Optimize result diversity using multiple algorithms.
    
    Ensures search results are diverse in:
    - Content (different topics/aspects)
    - Source (different documents)
    - Position (different pages within documents)
    
    Usage:
        optimizer = DiversityOptimizer()
        diverse = optimizer.optimize(results, top_k=10)
    """
    
    def __init__(self, config: Optional[DiversityConfig] = None):
        """Initialize diversity optimizer."""
        self.config = config or DiversityConfig()
    
    def optimize(
        self,
        results: List[SearchResult],
        top_k: int = 10,
        embeddings: Optional[List[List[float]]] = None,
    ) -> DiversityResult:
        """
        Apply diversity optimization to results.
        
        Args:
            results: Ranked search results
            top_k: Number of results to return
            embeddings: Optional embeddings for DPP (if not using text similarity)
            
        Returns:
            DiversityResult with diversified results
        """
        import time
        start_time = time.perf_counter()
        
        if not self.config.enabled or len(results) <= 1:
            return DiversityResult(results=results[:top_k])
        
        current_results = results.copy()
        
        # Apply source diversity first
        if self.config.intent_diversity_enabled:
            current_results = self._apply_source_diversity(current_results)
        
        # Apply MMR or DPP
        if self.config.dpp_enabled and embeddings is not None:
            current_results = self._apply_dpp(current_results, embeddings, top_k)
        else:
            current_results = self._apply_mmr(current_results, top_k)
        
        # Calculate diversity score
        diversity_score = self._calculate_diversity_score(current_results)
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        return DiversityResult(
            results=current_results[:top_k],
            removed_count=len(results) - len(current_results),
            diversity_score=diversity_score,
            optimization_time_ms=elapsed_ms,
        )
    
    def _apply_source_diversity(
        self,
        results: List[SearchResult],
    ) -> List[SearchResult]:
        """Apply document and page diversity constraints."""
        max_per_doc = self.config.max_per_document
        max_per_page = self.config.max_per_page
        
        # Track counts per document and page
        doc_counts: Dict[str, int] = {}
        page_counts: Dict[str, int] = {}
        
        diverse_results = []
        
        for result in results:
            doc_id = str(result.document_id)
            page_key = f"{doc_id}:{result.page}"
            
            # Check document limit
            doc_count = doc_counts.get(doc_id, 0)
            if doc_count >= max_per_doc:
                continue
            
            # Check page limit
            page_count = page_counts.get(page_key, 0)
            if page_count >= max_per_page:
                continue
            
            # Accept result
            diverse_results.append(result)
            doc_counts[doc_id] = doc_count + 1
            page_counts[page_key] = page_count + 1
        
        return diverse_results
    
    def _apply_mmr(
        self,
        results: List[SearchResult],
        top_k: int,
    ) -> List[SearchResult]:
        """Apply Maximal Marginal Relevance for content diversity."""
        if len(results) <= top_k:
            return results
        
        lambda_param = self.config.mmr_lambda
        
        def text_similarity(a: str, b: str) -> float:
            """Jaccard-like text similarity."""
            if not a or not b:
                return 0.0
            words_a = set(a.lower().split())
            words_b = set(b.lower().split())
            if not words_a or not words_b:
                return 0.0
            intersection = len(words_a & words_b)
            union = len(words_a | words_b)
            return intersection / union if union > 0 else 0.0
        
        # Get relevance scores
        def get_relevance(r: SearchResult) -> float:
            return r.rerank_score or r.final_score or r.rrf_score or r.semantic_score or 0.0
        
        selected: List[SearchResult] = []
        remaining = results.copy()
        
        # Select first by pure relevance
        remaining.sort(key=get_relevance, reverse=True)
        selected.append(remaining.pop(0))
        
        # Iteratively select by MMR
        while len(selected) < top_k and remaining:
            best_mmr = float("-inf")
            best_idx = 0
            
            for i, candidate in enumerate(remaining):
                relevance = get_relevance(candidate)
                
                # Calculate max similarity to selected results
                max_sim = 0.0
                for sel in selected:
                    sim = text_similarity(candidate.text or "", sel.text or "")
                    max_sim = max(max_sim, sim)
                
                # MMR score balances relevance and diversity
                mmr = lambda_param * relevance - (1 - lambda_param) * max_sim
                
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = i
            
            selected.append(remaining.pop(best_idx))
        
        return selected
    
    def _apply_dpp(
        self,
        results: List[SearchResult],
        embeddings: List[List[float]],
        top_k: int,
    ) -> List[SearchResult]:
        """
        Apply Determinantal Point Process for diversity.
        
        DPP models diversity through the determinant of a kernel matrix,
        naturally encouraging selection of diverse items.
        """
        try:
            import numpy as np
        except ImportError:
            logger.warning("NumPy not available for DPP, falling back to MMR")
            return self._apply_mmr(results, top_k)
        
        if len(results) != len(embeddings) or len(results) <= top_k:
            return results[:top_k]
        
        n = len(results)
        
        # Build quality scores
        qualities = np.array([
            r.rerank_score or r.final_score or r.rrf_score or 0.5
            for r in results
        ])
        
        # Build similarity matrix from embeddings
        emb_array = np.array(embeddings)
        
        # Normalize embeddings
        norms = np.linalg.norm(emb_array, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        emb_normalized = emb_array / norms
        
        # Compute similarity matrix
        sim_matrix = emb_normalized @ emb_normalized.T
        
        # Build L-ensemble kernel
        # L_ij = q_i * q_j * S_ij
        L = np.outer(qualities, qualities) * sim_matrix
        
        # Greedy MAP inference for DPP
        selected_indices = self._greedy_dpp(L, top_k)
        
        return [results[i] for i in selected_indices]
    
    def _greedy_dpp(self, L: "np.ndarray", k: int) -> List[int]:
        """Greedy MAP inference for DPP selection."""
        import numpy as np
        
        n = L.shape[0]
        selected = []
        remaining = list(range(n))
        
        for _ in range(min(k, n)):
            if not remaining:
                break
            
            # Select item that maximizes log det
            best_score = float("-inf")
            best_idx = remaining[0]
            
            for idx in remaining:
                # Score is diagonal element (quality) times diversity contribution
                if not selected:
                    score = L[idx, idx]
                else:
                    # Approximate contribution to determinant
                    sel_set = selected + [idx]
                    sub_L = L[np.ix_(sel_set, sel_set)]
                    try:
                        score = np.linalg.det(sub_L)
                    except np.linalg.LinAlgError:
                        score = 0.0
                
                if score > best_score:
                    best_score = score
                    best_idx = idx
            
            selected.append(best_idx)
            remaining.remove(best_idx)
        
        return selected
    
    def _calculate_diversity_score(self, results: List[SearchResult]) -> float:
        """Calculate a diversity score for the result set."""
        if len(results) <= 1:
            return 1.0
        
        # Measure average pairwise dissimilarity
        total_dissim = 0.0
        count = 0
        
        for i, r1 in enumerate(results):
            for r2 in results[i+1:]:
                # Text dissimilarity
                text_sim = self._text_similarity(r1.text or "", r2.text or "")
                dissim = 1.0 - text_sim
                total_dissim += dissim
                count += 1
        
        return total_dissim / count if count > 0 else 1.0
    
    def _text_similarity(self, a: str, b: str) -> float:
        """Calculate text similarity."""
        if not a or not b:
            return 0.0
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        if not words_a or not words_b:
            return 0.0
        intersection = len(words_a & words_b)
        union = len(words_a | words_b)
        return intersection / union if union > 0 else 0.0

class IntentDiversifier:
    """
    Diversify results based on detected query intents.
    
    Ensures that results cover multiple possible interpretations
    or aspects of the user's query.
    """
    
    def __init__(self, intents_per_result: int = 1):
        """Initialize intent diversifier."""
        self.intents_per_result = intents_per_result
        self._intent_keywords = {
            "definition": ["what is", "define", "meaning"],
            "procedure": ["how to", "steps", "guide"],
            "comparison": ["compare", "vs", "difference"],
            "example": ["example", "sample", "instance"],
            "reason": ["why", "because", "reason"],
            "location": ["where", "location", "place"],
            "time": ["when", "date", "time"],
        }
    
    def diversify_by_intent(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int = 10,
    ) -> List[SearchResult]:
        """
        Select results that cover different aspects of the query.
        
        Args:
            query: User query
            results: Ranked search results
            top_k: Number of results to return
            
        Returns:
            Diversified results covering multiple intents
        """
        # Detect query intents
        detected_intents = self._detect_intents(query)
        
        if not detected_intents:
            # No specific intents, return top results
            return results[:top_k]
        
        # Classify results by intent coverage
        intent_buckets: Dict[str, List[SearchResult]] = {
            intent: [] for intent in detected_intents
        }
        uncategorized: List[SearchResult] = []
        
        for result in results:
            result_intents = self._detect_intents(result.text or "")
            matched = False
            
            for intent in detected_intents:
                if intent in result_intents:
                    intent_buckets[intent].append(result)
                    matched = True
            
            if not matched:
                uncategorized.append(result)
        
        # Round-robin selection from intent buckets
        selected: List[SearchResult] = []
        seen_ids: Set[str] = set()
        
        while len(selected) < top_k:
            added_any = False
            
            for intent in detected_intents:
                if len(selected) >= top_k:
                    break
                
                bucket = intent_buckets[intent]
                for result in bucket:
                    result_id = str(result.chunk_id)
                    if result_id not in seen_ids:
                        selected.append(result)
                        seen_ids.add(result_id)
                        added_any = True
                        break
            
            # Fill with uncategorized
            if not added_any or not any(intent_buckets.values()):
                for result in uncategorized:
                    if len(selected) >= top_k:
                        break
                    result_id = str(result.chunk_id)
                    if result_id not in seen_ids:
                        selected.append(result)
                        seen_ids.add(result_id)
                break
        
        return selected
    
    def _detect_intents(self, text: str) -> List[str]:
        """Detect intents in text based on keywords."""
        text_lower = text.lower()
        detected = []
        
        for intent, keywords in self._intent_keywords.items():
            if any(kw in text_lower for kw in keywords):
                detected.append(intent)
        
        return detected
