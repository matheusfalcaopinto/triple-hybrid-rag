"""
Retrieval Evaluation Metrics for Triple-Hybrid-RAG

Implements standard information retrieval metrics:
- NDCG (Normalized Discounted Cumulative Gain)
- MRR (Mean Reciprocal Rank)
- Recall@k
- Precision@k
- F1@k
- Hit Rate

Reference:
- Manning et al. "Introduction to Information Retrieval"
- Järvelin & Kekäläinen "Cumulated Gain-Based Evaluation of IR Techniques"
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Union
from uuid import UUID

from triple_hybrid_rag.types import SearchResult

@dataclass
class RetrievalMetrics:
    """Complete set of retrieval evaluation metrics."""
    
    # Ranking metrics
    ndcg_at_5: float = 0.0
    ndcg_at_10: float = 0.0
    ndcg_at_20: float = 0.0
    
    mrr: float = 0.0
    
    # Recall metrics
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    recall_at_20: float = 0.0
    
    # Precision metrics
    precision_at_5: float = 0.0
    precision_at_10: float = 0.0
    precision_at_20: float = 0.0
    
    # F1 metrics
    f1_at_5: float = 0.0
    f1_at_10: float = 0.0
    f1_at_20: float = 0.0
    
    # Hit rate
    hit_rate: float = 0.0
    
    # Latency metrics
    latency_p50_ms: float = 0.0
    latency_p90_ms: float = 0.0
    latency_p99_ms: float = 0.0
    
    # Result count
    num_queries: int = 0
    total_relevant: int = 0
    total_retrieved: int = 0
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "ndcg@5": self.ndcg_at_5,
            "ndcg@10": self.ndcg_at_10,
            "ndcg@20": self.ndcg_at_20,
            "mrr": self.mrr,
            "recall@5": self.recall_at_5,
            "recall@10": self.recall_at_10,
            "recall@20": self.recall_at_20,
            "precision@5": self.precision_at_5,
            "precision@10": self.precision_at_10,
            "precision@20": self.precision_at_20,
            "f1@5": self.f1_at_5,
            "f1@10": self.f1_at_10,
            "f1@20": self.f1_at_20,
            "hit_rate": self.hit_rate,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p90_ms": self.latency_p90_ms,
            "latency_p99_ms": self.latency_p99_ms,
            "num_queries": self.num_queries,
        }
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"NDCG@10: {self.ndcg_at_10:.4f} | "
            f"MRR: {self.mrr:.4f} | "
            f"Recall@10: {self.recall_at_10:.4f} | "
            f"Precision@10: {self.precision_at_10:.4f}"
        )

def ndcg_at_k(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
    relevance_scores: Optional[Dict[str, float]] = None,
    k: int = 10,
) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at k.
    
    Args:
        retrieved_ids: List of retrieved document IDs in ranked order
        relevant_ids: Set of relevant document IDs
        relevance_scores: Optional graded relevance scores (default: binary)
        k: Cutoff rank
        
    Returns:
        NDCG@k score (0.0 to 1.0)
    """
    if not relevant_ids:
        return 0.0
    
    # Get top-k retrieved
    top_k = retrieved_ids[:k]
    
    # Calculate DCG
    dcg = 0.0
    for i, doc_id in enumerate(top_k):
        if doc_id in relevant_ids:
            rel = relevance_scores.get(doc_id, 1.0) if relevance_scores else 1.0
            # DCG formula: rel / log2(rank + 1)
            dcg += rel / math.log2(i + 2)  # +2 because i is 0-indexed
    
    # Calculate ideal DCG (IDCG)
    if relevance_scores:
        # Sort by relevance score
        sorted_rels = sorted(
            [relevance_scores.get(rid, 1.0) for rid in relevant_ids],
            reverse=True,
        )
    else:
        sorted_rels = [1.0] * len(relevant_ids)
    
    idcg = 0.0
    for i, rel in enumerate(sorted_rels[:k]):
        idcg += rel / math.log2(i + 2)
    
    if idcg == 0.0:
        return 0.0
    
    return dcg / idcg

def mrr(
    retrieved_ids_list: List[List[str]],
    relevant_ids_list: List[Set[str]],
) -> float:
    """
    Calculate Mean Reciprocal Rank across multiple queries.
    
    Args:
        retrieved_ids_list: List of retrieved ID lists (one per query)
        relevant_ids_list: List of relevant ID sets (one per query)
        
    Returns:
        MRR score (0.0 to 1.0)
    """
    if not retrieved_ids_list:
        return 0.0
    
    rr_sum = 0.0
    
    for retrieved_ids, relevant_ids in zip(retrieved_ids_list, relevant_ids_list):
        rr = 0.0
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_ids:
                rr = 1.0 / (i + 1)
                break
        rr_sum += rr
    
    return rr_sum / len(retrieved_ids_list)

def reciprocal_rank(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
) -> float:
    """
    Calculate Reciprocal Rank for a single query.
    
    Args:
        retrieved_ids: List of retrieved document IDs in ranked order
        relevant_ids: Set of relevant document IDs
        
    Returns:
        Reciprocal rank (0.0 to 1.0)
    """
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0

def recall_at_k(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
    k: int = 10,
) -> float:
    """
    Calculate Recall at k.
    
    Args:
        retrieved_ids: List of retrieved document IDs
        relevant_ids: Set of relevant document IDs
        k: Cutoff rank
        
    Returns:
        Recall@k score (0.0 to 1.0)
    """
    if not relevant_ids:
        return 0.0
    
    top_k = set(retrieved_ids[:k])
    relevant_retrieved = len(top_k & relevant_ids)
    
    return relevant_retrieved / len(relevant_ids)

def precision_at_k(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
    k: int = 10,
) -> float:
    """
    Calculate Precision at k.
    
    Args:
        retrieved_ids: List of retrieved document IDs
        relevant_ids: Set of relevant document IDs
        k: Cutoff rank
        
    Returns:
        Precision@k score (0.0 to 1.0)
    """
    if k == 0:
        return 0.0
    
    top_k = set(retrieved_ids[:k])
    relevant_retrieved = len(top_k & relevant_ids)
    
    return relevant_retrieved / k

def f1_at_k(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
    k: int = 10,
) -> float:
    """
    Calculate F1 score at k.
    
    Args:
        retrieved_ids: List of retrieved document IDs
        relevant_ids: Set of relevant document IDs
        k: Cutoff rank
        
    Returns:
        F1@k score (0.0 to 1.0)
    """
    p = precision_at_k(retrieved_ids, relevant_ids, k)
    r = recall_at_k(retrieved_ids, relevant_ids, k)
    
    if p + r == 0:
        return 0.0
    
    return 2 * p * r / (p + r)

def hit_rate(
    retrieved_ids_list: List[List[str]],
    relevant_ids_list: List[Set[str]],
    k: int = 10,
) -> float:
    """
    Calculate Hit Rate (Success Rate) at k.
    
    Hit rate is the fraction of queries where at least one
    relevant document appears in the top-k results.
    
    Args:
        retrieved_ids_list: List of retrieved ID lists (one per query)
        relevant_ids_list: List of relevant ID sets (one per query)
        k: Cutoff rank
        
    Returns:
        Hit rate (0.0 to 1.0)
    """
    if not retrieved_ids_list:
        return 0.0
    
    hits = 0
    for retrieved_ids, relevant_ids in zip(retrieved_ids_list, relevant_ids_list):
        top_k = set(retrieved_ids[:k])
        if top_k & relevant_ids:
            hits += 1
    
    return hits / len(retrieved_ids_list)

def average_precision(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
) -> float:
    """
    Calculate Average Precision for a single query.
    
    Args:
        retrieved_ids: List of retrieved document IDs
        relevant_ids: Set of relevant document IDs
        
    Returns:
        Average Precision score
    """
    if not relevant_ids:
        return 0.0
    
    precision_sum = 0.0
    relevant_count = 0
    
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            relevant_count += 1
            precision_at_i = relevant_count / (i + 1)
            precision_sum += precision_at_i
    
    return precision_sum / len(relevant_ids)

def mean_average_precision(
    retrieved_ids_list: List[List[str]],
    relevant_ids_list: List[Set[str]],
) -> float:
    """
    Calculate Mean Average Precision (MAP) across queries.
    
    Args:
        retrieved_ids_list: List of retrieved ID lists
        relevant_ids_list: List of relevant ID sets
        
    Returns:
        MAP score
    """
    if not retrieved_ids_list:
        return 0.0
    
    ap_sum = sum(
        average_precision(ret, rel)
        for ret, rel in zip(retrieved_ids_list, relevant_ids_list)
    )
    
    return ap_sum / len(retrieved_ids_list)

class MetricsCalculator:
    """
    Calculate and aggregate retrieval metrics.
    
    Usage:
        calc = MetricsCalculator()
        
        # Add query results
        for query in queries:
            results = retriever.search(query.text)
            calc.add_query(
                retrieved_ids=[str(r.chunk_id) for r in results],
                relevant_ids=query.relevant_chunk_ids,
                latency_ms=query.latency_ms,
            )
        
        # Get aggregate metrics
        metrics = calc.compute()
    """
    
    def __init__(self):
        """Initialize metrics calculator."""
        self._retrieved_ids_list: List[List[str]] = []
        self._relevant_ids_list: List[Set[str]] = []
        self._relevance_scores_list: List[Dict[str, float]] = []
        self._latencies: List[float] = []
    
    def add_query(
        self,
        retrieved_ids: List[str],
        relevant_ids: Set[str],
        relevance_scores: Optional[Dict[str, float]] = None,
        latency_ms: Optional[float] = None,
    ) -> None:
        """
        Add results for a single query.
        
        Args:
            retrieved_ids: Retrieved document IDs in ranked order
            relevant_ids: Set of relevant document IDs
            relevance_scores: Optional graded relevance scores
            latency_ms: Query latency in milliseconds
        """
        self._retrieved_ids_list.append(retrieved_ids)
        self._relevant_ids_list.append(relevant_ids)
        self._relevance_scores_list.append(relevance_scores or {})
        
        if latency_ms is not None:
            self._latencies.append(latency_ms)
    
    def add_query_from_results(
        self,
        results: List[SearchResult],
        relevant_ids: Set[str],
        relevance_scores: Optional[Dict[str, float]] = None,
        latency_ms: Optional[float] = None,
    ) -> None:
        """
        Add query results directly from SearchResult objects.
        
        Args:
            results: List of SearchResult objects
            relevant_ids: Set of relevant chunk IDs
            relevance_scores: Optional graded relevance
            latency_ms: Query latency
        """
        retrieved_ids = [str(r.chunk_id) for r in results]
        self.add_query(retrieved_ids, relevant_ids, relevance_scores, latency_ms)
    
    def compute(self) -> RetrievalMetrics:
        """
        Compute aggregate metrics across all queries.
        
        Returns:
            RetrievalMetrics with all computed values
        """
        n = len(self._retrieved_ids_list)
        
        if n == 0:
            return RetrievalMetrics()
        
        # Calculate NDCG at different k
        ndcg_5 = sum(
            ndcg_at_k(ret, rel, scores, k=5)
            for ret, rel, scores in zip(
                self._retrieved_ids_list,
                self._relevant_ids_list,
                self._relevance_scores_list,
            )
        ) / n
        
        ndcg_10 = sum(
            ndcg_at_k(ret, rel, scores, k=10)
            for ret, rel, scores in zip(
                self._retrieved_ids_list,
                self._relevant_ids_list,
                self._relevance_scores_list,
            )
        ) / n
        
        ndcg_20 = sum(
            ndcg_at_k(ret, rel, scores, k=20)
            for ret, rel, scores in zip(
                self._retrieved_ids_list,
                self._relevant_ids_list,
                self._relevance_scores_list,
            )
        ) / n
        
        # Calculate MRR
        mrr_val = mrr(self._retrieved_ids_list, self._relevant_ids_list)
        
        # Calculate Recall
        recall_5 = sum(
            recall_at_k(ret, rel, k=5)
            for ret, rel in zip(self._retrieved_ids_list, self._relevant_ids_list)
        ) / n
        
        recall_10 = sum(
            recall_at_k(ret, rel, k=10)
            for ret, rel in zip(self._retrieved_ids_list, self._relevant_ids_list)
        ) / n
        
        recall_20 = sum(
            recall_at_k(ret, rel, k=20)
            for ret, rel in zip(self._retrieved_ids_list, self._relevant_ids_list)
        ) / n
        
        # Calculate Precision
        precision_5 = sum(
            precision_at_k(ret, rel, k=5)
            for ret, rel in zip(self._retrieved_ids_list, self._relevant_ids_list)
        ) / n
        
        precision_10 = sum(
            precision_at_k(ret, rel, k=10)
            for ret, rel in zip(self._retrieved_ids_list, self._relevant_ids_list)
        ) / n
        
        precision_20 = sum(
            precision_at_k(ret, rel, k=20)
            for ret, rel in zip(self._retrieved_ids_list, self._relevant_ids_list)
        ) / n
        
        # Calculate F1
        f1_5 = sum(
            f1_at_k(ret, rel, k=5)
            for ret, rel in zip(self._retrieved_ids_list, self._relevant_ids_list)
        ) / n
        
        f1_10 = sum(
            f1_at_k(ret, rel, k=10)
            for ret, rel in zip(self._retrieved_ids_list, self._relevant_ids_list)
        ) / n
        
        f1_20 = sum(
            f1_at_k(ret, rel, k=20)
            for ret, rel in zip(self._retrieved_ids_list, self._relevant_ids_list)
        ) / n
        
        # Calculate Hit Rate
        hit_rate_val = hit_rate(self._retrieved_ids_list, self._relevant_ids_list, k=10)
        
        # Calculate latency percentiles
        latency_p50 = 0.0
        latency_p90 = 0.0
        latency_p99 = 0.0
        
        if self._latencies:
            sorted_latencies = sorted(self._latencies)
            latency_p50 = self._percentile(sorted_latencies, 50)
            latency_p90 = self._percentile(sorted_latencies, 90)
            latency_p99 = self._percentile(sorted_latencies, 99)
        
        # Count totals
        total_relevant = sum(len(rel) for rel in self._relevant_ids_list)
        total_retrieved = sum(len(ret) for ret in self._retrieved_ids_list)
        
        return RetrievalMetrics(
            ndcg_at_5=ndcg_5,
            ndcg_at_10=ndcg_10,
            ndcg_at_20=ndcg_20,
            mrr=mrr_val,
            recall_at_5=recall_5,
            recall_at_10=recall_10,
            recall_at_20=recall_20,
            precision_at_5=precision_5,
            precision_at_10=precision_10,
            precision_at_20=precision_20,
            f1_at_5=f1_5,
            f1_at_10=f1_10,
            f1_at_20=f1_20,
            hit_rate=hit_rate_val,
            latency_p50_ms=latency_p50,
            latency_p90_ms=latency_p90,
            latency_p99_ms=latency_p99,
            num_queries=n,
            total_relevant=total_relevant,
            total_retrieved=total_retrieved,
        )
    
    def _percentile(self, sorted_list: List[float], p: int) -> float:
        """Calculate percentile from sorted list."""
        if not sorted_list:
            return 0.0
        
        k = (len(sorted_list) - 1) * p / 100
        f = int(k)
        c = f + 1
        
        if c >= len(sorted_list):
            return sorted_list[-1]
        
        return sorted_list[f] + (k - f) * (sorted_list[c] - sorted_list[f])
    
    def reset(self) -> None:
        """Reset all accumulated data."""
        self._retrieved_ids_list.clear()
        self._relevant_ids_list.clear()
        self._relevance_scores_list.clear()
        self._latencies.clear()
