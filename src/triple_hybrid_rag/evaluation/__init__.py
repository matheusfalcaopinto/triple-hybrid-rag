"""
Evaluation Framework for Triple-Hybrid-RAG

This module provides comprehensive evaluation metrics and tools:
- Retrieval metrics: NDCG, MRR, Recall, Precision
- Quality assessment: LLM-as-judge
- Synthetic query generation
- A/B testing support
"""

from triple_hybrid_rag.evaluation.metrics import (
    RetrievalMetrics,
    MetricsCalculator,
    ndcg_at_k,
    mrr,
    recall_at_k,
    precision_at_k,
    f1_at_k,
    hit_rate,
)
from triple_hybrid_rag.evaluation.judge import (
    LLMJudge,
    JudgmentResult,
    RelevanceLevel,
)

__all__ = [
    # Metrics
    "RetrievalMetrics",
    "MetricsCalculator",
    "ndcg_at_k",
    "mrr",
    "recall_at_k",
    "precision_at_k",
    "f1_at_k",
    "hit_rate",
    # LLM Judge
    "LLMJudge",
    "JudgmentResult",
    "RelevanceLevel",
]
