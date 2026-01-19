"""
Unit tests for the evaluation framework.

Tests cover:
- Retrieval metrics (NDCG, MRR, Recall, Precision)
- LLM Judge
- Metrics Calculator
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from triple_hybrid_rag.evaluation.metrics import (
    RetrievalMetrics,
    MetricsCalculator,
    ndcg_at_k,
    mrr,
    recall_at_k,
    precision_at_k,
    f1_at_k,
    hit_rate,
    reciprocal_rank,
    average_precision,
    mean_average_precision,
)
from triple_hybrid_rag.evaluation.judge import (
    LLMJudge,
    JudgmentResult,
    RelevanceLevel,
    FaithfulnessResult,
    PairwiseResult,
)
from triple_hybrid_rag.types import SearchResult

# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_config():
    """Create a mock RAGConfig."""
    config = MagicMock()
    config.openai_api_key = "test-key"
    config.openai_base_url = "https://api.openai.com/v1"
    return config

@pytest.fixture
def sample_search_results():
    """Create sample SearchResult objects."""
    results = []
    for i in range(10):
        result = SearchResult(
            chunk_id=uuid4(),
            document_id=uuid4(),
            text=f"Sample text {i}",
            semantic_score=0.9 - i * 0.05,
        )
        results.append(result)
    return results

# ============================================================================
# NDCG Tests
# ============================================================================

class TestNDCG:
    """Tests for NDCG metric."""
    
    def test_perfect_ranking(self):
        """Test NDCG with perfect ranking."""
        # Retrieved in perfect order
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc1", "doc2", "doc3"}
        
        score = ndcg_at_k(retrieved, relevant, k=3)
        
        # Perfect ranking should give NDCG = 1.0
        assert score == pytest.approx(1.0, rel=0.01)
    
    def test_reversed_ranking(self):
        """Test NDCG with reversed ranking."""
        # Retrieved in reverse order
        retrieved = ["doc3", "doc2", "doc1"]
        relevant = {"doc1"}  # Only doc1 is relevant
        
        score_reversed = ndcg_at_k(retrieved, relevant, k=3)
        
        # Compare with optimal ranking
        retrieved_optimal = ["doc1", "doc2", "doc3"]
        score_optimal = ndcg_at_k(retrieved_optimal, relevant, k=3)
        
        # Reversed should have lower NDCG
        assert score_reversed < score_optimal
    
    def test_no_relevant(self):
        """Test NDCG with no relevant documents."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = set()  # No relevant docs
        
        score = ndcg_at_k(retrieved, relevant, k=3)
        
        assert score == 0.0
    
    def test_no_relevant_retrieved(self):
        """Test NDCG when no relevant docs are retrieved."""
        retrieved = ["doc4", "doc5", "doc6"]
        relevant = {"doc1", "doc2", "doc3"}
        
        score = ndcg_at_k(retrieved, relevant, k=3)
        
        assert score == 0.0
    
    def test_graded_relevance(self):
        """Test NDCG with graded relevance scores."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc1", "doc2", "doc3"}
        relevance_scores = {
            "doc1": 3.0,  # Highly relevant
            "doc2": 2.0,  # Relevant
            "doc3": 1.0,  # Somewhat relevant
        }
        
        score = ndcg_at_k(retrieved, relevant, relevance_scores, k=3)
        
        # Should be 1.0 because highest scores are ranked first
        assert score == pytest.approx(1.0, rel=0.01)
    
    def test_k_larger_than_results(self):
        """Test NDCG when k > len(retrieved)."""
        retrieved = ["doc1", "doc2"]
        relevant = {"doc1", "doc2", "doc3", "doc4", "doc5"}
        
        score = ndcg_at_k(retrieved, relevant, k=10)
        
        # Should still compute valid score
        assert 0.0 <= score <= 1.0

# ============================================================================
# MRR Tests
# ============================================================================

class TestMRR:
    """Tests for Mean Reciprocal Rank metric."""
    
    def test_first_position(self):
        """Test MRR when relevant doc is at first position."""
        retrieved_list = [["doc1", "doc2", "doc3"]]
        relevant_list = [{"doc1"}]
        
        score = mrr(retrieved_list, relevant_list)
        
        assert score == 1.0  # 1/1
    
    def test_second_position(self):
        """Test MRR when relevant doc is at second position."""
        retrieved_list = [["doc1", "doc2", "doc3"]]
        relevant_list = [{"doc2"}]
        
        score = mrr(retrieved_list, relevant_list)
        
        assert score == 0.5  # 1/2
    
    def test_multiple_queries(self):
        """Test MRR with multiple queries."""
        retrieved_list = [
            ["doc1", "doc2", "doc3"],  # First relevant at position 1
            ["doc4", "doc5", "doc6"],  # First relevant at position 3
        ]
        relevant_list = [
            {"doc1"},
            {"doc6"},
        ]
        
        score = mrr(retrieved_list, relevant_list)
        
        # (1/1 + 1/3) / 2 = 0.667
        assert score == pytest.approx(0.667, rel=0.01)
    
    def test_no_relevant_found(self):
        """Test MRR when no relevant doc is found."""
        retrieved_list = [["doc4", "doc5", "doc6"]]
        relevant_list = [{"doc1", "doc2", "doc3"}]
        
        score = mrr(retrieved_list, relevant_list)
        
        assert score == 0.0
    
    def test_empty_queries(self):
        """Test MRR with empty query list."""
        score = mrr([], [])
        
        assert score == 0.0

class TestReciprocalRank:
    """Tests for single-query Reciprocal Rank."""
    
    def test_first_position(self):
        """Test RR at first position."""
        rr = reciprocal_rank(["doc1", "doc2"], {"doc1"})
        assert rr == 1.0
    
    def test_third_position(self):
        """Test RR at third position."""
        rr = reciprocal_rank(["doc1", "doc2", "doc3"], {"doc3"})
        assert rr == pytest.approx(0.333, rel=0.01)
    
    def test_not_found(self):
        """Test RR when not found."""
        rr = reciprocal_rank(["doc1", "doc2"], {"doc3"})
        assert rr == 0.0

# ============================================================================
# Recall Tests
# ============================================================================

class TestRecall:
    """Tests for Recall@k metric."""
    
    def test_perfect_recall(self):
        """Test perfect recall."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc1", "doc2", "doc3"}
        
        score = recall_at_k(retrieved, relevant, k=3)
        
        assert score == 1.0
    
    def test_partial_recall(self):
        """Test partial recall."""
        retrieved = ["doc1", "doc2", "doc4", "doc5"]
        relevant = {"doc1", "doc2", "doc3"}  # 3 relevant, 2 retrieved
        
        score = recall_at_k(retrieved, relevant, k=4)
        
        assert score == pytest.approx(0.667, rel=0.01)  # 2/3
    
    def test_recall_with_k_cutoff(self):
        """Test recall respects k cutoff."""
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = {"doc1", "doc4", "doc5"}
        
        score_k2 = recall_at_k(retrieved, relevant, k=2)
        score_k5 = recall_at_k(retrieved, relevant, k=5)
        
        assert score_k2 == pytest.approx(0.333, rel=0.01)  # 1/3
        assert score_k5 == 1.0  # 3/3
    
    def test_no_relevant(self):
        """Test recall with no relevant documents."""
        retrieved = ["doc1", "doc2"]
        relevant = set()
        
        score = recall_at_k(retrieved, relevant, k=2)
        
        assert score == 0.0

# ============================================================================
# Precision Tests
# ============================================================================

class TestPrecision:
    """Tests for Precision@k metric."""
    
    def test_perfect_precision(self):
        """Test perfect precision."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc1", "doc2", "doc3"}
        
        score = precision_at_k(retrieved, relevant, k=3)
        
        assert score == 1.0
    
    def test_half_precision(self):
        """Test 50% precision."""
        retrieved = ["doc1", "doc4", "doc2", "doc5"]
        relevant = {"doc1", "doc2"}  # 2 of 4 retrieved are relevant
        
        score = precision_at_k(retrieved, relevant, k=4)
        
        assert score == 0.5
    
    def test_precision_k_zero(self):
        """Test precision at k=0."""
        retrieved = ["doc1", "doc2"]
        relevant = {"doc1"}
        
        score = precision_at_k(retrieved, relevant, k=0)
        
        assert score == 0.0
    
    def test_precision_cutoff(self):
        """Test precision respects k cutoff."""
        retrieved = ["doc1", "doc2", "doc3", "doc4"]
        relevant = {"doc1", "doc2"}
        
        score_k2 = precision_at_k(retrieved, relevant, k=2)
        score_k4 = precision_at_k(retrieved, relevant, k=4)
        
        assert score_k2 == 1.0  # 2/2
        assert score_k4 == 0.5  # 2/4

# ============================================================================
# F1 Tests
# ============================================================================

class TestF1:
    """Tests for F1@k metric."""
    
    def test_perfect_f1(self):
        """Test perfect F1."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc1", "doc2", "doc3"}
        
        score = f1_at_k(retrieved, relevant, k=3)
        
        assert score == 1.0
    
    def test_f1_calculation(self):
        """Test F1 calculation formula."""
        retrieved = ["doc1", "doc2", "doc4", "doc5"]
        relevant = {"doc1", "doc2", "doc3"}
        
        p = precision_at_k(retrieved, relevant, k=4)  # 2/4 = 0.5
        r = recall_at_k(retrieved, relevant, k=4)  # 2/3 = 0.667
        
        expected_f1 = 2 * p * r / (p + r)  # 2 * 0.5 * 0.667 / 1.167 ≈ 0.571
        actual_f1 = f1_at_k(retrieved, relevant, k=4)
        
        assert actual_f1 == pytest.approx(expected_f1, rel=0.01)
    
    def test_f1_zero_when_nothing_relevant(self):
        """Test F1 is zero when nothing relevant retrieved."""
        retrieved = ["doc4", "doc5"]
        relevant = {"doc1", "doc2"}
        
        score = f1_at_k(retrieved, relevant, k=2)
        
        assert score == 0.0

# ============================================================================
# Hit Rate Tests
# ============================================================================

class TestHitRate:
    """Tests for Hit Rate metric."""
    
    def test_perfect_hit_rate(self):
        """Test 100% hit rate."""
        retrieved_list = [
            ["doc1", "doc2"],
            ["doc3", "doc4"],
        ]
        relevant_list = [
            {"doc1"},
            {"doc3"},
        ]
        
        score = hit_rate(retrieved_list, relevant_list, k=2)
        
        assert score == 1.0
    
    def test_partial_hit_rate(self):
        """Test partial hit rate."""
        retrieved_list = [
            ["doc1", "doc2"],  # Hit
            ["doc5", "doc6"],  # Miss
        ]
        relevant_list = [
            {"doc1"},
            {"doc3"},
        ]
        
        score = hit_rate(retrieved_list, relevant_list, k=2)
        
        assert score == 0.5
    
    def test_hit_rate_respects_k(self):
        """Test hit rate respects k cutoff."""
        retrieved_list = [
            ["doc4", "doc5", "doc1"],  # Hit only at position 3
        ]
        relevant_list = [
            {"doc1"},
        ]
        
        score_k2 = hit_rate(retrieved_list, relevant_list, k=2)
        score_k3 = hit_rate(retrieved_list, relevant_list, k=3)
        
        assert score_k2 == 0.0
        assert score_k3 == 1.0

# ============================================================================
# Average Precision Tests
# ============================================================================

class TestAveragePrecision:
    """Tests for Average Precision."""
    
    def test_perfect_ap(self):
        """Test perfect AP."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc1", "doc2", "doc3"}
        
        ap = average_precision(retrieved, relevant)
        
        assert ap == 1.0
    
    def test_ap_calculation(self):
        """Test AP calculation."""
        # Relevant at positions 1, 3
        retrieved = ["doc1", "doc4", "doc2", "doc5"]
        relevant = {"doc1", "doc2"}
        
        # P@1 = 1/1 = 1.0, P@3 = 2/3 = 0.667
        # AP = (1.0 + 0.667) / 2 = 0.833
        ap = average_precision(retrieved, relevant)
        
        assert ap == pytest.approx(0.833, rel=0.01)
    
    def test_ap_no_relevant(self):
        """Test AP with no relevant documents."""
        ap = average_precision(["doc1", "doc2"], set())
        assert ap == 0.0

class TestMAP:
    """Tests for Mean Average Precision."""
    
    def test_map_calculation(self):
        """Test MAP calculation."""
        retrieved_list = [
            ["doc1", "doc2"],  # AP = 1.0
            ["doc3", "doc1"],  # AP = 1/2 = 0.5
        ]
        relevant_list = [
            {"doc1"},
            {"doc1"},
        ]
        
        map_score = mean_average_precision(retrieved_list, relevant_list)
        
        assert map_score == pytest.approx(0.75, rel=0.01)

# ============================================================================
# Metrics Calculator Tests
# ============================================================================

class TestMetricsCalculator:
    """Tests for MetricsCalculator."""
    
    def test_add_single_query(self):
        """Test adding a single query."""
        calc = MetricsCalculator()
        
        calc.add_query(
            retrieved_ids=["doc1", "doc2", "doc3"],
            relevant_ids={"doc1", "doc2"},
            latency_ms=100.0,
        )
        
        assert len(calc._retrieved_ids_list) == 1
        assert len(calc._latencies) == 1
    
    def test_compute_metrics(self):
        """Test computing aggregate metrics."""
        calc = MetricsCalculator()
        
        # Add queries
        calc.add_query(
            retrieved_ids=["doc1", "doc2", "doc3"],
            relevant_ids={"doc1"},
            latency_ms=100.0,
        )
        calc.add_query(
            retrieved_ids=["doc4", "doc5", "doc6"],
            relevant_ids={"doc4", "doc5"},
            latency_ms=150.0,
        )
        
        metrics = calc.compute()
        
        assert metrics.num_queries == 2
        assert metrics.mrr > 0
        assert metrics.latency_p50_ms > 0
    
    def test_compute_empty(self):
        """Test computing with no data."""
        calc = MetricsCalculator()
        
        metrics = calc.compute()
        
        assert metrics.num_queries == 0
        assert metrics.mrr == 0.0
    
    def test_add_query_from_results(self, sample_search_results):
        """Test adding query from SearchResult objects."""
        calc = MetricsCalculator()
        
        relevant_ids = {str(sample_search_results[0].chunk_id)}
        
        calc.add_query_from_results(
            results=sample_search_results[:5],
            relevant_ids=relevant_ids,
            latency_ms=100.0,
        )
        
        assert len(calc._retrieved_ids_list) == 1
        assert len(calc._retrieved_ids_list[0]) == 5
    
    def test_reset(self):
        """Test reset clears all data."""
        calc = MetricsCalculator()
        
        calc.add_query(["doc1"], {"doc1"})
        calc.reset()
        
        assert len(calc._retrieved_ids_list) == 0
        assert len(calc._latencies) == 0
    
    def test_metrics_to_dict(self):
        """Test RetrievalMetrics.to_dict()."""
        metrics = RetrievalMetrics(
            ndcg_at_10=0.85,
            mrr=0.75,
            recall_at_10=0.9,
        )
        
        d = metrics.to_dict()
        
        assert d["ndcg@10"] == 0.85
        assert d["mrr"] == 0.75
        assert d["recall@10"] == 0.9
    
    def test_metrics_str(self):
        """Test RetrievalMetrics.__str__()."""
        metrics = RetrievalMetrics(
            ndcg_at_10=0.85,
            mrr=0.75,
            recall_at_10=0.9,
            precision_at_10=0.8,
        )
        
        s = str(metrics)
        
        assert "NDCG@10" in s
        assert "MRR" in s
        assert "0.85" in s or "0.8500" in s

# ============================================================================
# LLM Judge Tests
# ============================================================================

class TestRelevanceLevel:
    """Tests for RelevanceLevel enum."""
    
    def test_from_score_boundaries(self):
        """Test score to level conversion."""
        assert RelevanceLevel.from_score(0.0) == RelevanceLevel.NOT_RELEVANT
        assert RelevanceLevel.from_score(0.19) == RelevanceLevel.NOT_RELEVANT
        assert RelevanceLevel.from_score(0.2) == RelevanceLevel.MARGINALLY_RELEVANT
        assert RelevanceLevel.from_score(0.4) == RelevanceLevel.RELEVANT
        assert RelevanceLevel.from_score(0.6) == RelevanceLevel.HIGHLY_RELEVANT
        assert RelevanceLevel.from_score(0.8) == RelevanceLevel.PERFECT
        assert RelevanceLevel.from_score(1.0) == RelevanceLevel.PERFECT
    
    def test_level_values(self):
        """Test level numeric values."""
        assert RelevanceLevel.NOT_RELEVANT.value == 0
        assert RelevanceLevel.PERFECT.value == 4

class TestJudgmentResult:
    """Tests for JudgmentResult dataclass."""
    
    def test_judgment_result_creation(self):
        """Test creating JudgmentResult."""
        result = JudgmentResult(
            query="test query",
            document_text="test document",
            relevance_score=0.75,
            relevance_level=RelevanceLevel.HIGHLY_RELEVANT,
            reasoning="Good match",
        )
        
        assert result.query == "test query"
        assert result.relevance_score == 0.75
        assert result.relevance_level == RelevanceLevel.HIGHLY_RELEVANT

class TestLLMJudge:
    """Tests for LLM Judge."""
    
    @pytest.mark.asyncio
    async def test_judge_relevance_mock(self, mock_config):
        """Test relevance judgment with mocked LLM response."""
        judge = LLMJudge(config=mock_config)
        
        # Mock the HTTP client
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": "SCORE: 3\nREASONING: Good relevance match"
                }
            }]
        }
        mock_response.raise_for_status = MagicMock()
        
        with patch.object(judge, '_call_llm', return_value="SCORE: 3\nREASONING: Good relevance"):
            result = await judge.judge_relevance(
                query="What is machine learning?",
                document_text="Machine learning is a subset of AI...",
            )
        
        assert result.relevance_score == 0.75  # 3/4
        assert result.relevance_level == RelevanceLevel.HIGHLY_RELEVANT
    
    def test_judge_initialization(self, mock_config):
        """Test judge initialization."""
        judge = LLMJudge(
            config=mock_config,
            model="gpt-4",
            timeout=60.0,
        )
        
        assert judge.model == "gpt-4"
        assert judge.timeout == 60.0

class TestFaithfulnessResult:
    """Tests for FaithfulnessResult dataclass."""
    
    def test_faithfulness_result_creation(self):
        """Test creating FaithfulnessResult."""
        result = FaithfulnessResult(
            answer="The capital is Paris.",
            context="France's capital city is Paris.",
            faithfulness_score=1.0,
            supported_claims=1,
            unsupported_claims=0,
        )
        
        assert result.faithfulness_score == 1.0
        assert result.supported_claims == 1

class TestPairwiseResult:
    """Tests for PairwiseResult dataclass."""
    
    def test_pairwise_result_creation(self):
        """Test creating PairwiseResult."""
        result = PairwiseResult(
            query="test query",
            winner="A",
            confidence=0.9,
            reasoning="A is more relevant",
        )
        
        assert result.winner == "A"
        assert result.confidence == 0.9

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
