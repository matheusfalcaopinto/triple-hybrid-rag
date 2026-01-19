from triple_hybrid_rag.rag import _apply_post_rerank_safety
from triple_hybrid_rag.types import SearchResult


def test_post_rerank_safety_filters_and_trims():
    results = [
        SearchResult(chunk_id="1", parent_id="1", document_id="1", text="a", rerank_score=0.9),
        SearchResult(chunk_id="2", parent_id="2", document_id="2", text="b", rerank_score=0.4),
        SearchResult(chunk_id="3", parent_id="3", document_id="3", text="c", rerank_score=0.2),
    ]

    filtered, refused, reason, max_score = _apply_post_rerank_safety(
        results,
        safety_threshold=0.6,
        denoise_alpha=0.5,
        denoise_enabled=True,
        top_k=2,
    )

    assert not refused
    assert reason is None
    assert max_score == 0.9
    assert len(filtered) == 1
    assert filtered[0].rerank_score == 0.9


def test_post_rerank_safety_refuses_when_below_threshold():
    results = [
        SearchResult(chunk_id="1", parent_id="1", document_id="1", text="a", rerank_score=0.3),
        SearchResult(chunk_id="2", parent_id="2", document_id="2", text="b", rerank_score=0.2),
    ]

    filtered, refused, reason, max_score = _apply_post_rerank_safety(
        results,
        safety_threshold=0.6,
        denoise_alpha=0.5,
        denoise_enabled=True,
        top_k=2,
    )

    assert refused
    assert reason == "Below safety threshold"
    assert filtered == []
    assert max_score == 0.3
