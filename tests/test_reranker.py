from triple_hybrid_rag.core.reranker import _parse_rerank_scores


def test_parse_rerank_scores_results_format():
    payload = {
        "results": [
            {"index": 0, "relevance_score": 0.9},
            {"index": 1, "relevance_score": 0.1},
        ]
    }

    scores = _parse_rerank_scores(payload, expected_len=2)
    assert scores == [0.9, 0.1]


def test_parse_rerank_scores_data_format():
    payload = {"data": [{"score": 0.3}, {"score": 0.6}]}
    scores = _parse_rerank_scores(payload, expected_len=2)
    assert scores == [0.3, 0.6]
