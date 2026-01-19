from triple_hybrid_rag.core.entity_extractor import (
    _parse_extraction_response,
    _normalize_name,
    GraphExtractionResult,
)
from triple_hybrid_rag.config import RAGConfig


def test_normalize_name():
    assert _normalize_name("Acme Corp.") == "acme corp."
    assert _normalize_name("  Foo/Bar  ") == "foobar"


def test_parse_extraction_response_defaults():
    config = RAGConfig()
    payload = {
        "entities": [
            {
                "chunk_id": "c1",
                "name": "Alice",
                "canonical_name": "alice",
                "entity_type": "PERSON",
                "mention_text": "Alice",
                "confidence": 0.9,
            },
            {
                "chunk_id": "c1",
                "name": "Unknown",
                "entity_type": "NOT_A_TYPE",
                "mention_text": "Unknown",
                "confidence": "0.7",
            },
        ],
        "relations": [
            {
                "chunk_id": "c1",
                "subject": "alice",
                "object": "unknown",
                "relation_type": "RELATED_TO",
                "confidence": 0.8,
            }
        ],
    }

    result = _parse_extraction_response(str(payload).replace("'", '"'), config)
    assert isinstance(result, GraphExtractionResult)
    assert len(result.entities) == 2
    assert result.entities[1].entity_type == "CONCEPT"
    assert result.relations[0].relation_type == "RELATED_TO"
