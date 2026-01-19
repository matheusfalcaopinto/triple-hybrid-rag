# Phase 3: Robustness & Edge Cases - Walkthrough

> **Completed:** January 15, 2026  
> **Total Tests Added:** 57 new tests (172 total RAG2 tests)  
> **Duration:** ~1 hour

---

## Executive Summary

Phase 3 focused on adding robustness features and testing edge cases to ensure production reliability:

1. **Retry Logic** - Added tenacity-based retry with exponential backoff for entity extraction
2. **Gundam Tiling E2E** - Comprehensive tests for large image OCR processing
3. **Skip Planning Tests** - Edge case tests for retrieval pipeline paths

All **172 RAG2 tests pass** (up from 115 in Phase 2).

---

## Task 3.1: Retry Logic for Entity Extraction

### Changes Made

**File:** `src/voice_agent/rag2/ingest.py`

Added tenacity retry decorator to handle transient failures during entity extraction:

```python
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
    RetryError,
)
import httpx

# ...

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((httpx.HTTPError, TimeoutError, ConnectionError, OSError)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
async def _extract_entities_for_parent(
    self,
    parent_id: str,
    parent_text: str,
    parent_context: Optional[str],
    document_title: str,
    document_id: str,
    db_parent_id: str,
    child_ids: List[str],
) -> Dict[str, int]:
    """Extract entities for a single parent chunk with retry logic."""
    # ...
```

### Retry Behavior

| Attempt | Wait Before Retry |
|---------|------------------|
| 1       | -                |
| 2       | 2 seconds        |
| 3       | 4 seconds        |
| Fail    | Original exception raised |

### Errors That Trigger Retry

- `httpx.HTTPError` - API communication errors
- `TimeoutError` - Connection timeouts
- `ConnectionError` - Network issues
- `OSError` - System-level I/O errors

### Errors That Do NOT Retry

- `RuntimeError` - Extraction returned errors (API rate limit, invalid response)
- `ValueError` - Invalid input
- Other application errors

### Test File

**File:** `tests/test_rag2_ingest.py` (19 tests)

| Test Class | Tests | Description |
|------------|-------|-------------|
| `TestIngestStats` | 3 | Stats dataclass validation |
| `TestIngestResult` | 2 | Result dataclass validation |
| `TestRAG2Ingestor` | 4 | Ingestor initialization |
| `TestEntityExtractionRetry` | 6 | **Retry logic verification** |
| `TestExtractEntitiesPipeline` | 3 | Pipeline error handling |
| `TestModuleLevelFunctions` | 1 | Module exports |

### Key Tests for Retry Logic

```python
# Successful retry after transient HTTP error
async def test_extract_entities_retry_on_http_error():
    """Retries twice, then succeeds on 3rd attempt."""
    # call_count == 3 after success

# Exhausts all retries
async def test_extract_entities_exhausts_retries():
    """All 3 attempts fail, raises original HTTPError."""
    # call_count == 3, raises httpx.HTTPError

# RuntimeError not retried
async def test_extract_entities_does_not_retry_runtime_error():
    """RuntimeError fails immediately (not transient)."""
    # call_count == 1
```

---

## Task 3.2: Gundam Tiling E2E Tests

### Background

Gundam Tiling is a strategy for processing high-resolution images (>1500px) by:
1. Splitting into overlapping tiles (default 1024px with 128px overlap)
2. Processing each tile with OCR independently
3. Merging results using fuzzy deduplication

### Test File

**File:** `tests/test_rag2_ocr_gundam.py` (28 tests)

| Test Class | Tests | Description |
|------------|-------|-------------|
| `TestGundamTilingConfig` | 2 | Config dataclass validation |
| `TestOCRResult` | 2 | Result metadata for Gundam mode |
| `TestGundamTilingShouldUse` | 4 | Activation threshold tests |
| `TestTileCalculation` | 5 | Tile coordinate calculation |
| `TestMergeStrategies` | 6 | Fuzzy/concat/vote merge tests |
| `TestGundamTilingE2E` | 4 | Full E2E with real images |
| `TestOCRProcessorModes` | 3 | Mode hierarchy and config |
| `TestConfidenceAggregation` | 2 | Confidence score averaging |

### Key Tests

```python
# Gundam activates for large images
async def test_gundam_activates_for_large_images():
    """2000x2000 image triggers Gundam Tiling."""
    should_tile = await ocr._should_use_gundam_tiling(large_img)
    assert should_tile

# Fuzzy merge deduplicates overlap
def test_merge_fuzzy_deduplicates():
    """Shared text between tiles appears only once."""
    overlap_count = merged_result.text.count("overlap region")
    assert overlap_count == 1

# Tile calculation respects max_tiles
def test_tile_calculation_respects_max_tiles():
    """Very large image capped at 16 tiles."""
    tiles = ocr._calculate_tiles(10000, 10000)
    assert len(tiles) <= 16
```

### Merge Strategies Tested

| Strategy | Behavior |
|----------|----------|
| `concat` | Simple concatenation with newlines |
| `fuzzy` | Fuzzy matching deduplicates overlaps (85% threshold) |
| `vote` | Confidence-weighted voting for overlaps |

---

## Task 3.3: Skip Planning Path Tests

### Background

The retrieval pipeline supports two optimization flags:
- `skip_planning=True` - Bypasses GPT-5 query planning, uses simple word split
- `skip_rerank=True` - Bypasses reranking step for faster retrieval

### Tests Added

**File:** `tests/test_rag2_retrieval.py` (10 new tests, 25 total)

| Test Class | Tests | Description |
|------------|-------|-------------|
| `TestRetrievalEdgeCases` | 7 | Skip planning/rerank, empty queries |
| `TestQueryPlanSkipPlanning` | 2 | QueryPlan from skip path |
| `TestNoResultsHandling` | 1 | Refused result for no candidates |

### Key Tests

```python
# Skip planning uses word split
async def test_retrieve_with_skip_planning():
    """Keywords = query.split(), no GPT-5 call."""
    result = await retriever.retrieve(query="test query words", skip_planning=True)
    assert result.query_plan.keywords == ["test", "query", "words"]
    assert result.timings["planning"] < 0.1  # Very fast

# Skip rerank avoids reranker
async def test_retrieve_with_skip_rerank():
    """Rerank timing minimal when skipped."""
    result = await retriever.retrieve(query="test", skip_rerank=True)
    # No significant rerank time

# Empty query handled gracefully
async def test_retrieve_empty_query():
    """Returns refused or empty, doesn't crash."""
    result = await retriever.retrieve(query="")
    assert result.refused or len(result.contexts) == 0
```

---

## Test Results Summary

### Before Phase 3
- RAG2 Tests: **115 passing**

### After Phase 3
- RAG2 Tests: **172 passing** (+57 tests)

### Test Distribution

| Test File | Tests |
|-----------|-------|
| `test_rag2_chunker.py` | 20 |
| `test_rag2_e2e.py` | 15 |
| `test_rag2_embedder.py` | 15 |
| `test_rag2_entity_e2e.py` | 19 |
| `test_rag2_graph_e2e.py` | 16 |
| `test_rag2_ingest.py` | 19 (new) |
| `test_rag2_integration.py` | 15 |
| `test_rag2_ocr_gundam.py` | 28 (new) |
| `test_rag2_retrieval.py` | 25 (+10) |
| **Total** | **172** |

---

## Files Created/Modified

### New Files
- `tests/test_rag2_ingest.py` - 19 tests for ingestion and retry logic
- `tests/test_rag2_ocr_gundam.py` - 28 tests for Gundam Tiling OCR

### Modified Files
- `src/voice_agent/rag2/ingest.py` - Added tenacity retry logic
- `tests/test_rag2_retrieval.py` - Added 10 edge case tests

---

## Dependencies

The retry logic uses **tenacity** which is already in the project dependencies:

```python
# pyproject.toml already has:
tenacity = "^8.x"
```

---

## Verification Commands

```bash
# Run all Phase 3 tests
pytest tests/test_rag2_ingest.py tests/test_rag2_ocr_gundam.py -v

# Run skip planning tests
pytest tests/test_rag2_retrieval.py::TestRetrievalEdgeCases -v

# Run all RAG2 tests
pytest tests/test_rag2*.py -v

# Result: 172 passed
```

---

## Next Steps: Phase 4 - Integration

Phase 4 will focus on:
1. Triple-hybrid retrieval integration test
2. End-to-end documentation

---

## Conclusion

Phase 3 successfully added:
- ✅ Robust retry logic for entity extraction (3 attempts, exponential backoff)
- ✅ Comprehensive Gundam Tiling tests (28 tests covering all strategies)
- ✅ Edge case tests for skip_planning and skip_rerank paths

The RAG 2.0 system now has **172 passing tests** covering robustness and edge cases.
