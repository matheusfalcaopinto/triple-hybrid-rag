# Phase 3: Robustness & Edge Cases

> **Priority:** MEDIUM - Production resilience  
> **Time:** 1 hour 15 minutes  
> **Status:** [ ] Not Started  
> **Depends On:** Phase 1 (PuppyGraph), Phase 2 (Module Validation)

---

## Objective

Add robustness features (retry logic) and test edge cases (Gundam Tiling, skip planning) to ensure production reliability.

---

## Task 3.1: Add Retry Logic to Entity Extraction

**Time:** 30 minutes  
**File:** `src/voice_agent/rag2/ingest.py`

### Current State
- ✅ Entity extraction integrated into ingestion
- ❌ No retry logic for GPT-5 API failures
- ❌ Single failure stops entire ingestion

### Implementation

Add retry decorator to `_extract_entities` method:

```python
# In src/voice_agent/rag2/ingest.py

# Add import at top
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
import httpx

# Modify _extract_entities method
class RAG2Ingestor:
    # ... existing code ...
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.HTTPError, TimeoutError, ConnectionError)),
        before_sleep=lambda retry_state: logger.warning(
            f"Entity extraction attempt {retry_state.attempt_number} failed, retrying..."
        ),
    )
    async def _extract_entities_with_retry(
        self,
        parent_id: str,
        parent_text: str,
        document_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Extract entities with retry logic."""
        try:
            result = await self.entity_extractor.extract(parent_text)
            
            if result.entities:
                stored = await self.entity_store.store_entities(
                    entities=result.entities,
                    relations=result.relations,
                    org_id=self.org_id,
                    document_id=document_id,
                    parent_chunk_id=parent_id,
                )
                return stored
            return None
            
        except Exception as e:
            logger.error(f"Entity extraction failed for parent {parent_id}: {e}")
            raise  # Let tenacity handle retry
    
    async def _extract_entities(
        self,
        stored_parents: Dict[str, Any],
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Extract entities from all parent chunks with retry logic."""
        if not self.entity_extraction_enabled:
            return {"entities": 0, "relations": 0}
        
        total_entities = 0
        total_relations = 0
        failed_parents = []
        
        for parent_id, parent_info in stored_parents.items():
            try:
                result = await self._extract_entities_with_retry(
                    parent_id=parent_id,
                    parent_text=parent_info["text"],
                    document_id=parent_info["document_id"],
                )
                
                if result:
                    total_entities += len(result.get("entity_ids", []))
                    total_relations += len(result.get("relation_ids", []))
                    
            except Exception as e:
                # All retries failed, log and continue
                logger.error(f"Entity extraction failed after retries for {parent_id}: {e}")
                failed_parents.append(parent_id)
        
        if failed_parents:
            logger.warning(f"Entity extraction failed for {len(failed_parents)} parents")
        
        return {
            "entities": total_entities,
            "relations": total_relations,
            "failed_parents": failed_parents,
        }
```

### Test

```python
# Add to tests/test_rag2_ingest.py

async def test_entity_extraction_retry_on_failure():
    """Test that entity extraction retries on transient failures."""
    from unittest.mock import AsyncMock, patch
    
    ingestor = RAG2Ingestor(
        org_id="test-org",
        entity_extraction_enabled=True,
    )
    
    # Mock extractor to fail twice then succeed
    call_count = 0
    async def mock_extract(text):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise httpx.HTTPError("Transient error")
        return ExtractionResult(entities=[], relations=[])
    
    with patch.object(ingestor.entity_extractor, 'extract', side_effect=mock_extract):
        result = await ingestor._extract_entities_with_retry(
            parent_id="test",
            parent_text="Test text",
            document_id="test-doc",
        )
    
    assert call_count == 3  # Retried twice
```

### Verification

```bash
pytest tests/test_rag2_ingest.py::test_entity_extraction_retry_on_failure -v
```

### Success Criteria
- [ ] Retry decorator added with 3 attempts
- [ ] Exponential backoff (2s, 4s, 8s)
- [ ] Failed parents logged but don't stop pipeline
- [ ] Test passes

---

## Task 3.2: Gundam Tiling E2E Test

**Time:** 30 minutes  
**File:** `tests/test_rag2_ocr_gundam.py` (new)

### Current State
- ✅ Gundam Tiling implemented in `ocr.py`
- ✅ Config dataclass exists
- ❌ No E2E test with actual large image
- ❌ Merge strategies not tested

### Implementation

```python
# tests/test_rag2_ocr_gundam.py
"""
E2E tests for Gundam Tiling OCR strategy.
"""

import io
import pytest
from PIL import Image, ImageDraw, ImageFont

from voice_agent.ingestion.ocr import (
    OCRProcessor,
    GundamTilingConfig,
    OCRResult,
)


class TestGundamTilingE2E:
    """E2E tests for Gundam Tiling OCR."""
    
    def create_large_test_image(self, width: int, height: int, text: str = "Test") -> bytes:
        """Create a test image with text."""
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)
        
        # Draw text in multiple positions
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        # Add text at various positions
        positions = [
            (50, 50), (50, 200), (50, 400), (50, 600),
            (500, 50), (500, 200), (500, 400), (500, 600),
            (1000, 50), (1000, 200), (1000, 400), (1000, 600),
        ]
        
        for i, (x, y) in enumerate(positions):
            if x < width and y < height:
                draw.text((x, y), f"{text} Block {i+1}", fill='black', font=font)
        
        # Save to bytes
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        return buf.getvalue()
    
    def test_gundam_tiling_activates_for_large_images(self):
        """Test that Gundam Tiling activates for images > min_image_size."""
        config = GundamTilingConfig(
            enabled=True,
            min_image_size=1500,
            tile_size=1024,
            overlap=128,
        )
        
        ocr = OCRProcessor(mode="base", gundam_config=config)
        
        # Create 2000x2000 image (should trigger tiling)
        large_img = self.create_large_test_image(2000, 2000)
        
        # Check if should use tiling
        import asyncio
        should_tile = asyncio.run(ocr._should_use_gundam_tiling(large_img))
        
        assert should_tile, "Should use Gundam Tiling for 2000x2000 image"
    
    def test_gundam_tiling_skips_small_images(self):
        """Test that Gundam Tiling skips small images."""
        config = GundamTilingConfig(
            enabled=True,
            min_image_size=1500,
        )
        
        ocr = OCRProcessor(mode="base", gundam_config=config)
        
        # Create 800x800 image (should NOT trigger tiling)
        small_img = self.create_large_test_image(800, 800)
        
        import asyncio
        should_tile = asyncio.run(ocr._should_use_gundam_tiling(small_img))
        
        assert not should_tile, "Should NOT use Gundam Tiling for 800x800 image"
    
    def test_tile_calculation(self):
        """Test tile coordinate calculation."""
        config = GundamTilingConfig(
            tile_size=1024,
            overlap=128,
            max_tiles=16,
        )
        
        ocr = OCRProcessor(gundam_config=config)
        
        # 2048x2048 should produce 4 tiles (2x2 grid)
        tiles = ocr._calculate_tiles(2048, 2048)
        
        assert len(tiles) >= 4, f"Expected at least 4 tiles, got {len(tiles)}"
        assert len(tiles) <= config.max_tiles, f"Should not exceed max_tiles"
        
        # Verify tiles have correct dimensions
        for x1, y1, x2, y2 in tiles:
            assert x2 - x1 <= config.tile_size
            assert y2 - y1 <= config.tile_size
            assert x1 >= 0 and y1 >= 0
    
    def test_merge_fuzzy_deduplication(self):
        """Test fuzzy merge removes duplicate lines."""
        config = GundamTilingConfig(
            merge_strategy="fuzzy",
            fuzzy_threshold=0.85,
        )
        
        ocr = OCRProcessor(gundam_config=config)
        
        # Simulate OCR results with overlapping text
        from voice_agent.ingestion.ocr import OCRResult
        
        results = [
            OCRResult(
                text="Line 1 from tile A\nLine 2 from tile A\nLine 3 overlap",
                confidence=0.9,
                has_tables=False,
                tables=[],
                mode_used="large",
            ),
            OCRResult(
                text="Line 3 overlap\nLine 4 from tile B\nLine 5 from tile B",
                confidence=0.85,
                has_tables=False,
                tables=[],
                mode_used="large",
            ),
        ]
        
        tiles = [(0, 0, 1024, 1024), (900, 0, 1924, 1024)]  # Overlapping
        
        merged_result = ocr._merge_tile_results(results, tiles, 1924, 1024)
        
        # "Line 3 overlap" should appear only once
        line_3_count = merged_result.text.count("Line 3")
        assert line_3_count == 1, f"Overlap line should be deduplicated, found {line_3_count} times"
    
    def test_merge_concat_strategy(self):
        """Test concat merge strategy."""
        config = GundamTilingConfig(merge_strategy="concat")
        
        ocr = OCRProcessor(gundam_config=config)
        
        results = [
            OCRResult(text="Tile 1 text", confidence=0.9, has_tables=False, tables=[], mode_used="large"),
            OCRResult(text="Tile 2 text", confidence=0.9, has_tables=False, tables=[], mode_used="large"),
        ]
        
        merged = ocr._merge_concat(results)
        
        assert "Tile 1 text" in merged
        assert "Tile 2 text" in merged
    
    def test_ocr_result_includes_tile_metadata(self):
        """Test that OCR result includes Gundam Tiling metadata."""
        config = GundamTilingConfig(enabled=True)
        
        ocr = OCRProcessor(gundam_config=config)
        
        # Create mock result as if from Gundam Tiling
        from voice_agent.ingestion.ocr import OCRResult
        
        result = OCRResult(
            text="Merged text",
            confidence=0.9,
            has_tables=False,
            tables=[],
            mode_used="gundam",
            tiles_processed=4,
            tile_confidences=[0.9, 0.85, 0.92, 0.88],
            metadata={
                "gundam_tiles": 4,
                "image_size": "2048x2048",
                "merge_strategy": "fuzzy",
            },
        )
        
        assert result.tiles_processed == 4
        assert len(result.tile_confidences) == 4
        assert result.metadata["gundam_tiles"] == 4
```

### Verification

```bash
pytest tests/test_rag2_ocr_gundam.py -v
```

### Success Criteria
- [ ] Tiling activates for large images
- [ ] Tiling skips small images
- [ ] Tile calculation produces correct coordinates
- [ ] Fuzzy merge deduplicates overlapping text
- [ ] Concat merge works
- [ ] Result includes tile metadata

---

## Task 3.3: Skip Planning Path Test

**Time:** 15 minutes  
**File:** `tests/test_rag2_retrieval.py` (add test)

### Current State
- ✅ `skip_planning=True` parameter exists
- ❌ No dedicated test for this path

### Implementation

Add to existing test file:

```python
# Add to tests/test_rag2_retrieval.py

class TestRetrievalEdgeCases:
    """Edge case tests for retrieval."""
    
    async def test_retrieve_with_skip_planning(self):
        """Test retrieval with query planning skipped."""
        retriever = RAG2Retriever(org_id="test-org")
        
        # Mock the search methods to avoid DB calls
        with patch.object(retriever, '_lexical_search', return_value=[]):
            with patch.object(retriever, '_semantic_search', return_value=[]):
                result = await retriever.retrieve(
                    query="test query words",
                    skip_planning=True,
                )
        
        # When skipping planning, keywords should be simple word split
        assert result.query_plan is not None
        assert result.query_plan.keywords == ["test", "query", "words"]
        assert result.query_plan.semantic_query_text == "test query words"
        assert result.query_plan.cypher_query is None
        
        # Should still have timing for "planning" (even if minimal)
        assert "planning" in result.timings
    
    async def test_retrieve_with_skip_rerank(self):
        """Test retrieval with reranking skipped."""
        retriever = RAG2Retriever(org_id="test-org")
        
        # Create mock candidates
        mock_candidates = [
            {"child_id": "c1", "parent_id": "p1", "document_id": "d1", "text": "Result 1", "page": 1},
        ]
        
        with patch.object(retriever, '_lexical_search', return_value=[]):
            with patch.object(retriever, '_semantic_search', return_value=mock_candidates):
                with patch.object(retriever, '_expand_to_parents', return_value=[]):
                    result = await retriever.retrieve(
                        query="test",
                        skip_rerank=True,
                    )
        
        # Should not have rerank timing
        assert "rerank" not in result.timings or result.timings.get("rerank", 0) < 0.001
    
    async def test_retrieve_empty_query(self):
        """Test retrieval with empty query."""
        retriever = RAG2Retriever(org_id="test-org")
        
        result = await retriever.retrieve(query="")
        
        # Should handle gracefully (refuse or return empty)
        assert result.refused or len(result.contexts) == 0
```

### Verification

```bash
pytest tests/test_rag2_retrieval.py::TestRetrievalEdgeCases -v
```

### Success Criteria
- [ ] skip_planning uses word split for keywords
- [ ] skip_rerank doesn't call reranker
- [ ] Empty query handled gracefully

---

## Verification Checklist

```
[ ] Task 3.1: Retry Logic
    [ ] tenacity import added
    [ ] Retry decorator on _extract_entities_with_retry
    [ ] Failed parents logged, don't stop pipeline
    [ ] Test passes

[ ] Task 3.2: Gundam Tiling E2E
    [ ] Test file created
    [ ] Tiling activation tested
    [ ] Merge strategies tested
    [ ] Metadata included in results

[ ] Task 3.3: Skip Planning Test
    [ ] Test added to existing file
    [ ] Skip planning path tested
    [ ] Skip rerank path tested
    [ ] Empty query tested
```

---

## Dependencies to Install

```bash
pip install tenacity Pillow
```

(Pillow may already be installed for OCR processing)
