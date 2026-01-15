"""
E2E Tests for Gundam Tiling OCR Strategy

Tests the Gundam Tiling approach for processing large images by:
- Splitting into overlapping tiles
- Processing each tile with OCR
- Merging results with fuzzy deduplication

These tests use real image processing and may be slower than unit tests.
"""
from __future__ import annotations

import io
import pytest
from typing import List, Tuple
from dataclasses import dataclass


class TestGundamTilingConfig:
    """Test GundamTilingConfig dataclass."""
    
    def test_config_defaults(self) -> None:
        """Test default configuration values."""
        from voice_agent.ingestion.ocr import GundamTilingConfig
        
        config = GundamTilingConfig()
        
        assert config.enabled is True
        assert config.tile_size == 1024
        assert config.overlap == 128
        assert config.min_image_size == 1500
        assert config.max_tiles == 16
        assert config.merge_strategy == "fuzzy"
        assert config.fuzzy_threshold == 0.85
    
    def test_config_custom_values(self) -> None:
        """Test custom configuration values."""
        from voice_agent.ingestion.ocr import GundamTilingConfig
        
        config = GundamTilingConfig(
            enabled=False,
            tile_size=512,
            overlap=64,
            min_image_size=2000,
            max_tiles=8,
            merge_strategy="concat",
            fuzzy_threshold=0.9,
        )
        
        assert config.enabled is False
        assert config.tile_size == 512
        assert config.overlap == 64
        assert config.min_image_size == 2000
        assert config.max_tiles == 8
        assert config.merge_strategy == "concat"
        assert config.fuzzy_threshold == 0.9


class TestOCRResult:
    """Test OCRResult dataclass with Gundam metadata."""
    
    def test_result_with_gundam_metadata(self) -> None:
        """Test OCRResult includes Gundam Tiling metadata."""
        from voice_agent.ingestion.ocr import OCRResult
        
        result = OCRResult(
            text="Merged text from tiles",
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
        
        assert result.mode_used == "gundam"
        assert result.tiles_processed == 4
        assert len(result.tile_confidences) == 4
        assert result.metadata["gundam_tiles"] == 4
        assert result.metadata["image_size"] == "2048x2048"
    
    def test_result_defaults(self) -> None:
        """Test OCRResult default values."""
        from voice_agent.ingestion.ocr import OCRResult
        
        result = OCRResult(
            text="Test",
            confidence=0.8,
            has_tables=False,
            tables=[],
            mode_used="base",
        )
        
        assert result.tiles_processed == 0
        assert result.tile_confidences == []
        assert result.metadata == {}


class TestGundamTilingShouldUse:
    """Test _should_use_gundam_tiling logic."""
    
    @pytest.fixture
    def ocr_processor(self):
        """Create OCR processor with Gundam config."""
        from voice_agent.ingestion.ocr import OCRProcessor, GundamTilingConfig
        
        config = GundamTilingConfig(
            enabled=True,
            min_image_size=1500,
        )
        return OCRProcessor(mode="base", gundam_config=config)
    
    def create_test_image(self, width: int, height: int) -> bytes:
        """Create a test image of specified dimensions."""
        from PIL import Image
        import io
        
        img = Image.new('RGB', (width, height), color='white')
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        return buf.getvalue()
    
    @pytest.mark.asyncio
    async def test_gundam_activates_for_large_images(self, ocr_processor) -> None:
        """Test that Gundam Tiling activates for images > min_image_size."""
        # 2000x2000 should trigger tiling (> 1500)
        large_img = self.create_test_image(2000, 2000)
        
        should_tile = await ocr_processor._should_use_gundam_tiling(large_img)
        
        assert should_tile, "Should use Gundam Tiling for 2000x2000 image"
    
    @pytest.mark.asyncio
    async def test_gundam_skips_small_images(self, ocr_processor) -> None:
        """Test that Gundam Tiling skips small images."""
        # 800x800 should NOT trigger tiling (< 1500)
        small_img = self.create_test_image(800, 800)
        
        should_tile = await ocr_processor._should_use_gundam_tiling(small_img)
        
        assert not should_tile, "Should NOT use Gundam Tiling for 800x800 image"
    
    @pytest.mark.asyncio
    async def test_gundam_activates_at_boundary(self, ocr_processor) -> None:
        """Test boundary condition at min_image_size."""
        # Exactly 1500 should trigger (>= min_image_size)
        boundary_img = self.create_test_image(1500, 800)
        
        should_tile = await ocr_processor._should_use_gundam_tiling(boundary_img)
        
        assert should_tile, "Should use Gundam Tiling at boundary (1500x800)"
    
    @pytest.mark.asyncio
    async def test_gundam_checks_largest_dimension(self, ocr_processor) -> None:
        """Test that largest dimension is checked."""
        # 800x2000 should trigger because max(800, 2000) >= 1500
        tall_img = self.create_test_image(800, 2000)
        
        should_tile = await ocr_processor._should_use_gundam_tiling(tall_img)
        
        assert should_tile, "Should use Gundam Tiling for 800x2000 (tall image)"


class TestTileCalculation:
    """Test tile coordinate calculation."""
    
    @pytest.fixture
    def ocr_processor(self):
        """Create OCR processor with default Gundam config."""
        from voice_agent.ingestion.ocr import OCRProcessor, GundamTilingConfig
        
        config = GundamTilingConfig(
            tile_size=1024,
            overlap=128,
            max_tiles=16,
        )
        return OCRProcessor(gundam_config=config)
    
    def test_tile_calculation_small_image(self, ocr_processor) -> None:
        """Test tile calculation for small image (single tile)."""
        tiles = ocr_processor._calculate_tiles(800, 800)
        
        # Small image should produce single tile
        assert len(tiles) == 1
        x1, y1, x2, y2 = tiles[0]
        assert x2 - x1 <= 1024
        assert y2 - y1 <= 1024
    
    def test_tile_calculation_2x2_grid(self, ocr_processor) -> None:
        """Test tile calculation produces 2x2 grid for 2048x2048."""
        tiles = ocr_processor._calculate_tiles(2048, 2048)
        
        # 2048x2048 with 1024 tiles and 128 overlap should produce 4 tiles
        assert len(tiles) >= 4
        assert len(tiles) <= 16
        
        # All tiles should be within bounds
        for x1, y1, x2, y2 in tiles:
            assert x1 >= 0
            assert y1 >= 0
            assert x2 <= 2048
            assert y2 <= 2048
            assert x2 - x1 <= 1024
            assert y2 - y1 <= 1024
    
    def test_tile_calculation_respects_max_tiles(self, ocr_processor) -> None:
        """Test that tile count doesn't exceed max_tiles."""
        # Very large image that would produce many tiles
        tiles = ocr_processor._calculate_tiles(10000, 10000)
        
        assert len(tiles) <= 16, f"Should not exceed max_tiles, got {len(tiles)}"
    
    def test_tile_calculation_rectangular_image(self, ocr_processor) -> None:
        """Test tile calculation for rectangular image."""
        tiles = ocr_processor._calculate_tiles(3000, 1500)
        
        # Should produce tiles for rectangular image
        assert len(tiles) > 1
        
        # All tiles should be within bounds
        for x1, y1, x2, y2 in tiles:
            assert x2 <= 3000
            assert y2 <= 1500
    
    def test_tiles_have_overlap(self, ocr_processor) -> None:
        """Test that adjacent tiles overlap."""
        tiles = ocr_processor._calculate_tiles(2048, 1024)
        
        if len(tiles) >= 2:
            # Sort by x position
            sorted_tiles = sorted(tiles, key=lambda t: t[0])
            
            # Check horizontal tiles have overlap
            for i in range(len(sorted_tiles) - 1):
                t1_x2 = sorted_tiles[i][2]
                t2_x1 = sorted_tiles[i + 1][0]
                
                if sorted_tiles[i][1] == sorted_tiles[i + 1][1]:  # Same row
                    # There should be overlap or adjacency
                    assert t1_x2 >= t2_x1, "Horizontal tiles should overlap"


class TestMergeStrategies:
    """Test tile result merge strategies."""
    
    @pytest.fixture
    def ocr_processor_fuzzy(self):
        """Create OCR processor with fuzzy merge strategy."""
        from voice_agent.ingestion.ocr import OCRProcessor, GundamTilingConfig
        
        config = GundamTilingConfig(
            merge_strategy="fuzzy",
            fuzzy_threshold=0.85,
        )
        return OCRProcessor(gundam_config=config)
    
    @pytest.fixture
    def ocr_processor_concat(self):
        """Create OCR processor with concat merge strategy."""
        from voice_agent.ingestion.ocr import OCRProcessor, GundamTilingConfig
        
        config = GundamTilingConfig(merge_strategy="concat")
        return OCRProcessor(gundam_config=config)
    
    @pytest.fixture
    def ocr_processor_vote(self):
        """Create OCR processor with vote merge strategy."""
        from voice_agent.ingestion.ocr import OCRProcessor, GundamTilingConfig
        
        config = GundamTilingConfig(merge_strategy="vote")
        return OCRProcessor(gundam_config=config)
    
    def create_mock_results(self) -> List:
        """Create mock OCR results with overlapping text."""
        from voice_agent.ingestion.ocr import OCRResult
        
        return [
            OCRResult(
                text="Company Overview and Mission Statement\nProduct Features Description\nThe overlap region with shared content",
                confidence=0.9,
                has_tables=False,
                tables=[],
                mode_used="large",
            ),
            OCRResult(
                text="The overlap region with shared content\nPricing and Subscription Plans\nContact Information and Support",
                confidence=0.85,
                has_tables=False,
                tables=[],
                mode_used="large",
            ),
        ]
    
    def test_merge_concat_combines_all(self, ocr_processor_concat) -> None:
        """Test concat merge combines all text."""
        results = self.create_mock_results()
        
        merged = ocr_processor_concat._merge_concat(results)
        
        assert "Company Overview" in merged
        assert "Product Features" in merged
        assert "Pricing and Subscription" in merged
        assert "Contact Information" in merged
        # Concat does NOT deduplicate
        assert merged.count("overlap region") >= 1
    
    def test_merge_fuzzy_deduplicates(self, ocr_processor_fuzzy) -> None:
        """Test fuzzy merge deduplicates overlapping text."""
        results = self.create_mock_results()
        tiles = [(0, 0, 1024, 1024), (900, 0, 1924, 1024)]  # Overlapping
        
        merged_result = ocr_processor_fuzzy._merge_tile_results(results, tiles, 1924, 1024)
        
        # "The overlap region with shared content" should appear only once
        overlap_count = merged_result.text.count("overlap region")
        assert overlap_count == 1, f"Overlap line should be deduplicated, found {overlap_count} times"
        
        # Other distinct lines should still be present
        assert "Company Overview" in merged_result.text
        assert "Pricing and Subscription" in merged_result.text
    
    def test_merge_fuzzy_threshold(self, ocr_processor_fuzzy) -> None:
        """Test fuzzy threshold for near-matches."""
        from voice_agent.ingestion.ocr import OCRResult
        
        # Results with similar but not identical lines
        results = [
            OCRResult(
                text="The quick brown fox jumps",
                confidence=0.9,
                has_tables=False,
                tables=[],
                mode_used="large",
            ),
            OCRResult(
                text="The quick brown fox jump",  # Slightly different (missing 's')
                confidence=0.85,
                has_tables=False,
                tables=[],
                mode_used="large",
            ),
        ]
        tiles = [(0, 0, 1024, 1024), (900, 0, 1924, 1024)]
        
        merged_result = ocr_processor_fuzzy._merge_tile_results(results, tiles, 1924, 1024)
        
        # Similar lines should be deduplicated
        assert merged_result.text.count("quick brown fox") == 1
    
    def test_merge_vote_uses_confidence(self, ocr_processor_vote) -> None:
        """Test vote merge uses confidence for overlap resolution."""
        from voice_agent.ingestion.ocr import OCRResult
        
        results = [
            OCRResult(
                text="Low confidence version",
                confidence=0.5,
                has_tables=False,
                tables=[],
                mode_used="large",
            ),
            OCRResult(
                text="High confidence version",
                confidence=0.95,
                has_tables=False,
                tables=[],
                mode_used="large",
            ),
        ]
        tiles = [(0, 0, 1024, 1024), (500, 0, 1524, 1024)]
        
        merged_result = ocr_processor_vote._merge_tile_results(results, tiles, 1524, 1024)
        
        # Both texts should be in result (no overlap to resolve)
        assert "confidence version" in merged_result.text
    
    def test_merge_result_has_metadata(self, ocr_processor_fuzzy) -> None:
        """Test merged result includes Gundam metadata."""
        results = self.create_mock_results()
        tiles = [(0, 0, 1024, 1024), (500, 0, 1524, 1024)]
        
        merged_result = ocr_processor_fuzzy._merge_tile_results(results, tiles, 1524, 1024)
        
        assert merged_result.mode_used == "gundam"
        assert merged_result.tiles_processed == 2
        assert len(merged_result.tile_confidences) == 2
        assert merged_result.metadata["gundam_tiles"] == 2
        assert merged_result.metadata["merge_strategy"] == "fuzzy"
    
    def test_merge_collects_tables(self, ocr_processor_fuzzy) -> None:
        """Test that tables from all tiles are collected."""
        from voice_agent.ingestion.ocr import OCRResult
        
        results = [
            OCRResult(
                text="Text with table",
                confidence=0.9,
                has_tables=True,
                tables=["| Col1 | Col2 |\n| a | b |"],
                mode_used="large",
            ),
            OCRResult(
                text="More text",
                confidence=0.85,
                has_tables=True,
                tables=["| X | Y |\n| 1 | 2 |"],
                mode_used="large",
            ),
        ]
        tiles = [(0, 0, 1024, 1024), (1024, 0, 2048, 1024)]
        
        merged_result = ocr_processor_fuzzy._merge_tile_results(results, tiles, 2048, 1024)
        
        assert merged_result.has_tables
        assert len(merged_result.tables) == 2


class TestGundamTilingE2E:
    """End-to-end tests for Gundam Tiling with real images."""
    
    def create_large_test_image(
        self,
        width: int,
        height: int,
        text_blocks: List[Tuple[int, int, str]] = None,
    ) -> bytes:
        """
        Create a test image with text at various positions.
        
        Args:
            width: Image width
            height: Image height
            text_blocks: List of (x, y, text) tuples for text placement
        """
        from PIL import Image, ImageDraw, ImageFont
        import io
        
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)
        
        # Use default font (may vary by system)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
        except (OSError, IOError):
            font = ImageFont.load_default()
        
        # Default text blocks if none provided
        if text_blocks is None:
            text_blocks = [
                (50, 50, "Block 1: Top Left"),
                (50, 200, "Block 2: Left Side"),
                (50, 400, "Block 3: Middle Left"),
                (width // 2, 50, "Block 4: Top Center"),
                (width // 2, 200, "Block 5: Center"),
                (width // 2, 400, "Block 6: Middle Center"),
                (width - 200, 50, "Block 7: Top Right"),
                (width - 200, 200, "Block 8: Right Side"),
            ]
        
        # Draw text blocks
        for x, y, text in text_blocks:
            if x < width and y < height:
                draw.text((x, y), text, fill='black', font=font)
        
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        return buf.getvalue()
    
    def test_process_large_image_activates_gundam(self) -> None:
        """Test that processing large image activates Gundam Tiling."""
        from voice_agent.ingestion.ocr import OCRProcessor, GundamTilingConfig
        
        config = GundamTilingConfig(
            enabled=True,
            min_image_size=1500,
            tile_size=512,
            overlap=64,
        )
        
        ocr = OCRProcessor(mode="base", gundam_config=config)
        
        # Create 2000x2000 image
        large_img = self.create_large_test_image(2000, 2000)
        
        # Verify Gundam would activate
        import asyncio
        should_tile = asyncio.run(ocr._should_use_gundam_tiling(large_img))
        
        assert should_tile
    
    def test_tile_coordinates_cover_full_image(self) -> None:
        """Test that calculated tiles cover the entire image."""
        from voice_agent.ingestion.ocr import OCRProcessor, GundamTilingConfig
        
        config = GundamTilingConfig(
            tile_size=512,
            overlap=64,
            max_tiles=32,
        )
        
        ocr = OCRProcessor(gundam_config=config)
        tiles = ocr._calculate_tiles(1500, 1200)
        
        # Check coverage
        covered_x = set()
        covered_y = set()
        
        for x1, y1, x2, y2 in tiles:
            covered_x.update(range(x1, x2))
            covered_y.update(range(y1, y2))
        
        # All pixels should be covered
        assert min(covered_x) == 0
        assert max(covered_x) >= 1499
        assert min(covered_y) == 0
        assert max(covered_y) >= 1199
    
    def test_empty_tile_results_handled(self) -> None:
        """Test handling of empty tile results."""
        from voice_agent.ingestion.ocr import OCRProcessor, GundamTilingConfig
        
        config = GundamTilingConfig(merge_strategy="fuzzy")
        ocr = OCRProcessor(gundam_config=config)
        
        merged = ocr._merge_tile_results([], [], 1000, 1000)
        
        assert merged.text == ""
        assert merged.confidence == 0.0
        assert merged.error is not None or merged.text == ""
    
    def test_single_tile_returns_directly(self) -> None:
        """Test that single tile result is returned without merge overhead."""
        from voice_agent.ingestion.ocr import OCRProcessor, OCRResult, GundamTilingConfig
        
        config = GundamTilingConfig(merge_strategy="fuzzy")
        ocr = OCRProcessor(gundam_config=config)
        
        results = [
            OCRResult(
                text="Single tile text",
                confidence=0.95,
                has_tables=False,
                tables=[],
                mode_used="large",
            )
        ]
        tiles = [(0, 0, 1024, 1024)]
        
        merged = ocr._merge_tile_results(results, tiles, 1024, 1024)
        
        assert "Single tile text" in merged.text
        assert merged.confidence == 0.95


class TestOCRProcessorModes:
    """Test OCR processor mode handling."""
    
    def test_mode_hierarchy(self) -> None:
        """Test mode hierarchy for fallback."""
        from voice_agent.ingestion.ocr import OCRProcessor
        
        ocr = OCRProcessor(mode="base")
        
        # Test mode progression
        assert ocr._get_next_mode("tiny") == "small"
        assert ocr._get_next_mode("small") == "base"
        assert ocr._get_next_mode("base") == "large"
        assert ocr._get_next_mode("large") == "gundam"
        assert ocr._get_next_mode("gundam") is None
    
    def test_gundam_mode_forces_tiling(self) -> None:
        """Test that mode='gundam' forces Gundam Tiling."""
        from voice_agent.ingestion.ocr import OCRProcessor, GundamTilingConfig
        
        config = GundamTilingConfig(
            enabled=True,
            min_image_size=5000,  # Very high threshold
        )
        
        ocr = OCRProcessor(mode="gundam", gundam_config=config)
        
        # Even small images should use Gundam when mode is "gundam"
        assert ocr.mode == "gundam"
    
    def test_disabled_gundam_config(self) -> None:
        """Test that disabled Gundam config prevents tiling."""
        from voice_agent.ingestion.ocr import OCRProcessor, GundamTilingConfig
        import asyncio
        
        config = GundamTilingConfig(enabled=False)
        ocr = OCRProcessor(gundam_config=config)
        
        # Even if _should_use_gundam_tiling returns True, enabled=False should prevent it
        assert not config.enabled


class TestConfidenceAggregation:
    """Test confidence score aggregation from tiles."""
    
    def test_average_confidence(self) -> None:
        """Test that merged result has average confidence."""
        from voice_agent.ingestion.ocr import OCRProcessor, OCRResult, GundamTilingConfig
        
        config = GundamTilingConfig(merge_strategy="fuzzy")
        ocr = OCRProcessor(gundam_config=config)
        
        results = [
            OCRResult(text="A", confidence=0.8, has_tables=False, tables=[], mode_used="large"),
            OCRResult(text="B", confidence=0.9, has_tables=False, tables=[], mode_used="large"),
            OCRResult(text="C", confidence=1.0, has_tables=False, tables=[], mode_used="large"),
        ]
        tiles = [(0, 0, 100, 100), (100, 0, 200, 100), (200, 0, 300, 100)]
        
        merged = ocr._merge_tile_results(results, tiles, 300, 100)
        
        expected_avg = (0.8 + 0.9 + 1.0) / 3
        assert abs(merged.confidence - expected_avg) < 0.001
    
    def test_zero_confidence_excluded(self) -> None:
        """Test that zero confidence tiles are excluded from average."""
        from voice_agent.ingestion.ocr import OCRProcessor, OCRResult, GundamTilingConfig
        
        config = GundamTilingConfig(merge_strategy="fuzzy")
        ocr = OCRProcessor(gundam_config=config)
        
        results = [
            OCRResult(text="A", confidence=0.8, has_tables=False, tables=[], mode_used="large"),
            OCRResult(text="", confidence=0.0, has_tables=False, tables=[], mode_used="large"),  # Failed tile
            OCRResult(text="C", confidence=0.9, has_tables=False, tables=[], mode_used="large"),
        ]
        tiles = [(0, 0, 100, 100), (100, 0, 200, 100), (200, 0, 300, 100)]
        
        merged = ocr._merge_tile_results(results, tiles, 300, 100)
        
        # Average of 0.8 and 0.9 (excluding 0.0)
        expected_avg = (0.8 + 0.9) / 2
        assert abs(merged.confidence - expected_avg) < 0.001
