"""
OCR Processing Module

Handles optical character recognition for scanned documents and images:
- Qwen3-VL Vision API integration (OpenAI-compatible)
- DeepSeek OCR integration (legacy fallback)
- **Gundam Tiling**: Overlapping tile strategy for high-resolution documents
- Confidence tracking
- Retry logic with fallback modes
- Table detection and preservation
"""

import asyncio
import base64
import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import httpx

from voice_agent.config import SETTINGS
from voice_agent.observability.rag_metrics import rag_metrics

# Default OCR prompt - kept simple for compatibility with DeepSeek-OCR
OCR_PROMPT = """OCR this image. Extract all text exactly as it appears. For tables, use Markdown format. Output only the extracted text."""

logger = logging.getLogger(__name__)


class OCRMode(Enum):
    """OCR processing modes (quality vs speed/cost tradeoff)."""
    TINY = "tiny"
    SMALL = "small"
    BASE = "base"
    LARGE = "large"
    GUNDAM = "gundam"  # Highest quality - uses Gundam Tiling


@dataclass
class GundamTilingConfig:
    """
    Configuration for Gundam Tiling OCR strategy.
    
    Gundam Tiling splits large images into overlapping tiles,
    processes each tile separately, then intelligently merges
    the results using fuzzy matching to handle overlapping text.
    
    This dramatically improves OCR accuracy for:
    - High-resolution scanned documents (>2000px)
    - Dense text layouts
    - Tables and structured data
    - Multi-column documents
    """
    enabled: bool = True
    tile_size: int = 1024  # Tile dimensions in pixels
    overlap: int = 128  # Overlap between adjacent tiles (px)
    min_image_size: int = 1500  # Only tile images larger than this
    max_tiles: int = 16  # Maximum tiles per image to prevent explosion
    merge_strategy: str = "fuzzy"  # "fuzzy", "concat", or "vote"
    fuzzy_threshold: float = 0.85  # Similarity threshold for fuzzy merge


@dataclass
class OCRResult:
    """Result of OCR processing."""
    text: str
    confidence: float  # 0.0 to 1.0
    has_tables: bool
    tables: List[str]  # Markdown tables extracted
    mode_used: str
    error: Optional[str] = None
    retry_count: int = 0
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
    # Gundam Tiling metadata
    tiles_processed: int = 0
    tile_confidences: List[float] = field(default_factory=list)
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class OCRProcessor:
    """
    Process images and scanned documents with OCR.
    
    Uses DeepSeek OCR with configurable modes and retry logic.
    Falls back to higher quality modes if confidence is low.
    
    Gundam Mode:
        When mode="gundam" or for images > min_image_size pixels,
        uses Gundam Tiling to split large images into overlapping tiles,
        process each tile, and merge results with fuzzy matching.
    """
    
    def __init__(
        self,
        mode: str = "base",
        confidence_threshold: float = 0.6,
        retry_limit: int = 2,
        endpoint: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 60.0,
        gundam_config: Optional[GundamTilingConfig] = None,
    ):
        """
        Initialize OCR processor.
        
        Args:
            mode: OCR mode (tiny, small, base, large, gundam)
            confidence_threshold: Minimum confidence to accept result
            retry_limit: Maximum retry attempts with higher modes
            endpoint: DeepSeek OCR endpoint URL
            model: DeepSeek OCR model name
            timeout: Request timeout in seconds
            gundam_config: Configuration for Gundam Tiling strategy
        """
        self.mode = mode or SETTINGS.rag_ocr_mode
        self.confidence_threshold = confidence_threshold or SETTINGS.rag_ocr_confidence_threshold
        self.retry_limit = retry_limit or SETTINGS.rag_ocr_retry_limit
        self.endpoint = endpoint or SETTINGS.rag_ocr_endpoint
        self.model = model or SETTINGS.rag_ocr_model
        self.timeout = timeout
        
        # Gundam Tiling configuration
        self.gundam_config = gundam_config or GundamTilingConfig()
        
        # Mode hierarchy for fallback
        self._mode_hierarchy = ["tiny", "small", "base", "large", "gundam"]
    
    async def process_image(
        self,
        image_data: bytes,
        context: Optional[str] = None,
    ) -> OCRResult:
        """
        Process an image with OCR.
        
        Args:
            image_data: Raw image bytes (PNG, JPEG, etc.)
            context: Optional context hint for OCR
            
        Returns:
            OCRResult with extracted text and confidence
        """
        import time
        start_time = time.perf_counter()
        
        if not self.endpoint:
            # Fallback to local OCR or return placeholder
            result = await self._fallback_ocr(image_data)
            rag_metrics.ocr_pages_processed.inc()
            rag_metrics.ocr_duration.observe(time.perf_counter() - start_time)
            return result
        
        # Check if Gundam Tiling should be used
        use_gundam = self.mode == "gundam" or await self._should_use_gundam_tiling(image_data)
        
        if use_gundam and self.gundam_config.enabled:
            result = await self._process_with_gundam_tiling(image_data, context)
            rag_metrics.ocr_pages_processed.inc()
            rag_metrics.ocr_duration.observe(time.perf_counter() - start_time)
            return result
        
        current_mode = self.mode
        retry_count = 0
        result: Optional[OCRResult] = None
        
        while retry_count <= self.retry_limit:
            result = await self._call_ocr(image_data, current_mode, context)
            
            if result.error:
                logger.warning(f"OCR error with mode {current_mode}: {result.error}")
                retry_count += 1
                rag_metrics.ocr_retries.inc()
                current_mode = self._get_next_mode(current_mode)
                if current_mode is None:
                    rag_metrics.ocr_failures.inc()
                    rag_metrics.ocr_duration.observe(time.perf_counter() - start_time)
                    return result
                continue
            
            # Check confidence threshold
            if result.confidence >= self.confidence_threshold:
                result.retry_count = retry_count
                rag_metrics.ocr_pages_processed.inc()
                rag_metrics.ocr_duration.observe(time.perf_counter() - start_time)
                return result
            
            # Try higher mode
            logger.info(
                f"OCR confidence {result.confidence:.2f} below threshold "
                f"{self.confidence_threshold}, retrying with higher mode"
            )
            retry_count += 1
            rag_metrics.ocr_retries.inc()
            current_mode = self._get_next_mode(current_mode)
            
            if current_mode is None:
                result.retry_count = retry_count
                rag_metrics.ocr_pages_processed.inc()
                rag_metrics.ocr_duration.observe(time.perf_counter() - start_time)
                return result
        
        if result is None:
            rag_metrics.ocr_failures.inc()
            rag_metrics.ocr_duration.observe(time.perf_counter() - start_time)
            return OCRResult(
                text="",
                confidence=0.0,
                has_tables=False,
                tables=[],
                mode_used=self.mode,
                error="No OCR result obtained",
            )
        
        rag_metrics.ocr_pages_processed.inc()
        rag_metrics.ocr_duration.observe(time.perf_counter() - start_time)
        return result
    
    async def _call_ocr(
        self,
        image_data: bytes,
        mode: str,
        context: Optional[str] = None,
    ) -> OCRResult:
        """
        Call the OCR API using OpenAI-compatible vision endpoint (Qwen3-VL).
        
        For Qwen3-VL, mode is ignored since we use a single model.
        """
        try:
            # Encode image as base64 data URL
            image_b64 = base64.b64encode(image_data).decode("utf-8")
            
            # Detect image type from magic bytes
            image_type = "image/png"  # default
            if image_data[:3] == b'\xff\xd8\xff':
                image_type = "image/jpeg"
            elif image_data[:8] == b'\x89PNG\r\n\x1a\n':
                image_type = "image/png"
            elif image_data[:4] == b'GIF8':
                image_type = "image/gif"
            elif image_data[:4] == b'RIFF' and image_data[8:12] == b'WEBP':
                image_type = "image/webp"
            
            data_url = f"data:{image_type};base64,{image_b64}"
            
            # Build OCR prompt with optional context
            prompt = OCR_PROMPT
            if context:
                prompt = f"Context: {context}\n\n{prompt}"
            
            # Prepare OpenAI-compatible chat completion request
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt,
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": data_url,
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 2048,  # Reduced from 4096 to leave room for input tokens
                "temperature": 0.1,  # Low temperature for accurate OCR
            }
            
            # Call OpenAI-compatible endpoint
            endpoint_url = self.endpoint.rstrip("/")
            if not endpoint_url.endswith("/chat/completions"):
                endpoint_url = f"{endpoint_url}/chat/completions"
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    endpoint_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()
                data = response.json()
            
            # Parse response from OpenAI format
            text = ""
            if "choices" in data and len(data["choices"]) > 0:
                choice = data["choices"][0]
                if "message" in choice:
                    text = choice["message"].get("content", "")
                elif "text" in choice:
                    text = choice["text"]
            
            # Estimate confidence based on text quality (no API confidence available)
            confidence = self._estimate_confidence(text)
            
            # Extract tables if present
            tables = self._extract_tables(text)
            has_tables = len(tables) > 0
            
            return OCRResult(
                text=text,
                confidence=confidence,
                has_tables=has_tables,
                tables=tables,
                mode_used=f"qwen3-vl",
                metadata={"model": self.model, "usage": data.get("usage", {})},
            )
            
        except httpx.HTTPError as e:
            logger.error(f"OCR HTTP error: {e}")
            return OCRResult(
                text="",
                confidence=0.0,
                has_tables=False,
                tables=[],
                mode_used=mode,
                error=f"HTTP error: {str(e)}",
            )
        except Exception as e:
            logger.error(f"OCR error: {e}")
            return OCRResult(
                text="",
                confidence=0.0,
                has_tables=False,
                tables=[],
                mode_used=mode,
                error=str(e),
            )
    
    async def _fallback_ocr(self, image_data: bytes) -> OCRResult:
        """
        Fallback OCR when DeepSeek endpoint is not configured.
        Uses pytesseract if available, otherwise returns placeholder.
        """
        try:
            import pytesseract
            from PIL import Image
            import io
            
            image = Image.open(io.BytesIO(image_data))
            text = pytesseract.image_to_string(image, lang="por+eng")
            
            # Estimate confidence based on text quality
            confidence = self._estimate_confidence(text)
            tables = self._extract_tables(text)
            
            return OCRResult(
                text=text,
                confidence=confidence,
                has_tables=len(tables) > 0,
                tables=tables,
                mode_used="tesseract_fallback",
                metadata={"fallback": True},
            )
            
        except ImportError:
            logger.warning("No OCR backend available (pytesseract not installed)")
            return OCRResult(
                text="[OCR not available - install pytesseract or configure DeepSeek endpoint]",
                confidence=0.0,
                has_tables=False,
                tables=[],
                mode_used="none",
                error="No OCR backend available",
            )
        except Exception as e:
            logger.error(f"Fallback OCR failed: {e}")
            return OCRResult(
                text="",
                confidence=0.0,
                has_tables=False,
                tables=[],
                mode_used="tesseract_fallback",
                error=str(e),
            )
    
    def _estimate_confidence(self, text: str) -> float:
        """Estimate OCR confidence from text quality heuristics."""
        if not text or len(text.strip()) < 10:
            return 0.1
        
        # Heuristics for quality estimation
        total_chars = len(text)
        
        # Count "good" characters (letters, numbers, common punctuation)
        good_chars = sum(1 for c in text if c.isalnum() or c in " .,;:!?-\n\t")
        good_ratio = good_chars / total_chars if total_chars > 0 else 0
        
        # Check for garbled text (many special characters)
        special_chars = sum(1 for c in text if not c.isalnum() and c not in " .,;:!?-\n\t")
        special_ratio = special_chars / total_chars if total_chars > 0 else 0
        
        # Base confidence
        confidence = good_ratio * 0.8 + (1 - special_ratio) * 0.2
        
        # Adjust for text length (very short = lower confidence)
        if total_chars < 50:
            confidence *= 0.7
        elif total_chars < 100:
            confidence *= 0.85
        
        return min(max(confidence, 0.0), 1.0)
    
    def _extract_tables(self, text: str) -> List[str]:
        """Extract Markdown tables from OCR text."""
        tables = []
        lines = text.split("\n")
        
        table_lines = []
        in_table = False
        
        for line in lines:
            # Detect table row (starts and ends with |)
            stripped = line.strip()
            if stripped.startswith("|") and stripped.endswith("|"):
                in_table = True
                table_lines.append(stripped)
            elif in_table:
                # End of table
                if len(table_lines) >= 2:  # At least header + data
                    tables.append("\n".join(table_lines))
                table_lines = []
                in_table = False
        
        # Handle table at end of text
        if table_lines and len(table_lines) >= 2:
            tables.append("\n".join(table_lines))
        
        return tables
    
    def _get_next_mode(self, current_mode: str) -> Optional[str]:
        """Get the next higher quality mode for retry."""
        try:
            idx = self._mode_hierarchy.index(current_mode)
            if idx < len(self._mode_hierarchy) - 1:
                return self._mode_hierarchy[idx + 1]
        except ValueError:
            pass
        return None
    
    # ──────────────────────────────────────────────────────────────────────────
    # Gundam Tiling Methods
    # ──────────────────────────────────────────────────────────────────────────
    
    async def _should_use_gundam_tiling(self, image_data: bytes) -> bool:
        """
        Determine if Gundam Tiling should be used for this image.
        
        Checks image dimensions against min_image_size threshold.
        """
        try:
            from PIL import Image
            import io
            
            image = Image.open(io.BytesIO(image_data))
            width, height = image.size
            
            # Use Gundam Tiling for large images
            return max(width, height) >= self.gundam_config.min_image_size
        except Exception as e:
            logger.debug(f"Could not check image size for Gundam Tiling: {e}")
            return False
    
    def _calculate_tiles(
        self,
        width: int,
        height: int,
    ) -> List[Tuple[int, int, int, int]]:
        """
        Calculate tile coordinates with overlap.
        
        Args:
            width: Image width in pixels
            height: Image height in pixels
            
        Returns:
            List of (x1, y1, x2, y2) tile coordinates
        """
        tile_size = self.gundam_config.tile_size
        overlap = self.gundam_config.overlap
        max_tiles = self.gundam_config.max_tiles
        
        step = tile_size - overlap
        tiles = []
        
        # Calculate grid
        cols = math.ceil(width / step) if step > 0 else 1
        rows = math.ceil(height / step) if step > 0 else 1
        
        # Limit total tiles
        if cols * rows > max_tiles:
            # Reduce tile count by increasing step
            scale = math.sqrt((cols * rows) / max_tiles)
            step = int(step * scale)
            cols = math.ceil(width / step) if step > 0 else 1
            rows = math.ceil(height / step) if step > 0 else 1
        
        for row in range(rows):
            for col in range(cols):
                x1 = col * step
                y1 = row * step
                x2 = min(x1 + tile_size, width)
                y2 = min(y1 + tile_size, height)
                
                # Adjust start if we're at the edge
                if x2 == width and x2 - x1 < tile_size:
                    x1 = max(0, width - tile_size)
                if y2 == height and y2 - y1 < tile_size:
                    y1 = max(0, height - tile_size)
                
                tiles.append((x1, y1, x2, y2))
        
        # Deduplicate overlapping tiles
        seen = set()
        unique_tiles = []
        for tile in tiles:
            if tile not in seen:
                seen.add(tile)
                unique_tiles.append(tile)
        
        return unique_tiles[:max_tiles]
    
    async def _process_with_gundam_tiling(
        self,
        image_data: bytes,
        context: Optional[str] = None,
    ) -> OCRResult:
        """
        Process image using Gundam Tiling strategy.
        
        1. Split image into overlapping tiles
        2. OCR each tile concurrently
        3. Merge results using fuzzy matching
        """
        try:
            from PIL import Image
            import io
            
            image = Image.open(io.BytesIO(image_data))
            width, height = image.size
            
            # Calculate tiles
            tiles = self._calculate_tiles(width, height)
            logger.info(f"Gundam Tiling: processing {len(tiles)} tiles for {width}x{height} image")
            
            if len(tiles) <= 1:
                # Single tile, just do normal OCR
                return await self._call_ocr(image_data, "large", context)
            
            # Extract tile images
            tile_images = []
            for x1, y1, x2, y2 in tiles:
                tile_img = image.crop((x1, y1, x2, y2))
                buf = io.BytesIO()
                tile_img.save(buf, format="PNG")
                tile_images.append(buf.getvalue())
            
            # Process tiles concurrently
            semaphore = asyncio.Semaphore(4)  # Limit concurrency
            
            async def process_tile(tile_data: bytes, tile_idx: int) -> Tuple[int, OCRResult]:
                async with semaphore:
                    result = await self._call_ocr(tile_data, "large", context)
                    return tile_idx, result
            
            tasks = [process_tile(tile_data, idx) for idx, tile_data in enumerate(tile_images)]
            tile_results = await asyncio.gather(*tasks)
            
            # Sort by tile index to maintain reading order
            tile_results = sorted(tile_results, key=lambda x: x[0])
            
            # Merge results
            return self._merge_tile_results(
                [r for _, r in tile_results],
                tiles,
                width,
                height,
            )
            
        except ImportError:
            logger.warning("PIL not available, falling back to normal OCR")
            return await self._call_ocr(image_data, "large", context)
        except Exception as e:
            logger.error(f"Gundam Tiling failed: {e}, falling back to normal OCR")
            return await self._call_ocr(image_data, "large", context)
    
    def _merge_tile_results(
        self,
        results: List[OCRResult],
        tiles: List[Tuple[int, int, int, int]],
        width: int,
        height: int,
    ) -> OCRResult:
        """
        Merge OCR results from multiple tiles.
        
        Uses the configured merge strategy:
        - "concat": Simple concatenation with newlines
        - "fuzzy": Fuzzy matching to deduplicate overlap regions
        - "vote": Confidence-weighted voting for overlap regions
        """
        if not results:
            return OCRResult(
                text="",
                confidence=0.0,
                has_tables=False,
                tables=[],
                mode_used="gundam",
                error="No tile results",
            )
        
        strategy = self.gundam_config.merge_strategy
        
        if strategy == "concat":
            merged_text = self._merge_concat(results)
        elif strategy == "fuzzy":
            merged_text = self._merge_fuzzy(results, tiles)
        elif strategy == "vote":
            merged_text = self._merge_vote(results, tiles)
        else:
            merged_text = self._merge_fuzzy(results, tiles)
        
        # Calculate aggregate confidence
        confidences = [r.confidence for r in results if r.confidence > 0]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Collect tables from all tiles
        all_tables = []
        for r in results:
            all_tables.extend(r.tables)
        
        return OCRResult(
            text=merged_text,
            confidence=avg_confidence,
            has_tables=len(all_tables) > 0,
            tables=all_tables,
            mode_used="gundam",
            tiles_processed=len(results),
            tile_confidences=[r.confidence for r in results],
            metadata={
                "gundam_tiles": len(tiles),
                "image_size": f"{width}x{height}",
                "merge_strategy": strategy,
            },
        )
    
    def _merge_concat(self, results: List[OCRResult]) -> str:
        """Simple concatenation of tile texts."""
        texts = [r.text.strip() for r in results if r.text.strip()]
        return "\n\n".join(texts)
    
    def _merge_fuzzy(
        self,
        results: List[OCRResult],
        tiles: List[Tuple[int, int, int, int]],
    ) -> str:
        """
        Fuzzy merge: Remove duplicate lines that appear in overlapping regions.
        
        Uses similarity threshold to detect and deduplicate repeated text
        from adjacent tiles' overlap zones.
        """
        from difflib import SequenceMatcher
        
        if len(results) <= 1:
            return results[0].text if results else ""
        
        threshold = self.gundam_config.fuzzy_threshold
        merged_lines: List[str] = []
        
        for result in results:
            lines = result.text.strip().split("\n")
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check if this line is similar to any recent line
                is_duplicate = False
                for existing in merged_lines[-10:]:  # Check last 10 lines
                    similarity = SequenceMatcher(None, line, existing).ratio()
                    if similarity >= threshold:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    merged_lines.append(line)
        
        return "\n".join(merged_lines)
    
    def _merge_vote(
        self,
        results: List[OCRResult],
        tiles: List[Tuple[int, int, int, int]],
    ) -> str:
        """
        Confidence-weighted voting for overlap regions.
        
        For lines that appear in multiple tiles' overlap zones,
        use the version from the higher-confidence tile.
        """
        from difflib import SequenceMatcher
        
        if len(results) <= 1:
            return results[0].text if results else ""
        
        threshold = self.gundam_config.fuzzy_threshold
        
        # Collect all lines with their confidence scores
        line_scores: Dict[str, Tuple[str, float]] = {}  # normalized -> (original, confidence)
        
        for result in results:
            lines = result.text.strip().split("\n")
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                normalized = line.lower()
                
                # Check for similar existing lines
                best_match = None
                best_similarity = 0.0
                
                for existing_norm in line_scores:
                    similarity = SequenceMatcher(None, normalized, existing_norm).ratio()
                    if similarity >= threshold and similarity > best_similarity:
                        best_match = existing_norm
                        best_similarity = similarity
                
                if best_match:
                    # Update if this version has higher confidence
                    existing_line, existing_conf = line_scores[best_match]
                    if result.confidence > existing_conf:
                        del line_scores[best_match]
                        line_scores[normalized] = (line, result.confidence)
                else:
                    line_scores[normalized] = (line, result.confidence)
        
        # Return lines in order of appearance (using dict order preservation)
        return "\n".join(orig for orig, _ in line_scores.values())

    async def process_batch(
        self,
        images: List[bytes],
        max_concurrent: int = 4,
    ) -> List[OCRResult]:
        """
        Process multiple images concurrently.
        
        Args:
            images: List of image data bytes
            max_concurrent: Maximum concurrent OCR requests
            
        Returns:
            List of OCRResult for each image
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_limit(image_data: bytes) -> OCRResult:
            async with semaphore:
                return await self.process_image(image_data)
        
        tasks = [process_with_limit(img) for img in images]
        return await asyncio.gather(*tasks)
