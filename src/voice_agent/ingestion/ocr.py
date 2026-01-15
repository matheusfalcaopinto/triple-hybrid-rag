"""
OCR Processing Module

Handles optical character recognition for scanned documents and images:
- Qwen3-VL Vision API integration (OpenAI-compatible)
- DeepSeek OCR integration (legacy fallback)
- Confidence tracking
- Retry logic with fallback modes
- Table detection and preservation
"""

import asyncio
import base64
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

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
    GUNDAM = "gundam"  # Highest quality


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
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class OCRProcessor:
    """
    Process images and scanned documents with OCR.
    
    Uses DeepSeek OCR with configurable modes and retry logic.
    Falls back to higher quality modes if confidence is low.
    """
    
    def __init__(
        self,
        mode: str = "base",
        confidence_threshold: float = 0.6,
        retry_limit: int = 2,
        endpoint: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 60.0,
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
        """
        self.mode = mode or SETTINGS.rag_ocr_mode
        self.confidence_threshold = confidence_threshold or SETTINGS.rag_ocr_confidence_threshold
        self.retry_limit = retry_limit or SETTINGS.rag_ocr_retry_limit
        self.endpoint = endpoint or SETTINGS.rag_ocr_endpoint
        self.model = model or SETTINGS.rag_ocr_model
        self.timeout = timeout
        
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
