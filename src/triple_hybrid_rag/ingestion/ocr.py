"""
OCR Processing Module

Handles optical character recognition for scanned documents and images:
- Qwen3-VL Vision API integration (OpenAI-compatible)
- DeepSeek OCR support
- **Gundam Tiling**: Overlapping tile strategy for high-resolution documents
- Confidence tracking with heuristic estimation
- Retry logic with tenacity + fallback modes
- Table detection and preservation
- AUTO mode for intelligent OCR decision-making
"""

import asyncio
import base64
import logging
import math
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

import httpx
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from triple_hybrid_rag.config import RAGConfig, get_settings

if TYPE_CHECKING:
    from triple_hybrid_rag.ingestion.loaders import LoadedDocument

# Default OCR prompt - kept simple for compatibility
OCR_PROMPT = """OCR this image. Extract all text exactly as it appears. For tables, use Markdown format. Output only the extracted text."""

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# FILE EXTENSION CATEGORIES FOR AUTO MODE
# ═══════════════════════════════════════════════════════════════════════════════

# Files that NEVER need OCR (pure text formats)
TEXT_ONLY_EXTENSIONS: Set[str] = {
    ".txt", ".md", ".markdown", ".csv", ".json", ".jsonl",
    ".log", ".xml", ".html", ".htm", ".yaml", ".yml",
    ".toml", ".ini", ".cfg", ".conf", ".env", ".rst",
    ".tex", ".rtf", ".tsv", ".sql", ".py", ".js", ".ts",
    ".java", ".c", ".cpp", ".h", ".hpp", ".go", ".rs",
    ".rb", ".php", ".sh", ".bash", ".zsh", ".ps1",
    ".css", ".scss", ".sass", ".less", ".vue", ".jsx", ".tsx",
    ".kt", ".swift", ".m", ".mm", ".r", ".R", ".scala",
    ".pl", ".pm", ".lua", ".hs", ".elm", ".clj", ".ex", ".exs",
    ".erl", ".hrl", ".ml", ".mli", ".fs", ".fsi", ".fsx",
}

# Files that MAY need OCR (analysis required)
ANALYZABLE_EXTENSIONS: Set[str] = {
    ".pdf",   # Could be native or scanned
    ".docx", ".doc",  # Could have embedded images
    ".xlsx", ".xls",  # Could have image content
    ".pptx", ".ppt",  # Presentations with images
    ".odt", ".ods", ".odp",  # OpenDocument formats
}

# Files that ALWAYS need OCR (images)
IMAGE_EXTENSIONS: Set[str] = {
    ".png", ".jpg", ".jpeg", ".webp", ".tiff", ".tif",
    ".bmp", ".gif", ".heic", ".heif", ".ico", ".svg",
    ".raw", ".cr2", ".nef", ".arw", ".dng",
}


class OCRIngestionMode(Enum):
    """
    OCR ingestion mode for document processing.
    
    Determines which OCR provider to use during ingestion:
    - QWEN: Use Qwen3-VL for OCR
    - DEEPSEEK: Use DeepSeek OCR
    - LOCAL_VRAG: Use local Visual RAG API (LightOnOCR-2-1B)
    - OFF: Skip OCR, only extract native text
    - AUTO: System decides based on file analysis
    """
    QWEN = "qwen"
    DEEPSEEK = "deepseek"
    LOCAL_VRAG = "local_vrag"
    OFF = "off"
    AUTO = "auto"


class OCRMode(Enum):
    """OCR processing modes (quality vs speed/cost tradeoff)."""
    TINY = "tiny"
    SMALL = "small"
    BASE = "base"
    LARGE = "large"
    GUNDAM = "gundam"  # Highest quality - uses Gundam Tiling


@dataclass
class DocumentOCRAnalysis:
    """
    Analysis result for AUTO mode decision.
    
    Contains the recommended OCR mode and scoring details.
    """
    recommended_mode: OCRIngestionMode
    confidence: float  # 0.0 to 1.0 confidence in the recommendation
    reasons: List[str]  # Explanation of why this mode was chosen
    scores: Dict[str, float] = field(default_factory=dict)  # Individual factor scores
    
    def __str__(self) -> str:
        reasons_str = "; ".join(self.reasons)
        return f"OCRAnalysis(mode={self.recommended_mode.value}, confidence={self.confidence:.2f}, reasons=[{reasons_str}])"


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
    network_retry_count: int = 0
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
    # Gundam Tiling metadata
    tiles_processed: int = 0
    tile_confidences: List[float] = field(default_factory=list)
    latency_ms: float = 0.0
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class DocumentOCRAnalyzer:
    """
    Analyzes documents to determine if OCR would be beneficial.
    
    Used by AUTO mode to make intelligent decisions about OCR usage
    based on file type, text density, scanned page ratio, and image presence.
    """
    
    def __init__(self, config: Optional[RAGConfig] = None):
        """
        Initialize the analyzer.
        
        Args:
            config: RAG configuration with AUTO mode settings
        """
        self._config = config or get_settings()
        self._threshold = self._config.rag_ocr_auto_threshold
        self._preferred = self._config.rag_ocr_auto_preferred.lower()
    
    def analyze_file_path(self, file_path: str) -> DocumentOCRAnalysis:
        """
        Quick analysis based on file extension only.
        
        Args:
            file_path: Path to the file
            
        Returns:
            DocumentOCRAnalysis with recommendation
        """
        ext = Path(file_path).suffix.lower()
        
        # Pure text files - NEVER need OCR
        if ext in TEXT_ONLY_EXTENSIONS:
            return DocumentOCRAnalysis(
                recommended_mode=OCRIngestionMode.OFF,
                confidence=1.0,
                reasons=[f"File extension '{ext}' is a text-only format"],
                scores={"file_type": -1.0},
            )
        
        # Image files - ALWAYS need OCR
        if ext in IMAGE_EXTENSIONS:
            preferred_mode = self._get_preferred_mode()
            return DocumentOCRAnalysis(
                recommended_mode=preferred_mode,
                confidence=1.0,
                reasons=[f"File extension '{ext}' is an image format"],
                scores={"file_type": 1.0},
            )
        
        # Analyzable files - need deeper analysis
        if ext in ANALYZABLE_EXTENSIONS:
            return DocumentOCRAnalysis(
                recommended_mode=OCRIngestionMode.AUTO,  # Needs content analysis
                confidence=0.5,
                reasons=[f"File extension '{ext}' requires content analysis"],
                scores={"file_type": 0.0},
            )
        
        # Unknown extension - assume no OCR needed
        return DocumentOCRAnalysis(
            recommended_mode=OCRIngestionMode.OFF,
            confidence=0.7,
            reasons=[f"Unknown file extension '{ext}', assuming text-based"],
            scores={"file_type": -0.5},
        )
    
    def analyze_document(self, document: "LoadedDocument") -> DocumentOCRAnalysis:
        """
        Full analysis of a loaded document.
        
        Analyzes text density, scanned page ratio, and image presence
        to determine if OCR would be beneficial.
        
        Args:
            document: Loaded document with page content
            
        Returns:
            DocumentOCRAnalysis with recommendation and scoring details
        """
        ext = Path(document.file_path).suffix.lower()
        reasons: List[str] = []
        scores: Dict[str, float] = {}
        
        # 1. File type scoring (25% weight)
        if ext in TEXT_ONLY_EXTENSIONS:
            scores["file_type"] = -1.0
            reasons.append(f"Text-only format ({ext})")
            # Early return for text files
            return DocumentOCRAnalysis(
                recommended_mode=OCRIngestionMode.OFF,
                confidence=1.0,
                reasons=reasons,
                scores=scores,
            )
        elif ext in IMAGE_EXTENSIONS:
            scores["file_type"] = 1.0
            reasons.append(f"Image format ({ext})")
            # Early return for images
            return DocumentOCRAnalysis(
                recommended_mode=self._get_preferred_mode(),
                confidence=1.0,
                reasons=reasons,
                scores=scores,
            )
        else:
            scores["file_type"] = 0.0
            reasons.append(f"Analyzable format ({ext})")
        
        # 2. Text density scoring (30% weight)
        total_chars = sum(len(page.text.strip()) for page in document.pages)
        total_pages = len(document.pages) or 1
        chars_per_page = total_chars / total_pages
        
        # Low text density suggests scanned content
        if chars_per_page < 100:
            scores["text_density"] = 1.0
            reasons.append(f"Very low text density ({chars_per_page:.0f} chars/page)")
        elif chars_per_page < 500:
            scores["text_density"] = 0.5
            reasons.append(f"Low text density ({chars_per_page:.0f} chars/page)")
        elif chars_per_page < 1000:
            scores["text_density"] = 0.0
            reasons.append(f"Moderate text density ({chars_per_page:.0f} chars/page)")
        else:
            scores["text_density"] = -0.5
            reasons.append(f"High text density ({chars_per_page:.0f} chars/page)")
        
        # 3. Scanned page ratio scoring (25% weight)
        scanned_pages = sum(1 for page in document.pages if page.is_scanned)
        scanned_ratio = scanned_pages / total_pages if total_pages > 0 else 0
        
        if scanned_ratio >= 0.8:
            scores["scanned_ratio"] = 1.0
            reasons.append(f"Most pages are scanned ({scanned_ratio:.0%})")
        elif scanned_ratio >= 0.5:
            scores["scanned_ratio"] = 0.6
            reasons.append(f"Many pages are scanned ({scanned_ratio:.0%})")
        elif scanned_ratio >= 0.2:
            scores["scanned_ratio"] = 0.3
            reasons.append(f"Some pages are scanned ({scanned_ratio:.0%})")
        else:
            scores["scanned_ratio"] = -0.5
            reasons.append(f"Few/no scanned pages ({scanned_ratio:.0%})")
        
        # 4. Image presence scoring (20% weight)
        pages_with_images = sum(1 for page in document.pages if page.has_images)
        image_ratio = pages_with_images / total_pages if total_pages > 0 else 0
        
        if image_ratio >= 0.8:
            scores["image_presence"] = 0.8
            reasons.append(f"Most pages have images ({image_ratio:.0%})")
        elif image_ratio >= 0.5:
            scores["image_presence"] = 0.5
            reasons.append(f"Many pages have images ({image_ratio:.0%})")
        elif image_ratio >= 0.2:
            scores["image_presence"] = 0.2
            reasons.append(f"Some pages have images ({image_ratio:.0%})")
        else:
            scores["image_presence"] = 0.0
            reasons.append(f"Few/no images ({image_ratio:.0%})")
        
        # Calculate final weighted score
        final_score = (
            scores["file_type"] * 0.25 +
            scores["text_density"] * 0.30 +
            scores["scanned_ratio"] * 0.25 +
            scores["image_presence"] * 0.20
        )
        
        # Make decision based on threshold
        if final_score >= self._threshold:
            recommended_mode = self._get_preferred_mode()
            confidence = min(0.5 + final_score, 1.0)
            reasons.insert(0, f"OCR recommended (score={final_score:.2f} >= threshold={self._threshold})")
        else:
            recommended_mode = OCRIngestionMode.OFF
            confidence = min(0.5 + abs(final_score), 1.0)
            reasons.insert(0, f"OCR not needed (score={final_score:.2f} < threshold={self._threshold})")
        
        return DocumentOCRAnalysis(
            recommended_mode=recommended_mode,
            confidence=confidence,
            reasons=reasons,
            scores=scores,
        )
    
    def _get_preferred_mode(self) -> OCRIngestionMode:
        """Get the preferred OCR mode from configuration."""
        if self._preferred == "deepseek":
            return OCRIngestionMode.DEEPSEEK
        return OCRIngestionMode.QWEN


def resolve_ocr_mode(
    mode: str,
    document: Optional["LoadedDocument"] = None,
    file_path: Optional[str] = None,
    config: Optional[RAGConfig] = None,
) -> Tuple[OCRIngestionMode, Optional[DocumentOCRAnalysis]]:
    """
    Resolve the effective OCR mode for a document.
    
    Handles AUTO mode resolution and backward compatibility.
    
    Args:
        mode: OCR mode string from configuration
        document: Optional loaded document for content analysis
        file_path: Optional file path for extension-based analysis
        config: Optional RAG configuration
        
    Returns:
        Tuple of (resolved OCRIngestionMode, analysis if AUTO mode was used)
    """
    config = config or get_settings()
    mode_lower = mode.lower()
    
    # Handle backward compatibility with rag_deepseek_ocr_enabled
    if mode_lower == "auto" and config.rag_deepseek_ocr_enabled:
        # Legacy setting takes precedence when mode is auto
        logger.debug("Using legacy rag_deepseek_ocr_enabled=True setting")
        return OCRIngestionMode.DEEPSEEK, None
    
    # Direct mode mapping
    if mode_lower == "qwen":
        return OCRIngestionMode.QWEN, None
    elif mode_lower == "deepseek":
        return OCRIngestionMode.DEEPSEEK, None
    elif mode_lower == "local_vrag":
        return OCRIngestionMode.LOCAL_VRAG, None
    elif mode_lower == "off":
        return OCRIngestionMode.OFF, None
    elif mode_lower == "auto":
        # Perform AUTO analysis
        analyzer = DocumentOCRAnalyzer(config)
        
        if document is not None:
            analysis = analyzer.analyze_document(document)
        elif file_path is not None:
            analysis = analyzer.analyze_file_path(file_path)
        else:
            # No context, use preferred mode
            analysis = DocumentOCRAnalysis(
                recommended_mode=analyzer._get_preferred_mode(),
                confidence=0.5,
                reasons=["No document context, using preferred OCR provider"],
            )
        
        logger.info(f"AUTO OCR analysis: {analysis}")
        
        # Recurse if analysis returned AUTO (needs content analysis)
        if analysis.recommended_mode == OCRIngestionMode.AUTO:
            return analyzer._get_preferred_mode(), analysis
        
        return analysis.recommended_mode, analysis
    
    # Unknown mode, default to OFF
    logger.warning(f"Unknown OCR mode '{mode}', defaulting to OFF")
    return OCRIngestionMode.OFF, None


class OCRProcessor:
    """
    Process images and scanned documents with OCR.
    
    Uses Qwen3-VL or DeepSeek via OpenAI-compatible endpoint with configurable
    modes and retry logic. Falls back to higher quality modes if confidence is low.
    
    Supports OCRIngestionMode for selecting OCR provider:
    - QWEN: Qwen3-VL OCR
    - DEEPSEEK: DeepSeek OCR
    - OFF: No OCR processing
    - AUTO: Intelligent selection based on document analysis
    
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
        timeout: Optional[float] = None,
        gundam_config: Optional[GundamTilingConfig] = None,
        settings: Optional[RAGConfig] = None,
        ingestion_mode: Optional[OCRIngestionMode] = None,
    ):
        """
        Initialize OCR processor.
        
        Args:
            mode: OCR mode (tiny, small, base, large, gundam)
            confidence_threshold: Minimum confidence to accept result
            retry_limit: Maximum retry attempts with higher modes
            endpoint: Vision API endpoint URL
            model: Vision model name
            timeout: Request timeout in seconds
            gundam_config: Configuration for Gundam Tiling strategy
            settings: Optional Settings object
            ingestion_mode: OCR ingestion mode (QWEN, DEEPSEEK, OFF, AUTO)
        """
        self._settings = settings or get_settings()
        
        self.mode = mode
        self.confidence_threshold = confidence_threshold
        self.retry_limit = retry_limit
        
        # Determine OCR provider based on ingestion_mode or legacy settings
        self.ingestion_mode = ingestion_mode
        if ingestion_mode == OCRIngestionMode.DEEPSEEK:
            default_endpoint = self._settings.rag_deepseek_ocr_api_base
            default_model = self._settings.rag_deepseek_ocr_model
        elif ingestion_mode == OCRIngestionMode.QWEN:
            default_endpoint = self._settings.rag_ocr_api_base
            default_model = self._settings.rag_ocr_model
        elif self._settings.rag_deepseek_ocr_enabled:
            # Legacy compatibility
            default_endpoint = self._settings.rag_deepseek_ocr_api_base
            default_model = self._settings.rag_deepseek_ocr_model
        else:
            default_endpoint = self._settings.rag_ocr_api_base
            default_model = self._settings.rag_ocr_model

        self.endpoint = endpoint or default_endpoint
        self.model = model or default_model
        self.timeout = timeout or self._settings.rag_ocr_timeout
        
        # Gundam Tiling configuration
        self.gundam_config = gundam_config or GundamTilingConfig(
            enabled=self._settings.rag_gundam_tiling_enabled,
            tile_size=self._settings.rag_gundam_tile_size,
            overlap=self._settings.rag_gundam_overlap,
            min_image_size=self._settings.rag_gundam_min_image_size,
            max_tiles=self._settings.rag_gundam_max_tiles,
            merge_strategy=self._settings.rag_gundam_merge_strategy,
            fuzzy_threshold=self._settings.rag_gundam_fuzzy_threshold,
        )
        
        # Mode hierarchy for fallback
        self._mode_hierarchy = ["tiny", "small", "base", "large", "gundam"]
        
        # Metrics tracking
        self._total_pages = 0
        self._total_retries = 0
        self._total_network_retries = 0
        self._total_failures = 0
        self._total_latency = 0.0
    
    @classmethod
    def create_for_mode(
        cls,
        ingestion_mode: OCRIngestionMode,
        settings: Optional[RAGConfig] = None,
        **kwargs,
    ) -> "OCRProcessor":
        """
        Factory method to create an OCRProcessor for a specific ingestion mode.
        
        Args:
            ingestion_mode: The OCR ingestion mode to use
            settings: Optional RAG configuration
            **kwargs: Additional arguments passed to __init__
            
        Returns:
            Configured OCRProcessor instance
        """
        return cls(
            ingestion_mode=ingestion_mode,
            settings=settings,
            **kwargs,
        )
    
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
        start_time = time.perf_counter()
        
        # Check if OCR is disabled
        if self.ingestion_mode == OCRIngestionMode.OFF:
            return OCRResult(
                text="",
                confidence=0.0,
                has_tables=False,
                tables=[],
                mode_used="off",
                metadata={"skipped": True, "reason": "OCR mode is OFF"},
                latency_ms=(time.perf_counter() - start_time) * 1000,
            )
        
        if not self.endpoint:
            # Fallback to local OCR or return placeholder
            result = await self._fallback_ocr(image_data)
            result.latency_ms = (time.perf_counter() - start_time) * 1000
            self._total_pages += 1
            return result
        
        # Check if Gundam Tiling should be used
        use_gundam = self.mode == "gundam" or await self._should_use_gundam_tiling(image_data)
        
        if use_gundam and self.gundam_config.enabled:
            result = await self._process_with_gundam_tiling(image_data, context)
            result.latency_ms = (time.perf_counter() - start_time) * 1000
            self._total_pages += 1
            self._total_latency += result.latency_ms
            return result
        
        current_mode = self.mode
        retry_count = 0
        result: Optional[OCRResult] = None
        
        while retry_count <= self.retry_limit:
            result = await self._call_ocr(image_data, current_mode, context)
            
            if result.error:
                logger.warning(f"OCR error with mode {current_mode}: {result.error}")
                retry_count += 1
                self._total_retries += 1
                current_mode = self._get_next_mode(current_mode)
                if current_mode is None:
                    self._total_failures += 1
                    result.latency_ms = (time.perf_counter() - start_time) * 1000
                    return result
                continue
            
            # Check confidence threshold
            if result.confidence >= self.confidence_threshold:
                result.retry_count = retry_count
                result.latency_ms = (time.perf_counter() - start_time) * 1000
                self._total_pages += 1
                self._total_latency += result.latency_ms
                return result
            
            # Try higher mode
            logger.info(
                f"OCR confidence {result.confidence:.2f} below threshold "
                f"{self.confidence_threshold}, retrying with higher mode"
            )
            retry_count += 1
            self._total_retries += 1
            current_mode = self._get_next_mode(current_mode)
            
            if current_mode is None:
                result.retry_count = retry_count
                result.latency_ms = (time.perf_counter() - start_time) * 1000
                self._total_pages += 1
                self._total_latency += result.latency_ms
                return result
        
        if result is None:
            self._total_failures += 1
            return OCRResult(
                text="",
                confidence=0.0,
                has_tables=False,
                tables=[],
                mode_used=self.mode,
                error="No OCR result obtained",
                latency_ms=(time.perf_counter() - start_time) * 1000,
            )
        
        result.latency_ms = (time.perf_counter() - start_time) * 1000
        self._total_pages += 1
        self._total_latency += result.latency_ms
        return result
    
    async def _call_ocr(
        self,
        image_data: bytes,
        mode: str,
        context: Optional[str] = None,
    ) -> OCRResult:
        """
        Call the OCR API using OpenAI-compatible vision endpoint.
        
        Supports both Qwen3-VL and DeepSeek OCR endpoints.
        Uses tenacity retry for transient network errors.
        """
        network_retry_count = 0

        async def _execute_request() -> OCRResult:
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
            elif image_data[:4] == b'RIFF' and len(image_data) >= 12 and image_data[8:12] == b'WEBP':
                image_type = "image/webp"

            data_url = f"data:{image_type};base64,{image_b64}"

            # Build OCR prompt with optional context
            prompt = OCR_PROMPT
            if context:
                prompt = f"Context: {context}\n\n{prompt}"

            # Prepare OpenAI-compatible chat completion request
            # Calculate max_tokens dynamically to avoid context overflow
            # DeepSeek-OCR has 2048 context limit, so we use 1500 for output
            # to leave room for ~500 input tokens (prompt + image tokens)
            max_tokens = 1500
            
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
                            },
                        ],
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": 0.1,
            }

            # Call OpenAI-compatible endpoint
            endpoint_url = self.endpoint.rstrip("/")
            if not endpoint_url.endswith("/chat/completions"):
                endpoint_url = f"{endpoint_url}/chat/completions"

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                try:
                    response = await client.post(
                        endpoint_url,
                        json=payload,
                        headers={"Content-Type": "application/json"},
                    )
                    response.raise_for_status()
                except httpx.HTTPStatusError as e:
                    if self._is_retryable_http_error(e):
                        raise
                    logger.error(f"OCR HTTP error: {e}")
                    return OCRResult(
                        text="",
                        confidence=0.0,
                        has_tables=False,
                        tables=[],
                        mode_used=mode,
                        error=f"HTTP error: {str(e)}",
                    )
                except (httpx.TimeoutException, httpx.NetworkError):
                    raise
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

                try:
                    data = response.json()
                except Exception as e:
                    logger.error(f"OCR response parse error: {e}")
                    return OCRResult(
                        text="",
                        confidence=0.0,
                        has_tables=False,
                        tables=[],
                        mode_used=mode,
                        error=f"Invalid response JSON: {str(e)}",
                    )

            # Parse response from OpenAI format
            text = ""
            if "choices" in data and len(data["choices"]) > 0:
                choice = data["choices"][0]
                if "message" in choice:
                    text = choice["message"].get("content", "")
                elif "text" in choice:
                    text = choice["text"]

            # Estimate confidence based on text quality
            confidence = self._estimate_confidence(text)

            # Extract tables if present
            tables = self._extract_tables(text)
            has_tables = len(tables) > 0

            return OCRResult(
                text=text,
                confidence=confidence,
                has_tables=has_tables,
                tables=tables,
                mode_used=self.model,
                metadata={"model": self.model, "usage": data.get("usage", {})},
            )

        try:
            retryer = AsyncRetrying(
                retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError)),
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=1, min=1, max=10),
                reraise=True,
            )

            async for attempt in retryer:
                if attempt.retry_state.attempt_number > 1:
                    network_retry_count += 1
                    self._total_network_retries += 1
                with attempt:
                    result = await _execute_request()
                    result.network_retry_count = network_retry_count
                    return result

        except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError) as e:
            logger.error(f"OCR network error: {e}")
            return OCRResult(
                text="",
                confidence=0.0,
                has_tables=False,
                tables=[],
                mode_used=mode,
                error=str(e),
                network_retry_count=network_retry_count,
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
                network_retry_count=network_retry_count,
            )

        # Fallback return (should never be reached)
        return OCRResult(
            text="",
            confidence=0.0,
            has_tables=False,
            tables=[],
            mode_used=mode,
            error="Unexpected code path - no OCR result",
            network_retry_count=network_retry_count,
        )
    
    async def _fallback_ocr(self, image_data: bytes) -> OCRResult:
        """
        Fallback OCR when endpoint is not configured.
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
                text="[OCR not available - install pytesseract or configure endpoint]",
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

    @staticmethod
    def _is_retryable_http_error(error: httpx.HTTPStatusError) -> bool:
        """Determine if an HTTP error should be retried."""
        status = error.response.status_code if error.response else None
        if status is None:
            return False
        return status in {408, 429, 500, 502, 503, 504}
    
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
        total_network_retries = sum(r.network_retry_count for r in results)
        
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
            network_retry_count=total_network_retries,
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
    
    def get_metrics(self) -> Dict[str, Any]:
        """Return current metrics."""
        avg_latency = self._total_latency / self._total_pages if self._total_pages > 0 else 0.0
        return {
            "total_pages": self._total_pages,
            "total_retries": self._total_retries,
            "total_network_retries": self._total_network_retries,
            "total_failures": self._total_failures,
            "avg_latency_ms": avg_latency,
        }
