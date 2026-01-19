"""
Unit tests for OCR Ingestion Mode functionality.

Tests:
- OCRIngestionMode enum values
- File extension categorization
- DocumentOCRAnalyzer scoring logic
- Backward compatibility with deprecated settings
- resolve_ocr_mode function
"""

import pytest
from dataclasses import dataclass, field
from typing import List, Optional
from unittest.mock import MagicMock, patch

from triple_hybrid_rag.config import RAGConfig
from triple_hybrid_rag.ingestion.ocr import (
    OCRIngestionMode,
    OCRMode,
    DocumentOCRAnalysis,
    DocumentOCRAnalyzer,
    OCRProcessor,
    resolve_ocr_mode,
    TEXT_ONLY_EXTENSIONS,
    ANALYZABLE_EXTENSIONS,
    IMAGE_EXTENSIONS,
)


# ═══════════════════════════════════════════════════════════════════════════════
# MOCK CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MockPage:
    """Mock page for testing."""
    text: str = ""
    page_number: int = 1
    is_scanned: bool = False
    has_images: bool = False
    image_data: Optional[bytes] = None


@dataclass
class MockLoadedDocument:
    """Mock loaded document for testing."""
    file_path: str = "test.pdf"
    pages: List[MockPage] = field(default_factory=list)
    error: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════════
# TEST OCRIngestionMode ENUM
# ═══════════════════════════════════════════════════════════════════════════════

class TestOCRIngestionModeEnum:
    """Tests for OCRIngestionMode enum."""
    
    def test_enum_values_exist(self):
        """Test that all expected enum values exist."""
        assert OCRIngestionMode.QWEN.value == "qwen"
        assert OCRIngestionMode.DEEPSEEK.value == "deepseek"
        assert OCRIngestionMode.OFF.value == "off"
        assert OCRIngestionMode.AUTO.value == "auto"
    
    def test_enum_count(self):
        """Test that there are exactly 4 OCR modes."""
        assert len(OCRIngestionMode) == 4
    
    def test_enum_from_value(self):
        """Test creating enum from string value."""
        assert OCRIngestionMode("qwen") == OCRIngestionMode.QWEN
        assert OCRIngestionMode("deepseek") == OCRIngestionMode.DEEPSEEK
        assert OCRIngestionMode("off") == OCRIngestionMode.OFF
        assert OCRIngestionMode("auto") == OCRIngestionMode.AUTO
    
    def test_invalid_value_raises(self):
        """Test that invalid value raises ValueError."""
        with pytest.raises(ValueError):
            OCRIngestionMode("invalid")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST FILE EXTENSION CATEGORIES
# ═══════════════════════════════════════════════════════════════════════════════

class TestFileExtensionCategories:
    """Tests for file extension categorization."""
    
    def test_text_only_extensions_exist(self):
        """Test that common text extensions are in TEXT_ONLY_EXTENSIONS."""
        assert ".txt" in TEXT_ONLY_EXTENSIONS
        assert ".md" in TEXT_ONLY_EXTENSIONS
        assert ".json" in TEXT_ONLY_EXTENSIONS
        assert ".py" in TEXT_ONLY_EXTENSIONS
        assert ".csv" in TEXT_ONLY_EXTENSIONS
    
    def test_analyzable_extensions_exist(self):
        """Test that document formats are in ANALYZABLE_EXTENSIONS."""
        assert ".pdf" in ANALYZABLE_EXTENSIONS
        assert ".docx" in ANALYZABLE_EXTENSIONS
        assert ".doc" in ANALYZABLE_EXTENSIONS
        assert ".xlsx" in ANALYZABLE_EXTENSIONS
        assert ".pptx" in ANALYZABLE_EXTENSIONS
    
    def test_image_extensions_exist(self):
        """Test that image formats are in IMAGE_EXTENSIONS."""
        assert ".png" in IMAGE_EXTENSIONS
        assert ".jpg" in IMAGE_EXTENSIONS
        assert ".jpeg" in IMAGE_EXTENSIONS
        assert ".tiff" in IMAGE_EXTENSIONS
        assert ".webp" in IMAGE_EXTENSIONS
    
    def test_no_overlap_between_categories(self):
        """Test that categories don't overlap."""
        assert not TEXT_ONLY_EXTENSIONS & ANALYZABLE_EXTENSIONS
        assert not TEXT_ONLY_EXTENSIONS & IMAGE_EXTENSIONS
        assert not ANALYZABLE_EXTENSIONS & IMAGE_EXTENSIONS


# ═══════════════════════════════════════════════════════════════════════════════
# TEST DocumentOCRAnalyzer
# ═══════════════════════════════════════════════════════════════════════════════

class TestDocumentOCRAnalyzer:
    """Tests for DocumentOCRAnalyzer class."""
    
    @pytest.fixture
    def default_config(self):
        """Create a default RAGConfig for testing."""
        config = RAGConfig()
        config.rag_ocr_mode = "auto"
        config.rag_ocr_auto_preferred = "qwen"
        config.rag_ocr_auto_threshold = 0.3
        return config
    
    @pytest.fixture
    def analyzer(self, default_config):
        """Create an analyzer with default config."""
        return DocumentOCRAnalyzer(default_config)
    
    # --- File path analysis tests ---
    
    def test_analyze_text_file_returns_off(self, analyzer):
        """Test that text files return OFF mode."""
        analysis = analyzer.analyze_file_path("document.txt")
        
        assert analysis.recommended_mode == OCRIngestionMode.OFF
        assert analysis.confidence == 1.0
        assert "text-only" in analysis.reasons[0].lower()
    
    def test_analyze_markdown_file_returns_off(self, analyzer):
        """Test that markdown files return OFF mode."""
        analysis = analyzer.analyze_file_path("README.md")
        
        assert analysis.recommended_mode == OCRIngestionMode.OFF
        assert analysis.confidence == 1.0
    
    def test_analyze_python_file_returns_off(self, analyzer):
        """Test that Python files return OFF mode."""
        analysis = analyzer.analyze_file_path("script.py")
        
        assert analysis.recommended_mode == OCRIngestionMode.OFF
        assert analysis.confidence == 1.0
    
    def test_analyze_image_file_returns_ocr(self, analyzer):
        """Test that image files return OCR mode (preferred provider)."""
        analysis = analyzer.analyze_file_path("scan.png")
        
        assert analysis.recommended_mode == OCRIngestionMode.QWEN
        assert analysis.confidence == 1.0
        assert "image" in analysis.reasons[0].lower()
    
    def test_analyze_jpeg_file_returns_ocr(self, analyzer):
        """Test that JPEG files return OCR mode."""
        analysis = analyzer.analyze_file_path("photo.jpg")
        
        assert analysis.recommended_mode == OCRIngestionMode.QWEN
        assert analysis.confidence == 1.0
    
    def test_analyze_pdf_file_needs_content_analysis(self, analyzer):
        """Test that PDF files indicate need for content analysis."""
        analysis = analyzer.analyze_file_path("document.pdf")
        
        assert analysis.recommended_mode == OCRIngestionMode.AUTO
        assert analysis.confidence == 0.5
        assert "content analysis" in analysis.reasons[0].lower()
    
    def test_analyze_docx_file_needs_content_analysis(self, analyzer):
        """Test that DOCX files indicate need for content analysis."""
        analysis = analyzer.analyze_file_path("document.docx")
        
        assert analysis.recommended_mode == OCRIngestionMode.AUTO
        assert analysis.confidence == 0.5
    
    def test_analyze_unknown_extension_returns_off(self, analyzer):
        """Test that unknown extensions default to OFF."""
        analysis = analyzer.analyze_file_path("file.xyz")
        
        assert analysis.recommended_mode == OCRIngestionMode.OFF
        assert analysis.confidence == 0.7
        assert "unknown" in analysis.reasons[0].lower()
    
    # --- Document analysis tests ---
    
    def test_analyze_document_text_only_format(self, analyzer):
        """Test analyzing a document with text-only format."""
        doc = MockLoadedDocument(
            file_path="data.json",
            pages=[MockPage(text="a" * 1000, page_number=1)],
        )
        
        analysis = analyzer.analyze_document(doc)
        
        assert analysis.recommended_mode == OCRIngestionMode.OFF
        assert analysis.confidence == 1.0
    
    def test_analyze_document_image_format(self, analyzer):
        """Test analyzing a document with image format."""
        doc = MockLoadedDocument(
            file_path="scan.tiff",
            pages=[MockPage(text="", page_number=1)],
        )
        
        analysis = analyzer.analyze_document(doc)
        
        assert analysis.recommended_mode == OCRIngestionMode.QWEN
        assert analysis.confidence == 1.0
    
    def test_analyze_document_low_text_density_recommends_ocr(self, analyzer):
        """Test that low text density leads to OCR recommendation."""
        doc = MockLoadedDocument(
            file_path="scanned.pdf",
            pages=[
                MockPage(text="x" * 50, page_number=1, is_scanned=True),
                MockPage(text="y" * 30, page_number=2, is_scanned=True),
            ],
        )
        
        analysis = analyzer.analyze_document(doc)
        
        # Low text density + scanned pages should trigger OCR
        assert analysis.recommended_mode in (OCRIngestionMode.QWEN, OCRIngestionMode.DEEPSEEK)
    
    def test_analyze_document_high_text_density_skips_ocr(self, analyzer):
        """Test that high text density skips OCR."""
        doc = MockLoadedDocument(
            file_path="native.pdf",
            pages=[
                MockPage(text="x" * 2000, page_number=1, is_scanned=False),
                MockPage(text="y" * 2500, page_number=2, is_scanned=False),
            ],
        )
        
        analysis = analyzer.analyze_document(doc)
        
        assert analysis.recommended_mode == OCRIngestionMode.OFF
        assert "not needed" in analysis.reasons[0].lower()
    
    def test_analyze_document_scanned_pages_trigger_ocr(self, analyzer):
        """Test that mostly scanned pages trigger OCR."""
        doc = MockLoadedDocument(
            file_path="mixed.pdf",
            pages=[
                MockPage(text="", page_number=1, is_scanned=True),
                MockPage(text="", page_number=2, is_scanned=True),
                MockPage(text="", page_number=3, is_scanned=True),
                MockPage(text="some text", page_number=4, is_scanned=False),
            ],
        )
        
        analysis = analyzer.analyze_document(doc)
        
        assert "scanned" in str(analysis.reasons).lower()
    
    # --- Preferred mode tests ---
    
    def test_preferred_mode_qwen(self, default_config):
        """Test that preferred mode qwen returns QWEN."""
        default_config.rag_ocr_auto_preferred = "qwen"
        analyzer = DocumentOCRAnalyzer(default_config)
        
        analysis = analyzer.analyze_file_path("image.png")
        
        assert analysis.recommended_mode == OCRIngestionMode.QWEN
    
    def test_preferred_mode_deepseek(self, default_config):
        """Test that preferred mode deepseek returns DEEPSEEK."""
        default_config.rag_ocr_auto_preferred = "deepseek"
        analyzer = DocumentOCRAnalyzer(default_config)
        
        analysis = analyzer.analyze_file_path("image.png")
        
        assert analysis.recommended_mode == OCRIngestionMode.DEEPSEEK


# ═══════════════════════════════════════════════════════════════════════════════
# TEST resolve_ocr_mode FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

class TestResolveOCRMode:
    """Tests for resolve_ocr_mode function."""
    
    @pytest.fixture
    def default_config(self):
        """Create a default RAGConfig for testing."""
        config = RAGConfig()
        config.rag_ocr_mode = "auto"
        config.rag_ocr_auto_preferred = "qwen"
        config.rag_ocr_auto_threshold = 0.3
        config.rag_deepseek_ocr_enabled = False
        return config
    
    def test_resolve_qwen_mode(self, default_config):
        """Test resolving QWEN mode."""
        mode, analysis = resolve_ocr_mode("qwen", config=default_config)
        
        assert mode == OCRIngestionMode.QWEN
        assert analysis is None
    
    def test_resolve_deepseek_mode(self, default_config):
        """Test resolving DEEPSEEK mode."""
        mode, analysis = resolve_ocr_mode("deepseek", config=default_config)
        
        assert mode == OCRIngestionMode.DEEPSEEK
        assert analysis is None
    
    def test_resolve_off_mode(self, default_config):
        """Test resolving OFF mode."""
        mode, analysis = resolve_ocr_mode("off", config=default_config)
        
        assert mode == OCRIngestionMode.OFF
        assert analysis is None
    
    def test_resolve_auto_with_file_path(self, default_config):
        """Test resolving AUTO mode with file path."""
        mode, analysis = resolve_ocr_mode(
            "auto",
            file_path="document.txt",
            config=default_config,
        )
        
        assert mode == OCRIngestionMode.OFF
        assert analysis is not None
        assert analysis.recommended_mode == OCRIngestionMode.OFF
    
    def test_resolve_auto_with_image_file(self, default_config):
        """Test resolving AUTO mode with image file."""
        mode, analysis = resolve_ocr_mode(
            "auto",
            file_path="scan.png",
            config=default_config,
        )
        
        assert mode == OCRIngestionMode.QWEN
        assert analysis is not None
    
    def test_resolve_unknown_mode_defaults_to_off(self, default_config):
        """Test that unknown mode defaults to OFF."""
        mode, analysis = resolve_ocr_mode("invalid_mode", config=default_config)
        
        assert mode == OCRIngestionMode.OFF
        assert analysis is None
    
    def test_case_insensitive_mode(self, default_config):
        """Test that mode resolution is case insensitive."""
        mode1, _ = resolve_ocr_mode("QWEN", config=default_config)
        mode2, _ = resolve_ocr_mode("Qwen", config=default_config)
        mode3, _ = resolve_ocr_mode("qwen", config=default_config)
        
        assert mode1 == mode2 == mode3 == OCRIngestionMode.QWEN
    
    # --- Backward compatibility tests ---
    
    def test_backward_compat_deepseek_ocr_enabled(self, default_config):
        """Test backward compatibility with rag_deepseek_ocr_enabled."""
        default_config.rag_deepseek_ocr_enabled = True
        
        mode, analysis = resolve_ocr_mode("auto", config=default_config)
        
        # Legacy setting should take precedence
        assert mode == OCRIngestionMode.DEEPSEEK
        assert analysis is None


# ═══════════════════════════════════════════════════════════════════════════════
# TEST OCRProcessor MODE SUPPORT
# ═══════════════════════════════════════════════════════════════════════════════

class TestOCRProcessorModeSupport:
    """Tests for OCRProcessor mode selection support."""
    
    @pytest.fixture
    def default_config(self):
        """Create a default RAGConfig for testing."""
        return RAGConfig()
    
    def test_create_for_mode_qwen(self, default_config):
        """Test creating processor for QWEN mode."""
        processor = OCRProcessor.create_for_mode(
            OCRIngestionMode.QWEN,
            settings=default_config,
        )
        
        assert processor.ingestion_mode == OCRIngestionMode.QWEN
        assert processor.endpoint == default_config.rag_ocr_api_base
        assert processor.model == default_config.rag_ocr_model
    
    def test_create_for_mode_deepseek(self, default_config):
        """Test creating processor for DEEPSEEK mode."""
        processor = OCRProcessor.create_for_mode(
            OCRIngestionMode.DEEPSEEK,
            settings=default_config,
        )
        
        assert processor.ingestion_mode == OCRIngestionMode.DEEPSEEK
        assert processor.endpoint == default_config.rag_deepseek_ocr_api_base
        assert processor.model == default_config.rag_deepseek_ocr_model
    
    def test_create_for_mode_off(self, default_config):
        """Test creating processor for OFF mode."""
        processor = OCRProcessor.create_for_mode(
            OCRIngestionMode.OFF,
            settings=default_config,
        )
        
        assert processor.ingestion_mode == OCRIngestionMode.OFF
    
    @pytest.mark.asyncio
    async def test_process_image_off_mode_skips_ocr(self, default_config):
        """Test that OFF mode skips OCR processing."""
        processor = OCRProcessor.create_for_mode(
            OCRIngestionMode.OFF,
            settings=default_config,
        )
        
        # Dummy image data
        image_data = b"fake image data"
        
        result = await processor.process_image(image_data)
        
        assert result.text == ""
        assert result.mode_used == "off"
        assert result.metadata.get("skipped") is True


# ═══════════════════════════════════════════════════════════════════════════════
# TEST CONFIGURATION INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestConfigurationIntegration:
    """Tests for configuration integration."""
    
    def test_config_has_ocr_mode_field(self):
        """Test that RAGConfig has rag_ocr_mode field."""
        config = RAGConfig()
        
        assert hasattr(config, "rag_ocr_mode")
        assert config.rag_ocr_mode == "auto"  # default
    
    def test_config_has_ocr_auto_preferred_field(self):
        """Test that RAGConfig has rag_ocr_auto_preferred field."""
        config = RAGConfig()
        
        assert hasattr(config, "rag_ocr_auto_preferred")
        assert config.rag_ocr_auto_preferred == "qwen"  # default
    
    def test_config_has_ocr_auto_threshold_field(self):
        """Test that RAGConfig has rag_ocr_auto_threshold field."""
        config = RAGConfig()
        
        assert hasattr(config, "rag_ocr_auto_threshold")
        assert config.rag_ocr_auto_threshold == 0.3  # default
    
    def test_config_accepts_custom_values(self):
        """Test that config accepts custom values."""
        # This would normally be set via environment variables
        config = RAGConfig()
        config.rag_ocr_mode = "deepseek"
        config.rag_ocr_auto_preferred = "deepseek"
        config.rag_ocr_auto_threshold = 0.5
        
        assert config.rag_ocr_mode == "deepseek"
        assert config.rag_ocr_auto_preferred == "deepseek"
        assert config.rag_ocr_auto_threshold == 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# TEST DocumentOCRAnalysis DATACLASS
# ═══════════════════════════════════════════════════════════════════════════════

class TestDocumentOCRAnalysis:
    """Tests for DocumentOCRAnalysis dataclass."""
    
    def test_str_representation(self):
        """Test string representation of analysis."""
        analysis = DocumentOCRAnalysis(
            recommended_mode=OCRIngestionMode.QWEN,
            confidence=0.95,
            reasons=["Image format", "No text content"],
            scores={"file_type": 1.0, "text_density": 1.0},
        )
        
        str_repr = str(analysis)
        
        assert "qwen" in str_repr
        assert "0.95" in str_repr
        assert "Image format" in str_repr
    
    def test_default_scores(self):
        """Test that scores default to empty dict."""
        analysis = DocumentOCRAnalysis(
            recommended_mode=OCRIngestionMode.OFF,
            confidence=1.0,
            reasons=["Text file"],
        )
        
        assert analysis.scores == {}
