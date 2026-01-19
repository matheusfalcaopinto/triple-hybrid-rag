"""
End-to-end tests for OCR Ingestion Mode functionality.

Tests the full integration of OCR mode selection through the ingestion pipeline.
"""

import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from triple_hybrid_rag.config import RAGConfig
from triple_hybrid_rag.ingestion.ingest import Ingestor, IngestStats
from triple_hybrid_rag.ingestion.ocr import (
    OCRIngestionMode,
    OCRProcessor,
    OCRResult,
    resolve_ocr_mode,
)

# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def config():
    """Create a test configuration."""
    cfg = RAGConfig()
    cfg.rag_ocr_mode = "auto"
    cfg.rag_ocr_auto_preferred = "qwen"
    cfg.rag_ocr_auto_threshold = 0.3
    return cfg


@pytest.fixture
def mock_embedder():
    """Create a mock embedder that returns fake embeddings."""
    embedder = MagicMock()
    embedder.embed_texts = AsyncMock(return_value=[[0.1] * 1024])
    return embedder


@pytest.fixture
def mock_ocr_processor():
    """Create a mock OCR processor."""
    processor = MagicMock(spec=OCRProcessor)
    processor.process_image = AsyncMock(
        return_value=OCRResult(
            text="OCR extracted text",
            confidence=0.9,
            has_tables=False,
            tables=[],
            mode_used="qwen3-vl",
        )
    )
    return processor


# ═══════════════════════════════════════════════════════════════════════════════
# TEST OCR MODE SELECTION IN INGESTION
# ═══════════════════════════════════════════════════════════════════════════════

class TestOCRModeInIngestion:
    """Test OCR mode selection during document ingestion."""
    
    @pytest.mark.asyncio
    async def test_text_file_auto_mode_skips_ocr(self, config, mock_embedder):
        """Test that AUTO mode skips OCR for text files."""
        config.rag_ocr_mode = "auto"
        
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
            f.write("This is a plain text file with plenty of content.\n" * 50)
            txt_path = f.name
        
        try:
            # Create a mock loader that returns a text document
            mock_loader = MagicMock()
            mock_document = MagicMock()
            mock_document.file_path = txt_path
            mock_document.pages = [
                MagicMock(
                    text="This is a plain text file with plenty of content.\n" * 50,
                    page_number=1,
                    is_scanned=False,
                    has_images=False,
                    image_data=None,
                )
            ]
            mock_document.error = None
            mock_loader.load.return_value = mock_document
            
            # Create ingestor with mocks
            ingestor = Ingestor(
                config=config,
                loader=mock_loader,
                embedder=mock_embedder,
            )
            
            # Mock database storage
            with patch.object(ingestor, '_store_chunks', new_callable=AsyncMock) as mock_store:
                mock_store.return_value = (["p1"], ["c1"])
                
                result = await ingestor.ingest_file(txt_path)
                
                # OCR should be skipped for text files
                assert result.stats.ocr_pages_processed == 0
        finally:
            Path(txt_path).unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_off_mode_never_calls_ocr(self, config, mock_embedder, mock_ocr_processor):
        """Test that OFF mode never calls OCR processor."""
        config.rag_ocr_mode = "off"
        
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False, mode="wb") as f:
            f.write(b"fake pdf content")
            pdf_path = f.name
        
        try:
            # Create a mock loader that returns a scanned document
            mock_loader = MagicMock()
            mock_document = MagicMock()
            mock_document.file_path = pdf_path
            mock_document.pages = [
                MagicMock(
                    text="",
                    page_number=1,
                    is_scanned=True,
                    has_images=True,
                    image_data=b"fake image data",
                )
            ]
            mock_document.error = None
            mock_loader.load.return_value = mock_document
            
            ingestor = Ingestor(
                config=config,
                loader=mock_loader,
                embedder=mock_embedder,
                ocr_processor=mock_ocr_processor,
            )
            
            with patch.object(ingestor, '_store_chunks', new_callable=AsyncMock) as mock_store:
                mock_store.return_value = (["p1"], ["c1"])
                
                result = await ingestor.ingest_file(pdf_path)
                
                # OCR should be skipped even for scanned pages when mode is OFF
                assert result.stats.ocr_pages_processed == 0
        finally:
            Path(pdf_path).unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_qwen_mode_uses_qwen_processor(self, config, mock_embedder):
        """Test that QWEN mode creates processor with Qwen settings."""
        config.rag_ocr_mode = "qwen"
        
        # Resolve the mode
        mode, _ = resolve_ocr_mode("qwen", config=config)
        
        assert mode == OCRIngestionMode.QWEN
        
        # Create processor for this mode
        processor = OCRProcessor.create_for_mode(mode, settings=config)
        
        assert processor.ingestion_mode == OCRIngestionMode.QWEN
        assert processor.endpoint == config.rag_ocr_api_base
        assert processor.model == config.rag_ocr_model
    
    @pytest.mark.asyncio
    async def test_deepseek_mode_uses_deepseek_processor(self, config, mock_embedder):
        """Test that DEEPSEEK mode creates processor with DeepSeek settings."""
        config.rag_ocr_mode = "deepseek"
        
        mode, _ = resolve_ocr_mode("deepseek", config=config)
        
        assert mode == OCRIngestionMode.DEEPSEEK
        
        processor = OCRProcessor.create_for_mode(mode, settings=config)
        
        assert processor.ingestion_mode == OCRIngestionMode.DEEPSEEK
        assert processor.endpoint == config.rag_deepseek_ocr_api_base
        assert processor.model == config.rag_deepseek_ocr_model


# ═══════════════════════════════════════════════════════════════════════════════
# TEST AUTO MODE DECISION MAKING
# ═══════════════════════════════════════════════════════════════════════════════

class TestAutoModeDecisions:
    """Test AUTO mode decision making for different file types."""
    
    def test_auto_mode_text_file_returns_off(self, config):
        """Test that AUTO mode returns OFF for text files."""
        mode, analysis = resolve_ocr_mode(
            "auto",
            file_path="document.txt",
            config=config,
        )
        
        assert mode == OCRIngestionMode.OFF
        assert analysis is not None
        assert analysis.confidence == 1.0
    
    def test_auto_mode_markdown_file_returns_off(self, config):
        """Test that AUTO mode returns OFF for markdown files."""
        mode, analysis = resolve_ocr_mode(
            "auto",
            file_path="README.md",
            config=config,
        )
        
        assert mode == OCRIngestionMode.OFF
    
    def test_auto_mode_python_file_returns_off(self, config):
        """Test that AUTO mode returns OFF for Python files."""
        mode, analysis = resolve_ocr_mode(
            "auto",
            file_path="script.py",
            config=config,
        )
        
        assert mode == OCRIngestionMode.OFF
    
    def test_auto_mode_json_file_returns_off(self, config):
        """Test that AUTO mode returns OFF for JSON files."""
        mode, analysis = resolve_ocr_mode(
            "auto",
            file_path="data.json",
            config=config,
        )
        
        assert mode == OCRIngestionMode.OFF
    
    def test_auto_mode_png_file_returns_qwen(self, config):
        """Test that AUTO mode returns QWEN (preferred) for PNG files."""
        config.rag_ocr_auto_preferred = "qwen"
        
        mode, analysis = resolve_ocr_mode(
            "auto",
            file_path="scan.png",
            config=config,
        )
        
        assert mode == OCRIngestionMode.QWEN
        assert analysis is not None
        assert analysis.confidence == 1.0
    
    def test_auto_mode_jpeg_file_returns_preferred_provider(self, config):
        """Test that AUTO mode returns preferred provider for JPEG files."""
        config.rag_ocr_auto_preferred = "deepseek"
        
        mode, analysis = resolve_ocr_mode(
            "auto",
            file_path="photo.jpg",
            config=config,
        )
        
        assert mode == OCRIngestionMode.DEEPSEEK
    
    def test_auto_mode_tiff_file_returns_ocr(self, config):
        """Test that AUTO mode returns OCR for TIFF files."""
        mode, analysis = resolve_ocr_mode(
            "auto",
            file_path="scan.tiff",
            config=config,
        )
        
        assert mode in (OCRIngestionMode.QWEN, OCRIngestionMode.DEEPSEEK)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST BACKWARD COMPATIBILITY
# ═══════════════════════════════════════════════════════════════════════════════

class TestBackwardCompatibility:
    """Test backward compatibility with legacy settings."""
    
    def test_legacy_deepseek_ocr_enabled_takes_precedence(self, config):
        """Test that rag_deepseek_ocr_enabled=True overrides auto mode."""
        config.rag_ocr_mode = "auto"
        config.rag_deepseek_ocr_enabled = True
        
        mode, analysis = resolve_ocr_mode("auto", config=config)
        
        assert mode == OCRIngestionMode.DEEPSEEK
        assert analysis is None  # No analysis needed for legacy override
    
    def test_legacy_disabled_allows_auto_mode(self, config):
        """Test that rag_deepseek_ocr_enabled=False allows AUTO mode."""
        config.rag_ocr_mode = "auto"
        config.rag_deepseek_ocr_enabled = False
        
        mode, analysis = resolve_ocr_mode(
            "auto",
            file_path="scan.png",
            config=config,
        )
        
        # Should use preferred provider (qwen by default)
        assert mode == OCRIngestionMode.QWEN
        assert analysis is not None


# ═══════════════════════════════════════════════════════════════════════════════
# TEST CLI INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestCLIIntegration:
    """Test CLI argument integration."""
    
    def test_cli_ocr_mode_choices(self):
        """Test that CLI parser accepts valid OCR mode choices."""
        from triple_hybrid_rag.cli import _build_parser
        
        parser = _build_parser()
        
        # Test valid modes
        for mode in ["qwen", "deepseek", "off", "auto"]:
            args = parser.parse_args(["ingest", "test.txt", "--ocr-mode", mode])
            assert args.ocr_mode == mode
    
    def test_cli_ocr_mode_default_is_none(self):
        """Test that CLI ocr-mode defaults to None (use config)."""
        from triple_hybrid_rag.cli import _build_parser
        
        parser = _build_parser()
        args = parser.parse_args(["ingest", "test.txt"])
        
        assert args.ocr_mode is None


# ═══════════════════════════════════════════════════════════════════════════════
# TEST PROCESSOR BEHAVIOR
# ═══════════════════════════════════════════════════════════════════════════════

class TestProcessorBehavior:
    """Test OCRProcessor behavior with different modes."""
    
    @pytest.mark.asyncio
    async def test_off_mode_returns_empty_result(self, config):
        """Test that OFF mode returns empty OCR result."""
        processor = OCRProcessor.create_for_mode(
            OCRIngestionMode.OFF,
            settings=config,
        )
        
        result = await processor.process_image(b"fake image data")
        
        assert result.text == ""
        assert result.mode_used == "off"
        assert result.metadata.get("skipped") is True
        assert result.metadata.get("reason") == "OCR mode is OFF"
    
    def test_qwen_mode_uses_correct_endpoint(self, config):
        """Test that QWEN mode configures correct endpoint."""
        processor = OCRProcessor.create_for_mode(
            OCRIngestionMode.QWEN,
            settings=config,
        )
        
        assert processor.endpoint == config.rag_ocr_api_base
        assert processor.model == config.rag_ocr_model
    
    def test_deepseek_mode_uses_correct_endpoint(self, config):
        """Test that DEEPSEEK mode configures correct endpoint."""
        processor = OCRProcessor.create_for_mode(
            OCRIngestionMode.DEEPSEEK,
            settings=config,
        )
        
        assert processor.endpoint == config.rag_deepseek_ocr_api_base
        assert processor.model == config.rag_deepseek_ocr_model


# ═══════════════════════════════════════════════════════════════════════════════
# TEST ENVIRONMENT VARIABLE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestEnvironmentConfiguration:
    """Test configuration via environment variables."""
    
    def test_config_defaults(self):
        """Test default configuration values."""
        config = RAGConfig()
        
        assert config.rag_ocr_mode == "auto"
        assert config.rag_ocr_auto_preferred == "qwen"
        assert config.rag_ocr_auto_threshold == 0.3
    
    def test_config_can_be_modified(self):
        """Test that configuration can be modified."""
        config = RAGConfig()
        config.rag_ocr_mode = "deepseek"
        config.rag_ocr_auto_preferred = "deepseek"
        config.rag_ocr_auto_threshold = 0.5
        
        assert config.rag_ocr_mode == "deepseek"
        assert config.rag_ocr_auto_preferred == "deepseek"
        assert config.rag_ocr_auto_threshold == 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# TEST COMPLETE INGESTION FLOW
# ═══════════════════════════════════════════════════════════════════════════════

class TestCompleteIngestionFlow:
    """Test complete ingestion flow with OCR mode."""
    
    @pytest.mark.asyncio
    async def test_ingestion_with_auto_mode_text_file(self, config, mock_embedder):
        """Test complete ingestion of text file with AUTO mode."""
        config.rag_ocr_mode = "auto"
        
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
            content = "This is test content for ingestion.\n" * 100
            f.write(content)
            txt_path = f.name
        
        try:
            mock_loader = MagicMock()
            mock_document = MagicMock()
            mock_document.file_path = txt_path
            mock_document.pages = [
                MagicMock(
                    text=content,
                    page_number=1,
                    is_scanned=False,
                    has_images=False,
                    image_data=None,
                )
            ]
            mock_document.error = None
            mock_loader.load.return_value = mock_document
            
            ingestor = Ingestor(
                config=config,
                loader=mock_loader,
                embedder=mock_embedder,
            )
            
            with patch.object(ingestor, '_store_chunks', new_callable=AsyncMock) as mock_store:
                mock_store.return_value = (["p1"], ["c1", "c2", "c3"])
                
                result = await ingestor.ingest_file(txt_path)
                
                assert result.success
                assert result.stats.ocr_pages_processed == 0
                assert result.stats.documents_processed == 1
        finally:
            Path(txt_path).unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_ingestion_respects_mode_override(self, config, mock_embedder):
        """Test that OCR mode override parameter is respected."""
        config.rag_ocr_mode = "qwen"  # Default config
        
        # The _process_ocr method accepts ocr_mode_override parameter
        # This test verifies the mode resolution respects the override
        
        mode1, _ = resolve_ocr_mode("qwen", config=config)
        assert mode1 == OCRIngestionMode.QWEN
        
        mode2, _ = resolve_ocr_mode("off", config=config)
        assert mode2 == OCRIngestionMode.OFF
        
        mode3, _ = resolve_ocr_mode("deepseek", config=config)
        assert mode3 == OCRIngestionMode.DEEPSEEK
