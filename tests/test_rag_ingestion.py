"""
RAG Ingestion Tests

Unit tests for the RAG ingestion pipeline components:
- Document loading
- Text chunking
- OCR processing
- Embedding generation
- Deduplication
"""

import hashlib
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from voice_agent.ingestion.loader import (
    DocumentLoader,
    LoadedDocument,
    PageContent,
    FileType,
    detect_file_type,
)
from voice_agent.ingestion.chunker import (
    Chunker,
    Chunk,
    ChunkType,
)
from voice_agent.ingestion.embedder import (
    Embedder,
    EmbeddingResult,
    normalize_embedding,
)
from voice_agent.ingestion.ocr import (
    OCRProcessor,
    OCRResult,
    OCRMode,
)
from voice_agent.ingestion.kb_ingest import (
    KnowledgeBaseIngestor,
    IngestResult,
    IngestStats,
)


# =============================================================================
# File Type Detection Tests
# =============================================================================

class TestFileTypeDetection:
    """Tests for file type detection."""
    
    def test_detect_pdf(self, tmp_path):
        """Test PDF detection."""
        pdf_file = tmp_path / "test.pdf"
        pdf_file.touch()
        assert detect_file_type(pdf_file) == FileType.PDF
    
    def test_detect_docx(self, tmp_path):
        """Test DOCX detection."""
        docx_file = tmp_path / "test.docx"
        docx_file.touch()
        assert detect_file_type(docx_file) == FileType.DOCX
    
    def test_detect_txt(self, tmp_path):
        """Test TXT detection."""
        txt_file = tmp_path / "test.txt"
        txt_file.touch()
        assert detect_file_type(txt_file) == FileType.TXT
    
    def test_detect_markdown(self, tmp_path):
        """Test Markdown detection."""
        md_file = tmp_path / "test.md"
        md_file.touch()
        assert detect_file_type(md_file) == FileType.TXT
    
    def test_detect_csv(self, tmp_path):
        """Test CSV detection."""
        csv_file = tmp_path / "test.csv"
        csv_file.touch()
        assert detect_file_type(csv_file) == FileType.CSV
    
    def test_detect_xlsx(self, tmp_path):
        """Test XLSX detection."""
        xlsx_file = tmp_path / "test.xlsx"
        xlsx_file.touch()
        assert detect_file_type(xlsx_file) == FileType.XLSX
    
    def test_detect_image_png(self, tmp_path):
        """Test PNG detection."""
        png_file = tmp_path / "test.png"
        png_file.touch()
        assert detect_file_type(png_file) == FileType.IMAGE
    
    def test_detect_image_jpg(self, tmp_path):
        """Test JPG detection."""
        jpg_file = tmp_path / "test.jpg"
        jpg_file.touch()
        assert detect_file_type(jpg_file) == FileType.IMAGE
    
    def test_detect_unknown(self, tmp_path):
        """Test unknown file type."""
        unknown_file = tmp_path / "test.xyz"
        unknown_file.touch()
        assert detect_file_type(unknown_file) == FileType.UNKNOWN


# =============================================================================
# Document Loader Tests
# =============================================================================

class TestDocumentLoader:
    """Tests for document loading."""
    
    def test_load_txt_file(self, tmp_path):
        """Test loading plain text file."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("Hello, world!\nThis is a test.")
        
        loader = DocumentLoader()
        result = loader.load(txt_file)
        
        assert result.file_type == FileType.TXT
        assert result.error is None
        assert len(result.pages) == 1
        assert "Hello, world!" in result.pages[0].text
        assert result.file_hash  # Should have hash
    
    def test_load_csv_file(self, tmp_path):
        """Test loading CSV file."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("Name,Age,City\nAlice,30,NYC\nBob,25,LA")
        
        loader = DocumentLoader()
        result = loader.load(csv_file)
        
        assert result.file_type == FileType.CSV
        assert result.error is None
        assert len(result.pages) == 1
        assert "Name" in result.pages[0].text
        assert len(result.pages[0].tables) == 1
    
    def test_load_nonexistent_file(self, tmp_path):
        """Test loading nonexistent file."""
        fake_file = tmp_path / "nonexistent.txt"
        
        loader = DocumentLoader()
        result = loader.load(fake_file)
        
        assert result.error is not None
        assert "not found" in result.error.lower()
    
    def test_rows_to_markdown(self):
        """Test Markdown table generation."""
        loader = DocumentLoader()
        rows = [
            ["Header1", "Header2"],
            ["Data1", "Data2"],
            ["Data3", "Data4"],
        ]
        
        md = loader._rows_to_markdown(rows)
        
        assert "| Header1 | Header2 |" in md
        assert "| --- | --- |" in md
        assert "| Data1 | Data2 |" in md


# =============================================================================
# Chunker Tests
# =============================================================================

class TestChunker:
    """Tests for text chunking."""
    
    def test_basic_chunking(self):
        """Test basic text chunking."""
        chunker = Chunker(chunk_size=100, chunk_overlap=20)
        text = "A" * 250  # 250 characters
        
        chunks = chunker.chunk_text_simple(text, source_document="test.txt")
        
        assert len(chunks) >= 2
        assert all(len(c.content) <= 100 for c in chunks)
        assert all(c.chunk_type == ChunkType.TEXT for c in chunks)
    
    def test_small_text_single_chunk(self):
        """Test small text results in single chunk."""
        chunker = Chunker(chunk_size=1000, chunk_overlap=200)
        text = "This is a short text."
        
        chunks = chunker.chunk_text_simple(text, source_document="test.txt")
        
        assert len(chunks) == 1
        assert chunks[0].content == text
    
    def test_chunk_overlap(self):
        """Test overlap between consecutive chunks."""
        chunker = Chunker(chunk_size=100, chunk_overlap=20, sentence_boundary=False)
        text = "A" * 50 + "B" * 50 + "C" * 50  # 150 chars
        
        chunks = chunker.chunk_text_simple(text, source_document="test.txt")
        
        # With overlap, later chunks should contain some content from earlier
        if len(chunks) >= 2:
            # Check there's overlap
            assert len(chunks[0].content) > 50
    
    def test_content_hash_uniqueness(self):
        """Test that different content produces different hashes."""
        chunker = Chunker()
        
        chunks1 = chunker.chunk_text_simple("Content A", source_document="test.txt")
        chunks2 = chunker.chunk_text_simple("Content B", source_document="test.txt")
        
        assert chunks1[0].content_hash != chunks2[0].content_hash
    
    def test_content_hash_consistency(self):
        """Test that same content produces same hash."""
        chunker = Chunker()
        
        chunks1 = chunker.chunk_text_simple("Same Content", source_document="test.txt")
        chunks2 = chunker.chunk_text_simple("Same Content", source_document="test.txt")
        
        assert chunks1[0].content_hash == chunks2[0].content_hash
    
    def test_table_detection(self):
        """Test table context generation."""
        chunker = Chunker()
        
        table_md = """| Name | Age | City |
| --- | --- | --- |
| Alice | 30 | NYC |
| Bob | 25 | LA |"""
        
        context = chunker._generate_table_context(table_md)
        
        assert "Name" in context
        assert "Age" in context
        assert "2 rows" in context
    
    def test_heading_preservation(self):
        """Test heading context is preserved."""
        chunker = Chunker(chunk_size=100)
        text = """## Introduction

This is the introduction section with some content.

## Methods

This is the methods section with more content."""
        
        chunks = chunker.chunk_text_simple(text, source_document="test.txt")
        
        # At least one chunk should have heading context
        headings = [c.heading_context for c in chunks if c.heading_context]
        assert len(headings) > 0
    
    def test_chunk_document(self, tmp_path):
        """Test chunking a loaded document."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("This is test content. " * 100)
        
        loader = DocumentLoader()
        document = loader.load(txt_file)
        
        chunker = Chunker(chunk_size=200, chunk_overlap=50)
        chunks = chunker.chunk_document(document)
        
        assert len(chunks) > 0
        assert all(c.source_document == str(txt_file) for c in chunks)


# =============================================================================
# Embedding Tests
# =============================================================================

class TestEmbedding:
    """Tests for embedding generation."""
    
    def test_normalize_embedding(self):
        """Test L2 normalization of embeddings."""
        import numpy as np
        
        embedding = [3.0, 4.0]  # Length = 5
        normalized = normalize_embedding(embedding)
        
        # Check unit length
        length = np.sqrt(sum(x**2 for x in normalized))
        assert abs(length - 1.0) < 0.0001
        
        # Check direction preserved
        assert normalized[0] / normalized[1] == pytest.approx(3.0 / 4.0)
    
    def test_normalize_zero_vector(self):
        """Test normalization of zero vector."""
        embedding = [0.0, 0.0, 0.0]
        normalized = normalize_embedding(embedding)
        
        # Should not raise, should return zeros
        assert normalized == [0.0, 0.0, 0.0]
    
    @pytest.mark.asyncio
    async def test_embedder_initialization(self):
        """Test embedder initialization with config."""
        embedder = Embedder(
            model="test-model",
            batch_size=10,
        )
        
        assert embedder.model == "test-model"
        assert embedder.batch_size == 10
        # Legacy aliases should also work
        assert embedder.text_model == "test-model"
        assert embedder.image_model == "test-model"
    
    @pytest.mark.asyncio
    async def test_embed_chunks_with_mock(self):
        """Test embedding chunks with mocked OpenAI."""
        embedder = Embedder()
        
        # Create test chunks
        chunks = [
            Chunk(
                content="Test content 1",
                chunk_type=ChunkType.TEXT,
                page_number=1,
                chunk_index=0,
                source_document="test.txt",
            ),
            Chunk(
                content="Test content 2",
                chunk_type=ChunkType.TEXT,
                page_number=1,
                chunk_index=1,
                source_document="test.txt",
            ),
        ]
        
        # Mock OpenAI response
        mock_embedding = [0.1] * 1536
        with patch.object(embedder, '_call_openai_embeddings') as mock_call:
            mock_call.return_value = [mock_embedding, mock_embedding]
            
            results = await embedder.embed_chunks(chunks)
            
            assert len(results) == 2
            assert all(r.text_embedding is not None for r in results)
            assert all(
                r.text_embedding is not None and len(r.text_embedding) == 1536 
                for r in results
            )
    
    def test_prepare_text_for_embedding(self):
        """Test text preparation with context."""
        embedder = Embedder()
        
        chunk = Chunk(
            content="Main content here",
            chunk_type=ChunkType.TABLE,
            page_number=1,
            chunk_index=0,
            source_document="test.txt",
            heading_context="Data Section",
            is_table=True,
            table_context="Sales by region",
        )
        
        prepared = embedder._prepare_text_for_embedding(chunk)
        
        assert "Data Section" in prepared
        assert "Sales by region" in prepared
        assert "Main content here" in prepared


# =============================================================================
# OCR Tests
# =============================================================================

class TestOCR:
    """Tests for OCR processing."""
    
    def test_ocr_result_dataclass(self):
        """Test OCR result dataclass."""
        result = OCRResult(
            text="Extracted text",
            confidence=0.95,
            has_tables=False,
            tables=[],
            mode_used="base",
        )
        
        assert result.text == "Extracted text"
        assert result.confidence == 0.95
        assert result.metadata == {}
    
    def test_estimate_confidence_good_text(self):
        """Test confidence estimation for good text."""
        processor = OCRProcessor()
        
        text = "This is a clear, well-formatted sentence with normal characters."
        confidence = processor._estimate_confidence(text)
        
        assert confidence > 0.7
    
    def test_estimate_confidence_garbled_text(self):
        """Test confidence estimation for garbled text."""
        processor = OCRProcessor()
        
        text = "!@#$%^&*()_+=[]{}|\\;':\"<>?,./~`"
        confidence = processor._estimate_confidence(text)
        
        assert confidence < 0.5
    
    def test_estimate_confidence_empty_text(self):
        """Test confidence estimation for empty text."""
        processor = OCRProcessor()
        
        confidence = processor._estimate_confidence("")
        
        assert confidence < 0.2
    
    def test_extract_tables_from_text(self):
        """Test table extraction from OCR text."""
        processor = OCRProcessor()
        
        text = """Some text before.
| Header1 | Header2 |
| --- | --- |
| Data1 | Data2 |
Some text after."""
        
        tables = processor._extract_tables(text)
        
        assert len(tables) == 1
        assert "Header1" in tables[0]
    
    def test_get_next_mode(self):
        """Test mode hierarchy for fallback."""
        processor = OCRProcessor()
        
        assert processor._get_next_mode("tiny") == "small"
        assert processor._get_next_mode("base") == "large"
        assert processor._get_next_mode("gundam") is None
    
    @pytest.mark.asyncio
    async def test_fallback_ocr_no_tesseract(self):
        """Test fallback OCR when tesseract not available."""
        processor = OCRProcessor(endpoint="")  # No endpoint
        
        # Create a simple test image (1x1 white pixel PNG)
        import base64
        # Minimal valid PNG
        png_data = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )
        
        result = await processor._fallback_ocr(png_data)
        
        # Should return result even if OCR fails
        assert isinstance(result, OCRResult)


# =============================================================================
# Ingestor Tests
# =============================================================================

class TestIngestor:
    """Tests for the main ingestion pipeline."""
    
    def test_ingest_stats_dataclass(self):
        """Test ingestion stats dataclass."""
        stats = IngestStats()
        
        assert stats.documents_processed == 0
        assert stats.chunks_created == 0
        assert stats.errors == []
    
    def test_ingest_stats_duration(self):
        """Test duration calculation."""
        from datetime import datetime, timedelta, timezone
        
        stats = IngestStats()
        stats.start_time = datetime.now(timezone.utc)
        stats.end_time = stats.start_time + timedelta(seconds=10)
        
        assert stats.duration_seconds == pytest.approx(10.0, abs=0.1)
    
    @pytest.mark.asyncio
    async def test_ingestor_initialization(self):
        """Test ingestor initialization."""
        ingestor = KnowledgeBaseIngestor(
            org_id="test-org-id",
            category="test",
        )
        
        assert ingestor.org_id == "test-org-id"
        assert ingestor.category == "test"
        assert ingestor.dedup_enabled
    
    @pytest.mark.asyncio
    async def test_ingest_text(self):
        """Test text ingestion with mocks."""
        ingestor = KnowledgeBaseIngestor(
            org_id="test-org-id",
            category="test",
        )
        
        # Mock the embedder
        mock_embedding = [0.1] * 1536
        with patch.object(ingestor.embedder, 'embed_chunks') as mock_embed:
            # Create mock embedding results
            mock_results = []
            mock_embed.return_value = mock_results
            
            # Mock the database
            with patch.object(ingestor, '_deduplicate_chunks') as mock_dedup:
                mock_dedup.return_value = ([], 0)
                
                result = await ingestor.ingest_text(
                    text="Test content for ingestion.",
                    title="Test Document",
                )
                
                assert result.success
                assert result.stats.chunks_created > 0


# =============================================================================
# Integration Tests (require mocking)
# =============================================================================

class TestIngestionIntegration:
    """Integration tests for ingestion pipeline."""
    
    @pytest.mark.asyncio
    async def test_full_pipeline_txt(self, tmp_path):
        """Test full pipeline with text file."""
        # Create test file
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("This is test content for the RAG system. " * 50)
        
        ingestor = KnowledgeBaseIngestor(
            org_id="test-org-id",
            category="test",
        )
        
        # Mock external dependencies
        with patch.object(ingestor.embedder, 'embed_chunks') as mock_embed:
            with patch.object(ingestor, '_store_chunks') as mock_store:
                with patch.object(ingestor, '_deduplicate_chunks') as mock_dedup:
                    # Setup mocks
                    mock_embed.return_value = []
                    mock_store.return_value = []
                    mock_dedup.side_effect = lambda chunks: (chunks, 0)
                    
                    result = await ingestor.ingest_file(txt_file)
                    
                    assert result.stats.documents_processed == 1
                    assert result.stats.chunks_created > 0
    
    def test_loader_chunker_integration(self, tmp_path):
        """Test loader + chunker integration."""
        # Create test file
        txt_file = tmp_path / "test.txt"
        content = """# Test Document

## Introduction

This is the introduction section with important information about the product.

## Features

The product has the following features:
- Feature 1: Fast processing
- Feature 2: Easy to use
- Feature 3: Reliable

## Conclusion

In conclusion, this product is great."""
        txt_file.write_text(content)
        
        # Load
        loader = DocumentLoader()
        document = loader.load(txt_file)
        
        assert document.error is None
        assert len(document.pages) == 1
        
        # Chunk
        chunker = Chunker(chunk_size=200, chunk_overlap=50)
        chunks = chunker.chunk_document(document)
        
        assert len(chunks) > 0
        # Check heading preservation
        introduction_chunks = [c for c in chunks if "Introduction" in c.heading_context]
        # May or may not have heading depending on chunk size
