"""
Tests for RAG2 Ingestion Pipeline

Tests ingestion workflow including retry logic, entity extraction,
and error handling.
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass, field
from typing import List, Optional
import httpx


# ──────────────────────────────────────────────────────────────────────────────
# Test Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class MockExtractionResult:
    """Mock extraction result for testing."""
    entities: List = field(default_factory=list)
    relations: List = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class TestIngestStats:
    """Test IngestStats dataclass."""
    
    def test_stats_creation(self) -> None:
        """Test IngestStats can be created."""
        from voice_agent.rag2.ingest import IngestStats
        
        stats = IngestStats()
        
        assert stats.documents_registered == 0
        assert stats.documents_skipped == 0
        assert stats.parent_chunks_created == 0
        assert stats.child_chunks_created == 0
        assert stats.entities_extracted == 0
        assert stats.relations_extracted == 0
        assert stats.errors == []
    
    def test_stats_duration(self) -> None:
        """Test duration calculation."""
        from voice_agent.rag2.ingest import IngestStats
        from datetime import datetime, timezone, timedelta
        
        stats = IngestStats()
        stats.start_time = datetime.now(timezone.utc)
        stats.end_time = stats.start_time + timedelta(seconds=5)
        
        assert stats.duration_seconds == pytest.approx(5.0)
    
    def test_stats_duration_none(self) -> None:
        """Test duration when times not set."""
        from voice_agent.rag2.ingest import IngestStats
        
        stats = IngestStats()
        
        assert stats.duration_seconds == 0.0


class TestIngestResult:
    """Test IngestResult dataclass."""
    
    def test_result_creation_success(self) -> None:
        """Test successful IngestResult."""
        from voice_agent.rag2.ingest import IngestResult, IngestStats
        
        result = IngestResult(
            success=True,
            document_id="doc_123",
            stats=IngestStats(documents_registered=1),
        )
        
        assert result.success
        assert result.document_id == "doc_123"
        assert result.error is None
    
    def test_result_creation_failure(self) -> None:
        """Test failed IngestResult."""
        from voice_agent.rag2.ingest import IngestResult, IngestStats
        
        result = IngestResult(
            success=False,
            document_id=None,
            stats=IngestStats(errors=["Something went wrong"]),
            error="Failed to ingest",
        )
        
        assert not result.success
        assert result.document_id is None
        assert result.error == "Failed to ingest"


class TestRAG2Ingestor:
    """Test RAG2Ingestor class."""
    
    def test_ingestor_initialization(self) -> None:
        """Test ingestor initializes correctly."""
        from voice_agent.rag2.ingest import RAG2Ingestor
        
        ingestor = RAG2Ingestor(
            org_id="org_123",
            collection="test_collection",
        )
        
        assert ingestor.org_id == "org_123"
        assert ingestor.collection == "test_collection"
        assert ingestor.loader is not None
        assert ingestor.chunker is not None
        assert ingestor.embedder is not None
    
    def test_ingestor_with_entity_extraction_disabled(self) -> None:
        """Test ingestor with entity extraction disabled."""
        from voice_agent.rag2.ingest import RAG2Ingestor
        
        ingestor = RAG2Ingestor(
            org_id="org_123",
            entity_extraction_enabled=False,
        )
        
        assert not ingestor.entity_extraction_enabled
        assert ingestor.entity_extractor is None
    
    def test_compute_file_hash(self) -> None:
        """Test file hash computation."""
        import tempfile
        from pathlib import Path
        from voice_agent.rag2.ingest import RAG2Ingestor
        
        ingestor = RAG2Ingestor(org_id="test")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
            f.write(b"Test content")
            temp_path = Path(f.name)
        
        try:
            hash1 = ingestor._compute_file_hash(temp_path)
            hash2 = ingestor._compute_file_hash(temp_path)
            
            # Same file should produce same hash
            assert hash1 == hash2
            assert len(hash1) == 64  # SHA-256 hex digest
        finally:
            temp_path.unlink()
    
    def test_get_mime_type(self) -> None:
        """Test MIME type detection."""
        from pathlib import Path
        from voice_agent.rag2.ingest import RAG2Ingestor
        
        ingestor = RAG2Ingestor(org_id="test")
        
        assert ingestor._get_mime_type(Path("doc.pdf")) == "application/pdf"
        assert ingestor._get_mime_type(Path("doc.docx")) == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        assert ingestor._get_mime_type(Path("data.csv")) == "text/csv"
        assert ingestor._get_mime_type(Path("file.txt")) == "text/plain"
        assert ingestor._get_mime_type(Path("file.unknown")) == "application/octet-stream"


class TestEntityExtractionRetry:
    """Test entity extraction retry logic."""
    
    @pytest.mark.asyncio
    async def test_extract_entities_for_parent_success(self) -> None:
        """Test successful entity extraction for a single parent."""
        from voice_agent.rag2.ingest import RAG2Ingestor
        
        ingestor = RAG2Ingestor(
            org_id="test-org",
            entity_extraction_enabled=True,
        )
        
        # Mock entity extractor and store
        mock_result = MockExtractionResult(
            entities=[{"name": "Test Entity", "type": "PERSON"}],
            relations=[],
            errors=[],
        )
        
        mock_extractor = AsyncMock()
        mock_extractor.extract.return_value = mock_result
        ingestor.entity_extractor = mock_extractor
        
        mock_store = AsyncMock()
        mock_store.store_extraction.return_value = {
            "entities_created": 1,
            "relations_created": 0,
        }
        ingestor._entity_store = mock_store
        
        result = await ingestor._extract_entities_for_parent(
            parent_id="parent_1",
            parent_text="John Smith works at Acme Corp.",
            parent_context="Introduction",
            document_title="Test Document",
            document_id="doc_1",
            db_parent_id="db_parent_1",
            child_ids=["child_1"],
        )
        
        assert result["entities_created"] == 1
        assert result["relations_created"] == 0
        mock_extractor.extract.assert_called_once()
        mock_store.store_extraction.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_extract_entities_retry_on_http_error(self) -> None:
        """Test that entity extraction retries on HTTP errors."""
        from voice_agent.rag2.ingest import RAG2Ingestor
        
        ingestor = RAG2Ingestor(
            org_id="test-org",
            entity_extraction_enabled=True,
        )
        
        call_count = 0
        
        async def mock_extract(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise httpx.HTTPError("Transient server error")
            return MockExtractionResult(
                entities=[{"name": "Entity", "type": "PERSON"}],
                relations=[],
                errors=[],
            )
        
        mock_extractor = AsyncMock()
        mock_extractor.extract = mock_extract
        ingestor.entity_extractor = mock_extractor
        
        mock_store = AsyncMock()
        mock_store.store_extraction.return_value = {
            "entities_created": 1,
            "relations_created": 0,
        }
        ingestor._entity_store = mock_store
        
        result = await ingestor._extract_entities_for_parent(
            parent_id="parent_1",
            parent_text="Test text",
            parent_context=None,
            document_title="Test",
            document_id="doc_1",
            db_parent_id="db_parent_1",
            child_ids=[],
        )
        
        # Should have retried twice then succeeded
        assert call_count == 3
        assert result["entities_created"] == 1
    
    @pytest.mark.asyncio
    async def test_extract_entities_retry_on_timeout(self) -> None:
        """Test that entity extraction retries on timeout errors."""
        from voice_agent.rag2.ingest import RAG2Ingestor
        
        ingestor = RAG2Ingestor(
            org_id="test-org",
            entity_extraction_enabled=True,
        )
        
        call_count = 0
        
        async def mock_extract(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise TimeoutError("Connection timed out")
            return MockExtractionResult(
                entities=[],
                relations=[],
                errors=[],
            )
        
        mock_extractor = AsyncMock()
        mock_extractor.extract = mock_extract
        ingestor.entity_extractor = mock_extractor
        
        mock_store = AsyncMock()
        mock_store.store_extraction.return_value = {
            "entities_created": 0,
            "relations_created": 0,
        }
        ingestor._entity_store = mock_store
        
        result = await ingestor._extract_entities_for_parent(
            parent_id="parent_1",
            parent_text="Test text",
            parent_context=None,
            document_title="Test",
            document_id="doc_1",
            db_parent_id="db_parent_1",
            child_ids=[],
        )
        
        # Should have retried once then succeeded
        assert call_count == 2
        assert result["entities_created"] == 0
    
    @pytest.mark.asyncio
    async def test_extract_entities_retry_on_connection_error(self) -> None:
        """Test that entity extraction retries on connection errors."""
        from voice_agent.rag2.ingest import RAG2Ingestor
        
        ingestor = RAG2Ingestor(
            org_id="test-org",
            entity_extraction_enabled=True,
        )
        
        call_count = 0
        
        async def mock_extract(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Connection refused")
            return MockExtractionResult(
                entities=[{"name": "Entity", "type": "ORG"}],
                relations=[],
                errors=[],
            )
        
        mock_extractor = AsyncMock()
        mock_extractor.extract = mock_extract
        ingestor.entity_extractor = mock_extractor
        
        mock_store = AsyncMock()
        mock_store.store_extraction.return_value = {
            "entities_created": 1,
            "relations_created": 0,
        }
        ingestor._entity_store = mock_store
        
        result = await ingestor._extract_entities_for_parent(
            parent_id="parent_1",
            parent_text="Test text",
            parent_context=None,
            document_title="Test",
            document_id="doc_1",
            db_parent_id="db_parent_1",
            child_ids=[],
        )
        
        assert call_count == 2
        assert result["entities_created"] == 1
    
    @pytest.mark.asyncio
    async def test_extract_entities_exhausts_retries(self) -> None:
        """Test that entity extraction gives up after max retries."""
        from voice_agent.rag2.ingest import RAG2Ingestor
        
        ingestor = RAG2Ingestor(
            org_id="test-org",
            entity_extraction_enabled=True,
        )
        
        call_count = 0
        
        async def mock_extract(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise httpx.HTTPError("Persistent server error")
        
        mock_extractor = AsyncMock()
        mock_extractor.extract = mock_extract
        ingestor.entity_extractor = mock_extractor
        
        # With reraise=True, the original exception is raised after retries exhausted
        with pytest.raises(httpx.HTTPError):
            await ingestor._extract_entities_for_parent(
                parent_id="parent_1",
                parent_text="Test text",
                parent_context=None,
                document_title="Test",
                document_id="doc_1",
                db_parent_id="db_parent_1",
                child_ids=[],
            )
        
        # Should have tried 3 times
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_extract_entities_does_not_retry_runtime_error(self) -> None:
        """Test that RuntimeError from extraction errors is not retried (not transient)."""
        from voice_agent.rag2.ingest import RAG2Ingestor
        
        ingestor = RAG2Ingestor(
            org_id="test-org",
            entity_extraction_enabled=True,
        )
        
        call_count = 0
        
        async def mock_extract(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # Return errors in result - this triggers RuntimeError which is not retried
            return MockExtractionResult(
                entities=[],
                relations=[],
                errors=["API rate limit exceeded"],
            )
        
        mock_extractor = AsyncMock()
        mock_extractor.extract = mock_extract
        ingestor.entity_extractor = mock_extractor
        
        # RuntimeError is not in the retry list, so should fail immediately
        with pytest.raises(RuntimeError, match="Entity extraction returned errors"):
            await ingestor._extract_entities_for_parent(
                parent_id="parent_1",
                parent_text="Test text",
                parent_context=None,
                document_title="Test",
                document_id="doc_1",
                db_parent_id="db_parent_1",
                child_ids=[],
            )
        
        # Should have only tried once (no retry for RuntimeError)
        assert call_count == 1


class TestExtractEntitiesPipeline:
    """Test the _extract_entities pipeline method."""
    
    @pytest.mark.asyncio
    async def test_extract_entities_skips_when_disabled(self) -> None:
        """Test that entity extraction is skipped when disabled."""
        from voice_agent.rag2.ingest import RAG2Ingestor, IngestStats
        from voice_agent.rag2.chunker import ParentChunk
        
        ingestor = RAG2Ingestor(
            org_id="test-org",
            entity_extraction_enabled=False,
        )
        
        stats = IngestStats()
        
        await ingestor._extract_entities(
            doc_id="doc_1",
            title="Test",
            parents=[ParentChunk(
                id="p1", index_in_document=0, text="Test", token_count=10,
                page_start=1, page_end=1, children=[],
            )],
            stored_parents={"p1": {"db_id": "db_p1", "child_ids": []}},
            stats=stats,
        )
        
        # Should not have extracted anything
        assert stats.entities_extracted == 0
        assert stats.relations_extracted == 0
    
    @pytest.mark.asyncio
    async def test_extract_entities_handles_missing_parent(self) -> None:
        """Test that missing parents in stored_parents are skipped."""
        from voice_agent.rag2.ingest import RAG2Ingestor, IngestStats
        from voice_agent.rag2.chunker import ParentChunk
        
        ingestor = RAG2Ingestor(
            org_id="test-org",
            entity_extraction_enabled=True,
        )
        
        mock_extractor = AsyncMock()
        ingestor.entity_extractor = mock_extractor
        
        stats = IngestStats()
        
        await ingestor._extract_entities(
            doc_id="doc_1",
            title="Test",
            parents=[ParentChunk(
                id="p1", index_in_document=0, text="Test", token_count=10,
                page_start=1, page_end=1, children=[],
            )],
            stored_parents={},  # Empty - no stored parents
            stats=stats,
        )
        
        # Should not have called extractor
        mock_extractor.extract.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_extract_entities_continues_on_failure(self) -> None:
        """Test that pipeline continues after a parent fails."""
        from voice_agent.rag2.ingest import RAG2Ingestor, IngestStats
        from voice_agent.rag2.chunker import ParentChunk
        
        ingestor = RAG2Ingestor(
            org_id="test-org",
            entity_extraction_enabled=True,
        )
        
        call_count = 0
        
        async def mock_extract(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First parent fails (non-retryable)
                raise ValueError("Invalid text format")
            return MockExtractionResult(
                entities=[{"name": "Entity", "type": "PERSON"}],
                relations=[],
                errors=[],
            )
        
        mock_extractor = AsyncMock()
        mock_extractor.extract = mock_extract
        ingestor.entity_extractor = mock_extractor
        
        mock_store = AsyncMock()
        mock_store.store_extraction.return_value = {
            "entities_created": 1,
            "relations_created": 0,
        }
        ingestor._entity_store = mock_store
        
        stats = IngestStats()
        
        parents = [
            ParentChunk(
                id="p1", index_in_document=0, text="Fail text", token_count=10,
                page_start=1, page_end=1, children=[],
            ),
            ParentChunk(
                id="p2", index_in_document=1, text="Success text", token_count=10,
                page_start=2, page_end=2, children=[],
            ),
        ]
        
        stored_parents = {
            "p1": {"db_id": "db_p1", "child_ids": []},
            "p2": {"db_id": "db_p2", "child_ids": []},
        }
        
        await ingestor._extract_entities(
            doc_id="doc_1",
            title="Test",
            parents=parents,
            stored_parents=stored_parents,
            stats=stats,
        )
        
        # First failed, second succeeded
        assert stats.entities_extracted == 1
        assert len(stats.errors) >= 1
        assert call_count == 2  # Both parents were attempted


class TestModuleLevelFunctions:
    """Test module-level convenience functions."""
    
    def test_ingest_file_function_exists(self) -> None:
        """Test ingest_file function can be imported."""
        from voice_agent.rag2.ingest import ingest_file
        
        assert callable(ingest_file)
