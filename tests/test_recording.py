"""Tests for recording service."""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestRecordingService:
    """Test recording service functionality."""

    @pytest.fixture
    def temp_recordings_dir(self):
        """Create temporary directory for recordings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def mock_settings(self, temp_recordings_dir):
        """Mock settings with recording enabled."""
        with patch("voice_agent.services.recording.SETTINGS") as mock:
            mock.recording_enabled = True
            mock.recording_path = temp_recordings_dir
            mock.recording_format = "wav"
            mock.recording_sample_rate = 16000
            mock.recording_include_bot_audio = True
            yield mock

    def test_recording_service_init(self, mock_settings, temp_recordings_dir):
        """Test recording service initializes and creates directory."""
        from voice_agent.services.recording import RecordingService
        
        service = RecordingService()
        assert Path(temp_recordings_dir).exists()

    def test_generate_filename(self, mock_settings):
        """Test filename generation includes call_sid and timestamp."""
        from voice_agent.services.recording import RecordingService
        
        service = RecordingService()
        filename = service._generate_filename("test-call-123")
        
        assert "test-call-123" in filename
        assert filename.endswith(".wav")

    def test_generate_filename_with_suffix(self, mock_settings):
        """Test filename generation with suffix."""
        from voice_agent.services.recording import RecordingService
        
        service = RecordingService()
        filename = service._generate_filename("test-call-123", suffix="user")
        
        assert "test-call-123" in filename
        assert "_user" in filename
        assert filename.endswith(".wav")

    def test_save_recording_disabled(self, mock_settings):
        """Test save_recording returns None when disabled."""
        mock_settings.recording_enabled = False
        
        from voice_agent.services.recording import RecordingService
        
        service = RecordingService()
        mock_buffer = MagicMock()
        
        result = asyncio.run(service.save_recording(mock_buffer, "call-123"))
        assert result is None

    def test_save_recording_no_audio(self, mock_settings):
        """Test save_recording handles empty audio buffer."""
        from voice_agent.services.recording import RecordingService
        
        service = RecordingService()
        
        mock_buffer = MagicMock()
        mock_buffer.has_audio.return_value = False
        
        result = asyncio.run(service.save_recording(mock_buffer, "call-123"))
        assert result is None

    def test_save_recording_success(self, mock_settings, temp_recordings_dir):
        """Test successful recording save."""
        from voice_agent.services.recording import RecordingService
        
        service = RecordingService()
        
        # Mock audio buffer with some audio data (new API)
        mock_buffer = MagicMock()
        mock_buffer.has_audio.return_value = True
        mock_buffer.merge_audio_buffers.return_value = b"\x00\x00" * 8000  # 0.5 second of silence at 16kHz
        
        result = asyncio.run(service.save_recording(mock_buffer, "call-123", "+5511999999999"))
        
        assert result is not None
        assert Path(result).exists()
        assert "call-123" in result
        assert result.endswith(".wav")

    def test_write_wav_file(self, mock_settings, temp_recordings_dir):
        """Test WAV file writing."""
        from voice_agent.services.recording import RecordingService
        
        service = RecordingService()
        
        filepath = Path(temp_recordings_dir) / "test.wav"
        audio_data = b"\x00\x00" * 1600  # ~0.1 second
        
        service._write_wav_file(filepath, audio_data)
        
        assert filepath.exists()
        assert filepath.stat().st_size > 0


class TestGetRecordingService:
    """Test singleton getter."""

    def test_get_recording_service_singleton(self):
        """Test that get_recording_service returns singleton."""
        with patch("voice_agent.services.recording.SETTINGS") as mock:
            mock.recording_path = "/tmp/test_recordings"
            mock.recording_format = "wav"
            mock.recording_sample_rate = 16000
            
            # Reset singleton
            import voice_agent.services.recording as recording_module
            recording_module._recording_service = None
            
            from voice_agent.services.recording import get_recording_service
            
            service1 = get_recording_service()
            service2 = get_recording_service()
            
            assert service1 is service2
