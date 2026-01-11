"""
Tests for WhatsApp MCP tool (Twilio Implementation)
"""

from __future__ import annotations

import asyncio
import importlib
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import voice_agent.tools.whatsapp as whatsapp
from voice_agent.config import SETTINGS

# Force Twilio backend for these tests
SETTINGS.whatsapp_backend = "twilio"
importlib.reload(whatsapp)

# Aliases
TOOL_DEFINITIONS = whatsapp.TOOL_DEFINITIONS
send_audio = whatsapp.send_audio
send_document = whatsapp.send_document
send_generated_audio = whatsapp.send_generated_audio
send_image = whatsapp.send_image
send_location = whatsapp.send_location
send_media_message = whatsapp.send_media_message
send_media_message_async = whatsapp.send_media_message_async
send_template_message = whatsapp.send_template_message
send_template_message_async = whatsapp.send_template_message_async
send_text_message = whatsapp.send_text_message
send_text_message_async = whatsapp.send_text_message_async
send_video = whatsapp.send_video


def _get_whatsapp_modules():
    return (
        importlib.import_module("voice_agent.tools.whatsapp"),
        importlib.import_module("voice_agent.tools.whatsapp"),
    )


@pytest.fixture
def twilio_client(monkeypatch):
    mock_client = MagicMock()
    # Patch the directly imported module object that aliases point to
    monkeypatch.setattr(whatsapp, "get_twilio_client", lambda: mock_client)
    # Also patch media_base_url_configured to avoid setup checks failing
    if hasattr(whatsapp, "media_base_url_configured"):
        monkeypatch.setattr(whatsapp, "media_base_url_configured", lambda: True)
    return mock_client


class TestWhatsAppTools:
    """Test WhatsApp tool implementations with Twilio"""

    @patch("voice_agent.tools.whatsapp.os.getenv")
    def test_send_text_message_success(self, mock_getenv, twilio_client):
        """Test successful text message sending"""
        mock_getenv.return_value = "+14155238886"
        
        mock_message = MagicMock()
        mock_message.sid = "SMxxx123"
        mock_message.status = "queued"
        twilio_client.messages.create.return_value = mock_message

        result = send_text_message(
            to="+5511999990001",
            message="Hello from Twilio WhatsApp!",
        )

        assert result["success"] is True
        assert result["message_id"] == "SMxxx123"
        assert result["status"] == "queued"
        assert result["to"] == "+5511999990001"

    @patch("voice_agent.tools.whatsapp.os.getenv")
    def test_send_text_message_missing_from(self, mock_getenv, twilio_client):
        """Test text message without from number configured"""
        mock_getenv.return_value = None  # No TWILIO_WHATSAPP_FROM set
        
        result = send_text_message(
            to="+5511999990001",
            message="Test",
        )

        assert "error" in result
        assert result["setup_required"] is True

    @patch("voice_agent.tools.whatsapp.os.getenv")
    def test_send_image_success(self, mock_getenv, twilio_client):
        """Test successful image sending"""
        mock_getenv.return_value = "+14155238886"
        
        mock_message = MagicMock()
        mock_message.sid = "SMxxx456"
        mock_message.status = "queued"
        twilio_client.messages.create.return_value = mock_message

        result = send_image(
            to="+5511999990001",
            image_url="https://example.com/image.jpg",
            caption="Check this out!",
        )

        assert result["success"] is True
        assert result["message_id"] == "SMxxx456"
        assert result["media_url"] == "https://example.com/image.jpg"

    @patch("voice_agent.tools.whatsapp.os.getenv")
    def test_send_video_success(self, mock_getenv, twilio_client):
        """Test successful video sending"""
        mock_getenv.return_value = "+14155238886"
        
        mock_message = MagicMock()
        mock_message.sid = "SMxxx789"
        mock_message.status = "queued"
        twilio_client.messages.create.return_value = mock_message

        result = send_video(
            to="+5511999990001",
            video_url="https://example.com/video.mp4",
        )

        assert result["success"] is True
        assert result["message_id"] == "SMxxx789"

    @patch("voice_agent.tools.whatsapp.os.getenv")
    def test_send_document_success(self, mock_getenv, twilio_client):
        """Test successful document sending"""
        mock_getenv.return_value = "+14155238886"
        
        mock_message = MagicMock()
        mock_message.sid = "SMxxxabc"
        mock_message.status = "queued"
        twilio_client.messages.create.return_value = mock_message

        result = send_document(
            to="+5511999990001",
            document_url="https://example.com/document.pdf",
            filename="report.pdf",
        )

        assert result["success"] is True
        assert result["message_id"] == "SMxxxabc"

    @patch("voice_agent.tools.whatsapp.os.getenv")
    def test_send_audio_success(self, mock_getenv, twilio_client):
        """Test successful audio sending"""
        mock_getenv.return_value = "+14155238886"
        
        mock_message = MagicMock()
        mock_message.sid = "SMxxxdef"
        mock_message.status = "queued"
        twilio_client.messages.create.return_value = mock_message

        result = send_audio(
            to="+5511999990001",
            audio_url="https://example.com/audio.mp3",
        )

        assert result["success"] is True
        assert result["message_id"] == "SMxxxdef"

    @patch("voice_agent.tools.whatsapp.os.getenv")
    def test_send_location_success(self, mock_getenv, twilio_client):
        """Test successful location sending (as Google Maps link)"""
        mock_getenv.return_value = "+14155238886"
        
        mock_message = MagicMock()
        mock_message.sid = "SMxxxghi"
        mock_message.status = "queued"
        twilio_client.messages.create.return_value = mock_message

        result = send_location(
            to="+5511999990001",
            latitude=-23.5505,
            longitude=-46.6333,
            name="São Paulo",
            address="São Paulo, Brazil",
        )

        assert result["success"] is True
        assert result["message_id"] == "SMxxxghi"

    @patch("voice_agent.tools.whatsapp.os.getenv")
    def test_send_template_message_success(self, mock_getenv, twilio_client):
        """Test successful template message sending"""
        mock_getenv.return_value = "+14155238886"
        
        mock_message = MagicMock()
        mock_message.sid = "SMxxxjkl"
        mock_message.status = "queued"
        twilio_client.messages.create.return_value = mock_message

        result = send_template_message(
            to="+5511999990001",
            content_sid="HXxxxyyy",
            content_variables={"1": "João", "2": "Tomorrow"},
        )

        assert result["success"] is True
        assert result["message_id"] == "SMxxxjkl"
        assert result["template"] == "HXxxxyyy"

    def test_library_not_installed(self, monkeypatch):
        """Test error handling when Twilio library not installed"""
        monkeypatch.setattr(whatsapp, "TWILIO_AVAILABLE", False)

        result = send_text_message(
            to="+5511999990001",
            message="Test",
        )

        assert "error" in result
        assert result["install_required"] is True

    def test_missing_credentials(self, monkeypatch):
        """Test error handling when credentials not configured"""
        def _raise(*args, **kwargs):
            raise ValueError("TWILIO_ACCOUNT_SID not configured")

        monkeypatch.setattr(whatsapp, "get_twilio_client", _raise)
        
        result = send_text_message(
            to="+5511999990001",
            message="Test",
        )
        
        assert "error" in result
        assert result["setup_required"] is True

    def test_send_generated_audio_success(self, monkeypatch):
        async_generate = AsyncMock(return_value=(Path("generated.mp3"), "https://cdn/audio.mp3"))
        mock_send_audio = MagicMock(return_value={"success": True, "message_id": "SMGEN"})

        monkeypatch.setattr(whatsapp, "generate_audio_reply", async_generate)
        if hasattr(whatsapp, "media_base_url_configured"):
            monkeypatch.setattr(whatsapp, "media_base_url_configured", lambda: True)
        monkeypatch.setattr(whatsapp, "send_audio", mock_send_audio)

        result = asyncio.run(send_generated_audio("+1555000000", "Hello"))
        mock_send_audio.assert_called_once_with(
            "+1555000000",
            "https://cdn/audio.mp3",
            from_number=None,
        )
        assert result["success"] is True
        assert result["generated_media_url"] == "https://cdn/audio.mp3"
        assert Path(result["local_path"]).name == "generated.mp3"

    def test_send_generated_audio_requires_setup(self, monkeypatch):
        if hasattr(whatsapp, "media_base_url_configured"):
            monkeypatch.setattr(whatsapp, "media_base_url_configured", lambda: False)

        result = asyncio.run(send_generated_audio("+1555000000", "Hello"))
        assert result["success"] is False
        assert result["setup_required"] is True

    def test_tool_definitions_format(self):
        """Test that tool definitions have correct structure"""
        assert isinstance(TOOL_DEFINITIONS, list)
        expected_names = {
            "send_whatsapp_message",
            "send_whatsapp_image",
            "send_whatsapp_video",
            "send_whatsapp_document",
            "send_whatsapp_audio",
            "send_whatsapp_generated_audio",
            "send_whatsapp_location",
            "send_whatsapp_template",
        }
        actual_names = {tool["name"] for tool in TOOL_DEFINITIONS}
        assert actual_names == expected_names

        for tool in TOOL_DEFINITIONS:
            assert "description" in tool
            assert "parameters" in tool
            assert "required" in tool
            assert "handler" in tool

    def test_tool_names_unique(self):
        """Test that all tool names are unique"""
        names = [tool["name"] for tool in TOOL_DEFINITIONS]
        assert len(names) == len(set(names))

    def test_tool_handlers_valid(self):
        """Test that all handlers are callable"""
        for tool in TOOL_DEFINITIONS:
            assert callable(tool["handler"])


class TestWhatsAppAsyncWrappers:
    """Tests for async helper wrappers."""

    def test_send_text_message_async_wrapper(self, monkeypatch):
        async_mock = AsyncMock(return_value={"success": True})
        monkeypatch.setattr("voice_agent.tools.whatsapp.asyncio.to_thread", async_mock)
        result = asyncio.run(send_text_message_async("+1555000000", "Hello"))
        async_mock.assert_awaited_once()
        func = async_mock.call_args.args[0]
        assert func is send_text_message
        assert result["success"] is True

    def test_send_media_message_async_wrapper(self, monkeypatch):
        async_mock = AsyncMock(return_value={"success": True})
        monkeypatch.setattr("voice_agent.tools.whatsapp.asyncio.to_thread", async_mock)
        result = asyncio.run(send_media_message_async("+1555000000", "https://cdn/img.jpg"))
        async_mock.assert_awaited_once()
        func = async_mock.call_args.args[0]
        assert func is send_media_message
        assert result["success"] is True

    def test_send_template_message_async_wrapper(self, monkeypatch):
        async_mock = AsyncMock(return_value={"success": True})
        monkeypatch.setattr("voice_agent.tools.whatsapp.asyncio.to_thread", async_mock)
        result = asyncio.run(send_template_message_async("+1555000000", "HX123"))
        async_mock.assert_awaited_once()
        func = async_mock.call_args.args[0]
        assert func is send_template_message
        assert result["success"] is True


class TestWhatsAppIntegration:
    """Integration tests for WhatsApp tools"""

    @patch("voice_agent.tools.whatsapp.os.getenv")
    def test_send_multiple_message_types(self, mock_getenv, twilio_client):
        """Test sending different message types in sequence"""
        mock_getenv.return_value = "+14155238886"
        
        # Mock different message responses
        messages = [
            MagicMock(sid=f"SM{i}", status="queued")
            for i in range(3)
        ]
        twilio_client.messages.create.side_effect = messages

        # Send text
        result1 = send_text_message("+5511999990001", "Hello")
        assert result1["success"] is True

        # Send image
        result2 = send_image("+5511999990001", "https://example.com/img.jpg")
        assert result2["success"] is True

        # Send location
        result3 = send_location("+5511999990001", -23.5505, -46.6333)
        assert result3["success"] is True

        assert twilio_client.messages.create.call_count == 3

    @patch("voice_agent.tools.whatsapp.os.getenv")
    def test_phone_number_formatting(self, mock_getenv, twilio_client):
        """Test that phone numbers are properly formatted with whatsapp: prefix"""
        mock_getenv.return_value = "+14155238886"
        
        mock_message = MagicMock(sid="SMxxx", status="queued")
        twilio_client.messages.create.return_value = mock_message

        send_text_message("+5511999990001", "Test")

        # Check that the call was made with whatsapp: prefix
        call_args = twilio_client.messages.create.call_args
        assert call_args[1]["to"] == "whatsapp:+5511999990001"
        assert call_args[1]["from_"] == "whatsapp:+14155238886"
