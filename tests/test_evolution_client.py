"""
Tests for Evolution API WhatsApp Client

Tests the Evolution API client implementation for WhatsApp messaging.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from voice_agent.services.evolution_client import (
    EVOLUTION_AVAILABLE,
    EvolutionAPIError,
    EvolutionClient,
    get_evolution_client,
)


class TestEvolutionClient:
    """Test Evolution API client implementation"""

    def test_phone_number_formatting_with_plus(self):
        """Test formatting phone numbers with + prefix"""
        client = EvolutionClient(base_url="http://test", api_key="test")
        result = client.format_phone_number("+5511999990001")
        assert result == "5511999990001"

    def test_phone_number_formatting_without_plus(self):
        """Test formatting phone numbers without + prefix"""
        client = EvolutionClient(base_url="http://test", api_key="test")
        result = client.format_phone_number("5511999990001")
        assert result == "5511999990001"

    def test_phone_number_formatting_whatsapp_prefix(self):
        """Test formatting phone numbers with whatsapp: prefix"""
        client = EvolutionClient(base_url="http://test", api_key="test")
        result = client.format_phone_number("whatsapp:+5511999990001")
        assert result == "5511999990001"

    def test_client_initialization_with_params(self):
        """Test client initialization with explicit parameters"""
        client = EvolutionClient(
            base_url="https://api.example.com",
            api_key="my-api-key",
            instance_name="my-instance",
            instance_token="my-token",
        )
        assert client.base_url == "https://api.example.com"
        assert client.api_key == "my-api-key"
        assert client.instance_name == "my-instance"
        assert client.instance_token == "my-token"

    def test_client_headers(self):
        """Test that headers are correctly generated"""
        client = EvolutionClient(
            base_url="http://test",
            api_key="test-key",
            instance_token="bearer-token",
        )
        headers = client._get_headers()
        assert headers["Content-Type"] == "application/json"
        assert headers["apikey"] == "test-key"
        assert headers["Authorization"] == "Bearer bearer-token"

    def test_client_headers_without_instance_token(self, monkeypatch):
        """Test headers without instance token"""
        monkeypatch.setattr("voice_agent.services.evolution_client.SETTINGS.evolution_instance_token", "")
        client = EvolutionClient(
            base_url="http://test",
            api_key="test-key",
        )
        headers = client._get_headers()
        assert "Authorization" not in headers
        assert headers["apikey"] == "test-key"

    def test_send_text_success(self):
        """Test successful text message sending"""
        async def _test():
            client = EvolutionClient(
                base_url="http://test",
                api_key="test-key",
                instance_name="test-instance",
            )
            
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "key": {"id": "MSG123"},
                "status": "sent",
            }
            
            with patch.object(client, "_get_client") as mock_get_client:
                mock_http_client = AsyncMock()
                mock_http_client.post.return_value = mock_response
                mock_get_client.return_value = mock_http_client
                
                result = await client.send_text("+5511999990001", "Hello!")
                
                assert result["key"]["id"] == "MSG123"
                assert result["status"] == "sent"
        
        asyncio.run(_test())

    def test_send_media_success(self):
        """Test successful media message sending"""
        async def _test():
            client = EvolutionClient(
                base_url="http://test",
                api_key="test-key",
                instance_name="test-instance",
            )
            
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "key": {"id": "MEDIA123"},
                "status": "sent",
            }
            
            with patch.object(client, "_get_client") as mock_get_client:
                mock_http_client = AsyncMock()
                mock_http_client.post.return_value = mock_response
                mock_get_client.return_value = mock_http_client
                
                result = await client.send_media(
                    "+5511999990001",
                    "image",
                    "https://example.com/image.jpg",
                    caption="Test image",
                )
                
                assert result["key"]["id"] == "MEDIA123"
        
        asyncio.run(_test())

    def test_get_evolution_client_no_url(self, monkeypatch):
        """Test error when Evolution API URL is not configured"""
        monkeypatch.setattr("voice_agent.services.evolution_client.SETTINGS.evolution_api_url", "")
        
        with pytest.raises(ValueError) as exc_info:
            get_evolution_client()
        assert "EVOLUTION_API_URL not configured" in str(exc_info.value)

    def test_get_evolution_client_no_key(self, monkeypatch):
        """Test error when Evolution API key is not configured"""
        monkeypatch.setattr("voice_agent.services.evolution_client.SETTINGS.evolution_api_url", "http://test")
        monkeypatch.setattr("voice_agent.services.evolution_client.SETTINGS.evolution_api_key", "")
        
        with pytest.raises(ValueError) as exc_info:
            get_evolution_client()
        assert "EVOLUTION_API_KEY not configured" in str(exc_info.value)

    def test_evolution_available_flag(self):
        """Test that EVOLUTION_AVAILABLE is True"""
        assert EVOLUTION_AVAILABLE is True


class TestEvolutionAPIError:
    """Test Evolution API error handling"""

    def test_error_with_status_code(self):
        """Test error with status code"""
        error = EvolutionAPIError(
            "API error",
            status_code=400,
            response_data={"error": "Bad Request"},
        )
        assert error.status_code == 400
        assert error.response_data == {"error": "Bad Request"}
        assert "API error" in str(error)

    def test_error_without_status_code(self):
        """Test error without status code"""
        error = EvolutionAPIError("Connection failed")
        assert error.status_code is None
        assert error.response_data == {}
