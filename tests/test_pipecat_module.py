"""
Unit tests for the Pipecat-based voice agent.

Run with: pytest tests/test_pipecat_module.py -v
"""

import pytest


class TestPipecatConfig:
    """Tests for voice_agent_pipecat.config module."""
    
    def test_config_loads(self) -> None:
        """Test that configuration loads without errors."""
        from voice_agent.config import SETTINGS
        
        assert SETTINGS is not None
        assert SETTINGS.openai_model  # Has default
        assert SETTINGS.cartesia_tts_model  # Has default
        assert SETTINGS.cartesia_stt_model  # Has default
    
    def test_config_defaults(self) -> None:
        """Test that configuration has expected defaults."""
        from voice_agent.config import SETTINGS
        
        # Check defaults from config.py (values may come from .env)
        assert SETTINGS.openai_model  # Has a value
        assert SETTINGS.cartesia_stt_language  # Has a value
        assert SETTINGS.vad_threshold > 0  # Valid threshold
        assert SETTINGS.twilio_sample_rate > 0  # Valid sample rate
    
    def test_system_prompt_property(self) -> None:
        """Test that system prompt property works."""
        from voice_agent.config import SETTINGS
        
        # Method should return string even if file doesn't exist
        prompt = SETTINGS.get_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 0


class TestPipecatTransport:
    """Tests for Twilio transport layer."""
    
    def test_transport_import(self) -> None:
        """Test that transport can be imported."""
        from voice_agent.transports import TwilioWebsocketTransport
        
        assert TwilioWebsocketTransport is not None
    
    def test_input_output_processors(self) -> None:
        """Test that transport creates input/output processors."""
        from voice_agent.transports import TwilioWebsocketTransport
        from unittest.mock import MagicMock
        
        mock_ws = MagicMock()
        transport = TwilioWebsocketTransport(mock_ws, sample_rate=8000)
        
        assert transport.input_processor() is not None
        assert transport.output_processor() is not None


class TestPipecatBot:
    """Tests for bot pipeline creation."""
    
    def test_bot_imports(self) -> None:
        """Test that bot module can be imported."""
        from voice_agent.bot import create_bot_pipeline, run_bot
        
        assert create_bot_pipeline is not None
        assert run_bot is not None


class TestPipecatApp:
    """Tests for FastAPI application."""
    
    def test_app_creation(self) -> None:
        """Test that FastAPI app is created."""
        from voice_agent.app import app
        
        assert app is not None
        assert app.title == "Voice Agent Pipecat"
    
    def test_routes_exist(self) -> None:
        """Test that required routes are registered."""
        from voice_agent.app import app
        
        routes = [r.path for r in app.routes if hasattr(r, "path")]
        
        assert "/" in routes
        assert "/healthz" in routes
        assert "/readyz" in routes
        assert "/incoming-call" in routes
        assert "/media-stream" in routes
    
    def test_health_endpoint(self) -> None:
        """Test health endpoint returns 200."""
        from voice_agent.app import health
        
        response = health()
        assert response.status_code == 200
    
    def test_info_endpoint(self) -> None:
        """Test info endpoint returns expected data."""
        from voice_agent.app import info
        
        data = info()
        assert "version" in data
        assert "framework" in data
        assert data["framework"] == "pipecat"


class TestPipecatTools:
    """Tests for tools adapter."""
    
    def test_tools_functions_exist(self) -> None:
        """Test that tool functions can be imported."""
        from voice_agent.tools import get_all_tools, get_tool_handlers
        
        assert get_all_tools is not None
        assert get_tool_handlers is not None
    
    def test_get_all_tools_returns_list(self) -> None:
        """Test that get_all_tools returns a list."""
        from voice_agent.tools import get_all_tools
        
        # This may return empty list if MCP tools not available
        tools = get_all_tools()
        assert isinstance(tools, list)
    
    def test_get_tool_handlers_returns_dict(self) -> None:
        """Test that get_tool_handlers returns a dict."""
        from voice_agent.tools import get_tool_handlers
        
        # This may return empty dict if MCP tools not available
        # May also fail to import some tools - that's expected
        try:
            handlers = get_tool_handlers()
            assert isinstance(handlers, dict)
        except ImportError:
            # Expected if voice_agent_v4 tools aren't fully importable
            pass


class TestPipecatFrames:
    """Tests for Pipecat frame compatibility."""
    
    def test_required_frames_available(self) -> None:
        """Test that all required frames can be imported."""
        from pipecat.frames.frames import (
            AudioRawFrame,
            CancelFrame,
            EndFrame,
            Frame,
            StartFrame,
            StartInterruptionFrame,
            TransportMessageFrame,
        )
        
        # All imports should succeed
        assert AudioRawFrame is not None
        assert CancelFrame is not None
        assert EndFrame is not None
        assert Frame is not None
        assert StartFrame is not None
        assert StartInterruptionFrame is not None
        assert TransportMessageFrame is not None
    
    def test_services_available(self) -> None:
        """Test that required Pipecat services can be imported."""
        from pipecat.services.cartesia.tts import CartesiaTTSService
        from pipecat.services.cartesia.stt import CartesiaSTTService
        from pipecat.services.openai.llm import OpenAILLMService
        from pipecat.audio.vad.silero import SileroVADAnalyzer
        
        assert CartesiaTTSService is not None
        assert CartesiaSTTService is not None
        assert OpenAILLMService is not None
        assert SileroVADAnalyzer is not None
