"""
Comprehensive tests for the Pipecat Voice Agent migration.

Run with: pytest tests/test_pipecat_module.py tests/test_pipecat_integration.py -v
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any, Dict


# ──────────────────────────────────────────────────────────────────────────────
# Transport Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestTwilioTransport:
    """Tests for enhanced Twilio transport layer."""
    
    def test_transport_params_defaults(self) -> None:
        """Test TwilioTransportParams has correct defaults."""
        from voice_agent.transports.twilio_transport import TwilioTransportParams
        
        params = TwilioTransportParams()
        assert params.sample_rate == 8000
        assert params.enable_interruptions is True
        assert params.dtmf_interrupts_tts is False
        assert params.dtmf_collection_timeout == 10.0
    
    def test_digit_collector_basic(self) -> None:
        """Test DigitCollector basic digit collection."""
        from voice_agent.transports.twilio_transport import DigitCollector
        
        collector = DigitCollector(expected_length=4)
        collector.start()
        
        complete, digits = collector.add_digit("1")
        assert not complete
        assert digits == "1"
        
        complete, digits = collector.add_digit("2")
        complete, digits = collector.add_digit("3")
        complete, digits = collector.add_digit("4")
        
        assert complete
        assert digits == "1234"
    
    def test_digit_collector_terminator(self) -> None:
        """Test DigitCollector with terminator."""
        from voice_agent.transports.twilio_transport import DigitCollector
        
        collector = DigitCollector(terminator="#")
        collector.start()
        
        collector.add_digit("1")
        collector.add_digit("2")
        complete, digits = collector.add_digit("#")
        
        assert complete
        assert digits == "12"
    
    def test_digit_collector_ignores_non_digits(self) -> None:
        """Test DigitCollector ignores non-digit characters."""
        from voice_agent.transports.twilio_transport import DigitCollector
        
        collector = DigitCollector()
        collector.start()
        
        collector.add_digit("1")
        collector.add_digit("a")  # Should be ignored
        collector.add_digit("*")  # Should be ignored
        collector.add_digit("2")
        
        assert collector.get_partial() == "12"
    
    def test_transport_creation(self) -> None:
        """Test TwilioWebsocketTransport creation."""
        from voice_agent.transports.twilio_transport import (
            TwilioWebsocketTransport,
            TwilioTransportParams,
        )
        
        mock_ws = MagicMock()
        params = TwilioTransportParams(sample_rate=8000)
        transport = TwilioWebsocketTransport(mock_ws, params=params)
        
        assert transport.stream_sid is None
        assert transport.call_sid is None
        assert transport.input_processor() is not None
        assert transport.output_processor() is not None


# ──────────────────────────────────────────────────────────────────────────────
# Context Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestContext:
    """Tests for context management."""
    
    def test_customer_context_unknown(self) -> None:
        """Test CustomerContext for unknown customer."""
        from voice_agent.context import CustomerContext
        
        ctx = CustomerContext(phone="+5511999999999")
        
        assert not ctx.is_known
        context_str = ctx.to_context_string()
        assert "Unknown/New" in context_str
        assert "+5511999999999" in context_str
    
    def test_customer_context_known(self) -> None:
        """Test CustomerContext for known customer."""
        from voice_agent.context import CustomerContext
        
        ctx = CustomerContext(
            customer_id="test-123",
            phone="+5511999999999",
            name="João Silva",
            status="active",
            facts=[{"fact_type": "preference", "fact_value": "morning calls"}],
        )
        
        assert ctx.is_known
        context_str = ctx.to_context_string()
        assert "João Silva" in context_str
        assert "test-123" in context_str
        assert "morning calls" in context_str
    
    def test_build_system_prompt_with_context(self) -> None:
        """Test system prompt building with context injection."""
        from voice_agent.context import (
            CustomerContext,
            build_system_prompt_with_context,
        )
        
        base_prompt = "You are a helpful assistant."
        ctx = CustomerContext(
            customer_id="test-123",
            name="Test User",
            phone="+5511999999999",
        )
        
        result = build_system_prompt_with_context(base_prompt, ctx)
        
        assert "You are a helpful assistant." in result
        assert "[CUSTOMER CONTEXT]" in result
        assert "Test User" in result
    
    def test_build_system_prompt_without_context(self) -> None:
        """Test system prompt building without context."""
        from voice_agent.context import build_system_prompt_with_context
        
        base_prompt = "You are a helpful assistant."
        result = build_system_prompt_with_context(
            base_prompt,
            caller_phone="+5511999999999",
        )
        
        assert "You are a helpful assistant." in result
        assert "[CALLER CONTEXT]" in result


# ──────────────────────────────────────────────────────────────────────────────
# App Session Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestAppState:
    """Tests for FastAPI application state management."""
    
    @pytest.mark.asyncio
    async def test_session_lifecycle(self) -> None:
        """Test session add/remove lifecycle."""
        from voice_agent.app import AppState, SessionInfo
        
        # Create fresh instance for test
        state = AppState()
        
        # Add session
        session = await state.add_session(
            call_sid="test-call-123",
            caller_phone="+5511999999999",
            trace_id="trace-456",
        )
        
        assert state.active_calls == 1
        assert state.total_calls_handled == 1
        assert "test-call-123" in state.sessions
        assert session.status == "active"
        
        # Get session
        retrieved = state.get_session("test-call-123")
        assert retrieved is not None
        assert retrieved.caller_phone == "+5511999999999"
        
        # Remove session
        ended = await state.remove_session("test-call-123")
        assert ended is not None
        assert ended.status == "ended"
        assert state.active_calls == 0
        assert "test-call-123" not in state.sessions
    
    def test_health_endpoint(self) -> None:
        """Test health endpoint returns 200."""
        from voice_agent.app import health
        
        response = health()
        assert response.status_code == 200
    
    def test_info_endpoint(self) -> None:
        """Test info endpoint returns expected structure."""
        from voice_agent.app import info
        
        data = info()
        assert "version" in data
        assert "framework" in data
        assert data["framework"] == "pipecat"
        assert "config" in data
        assert "llm_model" in data["config"]


# ──────────────────────────────────────────────────────────────────────────────
# Bot Pipeline Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestBotPipeline:
    """Tests for bot pipeline creation."""
    
    def test_bot_module_imports(self) -> None:
        """Test bot module can be imported."""
        from voice_agent.bot import create_bot_pipeline, run_bot
        
        assert create_bot_pipeline is not None
        assert run_bot is not None


# ──────────────────────────────────────────────────────────────────────────────
# Tools Integration Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestToolsIntegration:
    """Tests for MCP tools integration."""
    
    def test_tools_loaded(self) -> None:
        """Test that tools are loaded correctly."""
        from voice_agent.tools import get_all_tools, get_tool_count
        
        tools = get_all_tools()
        count = get_tool_count()
        
        assert isinstance(tools, list)
        assert count > 0
        assert len(tools) == count
    
    def test_tools_format(self) -> None:
        """Test tools are in OpenAI function calling format."""
        from voice_agent.tools import get_all_tools
        
        tools = get_all_tools()
        
        if tools:
            tool = tools[0]
            assert "type" in tool
            assert tool["type"] == "function"
            assert "function" in tool
            assert "name" in tool["function"]
            assert "description" in tool["function"]
    
    def test_essential_tools_subset(self) -> None:
        """Test essential tools is a subset of all tools."""
        from voice_agent.tools import (
            get_all_tools,
            get_essential_tools,
            ESSENTIAL_TOOL_NAMES,
        )
        
        all_tools = get_all_tools()
        essential = get_essential_tools()
        
        assert len(essential) <= len(all_tools)
        
        # Check each essential tool is in the list
        for tool in essential:
            name = tool.get("function", {}).get("name")
            assert name in ESSENTIAL_TOOL_NAMES
    
    def test_handlers_match_tools(self) -> None:
        """Test that handlers exist for all tools."""
        from voice_agent.tools import get_all_tools, get_tool_handlers
        
        tools = get_all_tools()
        handlers = get_tool_handlers()
        
        tool_names = {t.get("function", {}).get("name") for t in tools}
        handler_names = set(handlers.keys())
        
        # All handlers should have corresponding tools
        assert handler_names == tool_names


# ──────────────────────────────────────────────────────────────────────────────
# Pipecat Services Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestPipecatServices:
    """Tests for Pipecat service availability."""
    
    def test_cartesia_services(self) -> None:
        """Test Cartesia services are available."""
        from pipecat.services.cartesia.tts import CartesiaTTSService
        from pipecat.services.cartesia.stt import CartesiaSTTService
        
        assert CartesiaTTSService is not None
        assert CartesiaSTTService is not None
    
    def test_openai_service(self) -> None:
        """Test OpenAI service is available."""
        from pipecat.services.openai.llm import OpenAILLMService
        
        assert OpenAILLMService is not None
    
    def test_silero_vad(self) -> None:
        """Test Silero VAD is available."""
        from pipecat.audio.vad.silero import SileroVADAnalyzer
        
        assert SileroVADAnalyzer is not None
    
    def test_pipeline_components(self) -> None:
        """Test pipeline components are available."""
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.pipeline.task import PipelineTask, PipelineParams
        
        assert Pipeline is not None
        assert PipelineTask is not None
        assert PipelineParams is not None


# ──────────────────────────────────────────────────────────────────────────────
# Config Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestConfig:
    """Tests for configuration management."""
    
    def test_config_loads(self) -> None:
        """Test configuration loads correctly."""
        from voice_agent.config import SETTINGS
        
        assert SETTINGS is not None
        assert SETTINGS.openai_model
        assert SETTINGS.cartesia_tts_model
        assert SETTINGS.cartesia_stt_model
    
    def test_system_prompt_method(self) -> None:
        """Test get_system_prompt method."""
        from voice_agent.config import SETTINGS
        
        prompt = SETTINGS.get_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 0
