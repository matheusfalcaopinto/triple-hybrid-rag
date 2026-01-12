"""
Pipecat Bot - Main Pipeline Definition

This module defines the core Pipecat pipeline that replaces the custom SessionActor.
It uses:
- CartesiaTTSService for text-to-speech
- CartesiaSTTService for speech-to-text  
- OpenAILLMService for LLM with function calling
- SileroVADAnalyzer for voice activity detection
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
try:
    from pipecat.audio.buffer import AudioBufferProcessor
    AUDIO_BUFFER_AVAILABLE = True
except ImportError:
    AUDIO_BUFFER_AVAILABLE = False
    AudioBufferProcessor = None

from .processors.vad_frame_processor import VADFrameProcessor

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
)
from pipecat.services.cartesia.stt import CartesiaSTTService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.openai.llm import OpenAILLMService

from .config import SETTINGS
from .context import (
    CustomerContext,
    SessionContext,
    prefetch_customer_context,
    build_system_prompt_with_context,
)
from .processors.idle_handler import IdleHandlerProcessor

logger = logging.getLogger("voice_agent_pipecat.bot")


async def create_bot_pipeline(
    transport: Any,
    *,
    caller_phone: str = "",
    call_sid: str = "",
    trace_id: str = "",
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_handlers: Optional[Dict[str, Callable]] = None,
    enable_prefetch: bool = True,
) -> Tuple[PipelineTask, Optional["AudioBufferProcessor"]]:
    """
    Create a Pipecat pipeline for voice agent processing.
    
    Args:
        transport: The transport layer (e.g., TwilioTransport)
        caller_phone: The phone number of the caller
        call_sid: The unique call identifier
        trace_id: Trace ID for observability
        tools: List of tool definitions for function calling
        tool_handlers: Dict mapping tool names to handler functions
        enable_prefetch: Whether to prefetch customer context
        
    Returns:
        Tuple of (PipelineTask, Optional[AudioBufferProcessor]) - task and audio buffer if enabled
    """
    start_time = time.monotonic()
    logger.info(
        "Creating Pipecat pipeline for call_sid=%s, caller=%s",
        call_sid,
        caller_phone,
    )
    
    # ──────────────────────────────────────────────────────────────────────────
    # Customer Context Prefetch
    # ──────────────────────────────────────────────────────────────────────────
    
    customer_context: Optional[CustomerContext] = None
    if enable_prefetch and caller_phone:
        try:
            customer_context = await asyncio.wait_for(
                prefetch_customer_context(caller_phone),
                timeout=SETTINGS.prefetch_timeout,
            )
            logger.info(
                "Customer context prefetched: known=%s, name=%s",
                customer_context.is_known,
                customer_context.name or "N/A",
            )
        except asyncio.TimeoutError:
            logger.warning("Customer prefetch timed out")
        except Exception as e:
            logger.warning("Customer prefetch failed: %s", e)
    
    # ──────────────────────────────────────────────────────────────────────────
    # Initialize AI Services
    # ──────────────────────────────────────────────────────────────────────────
    
    # Speech-to-Text (Cartesia Ink Whisper)
    stt = CartesiaSTTService(
        api_key=SETTINGS.cartesia_api_key,
        model=SETTINGS.cartesia_stt_model,
        language=SETTINGS.cartesia_stt_language,
    )
    
    # Text-to-Speech (Cartesia Sonic)
    tts = CartesiaTTSService(
        api_key=SETTINGS.cartesia_api_key,
        voice_id=SETTINGS.cartesia_voice_id,
        model=SETTINGS.cartesia_tts_model,
        sample_rate=SETTINGS.cartesia_sample_rate,  # Twilio µ-law 8kHz
    )
    
    # LLM (OpenAI with function calling)
    llm = OpenAILLMService(
        api_key=SETTINGS.openai_api_key,
        model=SETTINGS.openai_model,
        base_url=SETTINGS.openai_base_url,
    )
    
    # Register tools for function calling
    if tools and tool_handlers:
        for tool in tools:
            tool_name = tool.get("function", {}).get("name")
            if tool_name and tool_name in tool_handlers:
                llm.register_function(tool_name, tool_handlers[tool_name])
        logger.info("Registered %d tools with LLM", len(tool_handlers))
    
    # ──────────────────────────────────────────────────────────────────────────
    # Context Management
    # ──────────────────────────────────────────────────────────────────────────
    
    # Load and enhance system prompt
    base_prompt = SETTINGS.get_system_prompt()
    system_prompt = build_system_prompt_with_context(
        base_prompt,
        customer_context=customer_context,
        caller_phone=caller_phone,
    )
    
    # Create LLM context with system message
    messages = [
        {"role": "system", "content": system_prompt},
    ]
    
    context = OpenAILLMContext(messages=messages, tools=tools or [])
    context_aggregator = llm.create_context_aggregator(context)
    
    # ──────────────────────────────────────────────────────────────────────────
    # VAD (Voice Activity Detection)
    # ──────────────────────────────────────────────────────────────────────────
    
    vad_analyzer = SileroVADAnalyzer(
        sample_rate=getattr(transport, "_sample_rate", 8000),
        params=VADParams(
            confidence=SETTINGS.vad_threshold,
            stop_secs=SETTINGS.vad_min_silence_ms / 1000.0,
        ),
    )
    vad_processor = VADFrameProcessor(vad_analyzer)
    
    # ──────────────────────────────────────────────────────────────────────────
    # Audio Recording Buffer (optional)
    # ──────────────────────────────────────────────────────────────────────────
    
    audio_buffer: Optional[AudioBufferProcessor] = None
    if SETTINGS.recording_enabled and AUDIO_BUFFER_AVAILABLE:
        audio_buffer = AudioBufferProcessor(
            sample_rate=SETTINGS.recording_sample_rate,
            num_channels=1,
        )
        logger.info("Audio recording enabled for call_sid=%s", call_sid)
    elif SETTINGS.recording_enabled and not AUDIO_BUFFER_AVAILABLE:
        logger.warning("Recording enabled but AudioBufferProcessor not available")
    
    # ──────────────────────────────────────────────────────────────────────────
    # Idle Handler (User Inactivity Detection)
    # ──────────────────────────────────────────────────────────────────────────
    
    idle_handler = IdleHandlerProcessor(
        warning_seconds=SETTINGS.user_idle_warning_seconds,
        timeout_seconds=SETTINGS.user_idle_timeout_seconds,
        max_warnings=SETTINGS.user_idle_max_warnings,
        warning_message="Você ainda está aí? Posso ajudar em algo mais?",
        goodbye_message="Como não recebi resposta, vou encerrar a ligação. Obrigado por ligar!",
    )
    logger.info(
        "Idle handler configured: warn=%.1fs, timeout=%.1fs, max_warnings=%d",
        SETTINGS.user_idle_warning_seconds,
        SETTINGS.user_idle_timeout_seconds,
        SETTINGS.user_idle_max_warnings,
    )
    
    # ──────────────────────────────────────────────────────────────────────────
    # Build Pipeline
    # ──────────────────────────────────────────────────────────────────────────
    
    # The pipeline flows:
    # Input -> [AudioBuffer] -> VAD -> STT -> Context(user) -> LLM -> TTS -> Output -> Context(assistant)
    
    # Build pipeline with conditional audio buffer
    pipeline_processors = [
        transport.input_processor(),     # Receive audio from Twilio
    ]
    
    if audio_buffer:
        pipeline_processors.append(audio_buffer)  # Capture audio before VAD
    
    pipeline_processors.extend([
        vad_processor,                   # Voice activity detection (adapter)
        stt,                             # Speech-to-text
        idle_handler,                    # User inactivity detection
        context_aggregator.user(),       # Add user message to context
        llm,                             # LLM processing (with function calling)
        tts,                             # Text-to-speech
        transport.output_processor(),    # Send audio to Twilio
        context_aggregator.assistant(),  # Add assistant response to context
    ])

    
    pipeline = Pipeline(pipeline_processors)
    
    # ──────────────────────────────────────────────────────────────────────────
    # Create Pipeline Task
    # ──────────────────────────────────────────────────────────────────────────
    
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=SETTINGS.enable_metrics,
            enable_usage_metrics=SETTINGS.enable_usage_metrics,
        ),
    )
    
    setup_time = (time.monotonic() - start_time) * 1000
    logger.info(
        "Pipecat pipeline created in %.1fms for call_sid=%s",
        setup_time,
        call_sid,
    )
    
    return task, audio_buffer


async def run_bot(
    transport: Any,
    caller_phone: str = "",
    call_sid: str = "",
) -> None:
    """
    Run the bot pipeline to completion.
    
    Args:
        transport: The transport layer
        caller_phone: Caller's phone number
        call_sid: Call identifier
    """
    # Import tools dynamically to avoid circular imports
    from .tools import get_all_tools, get_tool_handlers
    
    tools = get_all_tools()
    handlers = get_tool_handlers()
    
    logger.info(
        "Starting bot with %d tools for call_sid=%s",
        len(tools), call_sid,
    )
    
    task, audio_buffer = await create_bot_pipeline(
        transport,
        caller_phone=caller_phone,
        call_sid=call_sid,
        tools=tools,
        tool_handlers=handlers,
    )
    
    try:
        await task.run()
    except asyncio.CancelledError:
        logger.info("Bot pipeline cancelled for call_sid=%s", call_sid)
    except Exception as e:
        logger.exception("Bot pipeline error for call_sid=%s: %s", call_sid, e)
        raise
    finally:
        # Save recording on call end
        if audio_buffer and SETTINGS.recording_enabled:
            try:
                from .services.recording import get_recording_service
                recording_service = get_recording_service()
                recording_path = await recording_service.save_recording(
                    audio_buffer,
                    call_sid=call_sid,
                    caller_phone=caller_phone,
                )
                if recording_path:
                    logger.info("Recording saved: %s", recording_path)
            except Exception as e:
                logger.error("Failed to save recording: %s", e)
        
        logger.info("Bot pipeline completed for call_sid=%s", call_sid)

