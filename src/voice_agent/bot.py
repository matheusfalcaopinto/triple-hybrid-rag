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

try:
    from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
    AUDIO_BUFFER_AVAILABLE = True
except ImportError:
    AUDIO_BUFFER_AVAILABLE = False
    AudioBufferProcessor = None

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
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
from .processors.tool_call_muter import ToolCallMuterProcessor
from .services.post_call import process_post_call

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
) -> Tuple[PipelineTask, Optional["AudioBufferProcessor"], "OpenAILLMContext"]:
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
    # Customer Context Prefetch (Parallel or Blocking based on config)
    # ──────────────────────────────────────────────────────────────────────────
    
    customer_context: Optional[CustomerContext] = None
    prefetch_task: Optional[asyncio.Task] = None
    
    if enable_prefetch and caller_phone:
        if SETTINGS.parallel_context_prefetch:
            # NON-BLOCKING: Start prefetch in background, inject later
            prefetch_task = asyncio.create_task(
                prefetch_customer_context(caller_phone),
                name=f"prefetch-{call_sid[:8]}"
            )
            logger.info(
                "Started background customer prefetch for %s",
                caller_phone[:6] + "****" if len(caller_phone) > 6 else caller_phone
            )
        else:
            # BLOCKING: Wait for prefetch to complete (legacy behavior)
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
        sample_rate=8000,  # Twilio uses 8kHz audio
    )
    
    # Text-to-Speech (Cartesia Sonic)
    # Import Language enum for Portuguese
    from pipecat.transcriptions.language import Language
    tts = CartesiaTTSService(
        api_key=SETTINGS.cartesia_api_key,
        voice_id=SETTINGS.cartesia_voice_id,
        model=SETTINGS.cartesia_tts_model,
        sample_rate=SETTINGS.cartesia_sample_rate,  # Twilio µ-law 8kHz
        params=CartesiaTTSService.InputParams(language=Language.PT),
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
    # Note: Initial greeting uses pre-recorded audio, not LLM-generated
    messages = [
        {"role": "system", "content": system_prompt},
    ]
    
    context = OpenAILLMContext(messages=messages, tools=tools or [])
    context_aggregator = llm.create_context_aggregator(context)
    
    # VAD is now handled by the transport's VADAnalyzer parameter
    
    # ──────────────────────────────────────────────────────────────────────────
    # Audio Recording Buffer (optional)
    # ──────────────────────────────────────────────────────────────────────────
    
    audio_buffer: Optional[AudioBufferProcessor] = None
    if SETTINGS.recording_enabled and AUDIO_BUFFER_AVAILABLE:
        audio_buffer = AudioBufferProcessor(
            sample_rate=SETTINGS.recording_sample_rate,
            num_channels=1,
        )
        # Start recording immediately
        audio_buffer.start_recording()
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
    # Tool Call Muter (Mute user briefly after tool calls start)
    # ──────────────────────────────────────────────────────────────────────────
    
    tool_call_muter = ToolCallMuterProcessor(
        mute_duration=SETTINGS.mute_function_call_duration,
        enabled=SETTINGS.mute_during_function_call,
    )
    if SETTINGS.mute_during_function_call:
        logger.info(
            "Tool call muter enabled - user muted for %.1fs after tool call starts",
            SETTINGS.mute_function_call_duration
        )
    else:
        logger.info("Tool call muter disabled")
    
    # ──────────────────────────────────────────────────────────────────────────
    # Build Pipeline
    # ──────────────────────────────────────────────────────────────────────────
    
    # The pipeline flows:
    # Input -> [AudioBuffer] -> STT -> ToolMuter -> IdleHandler -> Context(user) -> LLM -> TTS -> Output -> Context(assistant)
    # Note: VAD is now handled by the transport's VADAnalyzer
    # Note: ToolMuter drops user transcription while tools are executing
    
    # Build pipeline with conditional audio buffer
    pipeline_processors = [
        transport.input(),                # Receive audio from Twilio (via WebSocket)
    ]
    
    if audio_buffer:
        pipeline_processors.append(audio_buffer)  # Capture audio before STT
    
    pipeline_processors.extend([
        stt,                             # Speech-to-text
        tool_call_muter,                 # Mute user during tool execution
        idle_handler,                    # User inactivity detection
        context_aggregator.user(),       # Add user message to context
        llm,                             # LLM processing (with function calling)
        tts,                             # Text-to-speech
        transport.output(),              # Send audio to Twilio (via WebSocket)
        context_aggregator.assistant(),  # Add assistant response to context
    ])

    
    pipeline = Pipeline(pipeline_processors)
    
    # ──────────────────────────────────────────────────────────────────────────
    # Create Pipeline Task
    # ──────────────────────────────────────────────────────────────────────────
    
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=8000,    # Twilio uses 8kHz
            audio_out_sample_rate=8000,   # Twilio uses 8kHz
            enable_metrics=SETTINGS.enable_metrics,
            enable_usage_metrics=SETTINGS.enable_usage_metrics,
        ),
    )
    
    # ──────────────────────────────────────────────────────────────────────────
    # Background Context Injection (for parallel prefetch)
    # ──────────────────────────────────────────────────────────────────────────
    
    if prefetch_task is not None:
        async def inject_customer_context():
            """Inject customer context when prefetch completes."""
            try:
                customer_ctx = await asyncio.wait_for(
                    prefetch_task,
                    timeout=SETTINGS.prefetch_timeout
                )
                
                if customer_ctx and customer_ctx.is_known:
                    logger.info(
                        "Injecting customer context: id=%s, name=%s",
                        customer_ctx.customer_id,
                        customer_ctx.name or "Unknown"
                    )
                    
                    # Build enhanced prompt with customer data
                    enhanced_prompt = build_system_prompt_with_context(
                        base_prompt,
                        customer_context=customer_ctx,
                        caller_phone=caller_phone,
                    )
                    
                    # Update the LLM context with customer info
                    # This injects a new system message with the customer context
                    await task.queue_frames([
                        OpenAILLMContextFrame(
                            context=OpenAILLMContext(
                                messages=[{"role": "system", "content": enhanced_prompt}],
                                tools=tools or []
                            )
                        )
                    ])
                    logger.info("Customer context injected successfully")
                else:
                    logger.info("No customer context to inject (unknown caller)")
                    
            except asyncio.TimeoutError:
                logger.warning(
                    "Customer prefetch timed out for %s",
                    caller_phone[:6] + "****" if len(caller_phone) > 6 else caller_phone
                )
            except asyncio.CancelledError:
                logger.debug("Customer prefetch cancelled")
            except Exception as e:
                logger.error("Failed to inject customer context: %s", e)
        
        # Start context injection in background (fire-and-forget)
        asyncio.create_task(
            inject_customer_context(),
            name=f"inject-{call_sid[:8]}"
        )
    
    setup_time = (time.monotonic() - start_time) * 1000
    logger.info(
        "Pipecat pipeline created in %.1fms for call_sid=%s%s",
        setup_time,
        call_sid,
        " (context loading in background)" if prefetch_task else "",
    )
    
    return task, audio_buffer, context


async def run_bot(
    transport: Any,
    caller_phone: str = "",
    call_sid: str = "",
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_handlers: Optional[Dict[str, Callable]] = None,
    skip_greeting: bool = False,
) -> None:
    """
    Run the bot pipeline to completion.
    
    Args:
        transport: The transport layer
        caller_phone: Caller's phone number
        call_sid: Call identifier
        tools: Pre-cached tool definitions (from app.state)
        tool_handlers: Pre-cached tool handlers (from app.state)
        skip_greeting: If True, skip playing greeting (already sent via direct injection)
    """
    # Use pre-cached tools if provided, otherwise load dynamically
    if tools is None or tool_handlers is None:
        # Fallback: Import tools dynamically (slower path)
        from .tools import get_all_tools, get_tool_handlers
        tools = tools or get_all_tools()
        tool_handlers = tool_handlers or get_tool_handlers()
        logger.info("Tools loaded dynamically (not pre-cached)")
    else:
        logger.info("Using pre-cached tools from app.state")
    
    logger.info(
        "Starting bot with %d tools for call_sid=%s",
        len(tools), call_sid,
    )
    
    # Track call start time for duration calculation
    call_start_time = time.monotonic()
    
    task, audio_buffer, llm_context = await create_bot_pipeline(
        transport,
        caller_phone=caller_phone,
        call_sid=call_sid,
        tools=tools,
        tool_handlers=tool_handlers,
    )
    
    try:
        runner = PipelineRunner(handle_sigint=False)
        
        # Import frame types
        from pipecat.frames.frames import LLMRunFrame, OutputAudioRawFrame
        
        # Play pre-recorded greeting (unless already sent via direct injection)
        greeting_played = skip_greeting  # If skip_greeting=True, consider it already played
        if SETTINGS.greeting_audio_enabled and not skip_greeting:
            try:
                from .services.pre_recorded import get_clip_frames
                frames = get_clip_frames(SETTINGS.greeting_audio_clip)
                
                if frames:
                    logger.info(
                        "Queueing pre-recorded greeting: %d frames (~%.1fs)",
                        len(frames), len(frames) * 0.02  # 20ms per frame
                    )
                    # Queue audio frames to be sent when pipeline starts
                    audio_frames = [
                        OutputAudioRawFrame(
                            audio=audio_data,
                            sample_rate=8000,
                            num_channels=1,
                        )
                        for audio_data in frames
                    ]
                    await task.queue_frames(audio_frames)
                    greeting_played = True
                    
            except Exception as e:
                logger.warning("Failed to load pre-recorded greeting: %s", e)
        
        # If no pre-recorded greeting, trigger LLM greeting
        if not greeting_played:
            logger.info("Triggering LLM greeting")
            await task.queue_frames([LLMRunFrame()])
        
        # Register disconnect handler
        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            logger.info("Client disconnected")
            await task.cancel()
        
        # Just run the pipeline - FastAPIWebsocketTransport handles WebSocket internally
        await runner.run(task)
    except asyncio.CancelledError:
        logger.info("Bot pipeline cancelled for call_sid=%s", call_sid)
    except Exception as e:
        logger.exception("Bot pipeline error for call_sid=%s: %s", call_sid, e)
        raise
    finally:
        # Calculate call duration
        call_duration = time.monotonic() - call_start_time
        
        # Save recording on call end
        if audio_buffer and SETTINGS.recording_enabled:
            try:
                # Stop recording first
                audio_buffer.stop_recording()
                
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
        
        # Post-call processing: Save call summary to CRM
        if SETTINGS.auto_save_call_summary:
            try:
                # Extract messages from LLM context
                messages = list(llm_context.messages) if llm_context and hasattr(llm_context, 'messages') else []
                
                logger.info(
                    "Starting post-call processing: call_sid=%s, duration=%.1fs, messages=%d",
                    call_sid, call_duration, len(messages)
                )
                
                # Process post-call asynchronously (fire-and-forget)
                asyncio.create_task(
                    process_post_call(
                        call_sid=call_sid,
                        caller_phone=caller_phone,
                        duration=call_duration,
                        messages=messages,
                        customer_id=None,  # Will be looked up by phone
                    ),
                    name=f"post-call-{call_sid[:8]}"
                )
                
            except Exception as e:
                logger.error("Failed to start post-call processing: %s", e)
        
        logger.info("Bot pipeline completed for call_sid=%s", call_sid)

