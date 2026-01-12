"""
Twilio WebSocket Transport for Pipecat

This module provides a transport layer that bridges Twilio's Media Streams
WebSocket protocol with Pipecat's frame-based pipeline.

Features:
- Audio streaming (µ-law 8kHz)
- DTMF handling via DTMFFrame
- Barge-in/interruption support via clear audio
- Mark events for audio synchronization
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from fastapi import WebSocket

from pipecat.frames.frames import (
    AudioRawFrame,
    CancelFrame,
    DTMFFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
    StartFrame,
    StartInterruptionFrame,
    TransportMessageFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transports.base_transport import TransportParams
from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADState

logger = logging.getLogger("voice_agent_pipecat.transports.twilio")


# ──────────────────────────────────────────────────────────────────────────────
# DTMF Digit Collector
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DigitCollector:
    """Collects DTMF digits until a condition is met."""
    
    expected_length: Optional[int] = None
    terminator: str = "#"
    timeout_seconds: float = 10.0
    digits: List[str] = field(default_factory=list)
    start_time: float = 0.0
    
    @property
    def completed(self) -> bool:
        if self.expected_length and len(self.digits) >= self.expected_length:
            return True
        return False
    
    def start(self) -> None:
        self.start_time = time.monotonic()
        self.digits.clear()
    
    def reset(self) -> None:
        self.start_time = 0.0
        self.digits.clear()
    
    def is_timed_out(self) -> bool:
        if self.start_time == 0.0:
            return False
        return (time.monotonic() - self.start_time) > self.timeout_seconds
    
    def get_partial(self) -> str:
        return "".join(self.digits)
    
    def add_digit(self, digit: str) -> tuple[bool, str]:
        """Add a digit. Returns (is_complete, collected_digits)."""
        if digit == self.terminator:
            return True, "".join(self.digits)
        
        if not digit.isdigit():
            return False, "".join(self.digits)
        
        self.digits.append(digit)
        collected = "".join(self.digits)
        
        return self.completed, collected


# ──────────────────────────────────────────────────────────────────────────────
# Transport Parameters
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TwilioTransportParams:
    """Configuration parameters for Twilio transport."""
    
    sample_rate: int = 8000
    num_channels: int = 1
    
    # Interruption settings
    enable_interruptions: bool = True
    interruption_min_words: int = 1
    
    # DTMF settings
    dtmf_interrupts_tts: bool = False
    dtmf_collection_timeout: float = 10.0
    dtmf_phone_min_digits: int = 6
    
    # Mark tracking
    enable_marks: bool = True


# ──────────────────────────────────────────────────────────────────────────────
# Main Transport Class
# ──────────────────────────────────────────────────────────────────────────────

class TwilioWebsocketTransport:
    """
    Transport adapter for Twilio Media Streams WebSocket.
    
    Handles:
    - Receiving audio from Twilio (µ-law 8kHz)
    - Sending audio to Twilio
    - DTMF digit handling
    - Media stream lifecycle (start, connected, stop)
    - Mark events for audio synchronization
    - Barge-in via audio clear
    """
    
    def __init__(
        self,
        websocket: WebSocket,
        sample_rate: int = 8000,
        params: Optional[TwilioTransportParams] = None,
    ) -> None:
        self._websocket = websocket
        self._sample_rate = sample_rate
        self._params = params or TwilioTransportParams(sample_rate=sample_rate)
        self._stream_sid: Optional[str] = None
        self._call_sid: Optional[str] = None
        self._input_processor = TwilioInputProcessor(self)
        self._output_processor = TwilioOutputProcessor(self)
        self._running = False
        
        # DTMF collection
        self._digit_collector: Optional[DigitCollector] = None
        self._dtmf_callback: Optional[Callable[[str], None]] = None
        
        # Mark tracking
        self._pending_marks: Dict[str, float] = {}
        self._mark_callbacks: Dict[str, Callable[[], None]] = {}
        
        # Stats
        self._audio_frames_received = 0
        self._audio_frames_sent = 0
        
    @property
    def stream_sid(self) -> Optional[str]:
        return self._stream_sid
    
    @property
    def call_sid(self) -> Optional[str]:
        return self._call_sid
    
    @property
    def is_running(self) -> bool:
        return self._running
    
    def input_processor(self) -> FrameProcessor:
        """Get the input frame processor for the pipeline."""
        return self._input_processor
    
    def output_processor(self) -> FrameProcessor:
        """Get the output frame processor for the pipeline."""
        return self._output_processor
    
    # ──────────────────────────────────────────────────────────────────────────
    # DTMF Collection
    # ──────────────────────────────────────────────────────────────────────────
    
    def start_dtmf_collection(
        self,
        expected_length: Optional[int] = None,
        terminator: str = "#",
        timeout: Optional[float] = None,
        callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Start collecting DTMF digits."""
        self._digit_collector = DigitCollector(
            expected_length=expected_length,
            terminator=terminator,
            timeout_seconds=timeout or self._params.dtmf_collection_timeout,
        )
        self._digit_collector.start()
        self._dtmf_callback = callback
        logger.debug("Started DTMF collection (length=%s, terminator=%s)", 
                     expected_length, terminator)
    
    def stop_dtmf_collection(self) -> str:
        """Stop collecting and return collected digits."""
        if self._digit_collector:
            digits = self._digit_collector.get_partial()
            self._digit_collector = None
            self._dtmf_callback = None
            return digits
        return ""
    
    def _handle_dtmf(self, digit: str) -> None:
        """Handle incoming DTMF digit."""
        logger.debug("Received DTMF: %s", digit)
        
        if self._digit_collector:
            complete, collected = self._digit_collector.add_digit(digit)
            if complete and self._dtmf_callback:
                self._dtmf_callback(collected)
                self._digit_collector = None
    
    # ──────────────────────────────────────────────────────────────────────────
    # Audio Output
    # ──────────────────────────────────────────────────────────────────────────
    
    async def send_audio(self, audio: bytes) -> None:
        """Send audio data to Twilio."""
        if not self._stream_sid:
            logger.warning("Cannot send audio - no stream_sid")
            return
            
        payload = base64.b64encode(audio).decode("utf-8")
        message = {
            "event": "media",
            "streamSid": self._stream_sid,
            "media": {"payload": payload},
        }
        await self._websocket.send_json(message)
        self._audio_frames_sent += 1
    
    async def send_mark(self, name: str, callback: Optional[Callable[[], None]] = None) -> None:
        """Send a mark event to Twilio for synchronization."""
        if not self._stream_sid:
            return
            
        message = {
            "event": "mark",
            "streamSid": self._stream_sid,
            "mark": {"name": name},
        }
        await self._websocket.send_json(message)
        
        self._pending_marks[name] = time.monotonic()
        if callback:
            self._mark_callbacks[name] = callback
        
        logger.debug("Sent mark: %s", name)
    
    async def clear_audio(self) -> None:
        """Clear queued audio on Twilio side (for interruptions/barge-in)."""
        if not self._stream_sid:
            return
            
        message = {
            "event": "clear",
            "streamSid": self._stream_sid,
        }
        await self._websocket.send_json(message)
        logger.debug("Cleared audio queue (barge-in)")
    
    # ──────────────────────────────────────────────────────────────────────────
    # WebSocket Message Loop
    # ──────────────────────────────────────────────────────────────────────────
    
    async def run(self) -> None:
        """
        Run the transport - receive messages from Twilio WebSocket.
        This should be run as a task alongside the pipeline.
        """
        self._running = True
        
        try:
            while self._running:
                try:
                    data = await self._websocket.receive_text()
                    message = json.loads(data)
                    await self._handle_message(message)
                except Exception as e:
                    if "disconnect" in str(e).lower():
                        break
                    logger.exception("Error receiving WebSocket message: %s", e)
                    break
        finally:
            self._running = False
            await self._input_processor.push_frame(EndFrame())
            logger.info(
                "Transport stopped: frames_rx=%d, frames_tx=%d",
                self._audio_frames_received, self._audio_frames_sent,
            )
    
    async def _handle_message(self, message: dict) -> None:
        """Handle incoming Twilio WebSocket message."""
        event = message.get("event", "")
        
        if event == "connected":
            logger.info("Twilio stream connected")
            
        elif event == "start":
            self._stream_sid = message.get("streamSid")
            start_data = message.get("start", {})
            self._call_sid = start_data.get("callSid")
            logger.info(
                "Twilio stream started: stream_sid=%s, call_sid=%s",
                self._stream_sid, self._call_sid,
            )
            await self._input_processor.push_frame(StartFrame())
            
        elif event == "media":
            media = message.get("media", {})
            payload = media.get("payload", "")
            if payload:
                audio_bytes = base64.b64decode(payload)
                frame = InputAudioRawFrame(
                    audio=audio_bytes,
                    sample_rate=self._sample_rate,
                    num_channels=1,
                )
                await self._input_processor.push_frame(frame)
                self._audio_frames_received += 1
                
        elif event == "dtmf":
            # Handle DTMF digits
            dtmf_data = message.get("dtmf", {})
            digit = dtmf_data.get("digit", "")
            if digit:
                self._handle_dtmf(digit)
                # Also push as DTMFFrame for pipeline processing
                await self._input_processor.push_frame(DTMFFrame(dtmf=digit))
                
                # Optionally interrupt TTS on DTMF
                if self._params.dtmf_interrupts_tts:
                    await self.clear_audio()
                
        elif event == "mark":
            mark = message.get("mark", {})
            name = mark.get("name", "")
            logger.debug("Received mark: %s", name)
            
            # Track mark completion
            if name in self._pending_marks:
                del self._pending_marks[name]
            
            # Execute callback if registered
            if name in self._mark_callbacks:
                callback = self._mark_callbacks.pop(name)
                callback()
            
            await self._input_processor.push_frame(
                TransportMessageFrame(message={"type": "mark", "name": name})
            )
            
        elif event == "stop":
            logger.info("Twilio stream stopped")
            self._running = False
            await self._input_processor.push_frame(EndFrame())


# ──────────────────────────────────────────────────────────────────────────────
# Input Processor
# ──────────────────────────────────────────────────────────────────────────────

class TwilioInputProcessor(FrameProcessor):
    """
    Input processor that receives frames from the transport.
    Acts as the entry point for audio into the pipeline.
    """
    
    def __init__(self, transport: TwilioWebsocketTransport) -> None:
        super().__init__()
        self._transport = transport
        self._queue: asyncio.Queue[Frame] = asyncio.Queue()
    
    async def push_frame(self, frame: Frame) -> None:
        """Push a frame from the transport into the pipeline."""
        await self._queue.put(frame)
    
    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Process frames - for input, we pull from our queue."""
        pass
    
    async def run(self, context: Any = None) -> None:
        """Run the input processor - yield frames from the queue."""
        while True:
            frame = await self._queue.get()
            await self.push_frame_to_pipeline(frame, FrameDirection.DOWNSTREAM)
            if isinstance(frame, EndFrame):
                break


# ──────────────────────────────────────────────────────────────────────────────
# Output Processor  
# ──────────────────────────────────────────────────────────────────────────────

class TwilioOutputProcessor(FrameProcessor):
    """
    Output processor that sends frames to the transport.
    Acts as the exit point for audio from the pipeline.
    Handles barge-in by clearing audio on interruption frames.
    """
    
    def __init__(self, transport: TwilioWebsocketTransport) -> None:
        super().__init__()
        self._transport = transport
        self._is_speaking = False
    
    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Process outgoing frames - send audio to Twilio."""
        
        # Handle audio output
        if isinstance(frame, (AudioRawFrame, OutputAudioRawFrame)):
            await self._transport.send_audio(frame.audio)
            if not self._is_speaking:
                self._is_speaking = True
        
        # Handle interruptions (barge-in)
        elif isinstance(frame, StartInterruptionFrame):
            logger.debug("Barge-in detected - clearing audio")
            await self._transport.clear_audio()
            self._is_speaking = False
            
        elif isinstance(frame, CancelFrame):
            await self._transport.clear_audio()
            self._is_speaking = False
        
        # Track speaking state
        elif isinstance(frame, UserStartedSpeakingFrame):
            # User started speaking - prepare for potential barge-in
            pass
            
        elif isinstance(frame, UserStoppedSpeakingFrame):
            # User stopped speaking
            pass
        
        # Always pass frames downstream
        await self.push_frame(frame, direction)
