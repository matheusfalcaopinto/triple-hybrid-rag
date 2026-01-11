"""
WhatsApp WebRTC Transport for Pipecat

Custom transport that bridges WhatsApp voice calls (via Meta's WebRTC API)
to the Pipecat pipeline.

NOTE: This is a foundational implementation. Full WebRTC support requires:
- aiortc library for WebRTC stack
- Proper ICE candidate handling
- Audio codec support (Opus)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Optional

from pipecat.frames.frames import (
    AudioRawFrame,
    StartFrame,
    EndFrame,
    Frame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from ..config import SETTINGS

logger = logging.getLogger("voice_agent.transports.whatsapp_webrtc")


# Check for aiortc availability
try:
    from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate
    from aiortc.contrib.media import MediaPlayer, MediaRecorder
    AIORTC_AVAILABLE = True
except ImportError:
    AIORTC_AVAILABLE = False
    RTCPeerConnection = None
    RTCSessionDescription = None
    RTCIceCandidate = None
    logger.warning("aiortc not installed. WhatsApp WebRTC transport will not be functional.")


class WhatsAppWebRTCInputProcessor(FrameProcessor):
    """
    Processor that receives audio from WhatsApp WebRTC connection.
    
    Converts WebRTC audio tracks to Pipecat AudioRawFrames.
    """

    def __init__(self, sample_rate: int = 16000):
        super().__init__()
        self._sample_rate = sample_rate
        self._audio_queue: asyncio.Queue = asyncio.Queue()
        self._running = False

    async def start(self):
        """Start receiving audio."""
        self._running = True
        logger.info("WhatsApp WebRTC input processor started")

    async def stop(self):
        """Stop receiving audio."""
        self._running = False
        logger.info("WhatsApp WebRTC input processor stopped")

    async def receive_audio(self, audio_data: bytes):
        """
        Called when audio is received from WebRTC track.
        
        Args:
            audio_data: Raw PCM audio bytes
        """
        if self._running:
            await self._audio_queue.put(audio_data)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames."""
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self.start()
        elif isinstance(frame, EndFrame):
            await self.stop()

        await self.push_frame(frame, direction)

    async def run_audio_pump(self):
        """Pump audio from queue to pipeline."""
        while self._running:
            try:
                audio_data = await asyncio.wait_for(
                    self._audio_queue.get(),
                    timeout=0.1,
                )
                frame = AudioRawFrame(
                    audio=audio_data,
                    sample_rate=self._sample_rate,
                    num_channels=1,
                )
                await self.push_frame(frame, FrameDirection.DOWNSTREAM)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break


class WhatsAppWebRTCOutputProcessor(FrameProcessor):
    """
    Processor that sends audio to WhatsApp WebRTC connection.
    
    Converts Pipecat AudioRawFrames to WebRTC audio tracks.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        on_audio: Optional[Callable[[bytes], None]] = None,
    ):
        super().__init__()
        self._sample_rate = sample_rate
        self._on_audio = on_audio
        self._running = False

    async def start(self):
        """Start sending audio."""
        self._running = True
        logger.info("WhatsApp WebRTC output processor started")

    async def stop(self):
        """Stop sending audio."""
        self._running = False
        logger.info("WhatsApp WebRTC output processor stopped")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process outgoing frames."""
        await super().process_frame(frame, direction)

        if isinstance(frame, AudioRawFrame):
            if self._running and self._on_audio:
                try:
                    self._on_audio(frame.audio)
                except Exception as e:
                    logger.error("Error sending audio to WebRTC: %s", e)
        elif isinstance(frame, StartFrame):
            await self.start()
        elif isinstance(frame, EndFrame):
            await self.stop()

        await self.push_frame(frame, direction)


class WhatsAppWebRTCTransport:
    """
    Transport layer for WhatsApp WebRTC calls.
    
    Manages the WebRTC peer connection and audio tracks.
    """

    def __init__(
        self,
        call_id: str,
        sample_rate: int = 16000,
    ):
        self._call_id = call_id
        self._sample_rate = sample_rate
        self._pc: Optional[Any] = None  # RTCPeerConnection
        self._input_processor = WhatsAppWebRTCInputProcessor(sample_rate)
        self._output_processor = WhatsAppWebRTCOutputProcessor(
            sample_rate,
            on_audio=self._send_audio_to_webrtc,
        )
        self._audio_track = None
        self._running = False

    def input_processor(self) -> WhatsAppWebRTCInputProcessor:
        """Get the input processor for the pipeline."""
        return self._input_processor

    def output_processor(self) -> WhatsAppWebRTCOutputProcessor:
        """Get the output processor for the pipeline."""
        return self._output_processor

    async def initialize(self, sdp_offer: str) -> Optional[str]:
        """
        Initialize WebRTC connection with SDP offer.
        
        Args:
            sdp_offer: SDP offer from WhatsApp
            
        Returns:
            SDP answer to send back, or None if failed
        """
        if not AIORTC_AVAILABLE:
            logger.error("aiortc not available, cannot initialize WebRTC")
            return None

        try:
            # Create peer connection
            self._pc = RTCPeerConnection()

            # Handle incoming audio track
            @self._pc.on("track")
            def on_track(track):
                if track.kind == "audio":
                    logger.info("Received audio track from WhatsApp")
                    asyncio.create_task(self._receive_audio_from_track(track))

            # Handle connection state changes
            @self._pc.on("connectionstatechange")
            async def on_connectionstatechange():
                logger.info("WebRTC connection state: %s", self._pc.connectionState)
                if self._pc.connectionState == "failed":
                    await self.close()

            # Set remote description (offer)
            offer = RTCSessionDescription(sdp=sdp_offer, type="offer")
            await self._pc.setRemoteDescription(offer)

            # Create and set local description (answer)
            answer = await self._pc.createAnswer()
            await self._pc.setLocalDescription(answer)

            self._running = True
            logger.info("WhatsApp WebRTC transport initialized for call_id=%s", self._call_id)

            return self._pc.localDescription.sdp

        except Exception as e:
            logger.exception("Failed to initialize WebRTC: %s", e)
            return None

    async def _receive_audio_from_track(self, track):
        """Receive audio from WebRTC track and push to input processor."""
        try:
            while self._running:
                frame = await track.recv()
                # Convert to raw PCM
                # Note: actual implementation depends on aiortc audio frame format
                audio_data = frame.to_ndarray().tobytes()
                await self._input_processor.receive_audio(audio_data)
        except Exception as e:
            if self._running:
                logger.error("Error receiving audio from track: %s", e)

    def _send_audio_to_webrtc(self, audio_data: bytes):
        """Send audio to WebRTC track."""
        if self._audio_track:
            # Note: actual implementation would push to audio track
            pass

    async def add_ice_candidate(self, candidate: str, sdp_mid: str, sdp_mline_index: int):
        """Add ICE candidate from remote peer."""
        if not AIORTC_AVAILABLE or not self._pc:
            return

        try:
            ice_candidate = RTCIceCandidate(
                candidate=candidate,
                sdpMid=sdp_mid,
                sdpMLineIndex=sdp_mline_index,
            )
            await self._pc.addIceCandidate(ice_candidate)
        except Exception as e:
            logger.error("Error adding ICE candidate: %s", e)

    async def close(self):
        """Close the WebRTC connection."""
        self._running = False
        if self._pc:
            await self._pc.close()
            self._pc = None
        logger.info("WhatsApp WebRTC transport closed for call_id=%s", self._call_id)


# Registry of active WebRTC transports by call_id
_active_transports: dict[str, WhatsAppWebRTCTransport] = {}


def get_transport(call_id: str) -> Optional[WhatsAppWebRTCTransport]:
    """Get an active transport by call_id."""
    return _active_transports.get(call_id)


def register_transport(call_id: str, transport: WhatsAppWebRTCTransport):
    """Register an active transport."""
    _active_transports[call_id] = transport


def unregister_transport(call_id: str):
    """Unregister a transport."""
    _active_transports.pop(call_id, None)
