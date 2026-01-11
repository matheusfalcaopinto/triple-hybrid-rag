"""
Audio Recording Service

Handles saving call audio from Pipecat's AudioBufferProcessor to disk.
"""

from __future__ import annotations

import asyncio
import logging
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pipecat.audio.buffer import AudioBufferProcessor

from ..config import SETTINGS

logger = logging.getLogger("voice_agent.services.recording")


class RecordingService:
    """Service for managing call audio recordings."""

    def __init__(self):
        self._recordings_dir = Path(SETTINGS.recording_path)
        self._recordings_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Recording service initialized. Path: %s", self._recordings_dir)

    def _generate_filename(self, call_sid: str, suffix: str = "") -> str:
        """Generate unique filename for recording."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        suffix_str = f"_{suffix}" if suffix else ""
        return f"{call_sid}_{timestamp}{suffix_str}.{SETTINGS.recording_format}"

    async def save_recording(
        self,
        buffer: "AudioBufferProcessor",
        call_sid: str,
        caller_phone: str = "",
    ) -> Optional[str]:
        """
        Save audio buffer to file.

        Args:
            buffer: Pipecat AudioBufferProcessor with captured audio
            call_sid: Unique call identifier
            caller_phone: Caller's phone number for metadata

        Returns:
            Absolute path to saved recording, or None if failed
        """
        if not SETTINGS.recording_enabled:
            logger.debug("Recording disabled, skipping save")
            return None

        try:
            # Get audio data from buffer
            # The buffer provides separate user and bot audio
            user_audio = buffer.get_user_audio()
            bot_audio = buffer.get_bot_audio() if SETTINGS.recording_include_bot_audio else None

            if not user_audio and not bot_audio:
                logger.warning("No audio data in buffer for call_sid=%s", call_sid)
                return None

            # Generate filename
            filename = self._generate_filename(call_sid)
            filepath = self._recordings_dir / filename

            # Save as WAV file
            await asyncio.to_thread(
                self._write_wav_file,
                filepath,
                user_audio,
                bot_audio,
            )

            abs_path = str(filepath.absolute())
            logger.info(
                "Recording saved: %s (call_sid=%s, caller=%s)",
                abs_path, call_sid, caller_phone[:6] + "****" if caller_phone else "N/A",
            )
            return abs_path

        except Exception as e:
            logger.exception("Failed to save recording for call_sid=%s: %s", call_sid, e)
            return None

    def _write_wav_file(
        self,
        filepath: Path,
        user_audio: bytes,
        bot_audio: Optional[bytes],
    ) -> None:
        """Write audio data to WAV file (blocking, run in thread)."""
        # If we have both user and bot audio, mix them
        if bot_audio and user_audio:
            # Simple mixing: interleave or combine
            # For simplicity, we'll save user audio only in this version
            audio_data = user_audio
        else:
            audio_data = user_audio or bot_audio or b""

        with wave.open(str(filepath), "wb") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(SETTINGS.recording_sample_rate)
            wav_file.writeframes(audio_data)


# Global instance
_recording_service: Optional[RecordingService] = None


def get_recording_service() -> RecordingService:
    """Get or create the recording service singleton."""
    global _recording_service
    if _recording_service is None:
        _recording_service = RecordingService()
    return _recording_service
