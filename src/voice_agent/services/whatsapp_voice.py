"""
WhatsApp Voice Utilities - Stub for Standalone Pipecat Agent

The original whatsapp_voice.py depends on voice_agent.core.tts and
voice_agent.core.stt from the original voice pipeline.

For the Pipecat standalone agent, TTS is handled by Pipecat's Cartesia
integration, so these functions are stubbed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

from voice_agent.config import SETTINGS

logger = logging.getLogger("voice_agent.whatsapp_voice")


def media_base_url_configured() -> bool:
    """Check if media base URL is configured."""
    base = SETTINGS.whatsapp_media_base_url.strip()
    if base:
        return True
    fallback = SETTINGS.communication_webhook_base.strip()
    return bool(fallback)


async def generate_audio_reply(
    text: str,
    *,
    trace_id: str | None = None,
) -> Tuple[Path, str]:
    """
    Generate TTS audio and return path and URL.
    
    Note: In standalone Pipecat agent, use Pipecat's TTS service instead.
    This is a stub that raises NotImplementedError.
    """
    raise NotImplementedError(
        "generate_audio_reply is not available in standalone Pipecat agent. "
        "Use Pipecat's Cartesia TTS service instead."
    )
