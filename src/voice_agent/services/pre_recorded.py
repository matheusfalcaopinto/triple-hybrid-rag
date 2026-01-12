"""
Pre-recorded Audio Service

Provides utilities for loading and playing pre-recorded audio clips in µ-law format.
These are used for consistent, low-latency greetings and silence fillers.

Features:
- In-memory caching for instant access
- Pre-encoded Base64 frames for direct Twilio WebSocket injection
- Async preloading at application startup
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import WebSocket

logger = logging.getLogger("voice_agent.services.pre_recorded")

# Frame size for µ-law 8kHz mono (20ms = 160 bytes)
MULAW_FRAME_SIZE = 160
SAMPLE_RATE = 8000

# ══════════════════════════════════════════════════════════════════════════════
# In-Memory Audio Cache
# ══════════════════════════════════════════════════════════════════════════════

# Raw audio frames cache (bytes)
_audio_cache: Dict[str, List[bytes]] = {}

# Pre-encoded Base64 frames for direct Twilio injection
_mulaw_base64_cache: Dict[str, List[str]] = {}

# Flag to track if clips have been preloaded
_preloaded: bool = False


def get_audio_assets_dir() -> Path:
    """Get the audio assets directory path."""
    # Try package-relative first
    pkg_dir = Path(__file__).parent.parent.parent.parent / "audio_assets"
    if pkg_dir.exists():
        return pkg_dir
    # Fallback to cwd
    return Path.cwd() / "audio_assets"


def load_raw_audio(file_path: Path) -> bytes:
    """
    Load a raw µ-law audio file.
    
    Args:
        file_path: Path to the .raw audio file
        
    Returns:
        Raw audio bytes
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    with open(file_path, "rb") as f:
        return f.read()


def get_clip_frames(
    clip_name: str,
    base_dir: Optional[str] = None,
) -> List[bytes]:
    """
    Get audio frames for a named clip.
    
    Args:
        clip_name: Name of the clip (without extension)
        base_dir: Optional base directory (defaults to audio_assets/)
        
    Returns:
        List of 160-byte audio frames (20ms each at 8kHz)
    """
    if base_dir:
        audio_dir = Path(base_dir)
    else:
        audio_dir = get_audio_assets_dir()
    
    # Handle special case for random silence fillings
    if clip_name == "random_silence_filling":
        return _get_random_silence_filling(audio_dir)
    
    # Try various extensions
    for ext in [".raw", ".mulaw", ".ulaw"]:
        file_path = audio_dir / f"{clip_name}{ext}"
        if file_path.exists():
            audio_data = load_raw_audio(file_path)
            return _split_into_frames(audio_data)
    
    # Also check in subdirectory with clip name
    clip_dir = audio_dir / clip_name
    if clip_dir.is_dir():
        # Look for default file in the directory
        for ext in [".raw", ".mulaw", ".ulaw"]:
            default_file = clip_dir / f"{clip_name}{ext}"
            if default_file.exists():
                audio_data = load_raw_audio(default_file)
                return _split_into_frames(audio_data)
    
    # Try greetings.raw for "greetings" clip
    if clip_name == "greetings" or clip_name == "greeting":
        greetings_file = audio_dir / "greetings.raw"
        if greetings_file.exists():
            audio_data = load_raw_audio(greetings_file)
            return _split_into_frames(audio_data)
    
    raise FileNotFoundError(f"Audio clip not found: {clip_name}")


def _get_random_silence_filling(audio_dir: Path) -> List[bytes]:
    """Get a random silence filling clip."""
    silence_dir = audio_dir / "silence_fillings"
    if not silence_dir.exists():
        raise FileNotFoundError(f"Silence fillings directory not found: {silence_dir}")
    
    # Find all raw files
    files = list(silence_dir.glob("*.raw"))
    if not files:
        raise FileNotFoundError(f"No silence filling clips found in: {silence_dir}")
    
    # Pick random file
    chosen = random.choice(files)
    audio_data = load_raw_audio(chosen)
    logger.debug("Using random silence filling: %s", chosen.name)
    return _split_into_frames(audio_data)


def _split_into_frames(audio_data: bytes) -> List[bytes]:
    """
    Split raw audio data into 160-byte frames (20ms at 8kHz).
    
    Args:
        audio_data: Raw µ-law audio bytes
        
    Returns:
        List of 160-byte frames
    """
    frames = []
    for i in range(0, len(audio_data), MULAW_FRAME_SIZE):
        frame = audio_data[i:i + MULAW_FRAME_SIZE]
        # Pad last frame with µ-law silence (0xFF) if needed
        # Note: 0x00 in µ-law is a loud pop, 0xFF is silence
        if len(frame) < MULAW_FRAME_SIZE:
            padding_size = MULAW_FRAME_SIZE - len(frame)
            frame = frame + bytes([0xFF] * padding_size)
        frames.append(frame)
    return frames


def get_greeting_audio_bytes() -> Optional[bytes]:
    """
    Get the greeting audio as raw bytes.
    
    Returns:
        Raw µ-law audio bytes, or None if not found
    """
    try:
        audio_dir = get_audio_assets_dir()
        greetings_file = audio_dir / "greetings.raw"
        if greetings_file.exists():
            return load_raw_audio(greetings_file)
        return None
    except Exception as e:
        logger.warning("Failed to load greeting audio: %s", e)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Preloading & Caching Functions
# ══════════════════════════════════════════════════════════════════════════════

async def preload_all_clips() -> None:
    """
    Pre-load all audio clips into memory at application startup.
    
    This function:
    1. Loads raw audio files from disk
    2. Splits them into 20ms frames
    3. Pre-encodes frames as Base64 for direct Twilio injection
    
    Call this during application lifespan startup for instant greeting playback.
    """
    global _audio_cache, _mulaw_base64_cache, _preloaded
    
    clips_to_load = ["greetings"]  # Add more clip names as needed
    
    for clip_name in clips_to_load:
        try:
            frames = _load_clip_to_cache(clip_name)
            if frames:
                logger.info(
                    "Cached audio clip: %s (%d frames, ~%.1fs)",
                    clip_name, len(frames), len(frames) * 0.02
                )
        except FileNotFoundError:
            logger.warning("Audio clip not found for preloading: %s", clip_name)
        except Exception as e:
            logger.error("Failed to preload clip %s: %s", clip_name, e)
    
    _preloaded = True
    logger.info("Audio preloading complete. Cached clips: %s", list(_audio_cache.keys()))


def _load_clip_to_cache(clip_name: str) -> Optional[List[bytes]]:
    """
    Load a clip into both raw and Base64 caches.
    
    Args:
        clip_name: Name of the clip to load
        
    Returns:
        List of raw audio frames, or None if not found
    """
    global _audio_cache, _mulaw_base64_cache
    
    # Check if already cached
    if clip_name in _audio_cache:
        return _audio_cache[clip_name]
    
    audio_dir = get_audio_assets_dir()
    
    # Try various extensions and locations
    search_paths = [
        audio_dir / f"{clip_name}.raw",
        audio_dir / f"{clip_name}.mulaw",
        audio_dir / f"{clip_name}.ulaw",
        audio_dir / clip_name / f"{clip_name}.raw",
    ]
    
    for path in search_paths:
        if path.exists():
            audio_data = load_raw_audio(path)
            frames = _split_into_frames(audio_data)
            
            # Store in raw cache
            _audio_cache[clip_name] = frames
            
            # Pre-encode as Base64 for Twilio
            _mulaw_base64_cache[clip_name] = [
                base64.b64encode(frame).decode("ascii")
                for frame in frames
            ]
            
            return frames
    
    raise FileNotFoundError(f"Audio clip not found: {clip_name}")


def get_clip_frames_base64(clip_name: str) -> Optional[List[str]]:
    """
    Get pre-encoded Base64 frames for direct Twilio WebSocket injection.
    
    This is faster than get_clip_frames() because frames are pre-encoded.
    
    Args:
        clip_name: Name of the clip
        
    Returns:
        List of Base64-encoded audio frames, or None if not found
    """
    # Check pre-encoded cache first
    if clip_name in _mulaw_base64_cache:
        return _mulaw_base64_cache[clip_name]
    
    # Fallback: load and encode on demand
    try:
        _load_clip_to_cache(clip_name)
        return _mulaw_base64_cache.get(clip_name)
    except FileNotFoundError:
        return None


def is_preloaded() -> bool:
    """Check if audio clips have been preloaded."""
    return _preloaded


# ══════════════════════════════════════════════════════════════════════════════
# Direct WebSocket Greeting Injection
# ══════════════════════════════════════════════════════════════════════════════

async def send_greeting_direct(
    websocket: "WebSocket",
    stream_sid: str,
    clip_name: str = "greetings",
) -> bool:
    """
    Send pre-recorded greeting directly via Twilio WebSocket.
    
    This bypasses the Pipecat pipeline to achieve instant playback (<100ms).
    The greeting starts playing while the pipeline is still being created.
    
    Args:
        websocket: FastAPI WebSocket connection to Twilio
        stream_sid: Twilio stream SID from the "start" event
        clip_name: Name of the audio clip to play (default: "greetings")
        
    Returns:
        True if greeting was sent successfully, False otherwise
    """
    frames = get_clip_frames_base64(clip_name)
    
    if not frames:
        logger.warning("No greeting frames available for clip: %s", clip_name)
        return False
    
    logger.info(
        "Sending greeting directly via WebSocket: %d frames (~%.1fs)",
        len(frames), len(frames) * 0.02
    )
    
    try:
        frames_sent = 0
        for payload in frames:
            await websocket.send_json({
                "event": "media",
                "streamSid": stream_sid,
                "media": {"payload": payload}
            })
            frames_sent += 1
            # Pace at slightly under real-time to prevent buffer overflow
            # 20ms per frame, but we send slightly faster to account for network jitter
            await asyncio.sleep(0.018)
        
        logger.info("Greeting sent successfully: %d frames", frames_sent)
        return True
        
    except Exception as e:
        logger.error("Failed to send greeting via WebSocket: %s", e)
        return False


async def send_audio_frames_direct(
    websocket: "WebSocket",
    stream_sid: str,
    frames: List[bytes],
) -> bool:
    """
    Send raw audio frames directly via Twilio WebSocket.
    
    Args:
        websocket: FastAPI WebSocket connection
        stream_sid: Twilio stream SID
        frames: List of raw µ-law audio frames (160 bytes each)
        
    Returns:
        True if sent successfully
    """
    try:
        for frame in frames:
            payload = base64.b64encode(frame).decode("ascii")
            await websocket.send_json({
                "event": "media",
                "streamSid": stream_sid,
                "media": {"payload": payload}
            })
            await asyncio.sleep(0.018)
        return True
    except Exception as e:
        logger.error("Failed to send audio frames: %s", e)
        return False
