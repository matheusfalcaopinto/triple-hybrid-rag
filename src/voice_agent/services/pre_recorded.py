"""
Pre-recorded Audio Service

Provides utilities for loading and playing pre-recorded audio clips in µ-law format.
These are used for consistent, low-latency greetings and silence fillers.
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger("voice_agent.services.pre_recorded")

# Frame size for µ-law 8kHz mono (20ms = 160 bytes)
MULAW_FRAME_SIZE = 160
SAMPLE_RATE = 8000


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
        # Pad last frame if needed
        if len(frame) < MULAW_FRAME_SIZE:
            frame = frame + bytes(MULAW_FRAME_SIZE - len(frame))
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
