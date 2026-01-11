"""Structured trace helpers for consistent event logging.

This module centralizes trace event construction to ensure:
- Consistent payload structure across the codebase
- Proper field truncation to prevent log bloat
- Type safety and validation
"""

from __future__ import annotations

from .metrics import Trace

# Optional latency visualization hook
try:
    from . import latency_viz

    LATENCY_VIZ_ENABLED = True
except ImportError:
    LATENCY_VIZ_ENABLED = False

# Maximum lengths for truncation
_MAX_TEXT_LEN = 120
_MAX_TOKEN_LEN = 50
_MAX_ERROR_LEN = 200
_MAX_PREVIEW_LEN = 80


def _truncate(text: str, max_len: int) -> str:
    """Truncate text to max_len with ellipsis if needed."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


# ============================================================================
# Turn & Session Lifecycle
# ============================================================================


def trace_turn_begin(
    call_sid: str, turn_id: int, trace_id: str, *, seq: int = 0
) -> None:
    """Log the start of a new turn."""
    Trace(call_sid, turn_id, "turn_begin", trace_id=trace_id, seq=seq).log()
    if LATENCY_VIZ_ENABLED:
        latency_viz.on_turn_begin(call_sid, turn_id)


def trace_turn_reset(
    call_sid: str,
    turn_id: int,
    trace_id: str,
    *,
    old_turn: int,
    seq: int = 0,
) -> None:
    """Log turn reset after interruption."""
    Trace(
        call_sid,
        turn_id,
        "turn_reset",
        trace_id=trace_id,
        seq=seq,
        extra={"old": old_turn},
    ).log()


def trace_barge_in(
    call_sid: str, turn_id: int, trace_id: str, *, seq: int = 0
) -> None:
    """Log user interruption (barge-in)."""
    Trace(call_sid, turn_id, "barge_in", trace_id=trace_id, seq=seq).log()


def trace_twilio_session_start(
    call_sid: str, turn_id: int, trace_id: str, *, seq: int = 0
) -> None:
    """Log Twilio session start."""
    Trace(
        call_sid, turn_id, "twilio_session_start", trace_id=trace_id, seq=seq
    ).log()


def trace_twilio_session_closed(
    call_sid: str, turn_id: int, trace_id: str, *, seq: int = 0
) -> None:
    """Log Twilio session close."""
    Trace(
        call_sid, turn_id, "twilio_session_closed", trace_id=trace_id, seq=seq
    ).log()


def trace_twilio_mark(
    call_sid: str, turn_id: int, trace_id: str, *, mark_name: str, seq: int = 0
) -> None:
    """Log Twilio mark event."""
    Trace(
        call_sid,
        turn_id,
        f"twilio_mark:{mark_name}",
        trace_id=trace_id,
        seq=seq,
    ).log()


# ============================================================================
# VAD Events
# ============================================================================


def trace_vad_init(
    call_sid: str,
    turn_id: int,
    trace_id: str,
    *,
    vad_impl: str,
    seq: int = 0,
) -> None:
    """Log VAD initialization."""
    Trace(
        call_sid,
        turn_id,
        "vad_init",
        trace_id=trace_id,
        seq=seq,
        extra={"impl": vad_impl},
    ).log()


def trace_vad_debug(
    call_sid: str,
    turn_id: int,
    trace_id: str,
    *,
    metrics: dict,
    seq: int = 0,
) -> None:
    """Log VAD debug metrics."""
    Trace(
        call_sid,
        turn_id,
        "vad_debug",
        trace_id=trace_id,
        seq=seq,
        extra=dict(metrics),
    ).log()


# ============================================================================
# STT Events
# ============================================================================


def trace_stt_final(
    call_sid: str,
    turn_id: int,
    trace_id: str,
    *,
    text: str,
    seq: int = 0,
) -> None:
    """Log final STT transcription."""
    Trace(
        call_sid,
        turn_id,
        "stt_final",
        trace_id=trace_id,
        seq=seq,
        extra={"text": _truncate(text, _MAX_TEXT_LEN)},
    ).log()
    if LATENCY_VIZ_ENABLED:
        latency_viz.on_stt_final(call_sid, turn_id)


def trace_stt_flush(
    call_sid: str,
    turn_id: int,
    trace_id: str,
    *,
    reason: str,
    seq: int = 0,
) -> None:
    """Log STT flush request."""
    Trace(
        call_sid,
        turn_id,
        "stt_flush",
        trace_id=trace_id,
        seq=seq,
        extra={"reason": reason},
    ).log()


def trace_input_too_short(
    call_sid: str,
    turn_id: int,
    trace_id: str,
    *,
    text: str,
    seq: int = 0,
) -> None:
    """Log input rejected for being too short."""
    Trace(
        call_sid,
        turn_id,
        "input_too_short",
        trace_id=trace_id,
        seq=seq,
        extra={"text": _truncate(text, _MAX_TEXT_LEN)},
    ).log()


def trace_only_fillers(
    call_sid: str,
    turn_id: int,
    trace_id: str,
    *,
    text: str,
    seq: int = 0,
) -> None:
    """Log input rejected for containing only filler words."""
    Trace(
        call_sid,
        turn_id,
        "only_fillers",
        trace_id=trace_id,
        seq=seq,
        extra={"text": _truncate(text, _MAX_TEXT_LEN)},
    ).log()


# ============================================================================
# LLM Events
# ============================================================================


def trace_llm_first_token(
    call_sid: str, turn_id: int, trace_id: str, *, seq: int = 0
) -> None:
    """Log first LLM token received."""
    Trace(call_sid, turn_id, "llm_first_token", trace_id=trace_id, seq=seq).log()
    if LATENCY_VIZ_ENABLED:
        latency_viz.on_first_token(call_sid, turn_id)


def trace_llm_done(
    call_sid: str, turn_id: int, trace_id: str, *, seq: int = 0
) -> None:
    """Log LLM completion."""
    Trace(call_sid, turn_id, "llm_done", trace_id=trace_id, seq=seq).log()
    if LATENCY_VIZ_ENABLED:
        latency_viz.on_llm_done(call_sid, turn_id)


def trace_llm_error(
    call_sid: str,
    turn_id: int,
    trace_id: str,
    *,
    error: Exception | str,
    seq: int = 0,
) -> None:
    """Log LLM error."""
    error_str = repr(error) if isinstance(error, Exception) else error
    Trace(
        call_sid,
        turn_id,
        "llm_error",
        trace_id=trace_id,
        seq=seq,
        extra={"err": _truncate(error_str, _MAX_ERROR_LEN)},
    ).log()


def trace_buffering_token(
    call_sid: str,
    turn_id: int,
    trace_id: str,
    *,
    token: str,
    seq: int = 0,
) -> None:
    """Log LLM token buffering."""
    Trace(
        call_sid,
        turn_id,
        "buffering_token",
        trace_id=trace_id,
        seq=seq,
        extra={"token": _truncate(token, _MAX_TOKEN_LEN)},
    ).log()


# ============================================================================
# TTS Events
# ============================================================================


def trace_tts_flush(
    call_sid: str,
    turn_id: int,
    trace_id: str,
    *,
    chars: int,
    final: bool,
    preview: str,
    seq: int = 0,
) -> None:
    """Log TTS text flush."""
    Trace(
        call_sid,
        turn_id,
        "tts_flush",
        trace_id=trace_id,
        seq=seq,
        extra={
            "chars": chars,
            "final": final,
            "preview": _truncate(preview, _MAX_PREVIEW_LEN),
        },
    ).log()


def trace_tts_audio_start(
    call_sid: str, turn_id: int, trace_id: str, *, seq: int = 0
) -> None:
    """Log TTS audio stream start."""
    Trace(call_sid, turn_id, "tts_audio_start", trace_id=trace_id, seq=seq).log()
    if LATENCY_VIZ_ENABLED:
        latency_viz.on_tts_start(call_sid, turn_id)


def trace_tts_audio_first_chunk(
    call_sid: str, turn_id: int, trace_id: str, *, seq: int = 0
) -> None:
    """Log first TTS audio chunk received."""
    Trace(
        call_sid, turn_id, "tts_audio_first_chunk", trace_id=trace_id, seq=seq
    ).log()
    if LATENCY_VIZ_ENABLED:
        latency_viz.on_tts_first_chunk(call_sid, turn_id)


def trace_tts_audio_done(
    call_sid: str,
    turn_id: int,
    trace_id: str,
    *,
    chunks: int,
    seq: int = 0,
) -> None:
    """Log TTS audio stream completion."""
    Trace(
        call_sid,
        turn_id,
        "tts_audio_done",
        trace_id=trace_id,
        seq=seq,
        extra={"chunks": chunks},
    ).log()
    if LATENCY_VIZ_ENABLED:
        latency_viz.on_turn_complete(call_sid, turn_id)


def trace_tts_audio_cancelled(
    call_sid: str, turn_id: int, trace_id: str, *, seq: int = 0
) -> None:
    """Log TTS audio stream cancellation."""
    Trace(
        call_sid, turn_id, "tts_audio_cancelled", trace_id=trace_id, seq=seq
    ).log()


def trace_tts_audio_no_session(
    call_sid: str, turn_id: int, trace_id: str, *, seq: int = 0
) -> None:
    """Log missing TTS session."""
    Trace(
        call_sid, turn_id, "tts_audio_no_session", trace_id=trace_id, seq=seq
    ).log()


def trace_tts_error(
    call_sid: str,
    turn_id: int,
    trace_id: str,
    *,
    error: Exception | str,
    seq: int = 0,
) -> None:
    """Log TTS error."""
    error_str = repr(error) if isinstance(error, Exception) else error
    Trace(
        call_sid,
        turn_id,
        "tts_error",
        trace_id=trace_id,
        seq=seq,
        extra={"err": _truncate(error_str, _MAX_ERROR_LEN)},
    ).log()


# ============================================================================
# Audio Playback Events
# ============================================================================


def trace_play_greeting(
    call_sid: str,
    turn_id: int,
    trace_id: str,
    *,
    frames: int,
    seq: int = 0,
) -> None:
    """Log greeting audio playback."""
    Trace(
        call_sid,
        turn_id,
        "play_greeting",
        trace_id=trace_id,
        seq=seq,
        extra={"frames": frames},
    ).log()


def trace_play_silence_fill(
    call_sid: str,
    turn_id: int,
    trace_id: str,
    *,
    frames: int | None = None,
    seq: int = 0,
) -> None:
    """Log silence fill audio playback."""
    extra = {}
    if frames is not None:
        extra["frames"] = frames
    Trace(
        call_sid,
        turn_id,
        "play_silence_fill",
        trace_id=trace_id,
        seq=seq,
        extra=extra,
    ).log()


def trace_silence_fill_cancelled(
    call_sid: str, turn_id: int, trace_id: str, *, seq: int = 0
) -> None:
    """Log silence fill cancellation."""
    Trace(
        call_sid, turn_id, "silence_fill_cancelled", trace_id=trace_id, seq=seq
    ).log()


def trace_dtmf_input(
    call_sid: str,
    turn_id: int,
    trace_id: str,
    *,
    digit: str,
    track: str = "inbound_track",
    seq: int = 0,
) -> None:
    """Log DTMF digit input from phone keypad."""
    Trace(
        call_sid,
        turn_id,
        "dtmf_input",
        trace_id=trace_id,
        seq=seq,
        extra={"digit": digit, "track": track},
    ).log()
