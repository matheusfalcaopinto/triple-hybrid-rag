"""Ingest schemas for runtime -> backend events."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class CallInfo(BaseModel):
    """Call information from runtime."""

    provider: str = "twilio"
    provider_call_sid: str
    direction: str  # inbound, outbound
    from_e164: str
    to_e164: str


class TranscriptSegmentPayload(BaseModel):
    """Transcript segment payload."""

    segment_id: str
    speaker: str  # customer, agent, system
    text: str
    started_at: datetime
    ended_at: datetime
    confidence: float | None = None


class ToolCalledPayload(BaseModel):
    """Tool called payload."""

    tool_name: str
    arguments: dict[str, Any] | None = None


class ToolResultPayload(BaseModel):
    """Tool result payload."""

    tool_name: str
    result: dict[str, Any] | None = None
    error: str | None = None
    duration_ms: int | None = None


class RecordingAvailablePayload(BaseModel):
    """Recording available payload."""

    recording_url: str
    storage_provider: str = "s3"
    content_type: str = "audio/wav"
    duration_seconds: int | None = None


class CallStartedPayload(BaseModel):
    """Call started payload."""

    started_at: datetime


class CallEndedPayload(BaseModel):
    """Call ended payload."""

    ended_at: datetime
    duration_seconds: int
    final_status: str
    hangup_by: str | None = None


class RuntimeErrorPayload(BaseModel):
    """Runtime error payload."""

    component: str  # stt, tts, llm, transport
    message: str
    fatal: bool = False


class CallEventPayload(BaseModel):
    """Union of all event payloads."""

    # Generic event fields
    id: str
    type: Literal[
        "call_started",
        "call_ended",
        "transcript_segment",
        "tool_called",
        "tool_result",
        "recording_available",
        "runtime_error",
    ]
    occurred_at: datetime
    idempotency_key: str
    payload: dict[str, Any]


class IngestCallEventRequest(BaseModel):
    """Ingest call event request from runtime."""

    runtime_id: str
    establishment_id: str
    agent_id: str
    agent_version_id: str | None = None
    call: CallInfo
    event: CallEventPayload


class IngestResponse(BaseModel):
    """Ingest response."""

    status: str = "accepted"
