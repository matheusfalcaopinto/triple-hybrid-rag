"""Call schemas."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SentimentInfo(BaseModel):
    """Call sentiment information."""

    label: str | None = None  # positive, negative, neutral
    score: float | None = None
    computed_at: datetime | None = None


class CallResponse(BaseModel):
    """Call response."""

    model_config = ConfigDict(from_attributes=True)

    call_id: str
    provider: str
    provider_call_sid: str
    establishment_id: str
    agent_id: str | None = None
    agent_version_id: str | None = None
    status: str
    direction: str
    from_e164: str
    to_e164: str
    started_at: datetime | None = None
    answered_at: datetime | None = None
    ended_at: datetime | None = None
    duration_seconds: int | None = None
    sentiment: SentimentInfo | None = None
    hangup_by: str | None = None
    final_status: str | None = None


class CallsListResponse(BaseModel):
    """List of calls."""

    items: list[CallResponse]
    total: int | None = None
    cursor: str | None = None


class ActiveCallsResponse(BaseModel):
    """Active calls response."""

    items: list[CallResponse]


class CallEventResponse(BaseModel):
    """Call event response."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    call_id: str
    event_type: str
    occurred_at: datetime
    payload: dict[str, Any] | None = None


class TranscriptSegmentResponse(BaseModel):
    """Transcript segment response."""

    model_config = ConfigDict(from_attributes=True)

    segment_id: str
    speaker: str  # customer, agent, system
    text: str
    started_at: datetime
    ended_at: datetime
    confidence: float | None = None


class TranscriptResponse(BaseModel):
    """Transcript response with cursor pagination."""

    call_id: str
    items: list[TranscriptSegmentResponse]
    cursor: str | None = None
    is_complete: bool = False


class OutboundCallRequest(BaseModel):
    """Outbound call request."""

    to_e164: str = Field(..., pattern=r"^\+[1-9]\d{1,14}$")
    from_e164: str | None = None  # Use default if not provided
    agent_id: str | None = None
    agent_version_id: str | None = None
    lead_id: str | None = None
    campaign_id: str | None = None


class CallHandoffRequest(BaseModel):
    """Call handoff request."""

    reason: str | None = None


class CallHandoffResponse(BaseModel):
    """Call handoff response."""

    call_id: str
    status: str
    handoff_e164: str
