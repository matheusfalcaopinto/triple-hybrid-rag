"""Call and artifact models."""

from datetime import datetime

from sqlalchemy import DateTime, Enum, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from control_plane.db.models.base import Base, JSONB, TimestampMixin


class Call(Base, TimestampMixin):
    """Call model."""

    __tablename__ = "calls"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    establishment_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("establishments.id", ondelete="CASCADE"), nullable=False, index=True
    )
    agent_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("agents.id", ondelete="SET NULL"), nullable=True, index=True
    )
    agent_version_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("agent_versions.id", ondelete="SET NULL"), nullable=True
    )
    runtime_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("runtimes.id", ondelete="SET NULL"), nullable=True
    )

    # Provider info
    provider: Mapped[str] = mapped_column(
        Enum("twilio", "whatsapp", name="call_provider"),
        nullable=False,
        default="twilio",
    )
    provider_call_sid: Mapped[str] = mapped_column(
        String(100), unique=True, nullable=False, index=True
    )

    # Call direction and parties
    direction: Mapped[str] = mapped_column(
        Enum("inbound", "outbound", name="call_direction"),
        nullable=False,
    )
    from_e164: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    to_e164: Mapped[str] = mapped_column(String(20), nullable=False, index=True)

    # Status
    status: Mapped[str] = mapped_column(
        Enum(
            "queued",
            "ringing",
            "in_progress",
            "completed",
            "busy",
            "failed",
            "no_answer",
            "cancelled",
            "handoff",
            name="call_status",
        ),
        nullable=False,
        default="queued",
    )

    # Timestamps
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    answered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Hangup info
    hangup_by: Mapped[str | None] = mapped_column(String(50), nullable=True)
    final_status: Mapped[str | None] = mapped_column(String(50), nullable=True)

    # Sentiment (computed async)
    sentiment_label: Mapped[str | None] = mapped_column(String(20), nullable=True)
    sentiment_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    sentiment_computed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Campaign reference (if part of a campaign)
    campaign_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("campaigns.id", ondelete="SET NULL"), nullable=True
    )
    lead_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("leads.id", ondelete="SET NULL"), nullable=True
    )


class CallEvent(Base):
    """Call lifecycle events."""

    __tablename__ = "call_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    call_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("calls.id", ondelete="CASCADE"), nullable=False, index=True
    )
    event_type: Mapped[str] = mapped_column(String(50), nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    idempotency_key: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    payload: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default="now()"
    )


class CallTranscriptSegment(Base):
    """Transcript segment for a call."""

    __tablename__ = "call_transcript_segments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    call_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("calls.id", ondelete="CASCADE"), nullable=False, index=True
    )
    segment_id: Mapped[str] = mapped_column(String(100), nullable=False)
    speaker: Mapped[str] = mapped_column(
        Enum("customer", "agent", "system", name="speaker_type"),
        nullable=False,
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    ended_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    idempotency_key: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default="now()"
    )


class CallSummary(Base, TimestampMixin):
    """Call summary generated post-call."""

    __tablename__ = "call_summaries"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    call_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("calls.id", ondelete="CASCADE"), unique=True, nullable=False
    )
    summary_text: Mapped[str] = mapped_column(Text, nullable=False)
    key_points: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    action_items: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    customer_intent: Mapped[str | None] = mapped_column(String(100), nullable=True)
    resolution_status: Mapped[str | None] = mapped_column(String(50), nullable=True)
    model_used: Mapped[str | None] = mapped_column(String(100), nullable=True)


class CallRecording(Base, TimestampMixin):
    """Call recording reference."""

    __tablename__ = "call_recordings"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    call_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("calls.id", ondelete="CASCADE"), unique=True, nullable=False
    )
    recording_url: Mapped[str] = mapped_column(Text, nullable=False)
    storage_provider: Mapped[str] = mapped_column(String(50), nullable=False)  # s3, supabase, etc.
    content_type: Mapped[str] = mapped_column(String(50), default="audio/wav", nullable=False)
    duration_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    file_size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)


class CallToolLog(Base):
    """Tool call log for a call."""

    __tablename__ = "call_tool_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    call_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("calls.id", ondelete="CASCADE"), nullable=False, index=True
    )
    tool_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    arguments: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    result: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    idempotency_key: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default="now()"
    )
