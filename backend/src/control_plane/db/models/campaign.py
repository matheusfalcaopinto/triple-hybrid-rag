"""Campaign models."""

from datetime import datetime

from sqlalchemy import DateTime, Enum, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from control_plane.db.models.base import Base, JSONB, TimestampMixin


class Campaign(Base, TimestampMixin):
    """Campaign model for automated dialing."""

    __tablename__ = "campaigns"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    establishment_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("establishments.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)

    # Agent config
    agent_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("agents.id", ondelete="CASCADE"), nullable=False
    )
    agent_version_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("agent_versions.id", ondelete="SET NULL"), nullable=True
    )

    # Status
    status: Mapped[str] = mapped_column(
        Enum("created", "running", "paused", "completed", "cancelled", name="campaign_status"),
        nullable=False,
        default="created",
    )

    # Lead filter criteria
    lead_filter: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # Pacing
    pace_config: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default={"calls_per_minute": 6, "max_concurrent": 10},
    )

    # Schedule
    schedule: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default={
            "timezone": "UTC",
            "days": ["mon", "tue", "wed", "thu", "fri"],
            "start": "09:00",
            "end": "18:00",
        },
    )

    # Stats
    total_leads: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    leads_contacted: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    leads_completed: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # Timestamps
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class CampaignEnrollment(Base, TimestampMixin):
    """Enrollment of a lead in a campaign."""

    __tablename__ = "campaign_enrollments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    campaign_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("campaigns.id", ondelete="CASCADE"), nullable=False, index=True
    )
    lead_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("leads.id", ondelete="CASCADE"), nullable=False, index=True
    )
    status: Mapped[str] = mapped_column(
        Enum(
            "pending",
            "in_progress",
            "completed",
            "failed",
            "skipped",
            "retry",
            name="enrollment_status",
        ),
        nullable=False,
        default="pending",
    )
    attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    last_attempt_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    next_attempt_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    last_call_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("calls.id", ondelete="SET NULL"), nullable=True
    )


class CampaignRun(Base, TimestampMixin):
    """Campaign run session (for tracking individual runs)."""

    __tablename__ = "campaign_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    campaign_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("campaigns.id", ondelete="CASCADE"), nullable=False, index=True
    )
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(
        Enum("running", "completed", "paused", "error", name="run_status"),
        nullable=False,
        default="running",
    )
    calls_made: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    calls_answered: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error_message: Mapped[str | None] = mapped_column(String(500), nullable=True)
