"""Lead model."""

from datetime import datetime

from sqlalchemy import DateTime, Enum, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from control_plane.db.models.base import Base, JSONB, TimestampMixin


class Lead(Base, TimestampMixin):
    """Lead model for campaign dialing."""

    __tablename__ = "leads"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    establishment_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("establishments.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Contact info
    phone_e164: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    email: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Status
    status: Mapped[str] = mapped_column(
        Enum(
            "new",
            "contacted",
            "qualified",
            "converted",
            "not_interested",
            "invalid",
            "do_not_call",
            name="lead_status",
        ),
        nullable=False,
        default="new",
    )

    # Source
    source: Mapped[str | None] = mapped_column(String(100), nullable=True)  # e.g., "csv_import"
    import_job_id: Mapped[str | None] = mapped_column(String(36), nullable=True)

    # Custom fields
    custom_fields: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # Notes
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Last contact
    last_contacted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    last_call_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("calls.id", ondelete="SET NULL"), nullable=True
    )
    total_calls: Mapped[int] = mapped_column(default=0, nullable=False)
