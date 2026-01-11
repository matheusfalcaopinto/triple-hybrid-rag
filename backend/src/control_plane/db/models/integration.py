"""Integration models."""

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, DateTime, Enum, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from control_plane.db.models.base import Base, JSONB, TimestampMixin

if TYPE_CHECKING:
    from control_plane.db.models.establishment import Establishment


class Integration(Base, TimestampMixin):
    """Available integration types."""

    __tablename__ = "integrations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    type: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    icon_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    category: Mapped[str] = mapped_column(String(50), nullable=False)  # email, calendar, crm, etc.
    auth_type: Mapped[str] = mapped_column(
        Enum("oauth2", "api_key", "basic", name="auth_type"),
        nullable=False,
    )
    oauth_config: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)


class IntegrationConnection(Base, TimestampMixin):
    """Connection of an integration to an establishment."""

    __tablename__ = "integration_connections"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    establishment_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("establishments.id", ondelete="CASCADE"), nullable=False, index=True
    )
    integration_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("integrations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    display_name: Mapped[str] = mapped_column(String(100), nullable=False)

    # Status
    status: Mapped[str] = mapped_column(
        Enum(
            "pending_auth",
            "active",
            "error",
            "disabled",
            "expired",
            name="connection_status",
        ),
        nullable=False,
        default="pending_auth",
    )
    is_enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # Auth data (encrypted in production)
    auth_data: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    access_token_encrypted: Mapped[str | None] = mapped_column(Text, nullable=True)
    refresh_token_encrypted: Mapped[str | None] = mapped_column(Text, nullable=True)
    token_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Error tracking
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_error_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Usage tracking
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationships
    establishment: Mapped["Establishment"] = relationship(
        "Establishment", back_populates="integration_connections"
    )
