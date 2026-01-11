"""Establishment (tenant) models."""

from typing import TYPE_CHECKING

from sqlalchemy import Boolean, Enum, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from control_plane.db.models.base import Base, JSONB, StringArray, TimestampMixin

if TYPE_CHECKING:
    from control_plane.db.models.agent import Agent
    from control_plane.db.models.integration import IntegrationConnection
    from control_plane.db.models.runtime import Runtime
    from control_plane.db.models.user import EstablishmentUser


class Establishment(Base, TimestampMixin):
    """Establishment (tenant) model."""

    __tablename__ = "establishments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    timezone: Mapped[str] = mapped_column(String(50), default="UTC", nullable=False)
    locale: Mapped[str] = mapped_column(String(10), default="en-US", nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # Billing (stub)
    billing_plan: Mapped[str | None] = mapped_column(String(50), nullable=True)
    billing_status: Mapped[str | None] = mapped_column(String(50), nullable=True)

    # Relationships
    members: Mapped[list["EstablishmentUser"]] = relationship(
        "EstablishmentUser", back_populates="establishment", cascade="all, delete-orphan"
    )
    agents: Mapped[list["Agent"]] = relationship(
        "Agent", back_populates="establishment", cascade="all, delete-orphan"
    )
    phone_numbers: Mapped[list["PhoneNumber"]] = relationship(
        "PhoneNumber", back_populates="establishment", cascade="all, delete-orphan"
    )
    telephony_policy: Mapped["EstablishmentTelephonyPolicy | None"] = relationship(
        "EstablishmentTelephonyPolicy",
        back_populates="establishment",
        uselist=False,
        cascade="all, delete-orphan",
    )
    runtime: Mapped["Runtime | None"] = relationship(
        "Runtime", back_populates="establishment", uselist=False, cascade="all, delete-orphan"
    )
    integration_connections: Mapped[list["IntegrationConnection"]] = relationship(
        "IntegrationConnection", back_populates="establishment", cascade="all, delete-orphan"
    )


class EstablishmentTelephonyPolicy(Base, TimestampMixin):
    """Telephony policy for an establishment."""

    __tablename__ = "establishment_telephony_policies"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    establishment_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("establishments.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )

    # Handoff / transfer
    handoff_e164: Mapped[str | None] = mapped_column(String(20), nullable=True)

    # Call limits
    max_call_duration_seconds: Mapped[int] = mapped_column(Integer, default=900, nullable=False)

    # Business hours (JSON array)
    business_hours: Mapped[list | None] = mapped_column(JSONB, nullable=True)

    # Blacklist (array of E.164 numbers)
    blacklist_e164: Mapped[list[str] | None] = mapped_column(StringArray(), nullable=True)

    # Relationship
    establishment: Mapped["Establishment"] = relationship(
        "Establishment", back_populates="telephony_policy"
    )


class PhoneNumber(Base, TimestampMixin):
    """Phone number assigned to an establishment."""

    __tablename__ = "phone_numbers"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    establishment_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("establishments.id", ondelete="CASCADE"), nullable=False, index=True
    )
    e164: Mapped[str] = mapped_column(String(20), unique=True, nullable=False, index=True)
    provider: Mapped[str] = mapped_column(
        Enum("twilio", "whatsapp", name="phone_provider"),
        nullable=False,
        default="twilio",
    )
    routing_agent_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("agents.id", ondelete="SET NULL"), nullable=True
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    display_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    provider_sid: Mapped[str | None] = mapped_column(String(100), nullable=True)

    # Relationships
    establishment: Mapped["Establishment"] = relationship(
        "Establishment", back_populates="phone_numbers"
    )
