"""Agent and version models."""

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, DateTime, Enum, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from control_plane.db.models.base import Base, JSONB, TimestampMixin

if TYPE_CHECKING:
    from control_plane.db.models.establishment import Establishment


class Agent(Base, TimestampMixin):
    """Agent model."""

    __tablename__ = "agents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    establishment_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("establishments.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    active_version_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("agent_versions.id", ondelete="SET NULL"), nullable=True
    )

    # Relationships
    establishment: Mapped["Establishment"] = relationship("Establishment", back_populates="agents")
    versions: Mapped[list["AgentVersion"]] = relationship(
        "AgentVersion",
        back_populates="agent",
        cascade="all, delete-orphan",
        foreign_keys="AgentVersion.agent_id",
    )


class AgentVersion(Base, TimestampMixin):
    """Agent version model - contains the "brain" configuration."""

    __tablename__ = "agent_versions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    agent_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("agents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)  # e.g., "v1", "v2"
    status: Mapped[str] = mapped_column(
        Enum("draft", "published", "archived", name="version_status"),
        nullable=False,
        default="draft",
    )

    # Brain configuration
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    model_config: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default={
            "provider": "openai",
            "model": "gpt-4.1-mini",
            "temperature": 0.7,
            "max_tokens": 1024,
        },
    )
    tools_config: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default={
            "email": {"enabled": False},
            "calendar": {"enabled": False},
            "whatsapp_messaging": {"enabled": False},
        },
    )

    # Metadata
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    published_by_user_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )

    # Relationships
    agent: Mapped["Agent"] = relationship(
        "Agent", back_populates="versions", foreign_keys=[agent_id]
    )


class AgentDeployment(Base, TimestampMixin):
    """Tracks which agent version is deployed to which runtime."""

    __tablename__ = "agent_deployments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    agent_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("agents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    version_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("agent_versions.id", ondelete="CASCADE"), nullable=False
    )
    runtime_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("runtimes.id", ondelete="CASCADE"), nullable=False
    )
    status: Mapped[str] = mapped_column(
        Enum("deploying", "active", "failed", "rolled_back", name="deployment_status"),
        nullable=False,
        default="deploying",
    )
    deployed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    deployed_by_user_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
