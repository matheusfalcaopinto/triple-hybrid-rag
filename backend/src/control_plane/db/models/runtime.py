"""Runtime (Docker container) model."""

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, DateTime, Enum, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from control_plane.db.models.base import Base, JSONB, TimestampMixin

if TYPE_CHECKING:
    from control_plane.db.models.establishment import Establishment


class Runtime(Base, TimestampMixin):
    """Runtime instance (Docker container) model."""

    __tablename__ = "runtimes"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    establishment_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("establishments.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
        index=True,
    )

    # Docker info
    container_name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    container_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    image_tag: Mapped[str] = mapped_column(String(255), nullable=False)

    # Network
    host_port: Mapped[int] = mapped_column(Integer, unique=True, nullable=False)
    base_url: Mapped[str] = mapped_column(String(255), nullable=False)

    # Status
    status: Mapped[str] = mapped_column(
        Enum(
            "created",
            "starting",
            "running",
            "stopping",
            "stopped",
            "draining",
            "error",
            name="runtime_status",
        ),
        nullable=False,
        default="created",
    )
    is_ready: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    ready_issues: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    last_health_check_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Runtime info from agent
    agent_info: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # Environment variables (encrypted in production)
    env_config: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # Relationships
    establishment: Mapped["Establishment"] = relationship(
        "Establishment", back_populates="runtime"
    )
