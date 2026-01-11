"""User and authentication models."""

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, DateTime, Enum, ForeignKey, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from control_plane.db.models.base import Base, TimestampMixin

if TYPE_CHECKING:
    from control_plane.db.models.establishment import Establishment


class User(Base, TimestampMixin):
    """User account model."""

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(Text, nullable=False)
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_superuser: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    default_establishment_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("establishments.id", ondelete="SET NULL"), nullable=True
    )
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationships
    memberships: Mapped[list["EstablishmentUser"]] = relationship(
        "EstablishmentUser", back_populates="user", cascade="all, delete-orphan"
    )


class EstablishmentUser(Base, TimestampMixin):
    """User membership in an establishment with role."""

    __tablename__ = "establishment_users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    establishment_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("establishments.id", ondelete="CASCADE"), nullable=False, index=True
    )
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    role: Mapped[str] = mapped_column(
        Enum("admin", "operator", "viewer", name="user_role"),
        nullable=False,
        default="viewer",
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="memberships")
    establishment: Mapped["Establishment"] = relationship(
        "Establishment", back_populates="members"
    )


class UserInvitation(Base, TimestampMixin):
    """User invitation to join an establishment."""

    __tablename__ = "user_invitations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    establishment_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("establishments.id", ondelete="CASCADE"), nullable=False, index=True
    )
    email: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    role: Mapped[str] = mapped_column(
        Enum("admin", "operator", "viewer", name="user_role"),
        nullable=False,
        default="viewer",
    )
    invited_by_user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    status: Mapped[str] = mapped_column(
        Enum("pending", "accepted", "expired", "cancelled", name="invitation_status"),
        nullable=False,
        default="pending",
    )
    token: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: func.now() + func.make_interval(0, 0, 0, 7),  # 7 days
        nullable=False,
    )
