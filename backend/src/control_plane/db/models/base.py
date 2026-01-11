"""Base model and mixins."""

from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, JSON, MetaData, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import TypeDecorator
from sqlalchemy.dialects.postgresql import JSONB as PG_JSONB
from sqlalchemy.dialects.postgresql import ARRAY as PG_ARRAY

# Naming convention for constraints
convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class JSONB(TypeDecorator):
    """JSONB type that falls back to JSON for SQLite."""
    
    impl = JSON
    cache_ok = True
    
    def load_dialect_impl(self, dialect):
        if dialect.name == 'postgresql':
            return dialect.type_descriptor(PG_JSONB())
        else:
            return dialect.type_descriptor(JSON())


class StringArray(TypeDecorator):
    """String array type that falls back to JSON for SQLite."""
    
    impl = JSON
    cache_ok = True
    
    def load_dialect_impl(self, dialect):
        if dialect.name == 'postgresql':
            return dialect.type_descriptor(PG_ARRAY(String(255)))
        else:
            return dialect.type_descriptor(JSON())
    
    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        return list(value)
    
    def process_result_value(self, value, dialect):
        if value is None:
            return None
        return list(value)


class Base(DeclarativeBase):
    """Base class for all models."""

    metadata = MetaData(naming_convention=convention)

    def to_dict(self) -> dict[str, Any]:
        """Convert model to dictionary."""
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


class TimestampMixin:
    """Mixin for created_at and updated_at timestamps."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
