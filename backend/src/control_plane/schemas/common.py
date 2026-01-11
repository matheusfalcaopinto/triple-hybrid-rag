"""Common schemas."""

from typing import Any, Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class ErrorDetail(BaseModel):
    """Error detail."""

    code: str
    message: str
    details: dict[str, Any] | None = None


class ErrorResponse(BaseModel):
    """Error response wrapper."""

    error: ErrorDetail


class SuccessResponse(BaseModel):
    """Generic success response."""

    status: str = "ok"
    message: str | None = None


class PaginatedResponse(BaseModel, Generic[T]):
    """Paginated response wrapper."""

    items: list[T]
    total: int | None = None
    cursor: str | None = None
    has_more: bool = False
